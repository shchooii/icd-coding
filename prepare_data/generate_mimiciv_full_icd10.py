#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV(train/dev/test) -> unified feather + split feather (+ optional Zipf long-tail JSON)

- Default _id is hadm_id
- If id_map.csv (hadm_id -> new _id) is provided, replace _id in BOTH full and split
  to keep them consistent (important for train-only Zipf stats).
"""

import ast, json, re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf


# =========================================================
# (1) Config: fill these variables (no argparse)
# =========================================================
INPUT_DIR = None  # e.g., "/Users/houn/Desktop" (if set, uses train_full/dev_full/test_full.csv under this dir)
TRAIN_CSV = None  # e.g., "/path/train_full.csv" (used when INPUT_DIR is None)
DEV_CSV   = None  # e.g., "/path/dev_full.csv"
TEST_CSV  = None  # e.g., "/path/test_full.csv"

OUT_DIR = ""      # e.g., "/path/out" (required)

DEV_SPLIT_NAME = "val"  # "val" or "dev"

ID_MAP_CSV = None       # e.g., "/path/id_map.csv" with columns: hadm_id,_id ; set None to disable

BUILD_ZIPF_JSON = False
ZIPF_HEAD_MASS = 0.6
ZIPF_MEDIUM_MASS = 0.3


# =========================================================
# (2) ICD-10 Regex
# =========================================================
RE_CM  = re.compile(r'^[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?$')  # ICD-10-CM (diagnosis)
RE_PCS = re.compile(r'^[0-9A-HJ-NP-Z]{7}$')                           # ICD-10-PCS (procedure), 7 chars, no I/O

REQ_COLS_IN = {'subject_id','hadm_id','text','labels'}

REQ_COLS_OUT = [
    'note_id','subject_id','_id','note_type','note_seq','charttime','storetime',
    'text','icd10_proc','icd10_diag','target','num_words','num_targets'
]


# =========================================================
# (3) Parsing
# =========================================================
def parse_labels_cell(val):
    """Parse a labels cell into list[str]. Handles '[...]', tuples, ';', ',', or whitespace separated strings."""
    if isinstance(val, list):
        items = val
    elif isinstance(val, (tuple, np.ndarray)):
        items = list(val)
    elif val is None:
        items = []
    else:
        s = str(val).strip()
        if s == "" or s.lower() in ("nan", "none"):
            items = []
        elif (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                v = ast.literal_eval(s)
                items = list(v) if isinstance(v, (list, tuple, np.ndarray)) else [s]
            except Exception:
                items = re.split(r"[;,\s]+", s)
        else:
            items = s.split(";") if ";" in s else re.split(r"[;,\s]+", s)

    out = []
    for t in items:
        z = str(t).strip().upper()
        if z:
            out.append(z)
    return out


def split_icd_codes(labels):
    """Split labels(list[str]) into (diag, proc, target) with basic validation + fallback heuristics."""
    diag, proc = set(), set()
    for code in labels:
        if RE_PCS.fullmatch(code):
            proc.add(code); continue
        if RE_CM.fullmatch(code):
            diag.add(code); continue

        # fallback heuristics
        if "." in code and RE_CM.match(code):
            diag.add(code)
        elif len(code) == 7 and RE_PCS.match(code):
            proc.add(code)
        else:
            diag.add(code)

    diag = sorted(diag)
    proc = sorted(proc)
    target = sorted(set(diag + proc))
    return diag, proc, target


def to_pa_list_of_str(series: pd.Series):
    """Convert a pandas Series of list[str] to Arrow list<string>."""
    return pa.array([[str(x) for x in (v if isinstance(v, list) else [])] for v in series],
                    type=pa.list_(pa.string()))


# =========================================================
# (4) Build one split
# =========================================================
def build_one_from_csv(csv_path: Path, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[])
    miss = REQ_COLS_IN - set(df.columns)
    if miss:
        raise RuntimeError(f"{csv_path.name}: missing required columns: {sorted(miss)}")

    # Deterministic per-subject ordering for note_seq
    df['_row_order'] = np.arange(len(df))
    df['note_seq'] = df.groupby('subject_id')['_row_order'].rank(method='first').astype(int)

    labs = df['labels'].map(parse_labels_cell)

    diags, procs, targets = [], [], []
    for lab in labs:
        d, p, t = split_icd_codes(lab)
        diags.append(d)
        procs.append(p)
        targets.append(t)

    # Compute num_words (prefer 'length' if numeric)
    if 'length' in df.columns:
        def pick_len(text_val, length_val):
            s = str(length_val).strip()
            return int(s) if s.isdigit() else len(str(text_val).split())
        num_words = [pick_len(t, l) for t, l in zip(df['text'], df['length'])]
    else:
        num_words = df['text'].astype(str).map(lambda s: len(s.split())).tolist()

    out = pd.DataFrame({
        'note_id':    [f"{sid}-DS-{seq}" for sid, seq in zip(df['subject_id'], df['note_seq'])],
        'subject_id': df['subject_id'].astype(str).str.strip(),
        '_id':        df['hadm_id'].astype(str).str.strip(),  # Default: use hadm_id (can be replaced later)
        'note_type':  "DS",
        'note_seq':   df['note_seq'].astype(str),
        'charttime':  "",
        'storetime':  "",
        'text':       df['text'].astype(str),
        'icd10_proc': procs,
        'icd10_diag': diags,
        'target':     targets,
        'num_words':  num_words,
        'num_targets':[len(t) for t in targets],
    })[REQ_COLS_OUT]

    split_df = out[['_id']].copy()
    split_df['split'] = split_name
    return out, split_df


# =========================================================
# (5) ID replacement logic (core)
# =========================================================
def load_id_map(id_map_csv: Path) -> pd.DataFrame:
    """
    Load an ID mapping file.
    Expected columns: hadm_id, _id
    Returns a DataFrame with columns: old_id, new_id
    """
    mp = pd.read_csv(id_map_csv, dtype=str, keep_default_na=False, na_values=[])
    if not {'hadm_id','_id'}.issubset(mp.columns):
        raise RuntimeError("id_map must contain columns: hadm_id,_id")
    mp = mp[['hadm_id','_id']].drop_duplicates()
    mp['hadm_id'] = mp['hadm_id'].astype(str).str.strip()
    mp['_id'] = mp['_id'].astype(str).str.strip()
    return mp.rename(columns={'hadm_id':'old_id', '_id':'new_id'})


def apply_id_map_to_ids(ids: pd.Series, mp: pd.DataFrame) -> pd.Series:
    """
    Replace ids using mapping mp(old_id -> new_id).
    ids: pandas Series containing current _id values.
    """
    tmp = pd.DataFrame({'_id': ids.astype(str)})
    tmp = tmp.merge(mp, left_on='_id', right_on='old_id', how='left')
    repl = tmp['new_id'].astype(str).str.len() > 0
    out = np.where(repl, tmp['new_id'], tmp['_id'])
    return pd.Series(out, index=ids.index)


def apply_id_map(full: pd.DataFrame, split: pd.DataFrame, id_map_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the same hadm_id -> new _id mapping to BOTH:
      - full['_id']
      - split['_id']
    This keeps the ID space consistent across outputs.
    """
    mp = load_id_map(id_map_csv)

    full = full.copy()
    split = split.copy()

    full['_id'] = apply_id_map_to_ids(full['_id'], mp)
    split['_id'] = apply_id_map_to_ids(split['_id'], mp)

    print(f"[INFO] id_map applied to full and split | map_rows={len(mp)}")
    return full, split


# =========================================================
# (6) Zipf JSON (optional)
# =========================================================
def fit_zipf_s(counts_desc: np.ndarray) -> float:
    """Estimate Zipf exponent s from sorted counts via log-log linear fit."""
    ranks = np.arange(1, len(counts_desc)+1, dtype=np.float64)
    y = np.log(counts_desc.astype(np.float64))
    x = np.log(ranks)
    B, _A = np.polyfit(x, y, 1)
    s = -B
    return max(float(s), 0.5)


def zipf_mass_cuts(L: int, s: float, head_mass: float, medium_mass: float) -> tuple[int,int]:
    """Return rank cutoffs (head_end, medium_end) based on Zipf mass thresholds."""
    ranks = np.arange(1, L+1, dtype=np.float64)
    w = ranks ** (-s)
    p = w / w.sum()
    c = p.cumsum()
    r_head = int(np.searchsorted(c, head_mass, side='right'))
    r_med  = int(np.searchsorted(c, head_mass + medium_mass, side='right'))
    r_head = min(max(r_head, 1), L)
    r_med  = min(max(r_med,  r_head), L)
    return r_head, r_med


def build_zipf_json_from_train(full: pd.DataFrame, split: pd.DataFrame, out_json: Path,
                               head_mass=0.6, medium_mass=0.3):
    """
    Build a Zipf-based head/medium/tail split using TRAIN only.
    Output JSON includes counts per code and estimated Zipf exponent.
    """
    train_ids = set(split.loc[split["split"].astype(str).str.lower()=="train", "_id"].astype(str))
    ftrain = full[full["_id"].astype(str).isin(train_ids)]
    if ftrain.empty:
        raise RuntimeError("No train rows found when building Zipf JSON.")

    cnt = Counter()
    for row in ftrain['target']:
        if isinstance(row, list):
            for z in row:
                z = str(z).strip().upper()
                if z:
                    cnt[z] += 1

    pairs = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    codes, counts = zip(*pairs)
    counts = np.array(counts, dtype=np.int64)
    L = len(codes)

    s = fit_zipf_s(counts)
    r_head, r_med = zipf_mass_cuts(L, s, head_mass, medium_mass)

    head_pairs   = pairs[:r_head]
    medium_pairs = pairs[r_head:r_med]
    tail_pairs   = pairs[r_med:]

    def to_dict(ps): return {c: int(v) for c, v in ps}

    out = {
        "meta": {
            "basis": "zipf",
            "s": s,
            "head_mass": head_mass,
            "medium_mass": medium_mass,
            "tail_mass": max(0.0, 1.0 - head_mass - medium_mass),
            "L": int(L),
            "total_occurrences": int(counts.sum()),
            "ranks": {"head_end": int(r_head), "medium_end": int(r_med)}
        },
        "head":   to_dict(head_pairs),
        "medium": to_dict(medium_pairs),
        "tail":   to_dict(tail_pairs),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[ZIPF] wrote {out_json} | head={len(out['head'])}, medium={len(out['medium'])}, tail={len(out['tail'])}, s≈{s:.3f}")


# =========================================================
# (7) Main
# =========================================================
def main():
    out_dir = Path(OUT_DIR)
    if not str(out_dir).strip():
        raise RuntimeError("OUT_DIR must be set.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve CSV paths
    if INPUT_DIR:
        in_dir = Path(INPUT_DIR)
        train_csv = in_dir / "train_full.csv"
        dev_csv   = in_dir / "dev_full.csv"
        test_csv  = in_dir / "test_full.csv"
    else:
        if not (TRAIN_CSV and DEV_CSV and TEST_CSV):
            raise RuntimeError("Set INPUT_DIR or set TRAIN_CSV/DEV_CSV/TEST_CSV explicitly.")
        train_csv = Path(TRAIN_CSV)
        dev_csv   = Path(DEV_CSV)
        test_csv  = Path(TEST_CSV)

    for p in (train_csv, dev_csv, test_csv):
        if not p.exists():
            raise FileNotFoundError(str(p))

    # Build splits
    full_parts, split_parts = [], []
    for path, split_name in [(train_csv, "train"), (dev_csv, DEV_SPLIT_NAME), (test_csv, "test")]:
        fi, si = build_one_from_csv(path, split_name)
        full_parts.append(fi); split_parts.append(si)
        print(f"[OK] {path.name}: rows={len(fi)}")

    full  = pd.concat(full_parts,  ignore_index=True)
    split = pd.concat(split_parts, ignore_index=True)

    # Apply ID mapping to BOTH full and split (important)
    if ID_MAP_CSV:
        full, split = apply_id_map(full, split, Path(ID_MAP_CSV))

    # Write feather files
    table = pa.table({
        'note_id':     pa.array(full['note_id'].astype(str)),
        'subject_id':  pa.array(full['subject_id'].astype(str)),
        '_id':         pa.array(full['_id'].astype(str)),
        'note_type':   pa.array(full['note_type'].astype(str)),
        'note_seq':    pa.array(full['note_seq'].astype(str)),
        'charttime':   pa.array(full['charttime'].astype(str)),
        'storetime':   pa.array(full['storetime'].astype(str)),
        'text':        pa.array(full['text'].astype(str)),
        'icd10_proc':  to_pa_list_of_str(full['icd10_proc']),
        'icd10_diag':  to_pa_list_of_str(full['icd10_diag']),
        'target':      to_pa_list_of_str(full['target']),
        'num_words':   pa.array(pd.to_numeric(full['num_words'], errors='coerce').fillna(0).astype('int64')),
        'num_targets': pa.array(pd.to_numeric(full['num_targets'], errors='coerce').fillna(0).astype('int64')),
    })
    pf.write_feather(table, out_dir / "mimiciv_full_icd10.feather")

    pf.write_feather(pa.table({
        "_id":   pa.array(split["_id"].astype(str)),
        "split": pa.array(split["split"].astype(str)),
    }), out_dir / "mimiciv_full_icd10_split.feather")

    print("[DONE] wrote", out_dir / "mimiciv_full_icd10.feather")
    print("[DONE] wrote", out_dir / "mimiciv_full_icd10_split.feather")
    print("[CHECK] full.shape:", full.shape, "| split.shape:", split.shape)
    print("[CHECK] unique _id:", full['_id'].nunique(), "| dup _id rows:", int(full.duplicated('_id').sum()))

    # Optional Zipf JSON (train-only)
    if BUILD_ZIPF_JSON:
        build_zipf_json_from_train(
            full=full,
            split=split,
            out_json=out_dir / "icd10_longtail_split.json",
            head_mass=ZIPF_HEAD_MASS,
            medium_mass=ZIPF_MEDIUM_MASS
        )


if __name__ == "__main__":
    main()
