### Overview

This repository provides the official implementation for the paper
“Mitigating Imbalance in ICD Coding: A Comprehensive Evaluation of Loss Functions and Threshold Optimization.”

It focuses on:

* Dataset preparation and standardized splits
* Training and evaluation commands
* Threshold tuning strategies (single / per-class / per-group)

Figures and pretrained checkpoints are not included.

### Dataset Preparation

This repository supports two setups:

* **MIMIC-IV ICD-10 benchmark (7,942 labels)** following Edin et al. (SIGIR 2023)
* **MIMIC-IV-Full ICD-10 (26,096 labels)** following Nguyen et al. (“MIMIC-IV-ICD-data-processing”)

#### Accessing MIMIC datasets

You must obtain access via PhysioNet and complete the required CITI training.

#### MIMIC-IV benchmark preprocessing (Edin et al.)

1. Download MIMIC-IV and MIMIC-IV-Note from PhysioNet.
2. Configure dataset paths (example):

```python
DOWNLOAD_DIRECTORY_MIMICIV = "/path/to/mimiciv"
DOWNLOAD_DIRECTORY_MIMICIV_NOTE = "/path/to/mimiciv-note"
```

3. Run preprocessing:

```bash
python prepare_data/prepare_mimiciv.py
```

This generates standardized train/validation/test splits and processed inputs.

#### MIMIC-IV-Full preprocessing (Nguyen et al.)

For full-scale experiments (26,096 ICD-10 labels), we follow Nguyen et al.’s official pipeline and splits.

Expected directory layout:

```text
mimicdata/
  physionet.org/
    files/
      mimic4_icd10/
        ALL_CODES.csv
        ALL_CODES_filtered.csv
        disch_10_full.csv
        disch_10_filtered.csv
        notes_labeled.csv
        *_hadm_ids.csv
```

Follow the Nguyen repository instructions and run their scripts/notebooks to generate:

* filtered label space
* cleaned discharge summaries
* official train/val/test admission lists (`*_hadm_ids.csv`)

#### Optional: Converting pre-split CSVs to Feather

If you already have:

* `train_full.csv`, `dev_full.csv`, `test_full.csv`
  required columns: `subject_id, hadm_id, text, labels`

You can convert them into unified Feather files:

* `mimiciv_full_icd10.feather`
* `mimiciv_full_icd10_split.feather`

(If you remap IDs, apply the same mapping to both `full` and `split` outputs.)


### Training

```bash
python main.py experiment=mimiciv_icd10/plm_icd gpu=0
```

* `experiment=...` selects model/data/loss/threshold config
* `gpu=0` selects the GPU index


### Evaluation (with your checkpoint)

```bash
python main.py experiment=mimiciv_icd10/plm_icd gpu=0 \
  load_model=/path/to/your/model.ckpt trainer.epochs=0
```

This runs evaluation only.
Pretrained checkpoints are not distributed.

### Threshold Tuning

Supported strategies:

* Fixed threshold (0.5)
* Single global threshold
* Per-label thresholds
* Per-group thresholds (head/medium/tail)

You can switch strategies by changing **only** the function arguments: `type`, `average`, and `groups` (per-group only).

> You can switch thresholding strategies by changing the function arguments—specifically `type` (single / per-label / per-label2 (w sanity checks) / per-group), `average` (micro vs. macro), and `groups` (only required for per-group tuning). No other code changes are needed.

Example:

```python
best_f1, best_thr = f1_score_db_tuning(
    logits=val_probs,        # probabilities in [0,1]
    targets=val_targets,     # multi-hot labels
    groups=groups,           # required only for type="per_group"
    average="micro",
    type="per_group",
)
```

