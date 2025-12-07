# Mitigating Imbalance in ICD Coding: A Comprehensive Evaluation of Loss Functions and Threshold Optimization

This repository provides the official implementation for the paper
“Mitigating Imbalance in ICD Coding: A Comprehensive Evaluation of Loss Functions and Threshold Optimization.”

We present a unified framework for evaluating class-sensitive loss functions and threshold optimization strategies for automated ICD-10 coding on long-tailed clinical datasets (MIMIC-IV and MIMIC-IV-Full).

Our main findings are:

* Carefully designed loss functions substantially improve rare-code performance (Macro-F1)
* These gains can be achieved while preserving strong performance on frequent codes (Micro-F1)
* Simple but principled threshold tuning (especially per-group thresholds) is an essential complement to loss-level calibration

Figures from the paper are not included here. If you want to reuse them, please insert them manually at appropriate places in this README.

Note: Pretrained model checkpoints are not distributed in this repository. You must train models yourself or attach your own checkpoints.

## 1. Method Overview

### 1.1 Problem setting

We formulate automated ICD coding as long-tailed multi-label classification.

* Input: Discharge summaries
* Output: Multi-hot ICD-10 label vector (7,942+ codes on MIMIC-IV; 26,096 codes on MIMIC-IV-Full)
* Model: PLM-based encoder (PLM-ICD) with label-wise attention (LAAT-style decoder)
* Metrics: Micro-F1, Macro-F1, AUC-ROC (micro/macro), MAP, P@k, EMR

The baseline objective is standard Binary Cross-Entropy (BCE) over sigmoid probabilities.

### 1.2 Layer-wise taxonomy of long-tailed loss functions

We organize loss functions into five conceptual layers (L0–L4), from simple baselines to long-tailed multi-label specific objectives.

Reconstructed taxonomy:

| Layer | Key idea                    | Techniques (examples)                           |
| ----- | --------------------------- | ----------------------------------------------- |
| L0    | Baseline, uniform weighting | BCE                                             |
| L1    | Symmetric focusing          | Hill Loss, Focal Loss                           |
| L2    | Explicit class re-weighting | Class-Balanced (CB), Distribution-Balanced (DB) |
| L3    | Asymmetric / polynomial     | ASL, APL, RAL                                   |
| L4    | LT-MLC specific objectives  | MFM, COMIC (RLC + MFM + HTB)                    |

Intuition:

* L0: Treats all classes and samples equally; fails under extreme head–tail imbalance.
* L1: Focuses on hard examples but still symmetric between positive and negative labels.
* L2: Re-weights by class frequency to correct head–tail bias.
* L3: Decouples positive/negative behavior (asymmetric focusing and polynomial shaping).
* L4: Tailored to long-tailed multi-label classification with class-dependent focusing and auxiliary correction/distillation.

## 2. Implemented Models and Losses

### 2.1 PLM-based baseline (PLMICD)

PLMICD is a RoBERTa-based ICD coding model with a label-wise attention decoder.

Key components:

* Text encoder: RoBERTa-base-PM-M3-Voc (clinical PLM)
* Tokenization: RoBERTa BPE (SentencePiece family)
* Decoder: Label-wise attention (LAAT-style)
* Optimization: AdamW with learning rate 5e-5, dropout 0.2

Baseline settings (reconstructed from the paper):

| Item                | Setting                                |
| ------------------- | -------------------------------------- |
| Backbone PLM        | RoBERTa-base-PM-M3-Voc                 |
| Max sequence length | 3072 (processed in 128-token chunks)   |
| Decoder             | Label-wise attention (LAAT)            |
| Optimizer           | AdamW, lr = 5e-5                       |
| Batch size / Epochs | 16 / 20                                |
| Dropout             | 0.2                                    |
| Random seed         | 42                                     |
| Thresholding        | fixed / single / per-class / per-group |

Loss functions implemented for PLMICD include:

* L0: BCEWithLogitsLoss
* L1: Hill Loss, Focal Loss
* L2: Class-Balanced Loss (CB), Distribution-Balanced Loss (DB via ResampleLoss)
* L3: Asymmetric Loss (ASL), Asymmetric Polynomial Loss (APL), Robust Asymmetric Loss (RAL)
* L4: Multi-Grained Focal Loss (MFM), auxiliary variants used by COMIC

Most losses follow hyperparameter values from their original papers (for example, γ = 2.0 for Focal Loss, β = 0.9 for Class-Balanced Loss, recommended γ+, γ−, τ, λ for ASL/APL/RAL/MFM). We intentionally do not perform loss-specific hyperparameter tuning, in order to isolate the design effect of the loss itself.

### 2.2 COMIC-style multi-expert model (PLMICD2)

PLMICD2 implements the COMIC framework with three components:

* Reflective Label Corrector (RLC)
* Multi-Grained Focal Loss (MFM)
* Head–Tail Balancer (HTB) with multi-expert architecture (head expert Mh, tail expert Mt, balanced learner Mb)

Composite COMIC loss:

L_COMIC = λ_c · L_RLC + λ_m · L_MFM + λ_b · L_HTB

RLC attempts to correct missing positives under partial labeling; MFM performs class-dependent focusing; HTB balances head and tail experts and distills their knowledge into the central learner.

## 3. Dataset Preparation

Our experiments use the standardized splits and preprocessing pipeline from:

Edin et al. (SIGIR 2023)
“Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study”

### 3.1 Accessing MIMIC datasets

You must first request access to the MIMIC family of datasets via PhysioNet.

Reconstructed dataset table:

| Dataset       | Version | Source URL              | Description                       |
| ------------- | ------- | ----------------------- | --------------------------------- |
| MIMIC-III     | 1.4     | PhysioNet MIMIC-III     | ICD-9 codes, discharge summaries  |
| MIMIC-IV      | 2.2     | PhysioNet MIMIC-IV      | ICD-10 codes, structured EHR data |
| MIMIC-IV-Note | 2.2     | PhysioNet MIMIC-IV-Note | De-identified clinical notes      |

Access requires completion of the CITI training course on PhysioNet. Registration and approval are mandatory before downloading any data.

### 3.2 Preprocessing

1. Download the raw CSVs from PhysioNet.
   Example paths:

   * /path/to/mimiciii
   * /path/to/mimiciv
   * /path/to/mimiciv-note

2. Configure data paths in the codebase (for example, configs/data/mimiciv.yaml or a global settings module):

```python
DOWNLOAD_DIRECTORY_MIMICIII = "/path/to/mimiciii"
DOWNLOAD_DIRECTORY_MIMICIV = "/path/to/mimiciv"
DOWNLOAD_DIRECTORY_MIMICIV_NOTE = "/path/to/mimiciv-note"
```

3. Run preprocessing scripts:

```bash
# For MIMIC-III
python prepare_data/prepare_mimiciii.py

# For MIMIC-IV and MIMIC-IV-Note
python prepare_data/prepare_mimiciv.py
```

These scripts generate tokenized and split datasets that follow the Edin et al. protocol (approximately 8:1:1 train/validation/test ratio).

### 3.3 MIMIC-IV corpus statistics (benchmark split)

Reconstructed corpus-level statistics:

| Statistic                        | Value                            |
| -------------------------------- | -------------------------------- |
| Number of documents              | 122,279                          |
| Number of patients               | 65,659                           |
| Number of unique ICD-10 codes    | 7,942                            |
| Codes per instance (median, IQR) | 14 (9–20)                        |
| Words per document (median, IQR) | 1,492 (1,147–1,931)              |
| Documents Train / Val / Test (%) | 72.9 / 10.9 / 16.2               |
| Notes with missing codes (%)     | 0.0 / 0.5 / 0.1 (Train/Val/Test) |

### 3.4 MIMIC-IV-Full corpus statistics (extended split)

Extended experiments use the full ICD-10 label space:

| Statistic                        | Value                        |
| -------------------------------- | ---------------------------- |
| Number of documents              | 122,309                      |
| Number of patients               | 65,682                       |
| Number of unique ICD-10 codes    | 26,096                       |
| Codes per instance (median, IQR) | 15 (10–21)                   |
| Words per document (median, IQR) | 1,553 (1,192–2,010)          |
| Documents Train / Val / Test (%) | 90.3 / 3.3 / 6.4             |
| Notes with missing codes (%)     | 0.0 / 0.0 / 0.0 (all splits) |

## 4. Frequency Stratification and Thresholding

### 4.1 Head / Medium / Tail stratification

ICD-10 code frequencies follow a Zipf-like long tail. We stratify labels into three groups using cumulative frequency (not raw code count):

* Head: top 5% of label occurrences
* Medium: next 15%
* Tail: remaining 80%

This stratification is computed on the training split only and kept fixed for validation and test to avoid leakage.

### 4.2 Threshold tuning strategies

We study several decision threshold strategies:

* Fixed threshold: 0.5 for all classes
* Single global threshold:

  * Optimized for Micro-F1
  * Optimized for Macro-F1
* Per-class thresholds:

  * Individual threshold per ICD-10 code
* Per-group thresholds:

  * Separate thresholds for head, medium, and tail subsets
  * Optimized either for Micro-F1 or Macro-F1 on validation data

Thresholds are selected by sweeping values from 0.05 to 0.95 with step 0.01 on the validation set.

Per-group thresholds provide a robust trade-off between Micro-F1 and Macro-F1 and avoid instability seen in per-class tuning for large label spaces (C = 7,942 or 26,096).

## 5. Results (Reconstructed)

This section summarizes key results from the paper. For detailed numbers, please refer to the main manuscript.

### 5.1 Overall performance on MIMIC-IV ICD-10

Reconstructed summary table for main losses (test set):

| Loss       | AUC-ROC Micro | AUC-ROC Macro | F1 Micro | F1 Macro |   MAP |
| ---------- | ------------: | ------------: | -------: | -------: | ----: |
| BCE (L0)   |         99.27 |         96.84 |    59.22 |    22.44 | 62.86 |
| Focal (L1) |         99.31 |         97.05 |    59.03 |    23.92 | 62.70 |
| DB (L2)    |         99.21 |         96.68 |    58.67 |    23.39 | 62.19 |
| ASL (L3)   |         99.25 |         96.88 |    58.98 |    24.53 | 62.72 |
| APL (L3)   |         99.25 |         96.87 |    59.19 |    24.67 | 62.93 |
| RAL (L3)   |         99.17 |         96.56 |    59.55 |    24.52 | 63.22 |
| MFM (L4)   |         99.24 |         96.85 |    59.45 |    23.99 | 63.24 |
| COMIC (L4) |         99.22 |         96.75 |    59.51 |    25.13 | 63.11 |

Observations:

* Macro-F1 and MAP consistently improve over BCE with advanced losses.
* COMIC yields the highest Macro-F1 among all methods while preserving Micro-F1.
* RAL and MFM offer competitive trade-offs and strong MAP values.

### 5.2 Performance by frequency group

Reconstructed F1 scores (MIMIC-IV, test set), highlighting BCE vs COMIC:

| Loss  | F1 Micro (All / Head / Med / Tail) | F1 Macro (All / Head / Med / Tail) |
| ----- | ---------------------------------- | ---------------------------------- |
| BCE   | 59.23 / 63.79 / 42.43 / 22.85      | 22.44 / 50.69 / 32.49 / 12.98      |
| COMIC | 59.51 / 63.95 / 44.45 / 26.73      | 25.13 / 51.32 / 35.63 / 16.05      |

Key points:

* Head performance is preserved (similar Micro-F1 for head codes).
* Medium and tail codes see consistent improvements, especially tail Macro-F1 (from 12.98 to 16.05).
* COMIC achieves roughly 24 percent relative improvement in tail Macro-F1 over BCE.

### 5.3 Effect of threshold tuning (BCE vs COMIC)

Illustrative comparison of selected configurations:

BCE:

* Fixed 0.5: Macro-F1 22.43, Micro-F1 59.23
* Single Macro-opt threshold: Macro-F1 26.46, Micro-F1 52.12
* Per-group (Macro-opt): Macro-F1 26.56, Micro-F1 56.20

COMIC:

* Fixed 0.5: Macro-F1 25.13, Micro-F1 59.51
* Single Macro-opt threshold: Macro-F1 28.28, Micro-F1 55.58
* Per-group (Micro-opt): Macro-F1 27.45, Micro-F1 59.15

Takeaways:

* Advanced losses (such as COMIC) produce better-calibrated probabilities; they already improve Macro-F1 before tuning.
* Per-group thresholds offer a robust, interpretable setting with good Micro/Macro balance.
* Per-class thresholds are unstable and not recommended for large label spaces.

### 5.4 Extended results on MIMIC-IV-Full (26,096 labels)

Reconstructed summary for MIMIC-IV-Full:

| Method                             | F1 Macro | F1 Micro |  P@8 |
| ---------------------------------- | -------: | -------: | ---: |
| CAML                               |      4.1 |     52.7 | 64.4 |
| LAAT                               |      4.5 |     55.4 | 67.0 |
| JointLAAT                          |      5.7 |     55.9 | 66.9 |
| MSMN                               |      5.4 |     55.9 | 67.7 |
| CoRelation                         |      6.3 |     57.8 | 70.0 |
| PLM-ICD                            |      4.9 |     57.0 | 69.5 |
| PLM-ICD + COMIC                    |     17.2 |     58.7 | 70.4 |
| PLM-ICD + COMIC + per-group thres. |     20.6 |     58.8 | 70.4 |

Main message:

* COMIC dramatically improves Macro-F1 on extremely long-tailed label space.
* Per-group threshold calibration further boosts Macro-F1 by about 3.4 points, while keeping Micro-F1 and P@8 stable.
* Loss-level calibration and simple per-group thresholds act as complementary mechanisms.

## 6. Usage

Below are minimal usage examples. For full options, consult the configuration files and source code.

### 6.1 Training on MIMIC-IV (PLMICD with chosen loss)

Example: training PLMICD with the default configuration and a selected loss (for example, Focal, ASL, COMIC-style MFM).

```bash
python main.py experiment=mimiciv_icd10/plm_icd gpu=0
```

* experiment=mimiciv_icd10/plm_icd selects the configuration (model, data, loss, thresholds, optimizer).
* gpu=0 chooses the first GPU.

To switch losses inside PLMICD:

* Open the implementation file (for example, plmicd.py).
* Replace the default BCE loss with your chosen loss:

```python
class PLMICD(nn.Module):
    def __init__(self, num_classes: int, model_path: str, **kwargs):
        super().__init__()
        # encoder and decoder definition

        # default
        # self.loss = torch.nn.functional.binary_cross_entropy_with_logits

        # example: Focal loss
        # self.loss = FocalLoss()

        # example: Hill loss
        # self.loss = Hill()

        # example: ASL / APL / RAL / MFM
        # self.loss = AsymmetricLoss()
        # self.loss = AsymmetricPolynomialLoss()
        # self.loss = RobustAsymmetricLoss()
        # self.loss = MultiGrainedFocalLoss()

        # Make sure to pass cls_num_list / class_freq where required.
```

Losses that depend on class frequencies (for example, CB, DB, MFM, some COMIC components) require cls_num_list or class_freq to be passed at model construction.

### 6.2 Training with COMIC framework (PLMICD2)

To use the full COMIC framework (RLC + MFM + HTB), instantiate PLMICD2 (for example, in plmicd2.py):

```python
class PLMICD2(nn.Module):
    def __init__(self, num_classes: int, model_path: str, cls_num_list, **kwargs):
        super().__init__()
        # encoder and multiple experts (head, tail, balanced)

        self.rlc = ReflectiveLabelCorrectorLoss(...)
        self.mfm = MultiGrainedFocalLoss(...)
        self.htb = HeadTailBalancerLoss(PFM=self.mfm, ...)

    def _composite_loss(self, head_logits, tail_logits, bal_logits, labels):
        loss_r = self.rlc(bal_logits, labels)
        loss_m = self.mfm(bal_logits, labels)
        loss_b = self.htb(head_logits, tail_logits, bal_logits, labels)
        return self.lambda_r * loss_r + self.lambda_m * loss_m + self.lambda_b * loss_b
```

Then, in your training script, use PLMICD2 instead of PLMICD and keep the rest of the training loop unchanged.

### 6.3 Evaluation and inference

To run evaluation only (no further training), use a pretrained checkpoint:

```bash
python main.py experiment=mimiciv_icd10/plm_icd gpu=0 \
    load_model=/path/to/your/model.ckpt trainer.epochs=0
```

* load_model points to your checkpoint (for example, from a previous training run).
* trainer.epochs=0 disables further training and runs evaluation on the test set.

Note: This repository does not include pretrained weights; you must supply your own .ckpt file.

## 7. License

This project is released under the Apache 2.0 License.
For details, see the LICENSE file in this repository.

