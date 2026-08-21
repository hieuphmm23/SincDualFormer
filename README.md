# SincDualFormer

Official implementation of **SincDualFormer** and **SR-BandMix** for Motor Imagery EEG decoding.

> **SincDualFormer: A Dual-Scale Sinc-Filterbank Transformer Model with SR-BandMix Augmentation for Motor Imagery BCI**
> IEEE Journal of Biomedical and Health Informatics (J-BHI), 2026.

---
Equal Contribution: Hieu M. Pham and Trung M. Pham contributed equally to this work.

## 🏆 SOTA Comparison

Comparison with representative MI-EEG decoding methods on **BCI Competition IV-2a (Dataset I)** and **BCI Competition IV-2b (Dataset II)**.

For a fair architectural comparison, all methods are evaluated using the same **SR-BandMix** augmentation protocol. Results are averaged over five predefined random seeds: `0, 1, 2, 3, 4`.

| Method                    | IV-2a Accuracy (%) ↑ |        IV-2a Kappa ↑ | IV-2b Accuracy (%) ↑ |        IV-2b Kappa ↑ |
| :------------------------ | -------------------: | -------------------: | -------------------: | -------------------: |
| Deep ConvNet              |        78.34 ± 14.55 |        0.711 ± 0.194 |         85.44 ± 8.91 |        0.709 ± 0.178 |
| EEGNet                    |        78.38 ± 12.77 |        0.712 ± 0.170 |         87.39 ± 8.98 |        0.748 ± 0.180 |
| SHNN                      |        71.58 ± 11.56 |        0.621 ± 0.154 |        80.54 ± 12.38 |        0.611 ± 0.248 |
| CTNet                     |  <u>79.37 ± 9.50</u> | <u>0.725 ± 0.125</u> |  <u>87.44 ± 8.56</u> | <u>0.749 ± 0.171</u> |
| ADFCNN                    |        77.21 ± 12.57 |        0.696 ± 0.168 |         86.38 ± 8.90 |        0.727 ± 0.178 |
| EEG-DG                    |        73.97 ± 13.98 |        0.653 ± 0.186 |        84.43 ± 11.81 |        0.689 ± 0.252 |
| SMANet                    |        75.39 ± 11.75 |        0.672 ± 0.148 |        82.99 ± 10.06 |        0.660 ± 0.190 |
| **SincDualFormer (Ours)** |     **82.14 ± 8.57** |    **0.762 ± 0.114** |     **87.93 ± 9.10** |    **0.759 ± 0.182** |

**SincDualFormer achieves the best overall Accuracy and Cohen's Kappa on both public benchmarks under the SR-BandMix setting.**

* **BCI Competition IV-2a:** `82.14%` Accuracy, `0.762` Kappa
* **BCI Competition IV-2b:** `87.93%` Accuracy, `0.759` Kappa

**Bold** indicates the best result and <u>underline</u> indicates the second-best result.

> The complete subject-wise results, standard deviations over seeds, and statistical significance tests are reported in the paper.

---

## Repository Layout

```text
SincDualFormer/
├── config.py          # Dataset, experiment, paths, subjects, and seeds
├── run.py             # Unified experiment entry point
├── augmentation.py    # SR-BandMix and augmentation variants
├── train_2a.py        # BCI Competition IV-2a pipeline
├── train_2b.py        # BCI Competition IV-2b pipeline
├── requirements.txt   # Python dependencies
└── README.md
```

`train_2a.py` and `train_2b.py` are intentionally kept separate because the reported training configurations and model settings are dataset-specific. Keeping separate pipelines makes the experimental protocol explicit and avoids hiding differences required for faithful reproduction.

---

## Installation

We recommend **Python 3.10+** with a CUDA-enabled PyTorch installation.

Install the required dependencies with:

```bash
python -m pip install -r requirements.txt
```

The experiments reported in the paper were conducted on an **NVIDIA H100 80GB GPU**.

---

## Quick Start

The main experiment settings can be configured directly in `config.py`.

For example:

```python
"dataset": "2a",
"experiment": "main",
"data_root": "/path/to/BCICIV_2a_gdf",
"label_root": "/path/to/true_labels_2a",
```

Then run:

```bash
python run.py
```

Configuration values can also be supplied directly from the command line:

```bash
python run.py --dataset 2b --experiment main \
  --data-root /path/to/BCICIV_2b_gdf \
  --label-root /path/to/true_labels_2b \
  --output-root results
```

---

## Smoke Test

To verify the installation and pipeline using a single subject and a single seed:

```bash
python run.py --dataset 2a --experiment main \
  --subjects 1 --seeds 0 \
  --data-root /path/to/BCICIV_2a_gdf \
  --label-root /path/to/true_labels_2a
```

Use:

```bash
--dry-run
```

to inspect the selected configuration without loading EEG data or starting training.

---

## Experiments

The repository provides a unified interface for reproducing the main experiments and ablation studies reported in the paper.

| Experiment           | Available Modes                                              |
| :------------------- | :----------------------------------------------------------- |
| `main`               | `SR_BANDMIX`                                                 |
| `augmentation`       | `NO_AUG`, `SR_ONLY`, `BANDMIX_ONLY`, `SR_BANDMIX`            |
| `srbandmix_variants` | `FINE_GRAINED`, `NON_MU_BETA`, `FULL_8_30`, `COARSE_MU_BETA` |
| `xie`                | `XIE_ONIGA`                                                  |
| `architecture`       | `FULL` and eight `WO_*` ablation modes                       |

### Main Experiment

Run SincDualFormer with the proposed SR-BandMix augmentation:

```bash
python run.py --dataset 2a --experiment main
python run.py --dataset 2b --experiment main
```

---

### Augmentation Ablation

Compare training without augmentation, SR only, BandMix only, and the complete SR-BandMix strategy:

```bash
python run.py --dataset 2a --experiment augmentation
python run.py --dataset 2b --experiment augmentation
```

A specific mode can be selected using `--modes`:

```bash
python run.py --dataset 2a --experiment augmentation --modes SR_ONLY
```

Available modes:

```text
NO_AUG
SR_ONLY
BANDMIX_ONLY
SR_BANDMIX
```

---

### SR-BandMix Frequency-Band Study

Run the complete SR-BandMix spectral-design comparison:

```bash
python run.py --dataset 2a --experiment srbandmix_variants
python run.py --dataset 2b --experiment srbandmix_variants
```

The proposed `FINE_GRAINED` configuration partitions the MI-relevant **8–30 Hz** region into five sub-bands:

```text
8–12 Hz
12–16 Hz
16–20 Hz
20–24 Hz
24–30 Hz
```

The available variants are:

| Mode             | Description                                             |
| :--------------- | :------------------------------------------------------ |
| `FINE_GRAINED`   | Proposed five-band SR-BandMix                           |
| `NON_MU_BETA`    | Spectral mixing outside the canonical μ/β configuration |
| `FULL_8_30`      | Mixing over the complete 8–30 Hz ROI                    |
| `COARSE_MU_BETA` | Coarse μ/β-band mixing                                  |

All four configurations are defined centrally in `SR_BANDMIX_VARIANTS` in `augmentation.py`. `run.py` only selects the requested configuration.

---

### Architecture Ablation

Run the complete architecture:

```bash
python run.py --dataset 2a --experiment architecture --modes FULL
```

or compare selected architectural ablations:

```bash
python run.py --dataset 2b --experiment architecture \
  --modes FULL,WO_SINCNET
```

The `WO_*` configurations correspond to the component-removal experiments reported in the architecture ablation study.

---

## Dataset Settings

| Setting              | BCI Competition IV-2a | BCI Competition IV-2b |
| :------------------- | --------------------: | --------------------: |
| Classes              |                     4 |                     2 |
| EEG channels         |                    22 |                     3 |
| Sampling rate        |                250 Hz |                250 Hz |
| Input time points    |                  1000 |                  1000 |
| Transformer depth    |                     1 |                     6 |
| Batch size           |                    32 |                    32 |
| Epochs               |                  1000 |                  1000 |
| Dropout              |                   0.5 |                   0.4 |
| Long temporal kernel |                   126 |                    96 |
| Weight decay         |                `1e-4` |                `1e-3` |
| Label smoothing      |                   0.0 |                  0.05 |
| Optimizer            |                  Adam |                 AdamW |
| Learning rate        |                `1e-3` |                `1e-3` |
| Seeds                |       `0, 1, 2, 3, 4` |       `0, 1, 2, 3, 4` |
| Augmentation factor  |                     3 |                     3 |

---

## SR-BandMix Configuration

The default SR-BandMix configuration used in the main experiments is:

| Parameter           | Value                                   |
| :------------------ | :-------------------------------------- |
| Temporal segments   | 8                                       |
| Augmentation factor | 3                                       |
| Spectral ROI        | 8–30 Hz                                 |
| Number of sub-bands | 5                                       |
| Sub-bands           | 8–12 / 12–16 / 16–20 / 20–24 / 24–30 Hz |
| Mixing probability  | 0.5 per sub-band                        |

SR-BandMix combines:

1. **Segmentation–Reconstruction (SR)** in the time domain using same-class training trials.
2. **Fine-grained BandMix** in the frequency domain over the MI-relevant 8–30 Hz region.

---

## Reproducibility

All reported experiments use the predefined seeds:

```text
0, 1, 2, 3, 4
```

The IV-2a `SR_ONLY` implementation retains its original NumPy implementation and random-number-generation order to preserve the reported experimental behavior.

Dataset-specific settings are kept explicit rather than automatically unified across IV-2a and IV-2b.

---

## Code Availability

The source code supporting this work is publicly available in this repository.
