
## Repository layout of Sincdualformer and Sr-Bandmix augmentation

```text
SincDualFormer_Reproduction_Repo/
├── config.py          # edit dataset, experiment, paths, subjects and seeds
├── run.py             # single entry point
├── augmentation.py    # shared augmentation mode names and operators
├── train_2a.py        # IV-2a model, loader and training protocol
├── train_2b.py        # IV-2b model, loader and training protocol
├── requirements.txt
└── README.md
```

`train_2a.py` and `train_2b.py` remain separate because the reported settings
are dataset-specific. Combining them would obscure differences that are needed
to reproduce the results.

## Install

Use Python 3.10+ and an H100-compatible CUDA/PyTorch installation.

```bash
python -m pip install -r requirements.txt
```

## Basic use

Edit `config.py`, especially:

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

The same settings can be supplied without editing the file:

```bash
python run.py --dataset 2b --experiment main \
  --data-root /path/to/BCICIV_2b_gdf \
  --label-root /path/to/true_labels_2b \
  --output-root results
```

Use one subject and one seed for a smoke test:

```bash
python run.py --dataset 2a --experiment main \
  --subjects 1 --seeds 0 \
  --data-root /path/to/BCICIV_2a_gdf \
  --label-root /path/to/true_labels_2a
```

Add `--dry-run` to inspect the selected setting without loading EEG data.

## Experiments

| Setting | Modes |
| --- | --- |
| `main` | `SR_BANDMIX` |
| `augmentation` IV-2a | `NO_AUG`, `SR_ONLY`, `BANDMIX_ONLY`, `SR_BANDMIX` |
| `augmentation` IV-2b | `NO_AUG`, `SR_ONLY`, `BANDMIX_ONLY`, `SR_BANDMIX` |
| `srbandmix_variants` | `FINE_GRAINED`, `NON_MU_BETA`, `FULL_8_30`, `COARSE_MU_BETA` |
| `xie` | `XIE_ONIGA` |
| `architecture` | `FULL` and the eight `WO_*` modes |

Select a subset with `--modes`, for example:

```bash
python run.py --dataset 2a --experiment augmentation --modes SR_ONLY
python run.py --dataset 2b --experiment architecture --modes FULL,WO_SINCNET
```

Run the complete SR-BandMix band-design comparison with:

```bash
python run.py --dataset 2a --experiment srbandmix_variants
python run.py --dataset 2b --experiment srbandmix_variants
```

`FINE_GRAINED` is the proposed five-band setting
`8–12/12–16/16–20/20–24/24–30 Hz` and uses the unchanged main SR-BandMix path.
The other three modes correspond to the controlled variant scripts supplied
for IV-2a and IV-2b. All four named settings are defined together in
`SR_BANDMIX_VARIANTS` near the top of `augmentation.py`; `run.py` only selects
one of those shared settings.

The IV-2a `SR_ONLY` name is used consistently in the settings, output folders,
logs, and saved configurations. Its original NumPy implementation and RNG order
are retained.

## Dataset settings

| Setting | IV-2a | IV-2b |
| --- | ---: | ---: |
| Channels/classes | 22 / 4 | 3 / 2 |
| Transformer depth | 1 | 6 |
| Batch size | 32 | 32 |
| Epochs | 1000 | 1000 |
| Dropout | 0.5 | 0.4 |
| Long temporal kernel | 126 | 96 |
| Weight decay | 1e-4 | 1e-3 |
| Label smoothing | 0.0 | 0.05 |
| Optimizer | Adam | AdamW |
| Seeds | 0,1,2,3,4 | 0,1,2,3,4 |
| Augmentation factor | 3 | 3 |
