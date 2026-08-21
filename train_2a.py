#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete script to train a dual‑scale Sinc‑based transformer for the BCI Competition IV‑2a
dataset.

* Implements a dual‑scale convolutional frontend with learnable Sinc filterbanks
  fixed to the Mu (8–13 Hz) and Beta (13–30 Hz) bands. Conv and Sinc
  outputs are processed by separate spatial paths first, then concatenated
  after spatial filtering and fused by a learnable 1x1 pointwise layer.
  The Sinc bands are initialized to cover the full Mu/Beta ranges more evenly.
* Performs Segmentation & Reconstruction (S&R) data augmentation combined with
  BandMix augmentation across frequency bands. This variant changes only the
  temporal Conv setting: KERNEL_SHORT=64, KERNEL_LONG=126,
  DIL_SHORT=1, DIL_LONG=3, and uses Adam + cosine scheduler + dynamic per-epoch validation for IV-2a.
* Applies Euclidean Alignment (EA) per subject when enabled.
* During training the script prints detailed progress including the current
  training/validation accuracy and loss, as well as snapshots of BandSE
  weights and the learned Sinc filterband ranges at each epoch that
  achieves a new validation minimum.
* Generates diagnostic plots (temporal spectra, Sinc responses, spatial
  weights, BandSE statistics, t‑SNE visualisation) into a `viz_epXXX`
  subdirectory whenever a new best epoch is found.
* After training, evaluates on the held‑out test set and reports the final
  accuracy, Cohen’s kappa and p‑value versus chance.
* Supports running multiple seeds in a loop; each seed stores its
  predictions (`y_true.npy`/`y_pred.npy`), metrics JSON, and a copy of the
  configuration used for that run.

Usage:
    python train_2a.py

This version uses causal-masked self-attention in the Transformer encoder,
runs subjects 1..9 with fixed seeds [0,1,2,3,4], and uses the fixed
train-only class-pool augmentation.
"""

import os
import sys
import math
import time
import glob
import json
import datetime
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Suppress warnings from external libraries
warnings.filterwarnings("ignore")

try:
    import mne
    import scipy.io as sio
except ImportError:
    # lazily install mne and scipy if not present
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mne", "scipy"])
    import mne
    import scipy.io as sio

# H100 / CUDA runtime hygiene: reduce allocator fragmentation for long multi-seed runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.manifold import TSNE
from scipy.stats import binomtest

from augmentation import generate_augmentation, normalize_aug_mode, parse_bandmix_bands

# ============== User Config ==============
# The subject index (1–9). Set via SUBJECT_ID for single-subject main(),
# or SUBJECTS="2" / SUBJECTS="1,2,3" for the multi-subject runner.
def _parse_int_list_env(name: str, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return list(default)
    return [int(x.strip()) for x in str(raw).replace(";", ",").split(",") if x.strip()]

SUBJECT = int(os.environ.get("SUBJECT_ID", "1"))
SUBJECTS = _parse_int_list_env("SUBJECTS", range(1, 10))  # default: Subject 1..9
FIXED_SEEDS = _parse_int_list_env("FIXED_SEEDS", [0,1,2,3,4])
# Data roots; adjust to point to your GDF files and true label MAT files.
ROOT_GDF = os.environ.get("ROOT_GDF_2A", "/mnt/data/hieupm/Hieudzso1/Dataset/BCICIV_2a_gdf")
ROOT_LABELS = os.environ.get("ROOT_LABELS_2A", "/mnt/data/hieupm/Hieudzso1/Dataset/true_labels_2a")
AUG_MODE = normalize_aug_mode(os.environ.get("AUG_MODE", "SR_BANDMIX"))

# Clean output structure for this exact official-train-pool depth=1 ablation run.
OUTPUT_ROOT = os.environ.get(
    "OUTPUT_ROOT",
    os.path.join("sincdualformer", "srbandmix_a_depth_1_official_train_pool_architecture_ablation"),
)

# Architecture ablations. FULL is exactly the uploaded model.
# BRANCH_I  = Mu / long-kernel branch.
# BRANCH_II = Beta / short-kernel branch.
ALL_ABLATION_MODES = [
    "FULL",
    "WO_BRANCH_I",
    "WO_TRANSFORMER_ENCODER",
    "WO_BRANCH_II",
    "WO_SINCNET",
    "WO_TEMPORAL_CONV_LONG_SHORT",
    "WO_INCEPTION_TCN",
    "WO_POINTWISE_CONV_FUSION",
    "WO_BANDSE",
]


def _parse_ablation_modes_env():
    raw = os.environ.get("ABLATION_MODES", os.environ.get("ABLATION_MODE", ""))
    if raw is None or str(raw).strip() == "":
        return list(ALL_ABLATION_MODES)
    requested = [x.strip().upper() for x in str(raw).replace(";", ",").split(",") if x.strip()]
    unknown = [x for x in requested if x not in ALL_ABLATION_MODES]
    if unknown:
        raise ValueError(
            f"Unknown ablation mode(s): {unknown}. Allowed: {ALL_ABLATION_MODES}"
        )
    # Preserve requested order while removing duplicates.
    return list(dict.fromkeys(requested))


ABLATION_MODES = _parse_ablation_modes_env()

# Model/training hyperparameters
EPOCHS = 1000  # number of training epochs (500 recommended for final results)
BATCH_SIZE = 32
VALIDATE_RATIO = 0.20
HEADS = 2
EMB_DIM = 16
DEPTH = 1
# Fixed Transformer depth for the final 9-subject, 5-seed experiment.
# Intentionally locked to depth=1 so this script cannot accidentally run another depth.
DEPTH_LIST = [1]
EEG_F1_TOTAL = 8
EEG_D = 2
DROPOUT = 0.5
CAUSAL_ATTENTION = False  # True = causal mask; False = original full self-attention/no mask
N_AUG = 3
FLATTEN_SIZE = 320
KERNEL_SHORT = 64
KERNEL_LONG = 126  # Variant A: shorter Mu temporal kernel
DIL_SHORT = 1
DIL_LONG = 3     # Variant A: less sparse Mu temporal dilation
POOL1 = 8
POOL2 = 6
LR = 1e-3
# Adam + L2 weight decay. Default is conservative for SincDualFormer.
# The reported IV-2a setting uses the default value below.
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
# True = safer parameter groups: do not decay bias, BatchNorm/LayerNorm, or Sinc frequency params.
# False = apply Adam weight decay to every parameter.
WEIGHT_DECAY_SAFE_GROUPS = os.environ.get("WEIGHT_DECAY_SAFE_GROUPS", "1").lower() not in ("0", "false", "no")
LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", "0.0"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "1.0"))
COSINE_ETA_MIN_RATIO = float(os.environ.get("COSINE_ETA_MIN_RATIO", "0.05"))
BEST_BY = os.environ.get("BEST_BY", "val_loss").strip().lower()  # val_loss | val_acc_then_loss
SAMPLE_RATE = 250.0

# Sinc initialization strategy.
# Keep the architecture unchanged, but initialize the 4 Mu/Beta Sinc filters
# to cover the whole target bands instead of clustering at the lower edge.
# Mu  : 8–13 Hz  -> overlapping filters covering low/mid/high Mu.
# Beta: 13–30 Hz -> overlapping filters covering low/mid/high Beta.
SINC_INIT_MODE = "overlap_even_mu_beta"
SINC_INIT_BANDS_MU = ((8.0, 10.5), (9.0, 11.5), (10.0, 12.5), (10.5, 13.0))
SINC_INIT_BANDS_BETA = ((13.0, 18.0), (16.0, 21.0), (20.0, 25.0), (24.0, 30.0))

EA_APPLY = False
SEED = 0  # default seed; runner below uses FIXED_SEEDS = [0,1,2,3,4]

# H100 / CUDA performance knobs. These do not change the architecture.
# TF32 is usually beneficial on H100 for Conv/Linear/Attention throughput.
# Keep deterministic=True below for fair seed comparison; cache clearing is throttled
# because empty_cache() every epoch slows H100 runs without improving accuracy.
USE_TF32 = True
CLEAR_CACHE_EVERY = 50  # set 0 to disable; old code effectively used 1
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
PIN_MEMORY = torch.cuda.is_available()

# Epoch window (4 s → 1000 samples at 250 Hz)
TMIN, TMAX = 0.0, 3.996

# ============== Reproducibility ==============
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONHASHSEED"] = str(SEED)
cudnn.benchmark = False
cudnn.deterministic = True
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if USE_TF32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ============== Dataset utils ==============
def numberClassChannel(database_type: str = 'A') -> Tuple[int, int]:
    """Return the number of classes and channels for a given dataset type."""
    t = str(database_type).upper()
    if t == 'A':
        return 4, 22
    if t == 'B':
        return 2, 3
    raise ValueError(f"Unknown database_type={database_type}")


def set_global_seed(seed: int) -> None:
    """Set seeds across numpy, random, and torch for reproducibility."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    if USE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def _save_numpy_safe(path: str, arr: np.ndarray) -> None:
    """Save numpy array to disk, catching any exceptions."""
    try:
        np.save(path, arr)
    except Exception as e:
        print(f"[WARN] cannot save {path} -> {repr(e)}")


def save_run_artifacts(out_dir: str, seed: int, acc: float, kappa: float,
                       best_epoch: int, seconds: float, cfg: dict,
                       y_true=None, y_pred=None) -> None:
    """Write out predictions, metrics, and config for a run."""
    os.makedirs(out_dir, exist_ok=True)
    if y_true is not None and y_pred is not None:
        _save_numpy_safe(os.path.join(out_dir, "y_true.npy"), np.asarray(y_true))
        _save_numpy_safe(os.path.join(out_dir, "y_pred.npy"), np.asarray(y_pred))
    meta = dict(seed=int(seed), acc=float(acc), kappa=float(kappa),
                best_epoch=int(best_epoch), seconds=float(seconds))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _norm_ch_name(s: str) -> str:
    """Normalize EEG channel name for robust matching."""
    import re as _re
    return _re.sub(r"[^a-z0-9]", "", str(s).lower())


def _pick_22_eeg_indices_2a(raw):
    """Pick the 22 EEG channels of BCI Competition IV-2a.

    Original IV-2a GDF files usually contain 22 EEG channels followed by 3 EOG
    channels. MNE normally marks EOG correctly, but this function is defensive:
    it prefers non-EOG EEG picks and falls back to the first 22 non-EOG channel
    names if needed.
    """
    names = raw.info["ch_names"]
    eeg_picks = list(mne.pick_types(raw.info, eeg=True, eog=False, stim=False))
    non_eog_by_name = [i for i, n in enumerate(names) if "eog" not in str(n).lower()]

    if len(eeg_picks) >= 22:
        first22 = eeg_picks[:22]
        if all("eog" not in str(names[i]).lower() for i in first22):
            picks = list(first22)
        elif len(non_eog_by_name) >= 22:
            picks = list(non_eog_by_name[:22])
        else:
            picks = list(first22)
    elif len(non_eog_by_name) >= 22:
        picks = list(non_eog_by_name[:22])
    else:
        picks = list(range(min(22, len(names))))

    if len(picks) != 22:
        raise RuntimeError(f"[IV-2a] expected 22 EEG channels, got {len(picks)} from {names}")
    picked_names = [names[i] for i in picks]
    return picks, picked_names


def _read_raw_and_events_2a(gdf_path: str):
    """Read one IV-2a GDF file and return raw, events, event_id, and 22 EEG picks."""
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    picks, picked_names = _pick_22_eeg_indices_2a(raw)
    return raw, events, event_id, picks, picked_names


def _peek_events(tag: str, events: np.ndarray, event_id: dict) -> None:
    """Print a compact summary of events when debugging is needed."""
    # Keep logs clean by default. Uncomment if event debugging is needed.
    # try:
    #     from collections import Counter
    #     print(f"[{tag}] event_id keys:", list(event_id.keys())[:20])
    #     print(f"[{tag}] code counts:", dict(sorted(Counter(events[:, 2].tolist()).items())))
    # except Exception as e:
    #     print(f"[{tag}] peek error:", e)
    pass


def _lookup_event_id(event_id: dict, candidates):
    """Robustly look up an event code by candidate labels/tokens."""
    if not event_id:
        return None

    kv = {str(k).strip().lower(): v for k, v in event_id.items()}
    for c in candidates:
        c0 = str(c).strip().lower()
        if c0 in kv:
            return kv[c0]

    for c in candidates:
        c0 = str(c).strip().lower()
        for k, v in event_id.items():
            kk = str(k).strip().lower()
            if c0 in kk:
                return v
    return None


def _find_train_event_codes_2a(event_id: dict):
    """Find IV-2a class cue codes in label order 1..4.

    769 = left hand, 770 = right hand, 771 = foot, 772 = tongue.
    """
    left_id = _lookup_event_id(event_id, ["769", "t1", "left", "left hand", "class 1", "cue onset left"])
    right_id = _lookup_event_id(event_id, ["770", "t2", "right", "right hand", "class 2", "cue onset right"])
    foot_id = _lookup_event_id(event_id, ["771", "t3", "foot", "feet", "class 3", "cue onset foot"])
    tongue_id = _lookup_event_id(event_id, ["772", "t4", "tongue", "class 4", "cue onset tongue"])
    return left_id, right_id, foot_id, tongue_id


def _find_eval_event_code_2a(event_id: dict):
    """Find unknown/eval event code for IV-2a evaluation files."""
    return _lookup_event_id(event_id, ["783", "t0", "unknown", "cue unknown", "start of trial"])


def _epoch_by_ids(raw, events, ids, picks):
    """Epoch raw data based on selected event IDs."""
    ids = [i for i in ids if i is not None]
    sel = events[np.isin(events[:, 2], ids)] if ids else events
    if sel is None or sel.shape[0] == 0:
        sfreq = raw.info.get('sfreq', 250)
        n_times = int((TMAX - TMIN) * sfreq) + 1
        return np.empty((0, len(picks), n_times), dtype=np.float32), np.empty((0, 3), dtype=int)

    _, uniq_idx = np.unique(sel[:, 0], return_index=True)
    sel = sel[np.sort(uniq_idx)]

    epochs = mne.Epochs(
        raw, sel, None, TMIN, TMAX, picks=picks,
        baseline=None, preload=True,
        reject_by_annotation=False,
        event_repeated="merge", verbose=False
    )
    X = epochs.get_data(copy=True).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X, sel


def _labels_from_mat_2a(path: str, expect_n: int):
    """Load IV-2a labels from .mat file if available and matching expected length."""
    if (path is None) or (not os.path.exists(path)):
        return None

    mat = sio.loadmat(path)
    candidate_keys = ("classlabel", "labels", "label", "y", "true_y")
    for k in candidate_keys:
        if k in mat:
            y = np.asarray(mat[k]).reshape(-1).astype(np.int64)
            if len(y) == expect_n:
                return y

    # Fallback: accept first numeric vector matching expected trial count.
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        try:
            arr = np.asarray(v).reshape(-1)
            if len(arr) == expect_n and np.issubdtype(arr.dtype, np.number):
                return arr.astype(np.int64)
        except Exception:
            pass
    return None


def _derive_labels_from_event_codes_2a(sel_events, class_ids):
    """Derive 1..4 labels from IV-2a class cue events."""
    codes = sel_events[:, 2]
    y = np.zeros_like(codes, dtype=np.int64)
    for label, event_code in enumerate(class_ids, start=1):
        if event_code is not None:
            y[codes == event_code] = label
    if np.any(y == 0) and sel_events.shape[0] > 0:
        bad = np.unique(codes[y == 0]).tolist()
        raise RuntimeError(f"[IV-2a] Some trials are not class events: {bad}")
    return y


def _resolve_label_mat_2a(root_labels: str, subject: int, phase: str) -> Optional[str]:
    """Find label .mat for A01T/A01E style IV-2a files."""
    stem = f"A{subject:02d}{phase}"
    patterns = [f"{stem}.mat", f"{stem}.MAT", f"*{stem}.mat", f"*{stem}.MAT"]
    for pat in patterns:
        hits = glob.glob(os.path.join(root_labels, pat))
        if hits:
            hits.sort(key=lambda p: (os.path.basename(p).upper() != f"{stem}.MAT", len(os.path.basename(p))))
            return hits[0]
    return None


def extract_subject_2a(subject: int, root_gdf: str, root_labels: str):
    """Load one BCI Competition IV-2a subject.

    Expected files:
      - A01T.gdf ... A09T.gdf for training
      - A01E.gdf ... A09E.gdf for evaluation/test
    Labels returned are 1..4.
    """
    X_tr, y_tr, X_te, y_te = [], [], [], []

    for phase in ("T", "E"):
        is_train = (phase == "T")
        gdf_name = f"A{subject:02d}{phase}.gdf"
        gdf_path = os.path.join(root_gdf, gdf_name)
        if not os.path.exists(gdf_path):
            raise FileNotFoundError(f"Missing {gdf_path}")

        raw, events, event_id, picks, picked_names = _read_raw_and_events_2a(gdf_path)
        _peek_events(gdf_name, events, event_id)
        class_ids = _find_train_event_codes_2a(event_id)

        if is_train:
            wanted_ids = list(class_ids)
            X_s, sel_s = _epoch_by_ids(raw, events, wanted_ids, picks)
            y_s = _derive_labels_from_event_codes_2a(sel_s, class_ids)
        else:
            eval_id = _find_eval_event_code_2a(event_id)
            if eval_id is not None:
                wanted_ids = [eval_id]
            else:
                # Some converted/easy-label E files may already contain class events.
                wanted_ids = [i for i in class_ids if i is not None]
            X_s, sel_s = _epoch_by_ids(raw, events, wanted_ids, picks)

            mat_path = _resolve_label_mat_2a(root_labels, subject, phase)
            print(f"[LABEL {gdf_name}] file:", mat_path)
            y_s = _labels_from_mat_2a(mat_path, expect_n=X_s.shape[0])
            if y_s is None:
                # Fallback only works when E file has true class cue events, not unknown 783.
                try:
                    y_s = _derive_labels_from_event_codes_2a(sel_s, class_ids)
                except Exception as exc:
                    raise ValueError(
                        f"[IV-2a EVAL] missing/mismatched labels for {gdf_name}: "
                        f"epochs={X_s.shape[0]}, label_file={mat_path}"
                    ) from exc

        if len(y_s) != X_s.shape[0]:
            raise ValueError(f"{gdf_name}: epochs={X_s.shape[0]} but labels={len(y_s)}")

        if is_train:
            X_tr.append(X_s); y_tr.append(y_s)
        else:
            X_te.append(X_s); y_te.append(y_s)

    X_tr = np.concatenate(X_tr).astype(np.float32)
    y_tr = np.concatenate(y_tr).astype(np.int64)
    X_te = np.concatenate(X_te).astype(np.float32)
    y_te = np.concatenate(y_te).astype(np.int64)

    print(f"[IV-2a] Subject {subject} picked EEG channels ({len(picked_names)}): {picked_names}")
    print(f"[IV-2a] y_train counts: {dict(zip(*np.unique(y_tr, return_counts=True)))}")
    print(f"[IV-2a] y_test counts: {dict(zip(*np.unique(y_te, return_counts=True)))}")
    return X_tr, y_tr, X_te, y_te


# Backward-friendly alias
extract_subject = extract_subject_2a


# ============== Euclidean Alignment functions ==============
def euclidean_alignment_fit_transform(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N, C, T = X.shape
    Sigma_bar = np.zeros((C, C), dtype=np.float64)
    for n in range(N):
        Sigma_bar += (X[n] @ X[n].T) / float(T)
    w, v = np.linalg.eigh(Sigma_bar / float(N))
    w = np.clip(w, 1e-10, None)
    S = (v @ np.diag(1.0 / np.sqrt(w)) @ v.T).astype(np.float32)
    return np.einsum('ij,njt->nit', S, X, optimize=True).astype(np.float32), S


def euclidean_alignment_transform(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.einsum('ij,njt->nit', S, X, optimize=True).astype(np.float32)


# ============== Visualization & Analysis helpers ==============
import matplotlib.pyplot as plt
import scipy.signal as sg

def _fft_mag(kern: np.ndarray, fs: float, nfft: int = 2048):
    """Return frequency and magnitude of FFT of a kernel."""
    H = np.fft.rfft(kern, n=nfft)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return f, np.abs(H)


@torch.no_grad()
def plot_temporal_fft_adfcnn_style(conv2d: nn.Conv2d, fs: float, out_path: str, xlim=(0, 50)):
    """Plot normalised spectra of all kernels in a temporal convolution layer (ADFCNN style)."""
    W = conv2d.weight.detach().cpu().numpy()  # [F1, 1, 1, K]
    W = W[:, 0, 0, :]
    K = W.shape[1]
    nfft = 4096
    win = sg.windows.hann(K, sym=False)
    F_vals, A_vals, peaks = [], [], []
    for k in range(W.shape[0]):
        w = W[k] * win
        H = np.fft.rfft(w, n=nfft)
        f = np.fft.rfftfreq(nfft, d=1.0 / fs)
        a = (np.abs(H) ** 2) / fs
        a_max = a.max() if a.size else 0.0
        if a_max > 0:
            a = a / a_max
        F_vals.append(f); A_vals.append(a)
        mask = (f >= 4) & (f <= 45)
        pk = f[mask][np.argmax(a[mask])] if mask.any() else 0.0
        peaks.append(pk)
    order = np.argsort(np.asarray(peaks))
    plt.figure(figsize=(6, 4))
    for idx in order:
        plt.plot(F_vals[idx], A_vals[idx], alpha=0.9 if W.shape[0] <= 6 else 0.35)
    plt.axvspan(8, 13, alpha=0.12)
    plt.axvspan(13, 30, alpha=0.12)
    plt.xlim(*xlim); plt.ylim(0, 1.05)
    plt.xlabel("Frequency (Hz)"); plt.ylabel(r"$|H_k(f)|$ (norm.)")
    plt.title("Temporal Conv spectra (ADFCNN style)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.no_grad()
def plot_temporal_kernel_spectrum_adfcnn(
    conv2d: nn.Conv2d,
    fs: float,
    out_path: str,
    f_view: Tuple[float, float] = (0.0, 50.0),
    **kwargs,
):
    """Compatibility wrapper used by the training loop."""
    return plot_temporal_fft_adfcnn_style(conv2d, fs, out_path, xlim=f_view)


@torch.no_grad()
def plot_sinc_frequency_responses(sinc_fb_module, fs: float, out_path: str):
    """Plot the amplitude response of all learned Sinc filter bands."""
    sinc = sinc_fb_module
    low, high = sinc._get_bands()
    low = low.detach().cpu().numpy().astype(float)
    high = high.detach().cpu().numpy().astype(float)
    kernels = sinc._bandpass_kernels(
        torch.from_numpy(low).to(sinc.t.device, dtype=torch.float32),
        torch.from_numpy(high).to(sinc.t.device, dtype=torch.float32)
    ).detach().cpu().numpy()
    plt.figure(figsize=(6, 4))
    for i in range(kernels.shape[0]):
        f, mag = _fft_mag(kernels[i], fs, nfft=4096)
        plt.plot(f, mag, alpha=0.9 if kernels.shape[0] <= 6 else 0.35)
    plt.xlim(0, 50)
    plt.xlabel("Frequency (Hz)"); plt.ylabel("|H(f)|")
    plt.title("Sinc filter‑bank responses")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.no_grad()
def plot_temporal_conv_fft(conv2d: nn.Conv2d, fs: float, out_path: str, pick_idx=(0, 1, 2, 3)):
    """Plot the FFT magnitude for selected kernels of a temporal convolution."""
    W = conv2d.weight.detach().cpu().numpy()  # [F1, 1, 1, K]
    plt.figure(figsize=(6, 4))
    for i in pick_idx:
        if i >= W.shape[0]:
            break
        kern = W[i, 0, 0, :]
        f, mag = _fft_mag(kern, fs, nfft=4096)
        plt.plot(f, mag, label=f"kernel {i}")
    plt.axvspan(8, 13, alpha=0.1)
    plt.axvspan(13, 30, alpha=0.1)
    plt.xlim(0, 50)
    plt.xlabel("Frequency (Hz)"); plt.ylabel("|H(f)|")
    plt.title("Temporal Conv |FFT| (examples)")
    if len(pick_idx) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.no_grad()
def plot_spatial_weights_bar(dw_spatial: nn.Conv2d, out_path: str,
                             ch_names=('C3', 'Cz', 'C4'), pick_idx=(0, 1, 2, 3)):
    """Plot bar charts of depthwise spatial weights for selected filters."""
    W = dw_spatial.weight.detach().cpu().numpy()  # [F2,1,C,1]
    W = W[:, 0, :, 0]  # [F2, C]
    n = min(len(pick_idx), W.shape[0])
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 3))
    axs = np.atleast_1d(axs)
    for ax, i in zip(axs, pick_idx[:n]):
        if i >= W.shape[0]:
            break
        labels = list(ch_names) if len(ch_names) == W.shape[1] else [f"Ch{j + 1}" for j in range(W.shape[1])]
        ax.bar(labels, W[i])
        ax.axhline(0, lw=0.5, c='k')
        ax.set_title(f"Spatial filter {i}")
        ax.tick_params(axis='x', labelrotation=45)
    fig.suptitle("Depthwise‑spatial weights", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def plot_bandse_bars(w_mu_list, w_beta_list, out_path: str):
    """Plot bar charts of BandSE weights for MU and BETA branches."""
    w_mu = np.array(w_mu_list, dtype=float).reshape(-1)
    w_bt = np.array(w_beta_list, dtype=float).reshape(-1)
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].bar(np.arange(len(w_mu)), w_mu); axs[0].set_title("BandSE weights – MU")
    axs[1].bar(np.arange(len(w_bt)), w_bt); axs[1].set_title("BandSE weights – BETA")
    for ax in axs:
        ax.set_xlabel("band index"); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def bandse_stats(branch, loader, use_abs_energy: bool = True):
    """Compute mean/std/min/max and sparsity of BandSE weights for a branch over a dataset.

    For the separate-spatial frontend, Conv and Sinc have their own BandSE modules.
    We report the mean of Conv/Sinc gates to keep the old MU/BETA diagnostic format
    comparable. Detailed Conv/Sinc gates are still printed in quick snapshots.
    """
    W = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to('cuda', dtype=torch.float32, non_blocking=True) if torch.cuda.is_available() else xb.float()
            if hasattr(branch, "_bandse_weights_for_input"):
                w = branch._bandse_weights_for_input(xb, average=True)
            elif hasattr(branch, "_front_pre_se") and hasattr(branch, "band_se"):
                x_in = branch._front_pre_se(xb)
                if use_abs_energy:
                    s = x_in.pow(2).mean(dim=(2, 3))
                else:
                    s = x_in.mean(dim=(2, 3))
                w = branch.band_se.mlp(s)  # sigmoid inside mlp
            else:
                y_conv = branch.conv_t1(xb)
                y_fb = branch.fb(xb)
                x_in = y_conv + y_fb
                if use_abs_energy:
                    s = x_in.pow(2).mean(dim=(2, 3))
                else:
                    s = x_in.mean(dim=(2, 3))
                w = branch.band_se.mlp(s)
            W.append(w.detach().cpu().numpy())
    if not W:
        return {"mean": [], "std": [], "min": [], "max": [], "sparsity": 0.0}
    W = np.concatenate(W, axis=0)
    return {
        "mean": np.round(W.mean(0), 4).tolist(),
        "std": np.round(W.std(0, ddof=1), 4).tolist() if W.shape[0] > 1 else [0.0] * W.shape[1],
        "min": np.round(W.min(0), 4).tolist(),
        "max": np.round(W.max(0), 4).tolist(),
        "sparsity": float((W < 0.1).mean()),
    }


@torch.no_grad()
def plot_tsne_features(model: nn.Module, loader, out_path: str,
                       take_mean_token: bool = True, perplexity: int = 30,
                       label_names=None):
    """Plot t-SNE of encoder features for a dataset."""
    model.eval()
    Z, Y = [], []
    for xb, yb in loader:
        xb = xb.to('cuda', dtype=torch.float32, non_blocking=True) if torch.cuda.is_available() else xb.float()
        feats, _ = model(xb)
        if take_mean_token:
            z = feats.mean(dim=1).detach().cpu().numpy()
        else:
            z = feats[:, 0, :].detach().cpu().numpy()
        Z.append(z)
        # always move yb to CPU before converting to numpy
        Y.append(yb.detach().cpu().numpy())
    if not Z:
        return
    Z = np.concatenate(Z, axis=0)
    Y = np.concatenate(Y, axis=0)
    if Z.shape[0] < 5:
        return
    z2 = TSNE(n_components=2,
              perplexity=min(perplexity, max(5, Z.shape[0] // 3)),
              init='pca').fit_transform(Z)
    plt.figure(figsize=(5, 4))
    classes = sorted(set(Y.tolist()))
    for c in classes:
        m = (Y == c)
        if label_names is None:
            name = f"class {int(c)}"
        elif isinstance(label_names, (list, tuple)):
            name = label_names[int(c)] if int(c) < len(label_names) else f"class {int(c)}"
        else:
            name = label_names.get(int(c), f"class {int(c)}")
        plt.scatter(z2[m, 0], z2[m, 1], s=10, label=name, alpha=0.75)
    plt.legend()
    plt.title("t-SNE of encoder features")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============== Model definitions ==============
class BandSE(nn.Module):
    def __init__(self, n_bands: int, reduction: int = 4):
        super().__init__()
        mid = max(1, n_bands // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(n_bands, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, n_bands), nn.Sigmoid()
        )
    def forward(self, x):
        s = x.pow(2).mean(dim=(2, 3))
        w = self.mlp(s)
        return x * w[:, :, None, None]


class LearnableSincFB(nn.Module):
    def __init__(self, n_filters: int, kernel_size: int, sample_rate: float,
                 min_low_hz: float = 1.0, min_band_hz: float = 2.0,
                 max_high_hz: float = None,
                 fixed_range: Optional[Tuple[float, float]] = None,
                 init_bands: Optional[Tuple[Tuple[float, float], ...]] = None):
        super().__init__()
        self.n_filters = int(n_filters)
        self.kernel_size = int(kernel_size)
        self.fs = float(sample_rate)
        self.min_low_hz = float(min_low_hz)
        self.min_band_hz = float(min_band_hz)
        self.nyq = self.fs / 2.0
        self.max_high_hz = float(max_high_hz) if max_high_hz is not None else self.nyq - self.min_band_hz
        self.fixed_range = tuple(float(v) for v in fixed_range) if fixed_range is not None else None
        self.init_bands = init_bands

        n = torch.arange(self.kernel_size, dtype=torch.float32)
        t = (n - (self.kernel_size - 1) / 2.0) / self.fs
        self.register_buffer("t", t)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (self.kernel_size - 1))
        self.register_buffer("window", window)

        # Carefully initialize the Sinc bands.
        # Previous code used fixed 2-Hz bands whose Mu filters clustered around 8–11 Hz.
        # Here, explicit init_bands are used for Mu/Beta so the filters cover the full
        # intended range at epoch 0. The parameters remain learnable after initialization.
        f_low_init, band_init = self._make_initial_bands(init_bands)
        self.low_hz_ = nn.Parameter(f_low_init)
        self.band_hz_ = nn.Parameter(band_init)

    def _make_initial_bands(self, init_bands: Optional[Tuple[Tuple[float, float], ...]] = None):
        """Return low-frequency and bandwidth tensors for robust Sinc initialization."""
        if self.fixed_range is not None:
            fmin, fmax = self.fixed_range
        else:
            fmin, fmax = self.min_low_hz, self.max_high_hz

        # Preferred path: explicit hand-designed Mu/Beta bands from DualScalePatchEmbeddingCNN.
        if init_bands is not None:
            if len(init_bands) != self.n_filters:
                raise ValueError(
                    f"init_bands length ({len(init_bands)}) must match n_filters ({self.n_filters})."
                )
            lows = torch.tensor([float(lo) for lo, _ in init_bands], dtype=torch.float32)
            highs = torch.tensor([float(hi) for _, hi in init_bands], dtype=torch.float32)
        else:
            # Generic fallback for non-standard filter counts/ranges.
            # Use overlapping coverage when possible rather than forcing every band to min_band_hz.
            span = max(float(fmax - fmin), self.min_band_hz)
            if self.n_filters == 1:
                lows = torch.tensor([fmin], dtype=torch.float32)
                highs = torch.tensor([fmax], dtype=torch.float32)
            else:
                width = max(self.min_band_hz, span / max(1, self.n_filters - 1))
                width = min(width, span)
                lows = torch.linspace(fmin, fmax - width, steps=self.n_filters, dtype=torch.float32)
                highs = lows + width

        # Clamp safely inside the allowed range while preserving at least min_band_hz.
        lows = torch.clamp(lows, min=float(fmin), max=float(fmax - self.min_band_hz))
        highs = torch.maximum(highs, lows + self.min_band_hz)
        highs = torch.clamp(highs, max=float(fmax))
        band = (highs - lows).clamp(min=self.min_band_hz)
        return lows.contiguous(), band.contiguous()

    @staticmethod
    def _sinc(x):
        return torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.sin(math.pi * x) / (math.pi * x))
    def _bandpass_kernels(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return self._lp_kernel(high) - self._lp_kernel(low)
    def _get_bands(self):
        low = self.low_hz_
        band = torch.abs(self.band_hz_).clamp(min=self.min_band_hz)
        if self.fixed_range is not None:
            fmin, fmax = self.fixed_range
            low = torch.clamp(low, min=fmin, max=fmax - self.min_band_hz)
            band = torch.clamp(band, max=fmax - low)
            high = (low + band).clamp(max=fmax)
        else:
            low = torch.clamp(low, min=self.min_low_hz, max=self.max_high_hz - self.min_band_hz)
            high = (low + band).clamp(max=self.max_high_hz)
        return low, high
    def _lp_kernel(self, f: torch.Tensor) -> torch.Tensor:
        x = 2.0 * f.unsqueeze(1) * self.t.unsqueeze(0)
        h = 2.0 * f.unsqueeze(1) * self._sinc(x)
        h = h * self.window.unsqueeze(0)
        h = h / (h.abs().sum(dim=1, keepdim=True).clamp(min=1e-8))
        return h
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, one, C, T = x.shape
            assert one == 1
            x1 = x[:, 0]
        elif x.dim() == 3:
            B, C, T = x.shape
            x1 = x
        else:
            raise ValueError("Input must be (B,1,C,T) or (B,C,T)")
        low, high = self._get_bands()
        kernels = self._bandpass_kernels(low, high)
        weight = kernels.unsqueeze(1).to(x1.dtype)
        pad_left = self.kernel_size // 2
        pad_right = self.kernel_size - 1 - pad_left
        xbc = x1.reshape(B * C, 1, T)
        xbc = F.pad(xbc, (pad_left, pad_right))
        ybc = F.conv1d(xbc, weight=weight, stride=1, padding=0)
        y = ybc.view(B, C, self.n_filters, T).permute(0, 2, 1, 3).contiguous()
        return y

class SincFB2DAdapter(nn.Module):
    def __init__(self, n_filters: int, kernel_size: int, sample_rate: float,
                 fixed_range: Optional[Tuple[float, float]] = None,
                 init_bands: Optional[Tuple[Tuple[float, float], ...]] = None):
        super().__init__()
        self.fb = LearnableSincFB(
            n_filters,
            kernel_size,
            sample_rate,
            fixed_range=fixed_range,
            init_bands=init_bands,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fb(x)


class _InceptionTCN(nn.Module):
    def __init__(self, channels: int, ks=(8, 16, 32), dils=(1, 2, 3)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels, (1, k), padding='same', dilation=(1, d),
                      groups=channels, bias=False) for k, d in zip(ks, dils)
        ])
        self.pw = nn.Conv2d(channels * len(ks), channels, kernel_size=1, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(torch.cat([b(x) for b in self.branches], dim=1))


class _TemporalBranch(nn.Module):
    """One Mu/Beta temporal branch with separate Conv/Sinc spatial paths.

    Previous frontend:
        Conv(x), Sinc(x) -> concat -> 1x1 reduce to f1 -> BandSE -> shared spatial depthwise.

    New frontend:
        Conv(x) -> Conv-BandSE -> Conv spatial depthwise  -> f1*D
        Sinc(x) -> Sinc-BandSE -> Sinc spatial depthwise  -> f1*D
        concat after spatial -> 1x1 pointwise fuse -> f1*D

    This lets Conv and Sinc learn different C3/Cz/C4 spatial projections before the
    information is mixed, while keeping the output dimension unchanged for the
    downstream TCN/Transformer stack.
    """
    def __init__(self, f1: int, D: int, kernel_size: int, dil_temporal: int,
                 pooling_size1: int, pooling_size2: int, dropout_rate: float,
                 number_channel: int, sample_rate: float, fixed_range: Optional[Tuple[float, float]] = None,
                 init_bands: Optional[Tuple[Tuple[float, float], ...]] = None,
                 ablation_mode: str = "FULL"):
        super().__init__()
        self.ablation_mode = str(ablation_mode).upper()
        self.f1 = int(f1)
        self.D = int(D)
        self.f2 = self.f1 * self.D
        self.number_channel = int(number_channel)

        # Temporal extraction paths.
        self.conv_t1 = nn.Conv2d(1, self.f1, (1, kernel_size), padding='same', dilation=(1, dil_temporal), bias=False)
        self.fb = SincFB2DAdapter(self.f1, kernel_size, sample_rate, fixed_range=fixed_range, init_bands=init_bands)

        # Separate channel/band gates before spatial filtering. Keeping these separate
        # avoids forcing Conv/Sinc to share a single BandSE decision before they have
        # learned their own spatial projections. In WO_BANDSE, replace both learnable
        # gates with parameter-free identities; every other frontend operation remains
        # unchanged and tensor dimensions are preserved exactly.
        if self.ablation_mode == "WO_BANDSE":
            self.band_se_conv = nn.Identity()
            self.band_se_sinc = nn.Identity()
        else:
            self.band_se_conv = BandSE(n_bands=self.f1)
            self.band_se_sinc = BandSE(n_bands=self.f1)
        self.bn1_conv = nn.BatchNorm2d(self.f1)
        self.bn1_sinc = nn.BatchNorm2d(self.f1)

        # Separate spatial filters over C3/Cz/C4 for Conv and Sinc features.
        self.depthwise_spatial_conv = nn.Conv2d(
            self.f1, self.f2, (self.number_channel, 1), groups=self.f1, padding='valid', bias=False
        )
        self.depthwise_spatial_sinc = nn.Conv2d(
            self.f1, self.f2, (self.number_channel, 1), groups=self.f1, padding='valid', bias=False
        )

        # Fuse only after spatial filtering. This is the learnable matrix you wanted:
        # each output spatial channel can mix Conv-spatial and Sinc-spatial channels.
        # It is initialized as an average of matching channels to avoid doubling the
        # activation scale at epoch 0, but it remains fully learnable.
        self.spatial_pointwise_fuse = nn.Conv2d(2 * self.f2, self.f2, kernel_size=1, bias=False)
        self._init_spatial_pointwise_fuse_as_average()

        self.bn2 = nn.BatchNorm2d(self.f2)
        self.elu2 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, pooling_size1))
        self.drop1 = nn.Dropout(dropout_rate)
        self.itcn = _InceptionTCN(self.f2)
        self.bn3 = nn.BatchNorm2d(self.f2)
        self.elu3 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, pooling_size2))
        self.drop2 = nn.Dropout(dropout_rate)

    def _init_spatial_pointwise_fuse_as_average(self) -> None:
        """Initialize post-spatial Conv/Sinc fusion as 0.5*Conv + 0.5*Sinc per channel."""
        with torch.no_grad():
            self.spatial_pointwise_fuse.weight.zero_()
            for i in range(self.f2):
                self.spatial_pointwise_fuse.weight[i, i, 0, 0] = 0.5
                self.spatial_pointwise_fuse.weight[i, i + self.f2, 0, 0] = 0.5

    def _temporal_conv_sinc(self, x: torch.Tensor):
        # Dimension-preserving path ablations. FULL executes the original code path.
        if self.ablation_mode == "WO_SINCNET":
            y_conv = self.conv_t1(x)
            y_sinc = torch.zeros_like(y_conv)
        elif self.ablation_mode == "WO_TEMPORAL_CONV_LONG_SHORT":
            y_sinc = self.fb(x)
            y_conv = torch.zeros_like(y_sinc)
        else:
            y_conv = self.conv_t1(x)
            y_sinc = self.fb(x)
        return y_conv, y_sinc

    def _bandse_weights_for_input(self, x: torch.Tensor, average: bool = True) -> torch.Tensor:
        """Return effective Conv/Sinc BandSE weights for diagnostics.

        WO_BANDSE has no learned gate, so its effective multiplicative weights are
        exactly one. Reporting ones also keeps checkpoint diagnostics compatible.
        """
        y_conv, y_sinc = self._temporal_conv_sinc(x)
        if self.ablation_mode == "WO_BANDSE":
            w_conv = torch.ones(
                (y_conv.size(0), self.f1), device=y_conv.device, dtype=y_conv.dtype
            )
            w_sinc = torch.ones(
                (y_sinc.size(0), self.f1), device=y_sinc.device, dtype=y_sinc.dtype
            )
        else:
            w_conv = self.band_se_conv.mlp(y_conv.pow(2).mean(dim=(2, 3)))
            w_sinc = self.band_se_sinc.mlp(y_sinc.pow(2).mean(dim=(2, 3)))
        if average:
            return 0.5 * (w_conv + w_sinc)
        return torch.cat([w_conv, w_sinc], dim=1)

    def _spatial_features_pre_fuse(self, x: torch.Tensor):
        """Return Conv-spatial and Sinc-spatial tensors before pointwise fusion.

        In WO_BANDSE, band_se_conv and band_se_sinc are nn.Identity(), so the
        temporal features pass directly to the unchanged BatchNorm and spatial paths.
        """
        y_conv, y_sinc = self._temporal_conv_sinc(x)
        y_conv = self.bn1_conv(self.band_se_conv(y_conv))
        y_sinc = self.bn1_sinc(self.band_se_sinc(y_sinc))
        z_conv = self.depthwise_spatial_conv(y_conv)
        z_sinc = self.depthwise_spatial_sinc(y_sinc)
        return z_conv, z_sinc

    def _front_pre_fuse(self, x: torch.Tensor) -> torch.Tensor:
        z_conv, z_sinc = self._spatial_features_pre_fuse(x)
        return torch.cat([z_conv, z_sinc], dim=1)

    def _front(self, x: torch.Tensor) -> torch.Tensor:
        if self.ablation_mode == "WO_POINTWISE_CONV_FUSION":
            # Remove the learnable 1x1 fusion while preserving its f2-channel output.
            # A fixed matching-channel average is the parameter-free counterpart of
            # the original 0.5 Conv + 0.5 Sinc initialization.
            z_conv, z_sinc = self._spatial_features_pre_fuse(x)
            return 0.5 * (z_conv + z_sinc)
        return self.spatial_pointwise_fuse(self._front_pre_fuse(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from einops.layers.torch import Rearrange
        x = self.elu2(self.bn2(self._front(x)))
        x = self.drop1(self.pool1(x))
        if self.ablation_mode == "WO_INCEPTION_TCN":
            # Remove the entire Inception-TCN transform (including its post-TCN
            # BN/ELU), but keep the original second pooling and dropout stages.
            x = self.drop2(self.pool2(x))
        else:
            x = self.elu3(self.bn3(self.itcn(x)))
            x = self.drop2(self.pool2(x))
        x = Rearrange('b e (h) (w) -> b (h w) e')(x)
        return x


class DualScalePatchEmbeddingCNN(nn.Module):
    """Two parallel temporal branches (Mu and Beta bands) without cross attention."""
    def __init__(self, f1_total: int = 8, D: int = 2, kernel_short: int = 32, kernel_long: int = 96,
                 dil_short: int = 1, dil_long: int = 3, pooling_size1: int = 8, pooling_size2: int = 8,
                 dropout_rate: float = 0.3, number_channel: int = 22, emb_size: int = 16, sample_rate: float = 250.0,
                 ablation_mode: str = "FULL"):
        super().__init__()
        self.ablation_mode = str(ablation_mode).upper()
        f1_branch = max(1, f1_total // 2)
        MU_BAND = (8.0, 13.0)
        BETA_BAND = (13.0, 30.0)

        # Explicit full-band Sinc initialization for the current default f1_branch=4.
        # If f1_branch changes in future experiments, LearnableSincFB will fall back
        # to a generic overlapping initializer instead of silently using wrong lengths.
        mu_init_bands = SINC_INIT_BANDS_MU if (SINC_INIT_MODE == "overlap_even_mu_beta" and f1_branch == 4) else None
        beta_init_bands = SINC_INIT_BANDS_BETA if (SINC_INIT_MODE == "overlap_even_mu_beta" and f1_branch == 4) else None

        self.branch_mu = _TemporalBranch(f1_branch, D, kernel_long, dil_long,
                                         pooling_size1, pooling_size2, dropout_rate,
                                         number_channel, sample_rate, fixed_range=MU_BAND,
                                         init_bands=mu_init_bands,
                                         ablation_mode=self.ablation_mode)
        self.branch_beta = _TemporalBranch(f1_branch, D, kernel_short, dil_short,
                                           pooling_size1, pooling_size2, dropout_rate,
                                           number_channel, sample_rate, fixed_range=BETA_BAND,
                                           init_bands=beta_init_bands,
                                           ablation_mode=self.ablation_mode)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BRANCH_I = Mu/long branch; BRANCH_II = Beta/short branch.
        # The removed branch is replaced by zeros of identical shape, so the
        # Transformer embedding and classifier dimensions remain unchanged.
        if self.ablation_mode == "WO_BRANCH_I":
            xb = self.branch_beta(x)
            xa = torch.zeros_like(xb)
        elif self.ablation_mode == "WO_BRANCH_II":
            xa = self.branch_mu(x)
            xb = torch.zeros_like(xa)
        else:
            xa = self.branch_mu(x)
            xb = self.branch_beta(x)
        return torch.cat([xa, xb], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding: int, length: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.encoding[:, :x.shape[1], :].to(x.device))


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        drop_p: float = 0.5,
        fexp: int = 4,
        fdrop: float = 0.5,
        causal_attention: bool = True,
    ):
        super().__init__()
        self.causal_attention = bool(causal_attention)
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout=drop_p, batch_first=True)
        self.drop1 = nn.Dropout(drop_p)
        self.norm2 = nn.LayerNorm(emb_size)
        hidden = emb_size * fexp
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, hidden),
            nn.GELU(),
            nn.Dropout(fdrop),
            nn.Linear(hidden, emb_size),
        )
        self.drop2 = nn.Dropout(drop_p)
        self.register_buffer("_causal_mask_cache", torch.empty(0, 0, dtype=torch.bool), persistent=False)

    def _get_attn_mask(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Causal attention mask for tokens with shape [T, T].

        In torch.nn.MultiheadAttention, bool attn_mask=True means "do not attend".
        This upper-triangular mask blocks token t from seeing future tokens > t.
        """
        if not self.causal_attention:
            return None

        if (
            self._causal_mask_cache.numel() == 0
            or self._causal_mask_cache.size(0) != seq_len
            or self._causal_mask_cache.device != device
        ):
            self._causal_mask_cache = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1,
            )
        return self._causal_mask_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_mask = self._get_attn_mask(seq_len=h.size(1), device=h.device)
        attn_out = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Sequential):
    def __init__(self, heads: int, depth: int, emb_size: int, causal_attention: bool = True):
        super().__init__(
            *[
                TransformerEncoderBlock(
                    emb_size,
                    heads,
                    drop_p=DROPOUT,
                    causal_attention=causal_attention,
                )
                for _ in range(depth)
            ]
        )


class EEGTransformer(nn.Module):
    def __init__(self, heads=2, emb_size=16, depth=10, database_type='B',
                 f1_total=8, D=2, kernel_short=50, kernel_long=150,
                 dil_short=1, dil_long=4, pooling_size1=8, pooling_size2=8,
                 dropout_rate=0.5, number_channel=3, sample_rate=250.0, flatten_size=240,
                 causal_attention: bool = True, ablation_mode: str = "FULL"):
        super().__init__()
        self.ablation_mode = str(ablation_mode).upper()
        self.number_class, self.number_channel = numberClassChannel(database_type)
        self.emb_size = emb_size
        self.cnn = DualScalePatchEmbeddingCNN(f1_total, D, kernel_short, kernel_long, dil_short, dil_long,
                                              pooling_size1, pooling_size2, dropout_rate, self.number_channel,
                                              emb_size, sample_rate, ablation_mode=self.ablation_mode)
        self.position = PositionalEncoding(emb_size, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size, causal_attention=causal_attention)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(flatten_size, self.number_class))
    def forward(self, x: torch.Tensor):
        x_tok = self.cnn(x) * math.sqrt(self.emb_size)
        x_tok = self.position(x_tok)
        if self.ablation_mode == "WO_TRANSFORMER_ENCODER":
            feats = x_tok
        else:
            z = self.trans(x_tok)
            feats = x_tok + z
        logits = self.classifier(self.flatten(feats))
        return feats, logits



# ============== Optimizer helpers ==============
def build_adam_optimizer(model: nn.Module, lr: float, betas=(0.9, 0.999),
                          weight_decay: float = 0.0, safe_groups: bool = True):
    """Build Adam optimizer with safe weight-decay groups.

    This keeps the old IV-2a backbone unchanged. Weight decay is applied only to
    regular multi-dimensional weights. Biases, BatchNorm/LayerNorm parameters,
    and Sinc frequency parameters are excluded because decaying them can make
    Sinc filter edges drift in unstable ways.
    """
    weight_decay = float(weight_decay)
    if weight_decay <= 0.0:
        print(f"[Optimizer] Adam lr={lr:g}, betas={betas}, weight_decay=0", flush=True)
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas)

    if not safe_groups:
        print(
            f"[Optimizer] Adam lr={lr:g}, betas={betas}, weight_decay={weight_decay:g}, safe_groups=False",
            flush=True,
        )
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        is_no_decay = (
            p.ndim <= 1
            or name.endswith('.bias')
            or 'norm' in lname
            or 'bn' in lname
            or 'low_hz_' in lname
            or 'band_hz_' in lname
            or 'low_logit_' in lname
            or 'band_logit_' in lname
        )
        if is_no_decay:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = []
    if decay_params:
        param_groups.append({'params': decay_params, 'weight_decay': weight_decay})
    if no_decay_params:
        param_groups.append({'params': no_decay_params, 'weight_decay': 0.0})

    print(
        f"[Optimizer] Adam lr={lr:g}, betas={betas}, weight_decay={weight_decay:g}, "
        f"safe_groups={safe_groups}, decay_tensors={len(decay_params)}, no_decay_tensors={len(no_decay_params)}",
        flush=True,
    )
    return torch.optim.Adam(param_groups, lr=lr, betas=betas)

# ============== Experiment wrapper ==============
class ExP:
    """Encapsulate training, validation and testing for one subject."""
    def __init__(self, nsub, result_name, epochs, number_aug, heads, emb_size, depth, dataset_type,
                 f1_total, D, kernel_short, kernel_long, dil_short, dil_long,
                 pooling_size1, pooling_size2, dropout_rate, flatten_size, validate_ratio,
                 learning_rate, batch_size, use_aug=True, seed=0, ablation_mode="FULL",
                 aug_mode=AUG_MODE):
        self.nSub = nsub
        self.ablation_mode = str(ablation_mode).upper()
        if self.ablation_mode not in ALL_ABLATION_MODES:
            raise ValueError(f"Unknown ablation_mode={self.ablation_mode}")
        self.result_name = result_name
        os.makedirs(self.result_name, exist_ok=True)
        self.n_epochs = epochs
        self.number_augmentation = int(number_aug)
        self.validate_ratio = float(validate_ratio)
        self.lr = learning_rate
        self.batch_size = int(batch_size)
        self.aug_mode = normalize_aug_mode(aug_mode)
        self.use_aug = bool(use_aug) and self.aug_mode != "NO_AUG"
        self.dataset_type = "A"
        self.number_seg = 8
        self.bandmix_bands = parse_bandmix_bands()
        self.seed = int(seed)
        self.b1, self.b2 = 0.5, 0.999
        self.sample_rate = SAMPLE_RATE
        self.criterion_cls = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).cuda()
        # Validation uses plain CE, same style as the strong IV-2b version.
        self.criterion_val = torch.nn.CrossEntropyLoss().cuda()
        self.number_class, self.number_channel = numberClassChannel(dataset_type)
        self.model = EEGTransformer(
            heads=heads, emb_size=emb_size, depth=depth, database_type=dataset_type,
            f1_total=f1_total, D=D, kernel_short=kernel_short, kernel_long=kernel_long,
            dil_short=dil_short, dil_long=dil_long, pooling_size1=pooling_size1,
            pooling_size2=pooling_size2, dropout_rate=dropout_rate, flatten_size=flatten_size,
            number_channel=self.number_channel,
            causal_attention=CAUSAL_ATTENTION,
            ablation_mode=self.ablation_mode,
        ).cuda()
        self.model_filename = os.path.join(self.result_name, f'model_{self.nSub}_state.pth')

    def get_source_data(self):
        raise NotImplementedError("Set at runtime.")

    def _build_aug_class_pools(self, img: np.ndarray, label0: np.ndarray) -> None:
        """
        Build train-only class pools for SR-BandMix.

        This fixes the mini-batch augmentation bug: augmentation no longer
        assumes every random mini-batch contains samples from every class.
        Each class is sampled from the subject's official training data only.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        label0 = np.asarray(label0, dtype=np.int64).reshape(-1)
        self.aug_class_pools = []

        for cls in range(int(self.number_class)):
            idx = np.where(label0 == cls)[0]
            if idx.size == 0:
                print(f"[WARN] No TRAIN samples for class {cls}; augmentation will skip this class.", flush=True)
                pool = torch.empty(
                    (0, int(self.number_channel), int(img.shape[-1])),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                # img shape: (N, 1, C, T) -> pool shape: (N_cls, C, T)
                pool = torch.from_numpy(img[idx, 0]).float().contiguous().to(device)
            self.aug_class_pools.append(pool)

    def interaug(self, timg, label):
        """
        Hybrid S&R + BandMix augmentation as used in IV-2a.

        Fixed version:
        - Uses train-only subject-level class pools when available.
        - Does not fail when the current mini-batch has no samples from a class.
        - Keeps balanced n_per_class augmentation whenever every class exists in
          the training pool.
        """
        # Keep the original in-file SR-BandMix path for the main/architecture
        # experiments. Other settings use the shared named operators.
        if self.aug_mode != "SR_BANDMIX" or os.environ.get("BANDMIX_BANDS", "").strip():
            return generate_augmentation(self, timg, label)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.from_numpy(timg) if not isinstance(timg, torch.Tensor) else timg
        y = torch.from_numpy(label) if not isinstance(label, torch.Tensor) else label
        X = X.to(device).float()
        y = y.to(device).view(-1).long()

        K = int(self.number_class)
        C, L = int(self.number_channel), int(X.size(-1))
        n_per_class = int(self.number_augmentation) * max(1, self.batch_size // K)

        if n_per_class <= 0:
            return X.new_zeros((0, 1, C, L)), y.new_zeros((0,), dtype=torch.long)

        fs = float(self.sample_rate)
        S = max(1, int(getattr(self, "number_seg", 8)))
        seg_len = L // S
        seg_bounds = [(s * seg_len, (s + 1) * seg_len if s < S - 1 else L) for s in range(S)]

        bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
        p_choose_A = 0.5

        total_aug = n_per_class * K
        aug_data = torch.empty((total_aug, 1, C, L), dtype=X.dtype, device=device)
        aug_label = torch.empty((total_aug,), dtype=torch.long, device=device)

        freqs = torch.fft.rfftfreq(n=L, d=1.0 / fs).to(device)
        band_masks = torch.stack(
            [(freqs >= float(lo)) & (freqs < float(hi)) for lo, hi in bands],
            dim=0,
        ).float()
        BAND = band_masks.size(0)

        def get_class_pool(cls: int) -> torch.Tensor:
            # Preferred path: train-only class pools built once at the start of train().
            if hasattr(self, "aug_class_pools") and self.aug_class_pools is not None:
                pool = self.aug_class_pools[cls]
                if pool is not None and pool.numel() > 0:
                    return pool.to(device=device, dtype=X.dtype, non_blocking=True)

            # Fallback: current mini-batch pool. This avoids crashing even if
            # _build_aug_class_pools() was not called.
            idx = (y == cls).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                return X.new_zeros((0, C, L))
            return X.index_select(0, idx)[:, 0]

        def sr_recompose(pool_x: torch.Tensor, out_n: int) -> torch.Tensor:
            Nc = int(pool_x.size(0))
            if Nc <= 0:
                raise ValueError("Cannot perform S&R augmentation from an empty class pool.")

            out = torch.empty((out_n, C, L), dtype=pool_x.dtype, device=pool_x.device)
            for (st, ed) in seg_bounds:
                ridx = torch.randint(0, Nc, (out_n,), device=pool_x.device)
                out[:, :, st:ed] = pool_x.index_select(0, ridx)[:, :, st:ed]
            return out

        wp = 0
        for cls in range(K):
            pool = get_class_pool(cls)
            if pool.size(0) == 0:
                print(f"[WARN] Class {cls} has empty augmentation pool; skipping.", flush=True)
                continue

            A_sr = sr_recompose(pool, n_per_class)
            B_sr = sr_recompose(pool, n_per_class)

            Af = torch.fft.rfft(A_sr.float(), dim=-1)
            Bf = torch.fft.rfft(B_sr.float(), dim=-1)

            S_mask = (torch.rand((n_per_class, BAND), device=device) < p_choose_A).float()
            mA = torch.matmul(S_mask, band_masks)
            mB = torch.matmul(1.0 - S_mask, band_masks)
            covered = (mA + mB).clamp(max=1.0)
            mRest = 1.0 - covered

            Af_mix = Af * mA.unsqueeze(1) + Bf * mB.unsqueeze(1) + Af * mRest.unsqueeze(1)
            Y = torch.fft.irfft(Af_mix, n=L, dim=-1).to(dtype=X.dtype)

            block = Y.unsqueeze(1)
            end = wp + n_per_class
            aug_data[wp:end], aug_label[wp:end] = block, cls
            wp = end

        if wp == 0:
            return X.new_zeros((0, 1, C, L)), y.new_zeros((0,), dtype=torch.long)

        perm = torch.randperm(wp, device=device)
        return aug_data[:wp][perm].float(), aug_label[:wp][perm].long()

    def train(self):
        """Train with dynamic per-epoch validation, matching the strong IV-2b recipe.

        Key difference from the previous fixed-val variant:
        - No permanent 80/20 holdout split.
        - Each epoch shuffles the full official training set.
        - The tail of each mini-batch is used as temporary validation for that epoch.
        - Across epochs, every official training trial can be used for optimization.
        """
        img, label, test_data, test_label = self.get_source_data()

        # Apply Euclidean Alignment if enabled. Fit only on official train data.
        if EA_APPLY and img.shape[0] > 0:
            X_tr = img[:, 0, :, :]
            X_te = test_data[:, 0, :, :]
            X_tr_aligned, S = euclidean_alignment_fit_transform(X_tr)
            img[:, 0, :, :] = X_tr_aligned
            test_data[:, 0, :, :] = euclidean_alignment_transform(X_te, S)

        # Convert labels to 0-based indices once.
        y0 = np.asarray(label - 1, dtype=np.int64).reshape(-1)
        y_test0 = np.asarray(test_label - 1, dtype=np.int64).reshape(-1)

        print(
            f"[Ablation] mode={self.ablation_mode}",
            flush=True,
        )
        print(
            f"[DynamicVal] full_train={len(y0)} temporary_val_ratio={self.validate_ratio:.2f} "
            f"train_counts={dict(zip(*np.unique(y0, return_counts=True)))}",
            flush=True,
        )

        # Build augmentation pools from the official training data only.
        # This keeps full-data exposure like the strong IV-2b version.
        self._build_aug_class_pools(img, y0)

        print(
            f"[Augmentation] mode={self.aug_mode} n_aug={self.number_augmentation} "
            f"segments={int(getattr(self, 'number_seg', 8))} "
            f"bands={self.bandmix_bands} "
            f"source=official_training_class_pool",
            flush=True,
        )

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(img), torch.from_numpy(y0))
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(y_test0))
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        self.optimizer = build_adam_optimizer(
            self.model,
            lr=self.lr,
            betas=(self.b1, self.b2),
            weight_decay=WEIGHT_DECAY,
            safe_groups=WEIGHT_DECAY_SAFE_GROUPS,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.n_epochs),
            eta_min=self.lr * COSINE_ETA_MIN_RATIO,
        )

        best_epoch = 0
        best_val_loss = float('inf')
        best_val_acc = -1.0
        last_train_acc = 0.0
        last_train_loss = 0.0

        for e in range(self.n_epochs):
            self.model.train()
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
            )

            val_data_list, val_label_list = [], []
            total_correct, total_seen, total_loss_sum = 0, 0, 0.0

            for xb, yb in train_loader:
                # Dynamic validation: each shuffled batch contributes a temporary
                # validation tail; the remaining samples are optimized this epoch.
                n_val = max(1, int(self.validate_ratio * xb.shape[0]))
                if xb.shape[0] - n_val < 1:
                    n_val = max(0, xb.shape[0] - 1)
                if n_val > 0:
                    val_data_list.append(xb[-n_val:].clone())
                    val_label_list.append(yb[-n_val:].clone())
                    tr_x, tr_y = xb[:-n_val], yb[:-n_val]
                else:
                    tr_x, tr_y = xb, yb

                tr_x = tr_x.to('cuda', dtype=torch.float32, non_blocking=True)
                tr_y = tr_y.to('cuda', dtype=torch.long, non_blocking=True)

                if self.use_aug and self.number_augmentation > 0:
                    # IV-2a SR_ONLY retains its original full official-pool
                    # NumPy implementation and does not consume a torch shuffle
                    # after concatenation. Other modes keep the original path.
                    if self.aug_mode == "SR_ONLY":
                        aug_x, aug_y = self.interaug(self.allData, self.allLabel)
                    else:
                        aug_x, aug_y = self.interaug(tr_x, tr_y)
                    if aug_x.numel() > 0:
                        tr_x = torch.cat([tr_x, aug_x], dim=0)
                        tr_y = torch.cat([tr_y, aug_y], dim=0)
                        if self.aug_mode != "SR_ONLY":
                            perm = torch.randperm(tr_x.size(0), device=tr_x.device)
                            tr_x, tr_y = tr_x[perm], tr_y[perm]

                _, logits = self.model(tr_x)
                loss = self.criterion_cls(logits, tr_y)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if GRAD_CLIP and GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRAD_CLIP)
                self.optimizer.step()

                with torch.no_grad():
                    pred = logits.argmax(1)
                    total_correct += int((pred == tr_y).sum().item())
                    total_seen += int(tr_y.numel())
                    total_loss_sum += float(loss.item()) * int(tr_y.numel())

            scheduler.step()
            last_train_acc = total_correct / max(1, total_seen)
            last_train_loss = total_loss_sum / max(1, total_seen)

            # Evaluate on this epoch's dynamic validation tails.
            self.model.eval()
            val_x = torch.cat(val_data_list, dim=0).to('cuda', dtype=torch.float32, non_blocking=True)
            val_y = torch.cat(val_label_list, dim=0).to('cuda', dtype=torch.long, non_blocking=True)
            outs = []
            with torch.no_grad():
                vloader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(val_x, val_y),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )
                for vx, _ in vloader:
                    _, vz = self.model(vx)
                    outs.append(vz)
            val_logits = torch.cat(outs, dim=0)
            vloss = self.criterion_val(val_logits, val_y)
            vacc = (val_logits.argmax(1) == val_y).float().mean().item()
            lr_now = float(self.optimizer.param_groups[0]['lr'])

            if BEST_BY == "val_acc_then_loss":
                is_best = (vacc > best_val_acc + 1e-12) or (
                    abs(vacc - best_val_acc) <= 1e-12 and vloss.item() < best_val_loss - 1e-12
                )
            else:
                is_best = vloss.item() < best_val_loss - 1e-12

            if is_best:
                best_val_acc = float(vacc)
                best_val_loss = float(vloss.item())
                best_epoch = int(e)
                torch.save(self.model.state_dict(), self.model_filename)

                # Lightweight diagnostics only; no heavy t-SNE/plots by default.
                with torch.no_grad():
                    xb_diag, _ = next(iter(self.test_dataloader))
                    xb_diag = xb_diag.to('cuda', dtype=torch.float32, non_blocking=True)

                    def _bandse_quick_snapshot(branch, x):
                        if hasattr(branch, "_bandse_weights_for_input"):
                            w_avg = branch._bandse_weights_for_input(x, average=True).mean(0)
                            w_pair = branch._bandse_weights_for_input(x, average=False).mean(0)
                            f1 = branch.f1
                            return (
                                np.round(w_avg.detach().cpu().numpy(), 4).tolist(),
                                np.round(w_pair[:f1].detach().cpu().numpy(), 4).tolist(),
                                np.round(w_pair[f1:].detach().cpu().numpy(), 4).tolist(),
                            )
                        y_conv = branch.conv_t1(x)
                        y_fb = branch.fb(x)
                        x_in = y_conv + y_fb
                        w = branch.band_se.mlp(x_in.pow(2).mean(dim=(2, 3))).mean(0)
                        w = np.round(w.detach().cpu().numpy(), 4).tolist()
                        return w, w, w

                    w_mu, w_mu_conv, w_mu_sinc = _bandse_quick_snapshot(self.model.cnn.branch_mu, xb_diag)
                    w_beta, w_beta_conv, w_beta_sinc = _bandse_quick_snapshot(self.model.cnn.branch_beta, xb_diag)
                    sinc_mu = self.model.cnn.branch_mu.fb.fb
                    sinc_beta = self.model.cnn.branch_beta.fb.fb
                    low_mu, high_mu = sinc_mu._get_bands()
                    low_bt, high_bt = sinc_beta._get_bands()
                    bands_mu = list(zip(np.round(low_mu.detach().cpu().numpy(), 2), np.round(high_mu.detach().cpu().numpy(), 2)))
                    bands_bt = list(zip(np.round(low_bt.detach().cpu().numpy(), 2), np.round(high_bt.detach().cpu().numpy(), 2)))
                    print(f"[Best@{e}] ablation={self.ablation_mode} val_acc={vacc:.6f} val_loss={vloss.item():.7f}")
                    print(f"[Best@{e}] BandSE MU  AVG : {w_mu}")
                    print(f"[Best@{e}] BandSE MU  CONV: {w_mu_conv}")
                    print(f"[Best@{e}] BandSE MU  SINC: {w_mu_sinc}")
                    print(f"[Best@{e}] BandSE BETA AVG : {w_beta}")
                    print(f"[Best@{e}] BandSE BETA CONV: {w_beta_conv}")
                    print(f"[Best@{e}] BandSE BETA SINC: {w_beta_sinc}")
                    print(f"[Best@{e}] Sinc bands MU  (low,high Hz): {bands_mu}")
                    print(f"[Best@{e}] Sinc bands BETA(low,high Hz): {bands_bt}")

            print(
                f"{self.nSub}_{e} train_acc:{last_train_acc:.4f} train_loss:{last_train_loss:.6f}\t"
                f"val_acc:{vacc:.6f} val_loss:{vloss.item():.7f} lr:{lr_now:.8f}",
                flush=True,
            )

            if CLEAR_CACHE_EVERY and ((e + 1) % CLEAR_CACHE_EVERY == 0):
                torch.cuda.empty_cache()

        # Evaluation on official test set.
        self.model.load_state_dict(torch.load(self.model_filename, map_location='cuda'))
        self.model.eval()
        outs = []
        with torch.no_grad():
            for tx, _ in self.test_dataloader:
                _, tz = self.model(tx.to('cuda', dtype=torch.float32, non_blocking=True))
                outs.append(tz)
        logits = torch.cat(outs, dim=0)
        y_pred = logits.argmax(1).cpu()
        y_true = test_dataset.tensors[1].cpu()
        test_acc = (y_pred == y_true).float().mean().item()
        kappa = cohen_kappa_score(y_true.numpy(), y_pred.numpy())
        n = y_true.numel()
        k = int((y_true.numpy() == y_pred.numpy()).sum())
        p0 = 1.0 / float(self.number_class)
        pval = binomtest(k, n, p=p0, alternative="greater").pvalue
        cm = confusion_matrix(y_true.numpy(), y_pred.numpy(), labels=list(range(self.number_class)))
        print("[TEST] y_true counts:", dict(zip(*np.unique(y_true.numpy(), return_counts=True))))
        print("[TEST] y_pred counts:", dict(zip(*np.unique(y_pred.numpy(), return_counts=True))))
        print("[TEST] confusion matrix rows=true cols=pred:")
        print(cm)
        print(
            f"AUG {'ON' if self.use_aug else 'OFF'} | Best epoch: {best_epoch} | "
            f"best_val_acc={best_val_acc:.4f} | best_val_loss={best_val_loss:.6f} | "
            f"Test acc: {test_acc:.4f} | Kappa: {kappa:.4f} | p(vs chance)={pval:.3g}"
        )
        return test_acc, y_true, y_pred, best_epoch


# ============== Convenience runner ==============
def run_with_arrays(X_train, y_train, X_test, y_test, **cfg):
    """Instantiate ExP and run training given arrays for a subject."""
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)
    if y_train.min() == 0:
        y_train += 1
    if y_test.min() == 0:
        y_test += 1
    exp = ExP(
        nsub=cfg['SUBJECT'], result_name=cfg['result_dir'], epochs=cfg['EPOCHS'],
        number_aug=cfg['N_AUG'], heads=cfg['HEADS'], emb_size=cfg['EMB_DIM'], depth=cfg['DEPTH'],
        dataset_type='A', f1_total=cfg['EEG_F1_TOTAL'], D=cfg['EEG_D'],
        kernel_short=cfg['KERNEL_SHORT'], kernel_long=cfg['KERNEL_LONG'],
        dil_short=cfg['DIL_SHORT'], dil_long=cfg['DIL_LONG'], pooling_size1=cfg['POOL1'], pooling_size2=cfg['POOL2'],
        dropout_rate=cfg['DROPOUT'], flatten_size=cfg['FLATTEN_SIZE'], validate_ratio=cfg['VALIDATE_RATIO'],
        learning_rate=cfg['LR'], batch_size=cfg['BATCH_SIZE'],
        use_aug=cfg.get('AUG_MODE', AUG_MODE) != 'NO_AUG',
        seed=cfg.get('SEED', 0), ablation_mode=cfg.get('ABLATION_MODE', 'FULL'),
        aug_mode=cfg.get('AUG_MODE', AUG_MODE)
    )
    def _get_source_data(self):
        allData = np.expand_dims(X_train.astype(np.float32), axis=1)
        testData = np.expand_dims(X_test.astype(np.float32), axis=1)
        mean, std = allData.mean(), allData.std() + 1e-8
        self.allData = (allData - mean) / std
        self.allLabel = y_train
        self.testData = (testData - mean) / std
        self.testLabel = y_test
        return self.allData, self.allLabel, self.testData, self.testLabel
    import types
    exp.get_source_data = types.MethodType(_get_source_data, exp)
    return exp.train()


# ============== Main logic ==============
def main():
    print(f"[IV‑2a] Loading Subject {SUBJECT} from: {ROOT_GDF}, labels: {ROOT_LABELS}")
    X_tr, y_tr, X_te, y_te = extract_subject_2a(SUBJECT, ROOT_GDF, ROOT_LABELS)
    print(f"[Subject {SUBJECT}] Train: {X_tr.shape}, Test: {X_te.shape}")
    acc, _, _, _ = run_with_arrays(
        X_tr, y_tr, X_te, y_te,
        SUBJECT=SUBJECT,
        result_dir=os.path.join(OUTPUT_ROOT, "full", f"subject_{SUBJECT}", f"seed_{SEED}"),
        EPOCHS=EPOCHS, N_AUG=N_AUG, HEADS=HEADS, EMB_DIM=EMB_DIM, DEPTH=DEPTH,
        EEG_F1_TOTAL=EEG_F1_TOTAL, EEG_D=EEG_D,
        KERNEL_SHORT=KERNEL_SHORT, KERNEL_LONG=KERNEL_LONG,
        DIL_SHORT=DIL_SHORT, DIL_LONG=DIL_LONG,
        POOL1=POOL1, POOL2=POOL2, DROPOUT=DROPOUT,
        FLATTEN_SIZE=FLATTEN_SIZE, VALIDATE_RATIO=VALIDATE_RATIO,
        LR=LR, BATCH_SIZE=BATCH_SIZE, SEED=SEED,
        ABLATION_MODE=ABLATION_MODES[0],
    )
    print("=" * 60)
    print(f"[Subject {SUBJECT}] Final Accuracy (IV‑2a): {acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    """
    Run architecture ablations for the exact uploaded IV-2a implementation.

    Unchanged across modes:
      - Dataset loading and normalization
      - Official training-class-pool SR-BandMix augmentation
      - Dynamic validation
      - Optimizer, scheduler, checkpoint criterion and evaluation
      - All model/training hyperparameters
      - Subjects 1..9, seeds [0,1,2,3,4], Transformer depth=1

    Ablation mapping:
      - BRANCH_I  = Mu / long-kernel branch
      - BRANCH_II = Beta / short-kernel branch
    """
    print(f"[IV-2a] Attention mode: {'causal-masked self-attention' if CAUSAL_ATTENTION else 'full self-attention / no causal mask'}")
    print(f"[IV-2a] Run subjects={SUBJECTS} with fixed seeds={FIXED_SEEDS}")
    print(f"[IV-2a] Fixed Transformer depth={DEPTH_LIST[0]}")
    print(f"[IV-2a] Augmentation mode={AUG_MODE}")
    print(f"[IV-2a] Ablation modes={ABLATION_MODES}")
    print(f"[IV-2a] BRANCH_I=MU/LONG | BRANCH_II=BETA/SHORT")
    print(f"[IV-2a] GDF root: {ROOT_GDF}")
    print(f"[IV-2a] Label root: {ROOT_LABELS}")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"[IV-2a] Output root: {OUTPUT_ROOT}")

    # Remove legacy cross-ablation CSV files from earlier versions so that only
    # the independent CSV files inside each mode folder remain visible.
    for legacy_csv_name in (
        "all_runs.csv",
        "per_subject_5seed_summary.csv",
        "overall_ablation_summary.csv",
    ):
        legacy_csv_path = os.path.join(OUTPUT_ROOT, legacy_csv_name)
        if os.path.isfile(legacy_csv_path):
            os.remove(legacy_csv_path)
            print(f"[CSV cleanup] Removed legacy combined file: {legacy_csv_path}", flush=True)

    all_rows = []
    t0_all = time.time()

    base_cfg_dump = {
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "VALIDATE_RATIO": VALIDATE_RATIO,
        "HEADS": HEADS,
        "EMB_DIM": EMB_DIM,
        "DEPTH_DEFAULT": DEPTH,
        "DEPTH_LIST": DEPTH_LIST,
        "EEG_F1_TOTAL": EEG_F1_TOTAL,
        "EEG_D": EEG_D,
        "DROPOUT": DROPOUT,
        "CAUSAL_ATTENTION": CAUSAL_ATTENTION,
        "N_AUG": N_AUG,
        "AUG_MODE": AUG_MODE,
        "BANDMIX_BANDS": parse_bandmix_bands(),
        "FLATTEN_SIZE": FLATTEN_SIZE,
        "KERNEL_SHORT": KERNEL_SHORT,
        "KERNEL_LONG": KERNEL_LONG,
        "DIL_SHORT": DIL_SHORT,
        "DIL_LONG": DIL_LONG,
        "POOL1": POOL1,
        "POOL2": POOL2,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "WEIGHT_DECAY_SAFE_GROUPS": WEIGHT_DECAY_SAFE_GROUPS,
        "LABEL_SMOOTHING": LABEL_SMOOTHING,
        "GRAD_CLIP": GRAD_CLIP,
        "COSINE_ETA_MIN_RATIO": COSINE_ETA_MIN_RATIO,
        "BEST_BY": BEST_BY,
        "OPTIMIZER": "Adam_weight_decay_safe_groups" if WEIGHT_DECAY_SAFE_GROUPS else "Adam_weight_decay_all_params",
        "EA_APPLY": EA_APPLY,
        "SAMPLE_RATE": SAMPLE_RATE,
        "ATTENTION": "causal_masked_self_attention" if CAUSAL_ATTENTION else "full_self_attention_no_mask",
        "FIXED_SEEDS": FIXED_SEEDS,
        "DATASET": "BCI_Competition_IV_2a",
        "FRONTEND_FUSION": "separate_conv_sinc_spatial_then_1x1_pointwise_fuse",
        "TEMPORAL_KERNEL_VARIANT": "KSHORT64_KLONG126_DSHORT1_DLONG3",
        "SPATIAL_POINTWISE_FUSE_INIT": "0.5_conv_plus_0.5_sinc_matching_channels",
        "SINC_INIT_MODE": SINC_INIT_MODE,
        "SINC_INIT_BANDS_MU": SINC_INIT_BANDS_MU,
        "SINC_INIT_BANDS_BETA": SINC_INIT_BANDS_BETA,
        "USE_TF32": USE_TF32,
        "CLEAR_CACHE_EVERY": CLEAR_CACHE_EVERY,
        "NUM_WORKERS": NUM_WORKERS,
        "PIN_MEMORY": PIN_MEMORY,
        "ALL_ABLATION_MODES": ALL_ABLATION_MODES,
        "SELECTED_ABLATION_MODES": ABLATION_MODES,
        "BRANCH_I": "MU_LONG_KERNEL_BRANCH",
        "BRANCH_II": "BETA_SHORT_KERNEL_BRANCH",
        "WO_BANDSE": "replace_conv_and_sinc_BandSE_gates_with_parameter_free_identity",
    }

    for subject in SUBJECTS:
        print("\n" + "=" * 100)
        print(f"[IV-2a] Loading Subject {subject} | depth=1 | seeds={FIXED_SEEDS}")
        print("=" * 100)

        X_tr, y_tr, X_te, y_te = extract_subject_2a(subject, ROOT_GDF, ROOT_LABELS)
        print(f"[Subject {subject}] Train: {X_tr.shape}, Test: {X_te.shape}")

        subject_rows = []

        for ablation_mode in ABLATION_MODES:
            mode_rows = []
            mode_slug = ablation_mode.lower()

            for depth in DEPTH_LIST:
                depth = int(depth)
                depth_rows = []

                print("\n" + "#" * 100)
                print(f"[IV-2a] Subject {subject} | ABLATION={ablation_mode} | DEPTH={depth}")
                print("#" * 100)

                for seed in FIXED_SEEDS:
                    print("\n" + "-" * 100)
                    print(f"[IV-2a] Subject {subject} | ABLATION={ablation_mode} | DEPTH={depth} | seed={seed}")
                    print("-" * 100)

                    # Reset the exact same seed before every mode/run for fair comparison.
                    set_global_seed(seed)

                    t0 = time.time()
                    result_dir = os.path.join(
                        OUTPUT_ROOT,
                        mode_slug,
                        f"subject_{subject}",
                        f"seed_{seed}",
                    )

                    acc, y_true, y_pred, best_ep = run_with_arrays(
                        X_tr, y_tr, X_te, y_te,
                        SUBJECT=subject,
                        result_dir=result_dir,
                        EPOCHS=EPOCHS,
                        N_AUG=N_AUG,
                        HEADS=HEADS,
                        EMB_DIM=EMB_DIM,
                        DEPTH=depth,
                        EEG_F1_TOTAL=EEG_F1_TOTAL,
                        EEG_D=EEG_D,
                        KERNEL_SHORT=KERNEL_SHORT,
                        KERNEL_LONG=KERNEL_LONG,
                        DIL_SHORT=DIL_SHORT,
                        DIL_LONG=DIL_LONG,
                        POOL1=POOL1,
                        POOL2=POOL2,
                        DROPOUT=DROPOUT,
                        FLATTEN_SIZE=FLATTEN_SIZE,
                        VALIDATE_RATIO=VALIDATE_RATIO,
                        LR=LR,
                        BATCH_SIZE=BATCH_SIZE,
                        SEED=seed,
                        ABLATION_MODE=ablation_mode,
                    )

                    dur = time.time() - t0
                    y_true_np = y_true.numpy()
                    y_pred_np = y_pred.numpy()
                    kap = cohen_kappa_score(y_true_np, y_pred_np)

                    cfg_dump = dict(base_cfg_dump)
                    cfg_dump["SUBJECT"] = int(subject)
                    cfg_dump["SEED"] = int(seed)
                    cfg_dump["DEPTH"] = int(depth)
                    cfg_dump["ABLATION_MODE"] = ablation_mode

                    save_run_artifacts(
                        out_dir=result_dir,
                        seed=seed,
                        acc=acc,
                        kappa=kap,
                        best_epoch=best_ep,
                        seconds=dur,
                        cfg=cfg_dump,
                        y_true=y_true_np,
                        y_pred=y_pred_np,
                    )

                    row = {
                        "ablation_mode": ablation_mode,
                        "subject": int(subject),
                        "depth": int(depth),
                        "seed": int(seed),
                        "acc": float(acc),
                        "kappa": float(kap),
                        "best_epoch": int(best_ep),
                        "seconds": float(dur),
                        "result_dir": result_dir,
                    }
                    all_rows.append(row)
                    subject_rows.append(row)
                    mode_rows.append(row)
                    depth_rows.append(row)

                    print(
                        f"[Subject {subject} | {ablation_mode} | depth={depth} | seed={seed}] "
                        f"Final Accuracy: {acc:.4f} | kappa={kap:.4f} | "
                        f"best_epoch={best_ep} | time={dur/60:.1f}m"
                    )

                    del acc, y_true, y_pred, best_ep
                    torch.cuda.empty_cache()

                df_depth_tmp = pd.DataFrame(depth_rows)
                print("\n" + "*" * 100)
                print(
                    f"[Subject {subject} | {ablation_mode} | depth={depth}] "
                    f"{len(FIXED_SEEDS)}-seed acc: "
                    f"{df_depth_tmp['acc'].mean():.4f} ± {df_depth_tmp['acc'].std(ddof=1):.4f}"
                )
                print(
                    f"[Subject {subject} | {ablation_mode} | depth={depth}] "
                    f"{len(FIXED_SEEDS)}-seed kappa: "
                    f"{df_depth_tmp['kappa'].mean():.4f} ± {df_depth_tmp['kappa'].std(ddof=1):.4f}"
                )
                print("*" * 100)

            df_mode_tmp = pd.DataFrame(mode_rows)
            print("\n" + "=" * 100)
            print(f"[Subject {subject}] ABLATION={ablation_mode} DONE")
            print(df_mode_tmp.groupby(["ablation_mode", "depth"], as_index=False).agg(
                acc_mean=("acc", "mean"),
                acc_std=("acc", "std"),
                kappa_mean=("kappa", "mean"),
                kappa_std=("kappa", "std"),
                seconds_total=("seconds", "sum"),
            ).to_string(index=False))
            print("=" * 100)

        df_sub_tmp = pd.DataFrame(subject_rows)
        print("\n" + "=" * 100)
        print(f"[Subject {subject}] ALL SELECTED ABLATIONS DONE")
        print(df_sub_tmp.groupby(["ablation_mode", "depth"], as_index=False).agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
            seconds_total=("seconds", "sum"),
        ).to_string(index=False))
        print("=" * 100)

        del X_tr, y_tr, X_te, y_te
        torch.cuda.empty_cache()

    df_all = pd.DataFrame(all_rows)

    if df_all.empty:
        raise RuntimeError("No completed runs were found; no CSV summaries can be generated.")

    # Save a completely separate set of CSV files inside each ablation folder.
    # No cross-ablation CSV is written to OUTPUT_ROOT.
    saved_csv_paths = []
    mode_order = {mode: i for i, mode in enumerate(ALL_ABLATION_MODES)}

    for ablation_mode in ABLATION_MODES:
        mode_slug = ablation_mode.lower()
        mode_output_root = os.path.join(OUTPUT_ROOT, mode_slug)
        os.makedirs(mode_output_root, exist_ok=True)

        df_mode_all = (
            df_all[df_all["ablation_mode"] == ablation_mode]
            .copy()
            .sort_values(["depth", "subject", "seed"])
            .reset_index(drop=True)
        )

        if df_mode_all.empty:
            print(f"[WARN] No completed rows for ablation mode {ablation_mode}; CSV export skipped.", flush=True)
            continue

        # 1) Every seed/run belonging only to this ablation.
        csv_mode_all = os.path.join(mode_output_root, "all_runs.csv")
        df_mode_all.to_csv(csv_mode_all, index=False)

        # 2) Subject-level summary over the fixed seeds, belonging only to this ablation.
        df_mode_subject = (
            df_mode_all
            .groupby(["ablation_mode", "depth", "subject"], as_index=False)
            .agg(
                acc_mean_5seed=("acc", "mean"),
                acc_std_5seed=("acc", "std"),
                acc_min_5seed=("acc", "min"),
                acc_max_5seed=("acc", "max"),
                kappa_mean_5seed=("kappa", "mean"),
                kappa_std_5seed=("kappa", "std"),
                best_epoch_mean_5seed=("best_epoch", "mean"),
                seconds_total_5seed=("seconds", "sum"),
            )
            .sort_values(["depth", "subject"])
            .reset_index(drop=True)
        )
        csv_mode_subject = os.path.join(mode_output_root, "per_subject_5seed_summary.csv")
        df_mode_subject.to_csv(csv_mode_subject, index=False)

        # 3) Overall subject-level and all-run statistics, belonging only to this ablation.
        df_mode_overall = (
            df_mode_subject
            .groupby(["ablation_mode", "depth"], as_index=False)
            .agg(
                n_subjects=("subject", "nunique"),
                acc_mean=("acc_mean_5seed", "mean"),
                acc_std=("acc_mean_5seed", "std"),
                kappa_mean=("kappa_mean_5seed", "mean"),
                kappa_std=("kappa_mean_5seed", "std"),
                seconds_total=("seconds_total_5seed", "sum"),
            )
        )

        df_mode_run_summary = (
            df_mode_all
            .groupby(["ablation_mode", "depth"], as_index=False)
            .agg(
                n_runs=("acc", "size"),
                acc_mean_all_runs=("acc", "mean"),
                acc_std_all_runs=("acc", "std"),
                kappa_mean_all_runs=("kappa", "mean"),
                kappa_std_all_runs=("kappa", "std"),
            )
        )
        df_mode_overall = (
            df_mode_overall
            .merge(df_mode_run_summary, on=["ablation_mode", "depth"], how="left")
            .sort_values(["depth"])
            .reset_index(drop=True)
        )
        csv_mode_overall = os.path.join(mode_output_root, "overall_ablation_summary.csv")
        df_mode_overall.to_csv(csv_mode_overall, index=False)

        saved_csv_paths.append(
            (ablation_mode, csv_mode_all, csv_mode_subject, csv_mode_overall)
        )

        print("\n" + "-" * 100)
        print(f"[{ablation_mode}] Per-subject 5-seed summary")
        print(df_mode_subject.to_string(index=False))
        print(f"\n[{ablation_mode}] Overall summary")
        print(df_mode_overall.to_string(index=False))
        print("-" * 100)

    print("\n" + "=" * 100)
    print("[IV-2a] ARCHITECTURE ABLATION EXPERIMENT DONE")
    print("[CSV export] Each ablation has its own independent CSV files; no combined CSV was created.")
    for ablation_mode, csv_mode_all, csv_mode_subject, csv_mode_overall in saved_csv_paths:
        print(f"\n[{ablation_mode}]")
        print(f"  all runs:             {csv_mode_all}")
        print(f"  per-subject summary:  {csv_mode_subject}")
        print(f"  overall summary:      {csv_mode_overall}")
    print(f"\nTotal wall time: {(time.time() - t0_all) / 60:.1f}m")
    print("=" * 100)
