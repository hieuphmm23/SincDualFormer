"""Augmentation operators shared by IV-2a and IV-2b.

Mode names are consistent across datasets:
    NO_AUG, SR_ONLY, BANDMIX_ONLY, SR_BANDMIX, XIE_ONIGA

SR_BANDMIX has four visible band settings in ``SR_BANDMIX_VARIANTS``:
    FINE_GRAINED, NON_MU_BETA, FULL_8_30, COARSE_MU_BETA

The IV-2a SR_ONLY branch intentionally keeps the NumPy implementation and RNG
order of the original standalone SR-only experiment. Other modes use the
official-training class pools stored by the training script.
"""

from __future__ import annotations

import os

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Lets run.py inspect settings with --dry-run.
    torch = None


VALID_AUG_MODES = {
    "NO_AUG",
    "SR_ONLY",
    "BANDMIX_ONLY",
    "SR_BANDMIX",
    "XIE_ONIGA",
}

FINE_GRAINED_BANDS = (
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 30.0),
)

# SR-BandMix comparison settings supplied for both IV-2a and IV-2b.
# The operator and all training settings stay fixed; only these bands change.
# FINE_GRAINED is the proposed/main setting. The other three entries are the
# controlled variants from the supplied SR-BandMix variant scripts.
SR_BANDMIX_VARIANTS = {
    "FINE_GRAINED": FINE_GRAINED_BANDS,
    "NON_MU_BETA": ((4.0, 8.0), (30.0, 45.0)),
    "FULL_8_30": ((8.0, 30.0),),
    "COARSE_MU_BETA": ((8.0, 13.0), (13.0, 30.0)),
}


def get_sr_bandmix_variant(variant: str):
    """Return one named SR-BandMix band setting."""
    name = str(variant).strip().upper().replace("-", "_").replace(" ", "_")
    if name not in SR_BANDMIX_VARIANTS:
        raise ValueError(
            f"Unknown SR-BandMix variant {name!r}; "
            f"allowed={list(SR_BANDMIX_VARIANTS)}"
        )
    return name, SR_BANDMIX_VARIANTS[name]


def bands_to_env(bands) -> str:
    """Convert a band tuple to the compact runner format, for example ``8:30``."""
    return ",".join(f"{low:g}:{high:g}" for low, high in bands)


def normalize_aug_mode(mode: str) -> str:
    mode = str(mode).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "NONE": "NO_AUG",
        "NOAUG": "NO_AUG",
        "SR": "SR_ONLY",
        "S&R": "SR_ONLY",
        "BANDMIX": "BANDMIX_ONLY",
        "XIE": "XIE_ONIGA",
        "XIE_ONIGA_STYLE": "XIE_ONIGA",
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_AUG_MODES:
        raise ValueError(f"Unknown augmentation mode {mode!r}; allowed={sorted(VALID_AUG_MODES)}")
    return mode


def parse_bandmix_bands(raw: str | None = None):
    """Parse `8:12,12:16,...`; return the original fine-grained bands by default."""
    raw = os.environ.get("BANDMIX_BANDS", "") if raw is None else str(raw)
    if not raw.strip():
        return FINE_GRAINED_BANDS
    bands = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        lo, hi = item.split(":", 1)
        lo, hi = float(lo), float(hi)
        if not lo < hi:
            raise ValueError(f"Invalid band {item!r}: low edge must be below high edge")
        bands.append((lo, hi))
    if not bands:
        raise ValueError("BANDMIX_BANDS did not contain any valid band")
    return tuple(bands)


def _empty(exp, x, y):
    channels, length = int(exp.number_channel), int(x.shape[-1])
    return (
        x.new_zeros((0, 1, channels, length)),
        y.new_zeros((0,), dtype=torch.long),
    )


def _sr_only_iv2a_numpy(exp, timg, label):
    """Original IV-2a SR-only operator, including its NumPy RNG consumption."""
    timg = np.asarray(timg)
    label = np.asarray(label).reshape(-1)
    if timg.shape[-1] != 1000:
        raise ValueError(f"IV-2a SR_ONLY expects 1000 samples, got {timg.shape[-1]}")

    aug_data = []
    aug_label = []
    number_records = exp.number_augmentation * int(exp.batch_size / exp.number_class)
    number_segments = int(getattr(exp, "number_seg", 8))
    segment_length = 1000 // number_segments

    for class_index in range(exp.number_class):
        class_rows = np.where(label == class_index + 1)
        class_data = timg[class_rows]
        class_label = label[class_rows]
        if class_data.shape[0] == 0:
            raise RuntimeError(
                f"SR_ONLY found no official-training samples for class {class_index + 1}."
            )

        synthetic = np.zeros(
            (number_records, 1, exp.number_channel, 1000),
            dtype=np.float64,
        )
        for row in range(number_records):
            for segment in range(number_segments):
                # Keep this draw inside the inner loop: it matches the original
                # SR-only implementation's NumPy RNG order exactly.
                source_indices = np.random.randint(
                    0, class_data.shape[0], number_segments
                )
                start = segment * segment_length
                end = (segment + 1) * segment_length
                synthetic[row, :, :, start:end] = class_data[
                    source_indices[segment], :, :, start:end
                ]
        aug_data.append(synthetic)
        aug_label.append(class_label[:number_records])

    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    order = np.random.permutation(len(aug_data))
    aug_data = aug_data[order, :, :]
    aug_label = aug_label[order]
    return (
        torch.from_numpy(aug_data).cuda().float(),
        torch.from_numpy(aug_label - 1).cuda().long(),
    )


def _class_pool(exp, x, y, class_index, device):
    pools = getattr(exp, "aug_class_pools", None)
    if pools is not None:
        pool = pools[class_index]
        if pool is not None and pool.numel() > 0:
            return pool.to(device=device, dtype=x.dtype, non_blocking=True)
    indices = (y == class_index).nonzero(as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        return x.new_zeros((0, int(exp.number_channel), int(x.shape[-1])))
    return x.index_select(0, indices)[:, 0]


def _sr_bandmix_torch(exp, timg, label, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(timg) if not isinstance(timg, torch.Tensor) else timg
    y = torch.from_numpy(label) if not isinstance(label, torch.Tensor) else label
    x = x.to(device).float()
    y = y.to(device).view(-1).long()

    classes = int(exp.number_class)
    channels, length = int(exp.number_channel), int(x.shape[-1])
    per_class = int(exp.number_augmentation) * max(1, exp.batch_size // classes)
    if per_class <= 0:
        return _empty(exp, x, y)

    segments = max(1, int(getattr(exp, "number_seg", 8)))
    segment_length = length // segments
    bounds = [
        (s * segment_length, (s + 1) * segment_length if s < segments - 1 else length)
        for s in range(segments)
    ]

    bands = tuple(getattr(exp, "bandmix_bands", parse_bandmix_bands()))
    sample_rate = float(exp.sample_rate)
    frequencies = torch.fft.rfftfreq(n=length, d=1.0 / sample_rate).to(device)
    band_masks = None
    number_bands = 0
    if mode in {"BANDMIX_ONLY", "SR_BANDMIX"}:
        band_masks = torch.stack(
            [
                (frequencies >= float(low)) & (frequencies < float(high))
                for low, high in bands
            ],
            dim=0,
        ).float()
        number_bands = int(band_masks.shape[0])

    total = per_class * classes
    aug_data = torch.empty((total, 1, channels, length), dtype=x.dtype, device=device)
    aug_label = torch.empty((total,), dtype=torch.long, device=device)

    def sr_recompose(pool):
        count = int(pool.shape[0])
        if count <= 0:
            raise ValueError("Cannot perform SR_ONLY from an empty class pool")
        result = torch.empty(
            (per_class, channels, length), dtype=pool.dtype, device=pool.device
        )
        for start, end in bounds:
            indices = torch.randint(0, count, (per_class,), device=pool.device)
            result[:, :, start:end] = pool.index_select(0, indices)[:, :, start:end]
        return result

    def sample_whole(pool):
        count = int(pool.shape[0])
        if count <= 0:
            raise ValueError("Cannot sample from an empty class pool")
        indices = torch.randint(0, count, (per_class,), device=pool.device)
        return pool.index_select(0, indices).contiguous()

    def bandmix(first, second):
        first_fft = torch.fft.rfft(first.float(), dim=-1)
        second_fft = torch.fft.rfft(second.float(), dim=-1)
        choose_first = (
            torch.rand((per_class, number_bands), device=device) < 0.5
        ).float()
        mask_first = torch.matmul(choose_first, band_masks)
        mask_second = torch.matmul(1.0 - choose_first, band_masks)
        covered = (mask_first + mask_second).clamp(max=1.0)
        mixed_fft = (
            first_fft * mask_first.unsqueeze(1)
            + second_fft * mask_second.unsqueeze(1)
            + first_fft * (1.0 - covered).unsqueeze(1)
        )
        return torch.fft.irfft(mixed_fft, n=length, dim=-1).to(dtype=x.dtype)

    write_position = 0
    for class_index in range(classes):
        pool = _class_pool(exp, x, y, class_index, device)
        if int(pool.shape[0]) == 0:
            print(f"[WARN] Empty augmentation pool for class {class_index}; skipping.")
            continue
        if mode == "SR_ONLY":
            synthetic = sr_recompose(pool)
        elif mode == "BANDMIX_ONLY":
            synthetic = bandmix(sample_whole(pool), sample_whole(pool))
        else:
            synthetic = bandmix(sr_recompose(pool), sr_recompose(pool))

        end = write_position + per_class
        aug_data[write_position:end] = synthetic.unsqueeze(1)
        aug_label[write_position:end] = class_index
        write_position = end

    if write_position == 0:
        return _empty(exp, x, y)
    order = torch.randperm(write_position, device=device)
    return aug_data[:write_position][order].float(), aug_label[:write_position][order].long()


def _xie_oniga(exp, timg, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(timg) if not isinstance(timg, torch.Tensor) else timg
    y = torch.from_numpy(label) if not isinstance(label, torch.Tensor) else label
    x = x.to(device).float()
    y = y.to(device).view(-1).long()

    classes = int(exp.number_class)
    channels, length = int(exp.number_channel), int(x.shape[-1])
    per_class = int(exp.number_augmentation) * max(1, exp.batch_size // classes)
    if per_class <= 0:
        return _empty(exp, x, y)

    replace_length = max(1, min(length, int(round(float(exp.sample_rate)))))
    frequencies = torch.fft.rfftfreq(n=length, d=1.0 / float(exp.sample_rate)).to(device)
    low_mask = ((frequencies >= 7.0) & (frequencies <= 13.0)).float()
    high_mask = ((frequencies >= 14.0) & (frequencies <= 30.0)).float()
    if int(low_mask.sum().item()) == 0 or int(high_mask.sum().item()) == 0:
        raise RuntimeError("XIE_ONIGA frequency masks are empty")

    total = per_class * classes
    aug_data = torch.empty((total, 1, channels, length), dtype=x.dtype, device=device)
    aug_label = torch.empty((total,), dtype=torch.long, device=device)
    write_position = 0

    for class_index in range(classes):
        pool = _class_pool(exp, x, y, class_index, device)
        pool_size = int(pool.shape[0])
        if pool_size == 0:
            print(f"[WARN] Empty augmentation pool for class {class_index}; skipping.")
            continue
        if pool_size >= 3:
            source_indices = torch.rand((per_class, pool_size), device=device).topk(
                k=3, dim=1, largest=True, sorted=False
            ).indices
        else:
            source_indices = torch.randint(
                0, pool_size, (per_class, 3), device=device
            )

        sample1 = pool.index_select(0, source_indices[:, 0])
        sample2 = pool.index_select(0, source_indices[:, 1])
        sample3 = pool.index_select(0, source_indices[:, 2])

        max_start = length - replace_length
        if max_start > 0:
            starts = torch.randint(0, max_start + 1, (per_class,), device=device)
        else:
            starts = torch.zeros((per_class,), dtype=torch.long, device=device)
        time_axis = torch.arange(length, device=device).view(1, length)
        replace_mask = (
            (time_axis >= starts.view(-1, 1))
            & (time_axis < (starts + replace_length).view(-1, 1))
        )
        time_aug = torch.where(replace_mask.unsqueeze(1), sample2, sample1)

        time_fft = torch.fft.rfft(time_aug.float(), dim=-1)
        sample3_fft = torch.fft.rfft(sample3.float(), dim=-1)
        swap_low = torch.rand((per_class, 1, 1), device=device) < 0.5
        mixed_fft = torch.where(
            swap_low,
            sample3_fft * low_mask.view(1, 1, -1),
            time_fft * low_mask.view(1, 1, -1),
        )
        mixed_fft = mixed_fft + torch.where(
            swap_low,
            time_fft * high_mask.view(1, 1, -1),
            sample3_fft * high_mask.view(1, 1, -1),
        )
        synthetic = torch.fft.irfft(mixed_fft, n=length, dim=-1).to(dtype=x.dtype)

        end = write_position + per_class
        aug_data[write_position:end] = synthetic.unsqueeze(1)
        aug_label[write_position:end] = class_index
        write_position = end

    if write_position == 0:
        return _empty(exp, x, y)
    order = torch.randperm(write_position, device=device)
    return aug_data[:write_position][order].float(), aug_label[:write_position][order].long()


def generate_augmentation(exp, timg, label):
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to generate augmentation data")
    mode = normalize_aug_mode(exp.aug_mode)
    if mode == "NO_AUG":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(timg) if not isinstance(timg, torch.Tensor) else timg
        y = torch.from_numpy(label) if not isinstance(label, torch.Tensor) else label
        return _empty(exp, x.to(device).float(), y.to(device).long())
    if mode == "SR_ONLY" and exp.dataset_type == "A":
        return _sr_only_iv2a_numpy(exp, timg, label)
    if mode == "XIE_ONIGA":
        return _xie_oniga(exp, timg, label)
    return _sr_bandmix_torch(exp, timg, label, mode)
