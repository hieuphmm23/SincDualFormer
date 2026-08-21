#!/usr/bin/env python3
"""Run one repository setting from config.py, with optional CLI overrides."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from augmentation import SR_BANDMIX_VARIANTS, bands_to_env, get_sr_bandmix_variant
from config import CONFIG


ROOT = Path(__file__).resolve().parent

DEFAULT_MODES = {
    ("2a", "main"): ["SR_BANDMIX"],
    ("2b", "main"): ["SR_BANDMIX"],
    ("2a", "augmentation"): ["NO_AUG","SR_ONLY", "BANDMIX_ONLY", "SR_BANDMIX"],
    ("2b", "augmentation"): ["NO_AUG", "SR_ONLY", "BANDMIX_ONLY", "SR_BANDMIX"],
    ("2a", "srbandmix_variants"): list(SR_BANDMIX_VARIANTS),
    ("2b", "srbandmix_variants"): list(SR_BANDMIX_VARIANTS),
    ("2a", "xie"): ["XIE_ONIGA"],
    ("2b", "xie"): ["XIE_ONIGA"],
    ("2a", "architecture"): [
        "FULL", "WO_BRANCH_I", "WO_TRANSFORMER_ENCODER", "WO_BRANCH_II",
        "WO_SINCNET", "WO_TEMPORAL_CONV_LONG_SHORT", "WO_INCEPTION_TCN",
        "WO_POINTWISE_CONV_FUSION", "WO_BANDSE",
    ],
    ("2b", "architecture"): [
        "FULL", "WO_BRANCH_I", "WO_TRANSFORMER_ENCODER", "WO_BRANCH_II",
        "WO_SINCNET", "WO_TEMPORAL_CONV_LONG_SHORT", "WO_INCEPTION_TCN",
        "WO_POINTWISE_CONV_FUSION", "WO_BANDSE",
    ],
}


def csv(values):
    if isinstance(values, str):
        return values
    return ",".join(str(value) for value in values)


def parse_args():
    parser = argparse.ArgumentParser(description="SincDualFormer reproduction runner")
    parser.add_argument("--dataset", choices=["2a", "2b"])
    parser.add_argument(
        "--experiment",
        choices=["main", "augmentation", "srbandmix_variants", "xie", "architecture"],
    )
    parser.add_argument("--data-root")
    parser.add_argument("--label-root")
    parser.add_argument("--output-root")
    parser.add_argument("--subjects", help="for example: 1 or 1,2,3")
    parser.add_argument("--seeds", help="for example: 0 or 0,1,2,3,4")
    parser.add_argument("--modes", help="comma-separated subset of experiment modes")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalized_mode(mode):
    mode = str(mode).strip().upper().replace("-", "_").replace(" ", "_")
    return {"SR": "SR_ONLY", "BANDMIX": "BANDMIX_ONLY"}.get(mode, mode)


def launch(dataset, experiment, mode, settings, dry_run):
    train_file = ROOT / ("train_2a.py" if dataset == "2a" else "train_2b.py")
    output = Path(settings["output_root"]).expanduser().resolve()
    output = output / dataset / experiment
    if experiment != "architecture":
        output = output / mode.lower()

    env = os.environ.copy()
    suffix = dataset.upper()
    env[f"ROOT_GDF_{suffix}"] = str(Path(settings["data_root"]).expanduser().resolve())
    env[f"ROOT_LABELS_{suffix}"] = str(Path(settings["label_root"]).expanduser().resolve())
    env["OUTPUT_ROOT"] = str(output)
    env["SUBJECTS"] = csv(settings["subjects"])
    env["FIXED_SEEDS"] = csv(settings["seeds"])
    env["ABLATION_MODES"] = "FULL"
    env["AUG_MODE"] = "SR_BANDMIX"
    env.pop("BANDMIX_BANDS", None)

    if experiment in {"main", "augmentation", "xie"}:
        env["AUG_MODE"] = mode
    elif experiment == "srbandmix_variants":
        env["AUG_MODE"] = "SR_BANDMIX"
        variant, bands = get_sr_bandmix_variant(mode)
        env["SR_BANDMIX_VARIANT"] = variant
        # FINE_GRAINED deliberately leaves BANDMIX_BANDS unset so train_2a.py
        # or train_2b.py follows the unchanged main SR-BandMix implementation.
        if variant != "FINE_GRAINED":
            env["BANDMIX_BANDS"] = bands_to_env(bands)
    elif experiment == "architecture":
        env["AUG_MODE"] = "SR_BANDMIX"
        env["ABLATION_MODES"] = mode

    command = [sys.executable, str(train_file)]
    print(f"\nDataset      : {dataset}")
    print(f"Experiment   : {experiment}")
    print(f"Mode         : {mode}")
    print(f"Subjects     : {env['SUBJECTS']}")
    print(f"Seeds        : {env['FIXED_SEEDS']}")
    print(f"Output       : {output}")
    print(f"Command      : {' '.join(command)}")
    if dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


def main():
    args = parse_args()
    settings = dict(CONFIG)
    for key in ("dataset", "experiment", "data_root", "label_root", "output_root"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    if args.subjects:
        settings["subjects"] = args.subjects
    if args.seeds:
        settings["seeds"] = args.seeds

    dataset = str(settings["dataset"]).lower()
    experiment = str(settings["experiment"]).lower()
    key = (dataset, experiment)
    if key not in DEFAULT_MODES:
        raise SystemExit(f"Unsupported dataset/experiment combination: {key}")

    if args.modes:
        modes = [normalized_mode(item) for item in args.modes.split(",") if item.strip()]
    elif settings.get("modes"):
        modes = [normalized_mode(item) for item in settings["modes"]]
    else:
        modes = DEFAULT_MODES[key]

    allowed = set(DEFAULT_MODES[key])
    unknown = [mode for mode in modes if mode not in allowed]
    if unknown:
        raise SystemExit(f"Invalid modes {unknown}; allowed={sorted(allowed)}")

    # Architecture code already loops over a list of modes efficiently.
    if experiment == "architecture":
        modes = [",".join(modes)]

    if not args.dry_run:
        for path_key in ("data_root", "label_root"):
            if not Path(settings[path_key]).expanduser().is_dir():
                raise SystemExit(f"Directory does not exist: {settings[path_key]}")

    for mode in modes:
        return_code = launch(dataset, experiment, mode, settings, args.dry_run)
        if return_code != 0:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
