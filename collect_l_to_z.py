"""Collect alphabet samples from L to Z in sequence, then retrain the model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import collect_dataset
import train_model


LETTERS = tuple("LMNOPQRSTUVWXYZ")
DATASET_PATH = Path("dataset") / "gesture_dataset.csv"


def count_label(label: str) -> int:
    if not DATASET_PATH.exists():
        return 0

    dataset = pd.read_csv(DATASET_PATH)
    if "label" not in dataset.columns:
        return 0

    return int((dataset["label"] == label).sum())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect L-Z dataset samples sequentially.")
    parser.add_argument("--samples", type=int, default=120, help="Samples required per letter.")
    parser.add_argument("--auto", action="store_true", help="Capture samples automatically.")
    parser.add_argument("--interval", type=float, default=0.2, help="Auto capture interval.")
    return parser


def run_collect(label: str, samples: int, auto: bool, interval: float) -> int:
    args = ["collect_dataset.py", "--label", label, "--samples", str(samples)]
    if auto:
        args.extend(["--auto", "--interval", str(interval)])

    old_argv = sys.argv[:]
    try:
        sys.argv = args
        return collect_dataset.main()
    finally:
        sys.argv = old_argv


def main() -> int:
    args = build_parser().parse_args()

    for letter in LETTERS:
        existing_count = count_label(letter)
        if existing_count >= args.samples:
            print(f"{letter} sudah punya {existing_count} sample, skip.")
            continue

        print(f"Mulai collect huruf {letter}.")
        result = run_collect(letter, args.samples, args.auto, args.interval)
        if result != 0:
            print(f"Collect huruf {letter} gagal. Proses dihentikan.")
            return result

        current_count = count_label(letter)
        if current_count < args.samples:
            print(f"Huruf {letter} belum lengkap ({current_count}/{args.samples}). Proses dihentikan.")
            return 1

    print("Collect L-Z selesai. Training ulang model...")
    return train_model.main()


if __name__ == "__main__":
    raise SystemExit(main())
