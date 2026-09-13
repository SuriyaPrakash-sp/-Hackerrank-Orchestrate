#!/usr/bin/env python3
"""Compare a run's output against dataset/sample_requests.csv (the only
ground truth shipped with the challenge). Not part of the graded pipeline —
just a convenience script for further calibration.

Usage:
    python evaluation/score_samples.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    sample_out = ROOT / "sample_output.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--requests", "sample", "--output", str(sample_out)],
        check=True,
    )
    mine = pd.read_csv(sample_out)
    truth = pd.read_csv(ROOT.parent / "dataset" / "sample_requests.csv")
    m = mine.merge(truth, on="request_id", suffixes=("_mine", "_true"))

    for col in ["affordability_status", "recommended_payment_method",
                "earliest_date_for_full_payment", "spending_changes_needed"]:
        match = (m[f"{col}_mine"].astype(str) == m[f"{col}_true"].astype(str)).sum()
        print(f"{col}: {match}/{len(m)} exact matches")

    m["amt_diff_pct"] = (
        (m["amount_safe_to_pay_mine"].astype(float) - m["amount_safe_to_pay_true"].astype(float)).abs()
        / m["amount_safe_to_pay_true"].astype(float).clip(lower=1)
    )
    print(f"amount_safe_to_pay: median relative error = {m['amt_diff_pct'].median():.1%}")


if __name__ == "__main__":
    main()
