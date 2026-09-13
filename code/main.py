#!/usr/bin/env python3
"""Buy or Wait? — main entry point.

Usage:
    python main.py                     # process dataset/requests.csv -> ../output.csv
    python main.py --requests sample   # process dataset/sample_requests.csv instead
    python main.py --limit 10          # only process the first N requests (debugging)

Environment variables (all optional):
    DATASET_DIR   - override the dataset directory (default: ../dataset)
    OUTPUT_PATH   - override the output CSV path (default: ../output.csv)
    LLM_API_KEY   - enable live LLM evidence extraction for cache/rule misses
    LLM_BASE_URL, TEXT_MODEL, VISION_MODEL - LLM configuration
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config, data_loader, evidence
from src.fx import FxEngine
from src.pipeline import process_request
from src.usage import write_usage_report

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def main():
    parser = argparse.ArgumentParser(description="Buy or Wait? decision engine")
    parser.add_argument("--requests", choices=["main", "sample"], default="main",
                         help="Which request file to process (default: main = requests.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N requests")
    parser.add_argument("--output", type=str, default=None, help="Override output CSV path")
    args = parser.parse_args()

    t0 = time.time()
    print(f"[buy-or-wait] Loading dataset from {config.DATASET_DIR} ...")
    ds = data_loader.load_dataset()

    requests_df = ds.sample_requests if args.requests == "sample" else ds.requests
    if args.limit:
        requests_df = requests_df.head(args.limit)

    fx = FxEngine(ds.exchange_rates)

    print("[buy-or-wait] Resolving evidence (vision cache + message directives) ...")
    vision_facts = evidence.get_vision_facts(ds)
    directives_by_user = evidence.get_directives(ds)
    unmatched = directives_by_user.pop("_unmatched_count", 0)
    num_directives = sum(len(v) for v in directives_by_user.values())

    print(f"[buy-or-wait] Processing {len(requests_df)} requests ...")
    rows = []
    for i, (_, req) in enumerate(requests_df.iterrows(), 1):
        try:
            row = process_request(ds, fx, req.to_dict(), vision_facts, directives_by_user)
        except Exception as e:  # never let one bad request crash the whole run
            row = {
                "request_id": req["request_id"],
                "amount_safe_to_pay": "0",
                "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended",
                "payment_plan": "none",
                "earliest_date_for_full_payment": "",
                "spending_changes_needed": "none",
                "decision_explanation": f"Could not be evaluated safely due to an internal error: {e}",
            }
        rows.append(row)
        if i % 50 == 0 or i == len(requests_df):
            print(f"[buy-or-wait]   {i}/{len(requests_df)} done")

    import csv
    out_path = Path(args.output) if args.output else config.OUTPUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    write_usage_report(
        num_requests=len(rows),
        num_vision_cached=len(vision_facts),
        num_vision_live=0,
        num_users_with_directives=len(directives_by_user),
        num_directives=num_directives,
        num_unmatched_messages=unmatched,
        llm_calls_made=0,
    )

    dt = time.time() - t0
    print(f"[buy-or-wait] Wrote {len(rows)} rows to {out_path} in {dt:.1f}s")


if __name__ == "__main__":
    main()
