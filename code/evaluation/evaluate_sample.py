import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import data_loader, fx, pipeline, evidence

def evaluate():
    ds = data_loader.load_dataset()
    fx_engine = fx.FxEngine(ds.exchange_rates)
    vision_facts = evidence.get_vision_facts(ds)
    directives_by_user = evidence.get_directives(ds)

    sample_df = ds.sample_requests
    cols_to_check = [
        'amount_safe_to_pay',
        'affordability_status',
        'recommended_payment_method',
        'payment_plan',
        'earliest_date_for_full_payment',
        'spending_changes_needed'
    ]
    matches = {col: 0 for col in cols_to_check}
    total = len(sample_df)

    for _, req in sample_df.iterrows():
        pred = pipeline.process_request(ds, fx_engine, req.to_dict(), vision_facts, directives_by_user)
        for col in cols_to_check:
            actual = str(req[col]).strip() if pd.notna(req[col]) else ''
            predicted = str(pred[col]).strip()
            if actual == predicted:
                matches[col] += 1
            else:
                print(f"Mismatch {req['request_id']} [{col}]: Expected '{actual}' | Got '{predicted}'")

    print(f"\n--- Accuracy on dataset/sample_requests.csv ({total} rows) ---")
    for col, cnt in matches.items():
        print(f"  {col:32s}: {cnt}/{total} ({cnt/total*100:.1f}%)")

if __name__ == "__main__":
    evaluate()
