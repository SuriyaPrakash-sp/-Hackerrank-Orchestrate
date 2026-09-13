"""Deterministic Data Loader: reads all dataset/*.csv files into indexed
in-memory structures (pandas DataFrames + dict indexes) so downstream stages
never re-scan the raw CSVs."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from . import config


def _parse_date(s) -> Optional[date]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return None
    return pd.to_datetime(s).date()


@dataclass
class Dataset:
    requests: pd.DataFrame
    sample_requests: pd.DataFrame
    profiles: pd.DataFrame
    events: pd.DataFrame
    payment_options: pd.DataFrame
    messages: pd.DataFrame
    images: pd.DataFrame
    exchange_rates: pd.DataFrame

    # indexes
    profile_by_user: Dict[str, dict] = field(default_factory=dict)
    events_by_user: Dict[str, pd.DataFrame] = field(default_factory=dict)
    options_by_request: Dict[str, pd.DataFrame] = field(default_factory=dict)
    messages_by_user: Dict[str, pd.DataFrame] = field(default_factory=dict)
    images_by_event: Dict[str, dict] = field(default_factory=dict)
    images_by_request: Dict[str, List[dict]] = field(default_factory=dict)


def _read_csv(name: str) -> pd.DataFrame:
    path = config.DATASET_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset file not found: {path}. "
            f"Set DATASET_DIR env var if your dataset lives elsewhere."
        )
    return pd.read_csv(path, dtype=str, keep_default_na=True)


def load_dataset() -> Dataset:
    requests = _read_csv("requests.csv")
    sample_requests = _read_csv("sample_requests.csv")
    profiles = _read_csv("financial_profiles.csv")
    events = _read_csv("financial_events.csv")
    payment_options = _read_csv("request_payment_options.csv")
    messages = _read_csv("messages.csv")
    images = _read_csv("images.csv")
    exchange_rates = _read_csv("exchange_rates.csv")

    ds = Dataset(
        requests=requests,
        sample_requests=sample_requests,
        profiles=profiles,
        events=events,
        payment_options=payment_options,
        messages=messages,
        images=images,
        exchange_rates=exchange_rates,
    )

    # --- profile index ---
    for _, row in profiles.iterrows():
        ds.profile_by_user[row["user_id"]] = row.to_dict()

    # --- events index (by user) ---
    for uid, grp in events.groupby("user_id"):
        ds.events_by_user[uid] = grp.reset_index(drop=True)

    # --- payment options index (by request) ---
    for rid, grp in payment_options.groupby("request_id"):
        ds.options_by_request[rid] = grp.reset_index(drop=True)

    # --- messages index (by user) ---
    for uid, grp in messages.groupby("user_id"):
        ds.messages_by_user[uid] = grp.reset_index(drop=True)

    # --- images index ---
    for _, row in images.iterrows():
        d = row.to_dict()
        if isinstance(d.get("related_event_id"), str) and d["related_event_id"]:
            ds.images_by_event[d["related_event_id"]] = d
        rid = d.get("request_id")
        if isinstance(rid, str) and rid:
            ds.images_by_request.setdefault(rid, []).append(d)

    return ds
