"""Per-request orchestration: state reconstruction -> planning -> verification
-> explanation -> output row."""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Dict

import pandas as pd

from .money import D, fmt_amount
from .events import build_user_state
from .planner import build_candidates, select_best
from .verifier import verify
from .explain import build_explanation
from . import forecast as fc


def _format_spending_changes(candidate) -> str:
    refs = getattr(candidate, "_change_refs", None)
    if not refs:
        return "none"
    parts = []
    for kind, ref_id, amt in refs:
        if kind == "stop":
            parts.append(f"stop:{ref_id}")
        else:
            parts.append(f"reduce_to:{ref_id}:{fmt_amount(amt)}")
    return "|".join(parts)


def _format_payment_plan(candidate) -> str:
    return "|".join(f"{d.isoformat()}:{fmt_amount(amt)}" for d, amt in candidate.payments)


def process_request(ds, fx, request_row: dict, vision_facts: Dict[str, dict],
                     directives_by_user: Dict[str, list]) -> dict:
    user_id = request_row["user_id"]
    request_date = pd.to_datetime(request_row["request_date"]).date()
    desired = pd.to_datetime(request_row["desired_completion_date"]).date()
    requested_amount = D(request_row["requested_amount"])
    horizon_end = request_date + timedelta(days=90)

    profile = ds.profile_by_user[user_id]
    currency = profile["home_currency"]

    state = build_user_state(
        ds, user_id, fx, request_date, horizon_end,
        vision_facts, directives_by_user.get(user_id, []),
    )

    candidates, base_safe, baseline_earliest = build_candidates(ds, fx, state, request_row)
    best = select_best(candidates, desired)

    ok = verify(state, best, request_date)
    if not ok:
        best = None

    if best is None:
        status = "not_affordable"
        method = "not_recommended"
        payment_plan = "none"
        spending_changes = "none"
        min_reached = state.minimum_balance_to_keep
    else:
        status = best.status
        method = best.method
        payment_plan = _format_payment_plan(best)
        spending_changes = _format_spending_changes(best)
        changes_tuple = tuple((k, rid, a) for k, rid, a in getattr(best, "_change_refs", []))
        payments_dict = dict(best.payments)
        in_window = {d: a for d, a in payments_dict.items() if request_date <= d <= horizon_end}
        _, min_reached = fc.simulate(state, request_date, in_window, changes=changes_tuple)

    explanation = build_explanation(currency, best, requested_amount, state.minimum_balance_to_keep, min_reached)

    return {
        "request_id": request_row["request_id"],
        "amount_safe_to_pay": fmt_amount(base_safe),
        "affordability_status": status,
        "recommended_payment_method": method,
        "payment_plan": payment_plan,
        "earliest_date_for_full_payment": baseline_earliest.isoformat() if baseline_earliest else "",
        "spending_changes_needed": spending_changes,
        "decision_explanation": explanation,
    }
