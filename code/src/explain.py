"""Decision Explanation Generator.

Produces the short, user-facing `decision_explanation` string. Template
based (no LLM call needed) so it is fully deterministic and free; mirrors
the tone and structure seen in dataset/sample_requests.csv.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import List, Optional

from .money import fmt_amount
from .planner import Candidate

CATEGORY_LABELS = {
    "streaming": "streaming subscription",
    "music_subscription": "music subscription",
    "delivery_membership": "delivery membership",
    "cloud_storage": "cloud storage subscription",
    "entertainment": "entertainment spending",
    "dining": "dining out",
    "gym": "gym membership",
    "shopping": "discretionary shopping",
}


def _fmt_date(d: date) -> str:
    if not hasattr(d, "strftime"):
        return str(d)
    return f"{d.day} {d.strftime('%B %Y')}"


def _change_phrase(kind: str, category: str) -> str:
    label = CATEGORY_LABELS.get(category, category.replace("_", " "))
    return f"Stop the {label}" if kind == "stop" else f"Reduce the {label}"


def build_explanation(
    currency: str,
    candidate: Optional[Candidate],
    requested_amount: Decimal,
    minimum_balance_to_keep: Decimal,
    min_balance_reached: Decimal,
) -> str:
    floor = fmt_amount(min(minimum_balance_to_keep, min_balance_reached) if min_balance_reached < minimum_balance_to_keep
                        else minimum_balance_to_keep)
    if candidate is None:
        return (
            f"The full {currency} {fmt_amount(requested_amount)} request cannot be completed safely "
            f"within the next 90 days without breaking the {currency} {fmt_amount(minimum_balance_to_keep)} "
            f"minimum balance you want to keep."
        )

    if candidate.method == "full_payment" and candidate.status == "affordable_now":
        d, amt = candidate.payments[0]
        return (
            f"Pay {currency} {fmt_amount(amt)} today. This leaves at least {currency} {floor} "
            f"available over the next 90 days."
        )

    if candidate.method == "wait":
        d, amt = candidate.payments[0]
        return (
            f"Pay {currency} {fmt_amount(amt)} in full on {_fmt_date(d)}. Paying earlier would take "
            f"the balance below the {currency} {floor} minimum."
        )

    if candidate.method == "full_payment" and candidate.status == "affordable_with_plan":
        refs = getattr(candidate, "_change_refs", [])
        phrases = [_change_phrase(k, c) for k, c, _a in candidate.changes] if candidate.changes else []
        lead = ", then ".join(phrases) if phrases else "Adjust flexible spending"
        d, amt = candidate.payments[0]
        return (
            f"{lead}, then pay {currency} {fmt_amount(amt)} today. This leaves at least "
            f"{currency} {floor} available."
        )

    if candidate.method == "partial_payment":
        (d1, a1), (d2, a2) = candidate.payments
        return (
            f"Pay {currency} {fmt_amount(a1)} today and {currency} {fmt_amount(a2)} on {_fmt_date(d2)}. "
            f"This leaves at least {currency} {floor} available throughout."
        )

    if candidate.method == "installments":
        n = candidate.num_payments
        amt = candidate.payments[0][1]
        start = candidate.start_date
        return (
            f"Use {n} installments of {currency} {fmt_amount(amt)}, starting {_fmt_date(start)}. "
            f"This leaves at least {currency} {floor} available."
        )

    return "See payment_plan for details."
