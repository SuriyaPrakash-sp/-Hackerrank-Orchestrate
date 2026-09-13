"""Deterministic Verifier.

Re-simulates the planner's chosen candidate from scratch, independently of
however it was produced, so that a bug in candidate generation can never
silently ship an unsafe recommendation. If verification fails, the request
is downgraded to not_recommended/not_affordable rather than emitting a plan
that hasn't actually been checked.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from .events import UserFinancialState
from .planner import Candidate
from . import forecast as fc


def verify(state: UserFinancialState, candidate: Optional[Candidate], request_date: date) -> bool:
    if candidate is None:
        return True  # nothing to verify; not_recommended is always "safe"
    changes = tuple((k, ref_id, amt) for k, ref_id, amt in getattr(candidate, "_change_refs", []))
    payments = dict(candidate.payments)
    in_window = {d: a for d, a in payments.items() if request_date <= d <= request_date.__class__.fromordinal(
        request_date.toordinal() + 90)}
    if not in_window:
        return True
    safe, _ = fc.simulate(state, request_date, in_window, changes=changes)
    return safe
