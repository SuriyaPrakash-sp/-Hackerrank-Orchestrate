"""90-Day Forecast Simulator.

Builds a day-indexed net cash-flow delta for a user (explicit future events +
recurring-series projections, with optional what-if spending changes), then
answers two questions:

  * amount_safe_to_pay(day)   - via binary search (monotonic: paying more is
                                  never safer).
  * earliest_date_for_full_payment - via a day-by-day scan across the
                                  forecast horizon.

All money math uses Decimal.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from .config import FORECAST_HORIZON_DAYS, BINARY_SEARCH_TOLERANCE
from .money import D
from .events import UserFinancialState, RecurringSeries


Change = Tuple[str, str, Optional[Decimal]]  # (kind: 'stop'|'reduce', ref_event_id, new_amount_or_None)


def _apply_changes(series: List[RecurringSeries], changes: Tuple[Change, ...]) -> Dict[str, Decimal]:
    """Returns {ref_event_id: effective_amount} overrides for the given
    recurring series list, given a tuple of (kind, ref_event_id, amount) changes."""
    overrides: Dict[str, Decimal] = {}
    by_id = {s.ref_event_id: s for s in series}
    for kind, ref_id, amt in changes:
        s = by_id.get(ref_id)
        if s is None:
            continue
        if kind == "stop":
            overrides[ref_id] = Decimal("0")
        elif kind == "reduce":
            overrides[ref_id] = amt if amt is not None else (s.minimum_allowed_amount or Decimal("0"))
    return overrides


def build_daily_deltas(
    state: UserFinancialState,
    start: date,
    horizon_days: int = FORECAST_HORIZON_DAYS,
    changes: Tuple[Change, ...] = (),
) -> Dict[date, Decimal]:
    """Net signed delta per day (credit positive, debit negative) from
    explicit future events plus projected recurring occurrences, over
    [start, start+horizon_days]."""
    end = start + timedelta(days=horizon_days)
    deltas: Dict[date, Decimal] = {}

    def add(d: date, amount_signed: Decimal):
        if start <= d <= end:
            deltas[d] = deltas.get(d, Decimal("0")) + amount_signed

    for e in state.explicit_future:
        if e.date < start or e.date > end:
            continue
        signed = e.amount if e.direction == "credit" else -e.amount
        add(e.date, signed)

    overrides = _apply_changes(state.recurring, changes)

    for s in state.recurring:
        if not s.active:
            continue
        amount = overrides.get(s.ref_event_id, s.amount)
        if amount == 0:
            continue
        d = s.last_date + timedelta(days=s.interval_days)
        signed = amount if s.direction == "credit" else -amount
        # guard against pathological tiny intervals blowing up the loop
        steps = 0
        while d <= end and steps < 400:
            if d >= start:
                add(d, signed)
            d = d + timedelta(days=s.interval_days)
            steps += 1

    return deltas


def simulate(
    state: UserFinancialState,
    start: date,
    payments: Dict[date, Decimal],
    horizon_days: int = FORECAST_HORIZON_DAYS,
    changes: Tuple[Change, ...] = (),
) -> Tuple[bool, Decimal]:
    """Runs the day-by-day simulation with the given extra `payments`
    (date -> amount, debited) applied on top of the base cash flow. Returns
    (is_safe, minimum_balance_reached)."""
    deltas = build_daily_deltas(state, start, horizon_days, changes)
    balance = state.current_available_balance
    min_seen = balance
    min_required = state.minimum_balance_to_keep
    end = start + timedelta(days=horizon_days)
    d = start
    while d <= end:
        balance += deltas.get(d, Decimal("0"))
        if d in payments:
            balance -= payments[d]
        if balance < min_seen:
            min_seen = balance
        d += timedelta(days=1)
    return (min_seen >= min_required, min_seen)


def is_safe_amount(state: UserFinancialState, on: date, amount: Decimal,
                    changes: Tuple[Change, ...] = ()) -> bool:
    safe, _ = simulate(state, on, {on: amount}, changes=changes)
    return safe


def amount_safe_to_pay(state: UserFinancialState, request_date: date, cap: Decimal,
                        changes: Tuple[Change, ...] = ()) -> Decimal:
    """Largest amount in [0, cap] that stays safe if paid on request_date."""
    if cap <= 0:
        return Decimal("0")
    if is_safe_amount(state, request_date, cap, changes):
        return cap
    lo, hi = Decimal("0"), cap
    tol = D(BINARY_SEARCH_TOLERANCE)
    if not is_safe_amount(state, request_date, lo, changes):
        return Decimal("0")
    for _ in range(60):
        if hi - lo <= tol:
            break
        mid = (lo + hi) / 2
        if is_safe_amount(state, request_date, mid, changes):
            lo = mid
        else:
            hi = mid
    return lo


def earliest_date_for_full_payment(
    state: UserFinancialState, request_date: date, amount: Decimal,
    horizon_days: int = FORECAST_HORIZON_DAYS, changes: Tuple[Change, ...] = (),
) -> Optional[date]:
    """First date in [request_date, request_date+horizon] on which paying
    `amount` as a single lump sum keeps the forecast safe throughout the
    (fixed) 90-day window anchored at request_date."""
    if amount <= 0:
        return request_date
    end = request_date + timedelta(days=horizon_days)
    d = request_date
    while d <= end:
        if is_safe_amount(state, d, amount, changes):
            return d
        d += timedelta(days=1)
    return None


def simulate_plan(state: UserFinancialState, payments: Dict[date, Decimal],
                   request_date: date, horizon_days: int = FORECAST_HORIZON_DAYS) -> bool:
    """Safety check for a multi-payment plan (partial/installments): only
    payments that fall within the 90-day window are checked; the running
    balance must stay >= minimum on every day within that window."""
    in_window = {d: a for d, a in payments.items() if request_date <= d <= request_date + timedelta(days=horizon_days)}
    safe, _ = simulate(state, request_date, in_window, horizon_days=horizon_days)
    return safe
