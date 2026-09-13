"""Plan Generator + Ranker.

Builds every *eligible* candidate way of satisfying a request (pay now, wait,
pay now after permitted spending changes, partial payment, each supplied
installment option) and ranks them using the six tie-break rules from the
problem statement, encoded as a single sort key:

  1. Completes by desired_completion_date            -> meets_deadline
  2. Requires no spending changes                     -> uses_changes
  3. Minimizes total amount paid                      -> total_cost
  4. Starts earlier                                   -> start_date
  5. Uses fewer payments                               -> num_payments
  6. Lowest payment_option_id (final tiebreaker)       -> payment_option_id
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from itertools import combinations, product
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .money import D
from .events import UserFinancialState, eligible_flexible_targets
from . import forecast as fc

MAX_CHANGES = 3


@dataclass
class Candidate:
    method: str                 # 'full_payment' | 'wait' | 'partial_payment' | 'installments'
    status: str                 # affordable_now | affordable_with_plan | affordable_later
    payments: List[Tuple[date, Decimal]]
    changes: List[Tuple[str, str, Optional[Decimal]]] = field(default_factory=list)  # (kind, category, amount)
    payment_option_id: Optional[str] = None

    @property
    def total_cost(self) -> Decimal:
        return sum((amt for _, amt in self.payments), Decimal("0"))

    @property
    def start_date(self) -> date:
        return min(d for d, _ in self.payments)

    @property
    def last_date(self) -> date:
        return max(d for d, _ in self.payments)

    @property
    def num_payments(self) -> int:
        return len(self.payments)

    def meets_deadline(self, desired: date) -> bool:
        return self.last_date <= desired

    def sort_key(self, desired: date):
        return (
            0 if self.meets_deadline(desired) else 1,
            0 if not self.changes else 1,
            self.total_cost,
            self.start_date,
            self.num_payments,
            self.payment_option_id or "",
        )


def _search_spending_changes(state: UserFinancialState, request_date: date, horizon_end: date,
                              requested_amount: Decimal):
    targets = eligible_flexible_targets(state, horizon_end)
    variants = []
    for t in targets:
        opts = []
        if t.flexibility in ("stoppable", "reducible_or_stoppable") and t.category in state.stop_categories:
            opts.append(("stop", t.category, t.ref_event_id, None))
        if t.flexibility in ("reducible", "reducible_or_stoppable") and t.category in state.reduce_categories:
            amt = t.minimum_allowed_amount if t.minimum_allowed_amount is not None else Decimal("0")
            opts.append(("reduce", t.category, t.ref_event_id, amt))
        if opts:
            variants.append(opts)

    n = len(variants)
    for size in range(1, min(MAX_CHANGES, n) + 1):
        for idxs in combinations(range(n), size):
            for choice in product(*(variants[i] for i in idxs)):
                changes = tuple((kind, ref_id, amt) for kind, _cat, ref_id, amt in choice)
                if fc.is_safe_amount(state, request_date, requested_amount, changes=changes):
                    return [(kind, cat, ref_id, amt) for kind, cat, ref_id, amt in choice]
    return None


def build_candidates(
    ds, fx, state: UserFinancialState, request_row: dict,
) -> Tuple[List[Candidate], Decimal, Optional[date]]:
    request_date = pd.to_datetime(request_row["request_date"]).date()
    desired = pd.to_datetime(request_row["desired_completion_date"]).date()
    requested_amount = D(request_row["requested_amount"])
    allows_partial = str(request_row.get("allows_partial_payment", "")).strip().lower() == "true"
    horizon_end = request_date + timedelta(days=90)

    base_safe = fc.amount_safe_to_pay(state, request_date, requested_amount)
    baseline_earliest = fc.earliest_date_for_full_payment(state, request_date, requested_amount)

    candidates: List[Candidate] = []
    accepted = state.accepted_methods

    # (a) full payment today
    if "full_payment" in accepted and base_safe >= requested_amount:
        candidates.append(Candidate(
            method="full_payment", status="affordable_now",
            payments=[(request_date, requested_amount)],
        ))

    # (b) wait for full payment later
    if "full_payment" in accepted and baseline_earliest is not None and baseline_earliest != request_date:
        candidates.append(Candidate(
            method="wait", status="affordable_later",
            payments=[(baseline_earliest, requested_amount)],
        ))
    elif "full_payment" in accepted and baseline_earliest is not None and base_safe < requested_amount:
        # earliest == request_date but base_safe already said unsafe: contradiction guard, skip
        pass

    # (c) full payment today via permitted spending changes
    if "full_payment" in accepted and base_safe < requested_amount:
        combo = _search_spending_changes(state, request_date, horizon_end, requested_amount)
        if combo:
            candidates.append(Candidate(
                method="full_payment", status="affordable_with_plan",
                payments=[(request_date, requested_amount)],
                changes=[(k, c, a) for k, c, _rid, a in combo],
            ))
            # stash ref ids on the candidate for spending_changes_needed formatting
            candidates[-1].__dict__["_change_refs"] = [(k, rid, a) for k, _c, rid, a in combo]

    # (d) partial payment (exactly two payments; no spending changes)
    if (allows_partial and "partial_payment" in accepted
            and 0 < base_safe < requested_amount
            and baseline_earliest is not None and baseline_earliest <= desired):
        remaining = requested_amount - base_safe
        plan = {request_date: base_safe, baseline_earliest: remaining}
        if fc.simulate_plan(state, plan, request_date):
            candidates.append(Candidate(
                method="partial_payment", status="affordable_with_plan",
                payments=[(request_date, base_safe), (baseline_earliest, remaining)],
            ))

    # (e) installments (must match a supplied payment option exactly)
    if "installments" in accepted and state.max_installment_months is not None:
        options = ds.options_by_request.get(request_row["request_id"])
        if options is not None:
            for _, opt in options.iterrows():
                if opt.get("payment_method") != "installments":
                    continue
                n = int(D(opt["number_of_payments"]))
                freq = int(D(opt["payment_frequency_days"]))
                first = pd.to_datetime(opt["first_payment_date"]).date()
                per_payment = D(opt["payment_amount"])
                span_months = ((n - 1) * freq) / Decimal(30.44) if n > 1 else Decimal("0")
                if span_months > state.max_installment_months:
                    continue
                payments = [(first + timedelta(days=freq * i), per_payment) for i in range(n)]
                if payments[-1][0] > desired:
                    continue
                if not fc.simulate_plan(state, dict(payments), request_date):
                    continue
                candidates.append(Candidate(
                    method="installments", status="affordable_with_plan",
                    payments=payments, payment_option_id=opt["payment_option_id"],
                ))

    return candidates, base_safe, baseline_earliest


def select_best(candidates: List[Candidate], desired: date) -> Optional[Candidate]:
    if not candidates:
        return None
    return min(candidates, key=lambda c: c.sort_key(desired))
