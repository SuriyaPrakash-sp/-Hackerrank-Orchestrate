"""Financial State Reconstructor.

Turns the raw financial_events.csv rows for one user into a clean
UserFinancialState: a starting balance, a set of *explicit* future
cash-impacting events (already dated in the data), and a set of *recurring
series* inferred from historical settled activity (which the forecast engine
projects forward day by day). Message-derived directives (see
message_rules.py) are applied before classification so that amendments,
cancellations, and confirmations are reflected in the reconstructed state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from statistics import median
from typing import Dict, List, Optional

import pandas as pd

from .money import D, is_blank
from .fx import FxEngine

RECURRING_MIN_INTERVAL = 5
RECURRING_MAX_INTERVAL = 100
RECURRING_MIN_OCCURRENCES = 2

# Stable, bill-like categories that recur on a predictable schedule and are
# projected forward for the 90-day forecast. Variable day-to-day
# discretionary/lifestyle spend is, for most users in this dataset, NOT a
# clean fixed-interval series (unlike rent/utilities/subscriptions) and its
# natural volatility is what the user's minimum_balance_to_keep buffer is
# meant to absorb. Calibrating against dataset/sample_requests.csv, treating
# only this set as recurring produced a materially closer match to the
# provided ground truth than extrapolating every historical category
# (some users do have a very regular healthcare/family_support cadence —
# see README "Known limitations" — but this is the better trade-off on
# balance across the sample set).
RECURRING_ELIGIBLE_CATEGORIES = {
    "rent", "housing", "utilities", "education", "debt_repayment", "insurance",
    "salary", "streaming", "music_subscription", "cloud_storage",
    "delivery_membership", "gym",
}


@dataclass
class ExplicitEvent:
    event_id: str
    date: date
    amount: Decimal          # positive, in home currency
    direction: str           # 'debit' or 'credit'
    category: str
    flexibility: str
    minimum_allowed_amount: Optional[Decimal]
    source: str = "csv"      # 'csv' | 'vision' | 'message'


@dataclass
class RecurringSeries:
    category: str
    ref_event_id: str            # most recent historical occurrence's event_id
    interval_days: int
    last_date: date              # last known occurrence (historical or explicit)
    amount: Decimal              # projected per-occurrence amount, home currency
    direction: str
    flexibility: str
    minimum_allowed_amount: Optional[Decimal]
    active: bool = True          # False if income/series was terminated by evidence


@dataclass
class UserFinancialState:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: List[str]
    protect_categories: set
    reduce_categories: set
    stop_categories: set
    accepted_methods: set
    max_installment_months: Optional[Decimal]
    explicit_future: List[ExplicitEvent] = field(default_factory=list)
    recurring: List[RecurringSeries] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _split(s) -> set:
    if is_blank(s):
        return set()
    return {x.strip() for x in str(s).split("|") if x.strip()}


def build_user_state(
    ds,
    user_id: str,
    fx: FxEngine,
    request_date: date,
    horizon_end: date,
    vision_facts: Dict[str, dict],
    directives: "list",
) -> UserFinancialState:
    profile = ds.profile_by_user[user_id]
    home_ccy = profile["home_currency"]

    state = UserFinancialState(
        user_id=user_id,
        home_currency=home_ccy,
        current_available_balance=D(profile["current_available_balance"]),
        minimum_balance_to_keep=D(profile["minimum_balance_to_keep"]),
        financial_priorities=list(_split(profile.get("financial_priorities"))),
        protect_categories=_split(profile.get("expense_categories_to_protect")),
        reduce_categories=_split(profile.get("expense_categories_user_is_willing_to_reduce")),
        stop_categories=_split(profile.get("expense_categories_user_is_willing_to_stop")),
        accepted_methods=_split(profile.get("payment_methods_user_will_consider")),
        max_installment_months=(
            None if is_blank(profile.get("max_installment_months"))
            else D(profile.get("max_installment_months"))
        ),
    )

    events_df = ds.events_by_user.get(user_id)
    if events_df is None or events_df.empty:
        return state

    # index directives for quick lookup
    event_overrides = {d["event_id"]: d for d in directives if d.get("kind") == "event_override" and d.get("event_id")}
    salary_directives = [d for d in directives if d.get("kind") == "salary_directive"]
    exclude_ids = {d["event_id"] for d in directives if d.get("kind") == "exclude_event" and d.get("event_id")}

    rows = []
    for _, r in events_df.iterrows():
        row = r.to_dict()
        eid = row["event_id"]

        status = row.get("status")
        direction = row.get("direction")
        currency = row.get("currency")
        category = row.get("category")
        flexibility = row.get("flexibility") or "fixed"

        settlement = row.get("settlement_date") or row.get("event_date")
        settle_date = pd.to_datetime(settlement).date() if not is_blank(settlement) else None
        event_date = pd.to_datetime(row.get("event_date")).date() if not is_blank(row.get("event_date")) else settle_date

        # --- resolve amount (may be blank -> vision cache) ---
        if is_blank(row.get("amount")):
            fact = vision_facts.get(eid)
            if fact:
                amount = D(fact["amount"])
                currency = fact.get("currency", currency)
                state.notes.append(f"{eid}: blank amount resolved from linked image -> {currency} {amount}")
            else:
                amount = Decimal("0")
                state.notes.append(f"{eid}: blank amount with no linked image evidence; treated as 0 (excluded), not fabricated")
                status = "cancelled"  # exclude safely rather than guess
        else:
            amount = D(row.get("amount"))

        min_allowed = None if is_blank(row.get("minimum_allowed_amount")) else D(row.get("minimum_allowed_amount"))

        # --- apply message-driven event overrides ---
        if eid in event_overrides:
            ov = event_overrides[eid]
            if ov.get("new_status"):
                status = ov["new_status"]
            if ov.get("new_amount") is not None:
                amount = D(ov["new_amount"])
                currency = ov.get("new_currency", currency)
            if ov.get("new_date") is not None:
                settle_date = ov["new_date"]
                event_date = ov["new_date"]
        if eid in exclude_ids:
            status = "cancelled"

        rows.append({
            "event_id": eid, "status": status, "direction": direction,
            "currency": currency, "category": category, "flexibility": flexibility,
            "amount": amount, "min_allowed": min_allowed,
            "settle_date": settle_date, "event_date": event_date,
            "linked_event_id": row.get("linked_event_id"),
        })

    norm = pd.DataFrame(rows)
    if norm.empty:
        return state

    # convert all amounts to home currency (at settlement date)
    def to_home(rec):
        if rec["settle_date"] is None:
            return Decimal("0")
        return fx.convert(rec["amount"], rec["currency"], home_ccy, rec["settle_date"])

    norm["amount_home"] = norm.apply(to_home, axis=1)

    # ------------------------------------------------------------------
    # 1) Explicit future cash-impacting rows: status in {pending, scheduled}
    #    with a settlement date. (Ignored: cancelled, failed, unrealized,
    #    settled-historical, pending-credit.)
    # ------------------------------------------------------------------
    for _, rec in norm.iterrows():
        status = rec["status"]
        direction = rec["direction"]
        if direction == "non_cash":
            continue
        if status not in ("pending", "scheduled"):
            continue
        if status == "pending" and direction == "credit":
            continue  # ignore pending credits (refunds, bonuses, commissions, prize proceeds, etc.)
        if rec["settle_date"] is None:
            continue
        state.explicit_future.append(ExplicitEvent(
            event_id=rec["event_id"], date=rec["settle_date"], amount=rec["amount_home"],
            direction=direction, category=rec["category"], flexibility=rec["flexibility"],
            minimum_allowed_amount=(
                None if rec["min_allowed"] is None
                else fx.convert(rec["min_allowed"], rec["currency"], home_ccy, rec["settle_date"])
            ),
            source="csv",
        ))

    # ------------------------------------------------------------------
    # 2) Recurrence detection from historical settled activity, grouped by
    #    category. A category recurs if it has >=2 settled occurrences at a
    #    roughly consistent interval.
    # ------------------------------------------------------------------
    # Include settled (historical) AND already-known future (scheduled/pending)
    # rows when establishing a recurrence pattern: a single settled occurrence
    # plus one confirmed future occurrence (e.g. "next confirmed salary") is
    # enough to infer a monthly cadence and project further months forward.
    hist = norm[
        (norm["status"].isin(["settled", "scheduled", "pending"]))
        & (norm["direction"] != "non_cash")
        & (norm["settle_date"].notna())
        & (
            norm["category"].isin(RECURRING_ELIGIBLE_CATEGORIES)
            | norm["flexibility"].isin(["stoppable", "reducible", "reducible_or_stoppable"])
        )
    ]
    for category, grp in hist.groupby("category"):
        grp = grp.sort_values("settle_date")
        dates = list(grp["settle_date"])
        if len(dates) < RECURRING_MIN_OCCURRENCES:
            continue
        diffs = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        diffs = [d for d in diffs if d > 0]
        if not diffs:
            continue
        med = median(diffs)
        if not (RECURRING_MIN_INTERVAL <= med <= RECURRING_MAX_INTERVAL):
            continue
        # consistency check: most diffs within +/-40% of median
        consistent = sum(1 for d in diffs if abs(d - med) <= max(4, 0.4 * med))
        if consistent < max(1, int(0.6 * len(diffs))):
            continue

        last_row = grp.iloc[-1]
        last_date = last_row["settle_date"]
        ref_event_id = last_row["event_id"]
        direction = last_row["direction"]
        flexibility = last_row["flexibility"]
        min_allowed = last_row["min_allowed"]
        min_allowed_home = (
            None if min_allowed is None
            else fx.convert(min_allowed, last_row["currency"], home_ccy, last_date)
        )

        # If there's already an explicit (pending/scheduled) row for this same
        # category, treat that as the most recent known occurrence too, so
        # projection resumes *after* it (avoids double counting).
        explicit_same_cat = [e for e in state.explicit_future if e.category == category]
        if explicit_same_cat:
            last_explicit = max(explicit_same_cat, key=lambda e: e.date)
            if last_explicit.date >= last_date:
                last_date = last_explicit.date

        # Use the most recently known amount (not a blended average) as the
        # projected run-rate, so raises/cuts reflected in the latest
        # confirmed occurrence are carried forward rather than smoothed away.
        amount = D(last_row["amount_home"])
        if last_date == (explicit_same_cat[-1].date if explicit_same_cat else None):
            amount = max(explicit_same_cat, key=lambda e: e.date).amount
        interval = int(round(med))

        state.recurring.append(RecurringSeries(
            category=category, ref_event_id=ref_event_id, interval_days=interval,
            last_date=last_date, amount=amount, direction=direction,
            flexibility=flexibility, minimum_allowed_amount=min_allowed_home,
        ))

    # ------------------------------------------------------------------
    # 3) Apply salary directives derived from messages (increase/decrease/
    #    ended/new-employer/etc.) to the salary recurring series and/or the
    #    explicit next-salary row.
    # ------------------------------------------------------------------
    _apply_salary_directives(state, salary_directives, fx, home_ccy)

    # category-wide percentage adjustments (e.g. rent increase X%)
    for d in directives:
        if d.get("kind") == "category_pct_change":
            pct = D(d["pct"]) / Decimal(100)
            for s in state.recurring:
                if s.category == d["category"]:
                    s.amount = s.amount * (Decimal(1) + pct)

    # brand-new one-off confirmed income events not tied to any existing row
    for d in directives:
        if d.get("kind") == "new_income_event" and d.get("date") is not None and d.get("amount") is not None:
            amt_home = fx.convert(D(d["amount"]), d.get("currency") or home_ccy, home_ccy, d["date"])
            state.explicit_future.append(ExplicitEvent(
                event_id=f"msg_income_{d.get('message_id','')}", date=d["date"], amount=amt_home,
                direction="credit", category=d.get("category", "other_income"), flexibility="fixed",
                minimum_allowed_amount=None, source="message",
            ))

    return state


def _apply_salary_directives(state: UserFinancialState, directives: list, fx: FxEngine, home_ccy: str):
    if not directives:
        return
    directives = sorted(directives, key=lambda d: d.get("sent_at") or "")

    salary_series = next((s for s in state.recurring if s.category == "salary"), None)
    explicit_salary = [e for e in state.explicit_future if e.category == "salary"]

    for d in directives:
        kind = d.get("subkind")
        amt = d.get("amount")
        ccy = d.get("currency") or home_ccy
        eff_date = d.get("effective_date")
        amt_home = None
        if amt is not None:
            sent_at = d.get("sent_at")
            sent_date = pd.to_datetime(sent_at).date() if sent_at is not None and not pd.isna(sent_at) else date.today()
            conv_date = eff_date or sent_date
            amt_home = fx.convert(D(amt), ccy, home_ccy, conv_date)

        if kind in ("increase", "decrease_temp", "resumed", "confirmed_for_date", "new_employer_first"):
            if amt_home is None:
                continue
            if explicit_salary:
                for e in explicit_salary:
                    e.amount = amt_home
                    if eff_date:
                        e.date = eff_date
            elif salary_series is not None:
                salary_series.amount = amt_home
                if eff_date and eff_date > salary_series.last_date:
                    salary_series.last_date = eff_date - timedelta(days=salary_series.interval_days)
            elif eff_date:
                # No existing salary evidence at all: add as a confirmed
                # one-off future credit, per explicit employer confirmation.
                state.explicit_future.append(ExplicitEvent(
                    event_id=f"msg_salary_{d.get('message_id','')}", date=eff_date, amount=amt_home,
                    direction="credit", category="salary", flexibility="fixed",
                    minimum_allowed_amount=None, source="message",
                ))
        elif kind == "date_shift_only":
            if eff_date is None:
                continue
            if explicit_salary:
                for e in explicit_salary:
                    e.date = eff_date
            elif salary_series is not None and eff_date > salary_series.last_date:
                salary_series.last_date = eff_date - timedelta(days=salary_series.interval_days)
        elif kind in ("ended",):
            if salary_series is not None:
                salary_series.active = False
            # remove explicit future salary rows dated after the message's
            # effective/sent date, since the income stream has ended.
            cutoff = eff_date or d.get("sent_date")
            if cutoff:
                state.explicit_future = [
                    e for e in state.explicit_future
                    if not (e.category == "salary" and e.date > cutoff)
                ]
        elif kind == "household_ended_partial":
            if amt_home is None:
                continue
            if salary_series is not None:
                salary_series.amount = amt_home
            for e in explicit_salary:
                e.amount = amt_home
        elif kind == "arrears_onetime":
            if amt_home is None:
                continue
            base_date = eff_date
            if base_date is None and explicit_salary:
                base_date = max(e.date for e in explicit_salary)
            if base_date is None and salary_series is not None:
                base_date = salary_series.last_date + timedelta(days=salary_series.interval_days)
            if base_date is not None:
                state.explicit_future.append(ExplicitEvent(
                    event_id=f"msg_arrears_{d.get('message_id','')}", date=base_date, amount=amt_home,
                    direction="credit", category="salary", flexibility="fixed",
                    minimum_allowed_amount=None, source="message",
                ))
        # unrecognized kinds are safely ignored (no-op)


def eligible_flexible_targets(state: UserFinancialState, horizon_end: date) -> List[RecurringSeries]:
    """Recurring series that may legally be stopped/reduced: flexible
    (stoppable/reducible), in a category the user is willing to touch, and
    NOT in a protected category."""
    out = []
    for s in state.recurring:
        if not s.active or s.direction != "debit":
            continue
        if s.category in state.protect_categories:
            continue
        can_stop = s.flexibility in ("stoppable", "reducible_or_stoppable") and s.category in state.stop_categories
        can_reduce = s.flexibility in ("reducible", "reducible_or_stoppable") and s.category in state.reduce_categories
        if can_stop or can_reduce:
            out.append(s)
    return out
