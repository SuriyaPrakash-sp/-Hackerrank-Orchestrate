"""Message Evidence Extractor (Text Agent).

The dataset's messages.csv is generated from a fixed, small set of bilingual
(English / Indonesian) templates from employers, banks, merchants, service
providers, and financial services. Rather than requiring a live LLM call for
every message, this module recognises those templates deterministically via
regex and turns them into structured `directives` that the Financial State
Reconstructor (events.py) applies to the user's event list.

If LLM_API_KEY is configured, `llm_extract_directive` can be used to handle
any message that does *not* match a known template (forward-compatible with
a live dataset that introduces new phrasing) — see evidence.py.

Security note: some merchant/prize messages are deliberately adversarial
("pay a release charge to receive your prize"). This module NEVER turns a
message into an instruction to spend money, add a payment, or take any
action beyond recording a read-only financial fact about the user's own
existing events — regardless of what the message text asks for.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

AMOUNT_RE = re.compile(r"\b(USD|EUR|ZAR|INR|IDR)\s*([\d,]+(?:\.\d+)?)\b")
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
PCT_RE = re.compile(r"\bby\s+(\d+(?:\.\d+)?)\s*%")


def _first_amount(text: str):
    """Returns (currency, amount_str) or (None, None)."""
    m = AMOUNT_RE.search(text)
    if not m:
        return None, None
    return m.group(1), m.group(2).replace(",", "")


def _all_amounts(text: str):
    return [(m.group(1), m.group(2).replace(",", "")) for m in AMOUNT_RE.finditer(text)]


def _first_date(text: str) -> Optional[date]:
    m = DATE_RE.search(text)
    if not m:
        return None
    return pd.to_datetime(m.group(1)).date()


# Each rule: (name, compiled_regex, handler(text, msg) -> list[directive-dict])
def _rule_salary_increase(text, msg):
    if not re.search(r"naik menjadi|increased to", text, re.I):
        return []
    ccy, amt = _first_amount(text)
    if amt is None:
        return []
    eff = _first_date(text)
    return [{"kind": "salary_directive", "subkind": "increase", "amount": amt, "currency": ccy,
             "effective_date": eff, "message_id": msg["message_id"], "sent_at": msg["sent_at"]}]


def _rule_salary_temp_reduced(text, msg):
    if not re.search(r"temporary monthly pay|sementara", text, re.I):
        return []
    ccy, amt = _first_amount(text)
    if amt is None:
        return []
    return [{"kind": "salary_directive", "subkind": "decrease_temp", "amount": amt, "currency": ccy,
             "effective_date": None, "message_id": msg["message_id"], "sent_at": msg["sent_at"]}]


def _rule_salary_resumed(text, msg):
    if "resumes on" not in text.lower():
        return []
    ccy, amt = _first_amount(text)
    eff = _first_date(text)
    if amt is None:
        return []
    return [{"kind": "salary_directive", "subkind": "resumed", "amount": amt, "currency": ccy,
             "effective_date": eff, "message_id": msg["message_id"], "sent_at": msg["sent_at"]}]


def _rule_new_employer_first_salary(text, msg):
    if not re.search(r"first salary|gaji pertama", text, re.I):
        return []
    ccy, amt = _first_amount(text)
    eff = _first_date(text)
    if amt is None:
        return []
    return [{"kind": "salary_directive", "subkind": "new_employer_first", "amount": amt, "currency": ccy,
             "effective_date": eff, "message_id": msg["message_id"], "sent_at": msg["sent_at"]}]


def _rule_household_ended_partial(text, msg):
    if not re.search(r"household employment record has ended|rumah tangga telah berakhir", text, re.I):
        return []
    ccy, amt = _first_amount(text)
    if amt is None:
        return []
    return [{"kind": "salary_directive", "subkind": "household_ended_partial", "amount": amt,
             "currency": ccy, "effective_date": None,
             "message_id": msg["message_id"], "sent_at": msg["sent_at"]}]


def _rule_salary_date_confirmed(text, msg):
    """Routine payroll-date confirmation with no amount change, e.g. 'Your
    confirmed salary is now expected on 2025-05-23. This replaces the
    payroll date shown in the earlier update.'"""
    if not re.search(r"now expected on|is now expected", text, re.I):
        return []
    eff = _first_date(text)
    if eff is None:
        return []
    return [{"kind": "salary_directive", "subkind": "date_shift_only", "amount": None, "currency": None,
             "effective_date": eff, "message_id": msg["message_id"], "sent_at": msg["sent_at"]}]


def _rule_income_ended(text, msg):
    if not re.search(r"contract has ended|seasonal contract|employment has ended", text, re.I):
        return []
    sent_date = pd.to_datetime(msg["sent_at"]).date() if not pd.isna(msg["sent_at"]) else None
    return [{"kind": "salary_directive", "subkind": "ended", "amount": None, "currency": None,
             "effective_date": None, "sent_date": sent_date,
             "message_id": msg["message_id"], "sent_at": msg["sent_at"]}]


def _rule_arrears(text, msg):
    if "arrears adjustment" not in text.lower():
        return []
    amounts = _all_amounts(text)
    if len(amounts) < 2:
        return []
    (ccy1, amt1), (ccy2, amt2) = amounts[0], amounts[1]
    return [
        {"kind": "salary_directive", "subkind": "confirmed_for_date", "amount": amt1, "currency": ccy1,
         "effective_date": None, "message_id": msg["message_id"], "sent_at": msg["sent_at"]},
        {"kind": "salary_directive", "subkind": "arrears_onetime", "amount": amt2, "currency": ccy2,
         "effective_date": None, "message_id": msg["message_id"], "sent_at": msg["sent_at"]},
    ]


def _rule_rent_increase(text, msg):
    m = PCT_RE.search(text)
    if not m or "rent" not in text.lower():
        return []
    return [{"kind": "category_pct_change", "category": "rent", "pct": m.group(1),
             "message_id": msg["message_id"], "sent_at": msg["sent_at"]}]


def _rule_transfer_internal(text, msg):
    if "transfer between your two accounts" not in text.lower() and "kedua akun" not in text.lower():
        return []
    if not msg.get("related_event_id") or pd.isna(msg.get("related_event_id")):
        return []
    return [{"kind": "exclude_event", "event_id": msg["related_event_id"],
             "message_id": msg["message_id"], "sent_at": msg["sent_at"]}]


def _rule_invoice_approved(text, msg):
    if not re.search(r"client approved|klien menyetujui", text, re.I):
        return []
    ccy, amt = _first_amount(text)
    eff = _first_date(text)
    if amt is None:
        return []
    rel = msg.get("related_event_id")
    if rel and not pd.isna(rel):
        return [{"kind": "event_override", "event_id": rel, "new_status": "scheduled",
                 "new_amount": amt, "new_currency": ccy, "new_date": eff,
                 "message_id": msg["message_id"], "sent_at": msg["sent_at"]}]
    return [{"kind": "new_income_event", "category": "freelance_income", "amount": amt, "currency": ccy,
             "date": eff, "message_id": msg["message_id"], "sent_at": msg["sent_at"]}]


def _rule_debit_failed_still_open(text, msg):
    if "debit attempt failed" not in text.lower() and "upaya debit sebelumnya gagal" not in text.lower():
        return []
    rel = msg.get("related_event_id")
    if not rel or pd.isna(rel):
        return []
    return [{"kind": "event_override", "event_id": rel, "new_status": "pending",
             "message_id": msg["message_id"], "sent_at": msg["sent_at"]}]


def _rule_settlement_confirmed(text, msg):
    """Prize proceeds credited & closed / investment sale settled -> the
    linked event should be treated as settled cash (already reflected, or to
    be reflected, in the account) rather than a still-pending/unrealized
    amount."""
    if not re.search(r"reached your account after withholding|investment sale have settled|"
                      r"hasil penjualan investasi", text, re.I):
        return []
    rel = msg.get("related_event_id")
    if not rel or pd.isna(rel):
        return []
    return [{"kind": "event_override", "event_id": rel, "new_status": "settled",
             "message_id": msg["message_id"], "sent_at": msg["sent_at"]}]


def _rule_scam_ignore(text, msg):
    if (re.search(r"release charge|processing charge|biaya pencairan|biaya pemrosesan", text, re.I)
            and re.search(r"prize|hadiah", text, re.I)):
        return [{"kind": "ignored", "reason": "solicitation/prompt-injection attempt (pay-to-claim prize); "
                                               "not actionable financial evidence",
                 "message_id": msg["message_id"], "sent_at": msg["sent_at"]}]
    return []


_RULES = [
    _rule_scam_ignore,           # check adversarial pattern first
    _rule_salary_increase,
    _rule_salary_temp_reduced,
    _rule_salary_resumed,
    _rule_new_employer_first_salary,
    _rule_household_ended_partial,
    _rule_salary_date_confirmed,
    _rule_income_ended,
    _rule_arrears,
    _rule_rent_increase,
    _rule_transfer_internal,
    _rule_invoice_approved,
    _rule_debit_failed_still_open,
    _rule_settlement_confirmed,
]


def extract_all_directives(messages_df: pd.DataFrame) -> Dict[str, List[dict]]:
    """Returns {user_id: [directive, ...]} for every message in the dataset,
    using deterministic template rules. Messages matching no rule produce no
    directive (safe no-op) but are still counted for reporting."""
    by_user: Dict[str, List[dict]] = {}
    unmatched = 0
    for _, msg in messages_df.iterrows():
        text = str(msg.get("message_text") or "")
        matched = False
        for rule in _RULES:
            out = rule(text, msg)
            if out:
                matched = True
                by_user.setdefault(msg["user_id"], []).extend(out)
        if not matched:
            unmatched += 1
    by_user["_unmatched_count"] = unmatched  # stashed for the usage report
    return by_user
