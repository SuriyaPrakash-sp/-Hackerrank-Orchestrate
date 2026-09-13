"""Decimal-based money helpers. Never use binary floats for financial math."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Optional, Union

Number = Union[int, float, str, Decimal]

TWO_PLACES = Decimal("0.01")


def D(value: Optional[Number]) -> Decimal:
    """Safely coerce a value to Decimal. Blank/NaN -> Decimal('0') by caller's choice
    (callers that need to distinguish blank-from-zero must check before calling D)."""
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    try:
        s = str(value).strip()
        if s == "" or s.lower() == "nan":
            return Decimal("0")
        # strip thousands separators defensively
        s = s.replace(",", "")
        return Decimal(s)
    except InvalidOperation:
        return Decimal("0")


def is_blank(value) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.lower() == "nan"


def round2(value: Decimal) -> Decimal:
    return value.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def fmt_amount(value: Decimal) -> str:
    """Format a Decimal amount for CSV output using a minimal representation,
    matching dataset/sample_requests.csv style: round to 2dp, then drop
    trailing zeros (25256 -> '25256', 603.30 -> '603.3',
    15952906.67 -> '15952906.67')."""
    q = round2(value)
    s = f"{q:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"
