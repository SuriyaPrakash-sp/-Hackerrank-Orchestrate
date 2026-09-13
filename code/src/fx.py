"""Exchange-rate engine.

exchange_rates.csv only supplies a handful of directed pairs (e.g. EUR->ZAR,
USD->EUR, USD->IDR, USD->INR, EUR->USD), sampled roughly monthly. To convert
between any two of the five dataset currencies (INR, ZAR, IDR, USD, EUR) we
build a small directed graph (inverting a rate to traverse edges backwards)
and pick, for each edge, the rate row whose date is closest to the event's
settlement date.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import date, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .money import D


class FxEngine:
    def __init__(self, exchange_rates: pd.DataFrame):
        # rows_by_pair[(frm,to)] = list of (date, rate) sorted by date
        self.rows_by_pair: Dict[Tuple[str, str], List[Tuple[date, Decimal]]] = defaultdict(list)
        pairs = set()
        for _, r in exchange_rates.iterrows():
            frm, to = r["from_currency"], r["to_currency"]
            d = pd.to_datetime(r["rate_date"]).date()
            rate = D(r["rate"])
            self.rows_by_pair[(frm, to)].append((d, rate))
            pairs.add((frm, to))
        for k in self.rows_by_pair:
            self.rows_by_pair[k].sort(key=lambda x: x[0])

        # build adjacency for pathfinding (both directions traversable)
        self.adj: Dict[str, List[str]] = defaultdict(list)
        for frm, to in pairs:
            self.adj[frm].append(to)
            self.adj[to].append(frm)

    def _nearest_rate(self, frm: str, to: str, on: date) -> Optional[Decimal]:
        rows = self.rows_by_pair.get((frm, to))
        if rows:
            best = min(rows, key=lambda x: abs((x[0] - on).days))
            return best[1]
        rows_inv = self.rows_by_pair.get((to, frm))
        if rows_inv:
            best = min(rows_inv, key=lambda x: abs((x[0] - on).days))
            if best[1] != 0:
                return Decimal(1) / best[1]
        return None

    @lru_cache(maxsize=None)
    def _path(self, frm: str, to: str) -> Optional[Tuple[str, ...]]:
        if frm == to:
            return (frm,)
        seen = {frm}
        q = deque([[frm]])
        while q:
            path = q.popleft()
            node = path[-1]
            for nxt in self.adj.get(node, []):
                if nxt in seen:
                    continue
                new_path = path + [nxt]
                if nxt == to:
                    return tuple(new_path)
                seen.add(nxt)
                q.append(new_path)
        return None

    def convert(self, amount: Decimal, frm: str, to: str, on: date) -> Decimal:
        """Convert `amount` from currency `frm` to `to`, using the rate(s)
        closest to date `on`, chaining through intermediate currencies if a
        direct pair isn't available."""
        if frm == to or amount == 0:
            return amount
        path = self._path(frm, to)
        if not path:
            # Unknown pair: fail safe by returning the amount unconverted
            # rather than crashing the whole run.
            return amount
        value = amount
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            rate = self._nearest_rate(a, b, on)
            if rate is None:
                return amount
            value = value * rate
        return value
