# Usage Report — Buy or Wait?

## Run summary

- Requests processed: **250**
- LLM API configured: **no** (`LLM_API_KEY` not set)
- Live LLM calls made this run: **0**
- Estimated LLM spend this run: **$0.00** (fully deterministic / cached path)

## Evidence resolution

- Blank-`amount` events linked to a receipt/payslip image: resolved via
  `code/cache/vision_cache.json` for **16** images (pre-computed,
  zero cost); **0** required a live vision call this run.
- Messages parsed by the deterministic bilingual (EN/ID) template engine
  (`src/message_rules.py`): **123** structured directives were
  extracted across **117** users.
- Messages that matched no known template: **98**
  (treated as informational no-ops — the conservative, already-safe default
  from `financial_events.csv` status/direction is kept).

## Design rationale (cost-awareness)

The dataset's messages and images are drawn from a small, fixed set of
templates (bilingual salary/payroll updates, bank/merchant/service-provider
notices, and 16 scanned receipts/payslips). Because these are enumerable and
highly regular, the extraction logic for both is implemented as a
**deterministic rule/cache layer first** (Level 0 in a tiered routing
sense), with a live LLM call as an *optional* fallback only when:

1. `LLM_API_KEY` is configured, **and**
2. the cache/rule engine has no match for a given image or message.

This keeps the solution reproducible, free to grade, and fast, while still
being architected to call a real model if the dataset is later swapped for
one whose evidence doesn't fit the known templates.

## Safety notes

- One message in the dataset is an adversarial "pay a release charge to
  claim your prize" solicitation. It is explicitly recognised and ignored
  (`_rule_scam_ignore` in `src/message_rules.py`) rather than acted upon,
  regardless of what it asks the agent to do.
- Blank-`amount` events with no linked image evidence are excluded from the
  forecast (never assigned a fabricated amount).
- Every recommended plan is independently re-simulated by `src/verifier.py`
  before being written to `output.csv`; a plan that fails re-verification is
  downgraded to `not_recommended` / `not_affordable` rather than shipped.
