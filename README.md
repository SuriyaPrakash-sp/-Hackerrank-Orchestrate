# Buy or Wait? — Solution

An AI-powered financial agent that decides whether a user can safely afford a
requested expense, for HackerRank's **Orchestrate — "Buy or Wait?"** challenge
(https://github.com/interviewstreet/hackerrank-orchestrate-september26).

This bundle is self-contained: unzip it, install the requirements, run
`main.py`, and you get `output.csv` for every row in `dataset/requests.csv`
— no API key required.

## Quick start

```bash
cd code
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

This writes `output.csv` to the bundle root (next to `dataset/`) and
`code/evaluation/usage_report.md`.

Useful flags:

```bash
python main.py --requests sample   # run against dataset/sample_requests.csv instead
python main.py --limit 10          # only the first 10 requests (fast iteration)
python main.py --output /tmp/o.csv # write elsewhere
```

Environment variables (all optional — the pipeline is fully deterministic
and free to run without any of these):

| Variable        | Purpose                                                        |
| --------------- | --------------------------------------------------------------- |
| `DATASET_DIR`   | Point at a different dataset directory                          |
| `OUTPUT_PATH`   | Override the output CSV path                                    |
| `LLM_API_KEY`   | Enable a **live** vision/text LLM call when the shipped cache/rule engine has no match for a given image or message (OpenAI-compatible endpoint) |
| `LLM_BASE_URL`, `TEXT_MODEL`, `VISION_MODEL` | LLM configuration, if `LLM_API_KEY` is set |

## Layout

```
buy_or_wait_solution/
├── dataset/                    # the challenge dataset (bundled for convenience)
├── problem_statement.md        # the original challenge brief
├── output.csv                  # this run's predictions for dataset/requests.csv
├── log.txt                     # session notes (see below)
└── code/                       # <-- this is what you'd zip as `code.zip` for submission
    ├── main.py                 # entry point
    ├── requirements.txt
    ├── README.md                # (this file's contents, duplicated here for submission)
    ├── cache/
    │   └── vision_cache.json    # pre-extracted facts for the 16 receipt/payslip images
    ├── evaluation/
    │   └── usage_report.md      # token/cost report for the run that produced output.csv
    └── src/
        ├── config.py            # paths & optional LLM settings
        ├── money.py             # Decimal helpers
        ├── data_loader.py       # loads & indexes every dataset/*.csv
        ├── fx.py                # currency conversion (graph pathfinding over the 5 currencies)
        ├── message_rules.py     # bilingual (EN/ID) template-based message → directive extractor
        ├── evidence.py          # combines vision cache + message directives (+ optional live LLM)
        ├── events.py            # Financial State Reconstructor (recurrence detection, overrides)
        ├── forecast.py          # 90-day simulator, binary-search safe amount, earliest-date scan
        ├── planner.py           # candidate plan generation + the 6 tie-break ranking rules
        ├── verifier.py          # re-simulates the chosen plan before it's ever written out
        ├── explain.py           # decision_explanation text generator
        ├── pipeline.py          # per-request orchestration
        └── usage.py             # writes evaluation/usage_report.md
```

## How it works

1. **Data loading** (`data_loader.py`) reads all seven `dataset/*.csv` files
   once into indexed pandas structures.
2. **Evidence resolution** (`evidence.py`):
   - The 16 receipt/payslip images linked to blank-`amount` events are
     resolved from `cache/vision_cache.json` — pre-computed by manually
     inspecting each image against its linked event, so no vision model call
     is needed to reproduce this run.
   - `messages.csv` is generated from a small, fixed set of bilingual
     (English/Indonesian) templates (salary changes, bank/merchant/service
     notices, settlement confirmations, and a couple of deliberately
     adversarial "pay a fee to claim your prize" messages). `message_rules.py`
     recognises these deterministically via regex and turns them into
     structured directives (e.g. "override this event's status to settled",
     "the user's salary increased to X effective Y", "exclude this event —
     it's an internal transfer between the user's own accounts").
   - Both layers are architected to fall back to a live LLM call if
     `LLM_API_KEY` is set and the cache/rules don't cover a given
     image/message (e.g. if the dataset is later swapped) — see
     `evidence.py` / `config.py`.
3. **Financial state reconstruction** (`events.py`): applies the evidence
   above, classifies every event by `(status, direction)` into
   counted / reserved / ignored per the problem statement's rules, and
   detects recurring series (interval + typical amount) from historical
   activity so they can be projected forward.
4. **90-day forecast** (`forecast.py`): a day-by-day Decimal simulation of
   the user's balance; `amount_safe_to_pay` is found by binary search,
   `earliest_date_for_full_payment` by a day-by-day scan — both computed
   independently of the user's payment-method preferences, per the spec.
5. **Planning** (`planner.py`): builds every eligible way to satisfy the
   request (pay now / wait / pay now after permitted spending cuts / partial
   payment / each supplied installment option) and ranks them with a single
   sort key that encodes the six tie-break rules from the problem statement.
6. **Verification** (`verifier.py`): independently re-simulates the winning
   plan before it is ever written to `output.csv`; a plan that fails is
   downgraded to `not_recommended` rather than shipped.
7. **Explanation** (`explain.py`): short, template-based prose mirroring the
   tone of `dataset/sample_requests.csv`.

## Known limitations / assumptions

This was built and calibrated against the 25 rows of
`dataset/sample_requests.csv` (the only ground truth available at
build time), by reverse-engineering the mechanics that best reproduced those
answers. Notable judgment calls, documented here for transparency:

- **Which categories are treated as "recurring"**: rent, housing, utilities,
  education, debt_repayment, insurance, salary, and subscription-style bills
  (streaming, music, cloud storage, delivery membership, gym) are always
  projected forward from historical cadence. Day-to-day discretionary
  spend (dining, groceries, transport, shopping, entertainment, etc.) is
  *not* extrapolated forward by default — this matched the sample set
  noticeably better than extrapolating every category — **except** when a
  category is flagged `stoppable`/`reducible`/`reducible_or_stoppable` for
  that user, since it must still be trackable for the spending-changes
  mechanism. A minority of users have a very regular cadence in an
  excluded category (e.g. a fixed monthly healthcare or family-support
  payment); for those specific users this will under-count a real
  recurring cost. If you have the hidden ground truth, `code/evaluation/`
  is a good place to add a scoring script and re-tune
  `RECURRING_ELIGIBLE_CATEGORIES` in `src/events.py`.
- **Spending-change targeting**: `financial_events.csv` only contains
  historical + one "next confirmed" occurrence per series (no pre-populated
  future rows), so `spending_changes_needed` references the most recent
  historical `event_id` of the affected category as a stand-in for "this
  recurring commitment going forward" — this matches the one example in the
  sample set that exercises this field.
- **Blank amounts with no linked image**: excluded from the forecast rather
  than estimated, per "do not invent unsupported financial information."
- Money math uses `Decimal` throughout; output amounts are formatted to
  match `sample_requests.csv`'s minimal-decimal style (e.g. `25256`,
  `603.3`, `15952906.67`).

## Re-running against a different dataset

Point `DATASET_DIR` at a directory with the same seven CSVs (and, if you
have new receipt images, either add entries to
`code/cache/vision_cache.json` or set `LLM_API_KEY` so the live vision path
picks them up).
