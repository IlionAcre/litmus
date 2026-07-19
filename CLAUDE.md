# CLAUDE.md — Litmus

This file is the source of truth for any agent working on this project. It
records decisions already made and *why*, so they aren't re-litigated in a
future session, plus explicit guardrails on what not to change without asking.
Treat this as a living decision log: whenever a new architectural decision is
made, add it here with its rationale before moving on.

## Rule: read `AI_docs/` before doing anything else

Before taking any action in this project, read every file in the `AI_docs/`
folder. That folder holds agent-facing reference material — the detailed
execution roadmap (`AI_docs/PHASES.md`) and any other reference docs added
there over time. This file (`CLAUDE.md`) only holds high-level decisions and
guardrails; the operational detail (current phase, next concrete action, exit
criteria) lives in `AI_docs/`, not here.

## What this is

Litmus is an LLM evaluation and regression-testing tool. Given a prompt/model
config and a golden test set, it detects when a change causes a statistically
significant regression in accuracy, cost, or latency — not just "the output
looks different." See `README.md` for the user-facing pitch.

## Brand

Name: **Litmus**. Tagline: **"A real test, not a vibe check."** Don't rename
the project or change the tagline without the user explicitly asking.

## Architecture decisions (with rationale)

- **Storage: JSON per run + DuckDB as the query layer.** Each run is written
  as a JSON file (timestamp, prompt/model version, scores, cost, latency,
  per-case results) — git-diffable, human-readable, works identically
  whether written by a local run or a CI job. DuckDB queries the directory of
  JSON files directly for trends/comparisons/drill-downs, with no server
  process of its own.
  - Rejected: **SQLite as source of truth** — CI runners are ephemeral, a
    `.db` file has nowhere durable to live between runs.
  - Rejected: **S3 / hosted Postgres** — unnecessary infra for this project.
    (Earlier reasoning floated S3 to reuse a skill demonstrated in other
    portfolio projects; that reasoning no longer applies since those projects
    are being dropped from the portfolio. Judge storage choices on their own
    merits, not on skill-reuse narrative.)

- **API/dashboard: FastAPI**, explicitly kept per user preference. A
  static-HTML-only v1 was considered and rejected — the user wants a real
  running service. Default assumption: run locally via `litmus serve` first;
  it's deployable later the same way as the Portfolio-WebPage project
  (container + Cloud Run) without redesign, since DuckDB-over-JSON doesn't
  care whether the process is local or containerized.

- **CLI: `typer`**, not `argparse`/`click` directly — type-hint/Pydantic-native,
  consistent with the rest of the stack.

- **Multi-provider LLM calls + cost tracking: `litellm`** — specifically for
  its built-in `completion_cost()`, avoiding hand-rolled per-provider pricing
  tables.

- **Statistics: `scipy.stats`** for Mann-Whitney and bootstrap CIs. McNemar's
  test (for paired pass/fail rate comparisons) is implemented manually as a
  closed-form chi-square test rather than pulling in `statsmodels` for one
  function.

- **Semantic-similarity scoring: embeddings via `litellm.embedding()`**, not a
  local `sentence-transformers` model. Keeps the stack on one provider-access
  library instead of adding a heavy local-ML dependency (`torch`) for a single
  scorer. Trade-off accepted: this scorer needs network access, same as every
  other `litellm`-backed piece of the stack.

- **Python 3.12**, `uv`-managed, own git repo — matches the convention of
  sibling projects in the parent Portfolio directory.

- **`litellm` pinned to `<1.90`.** Versions >=1.90 (tested: 1.92.0, 1.93.0)
  bundle a Rust-accelerated component (`litellm-rust`) with no prebuilt wheel
  for win_amd64/cp312, forcing a source build that fails on this machine's
  Rust toolchain (`link: extra operand` — Git Bash's coreutils `link` shadows
  MSVC's `link.exe` on PATH). `litellm==1.89.6` installs from a wheel with no
  build step and has the full API surface this project needs
  (`completion`, `completion_cost`, `embedding`). Revisit this pin if the
  Windows dev environment's Rust/linker setup ever gets fixed, or if a future
  litellm release ships Windows wheels again — don't just bump blindly.

## Known gotcha: real wall-clock timing in tests is flaky

`litellm_call` (`llm.py`) measures latency via real `time.perf_counter()`.
Any test that runs it (even with `litellm.completion` mocked) gets real,
variable timing — under system load (e.g. running the full suite vs. a
single file) this noise alone can spuriously trip latency-based statistical
significance in `compare_runs()`, independent of any real behavior change.
Confirmed: `tests/test_cli.py`'s `compare` tests failed intermittently only
when run as part of the full suite, not in isolation. Fix used there:
monkeypatch `time.perf_counter` to a deterministic incrementing counter
(`itertools.count`) so latency is constant across baseline/candidate and the
test only exercises the behavior it's actually meant to test. Apply the same
pattern to any future test that exercises the live run→score→compare path
and asserts on pass/fail of the comparison result.

## Known gotcha: DuckDB schema inference on empty JSON arrays

`read_json_auto` infers a generic `JSON` type (not a typed struct list) for a
run whose `results`/`scores` arrays are empty (a run with zero test cases),
which then fails `avg()` with "no function matches avg(JSON)" — either for
that file alone, or for the whole glob once mixed with populated runs.
Found via manual reproduction while validating Phase 8's CI baseline-lookup
logic. Fixed in `trends.py` by declaring the JSON schema explicitly via
`read_json(..., columns={...})` instead of relying on `read_json_auto`'s
inference — this sidesteps per-file type inference entirely. `RunTrendPoint`'s
`pass_rate`/`mean_latency_ms`/`mean_cost_usd` are `float | None` because of
this (an average over zero test cases is undefined, not zero). Apply the
same explicit-schema pattern to any future DuckDB query added over `runs/`.

## Guardrails — don't do these without asking first

- Don't introduce a database server (Postgres, hosted SQLite-as-a-service,
  etc.) — the JSON + DuckDB approach is deliberate, not a placeholder.
- Don't add cloud infra (S3, queues, etc.) unless a specific need actually
  arises later — this project should stand alone.
- Don't swap FastAPI for Streamlit or a static-HTML-only dashboard — that
  tradeoff was already made and reversed once.
- Don't add scoring/statistics dependencies beyond `scipy` without checking
  first (e.g. don't reach for `statsmodels` just for McNemar's).
- Don't rename the project or change the tagline.
- Don't skip ahead in the milestone order (e.g. don't build the dashboard
  before the runner/scorer/comparison engine exist) without confirming.

## Roadmap / phase status

Detailed, step-by-step phases (entry/exit criteria, file-level tasks,
verification commands) live in **`AI_docs/PHASES.md`** — that file is the
actual execution roadmap and the current status/resume point. This section
stays high-level only; don't duplicate phase detail here.

1. Test-case schema + runner — **done**
2a/2b. Real execution + first CLI command — **done**
3a/3b. Scoring interface + exact/schema match — **done**
4. Scoring: semantic similarity — **done**
5. Scoring: LLM-as-judge — **done**
6a/6b/6c. Comparison engine with real statistical tests — **done**
7a/7b. Persistence (JSON results) + DuckDB trend queries — **done**
8. CI gate via GitHub Action — **done (built + locally validated only — not
   pushed or tested against a live PR; needs user go-ahead)**
9. Dashboard (FastAPI + DuckDB) — **done**
10. README case study with a real caught regression — **done**

(Phases 2, 3, 6, and 7 are each split into sub-phases in `AI_docs/PHASES.md`
— see that file for the full 16-checkpoint breakdown, the rationale for each
split, and the authoritative current status table/resume point. This list is
a high-level summary only and can lag — `AI_docs/PHASES.md` is truth.)

## New decision: `GEMINI_API_KEY` is the project's real provider credential

Phase 10 required making real LLM calls. The user added a real
`GEMINI_API_KEY` to `Litmus/.env` (gitignored — never read/print/log that
file or its value). `python-dotenv` was added as a dependency and `cli.py`
calls `load_dotenv()` at module level, so any `uv run litmus ...` invocation
has the key available automatically. The proven working model string is
`gemini/gemini-2.5-flash-lite` (routed correctly by `litellm` via the
`gemini/` provider prefix — no code changes needed). Note:
`gemini-2.0-flash`/`gemini-2.0-flash-lite` (the models originally assumed)
are deprecated as of this writing and return a 404 — use the 2.5 line.
The CI gate workflow (`eval-gate.yml`) and README quickstart were updated to
reference `GEMINI_API_KEY`/`gemini/gemini-2.5-flash-lite` to match.

## Status

All 16 checkpoints complete (72 tests passing). Full pipeline built and
tested: schema, loader, runner, real litellm execution, the `litmus
run`/`litmus compare`/`litmus serve` CLI commands, all three scorers, all
three statistical tests, DuckDB-backed trend queries, the CI gate workflow,
the FastAPI dashboard, and a real README case study (real Gemini calls,
p=0.00087 caught regression — see `README.md`). Remaining, not gaps but
explicit scope boundaries: Phase 8's workflow is locally validated only, not
pushed/tested against a live PR; no rendered dashboard frontend; no hosted
deployment. See `AI_docs/PHASES.md` for full detail.
