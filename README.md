# Litmus

**A real test, not a vibe check.**

Litmus is an LLM evaluation and regression-testing tool. Given a prompt + model
config and a golden test set, it detects when a prompt or model change causes a
statistically significant regression — tracking accuracy, cost, and latency over
time, not just "did the output change."

## Why

Most "LLM eval" projects are a script that diffs two outputs and eyeballs whether
it looks worse. Real AI teams need to know, with statistical confidence, whether
a prompt tweak or model swap actually degraded quality, and by how much it moved
cost/latency — ideally as a CI gate before it ships, not after a user complains.

## Architecture

- **Test case store** — versioned golden dataset (input, expected output/rubric,
  tags), as JSON/YAML files under `testsets/`, so test sets are diffable in git.
- **Runner** — executes each test case against a target (prompt version + model),
  capturing output, token cost, and latency.
- **Scorer** — pluggable strategies per test case: exact/schema match, semantic
  similarity (embeddings), and LLM-as-judge against a rubric. Pluggability is the
  differentiator over a naive string-diff eval.
- **Comparison engine** — given a baseline run and a candidate run, computes
  per-metric deltas with real significance testing (paired bootstrap or McNemar's
  for pass/fail rates, Mann-Whitney for latency/cost) rather than eyeballing
  percentages.
- **History store** — every run persisted with timestamp + prompt/model version,
  so trendlines are possible, not just single comparisons.
- **CI gate** — a GitHub Action that runs the suite on a PR touching prompts/model
  config and fails/comments if a regression is statistically significant.
- **Dashboard** — minimal UI for the latest comparison, historical trends, and
  drill-down into failing cases.

## Stack

- Python, FastAPI for the dashboard
- Pydantic for test-case/result schemas
- `litellm` for multi-provider support and embeddings
- `scipy.stats` for significance testing
- JSON per run as the source of truth, queried via DuckDB (no database server)
- Packaged as a pip-installable CLI (`litmus run`, `litmus compare`)

## Milestones

1. Test-case schema + runner producing raw results for one prompt/model
2. Scoring engine with 2-3 pluggable strategies
3. Comparison engine: baseline vs. candidate with real stats, not just deltas
4. Persistence + trend view across versions
5. GitHub Action CI gate (PR fails/comments on regression)
6. Dashboard
7. README case study: pick one narrow, legible domain (e.g. structured JSON
   extraction or a classification/routing prompt — not open-ended chat) and show
   a real caught regression with a p-value

## CI gate setup

`.github/workflows/eval-gate.yml` runs on any PR touching `testsets/**` or
`src/litmus/**`. It requires one repository secret:

- `OPENAI_API_KEY` — used by `litellm` to actually call a model when running
  the eval suite against the PR branch. (Swap the workflow's `--model` value
  and secret name if you're targeting a different provider.)

The workflow finds the most recent persisted run committed to `runs/` on the
base branch, runs the test set fresh against the PR branch, compares the two
via `litmus compare`, posts the result as a PR comment, and fails the job if
a statistically significant regression is flagged. If the base branch has no
persisted runs yet, the gate is skipped (nothing to compare against).

## Status

Core pipeline implemented and tested (63 tests passing): schema, loader,
runner, real `litellm` execution, the `litmus run`/`litmus compare` CLI
(persisting runs and comparing by run ID), all three scorers, all three
statistical tests, DuckDB-backed trend queries, and the CI gate workflow
(built and locally validated, not yet exercised against a live PR). Remaining:
dashboard, README case study.
