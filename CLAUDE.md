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

- **Statistics: `scipy.stats`.** McNemar's test (for paired pass/fail rate
  comparisons) is implemented manually as a closed-form chi-square test
  rather than pulling in `statsmodels` for one function. Latency/cost use a
  **paired bootstrap CI** for the mean difference, not Mann-Whitney U.
  - Revised decision (code review pass): Mann-Whitney U was removed
    entirely (`stats.py`/`mann_whitney_test`/`MannWhitneyResult` deleted,
    along with its tests) because it's the wrong tool for this project's
    only use case — latency/cost deltas are **paired** data (same
    `test_case_id`, before vs. after), and Mann-Whitney U is an
    independent-samples test. Using it silently under-used the pairing
    (still produced a valid p-value, just a methodologically weaker one).
    `bootstrap_diff_ci` already existed but was *also* wrong in the same way
    (it resampled baseline and candidate arrays independently rather than
    resampling per-unit differences) and was unused dead code besides. Fixed
    both problems in one move: `bootstrap_diff_ci` now resamples the
    per-unit differences (`candidate[i] - baseline[i]`) directly — a genuine
    paired bootstrap — and `compare.py` uses it for `latency_ms`/`cost_usd`.
    `MetricComparison` now carries either `p_value` (pass_rate) or
    `ci_low`/`ci_high` (latency_ms, cost_usd), never both; CLI/API output
    changed accordingly (`p=...` vs `95% CI=[...]`).

- **Semantic-similarity scoring: embeddings via `litellm.embedding()`**, not a
  local `sentence-transformers` model. Keeps the stack on one provider-access
  library instead of adding a heavy local-ML dependency (`torch`) for a single
  scorer. Trade-off accepted: this scorer needs network access, same as every
  other `litellm`-backed piece of the stack.

- **Python 3.12**, `uv`-managed, own git repo — matches the convention of
  sibling projects in the parent Portfolio directory.

- **Per-case error isolation, not batch-crashing.** `RunResult` and
  `ScoreResult` each carry an `error: str | None` field. A single test
  case's LLM call or scoring blowing up (rate limit, bad model name, auth
  failure, an unparseable LLM-judge response, ...) no longer crashes
  `litmus run` and loses every already-computed result — it's caught in
  `cli.py`'s `_run_and_score`, recorded with `error` set (raw_output/passed
  are meaningless sentinel placeholders in that case, not a real verdict),
  printed as `[ERROR]` (distinct from PASS/FAIL, never coerced into either),
  summarized at the end of the run's output, and the run is still persisted
  with whatever cases did succeed. `compare.py`'s alignment then excludes
  any case where either run recorded an error for it (see next item) so an
  errored case can't quietly count as a real pass/fail in the statistics.
  Fixed (was an accepted gap, now resolved): `trends.py`'s aggregates
  previously did **not** exclude errored cases the same way `compare.py`
  does — an errored case's sentinel 0.0 latency/cost/failed-pass dragged
  every trend average down, making a run with real failures look
  artificially *better* on cost/latency, not just incomplete. Fixed by
  excluding by error field independently per metric in `_TREND_QUERY`'s
  three subqueries: `pass_rate`'s subquery filters `WHERE s.error IS NULL`,
  `mean_latency_ms`/`mean_cost_usd`'s subqueries filter
  `WHERE r.error IS NULL` — independently, not jointly, since a
  scoring-only failure (`ScoreResult.error` set, `RunResult.error` unset)
  has real latency/cost data that should still count even though it's
  excluded from pass_rate.

- **Comparison alignment is visible, not silently narrowed.**
  `ComparisonReport` carries `common_case_count`, `baseline_only_ids`,
  `candidate_only_ids`, and `errored_ids`. Previously, `compare.py`'s
  `_align()` silently dropped any `test_case_id` not present in both runs
  with zero visibility — a testset changing between baseline/candidate runs
  (normal usage: cases added/removed) could narrow the comparison to a small
  overlap that looked identical to a full one in the output. Now the CLI
  (`compare` command) prints an explicit `WARNING: comparing N common test
  case(s)` block listing what was excluded and why whenever
  `report.has_mismatched_cases` is true, and the API includes the same
  fields in its JSON response — never buried in a field nobody surfaces.

- **Scorer defaults are Gemini models, matching the project's real
  credential.** `SemanticSimilarityScorer`'s default was
  `text-embedding-3-small` (OpenAI) and `LlmJudgeScorer`'s was `gpt-4o-mini`
  (OpenAI), while the only real provider credential anywhere in this project
  is `GEMINI_API_KEY` — neither scorer had ever actually been exercised
  against a real provider, only mocks. Changed defaults to
  `gemini/text-embedding-004` and `gemini/gemini-2.5-flash-lite`
  respectively (both overridable via constructor args) so they work out of
  the box against what's actually configured here.

- **`litellm` pinned to `<1.90`.** Versions >=1.90 (tested: 1.92.0, 1.93.0)
  bundle a Rust-accelerated component (`litellm-rust`) with no prebuilt wheel
  for win_amd64/cp312, forcing a source build that fails on this machine's
  Rust toolchain (`link: extra operand` — Git Bash's coreutils `link` shadows
  MSVC's `link.exe` on PATH). `litellm==1.89.6` installs from a wheel with no
  build step and has the full API surface this project needs
  (`completion`, `completion_cost`, `embedding`). Revisit this pin if the
  Windows dev environment's Rust/linker setup ever gets fixed, or if a future
  litellm release ships Windows wheels again — don't just bump blindly.
  - Revised (made platform-conditional): confirmed via PyPI's JSON API that
    `litellm` 1.92.0 ships real prebuilt `manylinux_2_28` wheels for
    `cp312` (both x86_64 and aarch64) with no build step needed, alongside
    the confirmed-missing `win_amd64` wheel. The Rust-build failure is
    genuinely Windows-only, so the blanket `<1.90` pin was unnecessarily
    capping Linux (including this project's own GitHub Actions CI runner)
    at an older version for no real reason. Changed to two PEP 508
    marker-conditioned entries in `pyproject.toml`: `litellm<1.90;
    sys_platform == 'win32'` and `litellm; sys_platform != 'win32'`.
    Verified `uv lock`/`uv sync`/the full test suite still work correctly
    on this Windows machine afterward. Note: the regenerated `uv.lock`
    still resolves to `1.89.6` on *every* platform right now, not just
    Windows, since `uv lock`'s incremental resolution prefers keeping an
    already-valid pinned version over gratuitously bumping it, and
    `1.89.6` still satisfies the now-unbounded non-Windows constraint too.
    This change only removes the artificial ceiling for the future, it
    doesn't itself force an immediate version bump. A future
    `uv lock --upgrade-package litellm` would let non-Windows environments
    move past 1.90 without touching the Windows constraint, whenever
    there's an actual reason to (no confirmed bugfix in 1.90+ relevant to
    this project's usage has been identified yet).

- **Structured logging: stdlib `logging` + a custom JSON-lines formatter, no
  new dependency.** Matches the same avoid-a-dependency-for-something-stdlib-
  can-do precedent as the McNemar's/`statsmodels` decision above. Writes to
  `logs/litmus.jsonl` (gitignored — a runtime artifact, unlike `runs/` which
  holds meaningful committed demo data) via a stdlib
  `logging.handlers.RotatingFileHandler` (5MB, 3 backups). Configured once
  via a Typer `@app.callback()` in `cli.py` (`--log-file`/`--log-level`,
  both with `envvar=` — `LITMUS_LOG_FILE`/`LITMUS_LOG_LEVEL`, mirroring the
  existing `LITMUS_RUNS_DIR` pattern), which covers `run`/`compare` and, since
  `litmus serve` runs `uvicorn.run` in-process, `api.py`'s route handlers too
  (all via `logging.getLogger("litmus")`). `configure_logging()` closes and
  clears any handlers already on that logger before adding a new one — it's
  re-invoked on every CLI call in the same process (every test in a pytest
  session included), and without clearing first, handlers would accumulate
  (duplicate log lines) and leak file handles onto deleted `tmp_path` dirs.
  `JsonFormatter.format()` uses `json.dumps(..., default=str)` since extra
  log fields aren't guaranteed JSON-native (e.g. `testset_dir` is a `Path`).
  Logged events: `run_started`, `case_result` (PASS/FAIL/ERROR),
  `run_completed`, `comparison_performed`, `comparison_mismatch`,
  `run_saved`, and one `api_request`/`api_error` pair per API route — all
  drawn from data already surfaced via the existing `typer.echo`/HTTP output,
  never expanding what's exposed, just adding a durable structured sink.
  Never logs `.env` contents or `GEMINI_API_KEY` (`runner.py`/`llm.py` stay
  untouched — deliberately thin passthroughs with no logging of their own;
  all log calls sit at the orchestration layer where context is already
  assembled). Scope boundary: `api.py` imported directly (e.g. `TestClient`
  in tests, or bypassing `litmus serve`) won't have file logging configured —
  acceptable, since an unconfigured logger is a silent no-op, not a crash,
  and `litmus serve` is the documented way to run it.
  - Revised (follow-up review pass): thread-safety of concurrent logging was
    previously only a documented-and-true claim (verified by reading stdlib's
    `Handler.handle()` source, never exercised by a test) — now covered by
    `test_concurrent_logging_from_multiple_threads_produces_no_corrupt_lines`
    (8 threads × 50 log calls, asserts every line is independently parseable
    and none are lost). Two CLI/storage observability gaps also closed: `run`
    now catches `TestCaseLoadError` and `compare` now catches
    `FileNotFoundError` from `load_run` — both log an ERROR event
    (`testset_load_failed`/`run_load_failed`) and exit cleanly (`Error: ...`,
    code 1) instead of an uncaught traceback that never reached the log file.
    `storage.load_runs` now logs `run_load_failed` for a corrupt/unparseable
    run file before re-raising the same exception (behavior unchanged, just
    now visible in the log).

- **A 4th comparison metric (`mean_score`) and a `power_warning` field.**
  `ScoreResult.score` (a continuous value populated for every scored case)
  was never actually read by `compare_runs()` — only the boolean `passed`
  was — so a model quietly getting less confident/similar before enough
  cases flip to move the pass rate went completely undetected. `mean_score`
  closes this gap using the *existing* paired `bootstrap_diff_ci` (no
  `stats.py` changes needed), flagged when the score got worse (lower) and
  the CI excludes zero, included in `any_flagged`. Caveat: `score` is only a
  genuinely continuous signal for `SemanticSimilarityScorer` (raw cosine
  similarity) or `LlmJudgeScorer` (the judge's own confidence score, see
  revised decision below) — `ExactMatchScorer` and `JsonSchemaMatchScorer`
  still set `score = 1.0/0.0` mirroring `passed`, so for a testset scored
  entirely by one of those two, `mean_score` is mathematically close to
  redundant with `pass_rate`. `TestCase.scorer` is per-case, so a testset can
  mix scorer types — in that case `mean_score` pools continuous and
  boolean-mirrored values into one aggregate, still informative but less
  interpretable than a homogeneously-scored testset.
  Separately, the tool never told you whether a test set was even large
  enough for its own statistics to be trustworthy — a 2-case test set got
  the same confident-looking p-value treatment as a 200-case one.
  `power_warning: str | None` (purely advisory — never affects `any_flagged`
  or exit code) fires on two independent, parameterized thresholds
  (`compare_runs(..., min_case_count: int = 10, min_discordant_pairs: int =
  10)`): too few total common cases for the bootstrap-based metrics, or too
  few McNemar discordant pairs (`b + c`, already computed by `mcnemar_test`)
  for the chi-square approximation. These are heuristic thresholds, not a
  formal statistical power calculation (that would require assuming a
  target effect size, which nothing here does). Both surfaced in `cli.py`'s
  `compare` output (a `NOTE:`-prefixed block, deliberately distinct from the
  `WARNING:`-prefixed mismatch block) and in `api.py`'s `_report_dict()`
  (hand-picked keys, not a blanket `asdict(report)` — adding a field to
  `ComparisonReport` alone does **not** make it appear in the API response,
  `_report_dict()` must be edited too).
  - Revised decision: both deferred follow-ups are now closed.
    `LlmJudgeScorer` emits a real continuous `score` (0.0-1.0, the judge's
    own confidence) instead of mirroring `passed` — `_JudgeVerdict` dropped
    `passed` from the judge's JSON contract entirely (`{"score": ...,
    "rationale": ...}`), and `passed = score >= threshold` is derived the
    same way `SemanticSimilarityScorer` derives its boolean, rather than
    asking the judge for a separate, independently-judged boolean (which
    risked self-contradiction, e.g. `passed: true, confidence: 0.2`).
    `threshold` defaults to `0.5` (a neutral midpoint — deliberately not
    copying `SemanticSimilarityScorer`'s `0.8`, since there's no reason to
    assume the same bias fits a judge's confidence scale) and is a
    constructor arg, same pattern. Score validated via pydantic
    `Field(ge=0.0, le=1.0)`, so an out-of-range value raises
    `ValidationError`, already caught by the existing parse `except` block —
    no structural change needed there. `mean_score` is now genuinely
    informative for `llm_judge`-scored cases, not just
    `semantic_similarity`-scored ones. This is a breaking change to the
    judge's wire contract, made deliberately — no external users yet, and
    keeping both a separate `passed` and `confidence` risked
    self-contradiction for no real benefit.

    `mcnemar_test()` gained an `exact_threshold: int = 25` parameter
    (heuristic, not a universal rule — separate from and not to be confused
    with `power_warning`'s `min_discordant_pairs=10`: that threshold asks
    "is there enough evidence for any test to have power," this one asks
    "which formula is still valid at this sample size"). Below the
    threshold, `b + c` discordant pairs get the **exact binomial test**
    (`scipy.stats.binomtest` — already an approved dependency, not
    `statsmodels`) instead of the continuity-corrected chi-square
    approximation, which is known to be unreliable at small samples.
    `McNemarResult` gained `method: str` (`"no_discordant_pairs"` /
    `"exact_binomial"` / `"chi_square"`) so callers can see which path ran.
    This required restructuring `tests/test_stats_mcnemar.py`: the existing
    `b=10, c=2` test framed itself as an *independent* cross-check of the
    chi-square implementation against `scipy.stats.binomtest` — but once the
    implementation itself started calling `binomtest` for that same
    small-sample case, the check silently stopped being independent (it
    became a direct-equality check against what the code now literally
    calls). Verified directly, not assumed: exact p=0.03857421875 vs.
    chi-square p~=0.04331 for that case — different formulas, same
    conclusion, which is exactly why small samples get routed differently.
    Reframed that test honestly as a direct-equality check, and added a new
    `b=20, c=8` (28 discordant pairs, above the threshold) test that
    preserves a genuine independent cross-check on the chi-square path
    (exact p~=0.03570 vs. chi-square p~=0.03764 — close, not identical,
    both agree on significance).
  - Follow-up review pass caught two more real gaps in this same feature,
    both now fixed:
    1. **`McNemarResult.method` never actually reached any output** — it was
       computed correctly but discarded before reaching `MetricComparison`
       (which had no `method` field at all), so a user had no way to tell
       which formula produced a given `pass_rate` p-value. Fixed: added
       `method: str | None = None` to `MetricComparison` (only meaningful
       for `pass_rate`; `None` for the bootstrap-based metrics), threaded
       through in `compare_runs()`, printed in `cli.py` as a suffix on the
       `p=...` line (e.g. `p=0.0386 (exact_binomial)`). Reaches the API for
       free via `_report_dict()`'s existing `asdict(report.pass_rate)`.
    2. **Every comparison threshold was unreachable from the CLI.** `alpha`,
       `confidence`, `min_case_count`, `min_discordant_pairs` were real
       `compare_runs()` parameters nobody could actually override without
       calling the Python API directly, and `compare_runs()` didn't even
       have an `exact_threshold` parameter to forward to `mcnemar_test()`.
       Fixed: added `exact_threshold: int = 25` to `compare_runs()`, and
       exposed all five as `litmus compare` options (`--alpha`,
       `--confidence`, `--min-case-count`, `--min-discordant-pairs`,
       `--exact-threshold`), documented in `README.md`'s "CLI reference"
       section.
  - **A live smoke test uncovered a real, separate bug** in the judge
    confidence-score feature above: Gemini routinely wraps its JSON response
    in a markdown code fence (```` ```json ... ``` ````) even when the
    prompt says "respond with ONLY a JSON object" — `json.loads` can't parse
    that, so **every real judge call would have failed** with
    `JudgeParseError` regardless of how well-calibrated the actual score
    was. No existing test caught this because all of them mock
    `litellm.completion` with hand-crafted clean JSON strings, never real
    model output quirks. Fixed with a `_strip_markdown_code_fence()` helper
    applied before `json.loads`. Once fixed, the actual calibration question
    (was this whole feature worth it, or does an LLM judge just cluster at
    0.0/0.5/1.0 when asked to self-report confidence?) was tested for real
    against `gemini/gemini-2.5-flash-lite` — a fixed customer-service rubric
    against 6 outputs spanning clearly-satisfying to clearly-failing,
    including 4 deliberately borderline/partial ones. Real scores returned:
    `1.0, 0.0, 0.3, 0.4, 0.4, 0.6` — genuine gradation, not clustering at the
    extremes, and the rationale text correctly tracked which specific
    rubric criteria each partial case missed. Conclusion: the judge
    confidence score works as intended in practice, not just in code; no
    prompt revision was needed.
  - **A further review pass found three more real gaps, all now fixed:**
    1. **The markdown-fence stripper was overfit to the one failure mode it
       was built from.** The original line-based version broke on a
       single-line fence with no internal newline (treated the whole line
       as "the opening marker" and deleted it wholesale, leaving `''`) and
       on prose before the fence (`"Here is my answer:\n\`\`\`json\n{...}\n\`\`\`"`
       — a realistic model habit even when told to respond with ONLY the
       JSON; `startswith("\`\`\`")` is `False` there, so the text came back
       completely untouched). Fixed by switching to a regex
       (`` ```(?:\w*\n)?(.*?)``` `` with `re.DOTALL`) that searches for a
       fenced block anywhere in the text rather than assuming the markers
       are the first/last *lines* — handles both cases plus every
       previously-working variant, verified directly.
    2. **No range validation on the new CLI thresholds.**
       `litmus compare --confidence 1.5` crashed with a raw, unhandled
       `ValueError` from `numpy.percentile` inside `bootstrap_diff_ci`
       (confirmed live: empty stdout, exit 1, no useful message);
       `--confidence` below 0 or `--alpha` outside `[0,1]` didn't crash but
       silently produced meaningless results (a degenerate always-flagged
       CI, or McNemar's always/never "significant"). Fixed via typer's
       `min=`/`max=` on the `--alpha`/`--confidence` options (now a clean
       usage error, exit code 2, not a crash) and `min=0` on the three
       integer thresholds.
    3. **Prompt injection into `LlmJudgeScorer` was structurally possible.**
       `case.rubric`/`result.raw_output` were embedded unescaped into the
       judge prompt — a system-under-test that itself echoes injected
       content (plausible when test data comes from real production inputs,
       as the README's own case study does with real support tickets) could
       in principle manipulate the judge's verdict. This is a two-hop risk
       (the SUT must itself be injectable, and the injected content must
       survive into the graded output), not a trivial one-step exploit, and
       full immunity isn't realistically achievable for any LLM-judge
       pattern — so the fix is a low-cost mitigation, not a heavy defense:
       `result.raw_output` is now wrapped in an explicit
       `<output_to_grade>...</output_to_grade>` delimiter with a sentence
       telling the judge everything inside is untrusted content to grade,
       not instructions to follow. Documented here as a known, accepted
       residual risk — not a guarantee of immunity — since this tool's
       normal usage (a single person authoring and running their own local
       test sets) makes it low-exploitability in practice, but real for the
       CI-gate-on-production-data use case.

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

## Known gotcha: `actions/checkout`'s narrow fetch refspec breaks `origin/<ref>` lookups

`.github/workflows/eval-gate.yml`'s baseline-lookup step used to `git fetch
origin "$BASE_REF" --depth=1` then `git archive "origin/$BASE_REF" -- runs`.
This silently never worked: `actions/checkout@v4` configures
`remote.origin.fetch` as a **narrow, single-ref refspec** for only the
checked-out PR branch (confirmed by reading `ref-helper.ts` in
`actions/checkout`'s source — unqualified branch refs get
`+refs/heads/<branch>:refs/remotes/origin/<branch>`, not a wildcard). A
later `git fetch origin "$BASE_REF"` for a *different* ref (the base branch,
not what was checked out) has no matching configured refspec, so it only
populates `FETCH_HEAD` — `origin/$BASE_REF` never resolves, and the
`git archive` call fails with `fatal: not a valid object name`, silently
swallowed by the `2>/dev/null || true` guard meant only for the
"no `runs/` dir on this branch yet" case. Net effect: the gate always
printed "no baseline found" and never actually compared anything. Confirmed
by reproducing both the bug and the fix in an isolated local repo with
`actions/checkout`'s exact narrow-refspec configuration (not just a plain
`git clone`, which sets up a wildcard refspec by default and masks this
entirely). Fixed by referencing `FETCH_HEAD` instead of `origin/$BASE_REF` —
`FETCH_HEAD` is unconditionally populated by any `git fetch`, regardless of
refspec configuration. Apply the same pattern (reference `FETCH_HEAD`, not
an assumed remote-tracking ref) to any future CI step that fetches a ref
other than the one `actions/checkout` already checked out.

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
8. CI gate via GitHub Action — **done (built, locally validated, and
   verified on a real PR against GitHub Actions; see the CI gate decision
   entries below)**
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

## New decision: consolidated config via standalone `litmus.toml`

Every tunable value in Litmus was a scattered hardcoded default: 11
`typer.Option` defaults across `cli.py`'s three commands, `DEFAULT_MODEL`/
`DEFAULT_THRESHOLD` in two scorer modules, `bootstrap_diff_ci`/
`mcnemar_test`'s numeric defaults, `logging_config.py`'s rotation settings,
`storage.py`'s `DEFAULT_RUNS_DIR` — none overridable except by editing
source or passing CLI flags one invocation at a time. New module
`src/litmus/config.py`: `LitmusConfig` (frozen dataclass, one field per
scattered value) + `load_config()`, reading a standalone `litmus.toml` at
the project root via stdlib `tomllib` (no new dependency —
`requires-python = ">=3.12"` guarantees it). Precedence: **CLI flag >
`litmus.toml` > hardcoded fallback**. A missing file/key all fall back
gracefully — no override is ever required; the actual `litmus.toml` ships
fully commented-out as documentation of every available key, not a live
override.

- **Revised from the original design**: this lived in `pyproject.toml`'s
  `[tool.litmus]` section at first (reusing the file that already holds
  `[tool.pytest.ini_options]`). Moved to a standalone file after explicitly
  weighing the tradeoff: tying Litmus's own config to `pyproject.toml`
  presupposes the *system under test* is itself a Python project, which
  isn't guaranteed — Litmus's actual pitch is testing any LLM-backed system
  (a Node backend, a bare API, anything), so a project being evaluated by
  Litmus might have no `pyproject.toml` at all, or an unrelated one. A
  standalone `litmus.toml` (flat top-level keys, no `[tool.litmus]`
  wrapper needed since the whole file belongs to Litmus, `[scorers.*]` for
  scorer-specific settings) mirrors the well-established `ruff.toml`/
  `pyproject.toml` precedent. No external users existed yet, so this was a
  full migration, not dual-location support — one unambiguous config
  location, no precedence-between-two-files complexity to reason about.

Config-file resolution is deliberately a CLI-layer concern only —
`stats.py`/`compare.py`/the scorer classes keep their exact original
function signatures and hardcoded defaults, staying pure and config-unaware.
Only `cli.py` (every `typer.Option`'s default now reads `CONFIG.<field>`
instead of a literal) and `scoring/registry.py` (`SemanticSimilarityScorer`/
`LlmJudgeScorer` now constructed with `model=`/`threshold=` from `CONFIG`)
touch the new module, translating resolved config into concrete arguments
before calling the existing, untouched business logic — kept the blast
radius small.

`load_config()` validates range-constrained fields itself (`alpha`/
`confidence` in `[0,1]`; `min_case_count`/`min_discordant_pairs`/
`exact_threshold`/`n_resamples`/`max_workers`/`port` non-negative) and
raises `ConfigError` naming `litmus.toml` and the offending key directly.
This was confirmed necessary, not just cautious:
tested empirically that Typer/Click's own `min=`/`max=` option constraints
*do* apply to a programmatically-supplied default the same as a
user-supplied value (so a bad config value can't silently produce nonsense
results either way) — but the resulting error message blames the CLI flag
(`Invalid value for '--alpha'`) even when the user never touched that flag
and the bad value came from the config file. Typer's constraint stays as a
safety net; `config.py`'s validation is the actual, accurate error surface.

`compare_runs()`/`mcnemar_test()`'s `n_resamples`/threshold values were
already real function parameters but had no CLI exposure at all before this
— added `--n-resamples` on `compare` for genuine completeness, matching the
pattern of the other five threshold options added in the prior review pass.

## New decision: concurrency for `litmus run` via `ThreadPoolExecutor`

`litmus run` processed test cases strictly sequentially — one blocking LLM
call at a time, confirmed by reading `_run_and_score` directly. Confirmed no
case depends on any other case (each is a pure, independent per-case
pipeline). The per-case logic (both try/except blocks) was extracted into
`_process_case(case, target) -> tuple[RunResult, ScoreResult]`, which never
raises (both exception paths already convert failures into sentinel error
results) — this property is what makes it safe to run inside a thread pool
with no extra exception plumbing. `_run_and_score(cases, target,
max_workers=...)`: sequential `for` loop when `max_workers<=1` (identical to
the original code, zero new risk on that path); otherwise
`ThreadPoolExecutor(max_workers=...).map(...)`, which returns results in
*input* order regardless of completion order, so results/scores come back
correctly aligned with no manual reordering needed.

**`ThreadPoolExecutor` chosen over `asyncio`/`litellm.acompletion`**
specifically because every existing test mocks the *synchronous*
`litellm.completion` directly via `monkeypatch.setattr` — confirmed the
installed `litellm==1.89.6` genuinely exposes `acompletion`/`aembedding`,
but switching to them would have forced every existing test's mock to
become async-compatible. A thread pool around the existing synchronous call
chain gets the concurrency without touching `llm.py`, `runner.py`, or a
single existing test's mocking convention.

New `LitmusConfig.max_workers` field, hardcoded fallback **4** — a genuine
default improvement (not just an opt-in flag nobody finds), sized modestly
to reduce real-world provider rate-limit exposure rather than maximize raw
throughput. New `--max-workers` option on `run` (`min=1`). Manually verified
with a real timing comparison (24 simulated-latency cases): **4.25x speedup**
at `max_workers=4` vs. `max_workers=1`, with byte-for-byte identical
per-case ordering and pass/fail results between the two — concurrency is a
performance change only, never a behavior change.

Two thread-safety subtleties were checked empirically, not assumed, during
planning:
- The shared `SCORERS` singleton instances only hold read-only
  `self.model`/`self.threshold` set at construction, never mutated during
  `.score()` — safe to call concurrently with no locking.
- `monkeypatch.setattr("module.attr", ...)` mutates the actual module
  object in memory, shared process-wide regardless of thread — confirmed
  via a throwaway test that a monkeypatched function called from inside a
  `ThreadPoolExecutor` worker is genuinely the patched version.
- **Caveat that matters for any future test in this area:**
  `_pin_deterministic_timing`'s shared `itertools.count()` is thread-safe
  for *uniqueness* (no duplicates/races across concurrent `next()` calls,
  confirmed) but is **not** deterministic *per-case* once workers
  interleave — a sequential run gives every case the exact same latency by
  construction; a concurrent run gives varying, non-reproducible values
  (other threads' `next()` calls land between a given case's own
  start/end pair). Never assert on specific `latency_ms` values in a
  concurrency test — assert on ordering/content/pass-fail parity instead.

## New decision: custom "use and modify, no redistribution" LICENSE

The repo is now public (pushed to `github.com/IlionAcre/litmus`), which
needed a license decision. The user explicitly wants: anyone may use and
modify the software (including for private/internal/company purposes), but
may **not** redistribute it, original or modified, to any third party in
any form. This doesn't match any standard OSI-approved license (MIT/
Apache/BSD all explicitly permit redistribution — that's core to what
makes them "open source" in the legal sense), so `LICENSE` is a custom,
non-SPDX text rather than a named license. GitHub will not show an
automatic license badge for it. Flagged to the user: this is not
lawyer-reviewed; treat it as a clear approximation of intent for a
portfolio project, not a substitute for real legal review if the stakes
ever become non-trivial. Don't swap this for a standard license (MIT,
Apache, etc.) without the user explicitly re-deciding — the whole point was
the no-redistribution clause a standard license wouldn't provide.

## New decision: `.gitattributes` added (`* text=auto`)

Every commit in this repo's history showed `LF will be replaced by CRLF`
warnings from git, since development happened entirely on Windows with no
line-ending normalization configured. Added `.gitattributes` with
`* text=auto` so git normalizes to LF in the repository regardless of
platform and converts to the local OS convention on checkout — prevents
spurious whole-file diffs for anyone cloning from Linux/Mac. Purely a repo
hygiene fix, no behavior change.

## New decision: standing CI gate testset switched to `testsets/routing_baseline`

`eval-gate.yml` previously ran against `testsets/example` (2 cases). Actually
attempting a live PR test (the first time this workflow was ever exercised
against a real GitHub Actions run, not just locally validated) surfaced two
real problems `testsets/example` could never have been caught by:

1. **Structurally no statistical power.** McNemar's test needs enough
   discordant pairs to reach significance; with only 2 cases, at most 1 can
   ever flip, which `mcnemar_test`'s own math can't call significant
   (`binomtest(0-or-1, 1, 0.5)` is never below any reasonable `alpha`). The
   gate could never have actually blocked a real regression on this
   testset, regardless of how it was wired.
2. **No committed baseline run existed for it at all.** The two runs
   already committed to `runs/` on `main` were the Phase 10 case-study data
   (`testsets/routing_baseline`/`routing_candidate`, 28 cases, `t01`..`t28`
   ids). The gate's baseline-selection logic (`query_trends()[-1]`, "the
   single most recent run on the base branch, project-wide, no testset
   scoping") would have picked one of those as "baseline" and compared it
   against a fresh `testsets/example` run, zero overlapping `test_case_id`s
   between them, which `compare_runs()` correctly raises `ValueError` for.
   That uncaught exception would have surfaced as a raw traceback in the PR
   comment instead of a clean comparison, not a clean gate failure.

Fixed by switching the gate's standing target to `testsets/routing_baseline`
(the already-proven, 28-case, real-power testset from the case study) and
persisting a fresh real baseline run against it
(`runs/fea35b06b52643ea9c5eeccb10b6bbeb.json`, `prompt_version="main"`,
28/28 passed, real Gemini call, ~$0.0003) so it's chronologically the
latest run on `main` (the two existing case-study runs are older;
`d8d16e16...`, the *bad* "candidate" prompt run, was previously the
chronologically latest of the two, which would have been picked as
"baseline" backwards).

**Known, not-yet-fixed limitation, surfaced by this same investigation:**
the gate's baseline selection still isn't scoped by testset, it just picks
the single most recent run on the base branch project-wide. This works
correctly today because there's only one actively-maintained testset the
gate targets, but if a second testset were ever added and its own runs
started getting committed to `main`, the "latest run overall" heuristic
could again pick an incompatible baseline. A real fix would need
`query_trends`/the workflow's baseline-selection query to filter by which
testset produced a run (e.g. by `test_case_id` overlap, or an explicit
testset tag), not just recency. Not fixed in this pass; worth doing if a
second testset is ever added to the gate's scope.

## New decision: `eval-gate.yml` needs an explicit `permissions:` block

The first-ever live PR run of the gate (PR #1, a deliberate test regression,
closed without merging) surfaced a real bug: the "Post comparison as a PR
comment" step failed with `HttpError: Resource not accessible by
integration`. The workflow never declared a `permissions:` block, so it
inherited this repo's default `GITHUB_TOKEN` permissions, which don't
include posting PR comments. Fixed by adding `permissions: pull-requests:
write` at the job level. Everything upstream of that step (checkout, sync,
fetch baseline, run eval suite, compare) worked correctly on the first real
run, including a genuinely good sign found along the way: every one of the
28 candidate-run cases hit `APIConnectionError: Missing Gemini API key`
(the `GEMINI_API_KEY` repo secret wasn't set yet at test time) and every
single one degraded cleanly to `[ERROR]` instead of crashing the run, and
`compare_runs()` correctly raised a clear "no common test_case_ids (all
cases errored)" instead of silently reporting a false result, both features
built earlier in this project working exactly as designed under a real,
unplanned failure mode, not just their own unit tests.

## Status

All 16 checkpoints complete (125 tests passing). Full pipeline built and
tested: schema, loader, runner, real litellm execution, the `litmus
run`/`litmus compare`/`litmus serve` CLI commands, all three scorers,
McNemar's + paired-bootstrap statistical tests, DuckDB-backed trend queries,
the CI gate workflow, the FastAPI dashboard, structured JSON-lines logging
(`logs/litmus.jsonl`, see decision above), and a real README case study
(real Gemini calls, p=0.00087 caught regression — see `README.md`).

A full close-reading code review pass (not just "tests pass") found and
fixed several real issues: the CI gate's baseline git-fetch bug (see Known
gotcha above), silent alignment-drop / no-error-visibility in `compare.py`,
no error handling around real LLM calls, an API 500 on incompatible runs,
OpenAI-specific scorer defaults with no real-provider credential to match,
unused/methodologically-wrong Mann-Whitney U, a non-independent statistical
test assertion, and no duplicate-test-case-id validation at load time. All
fixed with tests; see the decision entries above for what changed and why.

Remaining, not gaps but explicit scope boundaries: no rendered dashboard
frontend; no hosted deployment. Phase 8's workflow has since been verified
on a real PR against GitHub Actions, not just locally, see the CI gate
decision entries below. See `AI_docs/PHASES.md` for full phase detail.
