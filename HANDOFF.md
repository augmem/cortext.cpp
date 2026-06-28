# HANDOFF — validation sprint (2026-06-13, 02:32 UTC)

> **Private. Stays untracked. NEVER commit or push this file** (server paths + corpus details; repo is public).
> Corpus `/shared/Memory/Julie` is private personal correspondence — analyze programmatically, never quote intimate content.
> (Supersedes the June 9 handoff; that capacity-4 / pre-sweep content is obsolete.)

## Current state

**v8 context-blowout stress run is complete.** The only still-running related processes are
the two Ollama servers:

- summarizer/extractor Ollama PID **1825328** on `0.0.0.0:11435`
- judge Ollama PID **3593901** on `127.0.0.1:11434`

Out dir: `eval_runs/replay_v8_context_blowout/`.

Public repo state: pushed commit **0cee79af** (`Record v8 context-blowout eval`)
updates the judge prior-only fairness filter, the loss-audit aggregate, paper
section 9, and the rendered manuscript.

Benchmark gate:

- `benchmark_status.json`: `status=finished`, `benchmark_exit_code=0`
- probe stream: 30/30 rows, final event index 18,000
- `summary.json`: valid; `processed_text_messages=18000`, `media_attempted=128`,
  `media_processed=126`, `media_failures=2`, `wall_ms=61735825`,
  `peak_rss_mb=4547.234375`, `db_memory_count=37310`,
  `db_association_count=6377`, `consolidation_runs=206`

Judge gate:

- **Accepted artifact:** `judge_gemma4_12b_local_context128_prioronly.json`
- shards: 45 + 45 rows; merged rows: 90
- complete: `judgment_complete=true`, `expected_judgments=90`, `judged=90`,
  `missing_judgments=0`, `judge_validation={}`
- fairness: `no_future_context_violations=true`,
  `no_current_turn_context_inclusions=true`, prompt-fit true, media-map complete
- non-prior filter: 82 Cortext rows excluded (precheck: future/current before filter 52/30,
  after filter 0/0)
- **Do not use:** `judge_gemma4_12b_local_context128.json` failed fairness
  (`cortext_native_future_context_violations=52`) and is preserved only as an invalid
  diagnostic.
- **Do not use:** the original uncapped final judge; it exceeded local context
  (prompt > 131,072 tokens).

Accepted v8 results:

- row wins: Cortext 30/90, traditional chat RAG 9/90, full-history upper bound 23/90,
  compacting session 10/90, tie/unclear 16, insufficient_context 2
- sufficiency means/CIs: Cortext 3.14 [2.76, 3.51], RAG 2.57 [2.22, 2.91],
  full-history 3.70 [3.36, 4.02], compacting 2.90 [2.58, 3.24]
- noise means/CIs: Cortext 1.64 [1.32, 1.96], RAG 1.09 [0.82, 1.37],
  full-history 0.36 [0.21, 0.51], compacting 0.84 [0.66, 1.04]
- mean context tokens: Cortext 432.7, RAG 42,550.2, full-history 88,806.4
- Cortext token reduction vs RAG: aggregate 98.98%; bootstrap mean 98.61%
  [98.05%, 99.04%]

Interpretation: full-history remains the quality upper bound. Cortext is the top
row-win system at long horizon and spends roughly two orders of magnitude fewer
context tokens than RAG, but strict prior-only filtering exposes higher Cortext
noise and overlapping sufficiency CIs against RAG/compacting. This is the
full-stack stress verdict, not a per-mechanism promotion of every durable-structure
component.

**Failed attempt preserved:** `eval_runs/replay_v8_context_blowout_dead_probe13_20260613T0223/`
contains the run that stopped with stale `status: running`, no `summary.json`, and 13/30
probe rows. There was no benchmark/orchestrator process left and no recorded exit code.

## Done this sprint (all pushed; HEAD = 0cee79af)

- **WM capacity = 21.** Unimodal curve (7→10→14→21→42), peak 21; 3-rep confirm suff **3.35 [3.08,3.62]** vs baseline **2.62 [2.29,2.93]** (non-overlapping CIs). Wins flat ~22/39 across all capacities — capacity moves sufficiency, not win rate.
- **18-arm mechanism sweep complete** at WM21. Control: 22/39, suff 3.95, noise 0.67, 503 tok.
  - **Keep:** temporal_retrieval (removal −0.21 suff, noise ~doubles), predictive_bonus (removal −0.16 suff, +30% bloat).
  - **Weak / late-concentrated:** constructive_recall, media_source_blobs.
  - **Cut candidates:** metacognitive (removal mildly positive, no long-horizon story → top simplification), procedural_proactive (exact null).
  - **Deferred to v8:** daily_consolidation, graph_expansion, stm_ltm_graph_label_handoff, fact_boosts family.
  - **ACT-R gates (enable-on-top, must beat control to promote): ALL 6 FAIL.** Singles null, differ only in token cost (+18…+145). `actr_all_gates` **HARMFUL** — noise 0.95 vs 0.67, worst in sweep (independently-null perturbations compound into junk admission). `evidence_blending` only token-negative arm (−38) → lone long-horizon re-test candidate. Opt-in flag design vindicated.
- **Paper updated & pushed** (`docs/paper/sections/9_experimental.qmd`): capacity table, mechanism table, ACT-R Gate Promotion Arms subsection. Commits b03a1fcd, c0281003.
- **UTF-8 run-killer fixed** (commit **4f143943**). Corpus past msg ~2,400 carries raw non-UTF-8 (0xF7); `nlohmann::json::dump()` threw `type_error.316` inside `ConsolidationSummarize` and killed v8 take 1. Fix: `error_handler_t::replace` (U+FFFD) at the two sites carrying corpus text — `src/providers/ollama_provider.cpp` (chat body) + `src/capi.cpp` (context_to_json). Suite green (7,782 assertions). **v8 take 3 has passed the old crash point — fix confirmed live.** Failed run preserved at `eval_runs/replay_v8_context_blowout_failed_utf8/`.
- **v8 final eval completed and pushed** (commit **0cee79af**): accepted prior-only capped judge artifact is `eval_runs/replay_v8_context_blowout/judge_gemma4_12b_local_context128_prioronly.json`; paper section 9 and generated manuscript now report the completed long-horizon stress run.

## Gotchas / environment

- **Pushes:** ssh-agent dead → `git -c credential.helper='!gh auth git-credential' push https://github.com/augmem/cortext.git main`.
- **Build linker:** needs `LIBRARY_PATH=$HOME/.local/lib` or `-lsqlite3` fails to link. A bare piped build can mask a FAILED compile and silently run a stale binary — **always check the exit code explicitly**, don't trust a tail'd "All tests passed".
- **Judging modes:** single-rep live screens score ~0.6–0.7 suff higher than 3-rep. Orderings hold; never compare across modes. Runner copies the live cumulative artifact to `gemma4_12b_ollama_blind_judge_reps3.json` (misleading name — verify schema/repetition field; the genuine 3-rep file is `judge_3rep.json`).
- **Early-probe head-fakes:** probe-8/12 leads reverted in 4+ arms. Single-rep deltas <~0.15 suff are noise, not signal.
- **Win saturation:** at 1,200 msgs recency coverage pins wins ~22/39 for every arm; mechanism value only shows in suff/noise/tokens. v8 exists to break this.
- **Corpus scope:** full transcript is ~105k messages + ~5,900 media, Nov 2019→2026. v8's 18k ≈ first year — chosen for horizon coverage, NOT corpus completeness. A true full-corpus replay ≈ 6× (~3 days bench); one-time release-gate option, decide after v8 verdict.
- **Preserve, never delete:** `eval_runs/replay_ablation_20260611` (v4 control), `eval_runs/replay_v5_wm_partition`.

## Open tasks

- **#17** v8 monitoring — complete through accepted judge, paper §9 update, manuscript render, memory update, commit, and push.
- **#6** chat-replay protocol design — effectively done; close once v8 validates the long-horizon arm.
- **#8** convert remaining ablation benches to real AIST embeddings (decay_surface failing) — older, untouched this sprint.

## Memory
`~/.claude/.../memory/validation-sprint-state.md` current through the sweep + UTF-8 fix; `MEMORY.md` index line updated.
