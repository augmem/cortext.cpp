# Operation Matrix

This document defines the **exhaustive ablation universe** for Cortext's
operation-family sweep.

It is intentionally narrower than [metric-matrix.md](/Users/gabrielwillen/VSCode/cortext/docs/metric-matrix.md).
`metric-matrix.md` enumerates atomic metric application sites. That is useful
for local reasoning, but it is too large for a literal global power-set study.

The exhaustive bench therefore runs over **independently removable operation
families** that already have either:

1. an ablation-only internal toggle, or
2. a deterministic way to omit the operation inside the benchmark harness.

## Scope

The current full sweep covers **12 non-fact operation families**:

| bit | family | disable mode in bench | primary behaviors covered |
|---|---|---|---|
| `0` | `source_confidence` | `CORTEXT_DISABLE_SOURCE_CONF=1` | provenance-weighted retrieval filtering |
| `1` | `predictive_retrieval` | `CORTEXT_DISABLE_PREDICTIVE_RETRIEVAL_BONUS=1` | pre-activation ranking prior |
| `2` | `constructive_recall` | `CORTEXT_DISABLE_CONSTRUCTIVE_RECALL=1` | retrieval-time reconstruction ledger |
| `3` | `procedural_memory` | `cfg.procedural_enabled=false` | routine memory surfacing and updates |
| `4` | `metacognitive_control` | disable `TOT`, `unknown`, and confidence-decay subpaths | recovery, caution, and confidence decay |
| `5` | `affect_interrupt` | `cfg.affect_interrupt=false` | affect-based interrupt threshold relaxation |
| `6` | `affect_retrieval` | `cfg.affect_retrieval=false` | affect bonus in retrieval ranking |
| `7` | `flashbulb_consolidation` | omit `ApplyEmotionalConsolidation` in the bench | flashbulb tagging and emotional half-life |
| `8` | `neuromodulation` | disable all four downstream scales | write, competition, reconsolidation, value gain |
| `9` | `meta_learning` | `CORTEXT_DISABLE_META_LEARNING=1` | learned knob-to-prior adaptation |
| `10` | `reinforcement_edges` | `cfg.reinforcement_enabled=false` | usage-driven `reinforces` edges |
| `11` | `sequential_edges` | `cfg.sequential_edges_enabled=false` | episode-order graph edges |

The exhaustive search size is therefore:

* `2^12 = 4096` operation-family combinations

That is the full power set for this document.

## Why This Is The Right Universe

The goal of the sweep is not "toggle every line of code." The goal is to answer:

> If we remove a coherent operation family from the branch, do we lose a real
> behavior that the paper claims the system should have?

That means the right unit is not `blend.relevance` or `persist.arousal`; it is a
removable family such as `predictive_retrieval`, `metacognitive_control`, or
`sequential_edges`.

## Explicit Exclusions

These are excluded from the exhaustive family bench on purpose:

| excluded area | reason |
|---|---|
| bitemporal fact layer | already has its own deterministic ablation matrix in `cortext_bitemporal_ablation_bench` and related fact benches |
| boundary subcomponents (`pressure`, `surprisal`, `natural`) | currently better handled as a separate boundary-calibration family; mixing them into this 12-family sweep would blur causal interpretation |
| flashbulb subcomponents (`percentile`, `rate`, `arousal`) | kept as dedicated subcomponent ablations; the exhaustive family sweep treats emotional consolidation as one removable family |
| individual neuromodulator downstream scales | kept as dedicated subcomponent ablations; the exhaustive family sweep asks whether neuromodulation as a family matters |
| individual metacognitive subpaths | kept as dedicated subcomponent ablations; the exhaustive family sweep asks whether metacognitive control as a family matters |

## Extension Sweep: Boundary + Fact Families

After the 12-family non-fact sweep, we ran a second deterministic extension
matrix for the previously excluded boundary/fact families:

| bit | family | ablation mode in extension bench | primary behavior covered |
|---|---|---|---|
| `0` | `boundary_surprisal` | `CORTEXT_BOUNDARY_DISABLE_SURPRISAL=1` | surprisal contribution inside natural boundary scoring |
| `1` | `boundary_natural` | `CORTEXT_BOUNDARY_DISABLE_NATURAL=1` | support-gated boundary score itself |
| `2` | `fact_layer` | `RetrievalAblationOverride.fact_layer_enabled=false` | direct fact-linked retrieval lifting |
| `3` | `fact_history` | `RetrievalAblationOverride.history_enabled=false` | historical `valid_at` retrieval |
| `4` | `fact_stale_penalty` | `RetrievalAblationOverride.stale_penalty_strength=Off` | stale-fact suppression in current mode |
| `5` | `fact_provenance` | `RetrievalAblationOverride.provenance_mode=AnyFactMatch` | direct-link provenance discipline |

The extension search size is:

* `2^6 = 64` combinations

Current extension result log:

* `logs/boundary_fact_extension_ablation_20260404/boundary_fact_extension_ablation.log`

Key outcome:

* the minimal best extension mask keeps
  `boundary_natural,fact_layer,fact_history,fact_stale_penalty,fact_provenance`
* only `boundary_surprisal` drops out of the minimal best deterministic mask

Interpretation:

* `fact_layer`, `fact_history`, `fact_stale_penalty`, and `fact_provenance` are
  load-bearing in the deterministic extension suite and are not pruning
  candidates
* `boundary_surprisal` is redundant **inside this narrow micro-suite**, but it is
  not automatically a global deletion candidate because the longer-horizon
  TopicalChat boundary rerun in the paper still shows a real encoder-sensitive
  `no_surprisal` effect

## Subcomponent Completion Sweep

After the family and boundary/fact sweeps, we ran one more deterministic
completion pass over the currently separable **multi-toggle subcomponent
families**:

* boundary: `pressure`, `surprisal`, `natural`
* flashbulb: `percentile`, `rate`, `arousal`
* metacognitive control: `tot_recovery`, `unknown_caution`, `confidence_decay`
* affect: `interrupt_path`, `retrieval_path`
* neuromodulation: `write_scale`, `competition_scale`,
  `reconsolidation_scale`, `value_gain`

Benchmark:

* `examples/benchmark/cortext_subcomponent_matrix_bench`

Recorded run:

* `logs/flashbulb_rate_rewrite_20260404/subcomponent_matrix.log`

These are the multi-part families that currently expose independently removable
subpaths on the branch. Families such as `predictive_retrieval`,
`constructive_recall`, `procedural_memory`, `meta_learning`,
`reinforcement_edges`, and `sequential_edges` are already close to atomic at
the current implementation level, so they were not exploded further here.

### Boundary subcomponents

| subcomponent | mean marginal | max score without | essential for best | current reading |
|---|---:|---:|---:|---|
| `pressure` | 0.000000 | 3.000000 | 0 | redundant in this micro-suite |
| `surprisal` | 0.000000 | 3.000000 | 0 | redundant in this micro-suite |
| `natural` | 3.000000 | 0.000000 | 1 | core |

Best masks:

| rank | score | enabled | disabled |
|---|---:|---|---|
| 1 | 3.000000 | `natural` | `pressure,surprisal` |
| 2 | 3.000000 | `pressure,natural` | `surprisal` |
| 3 | 3.000000 | `surprisal,natural` | `pressure` |
| 4 | 3.000000 | `pressure,surprisal,natural` | `none` |

Interpretation:

* `natural` remains the real load-bearing boundary mechanism
* `pressure` and `surprisal` are redundant in this tiny calibration suite
* only `surprisal` received a live-model follow-up in the paper, and it did
  **not** survive as a safe global cut
* `pressure` is therefore only a deterministic micro-suite redundancy today,
  not yet a justified global deletion

### Flashbulb subcomponents

| subcomponent | mean marginal | max score without | essential for best | current reading |
|---|---:|---:|---:|---|
| `percentile` | 1.000000 | 1.000000 | 1 | required |
| `rate` | 0.000000 | 2.000000 | 0 | neutral stabilizer in this micro-suite |
| `arousal` | 1.000000 | 1.000000 | 1 | required |

Best masks:

| rank | score | enabled | disabled |
|---|---:|---|---|
| 1 | 2.000000 | `percentile,arousal` | `rate` |
| 2 | 2.000000 | `percentile,rate,arousal` | `none` |
| 3 | 1.000000 | `percentile` | `rate,arousal` |

Interpretation:

* `percentile` and `arousal` are the load-bearing flashbulb paths in the local
  suite
* after the rate rewrite, `rate` is no longer antagonistic: keeping it no
  longer hurts the best achievable flashbulb micro-suite score
* this lines up with the post-rewrite live EmbeddingGemma rerun in the paper,
  where baseline and `no_rate` became identical on the current workload

### Metacognitive subcomponents

| subcomponent | mean marginal | max score without | essential for best | current reading |
|---|---:|---:|---:|---|
| `tot_recovery` | 0.500000 | 2.000000 | 1 | required |
| `unknown_caution` | 1.000000 | 2.000000 | 1 | required |
| `confidence_decay` | 0.500000 | 2.000000 | 1 | required |

Interpretation:

* all three subpaths are load-bearing in the metacognitive micro-suite
* `unknown_caution` is the strongest single contributor
* there is no metacognitive subcomponent pruning candidate on the current
  deterministic evidence

### Affect subcomponents

| subcomponent | mean marginal | max score without | essential for best | current reading |
|---|---:|---:|---:|---|
| `interrupt_path` | 0.000000 | 1.000000 | 0 | redundant in this micro-suite |
| `retrieval_path` | 1.000000 | 0.000000 | 1 | required |

Interpretation:

* `retrieval_path` is the load-bearing affect subcomponent in the local
  deterministic ranking probe
* `interrupt_path` is redundant in the micro-suite but already failed live-model
  deletion confirmation in the paper
* so the correct current reading is: keep both unless the live stack changes

### Neuromodulator subcomponents

| subcomponent | mean marginal | max score without | essential for best | current reading |
|---|---:|---:|---:|---|
| `write_scale` | 1.000000 | 3.000000 | 1 | required |
| `competition_scale` | 1.000000 | 3.000000 | 1 | required |
| `reconsolidation_scale` | 1.000000 | 3.000000 | 1 | required |
| `value_gain` | 1.000000 | 3.000000 | 1 | required |

Interpretation:

* all four downstream neuromodulator scales are independently load-bearing in
  the deterministic suite
* there is no evidence here for pruning any neuromodulator subpath

## Scenario Map

Each family is exercised by at least one deterministic scenario in the
exhaustive bench, but the scenarios are scored as a **shared suite**, not as
isolated one-off proofs. This is important because the sweep is meant to expose
interaction structure, not just check that a toggle exists.

| scenario family | main families stressed |
|---|---|
| provenance-weighted retrieval | `source_confidence`, `metacognitive_control` |
| predictive/routine retrieval | `predictive_retrieval`, `procedural_memory`, `reinforcement_edges`, `sequential_edges` |
| reconstruction-sensitive retrieval | `constructive_recall`, `source_confidence` |
| affective interrupt and ranking | `affect_interrupt`, `affect_retrieval`, `flashbulb_consolidation` |
| adaptive plasticity | `neuromodulation`, `meta_learning`, `reinforcement_edges` |
| episode graph continuity | `sequential_edges`, `reinforcement_edges`, `procedural_memory` |

## Interpretation Rules

When reading the exhaustive bench:

| signal | interpretation |
|---|---|
| `max_score_without_family < full_mask_score` | that family is not globally removable on the deterministic suite |
| low average marginal contribution but nonzero `max_score_without` gap | the family is real but narrow; keep unless corpus reruns show no value |
| `max_score_without_family == full_mask_score` | candidate for removal or demotion |
| a smaller mask matches the full-mask score | the omitted families are deterministic pruning candidates |

## Relationship To Dedicated Benches

The exhaustive family sweep does **not** replace the dedicated benches. It
organizes them.

Use the dedicated benches when you need to know:

* which **subcomponent** inside `metacognitive_control` matters
* whether `flashbulb_percentile` matters more than `flashbulb_rate`
* whether neuromodulation helps more through write scaling or value gain

Use the exhaustive family sweep when you need to know:

* whether the branch still needs the family at all, and
* whether several families are jointly redundant on the current deterministic suite.

## Full Separable-Unit Matrix

The matrix is now complete at the **separable-unit** level.

The branch exposes **26 independently removable units** that are meaningful to
ablate without changing the public API:

* **19 non-boundary/fact units**
  * `source_confidence`
  * `predictive_retrieval`
  * `constructive_recall`
  * `procedural_proactive`
  * `metacog_tot_recovery`
  * `metacog_unknown_caution`
  * `metacog_confidence_decay`
  * `affect_interrupt`
  * `affect_retrieval`
  * `flashbulb_percentile`
  * `flashbulb_rate`
  * `flashbulb_arousal`
  * `neuromod_write_scale`
  * `neuromod_competition_scale`
  * `neuromod_reconsolidation_scale`
  * `neuromod_value_gain`
  * `meta_learning`
  * `reinforcement_edges`
  * `sequential_edges`
* **7 boundary/fact units**
  * `boundary_pressure`
  * `boundary_surprisal`
  * `boundary_natural`
  * `fact_layer`
  * `fact_history`
  * `fact_stale_penalty`
  * `fact_provenance`

If you flattened that into one monolithic power set, it would be:

* `2^26 = 67,108,864` combinations

We did **not** materialize that as one giant benchmark, because the deterministic
audit graph factorizes cleanly into:

* a **19-unit non-boundary/fact cluster**
* a **7-unit boundary/fact cluster**

Those clusters were each run exhaustively:

* `logs/flashbulb_rate_rewrite_20260404/full_unit_ablation.log`
  * `2^19 = 524,288` non-boundary/fact unit combinations
  * implemented by `examples/benchmark/cortext_full_unit_ablation_bench`
  * evaluated through 164 factorized local cluster states
* `logs/full_boundary_fact_unit_ablation_20260404/full_boundary_fact_unit_ablation.log`
  * `2^7 = 128` boundary/fact unit combinations
  * implemented by `examples/benchmark/cortext_full_boundary_fact_unit_ablation_bench`

This is the completion criterion for the current audit: every separable unit is
covered by an exhaustive deterministic matrix in its actual dependency cluster.

### Non-boundary/fact 19-unit result

Key summary from the 19-unit sweep:

* full mask score: `15/19`
* best score: `16/19`
* minimal best mask disables:
  * `source_confidence`
  * `affect_interrupt`
  * `flashbulb_rate`

Load-bearing in the unit sweep:

* `predictive_retrieval`
* `constructive_recall`
* `procedural_proactive`
* all three metacognitive subpaths
* `affect_retrieval`
* `flashbulb_percentile`
* `flashbulb_arousal`
* all four neuromodulator downstream scales
* `meta_learning`
* `reinforcement_edges`
* `sequential_edges`

Interpretation:

* `flashbulb_rate` is no longer the strongest non-boundary/fact pruning signal;
  after the rewrite it looks neutral rather than harmful
* `affect_interrupt` again looks redundant in the deterministic micro-suite, but
  it already failed live-model deletion confirmation in the paper
* `source_confidence` is not load-bearing in this clean deterministic unit suite,
  but the contradiction-heavy Ubuntu probe still shows it blocking poisoned
  memories end to end; treat it as workload-conditional, not fake

### Boundary/fact 7-unit result

Key summary from the 7-unit sweep:

* full mask score: `7/7`
* best score: `7/7`
* minimal best mask disables:
  * `boundary_pressure`
  * `boundary_surprisal`

Load-bearing in the unit sweep:

* `boundary_natural`
* `fact_layer`
* `fact_history`
* `fact_stale_penalty`
* `fact_provenance`

Interpretation:

* the fact sublayer remains fully load-bearing
* `boundary_pressure` and `boundary_surprisal` are both redundant in the narrow
  deterministic boundary/fact unit suite
* `boundary_surprisal` already failed live-model deletion confirmation, so it is
  not a safe global cut
* `boundary_pressure` is the remaining boundary-side candidate that still needs
  a live confirmation pass before any surgery

### Current removal picture

After the family sweep, subcomponent sweeps, live confirmation pass, 19-unit
non-boundary/fact sweep, and 7-unit boundary/fact sweep:

* plausible but still unconfirmed candidate: `boundary_pressure`
* low-evidence but no longer harmful after the rewrite: `flashbulb_rate`
* deterministic-only redundancies already rejected by live evidence:
  * `affect_interrupt`
  * `boundary_surprisal`

That is the current end state of the matrix. Further progress would require a
new scope, such as live-model confirmation for `boundary_pressure` or deeper
metric-site audits below the separable-unit level.

## Structural Deletion Audit

The matrices above answer a **behavioral** question:

> If this family or unit is disabled, do the claimed behaviors still hold?

They do **not** by themselves answer the stricter engineering question:

> If I physically delete this code, will the branch still build, run, and keep
> the public surface coherent?

This section captures the current structural-readiness view for the strongest
cut candidates.

### Candidate classes

| candidate | behavioral status | structural entanglement | public API impact | schema impact | first-surgery suitability |
|---|---|---|---|---|---|
| `flashbulb_rate` | low-evidence after rewrite, but no longer harmful | medium | none | yes | medium-low |
| `boundary_pressure` | plausible but unconfirmed cut candidate | low | none | no | high if live follow-up clears |
| `affect_interrupt` | failed live deletion confirmation | high | yes | no | low |
| `source_confidence` | workload-conditional, not fake | medium-high | no direct public flag, but broad retrieval coupling | yes | low |

### `flashbulb_rate`

Current structural touchpoints:

* runtime algorithm in `src/operations/emotion.cpp`
* state field in `include/cortext/processor/processor_context.hpp`
* persisted state hydration in `src/signal_processor.cpp`
* state schema column `flashbulb_rate` in `src/store/schema.cpp`
* benchmark and paper references

Implication:

* deleting `flashbulb_rate` is **not** a blind one-line removal
* it needs:
  * algorithm cleanup in emotional consolidation
  * removal of `flashbulb_rate_ewma` from processor state
  * removal of persisted state read/write
  * a schema migration strategy for the stale column

So `flashbulb_rate` is still removable in principle, but after the rewrite it
no longer looks urgent enough to outrank `boundary_pressure` as the first
surgery target.

### `boundary_pressure`

Current structural touchpoints:

* internal boundary logic in `src/operations/boundary.cpp`
* deterministic benches and matrix docs

Notably absent:

* no public config field
* no public C API field
* no persisted schema column dedicated to boundary pressure

Implication:

* if live-model confirmation clears it, `boundary_pressure` is the cleanest
  actual first surgery candidate
* its removal would mostly be internal boundary-logic simplification plus test
  and doc updates

### `affect_interrupt`

Current structural touchpoints:

* public processor/runtime config fields in `include/cortext/processor.hpp`
  and `include/cortext/cortext.hpp`
* C API fields in `include/cortext/capi.h`
* C API marshalling in `src/capi.cpp`
* chat/example config plumbing in `examples/chat/main.cpp`
  and `examples/topical_chat_analysis/main.cpp`
* interrupt gate behavior in `src/operations/interrupt_gate.cpp`

Implication:

* even if it had survived the behavioral audit, this would be a relatively
  expensive deletion because it crosses the public surface
* since it already failed live-model deletion confirmation, it should not be
  the first surgery target anyway

### `source_confidence`

Current structural touchpoints:

* retrieval filtering and ranking in `src/operations/graph_retrieval.cpp`
* reconstruction ledger metadata in
  `src/operations/constructive_recall_internal.cpp`
  and `src/store/schema.cpp`
* persisted column use in retrieval-side tables
* paper benchmarks and contradiction-heavy probe

Implication:

* this is not structurally trivial to remove
* more importantly, it is behaviorally conditional rather than dead: the Ubuntu
  contradiction probe still shows it blocking poisoned memories
* so it is not a good first surgery candidate

### Practical order

If the next phase is actual removal work, the pragmatic order is:

1. `boundary_pressure`
   * only after a live-model confirmation pass
2. `flashbulb_rate`
   * now a low-evidence cleanup candidate, but still requires a real
     state/schema cleanup
3. reconsider others only after those two are resolved

This keeps the design audit and the engineering deletion audit separate:

* **behavioral matrix** says what still earns its place
* **structural audit** says what can be excised cleanly first
