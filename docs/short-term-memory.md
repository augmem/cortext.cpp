# Short-Term Memory Layer

This document explains where a Cortext short-term memory (STM) layer fits in
the existing architecture and how the three knobs should affect it.

STM is a system-level layer, not an anchor-model feature. Anchor models may
eventually consume STM, but the first decision is simpler: does an unsurfaced
short-term layer improve Cortext's ordinary memory behavior enough to justify
the extra state, latency, and implementation complexity?

## Why Add STM

The manuscript already has:

- `signal_stream`: per-step accumulator centroid stream;
- `score_stream`: scalar score stream;
- `memory_stream`: written memory representatives;
- `recent_memory_centroids`: bounded recent memory centroids for the interrupt
  gate;
- working memory: surfaced active coherent memories;
- long-term memory: durable stored memory and graph retrieval.

What is missing is an unsurfaced recent evidence layer.

Working memory is intentionally small and curated. Long-term memory is durable
and retrieval-oriented. Neither is the right place for weak, recent, ordered
evidence that may be useful for near-future continuity, boundary decisions,
consolidation evidence, or later model-side attention but should not be
surfaced.

STM should be that missing layer:

```text
working memory      surfaced context, small, high-confidence
short-term memory   unsurfaced recent ingress trace, dense, ordered
long-term memory    durable retrieval store
```

The closest cognitive analogue in the current manuscript is Baddeley's episodic
buffer plus Cowan-style activated-but-not-focused memory: STM is available to
internal mechanisms but is not necessarily inside the focus of attention or
surfaced context.

## Placement In The Current Pipeline

The manuscript's write path is currently:

```text
signal ingress
  -> encoder
  -> accumulator update
  -> boundary/write decision
  -> working-memory gate at coherent memory boundary
  -> retrieval/interrupt/consolidation consumers
```

Add STM between encoded ingress and working-memory surfacing:

```text
signal ingress
  -> encoder
  -> STM append / update
  -> accumulator update
  -> boundary/write decision
  -> working-memory gate
  -> shadow STM diagnostics / eligible internal consumers
  -> retrieval consumes committed/surfaced state
```

For strict chronology in any shadow benchmark:

```text
STM_before_t + current_signal_t -> diagnostic_or_consumer_t
current_signal_t appended to STM_after_t for future steps
```

That prevents future leakage while still letting the current signal be evaluated
against prior local context.

## Relationship To Existing Buffers

STM should not replace existing buffers.

| Existing structure | Role | STM relationship |
|---|---|---|
| `signal_stream` | accumulator centroid history for scoring/adaptation | STM stores richer per-signal evidence, not just centroids |
| `score_stream` | score lookbacks | STM may store score metadata per item |
| `memory_stream` | written memory representatives | STM includes pre-write and post-write evidence |
| `recent_memory_centroids` | interrupt context | STM is broader and ordered; interrupt may keep its cheap deque |
| working memory | surfaced active memories | STM is unsurfaced and denser |
| temporal context `c_t` | slow drift / context reinstatement | STM item stores temporal context snapshot |
| accumulator | forms coherent memory units | STM preserves signals even before/without write |

## STM Item Contract

Initial benchmark-only item:

```text
ShortTermMemoryItem {
  stm_id
  source_id
  step_id
  timestamp
  modality
  semantic_vector
  anchor_key optional
  temporal_context_vector
  accumulator_centroid
  raw_signal_embedding optional
  evidence_vectors optional
  score_t
  boundary_score
  drift
  coherence
  surprisal
  salience
  confidence
  source_continuity
  anchor_link optional
  surfaced_to_working_memory
  committed_to_long_term
  boundary_id optional
  ttl / expires_at
}
```

Runtime consumers should receive tensors and compact metadata only. Labels,
target flags, entity ids, track ids, and gold actions must not enter runtime
inputs.

## Storage Scope

Start benchmark-only and in-memory.

Recommended first implementation:

```text
ProcessorContext
  per_source_short_term_memory: map<source_id, ShortTermMemoryBuffer>

ShortTermMemoryBuffer
  Append(item)
  SnapshotBeforeStep(source_id, step_id)
  Recent(max_items, filters)
  AnchorLinked(anchor_id, max_items)
  Expire(knobs, boundary_state)
  Compact(knobs)
```

No public API or database schema change is needed until STM proves useful. If
promotion later requires persistence, use an append-only optional table so old
databases still open.

## Knob Effects

STM should follow the manuscript's three-knob philosophy: knobs set rates and
bounds, not modes.

### Focus

Focus controls STM read selectivity.

High Focus:

- narrows which STM items consumers or diagnostics attend to;
- increases weight on relevance/context match;
- reduces diffuse source-continuity carryover;
- lowers read breadth into the model.

Low Focus:

- permits broader STM context;
- helps exploratory recall in ambiguous or under-specified situations;
- increases risk of stale/same-source distractions.

Proposed derived quantities:

```text
stm_read_k(F,S,T) =
  round(lerp(128, 32, F) * lerp(1.0, 1.15, S) * lerp(0.85, 1.20, T))

stm_relevance_floor(F) = lerp(0.05, 0.30, F)

stm_diversity_weight(F,S,T) =
  lerp(0.35, 0.08, F) * lerp(0.9, 1.15, S) * lerp(0.9, 0.75, T)
```

Interpretation: high Focus gives a smaller, cleaner attention field; low Focus
keeps a wider shadow context.

### Sensitivity

Sensitivity controls plasticity and capture of weak or novel evidence.

High Sensitivity:

- increases STM admission salience for novelty and surprise;
- retains weak observations long enough to test future confirmation;
- shortens ordinary unlinked retention when evidence fails to confirm;
- increases split/create responsiveness.

Low Sensitivity:

- captures less weak evidence;
- reduces churn;
- risks missing fleeting continuity evidence.

STM should append all ingress signals initially, so Sensitivity should affect
priority, pinning, and compaction rather than admission itself.

Proposed derived quantities:

```text
stm_pin_gain(S) = lerp(0.2, 0.8, S)
stm_unconfirmed_decay(S,T) = lerp(0.04, 0.18, S) * lerp(1.15, 0.75, T)
stm_boundary_flush_gain(S,T) = lerp(0.2, 0.9, S) * lerp(1.05, 0.8, T)
```

Interpretation: high Sensitivity quickly records and tests weak evidence, but
unconfirmed items should decay quickly to avoid source-continuity glue.

### Stability

Stability controls persistence and inertia.

High Stability:

- lengthens STM retention;
- increases tolerance for delayed local continuity;
- slows eviction across soft topic shifts;
- keeps recently closed or boundary-crossing evidence visible for rejection or
  consolidation audit longer.

Low Stability:

- shortens STM retention;
- supports rapid topic shifts;
- increases risk that delayed references lose needed evidence.

Proposed derived quantities:

```text
stm_capacity(T) = round(lerp(32, 160, T))
stm_retention_s(T) = round(lerp(120, 1800, T))
stm_closed_anchor_grace_s(T) = round(lerp(60, 900, T))
stm_compaction_interval_s(T) = round(lerp(30, 300, T))
```

Interpretation: high Stability gives STM enough temporal span for delayed
continuity, but consumer safety must prevent stale items from acting like live
context.

## Boundary Interaction

The manuscript already defines boundary probability from surprisal, coherence,
drift, topic shift, and gap signals.

STM should use the same boundary state:

- high boundary score decays unlinked STM;
- boundary-crossing items may remain as recent closed/stale evidence;
- explicitly linked STM survives boundary crossing for a grace window;
- STM compaction should happen after boundaries, not mid-memory.

Boundary policy:

```text
if boundary_score high:
  retain explicitly linked STM
  retain recent closed/stale evidence
  decay unlinked same-source carryover
  compact older unlinked STM into event summary candidates
```

This directly targets stale same-source carryover and topic-shift false
continuity.

## Working Memory Relationship

Working memory remains the surfaced active set. STM does not bypass the WM gate.

At memory boundaries:

```text
written memory may enter WM through existing WM gates
STM keeps the underlying recent signals whether or not the memory surfaced
```

This preserves weak recent evidence without increasing visible context.

Any future consumer should treat the two layers differently:

- WM: curated, high-confidence active context;
- STM: dense, unsurfaced evidence and recent alternatives.

## Retrieval Relationship

Retrieval should not use STM as a general result source.

Allowed:

- benchmark-only diagnostics read STM before retrieval;
- future ingress-side models may read STM before retrieval after shadow gates
  pass;
- consolidation may use STM as evidence when a memory has already been written
  or linked;
- shadow diagnostics may report STM attention.

Not allowed initially:

- surfacing STM items directly to users/models;
- adding STM to vector retrieval results;
- using STM as a hidden reranker that changes production retrieval.

This keeps STM from becoming another hidden retrieval reranker.

## Consolidation Relationship

The manuscript says consolidation operates on stored memory representatives and
is external-only. STM should respect that.

Initial rule:

- STM is not a consolidation candidate by itself.
- STM can contribute evidence packets to a written memory.
- STM can be compacted into event summaries for future context.
- Durable consolidation still starts from written memories.

If STM promotion is later needed, it should happen through explicit write or
consolidation APIs, not background mutation.

## Deferred Anchor Model Relationship

This is not the first STM validation target. If STM proves useful as a general
memory layer, an anchor model may later consume:

```text
current signal
working-memory snapshot
short-term-memory snapshot
active-anchor snapshot
recent-closed-anchor snapshot
```

It should bind over anchors, not over memories:

```text
UPDATE -> active anchor index
CLOSE -> close/stale anchor index
SPLIT -> optional parent anchor index + new anchor
CREATE -> new anchor
ABSTAIN -> no attachment
```

STM items are evidence tokens. Anchors are state tokens.

## Validation Before Consumer Training

Do this before training any model or production consumer that reads STM.

Required STM audits:

- no future leakage in `STM_before_t`;
- no labels or evaluation-only fields in runtime tensors;
- STM length distributions by split and source;
- STM/WM overlap rate;
- useful recent evidence present when WM has already dropped it;
- topic-shift controls include tempting same-source evidence;
- stale same-source evidence retained only as stale/closed evidence;
- no-continuity controls include tempting STM evidence;
- delayed local-continuity retention at distances 2-4 and 5-12;
- source-held-out and dataset-held-out split hygiene.

Required outputs:

```text
short_term_memory_snapshots_train.jsonl
short_term_memory_snapshots_val.jsonl
short_term_memory_snapshots_test.jsonl
short_term_memory_schema.json
short_term_memory_summary.json
short_term_memory_leakage_audit.json
short_term_memory_knob_audit.json
short_term_memory_failure_slices.json
```

Promotion to model training should require:

- evidence coverage is high enough for delayed local continuity;
- no-continuity and topic-shift slices are not trivially easy;
- STM does not collapse to working memory;
- STM does not contain future context;
- STM knob behavior matches the expected Focus/Sensitivity/Stability effects.

## Implementation Phases

### Phase 1: Shadow STM Buffer

Add an in-memory, benchmark-only STM buffer and export snapshots during replay.

No retrieval changes.
No public API changes.
No model training.

### Phase 2: STM General Replay Audit

Run broad replay with STM export.

Evaluate:

- coverage;
- leakage;
- overlap with WM;
- recent-continuity, topic-shift, stale same-source, and no-continuity slice
  quality;
- knob sweeps for STM length and retention.

### Phase 3: Consumer-Specific Export

Only after Phases 1-2 pass, build task-specific training data. For anchoring,
that could later look like:

```text
current + WM_before + STM_before + anchors_before -> labels
```

Step tensors are derived artifacts. Episode JSONL remains source of truth.

### Phase 4: Model Or Consumer Training

Train or enable a specific consumer only after STM itself proves useful.

Do not train this model before Phases 1-3 pass.

## Shadow-Only Experiment Plan

The first STM work should be benchmark-only. The purpose is to prove that STM
contains useful information not already available from existing Cortext buffers
and that the benefit is worth bounded state and latency.

No experiment in this section changes production retrieval behavior.

### Experiment 1: General STM Coverage Audit

Question:

> Does STM preserve useful recent evidence that working memory and
> `recent_memory_centroids` do not?

Run chronological replay and export `STM_before_t` for ordinary continuation,
topic-shift, delayed-local-continuity, and stale-same-source cases.

Compare evidence availability across:

- working memory only;
- `recent_memory_centroids`;
- memory accumulator window;
- STM;
- STM minus working-memory overlap;
- STM minus same-source-only items.

Metrics:

- useful recent evidence present before current step;
- delayed local-continuity evidence present at distances 2-4 and 5-12;
- stale same-source evidence retained but older/inactive;
- topic-shift and no-continuity controls with tempting STM evidence;
- STM/WM overlap rate;
- STM/recent-centroid overlap rate;
- mean/p50/p95 STM length.

Value condition:

- STM must improve target-evidence coverage over WM and
  `recent_memory_centroids`, especially for delayed references.

Kill condition:

- STM is mostly identical to WM or recent centroids;
- target evidence is still absent before weak references;
- no-continuity controls have no tempting STM evidence.

Outputs:

```text
stm_coverage_cases.csv
stm_coverage_summary.json
stm_overlap_audit.json
```

### Experiment 2: No-Leak Chronology Audit

Question:

> Can STM be exported without future leakage?

For each replay case, verify:

- every STM item has `step_id < current_step`;
- no label fields appear in STM runtime tensors;
- no target/candidate class/track id/gold action fields appear in runtime
  input;
- current signal is not already present in `STM_before_t`;
- current signal appears in `STM_after_t`.

Value condition:

- zero future-leak rows.

Kill condition:

- any future step appears in `STM_before_t`;
- label-only fields appear in runtime tensors.

Outputs:

```text
stm_leakage_audit.json
stm_leakage_failures.csv
```

### Experiment 3: STM Knob Sweep

Question:

> Do Focus, Sensitivity, and Stability affect STM in the way the architecture
> claims?

Run the same replay under a grid such as:

```text
F ∈ {0.3, 0.5, 0.7}
S ∈ {0.3, 0.5, 0.7}
T ∈ {0.3, 0.5, 0.7}
```

Metrics:

- STM length;
- STM retention time;
- STM read-k after Focus filtering;
- unconfirmed item half-life;
- boundary flush/decay rate;
- explicitly linked item retention;
- delayed-continuity evidence retention;
- stale same-source retention;
- no-continuity tempting evidence rate.

Expected directional checks:

- higher Focus reduces STM read breadth and increases relevance concentration;
- higher Sensitivity increases weak/novel evidence capture but decays
  unconfirmed evidence faster;
- higher Stability increases retention and delayed-continuity coverage;
- high Stability should not make stale evidence act like live context.

Kill condition:

- knobs do not move STM metrics monotonically in the expected direction;
- high Stability preserves stale false positives as active evidence;
- high Focus does not reduce diffuse source-continuity evidence.

Outputs:

```text
stm_knob_sweep_results.json
stm_knob_sweep_summary.csv
```

### Experiment 4: General STM Recall Utility Diagnostic

Question:

> If a diagnostic scorer can read STM, does useful recent-context recall improve
> without trivially increasing stale or topic-shift false positives?

This is not a production gate. It is a diagnostic scorer over exported
snapshots.

Compare:

- WM only;
- STM only;
- WM + STM;
- STM excluding same-source items;
- STM excluding explicitly linked items;
- shuffled STM order;
- time-reversed STM order.

Use simple diagnostic scorers first:

- best cosine to current;
- temporal-context cosine;
- MaxSim over evidence vectors if available;
- small non-trained attention diagnostic;
- no learned classifier initially.

Metrics:

- useful-evidence-vs-control AUC;
- useful evidence top-1/top-3;
- no-continuity false evidence risk;
- topic-shift false evidence risk;
- stale same-source evidence risk;
- delayed/prequential;
- source-held-out.

Value condition:

- WM + STM improves target evidence reachability over WM alone;
- shuffled/time-reversed STM degrades if order matters;
- no-continuity and topic-shift risk do not explode.

Kill condition:

- STM improves only target AUC while no-continuity/topic-shift risk remains
  unchanged or worse;
- shuffled STM performs the same as ordered STM;
- STM excluding same-source evidence performs as well as full STM, implying STM
  only added source-continuity noise.

Outputs:

```text
stm_recall_utility_results.json
stm_recall_utility_cases.csv
stm_recall_utility_failures.csv
```

### Experiment 5: STM Boundary Ablation

Question:

> Does boundary-aware STM retention matter?

Compare:

- no boundary decay;
- decay unlinked STM at boundaries;
- retain anchor-linked STM across boundaries;
- retain recently closed/stale anchor evidence;
- compact older STM into event summaries;
- hard FIFO only.

Metrics:

- stale same-source false carryover;
- wrong-active distinct evidence rate;
- no-anchor tempting evidence rate;
- delayed reference target retention;
- STM size and age distribution;
- boundary-crossing evidence survival.

Value condition:

- boundary-aware retention reduces stale carryover while preserving delayed
  target evidence.

Kill condition:

- hard FIFO matches boundary-aware STM;
- boundary-aware STM removes target evidence before delayed references;
- no boundary policy can reduce stale false evidence without destroying target
  retention.

Outputs:

```text
stm_boundary_ablation_results.json
stm_boundary_ablation_cases.csv
```

### Experiment 6: STM Complexity Budget

Question:

> Is STM cheap enough to justify the added architectural layer?

Measure:

- append/update latency;
- snapshot export latency;
- memory footprint per source;
- mean/p95 STM tensorization cost;
- model-input token count added by STM;
- replay throughput with STM enabled vs disabled.

Run with capacities:

```text
S ∈ {16, 32, 64, 128, 256}
```

Value condition:

- useful coverage appears at a bounded size, ideally 32-128 items;
- append/update overhead is negligible relative to encoding;
- tensorization is predictable and bounded.

Kill condition:

- useful coverage requires unbounded or very large STM;
- p95 overhead is incompatible with realtime ingress;
- memory footprint grows with long sessions despite TTL/compaction.

Outputs:

```text
stm_complexity_budget.json
stm_latency.csv
stm_memory_footprint.csv
```

### Experiment 7: STM Consumer Ablation

Question:

> Which downstream mechanisms actually benefit from STM?

Run shadow diagnostics with STM available to:

- boundary detection diagnostics;
- consolidation evidence diagnostics;
- retrieval diagnostics, shadow-only;
- ingress continuity diagnostics;
- anchor diagnostics, deferred / stress-test only;
- all diagnostics.

Metrics:

- boundary quality / over-segmentation;
- consolidation cluster purity;
- retrieval target reachability;
- no-continuity and stale false-positive risk.

Value condition:

- at least one non-retrieval consumer gains uniquely from STM;
- boundary, consolidation, or ingress-continuity diagnostics improve more from
  STM than retrieval diagnostics do.

Kill condition:

- only retrieval reachability improves;
- no boundary/consolidation/continuity metric improves;
- STM behaves like another retrieval candidate pool.

Outputs:

```text
stm_consumer_ablation_results.json
stm_consumer_ablation_summary.csv
```

### Experiment 8: Source-Continuity Trap Test

Question:

> Is STM providing useful recent evidence, or just same-source recency?

Construct/replay hard controls:

- same-source topic shifts;
- same-source stale semantically close;
- cross-source true continuation;
- cross-modal same-entity continuation.

Ablations:

- remove source metadata;
- remove recency metadata;
- remove semantic vectors;
- source/recency only;
- semantic only;
- boundary-only;
- STM order only.

Value condition:

- semantic + order + boundary evidence outperforms source/recency-only;
- cross-source or cross-modal true continuations remain detectable;
- same-source stale/topic-shift controls do not pass merely because of source
  continuity.

Kill condition:

- source/recency-only matches full STM;
- removing source metadata destroys all signal;
- same-source controls remain the dominant failure.

Outputs:

```text
stm_source_continuity_trap_results.json
stm_source_continuity_trap_cases.csv
```

### Experiment 9: STM Manual Inspection Pack

Question:

> Are the exported STM snapshots human-auditable and semantically plausible?

Create a small inspection bundle:

- 25 ordinary continuation cases;
- 25 no-continuity/topic-shift cases;
- 25 delayed local-continuity cases;
- 25 stale same-source cases.

For each case include:

- current signal text/evidence for audit only;
- STM item summaries or source refs;
- WM snapshot;
- active/closed anchors;
- expected continuity label for audit;
- whether useful/stale/tempting evidence appears in STM.

Value condition:

- a human can understand why STM should help or not help.

Kill condition:

- STM snapshots are opaque, redundant, or obviously missing the evidence needed
  for general continuity.

Outputs:

```text
stm_manual_inspection_cases.jsonl
stm_manual_inspection_summary.json
```

## Decision Matrix

STM is worth keeping only if the shadow experiments show all of:

- STM adds evidence coverage beyond WM/recent-centroid buffers;
- STM can be exported with zero chronology leakage;
- F/S/T move STM behavior in predictable directions;
- ordered STM beats shuffled or time-reversed STM on at least one general
  diagnostic;
- no-continuity and topic-shift controls are not made easier by label leakage or
  source continuity shortcuts;
- useful coverage appears at bounded size and latency.

If these fail, do not train a model on STM. In that case, the added layer is
complexity without evidence.

## First Shadow Proxy Run

Command:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --short-term-memory-shadow-experiments \
  --models=models \
  --output-dir build/short_term_memory_shadow
```

Status:

- cases: 421 repaired replay cases.
- references: 270.
- no-anchor controls: 151.
- production retrieval changed: false.
- runtime: shadow-only proxy over repaired replay candidates and recent signal
  embeddings.
- limitation: this first surface has reference/no-anchor rows only. Wrong-active
  and stale same-source appear as candidate evidence, but not as separate
  row-level control cases in this proxy pass.

Artifacts:

- `stm_coverage_cases.csv`
- `stm_coverage_summary.json`
- `stm_leakage_audit.json`
- `stm_leakage_failures.csv`
- `stm_knob_sweep_results.json`
- `stm_knob_sweep_summary.csv`
- `stm_anchor_reachability_results.json`
- `stm_anchor_reachability_cases.csv`
- `stm_anchor_reachability_failures.csv`
- `stm_boundary_ablation_results.json`
- `stm_boundary_ablation_cases.csv`
- `stm_complexity_budget.json`
- `stm_consumer_ablation_results.json`
- `stm_consumer_ablation_summary.csv`
- `stm_source_continuity_trap_results.json`
- `stm_source_continuity_trap_cases.csv`
- `stm_manual_inspection_cases.jsonl`
- `stm_manual_inspection_summary.json`
- `stm_shadow_experiment_results.json`

### Experiment Results

#### 1. Coverage Audit

STM increased target evidence coverage relative to surfaced WM:

| slice | cases | target in WM | target in STM |
|---|---:|---:|---:|
| distance 1 | 100 | 100 | 99 |
| distance 2-4 | 113 | 113 | 107 |
| distance 5-12 | 57 | 0 | 54 |
| all references | 270 | 213 | 260 |

This is the strongest pro-STM signal in the first pass: STM recovers delayed
target evidence that WM cannot hold.

#### 2. No-Leak Chronology Audit

Result:

- checked cases: 421.
- future leak rows: 0.
- current signal in `STM_before_t`: 0.
- label fields in runtime rows: 0.

The proxy construction passes the basic chronology test.

#### 3. Knob Sweep

The first proxy only showed Focus movement:

- F=0.3: mean STM items 14.0, target present 267 / 270.
- F=0.5: mean STM items 13.57, target present 260 / 270.
- F=0.7: mean STM items 12.87, target present 250 / 270.

Sensitivity and Stability did not materially change the proxy metrics because
the pass uses a repaired replay candidate window rather than a true STM buffer
with TTL, pinning, boundary decay, and compaction. This is an implementation
gap, not a validated knob result.

#### 4. STM Anchor Reachability Diagnostic

STM improved target evidence presence but not ranking or safe commitment:

| variant | mean items | target present | target top-3 | no-anchor tempting | ref/control AUC |
|---|---:|---:|---:|---:|---:|
| working memory | 4.00 | 213 / 270 | 165 / 270 | 131 / 151 | 0.1817 |
| STM | 13.57 | 260 / 270 | 87 / 270 | 144 / 151 | 0.1819 |
| STM minus WM | 9.72 | 54 / 270 | 21 / 270 | 129 / 151 | 0.0707 |
| time-reversed STM | 13.57 | 260 / 270 | 85 / 270 | 144 / 151 | 0.1819 |

Interpretation: a larger STM evidence field improves coverage, especially for
delayed references, but a naive cosine scorer over STM is not useful. Order did
not matter for this scorer, so a real STM model must demonstrate an advantage
over shuffled/time-reversed controls.

#### 5. Boundary Ablation

The boundary proxy was not decisive:

- hard FIFO / stability-retention proxy: target present 260 / 270, top-3 87.
- decay-unlinked proxy: target present 54 / 270, top-3 21.

The proxy shows the obvious tradeoff: aggressive boundary decay removes delayed
target evidence. A true STM buffer needs boundary-aware anchor-linked retention,
not simple FIFO or simple decay.

#### 6. Complexity Budget

The proxy item counts are bounded:

| capacity | mean items | p95 items |
|---:|---:|---:|
| 16 | 13.46 | 14 |
| 32 | 14.28 | 15 |
| 64 | 14.28 | 15 |
| 128 | 14.28 | 15 |
| 256 | 14.28 | 15 |

This repaired replay surface saturates around 15 prior candidate/evidence
items, so it cannot yet test a true 64-128 item STM budget.

#### 7. Consumer Ablation

STM helped coverage but not the simple anchor diagnostic:

- anchor diagnostic STM AUC: 0.1819.
- retrieval/WM diagnostic AUC: 0.1817.
- STM target top-3: 87 / 270.
- WM target top-3: 165 / 270.

No consumer should be promoted from this proxy result.

#### 8. Source-Continuity Trap

Source/recency-only remained competitive with full STM:

- full STM AUC: 0.1819.
- source/recency-only AUC: 0.1852.
- boundary-only AUC: 0.3944.
- order-only AUC: 0.5000, but with an artificial target-order artifact.

This is a warning sign. The first proxy does not prove entity evidence; it mostly
shows that expanded recent context is available and that naive scoring is
confounded by controls.

#### 9. Manual Inspection Pack

The run wrote `stm_manual_inspection_cases.jsonl` with audit-only current text,
active context, and STM item text references. These texts are not runtime
inputs; they are only for human inspection.

### Interpretation

The first STM proxy run does **not** justify model training. It supports a
narrower claim:

> A shadow STM layer can recover delayed target evidence absent from surfaced
> working memory.

It does not yet show:

- safe no-continuity abstention;
- topic-shift rejection;
- stale same-source rejection;
- useful order sensitivity;
- useful F/S/T Stability or Sensitivity behavior.

Next required experiment: implement a true chronological STM buffer, not just a
candidate-window proxy, and run it on a broader general-memory surface with
ordinary continuation, delayed continuation, topic-shift, stale same-source,
and no-continuity controls.

## General STM Shadow Proxy Run

After reframing STM as a general memory-system layer, we added a separate
benchmark mode:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --short-term-memory-general-shadow-experiments \
  --models=models \
  --output-dir build/short_term_memory_general_shadow
```

This pass is still benchmark-only and proxy-based. It does not change
production retrieval. The surface has 480 cases:

- 120 ordinary continuation positives;
- 120 delayed continuation positives;
- 120 same-dataset topic-shift controls;
- 120 stale same-source controls.

Artifacts:

- `stm_general_utility_cases.csv`
- `stm_general_utility_results.json`
- `stm_general_leakage_audit.json`
- `stm_general_knob_sweep_results.json`
- `stm_general_boundary_results.json`
- `stm_general_consumer_results.json`
- `stm_general_complexity_budget.json`
- `stm_general_failure_examples.csv`
- `stm_general_shadow_results.json`

### General Utility Results

| variant | mean items | target present | target top-3 | tempting controls | AUC | inverted AUC | zero-FPR | 5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| working memory | 4.00 | 120 / 240 | 97 / 240 | 196 / 240 | 0.1980 | 0.8020 | 0 | 1 |
| recent centroids | 8.00 | 180 / 240 | 81 / 240 | 209 / 240 | 0.2392 | 0.7608 | 0 | 3 |
| STM | 13.61 | 234 / 240 | 69 / 240 | 226 / 240 | 0.2442 | 0.7558 | 0 | 4 |
| STM minus WM | 9.77 | 118 / 240 | 27 / 240 | 216 / 240 | 0.1205 | 0.8795 | 0 | 4 |
| source/recency only | 4.00 | 120 / 240 | 120 / 240 | 240 / 240 | 0.2500 | 0.7500 | 0 | 120 |
| time-reversed STM | 13.61 | 234 / 240 | 69 / 240 | 226 / 240 | 0.2442 | 0.7558 | 0 | 4 |

Coverage by positive slice:

| slice | cases | target in WM | target in STM |
|---|---:|---:|---:|
| ordinary continuation | 120 | 120 | 116 |
| delayed continuation | 120 | 0 | 118 |

Control behavior:

| control slice | cases | tempting STM controls |
|---|---:|---:|
| topic shift | 120 | 110 |
| stale same-source | 120 | 116 |

Interpretation: STM strongly improves delayed-continuity coverage. Working
memory has no delayed targets in this proxy (`0 / 120`), while STM has
`118 / 120`. However, naive STM scoring is not a useful consumer: topic-shift
and stale same-source controls are also highly tempting, raw AUC is inverted,
and zero-FPR recovery is still zero.

### Chronology And Cost

The leakage audit passed:

- checked cases: 480;
- future-leak rows: 0;
- current-signal-in-STM-before rows: 0;
- label-fields-in-runtime rows: 0.

The capacity proxy saturates quickly:

| capacity | mean items | p95 items |
|---:|---:|---:|
| 16 | 12.91 | 14 |
| 32 | 14.26 | 15 |
| 64 | 14.26 | 15 |
| 128 | 14.26 | 15 |
| 256 | 14.26 | 15 |

This pass still cannot stress a true 64-128 item STM budget because the repaired
candidate-window proxy tops out around 15 items.

### Knob And Boundary Findings

The knob sweep again showed weak movement because this is not a true
chronological STM buffer. Focus changed item counts slightly, but
Sensitivity/Stability did not produce meaningful TTL, pinning, boundary decay,
or compaction behavior. Boundary ablations reproduced the coverage/safety
tradeoff: aggressive stale-only or STM-minus-WM views reduce target coverage but
do not remove enough tempting controls.

### General STM Decision

This general pass is more relevant than the anchor-shaped first pass. It says:

- STM is worth continuing as a possible evidence reservoir because it recovers
  delayed continuity that WM cannot hold.
- STM is not worth promoting as a readable context source yet because stale and
  topic-shift controls score too highly.
- The next implementation must be a true chronological buffer with real
  boundary decay, TTL, compaction, and order-sensitive consumers. A
  candidate-window proxy cannot answer the latency/complexity question by
  itself.

## Chronological STM Buffer Shadow Run

We then added a true chronological-buffer shadow mode:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --short-term-memory-chronological-shadow-experiments \
  --models=models \
  --output-dir build/short_term_memory_chronological_shadow
```

Unlike the candidate-window proxy, this pass builds `STM_before_t` from actual
prior turns in the source conversation. It still does not change production
retrieval and still uses proxy labels for ordinary continuation, delayed
continuation, topic-shift controls, and stale same-source controls.

Artifacts:

- `stm_chronological_cases.csv`
- `stm_chronological_results.json`
- `stm_chronological_knob_sweep_results.json`
- `stm_chronological_leakage_audit.json`
- `stm_chronological_complexity_budget.json`
- `stm_chronological_failure_examples.csv`
- `stm_chronological_shadow_results.json`

### Chronological Buffer Results

| variant | mean items | p95 items | target present | target top-3 | tempting controls | AUC | inverted AUC | zero-FPR | 5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| working memory | 4.00 | 4 | 120 / 240 | 97 / 240 | 181 / 240 | 0.2264 | 0.7736 | 1 | 4 |
| recent centroids | 8.00 | 8 | 180 / 240 | 81 / 240 | 201 / 240 | 0.2759 | 0.7241 | 2 | 4 |
| chronological STM | 18.54 | 26 | 234 / 240 | 65 / 240 | 224 / 240 | 0.2763 | 0.7237 | 1 | 5 |
| STM minus WM | 14.71 | 22 | 118 / 240 | 21 / 240 | 217 / 240 | 0.1240 | 0.8760 | 1 | 4 |
| source/recency only | 4.00 | 4 | 120 / 240 | 120 / 240 | 240 / 240 | 0.2500 | 0.7500 | 0 | 120 |
| time-reversed STM | 18.54 | 26 | 234 / 240 | 65 / 240 | 224 / 240 | 0.2763 | 0.7237 | 1 | 5 |

Coverage by positive slice:

| slice | cases | target in WM | target in chronological STM |
|---|---:|---:|---:|
| ordinary continuation | 120 | 120 | 116 |
| delayed continuation | 120 | 0 | 118 |

Control behavior:

| control slice | cases | tempting chronological STM controls |
|---|---:|---:|
| topic shift | 120 | 113 |
| stale same-source | 120 | 111 |

The chronology audit passed:

- checked cases: 480;
- future-leak rows: 0;
- current-signal-in-STM-before rows: 0;
- label-fields-in-runtime rows: 0.

The buffer size is still bounded on this surface:

| capacity | mean items | p95 items |
|---:|---:|---:|
| 16 | 15.31 | 16 |
| 32 | 19.45 | 27 |
| 64 | 19.45 | 27 |
| 128 | 19.45 | 27 |
| 256 | 19.45 | 27 |

The full benchmark took 3900.84 ms for 480 cases, with the synthetic complexity
loop itself taking 0.62 ms.

### Chronological Buffer Interpretation

This is the first result that tests STM as an actual prior-turn buffer. It
confirms the central positive result: STM recovers delayed continuity evidence
that WM drops. But it also confirms the central risk: stale and topic-shift
controls remain highly tempting, raw utility AUC remains inverted, and
time-reversed STM performs identically. The current simple cosine consumer is
therefore not using order or boundary structure.

The next useful STM experiment should not add more reachability metrics. It
should add real boundary-aware retention and an order-sensitive consumer, then
test whether those reduce topic-shift and stale same-source false positives
without losing delayed continuity evidence.

## Boundary / Order Consumer Shadow Run

We extended the chronological STM pass with four simple shadow consumers:

- `recency_weighted_semantic`;
- `boundary_decay_semantic`;
- `order_sensitive_trace`;
- `boundary_order_consumer`.

Command:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --short-term-memory-chronological-shadow-experiments \
  --models=models \
  --output-dir build/short_term_memory_chronological_shadow_boundary_order
```

### Boundary / Order Results

| variant | mean items | target present | top-3 | tempting controls | AUC | inverted AUC | zero-FPR | 5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plain STM | 18.54 | 234 / 240 | 65 | 224 / 240 | 0.2763 | 0.7237 | 1 | 5 |
| recency-weighted semantic | 9.21 | 193 / 240 | 113 | 95 / 240 | 0.3089 | 0.6911 | 1 | 9 |
| boundary-decay semantic | 10.83 | 208 / 240 | 92 | 187 / 240 | 0.2685 | 0.7315 | 1 | 4 |
| order-sensitive trace | 16.71 | 227 / 240 | 102 | 181 / 240 | 0.3161 | 0.6839 | 1 | 7 |
| boundary + order | 6.47 | 163 / 240 | 107 | 151 / 240 | 0.2685 | 0.7315 | 1 | 5 |
| time-reversed boundary + order | 6.47 | 163 / 240 | 107 | 151 / 240 | 0.2685 | 0.7315 | 1 | 5 |

Control temptation by class:

| variant | topic shift | stale same-source |
|---|---:|---:|
| plain STM | 113 / 120 | 111 / 120 |
| recency-weighted semantic | 41 / 120 | 54 / 120 |
| boundary-decay semantic | 95 / 120 | 92 / 120 |
| order-sensitive trace | 88 / 120 | 93 / 120 |
| boundary + order | 74 / 120 | 77 / 120 |

The result is mixed but useful:

- recency weighting gives the strongest false-positive reduction, from
  224 / 240 tempting controls to 95 / 240, but loses 41 positive targets;
- order-sensitive trace is the best reachability/safety tradeoff, retaining
  227 / 240 positives while reducing tempting controls to 181 / 240;
- boundary + order reduces tempting controls to 151 / 240 but loses too much
  evidence, retaining only 163 / 240 positives;
- time-reversed boundary + order matches normal boundary + order, so this
  consumer still does not prove useful sequence-order sensitivity.

### Boundary / Order Interpretation

This pass changes the answer slightly. STM is not merely an uncontrolled
candidate expansion: simple retention/read policies can reduce stale and
topic-shift pressure. But none of the hand-built policies are good enough to
justify production complexity. Low-FPR recovery remains near zero, raw AUC is
still inverted, and order controls still do not separate.

The next experiment should test a learned or calibrated shadow consumer over
chronological STM features, with the objective explicitly balanced between
delayed-evidence retention and stale/topic-shift rejection. If that still fails,
STM should remain an evidence reservoir only, not a readable context source.

## Calibrated STM Consumer Shadow Run

We added a small logistic/calibrated shadow consumer over chronological STM
features. It uses only runtime-compatible features derived from the current
embedding and `STM_before_t`:

- plain STM best score and margin;
- working-memory and recent-centroid best scores;
- recency-weighted, boundary-decay, trace, and boundary/order best scores;
- STM and WM counts;
- best recent score;
- best stale score;
- recent-minus-stale score gap;
- mean STM score;
- mean STM distance;
- current step index.

Labels are used only for training/evaluation in the benchmark. No production
retrieval behavior changes.

Command:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --short-term-memory-chronological-shadow-experiments \
  --models=models \
  --output-dir build/short_term_memory_chronological_shadow_calibrated
```

Artifacts:

- `stm_chronological_calibrated_cases.csv`
- `stm_chronological_calibrated_consumer_results.json`

### Calibrated Consumer Results

| validation | AUC | positives | controls | zero-FPR recovery | 5% FPR recovery | tempting stale | tempting topic-shift |
|---|---:|---:|---:|---:|---:|---:|---:|
| chronological holdout | 0.7362 | 96 | 96 | 10 | 21 | 17 / 48 | 15 / 48 |
| source-held-out | 0.6536 | 240 | 240 | 22 | 55 | 27 / 120 | 34 / 120 |

Mean source-held-out scores by class:

| class | mean score |
|---|---:|
| ordinary continuation | 0.5636 |
| delayed continuation | 0.4537 |
| stale same-source control | 0.3480 |
| topic-shift control | 0.4029 |

This is the first STM result that improves the safety side of the tradeoff.
Compared with plain chronological STM's `224 / 240` tempting controls, the
source-held-out calibrated consumer leaves `61 / 240` controls above 0.5
(`27` stale and `34` topic-shift). Zero-FPR recovery is no longer zero:
`22 / 240` on source-held-out and `10 / 96` on chronological holdout.

### Calibrated Consumer Interpretation

STM now has a plausible path to justify its complexity:

- chronological STM preserves delayed evidence that WM drops;
- hand-built read policies reduce false positives but cannot reach low-FPR
  operation;
- a small calibrated consumer gets nonzero low-FPR recovery and much lower
  stale/topic-shift firing.

This is not production evidence yet. The source-held-out AUC is only 0.6536,
delayed continuation scores are weaker than ordinary continuation scores, and
the benchmark still uses proxy labels over text dialogue replay. But the result
is strong enough to continue with a proper shadow STM buffer implementation and
better held-out surfaces.

## Retention-Policy STM Shadow Run

We then added runtime-compatible retention/read variants to approximate real
STM policies:

- `ttl_8_semantic`;
- `ttl_16_semantic`;
- `boundary_gate_semantic`;
- `boundary_soft_decay_semantic`.

Command:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --short-term-memory-chronological-shadow-experiments \
  --models=models \
  --output-dir build/short_term_memory_chronological_shadow_retention
```

### Retention Policy Results

| variant | mean items | target present | top-3 | tempting controls | AUC | zero-FPR | 5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain STM | 18.54 | 234 / 240 | 65 | 224 / 240 | 0.2763 | 1 | 5 |
| TTL-8 semantic | 7.47 | 174 / 240 | 88 | 172 / 240 | 0.2968 | 1 | 4 |
| TTL-16 semantic | 13.90 | 227 / 240 | 89 | 177 / 240 | 0.3236 | 1 | 4 |
| boundary gate semantic | 12.85 | 200 / 240 | 77 | 147 / 240 | 0.3910 | 1 | 6 |
| boundary soft decay | 14.38 | 222 / 240 | 80 | 187 / 240 | 0.3342 | 1 | 6 |
| recency-weighted semantic | 9.21 | 193 / 240 | 113 | 95 / 240 | 0.3089 | 1 | 9 |
| order-sensitive trace | 16.71 | 227 / 240 | 102 | 181 / 240 | 0.3161 | 1 | 7 |

Control temptation by selected policy:

| variant | topic shift | stale same-source |
|---|---:|---:|
| plain STM | 113 / 120 | 111 / 120 |
| TTL-16 semantic | 90 / 120 | 87 / 120 |
| boundary gate semantic | 68 / 120 | 79 / 120 |
| recency-weighted semantic | 41 / 120 | 54 / 120 |

The boundary gate has the best hand-built AUC and cuts controls from
224 / 240 to 147 / 240, but loses 34 positives. TTL-16 is the better retention
compromise: it preserves 227 / 240 positives while cutting tempting controls to
177 / 240. Recency weighting still gives the strongest control reduction, but
at a larger coverage cost.

The calibrated consumer with the new retention features produced:

| validation | AUC | zero-FPR recovery | 5% FPR recovery | tempting stale | tempting topic-shift |
|---|---:|---:|---:|---:|---:|
| chronological holdout | 0.7372 | 10 / 96 | 22 / 96 | 16 / 48 | 14 / 48 |
| source-held-out | 0.6384 | 26 / 240 | 52 / 240 | 22 / 120 | 35 / 120 |

Compared with the previous calibrated consumer, the new features slightly
increase source-held-out zero-FPR recovery (`26` vs `22`) and reduce stale
temptation (`22 / 120` vs `27 / 120`), but reduce source-held-out AUC
(`0.6384` vs `0.6536`). This is not a clean win.

### Retention Policy Interpretation

The retention run supports adding real TTL and boundary hooks to a shadow STM
buffer. They move the right metrics. But the hand-built policies still do not
produce a safe operating point, and the calibrated consumer does not clearly
improve once the retention features are added.

The next implementation step should be a processor-local shadow STM buffer with
instrumented TTL, boundary decay, and compaction so we can measure real
per-ingress overhead and state growth. More proxy scoring variants are unlikely
to answer the complexity question.

## Shadow Buffer Policy Simulation

We added a processor-like shadow buffer simulator to the chronological STM mode.
This still runs inside the benchmark binary, not production retrieval, but it
updates a per-conversation buffer chronologically and records state growth,
TTL/boundary retention behavior, and per-update cost.

Command:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --short-term-memory-chronological-shadow-experiments \
  --models=models \
  --output-dir build/short_term_memory_chronological_shadow_buffer_policy
```

Artifacts:

- `stm_chronological_buffer_policy_results.json`
- `stm_chronological_buffer_policy_summary.csv`

### Buffer Policy Results

| policy | mean items | p95 items | target present | top-3 | tempting controls | AUC | zero-FPR | 5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIFO-32 | 19.45 | 27 | 240 / 240 | 65 | 224 / 240 | 0.2763 | 1 | 5 |
| TTL-16 | 15.31 | 16 | 240 / 240 | 68 | 216 / 240 | 0.2927 | 1 | 5 |
| TTL-32 | 19.45 | 27 | 240 / 240 | 65 | 224 / 240 | 0.2763 | 1 | 5 |
| boundary-gate TTL-32 | 13.49 | 25 | 207 / 240 | 78 | 186 / 240 | 0.3182 | 1 | 5 |
| compact-after-16 | 16.05 | 17 | 240 / 240 | 68 | 216 / 240 | 0.2927 | 1 | 5 |

The buffer simulation over 96 real conversations produced:

| metric | value |
|---|---:|
| updates | 1486 |
| mean buffer items | 8.27 |
| p95 buffer items | 15 |
| mean update time | 2090.81 us |
| p95 update time | 3053.67 us |
| compactions | 0 |

The timing includes the benchmark encoder cache lookup / embedding path plus
buffer maintenance, so it should be treated as an upper-bound shadow estimate,
not as final production STM overhead.

### Buffer Policy Interpretation

TTL-16 and compact-after-16 are attractive for complexity: both cap p95 items
near 16-17 and preserve all positive evidence on this surface. But they only
reduce tempting controls from 224 / 240 to 216 / 240, which is not enough.
Boundary-gate TTL-32 reduces controls further to 186 / 240 and improves AUC to
0.3182, but loses 33 positive cases.

This confirms the current direction:

- a bounded STM buffer is feasible from a state-growth perspective;
- TTL alone is not enough for false-positive control;
- boundary gating helps but is too destructive without a better retention
  signal;
- a production candidate needs a calibrated consumer and richer boundary
  features, not just a larger FIFO.

The next experiment should move the buffer into processor state in shadow mode
so the latency measurement reflects actual ingress work rather than benchmark
reconstruction.

## Processor-Local Shadow STM Run

We added a disabled-by-default processor operation,
`UpdateShortTermMemoryShadow`, that maintains the STM buffer inside
`ProcessorContext`. The operation runs only when
`CORTEXT_STM_SHADOW_ENABLE=1`, stores state in memory, and does not alter
retrieval, storage, consolidation, or returned memories.

Command:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --short-term-memory-processor-shadow-experiments \
  --models=models \
  --output-dir build/short_term_memory_processor_shadow
```

Artifacts:

- `stm_processor_shadow_results.json`
- `stm_processor_shadow_summary.csv`
- `stm_processor_shadow_cases.csv`

### Processor-Local Results

| metric | value |
|---|---:|
| processed conversations | 96 |
| updates | 1486 |
| TTL | 16 steps |
| capacity | 32 items |
| mean STM size | 8.27 |
| p95 STM size | 15 |
| max STM size | 17 |
| compactions | 0 |
| mean shadow update time | 4.05 us |
| p95 shadow update time | 7.24 us |
| mean minimal processor time | 233.62 us |
| p95 minimal processor time | 367.11 us |

The processor-local run separates STM maintenance from the earlier
benchmark-side reconstruction cost. The buffer update itself is
microsecond-scale on this replay surface, while the minimal `SignalProcessor`
pass remains sub-millisecond. State growth is also bounded: TTL-16 / capacity-32
keeps p95 buffer size at 15 items and never triggers compaction on the 96
conversation replay.

### Processor-Local Interpretation

This result changes the complexity question. A real shadow STM buffer is not
expensive enough to reject on latency grounds, at least for text replay and the
current bounded policy. The remaining question is value: STM must feed a
consumer that improves downstream behavior without increasing false positives.

### Processor-Backed Calibrated Consumer

We then connected the calibrated chronological consumer to snapshots captured
from the processor-maintained shadow buffer. This is still benchmark-only: the
benchmark inserts a local capture operation before the STM update operation,
records the buffer that existed before each ingress step, and reconstructs the
same 480 utility/control cases from those processor snapshots.

Command:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --short-term-memory-processor-shadow-experiments \
  --models=models \
  --output-dir build/short_term_memory_processor_shadow_calibrated
```

Artifacts:

- `stm_processor_shadow_results.json`
- `stm_processor_shadow_utility_results.json`
- `stm_processor_shadow_utility_cases.csv`
- `stm_chronological_calibrated_consumer_results.json`

The expanded processor run produced:

| metric | value |
|---|---:|
| processed conversations | 480 |
| updates | 9434 |
| snapshots | 9434 |
| non-empty snapshots | 8954 |
| utility cases | 480 |
| mean STM size | 10.10 |
| p95 STM size | 17 |
| max STM size | 17 |
| mean shadow update time | 4.77 us |
| p95 shadow update time | 10.08 us |
| mean minimal processor time | 1028.81 us |
| p95 minimal processor time | 1980.01 us |

Plain processor-backed STM preserved the same high-recall / unsafe-control
shape as the reconstructed STM:

| variant | target present | target top-3 | tempting controls | AUC | zero-FPR | 5% FPR |
|---|---:|---:|---:|---:|---:|---:|
| working memory | 120 / 240 | 97 | 181 / 240 | 0.2264 | 1 | 4 |
| STM | 234 / 240 | 66 | 219 / 240 | 0.2880 | 1 | 5 |
| TTL-16 | 227 / 240 | 89 | 177 / 240 | 0.3236 | 1 | 4 |
| recency weighted | 193 / 240 | 113 | 95 / 240 | 0.3089 | 1 | 9 |
| boundary gate | 200 / 240 | 77 | 146 / 240 | 0.3931 | 1 | 6 |

The calibrated processor-backed consumer produced:

| validation | AUC | zero-FPR recovery | 5% FPR recovery | tempting stale | tempting topic-shift |
|---|---:|---:|---:|---:|---:|
| chronological holdout | 0.7493 | 20 / 96 | 23 / 96 | 14 / 48 | 13 / 48 |
| source-held-out | 0.6636 | 34 / 240 | 57 / 240 | 23 / 120 | 33 / 120 |

This is stronger than the reconstructed calibrated pass on the key safety
metric: source-held-out zero-FPR recovery increases from `22 / 240` to
`34 / 240`, and 5% FPR recovery increases from `55 / 240` to `57 / 240`.
The result is still not a production gate, but it does show that the useful
calibrated STM signal survives when the buffer is maintained by the real
ingress pipeline.

### Processor-Backed Interpretation

STM now clears the first two complexity checks:

- bounded state growth is stable on the replay surface;
- processor-local shadow update overhead is microsecond-scale;
- calibrated consumption improves low-FPR behavior over naive STM and survives
  source-held-out validation.

The remaining work is to make the consumer more principled and non-proxy. STM
should stay shadow-only until the consumer is driven by explicit STM state,
boundary features, and F/S/T-derived policy rather than a benchmark logistic
calibrator.

## Processor-Backed TTL / Capacity Sweep

We then stress-tested the processor-backed buffer policy rather than adding a
new consumer. The benchmark now honors `CORTEXT_STM_SHADOW_TTL_STEPS` and
`CORTEXT_STM_SHADOW_CAPACITY`, so the same processor snapshot path can be run
with different retention policies.

Commands:

```bash
CORTEXT_STM_SHADOW_TTL_STEPS=8 CORTEXT_STM_SHADOW_CAPACITY=16 \
  ./build/examples/benchmark/cortext_anchor_replay_bench \
  --short-term-memory-processor-shadow-experiments \
  --models=models \
  --output-dir build/short_term_memory_processor_shadow_sweep_ttl8_cap16

CORTEXT_STM_SHADOW_TTL_STEPS=16 CORTEXT_STM_SHADOW_CAPACITY=32 \
  ./build/examples/benchmark/cortext_anchor_replay_bench \
  --short-term-memory-processor-shadow-experiments \
  --models=models \
  --output-dir build/short_term_memory_processor_shadow_sweep_ttl16_cap32

CORTEXT_STM_SHADOW_TTL_STEPS=32 CORTEXT_STM_SHADOW_CAPACITY=32 \
  ./build/examples/benchmark/cortext_anchor_replay_bench \
  --short-term-memory-processor-shadow-experiments \
  --models=models \
  --output-dir build/short_term_memory_processor_shadow_sweep_ttl32_cap32

CORTEXT_STM_SHADOW_TTL_STEPS=16 CORTEXT_STM_SHADOW_CAPACITY=64 \
  ./build/examples/benchmark/cortext_anchor_replay_bench \
  --short-term-memory-processor-shadow-experiments \
  --models=models \
  --output-dir build/short_term_memory_processor_shadow_sweep_ttl16_cap64
```

Combined artifacts:

- `build/short_term_memory_processor_shadow_policy_sweep/stm_processor_policy_sweep_results.json`
- `build/short_term_memory_processor_shadow_policy_sweep/stm_processor_policy_sweep_summary.csv`

### Policy Sweep Results

| policy | mean size | p95 size | p95 update | plain target | plain controls | source AUC | source zero-FPR | source 5% FPR | stale tempting | topic tempting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTL8 / cap16 | 7.17 | 9 | 8.17 us | 191 / 240 | 202 / 240 | 0.6709 | 34 / 240 | 61 / 240 | 18 / 120 | 30 / 120 |
| TTL16 / cap32 | 10.10 | 17 | 11.58 us | 234 / 240 | 219 / 240 | 0.6636 | 34 / 240 | 57 / 240 | 23 / 120 | 33 / 120 |
| TTL32 / cap32 | 10.64 | 21 | 12.96 us | 234 / 240 | 224 / 240 | 0.6384 | 26 / 240 | 52 / 240 | 22 / 120 | 35 / 120 |
| TTL16 / cap64 | 10.10 | 17 | 10.21 us | 234 / 240 | 219 / 240 | 0.6636 | 34 / 240 | 57 / 240 | 23 / 120 | 33 / 120 |

### Policy Sweep Interpretation

The sweep argues against a long-retention FIFO. TTL32/cap32 preserves target
coverage but increases control carryover and reduces source-held-out recovery.
Increasing capacity from 32 to 64 does not change the observed state or metrics
on this surface, because TTL16 is already the limiting factor.

TTL8/cap16 is the interesting alternative. It cuts p95 STM size from 17 to 9
and improves source-held-out 5% FPR recovery from `57 / 240` to `61 / 240`, with
fewer stale/topic temptations. The cost is target coverage: plain target
presence falls from `234 / 240` to `191 / 240`. Since calibrated zero-FPR
recovery remains tied at `34 / 240`, TTL8/cap16 is a plausible low-latency
policy for conservative modes, while TTL16/cap32 remains the better default for
retention-sensitive modes.

This gives STM an F/S/T mapping candidate:

- higher Stability should push toward TTL16/cap32 or longer;
- lower Stability should push toward TTL8/cap16;
- Focus/Sensitivity should primarily affect read/consumer strictness, not the
  write buffer size.

### Extended Retention Frontier

We extended the same processor-backed sweep with TTL4/cap8, TTL8/cap32,
TTL12/cap24, and TTL24/cap32 to check whether the tradeoff is smooth.

Artifacts:

- `build/short_term_memory_processor_shadow_policy_frontier/stm_processor_policy_frontier_results.json`
- `build/short_term_memory_processor_shadow_policy_frontier/stm_processor_policy_frontier_summary.csv`

| policy | p95 size | target present | tempting controls | source AUC | zero-FPR | 5% FPR | stale tempting | topic tempting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TTL4 / cap8 | 5 | 131 / 240 | 187 / 240 | 0.7066 | 24 / 240 | 55 / 240 | 23 / 120 | 33 / 120 |
| TTL8 / cap16 | 9 | 191 / 240 | 202 / 240 | 0.6709 | 34 / 240 | 61 / 240 | 18 / 120 | 30 / 120 |
| TTL8 / cap32 | 9 | 191 / 240 | 202 / 240 | 0.6709 | 34 / 240 | 61 / 240 | 18 / 120 | 30 / 120 |
| TTL12 / cap24 | 13 | 234 / 240 | 208 / 240 | 0.6843 | 31 / 240 | 63 / 240 | 28 / 120 | 38 / 120 |
| TTL16 / cap32 | 17 | 234 / 240 | 219 / 240 | 0.6636 | 34 / 240 | 57 / 240 | 23 / 120 | 33 / 120 |
| TTL24 / cap32 | 21 | 234 / 240 | 224 / 240 | 0.6405 | 27 / 240 | 53 / 240 | 23 / 120 | 36 / 120 |
| TTL32 / cap32 | 21 | 234 / 240 | 224 / 240 | 0.6384 | 26 / 240 | 52 / 240 | 22 / 120 | 35 / 120 |

The frontier is smooth enough to be useful. TTL4/cap8 is too destructive:
source AUC is high because many controls are removed, but target presence falls
to `131 / 240` and zero-FPR recovery drops. TTL12/cap24 is a good retention
point: it preserves `234 / 240` target presence and gives the best 5% FPR
recovery (`63 / 240`), but its zero-FPR recovery is lower than TTL8/TTL16.
TTL24 and TTL32 are dominated: they retain no more targets than TTL12/TTL16 and
bring back more unsafe carryover.

The practical frontier is therefore:

- `TTL8/cap16` for conservative / low-Stability mode;
- `TTL12/cap24` when 5% FPR recovery matters more than strict zero-FPR;
- `TTL16/cap32` when retention and zero-FPR are both important.

### Processor-Backed F/S/T Read Sweep

We also swept the existing F/S/T read proxy over the processor-backed
TTL16/cap32 snapshots. This varies `StmReadK`, the relevance floor, and the
summary metrics used by the naive STM reader, but it does not change the write
buffer or calibrated consumer.

Artifact:

- `build/short_term_memory_processor_shadow_knob_sweep/stm_processor_shadow_knob_sweep_summary.csv`

Result ranges across the 27 combinations:

| metric | min | max |
|---|---:|---:|
| target present | 226 / 240 | 236 / 240 |
| tempting controls | 219 / 240 | 219 / 240 |
| AUC | 0.2879 | 0.2880 |
| zero-FPR recovery | 1 / 240 | 1 / 240 |
| 5% FPR recovery | 5 / 240 | 5 / 240 |

The F/S/T read proxy is effectively non-discriminative in this form. Focus
changes target coverage slightly, but control carryover and low-FPR recovery do
not move. This is useful because it prevents us from over-interpreting the
knobs: the current evidence supports F/S/T controlling STM write-retention
policy and calibrated consumer strictness, not the naive read-k / relevance
floor proxy.

### Calibrated Consumer Threshold Frontier

Given the negative read-sweep result, we analyzed the calibrated consumer as a
thresholded decision surface. This uses the processor-backed TTL16/cap32
calibrated scores and reports exact false-positive budgets rather than treating
the logistic score as a production gate.

Artifacts:

- `build/short_term_memory_processor_shadow_consumer_threshold_frontier/stm_consumer_threshold_frontier.json`
- `build/short_term_memory_processor_shadow_consumer_threshold_frontier/stm_consumer_threshold_frontier.csv`

Source-held-out frontier:

| FP budget | threshold | recovery | false positives | precision | recall | stale FP | topic FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0.8919 | 34 / 240 | 0 | 1.000 | 0.142 | 0 | 0 |
| 1% | 0.7940 | 42 / 240 | 3 | 0.933 | 0.175 | 0 | 3 |
| 2% | 0.7767 | 48 / 240 | 5 | 0.906 | 0.200 | 0 | 5 |
| 5% | 0.7079 | 55 / 240 | 12 | 0.821 | 0.229 | 1 | 11 |
| 10% | 0.6243 | 73 / 240 | 24 | 0.753 | 0.304 | 7 | 17 |
| 20% | 0.5332 | 104 / 240 | 48 | 0.684 | 0.433 | 18 | 30 |

Chronological holdout frontier:

| FP budget | threshold | recovery | false positives | precision | recall | stale FP | topic FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0.8566 | 38 / 240 | 0 | 1.000 | 0.158 | 0 | 0 |
| 1% | 0.8156 | 51 / 240 | 3 | 0.944 | 0.212 | 0 | 3 |
| 2% | 0.7921 | 55 / 240 | 5 | 0.917 | 0.229 | 1 | 4 |
| 5% | 0.7248 | 73 / 240 | 12 | 0.859 | 0.304 | 6 | 6 |
| 10% | 0.6411 | 113 / 240 | 24 | 0.825 | 0.471 | 11 | 13 |
| 20% | 0.5376 | 148 / 240 | 48 | 0.755 | 0.617 | 24 | 24 |

This is the cleanest current STM control surface. It shows a monotonic
precision/recall tradeoff and makes the source-held-out cost explicit:
relaxing from zero-FP to a strict 5% FP budget adds 21 recovered positives
(`34` to `55`) at the cost of 12 false positives, mostly topic shifts. This is
where Focus/Sensitivity should act in a future shadow consumer: Focus can raise
or lower the commit threshold, while Sensitivity can control how much recovery
is allowed before false-positive budgets are exceeded.

### Joint Policy / Threshold Frontier

Finally, we combined the retention frontier with calibrated threshold budgets.
This answers which write policy should be paired with each consumer strictness
level instead of selecting retention and threshold independently.

Artifacts:

- `build/short_term_memory_processor_shadow_policy_threshold_frontier/stm_policy_threshold_frontier.json`
- `build/short_term_memory_processor_shadow_policy_threshold_frontier/stm_policy_threshold_frontier.csv`
- `build/short_term_memory_processor_shadow_policy_threshold_frontier/stm_policy_threshold_best_by_budget.csv`

Best source-held-out policy by false-positive budget:

| FP budget | policy | p95 size | recovery | false positives | precision | threshold |
|---|---|---:|---:|---:|---:|---:|
| 0% | TTL8 / cap16 | 9 | 34 / 240 | 0 | 1.000 | 0.8271 |
| 1% | TTL12 / cap24 | 13 | 42 / 240 | 3 | 0.933 | 0.7927 |
| 2% | TTL12 / cap24 | 13 | 51 / 240 | 5 | 0.911 | 0.7535 |
| 5% | TTL12 / cap24 | 13 | 63 / 240 | 12 | 0.840 | 0.7059 |
| 10% | TTL12 / cap24 | 13 | 80 / 240 | 24 | 0.769 | 0.6214 |
| 20% | TTL4 / cap8 | 5 | 122 / 240 | 48 | 0.718 | 0.5479 |

Best chronological policy by false-positive budget:

| FP budget | policy | p95 size | recovery | false positives | precision | threshold |
|---|---|---:|---:|---:|---:|---:|
| 0% | TTL12 / cap24 | 13 | 39 / 240 | 0 | 1.000 | 0.8306 |
| 1% | TTL16 / cap32 | 17 | 51 / 240 | 3 | 0.944 | 0.8156 |
| 2% | TTL12 / cap24 | 13 | 56 / 240 | 5 | 0.918 | 0.7762 |
| 5% | TTL12 / cap24 | 13 | 76 / 240 | 12 | 0.864 | 0.7272 |
| 10% | TTL16 / cap32 | 17 | 113 / 240 | 24 | 0.825 | 0.6411 |
| 20% | TTL16 / cap32 | 17 | 148 / 240 | 48 | 0.755 | 0.5376 |

The joint frontier refines the default policy:

- `TTL8/cap16` is the best strict zero-FP source-held-out policy;
- `TTL12/cap24` is the best source-held-out policy for 1-10% FP budgets and
  the best chronological policy at 2-5%;
- `TTL16/cap32` is only favored when chronological high-recall budgets dominate.

This gives a concrete shadow mapping for knobs:

- low Stability or high Focus: `TTL8/cap16` with a zero-FP threshold;
- balanced mode: `TTL12/cap24` with a 2-5% FP threshold;
- high Stability / high recall mode: `TTL16/cap32` with a looser threshold.

### Distance Retention / Complexity Knee Diagnostic

We then audited the same processor-backed policy sweep by target distance and
control-score tail. This asks whether larger STM buffers add real delayed
continuity value or mainly preserve more tempting controls.

Artifacts:

- `build/short_term_memory_processor_shadow_distance_retention/stm_distance_retention_results.json`
- `build/short_term_memory_processor_shadow_distance_retention/stm_distance_retention_summary.csv`
- `build/short_term_memory_processor_shadow_distance_retention/stm_distance_retention_frontier.csv`
- `build/short_term_memory_processor_shadow_distance_retention/stm_distance_retention_cases.csv`

The current replay surface has immediate references at distance 1 and delayed
references at distances 5-12; it does not contain distance 2-4 cases, so that
bucket remains untested in this diagnostic.

| policy | p95 STM size | p95 update us | distance-1 target present | distance 5-12 target present | distance 5-12 top-3 | topic-shift p95 best score | stale p95 best score |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTL4 / cap8 | 5 | 7.48 | 116 / 120 | 15 / 120 | 5 / 120 | 0.811 | 0.837 |
| TTL8 / cap16 | 9 | 9.38 | 116 / 120 | 75 / 120 | 18 / 120 | 0.827 | 0.843 |
| TTL12 / cap24 | 13 | 9.75 | 116 / 120 | 118 / 120 | 17 / 120 | 0.834 | 0.848 |
| TTL16 / cap32 | 17 | 10.08 | 116 / 120 | 118 / 120 | 13 / 120 | 0.844 | 0.875 |
| TTL32 / cap32 | 21 | 10.43 | 116 / 120 | 118 / 120 | 13 / 120 | 0.844 | 0.875 |

This establishes a bounded-size knee. `TTL8/cap16` is the first useful delayed
retention point: it raises delayed target presence from `15 / 120` to
`75 / 120` while keeping p95 size at `9`. `TTL12/cap24` nearly saturates
delayed target presence at `118 / 120` with p95 size `13`. Larger buffers do not
recover more delayed targets and slightly worsen the control score tail,
especially stale same-source controls (`0.848` to `0.875` p95 best score).

The value of STM is therefore not unbounded capacity. It is a compact recent
evidence layer whose useful retention saturates around `TTL12/cap24` on this
surface. Future experiments should focus on safer consumers and better
calibration, not larger STM buffers.

### Presence vs Readout Gap Diagnostic

The distance-retention result says STM keeps the evidence. We then checked
whether a simple STM reader can actually surface that evidence. This diagnostic
compares target presence in STM against the target's naive top-3 rank by STM
score. It uses the same processor-backed policy sweep and does not change
retrieval.

Artifacts:

- `build/short_term_memory_processor_shadow_presence_readout_gap/stm_presence_readout_gap_results.json`
- `build/short_term_memory_processor_shadow_presence_readout_gap/stm_presence_readout_gap_summary.csv`
- `build/short_term_memory_processor_shadow_presence_readout_gap/stm_presence_readout_gap_cases.csv`

| policy | reference targets present | reference targets top-3 | readout gap | distance-1 present / top-3 | distance 5-12 present / top-3 |
|---|---:|---:|---:|---:|---:|
| TTL4 / cap8 | 131 / 240 | 87 / 240 | 44 | 116 / 82 | 15 / 5 |
| TTL8 / cap16 | 191 / 240 | 79 / 240 | 112 | 116 / 61 | 75 / 18 |
| TTL12 / cap24 | 234 / 240 | 72 / 240 | 162 | 116 / 55 | 118 / 17 |
| TTL16 / cap32 | 234 / 240 | 66 / 240 | 168 | 116 / 53 | 118 / 13 |
| TTL32 / cap32 | 234 / 240 | 65 / 240 | 169 | 116 / 52 | 118 / 13 |

This separates the STM storage question from the STM consumer question. Larger
STM buffers are useful reservoirs, but a naive best-score readout is not a safe
or sufficient consumer. At `TTL12/cap24`, STM retains `234 / 240` reference
targets, including `118 / 120` delayed targets, but only `72 / 240` references
and `17 / 120` delayed references place the target in the top 3. The readout
gap grows as STM capacity grows because more non-target recent evidence
competes with the retained target.

This is not a reason to reject STM. It is a reason not to expose STM directly
as another retrieval candidate pool. STM should remain a bounded substrate for
boundary, consolidation, and learned/structured consumers that can use order,
boundary state, and local score distribution instead of raw best cosine alone.

### Policy Robustness Diagnostic

We next asked whether the retained targets and tempting controls are stable
across retention policies or artifacts of a single TTL/capacity choice. For
each replay case, we counted how many of the eight tested policies retained the
target, placed the target in the top 3, or produced a high control score.

Artifacts:

- `build/short_term_memory_processor_shadow_policy_robustness/stm_policy_robustness_results.json`
- `build/short_term_memory_processor_shadow_policy_robustness/stm_policy_robustness_summary.csv`
- `build/short_term_memory_processor_shadow_policy_robustness/stm_policy_robustness_cases.csv`
- `build/short_term_memory_processor_shadow_policy_robustness/stm_policy_robustness_histogram.csv`

| class | count | target present all policies | target present any policy | target top-3 all policies | target top-3 any policy | target never top-3 | control >= 0.7 all policies | control >= 0.7 any policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ordinary continuation | 120 | 116 | 116 | 52 | 82 | 38 | - | - |
| delayed continuation | 120 | 15 | 118 | 1 | 26 | 94 | - | - |
| topic-shift control | 120 | - | - | - | - | - | 18 | 33 |
| stale same-source control | 120 | - | - | - | - | - | 25 | 34 |

This reinforces the previous split. Immediate-continuation target retention is
robust across policies. Delayed-continuation retention is policy-sensitive:
only `15 / 120` delayed targets survive every policy, but `118 / 120` appear
under at least one policy. Readout is far less robust: `94 / 120` delayed
continuations never place the target in the naive top 3 under any policy.

The control risk is also not just a large-buffer artifact. `18 / 120`
topic-shift controls and `25 / 120` stale same-source controls exceed a `0.7`
best-score threshold under every policy. The remaining work is not more
retention. It is consumer discrimination that can use STM's temporal structure
without treating every retained item as live context.

### Memory Footprint / Update Budget Diagnostic

We also turned the processor-backed sweep into a concrete memory and latency
budget. This estimates vector payload per active source for the current 256d
embedding path, a future 1280d semantic path, and an AAIT-style 1280d semantic
plus 128d anchor-key path. The estimates include a small metadata budget per
STM item but exclude allocator overhead and persistent storage.

Artifacts:

- `build/short_term_memory_processor_shadow_memory_budget/stm_memory_budget_results.json`
- `build/short_term_memory_processor_shadow_memory_budget/stm_memory_budget_summary.csv`

| policy | p95 STM items | p95 update us | p95 256d KiB / source | p95 AAIT KiB / source | p95 AAIT MiB / 100 sources |
|---|---:|---:|---:|---:|---:|
| TTL8 / cap16 | 9 | 8.17 | 10.41 | 50.91 | 4.97 |
| TTL12 / cap24 | 13 | 9.75 | 15.02 | 73.52 | 7.18 |
| TTL16 / cap32 | 17 | 11.58 | 19.64 | 96.14 | 9.39 |
| TTL32 / cap32 | 21 | 12.96 | 24.25 | 118.75 | 11.60 |

This answers the raw complexity question. At the useful `TTL12/cap24` retention
knee, p95 shadow update latency is under `10 us`. Even with a future 1280d
semantic vector plus 128d anchor key, p95 resident vector payload is about
`73.5 KiB` per active source, or `7.18 MiB` per 100 active sources. That is
small enough that STM should not be rejected on memory or update latency
grounds. The risk is consumer calibration and readout, not the bounded buffer
itself.

### STM Gate Review

We consolidated the latest STM diagnostics into explicit pass/mixed/fail gates
so the complexity decision is not hidden across separate tables.

Artifacts:

- `build/short_term_memory_processor_shadow_gate_review/stm_gate_review_results.json`
- `build/short_term_memory_processor_shadow_gate_review/stm_gate_review.csv`

| gate | status | metric | value | interpretation |
|---|---|---|---|---|
| bounded retention value | pass | TTL12/cap24 target presence | 234 / 240 | STM retains nearly all positive evidence at p95 size 13. |
| delayed retention value | pass | TTL12/cap24 delayed target presence | 118 / 120 | Delayed evidence is retained before the current step. |
| direct readout value | fail | TTL12/cap24 top-3 readout | 72 / 240 overall; 17 / 120 delayed | STM must not be exposed as direct retrieval expansion. |
| latency budget | pass | TTL12/cap24 p95 update latency | 9.75 us | Update cost is negligible relative to model encoding. |
| memory budget | pass | TTL12/cap24 AAIT-sized p95 payload | 7.18 MiB / 100 active sources | Memory footprint is not the blocker. |
| policy robust delayed | mixed | delayed target present any/all policies | any 118 / 120; all 15 / 120 | Delayed retention exists but depends on TTL/Stability. |
| control risk not persistent | fail | controls >=0.7 under all policies | topic 18 / 120; stale 25 / 120 | Some controls remain tempting regardless of retention policy. |
| threshold transfer | fail | TopicalChat-calibrated raw score on Taskmaster at nominal 5% FP | observed FPR 0.387 | Raw consumer calibration does not transfer. |
| local-feature transfer mitigation | mixed | avg/logistic local features on Taskmaster at nominal 5% FP | FPR 0.089 / 0.125 | Local features reduce but do not fix transfer damage. |
| production retrieval unchanged | pass | all STM diagnostics shadow-only | true | STM evidence remains benchmark/shadow-only. |

The gate decision is `continue_shadow_only`. STM is worth keeping as a bounded
internal substrate because retention, memory, and latency gates pass. It is not
ready as a retrieval-visible or direct-readout layer because naive readout,
persistent controls, and threshold transfer fail. The next useful work is a
safer STM consumer for boundary, consolidation, and continuity diagnostics, not
larger buffers or anchor-specific promotion.

### Safe Threshold Envelope Diagnostic

Finally, we converted the transfer failure into an explicit safe-threshold
envelope. For each score family and held-out split, we compare the threshold
chosen on the training side to the oracle threshold required to satisfy the
same false-positive budget on the held-out side. The oracle threshold is
diagnostic only; it is not a deployable calibration.

Artifacts:

- `build/short_term_memory_processor_shadow_safe_threshold_envelope/stm_safe_threshold_envelope_results.json`
- `build/short_term_memory_processor_shadow_safe_threshold_envelope/stm_safe_threshold_envelope.csv`
- `build/short_term_memory_processor_shadow_safe_threshold_envelope/stm_safe_threshold_envelope_review.csv`
- `build/short_term_memory_processor_shadow_safe_threshold_envelope/stm_safe_threshold_best_oracle.csv`

Selected held-out thresholds:

| split | score | nominal FP budget | train threshold | oracle held-out threshold | transfer FP / FPR | oracle recovery |
|---|---|---:|---:|---:|---:|---:|
| TopicalChat -> Taskmaster | raw | 0% | 0.7233 | 0.8678 | 9 / 0.054 | 24 / 168 |
| TopicalChat -> Taskmaster | raw | 5% | 0.4891 | 0.7249 | 65 / 0.387 | 48 / 168 |
| TopicalChat -> Taskmaster | avg(raw, percentile) | 0% | 1.9352 | 2.2735 | 4 / 0.024 | 26 / 168 |
| TopicalChat -> Taskmaster | avg(raw, percentile) | 5% | 1.4170 | 1.7656 | 18 / 0.107 | 49 / 168 |

This gives the safe operating envelope. A zero-FP Taskmaster threshold exists,
but it is much stricter than the TopicalChat-calibrated threshold and recovers
only `24 / 168` raw-score positives. The local-normalized ensemble improves the
damage profile but still needs a held-out threshold lift. This reinforces the
shadow-only decision: STM should report transfer-calibrated budgets and
explicit threshold margins before any consumer is treated as safe.

### False-Positive Taxonomy

We then decomposed the unsafe held-out selections by control class and
conversation concentration. This answers whether the remaining STM risk is just
stale evidence or a broader topic-shift/boundary problem.

Artifacts:

- `build/short_term_memory_processor_shadow_false_positive_taxonomy/stm_false_positive_taxonomy_results.json`
- `build/short_term_memory_processor_shadow_false_positive_taxonomy/stm_false_positive_taxonomy_summary.csv`
- `build/short_term_memory_processor_shadow_false_positive_taxonomy/stm_false_positive_taxonomy_cases.csv`

Selected TopicalChat-to-Taskmaster transfer cases:

| score / threshold | recovery | false positives | topic-shift FP | stale FP | precision | observed FPR |
|---|---:|---:|---:|---:|---:|---:|
| raw, trained 5% threshold | 105 / 168 | 65 / 168 | 37 | 28 | 0.618 | 0.387 |
| raw, oracle 5% threshold | 48 / 168 | 8 / 168 | 7 | 1 | 0.857 | 0.048 |
| avg(raw, percentile), trained 5% threshold | 67 / 168 | 18 / 168 | 14 | 4 | 0.788 | 0.107 |

Unsafe STM transfer is split across topic-shift and stale carryover, with topic
shifts slightly dominant. Raising the threshold can make the raw score safe on
the held-out side, but it cuts recovery by more than half. The local-normalized
ensemble reduces false positives substantially, but still exceeds the nominal
budget and leaves topic-shift controls as the largest residual source. Future
STM consumers need explicit topic-shift/boundary discrimination plus stale
filtering; stale filtering alone will not solve the safety problem.

### Control-Specific Frontier

We also evaluated topic-shift and stale same-source controls separately. This
keeps the same STM score families but calibrates thresholds against each
control class to see whether one scalar can be safe for both.

Artifacts:

- `build/short_term_memory_processor_shadow_control_specific_frontier/stm_control_specific_frontier_results.json`
- `build/short_term_memory_processor_shadow_control_specific_frontier/stm_control_specific_frontier.csv`
- `build/short_term_memory_processor_shadow_control_specific_frontier/stm_control_specific_auc_summary.csv`
- `build/short_term_memory_processor_shadow_control_specific_frontier/stm_control_specific_review.csv`

| score | topic AUC | topic zero-FP recovery | stale AUC | stale zero-FP recovery | all-control 5% recovery / FP |
|---|---:|---:|---:|---:|---:|
| raw | 0.658 | 31 / 240 | 0.711 | 52 / 240 | 63 / 240, 12 FP |
| source-centered | 0.674 | 24 / 240 | 0.736 | 38 / 240 | 64 / 240, 12 FP |
| source percentile | 0.663 | 8 / 240 | 0.733 | 0 / 240 | 62 / 240, 12 FP |
| avg(raw, percentile) | 0.668 | 34 / 240 | 0.730 | 55 / 240 | 65 / 240, 12 FP |

Topic-shift and stale controls are both hard, but they respond differently to
normalization. Raw scoring is better against stale at strict zero-FP
(`52 / 240`) than against topic shifts (`31 / 240`). Source-local percentile
normalization is especially conservative against stale controls at zero-FP but
does not improve topic-shift safety enough to be a full solution. No scalar
score cleanly handles both control classes. This further points to a structured
STM consumer that uses boundary state, source-local score distribution, and
stale-state features together.

### Conjunctive Consumer Diagnostic

To test a minimal structured consumer, we trained diagnostic-only two-feature
AND rules on the training side of each split. Each rule requires both feature
thresholds to pass, for example `raw >= t_raw` and
`source_percentile >= t_percentile`. This is still a static threshold consumer,
not a learned model and not a production retrieval change.

Artifacts:

- `build/short_term_memory_processor_shadow_conjunctive_consumer/stm_conjunctive_consumer_results.json`
- `build/short_term_memory_processor_shadow_conjunctive_consumer/stm_conjunctive_consumer_results.csv`
- `build/short_term_memory_processor_shadow_conjunctive_consumer/stm_conjunctive_consumer_review.csv`
- `build/short_term_memory_processor_shadow_conjunctive_consumer/stm_conjunctive_consumer_cases.csv`

Best held-out transfer rows selected by the test-side FP budget:

| split | nominal FP budget | rule | held-out recovery | held-out FP / FPR | train recovery / FP |
|---|---:|---|---:|---:|---:|
| TopicalChat -> Taskmaster | 0% | raw AND source percentile | 20 / 168 | 0 / 0.000 | 8 / 0 |
| TopicalChat -> Taskmaster | 2% | raw AND source z | 20 / 168 | 2 / 0.012 | 12 / 0 |
| TopicalChat -> Taskmaster | 5% | raw AND source centered | 66 / 168 | 17 / 0.101 | 21 / 3 |
| Taskmaster -> TopicalChat | 5% | raw AND source percentile | 11 / 72 | 1 / 0.014 | 48 / 8 |
| chronological 60/40 | 5% | raw AND source centered | 28 / 96 | 5 / 0.052 | 39 / 7 |

The AND rules reduce held-out false positives compared with scalar raw scores,
but they do so by becoming conservative. On the difficult TopicalChat ->
Taskmaster transfer, `raw AND source_percentile` at nominal zero-FP achieves
true zero-FP transfer but recovers only `20 / 168`. At nominal 5%, the best
reviewed AND rule recovers `66 / 168` with `17 / 168` false positives, much
safer than the raw scalar transfer (`105 / 168` recovery with `65 / 168` false
positives) but still over the intended 5% budget.

This is a useful negative result. Simple feature conjunction helps safety, but
two static thresholds are not enough to make STM a reliable consumer. The
consumer needs structured boundary/stale features or a learned calibration
surface with held-out transfer constraints.

### Failure Example Pack

To make the remaining consumer problem auditable, we exported a compact failure
pack rather than only aggregate metrics. The pack contains concrete replay case
IDs and score fields for the main failure categories.

Artifacts:

- `build/short_term_memory_processor_shadow_failure_pack/stm_failure_example_pack_results.json`
- `build/short_term_memory_processor_shadow_failure_pack/stm_failure_example_pack_summary.csv`
- `build/short_term_memory_processor_shadow_failure_pack/stm_failure_example_pack.csv`

| category | count | reason |
|---|---:|---|
| raw transfer false positives | 20 | Highest-scoring Taskmaster controls selected by the TopicalChat-calibrated raw 5% threshold. |
| conjunctive residual false positives | 17 | Controls still selected by the two-feature AND consumer under the TopicalChat -> Taskmaster 5% diagnostic. |
| delayed retained but buried | 30 | Delayed targets retained in `TTL12/cap24` STM but ranked outside naive top-3. |
| conservative zero-FP misses | 30 | Positives missed by the zero-FP conjunctive rule despite high raw score or retained evidence. |

This pack turns the aggregate conclusion into inspection targets. The unsafe
side contains high-scoring topic-shift and stale controls that survive both raw
and two-feature consumers. The missed-positive side contains delayed references
whose evidence is in STM but buried by other recent evidence, plus high-raw
positives rejected by conservative threshold conjunction. These are the cases a
future boundary/consolidation/continuity STM consumer must improve on.

### False-Positive Concentration Audit

We then checked whether the false positives are concentrated in a few bad
conversations or distributed across the held-out replay surface.

Artifacts:

- `build/short_term_memory_processor_shadow_fp_concentration/stm_fp_concentration_results.json`
- `build/short_term_memory_processor_shadow_fp_concentration/stm_fp_concentration_summary.csv`
- `build/short_term_memory_processor_shadow_fp_concentration/stm_fp_concentration_review.csv`
- `build/short_term_memory_processor_shadow_fp_concentration/stm_fp_concentration_conversations.csv`

| split / score | false positives | unique conversations | max FP / conversation | top-5 conversation share | topic FP | stale FP |
|---|---:|---:|---:|---:|---:|---:|
| TopicalChat -> Taskmaster, raw 5% train threshold | 65 | 26 | 4 | 0.277 | 37 | 28 |
| TopicalChat -> Taskmaster, avg(raw, percentile) 5% train threshold | 18 | 14 | 2 | 0.500 | 14 | 4 |
| TopicalChat -> Taskmaster, raw 5% oracle threshold | 8 | 7 | 2 | 0.750 | 7 | 1 |
| TopicalChat -> Taskmaster, raw zero-FP train threshold | 9 | 7 | 2 | 0.778 | 8 | 1 |

The main raw transfer failure is broad: `65` false positives span `26`
Taskmaster conversations, with at most `4` false positives in any one
conversation. The local-normalized ensemble reduces the count to `18`, but they
still span `14` conversations. Even the oracle-safe threshold leaves residual
topic-shift failures across multiple conversations. STM safety therefore needs
general boundary/topic-shift discrimination, not a small blocklist or
conversation-specific patch.

### Oracle Control Ablation

We used label-only oracle rejectors to estimate the headroom from two future
consumer signals: topic-shift/boundary rejection and stale-state filtering. The
oracle rejectors are diagnostic only; they are not runtime features.

Artifacts:

- `build/short_term_memory_processor_shadow_oracle_control_ablation/stm_oracle_control_ablation_results.json`
- `build/short_term_memory_processor_shadow_oracle_control_ablation/stm_oracle_control_ablation.csv`
- `build/short_term_memory_processor_shadow_oracle_control_ablation/stm_oracle_control_ablation_review.csv`

TopicalChat-to-Taskmaster, raw 5% transfer threshold:

| oracle rejector | recovery | false positives | topic FP | stale FP | precision |
|---|---:|---:|---:|---:|---:|
| none | 105 / 168 | 65 / 168 | 37 | 28 | 0.618 |
| reject topic-shift | 105 / 168 | 28 / 168 | 0 | 28 | 0.789 |
| reject stale | 105 / 168 | 37 / 168 | 37 | 0 | 0.739 |
| reject all controls | 105 / 168 | 0 / 168 | 0 | 0 | 1.000 |

This gives a target for the next STM consumer. Topic-shift/boundary rejection
has the larger first-order payoff, reducing false positives from `65` to `28`
without reducing recovered positives. Stale filtering alone reduces false
positives to `37`. Both signals are required for safe use. The fact that an
oracle can preserve `105 / 168` recovery at zero false positives means the STM
buffer itself contains useful evidence; the missing piece is a consumer that
can distinguish live continuity from boundary shifts and stale carryover.

### Consumer Requirement Envelope

We converted the oracle-control headroom into rejection-rate requirements for a
future STM consumer. These rates are computed over controls already selected by
the STM score threshold; they are diagnostic targets, not runtime features.

Artifacts:

- `build/short_term_memory_processor_shadow_consumer_requirement_envelope/stm_consumer_requirement_envelope_results.json`
- `build/short_term_memory_processor_shadow_consumer_requirement_envelope/stm_consumer_requirement_envelope.csv`
- `build/short_term_memory_processor_shadow_consumer_requirement_envelope/stm_consumer_requirement_review.csv`

TopicalChat-to-Taskmaster selected-control requirements:

| score operating point | target FP budget | selected controls | required rejections | equal rejection rate | topic rejection if stale perfect | stale rejection if topic perfect |
|---|---:|---:|---:|---:|---:|---:|
| raw nominal 5% | 0% | 65 | 65 | 1.000 | 1.000 | 1.000 |
| raw nominal 5% | 5% | 65 | 57 | 0.877 | 0.784 | 0.714 |
| raw nominal 0% | 5% | 9 | 1 | 0.111 | 0.000 | 0.000 |
| avg(raw, percentile) nominal 5% | 5% | 18 | 10 | 0.556 | 0.429 | 0.000 |
| avg(raw, percentile) nominal 0% | 5% | 4 | 0 | 0.000 | 0.000 | 0.000 |

The requirement is strong but measurable. If we keep the high-recovery raw 5%
operating point, a downstream rejector must remove `57 / 65` selected controls
to hit a true 5% held-out budget. If stale filtering were perfect, it would
still need to reject about `78%` of topic-shift selections; if topic-shift
rejection were perfect, it would still need to reject about `71%` of stale
selections. Starting from the safer `avg(raw, percentile)` score lowers the
requirement to `10 / 18` selected controls. This is the concrete target for a
future STM consumer: it must learn strong boundary/topic-shift rejection and
nontrivial stale filtering while preserving retained positives.

### Learned Scalar Rejector Diagnostic

We then tested whether a tiny post-score rejector over the existing scalar STM
features can meet that requirement. The rejector is a diagnostic logistic model
trained only on scalar/local features such as raw score, source-centered score,
source percentile, and simple interactions. It is benchmark-only and does not
change retrieval.

Artifacts:

- `build/short_term_memory_processor_shadow_learned_rejector/stm_learned_rejector_results.json`
- `build/short_term_memory_processor_shadow_learned_rejector/stm_learned_rejector_results.csv`
- `build/short_term_memory_processor_shadow_learned_rejector/stm_learned_rejector_review.csv`
- `build/short_term_memory_processor_shadow_learned_rejector/stm_learned_rejector_cases.csv`

TopicalChat-to-Taskmaster results:

| base score | base budget | rejector budget | held-out recovery | held-out FP | held-out FPR | positive loss from base selection |
|---|---:|---:|---:|---:|---:|---:|
| raw | 5% | 0% | 60 / 168 | 18 / 168 | 0.107 | 45 |
| raw | 5% | 5% | 92 / 168 | 52 / 168 | 0.310 | 13 |
| avg(raw, percentile) | 5% | 5% | 67 / 168 | 18 / 168 | 0.107 | 0 |

The scalar rejector does not meet the consumer requirement envelope. It can
reduce the raw 5% transfer failure from `65` false positives to `18`, but only
by losing `45` positives. If allowed to keep more positives, false positives
remain far above budget. Starting from `avg(raw, percentile)` preserves
positives but still leaves `18` false positives, matching the earlier fixed
ensemble. Scalar/local score features alone are therefore insufficient. The
next consumer needs additional boundary, event-shift, and stale-state signals.

### Score-Family Redundancy Audit

We then checked whether the scalar STM score families are independent enough to
support a better threshold-only consumer. The audit compares correlations,
top-decile overlap, and top-decile class composition across raw and local
normalization features.

Artifacts:

- `build/short_term_memory_processor_shadow_score_redundancy/stm_score_redundancy_results.json`
- `build/short_term_memory_processor_shadow_score_redundancy/stm_score_redundancy_correlations.csv`
- `build/short_term_memory_processor_shadow_score_redundancy/stm_score_top_decile_composition.csv`
- `build/short_term_memory_processor_shadow_score_redundancy/stm_score_top_decile_overlap.csv`
- `build/short_term_memory_processor_shadow_score_redundancy/stm_score_redundancy_auc.csv`

Key redundancy metrics:

| comparison | value |
|---|---:|
| raw vs source-percentile Spearman, all cases | 0.850 |
| raw vs source-percentile Spearman, controls only | 0.804 |
| raw/source-percentile top-decile Jaccard | 0.525 |
| raw top-decile composition | 44 positive, 4 topic-shift, 0 stale |
| source-percentile top-decile composition | 62 positive, 9 topic-shift, 3 stale |

The scalar score families are partially complementary but still highly
correlated. Source-local percentile scoring brings in more positives, including
more delayed positives, but it also admits topic-shift and stale controls. The
top-decile overlap between raw and percentile scores is already `0.525`, so the
features do not provide independent enough evidence for a threshold-only
consumer. This explains why fixed ensembles and scalar rejectors improve safety
modestly but plateau below the consumer requirement envelope.

### Dataset Score Drift Audit

We then measured score distribution drift directly. This explains why
TopicalChat-calibrated STM thresholds transfer poorly to Taskmaster.

Artifacts:

- `build/short_term_memory_processor_shadow_dataset_score_drift/stm_dataset_score_drift_results.json`
- `build/short_term_memory_processor_shadow_dataset_score_drift/stm_dataset_score_distribution.csv`
- `build/short_term_memory_processor_shadow_dataset_score_drift/stm_dataset_score_drift_ks.csv`
- `build/short_term_memory_processor_shadow_dataset_score_drift/stm_dataset_threshold_positions.csv`

Key drift metrics:

| metric | value |
|---|---:|
| Taskmaster minus TopicalChat raw control mean shift | 0.176 |
| Taskmaster minus TopicalChat raw control p95 shift | 0.233 |
| raw control KS distance | 0.478 |
| raw topic-shift KS distance | 0.468 |
| TopicalChat zero-FP raw threshold selecting TopicalChat controls | 0 / 72 |
| TopicalChat zero-FP raw threshold selecting Taskmaster controls | 9 / 168 |
| TopicalChat 5% raw threshold selecting TopicalChat controls | 3 / 72 |
| TopicalChat 5% raw threshold selecting Taskmaster controls | 65 / 168 |

The transfer failure is not random. Taskmaster controls have much higher raw
STM scores than TopicalChat controls, especially in the upper tail. A threshold
that is zero-FP on TopicalChat is already above `5%` FPR on Taskmaster; a
TopicalChat 5% threshold becomes `38.7%` FPR on Taskmaster. Local normalization
helps but does not eliminate this dataset drift. Safe STM consumers therefore
need either robust local calibration or features that detect boundary and
stale-state changes directly.

### Source-Held-Out Threshold Stability Audit

We then moved below dataset-level transfer and calibrated thresholds in a
leave-one-source-out loop. Each held-out conversation/source was evaluated with
thresholds chosen on controls from all other sources only.

Artifacts:

- `build/short_term_memory_processor_shadow_source_transfer/stm_source_heldout_transfer_results.json`
- `build/short_term_memory_processor_shadow_source_transfer/stm_source_heldout_transfer_summary.csv`
- `build/short_term_memory_processor_shadow_source_transfer/stm_source_heldout_transfer_cases.csv`

Key source-held-out metrics:

| score | FP budget | recovery | false positives | sources with any FP | p95 source FPR |
|---|---:|---:|---:|---:|---:|
| raw | 0% | 31 / 240 | 1 / 240 | 1 / 37 | 0.000 |
| raw | 5% | 62 / 240 | 12 / 240 | 9 / 37 | 0.333 |
| source-centered | 5% | 61 / 240 | 11 / 240 | 9 / 37 | 0.250 |
| source-percentile | 5% | 62 / 240 | 13 / 240 | 11 / 37 | 0.167 |

Aggregate source-held-out FPR can look acceptable while individual held-out
sources spike. Raw 5% calibration lands exactly at `12 / 240` false positives
overall, but `9 / 37` sources have at least one false positive and the p95
source FPR is `0.333`. Source-local normalization lowers some tail risk, but it
does not make source-level safety reliable. STM consumers should therefore
carry per-source safety margins or adaptive boundary checks, not only a global
threshold.

### Source-Risk-Constrained Frontier

Finally, we computed an oracle frontier that keeps the global false-positive
budget but also caps false positives per held-out source. This is not a
production thresholding strategy; it measures how much STM recovery survives
when safety is constrained at the conversation level instead of only in
aggregate.

Artifacts:

- `build/short_term_memory_processor_shadow_source_risk_frontier/stm_source_risk_frontier_results.json`
- `build/short_term_memory_processor_shadow_source_risk_frontier/stm_source_risk_frontier.csv`
- `build/short_term_memory_processor_shadow_source_risk_frontier/stm_source_risk_frontier_review.csv`
- `build/short_term_memory_processor_shadow_source_risk_frontier/stm_source_risk_selected_cases.csv`

At the 5% global false-positive budget:

| score | per-source cap | recovery | false positives | sources with FP | p95 source FPR | topic FP | stale FP |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | none | 63 / 240 | 11 / 240 | 9 / 37 | 0.333 | 10 | 1 |
| raw | max 1 FP/source | 53 / 240 | 6 / 240 | 6 / 37 | 0.167 | 5 | 1 |
| raw | zero FP/source | 31 / 240 | 0 / 240 | 0 / 37 | 0.000 | 0 | 0 |
| source-centered | none | 64 / 240 | 12 / 240 | 10 / 37 | 0.250 | 8 | 4 |
| source-centered | max 1 FP/source | 48 / 240 | 5 / 240 | 5 / 37 | 0.167 | 4 | 1 |
| source-percentile | none | 62 / 240 | 12 / 240 | 11 / 37 | 0.167 | 9 | 3 |
| source-percentile | max 1 FP/source | 35 / 240 | 3 / 240 | 3 / 37 | 0.167 | 2 | 1 |

The useful but narrow result is that STM can be made source-safer with a
threshold lift, but recovery drops quickly. Raw scoring loses `10` recovered
positives when capped at one false positive per source (`63` to `53`) and loses
`32` positives at zero source-level false positives (`63` to `31`). This makes
the consumer requirement sharper: STM is worth keeping as a substrate, but a
safe consumer cannot rely on threshold safety margins alone without giving back
much of the delayed-evidence benefit.

### Source-Risk Recovery Composition

We then decomposed the source-risk frontier to see what evidence is lost when
per-source safety is enforced.

Artifacts:

- `build/short_term_memory_processor_shadow_source_risk_recovery_composition/stm_source_risk_recovery_composition_results.json`
- `build/short_term_memory_processor_shadow_source_risk_recovery_composition/stm_source_risk_recovery_composition.csv`
- `build/short_term_memory_processor_shadow_source_risk_recovery_composition/stm_source_risk_lost_positive_summary.csv`
- `build/short_term_memory_processor_shadow_source_risk_recovery_composition/stm_source_risk_lost_positive_cases.csv`

At the 5% global false-positive budget:

| score | per-source cap | ordinary recovered | delayed recovered | Taskmaster recovered | TopicalChat recovered |
|---|---|---:|---:|---:|---:|
| raw | none | 45 | 18 | 54 | 9 |
| raw | max 1 FP/source | 41 | 12 | 45 | 8 |
| raw | zero FP/source | 28 | 3 | 24 | 7 |
| source-centered | none | 48 | 16 | 46 | 18 |
| source-centered | max 1 FP/source | 36 | 12 | 32 | 16 |
| source-percentile | none | 42 | 20 | 48 | 14 |
| source-percentile | max 1 FP/source | 33 | 2 | 27 | 8 |

The cap does not merely remove low-value hits. Raw scoring loses `6` delayed
positives when moving from no per-source cap to max-one-FP/source, and loses
`15` delayed positives at zero-FP/source. Source-percentile is even more fragile:
max-one-FP/source cuts delayed recovery from `20` to `2`. This means source
safety and delayed-evidence recovery are in direct tension for simple STM
scores. The next consumer needs boundary/stale discrimination that rejects
controls without raising the global threshold enough to bury delayed positives.

### Distance Rank-Decay Audit

We then measured whether distance failures are storage failures or readout
failures. This uses the retained-target cases from the TTL/capacity policy
sweep and records target rank and target-minus-best-nontarget margin.

Artifacts:

- `build/short_term_memory_processor_shadow_distance_rank_decay/stm_distance_rank_decay_results.json`
- `build/short_term_memory_processor_shadow_distance_rank_decay/stm_distance_rank_decay_summary.csv`
- `build/short_term_memory_processor_shadow_distance_rank_decay/stm_distance_rank_decay_examples.csv`

Key readout-decay metrics:

| group | retained | top-3 if retained | buried after top-3 | median rank | median target margin |
|---|---:|---:|---:|---:|---:|
| all distance 1 | 928 / 960 | 469 / 928 | 459 / 928 | 3 | -0.102 |
| all distance 5-12 | 755 / 960 | 110 / 755 | 645 / 755 | 8 | -0.252 |
| TTL12/cap24 distance 1 | 116 / 120 | 55 / 116 | 61 / 116 | 4 | -0.104 |
| TTL12/cap24 distance 5-12 | 118 / 120 | 17 / 118 | 101 / 118 | 8 | -0.248 |
| TTL8/cap16 distance 5-12 | 75 / 120 | 18 / 75 | 57 / 75 | 5 | -0.218 |
| TTL16/cap32 distance 5-12 | 118 / 120 | 13 / 118 | 105 / 118 | 10 | -0.252 |

This sharpens the STM value proposition. TTL12/cap24 retains delayed targets
almost perfectly (`118 / 120`), but naive ranking surfaces only `17 / 118` in
the top 3 and buries `101 / 118`. Larger retention makes the buried-rank problem
worse: TTL16/cap32 keeps the same delayed target presence but pushes the median
rank to `10`. STM therefore needs a consumer that attends over temporal
structure and boundary state; exposing more retained items to retrieval is not
enough.

### Readout-Depth Requirement Audit

The rank-decay result raises a concrete design question: how many retained STM
items must a future consumer inspect before delayed evidence becomes visible?
We swept top-K depth over the retained-target ranks.

Artifacts:

- `build/short_term_memory_processor_shadow_readout_depth_requirement/stm_readout_depth_results.json`
- `build/short_term_memory_processor_shadow_readout_depth_requirement/stm_readout_depth_summary.csv`
- `build/short_term_memory_processor_shadow_readout_depth_requirement/stm_readout_depth_requirement.csv`

Key readout-depth metrics:

| group | retained | top-3 of retained | top-5 of retained | top-8 of retained | top-12 of retained | top-16 of retained |
|---|---:|---:|---:|---:|---:|---:|
| TTL12/cap24 distance 1 | 116 / 120 | 0.474 | 0.612 | 0.810 | 0.991 | 1.000 |
| TTL12/cap24 distance 5-12 | 118 / 120 | 0.144 | 0.305 | 0.551 | 0.932 | 1.000 |
| TTL8/cap16 distance 5-12 | 75 / 120 | 0.240 | 0.507 | 0.880 | 1.000 | 1.000 |
| TTL16/cap32 distance 5-12 | 118 / 120 | 0.110 | 0.263 | 0.424 | 0.712 | 0.958 |

This gives a practical consumer target. A shallow top-3 STM readout misses most
delayed evidence even when the target is retained. At the useful TTL12/cap24
policy, top-8 reaches only `55.1%` of retained delayed targets, while top-12
reaches `93.2%`. The future STM consumer therefore needs to attend over roughly
8-12 recent items, not just expose the top few by scalar similarity. Longer
retention increases the required depth again, which reinforces the bounded
TTL12/cap24 default rather than a larger shadow buffer.

### Consumer Depth Cost/Benefit Estimate

We then combined the readout-depth results with the measured STM memory/update
budget to estimate how expensive a future STM consumer would be at different
inspection depths. This is a sizing audit, not an implementation benchmark.

Artifacts:

- `build/short_term_memory_processor_shadow_consumer_depth_cost_benefit/stm_consumer_depth_cost_benefit_results.json`
- `build/short_term_memory_processor_shadow_consumer_depth_cost_benefit/stm_consumer_depth_cost_benefit.csv`
- `build/short_term_memory_processor_shadow_consumer_depth_cost_benefit/stm_consumer_depth_requirements.csv`

For delayed distance 5-12 evidence:

| policy | target retained fraction | min K | delayed top-K of retained | estimated AAIT scan KiB p95 | p95 AAIT resident KiB/source |
|---|---:|---:|---:|---:|---:|
| TTL8/cap16 | 0.75 | 8 | 0.880 | 45.25 | 50.91 |
| TTL12/cap24 | 0.75 | 12 | 0.932 | 67.88 | 73.53 |
| TTL16/cap32 | 0.75 | 16 | 0.958 | 90.50 | 96.16 |
| TTL32/cap32 | 0.75 | 16 | 0.805 | 90.50 | 118.78 |

This gives a concrete engineering target. TTL12/cap24 reaches over 90% of
retained delayed evidence by inspecting 12 items, scanning about `67.9 KiB` of
AAIT-sized semantic+anchor vectors at p95. TTL16/cap32 needs 16 items and about
`90.5 KiB` for a similar retained-delay fraction, while TTL32/cap32 still does
not reach 90% by top-16. The likely default remains TTL12/cap24 with an STM
consumer that can attend over roughly 12 items; larger buffers add cost and make
ranking harder before they add clear value.

### Updated Complexity Gate Verdict

The latest audits supersede the earlier provisional gate review. We consolidated
retention, readout, transfer, source-risk, and cost results into an updated STM
complexity verdict.

Artifacts:

- `build/short_term_memory_processor_shadow_complexity_verdict/stm_complexity_verdict_results.json`
- `build/short_term_memory_processor_shadow_complexity_verdict/stm_complexity_verdict_gates.csv`
- `build/short_term_memory_processor_shadow_complexity_verdict/stm_complexity_verdict_summary.csv`

Gate summary:

| gate | status | value |
|---|---|---|
| bounded substrate | pass | 13 p95 items; 7.18 MiB / 100 sources; 9.75 us p95 update |
| delayed evidence retention | pass | TTL12/cap24 keeps 118 / 120 delayed targets |
| direct readout | fail | TTL12/cap24 surfaces only 17 / 118 retained delayed targets in top-3 |
| consumer depth target | pass | K=12; estimated 67.9 KiB AAIT-sized scan p95 |
| larger buffer value | fail | TTL16/cap32 needs K=16 / 90.5 KiB and worsens median rank to 10 |
| threshold transfer | fail | TopicalChat-calibrated raw 5% gives 65 / 168 FP on Taskmaster |
| source-level safety | fail | raw 5% has FP in 9 / 37 sources; p95 source FPR 0.333 |
| source-safety tradeoff | mixed | max-one-FP/source drops recovery 63 -> 53 and delayed 18 -> 12 |
| production retrieval | pass | unchanged |

Decision: keep STM as a shadow/internal substrate, but do not expose it as
direct retrieval expansion or a scalar-threshold readout. The current default
candidate is TTL12/cap24 with a bounded consumer depth around 12 items. The
next production-worthy experiment should be a bounded STM consumer with
boundary/stale discrimination and held-out/source-level calibration.

### Default Sensitivity Audit

We then checked whether TTL12/cap24 is actually a robust default candidate or
just an artifact of one table. This audit compares nearby shorter and longer
policies across delayed retention, readout depth, threshold recovery, control
tail, update cost, and AAIT-sized payload.

Artifacts:

- `build/short_term_memory_processor_shadow_default_sensitivity/stm_default_sensitivity_results.json`
- `build/short_term_memory_processor_shadow_default_sensitivity/stm_default_sensitivity_summary.csv`
- `build/short_term_memory_processor_shadow_default_sensitivity/stm_default_sensitivity_vs_ttl12.csv`

Key sensitivity metrics:

| policy | delayed retained | delayed top-12 of retained | 5% recovery | zero-FP recovery | control p95 | p95 items | AAIT MiB / 100 sources |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTL4/cap8 | 15 / 120 | 1.000 | 53 | 24 | 0.818 | 5 | 2.76 |
| TTL8/cap16 | 75 / 120 | 1.000 | 60 | 34 | 0.837 | 9 | 4.97 |
| TTL12/cap24 | 118 / 120 | 0.932 | 63 | 31 | 0.841 | 13 | 7.18 |
| TTL16/cap32 | 118 / 120 | 0.712 | 55 | 34 | 0.848 | 17 | 9.39 |
| TTL32/cap32 | 118 / 120 | 0.534 | 52 | 26 | 0.856 | 21 | 11.60 |

TTL12/cap24 remains the best default candidate. TTL8/cap16 is cheaper and has
shallower retained ranks, but it loses `43` delayed retained targets relative to
TTL12. TTL16/cap32 and TTL32/cap32 retain the same delayed targets as TTL12, but
top-12 delayed coverage falls to `0.712` and `0.534`, control tails rise, and
AAIT-sized memory grows. This supports a Stability-controlled default around
TTL12/cap24 rather than either a minimal buffer or a long-retention shadow
history.

### Incremental Value Over Minimal Buffer

We then treated TTL4/cap8 as a minimal recency buffer and measured what larger
STM policies buy relative to that baseline.

Artifacts:

- `build/short_term_memory_processor_shadow_incremental_value/stm_incremental_value_results.json`
- `build/short_term_memory_processor_shadow_incremental_value/stm_incremental_value_vs_minimal.csv`
- `build/short_term_memory_processor_shadow_incremental_value/stm_incremental_value_review.csv`

Incremental value over TTL4/cap8:

| policy | extra p95 items | extra AAIT MiB / 100 sources | delayed retained gain | delayed top-3 gain | 5% recovery gain | control p95 delta |
|---|---:|---:|---:|---:|---:|---:|
| TTL8/cap16 | +4 | +2.21 | +60 | +13 | +7 | +0.019 |
| TTL12/cap24 | +8 | +4.42 | +103 | +12 | +10 | +0.023 |
| TTL16/cap32 | +12 | +6.63 | +103 | +8 | +2 | +0.030 |
| TTL32/cap32 | +16 | +8.84 | +103 | +8 | -1 | +0.038 |

This makes the complexity tradeoff explicit. TTL12/cap24 buys nearly all of the
delayed-retention value over a minimal recency buffer (`+103` retained delayed
targets) at modest memory cost (`+4.42 MiB / 100 sources`). But the direct
readout gain is much smaller (`+12` delayed top-3 and `+10` 5%-budget
recoveries), which again says STM is valuable as latent substrate, not as a
retrieval-visible list.

### Policy Pareto Audit

We then formalized the policy tradeoff as a Pareto audit over latent-substrate,
direct-readout, safe-threshold, and balanced-default objectives.

Artifacts:

- `build/short_term_memory_processor_shadow_policy_pareto/stm_policy_pareto_results.json`
- `build/short_term_memory_processor_shadow_policy_pareto/stm_policy_pareto.csv`
- `build/short_term_memory_processor_shadow_policy_pareto/stm_policy_pareto_balanced_scores.csv`

Balanced normalized scores:

| policy | balanced score | delayed retained | 5% recovery | zero-FP recovery | top-12 delayed retained | AAIT MiB / 100 sources | control p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTL8/cap16 | 0.761 | 75 | 60 | 34 | 1.000 | 4.97 | 0.837 |
| TTL12/cap24 | 0.744 | 118 | 63 | 31 | 0.932 | 7.18 | 0.841 |
| TTL16/cap32 | 0.518 | 118 | 55 | 34 | 0.712 | 9.39 | 0.848 |
| TTL4/cap8 | 0.515 | 15 | 53 | 24 | 1.000 | 2.76 | 0.818 |
| TTL32/cap32 | 0.200 | 118 | 52 | 26 | 0.534 | 11.60 | 0.856 |

This prevents overclaiming the TTL12 default. TTL8/cap16 is the best
cost-weighted policy if shallow readout and footprint matter most. TTL12/cap24
is the better default only when delayed evidence retention is the priority,
because it keeps `118 / 120` delayed targets versus `75 / 120` for TTL8. Longer
retention policies are not justified by the current evidence: they retain the
same delayed targets as TTL12 but worsen readout depth, control tails, and
memory cost.

### Knob Policy Mapping Audit

We then mapped the non-dominated STM policies back to an F/S/T-style control
surface. This is diagnostic only; it records how knobs should affect STM if the
shadow layer is promoted later.

Artifacts:

- `build/short_term_memory_processor_shadow_knob_policy_mapping/stm_knob_policy_mapping_results.json`
- `build/short_term_memory_processor_shadow_knob_policy_mapping/stm_knob_policy_mapping.csv`
- `build/short_term_memory_processor_shadow_knob_policy_mapping/stm_knob_policy_risk_register.csv`

Diagnostic knob mapping:

| profile | S/T interpretation | policy | consumer depth | delayed retained | p95 items | AAIT MiB / 100 sources |
|---|---|---|---:|---:|---:|---:|
| low-stability / cost-conservative | low T or cost-sensitive S | TTL4/cap8 | 5 | 15 / 120 | 5 | 2.76 |
| balanced default | medium S/T | TTL8/cap16 | 8 | 75 / 120 | 9 | 4.97 |
| delayed-retention default | medium-high S/T | TTL12/cap24 | 12 | 118 / 120 | 13 | 7.18 |
| long retention diagnostic | very high S/T | TTL16/cap32+ | 16 | 118 / 120 | 17+ | 9.39+ |

The key knob rule is that retention and readout safety must not be the same
knob. Raising Stability/retention beyond TTL12 does not add delayed target
presence on this surface; it mainly increases rank competition, control tails,
and memory cost. A future production STM design should let Stability adjust
retention within the TTL8-12 range, while readout/commit thresholds remain
separately calibrated with held-out and source-level safety checks.

### Threshold Safety-Margin Sweep

We then tested whether a simple fixed margin above the calibrated STM threshold
could repair source-level safety without a richer consumer. This sweep uses the
TTL12/cap24 source-normalized cases and raises the nominal 5% threshold by a
fixed score margin.

Artifacts:

- `build/short_term_memory_processor_shadow_threshold_margin_sweep/stm_threshold_margin_sweep_results.json`
- `build/short_term_memory_processor_shadow_threshold_margin_sweep/stm_threshold_margin_sweep.csv`
- `build/short_term_memory_processor_shadow_threshold_margin_sweep/stm_threshold_margin_review.csv`
- `build/short_term_memory_processor_shadow_threshold_margin_sweep/stm_threshold_margin_requirements.csv`

Raw-score 5% threshold margin:

| margin | recovery | delayed recovery | false positives | sources with FP | p95 source FPR | precision |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 63 | 18 | 12 | 9 | 0.333 | 0.840 |
| 0.025 | 53 | 12 | 8 | 7 | 0.167 | 0.869 |
| 0.050 | 50 | 11 | 4 | 4 | 0.167 | 0.926 |
| 0.075 | 44 | 7 | 3 | 3 | 0.167 | 0.936 |
| 0.100 | 41 | 5 | 2 | 2 | 0.167 | 0.953 |
| 0.150 | 34 | 3 | 1 | 1 | 0.000 | 0.971 |

A fixed safety margin helps, but it is too blunt to be the consumer. Raising the
raw 5% threshold by `0.05` cuts false positives from `12` to `4`, but delayed
recovery falls from `18` to `11`. Raising by `0.10` leaves only `5` delayed
recoveries. Zero global false positives require a `0.20` margin and leave only
`1` delayed recovery. This reinforces the conclusion that STM needs a
boundary/stale-aware consumer, not only a stricter scalar threshold.

### Score-Overlap Band Audit

We then inspected where positives and controls occupy the same scalar-score
bands. This targets the ambiguity region a future STM consumer must resolve.

Artifacts:

- `build/short_term_memory_processor_shadow_score_overlap_bands/stm_score_overlap_bands_results.json`
- `build/short_term_memory_processor_shadow_score_overlap_bands/stm_score_overlap_band_summary.csv`
- `build/short_term_memory_processor_shadow_score_overlap_bands/stm_score_overlap_mixed_bands.csv`
- `build/short_term_memory_processor_shadow_score_overlap_bands/stm_score_overlap_cases.csv`

Raw-score overlap bands:

| raw band | positives | controls | ordinary | delayed | topic shift | stale | positive rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.50-0.60 | 30 | 38 | 17 | 13 | 19 | 19 | 0.441 |
| 0.60-0.70 | 30 | 14 | 9 | 21 | 8 | 6 | 0.682 |
| 0.70-0.756 | 13 | 10 | 6 | 7 | 7 | 3 | 0.565 |
| >=0.70 total | 63 | 14 | 45 | 18 | 11 | 3 | 0.818 |

The high-score region is mixed, not a clean commit zone. Raw scores above
`0.70` contain `63` positives and `14` controls. The critical 0.70-0.756 band
is especially ambiguous: it contains `7` delayed positives and `10` controls.
This is where a future STM consumer must use non-scalar evidence such as
boundary state, stale-state features, source drift, and temporal structure.

### Ambiguity-Band Oracle Headroom

We then measured the headroom available if a future non-scalar STM consumer
could perfectly reject controls inside the ambiguous score bands while keeping
positives. This is an oracle diagnostic, not a proposed runtime rule.

Artifacts:

- `build/short_term_memory_processor_shadow_ambiguity_oracle/stm_ambiguity_oracle_results.json`
- `build/short_term_memory_processor_shadow_ambiguity_oracle/stm_ambiguity_oracle_summary.csv`
- `build/short_term_memory_processor_shadow_ambiguity_oracle/stm_ambiguity_oracle_review.csv`
- `build/short_term_memory_processor_shadow_ambiguity_oracle/stm_ambiguity_oracle_cases.csv`

Raw-score oracle headroom:

| raw threshold | positives kept | delayed kept | controls to reject | topic reject | stale reject |
|---:|---:|---:|---:|---:|---:|
| 0.600 | 93 | 39 | 28 | 19 | 9 |
| 0.700 | 63 | 18 | 14 | 11 | 3 |
| 0.706 | 63 | 18 | 11 | 10 | 1 |
| 0.756 | 50 | 11 | 4 | 4 | 0 |
| 0.806 | 41 | 5 | 2 | 2 | 0 |

This shows why a richer consumer is worth testing. A fixed margin can reduce
controls, but it also discards delayed positives. An oracle consumer at
raw >= `0.60` would keep `93` positives, including `39` delayed positives, if it
could reject `28` controls. At raw >= `0.70`, it would keep `63` positives and
`18` delayed positives while rejecting only `14` controls. This makes the next
consumer target concrete: focus on the raw 0.6-0.8 overlap band and learn
boundary/stale rejection there without raising the threshold enough to bury
delayed positives.

### Consumer Rejector Requirement Audit

We converted the ambiguity-band oracle counts into concrete acceptance targets
for a future bounded STM consumer. This is still a shadow diagnostic: the
artifact answers how strong a control rejector must be before STM readout is
worth its complexity.

Artifacts:

- `build/short_term_memory_processor_shadow_consumer_rejector_requirements/stm_consumer_rejector_requirements_results.json`
- `build/short_term_memory_processor_shadow_consumer_rejector_requirements/stm_consumer_rejector_requirements.csv`
- `build/short_term_memory_processor_shadow_consumer_rejector_requirements/stm_consumer_rejector_targets.csv`

| score | threshold | positives | delayed | controls | reject rate for 5% FP | reject rate for 2% FP | keep positives to beat raw 5% | keep delayed to beat raw 5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.600 | 93 | 39 | 28 | 0.571 | 0.857 | 0.688 | 0.487 |
| raw | 0.700 | 63 | 18 | 14 | 0.143 | 0.714 | impossible | impossible |
| raw | 0.756 | 50 | 11 | 4 | 0.000 | 0.000 | impossible | impossible |
| source-centered | 0.150 | 91 | 34 | 27 | 0.556 | 0.852 | 0.703 | 0.559 |
| source-percentile | 0.875 | 62 | 20 | 12 | 0.000 | 0.667 | impossible | 0.950 |

The useful target is now quantitative. A raw >= `0.60` consumer has enough
recall headroom, but to fit the global 5% false-positive budget it must reject
`16 / 28` selected controls while keeping at least `64 / 93` positives and
`19 / 39` delayed positives. Raw >= `0.70` is safer but has almost no
retention slack: it only matches the prior raw 5% recovery if it keeps every
selected positive and every selected delayed reference. Raw >= `0.756` already
meets the 2% and 5% control budgets without a rejector, but it is below the
recovery baseline and loses most delayed cases.

This reframes the next consumer experiment. It should not optimize target rank
alone. It needs to operate near the raw 0.60 recall zone, reject at least about
57% of selected controls, and preserve roughly 69% of selected positives and
49% of selected delayed positives under held-out threshold calibration.

### Consumer Feasibility Grid

We then swept hypothetical consumer quality against those selected STM bands.
This is not a learned model; it asks whether plausible retain/reject rates
would actually beat the current raw 5% FP frontier.

Artifacts:

- `build/short_term_memory_processor_shadow_consumer_feasibility_grid/stm_consumer_feasibility_results.json`
- `build/short_term_memory_processor_shadow_consumer_feasibility_grid/stm_consumer_feasibility_grid.csv`
- `build/short_term_memory_processor_shadow_consumer_feasibility_grid/stm_consumer_feasibility_frontier.csv`

Minimum frontier to beat both raw 5% total recovery (`63`) and delayed recovery
(`18`):

| score | threshold | FP budget | feasible | control reject rate | positive keep rate | delayed keep rate | projected recovery / FP |
|---|---:|---|---|---:|---:|---:|---:|
| raw | 0.600 | zero FP | yes | 1.000 | 0.688 | 0.487 | 64 / 0 |
| raw | 0.600 | 2% FP | yes | 0.857 | 0.688 | 0.487 | 64 / 4 |
| raw | 0.600 | 5% FP | yes | 0.571 | 0.688 | 0.487 | 64 / 12 |
| raw | 0.700 | 5% FP | no | - | - | - | - |
| raw | 0.756 | 5% FP | no | - | - | - | - |
| source-centered | 0.150 | 5% FP | yes | 0.556 | 0.703 | 0.559 | 64 / 12 |
| source-percentile | 0.875 | 5% FP | no | - | - | - | - |

The feasibility grid narrows the design space. A raw >= `0.60` STM substrate can
beat the current frontier if a consumer keeps `64 / 93` selected positives,
keeps `19 / 39` selected delayed positives, and rejects `16 / 28` selected
controls. Source-centered scoring has similar safety requirements and slightly
harder retention requirements, so it is useful for domain balance but not a
cheaper safety path. Raw >= `0.70`, raw >= `0.756`, and source-percentile
`0.875` do not have enough recall headroom to beat both total and delayed
recovery. This keeps the decision shadow-only: STM has value as a substrate,
but only if a future consumer adds a real topic-shift/stale-control rejector.

### Source-Local Feature Gate Screen

We then tested whether the existing source-local score families can meet the
feasibility bar without adding a new model. Starting from the raw >= `0.60`
overlap band, we swept one- and two-feature gates over raw score,
source-centered score, source z-score, robust z-score, and source percentile.

Artifacts:

- `build/short_term_memory_processor_shadow_consumer_feature_gate_screen/stm_consumer_feature_gate_screen_results.json`
- `build/short_term_memory_processor_shadow_consumer_feature_gate_screen/stm_consumer_feature_gate_screen.csv`
- `build/short_term_memory_processor_shadow_consumer_feature_gate_screen/stm_consumer_feature_gate_frontier.csv`

Best gate frontiers:

| FP budget | useful gates found | best gate | positives | delayed | controls | precision |
|---|---:|---|---:|---:|---:|---:|
| zero FP | 0 | raw >= 0.869206 | 31 | 3 | 0 | 1.000 |
| 2% FP | 0 | raw >= 0.754816 | 51 | 11 | 4 | 0.927 |
| 5% FP | 6 | raw >= 0.608397 AND source-centered >= 0.223245 | 66 | 20 | 12 | 0.846 |

This is the first non-oracle indication that STM's high-recall overlap band can
be filtered into a useful 5% FP point without new runtime features: the best
simple gate keeps `66` positives and `20` delayed positives with `12` controls.
However, the same score-family gates do not meet the zero-FP or 2% FP bars.
They are therefore a cheap scaffold for a future consumer, not sufficient
evidence for direct STM readout.

### Feature-Gate Transfer Audit

Because the source-local feature gate was selected on the full surface, we then
repeated the gate selection under held-out dataset and chronological splits.
This tests whether score-family gates are real consumer evidence or only tuned
frontiers.

Artifacts:

- `build/short_term_memory_processor_shadow_feature_gate_transfer/stm_feature_gate_transfer_results.json`
- `build/short_term_memory_processor_shadow_feature_gate_transfer/stm_feature_gate_transfer_results.csv`
- `build/short_term_memory_processor_shadow_feature_gate_transfer/stm_feature_gate_transfer_by_source.csv`

Held-out 5% FP results:

| transfer | selected gate | recovery | delayed | FP | FPR | within budget |
|---|---|---:|---:|---:|---:|---|
| TopicalChat -> Taskmaster | source-centered >= 0.314287 | 31 | 7 | 3 | 0.018 | yes |
| Taskmaster -> TopicalChat | source-percentile >= 0.843750 | 13 | 2 | 2 | 0.028 | yes |
| chronological 60/40 | raw <= 0.707159 AND source-centered >= 0.058262 | 9 | 7 | 7 | 0.073 | no |
| fixed full-surface gate on Taskmaster | raw >= 0.608397 AND source-centered >= 0.223245 | 52 | 17 | 10 | 0.060 | no |
| fixed full-surface gate on TopicalChat | raw >= 0.608397 AND source-centered >= 0.223245 | 14 | 3 | 2 | 0.028 | yes |
| fixed full-surface gate on chronological holdout | raw >= 0.608397 AND source-centered >= 0.223245 | 28 | 9 | 5 | 0.052 | no |

This weakens the feature-gate result. Dataset-held-out selection stays within
the nominal 5% budget, but only by collapsing recovery well below the full
surface `66 / 20` result. Chronological transfer misses the 5% budget, and the
fixed full-surface gate also exceeds the 5% budget on Taskmaster and the
chronological holdout. Source-local score gates therefore remain useful as
diagnostics, not as a validated STM consumer.

### Feature-Gate Failure Taxonomy

We then decomposed the transfer failures by control class and missed-positive
class. This tests whether feature-gate failure is mostly stale carryover,
topic-shift confusion, or delayed-positive burial.

Artifacts:

- `build/short_term_memory_processor_shadow_feature_gate_failure_taxonomy/stm_feature_gate_failure_taxonomy_results.json`
- `build/short_term_memory_processor_shadow_feature_gate_failure_taxonomy/stm_feature_gate_failure_taxonomy_summary.csv`
- `build/short_term_memory_processor_shadow_feature_gate_failure_taxonomy/stm_feature_gate_failure_cases.csv`
- `build/short_term_memory_processor_shadow_feature_gate_failure_taxonomy/stm_feature_gate_fp_sources.csv`

Selected failure rows:

| gate / subset | positives | delayed | FP | topic FP | stale FP | delayed missed |
|---|---:|---:|---:|---:|---:|---:|
| fixed full-surface / all | 66 | 20 | 12 | 9 | 3 | 100 |
| fixed full-surface / Taskmaster | 52 | 17 | 10 | 7 | 3 | 67 |
| fixed full-surface / chronological holdout | 28 | 9 | 5 | 2 | 3 | 39 |
| TopicalChat->Taskmaster gate / Taskmaster | 31 | 7 | 3 | 3 | 0 | 77 |
| Taskmaster->TopicalChat gate / TopicalChat | 13 | 2 | 2 | 2 | 0 | 34 |
| chronological gate / holdout | 9 | 7 | 7 | 4 | 3 | 41 |

The main failure is not only stale same-source persistence. For the fixed
full-surface gate, `9 / 12` false positives are topic-shift controls. On
Taskmaster, `7 / 10` false positives are topic-shift controls. The held-out
dataset gates reduce false positives, but they do so by missing most delayed
positives: TopicalChat->Taskmaster keeps only `7 / 84` delayed positives, and
Taskmaster->TopicalChat keeps only `2 / 36`. A future STM consumer therefore
needs explicit boundary/topic-shift evidence. Stale decay alone cannot make
short-term memory worth the complexity.

### Topic-Boundary Oracle Decomposition

We then asked whether a perfect topic-shift rejector would be enough to make
the score-gate STM substrate useful, or whether stale rejection is equally
important. This is an oracle decomposition over the failure taxonomy, not a
runtime rule.

Artifacts:

- `build/short_term_memory_processor_shadow_topic_boundary_oracle/stm_topic_boundary_oracle_results.json`
- `build/short_term_memory_processor_shadow_topic_boundary_oracle/stm_topic_boundary_oracle.csv`
- `build/short_term_memory_processor_shadow_topic_boundary_oracle/stm_topic_boundary_oracle_review.csv`

Selected oracle scenarios:

| gate / subset | scenario | positives | delayed | projected FP | projected FPR | 2% FP | 5% FP |
|---|---|---:|---:|---:|---:|---|---|
| fixed full-surface / all | none | 66 | 20 | 12 | 0.050 | no | yes |
| fixed full-surface / all | reject all topic | 66 | 20 | 3 | 0.013 | yes | yes |
| fixed full-surface / Taskmaster | none | 52 | 17 | 10 | 0.060 | no | no |
| fixed full-surface / Taskmaster | reject all topic | 52 | 17 | 3 | 0.018 | yes | yes |
| fixed full-surface / chronological holdout | none | 28 | 9 | 5 | 0.052 | no | no |
| fixed full-surface / chronological holdout | reject all topic | 28 | 9 | 3 | 0.031 | no | yes |
| chronological gate / holdout | reject all topic | 9 | 7 | 3 | 0.031 | no | yes |
| fixed full-surface / all | reject topic and stale | 66 | 20 | 0 | 0.000 | yes | yes |

This gives a concrete consumer target. A topic-boundary rejector alone would
repair the fixed full-surface gate to the 2% and 5% budgets on the full surface
and Taskmaster while preserving `66` positives / `20` delayed positives overall
and `52` positives / `17` delayed positives on Taskmaster. It would also repair
the chronological holdout to the 5% budget. However, stale residuals still block
zero-FP and some 2% chronological transfer. The STM consumer architecture
should therefore prioritize boundary/topic-shift detection first and stale
closure second.

### Boundary Rejector Requirement Audit

We converted the topic-boundary oracle into explicit recall requirements for a
future boundary/stale consumer. This makes the acceptance bar concrete without
adding a runtime mechanism.

Artifacts:

- `build/short_term_memory_processor_shadow_boundary_rejector_requirements/stm_boundary_rejector_requirements_results.json`
- `build/short_term_memory_processor_shadow_boundary_rejector_requirements/stm_boundary_rejector_requirements.csv`
- `build/short_term_memory_processor_shadow_boundary_rejector_requirements/stm_boundary_rejector_doc_rows.csv`

Requirements for the fixed full-surface gate:

| subset | budget | original FP | topic FP | stale FP | reject needed | topic-only sufficient | topic recall needed | stale recall after perfect topic |
|---|---|---:|---:|---:|---:|---|---:|---:|
| all | zero FP | 12 | 9 | 3 | 12 | no | - | 1.000 |
| all | 2% FP | 12 | 9 | 3 | 8 | yes | 0.889 | 0.000 |
| all | 5% FP | 12 | 9 | 3 | 0 | yes | 0.000 | 0.000 |
| Taskmaster | 2% FP | 10 | 7 | 3 | 7 | yes | 1.000 | 0.000 |
| Taskmaster | 5% FP | 10 | 7 | 3 | 2 | yes | 0.286 | 0.000 |
| chronological holdout | 2% FP | 5 | 2 | 3 | 4 | no | - | 0.667 |
| chronological holdout | 5% FP | 5 | 2 | 3 | 1 | yes | 0.500 | 0.000 |

This gives a usable spec for the next STM consumer. For 5% FP, the boundary
rejector does not have to be perfect: the Taskmaster slice needs only `2 / 7`
topic false positives rejected, and the chronological holdout needs `1 / 2`.
For 2% FP, the all-surface fixed gate needs `8 / 9` topic FPs rejected, and
Taskmaster needs all `7 / 7`. Zero-FP and the stricter chronological 2% target
cannot be solved by topic rejection alone because stale residuals remain. The
consumer should therefore be evaluated as two separate heads or signals:
boundary/topic shift first, stale closure second.

### Boundary False-Positive Source Concentration

We then checked whether the remaining boundary false positives were concentrated
in a few conversations. If so, per-source caps or source-specific warmup could
be a cheaper alternative to a boundary detector.

Artifacts:

- `build/short_term_memory_processor_shadow_boundary_fp_source_concentration/stm_boundary_fp_source_concentration_results.json`
- `build/short_term_memory_processor_shadow_boundary_fp_source_concentration/stm_boundary_fp_source_concentration.csv`
- `build/short_term_memory_processor_shadow_boundary_fp_source_concentration/stm_boundary_fp_source_concentration_sources.csv`

| gate / subset | FP sources | total FP | topic FP | stale FP | max FP/source | FP removed by cap-1/source |
|---|---:|---:|---:|---:|---:|---:|
| fixed full-surface / all | 10 | 12 | 9 | 3 | 2 | 2 |
| fixed full-surface / Taskmaster | 9 | 10 | 7 | 3 | 2 | 1 |
| fixed full-surface / chronological holdout | 5 | 5 | 2 | 3 | 1 | 0 |
| chronological gate / holdout | 7 | 7 | 4 | 3 | 1 | 0 |

The false positives are broad, not a single-source pathology. A one-FP-per-source
oracle cap would remove only `2 / 12` false positives from the full fixed gate
and only `1 / 10` from Taskmaster; it removes none of the chronological-holdout
fixed-gate failures. Per-source caps may still be useful as risk governance, but
they are not a replacement for a boundary/topic-shift signal.

### Boundary Feature Separability Screen

We then tested whether the existing score-local features can themselves serve
as a boundary rejector inside the fixed STM gate. The screen measured AUC for
rejecting controls versus positives using raw score, source-centered score,
source z-score, robust z-score, and source percentile. It also swept single
feature thresholds as reject rules.

Artifacts:

- `build/short_term_memory_processor_shadow_boundary_feature_separability/stm_boundary_feature_separability_results.json`
- `build/short_term_memory_processor_shadow_boundary_feature_separability/stm_boundary_feature_separability_auc.csv`
- `build/short_term_memory_processor_shadow_boundary_feature_separability/stm_boundary_feature_threshold_screen.csv`
- `build/short_term_memory_processor_shadow_boundary_feature_separability/stm_boundary_feature_threshold_frontier.csv`

Best AUCs inside the fixed gate:

| subset / control | best feature | best AUC | direction |
|---|---|---:|---|
| all controls / all | source_z | 0.737 | lower rejects |
| all / topic-shift | raw | 0.709 | lower rejects |
| all / stale | source_z | 0.889 | lower rejects |
| Taskmaster / topic-shift | raw | 0.692 | lower rejects |

Single-threshold frontier:

| target | feasible | best rule | positives kept | delayed kept | FP left |
|---|---|---|---:|---:|---:|
| 5% FP + recovery | yes | source_robust_z <= 0.422748 rejects | 65 | 20 | 11 |
| 2% FP + recovery | no | - | - | - | - |
| zero FP + recovery | no | - | - | - | - |

The score-local features are not enough. Stale controls are somewhat separable,
but topic-shift controls are only weakly separable from positives inside the
fixed gate. No single score-local threshold can meet the 2% or zero-FP recovery
bar while keeping `64` positives and `19` delayed positives. This supports the
architecture requirement from the oracle audit: a future STM consumer needs new
stream/boundary evidence, not just another threshold over STM score variants.

### Boundary Stream-Proxy Separability

We then joined the fixed STM gate with processor utility fields to test whether
cheap stream proxies provide the missing boundary signal. Features included
current step, STM count, STM best score, step divided by STM count, and STM
pressure. These are still coarse counters, not true boundary/drift/coherence
features.

Artifacts:

- `build/short_term_memory_processor_shadow_boundary_stream_proxy_separability/stm_boundary_stream_proxy_results.json`
- `build/short_term_memory_processor_shadow_boundary_stream_proxy_separability/stm_boundary_stream_proxy_auc.csv`
- `build/short_term_memory_processor_shadow_boundary_stream_proxy_separability/stm_boundary_stream_proxy_threshold_screen.csv`
- `build/short_term_memory_processor_shadow_boundary_stream_proxy_separability/stm_boundary_stream_proxy_threshold_frontier.csv`

Best AUCs inside the fixed gate:

| subset / control | best stream proxy | best AUC | direction |
|---|---|---:|---|
| all controls / all | current_step | 0.776 | higher rejects |
| all / topic-shift | current_step | 0.742 | higher rejects |
| all / stale | current_step | 0.876 | higher rejects |
| all / topic-shift | stm_count | 0.700 | higher rejects |

Single-threshold frontier:

| target | feasible | best rule | positives kept | delayed kept | FP left |
|---|---|---|---:|---:|---:|
| 5% FP + recovery | yes | current_step >= 25 rejects | 65 | 19 | 11 |
| 2% FP + recovery | no | - | - | - | - |
| zero FP + recovery | no | - | - | - | - |

Coarse stream proxies are better than score-local features for identifying
late topic/stale failures, but they are still not a sufficient consumer.
`current_step >= 25` removes one positive and one stale false positive while
preserving the 5% budget, but no stream-proxy threshold reaches the 2% or
zero-FP recovery bar. This points to the actual feature requirement: the
consumer should use ingress boundary, drift, coherence, surprisal, or event
shift evidence, not just age/occupancy counters.

### Combined Score/Stream Gate Screen

Finally, we tested whether simple conjunctive reject rules over both score-local
and coarse stream-proxy features can meet the stricter consumer bar. The fixed
STM gate still started from `66` positives, `20` delayed positives, and `12`
controls.

Artifacts:

- `build/short_term_memory_processor_shadow_boundary_combined_gate_screen/stm_boundary_combined_gate_results.json`
- `build/short_term_memory_processor_shadow_boundary_combined_gate_screen/stm_boundary_combined_gate_screen.csv`
- `build/short_term_memory_processor_shadow_boundary_combined_gate_screen/stm_boundary_combined_gate_frontier.csv`

Combined-gate frontier:

| target | feasible | best rule | positives kept | delayed kept | FP left |
|---|---|---|---:|---:|---:|
| 5% FP + recovery | yes | source_centered <= 0.270233 AND step_over_stm_count >= 1.833333 rejects | 66 | 20 | 10 |
| 2% FP + recovery | no | - | - | - | - |
| zero FP + recovery | no | - | - | - | - |

The result is a useful stopping point for threshold-family diagnostics. Out of
`51,102` one- and two-feature rules, `10,667` can preserve the 5% recovery bar,
but none preserve `64` positives and `19` delayed positives while reaching the
2% or zero-FP bars. The best 5% rule rejects only one topic-shift and one stale
control without losing positives. This reinforces the current conclusion:
short-term memory is a valuable substrate, but its safe consumer requires real
ingress boundary/event evidence rather than more combinations of score and age
proxies.

### Consumer Stop-Condition Matrix

We then consolidated the STM consumer diagnostics into a stop-condition matrix.
This is the checkpoint for deciding whether to keep searching within
threshold/score/age proxy families.

Artifacts:

- `build/short_term_memory_processor_shadow_consumer_stop_conditions/stm_consumer_stop_condition_results.json`
- `build/short_term_memory_processor_shadow_consumer_stop_conditions/stm_consumer_stop_condition_matrix.csv`
- `build/short_term_memory_processor_shadow_consumer_stop_conditions/stm_consumer_acceptance_requirements.csv`

Stop-condition summary:

| question | result | evidence | implication |
|---|---|---|---|
| delayed evidence retained? | pass | distance 5-12 retained `118 / 120`; p95 STM size `13` | keep STM as a bounded substrate |
| realtime shadow cost? | pass | p95 update `9.75 us`; top-12 AAIT-sized scan `67.9 KiB` | cost does not block shadow STM |
| direct top-k readout? | fail | naive top-3 `72 / 240`; delayed top-3 `17 / 118` retained | do not expose STM as retrieval |
| scalar threshold transfer? | fail | TopicalChat 5% threshold gives Taskmaster `65 / 168` controls | no threshold-only consumer |
| score-local gates? | mixed | full-surface `66 / 20 / 12`, but transfer fails or loses recall | diagnostic only |
| score+stream gates? | fail strict | best `66 / 20 / 10`; no 2% or zero-FP recovery rule | stop threshold-family search |
| boundary/topic-shift hypothesis? | pass as hypothesis | fixed-gate FPs split `9` topic / `3` stale | next consumer needs boundary/event evidence |
| source caps? | fail | `12` FPs across `10` sources; cap-1/source removes only `2` | caps are governance only |

Acceptance requirements:

| requirement | status | bar | current evidence |
|---|---|---|---|
| bounded substrate | met | delayed retention near-complete with bounded p95 cost | `118 / 120` retained; p95 `13` items |
| direct readout | failed | top-k readout recovers delayed targets without deep scan | top-3 only `17 / 118` retained delayed |
| 5% consumer | partially met posthoc | keep >= `64` positives and >= `19` delayed with <= `12` FP | combined gate `66 / 20 / 10` on full surface |
| 2% consumer | failed | keep >= `64` positives and >= `19` delayed with <= `4` FP | `0 / 51102` combined rules |
| zero-FP consumer | failed | keep >= `64` positives and >= `19` delayed with `0` FP | `0 / 51102` combined rules |
| transfer safety | failed | held-out thresholds remain within budget | fixed gate exceeds 5% on Taskmaster and chronological holdout |
| next feature family | required | boundary/topic-shift plus stale closure evidence | topic oracle repairs 2%/5%; stale residuals block zero-FP |

Decision: continue STM shadow-only. The substrate is valuable enough to keep
testing, but threshold-family consumers are exhausted. The next valid
experiment needs actual ingress boundary, drift, coherence, surprisal, or
event-shift features plus stale closure. Production retrieval remains
unchanged.

### Boundary Policy Retrospective

Before starting new boundary-feature work, we reviewed the existing STM
boundary policy ablations to distinguish retention/decay policy from a true
boundary classifier. These older runs changed what the buffer retained around
boundaries; they did not provide a boundary/event-shift signal to the consumer.

Artifacts:

- `build/short_term_memory_processor_shadow_boundary_policy_retrospective/stm_boundary_policy_retrospective_results.json`
- `build/short_term_memory_processor_shadow_boundary_policy_retrospective/stm_boundary_policy_retrospective.csv`
- `build/short_term_memory_processor_shadow_boundary_policy_retrospective/stm_boundary_policy_retrospective_review.csv`

Boundary-policy retrospective:

| surface | variant | target present | target top-3 | tempting controls | zero-FPR recovery | 5% recovery |
|---|---|---:|---:|---:|---:|---:|
| old-style anchor | decay unlinked at boundary | 54 / 270 | 21 | 129 / 151 | - | - |
| old-style anchor | hard FIFO 32 | 260 / 270 | 87 | 144 / 151 | - | - |
| general STM | decay unlinked at boundary | 118 / 240 | 27 | 216 / 240 | 0 | 4 |
| general STM | hard FIFO 32 | 234 / 240 | 69 | 226 / 240 | 0 | 4 |
| general STM | stale-only | 118 / 240 | 27 | 216 / 240 | 0 | 4 |

The retrospective closes another tempting branch. Boundary-aware retention
policies trade target presence against tempting controls, but none create a
safe consumer. On the general STM surface, hard FIFO retains `234 / 240` targets
but also `226 / 240` tempting controls; decay variants reduce target presence to
`118 / 240` while still leaving `216 / 240` tempting controls. Zero-FPR recovery
stays `0` and 5% recovery stays `4` across these variants. The next boundary
experiment must add an event-shift / boundary feature to the consumer, not just
change when STM entries decay.

### Boundary Feature Availability Audit

We next audited whether the existing STM replay artifacts actually expose the
boundary/event-shift features required by the stop-condition matrix. They do
not. Cortext computes these fields during ingress, but the STM benchmark case
tables mostly contain only STM scores and coarse stream counters.

Artifacts:

- `build/short_term_memory_processor_shadow_boundary_feature_availability_audit/stm_boundary_feature_availability_results.json`
- `build/short_term_memory_processor_shadow_boundary_feature_availability_audit/stm_boundary_feature_artifact_audit.csv`
- `build/short_term_memory_processor_shadow_boundary_feature_availability_audit/stm_boundary_feature_runtime_sources.csv`
- `build/short_term_memory_processor_shadow_boundary_feature_availability_audit/stm_boundary_feature_next_export_schema.csv`

Artifact feature availability:

| artifact family | boundary score | drift | coherence | surprisal | topic shift | boundary type |
|---|---:|---:|---:|---:|---:|---:|
| score-normalization cases | no | no | no | no | no | no |
| processor shadow utility cases | no | no | no | no | no | no |
| boundary stream-proxy AUC | no | no | no | no | no | no |
| chronological boundary-order cases | no | no | no | no | no | no |
| general STM utility cases | no | no | no | no | no | no |
| legacy source-continuity trap cases | yes | no | no | no | no | no |

Runtime source availability:

| runtime source | fields available |
|---|---|
| `src/operations/boundary.cpp` | `boundary_score`, `coherence_prev`, `coherence_curr`, `coh_drop`, `drift_spike`, `surprisal_raw`, `surprisal_norm`, `coh_drop_norm`, `drift_norm`, `topic_shift`, `boundary_rate_ema`, `boundary_rate_mult` |
| `src/operations/coherence.cpp` | accumulator coherence, structural coherence, `drift_mag` / `d_step` |
| `src/operations/embedding_prediction_error.cpp` | `embedding_surprisal` |
| `src/operations/memory_storage.cpp` | persisted `boundary_score`, `drift_mag` on memories |
| `src/operations/short_term_memory_shadow.cpp` | STM item `boundary_score` used internally |

This closes the current posthoc threshold path. We cannot honestly test a
boundary-aware STM consumer from the existing CSVs because the required
components are absent. The next valid experiment is a benchmark-only export pass
that adds `boundary_score`, `boundary_type`, `drift_spike`, `drift_norm`,
`coh_drop`, `coh_drop_norm`, `surprisal_raw`, `surprisal_norm`, `topic_shift`,
`boundary_rate_ema`, and `boundary_rate_mult` to the STM case surface. Production
retrieval remains unchanged.

### Boundary Feature Export Pass

We then added the missing benchmark-only boundary diagnostics to the
processor-backed STM shadow path. This changed only the shadow benchmark:
production retrieval remains unchanged. The processor-backed STM experiment now
runs the real ingress coherence, accumulator, embedding-surprisal, and boundary
operations before the shadow STM update, then exports current-step boundary
diagnostics with each STM utility case.

Artifacts:

- `build/short_term_memory_processor_shadow_boundary_feature_export/stm_processor_shadow_utility_cases.csv`
- `build/short_term_memory_processor_shadow_boundary_feature_export/stm_boundary_feature_export_results.json`
- `build/short_term_memory_processor_shadow_boundary_feature_export/stm_boundary_feature_export_auc.csv`
- `build/short_term_memory_processor_shadow_boundary_feature_export/stm_boundary_feature_export_fixed_gate_screen.csv`

Boundary-feature export result:

| metric | value |
|---|---:|
| cases | 480 |
| positives / controls | 240 / 240 |
| delayed positives | 120 |
| target present in STM | 153 / 240 |
| target top-3 | 83 / 240 |
| tempting controls | 192 / 240 |
| utility-vs-control AUC | 0.2587 |
| inverted AUC | 0.7413 |
| mean STM count | 6.11 |

Best all-control rejection AUCs:

| feature | best AUC | direction |
|---|---:|---|
| `current_step` | 0.6501 | higher rejects |
| `current_surprisal_norm` | 0.6147 | higher rejects |
| `current_coh_drop_norm` | 0.5999 | higher rejects |
| `current_surprisal_raw` | 0.5964 | lower rejects |
| `current_drift_norm` | 0.5606 | lower rejects |
| `current_boundary_score` | 0.5003 | higher rejects |

Fixed `raw >= 0.60` plus one boundary/stream feature:

| budget | best feature rule | positives | delayed | false positives |
|---|---|---:|---:|---:|
| zero FP | reject `current_step >= 14` | 21 | 0 | 0 |
| 2% FP | reject `current_step >= 14` | 21 | 0 | 0 |
| 5% FP | reject `current_step >= 17` | 28 | 2 | 12 |

This is a negative consumer result. Exporting the true boundary diagnostics did
not create a safe STM readout. The actual `boundary_score` is effectively chance
for rejecting controls on this surface, and the normalized coherence/surprisal
features are only weakly useful. Running the real boundary stack also changes
the substrate: mean STM size drops to `6.11`, and target presence drops to
`153 / 240`, versus the earlier minimal shadow processor run that retained
`118 / 120` delayed positives with a p95 size near `13`. The old minimal run was
therefore an optimistic substrate probe, not the behavior of the real boundary
policy. The remaining STM question is no longer "can a simple boundary feature
make STM readable"; it is whether STM needs a different retention policy or a
learned/order-aware consumer that uses the larger unsurfaced context without
turning topic shifts and stale same-source items into commitments.

### Boundary Diagnostics Without Boundary Decay

To separate diagnostic availability from retention policy, we added a
benchmark-only switch, `CORTEXT_STM_SHADOW_DISABLE_BOUNDARY_DECAY=1`, that keeps
the real boundary/coherence/surprisal diagnostics but prevents hard boundary
events from truncating the shadow STM buffer to the most recent four items.
Production retrieval remains unchanged.

Artifacts:

- `build/short_term_memory_processor_shadow_boundary_diagnostics_no_decay/stm_boundary_diagnostics_no_decay_results.json`
- `build/short_term_memory_processor_shadow_boundary_diagnostics_no_decay/stm_boundary_decay_vs_no_decay_comparison.json`
- `build/short_term_memory_processor_shadow_boundary_diagnostics_no_decay/stm_boundary_diagnostics_no_decay_auc.csv`
- `build/short_term_memory_processor_shadow_boundary_diagnostics_no_decay/stm_boundary_diagnostics_no_decay_fixed_gate_screen.csv`

Boundary decay versus no-decay:

| setting | target present | target top-3 | tempting controls | utility AUC | case mean STM count |
|---|---:|---:|---:|---:|---:|
| real boundary decay | 153 / 240 | 83 / 240 | 192 / 240 | 0.2587 | 6.11 |
| diagnostics, no boundary decay | 234 / 240 | 66 / 240 | 219 / 240 | 0.2880 | 15.25 |

Fixed `raw >= 0.60` plus one feature on the no-decay surface:

| budget | best feature rule | positives | delayed | false positives |
|---|---|---:|---:|---:|
| zero FP | reject `current_step >= 14` | 21 | 0 | 0 |
| 2% FP | reject `current_step >= 15` | 22 | 1 | 4 |
| 5% FP | reject `current_step >= 16` | 28 | 4 | 9 |

This ablation restores the substrate but not the consumer. Disabling boundary
decay recovers `+81` target-present cases, but it also adds `+27` tempting
controls. The best low-FPR rules still recover almost no delayed positives.
So the issue is now cleanly split: hard boundary decay is too aggressive for STM
as an evidence reservoir, while boundary diagnostics alone are too weak to make
the enlarged reservoir safely readable.

### Calibrated STM Consumer Comparison

The processor-backed STM run also emits a calibrated logistic feature probe over
STM-derived scores, margins, recency/order features, and coarse boundary
pressure. This is still benchmark-only and is not production retrieval. The
first version mislabeled a dataset-held-out fold as `source_held_out`; we fixed
the case source id to `dataset:conversation_id` and reran with 37 true
source/conversation groups.

Artifacts:

- `build/short_term_memory_processor_shadow_true_source_heldout_fixed_comparison/stm_true_source_heldout_fixed_comparison.json`
- `build/short_term_memory_processor_shadow_true_source_heldout_fixed_comparison/stm_true_source_heldout_fixed_comparison.csv`

Calibrated consumer comparison:

| run | split | AUC | zero-FPR recovery | 5% FPR recovery | controls tempted |
|---|---|---:|---:|---:|---:|
| boundary decay | chronological | 0.7386 | 10 / 96 | 26 / 96 | 30 / 96 |
| boundary decay | dataset-held-out | 0.6849 | 16 / 240 | 55 / 240 | 57 / 240 |
| boundary decay | true source-held-out | 0.7630 | 26 / 240 | 72 / 240 | 67 / 240 |
| diagnostics, no boundary decay | chronological | 0.7493 | 20 / 96 | 23 / 96 | 27 / 96 |
| diagnostics, no boundary decay | dataset-held-out | 0.6636 | 34 / 240 | 57 / 240 | 56 / 240 |
| diagnostics, no boundary decay | true source-held-out | 0.7665 | 41 / 240 | 80 / 240 | 69 / 240 |

This is the first STM consumer result that is not trivially zero at zero FPR:
the no-decay substrate plus calibrated features recovers `41 / 240` positives
at zero FPR and `80 / 240` at 5% FPR under true source/conversation-held-out
validation. The result is still not sufficient for promotion. The score is a
small logistic probe over hand-built STM features, and tempting controls remain
high before thresholding. The useful conclusion is narrow: STM is not just a
buffer-size idea. It likely needs an explicit learned consumer over order,
recency, margin, and boundary/context features. Scalar thresholds and
one-feature boundary gates are exhausted.

### Calibrated Consumer Feature Ablation

We then ablated the calibrated no-decay consumer to test whether the positive
source-held-out result was just a chronology shortcut. The benchmark now writes
the calibrated feature matrix and feature-family ablations.

Artifacts:

- `build/short_term_memory_processor_shadow_feature_ablation_no_decay/stm_chronological_calibrated_features.csv`
- `build/short_term_memory_processor_shadow_feature_ablation_no_decay/stm_calibrated_feature_ablation_results.csv`
- `build/short_term_memory_processor_shadow_feature_ablation_no_decay/stm_feature_ablation_no_decay_summary.json`

True source-held-out feature ablation:

| feature set | AUC | zero-FPR recovery | 5% FPR recovery |
|---|---:|---:|---:|
| all features | 0.7665 | 41 / 240 | 80 / 240 |
| without boundary family | 0.7670 | 41 / 240 | 75 / 240 |
| without `current_step` | 0.7600 | 42 / 240 | 82 / 240 |
| recency/order only | 0.7589 | 41 / 240 | 81 / 240 |
| STM scores + order, no step | 0.7577 | 41 / 240 | 77 / 240 |
| score family only | 0.7033 | 15 / 240 | 39 / 240 |
| boundary family only | 0.6884 | 3 / 240 | 31 / 240 |
| `current_step` only | 0.6437 | 47 / 240 | 64 / 240 |

The calibrated consumer is not just a `current_step` shortcut. `current_step`
alone is useful but much weaker by AUC, and removing it preserves the main
source-held-out signal. The strongest compact explanation is order/recency over
the STM surface: recency/order-only and STM score/order without `current_step`
both stay near the full model. Boundary-family features are auxiliary; by
themselves they recover almost nothing at zero FPR. This makes the next valid
STM experiment a proper learned/order-aware consumer with stronger validation,
not more scalar boundary tuning.

### Source-Held-Out Recovery Composition

We then decomposed the no-decay calibrated consumer's true source-held-out
low-FPR recovery by positive and control class.

Artifacts:

- `build/short_term_memory_processor_shadow_feature_ablation_no_decay/stm_source_heldout_recovery_composition.json`
- `build/short_term_memory_processor_shadow_feature_ablation_no_decay/stm_source_heldout_recovery_composition.csv`
- `build/short_term_memory_processor_shadow_feature_ablation_no_decay/stm_source_heldout_source_group_failures.csv`

True source-held-out recovery composition:

| threshold | ordinary recovered | delayed recovered | stale FP | topic-shift FP |
|---|---:|---:|---:|---:|
| zero FPR | 35 / 120 | 6 / 120 | 0 / 120 | 0 / 120 |
| 5% FPR | 54 / 120 | 26 / 120 | 5 / 120 | 8 / 120 |

This narrows the conclusion. The calibrated STM consumer's zero-FPR recovery is
not only easy ordinary cases, but it is dominated by ordinary continuations.
Delayed recovery becomes more meaningful at 5% FPR (`26 / 120`) but is still far
from enough to justify surfacing STM directly. False positives at 5% FPR are
mixed between topic-shift and stale same-source controls, and the failures are
spread across sources rather than concentrated in one bad conversation. The
next learned consumer should therefore optimize delayed retention explicitly,
not just aggregate total low-FPR recovery.

### Delayed-Weighted Consumer Probe

We then tested whether the weak delayed recovery was only a training-objective
imbalance. This benchmark-only probe reused the exported no-decay STM calibrated
feature matrix and upweighted delayed positives during source-held-out logistic
training.

Artifacts:

- `build/short_term_memory_processor_shadow_delayed_weighted_consumer/stm_delayed_weighted_consumer_results.json`
- `build/short_term_memory_processor_shadow_delayed_weighted_consumer/stm_delayed_weighted_consumer_summary.csv`
- `build/short_term_memory_processor_shadow_delayed_weighted_consumer/stm_delayed_weighted_consumer_scores.csv`

True source-held-out delayed-weight sweep:

| delayed weight | AUC | zero ordinary | zero delayed | 5% ordinary | 5% delayed | 5% FP |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.7675 | 35 / 120 | 6 / 120 | 53 / 120 | 25 / 120 | 13 / 240 |
| 1.5 | 0.7668 | 34 / 120 | 6 / 120 | 49 / 120 | 25 / 120 | 13 / 240 |
| 2.0 | 0.7647 | 29 / 120 | 6 / 120 | 45 / 120 | 24 / 120 | 13 / 240 |
| 3.0 | 0.7596 | 26 / 120 | 6 / 120 | 47 / 120 | 24 / 120 | 13 / 240 |
| 4.0 | 0.7547 | 25 / 120 | 7 / 120 | 49 / 120 | 26 / 120 | 13 / 240 |
| 6.0 | 0.7428 | 19 / 120 | 5 / 120 | 38 / 120 | 23 / 120 | 13 / 240 |
| 8.0 | 0.7311 | 11 / 120 | 3 / 120 | 31 / 120 | 22 / 120 | 13 / 240 |

Upweighting delayed positives barely moves delayed recovery. The best 5% FPR
delayed recovery increases from `25 / 120` to `26 / 120`; zero-FPR delayed
recovery only reaches `7 / 120` at weight 4.0 while AUC and total recovery fall.
Delayed weakness is therefore not just class-weight imbalance. A useful STM
consumer needs features or objectives that explicitly model delayed evidence,
not a simple delayed-positive weight.

### Nonlinear STM Consumer Capacity Probe

We then tested whether the source-held-out STM surface is limited by the linear
consumer itself. This benchmark-only probe trained nonlinear consumers over the
same exported no-decay feature matrix. It did not change retrieval behavior or
add STM to production retrieval.

Artifacts:

- `build/short_term_memory_processor_shadow_nonlinear_consumer/stm_nonlinear_consumer_results.json`
- `build/short_term_memory_processor_shadow_nonlinear_consumer/stm_nonlinear_consumer_summary.csv`
- `build/short_term_memory_processor_shadow_nonlinear_consumer/stm_nonlinear_consumer_scores.csv`

True source-held-out nonlinear consumer results:

| consumer | AUC | zero ordinary | zero delayed | 5% ordinary | 5% delayed | 5% FP |
|---|---:|---:|---:|---:|---:|---:|
| linear | 0.7706 | 35 / 120 | 5 / 120 | 53 / 120 | 27 / 120 | 12 / 240 |
| linear, delayed weight 4 | 0.7616 | 27 / 120 | 7 / 120 | 45 / 120 | 23 / 120 | 12 / 240 |
| polynomial interactions | 0.7657 | 34 / 120 | 7 / 120 | 50 / 120 | 27 / 120 | 12 / 240 |
| MLP, 8 hidden | 0.7484 | 38 / 120 | 8 / 120 | 57 / 120 | 26 / 120 | 12 / 240 |
| MLP, 16 hidden | 0.7743 | 40 / 120 | 6 / 120 | 62 / 120 | 29 / 120 | 12 / 240 |

The nonlinear probe improves total low-FPR recovery but does not solve delayed
continuation. The 16-hidden-unit MLP raises 5% FPR recovery to `91 / 240`
(`62 / 120` ordinary and `29 / 120` delayed), but zero-FPR delayed recovery is
still only `6 / 120`. This says consumer capacity matters, but the current
scalar feature surface still does not provide a clean delayed-safe signal. The
next STM consumer should use ordered STM item structure directly, not only
aggregate case-level scalar features.

### STM Rank-Gap Audit

We then audited the processor-backed STM cases to separate "target retained"
from "target directly readable." This uses only benchmark artifacts and does
not expose STM to production retrieval.

Artifacts:

- `build/short_term_memory_processor_shadow_rank_gap_audit/stm_rank_gap_audit_results.json`
- `build/short_term_memory_processor_shadow_rank_gap_audit/stm_rank_gap_by_label.csv`
- `build/short_term_memory_processor_shadow_rank_gap_audit/stm_rank_gap_by_distance.csv`
- `build/short_term_memory_processor_shadow_rank_gap_audit/stm_rank_gap_cases.csv`
- `build/short_term_memory_processor_shadow_rank_gap_audit/stm_score_overlap.csv`

Rank depth by continuation class:

| class | target in STM | top-1 | top-3 | top-8 | top-16 | mean rank | median margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary continuation | 116 / 120 | 22 | 53 | 88 | 116 | 5.59 | -0.1028 |
| delayed continuation | 118 / 120 | 8 | 13 | 50 | 113 | 9.36 | -0.2525 |

Score overlap remains high. At the positive target median score (`0.4935`),
`202 / 240` controls have a best STM score above that level; even at the
positive target 90th percentile (`0.7210`), `58 / 240` controls remain above
threshold. The delayed target is therefore not absent. It is retained but buried
behind recent or tempting STM evidence, with a much worse target-vs-best
non-target margin than ordinary continuations. This confirms that another
case-level scalar threshold is unlikely to solve STM use. The next useful
experiment needs an ordered item-level STM export and a bounded consumer that
can inspect roughly the top 8-16 retained items.

### Ordered Item-Level STM Export

We added a shadow-only processor benchmark export for ordered STM items. This
does not change production retrieval; it writes the retained `STM_before_t`
items, per-item distances, raw/recency/boundary/trace scores, ranks, and
boundary diagnostics for the processor-backed STM cases.

Command:

```bash
CORTEXT_STM_SHADOW_DISABLE_BOUNDARY_DECAY=1 \
./build/examples/benchmark/cortext_anchor_replay_bench \
  --short-term-memory-processor-shadow-experiments \
  --models=models \
  --output-dir build/short_term_memory_processor_shadow_item_level_export
```

Artifacts:

- `build/short_term_memory_processor_shadow_item_level_export/stm_processor_shadow_item_level_cases.csv`
- `build/short_term_memory_processor_shadow_item_level_export/stm_item_level_export_audit.json`
- `build/short_term_memory_processor_shadow_item_level_export/stm_item_level_case_summary.csv`
- `build/short_term_memory_processor_shadow_item_level_export/stm_item_level_label_summary.csv`
- `build/short_term_memory_processor_shadow_item_level_export/stm_item_level_score_auc.csv`

The run wrote `7,706` ordered STM item rows for `480` cases. The item-level
summary:

| class | target present | raw top-1 | raw top-3 | raw top-8 | raw top-16 | recency top-1 | median margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary continuation | 120 / 120 | 22 | 53 | 88 | 118 | 71 | -0.1071 |
| delayed continuation | 120 / 120 | 8 | 13 | 50 | 114 | 0 | -0.2526 |

Positive item-vs-other-item AUCs:

| score | AUC |
|---|---:|
| raw score | 0.5458 |
| recency-weighted score | 0.6796 |
| boundary-soft score | 0.5896 |
| trace score | 0.6541 |
| inverse item distance | 0.7414 |
| item boundary score | 0.5014 |

This is useful as an export, but it is not yet a useful consumer. Recency
features explain immediate ordinary continuations and fail delayed continuation:
recency top-1 is `71 / 120` for ordinary but `0 / 120` for delayed. Raw
similarity keeps delayed targets within the top-16 (`114 / 120`) but only top-3
`13 / 120`. Boundary fields are near chance at item level. The next experiment
should therefore train or test a bounded ordered-item consumer over top 8-16 STM
items with explicit delayed objectives, rather than using recency or scalar
similarity alone.

### Bounded Item-Level Consumer Probe

We then trained a source-held-out item-level logistic probe over the ordered STM
export. The probe scored retained STM items using only runtime-compatible item
features, then compared the target item score on positive cases against the
best item score on controls. This is still benchmark-only and does not change
production retrieval.

Artifacts:

- `build/short_term_memory_processor_shadow_item_level_consumer/stm_item_level_consumer_results.json`
- `build/short_term_memory_processor_shadow_item_level_consumer/stm_item_level_consumer_summary.csv`
- `build/short_term_memory_processor_shadow_item_level_consumer/stm_item_level_consumer_item_scores.csv`

Representative source-held-out results:

| consumer | window | AUC | zero ordinary | zero delayed | 5% ordinary | 5% delayed | delayed top-3 |
|---|---|---:|---:|---:|---:|---:|---:|
| raw score | top-16 | 0.2874 | 0 / 120 | 1 / 120 | 1 / 120 | 4 / 120 | 13 / 120 |
| recency score | top-16 | 0.3089 | 1 / 120 | 0 / 120 | 6 / 120 | 0 / 120 | 8 / 120 |
| trace score | top-16 | 0.3158 | 1 / 120 | 0 / 120 | 7 / 120 | 0 / 120 | 13 / 120 |
| item logistic, scores + rank + distance + boundary | top-16 | 0.3168 | 3 / 120 | 0 / 120 | 13 / 120 | 0 / 120 | 0 / 120 |
| item logistic, scores + rank + distance + boundary | all | 0.3191 | 3 / 120 | 0 / 120 | 14 / 120 | 0 / 120 | 0 / 120 |

This probe is a negative result. The item-level logistic model mostly learns
recency/rank structure and improves immediate ordinary target ranking while
destroying delayed target ranking. It also fails the safe-readout objective:
positive target scores remain below control best-item scores, producing AUCs
well below 0.5 in the target-score-vs-control-best-score contract. The ordered
export is necessary, but scalar item features are not sufficient. A useful STM
consumer likely needs sequence-aware attention over the retained item list and
an objective that explicitly separates delayed evidence from tempting recent
controls.

### Sequence-Feature STM Consumer Probe

We then tested a compact case-level sequence consumer over ordered top-k STM
item summaries. Unlike the itemwise probe, this model sees list-level score
shape: top scores, gaps, entropy, score statistics, item distances, and boundary
diagnostic summaries. It still uses only runtime-compatible shadow features and
does not change production retrieval.

Artifacts:

- `build/short_term_memory_processor_shadow_sequence_consumer/stm_sequence_consumer_results.json`
- `build/short_term_memory_processor_shadow_sequence_consumer/stm_sequence_consumer_summary.csv`
- `build/short_term_memory_processor_shadow_sequence_consumer/stm_sequence_consumer_scores.csv`

True source-held-out results:

| consumer | window | AUC | zero ordinary | zero delayed | 5% ordinary | 5% delayed | 5% FP |
|---|---|---:|---:|---:|---:|---:|---:|
| logistic sequence | top-8 | 0.7511 | 12 / 120 | 1 / 120 | 46 / 120 | 27 / 120 | 12 / 240 |
| logistic sequence | top-16 | 0.7691 | 33 / 120 | 3 / 120 | 50 / 120 | 20 / 120 | 12 / 240 |
| MLP-10 sequence | top-8 | 0.7555 | 12 / 120 | 2 / 120 | 43 / 120 | 26 / 120 | 12 / 240 |
| MLP-10 sequence | top-16 | 0.7691 | 35 / 120 | 4 / 120 | 52 / 120 | 24 / 120 | 12 / 240 |

This is better than itemwise scalar ranking but still not enough. Sequence
summaries recover a sane source-held-out AUC (`~0.769`) and nonzero zero-FPR
recovery, but the recovery remains ordinary-heavy. The best top-16 sequence
probe gets only `4 / 120` delayed cases at zero FPR and `24 / 120` delayed
cases at 5% FPR, below the earlier scalar MLP's `29 / 120` delayed 5% recovery.
The result says ordered list shape matters, but handcrafted sequence summaries
do not extract the delayed-safe signal. The next STM consumer needs either
attention over item embeddings/micro-vectors or explicit delayed-event
supervision, not just summary statistics over item scores.

### Distance-Compensated STM Item Score Sweep

We then tested whether delayed targets are buried only because the item score
overweights recency. The diagnostic swept fixed formulas of the form
`base_score + lambda * log1p(item_distance)` over top-8 and top-16 STM items.
Positive `lambda` rewards older retained evidence; negative `lambda` favors
recency. The sweep is benchmark-only and does not change production retrieval.

Artifacts:

- `build/short_term_memory_processor_shadow_distance_compensation/stm_distance_compensation_results.json`
- `build/short_term_memory_processor_shadow_distance_compensation/stm_distance_compensation_summary.csv`
- `build/short_term_memory_processor_shadow_distance_compensation/stm_distance_compensation_cases.csv`

Best representative rows:

| base score | lambda | window | AUC | zero ordinary | zero delayed | 5% ordinary | 5% delayed | 5% FP |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| boundary-soft | -0.10 | top-16 | 0.3339 | 1 / 120 | 1 / 120 | 4 / 120 | 4 / 120 | 12 / 240 |
| boundary-soft | 0.00 | top-16 | 0.3335 | 0 / 120 | 1 / 120 | 2 / 120 | 4 / 120 | 12 / 240 |
| boundary-soft | 0.10 | top-16 | 0.3283 | 0 / 120 | 1 / 120 | 1 / 120 | 4 / 120 | 12 / 240 |
| raw score | -0.20 | top-16 | 0.3249 | 1 / 120 | 1 / 120 | 4 / 120 | 3 / 120 | 12 / 240 |
| raw score | 0.00 | top-16 | 0.2874 | 0 / 120 | 1 / 120 | 1 / 120 | 4 / 120 | 12 / 240 |

Distance compensation does not expose a delayed-safe signal. Rewarding older
items slightly improves delayed top-3 in some rows, but low-FPR delayed
recovery stays at `4 / 120` or worse and AUC remains far below 0.5 in the
target-vs-control-best contract. Penalizing older items helps ordinary recency
but suppresses delayed evidence. The failure is therefore not a simple recency
weighting error; older retained evidence and tempting controls overlap too much
under scalar item scores.

### Sequence Consumer Breakdown

We then decomposed the compact sequence consumer by dataset and control class to
check whether the failure was another score-scale transfer problem.

Artifacts:

- `build/short_term_memory_processor_shadow_sequence_breakdown/stm_sequence_breakdown_results.json`
- `build/short_term_memory_processor_shadow_sequence_breakdown/stm_sequence_global_breakdown.csv`
- `build/short_term_memory_processor_shadow_sequence_breakdown/stm_sequence_dataset_local_thresholds.csv`
- `build/short_term_memory_processor_shadow_sequence_breakdown/stm_sequence_control_auc.csv`

For the best compact sequence probe (`MLP-10`, top-16), global 5% FPR recovery
was:

| scope | AUC | recovery | ordinary | delayed | false positives | stale FP | topic FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 0.7691 | 76 / 240 | 52 / 120 | 24 / 120 | 12 / 240 | 3 | 9 |
| Taskmaster | 0.7721 | 56 / 168 | 37 / 84 | 19 / 84 | 9 / 168 | 3 | 6 |
| TopicalChat | 0.7834 | 20 / 72 | 15 / 36 | 5 / 36 | 3 / 72 | 0 | 3 |

Dataset-local 5% thresholds barely changed the picture: TopicalChat moved from
`20 / 72` to `21 / 72` recovery. This is different from the earlier scalar STM
score surface, where local calibration exposed much more TopicalChat signal.
For sequence summaries, the main failure is not dataset scale. Control AUCs show
the harder class: stale same-source separates better (`0.7977`) than topic-shift
controls (`0.7405`). The next STM consumer should therefore focus on
topic-shift/discourse-change evidence and delayed target discrimination rather
than another dataset-local threshold scheme.

### Topic-Shift Rejector Probe

We then tested whether the dominant topic-shift false positives could be removed
with a small second-stage rejector over runtime-compatible summary fields. The
rejector used only sequence score, STM item count, best item score, top-item
distance/boundary, and near-top density. It did not use target ranks or labels
as runtime features.

Artifacts:

- `build/short_term_memory_processor_shadow_topic_rejector/stm_topic_rejector_results.json`
- `build/short_term_memory_processor_shadow_topic_rejector/stm_topic_rejector_summary.csv`
- `build/short_term_memory_processor_shadow_topic_rejector/stm_topic_rejector_control_auc.csv`
- `build/short_term_memory_processor_shadow_topic_rejector/stm_topic_rejector_scores.csv`

Source-held-out results:

| model | AUC | zero ordinary | zero delayed | 5% ordinary | 5% delayed | stale FP | topic FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| base MLP-10 sequence | 0.7691 | 35 / 120 | 4 / 120 | 52 / 120 | 24 / 120 | 3 | 9 |
| runtime-feature rejector | 0.7678 | 37 / 120 | 2 / 120 | 51 / 120 | 24 / 120 | 4 | 8 |
| topic-weighted rejector | 0.7669 | 29 / 120 | 2 / 120 | 48 / 120 | 25 / 120 | 4 | 8 |
| interaction rejector | 0.7676 | 35 / 120 | 0 / 120 | 51 / 120 | 22 / 120 | 4 | 8 |

The rejector does not materially improve STM safety. It can remove one
topic-shift false positive at 5% FPR, but this comes with worse delayed
zero-FPR recovery, worse stale false positives, or lower ordinary recovery.
Control AUC barely moves: topic-shift AUC rises only from `0.7405` to `0.7444`
in the topic-weighted variant. The remaining topic-shift failures need richer
discourse-change evidence or item-level embeddings, not another small rejector
over the current summary fields.

### Oracle Headroom for Sequence STM

We then used labels offline to measure upper bounds for the current sequence
score. This is not a runtime method. It asks whether perfect rejection of a
control family would be enough, or whether delayed positives are intrinsically
below the current score thresholds.

Artifacts:

- `build/short_term_memory_processor_shadow_oracle_headroom/stm_oracle_headroom_results.json`
- `build/short_term_memory_processor_shadow_oracle_headroom/stm_oracle_headroom_budget.csv`
- `build/short_term_memory_processor_shadow_oracle_headroom/stm_oracle_headroom_threshold_transfer.csv`
- `build/short_term_memory_processor_shadow_oracle_headroom/stm_oracle_headroom_score_distributions.csv`

Oracle budget results for the `MLP-10` top-16 sequence score:

| scenario | AUC | zero recovery | zero delayed | 5% recovery | 5% delayed | 5% FP |
|---|---:|---:|---:|---:|---:|---:|
| all controls | 0.7691 | 39 / 240 | 4 / 120 | 76 / 240 | 24 / 120 | 12 / 240 |
| remove topic controls | 0.7977 | 41 / 240 | 5 / 120 | 81 / 240 | 26 / 120 | 6 / 120 |
| remove stale controls | 0.7405 | 39 / 240 | 4 / 120 | 69 / 240 | 21 / 120 | 6 / 120 |
| delayed vs controls only | 0.7342 | 4 / 120 | 4 / 120 | 24 / 120 | 24 / 120 | 12 / 240 |

The score distributions explain the ceiling:

| class | mean | p50 | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|
| ordinary continuation | 0.6380 | 0.6913 | 0.8756 | 0.8953 | 0.9065 |
| delayed continuation | 0.5438 | 0.5714 | 0.7047 | 0.8044 | 0.8828 |
| stale same-source control | 0.3470 | 0.3088 | 0.4326 | 0.5492 | 0.8503 |
| topic-shift control | 0.3991 | 0.3467 | 0.5238 | 0.6838 | 0.8639 |

Perfect topic-shift rejection would improve precision and raise 5% recovery
from `76 / 240` to `81 / 240`, but it would only lift delayed recovery from
`24 / 120` to `26 / 120`. Even removing all controls while keeping the baseline
thresholds leaves delayed recovery unchanged because most delayed positives are
below the score cutoffs. The bottleneck is therefore not only false-positive
filtering. The sequence score itself under-scores delayed continuations.

### Soft-Attention STM Summary Probe

We then tested a small attention-shaped pooling diagnostic over ordered STM item
features. The full parameter grid was too slow for the notification loop, so we
kept a fixed, small set of attention functions: uniform, raw-score attention,
trace attention, old-item emphasis, recent-item emphasis, and boundary-aware
attention over top-8 and top-16 STM items. This is still a hand-built
summary over scalar item features, not an embedding-attention model, and it does
not change production retrieval.

Artifacts:

- `build/short_term_memory_processor_shadow_soft_attention_consumer/stm_soft_attention_consumer_results.json`
- `build/short_term_memory_processor_shadow_soft_attention_consumer/stm_soft_attention_consumer_summary.csv`
- `build/short_term_memory_processor_shadow_soft_attention_consumer/stm_soft_attention_consumer_scores.csv`

Representative source-held-out results:

| attention | window | AUC | zero ordinary | zero delayed | 5% ordinary | 5% delayed | stale FP | topic FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | top-16 | 0.7693 | 38 / 120 | 2 / 120 | 55 / 120 | 22 / 120 | 2 | 10 |
| boundary-aware | top-16 | 0.7647 | 39 / 120 | 5 / 120 | 52 / 120 | 28 / 120 | 4 | 8 |
| recent-item | top-8 | 0.7689 | 25 / 120 | 3 / 120 | 49 / 120 | 34 / 120 | 5 | 7 |
| trace | top-8 | 0.7629 | 30 / 120 | 5 / 120 | 50 / 120 | 36 / 120 | 4 | 8 |
| old-item | top-8 | 0.7571 | 31 / 120 | 4 / 120 | 52 / 120 | 34 / 120 | 3 | 9 |

This is the first sign that attention-shaped STM aggregation can lift delayed
5% FPR recovery: top-8 trace attention reaches `36 / 120` delayed recovery,
above the compact sequence probe (`24 / 120`) and the earlier scalar MLP
(`29 / 120`). But it does not solve safe use. Zero-FPR delayed recovery is still
only `5 / 120`, and topic-shift false positives remain `7-10 / 120` at the 5%
budget. The result supports the next experiment direction, not promotion:
attention over ordered STM items is worth testing with real item embeddings or
micro-vectors, but scalar attention summaries are not enough.

### Soft-Attention FP Frontier

We then recomputed the small soft-attention variants with full per-case scores
and measured recovery at 0%, 2%, 5%, and 10% false-positive budgets.

Artifacts:

- `build/short_term_memory_processor_shadow_soft_attention_frontier/stm_soft_attention_frontier_results.json`
- `build/short_term_memory_processor_shadow_soft_attention_frontier/stm_soft_attention_frontier_summary.csv`
- `build/short_term_memory_processor_shadow_soft_attention_frontier/stm_soft_attention_frontier.csv`
- `build/short_term_memory_processor_shadow_soft_attention_frontier/stm_soft_attention_frontier_scores.csv`

Best delayed-recovery row at each false-positive budget:

| FP budget | attention | window | AUC | recovery | ordinary | delayed | FP mix |
|---:|---|---|---:|---:|---:|---:|---|
| 0% | boundary-aware | top-16 | 0.7644 | 44 / 240 | 39 / 120 | 5 / 120 | 0 stale, 0 topic |
| 2% | trace | top-16 | 0.7653 | 57 / 240 | 44 / 120 | 13 / 120 | 0 stale, 5 topic |
| 5% | trace | top-8 | 0.7628 | 86 / 240 | 50 / 120 | 36 / 120 | 4 stale, 8 topic |
| 10% | recent-item | top-8 | 0.7685 | 112 / 240 | 66 / 120 | 46 / 120 | 7 stale, 17 topic |

The frontier makes the tradeoff explicit. Soft attention improves delayed
recovery only once the false-positive budget is relaxed. At 5% FPR it is a real
improvement over the previous sequence summaries, but at 2% FPR delayed recovery
falls to `13 / 120`, and at zero FPR it is still `5 / 120`. This keeps STM in
the "useful substrate, not safe consumer" category. The next high-value
experiment is not more scalar summary tuning; it is exporting compact item
embeddings or micro-vectors for an actual attention consumer.

### Compact Item-Projection STM Attention

We added a benchmark-only compact projection export for STM items. During the
processor-backed shadow run, each current signal and retained STM item embedding
is projected into a deterministic 32-dimensional signed-bucket vector and
L2-normalized. This keeps the artifact small while exposing real vector
interaction signal instead of only scalar scores. Production retrieval remains
unchanged.

Command:

```bash
CORTEXT_STM_SHADOW_DISABLE_BOUNDARY_DECAY=1 \
./build/examples/benchmark/cortext_anchor_replay_bench \
  --short-term-memory-processor-shadow-experiments \
  --models=models \
  --output-dir build/short_term_memory_processor_shadow_projection_export
```

Artifacts:

- `build/short_term_memory_processor_shadow_projection_export/stm_processor_shadow_item_projection_cases.csv`
- `build/short_term_memory_processor_shadow_projection_export/stm_processor_shadow_item_level_cases.csv`
- `build/short_term_memory_processor_shadow_projection_export/stm_processor_shadow_summary.csv`

The projection export wrote `7,348` top-16 STM item rows. We then trained a
source-held-out logistic consumer over attended projected item vectors
(`attended`, `current - attended`, and `current * attended`) using several fixed
attention rules.

Consumer artifacts:

- `build/short_term_memory_processor_shadow_projection_attention_consumer/stm_projection_attention_results.json`
- `build/short_term_memory_processor_shadow_projection_attention_consumer/stm_projection_attention_summary.csv`
- `build/short_term_memory_processor_shadow_projection_attention_frontier.csv`
- `build/short_term_memory_processor_shadow_projection_attention_scores.csv`

Projection-attention results:

| attention | AUC | zero ordinary | zero delayed | 2% delayed | 5% ordinary | 5% delayed | stale FP | topic FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 0.8167 | 30 / 120 | 2 / 120 | -- | 55 / 120 | 25 / 120 | 3 | 9 |
| raw-score attention | 0.8094 | 26 / 120 | 6 / 120 | 21 / 120 | 54 / 120 | 36 / 120 | 4 | 8 |
| projection-dot attention | 0.8044 | 29 / 120 | 5 / 120 | -- | 52 / 120 | 28 / 120 | 4 | 8 |
| recent projection | 0.8043 | 29 / 120 | 6 / 120 | -- | 51 / 120 | 29 / 120 | 5 | 7 |

This is the strongest STM consumer signal so far by AUC: compact projection
attention reaches `0.8167`, above scalar and sequence-summary probes. It also
improves the 2% delayed frontier: raw-score projection attention recovers
`21 / 120` delayed cases at 2% FPR, versus `13 / 120` for scalar soft attention.
However, it still does not solve safe use. Zero-FPR delayed recovery is only
`6 / 120`, and 5% FPR delayed recovery (`36 / 120`) matches the best scalar
soft-attention row rather than exceeding it. Compact item vectors help, but the
current projection is not enough for promotion. The next useful test is either
full-dimension item embedding attention, micro-vector evidence, or a learned
sequence objective trained specifically to raise delayed positives without
admitting topic-shift controls.

### Projection Attention Oracle Breakdown

We then ran a label-aware oracle breakdown over the compact projection-attention
scores. This is not a production policy: it uses control labels only to measure
which false-positive family is limiting the shadow consumer and how much
recovery headroom exists if that family could be rejected by a future runtime
signal.

Artifacts:

- `build/short_term_memory_processor_shadow_projection_oracle/stm_projection_oracle_results.json`
- `build/short_term_memory_processor_shadow_projection_oracle/stm_projection_oracle_budget.csv`
- `build/short_term_memory_processor_shadow_projection_oracle/stm_projection_oracle_controls.csv`
- `build/short_term_memory_processor_shadow_projection_oracle/stm_projection_oracle_control_auc.csv`
- `build/short_term_memory_processor_shadow_projection_oracle/stm_projection_oracle_score_distributions.csv`

Control separation by family:

| attention | stale AUC | topic-shift AUC |
|---|---:|---:|
| uniform | 0.8440 | 0.7894 |
| raw-score projection | 0.8376 | 0.7811 |

Raw-score projection oracle budget:

| scenario | zero recovery | zero delayed | 5% recovery | 5% delayed | 5% FP mix |
|---|---:|---:|---:|---:|---|
| all controls | 32 / 240 | 6 / 120 | 90 / 240 | 36 / 120 | 4 stale, 8 topic |
| remove topic controls | 32 / 240 | 6 / 120 | 127 / 240 | 56 / 120 | 6 stale, 0 topic |
| remove stale controls | 50 / 240 | 13 / 120 | 82 / 240 | 33 / 120 | 0 stale, 6 topic |

Raw-score projection score distribution:

| class | mean | p50 | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|
| ordinary continuation | 0.6959 | 0.7530 | 0.9310 | 0.9928 | 0.9998 |
| delayed continuation | 0.6105 | 0.6623 | 0.8263 | 0.9114 | 0.9915 |
| stale same-source control | 0.3048 | 0.2654 | 0.4366 | 0.6087 | 0.9499 |
| topic-shift control | 0.3715 | 0.3320 | 0.5704 | 0.7218 | 0.8969 |

The projection vectors move STM from a weak scalar substrate to a stronger
shadow consumer surface, but the low-FPR blocker is still the high tail of
control scores. Topic-shift controls are harder on average than stale
same-source controls, while stale controls set the absolute zero-FPR threshold
for the raw-score attention run. Perfect topic rejection would expose much more
5% delayed headroom (`56 / 120`), but it is only an oracle result. Under all
controls, zero-FPR delayed recovery remains `6 / 120`, so compact projection
attention remains evidence that STM is useful, not evidence that it is ready to
change production retrieval.

### Projection Attention With Discourse Features

The oracle breakdown suggested that the projection consumer needs runtime
features that can explain discourse shift instead of just more item similarity.
We therefore ran a shadow-only source-held-out probe over the same compact
projection export, adding aggregate STM item diagnostics such as boundary
scores, coherence drop, drift, surprisal, topic pressure, recency-weighted
scores, and item-distance statistics. This probe is a separate Python logistic
consumer, so its absolute AUC should be compared within the table below rather
than mixed with the earlier C++ projection-attention consumer.

Artifacts:

- `build/short_term_memory_processor_shadow_projection_discourse_consumer/stm_projection_discourse_results.json`
- `build/short_term_memory_processor_shadow_projection_discourse_consumer/stm_projection_discourse_summary.csv`
- `build/short_term_memory_processor_shadow_projection_discourse_consumer/stm_projection_discourse_scores.csv`

Source-held-out discourse-feature probe:

| mode | AUC | stale AUC | topic AUC | zero recovery | zero delayed | 2% delayed | 5% recovery | 5% delayed | 5% FP mix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| projection only | 0.7828 | 0.8174 | 0.7483 | 20 / 240 | 3 / 120 | 13 / 120 | 74 / 240 | 25 / 120 | 3 stale, 9 topic |
| scalar discourse only | 0.7587 | 0.7976 | 0.7198 | 39 / 240 | 3 / 120 | 15 / 120 | 82 / 240 | 27 / 120 | 2 stale, 10 topic |
| projection + boundary | 0.7880 | 0.8368 | 0.7391 | 31 / 240 | 7 / 120 | 8 / 120 | 80 / 240 | 28 / 120 | 2 stale, 10 topic |
| projection + discourse | 0.7941 | 0.8374 | 0.7508 | 32 / 240 | 8 / 120 | 18 / 120 | 90 / 240 | 31 / 120 | 2 stale, 10 topic |

The result is directionally useful but not sufficient. Adding discourse
aggregates improves zero-FPR delayed recovery from `3 / 120` to `8 / 120` and
5% delayed recovery from `25 / 120` to `31 / 120` within this consumer. It also
improves stale separation. However, topic-shift separation barely moves
(`0.7483` to `0.7508` AUC), and the 5% false-positive budget is still mostly
topic-shift controls. This supports STM as a worthwhile substrate with
boundary/discourse features, but it also says that a production-quality
short-term memory consumer needs a stronger topic-shift or discourse-transition
signal before the complexity is justified.

### Two-Head Topic/Stale Rejector Probe

We next tested whether the topic-shift high tail could be reduced by splitting
the shadow consumer into separate heads: a continuation head, a topic-shift
rejector, and a stale-control rejector. For each source-held-out fold, the
rejector penalties were selected on the training sources only, then evaluated on
the held-out source. This remains a benchmark-only probe over the compact STM
projection/discourse features.

Artifacts:

- `build/short_term_memory_processor_shadow_two_head_rejector/stm_two_head_rejector_results.json`
- `build/short_term_memory_processor_shadow_two_head_rejector/stm_two_head_rejector_summary.csv`
- `build/short_term_memory_processor_shadow_two_head_rejector/stm_two_head_rejector_scores.csv`

Two-head rejector results:

| score family | AUC | stale AUC | topic AUC | zero delayed | 2% delayed | 5% delayed | 5% FP mix |
|---|---:|---:|---:|---:|---:|---:|---|
| continuation head | 0.7983 | 0.8413 | 0.7552 | 8 / 120 | 18 / 120 | 35 / 120 | 2 stale, 10 topic |
| two-head rejector | 0.7885 | 0.8264 | 0.7507 | 7 / 120 | 19 / 120 | 36 / 120 | 4 stale, 8 topic |

The separate rejectors do not provide the missing STM consumer. The combined
score slightly improves delayed recovery at 2% and 5% FPR and reduces topic
false positives at the 5% budget from `10` to `8`, but it pays for that by
lowering AUC and admitting more stale controls. This is a limited tradeoff, not
a qualitative improvement. The failure mode is not just that the logistic
consumer needs an extra topic head; the runtime feature surface still lacks a
clean discourse-transition signal.

### Projection Threshold Transfer Diagnostic

The next safety question was whether the stronger projection/discourse
consumers preserve false-positive budgets when thresholds are transferred
instead of selected on the same evaluation surface. We trained thresholds on
control cases from one split and applied them to held-out cases without
retuning. This is shadow-only and uses the existing projection, discourse, and
two-head score artifacts.

Artifacts:

- `build/short_term_memory_processor_shadow_projection_threshold_transfer/stm_projection_threshold_transfer_results.json`
- `build/short_term_memory_processor_shadow_projection_threshold_transfer/stm_projection_threshold_transfer_summary.csv`

5% threshold-transfer results:

| score family | transfer split | recovery | delayed | false positives | FP rate | FP mix |
|---|---|---:|---:|---:|---:|---|
| projection attention raw score | Taskmaster -> TopicalChat | 25 / 72 | 10 | 5 / 72 | 0.069 | 2 stale, 3 topic |
| projection attention raw score | TopicalChat -> Taskmaster | 61 / 168 | 24 | 7 / 168 | 0.042 | 1 stale, 6 topic |
| projection attention raw score | chronological 60/40 | 37 / 96 | 14 | 8 / 96 | 0.083 | 2 stale, 6 topic |
| projection + discourse | Taskmaster -> TopicalChat | 28 / 72 | 8 | 5 / 72 | 0.069 | 2 stale, 3 topic |
| projection + discourse | TopicalChat -> Taskmaster | 56 / 168 | 20 | 7 / 168 | 0.042 | 0 stale, 7 topic |
| projection + discourse | chronological 60/40 | 36 / 96 | 11 | 8 / 96 | 0.083 | 2 stale, 6 topic |
| two-head rejector | Taskmaster -> TopicalChat | 28 / 72 | 10 | 5 / 72 | 0.069 | 2 stale, 3 topic |
| two-head rejector | TopicalChat -> Taskmaster | 44 / 168 | 15 | 3 / 168 | 0.018 | 0 stale, 3 topic |
| two-head rejector | chronological 60/40 | 37 / 96 | 13 | 10 / 96 | 0.104 | 2 stale, 8 topic |

Source leave-one-out transfer is less alarming because each fold still trains on
both datasets: at the 5% budget, projection attention raw-score recovered
`90 / 240` with `13 / 240` false positives, and projection + discourse
recovered `86 / 240` with `12 / 240` false positives. The harder result is
dataset and chronological transfer. A nominal 5% threshold becomes `6.9%` on
Taskmaster-to-TopicalChat, `8.3%` on chronological projection/discourse, and
`10.4%` for the two-head chronological split. This confirms that the current STM
consumers are not just missing a better frontier; their thresholds are not
portable enough for production use. Any future STM consumer must report
threshold transfer explicitly before its within-surface recovery numbers are
treated as actionable.

### Projection Threshold Guardband

We then asked whether a conservative train-side false-positive budget can make
the projection/discourse consumers portable. For each score family, we selected
the largest train-side FP budget whose maximum held-out FP rate across
Taskmaster -> TopicalChat, TopicalChat -> Taskmaster, and chronological 60/40
stayed below a target transfer FP rate. This is a guardband diagnostic, not a
runtime policy.

Artifacts:

- `build/short_term_memory_processor_shadow_projection_threshold_guardband/stm_projection_threshold_guardband_results.json`
- `build/short_term_memory_processor_shadow_projection_threshold_guardband/stm_projection_threshold_guardband_summary.csv`

Guardband needed to stay under 5% transfer FP:

| score family | selected train FP budget | max held-out FP | aggregate recovery | aggregate delayed | aggregate FP mix |
|---|---:|---:|---:|---:|---|
| projection attention raw score | 2% | 0.031 | 68 | 19 | 4 stale, 1 topic |
| projection attention uniform | 2% | 0.042 | 73 | 16 | 4 stale, 5 topic |
| projection + discourse | 2% | 0.036 | 92 | 31 | 2 stale, 7 topic |
| two-head continuation | 2% | 0.030 | 87 | 28 | 2 stale, 6 topic |
| two-head rejector | 0% | 0.021 | 52 | 13 | 0 stale, 3 topic |

The useful point is the cost of portability. Projection + discourse can stay
under 5% transfer FP when trained at a 2% guardband and still recovers `92`
positives across the three transfer splits, including `31` delayed cases. But
that is substantially less than its within-surface 5% frontier, and the
two-head rejector must fall all the way to a zero-FP train threshold to satisfy
the same transfer target. This strengthens the non-promotion conclusion:
current STM consumers contain real signal, but practical use requires either a
portable calibration mechanism or a stronger discourse-transition feature.

### Online Score-Calibration Probe

We then tested whether the threshold-transfer problem is mostly score-scale
drift. Using the existing projection/discourse score files, we compared raw
scores against offline percentiles and online prior-only percentiles. The
offline percentile variants are diagnostic only because they use the whole
source/global score distribution. The online variants use only earlier scores in
source order, with no labels.

Artifacts:

- `build/short_term_memory_processor_shadow_projection_online_calibration/stm_projection_online_calibration_results.json`
- `build/short_term_memory_processor_shadow_projection_online_calibration/stm_projection_online_calibration_summary.csv`
- `build/short_term_memory_processor_shadow_projection_online_calibration/stm_projection_online_calibration_guardband.csv`
- `build/short_term_memory_processor_shadow_projection_online_calibration/stm_projection_online_calibration_scores.csv`

5% transfer guardband after calibration:

| score family | calibration | train FP budget | max held-out FP | aggregate recovery | aggregate delayed | aggregate FP mix |
|---|---|---:|---:|---:|---:|---|
| projection attention raw score | raw | 2% | 0.031 | 68 | 19 | 4 stale, 1 topic |
| projection attention raw score | online global percentile | 2% | 0.028 | 55 | 12 | 5 stale, 1 topic |
| projection attention raw score | source percentile, offline | 0% | 0.014 | 38 | 10 | 2 stale, 0 topic |
| projection + discourse | raw | 2% | 0.036 | 92 | 31 | 2 stale, 7 topic |
| projection + discourse | online global percentile | 2% | 0.048 | 89 | 27 | 3 stale, 7 topic |
| projection + discourse | source percentile, offline | 0% | 0.028 | 38 | 9 | 2 stale, 2 topic |
| two-head continuation | raw | 2% | 0.030 | 87 | 28 | 2 stale, 6 topic |
| two-head continuation | online global percentile | 2% | 0.042 | 79 | 24 | 3 stale, 6 topic |
| two-head continuation | source percentile, offline | 2% | 0.028 | 68 | 18 | 2 stale, 2 topic |

This is a negative calibration result. Online global percentile calibration
slightly stabilizes some false-positive rates, but it costs more delayed
recovery than it saves. Online source percentile calibration is unusable here:
the per-source case history is too sparse and collapses recovery to zero under
the transfer guardband. Offline source percentiles reduce false positives, but
they also discard most recovery and are not a valid runtime mechanism. The raw
projection + discourse score remains the best guardbanded tradeoff, so the STM
problem is not just score-scale drift.

### STM Cost-Benefit Frontier

We then summarized the existing processor-backed TTL/cap sweeps as a
cost-benefit frontier. This is not a new consumer; it asks whether STM's
streaming buffer cost grows in proportion to the recovery signal already
measured by the calibrated shadow consumer.

Artifacts:

- `build/short_term_memory_processor_shadow_cost_benefit/stm_cost_benefit_results.json`
- `build/short_term_memory_processor_shadow_cost_benefit/stm_cost_benefit_summary.csv`

TTL/cap cost-benefit summary:

| TTL / cap | mean STM size | p95 STM size | mean update us | p95 update us | source 0% recovery | source 5% recovery | chronological 5% recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 / 8 | 4.49 | 5 | 3.82 | 7.58 | 24 | 55 | 26 |
| 8 / 16 | 7.17 | 9 | 4.13 | 8.17 | 34 | 61 | 24 |
| 8 / 32 | 7.17 | 9 | 4.13 | 8.42 | 34 | 61 | 24 |
| 12 / 24 | 9.03 | 13 | 4.58 | 9.75 | 31 | 63 | 22 |
| 16 / 32 | 10.10 | 17 | 5.53 | 11.58 | 34 | 57 | 23 |
| 24 / 32 | 10.62 | 21 | 5.02 | 10.62 | 27 | 53 | 26 |
| 32 / 32 | 10.64 | 21 | 5.97 | 12.96 | 26 | 52 | 22 |

The engineering cost of the shadow STM update is small in these runs:
single-digit microseconds on average and p95 below `13 us` even at TTL32/cap32.
The value frontier, however, saturates quickly. TTL8/cap16 recovers most of the
available source-held-out signal (`61` at 5% FPR and `34` at zero FPR), while
TTL12/cap24 adds only two 5% recoveries and loses three zero-FPR recoveries.
Longer TTLs increase retained size and latency without improving source-held-out
recovery. If STM is added, the current evidence favors a small bounded buffer
around TTL8-TTL12 rather than a broad hidden memory layer. The blocker is not
update cost; it is consumer safety and portable calibration.

### STM Promotion Gate

Finally, we collapsed the STM shadow evidence into an explicit promotion gate.
The gate separates two decisions: whether STM looks useful as a bounded
internal substrate, and whether any current STM consumer is safe enough to use.

Artifacts:

- `build/short_term_memory_processor_shadow_promotion_gate/stm_promotion_gate_results.json`
- `build/short_term_memory_processor_shadow_promotion_gate/stm_promotion_gate_criteria.csv`

Promotion criteria:

| criterion | result | evidence |
|---|---|---|
| bounded shadow update latency | pass | small-buffer p95 update <= 9.75 us |
| bounded STM size | pass | small-buffer p95 size <= 13 items |
| measurable recovery value | pass | TTL12/cap24 source 5% recovery 63, zero-FPR recovery 31 |
| transfer FP budget guarded | pass | projection + discourse max held-out FP 0.036 at a 2% train guardband |
| portable calibration improves raw | fail | online global percentile delayed 27 vs raw 31; max FP 0.048 vs 0.036 |
| control tail is low | fail | projection + discourse guardband FP mix: 2 stale, 7 topic |
| delayed transfer recovery sufficient | fail | projection + discourse guardband delayed recovery 31 |

The decision is split. STM has enough value to continue designing a small
bounded substrate: latency and size are acceptable, and the shadow buffer
contains measurable recovery signal. No current STM consumer should be promoted.
The remaining failures are not cost failures; they are safety failures:
calibration does not beat the raw score, topic-shift controls remain in the
high-score tail, and delayed transfer recovery is still too low once the
consumer is guardbanded. The next design work should specify the STM substrate
independently from retrieval behavior, then keep consumers shadow-only until the
portable calibration and topic-transition gates pass.

### STM Substrate Stability Audit

After the promotion gate, we audited the substrate itself across datasets,
conversations, and TTL/cap settings. This asks whether the proposed small STM
buffer is stable enough to design around, independent of whether a retrieval
consumer is ready.

Artifacts:

- `build/short_term_memory_processor_shadow_substrate_stability/stm_substrate_stability_results.json`
- `build/short_term_memory_processor_shadow_substrate_stability/stm_substrate_stability_summary.csv`
- `build/short_term_memory_processor_shadow_substrate_stability/stm_substrate_stability_by_dataset.csv`
- `build/short_term_memory_processor_shadow_substrate_stability/stm_substrate_stability_worst_conversations.csv`
- `build/short_term_memory_processor_shadow_substrate_stability/stm_substrate_reachability_by_dataset.csv`

Substrate stability:

| TTL / cap | mean size | p95 size | max size | p95 update us | max update us | dataset size spread |
|---|---:|---:|---:|---:|---:|---:|
| 4 / 8 | 4.49 | 5 | 5 | 7.58 | 24.62 | 0.19 |
| 8 / 16 | 7.17 | 9 | 9 | 8.17 | 25.92 | 0.69 |
| 12 / 24 | 9.03 | 13 | 13 | 9.75 | 59.29 | 1.49 |
| 32 / 32 | 10.64 | 21 | 30 | 12.96 | 81.79 | 3.39 |

Delayed-target retention by dataset:

| TTL / cap | dataset | delayed target in STM | delayed target top-3 |
|---|---|---:|---:|
| 8 / 16 | Taskmaster | 0.631 | 0.167 |
| 8 / 16 | TopicalChat | 0.611 | 0.111 |
| 12 / 24 | Taskmaster | 1.000 | 0.167 |
| 12 / 24 | TopicalChat | 0.944 | 0.083 |

This audit strengthens the substrate case and weakens the consumer case. TTL12
keeps nearly all delayed targets present in STM while staying bounded
(p95 size `13`, p95 update `9.75 us`), so the "shadow DOM" style STM buffer is
not obviously too large or too slow. But target presence is not enough: delayed
targets remain rarely top-3, especially on TopicalChat. TTL32 increases
worst-case size and update outliers without fixing the consumer problem. The
substrate design should therefore preserve ordered STM items for downstream
attention, but retrieval should not simply surface STM by local score.

### STM Order-Policy Audit

We then audited whether the retained STM items can be safely exposed by a simple
ordering policy. Using the exported item-level STM rows, we compared target
rank under raw score, recency, trace score, boundary-gated score, and a simple
recency-then-raw policy. This is a shadow-only analysis over item ordering; it
does not change retrieval.

Artifacts:

- `build/short_term_memory_processor_shadow_order_policy_audit/stm_order_policy_results.json`
- `build/short_term_memory_processor_shadow_order_policy_audit/stm_order_policy_summary.csv`
- `build/short_term_memory_processor_shadow_order_policy_audit/stm_order_policy_combined.csv`
- `build/short_term_memory_processor_shadow_order_policy_audit/stm_order_policy_cases.csv`

Target rank by ordering policy:

| policy | class | mean rank | top-1 | top-3 | top-5 | top-8 |
|---|---|---:|---:|---:|---:|---:|
| raw score | ordinary | 5.93 | 0.183 | 0.442 | 0.533 | 0.742 |
| raw score | delayed | 9.46 | 0.067 | 0.108 | 0.258 | 0.417 |
| recency | ordinary | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 |
| recency | delayed | 8.50 | 0.000 | 0.000 | 0.125 | 0.500 |
| trace score | ordinary | 2.96 | 0.475 | 0.742 | 0.875 | 0.942 |
| trace score | delayed | 8.71 | 0.000 | 0.108 | 0.225 | 0.517 |
| recency then raw top-4 | ordinary | 2.29 | 0.333 | 0.808 | 1.000 | 1.000 |
| recency then raw top-4 | delayed | 10.98 | 0.000 | 0.000 | 0.075 | 0.317 |

This explains why STM looks valuable as a substrate but unsafe as a direct
surface. Ordinary continuations can be recovered by recency-like policies, but
delayed targets are buried by every simple ordering. Raw score keeps `95%` of
delayed targets somewhere in the top-16, yet only `10.8%` are top-3. Recency
makes ordinary continuations trivial and delayed continuations invisible.
Therefore, a useful STM design should preserve ordered item sets and vector
features for downstream attention or sequence models. It should not expose only
a top-k list chosen by recency, raw score, or trace score.

### STM Attention-Window Audit

The order-policy audit implies that the downstream consumer needs a bounded
window of ordered STM items rather than just the best few items. We therefore
measured the attention window size required to include target items under each
simple ordering policy.

Artifacts:

- `build/short_term_memory_processor_shadow_attention_window_audit/stm_attention_window_results.json`
- `build/short_term_memory_processor_shadow_attention_window_audit/stm_attention_window_summary.csv`
- `build/short_term_memory_processor_shadow_attention_window_audit/stm_attention_window_best_delayed.csv`

Window size needed for target inclusion:

| policy | class | coverage @3 | coverage @8 | coverage @12 | coverage @16 | window for 80% | window for 95% |
|---|---|---:|---:|---:|---:|---:|---:|
| raw score | ordinary | 0.442 | 0.742 | 0.867 | 0.983 | 10 | 16 |
| raw score | delayed | 0.108 | 0.417 | 0.700 | 0.950 | 16 | 16 |
| recency | ordinary | 1.000 | 1.000 | 1.000 | 1.000 | 1 | 1 |
| recency | delayed | 0.000 | 0.500 | 1.000 | 1.000 | 12 | 12 |
| trace score | ordinary | 0.742 | 0.942 | 0.958 | 0.983 | 4 | 10 |
| trace score | delayed | 0.108 | 0.517 | 0.817 | 0.983 | 12 | 16 |

The design implication is concrete. If STM is only used for ordinary
continuations, a tiny recency/trace window is enough. If STM is meant to help
delayed continuity, a downstream model must see roughly `12-16` ordered STM
items. Smaller top-3 or top-5 views discard most of the delayed signal before
the consumer has a chance to reason over it. This supports a shadow-DOM style
STM interface: bounded, ordered, and vector-rich, with the consumer responsible
for attention over the window.

### Projection Attention Window Sweep

We then tested whether the inclusion-window result translates into actual
consumer behavior. Using the compact projection vectors, we trained the same
source-held-out projection-attention consumer while limiting the visible STM
window to top-4, top-8, top-12, or top-16 items.

Artifacts:

- `build/short_term_memory_processor_shadow_projection_window_sweep/stm_projection_window_sweep_results.json`
- `build/short_term_memory_processor_shadow_projection_window_sweep/stm_projection_window_sweep_summary.csv`
- `build/short_term_memory_processor_shadow_projection_window_sweep/stm_projection_window_sweep_frontier.csv`
- `build/short_term_memory_processor_shadow_projection_window_sweep/stm_projection_window_sweep_scores.csv`

Best delayed-recovery row by false-positive budget:

| FP budget | window | attention | AUC | recovery | delayed | FP mix |
|---:|---:|---|---:|---:|---:|---|
| 0% | 16 | recent | 0.7805 | 36 / 240 | 8 / 120 | 0 stale, 0 topic |
| 2% | 12 | recent | 0.7640 | 52 / 240 | 20 / 120 | 2 stale, 2 topic |
| 5% | 4 | raw score | 0.7351 | 69 / 240 | 34 / 120 | 6 stale, 6 topic |
| 10% | 16 | uniform | 0.8002 | 122 / 240 | 53 / 120 | 7 stale, 17 topic |

The window sweep shows why STM should expose an ordered window rather than a
single fixed top-k policy. Larger windows help at strict false-positive budgets:
the zero-FP and 2% delayed frontiers use windows `16` and `12`. At 5% FPR, a
small top-4 raw-score window is more aggressive and recovers `34 / 120` delayed
cases, but with weaker AUC and balanced stale/topic false positives. This is
not a promotion result; it is a consumer-design constraint. The right window
size depends on the risk budget, so the substrate should retain enough ordered
items for downstream policy to choose conservatively.

### Projection Window Threshold Transfer

We then repeated the window sweep with threshold transfer. Thresholds were
trained on one dataset or chronological prefix and applied to held-out splits,
then we selected the largest train-side FP budget whose maximum held-out FP rate
stayed below the target. This asks whether the apparent top-4 5% result survives
portable calibration.

Artifacts:

- `build/short_term_memory_processor_shadow_projection_window_transfer/stm_projection_window_transfer_results.json`
- `build/short_term_memory_processor_shadow_projection_window_transfer/stm_projection_window_transfer_summary.csv`
- `build/short_term_memory_processor_shadow_projection_window_transfer/stm_projection_window_transfer_guardband.csv`
- `build/short_term_memory_processor_shadow_projection_window_transfer/stm_projection_window_transfer_frontier.csv`

Transfer-stable window frontier:

| target transfer FP | window | attention | train FP budget | max held-out FP | aggregate recovery | aggregate delayed | FP mix |
|---:|---:|---|---:|---:|---:|---:|---|
| 2% | 12 | recent | 0% | 0.012 | 43 | 14 | 0 stale, 2 topic |
| 5% | 12 | recent | 2% | 0.024 | 70 | 27 | 3 stale, 3 topic |
| 10% | 12 | recent | 5% | 0.094 | 103 | 46 | 10 stale, 12 topic |

The transfer audit changes the interpretation of the within-surface sweep. The
top-4 raw-score row looked attractive at 5% FPR within the same surface, but
after transfer guardbanding it recovers only `15` delayed cases at the 5% target.
Window-12 with recent attention is the stable choice across all transfer
targets, recovering `27` delayed cases at the 5% target with max held-out FP
`0.024`. The STM substrate should therefore retain at least a 12-item ordered
window for delayed continuity. Smaller windows are too brittle once thresholds
must transfer.

### Window-12 Recent Failure Analysis

We then inspected the transfer-stable `window=12`, recent-attention policy at
the 5% transfer target. This uses the 2% train-side guardband thresholds from
the previous experiment and reports which classes survive or fail under those
transferred thresholds.

Artifacts:

- `build/short_term_memory_processor_shadow_window12_recent_failure_analysis/stm_window12_recent_failure_results.json`
- `build/short_term_memory_processor_shadow_window12_recent_failure_analysis/stm_window12_recent_failure_summary.csv`
- `build/short_term_memory_processor_shadow_window12_recent_failure_analysis/stm_window12_recent_failure_cases.csv`
- `build/short_term_memory_processor_shadow_window12_recent_failure_analysis/stm_window12_recent_failure_examples.csv`

Class selection under transferred thresholds:

| split | ordinary selected | delayed selected | stale FP | topic FP |
|---|---:|---:|---:|---:|
| Taskmaster -> TopicalChat | 7 / 36 | 5 / 36 | 1 / 36 | 0 / 36 |
| TopicalChat -> Taskmaster | 25 / 84 | 15 / 84 | 1 / 84 | 3 / 84 |
| chronological 60/40 | 11 / 48 | 7 / 48 | 1 / 48 | 0 / 48 |

Aggregated across these transfer evaluations, the policy recovers `70`
positives but misses `125` ordinary and `141` delayed positives, while allowing
only `6` false positives (`3` stale, `3` topic). The stable policy is therefore
safe because it is conservative, not because it understands delayed continuity.
The false positives are rare high-scoring stale/topic outliers; the much larger
failure is missed positives sitting just below the transferred thresholds. This
confirms that the substrate is retaining useful information, but the current
projection consumer does not extract enough of it under portable thresholds.

### Dual-Window Consumer Probe

Because the window-12 recent policy was safe but conservative, we tested a
dual-window shadow consumer that combines the stable `window=12` recent score
with the more aggressive `window=4` raw score. This is still a diagnostic over
already-computed STM scores; it does not change production retrieval.

Artifacts:

- `build/short_term_memory_processor_shadow_dual_window_probe/stm_dual_window_probe_results.json`
- `build/short_term_memory_processor_shadow_dual_window_probe/stm_dual_window_probe_summary.csv`
- `build/short_term_memory_processor_shadow_dual_window_probe/stm_dual_window_probe_guardband.csv`
- `build/short_term_memory_processor_shadow_dual_window_probe/stm_dual_window_probe_frontier.csv`

Best safe transfer points:

| target transfer FP | score family | train FP budget | max held-out FP | recovery | delayed | FP mix |
|---:|---|---:|---:|---:|---:|---|
| 2% | `w12_recent` | 0% | 0.0119 | 43 | 14 | 0 stale, 2 topic |
| 5% | `avg_w12_w4` | 2% | 0.0312 | 60 | 32 | 4 stale, 3 topic |
| 10% | `logistic_w12_w4` | 5% | 0.0714 | 108 | 57 | 8 stale, 12 topic |

The dual-window average improves delayed recovery at the 5% transfer point
(`32` delayed positives versus `27` for `w12_recent`) while staying below the
5% held-out false-positive target. It does not dominate the conservative
window-12 policy overall: total recovery drops from `70` to `60` and false
positives increase from `6` to `7` at the same 5% target. At the 10% target,
the logistic dual-window probe improves delayed recovery but carries a larger
topic/stale false-positive tail. The useful result is architectural rather
than promotional: STM should expose multiple ordered-window summaries or raw
ordered items to a future consumer, because short and long windows carry
different ordinary/delayed-risk tradeoffs.

We then split-audited the 5% transfer result:

- `build/short_term_memory_processor_shadow_dual_window_split_audit/stm_dual_window_split_audit.json`
- `build/short_term_memory_processor_shadow_dual_window_split_audit/stm_dual_window_split_audit.csv`
- `build/short_term_memory_processor_shadow_dual_window_split_audit/stm_dual_window_split_deltas.csv`

For `avg_w12_w4` against `w12_recent` at the same 5% budget, the gain is
split-specific rather than universal:

| split | recovery delta | delayed delta | false-positive delta |
|---|---:|---:|---:|
| Taskmaster -> TopicalChat | -5 | 0 | -3 |
| TopicalChat -> Taskmaster | +10 | +8 | +3 |
| chronological 60/40 | -12 | -3 | -5 |

This narrows the interpretation. Dual-window scoring can recover delayed
Taskmaster positives that the conservative window-12 consumer misses, but it
also changes the calibration tradeoff by split. That makes it useful evidence
for a richer STM read surface, not a stable consumer policy.

Finally, we audited split-local Pareto choices across the dual-window score
families:

- `build/short_term_memory_processor_shadow_dual_window_pareto_audit/stm_dual_window_pareto_audit.json`
- `build/short_term_memory_processor_shadow_dual_window_pareto_audit/stm_dual_window_pareto_by_split.csv`
- `build/short_term_memory_processor_shadow_dual_window_pareto_audit/stm_dual_window_global_guardband_top5.csv`

At a 5% held-out false-positive target, each split selects a different best
readout:

| split | split-local best score | recovery | delayed | FP / FPR |
|---|---|---:|---:|---:|
| chronological 60/40 | `weighted_30_w12_70_w4` | 23 | 11 | 4 / 0.0417 |
| Taskmaster -> TopicalChat | `weighted_30_w12_70_w4` | 17 | 9 | 3 / 0.0417 |
| TopicalChat -> Taskmaster | `avg_w12_w16uniform` | 65 | 28 | 6 / 0.0357 |

The global guardband still chooses `w12_recent` at the 5% target (`70`
aggregate recovery, `27` delayed, `6` false positives, max held-out FPR
`0.0238`). At the 10% target, the global best shifts to `avg_w12_w16uniform`
(`116` recovery, `49` delayed, `16` false positives, max FPR `0.0625`), while
`logistic_w12_w4` recovers more delayed positives (`57`) at the cost of more
false positives (`20`). This confirms that the STM substrate has multiple
useful views, but split-local optimization is not portable enough to become a
runtime policy.

We also audited objective choice directly:

- `build/short_term_memory_processor_shadow_stm_objective_tradeoff_audit/stm_objective_tradeoff_audit.json`
- `build/short_term_memory_processor_shadow_stm_objective_tradeoff_audit/stm_objective_tradeoff_audit.csv`

Under the same transferred false-positive targets, maximizing total recovery
and maximizing delayed recovery choose different STM readouts:

| target FPR | objective | score family | recovery | delayed | FP | max FPR |
|---:|---|---|---:|---:|---:|---:|
| 2% | max total | `avg_w12_w16recent` | 47 | 12 | 1 | 0.0060 |
| 2% | max delayed | `w12_recent` | 43 | 14 | 2 | 0.0119 |
| 5% | max total | `w12_recent` | 70 | 27 | 6 | 0.0238 |
| 5% | max delayed | `avg_w12_w4` | 60 | 32 | 7 | 0.0312 |
| 10% | max total | `avg_w12_w16uniform` | 116 | 49 | 16 | 0.0625 |
| 10% | max delayed | `logistic_w12_w4` | 108 | 57 | 20 | 0.0714 |

This reinforces the design constraint: an STM substrate can support different
read objectives, but a future consumer must declare which objective it serves
and must pass transfer guardbands for that objective. Treating "more recovery"
as a single scalar hides the ordinary/delayed tradeoff.

The marginal cost of optimizing for delayed recovery is explicit in:

- `build/short_term_memory_processor_shadow_stm_delayed_objective_cost_audit/stm_delayed_objective_cost_audit.json`
- `build/short_term_memory_processor_shadow_stm_delayed_objective_cost_audit/stm_delayed_objective_cost_audit.csv`

Compared with the max-total-recovery readout at each transferred FPR target,
the delayed-optimized readout changes the aggregate outcomes as follows:

| target FPR | readout change | delayed gain | ordinary delta | FP delta | total delta |
|---:|---|---:|---:|---:|---:|
| 2% | `avg_w12_w16recent` -> `w12_recent` | +2 | -6 | +1 | -4 |
| 5% | `w12_recent` -> `avg_w12_w4` | +5 | -15 | +1 | -10 |
| 10% | `avg_w12_w16uniform` -> `logistic_w12_w4` | +8 | -16 | +4 | -8 |

Delayed recovery is therefore not free. The current score families can trade
ordinary recovery and additional false positives for delayed positives, but
they do not provide a universally better readout. This supports keeping STM as
a richer substrate while requiring downstream consumers to choose and justify
their recovery objective.

We then made the robustness criterion stricter by selecting readouts that
maximize the worst split rather than aggregate totals:

- `build/short_term_memory_processor_shadow_stm_minimax_readout_audit/stm_minimax_readout_audit.json`
- `build/short_term_memory_processor_shadow_stm_minimax_readout_audit/stm_minimax_readout_audit.csv`
- `build/short_term_memory_processor_shadow_stm_minimax_readout_audit/stm_minimax_readout_candidates.csv`

Selected readouts:

| target FPR | objective | score family | aggregate recovery | delayed | min split recovery | min split delayed | max FPR |
|---:|---|---|---:|---:|---:|---:|---:|
| 2% | max worst recovery | `avg_w12_w16recent` | 47 | 12 | 8 | 3 | 0.0060 |
| 2% | max worst delayed | `w12_recent` | 43 | 14 | 7 | 3 | 0.0119 |
| 5% | max worst recovery | `weighted_70_w12_30_w4` | 63 | 29 | 13 | 7 | 0.0312 |
| 5% | max aggregate recovery | `w12_recent` | 70 | 27 | 12 | 5 | 0.0238 |
| 10% | max worst recovery | `max_w12_w4` | 103 | 44 | 23 | 11 | 0.0729 |
| 10% | max aggregate recovery | `avg_w12_w16uniform` | 116 | 49 | 21 | 9 | 0.0625 |

Worst-split optimization changes the chosen STM view. At 5%, the robust
readout sacrifices aggregate recovery (`63` versus `70`) but improves the
weakest split (`13` versus `12`) and weakest delayed split (`7` versus `5`).
At 10%, the robust readout again gives up aggregate recovery (`103` versus
`116`) for better worst-split recovery and delayed recovery. This is a further
non-promotion result: if STM is exposed, the API should preserve enough
ordered-window information for downstream consumers, but the current scalar
readouts are not stable under aggregate, delayed, and minimax objectives at
the same time.

We finally computed a multi-objective regret audit over the same feasible
readouts:

- `build/short_term_memory_processor_shadow_stm_multiobjective_regret_audit/stm_multiobjective_regret_audit.json`
- `build/short_term_memory_processor_shadow_stm_multiobjective_regret_audit/stm_multiobjective_regret_summary.csv`
- `build/short_term_memory_processor_shadow_stm_multiobjective_regret_audit/stm_multiobjective_regret_candidates.csv`

The objective set was aggregate recovery, aggregate delayed recovery, worst
split recovery, worst split delayed recovery, and false positives. The
least-max-regret choices were:

| target FPR | least-regret readout | recovery | delayed | FP | min split recovery | min split delayed | max regret |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2% | `avg_w12_w16recent` | 47 | 12 | 1 | 8 | 3 | 2 |
| 5% | `w12_recent` | 70 | 27 | 6 | 12 | 5 | 5 |
| 10% | `avg_w12_w16uniform` | 116 | 49 | 16 | 21 | 9 | 15 |

The 2% case is close to a stable scalar choice, missing only the best delayed
aggregate by `2`. The 5% and 10% cases are not close: the least-regret readout
still misses `5` delayed positives at 5%, and at 10% it misses `8` delayed
positives plus `2` worst-split delayed positives while carrying `15` extra
false positives relative to the minimum-FP feasible option. This is the
clearest scalar-readout result: STM has usable signal, but a single promoted
score cannot satisfy the competing objectives without a trained or explicitly
objective-aware consumer.

As a synthesis, we audited which primitive STM views are required to reproduce
the objective winners above:

- `build/short_term_memory_processor_shadow_stm_read_surface_requirement_audit/stm_read_surface_requirement_audit.json`
- `build/short_term_memory_processor_shadow_stm_read_surface_requirement_audit/stm_read_surface_requirement_winners.csv`
- `build/short_term_memory_processor_shadow_stm_read_surface_requirement_audit/stm_read_surface_requirement_families.csv`

Across `21` winner selections from the objective-tradeoff, minimax,
multi-objective-regret, and global-guardband audits, the winning score families
use these primitive views:

| primitive STM view | why it is needed |
|---|---|
| `window4_raw` | delayed-optimized and worst-split aggressive readouts |
| `window12_recent` | conservative 5% aggregate and least-regret readout |
| `window16_recent` | 2% conservative aggregate / least-regret readout |
| `window16_uniform` | 10% aggregate and least-regret readout |

The winning derived score families include `w12_recent`,
`avg_w12_w16recent`, `avg_w12_w16uniform`, `avg_w12_w4`,
`logistic_w12_w4`, `max_w12_w4`, and `weighted_70_w12_30_w4`. Therefore the
read surface implied by these experiments is not "one STM score." It is a
bounded ordered/windowed STM state from which consumers can compute short,
medium, and long-window views under their own objective and guardband. This is
the most concrete substrate requirement from the STM experiment sequence so
far.

We then ablated those primitive views from the winner set:

- `build/short_term_memory_processor_shadow_stm_view_ablation_audit/stm_view_ablation_audit.json`
- `build/short_term_memory_processor_shadow_stm_view_ablation_audit/stm_view_ablation_by_view.csv`
- `build/short_term_memory_processor_shadow_stm_view_ablation_audit/stm_view_ablation_removed_view.csv`
- `build/short_term_memory_processor_shadow_stm_view_ablation_audit/stm_view_ablation_coverage_sets.csv`

Removing each view makes the following objective winners unavailable:

| removed primitive view | unavailable winners | fraction |
|---|---:|---:|
| `window12_recent` | 21 / 21 | 1.000 |
| `window4_raw` | 6 / 21 | 0.286 |
| `window16_recent` | 5 / 21 | 0.238 |
| `window16_uniform` | 4 / 21 | 0.190 |

The ablation confirms that `window12_recent` is the central stable view, but
the other views are not decorative. `window4_raw` is needed for delayed and
worst-split aggressive objectives, `window16_recent` for the 2% conservative
aggregate/least-regret objective, and `window16_uniform` for the 10%
aggregate/least-regret objective. That argues for a compact four-view read
surface, not a single score and not an unbounded memory dump.

We then checked whether the four-view surface breaks the STM complexity budget:

- `build/short_term_memory_processor_shadow_stm_four_view_budget_audit/stm_four_view_budget_audit.json`
- `build/short_term_memory_processor_shadow_stm_four_view_budget_audit/stm_four_view_budget_audit.csv`

Window-16 support requires a larger retained STM horizon than the previously
preferred TTL12/cap24 substrate. The measured substrate tradeoff is:

| substrate | supports window16 | p95 size | p95 update us | delta p95 update | source 5% recovery | prior cost-benefit dominated |
|---|---:|---:|---:|---:|---:|---|
| TTL12 / cap24 | no | 13 | 9.750 | 0.000 | 63 | no |
| TTL16 / cap32 | yes | 17 | 11.583 | +1.833 | 57 | yes |
| TTL16 / cap64 | yes | 17 | 10.208 | +0.458 | 57 | yes |

The latency/size cost of a window-16-capable substrate is small in absolute
terms: TTL16/cap64 adds `4` p95 items and `0.458 us` p95 STM update latency
over TTL12/cap24. However, the old simple consumer did not improve on that
larger substrate (`57` source-held-out 5% recovery versus `63` for
TTL12/cap24), so the extra horizon is justified only if STM is exposed as a
multi-view substrate for downstream consumers. It should not be interpreted as
evidence that larger TTL alone improves retrieval.

We then summarized the complexity/coverage curve for primitive STM views:

- `build/short_term_memory_processor_shadow_stm_view_subset_efficiency_audit/stm_view_subset_efficiency_audit.json`
- `build/short_term_memory_processor_shadow_stm_view_subset_efficiency_audit/stm_view_subset_efficiency_best.csv`
- `build/short_term_memory_processor_shadow_stm_view_subset_efficiency_audit/stm_view_subset_efficiency_all.csv`

Best objective-winner coverage by view count:

| primitive view count | best view set | covered winners |
|---:|---|---:|
| 1 | `window12_recent` | 6 / 21 |
| 2 | `window12_recent`, `window4_raw` | 12 / 21 |
| 3 | `window12_recent`, `window16_recent`, `window4_raw` | 17 / 21 |
| 4 | `window12_recent`, `window16_recent`, `window16_uniform`, `window4_raw` | 21 / 21 |

The best two-view surface covers the stable 5% and aggressive short-window
objectives but misses the 2% and 10% long-window aggregate/least-regret
objectives. The best three-view surface still misses the 10% `window16_uniform`
objective family. Full objective coverage requires all four primitive views.
This makes the complexity tradeoff explicit: a one- or two-view surface may be
reasonable for a minimal implementation, but it would knowingly discard parts
of the observed STM value.

We then converted the view-count curve into implementation tiers:

- `build/short_term_memory_processor_shadow_stm_read_surface_tier_audit/stm_read_surface_tier_audit.json`
- `build/short_term_memory_processor_shadow_stm_read_surface_tier_audit/stm_read_surface_tier_audit.csv`

| tier | primitive views | substrate | covered winners | p95 size | p95 update us | old consumer source 5% |
|---|---|---|---:|---:|---:|---:|
| minimal stable | `window12_recent` | TTL12/cap24 | 6 / 21 | 13 | 9.750 | 63 |
| core short/medium | `window12_recent`, `window4_raw` | TTL12/cap24 | 12 / 21 | 13 | 9.750 | 63 |
| extended long-recent | + `window16_recent` | TTL16/cap64 | 17 / 21 | 17 | 10.208 | 57 |
| full four-view | + `window16_uniform` | TTL16/cap64 | 21 / 21 | 17 | 10.208 | 57 |

The two-view core surface doubles objective coverage over the one-view surface
without increasing the retained horizon. The full four-view surface covers all
recent objective winners and costs only `+4` p95 items and `+0.458 us` p95 STM
update latency relative to TTL12/cap24, but it depends on a substrate that was
dominated for the old simple consumer. This gives a practical implementation
choice: use the two-view surface if minimizing complexity now, or the four-view
surface if the goal is to preserve the full observed STM signal for future
consumers. Neither choice changes production retrieval until a consumer passes
transfer guardbands.

The resulting STM decision gate is:

- `build/short_term_memory_processor_shadow_stm_decision_gate_audit/stm_decision_gate_audit.json`
- `build/short_term_memory_processor_shadow_stm_decision_gate_audit/stm_decision_gate_criteria.csv`
- `build/short_term_memory_processor_shadow_stm_decision_gate_audit/stm_decision_gate_tiers.csv`

| component | status | action |
|---|---|---|
| STM substrate boundedness | pass | keep as bounded shadow substrate candidate |
| STM direct retrieval consumer | blocked | do not change production retrieval |
| single scalar STM score | blocked | do not expose one score as policy |
| two-view core read surface | candidate | use for minimal shadow interface if complexity must stay TTL12 |
| four-view full read surface | candidate | preferred research substrate if preserving observed signal matters |
| larger TTL as retrieval improvement | blocked | do not argue larger TTL alone improves retrieval |
| future STM consumer | required | train/evaluate shadow-only with declared objective and transfer guardbands |

This is the current experimental stopping point. STM is justified as a bounded
substrate/read surface, not as a production retrieval consumer. The next
implementation decision should be API shape: either the minimal two-view
TTL12/cap24 surface or the full four-view TTL16/cap64 surface. Retrieval
behavior should remain unchanged until a downstream consumer passes the same
transfer and control-tail gates.

### Dataset and Control-Class Breakdown

We then audited the recommended policy/threshold points by dataset and control
class.

Artifacts:

- `build/short_term_memory_processor_shadow_dataset_breakdown/stm_policy_dataset_breakdown.json`
- `build/short_term_memory_processor_shadow_dataset_breakdown/stm_policy_dataset_breakdown.csv`
- `build/short_term_memory_processor_shadow_dataset_breakdown/stm_policy_dataset_cases.csv`

Source-held-out dataset breakdown:

| config | policy | threshold | Taskmaster recovery / FP | TopicalChat recovery / FP |
|---|---|---:|---:|---:|
| conservative zero-FP | TTL8 / cap16 | 0.8271 | 27 / 168, 0 FP | 7 / 72, 0 FP |
| balanced 2% FP | TTL12 / cap24 | 0.7535 | 43 / 168, 5 FP | 8 / 72, 0 FP |
| balanced 5% FP | TTL12 / cap24 | 0.7059 | 54 / 168, 11 FP | 9 / 72, 1 FP |
| high-recall 10% FP | TTL12 / cap24 | 0.6214 | 68 / 168, 22 FP | 12 / 72, 2 FP |

Control-class breakdown:

| config | ordinary recovery | delayed recovery | stale FP | topic-shift FP |
|---|---:|---:|---:|---:|
| conservative zero-FP | 30 / 120 | 4 / 120 | 0 / 120 | 0 / 120 |
| balanced 2% FP | 40 / 120 | 11 / 120 | 0 / 120 | 5 / 120 |
| balanced 5% FP | 45 / 120 | 18 / 120 | 2 / 120 | 10 / 120 |
| high-recall 10% FP | 51 / 120 | 29 / 120 | 7 / 120 | 17 / 120 |

This breakdown sharpens the interpretation. The current STM consumer is useful
but not uniformly useful: Taskmaster carries most of the recovery, while
TopicalChat remains low-recall. The false-positive burden is mostly topic
shift, not stale same-source carryover. That means a future shadow consumer
should not focus only on stale decay; it also needs topic-shift / discourse
boundary discrimination.

### Dataset-Local Threshold Diagnostic

We then tested whether TopicalChat's low recovery was caused by weak STM signal
or by a global threshold calibrated mostly on Taskmaster. This is diagnostic
only: per-dataset thresholds are not a production strategy, but they reveal
whether the score scale shifts across datasets.

Artifacts:

- `build/short_term_memory_processor_shadow_dataset_local_thresholds/stm_dataset_local_thresholds.json`
- `build/short_term_memory_processor_shadow_dataset_local_thresholds/stm_dataset_local_thresholds.csv`
- `build/short_term_memory_processor_shadow_dataset_local_thresholds/stm_dataset_local_best_by_budget.csv`

Best dataset-local policies by false-positive budget:

| dataset | FP budget | policy | recovery | false positives | precision | ordinary | delayed | threshold |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Taskmaster | 0% | TTL16 / cap32 | 28 / 168 | 0 | 1.000 | 25 | 3 | 0.8919 |
| Taskmaster | 2% | TTL16 / cap32 | 39 / 168 | 4 | 0.907 | 31 | 8 | 0.7809 |
| Taskmaster | 5% | TTL8 / cap16 | 49 / 168 | 9 | 0.845 | 33 | 16 | 0.6922 |
| Taskmaster | 10% | TTL12 / cap24 | 58 / 168 | 17 | 0.773 | 37 | 21 | 0.6832 |
| TopicalChat | 0% | TTL12 / cap24 | 8 / 72 | 0 | 1.000 | 8 | 0 | 0.7233 |
| TopicalChat | 2% | TTL12 / cap24 | 12 / 72 | 2 | 0.857 | 11 | 1 | 0.6244 |
| TopicalChat | 5% | TTL12 / cap24 | 21 / 72 | 4 | 0.840 | 15 | 6 | 0.4891 |
| TopicalChat | 10% | TTL16 / cap32 | 31 / 72 | 8 | 0.795 | 22 | 9 | 0.4343 |

The TopicalChat result changes the diagnosis. Under the global 5% threshold,
TopicalChat recovered only `9 / 72`; with a dataset-local 5% threshold, it
recovers `21 / 72` at similar precision (`0.840`). So TopicalChat does contain
some usable STM signal, but its score scale is lower. A future shadow consumer
needs calibration features that explain corpus/domain shift without using
dataset identity directly. Candidate features include score distribution
statistics, boundary pressure, discourse entropy, source duration, and recent
topic-shift indicators.

### Source-Local Score Normalization Diagnostic

To test a dataset-free calibration feature, we normalized the TTL12/cap24
source-held-out scores within each conversation/source group. This is still
diagnostic: the benchmark groups are not a streaming production calibrator. The
goal was only to test whether local score statistics can reduce the Taskmaster /
TopicalChat scale mismatch.

Artifacts:

- `build/short_term_memory_processor_shadow_score_normalization/stm_score_normalization_results.json`
- `build/short_term_memory_processor_shadow_score_normalization/stm_score_normalization_frontier.csv`
- `build/short_term_memory_processor_shadow_score_normalization/stm_score_normalization_cases.csv`

At the 5% false-positive budget:

| score | recovery | precision | Taskmaster recovery | TopicalChat recovery | stale FP | topic FP |
|---|---:|---:|---:|---:|---:|---:|
| raw calibrated score | 63 / 240 | 0.840 | 54 | 9 | 2 | 10 |
| source-centered | 60 / 240 | 0.833 | 42 | 18 | 4 | 8 |
| source z-score | 61 / 240 | 0.836 | 46 | 15 | 3 | 9 |
| source percentile | 62 / 240 | 0.838 | 48 | 14 | 3 | 9 |
| source robust z | 43 / 240 | 0.782 | 26 | 17 | 5 | 7 |

At zero false positives:

| score | recovery | Taskmaster recovery | TopicalChat recovery |
|---|---:|---:|---:|
| raw calibrated score | 31 / 240 | 24 | 7 |
| source-centered | 24 / 240 | 13 | 11 |
| source z-score | 27 / 240 | 16 | 11 |
| source robust z | 13 / 240 | 5 | 8 |

The diagnostic does not beat the raw score on total recovery, but it improves
domain balance. At 5% FP, source-centering doubles TopicalChat recovery
(`9` to `18`) while reducing Taskmaster recovery (`54` to `42`) and keeping
precision similar. This suggests the future consumer should include local
score-distribution features alongside the raw calibrated score, then let a
held-out calibration decide when normalization is helpful.

### Raw + Local-Normalization Ensemble Diagnostic

We then tested fixed, training-free score combinations over the TTL12/cap24
scores to see whether local normalization can complement the raw calibrated
score. Features were standardized using control-score statistics only in this
diagnostic; no dataset identity or production retrieval path was used.

Artifacts:

- `build/short_term_memory_processor_shadow_score_ensemble/stm_score_ensemble_results.json`
- `build/short_term_memory_processor_shadow_score_ensemble/stm_score_ensemble_frontier.csv`

At the 5% false-positive budget:

| score | recovery | precision | Taskmaster recovery | TopicalChat recovery | delayed recovery | stale FP | topic FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw calibrated | 63 / 240 | 0.840 | 54 | 9 | 18 | 2 | 10 |
| source-centered | 60 / 240 | 0.833 | 42 | 18 | 15 | 4 | 8 |
| max(raw, centered) | 63 / 240 | 0.840 | 47 | 16 | 17 | 2 | 10 |
| avg(raw, centered) | 60 / 240 | 0.833 | 47 | 13 | 15 | 4 | 8 |
| avg(raw, source z) | 60 / 240 | 0.833 | 48 | 12 | 16 | 4 | 8 |
| avg(raw, source percentile) | 65 / 240 | 0.844 | 53 | 12 | 18 | 2 | 10 |

At the 2% false-positive budget:

| score | recovery | precision | Taskmaster recovery | TopicalChat recovery | delayed recovery | stale FP | topic FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw calibrated | 51 / 240 | 0.911 | 43 | 8 | 11 | 0 | 5 |
| source-centered | 46 / 240 | 0.902 | 31 | 15 | 10 | 1 | 4 |
| max(raw, centered) | 51 / 240 | 0.911 | 38 | 13 | 10 | 1 | 4 |
| avg(raw, centered) | 54 / 240 | 0.915 | 42 | 12 | 11 | 0 | 5 |
| avg(raw, source percentile) | 51 / 240 | 0.911 | 42 | 9 | 12 | 0 | 5 |

This is a modest but useful result. Fixed combinations do not eliminate the
domain shift, but they show local distribution features can complement the raw
score. At 5% FP, `avg(raw, source percentile)` improves total recovery from
`63` to `65` without reducing precision. At 2% FP, `avg(raw, centered)` improves
recovery from `51` to `54` and slightly improves precision. The next real
consumer should therefore train or calibrate over raw score plus source-local
distribution features, rather than choosing one score family manually.

### Held-Out Consumer Transfer Diagnostic

We then made the calibration test stricter. Instead of choosing thresholds on
the evaluation set, we trained / calibrated on one split and transferred the
thresholds to a held-out dataset or chronological holdout. This used no dataset
identity at runtime, but the diagnostic deliberately reports dataset-held-out
failure modes.

Artifacts:

- `build/short_term_memory_processor_shadow_consumer_transfer/stm_consumer_transfer_results.json`
- `build/short_term_memory_processor_shadow_consumer_transfer/stm_consumer_transfer_results.csv`

At the nominal 5% false-positive budget:

| transfer | score | recovery | false positives | precision | observed FPR |
|---|---|---:|---:|---:|---:|
| train TopicalChat -> test Taskmaster | raw | 105 / 168 | 65 / 168 | 0.618 | 0.387 |
| train TopicalChat -> test Taskmaster | avg(raw, percentile) | 60 / 168 | 15 / 168 | 0.800 | 0.089 |
| train TopicalChat -> test Taskmaster | logistic local features | 68 / 168 | 21 / 168 | 0.764 | 0.125 |
| train Taskmaster -> test TopicalChat | raw | 8 / 72 | 0 / 72 | 1.000 | 0.000 |
| train Taskmaster -> test TopicalChat | avg(raw, percentile) | 10 / 72 | 1 / 72 | 0.909 | 0.014 |
| train Taskmaster -> test TopicalChat | logistic local features | 13 / 72 | 2 / 72 | 0.867 | 0.028 |
| chronological 60/40 | raw | 26 / 96 | 3 / 96 | 0.897 | 0.031 |
| chronological 60/40 | avg(raw, percentile) | 28 / 96 | 7 / 96 | 0.800 | 0.073 |
| chronological 60/40 | logistic local features | 24 / 96 | 5 / 96 | 0.828 | 0.052 |

At nominal zero false positives:

| transfer | score | recovery | false positives | precision | observed FPR |
|---|---|---:|---:|---:|---:|
| train TopicalChat -> test Taskmaster | raw | 48 / 168 | 9 / 168 | 0.842 | 0.054 |
| train TopicalChat -> test Taskmaster | avg(raw, percentile) | 41 / 168 | 4 / 168 | 0.911 | 0.024 |
| train TopicalChat -> test Taskmaster | logistic local features | 28 / 168 | 2 / 168 | 0.933 | 0.012 |
| train Taskmaster -> test TopicalChat | raw | 7 / 72 | 0 / 72 | 1.000 | 0.000 |
| train Taskmaster -> test TopicalChat | avg(raw, percentile) | 8 / 72 | 0 / 72 | 1.000 | 0.000 |
| train Taskmaster -> test TopicalChat | logistic local features | 8 / 72 | 0 / 72 | 1.000 | 0.000 |
| chronological 60/40 | raw | 12 / 96 | 0 / 96 | 1.000 | 0.000 |
| chronological 60/40 | avg(raw, percentile) | 14 / 96 | 0 / 96 | 1.000 | 0.000 |
| chronological 60/40 | logistic local features | 12 / 96 | 0 / 96 | 1.000 | 0.000 |

This is the strongest caution so far. Local features improve some transfers,
but threshold calibration is not stable across datasets. Training on TopicalChat
and testing on Taskmaster badly violates the intended false-positive budget for
the raw score. The fixed ensemble and logistic local-feature consumer reduce
that damage, but they do not make the budget reliable. The next STM consumer
must therefore report threshold transfer, not just within-surface budget
frontiers, before it can be treated as a safe shadow candidate.

## Current Experimental Decision

The latest STM decision gate is recorded in:

- `build/short_term_memory_processor_shadow_stm_decision_gate_audit/stm_decision_gate_audit.json`
- `build/short_term_memory_processor_shadow_stm_decision_gate_audit/stm_decision_gate_criteria.csv`
- `build/short_term_memory_processor_shadow_stm_decision_gate_audit/stm_decision_gate_tiers.csv`

The decision is:

| component | status |
|---|---|
| bounded STM substrate | pass |
| direct retrieval consumer | blocked |
| single scalar STM score | blocked |
| two-view core read surface | candidate |
| four-view full read surface | candidate |
| larger TTL as retrieval improvement | blocked |
| future consumer | required, shadow-only |

STM is experimentally justified as a bounded internal substrate/read surface.
It is not justified as a production retrieval consumer. The practical API
choice is between a minimal two-view TTL12/cap24 surface
(`window12_recent`, `window4_raw`) and a full four-view TTL16/cap64 surface
that also exposes `window16_recent` and `window16_uniform`. Production
retrieval should remain unchanged until a downstream consumer declares its
objective and passes transfer guardbands.

## STM Attention Anchor Rerun

We then reran the anchor-attention question using the proposed STM substrate
instead of a special anchor model. This was a shadow-only benchmark mode:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --stm-attention-anchor-shadow-only \
  --models=models \
  --output-dir build/stm_attention_anchor_shadow_ttl16_cap64_full
```

The benchmark builds `STM_before_t` chronologically with the proposed
TTL16/cap64 substrate, then evaluates attention readouts over:

- `window4_raw`
- `window12_recent`
- `window16_recent`
- `window16_uniform`
- `core_two_view`
- `proposed_four_view`
- `proposed_four_view_entropy_margin_gate`

Artifacts:

- `build/stm_attention_anchor_shadow_ttl16_cap64_full/stm_attention_anchor_results.json`
- `build/stm_attention_anchor_shadow_ttl16_cap64_full/stm_attention_anchor_ablation.csv`
- `build/stm_attention_anchor_shadow_ttl16_cap64_full/stm_attention_anchor_scores.csv`
- `build/stm_attention_anchor_shadow_ttl16_cap64_full/stm_attention_anchor_failure_examples.csv`

Full 480-case results:

| STM attention view | target/control AUC | inverted AUC | target top-3 | zero-FPR recovery | 5% FPR recovery |
|---|---:|---:|---:|---:|---:|
| window4 raw | 0.1185 | 0.8815 | 97 / 240 | 1 / 240 | 6 / 240 |
| window12 recent | 0.1371 | 0.8629 | 90 / 240 | 2 / 240 | 10 / 240 |
| window16 recent | 0.1372 | 0.8628 | 90 / 240 | 2 / 240 | 10 / 240 |
| window16 uniform | 0.1216 | 0.8784 | 83 / 240 | 2 / 240 | 6 / 240 |
| core two-view | 0.1342 | 0.8658 | 93 / 240 | 2 / 240 | 10 / 240 |
| proposed four-view | 0.1348 | 0.8652 | 89 / 240 | 2 / 240 | 10 / 240 |
| proposed four-view + entropy/margin gate | 0.1536 | 0.8464 | 89 / 240 | 2 / 240 | 10 / 240 |

The 120-case control run of the existing repaired anchor attention benchmark
remained better on the same broad question: direct hierarchical attention had
delayed AUC 0.589 with 1 / 36 zero-FPR recovery, while the attention readouts
reached 4-5 / 36 low-FPR recovery. The STM-attention rerun therefore does not
replace the special anchor-model question.

Interpretation: proposed STM gives a useful bounded evidence substrate, but
attention over STM alone still behaves like a high-recall recent-evidence
finder. Topic-shift and stale controls can attract strong STM attention, so the
missing piece is still a commitment/verifier signal, not merely a larger
attention field. This result keeps STM as a substrate/read surface candidate,
not as an anchor mechanism.

## Key Design Decision

STM should be added to Cortext as a general internal memory layer first. The
specific consumers, including any future anchor model, are downstream.

If STM is added only as a model input hack, it will be hard to reason about
knob behavior, consolidation, and retrieval boundaries. If STM is a first-class
streaming buffer with clear F/S/T behavior, it becomes a reusable substrate for
consolidation, continuity modeling, and future multimodal event modeling.

### Real-Media Episode Ablation

We added `cortext_real_multimodal_episode_bench` and
`tools/prepare_real_multimodal_episode_assets.py` to test with actual media.
The asset script downloads public Wikimedia media, converts images to
`384x384` RGB and audio to `16 kHz` mono float32 PCM, and records source URLs,
licenses, and SHA-256 hashes. The benchmark uses the real Cortext API:
`ProcessImage`, `ProcessAudio`, and `ProcessText`. It does not construct
embeddings directly and does not set forced boundaries.

Artifacts:

- `build/real_multimodal_episode_assets/real_multimodal_episode_assets_manifest.json`
- `build/real_multimodal_episode_bench/real_multimodal_episode_summary.json`
- `build/real_multimodal_episode_bench/real_multimodal_episode_cases.json`
- `build/real_multimodal_episode_bench/real_multimodal_episode_cases.csv`
- `build/real_multimodal_episode_bench_no_signal_filter/real_multimodal_episode_summary.json`

| scenario | dog image+name fused | dog image+audio fused | all dog modalities fused | event split success | mixed-event memory |
|---|---:|---:|---:|---:|---:|
| dog image + text + Bailey audio | 1 | 0 | 0 | n/a | 0 |
| dog image/text/audio then car image/text/audio | 1 | 0 | 0 | 0 | 1 |
| dog image/audio then car image/audio | 1 | 1 | 0 | 1 | 0 |

Interpretation: the real-media path is not ready to call solved. The
image/audio-only sequence works, but inserting text between the dog image and
Bailey audio causes the text step to write the image+text dog memory while the
Bailey audio stays open. In the full dog-then-car sequence, that open Bailey
audio carries into the car-crash image and creates a mixed dog/car memory. This
points to boundary/write timing around multimodal same-episode audio, not to a
source-id issue.

The no-signal-filter rerun confirms the filter is not responsible. The
benchmark explicitly reports `signal_filter_used=false` and processes every
event through the real Cortext API. The full dog-then-car sequence still
processed `6/6` events, still failed dog image+audio fusion, and still produced
one mixed dog/car memory.
