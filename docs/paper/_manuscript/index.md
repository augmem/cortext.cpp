# Cortext: LLM-Free Long-Horizon Memory
Gabriel Willen
2026-10-01

# Abstract

We present Cortext, a local multimodal memory engine that stores and
recalls long conversational and sensory streams without calling a
language model on either path.

*Write path.* An online Bayesian change-point segmenter over embedding
statistics closes each source stream into coherent units, and every
closed unit is stored. A stored unit is indexed at three granularities:
a pooled unit vector, one shared-space vector per member signal, and 16
int8 late-interaction codes per text or image payload, up to 16. Its
text also enters an FTS5 keyword index.

*Read path.* A query builds a dense seed list over pooled unit vectors,
into which best-member hits are merged. When auxiliary channels return
hits, it fuses that list by reciprocal-rank fusion with BM25 keyword
seeds, one gap-corrected list per other stored modality, and an optional
reference-time list. A 0.1M-parameter late-interaction MaxSim reranker
over frozen AIST-87M position features then reorders the top 64.
Hydration attaches adjacent turns, superseded predecessors and
superseding successors, each with its own timestamp, and fits the packet
to a caller-supplied byte budget.

*Results.* Under one shared answer model, judge and 34,800-byte budget
on LoCoMo, Cortext scores 86.2%, against 89.0% for Mem0 and 81.0% for
Graphiti. It makes zero write-time model calls and recalls in 9.7 ms at
the median.

*Negative result.* A leave-one-out ablation on evaluations of up to 111k
turns tested the bio-inspired lifecycle mechanisms of earlier versions:
strength decay and eviction, reconsolidation, clustering consolidation,
retrieval-induced forgetting, synaptic tagging and neuromodulation. Each
was inert at its defaults or hurt recall once it fired. Eviction alone
cut packet recall from 0.630 to 0.086. We report which benchmarks
overlap the reranker’s training data. The original three-knob control
design (Focus, Sensitivity, Stability) remains as a macro-control
surface over a calibrated default profile.

**Keywords:** long-term memory, event segmentation, hybrid retrieval,
reciprocal-rank fusion, late interaction, multimodal retrieval, negative
results

# Introduction

An assistant that runs for months accumulates more history than any
context window holds. The common answer is to spend language-model calls
at write time: extract facts, summarize sessions, maintain a knowledge
graph, and retrieve from those products later (Packer et al. 2023;
Chhikara et al. 2025; Rasmussen et al. 2025). This works, but it makes
memory as expensive and as slow as the model. It rewrites what was said
into what a model inferred, and it cannot run on a device without one.

Cortext takes the other path. It stores what was observed, as segmented
units of the original signals, and puts the work into retrieval and
hydration. Neither path calls a language model. The only learned
components are a frozen multimodal encoder (AIST-87M) and a
0.1M-parameter late-interaction reranker over its per-position features.

This paper is a rewrite of the earlier Cortext specification, which
presented a three-knob (F, S, T) cognitive architecture: twelve metrics
blended by recursive least squares, mood and affect, a homeostatic write
threshold, and a family of bio-inspired lifecycle mechanisms.
Long-horizon ablation showed the lifecycle mechanisms did not earn their
place (<a href="#sec-negative" class="quarto-xref">Section 14</a>), and
removing the write threshold raised recall. What drives recall in the
current (v5) engine is a different set of mechanisms. The paper is
organized around them.

The contributions are:

1.  **LLM-free online segmentation, storing every closed unit**
    (<a href="#sec-thresholding" class="quarto-xref">Section 5</a>). A
    calibrated change-point posterior over surprisal, drift, coherence,
    topic shift and inter-arrival gaps decides unit boundaries per
    source stream. The write rule is
    `write_memory = force_write ∨ should_flush`.
2.  **Multi-granularity hybrid retrieval**
    (<a href="#sec-retrieval" class="quarto-xref">Section 6</a>). Dense
    seeds score pooled unit vectors, and best-member hits raise or join
    them in the same list. Reciprocal-rank fusion (k = 60) then combines
    that list with FTS5 BM25, one z-gated list per other stored
    modality, and a reference-time list, whenever any of those returns
    hits.
3.  **A modality-agnostic late-interaction reranker**
    (<a href="#sec-retrieval" class="quarto-xref">Section 6</a>). Each
    stored member keeps up to 16 int8 spherical k-means codes of its
    encoder positions, at most about 1 KB. Queries rerank the top 64 by
    gated MaxSim with a learned first-stage rank prior, in 0.4 to 3 ms.
4.  **Hydration and evidence fitting**
    (<a href="#sec-hydration" class="quarto-xref">Section 7</a>):
    -   contiguity with adjacent turns;
    -   superseded-predecessor and superseding-successor attachment
        through `supersedes` edges;
    -   standing directives outside the ranked list;
    -   per-blob content timestamps;
    -   a per-call evidence budget that trims grouped units to their
        best members before dropping whole memories.
5.  **Negative results**
    (<a href="#sec-negative" class="quarto-xref">Section 14</a>). A
    leave-one-out study removed:
    -   forgetting through strength decay and eviction;
    -   reconsolidation;
    -   clustering consolidation;
    -   retrieval-induced forgetting;
    -   synaptic tagging;
    -   the NE-like and DA-like neuromodulators;
    -   meta-learned priors and several feedback loops.

    It also found that short-horizon blind-judge screens had reported
    harm from removing mechanisms whose packets were byte-identical.
6.  **Realtime systems design**
    (<a href="#sec-optimization" class="quarto-xref">Section 17</a>):
    -   a content-addressed embedding arena;
    -   a sparse in-memory route;
    -   off-thread WAL checkpointing;
    -   sound family-collapse bounds;
    -   a top-k BM25 collector.

    Together these keep query p95 at 44.5 ms on a 443k-turn store.

<a href="#sec-overview" class="quarto-xref">Section 3</a> gives the
whole pipeline and says exactly what the three knobs still influence.
<a href="#sec-math" class="quarto-xref">Section 4</a> fixes notation and
the knob maps. Then come segmentation and the write path
(<a href="#sec-thresholding" class="quarto-xref">Section 5</a>),
retrieval (<a href="#sec-retrieval" class="quarto-xref">Section 6</a>)
and hydration
(<a href="#sec-hydration" class="quarto-xref">Section 7</a>), which form
the main line. The per-signal control stack follows
(<a href="#sec-core-adaptation" class="quarto-xref">Section 8</a>,
<a href="#sec-structural-metrics" class="quarto-xref">Section 9</a>),
with working memory and supersession
(<a href="#sec-advanced" class="quarto-xref">Section 10</a>), the
streaming interrupt gate
(<a href="#sec-interrupt" class="quarto-xref">Section 11</a>),
maintenance
(<a href="#sec-consolidation" class="quarto-xref">Section 12</a>), and
the record of removed reinforcement and decay
(<a href="#sec-reinforcement" class="quarto-xref">Section 13</a>).
<a href="#sec-negative" class="quarto-xref">Section 14</a> presents the
ablation, <a href="#sec-experimental" class="quarto-xref">Section 15</a>
the full experimental record with reranker-overlap separation,
<a href="#sec-implementation" class="quarto-xref">Section 16</a>
implementation detail, and
<a href="#sec-optimization" class="quarto-xref">Section 17</a> the
systems architecture. The historical optimization ledger is preserved in
Appendix E (`docs/paper/appendix/optimization_ledger.qmd`).

# Related Work

## LLM Memory Systems and Benchmarks

MemGPT (Packer et al. 2023) pages context in and out of a model’s window
under model control. Mem0 (Chhikara et al. 2025) extracts and
consolidates facts with a model at write time, and Zep (Rasmussen et al.
2025) (with its open-source engine Graphiti) maintains a temporal
knowledge graph built by model calls. Cortext differs in kind: it makes
no model call on either path, and it stores observed signals rather than
model-written facts.
<a href="#sec-v5-benchmarks" class="quarto-xref">Section 15.2</a>
compares it with Mem0 and Graphiti under one shared pipeline on LoCoMo
(Maharana et al. 2024), and with published results on LongMemEval (Wu et
al. 2024), BEAM, DolphinBench and MemLens.

## Event Segmentation

Cortext’s unit of memory comes from event segmentation theory (Zacks and
Swallow 2007): people segment continuous experience at points of
prediction error, and those boundaries structure later recall. EM-LLM
(Fountas et al. 2024) segments token streams into episodic events by
surprise, refined by graph-theoretic cohesion. Cortext adapts the idea
to any embedding stream. EMA prediction error supplies surprise,
within-unit cosine coherence supplies cohesion, and drift, topic shift
and inter-arrival gaps supply further evidence for a calibrated
change-point posterior
(<a href="#sec-thresholding" class="quarto-xref">Section 5</a>). Because
it operates on embeddings, the same segmenter serves text turns, audio
chunks and images.

## Hybrid and Late-Interaction Retrieval

BM25 (Robertson and Zaragoza 2009) remains a strong lexical baseline,
and fusing lexical and dense rankings by reciprocal rank (Cormack,
Clarke, and Buettcher 2009) is a robust, training-free way to combine
them. Cortext uses RRF with the standard k = 60 over the dense list
(with member hits merged in), the keyword list, the per-modality
cross-modal lists and the temporal list. Late-interaction retrieval
(Khattab and Zaharia 2020) scores a query against a document’s per-token
vectors by MaxSim. Cortext applies it as a reranker over frozen encoder
positions of any modality. Each stored member compresses its positions
to at most 16 int8 spherical k-means codes, which keeps storage near 1
KB per payload without a shared codebook. The sparse candidate route
follows the HNSW construction (Malkov and Yashunin 2018).

## Working Memory Models

The working memory component draws conceptually on Cowan (2001)
embedded-processes model, which posits a capacity limit of about 4±1
chunks for the focus of attention. That figure contrasts with Miller
(1956) earlier estimate of 7±2 items, which later research showed
conflates chunking with raw capacity (Cowan 2010). Cortext’s production
slots are larger application-level memory units rather than cognitive
chunks. The shipped F/S-derived capacity is 15–27 slots, 21 at neutral
knobs, selected from the product evaluation in
<a href="#sec-experimental" class="quarto-xref">Section 15</a>. The
biological result motivates a bounded set, not a claim that the software
reproduces human span. The episodic buffer of Baddeley (2000) aligns
with binding information across modalities through one shared vector
space.

## Consolidation, Reconsolidation and Forgetting

Complementary learning systems (McClelland, McNaughton, and O’Reilly
1995) motivate a fast episodic store beside slower structure, and
reconsolidation theory (Nader, Schafe, and Le Doux 2000; Nader 2003)
holds that retrieved memories become labile and can be modified before
restabilizing. Earlier Cortext versions implemented the following, each
with knob-derived rates:

-   reconsolidation drift during a lability window;
-   clustering consolidation into abstraction nodes;
-   multi-timescale strength decay with eviction;
-   retrieval-induced forgetting (Anderson, Bjork, and Bjork 1994).

All four were evaluated and removed
(<a href="#sec-negative" class="quarto-xref">Section 14</a>).
Reconsolidation was neutral to harmful, and consolidation never fired at
its defaults and did not help when forced. Eviction deleted 92–99.7% of
stores once its gate opened. The engine now keeps the episodic substrate
(Tulving 1972) and no consolidated semantic tier. Its only
consolidation-like step is maintenance of the retrieval route
(<a href="#sec-consolidation" class="quarto-xref">Section 12</a>).

## Emotional Influences on Memory

McGaugh (2004) showed that arousal modulates memory consolidation
through amygdala-mediated effects on hippocampal encoding (LaBar and
Cabeza 2006). Earlier Cortext versions implemented emotional
consolidation and an emotional cascade over graph neighbours. Both were
removed after leaving every packet byte-identical
(<a href="#sec-negative" class="quarto-xref">Section 14</a>). The
per-signal emotion projection onto a dimensional valence-arousal model
(Russell 1980) remains in the control stack. Its valence and arousal
feed the composite score and the streaming interrupt gate
(<a href="#sec-core-adaptation" class="quarto-xref">Section 8</a>). The
emotion threshold delta and the mood integrator were removed with the
dynamic threshold.

## Adaptive Control

Earlier versions steered a write threshold with a homeostatic
controller, following proportional-integral setpoint control (Åström and
Murray 2008), with an effective-sample-size reliability estimate from
sequential Monte Carlo (Liu and Chen 1998). Once every closed unit was
stored, the controller had no consumer, and it was removed
(<a href="#sec-spike-threshold" class="quarto-xref">Section 5.2</a>).
The adaptive pieces that remain are EWMA rate and cadence estimates, an
uncertainty estimate that sets learning rates, and
recursive-least-squares blending of the composite score
(<a href="#sec-structural-metrics" class="quarto-xref">Section 9</a>).

# System Overview

Cortext turns a stream of signals (text turns, audio chunks, images)
into a SQLite store, and turns a later query into a small, budgeted
packet of evidence. No language model runs on either path. The only
learned components are the frozen AIST-87M multimodal encoder and a
0.1M-parameter late-interaction reranker over the encoder’s per-position
features. This section gives the whole pipeline in one place. The
sections after it specify each stage.

## Write Path

``` text
signal (any modality)
  -> AIST embedding (256-d shared space) + per-position features
  -> per-source accumulator: running centroid, drift, coherence, surprisal
  -> online boundary segmentation (Bayesian change-point posterior)
  -> on a boundary: store the closed unit
       memories row (representative embedding e_rep)
       signals rows + object-store payloads (BLAKE3 content ids)
       memory_signal_vectors: one shared-space vector per stored payload
       memory_signal_li_codes: up to 16 int8 late-interaction codes per text or image payload
       memory_text_fts: FTS5 keyword index over long-term text
       standing_signals: lasting directives
       supersedes edges to similar older memories
  -> working-memory admission (bounded slot set)
```

1.  **Segmentation decides what a memory is**
    (<a href="#sec-thresholding" class="quarto-xref">Section 5</a>).
    Each source stream keeps an open unit. A calibrated boundary
    posterior over surprisal, drift, within-unit coherence, topic shift
    and inter-arrival gaps closes the unit at a natural event boundary.
    A capacity ceiling, a pressure ratio, an inactivity boost or an
    explicit `Flush()` can also close it.
2.  **Every closed unit is stored.** The write rule is
    `write_memory = force_write ∨ should_flush`, where `force_write` is
    set by Durable retention. An earlier score threshold discarded
    finished units and cost 7–11% of gold evidence turns
    (<a href="#sec-write-pacing" class="quarto-xref">Section 5.1</a>).
    Retention policies choose who closes units: `Natural` leaves it to
    the segmenter, `Durable` closes one per call, `Boundary` forces the
    edge only, and `Ephemeral` stores nothing.
3.  **A stored unit is indexed at three granularities.** The unit has
    one pooled vector. Each of its member signals keeps its own
    shared-space vector. Each member payload also keeps late-interaction
    codes. Its text enters the keyword index.

## Read Path

``` text
Ephemeral query (any modality; optional reference time T, as_of)
  -> dense unit seeds        (sparse route or exact; family collapse)
  -> member seeds            (best member cosine, max with unit cosine)
  -> keyword seeds           (FTS5 BM25, text queries)
  -> cross-modal seeds       (per-modality z-gated lists)
  -> temporal seeds          (nearest to T, when T is set)
  -> reciprocal-rank fusion  (k = 60)
  -> neighbour deferral
  -> late-interaction MaxSim rerank of the top 64
  -> cut to the output size (11 at neutral knobs)
  -> hydration: contiguity, superseded predecessors, superseding successors,
                standing block, per-blob timestamps
  -> evidence fitting to the caller's byte budget
```

Retrieval (<a href="#sec-retrieval" class="quarto-xref">Section 6</a>)
is multi-granularity and hybrid. Dense seeds score whole units. Member
seeds rescue a unit whose pooled vector sits between its members. BM25
catches rare names and numbers. Cross-modal lists correct for the
modality gap, and the temporal channel answers “around then” queries.
Hydration (<a href="#sec-hydration" class="quarto-xref">Section 7</a>)
then rebuilds what a person would recall around each hit: the adjacent
turns, the earlier version of an updated fact, the newer restatement,
and the standing directives. Finally it fits the packet to the reader’s
budget.

## Where the Three Knobs Still Act

Focus, Sensitivity and Stability (F, S, T) began as the core of the
architecture. Every rate, threshold and decay was derived from them, and
a homeostatic threshold controller gated persistence. In v5 they are a
macro-control surface over a calibrated default profile. The defaults (F
= S = T = 0.5) are the evaluated operating point. The knobs still change
behavior through closed-form maps:

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr>
<th>Stage</th>
<th>Knob-derived quantities</th>
</tr>
</thead>
<tbody>
<tr>
<td>segmentation</td>
<td>boundary hazard, change-point threshold <code>b_thresh(F,S)</code>,
capacity ceiling <code>max_signals(F,S,T)</code>, pressure ratio,
inactivity policy</td>
</tr>
<tr>
<td>composite score</td>
<td>the 12 metric maps and the RLS bootstrap coefficients and forgetting
factor (<a href="#sec-structural-metrics"
class="quarto-xref">Section 9</a>)</td>
</tr>
<tr>
<td>working memory</td>
<td>capacity 15–27 slots (21 at neutral), admission weights, chunk
budget</td>
</tr>
<tr>
<td>retrieval breadth</td>
<td>seed count (37 at neutral), seed search breadth (345), output size
(11)</td>
</tr>
<tr>
<td>supersession</td>
<td>similarity band, duplicate threshold, edge cap</td>
</tr>
<tr>
<td>streaming gate</td>
<td>pacing thresholds and interrupt-gate thresholds (<a
href="#sec-interrupt" class="quarto-xref">Section 11</a>)</td>
</tr>
<tr>
<td>sparse route</td>
<td>capacities <span class="math inline"><em>C</em></span>, <span
class="math inline"><em>B</em></span>, <span
class="math inline"><em>A</em></span> (<a href="#sec-optimization"
class="quarto-xref">Section 17</a>)</td>
</tr>
<tr>
<td>maintenance hints</td>
<td>throughput trough fraction and escalation (<a
href="#sec-consolidation" class="quarto-xref">Section 12</a>)</td>
</tr>
</tbody>
</table>

The v5 recall machinery is not knob-derived. Its constants were set by
ablation and hill-climbing on long-horizon evaluations:

-   RRF k = 60 and the channel limits;
-   the late-interaction depth of 64;
-   the contiguity, supersession and standing budgets;
-   the 480-byte media charge.

A reader should not expect turning a knob to retune them.

The composite-score stack still runs per signal and still changes
packets. It comprises the emotion projection, 12 metrics and
recursive-least-squares blending. With the RLS blender off, 1,980 of
1,986 LoCoMo retrieved sets differ (Jaccard 0.29). With the composite
score off, all 1,986 differ (Jaccard 0.08). The score has three
consumers, and none of them gates whether a unit is stored:

-   **The stored embedding.** The accumulator centroid at the unit’s
    highest score, `e_peak`, is blended into the representative
    embedding
    (<a href="#sec-rep-embedding" class="quarto-xref">Section 5.1.7</a>).
    It is also the fallback topic anchor for boundary detection.
-   **Working-memory admission.** The window score `S_window` ranks
    closed units for admission
    (<a href="#sec-wm-gating" class="quarto-xref">Section 10.2.2</a>).
-   **Uncertainty.** The recent-score variance is a term of the
    uncertainty estimate, which sets adaptation rates
    (<a href="#sec-math" class="quarto-xref">Section 4</a>).

The homeostatic threshold controller, the mood integrator and the spike
bypass that once sat on top of this stack were removed. None of them
reached a packet.

## What Was Removed

The 2026-10 leave-one-out ablation
(<a href="#sec-negative" class="quarto-xref">Section 14</a>) removed the
bio-inspired lifecycle mechanisms that earlier versions of this paper
presented as the architecture’s core:

-   multi-timescale strength traces, decay and strength-based eviction;
-   reconsolidation drift and retrieval-time reconstruction;
-   retrieval-induced forgetting (lateral inhibition);
-   clustering consolidation into `ASSOCIATION` centroids, and graph
    expansion;
-   synaptic tagging and capture;
-   emotional consolidation and the emotional cascade;
-   the NE-like and DA-like neuromodulators, effect scales and the
    encode/retrieve oscillator;
-   meta-learned knob priors;
-   predictive pre-activation;
-   pattern-separation sparse keys and the procedural lane;
-   Soft Anchors;
-   serial-position effects;
-   the focus, sensitivity, stability and influence feedback loops.

A follow-up prune removed what had become unread once every closed unit
was stored. Every removal left packets byte-identical:

-   the dynamic threshold controller, its hysteresis and 19 knobs;
-   the spike bypass;
-   the 6D mood integrator;
-   the stored source-reliability metadata and the per-memory context
    column;
-   the dead `affect_retrieval`, `reinforcement_enabled` and
    `sequential_edges_enabled` configuration flags;
-   ten always-default `Context::Memory` metric fields.

Each was either inert at the default knobs, leaving packets
byte-identical, or measurably harmful once made to fire. Later sections
keep a short record of each removed mechanism where it used to live, so
a reader can see what was tried. None of them describes current
behavior.

# Mathematical Foundations

## Notation and Primitives

We establish the following notation used throughout this paper. Let ε(F,
S, T) = 10^(−8 + 2T) denote a small stability-dependent constant for
numerical stability (we write ε as shorthand for ε(F, S, T)). All knob
values F, S, T lie in the closed interval \[0, 1\].

Core mathematical primitives:

    lerp(a, b, x) = a + (b − a) × x
    clamp(v, lo, hi) = max(lo, min(v, hi))
    sigmoid(z) = 1 / (1 + exp(−z))
    EWMA(prev, x, α) = (1 − α) × prev + α × x

Weight normalization and blending for combining multiple signals:

    normalize(w) = w / max(sum(w), ε) # w is a weight vector

Edge case: if sum(w) \< ε, return uniform weights (1/|w|) to avoid
division by zero or invalid probability distributions.

    blend(values, weights) = Σᵢ values[i] × weights[i]

Note: normalize() operates on weight vectors, not scalars. When used
with scalar expressions like lerp(), collect the scalars into a vector
first: normalize(\[lerp(…), lerp(…), …\]). The blend() function assumes
pre-normalized weights summing to 1.

For vectors, we define cosine similarity as cos(u, v) = u·v / (‖u‖ ×
‖v‖), and safe L2 normalization as l2_normalize(v) = v / max(‖v‖, ε).
Shannon entropy is computed in nats: H(p) = −Σᵢ pᵢ ln(pᵢ).

The temporal decay function follows exponential dynamics with
configurable half-life:

    decay(x, τ_half, Δt) = x × exp(−ln(2) × Δt / max(τ_half, τ_min(T)))

where τ_min(T) = lerp(60, 300, T) seconds provides a stability-derived
floor to prevent numerical instability from near-zero half-lives.

### Units Convention

To ensure consistency and avoid unit mismatch errors, the following
conventions apply throughout:

**Input timestamps:** All input timestamps (t) are specified in
milliseconds since epoch, consistent with standard system time APIs.

**Internal time representation:** All internal time calculations operate
in seconds. We define:

    now_s() = system_time_ms / 1000  # returns seconds
    now_ms() = system_time_ms         # returns milliseconds
    to_s(ts_ms) = ts_ms / 1000       # ms → seconds conversion

**Stored timestamps:** All stored timestamps are milliseconds since Unix
epoch.

**Naming contract (canonical):** Use the variable names below
consistently throughout this document. Units: stored timestamps are
integers in milliseconds (commonly suffixed \*\_ts, and also appearing
as timestamp/created_at/last_rate_timestamp); derived time intervals in
seconds use the **s suffix. Accumulator variables: t_start,
last_signal_ts, drift_acc, eta_acc, coherence_prev, emo_max,
arousal_sum, drift_accum, drift_at_last_interrupt, drift_acc_pacing,
x_last_check. Global variables: signals_processed, u_uncertainty,
m_rate, dt_ema, last_rate_timestamp, last_retrieval_ts, retention_ema,
consolidation_rate_floor, consolidation_rate_peak, last_embedding,
x_pred_ema. Weight naming rule: weight** and **weight variables (e.g.,
weight_relevance, mismatch_weight, weight_surprise) are control
parameters; w** variables (e.g., w_relevance, w_mismatch, … w_arousal)
are composite-score blender weights.

**Drift naming note:** `drift_step_t` (also called *drift_increment*) is
the instantaneous cosine distance between consecutive accumulator
centroids μ_acc(t) and μ_acc(t−1). `drift_acc` accumulates drift_step_t
within the current memory unit for boundary decisions; `drift_accum`
tracks cumulative centroid drift used for interrupt refractory pressure.
Treat these as distinct signals.

**Time-index convention:** Per-step computed values use a `_t` suffix
(e.g., score_t, u_t). Retained state uses the bare name (e.g.,
u_uncertainty). When a step proposes a new value, compute the `_t`
variant and assign the bare name at end of step (see Appendix D).

**Time deltas:** All Δt values, elapsed times (mem_elapsed, signal_gap,
idle_for), and time comparisons operate in seconds:

    Δt ← now_s() − to_s(last_rate_timestamp)   # seconds
    mem_elapsed ← now_s() − to_s(t_start)     # seconds
    signal_gap ← now_s() − to_s(last_signal_ts)  # seconds

**Time constants:** All time-related constants are specified with
explicit units (e.g., τ_min(T) in seconds). Gap scaling returns values
in seconds (gap_scale multiplies dt_ema to form gap_ref_s).

### Knob-Derived and Fixed Constants

The segmentation, working-memory, retrieval-breadth, supersession,
streaming and route-capacity parameters are closed-form functions of the
three knobs (F, S, T) and of stream state. The v5 recall and hydration
mechanisms are not. Their constants were fixed by ablation and
hill-climbing on long-horizon evaluations
(<a href="#sec-retrieval" class="quarto-xref">Section 6</a>,
<a href="#sec-hydration" class="quarto-xref">Section 7</a>):

-   RRF k = 60 and the per-channel limits;
-   the late-interaction depth of 64;
-   the contiguity, supersession and standing budgets;
-   the 480-byte media charge.

Numerical floors such as ε(F, S, T), unit conversions and schema
invariants are fixed as well. Earlier versions of this paper claimed
that no behavioral constant was fixed; that is no longer true, and the
fixed constants are labeled where they appear.

## Temporal Semantics

Cortext treats durable memories as **event-time observations**: a memory
is written when a coherent unit is detected, and its stored span
`[start_ts, end_ts]` records when that unit was observed. Each stored
signal keeps its own timestamp, and retrieved memories return one
timestamp per content blob
(<a href="#sec-hydration" class="quarto-xref">Section 7</a>). The engine
maintains no separate structured fact layer with valid-time and
transaction-time intervals. Temporal reasoning therefore operates over
memory and signal timestamps directly.

For an episodic memory `m_i`, the event timestamp is:

    t_event(m_i) = created_at(m_i)

Retrieval consumes event time in three places: the write-exclusion bound
(a query never retrieves its own open unit), the temporal seed channel
and `as_of` cutoff for queries that carry a reference time
(<a href="#sec-retrieval" class="quarto-xref">Section 6</a>), and the
contiguity gap that decides which adjacent turns belong to the same
stretch of conversation
(<a href="#sec-hydration" class="quarto-xref">Section 7</a>). Engine
time never runs behind the signals the store holds: if a supplied clock
is behind, the implicit “now” is the newest stored signal plus one
millisecond. The accumulator’s temporal-context vector serves boundary
topic-shift detection only; as noted in
<a href="#sec-temporal-context" class="quarto-xref">Section 10.2</a>, it
is not a retrieval-score term. It lives in memory only; migration 56
dropped the per-memory context column.

    t_retrieve(q, m_i) = q.timestamp - t_event(m_i)

This keeps the live store aligned with the episodic memory stream and
avoids a second assertion lifecycle that must be extracted, corrected,
superseded, and retrieved separately.

### Buffers

The specification uses distinct streaming buffers:

**signal_stream:** The per-step accumulator centroid stream (μ_acc after
incorporating the current signal). This stream is used for scoring,
uncertainty estimation, and threshold/rate updates; raw signal
embeddings are tracked separately inside the accumulator for
persistence.

**score_stream:** The per-signal scalar score stream. At each step,
append score_t to score_stream. Any score lookbacks operate on
score_stream.

**memory_stream:** The stream of written memory representatives (e_rep
for completed units) used for retrieval and focus-spread computations.

**recent_memory_centroids:** A bounded deque of recent memory centroids
(μ_acc) used by the interrupt gate.

    win_mem_ctx(T) = round(lerp(4, 32, T))  # max memories in interrupt context
    On successful write_memory: recent_memory_centroids.append(l2_normalize(μ_acc));
    recent_memory_centroids ← tail(recent_memory_centroids, win_mem_ctx(T))

## The Three-Knob Philosophy

Cortext exposes three continuous control parameters. In v5 they are a
macro-control surface over a calibrated default profile (F = S = T =
0.5, the evaluated operating point), not the mechanism that decides what
is stored or recalled
(<a href="#sec-overview" class="quarto-xref">Section 3</a>).

**Focus (F ∈ \[0, 1\]):** selectivity. Higher Focus narrows retrieval
breadth (fewer seeds, a smaller output), raises the change-point
threshold while lowering the unit capacity ceiling, enlarges working
memory, and tightens duplicate suppression.

**Sensitivity (S ∈ \[0, 1\]):** reactivity. Higher Sensitivity raises
the boundary hazard (shorter units), raises affect gain in the emotion
projection and the interrupt gate, shrinks working memory (faster
turnover), and makes streaming retrieval checks more frequent.

**Stability (T ∈ \[0, 1\]):** inertia. Higher Stability widens context
windows, slows the adaptive score statistics and the RLS blender,
enlarges the unit capacity ceiling and the sparse-route capacities, and
requires a deeper throughput trough before recommending maintenance.

The design principle that knobs set rates rather than modes still holds.
Behavior changes continuously with the knobs, and there are no discrete
operational modes. Earlier versions also tied Stability to memory
half-lives and eviction. Those were removed
(<a href="#sec-negative" class="quarto-xref">Section 14</a>).

### Midpoint-Biased Knob Mapping

To make **F = S = T = 0.5** a neutral, all‑arounder operating point, a
midpoint bias is applied to Focus and Sensitivity before parameters are
derived. `BiasMid` is a clamped quadratic remap,
*x*′ = clamp(*x* − 4*b* *x*(1 − *x*), 0, 1). It fixes both endpoints and
moves the midpoint by −*b*. It is applied at about 300 call sites. It is
not an affine shift, so no change of a formula’s intercept reproduces
it, and it was not folded into the constants:

    F̃ = clamp(F − β_F × 4F(1 − F), 0, 1)
    S̃ = clamp(S − β_S × 4S(1 − S), 0, 1)
    β_F = β_S = 0.10

Unless explicitly stated otherwise, knob‑derived formulas in the paper
use **F̃** and **S̃** in place of the raw UI knobs. Stability (T) is not
biased by this remap. Two further families use the same remap with other
biases. Affective gain uses a lighter Sensitivity bias. Retrieval policy
uses `RetrievalFocusBias` (*b* = −0.175), `RetrievalSensitivityBias`
(*b* = 0.325) and `RetrievalStabilityBias` (*b* = −0.25). A small number
of legacy/cold-start implementation priors still consume raw UI knobs
directly and are called out where they matter.

## Knob-Derived Parameters

Most system tunables derive from the three primary knobs. This section
catalogs the key derivations; fixed invariants are explicitly labeled in
their respective sections.

### Context Windows and Temporal Scales

    n_ctx(T) = round(lerp(32, 256, T))
    win_score(T) = round(lerp(20, 120, T))
    win_rate_s(T) = round(lerp(60, 300, T))

The context window n_ctx determines how many recent items inform
relevance computation. The scoring window win_score controls the
lookback for the score variance in uncertainty. The rate window
win_rate_s is the update factor of the maintenance throughput floor
(<a href="#sec-consolidation" class="quarto-xref">Section 12</a>).

### Hysteresis and Rate Targets (Removed)

Earlier builds derived a hysteresis band `lerp(0.02, 0.25, T)` and a
write-rate target `lerp(0.2, 5.0, S)` writes per minute for the dynamic
threshold controller. Both were removed with the controller
(<a href="#sec-spike-threshold" class="quarto-xref">Section 5.2</a>).
Other fixed values appear as schema invariants, vector dimensions,
parser guardrails, numeric epsilons, migration defaults, benchmark
protocol settings and the v5 recall constants listed above. None of
these is an additional user-facing knob.

### Retrieval Breadth

Retrieval uses two F/S/T-derived breadths rather than one global top-k.
The candidate discovery pass stays broad so durable memories remain
reachable. Retrieval uses its own midpoint biases (`RetrievalFocusBias`,
and likewise for S and T), shifted so that neutral knobs land on the
operating point a judged sweep selected:

    seed_k(F) = round(lerp(96, 8, RetrievalFocusBias(F)))   # 37 at neutral

Before the vector-seed cap is applied, near-duplicate embedding families
are collapsed to one current representative. Superseded targets stay
eligible: excluding them was removed after the engine ablation
(<a href="#sec-experimental" class="quarto-xref">Section 15</a>). Family
lookup normalizes each embedding and partitions its coordinates into 32
deterministic blocks. For a candidate `x` and representative `y`,
Cauchy–Schwarz gives the recall-safe upper bound

    cos(x, y) <= sum_b ||x_b||_2 ||y_b||_2

When that bound is below the cosine family threshold, the pair cannot be
a duplicate and the full dot product is skipped. All remaining pairs
receive an exact cosine check, so the screen cannot introduce a
false-negative partition while avoiding broad exact comparisons on
diverse vectors. The prompt-facing output is then narrowed after vector
relevance and bounded graph expansion:

    selected_k(F,T) = round(lerp(20, 8, RetrievalFocusBias(F))
                            * lerp(1.08, 0.92, RetrievalStabilityBias(T)))   # 11 at neutral

Dense vector relevance is the first-stage score:

    score(m) = map01(cos(q, embedding(m)))

For queries, member, keyword, cross-modal and temporal channels then
join this list by reciprocal-rank fusion, and a late-interaction model
reranks the head of the pool
(<a href="#sec-retrieval" class="quarto-xref">Section 6</a>).

Earlier builds added an event-time recency tie-break,
`(1 − semantic) × 10^-6 × temporal_rank(m)`. The round-3 ablation
(2026-10) dropped it with every packet byte-identical (LongMemEval-S
0/48, Eval 1 0/250, BEAM-10M conversation 7 0/20), so it was removed.

Earlier builds added a bounded graph bonus from one-hop association
expansion. It was removed with clustering consolidation
(<a href="#sec-experimental" class="quarto-xref">Section 15</a>),
because no traversable edges remained.

This split lets low Focus explore many durable-memory seeds while
keeping the application prompt bounded by a compact final surface. The
intermediate seed search breadth (345 at neutral) is likewise derived
from Focus, Sensitivity, and Stability rather than exposed as an
additional user setting. Neither `source_id` nor modality participates
in candidate discovery, family collapse, scoring, or final selection.
Earlier builds also stored a knob-derived source-reliability prior and a
contradiction count per memory. Nothing read them, and migration 56
dropped both.

### Experiential Mass and Maturity

The system tracks accumulated experience through a maturity function:

    τ_m(T) = lerp(10.0, 200.0, T)
    maturity(t) = 1 − exp(−count / τ_m(T))

where count is the total number of signals processed. Higher Stability
slows the approach to maturity. Maturity is now the cold-start fallback
for uncertainty (below). It also used to anneal the safety bounds of the
removed dynamic threshold.

## Uncertainty Estimation

Uncertainty u(t) ∈ \[0, 1\] modulates learning rates and evidence
weighting. The raw uncertainty estimate blends multiple signals:

    var_score_max(S) = lerp(0.15, 0.35, S)
    recent_scores ← tail(score_stream, win_score(T))
    if |recent_scores| < 2:
        var_recent_norm ← 0
    else:
        var_recent_norm ← clamp(var(recent_scores) / var_score_max, 0, 1)
    coherence_complement = 1 − coherence_struct_t  # uses structural coherence

When prediction error signals are available, novelty and surprisal are
blended:

    novelty_surprise = blend([novelty_t, surprisal_t],
                            weights = normalize([S, 1 − T]))

The final raw uncertainty combines these components with knob-derived
weights:

Normative note (MUST): focus_spread_t is the per-step derived
focus-spread metric from
<a href="#sec-focus-spread" class="quarto-xref">Section 9.1.2</a> and is
available each step when computing u(t).

    weights_u = normalize([S, F, 1 − T, S × (1 − T)])
    u_raw(t) = clamp(blend([var_recent_norm, focus_spread_t,
                            coherence_complement, novelty_surprise],
                           weights = weights_u), 0, 1)

Smoothed uncertainty applies EWMA with a stability-dependent rate:

    α_u(T) = 0.10 + (1 − T) × 0.60
    u(t) = EWMA(u(t−1), u_raw(t), α = α_u(T))

When structural metrics are unavailable, the fallback is u_raw(t) = 1 −
maturity(t), ensuring high uncertainty during early operation.

# Online Segmentation and the Write Path

The write path decides what a memory is. Each source stream keeps one
open unit. A calibrated change-point posterior closes the unit at a
natural event boundary, and every closed unit is stored. No language
model segments, summarizes or filters the stream. The inputs are
embedding statistics defined in
<a href="#sec-structural-metrics" class="quarto-xref">Section 9</a>:
surprisal against an EMA predictor, instantaneous drift, within-unit
coherence, topic shift against the temporal context, and inter-arrival
gaps.

The section specifies segmentation and the write commit. It ends with a
short record of the removed dynamic threshold θ_dynamic, which once
gated persistence.

**Knob note:** F and S here refer to midpoint‑biased values (F̃, S̃) from
<a href="#sec-math" class="quarto-xref">Section 4</a>.

## Memory Accumulation and Event Segmentation

The write gate operates per step, but coherent “thoughts” often span
multiple signals. Composite scoring is computed on the current
accumulator centroid (μ_acc), so write decisions are driven by the
accumulated group rather than any single raw signal. This section
introduces memory-level accumulation that groups signals into natural
units before storage decisions, inspired by Event Segmentation Theory
(Zacks and Swallow 2007).

This approach draws from EM-LLM (Fountas et al. 2024), which segments
token sequences into episodic events using surprise-based boundary
detection refined by graph-theoretic cohesion metrics. Their work shows
that combining prediction error signals with within-segment coherence
produces boundaries correlated with human event perception. Our
adaptation uses EMA prediction error (`surprisal_t`) for surprise and
cosine similarity for cohesion, while drift provides auxiliary boundary
pressure, enabling modality-agnostic operation across text tokens, audio
chunks, video frames, or any signal stream.

### Memory Accumulator State

Each source stream maintains accumulator state:

    μ_acc ← 256d running mean embedding
    c_t ← temporal context vector (slow drift; @sec-temporal-context)
    drift_acc ← accumulated drift within memory
    s_sum ← sum of accumulator scores in memory
    s_max ← max accumulator score in memory
    n ← count of signals in memory
    e_peak ← accumulator centroid at highest score
    emo_max ← 0  # max emotion_intensity_t within the unit
    arousal_sum ← 0  # sum of arousal_t within the unit (for avg)
    t_start ← now_ms()  # timestamp of accumulation start (ms since epoch)
    last_signal_ts ← now_ms()  # timestamp of previous signal (ms, for gap detection)
    eta_acc ← 0  # drift EWMA state
    coherence_prev ← 0  # previous coherence (initialize to 0)
    acc_signals_window ← []  # ring buffer of recent embeddings for coherence

Accumulator state is live runtime staging state, not durable memory.
`Flush()` closes every open unit as at a boundary; if the process
restarts before a boundary or flush commits a unit, the unfinished
accumulator window is lost. Committed memories, signal rows, and working
memory remain the durability boundary.

Reset behavior: reset_accumulator() clears μ_acc, drift_acc, s_sum,
s_max, n, e_peak, emo_max, arousal_sum, eta_acc, coherence_prev, and
sets acc_signals_window ← \[\] (and refreshes t_start/last_signal_ts for
the next unit).

    reset_accumulator(): acc_signals_window ← []  # MUST clear coherence window at boundaries

On each signal, update running statistics (note: n is the count before
this signal):

    signal_gap_s ← now_s() − to_s(last_signal_ts)  # compute BEFORE updating last_signal_ts

    win_coh(T) = round(lerp(8, 32, T))  # coherence window size
    acc_signals_window ← tail(acc_signals_window, win_coh(T))

    μ_acc ← (n × μ_acc + x_t) / (n + 1)
    n ← n + 1
    drift_gain(S,T) = clamp(0.5 × scale(S,T), 0.35, 0.65)
    drift_acc ← drift_acc + drift_mag_t × drift_gain(S,T)
    emo_max ← max(emo_max, emotion_intensity_t)
    arousal_sum ← arousal_sum + arousal_t
    last_signal_ts ← now_ms()  # update timestamp for next gap calculation (ms)

    # After composite scoring on μ_acc:
    s_sum ← s_sum + score_t
    if score_t > s_max:
        s_max ← score_t; e_peak ← μ_acc

### Hybrid Drift and Coherence Tracking

Boundary detection combines instantaneous drift with within-memory
coherence tied to the evolving centroid μ_acc:

    ε_noise(T) = lerp(0.01, 0.05, 1 − T)  # stability-derived noise floor
    # Use instantaneous drift_step_t for boundary velocity
    d_step ← max(drift_step_t − ε_noise(T), 0)
    eta_prev ← eta_acc  # baseline before updating EWMA

Memory coherence tracks similarity within the current memory
accumulation window (range \[−1, 1\] from mean cosine):

    mu_curr ← μ_acc  # current accumulator centroid
    current_window ← acc_signals_window  # embeddings from the current unit before mu_curr
    if |current_window| == 0:
        coherence_curr ← 1.0  # empty-window fallback
    else:
        coherence_curr ← mean([cos(mu_curr, x_i) for x_i in current_window])
    # After computing coherence_curr, append mu_curr for the next step
    acc_signals_window.append(mu_curr); acc_signals_window ← tail(acc_signals_window, win_coh(T))
    # coherence_prev is the stored coherence value from the previous step

Note: coherence_mem is distinct from coherence_struct
(<a href="#sec-struct-coherence" class="quarto-xref">Section 9.1.1</a>).
The former tracks within-memory similarity using raw mean cosine, while
the latter measures variance-based integration with broader context.
This dual-signal approach mirrors EM-LLM’s boundary detection mechanism.
In their formulation, boundaries occur where surprise (token-level
prediction error) exceeds a threshold and segment cohesion drops. In
Cortext, surprisal_t provides the prediction-error signal, while drift
spike provides auxiliary boundary pressure and coherence_mem drop
captures within-memory similarity degradation.

### Natural Boundary Detection (Bayesian Change-Point)

Segmentation is framed as online change-point inference. We compute a
**calibrated boundary probability** rather than a hand-tuned score. The
hazard is knob-controlled; the likelihood is driven by surprisal,
cohesion drop, drift, and temporal gaps, with a coherence/topic
**support gate** to downweight isolated spikes.

    # Hazard (prior boundary probability, calibrated to target rate)
    h_base(F,S,T) = lerp(0.03, 0.18, S) × lerp(1.2, 0.8, T)
    h_t_base ← clamp(h_base × (0.8 + 0.2(1 − F)), 0, 0.5)
    target_rate(F,S,T) = clamp(lerp(0.12, 0.35, S) × lerp(1.1, 0.7, F) × lerp(1.0, 0.7, T), 0.05, 0.45)
    rate_gain(S,T) = clamp(lerp(0.4, 1.0, S) × lerp(1.1, 0.8, T), 0.2, 1.2)
    rate_mult ← clamp(1 + rate_gain × (target_rate − boundary_rate_ema), 0.5, 1.5)
    # support uses normalized coherence/topic signals (defined below)
    h_t ← clamp(h_t_base × rate_mult × lerp(0.8, 1.3, support), 0, 0.5)

    # Drift spike (same signal, now used for likelihood)
    ε0(T) = lerp(0.005, 0.02, 1 − T)  # cold-start guard for eta_prev
    if eta_prev < ε0(T):
        drift_spike ← 0
    else:
        drift_spike ← (d_step − eta_prev) / max(eta_prev, ε)
    eta_acc ← EWMA(eta_prev, d_step, α = lerp(0.3, 0.1, T))
    coh_drop01 ← clamp((coherence_prev − coherence_curr) / 2, 0, 1)
    coherence_prev ← coherence_curr  # update for next step

    # Adaptive gap signal (dynamic, not a hard trigger)
    dt_ref ← max(dt_ema, dt_floor(T))  # EWMA of inter-arrival Δt in seconds (robust to jitter)
    gap_scale(T) = lerp(3.0, 8.0, T)  # expected pause multiplier
    gap_ref_s ← gap_scale(T) × dt_ref
    gap_z ← (signal_gap_s − gap_ref_s) / max(gap_ref_s, ε)
    gap_score ← sigmoid(gap_z)

    # Local normalization (within-episode, contrast-boosted)
    var_floor(S,T) = BoundaryVarianceFloor(S,T)
    warmup_signals(S,T) = BoundaryWarmupSignals(S,T)
    ema_mean_x ← EWMA(ema_mean_x, x, α_local(T))
    ema_var_x  ← EWMA(ema_var_x, (x − ema_mean_x)^2, α_local(T))
    norm_gain(S,T) = clamp(lerp(1.6, 3.0, S) × lerp(1.05, 0.95, T), 1.0, 3.2)
    x̂ ← sigmoid(norm_gain × (x − ema_mean_x) / sqrt(max(ema_var_x, var_floor) + ε))
    # warm-up uses raw x until warmup_signals(S,T)

    # Topic shift (embedding-only, temporal context anchor)
    topic_shift ← 1 − map01(cos(e_t, c_t))

    # Likelihood of a boundary given observations
    support_policy ← BoundarySupportPolicyForKnobs(F,S,T, coĥ, topiĉ)
    support ← support_policy.support
    support_gate ← support_policy.support_gate   # downweight isolated spikes
    gap_gate ← support_policy.gap_gate
    support_boost ← support_policy.support_boost  # emphasize coherent/topic shifts
    w_g(F,S,T) = clamp(0.01 × lerp(1.10, 0.90, F)
                             × lerp(0.90, 1.10, S)
                             × lerp(0.90, 1.10, T), 0.004, 0.020)
    [w_s, w_d, w_c, w_t, w_g] ← [0.34 + 0.14S, 0.18 + 0.05(1 − T),
                                0.36 + 0.16F, 0.30 + 0.22S, w_g(F,S,T)]
    [w_s, w_d, w_c, w_t, w_g] ← normalize([w_s×support_gate, w_d×support_gate,
                                          w_c×support_boost, w_t×support_boost, w_g×gap_gate])
    z_center(S,T) = clamp(lerp(0.44, 0.30, S) × lerp(1.05, 0.95, T), 0.26, 0.58)
    change_point_policy ← BoundaryChangePointPolicyForKnobs(S,T,support)
    z_center_eff ← change_point_policy.z_center_eff
    z_t ← w_s×(surprisal̂ − z_center_eff) +
          w_d×(drift̂ − z_center_eff) +
          w_c×(coĥ − z_center_eff) +
          w_t×(topiĉ − z_center_eff) +
          w_g×(gap_score − z_center_eff)
    k_cp_eff ← change_point_policy.k
    lik_boundary ← sigmoid(k_cp_eff × z_t)

    # Bayesian update for boundary posterior
    boundary_score ← (h_t × lik_boundary) /
                      max(ε, h_t × lik_boundary + (1 − h_t) × (1 − lik_boundary))
    boundary_score ← clamp(boundary_score, 0, 1)

Boundary threshold and limits:

    b_thresh(F, S) = lerp(0.48, 0.66, F) × lerp(1.1, 0.9, S)
    max_mem_drift(S) = lerp(0.8, 2.0, S)  # cumulative drift cap

Pressure-capacity ratio (continuous flush trigger derived from knobs):

    capacity_scale(T) = (1 + T)^2  # higher stability = larger capacity
    [capacity, pressure] ← BoundaryPressurePolicyForKnobs(S,T, drift_acc)
    saturation_ratio ← pressure / max(capacity, ε)
    k_flush(S,T) = k_surprise(S,T)
    pressure_score ← sigmoid((saturation_ratio − 1) × k_flush(S,T))

Hard capacity ceiling (max signals per memory; knob-derived):

    base = lerp(4.0, 12.0, T)
    f_scale = lerp(1.00, 0.35, F)
    s_scale = lerp(1.00, 0.55, S)
    floor = round(lerp(2, 3, T))
    max_signals(F,S,T) = max(floor, round(base × f_scale × s_scale))

Adaptive cadence (gap normalization only; no timeouts):

    dt_ref ← max(dt_ema, dt_floor(T))
    gap_ref_s ← gap_scale(T) × dt_ref

Fallback boundary floor (applies to capacity/pressure only):

    boundary_floor(F,S,T) = clamp(lerp(0.05, 0.15, S) × lerp(1.2, 0.8, F) × lerp(1.1, 0.9, T),
                                  0.02, 0.25)

Trigger memory flush when:

    mem_elapsed ← now_s() − to_s(t_start)
    should_flush = (boundary_score > b_thresh(F, S)) OR
                   (pressure_score > b_thresh(F, S) AND boundary_score ≥ boundary_floor(F,S,T)) OR
                   (n ≥ max_signals(F,S,T) AND boundary_score ≥ boundary_floor(F,S,T))

where signal_gap_s = now_s() − to_s(last_signal_ts) is computed at the
start of signal processing (before last_signal_ts is updated). Gap
timing influences boundary_score via gap_score only; there is no
time-based flush.

**Boundary type note:** When the max-signals ceiling triggers a flush,
set `boundary_type = "capacity"` for the enclosing episode. This is a
hard safety boundary (not a heuristic score) and is still fully
knob-derived.

### Capacity Flush and the Removed Spike Bypass

The capacity clause of `should_flush` is its own operation,
`CheckCapacityFlush`. It never runs for `Ephemeral` queries, so a probe
cannot turn an open Natural unit into a capacity boundary. When it fires
it marks the boundary type `capacity` and finalizes the episode.

Earlier builds also had a spike bypass, a “flashbulb” flush: a signal
whose composite score exceeded the dynamic threshold θ_dynamic by a
knob-derived margin closed its unit at once and forced its write. On the
parity sets the spike branch never fired, and only the capacity cap
closed units. It was removed, with the margin and its flag, and the
dynamic threshold went with it
(<a href="#sec-spike-threshold" class="quarto-xref">Section 5.2</a>).

### Inactivity Boost

We avoid hard timeouts for episode boundaries. Instead, when the stream
becomes quiet and natural indicators are weak, inactivity softly boosts
the boundary score:

    inactivity_policy ← BoundaryInactivityPolicyForKnobs(S,T)
    support_relax = exp(−inactivity_policy.support_relax_rate × gap_z⁺)
    gap_ratio = signal_gap_s / max(dt_ema, ε)
    gap_z_inact = log1p(max(gap_ratio − 1, 0))
    gap_score_inact = sigmoid(inactivity_policy.gap_score_scale × gap_z_inact)
    inactivity = gap_score_inact × (1 − support × support_relax)
    gap_z⁺ = max(0, gap_z_inact)
    inactivity_scale = exp(inactivity_policy.inactivity_exp_rate × gap_z⁺)
    boundary_score ← boundary_score + inactivity_policy.inactivity_weight × inactivity × inactivity_scale
    gap_force = sigmoid(inactivity_policy.gap_force_k ×
                       (gap_z_inact − inactivity_policy.gap_force_center))
    boundary_score ← max(boundary_score, gap_force)

This closes episodes after prolonged inactivity without forcing
premature flushes during coherent streams. The signal gap is measured
relative to the observed cadence (dt_ema), so inactivity remains dynamic
and adapts to the current conversational pace rather than a fixed
timeout.

### Window Score and Write Commit

Memory-level score combines peak and average with coverage bonus:

    s_avg ← s_sum / max(n, 1)
    α(F) = lerp(0.3, 0.7, F)  # peak vs avg weight
    coverage ← min(n / n_ctx(T), 1.0)  # memory completeness
    β(S) = lerp(0.05, 0.15, S)  # coverage weight
    S_window ← α(F) × s_max + (1 − α(F)) × s_avg + β(S) × coverage

S_window ranks the closed unit for working-memory admission
(<a href="#sec-wm-gating" class="quarto-xref">Section 10.2.2</a>). It no
longer decides storage: every unit that closes is written.

    write_memory = force_write OR should_flush   # unit closed and non-Ephemeral
    # force_write: Durable retention; should_flush includes the capacity flush

Boundary detection already decides that a unit is complete. An earlier
`S_window > θ_dynamic × M_write_refrac` threshold (with a write
refractory `M_write_refrac = 1 + k exp(−Δt_write/τ)`) discarded finished
units: under per-conversation Natural ingest it dropped 7–11% of gold
evidence turns (stored without it: LongMemEval-S 72 → 81 of 81, Eval1
371 → 399 of 400), and removing it raised retrieved-only packet gold
recall by +0.086 (S) and +0.054 (Eval1) for +6% store bytes. Dropping
only the refractory kept most of the loss (S 73 of 81, +0.012). The
threshold and refractory are removed.

The defensive rule for timestamps applies to boundary gaps and retrieval
pacing: non-increasing timestamps produce a zero elapsed interval. They
cannot create an unsigned wraparound boundary or bypass the
adjacent-retrieval limiter.

Normative rule (MUST): if should_flush is true, the current unit is
finalized, written, and reset_accumulator() runs. This prevents
perpetual should_flush states (time cap / drift cap / gap cap).

Representative embedding blends accumulator mean with peak:

### Representative Embedding

This is where the composite score reaches storage. `e_peak` is the
accumulator centroid at the unit’s highest composite score, so the score
decides which point of the unit the stored vector leans toward. `e_peak`
also stands in as the boundary topic anchor when `c_t` is unavailable.

    ρ(F) = lerp(0.2, 0.6, F)  # mean vs peak blend
    e_rep ← l2_normalize(ρ(F) × μ_acc + (1 − ρ(F)) × e_peak)

    # Edge-case guard
    if |μ_acc| = 0: e_rep ← e_peak
    if |e_peak| = 0 OR dim(e_peak) ≠ dim(μ_acc): e_rep ← μ_acc

On write: store e_rep with metadata {n, s_max, s_avg, drift_acc,
mem_elapsed, s_emotion_max=emo_max, s_arousal_avg=arousal_sum / max(n,
1), boundary_score}. `c_t` stays in memory for boundary detection and is
not persisted (migration 56 dropped the per-memory context column).
Append e_rep to memory_stream and l2_normalize(μ_acc) to
recent_memory_centroids. Reset accumulator for next unit.

## Dynamic Threshold Controller (Removed)

Earlier builds kept an adaptive write threshold θ_dynamic. It blended a
knob-derived prior with the 90th percentile of recent composite scores,
weighted by uncertainty. A homeostatic correction then steered it toward
a Sensitivity-derived write-rate target, with an effective-sample-size
reliability estimate (Liu and Chen 1998). Sensitivity, precision,
emotion and mood deltas moved it further, inside a Stability-derived
hysteresis band and maturity-annealed safety bounds.

The controller lost its consumers in two steps. First, every closed unit
is stored
(<a href="#sec-write-pacing" class="quarto-xref">Section 5.1</a>).
Second, the spike branch, its last reader, never fired. The threshold,
its hysteresis, the four adjustment terms (`UpdatePrecisionDelta`,
`InitializeStabilityPriors`), the rate target, the bias-corrected rate
and 19 knobs were then removed, with packets byte-identical (migration
58).

`UpdateThreshold` is now `UpdateSignalHistory`. It keeps the two values
other operations read:

-   the recent composite-score window, which feeds the score-variance
    term of uncertainty
    (<a href="#sec-math" class="quarto-xref">Section 4</a>);
-   the inter-signal time EWMA `dt_ema`, which feeds the boundary gap
    evidence and streaming pacing.

Switching `UpdateSignalHistory` off changes packets (retrieved-set
Jaccard 0.53 on LoCoMo), so it is live.

### Signal Cadence and Write Rate

The two surviving rate estimates are:

    Δt ← max(now_s() − to_s(last_rate_timestamp), dt_min(T))   # 1 s before any
    dt_min(T) = 10^(−4 + 2T)
    τ_dt(T) = lerp(0.5, 2.0, T)
    dt_ema ← EWMA(dt_ema, Δt, α = 1 − exp(−Δt / τ_dt(T)))      # UpdateSignalHistory

    dt_base ← max(dt_ema, dt_floor(T)),  dt_floor(T) = lerp(0.1, 0.5, T)
    τ_rate ← max(2^(3T) × dt_base, dt_floor(T))
    m_rate ← EWMA(m_rate, 60 × write_t / Δt, α = 1 − exp(−Δt / τ_rate))  # UpdateRateState

Here `dt_ema` normalizes the gap evidence of boundary detection and the
adjacent-retrieval window of streaming pacing. `m_rate` is the observed
write rate that the maintenance throughput classifier reads
(<a href="#sec-consolidation" class="quarto-xref">Section 12</a>).

# Multi-Granularity Hybrid Retrieval

A query is an `Ephemeral` signal of any modality. It is embedded like
any other signal, runs inside a transaction that is rolled back, and
leaves no trace in the store. This section specifies how its candidate
pool is built, fused and reranked.
<a href="#sec-hydration" class="quarto-xref">Section 7</a> specifies
what is attached to the selected memories afterwards.

Ingest-time retrieval, which feeds the streaming interrupt gate
(<a href="#sec-interrupt" class="quarto-xref">Section 11</a>), uses only
the dense unit channel. Lexical seeds were measured at ingest too: they
cost per-turn time without improving recall.

## Notation

-   *q*: the query, with unit embedding
    *e*<sub>*q*</sub> ∈ ℝ<sup>256</sup>, modality *μ*<sub>*q*</sub>,
    text payload (if any), and optional reference time *T*<sub>*q*</sub>
    with flag `as_of`.
-   ℳ: the long-term memories. Memory *m* has a pooled representative
    *e*<sub>*m*</sub>
    (<a href="#sec-rep-embedding" class="quarto-xref">Section 5.1.7</a>),
    a time span \[start<sub>*m*</sub>, end<sub>*m*</sub>\], and member
    signals *s* ∈ *m*.
-   Member *s* has a stored shared-space vector *v*<sub>*s*</sub>
    (`memory_signal_vectors`, migration 50, keyed by the payload’s
    content id) and a late-interaction code set *Z*<sub>*s*</sub>
    (`memory_signal_li_codes`, migration 54).
-   *τ*<sub>*x*</sub>: the exclusion time. It is the start of the
    query’s still-open write unit, so a query never retrieves itself.
    With `as_of` set,
    *τ*<sub>*x*</sub> ← min (*τ*<sub>*x*</sub>, *T*<sub>*q*</sub> + 1)
    on every channel.
-   map01 (*c*) = (*c* + 1)/2.

A memory is *seed-eligible* when it is `LONG_TERM`, has a vector surface
and started before *τ*<sub>*x*</sub>. Every channel resolves its hits
through the same eligibility test.

## Dense Unit Seeds

The base channel ranks memories by
map01 (cos (*e*<sub>*q*</sub>, *e*<sub>*m*</sub>)) over the current
long-term surface. Above a knob-derived population, candidates come from
the sparse route, and the exact SQL path runs whenever the route cannot
certify a full seed set
(<a href="#sec-optimization" class="quarto-xref">Section 17</a>).

Before the seed cap, near-duplicate embedding families collapse to one
representative. The sound block-norm screens of
<a href="#sec-optimization" class="quarto-xref">Section 17</a> keep this
from costing quadratic work. Superseded memories stay eligible:
excluding them was measured as harmful
(<a href="#sec-negative" class="quarto-xref">Section 14</a>).

The seed count is round (lerp (96, 8, *f*)), where *f* is the retrieval
Focus bias, and the seed search breadth expands it further. At neutral
knobs the engine seeds 37 memories from a search breadth of 345 and
returns 11.

## Member Seeds

A unit grouped from several signals has one pooled vector. That vector
sits between its members and is close to none of them. A query about one
member therefore ranks the unit far below where that member alone would
rank. The member channel scores each multi-signal memory by its best
member:

seed(*m*) = max (map01 (cos (*e*<sub>*q*</sub>, *e*<sub>*m*</sub>)), max<sub>*s* ∈ *m*</sub>map01 (cos (*e*<sub>*q*</sub>, *v*<sub>*s*</sub>))).

The 20 memories with the best members join the seed set, or raise the
seed score of a memory already in it. The seed list is then re-sorted.
Member vectors stay resident as int8 codes with one scale per vector
(dimension + 4 bytes per member). Single-signal memories are never held,
because their member vector is their memory vector. Each hit is resolved
to its current memory through the store before use.

Member scoring was measured as the third of three stages against the
Natural baseline. The first was query focus, which keeps a grouped
memory’s query-nearest members and was the precursor of evidence fitting
(<a href="#sec-hydration" class="quarto-xref">Section 7</a>). The second
was the per-signal standing set. On BEAM, the three stages together
raised the gold-turn share of the 8,000-token packet from 0.385 / 0.195
/ 0.125 to 0.617 / 0.420 / 0.272 at 100K / 1M / 10M. The
past.dev-protocol score, answered and judged by gpt-5.6-luna, rose from
0.793 / 0.750 / 0.651 to 0.847 / 0.813 / 0.718 (+0.054 \[0.019, 0.095\],
+0.062 \[0.044, 0.081\], +0.067 \[0.029, 0.100\]). Measured on its own,
member scoring moved the score by +0.018 \[+0.002, +0.034\] at 100K and
+0.013 \[−0.003, +0.032\] at 1M, and raised 1M gold share by +0.036
\[+0.022, +0.050\] (hill-climb record;
<a href="#sec-implementation" class="quarto-xref">Section 16</a>).

## Keyword Seeds: FTS5 BM25

Every stored long-term text memory is indexed in a contentless SQLite
FTS5 table, `memory_text_fts` (migration 49). Its rowid is the memory
id, its tokens are porter-stemmed `unicode61`, and the text itself stays
in the object store. The index row is written in the savepoint that
writes the memory, so it commits, rolls back and is deleted with it.

A text query ORs at most 16 of its non-stopword words, each as a quoted
FTS5 string. It drops any word whose index term occurs in more than
max (64, 0.05*N*) of the *N* memories. Such terms carry little BM25
weight, and their posting lists dominated query time. The 20 best
eligible hits by `bm25()` (Robertson and Zaragoza 2009) form the keyword
list, collected without sorting every match
(<a href="#sec-optimization" class="quarto-xref">Section 17</a>).

The channel targets turns that share a query’s rare words (names,
numbers) but sit far down the vector ranking. The shipped channel (FTS5
with the document-frequency cut, commit ec30c69f) was measured against
the pruned vector-only engine (47c18ec9), with Haiku judging (rubric
v2):

-   Eval 1, 250 cases: packet gold-turn recall rose from 0.597 to 0.787
    (70 cases better, 6 worse). Judged sufficiency rose by +0.132 ±
    0.065 (train) and +0.222 ± 0.060 (test).
-   LongMemEval-M, 48 haystacks of about 4.9k turns: packet recall rose
    from 0.407 to 0.667, and judged sufficiency from 0.530 to 0.821
    (+0.292 ± 0.112). Query p95 and ingest time stayed within 2%.

PR #28 reports the larger v2-against-47c18ec9 comparison (Eval 1 recall
0.597 → 0.809, sufficiency +0.149 ± 0.063 / +0.218 ± 0.059). That
comparison includes the other v2 changes as well as the keyword channel.

## Cross-Modal Seeds

AIST maps text, audio and images into one space, but cosines across
modalities sit far below cosines within one. A matching image scores
about as high as unrelated text. Raw cosine would therefore never let an
image answer a text query. The cross-modal channel ranks each stored
modality *μ* ≠ *μ*<sub>*q*</sub> in its own list and gap-corrects it
with a per-query z-score over that modality’s whole long-term population
*P*<sub>*μ*</sub> of signal vectors:

*z*<sub>*s*</sub> = (cos (*e*<sub>*q*</sub>, *v*<sub>*s*</sub>) − *c̄*<sub>*μ*</sub>)/*σ*<sub>*μ*</sub>,

where *c̄*<sub>*μ*</sub> and *σ*<sub>*μ*</sub> are the mean and standard
deviation of cos (*e*<sub>*q*</sub>, *v*<sub>*s*</sub>) over
*s* ∈ *P*<sub>*μ*</sub>. A memory takes its best member’s *z*. It is
kept when *z* ≥ 1.5 and |*P*<sub>*μ*</sub>| ≥ 3, and each modality
contributes at most 3 memories. Same-modality recall stays with the
dense channel. Populations stay resident per modality and advance
incrementally
(<a href="#sec-optimization" class="quarto-xref">Section 17</a>).

This channel shipped with the other multimodal changes: per-stream
signal filtering, mixed-modality memories that keep every payload, and a
vectorized image tower. Together they raised MemLens image-evidence
recall from 0.558 to 0.744 against an oracle ceiling of 0.867, and
lowered the misleading rate from 0.028 to 0.009, with text packets
unchanged (PR #28).

## Temporal Seeds

A query may carry a reference time *T*<sub>*q*</sub>, in the past or the
future (`Cortext::RetrievalTime`). The engine treats it as signal
metadata compared with stored times, never as text, so it works the same
for every modality. The temporal channel returns the *K* = 10 eligible
memories nearest *T*<sub>*q*</sub> by interval distance:

*d*(*m*, *T*<sub>*q*</sub>) = max (0, start<sub>*m*</sub> − *T*<sub>*q*</sub>, *T*<sub>*q*</sub> − end<sub>*m*</sub>),

which is 0 when *T*<sub>*q*</sub> falls inside
\[start<sub>*m*</sub>, end<sub>*m*</sub>\] and otherwise the distance to
the nearer edge.

One statement with three range scans, over partial indexes on long-term
`start_ts` and `end_ts` (migration 55), each limited to 2*K* + 4 rows,
finds them in *O*(log *N* + *K*). A test asserts the query plan has no
`SCAN memories`. The channel only adds candidates and rank mass, with
weight 0.5, so a memory found only by time scores 0.5/61 at best. That
ranks it after every dense seed while the dense list has at most 62
entries (1/(60 + *r*) \> 0.5/61 for *r* ≤ 62). The dense seed cap can
reach 96 at low Focus, and the late-interaction rerank after fusion can
still promote such a memory. With `as_of`, the cutoff
*τ*<sub>*x*</sub> ≤ *T*<sub>*q*</sub> + 1 also applies to attached later
turns and restatements. Unset, retrieval is byte-identical to the engine
without the channel (LoCoMo clock replay, 497 queries).

An oracle probe set *T*<sub>*q*</sub> to each LoCoMo question’s earliest
gold turn. Packet evidence hits rose from 1,767 to 1,783 of 1,979
questions (17 gained, 1 lost), and to 0.906 at weight 1.0. Query p50
rose by 0.07 to 0.17 ms (PR #31). This bounds the mechanism; it is not a
benchmark score.

Earlier builds also added an event-time recency tie-break to the dense
score. The round-3 ablation left every packet byte-identical without it,
so it was removed
(<a href="#sec-math" class="quarto-xref">Section 4</a>).

## Reciprocal-Rank Fusion

When any of the lexical, cross-modal or temporal channels returns hits,
the pool is ordered by reciprocal-rank fusion (Cormack, Clarke, and
Buettcher 2009):

RRF(*m*) = ∑<sub>ℓ</sub>*w*<sub>ℓ</sub> / (*k* + *r*<sub>ℓ</sub>(*m*)),   *k* = 60,

where *r*<sub>ℓ</sub>(*m*) is *m*’s 1-based rank in list ℓ. The dense
seed list (after member re-sorting) counts once with *w* = 1. The
keyword list and each per-modality cross-modal list count with *w* = 1,
and the temporal list with *w* = 0.5. Ties fall back to the semantic
score and then to memory id. Only the order changes; every candidate
keeps its semantic score. *k* = 60 is the constant of Cormack, Clarke,
and Buettcher (2009); experiment-hook builds can override it, release
builds cannot. Without any auxiliary hits, the pool stays in dense
order.

## Neighbour Deferral

Hydration attaches each anchor’s adjacent turns
(<a href="#sec-hydration" class="quarto-xref">Section 7</a>). A
candidate that is such a neighbour of a higher-ranked pick would spend
an anchor slot on content the packet already carries. Examples are a
reply next to its question, or the other half of an exchange. Within the
output size, such candidates are moved behind the picks, in order, so a
short list still fills. This costs one primary-key range lookup per
pick.

## Late-Interaction MaxSim Rerank

After deferral, the head of the pool is reranked by a learned
late-interaction model (Khattab and Zaharia 2020) over the frozen AIST
encoder (`AIST-87M-LI.gguf`, `src/models/aist_late_interaction.cpp`).
The constants below are read from the shipped model file.

**Representation.** Every item becomes a set of unit vectors in one
*D* = 64-dimensional space. Items here are queries and stored members,
in any modality. The set has two sources.

-   *Position features* through a per-tower linear map: text token
    hidden states (384-d, at most 256 positions) and image patch
    features (12 × 12 × 960-d). Audio has no trained map yet.
-   The item’s 256-d shared-space vector through one map shared by every
    modality.

**Memory side.** A stored payload’s raw encoder positions (384-d text,
960-d image) are clustered into *K* = 16 spherical k-means centroids (8
Lloyd iterations, deterministic linspace initialization). Only then are
the centroids mapped into the 64-d space, normalized and quantized to
int8. A payload with 16 or fewer positions keeps all of its rows, so a
blob holds up to 16 codes: at most 1,028 bytes, a 4-byte header plus
16 × 64 codes. Codes are per member, computed at write time; there is no
shared codebook. Its stored global vector is mapped and int8-rounded at
query time the same way.

**Query side.** The query keeps every mapped position, plus its mapped
global vector. Text queries are capped at 128 positions; image queries
keep all 144 patches. Each vector *q*<sub>*r*</sub> gets a softmax gate
weight *w*<sub>*r*</sub> from a learned linear gate.

**Scoring.** For member *s* with vector set
*V*<sub>*s*</sub> = *Z*<sub>*s*</sub> ∪ {*g*<sub>*s*</sub>},

*σ*(*s*) = ∑<sub>*r*</sub>*w*<sub>*r*</sub>max<sub>*v* ∈ *V*<sub>*s*</sub></sub>⟨*q*<sub>*r*</sub>, *v*⟩.

Memory *m* at first-stage pool position *ρ*<sub>*m*</sub> (0-based)
scores a soft maximum over its first 8 members plus a rank prior:

LI(*m*) = *α* ⋅ *τ*log ∑<sub>*s* ∈ *m*</sub>*e*<sup>*σ*(*s*)/*τ*</sup> + *β* ⋅ log (1 + *ρ*<sub>*m*</sub>)/log 101.

The shipped constants are *α* = 20.46,
*τ* = *e*<sup>−3.456</sup> ≈ 0.032 and *β* = −0.308. The rank prior is
how the rerank blends with the first stage. At position 63 it subtracts
0.28, which is worth about 0.014 of MaxSim similarity after the *α*
scale: a small learned preference for the fused order, not a veto.

**Depth and slots.** The rerank covers the top 64 candidates. Depth 64
keeps its cost at 0.4 to 3 ms per query. A memory none of whose members
has codes keeps its slot, and the coded memories reorder among the
remaining slots. Uncoded memories come from towers without a trained map
(audio) and from signals stored before migration 54. One global vector
cannot compete with a code set, so moving these memories would bias
against their modality. A query without position vectors keeps the
engine order.

The rerank runs after neighbour deferral on purpose. Deferring the
neighbours of the reranked picks instead lowered Eval 1 packet recall
from 0.938 to 0.892.

**Training.** The model has 102,469 float32 weights (410 KB). It is
distilled from the `ms-marco-MiniLM-L6-v2` cross-encoder with listwise
KL at temperature 2, plus a multi-positive listwise loss on gold
memories. Image members get soft labels through their BLIP captions and
hard labels through MemLens gold. Training pools come from
LongMemEval-S, the LongMemEval conversations of Eval 1, DolphinBench,
MemLens and the BEAM 1M train conversations. Held-out splits are 30% of
Eval 1 and BEAM 1M conversations, one DolphinBench persona, 20% of
LongMemEval and MemLens questions, and the whole 108-question MemLens
image evaluation set. LoCoMo is never trained on.
<a href="#sec-experimental" class="quarto-xref">Section 15</a> separates
trained-on and zero-shot benchmarks accordingly.

**Effect.** Packet gold-turn recall rose on every set (PR #28,
<a href="#sec-implementation" class="quarto-xref">Section 16</a>):

-   Eval 1: 0.846 to 0.938, and 0.878 to 0.955 on its conversations held
    out from training.
-   LongMemEval-S: 0.808 to 0.992 (14 unseen questions 0.857 to 1.000).
-   LongMemEval-M: 0.770 to 0.894.
-   MemLens: text gold 0.415 to 0.575, image gold 0.344 to 0.396.
-   BEAM at a 34,800-byte budget: 0.615 / 0.413 / 0.274 to 0.696 / 0.518
    / 0.369 at 100K / 1M / 10M. 100K and 10M were never trained on.

End-to-end BEAM scores (gpt-6.1-sol low answer and judge) rose from
0.816 / 0.778 / 0.667 to 0.843 / 0.799 / 0.699 (+0.027 \[+0.006,
+0.049\], +0.021 \[+0.005, +0.036\], +0.033 \[+0.003, +0.063\]). Query
p95 stayed within ×1.00 to ×1.14 of the engine without the rerank, from
10k to 443k memories. The codes added 11% to Eval 1 store size.

## Standing Memories

Some statements must shape answers long after they leave query
similarity: “Always include version numbers when I ask about libraries”,
“From now on, answer in French”. The engine keeps them in a small set
ranked on its own.

-   **Membership.** Membership belongs to the stored signal, not its
    memory (`standing_signals`, migration 53, replacing the memory-level
    set of migration 51). Under Natural retention a directive is one
    turn of a grouped memory several kilobytes long, which a
    memory-level check never admits.
-   **Detector.** A pattern check over the signal’s text, at most 480
    bytes: a clause opening with *always*, *never*, *from now on*,
    *going forward*, *whenever I*, *I prefer* and similar, optionally
    after *please*. This is today’s only detector.
-   **Capacity.** The set keeps the newest 256 members.
-   **Query.** Every query ranks the members written before
    *τ*<sub>*x*</sub> by RRF (*k* = 60) of two lists. One is the cosine
    of *e*<sub>*q*</sub> to the member’s own stored vector. The other is
    BM25 (*k*<sub>1</sub> = 1.2, *b* = 0.75) over the member’s 6-byte
    word stems, with document frequencies from the set itself.
    Store-wide frequencies would discount the topic words that tell
    directives apart.
-   **Block.** The best 5, within 2 KiB, form a standing block
    (`Context::standing_memory`) ahead of the retrieved memories. Each
    entry holds only the directive signals of one memory.

The block takes no retrieved slot. A consumer that cuts the retrieved
list at *k* keeps the same *k* retrieved memories. When standing items
took ranked slots instead, LongMemEval-M gold recall at *k* = 11 fell by
0.087.

On BEAM, PR #28 reports the share of instruction-following gold turns
(the instruction and its acknowledgement) for the standing block
(b5447e25) against the engine without it. The share rose from 0.44 /
0.26 / 0.00 to 0.86 / 0.81 / 0.44 at 100K / 1M / 10M.
<a href="#sec-implementation" class="quarto-xref">Section 16</a>
reports, for the same channel, an overall gold-turn share gain of 0.070
/ 0.075 / 0.053, no other category losing more than 0.01, and query p95
within 0.98 / 0.96 / 1.14 of the engine without it.

## Ranking Invariants

-   `source_id` and modality are never ranking inputs. Modality selects
    only which tower map and which cross-modal populations apply.
-   Only the open write unit, and with `as_of` everything that starts
    after *T*<sub>*q*</sub>, is excluded. `as_of` does not filter
    working memory, or the content of a retrieved memory that spans
    *T*<sub>*q*</sub> (PR #31).
-   Every auxiliary channel adds candidates and rank mass and removes
    none.
-   Every stage ends in a deterministic tie order (score, then memory
    id).

# Hydration and Evidence Fitting

Retrieval (<a href="#sec-retrieval" class="quarto-xref">Section 6</a>)
selects a ranked set of memories, 11 at neutral knobs. Hydration then
turns each selection into evidence. It loads the memory’s payloads and
attaches the context a person would recall around the hit. Finally it
fits the whole packet to the caller’s budget. Hydration never changes
the ranking, the pool, or the number of retrieved memories, and it runs
only after selection, so its cost follows the packet rather than the
store.

## Packet Structure

A query’s `Context` carries three blocks:

-   `standing_memory`: the standing block
    (<a href="#sec-retrieval" class="quarto-xref">Section 6</a>). It
    holds at most 5 directive signals within 2 KiB and is not counted in
    the retrieved top-k.
-   `retrieved_memory`: the ranked memories. Each has its own payloads
    plus attached blobs (adjacent turns, superseded predecessors,
    superseding successors).
-   `working_memory`: the active slots in insertion order
    (<a href="#sec-advanced" class="quarto-xref">Section 10</a>).

Every retrieved memory exposes `content` (one entry per blob),
`content_mimetypes` and `content_timestamps`, index-aligned. A memory
that holds signals of several modalities has modality `"mixed"` and
returns every payload with its own MIME type. Earlier builds took a
majority vote, which dropped minority-modality payloads.

## Contiguity Hydration

Recalling one moment of a conversation brings back what was said around
it. This is the temporal-contiguity effect of episodic recall, and it
fades with distance. A retrieved turn is rarely self-contained: a
question’s answer, the statement an acknowledgement refers to, or the
user message a reply was written for usually sits one or two turns away.
For each of the top 11 anchors, the engine attaches adjacent turns as
extra blobs of the anchor:

``` text
neighbours(anchor) = adjacent LONG_TERM rows in write order
                     within 16 ids and 30 minutes of the anchor,
                     same source stream, any modality,
                     not hidden by a correction from the same stretch
pass 1 (lag 1): for anchors in rank order: successor, then predecessor
pass 2 (lag 2): with what budget is left, the turns two steps away
charge: text clipped to 480 B; other media 480 B each
budget: 12 KiB per context
```

Forward association is stronger than backward, so the successor (usually
the reply) is attached first.

-   **Clip window.** Text is clipped on a UTF-8 boundary. The first 64
    bytes (the turn header: time, speaker) stay, followed by the
    480-byte window that covers the most distinct query terms.
-   **Cost.** All lag-1 probes run in one statement and all lag-2 probes
    in another. Each probe is a primary-key range lookup plus one
    indexed supersession check.
-   **Interaction with retrieval.** Neighbour deferral
    (<a href="#sec-retrieval" class="quarto-xref">Section 6</a>) keeps
    retrieval from spending anchor slots on turns this step will attach
    anyway.

**Effect.** Contiguity is the largest single hydration effect in the
ablation record. Switching it off costs 0.06 to 0.09 packet gold-turn
recall (<a href="#sec-experimental" class="quarto-xref">Section 15</a>).
The round-3 ablation found the earlier 6 KiB budget and lag-2 cap (top 6
anchors) binding. Raising the budget to 12 KiB and giving every anchor
lag-2 turns raised packet recall from 0.642 to 0.691 on LongMemEval-S
and from 0.597 to 0.629 on Eval 1, with 13 cases better and none worse.
Haiku-judged sufficiency rose by +0.048 ± 0.056 and +0.037 ± 0.019, for
+2.2 KB per Eval 1 packet. Budgets of 16 and 24 KiB gave packets
identical to 12 KiB. Clipping neighbours to 240 bytes raised a
clipped-text recall proxy but not judged sufficiency, so the clip stays
at 480 bytes.

## Superseded-Predecessor and Superseding-Successor Hydration

At write time each new memory gets directed `supersedes` edges to the
few older memories whose embeddings fall in a knob-derived similarity
band, and the edge weight grows with similarity
(<a href="#sec-advanced" class="quarto-xref">Section 10</a>). These
edges are topical, not only corrections. Retrieval does not demote
either endpoint. Hydration uses the edges in both directions:

-   **Predecessors.** For each of the top 11 anchors, the at most 2
    older memories it supersedes (strongest edges first) are attached
    *before* the anchor’s own content. Each is clipped to 512 bytes,
    under 4 KiB per packet. People do not lose the earlier version of a
    fact when they learn the newer one. Recalling the update brings back
    the episode it updated, and in these conversations the “obsolete”
    turn is often exactly what the question asks about.
-   **Successors.** For each of the top 11 anchors, the newest memory
    that supersedes it is attached *after* its content. It is clipped to
    512 bytes, under 4 KiB per packet, with one indexed lookup on the
    edge target for all anchors. A question phrased like a fact’s first
    statement retrieves that older version, while the latest restatement
    ranks below it.

With both attachments, each anchor’s evidence reads oldest to newest.

**Effect.** Switching off supersession edges and predecessor hydration
together changed 47/48 LongMemEval-S and 249/250 Eval 1 packets and cost
0.037, 0.025 and 0.015 packet recall (S, Eval 1, BEAM). Haiku-judged
sufficiency moved −0.036 ± 0.052 (S) and −0.016 ± 0.013 (Eval 1)
(<a href="#sec-experimental" class="quarto-xref">Section 15</a>).
Predecessor hydration used to skip a target from the anchor’s own source
stream within 30 minutes. Removing that skip raised BEAM-10M
conversation 7 recall from 0.103 to 0.118 for 0.5 KB more packet
(<a href="#sec-advanced" class="quarto-xref">Section 10</a>). Successor
hydration added 0.005 gold-turn share at each BEAM scale and +0.023
\[0.010, 0.036\] to the Haiku-answered 10M score
(<a href="#sec-implementation" class="quarto-xref">Section 16</a>).

With `as_of`, both attachments and contiguity skip memories that start
after *T*<sub>*q*</sub>.

## Per-Blob Content Timestamps

Attached blobs can come from weeks earlier (predecessors), from later
(successors), or from adjacent turns. The anchor’s single `timestamp`
does not date them. Every retrieved memory therefore carries one stored
signal time per content blob, in content order, with 0 meaning unknown
(PR #30).

-   **Invariant.** Whenever `content` is non-empty,
    `content_timestamps.size() == content.size()`.
-   **Coverage.** Every path that builds or reshapes content carries the
    source signal’s time: signal and memory blob hydration,
    working-memory hydration, contiguity, predecessor and successor
    attachment, standing joins, and evidence fitting.
-   **Modality.** Time stays signal metadata, never content.

**Known limit.** A multi-signal blob, such as a Natural unit’s joined
text, reports the time of its first signal.

**Parity.** No ranking, selection, hydration choice or budget changed.
Full LoCoMo (1,986 queries) returned byte-identical text with 80,422
parts, none with timestamp 0.

## Evidence Budget

The engine cannot know a reader’s context budget, so the caller may pass
one per query in bytes (`EvidenceBudget`, on every `Process*` call and
in the C API). Text is charged by its bytes and every other payload by
`kMediaEvidenceBytes` = 480. Unset, memories come back whole. Earlier
builds applied a fixed 3 KiB cap per memory, which cost judged Eval 1
sufficiency −0.016 ± 0.016 when the reader could see the whole packet.

Set, the fit runs once the packet is complete. It covers the standing
block, the retrieved memories and their attachments:

``` text
budget' = budget - bytes(standing block)
for m in retrieved, in rank order:
    floor(m) = for a grouped memory whose members all have stored vectors:
                 its best member per modality by cos(e_q, v_s);
               otherwise: the whole memory
    join m at floor(m) with its attached blobs if they fit,
    else at floor(m) alone, else leave m out
member_cap = largest c (binary search) such that the packet fits when
             every grouped memory keeps its floor plus its other members,
             best first, while their bytes stay within c
kept members return in their original order, between the attachments
```

The best member is chosen per modality because cosines are not
comparable across modalities: a text query sits closer to every text
member than to any image. The fit keeps more distinct memories trimmed
before it drops lower-ranked ones. It costs one indexed signal read plus
member payload reads, and only for packets over budget.

**Effect.** Under Natural retention with one source per conversation,
BEAM stores units of three to five turns, about 10 KB each with
attachments. An 8,000-token budget then held three or four whole
memories. With the BEAM reader’s budget passed (34,800 bytes), the
gold-turn share of its context matched the old 3 KiB cap (−0.002 /
+0.001 / +0.002 at 100K / 1M / 10M). The end-to-end score was 0.842 /
0.806 / 0.715 against 0.844 / 0.811 / 0.695. Unset, the whole memories
overflow that reader, and gold share falls by 0.144 / 0.115 / 0.093
(<a href="#sec-implementation" class="quarto-xref">Section 16</a>).
Every v5 benchmark in
<a href="#sec-experimental" class="quarto-xref">Section 15</a> runs at
the 34,800-byte budget.

# The Macro-Control Surface: Knob Priors and Affect

This section and
<a href="#sec-structural-metrics" class="quarto-xref">Section 9</a>
specify the per-signal control stack that earlier versions of this paper
presented as the core of the system. It still runs on every signal, in
the engine’s `CoreStage`. It changes packets: switching off the RLS
blender changes 1,980 of 1,986 LoCoMo retrieved sets (Jaccard 0.29),
switching off the composite score changes all of them (Jaccard 0.08),
and switching off the emotion projection changes 171 (Jaccard 0.975). It
reaches behavior through four paths:

-   **The stored embedding.** The composite score picks `e_peak`, the
    accumulator centroid at the unit’s highest score, which is blended
    into the representative embedding
    (<a href="#sec-rep-embedding" class="quarto-xref">Section 5.1.7</a>).
-   **Working-memory admission.** The window score ranks closed units
    for admission
    (<a href="#sec-wm-gating" class="quarto-xref">Section 10.2.2</a>).
-   **Uncertainty.** Score variance enters the uncertainty estimate,
    which feeds back into the learning rates below.
-   **Streaming.** The ACh-like encode/retrieve bias paces streaming
    retrieval checks between boundaries, and valence, arousal and
    emotion intensity feed the interrupt gate’s affect drive
    (<a href="#sec-interrupt" class="quarto-xref">Section 11</a>).

Nothing in this section is a write gate: every closed unit is stored
(<a href="#sec-thresholding" class="quarto-xref">Section 5</a>). Nothing
here enters query ranking either. Retrieval
(<a href="#sec-retrieval" class="quarto-xref">Section 6</a>) carries the
v5 recall results.

Each algorithm consists of a prior computation (executed at
initialization) and a dynamic update (executed per signal).

## Focus-Driven Selectivity

Focus governs perceptual selectivity through relevance weighting and
attention width.

### Focus Priors

Given Focus knob F ∈ \[0, 1\], initialize Focus control variables:

    weight_relevance = sigmoid(2F − 1)
    coverage_gain_floor = 0.3 + 0.7F
    mismatch_weight = 1 − F
    attention_width = lerp(π, 0.1π, F)

The attention width (in radians) controls the angular spread of the
receptive field in embedding space. High Focus produces narrow attention
(0.1π), while low Focus permits broad capture (π).

### Dynamic Focus Update

At each signal event t with accumulator centroid μ_acc (the running
centroid of the unflushed signal group):

    recent_context ← tail(signal_stream, n_ctx(T))  # signal_stream stores μ_acc
    if |recent_context| == 0:
        μ_ctx ← 0_vector; observed_cosine ← 0  # map01(0)=0.5
    else:
        μ_ctx ← mean(recent_context)
        observed_cosine ← cos(μ_acc, μ_ctx)
    weight_relevance_t ← EWMA(weight_relevance,
                          map01(observed_cosine), α = α_F(t))

    weight_relevance ← weight_relevance_t

where map01(z) = clamp((z + 1) / 2, 0, 1) transforms cosine values from
\[−1, 1\] to \[0, 1\].

The learning rate α_F(t) is modulated by uncertainty:

    α_min_F = 0.05; α_span_F = 0.45
    α_F(t) = α_min_F + F × α_span_F × u(t)

High uncertainty increases learning rate, allowing faster adaptation
when the environment is volatile. The Focus knob scales the uncertainty
responsiveness.

## Sensitivity-Driven Plasticity

Sensitivity governs learning speed, emotional responsiveness, and
novelty capture.

### Sensitivity Priors

Given Sensitivity knob S ∈ \[0, 1\], initialize Sensitivity control
variables:

    weight_novelty = 0.3 + 0.7S
    weight_surprise = 0.2 + 0.8S
    weight_valence = 0.4 + 0.6S
    weight_arousal = S
    emotion_gain = exp(1.5S)
    score_gain = exp(2S)

### Emotional Projection

When emotion category centroids are available, the system projects input
embeddings onto a discrete emotion space C = {anger, fear, joy, love,
sadness, surprise}. Categories inspired by Russell (1980) circumplex
model are projected onto the valence-arousal plane with the following
coordinates:

    v_map = {anger: −0.9, fear: −0.8, sadness: −0.9,
             joy: +0.9, love: +0.8, surprise: 0.0}
    a_map = {anger: +0.9, fear: +0.9, sadness: +0.3,
             joy: +0.6, love: +0.5, surprise: +0.8}

The projection procedure:

    raw_cos_c ← cos(μ_acc, centroids[c]) for each c ∈ C
    if all raw_cos_c ≤ 0:
        p_c ← uniform(1/6)
        emotion_intensity_t ← 0; valence_t ← 0.5; arousal_t ← 0
    else:
        logits_c ← max(0, raw_cos_c)
        β(S) = 8 + 24S  # softmax inverse temperature
        γ(S) = lerp(0.5, 0.25, S)  # intensity exponent (higher S boosts affect)
        g(S) = lerp(1.0, 2.2, S)   # gain to amplify affect strength
        p_c ← softmax(β(S) × logits_c)
        peak ← max_c(p_c)
        confidence ← 1 − H(p_c) / ln(6)
        emotion_intensity_t ← clamp((peak × confidence)^{γ(S)} × g(S), 0, 1)
        valence_t ← (Σ_c p_c × v_map[c] + 0.9) / 1.8
        arousal_t ← clamp(Σ_c p_c × a_map[c], 0, 1)

The emotion intensity combines peak probability with distributional
confidence through a Sensitivity-dependent power transform of their
product. It is high only when a category dominates and the distribution
has high confidence.

### Emotion Threshold Delta and Mood Integration (Removed)

The emotion projection above stays. Its valence, arousal and intensity
feed the valence and arousal metrics of the composite score
(<a href="#sec-structural-metrics" class="quarto-xref">Section 9</a>)
and the interrupt gate’s affect drive
(<a href="#sec-interrupt" class="quarto-xref">Section 11</a>).

Earlier builds also used it in two other ways. First, a threshold delta,
`−0.10 S × emotion_intensity × (0.5 + 0.5 × arousal)`, loosened the
dynamic threshold on salient signals, following McGaugh (2004). Second,
a 6D mood vector *M*<sub>*t*</sub> integrated the centred category
probabilities with a Sensitivity-derived reactivity
`lerp(0.01, 0.20, S)` and a Stability-derived half-life
`lerp(30, 600, T)` seconds. It added a second threshold delta
proportional to the norm of *M*<sub>*t*</sub> divided by √6. Mood’s only
effect was that delta.

Both deltas, the mood state and `memories.ambient_mood` were removed
with the dynamic threshold controller
(<a href="#sec-spike-threshold" class="quarto-xref">Section 5.2</a>),
with packets byte-identical (migrations 57 and 58).

## Stability-Driven Persistence (Removed)

Earlier builds derived the hysteresis band of the write threshold from
Stability (`InitializeStabilityPriors`). It was removed with the
threshold.

### Half-Life Adaptation (Removed)

Earlier builds also kept an adaptive memory half-life. It started at
`base_half_life(T)` and moved each signal by an EWMA toward
`base_half_life(T) × (1 + 0.25 × zscore_ret + ΔHalfLife_adj)`. Here
`zscore_ret` was the z-scored mean age of memories whose strength stayed
above `PeripheryCutoff(T)`, and `ΔHalfLife_adj` came from stability
feedback (<a href="#sec-feedback" class="quarto-xref">Section 13.4</a>).
Derived state included `rate_decay`, `periphery_half_life`,
`salience_half_life`, `drift_weight` and `retention_ema`.
Memory-strength decay was the only consumer of the half-life. The engine
ablation study (2026-10) removed memory strength
(<a href="#sec-reinforcement" class="quarto-xref">Section 13</a>), and
then this adaptation and its persisted state, which no operation read
any more.

## Neuromodulator Layer (Internal Mapping)

To keep the three-knob UI while enabling more biologically faithful
control, Cortext maps the knobs and internal signals to continuous
neuromodulator-like variables. These are **not** additional user knobs;
they are latent control signals derived from F/S/T and real-time stream
statistics.

One modulator remains:

-   **ACh-like (acetylcholine):** encode vs retrieve bias.

Its baseline comes from the knobs and its phasic part from the current
signal:

    ACh_base(F,S,T) = clamp(0.15 + 0.55S + 0.25(1 − T) − 0.15F, 0, 1)
    ACh_t ← clamp(ACh_base + 0.35 × novelty_t − 0.20 × retrieval_pressure, 0, 1)
    encode_bias ← ACh_t × (0.7 + 0.3S)
    retrieve_bias ← 1 − encode_bias

Where `retrieval_pressure` is the normalized retrieval queue-depth
input. The production operation context does not currently populate
queue depth, so this term is zero in the shipped full pipeline.
`retrieve_bias` gates streaming retrieval checks between boundaries
(<a href="#sec-interrupt" class="quarto-xref">Section 11</a>).

Earlier builds also kept an NE-like modulator,
`NE_t = clamp(NE_base + 0.50 × surprisal_t + 0.30 × arousal_t, 0, 1)`,
and “effect scales” that multiplied the write-gate threshold by
`1 − 0.3 × NE_t`, reconsolidation drift by `1 + 0.4 × ACh_t` and
retrieval competition by `1 + 0.5 × NE_t`. Competition and
reconsolidation were removed by the engine ablation study (2026-10).
Switching all effect scales off changed 5/48 LongMemEval-S, 32/250 Eval
1 and 7/20 BEAM-10M conversation 7 packets, left gold-turn recall within
one case, and moved Haiku-judged sufficiency by 0.000, −0.001 ± 0.006
and +0.028 ± 0.054; the MSC replay changed 7/39 probe packets, five only
in order. The write scale and NE were then removed (migration 47).

Earlier builds also kept a DA-like modulator,
`DA_t = clamp(DA_base + max(0, δ_reward_t), 0, 1)`, driven by a reward
prediction error `δ_reward_t = outcome_t − EWMA(outcome_t)` from the
metric blender. Its only consumer was `value_update_gain` for the
procedural value table
(<a href="#sec-pattern-separation" class="quarto-xref">Section 10.14</a>).
The engine ablation study (2026-10) removed that table, then the DA
modulator, the reward prediction error and their persisted state, with
no packet change.

## Oscillatory Gating (Removed)

Earlier builds multiplied `encode_bias` by a slow oscillator,
`0.6 + 0.4 × (0.5 + 0.5 × sin φ_t)` with `φ_t` advancing at
`ω(S,T) = lerp(0.03, 0.12, S) × lerp(1.2, 0.8, T)` rad/s. Its only
effect was on `retrieve_bias`, which the streaming pacer consults only
between boundaries; accepted boundaries, forced checks and queries
bypass it. The 2026-10 MSC rerun on the current engine (25 validation
dialogs, 1,190 turns, 39 probes, capacity 21) left every probe packet
byte-identical with the oscillator off, under both Durable and Natural
retention, so the engine ablation study removed it and its persisted
phase (<a href="#sec-experimental" class="quarto-xref">Section 15</a>).

## Knob-to-Parameter Priors (Meta-Learning Removed)

All knob-derived ranges and coefficient tables (e.g., bootstrap
coefficients for RLS, lerp endpoints, base rates) are fixed **species
priors**: closed-form functions of F/S/T. `attention_width_prior` is
`lerp(W_min, W_max, 1 − F)`, `base_rate_prior` is `BaseRatePrior(S)`,
and `hysteresis_band_prior` is `BaseBandPrior(T)`.

Earlier builds made three of these priors learnable
(`p = lerp(a, b, sigmoid(α_F F + α_S S + α_T T + β))`, updated from
write acceptance, retrieval use, uncertainty and reward, and stored in a
`meta_learning_coeffs` table). The engine ablation study (2026-10)
removed it. Skipping the learner and using the closed-form priors left
every retrieval packet byte-identical on Eval 1 (0/250), LongMemEval-S
(0/48) and two full-length BEAM-10M conversations (0/40, 42.6k turns),
with no measurable cost change. Migration 35 drops the table.

# Structural Metrics and Composite Scoring

Two groups of signals are defined here. The embedding-derived signals
(structural coherence, focus spread, instantaneous drift and EMA
prediction error) feed boundary segmentation
(<a href="#sec-thresholding" class="quarto-xref">Section 5</a>) and the
streaming gate
(<a href="#sec-interrupt" class="quarto-xref">Section 11</a>). The
composite score built from the 12 metrics below is live: it changes
every LoCoMo retrieved set when switched off (Jaccard 0.08), and the RLS
blender that weights it changes 1,980 of 1,986 (Jaccard 0.29). It is not
a write gate. Its consumers are the ones listed in
<a href="#sec-core-adaptation" class="quarto-xref">Section 8</a>:

-   the peak embedding `e_peak`, which shapes each stored representative
    embedding;
-   the window score for working-memory admission;
-   the score variance in the uncertainty estimate.

Context definitions (used throughout Table 1 and structural metrics):

    recent_context ← tail(signal_stream, n_ctx(T))  # signal_stream stores μ_acc
    if |recent_context| == 0:
        μ_ctx ← 0_vector  # define cos(μ_acc, μ_ctx)=0 so map01(cos)=0.5
    else:
        μ_ctx ← mean(recent_context)

## Embedding-Derived Metrics

### Structural Coherence

Structural coherence (coherence_struct) measures integration of the
current accumulator centroid with the broader context window. This
metric is distinct from memory coherence (coherence_mem, defined in
<a href="#sec-drift-coherence" class="quarto-xref">Section 5.1.2</a>)
which tracks within-memory similarity:

    coh_neutral(T) = lerp(0.45, 0.55, T)
    if |recent_context| < 2:
        coherence_struct_t ← coh_neutral(T)  # neutral, stability-derived
    else:
        raw ← var([cos(μ_acc, c) for c in recent_context])
        coherence_struct_t ← 1 − clamp(raw, 0, 1)  # range [0, 1]

High structural coherence (low variance in similarities) indicates the
current accumulator centroid fits consistently with context. The
effective Focus is modulated: F_eff = F × (0.5 + 0.5 ×
coherence_struct_t).

**Effective Focus usage:** F_eff is a diagnostic modulation of Focus.
Unless a formula explicitly references F_eff, the specification uses the
midpoint‑biased knobs F̃ and S̃ (see
<a href="#sec-math" class="quarto-xref">Section 4</a>) for all
knob‑derived controls and Table 1 metric formulas.

### Focus Spread

Focus spread quantifies the entropy of attention over nearest neighbors:

    k ← k_neighbors(T) = round(lerp(8, 32, T))

Normative note (MUST): kNN_similarities MUST be computed by querying
memory_stream with q = μ_acc and k = k_neighbors(T) (not
recent_context).

    if |memory_stream| == 0:
        focus_spread_t ← 0  # cold-start fallback
    else:
        k_eff ← min(k, |memory_stream|)
        if k_eff < 2:
            focus_spread_t ← 0  # avoid degenerate entropy
        else:
            kNN_similarities ← topK(vector_search(μ_acc, k=k_eff))
            # attention_width sharpens or flattens similarities
            kNN_similarities ← clamp(kNN_similarities × (π / attention_width), −1, 1)
            p ← softmax(kNN_similarities)
            focus_spread_t ← H(p) / ln(k_eff)

Values near 1 indicate diffuse attention; values near 0 indicate
concentrated attention. The effective Focus is further modulated: F_eff
← F_eff × (1 − focus_spread_t).

### Instantaneous Drift

Drift measures the instantaneous step between consecutive accumulator
centroids rather than a smoothed centroid lag:

    if x_{t−1} is unset:
        drift_step_t ← 0
    else:
        drift_step_t ← cosine_dist(μ_acc_t, μ_acc_{t−1})  # 1 − cos(·), range [0, 2]
    drift_mag_t ← drift_step_t

This uses the current centroid velocity in embedding space, ensuring
drift responds immediately to context shifts. A threshold defines an
informational drift-boundary signal:

    drift_threshold ← lerp(0.10, 0.35, T)
    drift_boundary_t ← (drift_mag_t > drift_threshold)

Normative note (MUST): drift_boundary_t is informational and MUST NOT
trigger a memory flush on its own. Memory flush decisions are defined by
should_flush in
<a href="#sec-boundary" class="quarto-xref">Section 5.1.3</a>.

The instantaneous drift_step_t also informs boundary pressure and
focus-width feedback. All other composite metrics in this section use
the accumulator centroid μ_acc as the effective signal representation.

### Real-Time Prediction Error (EMA)

We model real-time expectation using an exponential moving average (EMA)
of the accumulator centroid stream. The predictor state `x_pred_ema`
represents the current expectation; low Stability (T) adapts quickly,
high Stability adapts slowly.

We treat orthogonal or anti-correlated vectors (cos ≤ 0) as maximum
surprise.

    β_pred(T) = lerp(0.25, 0.02, T)
    err_ref(S) = lerp(0.25, 0.05, S)
    k_surprise(S,T) = lerp(6, 14, S) × lerp(1.1, 0.9, T)

    if x_pred_ema is unset:
        x_pred_ema ← μ_acc  # initialize expectation
        surprisal_t ← 0
    else:
        prediction_error_t ← 1 − max(0, cos(μ_acc, x_pred_ema))
        surprisal_t ← sigmoid((prediction_error_t − err_ref(S)) * k_surprise(S,T))
        x_pred_ema ← l2_normalize((1 − β_pred(T)) × x_pred_ema +
                                   β_pred(T) × μ_acc)

This formulation captures prediction failure directly from the stream,
without requiring an explicit kinematic trend model.

## Composite Score Computation

The system computes 12 metrics that blend into a composite write score:

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr>
<th style="text-align: left;">Metric</th>
<th style="text-align: left;">Knob</th>
<th style="text-align: left;">Expression</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Relevance</td>
<td style="text-align: left;">↑F</td>
<td
style="text-align: left;"><code>relevance_t = clamp(map01(cos(μ_acc, μ_ctx)) × (0.5 + 0.5F), 0, 1)</code></td>
</tr>
<tr>
<td style="text-align: left;">Mismatch</td>
<td style="text-align: left;">↓F, ↑S</td>
<td style="text-align: left;"><code>(1 − F) × S × novelty_t</code></td>
</tr>
<tr>
<td style="text-align: left;">Surprise</td>
<td style="text-align: left;">↑S, ↓T</td>
<td
style="text-align: left;"><code>surprisal_t × S × (1 − 0.5T)</code></td>
</tr>
<tr>
<td style="text-align: left;">Rarity</td>
<td style="text-align: left;">↑F, ↓T</td>
<td
style="text-align: left;"><code>rarity_t × (0.5 + 0.5F) × (1 − 0.2T)</code></td>
</tr>
<tr>
<td style="text-align: left;">Drift</td>
<td style="text-align: left;">↓T</td>
<td
style="text-align: left;"><code>(drift_mag_t / 2) × (1 − T)</code></td>
</tr>
<tr>
<td style="text-align: left;">Utility</td>
<td style="text-align: left;">↑F, ↓S</td>
<td
style="text-align: left;"><code>ΔSSE × (0.5 + 0.5F) × (1 − 0.3S)</code></td>
</tr>
<tr>
<td style="text-align: left;">Salience</td>
<td style="text-align: left;">F, S</td>
<td
style="text-align: left;"><code>(rarity_t + novelty_t) / 2 × (F + S) / 2</code></td>
</tr>
<tr>
<td style="text-align: left;">Valence</td>
<td style="text-align: left;">S, ↓T</td>
<td style="text-align: left;"><code>valence_t</code></td>
</tr>
<tr>
<td style="text-align: left;">Arousal</td>
<td style="text-align: left;">S, ↓T</td>
<td style="text-align: left;"><code>arousal_t</code></td>
</tr>
<tr>
<td style="text-align: left;">Contradiction</td>
<td style="text-align: left;">↑S, ↓F</td>
<td style="text-align: left;"><code>max(0, S − F)</code></td>
</tr>
<tr>
<td style="text-align: left;">Periphery</td>
<td style="text-align: left;">↑T</td>
<td style="text-align: left;"><code>(1 − relevance_t) × T</code></td>
</tr>
<tr>
<td style="text-align: left;">Coverage</td>
<td style="text-align: left;">↑F</td>
<td style="text-align: left;"><code>F × relevance_t</code></td>
</tr>
</tbody>
</table>

*Table 1: Metric definitions and knob dependencies. Arrows indicate
direction of influence.*

## Metric Weight Blending

Metric weights adapt online using recursive least squares (RLS) to
minimize prediction error between composite scores and **downstream
utility outcomes**. This avoids circular fitting to relevance. In
production the explicit user term `o_user` is 0. Initial weights derive
from bootstrap coefficients:

Blender weights are maintained as variables w\_\* (do not confuse these
with control weights like weight_relevance or mismatch_weight). We
denote the 12-element blender weight vector as:

    W_blend = [w_relevance, w_mismatch, w_surprise, w_rarity, w_drift,
              w_utility, w_salience, w_valence, w_arousal, w_contradiction,
              w_periphery, w_coverage]

The index i used below (e.g., w_bootstrap\[i\], w_rls\[i\]) follows this
ordering.

    w_bootstrap[i] ← sigmoid(c_F[i]×F + c_S[i]×S + c_T[i]×T + d_i)

Before normalization, control weights modulate the blender weights:

    w_relevance ← w_relevance × weight_relevance
    w_mismatch  ← w_mismatch  × mismatch_weight
    w_surprise  ← w_surprise  × weight_surprise
    w_valence   ← w_valence   × weight_valence × emotion_gain
    w_arousal   ← w_arousal   × weight_arousal × emotion_gain
    w_coverage  ← w_coverage  × coverage_gain_floor

Composite score scaling applies after weight normalization:

    score_t ← clamp(score_raw × score_gain, 0, 1)

Bootstrap coefficient defaults (canonical; used for initialization):

<table>
<thead>
<tr>
<th>Metric</th>
<th>c_F</th>
<th>c_S</th>
<th>c_T</th>
<th>d</th>
</tr>
</thead>
<tbody>
<tr>
<td>relevance</td>
<td>1.4</td>
<td>0.0</td>
<td>0.4</td>
<td>−1.0</td>
</tr>
<tr>
<td>mismatch</td>
<td>−1.0</td>
<td>1.0</td>
<td>0.0</td>
<td>−0.5</td>
</tr>
<tr>
<td>surprise</td>
<td>0.0</td>
<td>1.5</td>
<td>−0.5</td>
<td>−0.75</td>
</tr>
<tr>
<td>rarity</td>
<td>0.9</td>
<td>0.0</td>
<td>−0.3</td>
<td>0.05</td>
</tr>
<tr>
<td>drift</td>
<td>0.0</td>
<td>0.0</td>
<td>−1.0</td>
<td>0.0</td>
</tr>
<tr>
<td>utility</td>
<td>0.85</td>
<td>−0.45</td>
<td>0.0</td>
<td>0.075</td>
</tr>
<tr>
<td>salience</td>
<td>1.0</td>
<td>1.0</td>
<td>0.0</td>
<td>−1.0</td>
</tr>
<tr>
<td>valence</td>
<td>0.0</td>
<td>1.02</td>
<td>−0.42</td>
<td>−0.11</td>
</tr>
<tr>
<td>arousal</td>
<td>0.0</td>
<td>1.8</td>
<td>−0.2</td>
<td>−0.9</td>
</tr>
<tr>
<td>contradiction</td>
<td>−2.0</td>
<td>2.0</td>
<td>0.0</td>
<td>−1.0</td>
</tr>
<tr>
<td>periphery</td>
<td>0.0</td>
<td>0.0</td>
<td>1.0</td>
<td>−1.0</td>
</tr>
<tr>
<td>coverage</td>
<td>2.0</td>
<td>0.0</td>
<td>0.0</td>
<td>−1.0</td>
</tr>
</tbody>
</table>

These coefficients define a canonical initialization map from the three
knobs; they do not introduce additional runtime thresholds and may be
refit from human data without changing the rest of the system.

### RLS Weight Adaptation (Canonical)

RLS fits the blender weights directly in **weight-space**, using the
current metrics as predictors and an observed **outcome signal** as the
target. State is retained as a weight vector `w` and covariance matrix
`P`.

Target and predictor:

    x_t ← [metric_i] (each metric clamped to [0,1])
    y_t ← outcome_t  # observed downstream utility (in [0,1])

Outcome definition (utility target; normalized to \[0,1\]):

    o_use ← 1 if a retrieved memory was injected into context and used, else 0
    o_pred ← ΔSSE  # prediction-error reduction proxy (Appendix B)
    o_unc ← clamp(1 − u_uncertainty, 0, 1)
    o_user ← user_accept_t  # explicit accept/correct signal if available, else 0

    [w_use, w_pred, w_unc, w_user] ← normalize([0.4 + 0.2F, 0.3 + 0.2S, 0.2 + 0.2T, 0.1])
    outcome_t ← clamp(w_use×o_use + w_pred×o_pred + w_unc×o_unc + w_user×o_user, 0, 1)

Update cadence and forgetting:

    k_update = round(lerp(1, 8, T))  # update every k_update signals
    λ(T) = 0.90 + 0.09T              # forgetting factor

RLS update (performed when step % k_update == 0):

    y_hat ← wᵀ x_t
    e_t ← y_t − y_hat
    K ← (P x_t) / (λ + x_tᵀ P x_t)
    w ← clamp(w + K × e_t, 0, 1)
    P ← (I − K x_tᵀ) P / λ

Note: RLS fits **unnormalized** weights `w`; normalization happens later
in composite score computation.

Initialization:

    w_0 ← w_bootstrap   # from the coefficient table above
    P_init(T) = lerp(500, 2000, 1 − T)
    P_0 ← diag(P_init(T))

Interpretation-only window size:

    N ← round(lerp(64, 512, T))  # effective fitting window (interpretation only)

The fitted weights blend with bootstrap weights based on RLS confidence:

    τ_rls ← lerp(20.0, 80.0, T)
    t ← signals_processed  # global step count (maturity)
    confidence_rls ← 1 − exp(−t / τ_rls)
    w_rls01[i] ← clamp(w[i], 0, 1)  # constrain fitted weights to mixture range
    weight_i(t) ← clamp((1 − confidence_rls) × w_bootstrap[i] +
                          confidence_rls × w_rls01[i], 0, 1)

### Score Normalization

Composite score computation requires careful normalization:

    for each metric i:
        m01[i] = clamp(metric[i], 0, 1)
    weight_sum ← Σ weights[i]
    if weight_sum < ε: return 0
    weights_norm[i] ← weights[i] / weight_sum
    score ← clamp(Σ weights_norm[i] × m01[i], 0, 1)

Invariant (MUST): all 12 Table 1 metric values are defined on \[0, 1\].

Weight normalization is critical: with 12 metrics and raw weights
averaging ~0.6, the sum approaches 7.2. Without normalization, weighted
sums would saturate and collapse variance.

# Working Memory, Supersession and Removed Cognitive Mechanisms

This section specifies the surviving cognitively motivated components:
bounded working memory, the temporal context vector, and graph-native
belief revision through `supersedes` edges. It also keeps a short record
of the higher-order mechanisms the 2026-10 ablation removed, with their
measured effects. The removed mechanisms are reconsolidation, retrieval
competition, predictive pre-activation, serial-position effects,
emotional consolidation, synaptic tagging, Soft Anchors, and pattern
separation with its procedural lane
(<a href="#sec-negative" class="quarto-xref">Section 14</a>).

**Knob note:** Unless explicitly stated otherwise, F and S in this
section refer to midpoint‑biased values (F̃, S̃) defined in
<a href="#sec-math" class="quarto-xref">Section 4</a>.

## Working Memory Gates

Motivated by bounded working-memory models such as Cowan (2001), Cortext
maintains a limited number of active application-level memory units.
These slots are not asserted to be human cognitive chunks. Working
memory holds coherent memories as defined in
<a href="#sec-write-pacing" class="quarto-xref">Section 5.1</a>,
preserving the full content and signal sequence:

    base_capacity = round(3 × (lerp(8, 6, S̃) + lerp(−1, 1, F̃)))

This yields 15–27 slots and 21 at neutral knobs. High Sensitivity
reduces capacity (faster turnover), while high Focus increases the slot
budget. The operating point comes from the capacity study in
<a href="#sec-experimental" class="quarto-xref">Section 15</a> rather
than from a direct numerical fit to human span.

### Active Memory Structure

Each active memory holds a coherent memory with its full content and
metadata:

    memory.content ← concatenated signal blobs
    memory.source_id ← source identifier (e.g., 'user', 'assistant')
    memory.modality ← primary modality ('text', 'audio', 'image')
    memory.blob_ids ← [blob_1, blob_2, ..., blob_n]  # blob refs
    memory.embedding ← e_rep  # representative embedding (@sec-rep-embedding)
    memory.signals ← [x_1, x_2, ..., x_n]  # ordered signal embeddings
    memory.source_origin ← "source"  # constant class; provenance stays in source_id
    memory.evidence_packets ← ordered immutable signal rows
    memory.current_view ← latest reconstruction blob if present, else concatenated signal blobs
    memory.metadata ← {n, s_max, s_avg, drift_acc, mem_elapsed,
                       s_emotion_max, s_arousal_avg}

The emotional metrics (s_emotion_max, s_arousal_avg) are accumulated
during memory formation. Their original consumer, Emotional
Consolidation, was removed (<a href="#sec-emotional-consolidation"
class="quarto-xref">Section 10.12</a>). The `salience` and `arousal`
fields of returned working-memory items still carry the slot’s peak
score and mean arousal. The temporal context `c_t` is not stored with
the memory; migration 56 dropped that column, and `c_t` lives in the
accumulator only.

## Temporal Context State (Time-Cells Analogue)

Beyond the accumulator centroid μ_acc, Cortext maintains a slowly
drifting temporal context vector `c_t`. It anchors the topic-shift term
of boundary detection
(<a href="#sec-boundary" class="quarto-xref">Section 5.1.3</a>). Using
it for ordered recall through **context reinstatement** remains a
proposed extension, described below; it is not shipped.

    α_c(S,T) = AccumulatorTemporalContextAlpha(S,T)
    c_t ← l2_normalize((1 − α_c(S,T)) × c_{t−1} + α_c(S,T) × μ_acc)

At neutral Sensitivity, `α_c` follows `lerp(0.06, 0.01, T)`; the
production policy applies a bounded Sensitivity adjustment. The runtime
uses `c_t` as the topic-shift anchor in boundary detection. It no longer
persists `c_t` with each memory: nothing read the stored copy, and
migration 56 dropped it. Production retrieval does not add a
`cos(c_t, m.context)` score or reinstate `c_t` from a retrieved memory;
it queries from the current accumulator and recent-context surfaces
instead. Because the per-memory context is no longer stored, the
following context-score/reinstatement rule would need it restored. The
rule is therefore a proposed extension, not shipped behavior:

    w_ctx(F,S,T) = lerp(0.15, 0.45, F) × lerp(1.0, 0.85, S)
    association_boost(F,S,T) = lerp(0.015, 0.06, S) × lerp(1.0, 0.7, F) × lerp(1.0, 0.9, T)
    score_retrieval(m) ← (1 − w_ctx) × cos(q, m.embedding) +
                          w_ctx × cos(c_t, m.context) +
                          association_boost(F,S,T) × I[m.kind = ASSOCIATION]

Proposed reinstatement would update `c_t` toward `m.context` after
retrieval.

### Maintenance Cost

Maintenance incurs cognitive cost:

    maintenance_budget = 7 × lerp(0.05, 0.15, S̃)
    maintenance_cost_per_memory = maintenance_budget / max(base_capacity, 1)
    complexity_penalty = manifold_complexity × lerp(0.5, 1.5, S)

The legacy seven-slot normalization keeps the total cost of a full
configured working-memory window invariant across capacity settings; it
does not mean the shipped capacity is seven.

Passive strength decay is incremental. Access recency and decay
accounting use separate timestamps so repeated maintenance passes charge
each elapsed interval once without making an untouched slot appear
recently accessed:

    Δt_decay ← now_s() − strength_ts
    if Δt_decay > 0:
      strength ← max(strength_floor(F,S,T),
                     strength − maintenance_cost_per_memory × Δt_decay)
      strength_ts ← now_s()
    # last_ts changes only on chunking or insertion

The active floor is applied only while charging positive elapsed time. A
same-time or out-of-order signal therefore cannot recharge a slot that
was legitimately persisted below the current floor under an earlier knob
setting.

Persisted working-memory rows mirror `strength_ts` as
`strength_updated_at`. Load-time decay advances the same timestamp and
leaves changed slot metadata dirty so the next normal process persists
both values.

The manifold_complexity is defined as a normalized local variability
proxy (see Appendix B): manifold_complexity ← clamp((1 −
mean_cos_window) / 2, 0, 1).

### Memory-Level Gating

Gating thresholds determine entry and chunking:

    chunking_threshold = lerp(0.7, 0.9, F)
    gate_threshold = lerp(0.1, 0.4, F)
    memory_dedication_strength = lerp(0.3, 0.9, T)

Working memory gating evaluates coherent memories at accumulation
boundaries
(<a href="#sec-boundary" class="quarto-xref">Section 5.1.3</a>), not
individual signals. The WM update occurs **before** accumulator reset so
each slot preserves ordered signal blobs for content hydration:

    on_memory_boundary:
        [α, β, γ] ← normalize([lerp(0.55, 0.70, F) × (1 + 0.10(T − 0.5)),
                            lerp(0.20, 0.35, F) × (1 − 0.08(T − 0.5)),
                            lerp(0.10, 0.30, S) × (1 − 0.12(T − 0.5))])
        memory_benefit ← α × S_window + β × relevance_to_task(μ_acc, task_context) +
                           γ × novelty_to_set(μ_acc, {m.embedding | m ∈ active_memories})
        margin ← memory_benefit − gate_threshold
        k ← |active_memories|
        C ← max(base_capacity, 1)
        p_cap ← 3
        capacity_pressure(k, C) ← 1 + max(0, (k − C) / C)^p_cap
        raw_cost ← (maintenance_cost_per_memory × k + complexity_penalty) ×
                   capacity_pressure(k, C)
        total_cost ← raw_cost / (1 + raw_cost)  # squash to [0, 1)
        accept_memory = (margin ≥ total_cost)

Normative note (MUST): total_cost is normalized to \[0, 1\] to keep
gating on the same score scale as margin. This avoids cold-start or
capacity regimes where costs can exceed 1 and make acceptance
vanishingly rare.

Note that total_cost is computed from existing active memories only (k
is the current count), so k=0 yields no bootstrap penalty. Capacity
pressure activates only when k \> C.

When active memory is at capacity, eviction ranks slots by the usual
dedication/recency eviction score. Exact score ties are deterministic:
evict the weaker slot first, then the older slot, then the lowest stable
slot index. This keeps replay and hydration reproducible without
changing the strength/recency policy.

### Chunking at Memory Level

Chunking operates on memory embeddings to merge related content from the
same source:

    similar_memories ← {m ∈ active_memories | cos(m.embedding, memory.e_rep) > chunking_threshold AND
                                  m.source_id == memory.source_id}
    chunk_max(T) = round(lerp(8, 24, T))  # signals per slot (16 at T = 0.5)
    best ← argmax cos over similar_memories
    if best exists AND best.n_signals + memory.n_signals ≤ chunk_max(T):
        merge_into_chunk(best, memory)
    else:
        insert_new_slot(memory)  # capacity eviction as for any insert

The chunk budget bounds what one slot carries into every packet. Without
it, per-conversation Natural ingest let a slot absorb a whole stream: on
BEAM (22,652 turns) three slots held 21,793 signals, each query packet
carried ~0.9 MB of working-memory text, and consolidation p95 reached 10
s. With `chunk_max` = 16 the same stream keeps gold recall (.110
retrieved), halves the working-memory text, cuts ingest instructions
62%, and brings consolidation p95 to 0.31 s. Durable ingest never chunks
(each turn keeps its own trace).

Earlier builds also rehearsed slots: an incoming unit within
`lerp(0.5, 0.7, F)` cosine of a slot, or a retrieved memory within the
chunking threshold, raised that slot’s strength by
`lerp(0.5, 2.0, S) × base_delta`. The round-3 ablation (2026-10) found
rehearsal changed 12/48 LongMemEval-S and 33/250 Eval 1 packets
(working-memory order only) and none on BEAM-10M conversation 7, with
gold-turn recall unchanged and Haiku-judged sufficiency +0.000 (S) and
+0.002 ± 0.003 (Eval 1); removing it saved 3–4% of instructions, so it
was removed.

**Implementation note:** durable ingress inputs preserve an ordered
working-memory trace independent of the literal `source_id` string.
`source_id` is still used for opaque provenance and exact same-source
grouping, but Cortext does not parse roles such as user, assistant,
speaker, or device from it. Internal maintenance work follows the same
rule: explicit consolidation is driven by a force flag on the signal,
not by a magic `source_id` value. Prompt hydration orders active slots
by insertion/start time rather than `last_access`; otherwise slot
updates can scramble the visible recent context even when the underlying
memories are correct.

## Two-Layer Memory Surface

Cortext separates live memory state into two layers. They differ by
latency and commitment, not by modality.

1.  **Working memory (WM)** is the bounded active slot set used for
    recent-turn continuity, overlap suppression and the `working_memory`
    block of a packet. It stores coherent units with their ordered
    evidence. It is the most volatile layer and serves exact recent
    context rather than broad semantic search.
2.  **Long-term memory (LTM)** is the durable store. Each closed unit is
    a `LONG_TERM` row in `memories`, with its pooled representative
    embedding. Its signals are `signals` rows with object-store
    payloads, addressed by BLAKE3 content id. Four per-payload or
    per-memory indexes serve retrieval
    (<a href="#sec-retrieval" class="quarto-xref">Section 6</a>):
    -   `memory_signal_vectors`: one shared-space vector per payload;
    -   `memory_signal_li_codes`: late-interaction codes per payload;
    -   `memory_text_fts`: FTS5 keyword index over long-term text;
    -   `standing_signals`: the standing set.

    The only edges are directed `supersedes` edges in `associations`.

There is no processor-local short-term shadow graph and no fact
assertion store. Clustering consolidation no longer writes `ASSOCIATION`
centroids, `derived_from` edges or typed graph edges
(<a href="#sec-consolidation" class="quarto-xref">Section 12</a>). Older
stores may still hold such rows, and the runtime neither writes nor
traverses them.

The control flow is:

``` text
signal embedding
  -> accumulator update and boundary segmentation
  -> on a boundary: LTM unit write (+ indexes, + supersedes edges)
  -> WM admission of the closed unit
query embedding
  -> hybrid retrieval over LTM (@sec-retrieval)
  -> hydration and evidence fitting (@sec-hydration)
Consolidate()
  -> seal and recenter the sparse retrieval route (@sec-optimization)
```

Every step is embedding-only or metadata-only and modality-agnostic, and
no step calls a language model.

## Soft Anchor Formation (Removed)

Earlier builds ran an ingress-time Soft Anchor operation after memory
storage. It kept provisional subject/entity continuity states (semantic,
entity and full centroids with support and contradiction counts) and
linked each stored memory to up to three of them in `soft_anchors` and
`soft_anchor_links`; hydration exposed the links as
`Context::Memory::soft_anchors`. Retrieval never ranked on them. The
engine ablation study (2026-10) found packets byte-identical without the
operation on LongMemEval-S 0/48, Eval 1 0/250 and BEAM-10M conv 7 0/20,
at a 0.5-1% ingest instruction cost, and removed the operation, its
knobs, the public `Context::Memory::soft_anchors` / output diagnostics,
the C API JSON fields and the 1536-dimensional `soft_anchor_embedding`
signal slot. Migration 48 drops both tables.

## Retrieval Uncertainty

The current engine does not maintain a persistent higher-order
confidence state. Uncertainty is expressed directly through retrieval
scores, rank margins, coverage, candidate diversity, and source/evidence
metadata:

    rank_margin(q) = score_1(q) - score_2(q)
    retrieval_uncertainty(q) = 1 - clamp(rank_margin(q) / margin_ref(F,S,T), 0, 1)

The engine does not compute this quantity; it is a reading the caller
can take from the returned scores. This keeps uncertainty local to the
retrieval event. Callers can use the ranked candidate packet and
evidence surfaces to decide whether to answer, ask for clarification, or
expose multiple possible memories without the engine carrying a separate
persistent confidence machine.

## Memory Reconsolidation and Retrieval Reconstruction (Removed)

Earlier builds moved stored embeddings after use. Following Nader,
Schafe, and Le Doux (2000), a retrieved memory became labile
(`τ_labile = lerp(30, 300, T)`, susceptibility
`(1 − T) × (0.5 + 0.5 × S)`) and drifted toward the current context by
`(1 − T) × S × lability × contextual_relevance × lerp(0.2, 0.02, T)`,
scaled by `1 + 0.4 × ACh_t`. Retrieval itself also re-encoded its top
candidates as a bounded blend of the stored embedding and the query
(`blend ∈ [lerp(0.04, 0.07, S), …]`, at most
`RetrievalReconstructionUpdateCount` updates per signal). Both appended
versions to `memory_reconstructions` and moved the current retrieval
surface. An earlier ripple over association edges was removed first
because nothing wrote those edges.

The round-3 leave-one-out (2026-10) switched both off together on the
full-length evaluations. LongMemEval-S packet gold-turn recall was
unchanged (0.642) and Haiku-judged sufficiency moved 0.000; Eval 1
recall rose 0.597 → 0.611 (6 cases better, 1 worse), judged +0.010 ±
0.016; BEAM-10M conversation 7 rose 0.103 → 0.110, judged +0.014 ±
0.027; an eight-episode LongMemEval-M subset (39k turns) kept recall at
0.462, judged 0.000. Instructions fell 5–11% and stores shrank 12–23%.
Each source alone was also neutral. The earlier phase-1 “embedding
movement” arm that measured harm also switched off the constructive
reconstruction representation, so its delta came from that
representation change rather than from movement. The engine ablation
study (2026-10) therefore removed both movement sources, their knobs,
the ACh reconsolidation scale and the per-memory lability columns
(migration 46). Stored memories keep the reconstruction ledger written
at storage time, so existing stores that already hold moved versions are
still read through their latest reconstruction.

## Stored Source Metadata and Supersession

Each memory stores opaque `source_id` provenance and the constant origin
class `source_origin = "source"`; the application’s provenance stays in
`source_id`. Earlier builds also stored a knob-derived
`source_reliability` prior, a `source_contradiction_count` that
supersession incremented, and `source_last_verified_ts`. They were meant
for a proposed source-confidence gate that never shipped, and nothing
read them. Migration 56 dropped all three.

The shipped belief-revision path is graph-native and modality-agnostic.
On a durable memory write, the stored embedding is compared with a
knob-bounded nearest-neighbor set of prior memory embeddings. If
pairwise similarity lands in the supersession band, the runtime writes a
directed `supersedes` edge from the new memory to the older memory:

    θ_topic ← SupersessionSimilarityThreshold(F,S,T)
    θ_dup   ← SupersessionDuplicateThreshold(F,S,T)

    if θ_topic ≤ cos(e_new, e_old) < θ_dup:
        associations.add(e_new → e_old, type = "supersedes",
                         weight = SupersessionEdgeWeight(cos,F,S,T))

Retrieval treats `supersedes` as a directional family relation rather
than a traversable activation edge. Packet hydration uses it in both
directions: a retrieved replacement gets its superseded predecessors,
and a retrieved older version gets its newest superseding successor
(<a href="#sec-hydration" class="quarto-xref">Section 7</a>). Earlier
builds also removed superseded targets before the vector seed cap. The
engine ablation study (2026-10) measured that exclusion as harmful and
removed it. Keeping superseded targets eligible raised the share of gold
turns among the seed anchors by +0.16 on Eval 1, +0.14 on LongMemEval-S,
+0.16 on LongMemEval-M and +0.05 on BEAM-10M. Packet recall rose by
+0.017, +0.025, +0.000 and +0.039. In these conversations an “obsolete”
turn is often the evidence a question asks about. The number of written
supersession edges is capped by `SupersessionMaxEdges(F,S,T)`.

Predecessor hydration used to skip a target from the anchor’s own source
stream within 30 minutes, treating it as an in-conversation correction.
The round-3 leave-one-out (2026-10) found the skip changed no packet on
LongMemEval-S (0/48) or Eval 1 (0/250), because contiguity hydration
already attaches such nearby turns, while on BEAM-10M conversation 7
dropping it raised packet gold-turn recall from 0.103 to 0.118 (2 cases
better, none worse) for 0.5 KB more packet. The skip was removed.

## Reconstruction Ledger

Each memory keeps a two-layer view, a remnant of the removed
constructive-recall design:

-   **Evidence packets** remain the immutable ordered `signals` rows
    attached to a memory.
-   **Reconstructions** are versioned rows in `memory_reconstructions`,
    each carrying `{embedding_id, blob_id, ts, uncertainty, trigger}`.

<!-- -->

    on_memory_write:
        append(memory_reconstructions,
               {embedding_id = e_rep,
                blob_id = current_blob,
                ts = now,
                uncertainty = 0,
                trigger = 'initial'})

During retrieval the current embedding is the latest reconstruction (or
the base embedding when none exists). Since the 2026-10 ablation nothing
appends further versions, so for new stores it equals the stored
embedding.

Hydration resolves `memory.current_view` from the latest reconstruction
blob when one exists; otherwise it falls back to the immutable evidence
packets.

## Retrieval Competition (Removed)

Earlier builds made retrieved memories compete through lateral
inhibition, modeling retrieval-induced forgetting (Anderson, Bjork, and
Bjork 1994): winners within `inhibition_radius = lerp(0.5, 0.85, F)` of
the context suppressed losers by
`lerp(0.1, 0.01, T) × (1 − winning_activation)`, scaled by
`1 + 0.5 × NE_t`, and suppression recovered over `lerp(300, 1800, T)`
seconds through a lazily anchored log clock. The suppressed value
lowered `memories.strength`, but no retrieval, hydration or gating step
reads strength for long-term memories, so suppression never reached a
packet. Skipping the competition operation left every packet
byte-identical on LongMemEval-S (0/48) and Eval 1 (0/250). Its only
other effect was operational: the active-epoch bound was the sole source
of `Required` consolidation hints, so without it BEAM-10M conversation 7
consolidated once instead of 44 times and the sparse route stopped
recentering. Measured together with the in-conversation predecessor
change described above, all 20 BEAM packets changed, gold-turn recall
moved 0.103 → 0.110 (0.118 with the predecessor change alone) and
Haiku-judged sufficiency by −0.014 ± 0.048 (n = 18). On a 111,527-turn
chain of five BEAM-10M conversations the same build recalled 0.082 vs
0.062 of gold turns (13 cases better, 4 worse), judged +0.008 ± 0.040 (n
= 89), with 13.5% fewer instructions retired. The engine ablation study
(2026-10) removed the operation, the lazy recovery clock, its tables and
its active-epoch projection.

## Predictive Pre-activation (Removed)

Earlier builds kept a per-memory `pre_activation` value. An expectation
state (`x_pred_ema`) and recent context raised it, it decayed at
`lerp(0.7, 0.3, T)` per turn, and high surprise refreshed it faster.
After the hard cutover, graph retrieval no longer read the field. The
research ranking prior
`predictive_bonus = lerp(0.05, 0.20, FocusBias(F)) × lerp(1.0, 0.85, T) × pre_activation`
was never part of the production ranker. The engine ablation study
(2026-10) removed the operation, the column and its indexes. Skipping
the operation left every packet byte-identical: Eval 1 0/250,
LongMemEval-S 0/48, BEAM-10M 0/40. Instructions retired changed by -1%,
+3% and -1%, which is noise.

## Serial Position Effects (Removed)

Earlier builds modelled primacy, recency and von Restorff
distinctiveness (Murdock Jr 1962; Hunt 1995). They derived a per-signal
multiplier from knob-dependent windows
(`primacy_window = round(lerp(5, 2, F))`,
`recency_window = round(lerp(7, 3, F))`,
`von_restorff_multiplier = lerp(1.5, 3.0, S)` and others) and applied it
to memory-strength reinforcement. Memory strength was removed
(<a href="#sec-reinforcement" class="quarto-xref">Section 13</a>), so
nothing consumed the multiplier, and the engine ablation study (2026-10)
removed the two operations. Phase-1 ablation, as part of the
minor-feedback group, left every packet byte-identical: Eval 1 0/250,
LongMemEval-S 0/48, BEAM-10M 0/40.

## Emotional Consolidation and Cascade (Removed)

Earlier builds marked high-intensity, high-arousal memories as flashbulb
memories, following McGaugh (2004). A flashbulb memory got a half-life
bonus, detail suppression and gist-component settings. An emotional
cascade then propagated decayed intensity and half-life bonus to graph
neighbours within a knob-derived radius. The persisted outputs
(`flashbulb`, `emotional_intensity`, `half_life_bonus`,
`detail_suppression`, `gist_components`, `cascade_radius` and
`cascade_decay`) fed memory-strength decay and the cascade itself.
Retrieval never read them. The engine ablation study (2026-10) removed
both operations, their private metadata cache and the columns. Phase-1
ablation of the affect group left every packet byte-identical: Eval 1
0/250, LongMemEval-S 0/48, BEAM-10M 0/40. Sensitivity’s per-signal
emotion and mood projections, and their threshold deltas, are a separate
mechanism and remain.

## Synaptic Tagging and Capture (Removed)

Earlier builds tagged the temporal neighbours of a high-surprise or
high-arousal signal (`tag_strength`, `tag_expires_at`). A tagged memory
got a consolidation-score bonus of `lerp(0.10, 0.25, S) × tag_strength`.
That score was the tag’s only consumer. Clustering consolidation was
removed
(<a href="#sec-consolidation" class="quarto-xref">Section 12</a>), and
the engine ablation study (2026-10) removed tagging and its columns with
it. Phase-1 ablation left every packet byte-identical: Eval 1 0/250,
LongMemEval-S 0/48, BEAM-10M 0/40. The June research-branch result that
tagging removal hurt
(<a href="#sec-experimental" class="quarto-xref">Section 15</a>) was
measured on a branch where consolidation was live, so it does not apply
to the current engine.

## Pattern Separation Sparse Index and Procedural Lane (Removed)

Earlier builds kept two processor-local side tables keyed by a sparse
signature of an embedding (the signs of its
`RetrievalSparseKeySize(F,S,T)` largest-magnitude dimensions):
`index_store[sparse_key(e_rep)]` listed every stored memory id, and
`procedural_store[sparse_key(μ_acc)][memory_id]` held a routine value
`Q ← Q + value_update_gain × δ_reward × (1 − Q)` updated when a
retrieved memory was used. Neither table was persisted and no operation
read either one: retrieval seeds come from the dense vector surface
only. The tables were still journaled for rollback on every signal and
grew by one entry per stored memory. The engine ablation study (2026-10)
removed both tables, the sparse key and its knob with no packet change.
The public `procedural_enabled` configuration flag (C++ `Config`, C ABI
`cortext_config`, and the bindings) then controlled nothing and was
removed with the unused procedural seed and candidate-blend knobs. The
historical `procedural_proactive` ablation in
<a href="#sec-experimental" class="quarto-xref">Section 15</a> evaluated
an earlier research-branch path and was an exact null.

# Interrupt Gate and Streaming Integration

The interrupt gate controls when retrieved memories enter active context
during streaming generation. The gate balances novelty value against
disruption cost. The interrupt gate operates on memory-level context,
using centroids (μ_acc) rather than individual signal embeddings for
novelty and relevance computation. The interrupt thresholds, candidate
weights, maturity scaling, boundary multiplier, and refractory dynamics
in this section are named policies derived from the three knobs (F, S,
T). Eval-only ablation diagnostics exist only in builds compiled with
`CORTEXT_EXPERIMENT_HOOKS=ON`; default production builds ignore those
environment controls.

**Note:** As defined in
<a href="#sec-math" class="quarto-xref">Section 4</a>, all knob symbols
here use the midpoint‑biased values F̃ and S̃ (T is unmodified). For
affective gain, we use a lighter bias ( \_{affect} = (S, -0.06) ) to
preserve mid‑range affect modulation.

## Marginal Utility Computation

Novelty thresholds scale with knobs and refractory state:

    τ_novelty = lerp(0.12, 0.32, F) × (1 − 0.12S) × (1 + 0.25T)
    τ_mu = lerp(0.08, 0.18, F) × (1 − 0.4S) × (1 + 0.4T)
    retrieval_thresh(F) = lerp(0.12, 0.45, F)
    retrieval_thresh_interrupt(F,S) = retrieval_thresh(F) × (1 − 0.12S)
    affect_relax_coeff(S) = lerp(0.08, 0.28, S_affect)
    affect_gain(S) = lerp(1.0, 2.4, S_affect)
    [w_arousal, w_emotion, w_salience] = AffectDriveWeights(S)
    affect_drive = clamp(affect_gain(S) × (w_arousal × arousal + w_emotion × emotion_intensity + w_salience × salience), 0, 1)

Refractory dynamics suppress rapid successive interrupts:

Interrupt state (per stream):

    prev_x is unset on the first signal of a stream; set prev_x ← μ_acc after processing each signal
    first_step ← (prev_x is unset for this stream)
    if first_step:
        # cold start: do not update drift_accum
        drift_accum ← drift_accum
    else:
        drift_accum ← drift_accum + cosine_dist(μ_acc, prev_x)  # cumulative drift
    Δ ← drift_accum − drift_at_last_interrupt  # cumulative drift since last interrupt
    τ_refrac = lerp(24, 96, T) × lerp(1.4, 1.0, S)
    k_refrac = lerp(0.20, 0.05, T) × lerp(0.8, 1.2, F)
    M_refrac = 1.0 + k_refrac × exp(−Δ / τ_refrac)
    boundary_mult_eff = boundary_mult × (1 − 0.20S)  # relax non-boundary MU at higher S

On interrupt: set drift_at_last_interrupt ← drift_accum (resetting Δ to
0 for subsequent signals).

Effective thresholds incorporate refractory pressure:

    τ_novelty_eff = τ_novelty × M_refrac
    acc_maturity = clamp(n / win_coh(T), 0, 1)  # n = signals accumulated in current unit
    τ_mu_eff = τ_mu × M_refrac × (1 + (1 − acc_maturity) × lerp(0.4, 1.0, T))
    retrieval_thresh_eff = retrieval_thresh_interrupt(F,S) × (1 − affect_relax_coeff(S) × affect_drive)
    boundary_mult_eff = boundary_mult × (1 − 0.20S) × (1 − affect_relax_coeff(S) × affect_drive)

We experimented with scaling the interrupt threshold based on candidate
diversity, but it regressed abort rates on noisy chat and was removed.
The current gate relies on the affect‑relaxed threshold above.

## Marginal Utility Score

The marginal utility (MU) of a candidate memory combines five factors.
Context comparisons use memory centroids rather than individual signal
embeddings:

    # Context window contains recent memory centroids, not individual signals
    ctx_window ← recent_memory_centroids  # bounded deque of recent memory centroids (μ_acc)
    q_retrieval ← l2_normalize(μ_acc)  # accumulator centroid query
    wm_set ← {embedding(w) | w ∈ working_memory}
    included_set ← wm_set ∪ ctx_window  # context inclusion set for redundancy/coverage

Fallback: if included_set is empty, treat redundancy(·, included_set) =
0. If wm_set is empty, set overlap_star = −1.

    if |ctx_window| == 0:
        ctx_centroid ← q_retrieval  # fallback: use current unit centroid
    else:
        ctx_centroid ← mean(ctx_window)  # centroid of recent memory centroids

    weights_mu_raw = [lerp(0.40, 0.60, F),   # coverage gain
             lerp(0.35, 0.25, F),   # relevance
             lerp(0.15, 0.25, S),   # redundancy penalty
             lerp(0.15, 0.25, S),   # incoherence penalty
             lerp(0.20, 0.50, S)]   # surprise bonus
    [weight_cov, weight_rel, weight_red, weight_incoh, weight_surp] = normalize(weights_mu_raw)

    novelty_ctx = 1.0 if |ctx_window| == 0 else clamp((1 − max_{c ∈ ctx_window} cos(candidate, c)) / 2, 0, 1)
    surprise_bonus = surprisal_t × novelty_ctx × acc_maturity

    mu = weight_cov × coverage_gain(candidate | included_set) +
          weight_rel × map01(cos(candidate, ctx_centroid)) −
          weight_red × redundancy(candidate, included_set) −
          weight_incoh × (1 − coherence_struct_t) +
          weight_surp × surprise_bonus

With map01 applied, each MU term is in \[0, 1\], so μ is calibrated to
the same range as τ_mu.

## Gate Decision Logic

Duplicate suppression threshold:

    dup_thresh = lerp(0.985, 0.95, F) × (0.98 + 0.02T)
    K = round(lerp(10, 6, F))  # candidates to evaluate

Higher Focus lowers `dup_thresh`, making duplicate suppression stricter.

### Write Exclusion Filter

Memories stored during the current accumulation unit are excluded from
interrupt consideration to prevent self-triggering. Using the
accumulation start timestamp ensures all memories written within the
current unit are excluded:

    # Exclude memories written during current accumulation to prevent self-triggering
        write_exclusion_ts ← t_start  # start timestamp from Accumulator State section
    candidates_eligible ← {c ∈ candidates | c.created_at < write_exclusion_ts}

This filter is applied before novelty and marginal utility evaluation.
All subsequent gate logic operates on candidates_eligible rather than
the raw candidate set, preventing recursive triggering within a coherent
thought unit.

**Working memory exclusion (normative):** retrieval must also exclude
candidates that overlap with active working‑memory slots. This keeps
“what I just said” in working memory rather than re‑injecting it from
long‑term memory. The exclusion uses the same duplication threshold as
the gate:

    dup_thresh = lerp(0.985, 0.95, F) × (0.98 + 0.02T)
    candidates_eligible ← {c ∈ candidates_eligible | max_{w∈WM} cos(c, w) < dup_thresh}

Boundary-aware override permits lower-threshold interrupts at natural
boundaries:

    boundary_mult = lerp(1.3, 2.0, F) × lerp(1.1, 0.9, S) × lerp(1.4, 0.6, T)

The gate permits interrupt when:

    at_drift_boundary = should_flush  # boundary signal from the current accumulator
    if |candidates_eligible| == 0:
        allow_interrupt = false  # cold-start / empty-store fallback
    else:
        candidate_star = argmax_{c ∈ candidates_eligible} mu(c)
        mu_star = mu(candidate_star)
        rel_star = map01(cos(candidate_star, ctx_centroid))
        if |wm_set| == 0:
            overlap_star = −1.0
        else:
            overlap_star = max_{y ∈ wm_set} cos(candidate_star, y)
        if |ctx_window| == 0:
            novelty_star = 1.0
        else:
            max_cos = max_{c ∈ ctx_window} cos(candidate_star, c)  # in [−1, 1]
            novelty_star = clamp((1 − max_cos) / 2, 0, 1)
        allow_interrupt =
            (rel_star ≥ retrieval_thresh_eff) AND
            (novelty_star ≥ τ_novelty_eff OR mu_star ≥ τ_mu_eff) AND
            (overlap_star < dup_thresh) AND
            (at_drift_boundary OR mu_star ≥ boundary_mult_eff × τ_mu_eff)

This logic suppresses low-drift interrupts unless the marginal utility
substantially exceeds threshold, while permitting normal-threshold
interrupts at natural transition points.

## Interrupt-triggered Accumulator Abort

When the interrupt gate allows a retrieval outside a flush event, we
mark a **pending abort** for the current accumulator to avoid persisting
partial thoughts. On the next signal, we compare similarity to the
selected memory versus the current accumulator centroid. If the new
signal aligns more with the selected memory, we treat the interrupt as
**accepted** and drop the partial unit; otherwise we **resume** the
accumulator. This prevents “half‑utterances” from being stored when a
memory whisper redirects attention, while still allowing the speaker to
ignore the interrupt and continue seamlessly. This uses only embeddings
already present in the system (no new constants).

    if allow_interrupt AND NOT should_flush:
        pending_abort ← true
        pending_mem ← selected_candidate_embedding

    if pending_abort:
        sim_mem = cos(x_t, pending_mem)
        sim_acc = cos(x_t, μ_acc)
        margin = InterruptAbortAcceptanceMargin(F,S,T)
        if sim_mem > sim_acc + margin:        # accepted → drop partial unit
            reset_accumulator()
        pending_abort ← false

## Streaming Pacing

Streaming retrieval is gated by cumulative drift rate within the
accumulation unit. Retrieval checks trigger when drift exceeds
threshold, at boundaries, or for explicit ephemeral query ingress.
Ephemeral query ingress runs retrieval without granting write
permission, preserving the public “query without polluting memory”
contract. Retrieval uses q_retrieval (the accumulator centroid) captured
before any accumulator reset for this step. Retrieval returns a
candidate pool already filtered by write‑exclusion and WM‑overlap rules
and diversified (MMR‑style); the interrupt gate then applies
novelty/utility thresholds and redundancy penalties.

    # Pacing tracks drift within current memory formation

where cosine_dist(u, v) = 1 − cos(u, v).

    first_step ← (x_last_check is unset for this stream)  # MUST occur once per stream
    if first_step: x_last_check ← μ_acc; drift_acc_pacing ← 0
    drift_acc_pacing += cosine_dist(μ_acc, x_last_check)
    pacing_thresh(S) = lerp(0.3, 0.05, S)
    max_wait_drift(F) = lerp(1.2, 0.30, F)
    adjacent_window(F) = round(lerp(6, 1, F))

    since_last_s ← if last_retrieval_ts == 0 then +∞
                   else if now_ms() <= last_retrieval_ts then 0
                   else (now_ms() − last_retrieval_ts) / 1000
    min_gap_s ← adjacent_window(F) × dt_ema
    adjacent_ok ← (since_last_s ≥ min_gap_s)
    force_check ← (drift_acc_pacing > max_wait_drift(F))
    ephemeral_query ← (retention == Ephemeral)
    # Natural (default): force_boundary=false, force_write=false — episode algorithms.
    # Durable: force_boundary=true, force_write=true — explicit chat-turn commit.
    # Boundary: force_boundary=true, force_write=false — close unit; write gate still applies.

    # Retrieval triggered when drift exceeds threshold, at memory boundary,
    # ephemeral query ingress, or when drift exceeds max_wait_drift.
    # Adjacent-window throttling is bypassed on boundaries, force_check,
    # and ephemeral_query.
    if ephemeral_query OR (
       (drift_acc_pacing > pacing_thresh(S) OR should_flush OR force_check) AND
       (adjacent_ok OR should_flush OR force_check)):
        trigger_check(); x_last_check ← μ_acc; drift_acc_pacing ← 0; last_retrieval_ts ← now_ms()

High Sensitivity produces frequent checks triggered by small content
shifts; high Focus enforces strict drift limits. Memory boundaries
(<a href="#sec-boundary" class="quarto-xref">Section 5.1.3</a>) also
trigger retrieval checks to ensure context updates align with natural
thought transitions. Ephemeral query ingress bypasses pacing throttles
for the current signal only; storage remains controlled by the
retention-aware write gate.

# Consolidation as Route Maintenance

`Consolidate()` is now a maintenance call. It seals and recenters the
sparse retrieval route on a maintenance-only transaction path beside the
normal signal pipeline. Clustering consolidation is gone: shallow replay
into `ASSOCIATION` centroids and the typed-edge graph build were removed
in the 2026-10 engine ablation (see “Clustering Consolidation (Removed)”
below). Earlier hard cuts had already removed the generative
consolidation stack.

## Complementary Learning Systems Split

Cortext keeps the Complementary Learning Systems split at the level that
matters for the product:

-   **Fast episodic write path:** coherent memories are written
    immediately with source payloads, embeddings, timestamps, and graph
    metadata.
-   **Bounded live state:** working memory supports immediate continuity
    without scanning the whole store.
-   **Slow maintenance:** explicit `Consolidate()` calls, made during
    observed throughput troughs, seal the incremental sparse-route epoch
    and recenter its activation entry.

The engine no longer keeps a slow durable graph tier. Its clustering
produced no measurable quality gain (see below).

## Consolidation Triggers

Consolidation is external-only. The runtime never schedules it by
itself. Callers invoke `Consolidate()` or process a maintenance signal
with `Signal::force_consolidation=true`. This keeps production
scheduling under the application’s control while still letting the
engine derive an ordered urgency state from the three knobs.

Ordinary retained and ephemeral signals use the normal operation chain.
A forced request runs no per-signal operations; the processor only seals
and recenters the sparse retrieval route. Maintenance therefore does not
create an episode, signal, working-memory row, long-term memory, blob,
or searchable embedding, and it does not update normal throughput
observations. The single `consolidation_state` output is an ordered hint
for the caller; it does not schedule maintenance work by itself. It is
the more urgent of two classifiers. The first reads sparse-route growth.
Once a route is published, *C* memories written since the last
maintenance commit recommend a pass, and *C*×
`ConsolidationEscalationMultiplier(T)` require one, with *C* the route
capacity of
<a href="#sec-optimization" class="quarto-xref">Section 17</a>. A
maintenance commit resets the count. The second is the throughput
classifier below. Both are source- and modality-independent.

Earlier builds raised `Required` from the active epoch of
retrieval-induced forgetting. When retrieval competition was removed,
BEAM-10M conversation 7 consolidated once in 22,652 turns and its route
never recentered. The growth rule brought it back to 44 consolidations
and 43 recenters.

A caller can use the engine hints as the external scheduling rule. Since
the last committed forced request, including a coherent no-op with no
persisted cluster, the engine maintains the observed write-rate floor
*r*<sub>*f**l**o**o**r*</sub> and peak *r*<sub>*p**e**a**k*</sub>. The
peak is monotonic within that interval. A new lower observation lowers
the floor immediately; otherwise the floor moves upward with an update
factor 1/*W*<sub>*r**a**t**e*\_*s**e**c**o**n**d**s*</sub>(*T*). Before
a nonzero range has been observed, the throughput hint remains off. A
persisted initialization bit distinguishes a real zero-rate observation
from an uninitialized range, so a later positive sample updates from the
observed zero instead of overwriting it as bootstrap. Pre-migration rows
leave that bit false even when legacy rate-control ticks are positive,
because those ticks do not prove that a throughput range was observed;
the first post-migration sample establishes the floor and peak.

The knob-derived fraction is:

``` text
p(F,S,T) = clamp(
  0.5
  × lerp(1.10, 0.75, focus_bias(F))
  × lerp(0.80, 1.20, sensitivity_bias(S))
  × lerp(1.10, 0.80, T),
  0.20,
  0.70)
```

Higher focus and stability require a deeper trough; higher sensitivity
permits an earlier recommendation. The normalized throughput position
and ordered state are:

``` text
range = r_peak - r_floor
z = (m_rate - r_floor) / range
p_required(F,S,T) = p(F,S,T) / consolidation_escalation(T)
p_rearm(F,S,T) = p(F,S,T)
                 + (1 - p(F,S,T)) * T
p_drift(F,S,T) = 1 - p(F,S,T)

throughput_state =
  None,        if armed = false
               OR memories_since_consolidation <= 0
               OR range <= numeric_epsilon
               OR z > p(F,S,T)
  Required,    if z <= p_required(F,S,T)
  Recommended, otherwise
```

The required branch is evaluated first. Since
`consolidation_escalation(T)` is always greater than one, the required
boundary is strictly below the recommendation boundary after every floor
or peak update. Elapsed wall or event time and arbitrary durable memory
counts do not enter the throughput decision; backlog is only a
positive-work guard. The caller-facing state is `Required` whenever the
bounded active epoch has reached a safety ceiling, and otherwise equals
`throughput_state`. An explicit force request starts consolidation
independently and its output state is `None`. A committed forced request
acknowledges the active excursion by persisting `armed=false`, even when
no coherent cluster exists. The acknowledgment resets the next
event-derived range to the current rate: both floor and peak become that
rate. This prevents a historical spike from suppressing recommendations
after a throughput regime shift. Ordinary throughput observations
establish a new peak and rearm the classifier only after the observed
relative range is material:
(*r*<sub>*p**e**a**k*</sub> − *r*<sub>*f**l**o**o**r*</sub>)/max (*r*<sub>*p**e**a**k*</sub>, *ϵ*) ≥ *p*<sub>*d**r**i**f**t*</sub>.
Because *p* is clamped to \[0.20, 0.70\], the material drawdown remains
between 0.30 and 0.80 across the complete knob domain, including
*T* = 1. Once material, either recovery to
*z* ≥ *p*<sub>*r**e**a**r**m*</sub> or sustained degradation to
*z* ≤ *p* rearms the classifier. The lower branch prevents a gradually
falling write rate from remaining permanently silent after
acknowledgment, while the material-range guard keeps stable jitter from
repeatedly rearming. Failed transactions restore the prior armed state
and range. No count or elapsed-time horizon controls the reset. The
floor, peak, initialization, and armed fields are stored in the
singleton processor state row and participate in rollback, restart, and
ephemeral-query snapshot restoration. The engine can recommend
consolidation, but the API call is the only way to start it. `source_id`
and modality are never hint inputs or maintenance commands.

## Clustering Consolidation (Removed)

Earlier builds ran five operations on every maintenance signal:
`EvaluateConsolidation`, `ConsolidationGate` (scoring),
`ConsolidationCluster`, `ConsolidationShallow` and
`BuildGraphFromConsolidation`. They scored weak long-term memories as
`T·strength − F·redundancy + S·connectivity + T·stability + tag` against
`PeripheryCutoff(T)`. They clustered the candidates at
`MergeThreshold(F)`, wrote `ASSOCIATION` centroids with `derived_from`
edges, and derived `co_occurs`, `similar_to`, `causes`, `implies` and
episode edges.

The 2026-10 engine ablation found the pipeline inert at the default
knobs. Per-memory stability started at 1.0 and never fell, because
stability feedback updated only used memories and their raw-cosine gain
is practically never ≤ 0. That made `T·stability = 0.5` exceed the 0.115
floor, so scoring kept 0 of about 1,234 inputs per pass over 85k BEAM
turns. A fix that lowers the stability of retrieved-but-unused memories
made the pipeline fire. On LongMemEval-S it ran 112 passes, wrote 112
`ASSOCIATION` nodes, clustered 1,703 memories and wrote 11.1k typed
edges. On a 22.6k-turn BEAM-10M conversation it clustered 490 memories
into 12.8k edges. It still did not earn its keep:

-   Packet gold-turn recall moved -0.012 on S and -0.010 on Eval 1, and
    was unchanged on BEAM.
-   Haiku judge (rubric v2) sufficiency moved +0.012 ± 0.023 on S and
    -0.009 ± 0.023 on Eval 1. Misleading was flat.
-   Instructions retired rose 1–2%.
-   Firing also opened the strength-based eviction gate. With the
    storage floor at 0, eviction removed 94% of LongMemEval-S memories
    (22,390 of 23,868), and packet recall fell from 0.630 to 0.086
    (<a href="#sec-reinforcement" class="quarto-xref">Section 13</a>).

The pipeline, its knobs (`MergeThreshold`, `MinClusterSize`,
`ConsolidationMaxClusters`, graph-build thresholds and others) and its
operation-context plumbing were removed. Maintenance signals now run no
per-signal operations. The sparse route recenters on the active-signal
ring centroid, which was already the fallback whenever no cluster
centroid existed. At the default knobs that was every pass.

## Reconstruction Surface (Legacy Stores)

Reconsolidation and retrieval-time reconstruction were removed
(<a href="#sec-negative" class="quarto-xref">Section 14</a>). New stores
write one initial reconstruction per memory at storage time and never
append another, so their current surface equals the stored embedding.
Stores written by earlier builds may hold appended reconstructions.
Retrieval still reads the `current_memory_embeddings` surface first and
falls back to the latest reconstruction when a cached current row is
stale. Older signal rows, shared base embeddings and sibling memories
that point at an original embedding are preserved.

## What Was Removed

The following systems are no longer part of the runtime:

-   mode-selected semantic batch replay and public mode enums,
-   internal decoder-backed semantic operations,
-   vendored decoder runtime integration,
-   remote/local decoder adapter registry,
-   static taxonomy loading,
-   STM shadow graph promotion,
-   persistent confidence-monitoring state,
-   bitemporal fact assertion/cache/evidence tables,
-   fact-aware retrieval and eviction policies,
-   clustering consolidation (candidate scoring, clustering, shallow
    `ASSOCIATION` replay, typed-edge graph build),
-   the memory-lifecycle mechanisms of
    <a href="#sec-negative" class="quarto-xref">Section 14</a>: strength
    decay and eviction, reconsolidation, retrieval competition, synaptic
    tagging, emotional consolidation and the NE-like and DA-like
    modulators.

The removal is a hard cut, not a compatibility shim. New databases no
longer create the removed tables, public bindings no longer expose the
removed mode fields, and replay reports no longer emit the removed debug
metrics.

## Latency Discipline

Maintenance work is the sparse-route seal (bounded by the dirty epoch
delta) and one recenter. It does not scale with the number of stored
memories, source text length or multimodal payload size.

# Reinforcement, Decay and Feedback (Removed)

This section is a removal record. Earlier versions of the architecture
reinforced retrieved memories, decayed and evicted weak ones, and fed
retrieval outcomes back into the knob-derived controls. The 2026-10
ablation removed all of it
(<a href="#sec-negative" class="quarto-xref">Section 14</a>). What
remains is a usage counter that backs public fields. Nothing in this
section ranks, gates or deletes memories in the current engine.

## Memory Strength and Eviction (Removed)

Earlier builds kept multi-timescale strength traces per memory (fast,
medium, slow and ultra-slow, mixed into `strength`). Every retrieval
reinforced or decayed them, and memories whose effective strength fell
below `PeripheryCutoff(T)` were evicted. Eviction waited for two gates.
The first was a consolidation watermark: a memory had to predate
`last_consolidation_ts`. The second was a storage floor, by default 500
MB per file-backed store.

The engine ablation study (2026-10) removed the traces, the
reinforcement and decay update, eviction, its storage-pressure gate and
its `memory_evictions` audit log. At the default knobs eviction never
ran. The consolidation watermark was never set, because clustering
consolidation never produced a candidate
(<a href="#sec-consolidation" class="quarto-xref">Section 12</a>).
Phase-1 ablation of the strength update left every packet
byte-identical: Eval 1 0/250, LongMemEval-S 0/48, BEAM-10M 0/40. A fix
that made consolidation fire also opened the eviction gate. With the
storage floor at 0, eviction then deleted 94% of LongMemEval-S memories,
92% of Eval 1 memories and 99.7% of a 22.6k-turn BEAM-10M conversation.
Packet gold-turn recall fell from 0.630 to 0.086, from 0.587 to 0.084,
and from 0.103 to 0. Turn timestamps span weeks to months, and strength
decayed against a half-life of minutes to hours, so almost every memory
looked “weak” once the gate opened.

`strength` now holds only its knob-derived initial value; long-term
memories never update it, and working-memory slots persist their own
slot strength in it. A slim `RecordMemoryUsage` step keeps the retrieval
counters that hydrated memories expose: `retrieved_count`, `used_count`,
`last_used` and `last_access`. Store growth is unbounded again. A
bounded store needs a forgetting policy that is measured against recall
first.

## Contextual Gain (Definitions)

Contextual gain was the per-retrieval signal of the removed feedback
loops below. Usage detection still computes it, but no operation reads
it:

    contextual_gain_t(m) ← cos(μ_acc, m.embedding)    # if m was retrieved and used (range [−1, 1])
    contextual_gain_t(m) ← undefined                # otherwise

When undefined, treat contextual_gain_t(m) as 0 in downstream updates.
It is not persisted (the smoothed per-memory column was removed with
memory strength).

Per-memory strength initializes at creation:

    strength(m)_init = memory_initial_strength(F,S,T)

Retrieved vs used:

    retrieved(m) = memory returned by vector retrieval
    used(m) = retrieved(m) AND injected into active context after gate decisions

Contextual gain is only measured for used memories.

## Reinforcement Graph Maintenance

The normal interrupt gate injects at most one selected memory into
active context per dispatch. Production retrieval therefore does not
synthesize a pairwise *reinforces* edge from its broader retrieved
packet: packet co-occurrence is not evidence that two memories were used
together. This also prevents repeated noisy packets from turning
boilerplate into durable graph hubs.

Retrieval no longer traverses association edges, so existing
*reinforces* edges in older stores are inert.

## Causal Feedback Loop

The system tracks causal influence of retrieved memories on generation
quality through contextual gain—the semantic alignment (cosine
similarity) between the current signal embedding and the retrieved
memory embedding.

### Focus Feedback (Removed)

Earlier builds adjusted `weight_relevance` and `attention_width` from
the contextual gain of used memories. They also dilated the attention
width with drift and pulled it back toward its prior every turn. The
engine ablation study (2026-10) removed this step. With it switched off
as part of the minor-feedback group, packets stayed byte-identical: Eval
1 0/250, LongMemEval-S 0/48, BEAM-10M 0/40. `weight_relevance` and
`attention_width` now stay at their knob-derived priors unless
`UpdateFocus` moves them.

### Sensitivity Feedback (Removed)

Earlier builds nudged `weight_novelty` by
`η_base × (novelty_reward × contextual_gain − redundancy)` for used
memories, and persisted a per-memory `redundancy` that fed only the
clustering-consolidation score. The engine ablation study (2026-10)
removed the step, the `redundancy` column, and the never-written
`connectivity` column. With the step switched off as part of the
minor-feedback group, packets stayed byte-identical: Eval 1 0/250,
LongMemEval-S 0/48, BEAM-10M 0/40.

### Stability Feedback (Removed)

Earlier builds raised the per-memory `stability` of used memories with
positive contextual gain (`+0.05`) and lowered it otherwise (`× 0.95`).
The mean stability of used memories adjusted the half-life target
(<a href="#sec-stability-update" class="quarto-xref">Section 8.3.1</a>).
Stability had two consumers, the clustering-consolidation score and the
half-life adaptation, and both are gone. The engine ablation study
(2026-10) removed the step, `memories.stability` and the half-life
state. Contextual gain is a raw cosine and practically never ≤ 0, so
stability never fell below its initial value. As a result the
consolidation score’s `T·stability` term alone kept every memory above
the candidate floor
(<a href="#sec-consolidation" class="quarto-xref">Section 12</a>). With
the step switched off as part of the minor-feedback group, packets
stayed byte-identical: Eval 1 0/250, LongMemEval-S 0/48, BEAM-10M 0/40.

## Influence Feedback (Removed)

Earlier builds ran an influence-feedback operation (Algorithm 19). It
blended contextual gain with neutral generation-similarity and drift
terms into per-memory `influence`, `sustained_influence` and
`mean_influence` columns. Nothing downstream read those columns. The
engine ablation study (2026-10) removed the operation and its columns.
With it switched off as part of the minor-feedback group, packets stayed
byte-identical: Eval 1 0/250, LongMemEval-S 0/48, BEAM-10M 0/40.

# Negative Results: Bio-Inspired Lifecycle Mechanisms Under Long-Horizon Ablation

Earlier versions of this paper presented these memory-lifecycle
mechanisms as the architecture’s core:

-   forgetting through strength decay and eviction;
-   reconsolidation;
-   consolidation into abstractions;
-   retrieval-induced forgetting;
-   synaptic tagging;
-   NE- and DA-like neuromodulation;
-   emotional consolidation.

Each is well motivated by the memory literature. In October 2026 we
tested every one of them by leave-one-out on long evaluations, and none
survived. This section reports that result. We think it is the paper’s
most transferable finding: on long-horizon conversational recall, the
lifecycle mechanisms either never fired at their calibrated defaults, or
hurt recall once they did.

## Protocol

The study switched off one engine subsystem at a time, using a build
with experiment hooks. The hooks are not part of the release.

-   **Evaluations.**
    -   Eval 1: 250 cases over LoCoMo and LongMemEval conversations.
    -   LongMemEval-S: 48 cases.
    -   LongMemEval-M: every second episode (24), or an eight-episode
        subset of 39k turns.
    -   BEAM-10M: full-length conversations of 19k to 24k turns each.
    -   A 111,527-turn chain of five BEAM-10M conversations, for scale
        effects.
-   **Tier-1 metrics, judge-free.**
    -   Gold-turn recall in the packet and in the seed set.
    -   Query p95 and ingest time per turn.
    -   Instructions retired, peak RSS and store size.
-   **Tier-2 metrics.** A Haiku judge (rubric v2) scored only packets
    that changed, with a 95% interval on each “switched off minus on”
    delta.
-   **Decision rule.** A subsystem was removed when switching it off
    measurably helped quality, or when it left every packet
    byte-identical with no cost benefit.
-   **Recheck.** Each removal landed as its own commit. That commit had
    to build, pass the full test suite, and give byte-identical packets
    against its parent on LongMemEval-S, Eval 1 and BEAM-10M
    conversation 7.

The full tables are in
<a href="#sec-experimental" class="quarto-xref">Section 15</a>. The
headline effects follow.

## Forgetting Was Inert, Then Catastrophic

Multi-timescale strength traces decayed against half-lives of minutes to
hours, and memories below `PeripheryCutoff(T)` were evicted
(<a href="#sec-reinforcement" class="quarto-xref">Section 13</a>).
Eviction waited for two gates: a consolidation watermark and a 500 MB
storage floor. At the default knobs the watermark was never set, so
eviction never ran, and ablating the strength update changed 0/250, 0/48
and 0/40 packets.

A fix that made consolidation fire also opened the eviction gate. With
the storage floor at 0, eviction then deleted:

<table>
<thead>
<tr>
<th>Store</th>
<th style="text-align: right;">Memories evicted</th>
<th style="text-align: right;">Packet gold-turn recall</th>
</tr>
</thead>
<tbody>
<tr>
<td>LongMemEval-S</td>
<td style="text-align: right;">94% (22,390 of 23,868)</td>
<td style="text-align: right;">0.630 → 0.086</td>
</tr>
<tr>
<td>Eval 1</td>
<td style="text-align: right;">92%</td>
<td style="text-align: right;">0.587 → 0.084</td>
</tr>
<tr>
<td>BEAM-10M conversation 7 (22.6k turns)</td>
<td style="text-align: right;">99.7%</td>
<td style="text-align: right;">0.103 → 0</td>
</tr>
</tbody>
</table>

The cause is a mismatch of time scales. Conversation timestamps span
weeks to months, and a half-life of minutes makes almost every memory
look weak once the gate opens. The engine now has no forgetting policy
and its store grows without bound. A bounded store needs a forgetting
policy that is measured against recall first.

## Consolidation Into Abstractions Never Fired, and Did Not Help When Forced

Clustering consolidation scored weak memories, clustered them into
`ASSOCIATION` centroids with `derived_from` edges, and built typed edges
(`co_occurs`, `similar_to`, `causes`, `implies`). At the default knobs
it was inert: per-memory stability started at 1.0 and never fell, so the
score’s *T* ⋅ stability = 0.5 term alone exceeded the 0.115 candidate
floor. Scoring kept 0 of about 1,234 inputs per pass over 85k BEAM
turns.

A stability fix made it fire. On LongMemEval-S it ran 112 passes, wrote
112 `ASSOCIATION` nodes, clustered 1,703 memories and wrote 11.1k typed
edges. On BEAM-10M conversation 7 it clustered 490 memories into 12.8k
edges. It still did not earn its keep:

-   Packet recall moved −0.012 (S), −0.010 (Eval 1) and 0 (BEAM).
-   Judged sufficiency moved +0.012 ± 0.023 and −0.009 ± 0.023.
-   Instructions retired rose 1–2%.
-   Firing opened the eviction gate above.

Clustering, graph expansion (whose only traversable edges it wrote) and
linked-source hydration were removed together
(<a href="#sec-consolidation" class="quarto-xref">Section 12</a>).

## Reconsolidation Was Neutral to Harmful

Reconsolidation drifted a retrieved memory’s embedding toward the
current context during a lability window, and retrieval re-encoded its
top candidates toward the query
(<a href="#sec-advanced" class="quarto-xref">Section 10</a>). Switching
both off together:

<table>
<thead>
<tr>
<th>Evaluation</th>
<th style="text-align: right;">Packet recall (on → off)</th>
<th style="text-align: right;">Judged sufficiency delta</th>
</tr>
</thead>
<tbody>
<tr>
<td>LongMemEval-S</td>
<td style="text-align: right;">0.642 → 0.642</td>
<td style="text-align: right;">0.000</td>
</tr>
<tr>
<td>Eval 1</td>
<td style="text-align: right;">0.597 → 0.611 (6 better, 1 worse)</td>
<td style="text-align: right;">+0.010 ± 0.016</td>
</tr>
<tr>
<td>BEAM-10M conversation 7</td>
<td style="text-align: right;">0.103 → 0.110</td>
<td style="text-align: right;">+0.014 ± 0.027</td>
</tr>
<tr>
<td>LongMemEval-M subset (39k turns)</td>
<td style="text-align: right;">0.462 → 0.462</td>
<td style="text-align: right;">0.000</td>
</tr>
</tbody>
</table>

Instructions fell 5–11% and stores shrank 12–23%. Each movement source
alone was also neutral. An earlier phase-1 arm had measured harm from
removing “embedding movement”. That arm had also switched off the
reconstruction representation, so its delta came from the representation
change rather than from movement.

## Retrieval-Induced Forgetting Never Reached a Packet

Lateral inhibition suppressed retrieval losers in `memories.strength`
(Anderson, Bjork, and Bjork 1994). No retrieval, hydration or gating
step reads strength for long-term memories, so suppression changed 0/48
and 0/250 packets. Its only effect was operational: its active-epoch
bound was the sole source of `Required` consolidation hints, which the
engine now derives from sparse-route growth instead
(<a href="#sec-optimization" class="quarto-xref">Section 17</a>). On the
111,527-turn chain, the build without it recalled 0.082 of gold turns
against 0.062 (13 cases better, 4 worse), judged +0.008 ± 0.040, with
13.5% fewer instructions.

## Filtering Out “Obsolete” Memories Hurt

Belief revision used to remove superseded memories from the seed set.
Removing that exclusion helped:

-   Packet recall rose by +0.025 (S), +0.018 (Eval 1) and +0.066
    (BEAM-10M conversation 7).
-   The share of gold turns among the seeds rose by +0.136, +0.158 and
    +0.081.

In long conversations an earlier version of a fact is often what a
question asks about. The `supersedes` edges stayed, and now drive
predecessor and successor hydration instead
(<a href="#sec-hydration" class="quarto-xref">Section 7</a>).

The same pattern held for the write gate. A score threshold that
discarded finished units dropped 7–11% of gold evidence turns under
Natural ingest. Storing every closed unit raised retrieved-only packet
recall by +0.086 (LongMemEval-S) and +0.054 (Eval 1) for 6% more store
bytes (<a href="#sec-write-pacing" class="quarto-xref">Section 5.1</a>).

## Most Mechanisms Were Inert at the Defaults

These subsystems left every packet byte-identical when switched off
(Eval 1 0/250, LongMemEval-S 0/48, BEAM-10M 0/40 or 0/20, and for the
oscillator MSC 0/39). Each was removed
(<a href="#sec-experimental" class="quarto-xref">Section 15</a>):

-   meta-learned knob priors;
-   predictive pre-activation;
-   influence, focus, sensitivity and stability feedback;
-   serial-position effects;
-   synaptic tagging, whose only consumer was the consolidation score;
-   emotional consolidation and the emotional cascade;
-   the DA-like modulator and reward prediction error;
-   the encode/retrieve oscillator;
-   the temporal tie-break in scoring;
-   Soft Anchors;
-   the pattern-separation sparse index and procedural value table.

The NE-like modulator and its effect scales changed a few packets (5/48,
32/250, 7/20), with recall within one case and judged deltas of 0.000,
−0.001 ± 0.006 and +0.028 ± 0.054. Working-memory rehearsal reordered
working-memory slots in 12/48 and 33/250 packets, with recall unchanged.
Both were removed.

## Lesson: Re-Measure on the Shipped Engine, and Diff Packets Before Judging

The October results overturn several earlier research-branch verdicts.
Those verdicts came from blind-judge screens over short corpora
(<a href="#sec-experimental" class="quarto-xref">Section 15</a>), and
they failed for two different reasons.

<table>
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<thead>
<tr>
<th>Mechanism</th>
<th>Earlier verdict</th>
<th>October result</th>
<th>Why they differ</th>
</tr>
</thead>
<tbody>
<tr>
<td>encode/retrieve oscillator</td>
<td>removal hurts: −0.49 sufficiency (July, 3 reps)</td>
<td>same MSC protocol rerun: all 39 probe packets byte-identical without
it, under Durable and Natural</td>
<td>judge variance</td>
</tr>
<tr>
<td>synaptic tagging</td>
<td>removal hurts: −0.20 (July, 3 reps)</td>
<td>0/40 packets changed</td>
<td>July branch had live consolidation, the tag’s only consumer</td>
</tr>
<tr>
<td>graph expansion</td>
<td>unresolved, leaning negative (long-horizon sweep)</td>
<td>0/40 packets changed</td>
<td>only non-traversable <code>supersedes</code> edges existed on the
current engine</td>
</tr>
<tr>
<td>daily consolidation</td>
<td>retained by default (long-horizon sweep)</td>
<td>inert at defaults; −0.012 / −0.010 recall when forced</td>
<td>the current engine never fired it</td>
</tr>
<tr>
<td>emotion/mood threshold cascade</td>
<td>measured harm: −0.43, loses on all three paired repetitions
(long-horizon sweep)</td>
<td>emotion and mood threshold deltas removed with the dynamic
threshold, packets byte-identical</td>
<td>measured while the threshold still gated writes</td>
</tr>
</tbody>
</table>

A sufficiency delta cannot come from identical packets. The July
oscillator delta, measured on the same protocol, was therefore judge
variance, which the July confirmation itself put at up to 0.32 between
repetitions. The other rows were measured on a research-branch engine
whose mechanisms interacted differently.

We now apply two rules:

1.  Ablate on the engine that ships.
2.  Before spending judge budget on an arm, diff its packets against the
    control. Judge only the cases that changed, and report how many
    changed.

## What Survived

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr>
<th>Mechanism</th>
<th>Evidence for keeping it</th>
</tr>
</thead>
<tbody>
<tr>
<td>boundary segmentation, store every closed unit</td>
<td>write-gate removal above</td>
</tr>
<tr>
<td>contiguity hydration</td>
<td>off: −0.06 to −0.09 packet recall</td>
</tr>
<tr>
<td><code>supersedes</code> edges and predecessor hydration</td>
<td>off: −0.037 / −0.025 / −0.015 recall; judged −0.036 ± 0.052 (S)</td>
</tr>
<tr>
<td>working memory</td>
<td>off: −0.012 / −0.015 recall; MemLens image evidence −0.036 ±
0.029</td>
</tr>
<tr>
<td>SQLite sparse route</td>
<td>off at 111k turns: +26% instructions, ingest 95 → 170 ms/turn</td>
</tr>
<tr>
<td>usage counters</td>
<td>feed the public <code>retrieved_count</code> /
<code>used_count</code> fields</td>
</tr>
</tbody>
</table>

After all round-3 removals, measured against the merged pruned engine:

<table>
<colgroup>
<col style="width: 20%" />
<col style="width: 26%" />
<col style="width: 26%" />
<col style="width: 26%" />
</colgroup>
<thead>
<tr>
<th></th>
<th style="text-align: right;">LongMemEval-S</th>
<th style="text-align: right;">Eval 1</th>
<th style="text-align: right;">BEAM-10M conv 7</th>
</tr>
</thead>
<tbody>
<tr>
<td>packet gold-turn recall</td>
<td style="text-align: right;">0.642 → 0.691</td>
<td style="text-align: right;">0.597 → 0.644</td>
<td style="text-align: right;">0.103 → 0.118</td>
</tr>
<tr>
<td>judged sufficiency delta</td>
<td style="text-align: right;">+0.060 ± 0.057</td>
<td style="text-align: right;">+0.039 ± 0.021</td>
<td style="text-align: right;">+0.014 ± 0.048</td>
</tr>
<tr>
<td>instructions retired</td>
<td style="text-align: right;">−12%</td>
<td style="text-align: right;">−12%</td>
<td style="text-align: right;">−24%</td>
</tr>
<tr>
<td>peak RSS</td>
<td style="text-align: right;">−14%</td>
<td style="text-align: right;">−3%</td>
<td style="text-align: right;">−17%</td>
</tr>
</tbody>
</table>

Engine source (`src/` and `include/`) shrank by a net 7,075 lines (PR
#28). The v5 recall mechanisms of
<a href="#sec-retrieval" class="quarto-xref">Section 6</a> and
<a href="#sec-hydration" class="quarto-xref">Section 7</a> were then
hill-climbed on this pruned base.

## Scope of the Claim

These are ablations of one engine on conversational long-horizon recall,
at its calibrated default knobs. They do not show that forgetting,
consolidation or reconsolidation are useless in general. They show three
narrower things:

-   The specific forms implemented here never fired at the defaults.
-   When made to fire, they cost recall or cost nothing.
-   Short-horizon judged screens were not able to tell.

A bounded store still needs a forgetting policy. Any future one should
be gated on packet recall over month-scale timestamps before it ships.

A follow-up prune then removed what the write-path change had orphaned:

-   the dynamic threshold controller and its 19 knobs;
-   the spike bypass, which never fired on the parity sets;
-   the 6D mood integrator, whose only effect was a threshold delta;
-   stored source-reliability metadata and the per-memory context
    column;
-   three dead configuration flags.

Every removal left packets byte-identical
(<a href="#sec-overview" class="quarto-xref">Section 3</a>).

Two per-signal pieces of the original affect and neuromodulation design
remain, and both are live:

-   the ACh-like encode/retrieve bias, which paces streaming retrieval
    checks;
-   the emotion projection, whose valence and arousal feed the composite
    score and the interrupt gate
    (<a href="#sec-core-adaptation" class="quarto-xref">Section 8</a>).
    Switching it off changes 171 of 1,986 LoCoMo retrieved sets.

The composite score itself is live too
(<a href="#sec-structural-metrics" class="quarto-xref">Section 9</a>).
It is not a write gate.

# Experimental Results

This section records the evidence for the current engine, newest first.
It opens with the October 2026 v5 benchmarks, then the 2026-10 engine
ablation (summarized in
<a href="#sec-negative" class="quarto-xref">Section 14</a>), then the
earlier hard-cut, release and research-branch records. Older adapter-era
and internal-decoder experiments are preserved in git history, not in
the current manuscript, because those systems are no longer part of the
runtime being evaluated. Results older than October 2026 were measured
on engines that still contained mechanisms
<a href="#sec-negative" class="quarto-xref">Section 14</a> later
removed; each such subsection says so.

## Current Evaluation Scope

The evaluated system is the v5 engine described in
<a href="#sec-overview" class="quarto-xref">Section 3</a>. It contains:

-   online boundary segmentation, storage of every closed unit, working
    memory and `supersedes` edges;
-   multi-granularity hybrid retrieval with late-interaction reranking
    (<a href="#sec-retrieval" class="quarto-xref">Section 6</a>);
-   hydration and evidence fitting
    (<a href="#sec-hydration" class="quarto-xref">Section 7</a>);
-   explicit maintenance (sparse-route seal and recenter); and
-   C and C++ entry points in this repository, with Python, Go, Dart,
    JavaScript and WebAssembly packages in sibling repositories,
    exposing only the retained processing, retrieval, consolidation,
    reset and embedding APIs.

## v5 End-to-End Benchmarks

These results are copied from the README Benchmarks section and the v5
pull request (PR #28). They were measured in October 2026 on the v5
engine, macOS arm64.

**Protocol.** Each benchmark’s conversations were replayed into the
engine as they happened: live ingest, Natural retention, and
consolidation when the engine recommended it. Each question was then
asked against the final store. The retrieved context went to an answer
model and a judge, using each benchmark’s own published QA prompts. The
answer model and judge were both `gpt-6.1-sol` at low reasoning effort,
and packets were capped at a 34,800-byte evidence budget (about 8k
tokens). No LLM call was made to store or recall.

### Reranker Training Overlap

The late-interaction reranker
(<a href="#sec-retrieval" class="quarto-xref">Section 6</a>) was trained
on pools from some of these benchmarks. Scores on those benchmarks are
not zero-shot. The table separates them. Held-out splits are 20% of
LongMemEval and MemLens questions, 30% of BEAM 1M and Eval 1
conversations, and one DolphinBench persona. The 108-question MemLens
image evaluation set was excluded entirely.

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr>
<th>Benchmark</th>
<th>Reranker training overlap</th>
<th>Zero-shot evidence</th>
</tr>
</thead>
<tbody>
<tr>
<td>LoCoMo (1,540 q)</td>
<td>none: excluded from training (CC-BY-NC)</td>
<td>whole benchmark</td>
</tr>
<tr>
<td>BEAM 100K and 10M</td>
<td>none: never trained on</td>
<td>whole benchmark</td>
</tr>
<tr>
<td>BEAM 1M</td>
<td>70% of conversations trained on</td>
<td>held-out 30% only</td>
</tr>
<tr>
<td>LongMemEval-S (500 q)</td>
<td>trained on; 20% of LongMemEval questions held out</td>
<td>115 unseen questions</td>
</tr>
<tr>
<td>LongMemEval-M (100 q)</td>
<td>same LongMemEval questions as S</td>
<td>not separately reported</td>
</tr>
<tr>
<td>DolphinBench (600 tests)</td>
<td>trained on, one persona held out</td>
<td>not separately reported</td>
</tr>
<tr>
<td>MemLens agent subset (195 q)</td>
<td>trained on; 20% of MemLens questions held out</td>
<td>56 unseen questions</td>
</tr>
<tr>
<td>Eval 1 (internal, 250 q)</td>
<td>LongMemEval conversations, 70% trained on; LoCoMo conversations
never</td>
<td>held-out conversations</td>
</tr>
</tbody>
</table>

### Zero-Shot Results

<table>
<colgroup>
<col style="width: 30%" />
<col style="width: 40%" />
<col style="width: 30%" />
</colgroup>
<thead>
<tr>
<th>Benchmark</th>
<th style="text-align: right;">Cortext</th>
<th>Comparison</th>
</tr>
</thead>
<tbody>
<tr>
<td>LoCoMo, same pipeline (1,540 q)</td>
<td style="text-align: right;">86.2% (87.9% on corrected keys)</td>
<td>Mem0 OSS 2.2.1 89.0%, Graphiti 0.31.1 81.0% under the same answerer,
judge and budget</td>
</tr>
<tr>
<td>BEAM 100K / 10M, ExaBase protocol</td>
<td style="text-align: right;">83.8 / 70.8</td>
<td>published: past.dev 92.1 / 85.0, Exabase M-1 76.9 / 68.0</td>
</tr>
<tr>
<td>BEAM 100K / 10M, AMB protocol</td>
<td style="text-align: right;">72.2 / 53.6</td>
<td>published: Hindsight 73.4 / 64.1, Honcho 63.0 / 40.6</td>
</tr>
<tr>
<td>LongMemEval-S, unseen 115 q</td>
<td style="text-align: right;">96.5%</td>
<td>n/a</td>
</tr>
<tr>
<td>MemLens, unseen 56 q</td>
<td style="text-align: right;">78.6%</td>
<td>n/a</td>
</tr>
</tbody>
</table>

### Results on Benchmarks That Overlap Reranker Training

<table>
<colgroup>
<col style="width: 30%" />
<col style="width: 40%" />
<col style="width: 30%" />
</colgroup>
<thead>
<tr>
<th>Benchmark</th>
<th style="text-align: right;">Cortext</th>
<th>Published (own protocols, not re-run)</th>
</tr>
</thead>
<tbody>
<tr>
<td>LongMemEval-S (500 q)</td>
<td style="text-align: right;">93.8%</td>
<td>Mem0 94.4, Hindsight 91.4, EmergenceMem 86.0, Supermemory 85.2, Zep
71.2</td>
</tr>
<tr>
<td>LongMemEval-M (100 q)</td>
<td style="text-align: right;">78.0%</td>
<td>n/a</td>
</tr>
<tr>
<td>BEAM 1M, ExaBase / AMB protocol</td>
<td style="text-align: right;">80.0 / 70.4</td>
<td>past.dev 90.7, Exabase M-1 75.0 / Hindsight 73.9, Honcho 63.1, Mem0
64.1</td>
</tr>
<tr>
<td>DolphinBench (600 tests, agent with memory tools)</td>
<td style="text-align: right;">87.0%</td>
<td>Mem0 + Hermes 70.7, Hindsight + Hermes 69.5, Honcho + Hermes 68.5
(published rows used a <code>gpt-5.6-luna</code> agent)</td>
</tr>
<tr>
<td>MemLens agent subset (195 q, 32k, images)</td>
<td style="text-align: right;">72.3%</td>
<td>Mem0 43.2 (text captions), MemOS 36.5, MemAgent-7B 32.8</td>
</tr>
</tbody>
</table>

The competitor numbers in the last column of both tables are the
vendors’ or papers’ own, each with its own answer model, judge and
protocol. Only the LoCoMo comparison was run under one shared pipeline,
so read the other gaps as directional.

On LoCoMo, Mem0’s 2.8-point lead comes almost entirely from multi-hop
(91.8% against 83.3%) and temporal questions (89.1% against 85.1%).
There its write-time LLM writes consolidated, dated facts. Cortext made
no write-time model call, against 544 calls and 5.7M tokens for Mem0 and
4,151 calls and 14.4M tokens for Graphiti. It recalled at 9.7 / 13.7 ms
p50 / p95, against 167 / 250 ms and 143 / 157 ms. Cortext led on
open-domain questions (64.6% against 62.5% and 61.5%).

### Stage Measurements

PR #28 measured each v5 change against its own predecessor, with both
arms graded by one judge. Comparisons are per stage, not one end-to-end
base-against-final run, and the judges changed across the program
(Haiku, then `gpt-5.6-luna`, then `gpt-6.1-sol` low). Packet gold-turn
recall is judge-free.

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr>
<th>Stage</th>
<th>Evaluation</th>
<th>Result</th>
</tr>
</thead>
<tbody>
<tr>
<td>FTS5 keyword channel</td>
<td>Eval 1 (250 q)</td>
<td>recall 0.597 → 0.809; sufficiency train +0.149 ± 0.063, test +0.218
± 0.059 (Haiku)</td>
</tr>
<tr>
<td>FTS5 keyword channel</td>
<td>LongMemEval-S (48) / M (48 haystacks)</td>
<td>recall 0.642 → 0.840 / 0.407 → 0.679; sufficiency +0.185 ± 0.104 /
+0.292 ± 0.111</td>
</tr>
<tr>
<td>store every closed unit</td>
<td>Eval 1</td>
<td>sufficiency +0.047 ± 0.025 (Natural vs Durable)</td>
</tr>
<tr>
<td>multimodal R1–R4</td>
<td>MemLens (108 q, image evidence)</td>
<td>0.558 → 0.744 (ceiling 0.867); misleading 0.028 → 0.009</td>
</tr>
<tr>
<td>v5 engine vs base</td>
<td>BEAM e2e 100K / 1M / 10M</td>
<td>0.773 / 0.702 / 0.617 → 0.816 / 0.778 / 0.667</td>
</tr>
<tr>
<td>late-interaction rerank</td>
<td>BEAM e2e 100K / 1M / 10M</td>
<td>0.816 / 0.778 / 0.667 → 0.843 / 0.799 / 0.699; 100K and 10M never
trained on</td>
</tr>
<tr>
<td>late-interaction rerank</td>
<td>Eval 1 judged</td>
<td>0.955 → 0.973 (+0.018 ± 0.018); reranker-unseen 0.937 → 0.968</td>
</tr>
<tr>
<td>clock fix</td>
<td>DolphinBench (600)</td>
<td>78.8% → 84.0% (+5.2 ± 3.4 pp); 31–365-day slice 0.514 → 0.844</td>
</tr>
<tr>
<td>late-interaction rerank (pre-clock-fix base)</td>
<td>DolphinBench (600)</td>
<td>78.8% → 81.2%</td>
</tr>
</tbody>
</table>

The DolphinBench reranker run predates the clock fix. Final `main`,
which has both the clock fix and the reranker, was not rerun on the full
600 for PR #28; the 87.0% above comes from the later README scoreboard
run. The FTS5 rows compare the v2 engine with the pruned engine at
47c18ec9, so they include the other v2 changes;
<a href="#sec-implementation" class="quarto-xref">Section 16</a> reports
the channel-only effect (Eval 1 recall 0.60 → 0.79).

The evaluated system does not contain internal text-generation backends,
adapter registries, decoder-backed semantic batch operations, static
taxonomy loading, bitemporal fact tables, shadow STM graph promotion, or
persistent confidence-monitoring state. Blind-judge and release-protocol
tools that remain under `scripts/` and `tools/` are explicit experiment
harnesses, not linked runtime paths. Future quality studies should stay
in those harnesses or outside the repository; they must not become
hidden production options.

## Hard-Cutover Verification

The primary current experiment is a structural one: remove the old
semantic batch path and prove that the remaining engine still builds,
tests, and exposes only the intended surfaces. The verification pass
used the following gates.

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr>
<th>Gate</th>
<th>Evidence</th>
</tr>
</thead>
<tbody>
<tr>
<td>runtime code removal</td>
<td>repository scans over <code>include/</code>, <code>src/</code>,
<code>bindings/</code>, <code>CMakeLists.txt</code>, and runtime
examples find no retained internal decoder, adapter-registry, semantic
batch, static taxonomy, fact-layer, or processor-local STM-shadow
production path; experiment harnesses under <code>scripts/</code> and
<code>tools/</code> are explicit non-runtime tooling</td>
</tr>
<tr>
<td>dependency removal</td>
<td><code>.gitmodules</code> and the worktree no longer contain the
removed decoder/runtime submodules; the remaining third-party tree is
limited to SQLite/vector/object storage, opt-out OpenTelemetry, and
build/test dependencies</td>
</tr>
<tr>
<td>API removal</td>
<td>public consolidation mode enums and mode-specific C API calls are
gone; bindings no longer expose the removed mode fields</td>
</tr>
<tr>
<td>current behavior</td>
<td><code>Cortext::Consolidate()</code> and
<code>Signal::force_consolidation</code> drive the one retained
consolidation path; graph retrieval returns source-backed memory context
from vector evidence</td>
</tr>
<tr>
<td>paper consistency</td>
<td>this section, <a href="#sec-consolidation"
class="quarto-xref">Section 12</a>, <a href="#sec-implementation"
class="quarto-xref">Section 16</a>, and <a href="#sec-optimization"
class="quarto-xref">Section 17</a> describe the current branch rather
than deleted adapter-era workflows</td>
</tr>
</tbody>
</table>

This is intentionally a hard cut. No compatibility aliases, deprecated
enum values, dormant operation classes, hidden configuration toggles, or
legacy comments remain to imply that the deleted runtime paths can still
be selected.

## Regression Results

The following commands and counts are the historical hard-cutover
verification record:

``` bash
cmake -S . -B build/codex-hardcut-check \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build build/codex-hardcut-check -j

./build/codex-hardcut-check/tests/cortext_tests '~[cortext]~[aist]'

./build/codex-hardcut-check/tests/cortext_tests \
  '[operations][graph][retrieval],[operations][constructive_recall][graph],[state_persistence][working_memory][decay]'

./build/codex-hardcut-check/tests/cortext_tests \
  'AIST Matryoshka truncation normalizes output'
```

The broad filtered suite passed **429 test cases** and **2,489
assertions**. The targeted graph/reconstruction/state tests passed **5
test cases** and **16 assertions**. The AIST embedding unit passed **1
test case** and **2 assertions**.

An examples-enabled build also completed after configuring with
`CORTEXT_BUILD_EXAMPLES=ON`.

After the July 12 safety fixes documented below, the current
warnings-as-errors debug tree passed the complete default native suite:
**509 test cases** and **4,510 assertions** in 483.53 seconds. The
source lists 510 cases; one hidden Catch2 case is intentionally excluded
from the default invocation. Focused regression, store/binding,
performance, and model-backed C API filters also passed, as did Go tests
and vet, Dart analysis and tests, Python validation, the Node addon
build, and the WebAssembly bundle build.

On 2026-07-07, the belief-revision supersession patch added focused
modality-agnostic regression/evals. The storage probe uses an
image-modality signal and verifies that the new memory writes a directed
`supersedes` edge to an older memory whose embedding sits in the
knob-derived supersession band, without inspecting payload text. The
public API probe runs durable stale facts, durable corrections, and
ephemeral recall queries through `ProcessTextAt` at default knobs; it
verifies that `supersedes` edges are written and corrected vet/pill
facts outrank the stale facts. The retrieval probe presents an older
exact embedding match and a newer less-exact replacement connected by
`supersedes`; graph retrieval ranks the replacement first, keeps the
older memory in the packet, and records a nonzero supersession penalty
in the activation ledger.

``` bash
./build/tests/cortext_tests \
  "MemoryStorage writes modality-agnostic supersedes edges"

./build/tests/cortext_tests \
  "Cortext durable corrections supersede stale facts at default knobs"

./build/tests/cortext_tests \
  "Graph retrieval demotes superseded stale memories"
```

Both focused probes passed locally. This is a correctness eval for the
stale fact failure mode, not a broad retrieval-quality claim.

The patch was also re-screened with the local synthetic mechanism eval
at default knobs:

``` bash
scripts/run_mechanism_eval.sh logs/mechanism_eval_supersession_20260707T151403Z
```

Both `source_conf_on` and `source_conf_off` arms completed with return
code 0 over 19 turns and 19 writes. Their retrieval metrics were
unchanged between arms: retrieval turn rate 0.947368, average retrieved
candidates 7.94444, retrieval overlap mean 0.0555556, and retrieval
context-overlap mean 0.341005. The resulting databases each contained
115 `reinforces` edges and 34 `supersedes` edges, confirming that the
supersession write path is active in a normal harness run. This is a
local no-regression smoke screen for retrieval mechanics, not a
blind-judge quality result.

The mechanism-eval directory above was local run output and is not
checked into this repository; the numerical result is retained as a
historical run record, not a checkout-reproducible aggregate artifact.

On 2026-07-14, the public retention path was re-screened after making
`Retention::Ephemeral` transactionally read-only. The focused regression
suite verified that public ephemeral queries retrieve durable API
memories, expose a boundary packet, and leave memories, signals, current
embeddings, associations, reconstructions, retrieval/use counters,
lability, suppression, strength, episodes, and persisted processor state
unchanged. A prior CLI smoke test stored “The garage door code is
8841.”, ran an ephemeral `recall "garage door code"`, retrieved the
stored memory, and confirmed the SQLite memory/signal counts were
unchanged before and after recall.

The same 2026-07-14 screen exercised a deterministic 512-message
agent-style corpus with repeated tool acknowledgements and sixteen
buried copies of the sentence “The same-session recall test returned
teal.” A packaged JavaScript ephemeral query returned four family
representatives, with needle-bearing memories at ranks 0 and 1. Every
returned item exposed numeric `relevance` and `composite_score`; the
runtime object shape remained unchanged and did not add the undocumented
`rel` alias. This is a deterministic source-health probe, not a claim
about the unavailable reporter corpus.

The family screen also places 430 cosine-near distractors ahead of a
distinctive target. The distractors deliberately swap their eighth- and
ninth-strongest coordinates, reproducing the false-negative boundary of
a lossy top-feature fingerprint. Both the normal cached path and the
cache-rebuild path retain the target after a sound block-norm upper
bound and exact cosine family checks. The rebuild installs the recovered
surface so subsequent requests do not repeat the full SQL scan. A failed
rebuild installs a failure sentinel so later requests do not retry the
two complete surface scans. Shared base vectors keep all long-term
memory metadata alternatives behind one stored vector so the shared
vector surfaces through exactly one eligible sibling. Retrieval fallback
executes demand-driven ordered pages over the current and historical
vector surfaces. Every SQL page applies kind and timestamp eligibility
before distance ordering and a knob-derived top-k page limit; one C++
cosine-family index spans the pages until the ordinary seed budget is
full or the surface is exhausted. Processor hydration records
persisted-current currency separately from processor-surface
completeness, and the private search cache follows the processor’s
latest reconstruction vectors even when current-table writes are
disabled. A valid-cache regression places a full F/S/T-derived seed
breadth of distinct base families ahead of a target and then maps every
base to one latest family; the target remains visible because cache
ordering and family accounting use the latest vectors. The same test
invalidates the cache and proves that an ephemeral query with a complete
processor surface still finds the target without SQL or registry
mutation. Two additional regressions cover incomplete-processor recovery
using an explicit non-materializing policy and a disabled-current-write
experiment hook; the SQL fallback substitutes latest reconstructions
before family accounting. A separate disable-ablation regression removes
the private cache after reconstruction and verifies that rebuild ranks
the base embedding, matching restart hydration. A dedicated
invalid-neighbor regression places 900 closer `WORKING` rows before the
eligible long-term target and still requires the target to surface.
Another regression inserts 600 eligible historical memories with
distinct embedding identifiers and byte-distinct normalized vectors
inside one cosine family before a distinct target. The first 477-row
page cannot fill the semantic-family budget, so the fallback requests a
second page and the target surfaces. Current per-memory reconstructions
remain distinct rows while family accounting uses their current vectors.
The adversarial fallback regression contains 430 cosine-near distractors
plus 900 distant rows: the distinctive target remains selected while
returned SQL rows stay at or below the second-stage knob-derived bound
(477 for the test knobs), rather than materializing all 1,331 eligible
rows. An ephemeral request never installs recovery state in the process
registry. A separate 256-vector orthogonal-basis test exercises the
broad non-duplicate surface and requires exact family comparisons to
remain below one quarter of the all-pairs count.

The recovery performance probe compares the repair with exact pushed
pull-request head `cb5b8f3c` under matched AppleClang Release builds,
test objects, store generators, and query vectors. Every tenth memory
has a non-materialized reconstruction and each process executes 50 timed
ephemeral requests after seeding. Three alternating-order exact-final
pairs at 1,915 rows produced median process p50/p95 values of
9.445/9.641 ms for the repaired complete-processor path versus
19.144/19.518 ms for the comparator; paired differences were
-9.699/-9.884 ms and all three were negative. At 8,000 rows the
corresponding medians were 39.381/39.985 versus 73.276/74.217 ms, with
paired differences of -33.867/-34.198 ms and all three negative. An
incomplete-processor diagnostic exercises the latest-reconstruction SQL
union separately. Its paired median p50/p95 differences were
-0.117/-0.030 ms at 1,915 rows and -2.528/-5.881 ms at 8,000 rows; five
of six metric-by-pair results at 1,915 and all six at 8,000 were
favorable. These remain recovery-path diagnostics, not promoted to a
zero-margin non-inferiority result. The registered 18-pair fresh-process
benchmark below remains the normal-path performance gate. The direct
probe isolates query execution from database seeding and verifies a
nonempty retrieved set on every timed request. This is recovery
latency/scaling evidence, not an ANN claim or a statistical
production-latency estimate: sqlite-vec remains a brute-force linear
scan, while each SQL `LIMIT` bounds only the rows returned to C++ and
decoded downstream. The historical window partition may still sort and
materialize the full eligible history internally before the limit. The
measured unique-vector populations fill the semantic seed budget in one
page.

Performance was compared against the exact 1.2.1 macOS arm64 prebuild
using two unmeasured warm-up pairs and 18 measured pairs. Each pair
alternated baseline-first/candidate-first order with seed 20260714 and
used fresh processes and fresh databases. The zero-margin paired-median
gate and one-sided 95% bootstrap upper bound passed for all eight
metrics. The schema-v5 artifact binds the result to the exact baseline
addon and executed package wrapper, candidate addon and executed package
wrapper, candidate core runtime, full CMake-cache digest, actual
candidate build configuration and compiler, candidate build-source
digest, final review tree, and exact base-to-tree patch digest. Every
measured worker also records the core runtime it actually loaded and
verifies its path and SHA-256 against that candidate runtime; final
binding fails if any measured binary, build input, build configuration,
package-level JavaScript wrapper, or source changes. The validator
schema and dependency manifest ship beside the benchmark, baseline and
candidate workers both scrub runtime-loader overrides, and packaged
prebuild discovery uses Node architecture tags:

<table>
<colgroup>
<col style="width: 15%" />
<col style="width: 21%" />
<col style="width: 21%" />
<col style="width: 21%" />
<col style="width: 21%" />
</colgroup>
<thead>
<tr>
<th>Metric</th>
<th style="text-align: right;">Baseline median</th>
<th style="text-align: right;">Candidate median</th>
<th style="text-align: right;">Paired median difference</th>
<th style="text-align: right;">95% upper bound</th>
</tr>
</thead>
<tbody>
<tr>
<td>retrieval p50 (ms)</td>
<td style="text-align: right;">7.228</td>
<td style="text-align: right;">3.035</td>
<td style="text-align: right;">-4.214</td>
<td style="text-align: right;">-4.162</td>
</tr>
<tr>
<td>retrieval p95 (ms)</td>
<td style="text-align: right;">11.232</td>
<td style="text-align: right;">8.282</td>
<td style="text-align: right;">-2.971</td>
<td style="text-align: right;">-2.548</td>
</tr>
<tr>
<td>ingestion p50 (ms)</td>
<td style="text-align: right;">16.237</td>
<td style="text-align: right;">6.415</td>
<td style="text-align: right;">-9.781</td>
<td style="text-align: right;">-9.700</td>
</tr>
<tr>
<td>ingestion p95 (ms)</td>
<td style="text-align: right;">27.428</td>
<td style="text-align: right;">10.319</td>
<td style="text-align: right;">-17.327</td>
<td style="text-align: right;">-16.912</td>
</tr>
<tr>
<td>retrieval throughput (ops/s)</td>
<td style="text-align: right;">115.728</td>
<td style="text-align: right;">272.989</td>
<td style="text-align: right;">-157.876<a href="#fn1"
class="footnote-ref" id="fnref1"
role="doc-noteref"><sup>1</sup></a></td>
<td style="text-align: right;">-156.074</td>
</tr>
<tr>
<td>ingestion throughput (ops/s)</td>
<td style="text-align: right;">59.372</td>
<td style="text-align: right;">156.712</td>
<td style="text-align: right;">-97.913<a href="#fn2"
class="footnote-ref" id="fnref2"
role="doc-noteref"><sup>2</sup></a></td>
<td style="text-align: right;">-97.280</td>
</tr>
<tr>
<td>peak resident bytes</td>
<td style="text-align: right;">1,055,596,544</td>
<td style="text-align: right;">785,063,936</td>
<td style="text-align: right;">-270,524,416</td>
<td style="text-align: right;">-269,574,144</td>
</tr>
<tr>
<td>database bytes</td>
<td style="text-align: right;">8,982,528</td>
<td style="text-align: right;">6,008,832</td>
<td style="text-align: right;">-2,973,696</td>
<td style="text-align: right;">-2,973,696</td>
</tr>
</tbody>
</table>
<section id="footnotes" class="footnotes footnotes-end-of-document"
role="doc-endnotes">
<hr />
<ol>
<li id="fn1"><p>For higher-is-better throughput metrics, the registered
difference is baseline minus candidate; for all other metrics it is
candidate minus baseline. Non-positive values pass the zero-margin
gate.<a href="#fnref1" class="footnote-back"
role="doc-backlink">↩︎</a></p></li>
<li id="fn2"><p>For higher-is-better throughput metrics, the registered
difference is baseline minus candidate; for all other metrics it is
candidate minus baseline. Non-positive values pass the zero-margin
gate.<a href="#fnref2" class="footnote-back"
role="doc-backlink">↩︎</a></p></li>
</ol>
</section>

### Reporter-Corpus Consolidate-on-Recommend Control

On 2026-07-15, the 1,915-message whole-message durable replay was rerun
on fresh isolated databases with the exact seven-query manifest. Every
arm used the same generic source id, default text modality, and
`F=0.45`, `S=0.5`, `T=0.5`. The policy treatment called explicit
consolidation exactly once after each durable result whose consolidation
state was `recommended` or `required`; ephemeral query results never
triggered consolidation, and there was no unconditional final call.

<table>
<colgroup>
<col style="width: 13%" />
<col style="width: 17%" />
<col style="width: 17%" />
<col style="width: 17%" />
<col style="width: 17%" />
<col style="width: 17%" />
</colgroup>
<thead>
<tr>
<th>Arm</th>
<th style="text-align: right;">Recommendation outputs</th>
<th style="text-align: right;">Consolidations</th>
<th style="text-align: right;">Top-12 hits</th>
<th style="text-align: right;">Ingest wall time</th>
<th style="text-align: right;">Seven-query wall time</th>
</tr>
</thead>
<tbody>
<tr>
<td>released 1.2.1 policy</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">1/7</td>
<td style="text-align: right;">167.313 s</td>
<td style="text-align: right;">340.468 ms</td>
</tr>
<tr>
<td>current candidate, no consolidation</td>
<td style="text-align: right;">1,546</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">2/7</td>
<td style="text-align: right;">52.085 s</td>
<td style="text-align: right;">251.387 ms</td>
</tr>
<tr>
<td>current candidate, consolidate on recommendation</td>
<td style="text-align: right;">124</td>
<td style="text-align: right;">124</td>
<td style="text-align: right;">1/7</td>
<td style="text-align: right;">70.593 s</td>
<td style="text-align: right;">277.151 ms</td>
</tr>
</tbody>
</table>

The released booleans did not fire on any of the 1,915 writes, so that
arm reproduced the old never-consolidated behavior. The current
no-consolidation control emitted 28 `recommended` and 1,518 `required`
results. In the treatment, the first call occurred at write 371. Calls
then clustered into six consecutive bursts at writes 371–417, 689–711,
947–965, 1228–1239, 1547–1558, and 1885–1895. Each explicit call
returned state `none`, but the next durable write could immediately emit
`required`; the policy therefore produced 6 recommended calls followed
by 118 required calls rather than one call per throughput event. The 124
calls consumed 8.372 s in aggregate (median 59.846 ms, p95 135.009 ms,
maximum 164.979 ms).

Against the matched current-candidate control, the treatment lost the
`alice_gate_launch_code` top-12 hit and gained no top-12 hit. It did
return the `pre_llm_plugin_hook` needle at rank 14, so the full
variable-length packets contained two needles in both arms while the
fixed top-12 score regressed from 2/7 to 1/7. All seven needle counts
remained byte-content-equivalent and all needle-bearing rows remained
`LONG_TERM`; the failure is not retention.

The treatment also produced 37 additional long-term rows, 130
association memories, 167 additional current-memory embedding rows,
5,365 additional graph edges, 90 additional signal rows, and 1,867,776
additional database bytes before the queries. Its ingest wall time was
18.509 s (+35.54%) above the matched control, and its query wall time
was 25.764 ms (+10.25%) higher. This is one paired local run, not a
statistical production-latency estimate, but it fails the requested
non-regression gate and shows that callers must not treat every
non-`none` output as an unlatched level-triggered command.

The earlier 3/7 logical replay remains useful retrieval evidence but is
not the matched causal control for this experiment: candidate retrieval
queried a byte-identical copy of the released-ingested database. The
sanitized control artifact has SHA-256
`69db0e3337b708ce14c11dbc5a91883b2a4b825023df602206b543f8917f8909`. It
contains only labels and hashes, not transcript text, query plaintext,
needles, or memory content. This experiment added no source-id-specific
or modality-specific behavior and does not establish seven-of-seven
recall, production performance, issue closure, or release readiness.

### Consolidation Integrity Repair

The same matched replay was repeated after making recommendation
consumption edge-triggered, isolating forced maintenance from normal
ingestion, removing forced candidate/cluster fallbacks, limiting graph
construction to newly persisted clusters, preserving direct long-term
family representatives before graph expansion can compete, and
relevance-ordering linked sources before the compact hydration limit.
Both arms used fresh databases and the same 1,915 messages, seven
queries, source metadata, retention settings, model, runtime, and F/S/T
values.

<table>
<colgroup>
<col style="width: 13%" />
<col style="width: 17%" />
<col style="width: 17%" />
<col style="width: 17%" />
<col style="width: 17%" />
<col style="width: 17%" />
</colgroup>
<thead>
<tr>
<th>Arm</th>
<th style="text-align: right;">Non-<code>none</code> outputs</th>
<th style="text-align: right;">Consolidations</th>
<th style="text-align: right;">Top-12 hits</th>
<th style="text-align: right;">Ingest wall time</th>
<th style="text-align: right;">Seven-query wall time</th>
</tr>
</thead>
<tbody>
<tr>
<td>repaired candidate, no consolidation</td>
<td style="text-align: right;">1,558</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">2/7</td>
<td style="text-align: right;">46.719 s</td>
<td style="text-align: right;">206.878 ms</td>
</tr>
<tr>
<td>repaired candidate, consolidate on recommendation</td>
<td style="text-align: right;">1</td>
<td style="text-align: right;">1</td>
<td style="text-align: right;">2/7</td>
<td style="text-align: right;">46.732 s</td>
<td style="text-align: right;">194.543 ms</td>
</tr>
</tbody>
</table>

The treatment emitted `recommended` at write 380 and ran one 22.630 ms
maintenance transaction. The call acknowledged the current throughput
event, reset its derived floor/peak, and cleared the armed latch.
Recommendation timing varies with the observed early ingest rate, rather
than an arbitrary message-count or elapsed-time cadence. The call added
zero memory rows, long-term rows, working rows, current embeddings,
graph edges, association memories, signal rows, or processed-signal
count. Both independently built fresh databases finished at 23,060,480
and 23,064,576 bytes, respectively. The hashed raw snapshots include the
armed latch and nonzero-connectivity count before the call, after the
call, and at final query boundaries. They record the acknowledgment
clearing `armed` and zero memories with nonzero persisted connectivity
in both arms, closing the hidden global-write path found during the
first repair replay. All seven needle counts remained `LONG_TERM` with
counts 12, 21, 44, 2, 11, 2, and 7. Both arms retained the same two
top-12 hits and ended with 1,915 long-term memories, 1,915 current
embeddings, and 1,915 processed signals. Every ephemeral-query durable
delta remained zero.

The reporter replay is a functional matched control, not a latency
estimator. Performance was therefore checked separately against the
frozen released baseline with two warm-up pairs and 18 measured
fresh-process, fresh-database pairs, alternating order at the registered
zero non-inferiority margin. All eight gates passed.
Candidate-minus-baseline latency medians and one-sided 95% upper bounds
were -4.214/-4.162 ms for retrieval p50, -2.971/-2.548 ms for retrieval
p95, -9.781/-9.700 ms for ingestion p50, and -17.327/-16.912 ms for
ingestion p95. Higher-is-better throughput uses baseline minus candidate
and was likewise negative at both point and upper bound; peak RSS and
database bytes also passed.

The JavaScript binding contract now loads the measured native addon,
executes an ephemeral text request, and checks the serialized runtime
object as well as the TypeScript declaration: `consolidation_state` is
present and the two old boolean fields are absent. The sanitized repair
artifact contains no transcript text, query plaintext, needle plaintext,
or memory content. Its content address is recorded in the project proof
ledger rather than embedded here, avoiding a self-reference between the
artifact’s reviewed-tree identity and this manuscript. This repair
remains generic: it adds no source-id or modality branch and does not
claim seven-of-seven recall, production latency, merge readiness, or
release readiness.

On 2026-07-11, retention was expanded to four policies: `Natural`
(default when omitted: `force_boundary=false`, `force_write=false` so
accumulator boundary/write-gate algorithms decide), `Durable`
(historical chat convenience: force both flags), `Boundary` (force turn
edge only), and `Ephemeral` (no-store query). Binding surfaces expose
retention through process options; CLI remember continues to use
`Durable` and recall uses `Ephemeral`.

On 2026-07-12, a safety regression screen covered the retained ingress
and storage paths. Focused tests injected a root-transaction commit
failure and verified that a retry persisted the same working-memory
signal record; applied two passive WM maintenance passes and verified
linear rather than compounded decay; replayed a backward timestamp
without a forced gap boundary; retained an emotional source sixty
seconds old inside the five-minute cascade window; evicted a memory with
a distinct signal embedding and verified the vector row was reclaimed;
and verified supersession/graph writes invalidate cached fanout. Binding
checks reject undersized images before native inference, legacy media
JSON entry points remain Durable, and WASM text ingress now exposes all
four retention policies with Natural as its default.

The same patch ran the deterministic steady-state live-loop harness for
1,000 synthetic signals in a warnings-as-errors debug build. It
completed in 25.809 s (38.746 signals/s) with 238 writes, 239
boundaries, 231 interrupts, 5,461 total candidates, 253 final memories,
46 associations, and 1,355 embeddings. The performance-tag regression
tests also passed. A matched AppleClang Release comparison against the
unchanged parent tree then alternated five 1,000-signal runs per build.
Median throughput was 180.140 signals/s for the parent and 179.580
signals/s for the patched tree (-0.31%); means were 177.919 and 177.354
signals/s (-0.32%), respectively. Every run produced identical writes,
boundaries, interrupts, candidate totals, threshold and effective-focus
sums, and final memory, association, and embedding counts. The observed
throughput difference is within run-to-run variation. This comparison
covers the core operation and persistence loop with synthetic
embeddings, not model inference or language-runtime overhead. At that
measured head, the implemented checks added no per-signal query or eager
graph rebuild. The later passive-decay durability repair adds at most
one bounded memory-row update per changed live slot during normal
persistence and is outside this historical measurement.

Separately, the July 8 retention work was run through the normalized
memory-eval smoke slice with all configured benchmark adapters, using
durable history ingestion and ephemeral queries through the public API:

``` bash
scripts/run_memory_evals.py --profile smoke --benchmarks all --no-prepare \
  --out logs/memory_evals/release_1_1_10_smoke_20260708T043734Z \
  --max-episodes 1 --max-queries 1 --answer-mode none

scripts/run_memory_eval_judges.py \
  --run logs/memory_evals/release_1_1_10_smoke_20260708T043734Z \
  --judges packet,codex,grok,agy --include-unavailable --timeout-s 180
```

The smoke slice returned nonempty retrieval packets for every query
(`queries=6`, aggregate `retrieved_count=66`, `retrieval_empty=0`).
Packet-answer accuracy stayed at 0.0 over the scored slice, while the
Codex, Grok, and Antigravity adapters each scored 0.2 over five scored
queries. This is a regression check for public ephemeral retrieval and
harness wiring, not a state-of-the-art quality claim.

That `logs/memory_evals/...` directory is local-only and is not included
in the paper artifact bundle. The checked-in manuscript therefore treats
these smoke numbers as a historical harness record rather than
independently auditable raw evidence.

## Retrieval Behavior After the v1 Hard Cut (Historical)

This subsection describes the v1 hard-cut engine of June 2026, before
the 2026-10 ablation removed association seeding, graph expansion and
retrieval-time reconstruction. Current retrieval is specified in
<a href="#sec-retrieval" class="quarto-xref">Section 6</a>.

The highest-risk behavioral regression was graph retrieval. Removing
semantic summaries, fact rows, label-bank expansion, and
temporal-retrieval helpers left retrieval responsible for carrying
long-horizon context through retained memory embeddings, association
edges, and constructive reconstruction state.

The retained retrieval path now:

-   seeds from current durable `LONG_TERM` and `ASSOCIATION` embeddings;
-   excludes current writes by timestamp before ranking;
-   expands through retained graph edges only;
-   uses source-backed memory hydration for association products;
-   reads the latest constructive reconstruction surface when enabled;
    and
-   skips reconstruction writes entirely when constructive recall is
    disabled.

The targeted tests cover source ordering, association expansion,
current-write exclusion, reconstruction-aware ranking, and the
disabled-reconstruction path. This is the minimum behavior required for
the v1 hard-cut engine to stay useful without hidden semantic
extraction.

### Graph Retrieval Release Probe

A June 2026 regression probe compared a preserved old replay binary
against the current source tree using the same local AIST GGUF model.
The model rotation was therefore not sufficient to explain the observed
A/B drop. The main behavioral regression was a reconsolidation policy
that appended reconstruction records without advancing
`current_memory_embeddings`, leaving graph retrieval on stale memory
surfaces.

The experimental online label-bucket graph was removed from the
production release path. Runtime storage no longer builds
embedding-label buckets or writes their synthetic graph edges, retrieval
no longer reads those buckets or edge types, and hydration no longer
scans experiment-local label-edge tables. The retained graph behavior is
the durable source, reconstruction, and association path described
above.

After restoring current-surface advancement and removing a
maintenance-node hydration skip that was not present in the preserved
replay binary, the dense 578-message replay matched the old binary on
processed text messages, media events, probe events, memory rows, and
association rows. With the default bundled GGML build, the remaining
difference was 1 retrieved-list diff and 1 ranked-candidate diff across
383 probes. The full one-year sparse probe matched the old binary on the
same structural counts and left 1 retrieved-list diff and 1
ranked-candidate diff across 31 probes.

The two residual rank flips were traced to query-vector numeric drift
rather than graph state. For the compared candidate memories, the old
and current DBs contained identical memory/reconstruction embedding
hashes, while the probe signal embedding hashes differed for the same
embedding ids. Forcing the host fallback text path by disabling the full
GGML graph produced far larger replay drift. Rebuilding the current
source tree with the same dynamic system GGML libraries used by the
preserved old binary restored exact replay agreement: the dense replay
produced `retr_diffs=0` and `rank_diffs=0` across 383 probes, and the
full sparse replay produced `retr_diffs=0` and `rank_diffs=0` across 31
probes.

A fresh blind-judge run on 2026-06-28 then evaluated the current source
tree on the same one-year sparse replay. The run processed 2,633 text
messages after 102,166 skipped transcript messages, emitted 31 probes,
and judged text-only structurally normalized blind packets with
Gemma4-12B-AWQ served by vLLM at a 131,072-token context window. The
protocol used three repetitions per probe, `judge_seed=42`, 2,000
probe-bootstrap samples, `--max-media-per-system 0`, and
`--blind-packets`; it completed 93/93 judgments with no missing rows.

<table>
<colgroup>
<col style="width: 13%" />
<col style="width: 17%" />
<col style="width: 17%" />
<col style="width: 17%" />
<col style="width: 17%" />
<col style="width: 17%" />
</colgroup>
<thead>
<tr>
<th>Outcome</th>
<th style="text-align: right;">Probe-majority wins</th>
<th style="text-align: right;">Raw wins</th>
<th style="text-align: right;">Raw win rate</th>
<th style="text-align: right;">Probe-bootstrap 95% CI</th>
<th style="text-align: right;">Mean context tokens</th>
</tr>
</thead>
<tbody>
<tr>
<td>Cortext native</td>
<td style="text-align: right;">14/31</td>
<td style="text-align: right;">47/93</td>
<td style="text-align: right;">0.505</td>
<td style="text-align: right;">[0.387, 0.624]</td>
<td style="text-align: right;">467</td>
</tr>
<tr>
<td>Traditional chat RAG</td>
<td style="text-align: right;">2/31</td>
<td style="text-align: right;">16/93</td>
<td style="text-align: right;">0.172</td>
<td style="text-align: right;">[0.108, 0.247]</td>
<td style="text-align: right;">7,447</td>
</tr>
<tr>
<td>Full-history upper bound</td>
<td style="text-align: right;">1/31</td>
<td style="text-align: right;">3/93</td>
<td style="text-align: right;">0.032</td>
<td style="text-align: right;">[0.000, 0.086]</td>
<td style="text-align: right;">15,974</td>
</tr>
<tr>
<td>Tie / unclear</td>
<td style="text-align: right;">10/31</td>
<td style="text-align: right;">27/93</td>
<td style="text-align: right;">0.290</td>
<td style="text-align: right;">n/a</td>
<td style="text-align: right;">n/a</td>
</tr>
<tr>
<td>No-majority split</td>
<td style="text-align: right;">4/31</td>
<td style="text-align: right;">n/a</td>
<td style="text-align: right;">n/a</td>
<td style="text-align: right;">n/a</td>
<td style="text-align: right;">n/a</td>
</tr>
</tbody>
</table>

Cortext’s single-repetition win counts were 14, 17, and 16 out of 31, so
the three-repetition average is 15.7 wins per 31-probe pass and recovers
the historical 15-win A/B level. The token evidence remained strongly in
Cortext’s favor: mean Cortext context was 467 tokens versus 7,447 for
traditional RAG, a 93.7% reduction. The final six probes triggered
full-history prompt-fit trimming on all three repetitions, dropping 82
to 480 oldest full-history documents while preserving
`judge_prompt_fits_context_window` and
`full_history_prompt_fits_judge_context`; this constrains the
full-history comparator rather than the Cortext packet. The aggregate
artifact is
`docs/paper/artifacts/release_eval_20260628_gemma4_vllm/current_sparse_1y_system_ggml_20260628T0713Z/judge_vllm_gemma4_12b_awq_131k_rep3.json`.

### Hosted Meta MSC Frontier-Judge Probe

On 2026-06-30, a hosted frontier-judge probe evaluated a public Meta
Multi-Session Chat validation slice materialized from the Hugging Face
`nayohan/multi_session_chat` mirror. The materializer produced a
deterministic chat replay with 708 dataset rows and 9,130 text turns.
The replay used the default production knobs (`F=S=T=0.5`), daily
source-time consolidation at 02:00 UTC, a 49,152-token active-history
budget for the standard chat baseline, a 5,000-event warmup, and
500-event probe stride, yielding 9 judged probes.

The judge was hosted `gpt-5.5` through OpenAI Chat Completions. The
protocol used three blind repetitions per probe, `judge_seed=42`, 2,000
probe-bootstrap samples, `--max-media-per-system 0`, a 1,000,000-token
judge-context setting, and four text-only systems:

-   Cortext native working memory plus long-term graph retrieval;
-   traditional chat+RAG, using rolling text history until compaction
    plus text RAG hits from the same prior event stream;
-   full-history upper bound over all prior text history; and
-   a hosted `gpt-5.5` compacting-session rollup baseline.

The run completed 27/27 judgments. The strict gates passed:
`judge_prompt_fits_context_window`,
`full_history_prompt_fits_judge_context`,
`no_future_context_violations`, `no_current_turn_context_inclusions`,
`blind_prompt_hidden_labels_absent`, `traditional_chat_rag_text_only`,
and `full_history_text_only`.

<table>
<colgroup>
<col style="width: 9%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
</colgroup>
<thead>
<tr>
<th>Outcome</th>
<th style="text-align: right;">Probe-majority wins</th>
<th style="text-align: right;">Raw wins</th>
<th style="text-align: right;">Raw win rate</th>
<th style="text-align: right;">Probe-bootstrap 95% CI</th>
<th style="text-align: right;">Mean sufficiency</th>
<th style="text-align: right;">Mean noise</th>
<th style="text-align: right;">Mean context tokens</th>
</tr>
</thead>
<tbody>
<tr>
<td>Cortext native</td>
<td style="text-align: right;">7/9</td>
<td style="text-align: right;">21/27</td>
<td style="text-align: right;">0.778</td>
<td style="text-align: right;">[0.519, 0.963]</td>
<td style="text-align: right;">4.41</td>
<td style="text-align: right;">1.85</td>
<td style="text-align: right;">998</td>
</tr>
<tr>
<td>Traditional chat RAG</td>
<td style="text-align: right;">0/9</td>
<td style="text-align: right;">0/27</td>
<td style="text-align: right;">0.000</td>
<td style="text-align: right;">[0.000, 0.000]</td>
<td style="text-align: right;">4.67</td>
<td style="text-align: right;">4.70</td>
<td style="text-align: right;">49,196</td>
</tr>
<tr>
<td>Full-history upper bound</td>
<td style="text-align: right;">0/9</td>
<td style="text-align: right;">1/27</td>
<td style="text-align: right;">0.037</td>
<td style="text-align: right;">[0.000, 0.111]</td>
<td style="text-align: right;">4.52</td>
<td style="text-align: right;">5.00</td>
<td style="text-align: right;">185,439</td>
</tr>
<tr>
<td>Hosted compaction rollup</td>
<td style="text-align: right;">1/9</td>
<td style="text-align: right;">5/27</td>
<td style="text-align: right;">0.185</td>
<td style="text-align: right;">[0.037, 0.444]</td>
<td style="text-align: right;">4.63</td>
<td style="text-align: right;">3.63</td>
<td style="text-align: right;">n/a</td>
</tr>
<tr>
<td>No-majority split</td>
<td style="text-align: right;">1/9</td>
<td style="text-align: right;">n/a</td>
<td style="text-align: right;">n/a</td>
<td style="text-align: right;">n/a</td>
<td style="text-align: right;">n/a</td>
<td style="text-align: right;">n/a</td>
<td style="text-align: right;">n/a</td>
</tr>
</tbody>
</table>

Mean Cortext context was 998 tokens versus 49,196 for traditional
chat+RAG, a 97.97% context-token reduction with probe-bootstrap 95% CI
\[97.77%, 98.17%\]. This hosted public-benchmark result therefore
supports a strong token-reduction, noise-reduction, and blind-win claim:
Cortext won 7 of 9 probes by majority, with one additional probe split
across Cortext, full history, and compaction. It does not support a
completed sufficiency-match claim: mean sufficiency was lower for
Cortext (4.41) than traditional chat+RAG (4.67) and the hosted
compaction rollup baseline (4.63). The aggregate artifact is
`docs/paper/artifacts/msc_frontier_late_200dlg_gpt55_20260630T053427Z/judge_openai_gpt55_four_system_clean.json`.

### TencentDB Agent Memory MSC Comparator

The MSC judge harness now accepts externally materialized recall packets
so Cortext can be compared directly against TencentDB Agent Memory
(`https://github.com/TencentCloud/TencentDB-Agent-Memory`) under the
same blind packet protocol. The comparator is configured through
`judge_systems.json` using an `external_packet_systems` entry named
`tencentdb_agent_memory`. Its packet file is JSONL keyed by
`event_index` or `probe_event_index`; each row provides either `text`,
`content`, or an `items` array of recalled memory text. The harness
converts those rows into synthetic prior text evidence, applies the
configured token budget, blinds packet identities, and records
TencentDB’s scores, wins, and context-token counts alongside Cortext and
the RAG baselines.

For the local rerun protocol, `scripts/run_msc_frontier_judge.sh`
defaults to the OpenAI-compatible vLLM judge at
`http://127.0.0.1:8000/v1`, model `qwen-omni-judge`
(`Qwen/Qwen2.5-Omni-7B-AWQ`), with an 8,192-token prompt budget. Set
`TENCENTDB_PACKET_JSONL=/path/to/tencentdb_packets.jsonl` to include
TencentDB Agent Memory in the MSC comparison.

On 2026-07-06, a bounded local smoke comparison used 20 MSC validation
dialogs (970 turns), nine 100-event probes, one Qwen-Omni judge
repetition per probe, and only two judged systems: Cortext native and
the externally materialized TencentDB Agent Memory packet. The wider
five-system comparison did not fit the local 8,192-token judge context
window, so the reported result is a direct Cortext-versus-TencentDB
screen rather than a full RAG-ablation sweep. TencentDB Agent Memory
produced eight recalled items across the nine probes, with empty recall
packets at probe events 100, 200, 300, 400, and 800.

<table>
<colgroup>
<col style="width: 9%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
</colgroup>
<thead>
<tr>
<th>Outcome</th>
<th style="text-align: right;">Raw wins</th>
<th style="text-align: right;">Raw win rate</th>
<th style="text-align: right;">Probe-bootstrap 95% CI</th>
<th style="text-align: right;">Mean relevance</th>
<th style="text-align: right;">Mean sufficiency</th>
<th style="text-align: right;">Mean noise</th>
<th style="text-align: right;">Mean context tokens</th>
</tr>
</thead>
<tbody>
<tr>
<td>Cortext native</td>
<td style="text-align: right;">7/9</td>
<td style="text-align: right;">0.778</td>
<td style="text-align: right;">[0.444, 1.000]</td>
<td style="text-align: right;">3.22</td>
<td style="text-align: right;">3.56</td>
<td style="text-align: right;">1.00</td>
<td style="text-align: right;">714</td>
</tr>
<tr>
<td>TencentDB Agent Memory</td>
<td style="text-align: right;">1/9</td>
<td style="text-align: right;">0.111</td>
<td style="text-align: right;">[0.000, 0.333]</td>
<td style="text-align: right;">0.89</td>
<td style="text-align: right;">0.78</td>
<td style="text-align: right;">1.89</td>
<td style="text-align: right;">12</td>
</tr>
<tr>
<td>Tie / unclear</td>
<td style="text-align: right;">1/9</td>
<td style="text-align: right;">n/a</td>
<td style="text-align: right;">n/a</td>
<td style="text-align: right;">n/a</td>
<td style="text-align: right;">n/a</td>
<td style="text-align: right;">n/a</td>
<td style="text-align: right;">n/a</td>
</tr>
</tbody>
</table>

This small local run favors Cortext on quality while showing much
smaller TencentDB packets in this configuration. The result should not
be read as a production TencentDB benchmark: the TencentDB run used the
local Qwen model for both memory extraction and judging, ran without a
TencentDB embedding service, and logged several extraction/LLM parse
failures. The aggregate artifact is
`eval_runs/msc_tencentdb_qwen_20260706T213907Z/judge_vllm_qwen_omni.json`,
a local-only path that is not checked into this repository. Consequently
this table is a historical comparator record and cannot be independently
audited from the PR checkout; the checked-in artifact-backed MSC results
below and above carry stronger provenance.

### 128k RAG-Ablation Probe

A follow-up ablation used the same 9 MSC probes, three hosted `gpt-5.5`
blind repetitions per probe, `judge_seed=42`, and a 128,000-token
judge-context cap. This run removed the full-history arm and compared
Cortext against capped RAG-style packet variants:

-   semantic vector RAG only, top 5 prior hits;
-   deterministic lexical keyword RAG only, top 5 prior hits;
-   rolling-window chat context capped at 16k estimated packet tokens;
-   hybrid rolling-window plus vector RAG capped at 16k estimated packet
    tokens;
-   the existing hosted compaction-session rollup, capped to 8k
    estimated packet tokens for the judge prompt.

The run completed 27/27 judgments. The max estimated judge prompt was
116,425 tokens under the 128,000-token cap, and the strict
prompt/context/fairness gates passed. Actual OpenAI API usage captured
from responses was 1,871,994 prompt tokens and 39,001 completion tokens
across 27 judge requests.

<table>
<colgroup>
<col style="width: 9%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
</colgroup>
<thead>
<tr>
<th>Outcome</th>
<th style="text-align: right;">Probe-majority wins</th>
<th style="text-align: right;">Raw wins</th>
<th style="text-align: right;">Raw win rate</th>
<th style="text-align: right;">Probe-bootstrap 95% CI</th>
<th style="text-align: right;">Mean sufficiency</th>
<th style="text-align: right;">Mean noise</th>
<th style="text-align: right;">Mean context tokens</th>
</tr>
</thead>
<tbody>
<tr>
<td>Cortext native</td>
<td style="text-align: right;">6/9</td>
<td style="text-align: right;">19/27</td>
<td style="text-align: right;">0.704</td>
<td style="text-align: right;">[0.407, 0.926]</td>
<td style="text-align: right;">4.48</td>
<td style="text-align: right;">1.89</td>
<td style="text-align: right;">816</td>
</tr>
<tr>
<td>Semantic vector RAG</td>
<td style="text-align: right;">0/9</td>
<td style="text-align: right;">0/27</td>
<td style="text-align: right;">0.000</td>
<td style="text-align: right;">[0.000, 0.000]</td>
<td style="text-align: right;">0.67</td>
<td style="text-align: right;">3.46</td>
<td style="text-align: right;">88</td>
</tr>
<tr>
<td>Lexical keyword RAG</td>
<td style="text-align: right;">0/9</td>
<td style="text-align: right;">0/27</td>
<td style="text-align: right;">0.000</td>
<td style="text-align: right;">[0.000, 0.000]</td>
<td style="text-align: right;">0.96</td>
<td style="text-align: right;">3.57</td>
<td style="text-align: right;">224</td>
</tr>
<tr>
<td>Rolling-window chat</td>
<td style="text-align: right;">0/9</td>
<td style="text-align: right;">0/27</td>
<td style="text-align: right;">0.000</td>
<td style="text-align: right;">[0.000, 0.000]</td>
<td style="text-align: right;">4.63</td>
<td style="text-align: right;">4.48</td>
<td style="text-align: right;">15,999</td>
</tr>
<tr>
<td>Hybrid chat+vector RAG</td>
<td style="text-align: right;">0/9</td>
<td style="text-align: right;">0/27</td>
<td style="text-align: right;">0.000</td>
<td style="text-align: right;">[0.000, 0.000]</td>
<td style="text-align: right;">4.78</td>
<td style="text-align: right;">4.43</td>
<td style="text-align: right;">15,999</td>
</tr>
<tr>
<td>Hosted compaction rollup</td>
<td style="text-align: right;">3/9</td>
<td style="text-align: right;">8/27</td>
<td style="text-align: right;">0.296</td>
<td style="text-align: right;">[0.074, 0.593]</td>
<td style="text-align: right;">4.70</td>
<td style="text-align: right;">3.32</td>
<td style="text-align: right;">7,110</td>
</tr>
</tbody>
</table>

This ablation supports a narrower claim than the full-history frontier
result: under a 128k judge-context cap, Cortext retained most wins by
both probe majority (6/9) and row count (19/27) while substantially
reducing context and judged noise against retrieval-only,
rolling-context, hybrid RAG, and compaction-style packet variants. The
aggregate artifact is
`docs/paper/artifacts/msc_rag_ablation_128k_gpt55_20260630T_actual/judge_openai_gpt55_rag_ablation_128k.json`.

### Local Temporal-Scoring Fix A/B

On 2026-07-01, after the Fable audit identified the graph-retrieval
temporal term as effectively dead code, we ran a small local A/B on the
same 9-probe MSC slice. The baseline artifact was produced from the
pre-fix binary, and the candidate artifact used a bounded exponential
age decay that stays in \[0, 1\] over multi-month ages and applies the
temporal weight once. Both arms used Gemma4-12B-AWQ through vLLM, a
262,144-token judge context window, one blind repetition per probe,
`judge_seed=42`, `--max-media-per-system 0`, and three systems only:
Cortext native, traditional chat+RAG, and full-history upper bound. The
compacting-session arm was excluded from this local screen.

<table>
<colgroup>
<col style="width: 9%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 12%" />
</colgroup>
<thead>
<tr>
<th>arm</th>
<th style="text-align: right;">Cortext wins</th>
<th style="text-align: right;">RAG wins</th>
<th style="text-align: right;">full-history wins</th>
<th style="text-align: right;">tie/unclear</th>
<th style="text-align: right;">Cortext composite</th>
<th style="text-align: right;">Cortext temporal correctness</th>
<th style="text-align: right;">mean ingest ms</th>
</tr>
</thead>
<tbody>
<tr>
<td>pre-fix baseline</td>
<td style="text-align: right;">3/9</td>
<td style="text-align: right;">1/9</td>
<td style="text-align: right;">1/9</td>
<td style="text-align: right;">4/9</td>
<td style="text-align: right;">4.14</td>
<td style="text-align: right;">2.89</td>
<td style="text-align: right;">43.13</td>
</tr>
<tr>
<td>bounded age decay</td>
<td style="text-align: right;">2/9</td>
<td style="text-align: right;">2/9</td>
<td style="text-align: right;">0/9</td>
<td style="text-align: right;">5/9</td>
<td style="text-align: right;">2.89</td>
<td style="text-align: right;">1.67</td>
<td style="text-align: right;">42.69</td>
</tr>
</tbody>
</table>

The temporal fix therefore passed the mechanical correctness test but
did not improve this small MSC judge screen: Cortext lost one row win,
the composite score fell by 1.25 points, and mean temporal-correctness
fell by 1.22. Replay performance moved slightly in the desired
direction: mean ingest time fell 1.01 percent, mean processor time fell
1.34 percent, wall time fell 2.14 percent, peak RSS fell 0.46 percent,
and retrieved items during ingest fell from 116 to 113. The result is
too small for a release-quality quality verdict, but it is strong enough
to prevent claiming the temporal fix as an MSC-quality win without a
larger frozen-probe rerun. Artifacts:
`docs/paper/artifacts/msc_gemma4_temporal_baseline_20260701T200553Z/no_compaction_judge/judge_vllm_gemma4_12b_awq_ctx262k_nocomp_rep1.json`
and
`docs/paper/artifacts/msc_gemma4_temporal_fix_20260701T204155Z/no_compaction_judge/judge_vllm_gemma4_12b_awq_ctx262k_nocomp_rep1.json`.

## Research-Branch Ablations and Negative Results

These results predate the June 2026 hard cut and ran on the research
branch. Unless noted, each arm is a live single-repetition Gemma judge
screen on the 1,200-message personal corpus, so the numbers are not
comparable to the hosted frontier and 128k probes above, which use
different judges, repetition counts, and corpora. We record them so
rejected mechanisms are not re-proposed without new long-horizon
evidence, and so the operating points they set stay documented.

### Working-Memory Capacity Study

Capacity is overridable per arm. Per-slot maintenance cost is normalized
so a full working memory costs the same total regardless of capacity, so
gate strictness is capacity-invariant and the arms measure window value
alone. Each arm is a single-repetition live-judge screen against the
three-repetition baseline.

<table>
<thead>
<tr>
<th style="text-align: right;">capacity</th>
<th style="text-align: right;">wins</th>
<th style="text-align: right;">sufficiency</th>
<th style="text-align: right;">mean tokens</th>
<th style="text-align: right;">savings vs RAG</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">7 (baseline)</td>
<td style="text-align: right;">22/39 (56%)</td>
<td style="text-align: right;">3.40</td>
<td style="text-align: right;">290</td>
<td style="text-align: right;">95%</td>
</tr>
<tr>
<td style="text-align: right;">10</td>
<td style="text-align: right;">22/39 (56%)</td>
<td style="text-align: right;">3.51</td>
<td style="text-align: right;">331</td>
<td style="text-align: right;">95%</td>
</tr>
<tr>
<td style="text-align: right;">14</td>
<td style="text-align: right;">23/39 (59%)</td>
<td style="text-align: right;">3.64</td>
<td style="text-align: right;">377</td>
<td style="text-align: right;">94%</td>
</tr>
<tr>
<td style="text-align: right;">21</td>
<td style="text-align: right;">22/39 (56%)</td>
<td style="text-align: right;">3.95</td>
<td style="text-align: right;">503</td>
<td style="text-align: right;">92%</td>
</tr>
<tr>
<td style="text-align: right;">42</td>
<td style="text-align: right;">23/39 (59%)</td>
<td style="text-align: right;">3.82</td>
<td style="text-align: right;">758</td>
<td style="text-align: right;">88%</td>
</tr>
</tbody>
</table>

Win rate stays inside a 56 to 59 percent band, within single-repetition
judge variance. Sufficiency, the was-there-enough-context score a wider
window should move, climbs strictly through capacity 21 and then
declines at 42, which pays 1.5 times the context tokens for less
sufficiency. The curve is unimodal with its peak at 21, so the doubling
search stops there. A three-repetition judge pass over the capacity-21
arm confirmed the screen. Capacity 21 is the operating point the
production runtime ships: `WMBaseCapacity` in
`include/cortext/core/knobs.hpp` documents a capacity range of \[15,
27\] with 21 at neutral knobs, and the mechanism sweep below ran at it.

### Single-Mechanism Removal Arms

The removal record has a June legacy group and a July local-Qwen group
and should not be read as one pooled table. The June capacity-21 sweep
used the legacy 39-probe control: 22/39 wins, 3.95 sufficiency, 0.67
noise, and 503 packet tokens. The July 6 neuromodulator follow-up used
an MSC validation slice with the same capacity-21 stack shape but a
different local Qwen-Omni 32k judge setup. Its measured same-protocol
control was 17/39 wins, 3.50 sufficiency, 1.21 noise, and 755 packet
tokens. Because the July control moved by roughly the same amount as the
four removal arms, July deltas below are computed only against measured
July controls.

The June win column saturates from recency coverage at 1,200 messages,
so mechanism value in that group shows in sufficiency, noise, and tokens
rather than wins. The July confirmation calibrates the local-Qwen
screen: matched per-repetition sufficiency-delta SD reaches 0.32, so
single-repetition July deltas under roughly 0.3 are unstable evidence.
The June single-repetition arms carry the same caution qualitatively,
but that older judge/protocol needs its own repetition-variance
measurement before relitigating verdicts. The June consolidation-family
arms are expected-null at this length, because consolidation exists to
keep month-old content reachable and a short replay serves most probes
from recent memory.

Before the July 6 follow-up, synaptic-tag targeting was fixed to tag the
stored spike memory and same-source temporal neighbors rather than the
newest memories by `created_at`. The follow-up then ran the measured
control plus four neuromodulator-layer removals: neuromodulator effect
scales, synaptic tagging, the encode/retrieve oscillator, and the
emotion/mood threshold cascade. All five July arms completed 39/39
judgments and passed the prompt-fit and context-leak gates. The
aggregate artifact is
`docs/paper/artifacts/neuromodulator_mechanism_sweep_20260706T232135Z/mechanism_sweep_summary.json`.

The June removal arms below are read only against the June control. The
`reading` column records the decision made in that research branch at
the time; it is not an inventory of mechanisms consumed by the current
production ranker.

<table>
<colgroup>
<col style="width: 13%" />
<col style="width: 18%" />
<col style="width: 18%" />
<col style="width: 18%" />
<col style="width: 18%" />
<col style="width: 13%" />
</colgroup>
<thead>
<tr>
<th>removed mechanism</th>
<th style="text-align: right;">wins</th>
<th style="text-align: right;">sufficiency</th>
<th style="text-align: right;">noise</th>
<th style="text-align: right;">tokens</th>
<th>reading</th>
</tr>
</thead>
<tbody>
<tr>
<td>temporal_retrieval</td>
<td style="text-align: right;">22/39</td>
<td style="text-align: right;">3.74</td>
<td style="text-align: right;">0.97</td>
<td style="text-align: right;">543</td>
<td>largest hygiene loss; retained</td>
</tr>
<tr>
<td>predictive_bonus</td>
<td style="text-align: right;">22/39</td>
<td style="text-align: right;">3.79</td>
<td style="text-align: right;">0.86</td>
<td style="text-align: right;">655</td>
<td>30% packet bloat, ties triple; historical retain, absent from
current ranker</td>
</tr>
<tr>
<td>metacognitive</td>
<td style="text-align: right;">24/39</td>
<td style="text-align: right;">3.85</td>
<td style="text-align: right;">0.74</td>
<td style="text-align: right;">473</td>
<td>removal mildly positive; cut in v1</td>
</tr>
<tr>
<td>daily_consolidation</td>
<td style="text-align: right;">23/39</td>
<td style="text-align: right;">4.03</td>
<td style="text-align: right;">0.41</td>
<td style="text-align: right;">352</td>
<td>removal mildly positive; deferred to long horizon</td>
</tr>
<tr>
<td>constructive_recall</td>
<td style="text-align: right;">22/39</td>
<td style="text-align: right;">3.79</td>
<td style="text-align: right;">0.74</td>
<td style="text-align: right;">430</td>
<td>late-corpus sufficiency cost; weak retain</td>
</tr>
<tr>
<td>media_source_blobs</td>
<td style="text-align: right;">21/39</td>
<td style="text-align: right;">3.77</td>
<td style="text-align: right;">0.76</td>
<td style="text-align: right;">198</td>
<td>real late-corpus cost; 60% of packet weight</td>
</tr>
<tr>
<td>graph_expansion</td>
<td style="text-align: right;">21/39</td>
<td style="text-align: right;">3.90</td>
<td style="text-align: right;">0.71</td>
<td style="text-align: right;">507</td>
<td>null</td>
</tr>
<tr>
<td>fact_boosts</td>
<td style="text-align: right;">21/39</td>
<td style="text-align: right;">3.87</td>
<td style="text-align: right;">0.59</td>
<td style="text-align: right;">502</td>
<td>negligible</td>
</tr>
<tr>
<td>temporal_fact_boosts</td>
<td style="text-align: right;">22/39</td>
<td style="text-align: right;">3.74</td>
<td style="text-align: right;">0.97</td>
<td style="text-align: right;">543</td>
<td>identical to temporal alone; no interaction</td>
</tr>
<tr>
<td>procedural_proactive</td>
<td style="text-align: right;">22/39</td>
<td style="text-align: right;">3.97</td>
<td style="text-align: right;">0.59</td>
<td style="text-align: right;">502</td>
<td>exact null</td>
</tr>
<tr>
<td>stm_shadow</td>
<td style="text-align: right;">22/39</td>
<td style="text-align: right;">3.97</td>
<td style="text-align: right;">0.70</td>
<td style="text-align: right;">500</td>
<td>exact null</td>
</tr>
<tr>
<td>stm_ltm_graph_label_handoff</td>
<td style="text-align: right;">22/39</td>
<td style="text-align: right;">3.97</td>
<td style="text-align: right;">0.68</td>
<td style="text-align: right;">500</td>
<td>exact null; deferred to long horizon</td>
</tr>
</tbody>
</table>

In the June research branch, temporal retrieval and predictive
pre-activation stayed because their removal cost measurable packet
quality or packet size. The hard-cutover production graph ranker retains
temporal scoring but stopped reading `pre_activation`, and the 2026-10
engine ablation removed predictive pre-activation entirely (see the
removal table below), so that predictive verdict does not describe
current runtime behavior. Removing the metacognitive layer was mildly
positive and had no long-horizon story to defer to, which supported
cutting it in v1. Removing daily consolidation was also mildly positive
at this horizon, but we deferred that arm because consolidation-family
verdicts require long-horizon runs. The remaining removal arms were null
or long-horizon deferred as labeled.

The July single-repetition follow-up rows below are read only against
the July 6 measured control and are treated as a preliminary screen.

<table>
<colgroup>
<col style="width: 11%" />
<col style="width: 15%" />
<col style="width: 15%" />
<col style="width: 15%" />
<col style="width: 15%" />
<col style="width: 15%" />
<col style="width: 11%" />
</colgroup>
<thead>
<tr>
<th>arm</th>
<th style="text-align: right;">wins</th>
<th style="text-align: right;">sufficiency</th>
<th style="text-align: right;">noise</th>
<th style="text-align: right;">tokens</th>
<th style="text-align: right;">delta vs July control</th>
<th>reading</th>
</tr>
</thead>
<tbody>
<tr>
<td>control</td>
<td style="text-align: right;">17/39</td>
<td style="text-align: right;">3.50</td>
<td style="text-align: right;">1.21</td>
<td style="text-align: right;">755</td>
<td style="text-align: right;">0.00</td>
<td>measured control</td>
</tr>
<tr>
<td>neuromodulator_effect_scales removed</td>
<td style="text-align: right;">14/39</td>
<td style="text-align: right;">3.28</td>
<td style="text-align: right;">0.99</td>
<td style="text-align: right;">755</td>
<td style="text-align: right;">-0.22</td>
<td>preliminary removal-hurts; confirm required</td>
</tr>
<tr>
<td>synaptic_tagging removed</td>
<td style="text-align: right;">16/39</td>
<td style="text-align: right;">3.61</td>
<td style="text-align: right;">1.36</td>
<td style="text-align: right;">759</td>
<td style="text-align: right;">+0.10</td>
<td>preliminary null</td>
</tr>
<tr>
<td>encode_retrieve_oscillator removed</td>
<td style="text-align: right;">18/39</td>
<td style="text-align: right;">3.48</td>
<td style="text-align: right;">1.12</td>
<td style="text-align: right;">755</td>
<td style="text-align: right;">-0.03</td>
<td>preliminary null</td>
</tr>
<tr>
<td>emotion_mood_threshold_cascade removed</td>
<td style="text-align: right;">19/39</td>
<td style="text-align: right;">3.48</td>
<td style="text-align: right;">1.16</td>
<td style="text-align: right;">751</td>
<td style="text-align: right;">-0.03</td>
<td>preliminary null</td>
</tr>
</tbody>
</table>

A July 7 confirmation reran the same July protocol with three blind
judge repetitions per probe. All five arms completed 117/117 judgments
with no missing rows; malformed judge outputs were retried and
recovered. The aggregate artifact is
`docs/paper/artifacts/neuromodulator_mechanism_confirm_20260707T002126Z/mechanism_sweep_summary.json`.

<table style="width:100%;">
<colgroup>
<col style="width: 10%" />
<col style="width: 13%" />
<col style="width: 13%" />
<col style="width: 13%" />
<col style="width: 13%" />
<col style="width: 13%" />
<col style="width: 13%" />
<col style="width: 10%" />
</colgroup>
<thead>
<tr>
<th>arm</th>
<th style="text-align: right;">wins</th>
<th style="text-align: right;">sufficiency</th>
<th style="text-align: right;">noise</th>
<th style="text-align: right;">tokens</th>
<th style="text-align: right;">delta vs July 7 control</th>
<th style="text-align: right;">delta rep SD</th>
<th>reading</th>
</tr>
</thead>
<tbody>
<tr>
<td>control</td>
<td style="text-align: right;">50/117</td>
<td style="text-align: right;">3.57</td>
<td style="text-align: right;">1.27</td>
<td style="text-align: right;">755</td>
<td style="text-align: right;">0.00</td>
<td style="text-align: right;">0.00</td>
<td>measured control</td>
</tr>
<tr>
<td>neuromodulator_effect_scales removed</td>
<td style="text-align: right;">42/117</td>
<td style="text-align: right;">3.44</td>
<td style="text-align: right;">1.12</td>
<td style="text-align: right;">755</td>
<td style="text-align: right;">-0.13</td>
<td style="text-align: right;">0.16</td>
<td>below signal floor; no cut</td>
</tr>
<tr>
<td>synaptic_tagging removed</td>
<td style="text-align: right;">44/117</td>
<td style="text-align: right;">3.38</td>
<td style="text-align: right;">1.09</td>
<td style="text-align: right;">759</td>
<td style="text-align: right;">-0.20</td>
<td style="text-align: right;">0.03</td>
<td>removal hurts; retained</td>
</tr>
<tr>
<td>encode_retrieve_oscillator removed</td>
<td style="text-align: right;">39/117</td>
<td style="text-align: right;">3.08</td>
<td style="text-align: right;">1.21</td>
<td style="text-align: right;">755</td>
<td style="text-align: right;">-0.49</td>
<td style="text-align: right;">0.28</td>
<td>removal hurts; retained</td>
</tr>
<tr>
<td>emotion_mood_threshold_cascade removed</td>
<td style="text-align: right;">45/117</td>
<td style="text-align: right;">3.44</td>
<td style="text-align: right;">1.16</td>
<td style="text-align: right;">751</td>
<td style="text-align: right;">-0.14</td>
<td style="text-align: right;">0.32</td>
<td>null/deferred; no cut</td>
</tr>
</tbody>
</table>

A 2026-10 rerun of this protocol on the pruned engine (same 25
validation dialogs, warmup 20, probe stride 30, capacity 21; 1,190
turns, 39 probes) compared the packets themselves before judging. With
the oscillator off, all 39 probe packets were byte-identical to the
control under both Durable and Natural retention. With effect scales
off, 7/39 differed: five only in order and two in which contiguous turns
were attached. A sufficiency delta cannot come from identical packets,
so the July oscillator delta was judge variance (the July arm’s mean
packet tokens, 755, also matched the control). The oscillator was
removed (artifacts: `.claude/hillclimb/ablation/ablate3/msc/`). The July
reading follows for the record.

The July confirmation reverses the preliminary oscillator-null read.
Removing the encode/retrieve oscillator costs 0.49 sufficiency, the
largest July neuromodulator loss, and all three matched repetition
deltas are negative, so that per-signal fast dynamic stays. Synaptic
tagging also clears the 0.15 signal floor after the targeting fix with
the most stable July delta (rep SD 0.03), and is retained on both
correctness and short-horizon quality grounds. Neuromodulator effect
scales and the emotion/mood threshold cascade remain directionally
negative but below the signal floor in this confirmation; neither earns
a hard cut from this screen. Effect scales is not worth more
short-horizon spend, and the emotion/mood cascade remains deferred
because its slow mood integrator has a plausible long-horizon role.

### Long-Horizon Deferred-Mechanism Sweep

The deferred-family mechanism sweep then reused the 18,000-message
context-blowout protocol, capacity 21, 31 probes at stride 600 including
the initial probe, and three blind Qwen-Omni 32k judge repetitions per
probe. Each arm compared Cortext native against the same RAG,
full-history, and compacting session packet baselines as the stress run,
but the table below reports only the Cortext-native rows because the
question was per-mechanism removal value. All seven arms completed 93/93
judgments, the four local 32k judge servers reported
`max_model_len=32768`, and `--require-judge-prompt-fit` produced no
prompt-fit or context-window failures. Malformed judge outputs were
retried and recovered. The aggregate artifact is
`docs/paper/artifacts/long_horizon_mechanism_sweep_20260707T022225Z/mechanism_sweep_summary.json`.

<table style="width:100%;">
<colgroup>
<col style="width: 8%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 8%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 8%" />
<col style="width: 11%" />
<col style="width: 8%" />
</colgroup>
<thead>
<tr>
<th>arm</th>
<th style="text-align: right;">wins</th>
<th style="text-align: right;">sufficiency</th>
<th>rep sufficiency</th>
<th style="text-align: right;">noise</th>
<th style="text-align: right;">tokens</th>
<th style="text-align: right;">mean delta</th>
<th>paired deltas</th>
<th style="text-align: right;">delta rep SD</th>
<th>reading</th>
</tr>
</thead>
<tbody>
<tr>
<td>control</td>
<td style="text-align: right;">27/93</td>
<td style="text-align: right;">3.10</td>
<td>3.40 / 3.05 / 2.84</td>
<td style="text-align: right;">1.42</td>
<td style="text-align: right;">223</td>
<td style="text-align: right;">0.00</td>
<td>0.00 / 0.00 / 0.00</td>
<td style="text-align: right;">0.00</td>
<td>measured control</td>
</tr>
<tr>
<td>emotion_mood_threshold_cascade removed</td>
<td style="text-align: right;">26/93</td>
<td style="text-align: right;">2.67</td>
<td>2.67 / 2.78 / 2.55</td>
<td style="text-align: right;">1.58</td>
<td style="text-align: right;">224</td>
<td style="text-align: right;">-0.43</td>
<td>-0.73 / -0.27 / -0.29</td>
<td style="text-align: right;">0.26</td>
<td>measured harm; retained</td>
</tr>
<tr>
<td>neuromodulator_effect_scales removed</td>
<td style="text-align: right;">29/93</td>
<td style="text-align: right;">2.94</td>
<td>2.89 / 3.06 / 2.87</td>
<td style="text-align: right;">1.36</td>
<td style="text-align: right;">222</td>
<td style="text-align: right;">-0.16</td>
<td>-0.52 / +0.01 / +0.03</td>
<td style="text-align: right;">0.31</td>
<td>retained by default; no cut</td>
</tr>
<tr>
<td>daily_consolidation removed</td>
<td style="text-align: right;">25/93</td>
<td style="text-align: right;">2.95</td>
<td>2.97 / 3.08 / 2.80</td>
<td style="text-align: right;">1.39</td>
<td style="text-align: right;">222</td>
<td style="text-align: right;">-0.15</td>
<td>-0.44 / +0.02 / -0.04</td>
<td style="text-align: right;">0.25</td>
<td>retained by default; no cut</td>
</tr>
<tr>
<td>graph_expansion removed</td>
<td style="text-align: right;">19/93</td>
<td style="text-align: right;">2.84</td>
<td>2.74 / 2.98 / 2.81</td>
<td style="text-align: right;">1.66</td>
<td style="text-align: right;">221</td>
<td style="text-align: right;">-0.26</td>
<td>-0.66 / -0.08 / -0.03</td>
<td style="text-align: right;">0.35</td>
<td>retained by default; unresolved negative</td>
</tr>
<tr>
<td>stm_ltm_graph_label_handoff removed</td>
<td style="text-align: right;">19/93</td>
<td style="text-align: right;">2.81</td>
<td>3.06 / 2.79 / 2.57</td>
<td style="text-align: right;">1.53</td>
<td style="text-align: right;">223</td>
<td style="text-align: right;">-0.29</td>
<td>-0.35 / -0.26 / -0.26</td>
<td style="text-align: right;">0.05</td>
<td>measured harm; retained</td>
</tr>
<tr>
<td>synaptic_tag_ttl removed</td>
<td style="text-align: right;">22/93</td>
<td style="text-align: right;">2.86</td>
<td>2.94 / 2.61 / 3.04</td>
<td style="text-align: right;">1.66</td>
<td style="text-align: right;">220</td>
<td style="text-align: right;">-0.24</td>
<td>-0.47 / -0.44 / +0.20</td>
<td style="text-align: right;">0.38</td>
<td>retained by default; no cut</td>
</tr>
</tbody>
</table>

The long-horizon run closed the deferred cut question for v1 at the
time: no removal improved the stack, so no additional mechanism earned a
hard cut. The 2026-10 ablation later reversed this on the current engine
(<a href="#sec-negative" class="quarto-xref">Section 14</a>). Its packet
diffs showed that several of these mechanisms never reached a packet.
The table therefore uses two verdict tiers rather than treating all
negative mean deltas as equally resolved. The emotion/mood threshold
cascade and STM/LTM graph-label handoff are retained on measured harm.
Emotion/mood loses on all three paired repetitions, and handoff has the
most stable long-horizon loss (rep SD 0.05).

The remaining four arms stay by default: their removals do not help, but
their harm is not resolved by this screen once the paired repetitions
are visible. The control’s first repetition was high (3.40 versus 3.05
and 2.84), which makes mean-vs-mean losses more negative than the later
paired repetitions for effect scales, daily consolidation, and graph
expansion. Effect scales and daily consolidation are null or positive in
the later paired repetitions. Tag TTL is mixed. Graph expansion has the
clearest negative lean among the default-retain group, but the arc is
still null at short horizon and unresolved-leaning-negative at long
horizon, not a confirmed positive-value result. The TTL arm disables tag
expiry while leaving tag assignment enabled, so it is distinct from the
July short-horizon synaptic-tagging removal.

### ACT-R Gate Promotion Arms

Six arms inverted the logic. Each enabled one opt-in ACT-R-derived
scoring gate on top of the full default stack, so a gate earned a place
in the default only by beating the control it would ship into.

<table>
<colgroup>
<col style="width: 13%" />
<col style="width: 18%" />
<col style="width: 18%" />
<col style="width: 18%" />
<col style="width: 18%" />
<col style="width: 13%" />
</colgroup>
<thead>
<tr>
<th>enabled gate</th>
<th style="text-align: right;">wins</th>
<th style="text-align: right;">sufficiency</th>
<th style="text-align: right;">noise</th>
<th style="text-align: right;">tokens</th>
<th>reading</th>
</tr>
</thead>
<tbody>
<tr>
<td>base_level_availability</td>
<td style="text-align: right;">22/39</td>
<td style="text-align: right;">3.82</td>
<td style="text-align: right;">0.69</td>
<td style="text-align: right;">521</td>
<td>null; redundant with recency boosts</td>
</tr>
<tr>
<td>recent_retrieval_inhibition</td>
<td style="text-align: right;">22/39</td>
<td style="text-align: right;">4.00</td>
<td style="text-align: right;">0.64</td>
<td style="text-align: right;">551</td>
<td>null at +48 tokens</td>
</tr>
<tr>
<td>partial_matching_penalty</td>
<td style="text-align: right;">22/39</td>
<td style="text-align: right;">3.92</td>
<td style="text-align: right;">0.64</td>
<td style="text-align: right;">648</td>
<td>null at +29% packet size</td>
</tr>
<tr>
<td>evidence_blending</td>
<td style="text-align: right;">23/39</td>
<td style="text-align: right;">3.97</td>
<td style="text-align: right;">0.67</td>
<td style="text-align: right;">465</td>
<td>null; only arm cheaper than control</td>
</tr>
<tr>
<td>evidence_confidence</td>
<td style="text-align: right;">22/39</td>
<td style="text-align: right;">3.97</td>
<td style="text-align: right;">0.67</td>
<td style="text-align: right;">503</td>
<td>exact null; identical to control</td>
</tr>
<tr>
<td>all_gates</td>
<td style="text-align: right;">23/39</td>
<td style="text-align: right;">3.72</td>
<td style="text-align: right;">0.95</td>
<td style="text-align: right;">484</td>
<td>harmful; perturbations compound</td>
</tr>
</tbody>
</table>

No gate earned promotion. The five single-gate arms held the win column
at 22 or 23 of 39 with quality deltas inside judge noise, differing only
in token cost. The combined arm was actively harmful: with all four
scoring gates active, noise rose from 0.67 to 0.95 and sufficiency fell
0.23 to 3.72, the worst quality profile in the sweep. Evidence blending
was the one arm cheaper than the control and is the lone token-favorable
candidate worth a long-horizon re-test. These rejected scoring gates are
distinct from the ACT-R-inspired activation ledger that the current
retrieval trace retains; the ledger is observability, not a scoring
gate.

### Research-Branch Negative Results Summary (Superseded)

This summary records the June–July 2026 research-branch verdicts. The
2026-10 ablation on the current engine supersedes them
(<a href="#sec-negative" class="quarto-xref">Section 14</a>). In
particular, the emotion and mood threshold deltas that this summary
retains “on measured harm” were later removed with the dynamic
threshold, and packets stayed byte-identical.

The removal record separates four outcomes. The ACT-R gate-promotion
branch still rejects the activation-gating direction for the current v1
runtime: all six gate-promotion arms failed to beat the default stack,
and the `all_gates` arm was harmful because sufficiency fell from 3.95
to 3.72 while noise rose from 0.67 to 0.95. The metacognitive layer was
cut because its removal was mildly positive at this horizon and it had
no deferred long-horizon case. The July neuromodulator confirmation does
not support a hard cut to the tested neuromodulator mechanisms:
oscillator and synaptic-tagging removals are harmful, while
neuromodulator effect scales and the emotion/mood threshold cascade
remain below the short-horizon signal floor. The long-horizon
deferred-family sweep then resolves the remaining cut candidates by the
hard-cut rule: none of the six removals improves the stack, so no
additional cut ships. The retained rows split into two evidentiary
tiers. Emotion/mood cascade and STM/LTM graph-label handoff are retained
on measured harm; neuromodulator effect scales, daily consolidation,
graph expansion, and synaptic-tag TTL are retained by default because
removal shows no benefit and the apparent harm is below this screen’s
paired-repetition resolution. The 2026-10 engine ablation later removed
clustering consolidation, graph expansion and synaptic tagging on the
current engine, where consolidation never fired at default knobs (see
the removal table below). These verdicts do not remove the
ACT-R-inspired activation ledger retained in retrieval traces; that
ledger records ranking evidence for observability and audits, not an
additional scoring gate.

### Working-Memory Partition Failure

A June 11 experiment (commit 08902baf) split working memory into a pure
FIFO recent ring plus a scored associative slice, so the most recent
turns would be structural rather than a salience-competition outcome.
The motivation was a release loss audit in which 15 of the judged losses
were chronologically coherent context losses against the RAG recency
window, against a single retrieval failure. The partition regressed
quality. The streaming fail-fast judge killed the run at milestone 28,
where Cortext trended 1.62 composite points below traditional chat RAG
against a 0.5 floor, at 11 of 28 row wins. Blob-level dedup kept only
the recent copy of any memory held in both slices, so the associative
slice largely duplicated the ring rather than widening it. We reverted
the change (commit 72bb86c5). The reading is that guaranteeing
structural recency did not recover the chronology losses, so the driver
was the comparator recency window, not a missing recency guarantee in
working memory.

### Long-Horizon Context-Blowout Stress Run

The sweep deferred the consolidation family to a long horizon.
Consolidation exists to keep old content reachable, so the release
verdict needs a run where windowed strategies actually lose the early
conversation. We ran the capacity-21 stack for 18,000 text messages plus
128 media candidates (126 processed, 2 failed), with 30 probes at stride
600 and three blinded Gemma 4 12B local-judge repetitions per probe. The
judge compared four packets: Cortext native working memory plus
associative retrieval, traditional chat RAG with a 49,152-token
active-history budget and deterministic compaction, a prior full-history
upper bound, and a simulated compacting session that summarizes itself
near the same budget.

An early capped judge over this run was invalidated: its frozen Cortext
packets still carried future-context rows and failed the fairness gate
with 52 future-context violations. The accepted rerun added prior-only
packet filtering, dropping every non-prior Cortext row before packet
construction and recording the exclusion counts. That filter, in
`tools/judge_chat_replay_live_run.py`, is current tooling and is the
origin of the `no_future_context_violations` gate the MSC runs above
cite. The accepted artifact filtered 82 non-prior rows, capped each
packet at 128 items for judge presentation (the first uncapped prompt
had exceeded the local window), and completed 90/90 judgments with zero
judge-validation failures, no future-context violations, and no
current-turn inclusions under the 131,072-token judge window.

<table>
<colgroup>
<col style="width: 11%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
</colgroup>
<thead>
<tr>
<th>system</th>
<th style="text-align: right;">Probe-majority wins</th>
<th style="text-align: right;">row wins</th>
<th style="text-align: right;">probe-bootstrap win rate</th>
<th style="text-align: right;">sufficiency</th>
<th style="text-align: right;">noise</th>
<th style="text-align: right;">mean context tokens</th>
</tr>
</thead>
<tbody>
<tr>
<td>Cortext native</td>
<td style="text-align: right;">8 / 30</td>
<td style="text-align: right;">30 / 90</td>
<td style="text-align: right;">0.33 [0.22, 0.44]</td>
<td style="text-align: right;">3.14 [2.76, 3.51]</td>
<td style="text-align: right;">1.64 [1.32, 1.96]</td>
<td style="text-align: right;">433</td>
</tr>
<tr>
<td>traditional chat RAG</td>
<td style="text-align: right;">2 / 30</td>
<td style="text-align: right;">9 / 90</td>
<td style="text-align: right;">0.10 [0.03, 0.19]</td>
<td style="text-align: right;">2.57 [2.22, 2.91]</td>
<td style="text-align: right;">1.09 [0.82, 1.37]</td>
<td style="text-align: right;">42,550</td>
</tr>
<tr>
<td>full-history upper bound</td>
<td style="text-align: right;">5 / 30</td>
<td style="text-align: right;">23 / 90</td>
<td style="text-align: right;">0.26 [0.17, 0.34]</td>
<td style="text-align: right;">3.70 [3.36, 4.02]</td>
<td style="text-align: right;">0.36 [0.21, 0.51]</td>
<td style="text-align: right;">88,806</td>
</tr>
<tr>
<td>compacting session</td>
<td style="text-align: right;">1 / 30</td>
<td style="text-align: right;">10 / 90</td>
<td style="text-align: right;">0.11 [0.06, 0.18]</td>
<td style="text-align: right;">2.90 [2.58, 3.24]</td>
<td style="text-align: right;">0.84 [0.66, 1.04]</td>
<td style="text-align: right;">n/a</td>
</tr>
</tbody>
</table>

Five probes resolved as tie/unclear majorities and nine had no majority
across the three repetitions. Full history remains the quality upper
bound: best sufficiency and lowest noise. It is not a deployable memory
strategy, and the judge had to drop 123,359 oldest full-history items
across probes to keep prompts bounded. Cortext is the highest row-win
system at this horizon and uses 432.7 mean context tokens against
42,550.2 for traditional chat RAG, an aggregate 98.98 percent reduction
(probe-bootstrap savings mean 98.61 percent, 95% CI \[98.05%, 99.04%\]).
The tradeoff is honest: after strict prior-only filtering, Cortext
carries more noise than the windowed baselines, and its sufficiency
confidence interval overlaps the compacting and RAG arms. This is not a
per-mechanism promotion of any durable-structure component. It is the
full-stack release stress verdict: at a horizon where windowed
strategies compact or discard old context, Cortext stays competitive
with the full-history upper bound while spending roughly two orders of
magnitude less context than chat RAG. The artifact is
`docs/paper/artifacts/replay_v8_context_blowout/judge_gemma4_12b_local_context128_prioronly.json`.

## Engine Ablation Removals (2026-10)

A leave-one-out study switched off each engine subsystem in turn. It
used a build with experiment hooks; the hooks are not part of the
release. The evals were Eval 1 (250 cases), LongMemEval-S (48), every
second LongMemEval-M episode (24), and full-length BEAM-10M
conversations (19-24k turns each). Tier-1 metrics are judge-free:
gold-turn recall in the packet and in the seed set, query p95, ingest
per turn, instructions retired, peak RSS and store size. A subsystem was
removed when switching it off measurably helped quality, or when it left
every packet byte-identical with no cost benefit. Each removal was then
rechecked on its own commit: build, the full test suite, and
byte-identical packets against the parent commit on LongMemEval-S (48),
Eval 1 (250) and BEAM-10M conversation 7 (about 21k turns), unless the
entry says otherwise.

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr>
<th>Removed</th>
<th>Ablation evidence (packets changed; recall deltas)</th>
</tr>
</thead>
<tbody>
<tr>
<td>meta-learned knob priors (<code>ApplyMetaLearning</code>,
<code>meta_learning_coeffs</code>)</td>
<td>0/250 Eval 1, 0/48 S, 0/40 BEAM; no cost change. The priors are the
closed-form F/S/T functions again</td>
</tr>
<tr>
<td>predictive pre-activation
(<code>ApplyPredictivePreActivation</code>,
<code>memories.pre_activation</code> and its indexes)</td>
<td>0/250, 0/48, 0/40. No ranker read the field</td>
</tr>
<tr>
<td>influence feedback (<code>ApplyInfluenceFeedback</code>,
<code>memories.influence/sustained_influence/mean_influence</code>,
<code>state.sustained_influence</code>)</td>
<td>0/250, 0/48, 0/40 (as part of the minor-feedback group). The columns
were write-only</td>
</tr>
<tr>
<td>focus feedback (<code>ApplyFocusFeedback</code>)</td>
<td>0/250, 0/48, 0/40 (as part of the minor-feedback group)</td>
</tr>
<tr>
<td>supersession exclusion (seed-time filter of <code>supersedes</code>
targets, its eligibility sidecar and the SQL <code>instr()</code>
superseded-id filter)</td>
<td>Not neutral: every packet changes, and recall rises. Packet
gold-turn recall +0.025 on S, +0.018 on Eval 1, +0.066 on BEAM-10M
conversation 7. Seed gold share +0.136, +0.158 and +0.081. Haiku judge
(rubric v2) sufficiency -0.006 ± 0.051 on S and -0.005 ± 0.039 on Eval
1. Misleading 0.021 → 0.021 and 0.022 → 0.016. Instructions +2.5%.
<code>supersedes</code> edges and superseded-predecessor hydration
stay</td>
</tr>
<tr>
<td>clustering consolidation (<code>EvaluateConsolidation</code>,
scoring gate, clusterer, shallow <code>ASSOCIATION</code> replay,
typed-edge graph build)</td>
<td>Inert at default knobs: 0 candidates per pass because stability
never fell below 1.0. With a stability fix that made it fire, packet
recall moved -0.012 on S, -0.010 on Eval 1 and 0 on BEAM. Judge
sufficiency +0.012 ± 0.023 on S and -0.009 ± 0.023 on Eval 1.
Instructions +1–2%. Packets are byte-identical to the inert pipeline
after removal</td>
</tr>
<tr>
<td>graph expansion (one-hop traversal of
<code>co_occurs</code>/<code>similar_to</code>/<code>reinforces</code>/<code>causes</code>/<code>derived_from</code>/episode
edges with a bounded graph bonus)</td>
<td>0/250, 0/48, 0/40 in phase 1, where only <code>supersedes</code>
edges existed. After clustering consolidation was removed, no
traversable edge is written</td>
</tr>
<tr>
<td>memory strength and eviction (<code>UpdateMemoryStrength</code>
traces, reinforcement and decay, periphery-cutoff eviction,
storage-pressure gate, <code>memory_evictions</code>)</td>
<td>0/250, 0/48, 0/40. Eviction never ran at default knobs. Once a
stability fix opened its gate (storage floor 0), it deleted 94% of S,
92% of Eval 1 and 99.7% of BEAM conversation 7 memories, and packet
recall fell 0.630 → 0.086, 0.587 → 0.084 and 0.103 → 0. Retrieval
counters moved to <code>RecordMemoryUsage</code></td>
</tr>
<tr>
<td>serial position effects (<code>ApplySerialPositionEffects</code>,
<code>ApplySerialPositionMultiplier</code>)</td>
<td>0/250, 0/48, 0/40 (minor-feedback group). The multiplier only fed
memory-strength reinforcement</td>
</tr>
<tr>
<td>synaptic tagging (<code>ApplySynapticTagging</code>,
<code>tag_strength</code>, <code>tag_expires_at</code>)</td>
<td>0/250, 0/48, 0/40. Its only consumer was the consolidation score.
The June research-branch “removal hurts” verdict was measured with
consolidation live</td>
</tr>
<tr>
<td>sensitivity feedback (<code>ApplySensitivityFeedback</code>,
<code>memories.redundancy</code>, the unwritten
<code>connectivity</code>)</td>
<td>0/250, 0/48, 0/40 (minor-feedback group). <code>redundancy</code>
only fed the consolidation score</td>
</tr>
<tr>
<td>stability feedback and half-life adaptation
(<code>ApplyStabilityFeedback</code>, <code>UpdateStability</code>,
<code>memories.stability</code>, persisted
half-life/rate-decay/drift-weight/retention state)</td>
<td>0/250, 0/48, 0/40 (minor-feedback group). Their only consumers were
the consolidation score and memory-strength decay</td>
</tr>
<tr>
<td>emotional consolidation and cascade
(<code>ApplyEmotionalConsolidation</code>,
<code>PropagateEmotionalCascade</code>, emotional-metadata cache,
<code>flashbulb</code>/<code>emotional_intensity</code>/<code>half_life_bonus</code>/<code>detail_suppression</code>/<code>gist_components</code>/<code>cascade_*</code>
columns)</td>
<td>0/250, 0/48, 0/40 (affect group). Outputs fed only memory-strength
decay and the cascade</td>
</tr>
<tr>
<td>linked-source hydration (<code>LABEL</code>/<code>ASSOCIATION</code>
routing rows resolved through
<code>has_label</code>/<code>derived_from</code> routes, query-aware
linked ordering, compact-item limit)</td>
<td>0/250, 0/48, 0/40 with query-aware ordering off. With clustering
consolidation removed, nothing writes those routes</td>
</tr>
</tbody>
</table>

Working-memory items with equal start timestamps were returned in
whatever order SQLite’s chosen index produced. Dropping the strength and
affect columns and indexes changed that plan, which reordered tied
working-memory slots in 45/48 S and 244/250 Eval 1 packets (same items,
same retrieved list). Hydration now breaks ties by `memory_id`, so the
order no longer depends on the query plan.

Five subsystems changed packets when switched off but did not clearly
help or hurt on recall alone. They were judged with Haiku (rubric v2)
against the pipeline above. S and Eval 1 used one repetition per changed
case; BEAM-10M conversation 7 used two. Each delta is “switched off
minus on” with a 95% interval.

<table>
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<thead>
<tr>
<th>Subsystem</th>
<th>Judge sufficiency delta</th>
<th>Recall and cost</th>
<th>Decision</th>
</tr>
</thead>
<tbody>
<tr>
<td>embedding movement (reconsolidation drift)</td>
<td>S -0.048 ± 0.045, Eval 1 -0.032 ± 0.029; misleading +0.012 on Eval
1</td>
<td>packet recall -0.025 / -0.008; instructions -18%, store 380 → 265
MB</td>
<td>kept in round 2; removed in round 3 (below): this arm also switched
off the reconstruction representation, and movement alone is
neutral</td>
</tr>
<tr>
<td><code>supersedes</code> edges and superseded hydration</td>
<td>S -0.036 ± 0.052 (sufficient -0.095 ± 0.090), Eval 1 -0.016 ±
0.013</td>
<td>packet recall -0.037 / -0.025; instructions -6%</td>
<td>kept: switching it off hurts</td>
</tr>
<tr>
<td>working memory</td>
<td>S 0.000 ± 0.047 (sufficient -0.024), Eval 1 -0.014 ± 0.015
(sufficient -0.018)</td>
<td>packet recall -0.012 / -0.015; instructions -25%</td>
<td>kept: every sign is negative, though only just inside the noise;
round 3 measured its packet cost (below)</td>
</tr>
<tr>
<td>neuromodulator effect scales and encode/retrieve oscillator</td>
<td>S 0.000, Eval 1 0.000 ± 0.009, BEAM +0.028 ± 0.054</td>
<td>packet recall -0.003 on Eval 1; no cost change</td>
<td>both removed in round 3 (below)</td>
</tr>
<tr>
<td>SQLite sparse route (BEAM only; S and Eval 1 packets are identical
with it off)</td>
<td>BEAM +0.042 ± 0.059 (n = 18)</td>
<td>recall equal; instructions -22%, store 372 → 279 MB, query p95 906 →
811 ms</td>
<td>kept: at 111k turns it is the cheaper path (round 3, below)</td>
</tr>
</tbody>
</table>

### Round 3: leave-one-out inside the kept subsystems (2026-10)

Round 3 recursed into the subsystems the second round kept, on the
merged pruned engine. Each feature was switched off alone
(experiment-only hooks) on LongMemEval-S (48 cases), Eval 1 (250),
BEAM-10M conversation 7 (22.6k turns, 20 cases) and, where it mattered,
an eight-episode LongMemEval-M subset (39k turns). Query-time-only
parameters were swept by re-querying the stored baseline databases,
which reproduced the baseline packets exactly. Every feature fired at
full length (BEAM conversation 7: 45,290 `supersedes` edges, 4,520
reconsolidation and 14,237 retrieval reconstructions, 15,767 labile
memories, 18,162 suppressed memories, 5,403 used memories, 21
working-memory slots, 2 soft anchors). Tier 1 is packet gold-turn
recall; Haiku (rubric v2) judged only packets that changed. Instruction
counts come from `/usr/bin/time -l` because the machine was shared.

<table>
<colgroup>
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
</colgroup>
<thead>
<tr>
<th>Feature</th>
<th>Packets changed (S / Eval 1 / BEAM)</th>
<th>Recall and judge</th>
<th>Cost</th>
<th>Decision</th>
</tr>
</thead>
<tbody>
<tr>
<td>reconsolidation ripple over association edges</td>
<td>never fires: only <code>supersedes</code> edges exist</td>
<td>none</td>
<td>—</td>
<td>removed</td>
</tr>
<tr>
<td>association fanout cache, association embedding cache, suppression
id sets, sparse-key index and procedural table, unread retrieval-surface
fields, superseded penalty, prediction horizon</td>
<td>write-only or unread</td>
<td>identical by construction, tip identity 0/48 and 0/250</td>
<td>first query after a flush −54 ms at 20k turns; RSS −10% on BEAM</td>
<td>removed</td>
</tr>
<tr>
<td>DA modulator and reward prediction error</td>
<td>consumer removed</td>
<td>identical</td>
<td>—</td>
<td>removed</td>
</tr>
<tr>
<td>encode/retrieve oscillator</td>
<td>MSC 0/39 (Durable and Natural)</td>
<td>identical</td>
<td>—</td>
<td>removed</td>
</tr>
<tr>
<td>temporal tie-break in retrieval scoring</td>
<td>0/48, 0/250, 0/20</td>
<td>identical</td>
<td>—</td>
<td>removed</td>
</tr>
<tr>
<td>in-conversation skip in superseded-predecessor hydration</td>
<td>0/48, 0/250, 17/20</td>
<td>BEAM 0.103 → 0.118 (2 better)</td>
<td>+0.5 KB packet on BEAM</td>
<td>removed</td>
</tr>
<tr>
<td>retrieval competition and RIF state</td>
<td>0/48, 0/250, 0/20 with the op skipped; the epoch was the only
<code>Required</code> consolidation hint</td>
<td>111k chain recall 0.062 → 0.082, judge +0.008 ± 0.040</td>
<td>−5% to −15% instructions</td>
<td>removed</td>
</tr>
<tr>
<td>reconsolidation and retrieval-time reconstruction (together)</td>
<td>48/48, 241/250, 20/20, 8/8</td>
<td>S equal (judge 0.000); Eval 1 +0.014 (judge +0.010 ± 0.016); BEAM
+0.007 (+0.014 ± 0.027); M equal (0.000)</td>
<td>−5% to −11% instructions, stores −12% to −23%</td>
<td>removed</td>
</tr>
<tr>
<td>working-memory rehearsal</td>
<td>12/48, 33/250, 0/20</td>
<td>recall equal; judge 0.000 / +0.002 ± 0.003</td>
<td>−3% to −4%</td>
<td>removed</td>
</tr>
<tr>
<td>neuromodulator effect scales (write scale) and NE</td>
<td>5/48, 32/250, 7/20</td>
<td>recall within one case; judge 0.000 / −0.001 ± 0.006 / +0.028 ±
0.054</td>
<td>—</td>
<td>removed</td>
</tr>
<tr>
<td>contiguity budget 6 KiB → 12 KiB and lag 2 for all 11 anchors</td>
<td>query-time sweep</td>
<td>S 0.642 → 0.691, Eval 1 0.597 → 0.629; judge +0.048 ± 0.056 / +0.037
± 0.019</td>
<td>+2 KB per packet</td>
<td>adopted</td>
</tr>
<tr>
<td><code>supersedes</code> edge writing / predecessor hydration</td>
<td>47/48, 249/250</td>
<td>−0.037 / −0.025 / −0.015 recall</td>
<td>−5% instructions off</td>
<td>kept</td>
</tr>
<tr>
<td>contiguity hydration</td>
<td>all</td>
<td>−0.06 / −0.09 recall off</td>
<td>—</td>
<td>kept</td>
</tr>
<tr>
<td>working-memory chunking</td>
<td>0/48, 0/250, 0/20, MSC 0/39</td>
<td>never merged a slot</td>
<td>—</td>
<td>kept pending a workload where it fires</td>
</tr>
<tr>
<td>family collapse in retrieval</td>
<td>5/48, 18/250, 10/20</td>
<td>recall within one case</td>
<td>−1% to −5% off</td>
<td>kept (candidate: wired through the sparse-route cache)</td>
</tr>
<tr>
<td>usage detection and usage counters</td>
<td>0/48, 0/250, 0/20</td>
<td>identical</td>
<td>−1% each</td>
<td>kept: they feed the public
<code>retrieved_count</code>/<code>used_count</code> fields</td>
</tr>
<tr>
<td>soft anchors</td>
<td>0/48, 0/250, 0/20</td>
<td>identical</td>
<td>−0.5% to −1%</td>
<td>removed (public API, user-approved)</td>
</tr>
<tr>
<td>working-memory packet hydration</td>
<td>48/48, 250/250, 20/20</td>
<td>−0.012 / −0.015 / 0 recall; judge +0.012 ± 0.053 / −0.008 ±
0.016</td>
<td>packets −66% / −60% / −84%</td>
<td>kept: public API, flagged</td>
</tr>
<tr>
<td>SQLite sparse route at 111,527 turns (five BEAM conversations
chained)</td>
<td>94/99</td>
<td>off: recall 0.076 vs 0.062, judge +0.020 ± 0.052</td>
<td>off: +26% instructions, ingest 170 vs 95 ms/turn past 100k, query
median 45 vs 33 ms, RSS 7.4 vs 6.7 GB</td>
<td>kept</td>
</tr>
</tbody>
</table>

After all round-3 commits, the engine against the merged pruned engine:
packet gold-turn recall 0.642 → 0.691 (S), 0.597 → 0.644 (Eval 1, 19
cases better, 1 worse), 0.103 → 0.118 (BEAM); Haiku-judged sufficiency
+0.060 ± 0.057, +0.039 ± 0.021 and +0.014 ± 0.048; instructions retired
−12% (S), −12% (Eval 1), −24% (BEAM); peak RSS −14%, −3% and −17%.
Artifacts: `.claude/hillclimb/ablation/ablate3/`.

## Experimental Interpretation

The research question changed twice. The v1 hard cut asked how much
long-horizon utility remains when Cortext is restricted to
embedding-space memory, graph structure, shallow replay and bounded
reconstruction. The 2026-10 ablation answered that the lifecycle
mechanisms contributed nothing measurable on long horizons
(<a href="#sec-negative" class="quarto-xref">Section 14</a>). v5 then
asks a different question:

> How far can LLM-free segmentation, hybrid multi-granularity retrieval
> and a small late-interaction reranker go against systems that spend
> LLM calls at write time?

On the one benchmark run under a shared pipeline (LoCoMo), Cortext is
within 2.8 points of Mem0 and ahead of Graphiti, with no write-time
model calls and recall about 17× faster. On the zero-shot BEAM scales,
it is ahead of Exabase M-1 and behind past.dev under the ExaBase
protocol. Most other gaps come from vendors’ own protocols and are
directional only. Several of the strongest numbers (LongMemEval-S,
DolphinBench, MemLens) are on benchmarks that overlap the reranker’s
training pools. The held-out slices of those benchmarks are the fair
comparison. Broader claims still need replicated harnesses against
frozen probes, with evaluator state outside the production library.

## Reproducibility Notes

SQLite is built from the vendored `third_party/sqlite` source tree.
Native verification does not require a system SQLite development package
or shared `libsqlite3` dependency.

`QUARTO_DISABLE_GIT=1 QUARTO_DISABLE_GITHUB=1 quarto render docs/paper`
is the manuscript regeneration command used for local verification.

# Implementation Considerations

## Embedding-First Operations

The current runtime is embedding- and graph-first. Online scoring,
boundary detection, write gating, graph updates, retrieval,
working-memory maintenance, all operate over embeddings, bounded
processor state, and SQLite graph rows. Raw text, audio, and image
payloads are used to produce embeddings and to hydrate returned memories
for applications; hydrated payloads do not feed back into online
ranking.

The hard-cut runtime no longer contains an internal generative model
stack. There is no semantic batch stack, adapter registry, mode-selected
replay, attached static taxonomy, persistent confidence monitor,
bitemporal fact store, or short-term shadow graph. The remaining model
dependency in the core loop is the configured multimodal embedding path.

Retrieved memories are hydrated only after retrieval has selected a
compact ranked set. A retrieved memory without hydrated content is
skipped. Earlier builds resolved `LABEL` and `ASSOCIATION` routing rows
to linked source memories through `has_label` / `derived_from` routes,
ranked those sources query-aware before truncation, and gave them their
own relevance and composite scores. The engine ablation study (2026-10)
removed clustering consolidation, which was the only writer of those
routes
(<a href="#sec-consolidation" class="quarto-xref">Section 12</a>), and
with it linked-source hydration. A displayed memory carries the
relevance and composite score of its own retrieval trace. The C API and
JavaScript addon serialize the same memory object.

Each retrieved text turn is also hydrated with the adjacent turns of its
own source stream (temporal contiguity): the immediate successor and
predecessor of every anchor in rank order, then the turns two steps
away, each clipped to 480 bytes under one 12 KiB budget per context. The
round-3 ablation (2026-10) measured the earlier 6 KiB budget and lag-2
cap (top 6 anchors) as binding. Raising the budget to 12 KiB and giving
every anchor lag-2 turns raised packet gold-turn recall from 0.642 to
0.691 on LongMemEval-S and from 0.597 to 0.629 on Eval 1 (13 cases
better, none worse) for +2.2 KB per Eval 1 packet, and Haiku-judged
sufficiency by +0.048 ± 0.056 (S) and +0.037 ± 0.019 (Eval 1). Budgets
of 16 and 24 KiB gave packets identical to 12 KiB. Switching contiguity
off costs 0.06 to 0.09 packet recall; clipping neighbours to 240 bytes
raised the clipped-text recall proxy but not judged sufficiency, so the
480-byte clip stays.

## Durable Memory Surface

The current v1 runtime uses two durable graph layers plus one bounded
live layer:

-   **Working memory:** active coherent memories used for recent-turn
    continuity, overlap suppression, and prompt reconstruction.
-   **Long-term graph memory:** persisted `memories`, `signals`,
    `embeddings`, `associations`, and reconstruction rows.

Long-term graph memory is the durable substrate. `memories` stores
working and long-term nodes; `associations` stores the directional
`supersedes` edges that memory storage writes. Older stores may still
hold `LABEL`/`ASSOCIATION` nodes and typed edges from removed
consolidation paths; the current runtime neither writes nor traverses
them.

`source_id` is treated as opaque provenance and exact-equality grouping
only. Cortext does not infer application roles, speaker names,
modalities, maintenance commands, or consolidation behavior from
reserved source strings. Explicit maintenance is driven by API calls and
`Signal::force_consolidation`, not magic source identifiers.

Per-source accumulator state (open unit, drift, pacing, refractory)
lives only in process memory and is never persisted. After each root
signal the processor keeps every source with an open unit and at most
256 closed sources, evicting the least recently active closed ones; an
evicted source restarts from fresh state exactly as it would after an
engine restart. Hosts that mint one `source_id` per turn or per call
therefore hold constant per-source memory instead of ~4.5 KB per id. The
AIST text encoder’s repeat-text cache is a 256 KiB byte-bounded LRU for
the same reason: benchmark streams repeat fewer than 0.6% of their
texts, so an unbounded map grew with every ingested turn.

SQLite parameters are checked at bind time, including exact
statement/argument counts, so `SQLITE_TOOBIG`, `SQLITE_RANGE`, and
allocation failures cannot be silently converted into NULL values.
`Process()` and `Flush()` both snapshot mutable processor state until
the database transaction commits; rollback restores working-memory dirty
flags and persisted-record cursors. No operation traverses association
edges at runtime, so the engine keeps no in-memory association fanout
index. An earlier fanout cache served only the reconsolidation ripple;
after that was removed it was rebuilt from a full `associations` scan on
the first query after every flush (54 ms at 20k BEAM turns) and never
read.

Natural and Durable retention enter the same operation chain and
accumulate the current signal before boundary evaluation. Natural leaves
boundary and write acceptance to the accumulator algorithms. Durable
adds an explicit boundary and accepted flush of that same pending unit;
it does not construct a second single-signal memory alongside the
accumulated unit. The Durable packet is the last signal in the flushed
unit exactly once.

`Flush()` closes every source’s open unit before it commits: each unit
takes the boundary path of its last signal (write gate, storage, working
memory, accumulator reset), oldest activity first, under the boundary
type `flush`. Natural units that had not yet met a boundary (a
conversation’s last turns, a two-turn conversation below the boundary
warm-up) were otherwise volatile and lost at the next restart; under
per-conversation Natural ingest that was 19% of turns. Units closed by
`Flush()` reach the sparse route as ordinary writes do (dirty rows
sealed by a later signal), so the published route survives the flush. A
failed `Flush()` rolls back and reopens exactly the units it closed; its
snapshot copies only those accumulators, so it stays independent of idle
sources and store size.

WAL checkpoints never run on the ingest thread. Each file-backed SQLite
store owns a background checkpointer (its own connection and thread)
that runs PASSIVE checkpoints when the WAL reaches the auto-checkpoint
size (1,000 pages) and after every `Flush()`; commits only request one.
`Flush()` therefore returns once its transaction is in the WAL
(process-crash durable), and the background checkpoint syncs it against
power loss; recovery replays only whole committed transactions, so the
store is consistent after either crash. Durable retention’s post-commit
barrier is one fsync of the WAL that holds its commit (power-loss
durable when `Process()` returns) instead of a TRUNCATE checkpoint;
copying the WAL into the database is left to the background
checkpointer. Under synthetic disk pressure (8 writers, 256 MB each with
`F_FULLFSYNC`) a foreground Flush checkpoint stalled 99-161 ms and
commits carrying SQLite’s foreground auto-checkpoint up to 378 ms; with
the background checkpointer Flush took at most 5 ms and the remaining
commit stalls were the WAL page writes themselves.

Resident embeddings live in one process-wide arena. Every in-memory
consumer of a stored vector (the processor retrieval surface, the
historical and current supersession/seed surfaces, rollback snapshots,
and borrowed supersession candidates) holds an 8-byte handle to an
immutable float32 row instead of its own copy. Rows are
content-addressed bitwise and reference-counted, so the
retrieval-surface, historical, and current entries of one memory share a
single 1 KiB row, a reconstruction is a new row (copy-on-write), and a
rollback snapshot copies handles rather than vectors. Stored embedding
rows are append-only in SQLite (inserted and deleted, never updated),
which is what makes content addressing exact. Rows sit in 64-byte
aligned chunks that never move; exact scans read them in place,
replacing the two flat search buffers that duplicated every historical
and current vector, and evaluate the same per-row Eigen expressions as
those buffers did, so ranking is bit-identical. The cosine-family
collapse keeps only a side table of block norms per historical entry and
re-expands the normalized vector from the row on use.

### Retrieval-inhibition state (removed)

Migration 28 had added a lazy retrieval-induced-forgetting
representation: a global log-recovery clock, generation resets,
per-memory anchored inhibition rows, a connection-local effective-value
view, and an in-memory SQLite active-epoch projection that could raise
the `Required` consolidation hint. Caller-supplied SQL was wrapped to
materialize and rebuild that state around every statement. Retrieval
competition was removed
(<a href="#sec-advanced" class="quarto-xref">Section 10</a>), and
migration 45 drops the three tables and the `suppression`,
`suppression_ts`, `suppression_count` and `recovery_time_start` columns.
Caller SQL still invalidates every database-derived processor cache
before it executes.

### Bounded activation observation shadow

A separate private experiment observes whether long-lived retrieval can
use a fixed-size sparse activation graph. It is disabled unless the
internal `CORTEXT_BOUNDED_ACTIVATION_SHADOW` flag is set, cannot select
candidates or scores, and never writes canonical tables.
`SignalProcessor` prepares a lookup against the last published shadow
generation, commits the ordinary persistent SQLite transaction, and only
then publishes the immutable observation once. Nested processing
journals its prepared observations while SQLite uses savepoints;
releasing a nested savepoint never publishes. The outermost root commit
publishes the journal in event order, while an outer rollback discards
the complete journal. Rollback, cancellation, or commit failure
therefore leaves the published generation unchanged. Natural and Durable
share this publication edge; Durable’s later checkpoint cannot replay or
reverse it. A duplicate, generation gap or regression, digest mismatch,
or shadow-local failure disables and discards the disposable graph
without changing the canonical result.

The graph contains *R* = 8 + round (24*S*) roots and at most
*K* = 128 + round (512*F* + 512*T*) leaves. Root and leaf beams,
representatives, neighbor degree, activated consolidation leaves, and
cadence are likewise fixed functions of F/S/T. Storage is initialized
once to *R* + *K* + *K**E* embedding slots and *K**D* neighbor slots.
Normal mutation, recall, and internal consolidation each have
closed-form comparison ceilings derived from those capacities; no source
identifier or modality participates in routing or capacity.
Consolidation rebalances leaves, updates root centroids, and refreshes
only a fixed number of activated leaf neighborhoods.

The disposable graph is unavailable after restart until an explicit
measured rebuild from authoritative SQLite completes. That rebuild
currently visits retained history linearly, so it remains a named
production-cutover blocker. The experiment therefore establishes
implementation and measurement seams for fixed sparse activation; it
does not replace exact retrieval, make the shadow authoritative, or
claim bounded startup or live behavior.

### Packed sparse-route hybrid candidate

The follow-on benchmark candidate replaces the centroid shadow’s single
lossy route with two independently constructed sparse graphs and a
persistent 512-neighbor route for each sealed embedding. Query work is
fixed independently of retained-history size: each graph receives at
most 1,280 exact embedding comparisons, and the merged candidates are
reranked by the same cosine distance and identity tie order as the exact
control. During a 512-event active epoch, each new embedding may propose
at most 64 reciprocal route updates. Building a sealed graph is likewise
capped at 16,384 construction comparisons per graph. These capacities
are functions of the engine knobs and embedding geometry; neither
`source_id` nor modality participates in routing.

The benchmark persistence layout keeps SQLite authoritative. A sealed
snapshot stores normalized vectors, the two packed adjacency arrays, and
packed routes as four SQLite blobs, with 1,024 deterministic anchor rows
used to enter the graphs after restart. The current epoch remains a
bounded connection-local SQLite database. Restart opens the four blobs
and visits only metadata plus the anchor rows; it does not replay
retained embeddings. Consolidation is the epoch boundary: it seals
derived routes and then begins a fresh active epoch, producing the
intended sawtooth without introducing a second Natural/Durable write
algorithm. Durable would remain the shared write followed only by its
checkpoint barrier.

This layout is still a benchmark design, not the current production
retrieval path. The first quality corpus contained all embedding rows,
including signal-only rows that the public seed path does not rank as
long-term memories; its control was therefore mislabeled as current
public retrieval. A corrected export uses current long-term memory
surfaces keyed by memory identity. The two-graph design reached exact
top-1 and recall-at-16 of 1.0 over 3,172 such surfaces, but that static
population still omitted runtime timestamp eligibility, supersession
filtering, current-surface selection, and family collapse. A subsequent
15,695-packet run captured 512 actual public retrieval controls across
events 6–15,691. Three standalone attempts reconstructed the
point-in-time route from the final database: all surface versions, one
current node per memory, and base plus replaceable current
reconstruction. Their apparent identity coverage remained near 0.984 and
top-1 near 0.914. An exhaustive dual-surface oracle reproduced those
misses exactly, showing that approximate graph search was not the cause.
The final database cannot reproduce every intermediate current surface
after reconsolidation mutates `memories.embedding_id`; therefore those
quality labels are invalidated rather than treated as candidate
failures. A valid follow-on captures the live surface mutation stream
during replay. No variant is enabled in production. The prototype’s
source graph construction and snapshot writer also have not yet passed
the engine’s transaction-failure, consolidation-publication, end-to-end
latency, or full-replay gates. The description records the algorithm
precisely so those gates test one content-addressed design rather than
an informal approximation.

The replacement evaluation seam is an explicit private live-surface
mutation trace. It is disabled by default and enabled only by the replay
harness. After each successful process event it records ordered
long-term surface upserts and removals as memory identity, embedding
identity, and the 256-dimensional post-encoding vector;
consolidation-phase mutations are recorded separately after the process
phase. Normal execution neither stores nor copies these vectors. The
trace carries no routing branch for `source_id` or modality, and its raw
vectors remain private. A paired capture-off/capture-on replay must
retain exact public behavior and logical SQLite content before the trace
can be used as candidate evidence.

The direct-control harness now observes the engine rather than
reimplementing its eligibility rules. On explicitly enabled benchmark
runs, `GraphAugmentedRetrieveCandidates` exposes its already-ranked
direct seeds to a thread-local trace after timestamp exclusion,
supersession filtering, current-surface selection, family collapse,
exact scoring, and deterministic tie ordering. The Natural replay
records those seed ranks together with the graph-expanded and hydrated
public ranks only on retrieval-active events, then selects an inclusive
even sample from that event stream. Normal execution does not construct
the extra seed trace. A paired 200-event replay produced the same
behavior digest and canonical logical-database digest with and without
capture, so the observer is behavior- and storage-neutral at that scope.
Query vectors remain only in the private proof artifact; the
paper-facing audit retains their digests and aggregate coverage. The
complete 15,695-event, 512-query control and live candidate comparison
are reported in the optimization evaluation. This instrumentation
remains a private proof seam rather than a production API.

### Integrated HNSW candidate route

The sparse route is wired into the existing
`GraphAugmentedRetrieveCandidates` operation; it is not a second
retrieval or ingestion API. SQLite-backed routing is the normal internal
path, with a private rollback switch retained for experiments. Natural
and Durable update the same current-memory surface and Durable adds only
its post-commit SQLite checkpoint barrier. Routing consumes only the
256-dimensional post-encoding vector and memory identity, so modality
and the opaque `source_id` never select graph structure, capacity, or
rank behavior.

Every operational work limit is derived from the three public knobs.
After clamping *F*, *S*, and *T* independently to \[0, 1\], the
candidate capacity and legacy-store backfill batch are

*C*(*F*, *S*, *T*) = round (256 + 256*F* + 128*S* + 128*T*),

*B*(*F*, *S*, *T*) = round (64 + 64*F* + 32*S* + 32*T*).

The downstream activation target is independently named:

*A*(*F*, *S*, *T*) = 2*C*(*F*, *S*, *T*) + 2*B*(*F*, *S*, *T*).

The graph neighbor limit is *m**a**x*(8, ⌊*B*/2⌋), the level-zero link
limit is *m**a**x*(16, ⌊*C*/4⌋), and maximum hierarchy level is
⌈log<sub>2</sub>*C*⌉. At each level a new node may update at most
*m**a**x*(2, ⌊*B*/16⌋) reciprocal rows. HNSW construction effort is
*m**a**x*(32, round (25*C*/64)) and query effort is
*m**a**x*(*C*, round (5*C*/2)). The bootstrap, maximum public
visited-node, maximum public queue, construction-node, and
construction-queue limits are respectively 2*C*, 9*C*, 9*C*, *C* + *B*,
and 2*B*. The 9*C* quantity is a hard routing-inspection ceiling, not
the activated working set. After exact reranking, no more than *A*
identities may proceed into eligibility, family collapse, and final
ranking. Exact cosine family checks have a separate 2*C* per-event
ceiling. The breadth-first adjacency fetch batch is
*m**a**x*(8, ⌊*B*/4⌋). The larger query and bounded cache coefficients
were selected by copied-late quality probes: 2*C*/*C* and 3*C*/2*C*
search candidates missed the exact-identity gate; 8*C*/4*C* with a 16*C*
cache passed on the initial store but missed one exact top-1 after the
bounded supersession surface changed the live graph. A later expanding
5*C* → 12*C* envelope restored exact sampled quality but made both query
and construction work grow with query age. The retained candidate resets
public route inspection to 8*C*, advances it by *R* only until the 9*C*
ceiling, fixes downstream activation at *A* = 2*C* + 2*B*, and reserves
the smaller *C* + *B*/2*B* envelope for graph construction. These are
knob-derived finite limits, not midpoint constants.

Sparse activation is candidate generation rather than an exact
completeness certificate. The route first rejects an activated proposal
when ordinary eligibility leaves fewer rows than the requested seed
count. It then performs the exact cosine-family collapse and repeats the
cardinality check. This second check matters when the sparse proposal
contains enough eligible memory rows but several belong to the same
semantic family; if fewer distinct families remain, the operation
rejects the sparse cache certificate and reopens the complete
processor-surface or authoritative SQL seed path instead of returning a
silently underfilled result.

Consolidation changes activation locality without changing SQLite
authority or introducing a second route. Restart opens at the 9*C*
ceiling because in-process query age is not durable; every successful
recenter resets the shared path to the 8*C* floor, and subsequent
retrieval-active queries advance queue effort and actual-node ceiling
together by *R* until 9*C*. A successful route seal normalizes the
consolidation embedding, runs the canonical bounded search around that
semantic center, and persists at most *A* resulting memory identities as
the next activation snapshot. The canonical maximum-level entry
continues to own global HNSW descent. Ordinary retrieval loads the
packed snapshot through its independent *A* row budget and uses all
valid snapshot identities as deterministic level-zero entry points.
Neighbor rows discovered from those entry points remain charged to the
current canonical 8*C*–9*C* budget, so consolidation can steer the walk
without widening either envelope. The internal rebuild of a replacement
snapshot deliberately excludes the prior snapshot’s identities from its
frontier seeds; only the canonical route and the new consolidation
embedding may choose the replacement centroid neighborhood. The same
identities may still be rediscovered through canonical adjacency or the
active indexed fill and retained by exact reranking; this rule removes
stale seeding authority, not otherwise relevant memories. The internal
search used to construct a recenter snapshot does not advance the
retrieval-age ramp. If snapshot persistence fails, both queue effort and
the actual-node budget remain at their pre-attempt values; only a
successful recenter resets them to 8*C*. The floor, increment, and
ceiling are derived from F/S/T rather than independently tunable fourth
knobs. Every query records its ceiling and actual visited rows;
canonical rows above the current ceiling or above 9*C* fail the audit.
The persisted snapshot remains independently charged under its *A*
ceiling and is joined with the canonical candidates for final exact
reranking. The returned set makes the existing formula explicit: its
best 2*C* canonical identities are protected, while consolidation may
supply the remaining 2*B* identities in *A* = 2*C* + 2*B*. The final
selected set is reranked by exact distance. Total fetched-row work is
therefore bounded by 9*C* + *A*. A separate returned-identity counter
still fails above *A*, so a changed semantic center cannot be mistaken
for an unbounded working set. The normalized center, snapshot
generation, and packed identity list live in the single SQLite
route-metadata row and survive reopen; malformed, stale, duplicate, or
over-capacity metadata invalidates the route. Graph and memory rows
remain authoritative in SQLite.

Sparse HNSW activation is candidate generation rather than an
eligibility-complete seed surface. Routing nodes such as legacy
`ASSOCIATION` memories in older stores may consume activated slots and
are then removed by the public long-term seed predicate. If that
filtering leaves fewer than the requested candidate limit, the operation
discards the underfilled sparse proposal and executes the existing exact
SQL seed path. Consequently the fixed sparse envelope is not
misrepresented as complete after operation-specific filtering; the
fallback may still perform store-sized exact work.

Activation-snapshot rows are read from the same resident generation
graph as every other traversal row, so no query decodes a node row from
SQLite; query distances are still computed exactly per query. The
resident graph mirrors the one persisted route (each seal commits to
SQLite before its rows are replaced in memory) rather than acting as a
second write path, and it contains no modality or source-id predicate.

Earlier builds also bounded the emotional cascade’s breadth-first
propagation with a first-source-wins winner collapse and three
F/S/T-derived limits. The cascade was removed with emotional
consolidation
(<a href="#sec-negative" class="quarto-xref">Section 14</a>), and these
limits went with it; Appendix E keeps the measurements.

The cycle auditor classifies retrieval independently from whole-engine
process shape. It selects retrieval-active events from explicit route
activity, queue, ceiling, visit, distance, snapshot, or activation
evidence. In sawtooth mode it requires the exact F/S/T-derived 8*C*
reset floor, *R* increment, and 9*C* ceiling, canonical visits no
greater than the current ceiling, snapshot visits no greater than *A*,
combined row visits no greater than 9*C* + *A*, nonzero unique distance
work no greater than the fetched-row union, and activated identities no
greater than *A*. Mature cycles must show the reset on consolidation and
pass the preregistered peak/trough, resampled-shape, and late-template
gates. A route-active event with missing or zero work therefore fails
rather than disappearing from the sample. Missing recenter evidence, a
changed post-consolidation canonical ceiling, or any work value above
its declared derived envelope also fails closed. Activated-identity
overlap is a separate centroid movement measurement; without it the
classifier and work envelope may pass, but the full retrieval-cycle
contract remains unproven. Process, throughput, and operation-slope
gates remain separate, so bounded routing cannot conceal a rising write
path.

At neutral knobs these formulas retain the evaluated midpoint:
*C* = 512, *B* = 128, *A* = 1, 280, 64 neighbors, 128 level-zero links,
maximum level 9, 8 reciprocal updates per level, construction effort
200, query effort 1,280, and 1,024 exact family comparisons. The
canonical floor is 4,096 rows, the increment is 8, and the
canonical-plus-snapshot row ceiling is 9*C* + *A* = 5, 888 at neutral
knobs. The all-low and all-high endpoints are not aliases for this
midpoint; the complete 3 × 3 × 3 structural grid is tested.

When the current surface contains at most *C* rows, the exact scan is
already inside the bound. Above it, one deterministic persisted HNSW
hierarchy proposes a sparse frontier. The same SQLite route then
completes the current query’s knob-bounded routing envelope with a
deterministic indexed slice of active rows, wrapping once around the
canonical entry identity, until the current 8*C*–9*C* actual-node
ceiling is reached or the surface is exhausted. Every routed row—whether
reached by an HNSW edge or by envelope completion—is exactly reranked.
The independently persisted consolidation snapshot then contributes at
most *A* additional rows to the same exact union, after which the
operation returns at most *A* memory identities to the existing
downstream path. Timestamp eligibility, current-surface membership,
long-term kind, cosine-family collapse, final scoring, and deterministic
ties remain authoritative. This sparse activation route fixes
disconnected-node blindness without introducing a second retrieval or
write path; HNSW controls locality while the derived envelope controls
exact coverage and work.

Deterministic envelope completion reads only rows with `active=1`
through the existing `(generation, active, memory_id)` index. Sealed
removals therefore do not consume the fixed node budget or hide an
isolated active identity just beyond an inactive slice. CMake and Zig
compile the same pinned HNSW safety patch. The Zig graph copies the
header-only dependency into a build-local declared output and patches
that copy, leaving the content-addressed global package cache immutable
and safe for concurrent builds. The patch is checked and applied with
the build-local output as the working directory, so no host-native
output path is passed through Git’s path parser. The first replacement
Windows run still rejected the patch content check: Windows checkout had
converted the unified diff to CRLF, which reproduces locally as a
corrupt patch even though the same bytes with LF apply cleanly. The
repository therefore marks this exact patch `text eol=lf`; this is a
checkout invariant, not a platform-specific algorithm branch. The patch
itself is also a declared build input, so either content or
checkout-contract changes invalidate the preparation step instead of
reusing a stale patched output. Local POSIX and CRLF-adversarial proof
pass; replacement Windows CI remains the execution owner. The Zig
package manifest includes the `cmake` directory so downstream package
consumers receive both the preparation script and its content-checked
patch rather than only the top-level build graph.

When a sealed removal targets the canonical hierarchy entry, replacement
is selected by highest level and then lowest memory id through the
`(generation, active, level DESC, memory_id ASC)` index. The query reads
at most the staged removal count plus one row: at most that many leading
rows can be excluded by the same seal. The replacement’s level becomes
the new `max_level`, so restart never begins from an inactive entry or
asks a lower level node for nonexistent upper adjacency. If the removed
node later becomes active again without changing its embedding,
row-addressed sealing preserves its persisted links and level and
promotes it back to the canonical entry whenever that level exceeds the
current active maximum, or when it has the same maximum level and the
lower memory id required by the canonical `(level DESC, memory_id ASC)`
order. Thus metadata continues to name the highest active hierarchy root
with its deterministic tie after both deletion and reactivation. If any
sealed removal overlaps the persisted consolidation activation snapshot,
the snapshot entry, generation, centroid, and identity list are cleared
in the same transaction; in-process and restarted queries then use the
canonical route until the next successful recenter builds a new bounded
snapshot. Restart validates that the metadata entry exists in the same
generation, is active, and carries the recorded maximum level. A legacy
route with a dead or mismatched entry fails closed to exact retrieval
and the same bounded builder, rather than waiting for a future removal
to repair already-stale metadata. It likewise resolves the at-most-*A*
persisted activation identities and requires every one to remain active
in the same generation, covering snapshots written by the prior
implementation before removal-overlap clearing existed.

Within one activation query, the row-addressed route stores each
persisted node’s exact query distance beside that query’s local node
view. Higher-level descent, level-zero expansion, best-set maintenance,
and the final exact activation rerank reuse that value instead of
recomputing the same 256-element distance. The canonical cache is
query-local and bounded by the same 9*C* maximum fetched-node ceiling;
snapshot materialization is separately bounded by *A*, and both are
discarded after the call. Pending updated embeddings use a separate
distance evaluation from their older persisted row, so an identity
cannot reuse a stale vector merely because its numeric ID is unchanged.
The route records both visited nodes and exact distance evaluations, and
requires the latter to be at most the unique union of canonical,
snapshot, and knob-bounded pending-delta rows. This changes only
repeated arithmetic: activated identities, order, downstream
eligibility, SQLite authority, and Natural/Durable ownership are
unchanged.

SQLite remains canonical for both memory data and the row-addressed
graph. Active graph metadata and node rows are separate from an
unpublished build record. An existing store is backfilled in batches no
larger than *B* and the build is invisible to retrieval until its active
count matches the authoritative current surface. Exact retrieval remains
the candidate and the control during this interval. A durable
dirty-identity journal records every route upsert or removal in the
caller’s authoritative transaction. When the complete active journal
contains at most *C* identities, retrieval stages those authoritative
current embeddings or removals as an exact bounded delta over the
persisted hierarchy. More than *C* dirty identities fails closed to the
exact path. The journal query reads at most *C* + 1 identities only to
distinguish a complete *C*-row delta from overflow. At neutral knobs
this live-journal limit is therefore 512 and its overflow sentinel is
row 513. Historical ordered backfill remains independently bounded by
*B*. After advancing at most *B* = 128 historical rows, the
implementation tests whether the in-memory ordered iterator has another
entry. Thus 129 is the derived logical *B* + 1 boundary, not a physical
129th-row read and never a 129-row work batch. Opening the sparse route
loads one metadata row, validates at most *A* packed activation
identities from that row, and then loads every node row of the route
generation into memory once, in fixed 4,096-row chunks. Traversal reads
adjacency and vectors only from that resident graph; each committed seal
replaces exactly the rows it wrote, and SQLite remains the durable
authority the graph is reloaded from after restart or recovery. A
resident node keeps its adjacency and a handle to an arena row: the
memory’s current raw row when normalizing it reproduces the stored
normalized vector bit for bit (the distance then normalizes each
component on the fly, exactly as the stored vector was formed),
otherwise the stored vector itself. The per-query 9*C* canonical plus
*A* snapshot row envelope is unchanged, including the slot an identity
outside the generation consumes. On the 111k-node BEAM-chain snapshot,
513 activated searches return identical identity lists (same hash) and
per-search CPU falls from 16.4/24.8 ms (p50/p95) with per-node SQLite
fetches to 3.8/4.6 ms. Route opening is therefore *O*(generation), like
`SignalProcessor` startup, which already materializes the complete
current memory surface before retrieval begins.

Live writes do not grow an unbounded in-memory construction delta for
either active or unpublished routes. Each construction edge drains at
most *C* dirty identities from the SQLite journal and may independently
advance at most *B* ordered historical backfill rows. The larger dirty
bound is still knob-derived and lets an epoch that changes more than one
backfill batch converge instead of permanently chasing its own journal.
A successful seal deletes only the dirty identities included in that
bounded root set; any remainder stays authoritative for the next edge,
and an unpublished build cannot publish until both ordered backfill and
the journal are complete. Thus a busy active route or legacy-store
migration cannot turn the sparse index into a second unbounded write
path.

Normal writes append only their memory identities to the durable
journal. A successful seal writes changed nodes and knob-bounded
reciprocal neighbors, advances graph metadata, and clears only its
staged root identities in the same SQLite transaction. A full neighbor
list admits a closer new node by replacing a member, but never a member
for which that link is its only level-0 inbound link (the resident graph
keeps the exact level-0 in-degree of every row); and a new node that no
neighbor admitted by distance is linked from its nearest neighbor that
can give up such a member. Distance-only replacement had left 1.5% of
nodes with no inbound link and 2.9% unreachable from the entry at 443k
nodes. Neighbor proposal during this seal uses the *C* + *B* node and
2*B* queue construction envelope, not the public 9*C* activation
envelope. If an existing active node’s normalized embedding moves beyond
the fixed 10<sup>−6</sup> cosine-distance tolerance, the generation is
invalidated rather than pairing the new vector with stale adjacency.
Exact retrieval remains authoritative while the existing bounded
backfill constructs a replacement generation. The rejected
incremental-relink experiment is reported in the optimization section. A
failed authoritative transaction cannot leave a durable route mutation.
An ordinary interrupted or failed seal leaves the identities journaled
for restaging. Intentional moved-embedding invalidation instead abandons
the incoherent generation and clears its journal; the authoritative
current surface feeds the replacement bounded build while exact
retrieval remains authoritative. This gives consolidation a real
route-epoch boundary without a store-sized graph rewrite.

Application hydration uses a bounded-output rule when a memory has no
materialized memory or reconstruction blob. SQLite retains every
authoritative signal. The fallback first applies the memory’s modality
predicate in SQLite, then walks newest-first keyset pages and loads
payloads until either *B* rows pass the same modality, MIME, and
payload-shape predicate used by the public surface or that memory’s
candidates are exhausted. Accepted payloads are returned in
chronological order. Thus neutral knobs hydrate at most 128 fallback
signal payloads per returned memory, all-low hydrates 64, and all-high
hydrates 192. The accepted-output bound uses neither modality nor
`source_id` as a budget input, while filtering prevents a newer run of
nonmatching or mislabelled payloads from hiding an older matching
payload. This correction does mean that malformed legacy history can
require more than *B* candidate rows to be inspected; bounded output is
claimed, but bounded fallback scan work is not. The runtime records the
derived output limit and the number of fallback payloads actually
materialized on each public call.

The same bounded activation surface now serves supersession selection
after the current memory population exceeds *C*. `MemoryStorage` asks
the persisted route for at most *A* identities, applies self, timestamp,
and kind eligibility to that activated set before the knob-derived
candidate cut, and then exactly reranks the surviving current
embeddings. Threshold, duplicate-band, edge-count, and deterministic tie
rules remain unchanged. This prevents a nearer ineligible activated
identity from consuming the top-k slice and hiding an eligible duplicate
immediately behind it. On this sparse path the latest current embedding
is the memory’s active centroid. A bounded HNSW result is now treated
only as a candidate set, never as proof that it covers the exact current
population. The historical cache’s exact coverage/ranking proof
therefore remains active after sparse routing; when that proof cannot be
established, the existing SQLite fallback remains enabled. This prevents
an approximate current result from suppressing a predecessor outside the
activated subset. It deliberately permits history-sized supersession
verification work on a write, so the 9*C* + *A* retrieval-row ceiling
does not imply flat supersession-write cost. This is the accepted
correctness side of the write-throughput/retrieval-symmetry tradeoff and
keeps production-wide boundedness unclaimed. Until the route is
published, when the dirty delta exceeds *C*, or when route validation
fails, the operation retains the exact fallback. This is one shared,
modality- and source-agnostic candidate route for Natural and Durable;
it does not add a second write API or make the connection-local cache
authoritative. The 4,000-event knob ablation predates this
exact-coverage correction and its remaining full-horizon limits are
reported in the optimization section.
`TRACE[state:integrated_hnsw_sparse_route_experiment]` and
`TRACE[state:sqlite_hnsw_knob_ablation]`.

Rollback snapshots move the private accumulator map, working-memory
records, and their blob-reference vectors out of the generic context
copy and restore their ownership immediately afterward. The active
source’s bounded scalar accumulator fields remain in the rollback
snapshot. An ordinary non-boundary Natural packet therefore journals
original vector lengths and appends in place without copying the prior
unit. A private rollback hook takes one deep backup of the active source
immediately before a path may mutate or clear its pre-existing records:
accepted storage, an accumulator flush reset, or an accepted interrupt
reset. Working-memory appends roll back by original vector length;
eviction moves the erased slot’s record-vector ownership into the
journal and maps surviving slot identities without copying their
history. A separate full-map hook is reserved for operations that
explicitly mutate foreign-source accumulator topology. Failure restores
the applicable ownership or backup; successful processing discards it.
This changes snapshot ownership only, not public state layout,
accumulator scoring, boundary decisions, write acceptance, or retrieval
ranking. The ownership journal is enabled only for the private
production operation root, whose mutation sites carry that contract. A
caller-supplied `IOperation` root retains the automatic complete-context
snapshot, preserving exact rollback for hookless destructive operations
and synchronous nested processing without a new public API requirement.

## Current Operation Pipeline

The production operation chain is:

``` text
CoreStage:
  embedded centroid init, focus and sensitivity priors, coherence,
  accumulation, drift, focus spread, prediction error, uncertainty,
  focus/sensitivity (emotion projection), effective focus, metrics,
  neuromodulators (ACh), RLS metric weights, composite score,
  accumulator scores, signal history (score window, dt_ema),
  recent context, boundary, capacity flush, write gate

StorageStage:
  memory storage, signal metrics

RetrievalStage:
  streaming pacing, graph-augmented retrieval, rate/MNI gating,
  memory-usage detection

FeedbackStage:
  memory usage counters, working memory, accumulator reset
```

`UpdateMood`, `UpdatePrecisionDelta`, `InitializeStabilityPriors` and
the threshold half of `UpdateThreshold` were removed (migrations 57 and
58). `CheckSpikeBypass` became `CheckCapacityFlush`
(<a href="#sec-thresholding" class="quarto-xref">Section 5</a>). Probe
mode intentionally runs only `CoreStage` and `RetrievalStage` so replay
probes can inspect retrieval without mutating the durable store.

## Consolidation Implementation

Consolidation is explicit. The runtime does not schedule it
automatically and does not expose mode variants. Callers invoke
`Consolidate()` or inject a signal with `force_consolidation=true`; the
signal dispatcher then runs no operations, and the processor seals and
recenters the sparse retrieval route. Clustering consolidation was
removed in the 2026-10 engine ablation
(<a href="#sec-consolidation" class="quarto-xref">Section 12</a>). The
normal core, storage, retrieval, feedback, episode, object-store,
working-memory, signal-counter, and rate-observation paths are not
maintenance steps.

Emotional consolidation and the emotional cascade were removed in the
2026-10 engine ablation, together with their private emotional-metadata
cache, fixed point and topology footprint
(<a href="#sec-emotional-consolidation"
class="quarto-xref">Section 10.12</a>).

## Retrieval Implementation

<a href="#sec-retrieval" class="quarto-xref">Section 6</a> and
<a href="#sec-hydration" class="quarto-xref">Section 7</a> specify the
v5 retrieval and hydration algorithms. This section records
implementation detail and stage measurements behind them.

Read-only (ephemeral) text queries add a lexical seed channel to the
vector seeds. Each stored long-term text memory is indexed in a
contentless SQLite FTS5 table (`memory_text_fts`, rowid = memory id,
porter-stemmed `unicode61` tokens) inside the savepoint that writes the
memory, so the index commits, rolls back, and is deleted with it; the
text itself stays in the object store. A query ORs its non-stopword
words (at most 16, each a quoted FTS5 string), dropping words whose
index term occurs in more than max (64, 0.05*N*) of the *N* memories:
such terms carry little BM25 weight, and their posting lists dominated
query time on a 22.6k-turn store (FTS5 time per query fell from about 14
ms to 2 ms). A connection-local scratch FTS5 table with the index’s
tokenizer maps query words to index terms, and an `fts5vocab` table
supplies their document frequencies. The query takes the 20 best
`bm25()` hits that pass the same seed eligibility as vector seeds
(long-term, vector-seed eligible, written before the exclusion time).
Vector seed ranks and lexical ranks are fused by reciprocal rank,
∑1/(60 + *r*), which orders the output; the semantic score of every
candidate is unchanged. The channel targets turns that share a query’s
rare words (names, numbers) but sit far down the vector ranking: on the
250-case packet-sufficiency evaluation, gold-turn recall in the packet
rose from 0.60 to 0.79 and judged sufficiency rose by 0.13 (train) and
0.22 (test); on the 48 LongMemEval-M haystacks (about 4.9k turns each)
packet recall rose from 0.41 to 0.67 and judged sufficiency from 0.53 to
0.82, with query p95 and ingest time within 2% of the vector-only
engine. Ingest-time retrieval stays vector-only; lexical seeds there
cost per-turn time without improving recall.

A query may carry a reference time *T* (`Cortext::RetrievalTime`), in
the past or the future, as signal metadata compared with stored memory
times, never as text. Unset, retrieval is unchanged (packets
byte-identical on the LoCoMo clock replay). Set, a temporal seed channel
(`src/operations/temporal_seed_internal.cpp`) returns the *K* = 10
long-term memories nearest *T* by interval distance: 0 when *T* falls
inside \[start, end\], else the distance to the nearer edge. Three range
scans over partial indexes on long-term `start_ts` and `end_ts` (schema
migration 55), each limited to 2*K* + 4 rows, find them in
*O*(log *N* + *K*). The list joins the reciprocal-rank fusion with
weight 0.5, 0.5/(60 + *r*), so a memory found only by time ranks after
every dense seed, while a dense or keyword seed near *T* rises; the
channel adds candidates and removes none. Only the query’s still-open
write unit stays excluded. With `as_of`, every seed path and the
attached later turns and restatements also exclude memories that start
after *T*, an as-of view of what was stored by then. An oracle probe
that sets *T* to each LoCoMo question’s earliest gold turn raised
evidence hits in the packet from 1767 to 1783 of 1979 questions (17
gained, 1 lost), with query p50 0.07 to 0.17 ms higher and p95 within
the spread between run orders; it bounds the mechanism, not a benchmark
score.

Queries rerank the head of the fused candidate pool with a learned
late-interaction model over the frozen AIST encoder
(`models/AIST-87M-GGUF/AIST-87M-LI.gguf`, 102,469 weights or 0.10M
parameters, `src/models/aist_late_interaction.cpp`; training in
`tools/reranker`). Every item, in any modality, becomes a set of unit
vectors in one 64-d space. The first source is its AIST position
features: text token hidden states and 12x12 image patches, each tower
with its own linear map. The second is its stored 256-d retrieval
vector, through one map shared by all modalities. A stored signal keeps
16 spherical k-means codes of its positions as int8 (table
`memory_signal_li_codes`, schema migration 54, a 1,028-byte blob per
stored payload). A query keeps all of its positions, each with a learned
gate weight. A member scores the gate-weighted mean over query vectors
of the best cosine to its own vectors (MaxSim); a memory scores a soft
maximum over its first eight members plus a rank prior. The model is
distilled from a text cross-encoder (MiniLM-L6) with listwise KL. A
listwise loss on gold memories is added. Image members get soft labels
through their captions and hard labels through MemLens gold, so
cross-modal pairs are trained directly.

The rerank runs on the top 64 candidates after neighbour deferral and
before the output cut. Deferring the neighbours of the reranked picks
instead lowered Eval1 packet recall from 0.938 to 0.892. Some memories
have no member with codes: towers without a trained map (audio) and
signals stored before the model. These keep their slots, and the coded
memories reorder among the remaining ones; a lone global vector cannot
compete with a code set. A query without position vectors keeps the
engine order.

Packet gold-turn recall rose on every set. Eval1 rose from 0.846 to
0.938; its conversations held out from training rose from 0.878 to
0.955. On LongMemEval-S, S rose from 0.808 to 0.992 and M from 0.770 to
0.894. On MemLens, text gold rose from 0.415 to 0.575 and image gold
from 0.344 to 0.396. On BEAM at a 34,800-byte budget, recall at 100K /
1M / 10M rose from 0.615 / 0.413 / 0.274 to 0.696 / 0.518 / 0.369; 100K
and 10M were never trained on.

Judged packet sufficiency was near ceiling. Eval1 went from 0.955 to
0.973 and the MemLens image rubric from 0.852 to 0.878. On BEAM end to
end, the judged score at 100K / 1M / 10M moved from 0.816 / 0.778 /
0.667 to 0.843 / 0.799 / 0.699. DolphinBench passed 0.812 against 0.788
(600 tests), with abstention flat to slightly lower.

The rerank step costs 1 to 3 ms per query. Query p95 stayed within 1.02x
to 1.14x of the engine without it at 10K to 443K memories. Its codes
added 11% to Eval1 store size.

Standing memories are a small query-independent channel for lasting
directives and preferences (“Always include version numbers when I ask
about libraries”, “From now on, answer in French”). Membership is a
property of the stored signal that states the directive (table
`standing_signals`, schema migration 53, which replaced the memory-level
set of migration 51), set when its memory is written: under Natural
retention a directive is one turn of a grouped memory several kilobytes
long, which a memory-level check never admits. Today’s only detector is
a pattern check over a signal’s text (at most 480 bytes, a clause
opening with *always*, *never*, *from now on*, *going forward*,
*whenever I*, *I prefer* and similar, optionally after *please*). The
set keeps the newest 256 members; stores written before migration 53
fill it once on open. Every query ranks the members written before it
within the set, fusing cosine to the query (the signal’s own stored
vector) with a BM25 rank by reciprocal rank. The BM25 runs over word
stems (first six bytes of each non-stopword word) stored with each
member, with document frequencies from the set itself: store-wide
frequencies would discount the topic words that tell directives apart,
and the store’s FTS5 index cannot be restricted to the set cheaply (a
full posting-list scan cost about 20 ms per query on a 10M store). The
best five, within 2 KiB, form a standing block of their own ahead of the
retrieved memories (`Context::standing_memory`), each entry holding only
the directive signals of one memory, never its whole memory or
neighbours. The block takes no retrieved slot: a consumer that cuts the
retrieved list at k keeps the same k retrieved memories (LongMemEval-M
gold recall at k = 11 fell by 0.087 while standing items took ranked
slots, and by 0.024 \[-0.071, 0.000\] with the block). Query p95 at 100K
/ 1M / 10M stays within 0.98 / 0.96 / 1.14 of the engine without the
channel. A directive is stated once and must shape answers long after it
falls out of query similarity: on BEAM the share of
instruction-following gold turns (the instruction and its
acknowledgement) in the 8k-token packet was 0.44 / 0.26 / 0.00 at 100K /
1M / 10M tokens before, and 0.86 / 0.81 / 0.44 after (PR #28, standing
block b5447e25 against the engine without it); gold-turn share in that
packet rose by 0.070 / 0.075 / 0.053 overall, with no other category
losing more than 0.01.

Knowledge updates also reach the packet from the newer side. A query
phrased like a fact’s first statement retrieves that older version,
while the latest restatement, which links to it with a `supersedes`
edge, ranks below it. For each of the eleven best-ranked memories,
packet hydration therefore appends the newest superseding memory’s text
(at most 512 bytes) after the anchor’s own content, so the packet reads
oldest to newest; at most 4 KiB per packet, in one indexed lookup on the
edge target for all anchors. On BEAM this added 0.005 gold-turn share at
each scale and 0.023 \[0.010, 0.036\] to the Haiku-answered 10M score.

Memories grouped from several signals need two more query-time steps.
Under Natural retention with one source per conversation, BEAM
conversations are stored as units of three to five turns (about 10 KB
each with their attached neighbours), so an 8,000-token evidence budget
held three or four retrieved memories, and a unit’s pooled vector sits
between its members, close to none of them. *Evidence budget*: the
engine cannot know a reader’s budget, so a caller may pass one per
query, in bytes (text by its bytes, other media 480 bytes each); unset,
memories come back whole. Set, the standing block and the retrieved
memories with their attached neighbours are fitted to it once the packet
is complete: memories join in rank order at their floor (a grouped
memory’s member most similar to the query in each modality, by stored
shared-space vector, `memory_signal_vectors`), with their neighbours
when those fit, without them when only the memory does, or not at all;
then one member cap shared by every grouped memory grows to the largest
with which the packet fits, kept members in their original order. More
distinct memories are kept trimmed before lower-ranked ones are left
out. *Member scoring*: query seeding scores a grouped memory by the
higher of its pooled cosine and its best member’s cosine, and the twenty
memories with the best members join the seeds. Member vectors of grouped
LONG_TERM memories stay resident as int8 codes (dimension + 4 bytes per
member, read incrementally by signal id; single-signal memories are not
held, so Durable stores are unchanged), and each hit is resolved to its
current memory through the store. Opening a file store in WAL mode reads
the stored member vectors on a read-only connection and thread of its
own while the rest of open runs, and the first query adopts that read:
at 10M the first query after open took 54 ms when it read them itself
and takes 26 ms (26 ms without the member channel). On BEAM under
Natural ingest the gold-turn share of the 8,000-token packet rose from
0.385 / 0.195 / 0.125 to 0.617 / 0.420 / 0.272 at 100K / 1M / 10M
(Durable ingest: 0.675 / 0.472 / 0.296), and the past.dev-protocol score
answered and judged by gpt-5.6-luna from 0.793 / 0.750 / 0.651 to 0.847
/ 0.813 / 0.718 (+0.054 \[0.019, 0.095\], +0.062 \[0.044, 0.081\],
+0.067 \[0.029, 0.100\]); knowledge update rose most (+0.24 at 1M). A
fixed 3 KiB member cap per memory, the first form of this trimming, cost
judged Eval 1 sufficiency -0.016 ± 0.016 when the reader saw the whole
packet (LongMemEval-S -0.006 ± 0.026, LongMemEval-M -0.006 ± 0.043,
MemLens +0.000 ± 0.026); the budget leaves such packets whole. Unset,
judged against the Natural baseline by gpt-5.6-luna (both arms, four
repetitions), Eval 1 is -0.006 ± 0.012 and LongMemEval-S -0.003 ± 0.013
(two repetitions: LongMemEval-M +0.018 ± 0.020, MemLens +0.008 ± 0.025).
With the BEAM reader’s budget passed (8,000 luna tokens, 34,800 bytes)
the gold-turn share of its context equals the 3 KiB cap’s (-0.002 /
+0.001 / +0.002) and the e2e score is 0.842 / 0.806 / 0.715 against
0.844 / 0.811 / 0.695 (paired -0.002 \[-0.018, 0.014\], -0.005 \[-0.019,
0.008\], +0.021 \[0.005, 0.037\]); unset, the whole memories overflow
that reader (gold share -0.144 / -0.115 / -0.093). Paired query p50 /
p95 at 1M and 10M, first queries after open included, are within 1.05 /
1.04 and 1.07 / 1.03 of the engine without both steps.

Production retrieval discovers a broad but bounded vector seed set from
durable memory embeddings. Before the seed cap, it collapses
near-duplicate embedding families to one representative; superseded
targets stay eligible. The family index accumulates exact squared
coordinate differences in 16-coordinate vectorized blocks and rejects a
pair as soon as a completed block exceeds the complete squared-distance
limit implied by the cosine threshold. Surviving pairs use complementary
32-block Cauchy–Schwarz upper bounds in the stored coordinate system and
a normalized Walsh–Hadamard basis, followed by exact cosine checks. The
partial distance condition remains sound because every omitted term is
nonnegative; both upper bounds are also sound, so none can exclude a
pair meeting the cosine threshold. The complementary bases avoid broad
exact pairwise work for both sparse coordinate-aligned vectors and dense
Hadamard-aligned vectors; arbitrary adversarial surfaces may still reach
the exact fallback. One event may execute at most 2*C* exact cosine
checks across all family-collapse stages. Exhaustion conservatively
admits the current candidate to the independently bounded rank/output
stages rather than performing a store-sized comparison tail. The
normalized vector and both block-norm summaries are immutable functions
of an embedding, so the private surface computes them on first family
use and retains them with that entry. Replacing a current surface entry
replaces its summaries as part of the same cache mutation. This removes
repeated normalization and Walsh–Hadamard construction without changing
family order, comparison arithmetic, or fallback behavior. If the
private vector cache is unavailable, durable processing rebuilds it once
from the complete persisted surface and immediately applies the same
family filter before truncation. A failed rebuild installs a failure
sentinel rather than repeating the two complete surface scans on every
request. A unique cached vector retains every long-term memory metadata
alternative that shares its base embedding, and query-time eligibility
chooses an eligible sibling without duplicating vector storage. The
cache separately indexes entries that have at least one long-term
metadata alternative. Retrieval applies the existing timestamp,
current-surface, and supersession predicates to that subset before
evaluating vector distance, so signal-only historical embeddings do not
enter the distance kernel. This is an execution-only index: it does not
change the eligible set, distance expression, tie order, family
collapse, or source and modality semantics. An internal execution
sidecar is keyed by processor-context identity and is not part of the
public processor layout. Its earlier contents (the retrieval-inhibition
active epoch, the supersession-target set and the emotional-metadata
cache) served retrieval competition, supersession exclusion and the
emotional cascade. All three were removed by the ablation study. The
fallback executes a demand-driven sequence of ordered SQL pages over the
current and historical vector surfaces. Each page joins memory metadata
and applies kind and timestamp eligibility before distance ordering,
excludes a historical row when that memory already has a current
reconstruction, collapses byte-identical historical vectors, and uses a
second application of the existing F/S/T-derived seed-search breadth as
its row bound. One in-memory cosine-family index spans every page;
paging stops when the ordinary semantic-family seed budget is full or
the eligible surface is exhausted. Processor hydration tracks
persisted-current-surface currency and processor-surface completeness
independently. The private current-vector search cache is built and
incrementally updated from the processor’s latest reconstruction
surface, even when a test or storage policy intentionally omits the
corresponding persisted-current write. A valid cache therefore performs
distance ordering and cosine-family accounting on latest vectors. If
that cache is unavailable while the processor surface is complete and
original-base lineage remains available, durable retrieval rebuilds a
current-only cache from the processor surface and ephemeral retrieval
scans that surface directly without mutating the registry. Missing base
lineage or an incomplete processor surface falls back to SQL; in that
recovery case the query substitutes each latest reconstruction before
cross-page family accounting. Base families that converge after
reconstruction therefore consume one current-family slot and cannot stop
selection early. Current per-memory reconstructions remain distinct rows
and the corresponding base row for that same memory is excluded. The
same registry keeps an exact index of supersession-eligible historical
memory rows and the base lineage of each current row. Memory storage
first ranks the current surface and records its exact cutoff. When
current and historical populations are identical except for explicitly
tracked changed memories, deterministic embedding-id and memory-id tie
orders agree, and every changed historical row is outside that cutoff,
the current result proves that the historical top-k contributes no
unseen candidate. The historical scan is then skipped; any ambiguous
lineage, reversed tie order, or failed cutoff proof falls back to the
complete eligible historical scan. A historical embedding referenced by
more than one eligible memory also disables both current-population
shortcuts: the historical query ranks embeddings before expanding every
memory sibling, whereas the current surface ranks per-memory rows. The
complete embedding-level pass therefore preserves sibling fanout even
when the per-memory candidate limit is saturated. This proof changes
neither the candidate population nor the chosen supersession edges, but
the current surface distance pass remains proportional to that surface.
Above the sparse cutover the current result comes from the bounded
SQLite route and carries no exact cutoff, so memory storage no longer
verifies it against the complete historical population: it ranks only
the tracked changed memories’ historical embeddings exactly and accepts
the route’s approximate current neighbours for every unchanged memory.
Candidate work is then independent of store size, and the written
`supersedes` edges are as approximate as the route. Rollback and flush
recovery also rehydrate each current entry’s original base ID from
`memories.embedding_id`; they never infer it from the reconstructed
surface. When constructive recall is disabled, both restart hydration
and cache-loss rebuild select the base embedding and ignore
reconstruction/current surfaces. This recovery path preserves an
eligible target behind 900 closer ineligible rows and surfaces a shared
embedding through exactly one of its memory alternatives. A separate
600-member byte-distinct cosine-near family requires two bounded pages
and cannot hide the next semantic target. The ordinary 430-family
regression remains one page and returns no more than the 477-row default
page bound rather than materializing all 1,331 eligible rows. The
embedded sqlite-vec implementation is brute-force, so every requested
page scans the persisted vector surfaces; the optimization bounds rows
returned to C++ and downstream embedding decoding, not the scan’s
asymptotic order or the historical window partition’s internal sort and
materialization work. Ephemeral retrieval uses the fallback without
installing volatile registry state. It returns a compact selected set
from those seeds, ranked by vector similarity alone. Graph expansion
over association edges was removed in the 2026-10 engine ablation: once
clustering consolidation was gone, no traversable edge type was ever
written. `source_id` and modality are metadata, not ranking inputs. When
a current embedding exists, its redundant historical base embedding is
omitted from the historical seed search because the authoritative
surface was already searched. Current writes are excluded by timestamp,
and working-memory overlap filters prevent the retrieval result from
echoing the active memory tail. One dispatch exposes at most one
actually used retrieval candidate, so retrieval does not synthesize
reinforcement pair edges from the packet. Historical public knob helpers
for the removed packet-pair writer remain source-compatible
calibrations; only the step calibration still feeds the production
pruning floor.

The retrieval trace path records a ranking ledger for selected and
rejected candidates. Production graph retrieval populates composite
score, seed relevance, temporal score, and the base field of the
activation ledger. The shared trace structure still contains reserved
processor, durable-source, and evidence fields, but this path leaves
them at zero; their presence is not evidence that those terms
participate in ranking. Deleted label-graph and fact-layer fields are
not emitted.

`Retention::Ephemeral` executes this read path inside a transaction,
assembles the retrieval result, rolls back the transaction and object
transaction, and restores the exact pre-query processor snapshot, so
querying cannot change durable state. Earlier builds also had to skip
retrieval-time reconstruction, usage reinforcement, reconsolidation and
strength mutation for queries; those mechanisms no longer exist.

Schema migrations 25–27 add `consolidation_rate_floor`,
`consolidation_rate_peak`, initialization, and the armed latch to the
singleton processor state. Runtime hinting reads their in-memory private
mirror, so the throughput recommendation adds no per-signal SQL query.
Processor construction loads or initializes the mirror, transaction
rollback and ephemeral processing restore it with the full context
snapshot, successful state persistence writes both values, and processor
teardown erases its lifecycle-owned entry. A committed maintenance
request disarms the hint and resets the next event-derived floor and
peak to the acknowledged rate; ordinary rate recovery establishes a new
range and rearms at the F/S/T-derived threshold. A material relative
drawdown can also rearm at the lower envelope edge, so sustained slow
degradation cannot follow the moving floor indefinitely; the same
material-range threshold suppresses rearming from ordinary jitter. This
event boundary lets a lower throughput regime recover independently of
an old spike without a count or wall-clock horizon. Empty replay is
acknowledged, while a rolled-back request restores the exact prior latch
and range. State-row restoration does not depend on a nonzero
processed-signal counter, so a maintenance-only disarmed state survives
restart.

## Computational Complexity

### Bounded Active Signal-Vector Snapshot

Migration 30 removes accepted signal vectors from the store-wide
sqlite-vec search surface. The durable `signals` row and object-store
payload remain the authoritative event record, and
`signals.embedding_id` references the memory or working-memory centroid
that represents that event in global retrieval. Exact vectors for only
the active tail are retained in the internal
`cortext_active_signal_embeddings` SQLite table. This is one
SQLite-backed write path, not a second durability system: Natural
ingestion appends through the common path, and Durable adds the existing
flush/checkpoint boundary.

The exact-vector ring must cover both the historical batch and every
signal that restart can place in recent context. Its capacity is

*Q*(*F*, *S*, *T*) = max  (*B*(*F*, *S*, *T*), *N*<sub>*c**t**x*</sub>(*T*) + *K*<sub>*c**t**x*</sub>(*T*)),

after independently clamping each knob to \[0, 1\]. It therefore
contains 64, 151, and 266 rows at the all-low, midpoint, and all-high
corners. The logical rounding rule is C++ `std::lround`, so half-integer
ties round away from zero. For example, *F* = 1/128, *S* = *T* = 0
yields 65 rather than 64. The logical *B* + 1 probe remains an
overflow/completion check; it is not a fetched or processed extra row.
When the table is below *Q*, an upsert occupies its lowest free bounded
slot. When it is full, replacement is owned by the oldest
`(timestamp, signal_id)` frontier; an out-of-order vector older than
that frontier is ignored. Deletion gaps accept the next presented exact
vector and the timestamp frontier resumes ownership as soon as the table
is full. Existing signal identities update in place. Thus row-id gaps
and out-of-order ingestion cannot make a numerically newer identifier
displace a more recent event, and every ordering operation is over at
most *Q* rows. One accepted pipeline event may persist multiple
accumulated signal records and therefore perform multiple bounded
upserts; the measured Natural maximum of 14 ring-table mutations came
from the earlier *B*-capacity experiment and is not a new *Q*-capacity
benchmark. If knobs change, the table rehashes only the newest rows that
fit the newly resolved *Q* and remains bounded by the old or new
knob-derived capacity throughout the transition.

When the ring contains rows, recent-context restart hydration admits
only signals with an exact ring vector; it does not silently substitute
a memory centroid for a missing per-signal vector. Migration 30 itself
creates an empty ring, so the first capacity check copies up to the
newest *Q* legacy signal vectors only when their `signals.embedding_id`
is certifiably distinct from the owning memory centroid (or the signal
has no owner). The main MemoryStorage path performs that check before it
inserts aggregate-linked rows. The auxiliary working-memory persistence
path may insert its new aggregate-linked row first, but the same
`Upsert` immediately replaces that row’s ring slot with the exact event
vector, leaving the newest *Q* slots exact in either ordering. A truly
empty legacy database remains empty. Once any exact ring row exists, an
older store whose prior ring was smaller can temporarily restore a
shorter exact context until new writes fill *Q*, rather than mixing
aggregate vectors into an apparently full exact window. If a later
cascade empties the ring while flattened signal rows remain, the same
certification rule leaves those rows out and begins the new ring only
from exact vectors supplied by later writes. The global vector index
consequently contains memory centroids rather than a growing population
of signal-only decoys. The schema and routing code contain no modality
or `source_id` branch; text, audio, image, shared, and opaque sources
enter after encoding through the same bounded timestamp frontier. Direct
regressions cover mixed modality/source labels, knob changes, all 27
low/mid/high F/S/T combinations, and a 151-vector neutral restart whose
restored identities equal the exact signal vectors. A separate upgrade
regression starts from 150 legacy exact signals, performs the first
post-migration aggregate-linked write, and restores all 151 exact
vectors; it passes 155 assertions. This does not claim bounded
whole-engine restart.

Emotional propagation separately limits enqueued mutation statements to
the existing activation target *A* = 2*C* + 2*B*. This is a hard bound
on writes, not a claim that the current breadth-first emotional
traversal is fixed work. The full traversal remains an explicitly
unresolved optimization surface. A pass that reaches *A* is an
incomplete priority prefix and is not recorded as an emotional fixed
point; a persistent processor context therefore revisits its unresolved
tail on the next event. Commit-table mutation profiling is installed
whenever work-counter profiling is enabled and no longer depends on the
separate consolidation-epoch profiler.

The dominant per-signal cost is exact candidate discovery over the
durable embedding and association surface. With `N` stored memories and
embedding dimension `d`, the current write path computes exact
current/base distances in `O(Nd)` and performs `O(N)` bounded-top-k
selection before writing a knob-capped number of supersession edges.
Working-memory updates operate over bounded live state and remain
independent of total store size.

For an open Natural unit containing `U` signal records, the ordinary
rollback snapshot performs constant-time accumulator-map ownership
transfer plus record-vector ownership moves per working-memory slot and
does not copy the `U` records. Appending the next record remains
amortized `O(1)`. A boundary, Durable flush, or other destructive
active-source transition may copy `O(U)` active-unit records once for
exact rollback, then persist `O(U)` records. Working-memory append and
eviction rollback moves or trims owned vectors and does not copy
unrelated or cumulative working-memory history. This removes the prior
cumulative `O(U^2)` record-copy path across a long open unit; it does
not change the separate `O(Nd)` exact durable candidate search.

The safety checks preserve these bounds: bind-result validation is one
branch per existing parameter, working-memory decay advances one
timestamp per bounded slot and schedules at most one memory-row update
per changed slot during normal persistence, and graph writes perform
only O(1) cache invalidation. None adds a store-size-dependent query or
eager graph reconstruction; working-memory persistence remains bounded
by the configured live-slot capacity.

The retained long-horizon optimization target is that mean process
latency stays approximately flat as the database grows. The July 2026
exact-replay audit did not prove that target. At the end of the
2,636-event replay, the private historical surface contained 6,762
embedding entries but only 6,622 bit-identical vector groups. Exact
grouping could therefore remove at most 140 distance evaluations
(2.0704%); 97.9296% of the population still had to be evaluated. A
non-selecting previous-query triangle-bound probe likewise evaluated
99.975% of historical candidates on average over 1,000 events and
99.804% in the final 100.

The retained implementation removes repeated SQL, row decoding,
duplicate ranking, and redundant query-norm work, but the exact
current/base population and tie contract leave candidate discovery
linear in durable history on this corpus. Further asymptotic gains
require an approved change to candidate semantics,
consolidation/tiering/eviction semantics, storage/index backend, or
public state ownership. They do not depend on hidden decoder calls, and
the current result must not be described as flat storage or flat
throughput.

# Systems Architecture and Performance

This section describes the storage and execution machinery that keeps
the v5 engine realtime at hundreds of thousands of stored turns. It
covers only surviving architecture. The full optimization ledger,
including micro-optimizations and proof traces for subsystems the
2026-10 ablation later removed, is preserved verbatim in Appendix E
(`docs/paper/appendix/optimization_ledger.qmd`).

The design rule throughout is that SQLite stays authoritative. Every
resident structure below is a cache or index that can be rebuilt from
SQLite, and retrieval results are byte-identical whether a structure is
warm or rebuilt. None of these structures branches on `source_id` or
modality.

## Content-Addressed Embedding Arena

Several in-memory consumers hold the same stored vectors: the processor
retrieval surface, the historical and current supersession and seed
surfaces, rollback snapshots and borrowed supersession candidates.
Before the arena each kept its own copy. Now one process-wide arena
holds each vector once, as an immutable, reference-counted float32 row,
and every consumer holds an 8-byte handle.

-   **Content addressing.** Rows are keyed by their exact bits. Stored
    embedding rows are append-only in SQLite (inserted and deleted,
    never updated), so equal bits always mean the same vector and
    sharing is exact.
-   **Copy-on-write.** A new representation of a memory is a new row. A
    rollback snapshot copies handles, not vectors.
-   **Layout.** Rows sit in 64-byte-aligned chunks that never move.
    Exact scans read rows in place and evaluate the same per-row Eigen
    expressions as the flat buffers they replaced, so ranking is
    bit-identical.
-   **Family collapse.** The cosine-family index keeps only a side table
    of block norms per historical entry and re-expands the normalized
    vector from the row on use.

Together with the in-memory route below, the arena cut route search at
111k turns from 16.4/24.8 ms to 3.8/4.6 ms (p50/p95) and peak RSS by 614
MB, with packets byte-identical (PR #28).

## Rollback Journal

`Process()` and `Flush()` must restore exact processor state when their
SQLite transaction fails, and an Ephemeral query must leave no trace. A
naive full context snapshot copied every open accumulator and
working-memory record on each signal. That made a long Natural unit of
`U` signals cost `O(U^2)` record copies in total.

The engine journals ownership instead:

-   An ordinary non-boundary signal records only the original vector
    lengths of the active source, then appends in place: amortized
    `O(1)`.
-   A path that may mutate or clear prior records (accepted storage,
    accumulator reset after a flush, or an accepted interrupt reset)
    takes one deep backup of the active source first, so that transition
    copies `O(U)` once.
-   Working-memory appends roll back by original length. An evicted
    slot’s record vectors move into the journal instead of being copied.
-   Success discards the journal. Failure restores ownership or the
    backup.

A caller-supplied `IOperation` root keeps the automatic full-context
snapshot, so hookless custom operations still roll back exactly. The
private production root is the only one whose mutation sites carry the
journal contract.

## Sparse Route and In-Memory Route

Above a knob-derived population, dense seeding does not scan every
vector. It walks a sparse, persisted HNSW-style route (Malkov and
Yashunin 2018) over the current long-term surface. Its capacities are
functions of the knobs. With *F*, *S* and *T* clamped to \[0, 1\]:

*C* = round (256 + 256*F* + 128*S* + 128*T*),  *B* = round (64 + 64*F* + 32*S* + 32*T*),  *A* = 2*C* + 2*B*.

At neutral knobs *C* = 512, *B* = 128 and *A* = 1, 280. A query inspects
at most 8*C* to 9*C* canonical rows (the floor resets on each recenter
and the ceiling rises by *R* = max (2, ⌊*B*/16⌋) per query), plus at
most *A* rows from the consolidation snapshot. Every routed row is
reranked by exact cosine, and at most *A* identities go on to
eligibility, family collapse and final scoring. If eligibility or family
collapse leaves the proposal short of the requested seed count, the
route discards it and the exact SQL seed path runs instead. A short
result is never returned as complete.

-   **Epochs.** New memories join a dirty journal in the writing
    transaction. `Consolidate()` seals at most *C* dirty identities into
    the route and recenters its activation snapshot on the active-signal
    ring centroid. That is all maintenance does now
    (<a href="#sec-consolidation" class="quarto-xref">Section 12</a>).
    The engine recommends a pass once *C* memories have been written
    since the last one, and requires one at *C*×
    `ConsolidationEscalationMultiplier(T)`. This restored 44
    consolidations on BEAM-10M conversation 7 after retrieval
    competition, the previous hint source, was removed.
-   **Admission.** A full neighbour list admits a closer node by
    replacing a member, but never a member for which that link is its
    only level-0 inbound link. Distance-only replacement had left 1.5%
    of nodes with no inbound link and 2.9% unreachable at 443k nodes.
    After the fix, 98.99% of nodes are reachable and none lacks an
    inbound link (PR #28).
-   **In-memory route.** Opening the route loads its generation’s node
    rows once, in 4,096-row chunks, into a resident graph whose nodes
    hold adjacency and an arena handle. Traversal never reads SQLite,
    and each committed seal replaces exactly the rows it wrote. The
    resident graph mirrors the persisted route; it is not a second write
    path.
-   **Supersession reuse.** Above the cutover, write-time supersession
    search reuses the route instead of scanning history. Historical rows
    visited per turn on BEAM conversation 7 fell from up to 43k to 0,
    and the written edges are approximate (Jaccard 0.985 against the
    exact scan).

The round-3 ablation kept the route on cost alone
(<a href="#sec-experimental" class="quarto-xref">Section 15</a>). At
111,527 chained turns, switching it off raised instructions by 26%,
ingest past 100k turns from 95 to 170 ms per turn, query median from 33
to 45 ms and RSS from 6.7 to 7.4 GB. The route is approximate, though:
with it off, packet gold-turn recall was 0.076 against 0.062 with it on,
and judged sufficiency moved +0.020 ± 0.052.

## WAL Checkpointer Thread

Each file-backed SQLite store owns a background checkpointer: its own
connection and thread. It runs PASSIVE checkpoints when the WAL reaches
the auto-checkpoint size (1,000 pages) and after every `Flush()`.
Commits only request a checkpoint.

-   `Flush()` returns once its transaction is in the WAL, which is
    durable against a process crash. The background checkpoint then
    syncs it against power loss.
-   Durable retention’s post-commit barrier is one fsync of the WAL
    holding its commit, instead of a TRUNCATE checkpoint.
-   Writes take the lock at `BEGIN IMMEDIATE`, so `busy_timeout` covers
    every signal. This removed “database is locked” errors under ingest
    plus concurrent Ephemeral probes.

Under synthetic disk pressure (8 writers, 256 MB each with
`F_FULLFSYNC`), a foreground `Flush()` checkpoint stalled 99 to 161 ms.
With the checkpointer, `Flush()` took at most 5 ms
(<a href="#sec-implementation" class="quarto-xref">Section 16</a>).
Under the I/O-pressure test in PR #28, maximum consolidation time fell
from 728 to 103 ms and maximum `Flush()` time from 696 to 89 ms.

## Family-Collapse Bounds

Before the seed cap, near-duplicate embedding families collapse to one
representative (<a href="#sec-math" class="quarto-xref">Section 4</a>).
Comparing every candidate with every representative would be quadratic,
so each pair passes through sound filters before any exact cosine:

1.  **Partial squared distance.** Coordinate differences accumulate in
    16-coordinate vectorized blocks. A pair is rejected as soon as a
    completed block exceeds the squared-distance limit implied by the
    cosine threshold. Every omitted term is nonnegative, so this never
    rejects a true duplicate.
2.  **Cauchy–Schwarz block bounds in two bases.** Coordinates are split
    into 32 blocks. For normalized *x* and *y*,
    cos (*x*, *y*) ≤ ∑<sub>*b*</sub>∥*x*<sub>*b*</sub>∥∥*y*<sub>*b*</sub>∥.
    The bound is computed both in the stored basis and in a normalized
    Walsh–Hadamard basis *H*, using
    cos (*x*, *y*) = cos (*H**x*, *H**y*). Vectors that are sparse in
    coordinates are dense after *H* and vice versa, so one of the two
    bounds is tight for both kinds of surface.
3.  **Exact cosine** for the survivors.

Both bounds are upper bounds, so neither can exclude a pair that meets
the threshold. The normalized vector and both block-norm summaries are
immutable functions of an embedding, computed on first use and cached
with the surface entry. One event may run at most 2*C* exact checks.
Once that budget is spent, the remaining candidates are admitted rather
than compared, which keeps the work bounded at the cost of an occasional
duplicate in the seed set. A 256-vector orthogonal-basis regression
requires exact comparisons to stay below a quarter of the all-pairs
count.

## FTS5 Top-k Collector

The keyword channel
(<a href="#sec-retrieval" class="quarto-xref">Section 6</a>) runs BM25
over a contentless FTS5 index. Two changes keep it bounded on large
stores.

-   **Document-frequency cut.** Query words whose index term occurs in
    more than max (64, 0.05*N*) of *N* memories are dropped before the
    query runs. Their posting lists dominated query time but carry
    little BM25 weight. On a 22.6k-turn store, FTS5 time per query fell
    from about 14 ms to 2 ms. The df probe itself stops at the cut.
-   **Top-k collector.** FTS5’s `ORDER BY rank` scores every matching
    row with `bm25()`, one document-size read each, and sorts them all.
    One six-word question matched 32.8k rows: 2 ms to match and 42 ms to
    rank. The engine registers an FTS5 auxiliary function,
    `cortext_bm25_topk`, that computes the same score term for term and
    keeps the best 2 × 20 + 4 rows in a heap. The over-fetch covers hits
    that later fail seed eligibility. It also skips the document-size
    read and the score of a row whose upper bound cannot enter the heap.
    At 443k turns, query CPU fell from 57.9/100.9 ms to 24.6/37.6 ms
    (p50/p95) (PR #28).

## Resident Surfaces

Query-time channels read from resident structures that advance
incrementally instead of reloading.

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr>
<th>Surface</th>
<th>Contents</th>
<th>Maintenance</th>
</tr>
</thead>
<tbody>
<tr>
<td>retrieval surface</td>
<td>current long-term vector and metadata per memory</td>
<td>kept across read-only queries and across <code>Flush()</code>, not
reloaded; loaded as a stream at open</td>
</tr>
<tr>
<td>member index</td>
<td>int8 codes (dimension + 4 bytes) of the member signals of
multi-signal long-term memories</td>
<td>reads signal ids above its mark; a WAL store starts a warm read on
its own connection and thread at open, and the first query adopts
it</td>
</tr>
<tr>
<td>cross-modal populations</td>
<td>one aligned matrix per stored modality of long-term signal
vectors</td>
<td>loaded on first use by a query of another modality; then reads only
rows above a long-term mark</td>
</tr>
<tr>
<td>standing set</td>
<td>at most 256 directive signals with their word stems</td>
<td>maintained in the writing transaction</td>
</tr>
<tr>
<td>sparse route</td>
<td>resident node graph of the published generation</td>
<td>replaced row-for-row by each committed seal</td>
</tr>
</tbody>
</table>

What these surfaces bought, by the stage measurements in PR #28 and
<a href="#sec-implementation" class="quarto-xref">Section 16</a>:

-   Keeping the retrieval surface across queries took the query median
    at 20k turns from 690 ms to 14.9 ms.
-   Streaming the surface at open took 443k-turn reopen RSS from 6.77 to
    4.57 GB.
-   Resident cross-modal populations took queries over 125k image
    vectors from 204/243 ms to 20/26 ms.
-   Reading the member index at open, and advancing it past non-member
    rows, took 443k-turn queries from 400 ms to 30.7/44.5 ms p50/p95.
    The first query after open at BEAM-10M takes 26 ms instead of 54 ms.

The late-interaction codes are not resident. A query reads them for its
top 64 candidates only, by one indexed statement.

## Latency Summary

All rows are measurements already reported in the repository. They come
from different stages and stores, so read each row on its own rather
than as one benchmark run.

<table>
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<thead>
<tr>
<th>Path</th>
<th>Store</th>
<th>Measurement</th>
<th>Source</th>
</tr>
</thead>
<tbody>
<tr>
<td>recall (end to end)</td>
<td>LoCoMo, 1,540 questions</td>
<td>9.7 / 13.7 ms p50 / p95</td>
<td>README Benchmarks</td>
</tr>
<tr>
<td>query</td>
<td>443k-turn chain, v5</td>
<td>30.7 / 44.5 ms p50 / p95; reopen RSS 4.57 GB</td>
<td>PR #28</td>
</tr>
<tr>
<td>route search</td>
<td>111k-turn BEAM chain</td>
<td>3.8 / 4.6 ms p50 / p95 (was 16.4 / 24.8)</td>
<td><a href="#sec-implementation"
class="quarto-xref">Section 16</a></td>
</tr>
<tr>
<td>keyword channel CPU</td>
<td>443k turns</td>
<td>24.6 / 37.6 ms p50 / p95 (was 57.9 / 100.9)</td>
<td>PR #28</td>
</tr>
<tr>
<td>late-interaction rerank</td>
<td>10k to 443k memories</td>
<td>0.4 to 3 ms per query; query p95 ×1.00 to ×1.14</td>
<td>PR #28, <a href="#sec-retrieval"
class="quarto-xref">Section 6</a></td>
</tr>
<tr>
<td>first query after open</td>
<td>BEAM-10M</td>
<td>26 ms (was 54)</td>
<td><a href="#sec-implementation"
class="quarto-xref">Section 16</a></td>
</tr>
<tr>
<td>Natural ingest</td>
<td>BEAM-10M conversation 7</td>
<td>8.8 / 19.7 / 67.4 ms p50 / p95 / max</td>
<td>PR #28</td>
</tr>
<tr>
<td>image turn ingest</td>
<td>native image tower</td>
<td>22.9 / 27.9 ms</td>
<td>PR #28</td>
</tr>
<tr>
<td>consolidation (maintenance)</td>
<td>BEAM-10M conversation 7</td>
<td>26.0 / 32.6 ms wall p50 / p95</td>
<td>PR #28</td>
</tr>
<tr>
<td><code>Flush()</code> under disk pressure</td>
<td>synthetic, 8 writers</td>
<td>at most 5 ms (foreground checkpoint 99 to 161 ms)</td>
<td><a href="#sec-implementation"
class="quarto-xref">Section 16</a></td>
</tr>
</tbody>
</table>

Realtime targets were p95 under 50, 100 and 150 ms at 10k, 100k and 200k
memories, and every reranker configuration stayed inside them (PR #28).

## Remaining Costs

The engine is not flat in store size, and this section does not claim it
is.

-   sqlite-vec is a brute-force scan. Below the sparse cutover, and on
    every route fallback, exact seeding is linear in the current
    surface.
-   Write-time supersession verification can still do history-sized work
    below the cutover, or when the route is unpublished or invalid.
-   Opening a store is *O*(generation). It materializes the current
    surface and the resident route.
-   Hydration cost follows the packet: contiguity, supersession
    attachments and evidence fitting are bounded per anchor, but payload
    reads scale with the bytes returned.

Further asymptotic gains need a change to candidate semantics, a storage
or index backend change, or a tiering policy. A tiering policy is a
forgetting policy, and
<a href="#sec-negative" class="quarto-xref">Section 14</a> shows how
badly an unmeasured one fails.

# Conclusion and Future Directions

Cortext stores what it observed and puts its effort into retrieval.
Online segmentation decides what a memory is, and every closed unit is
kept. Retrieval works at three granularities over unit vectors, member
vectors and late-interaction codes, and fuses dense, keyword,
cross-modal and temporal evidence by reciprocal rank. A 0.1M-parameter
reranker over a frozen encoder reorders the head of the pool. Hydration
then rebuilds the context around each hit and fits it to the reader’s
budget. With no language model on either path, this reaches 86.2% on
LoCoMo under a shared pipeline, 2.8 points behind a system that spends
544 model calls at write time, with recall about 17× faster.

The negative result matters as much as the system. The bio-inspired
lifecycle mechanisms that earlier versions of this paper put at the
centre did not survive long-horizon ablation. Short-horizon judged
screens had also attributed effects to mechanisms whose packets never
changed. We recommend two practices to memory-system work: diff packets
before judging, and ablate on the engine that ships.

Future work:

1.  **A measured forgetting policy.** The store grows without bound. Any
    tiering or eviction must first be gated on packet recall over
    month-scale timestamps.
2.  **Out-of-pool gold.** At BEAM 1M and 10M, 42% and 67% of gold turns
    never enter the candidate pool. This, not reranking, is the next
    recall lever.
3.  **Reranker coverage.** The reranker needs an audio position map and
    re-encoding of stored codes when the reranker model changes.
4.  **Zero-shot evaluation.** The reranker should be evaluated on more
    benchmarks it was never trained on.
5.  **Standing-directive detection.** The current detector is a pattern
    check, and a learned one could replace it.

# References

Anderson, Michael C, Robert A Bjork, and Elizabeth L Bjork. 1994.
“Remembering Can Cause Forgetting: Retrieval Dynamics in Long-Term
Memory.” *Journal of Experimental Psychology: Learning, Memory, and
Cognition* 20 (5): 1063.

Åström, Karl Johan, and Richard M Murray. 2008. *Feedback Systems: An
Introduction for Scientists and Engineers*. Princeton university press.

Baddeley, Alan. 2000. “The Episodic Buffer: A New Component of Working
Memory?” *Trends in Cognitive Sciences* 4 (11): 417–23.

Chhikara, Prateek et al. 2025. “Mem0: Building Production-Ready AI
Agents with Scalable Long-Term Memory.” *arXiv Preprint
arXiv:2504.19413*.

Cormack, Gordon V, Charles L A Clarke, and Stefan Buettcher. 2009.
“Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank
Learning Methods.” In *Proceedings of the 32nd International ACM SIGIR
Conference on Research and Development in Information Retrieval*,
758–59.

Cowan, Nelson. 2001. “The Magical Number 4 in Short-Term Memory: A
Reconsideration of Mental Storage Capacity.” *Behavioral and Brain
Sciences* 24 (1): 87–114.

———. 2010. “The Magical Mystery Four: How Is Working Memory Capacity
Limited, and Why?” *Current Directions in Psychological Science* 19 (1):
51–57.

Fountas, Zafeirios et al. 2024. “EM-LLM: Episodic Memory in Large
Language Models.” *arXiv Preprint*.

Hunt, R Reed. 1995. “The Subtlety of Distinctiveness: What von Restorff
Really Did.” *Psychonomic Bulletin & Review* 2 (1): 105–12.

Khattab, Omar, and Matei Zaharia. 2020. “ColBERT: Efficient and
Effective Passage Search via Contextualized Late Interaction over BERT.”
In *Proceedings of the 43rd International ACM SIGIR Conference on
Research and Development in Information Retrieval*, 39–48.

LaBar, Kevin S, and Roberto Cabeza. 2006. “Cognitive Neuroscience of
Emotional Memory.” *Nature Reviews Neuroscience* 7 (1): 54–64.

Liu, Jun S, and Rong Chen. 1998. “Sequential Monte Carlo Methods for
Dynamic Systems.” *Journal of the American Statistical Association* 93
(443): 1032–44.

Maharana, Adyasha, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco
Barbieri, and Yuwei Fang. 2024. “Evaluating Very Long-Term
Conversational Memory of LLM Agents.” In *Proceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics*.

Malkov, Yu A, and Dmitry A Yashunin. 2018. “Efficient and Robust
Approximate Nearest Neighbor Search Using Hierarchical Navigable Small
World Graphs.” *IEEE Transactions on Pattern Analysis and Machine
Intelligence* 42 (4): 824–36.

McClelland, James L, Bruce L McNaughton, and Randall C O’Reilly. 1995.
“Why There Are Complementary Learning Systems in the Hippocampus and
Neocortex: Insights from the Successes and Failures of Connectionist
Models of Learning and Memory.” *Psychological Review* 102 (3): 419.

McGaugh, James L. 2004. “The Amygdala Modulates the Consolidation of
Memories of Emotionally Arousing Experiences.” *Annual Review of
Neuroscience* 27: 1–28.

Miller, George A. 1956. “The Magical Number Seven, Plus or Minus Two:
Some Limits on Our Capacity for Processing Information.” *Psychological
Review* 63 (2): 81.

Murdock Jr, Bennet B. 1962. “The Serial Position Effect of Free Recall.”
*Journal of Experimental Psychology* 64 (5): 482.

Nader, Karim. 2003. “Memory Traces Unbound.” *Trends in Neurosciences*
26 (2): 65–72.

Nader, Karim, Glenn E Schafe, and Joseph E Le Doux. 2000. “Fear Memories
Require Protein Synthesis in the Amygdala for Reconsolidation After
Retrieval.” *Nature* 406 (6797): 722–26.

Packer, Charles et al. 2023. “MemGPT: Towards LLMs as Operating
Systems.” *arXiv Preprint arXiv:2310.08560*.

Rasmussen, Preston et al. 2025. “Zep: A Temporal Knowledge Graph
Architecture for Agent Memory.” *arXiv Preprint arXiv:2501.13956*.

Robertson, Stephen, and Hugo Zaragoza. 2009. “The Probabilistic
Relevance Framework: BM25 and Beyond.” *Foundations and Trends in
Information Retrieval* 3 (4): 333–89.

Russell, James A. 1980. “A Circumplex Model of Affect.” *Journal of
Personality and Social Psychology* 39 (6): 1161.

Tulving, Endel. 1972. “Episodic and Semantic Memory.” In *Organization
of Memory*, 1:381–403. Academic Press.

Wu, Di, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong
Yu. 2024. “LongMemEval: Benchmarking Chat Assistants on Long-Term
Interactive Memory.” *arXiv Preprint arXiv:2410.10813*.

Zacks, Jeffrey M, and Kymberly M Swallow. 2007. “Event Segmentation.”
*Current Directions in Psychological Science* 16 (2): 80–84.

# Appendix A. State Variables Map

This appendix enumerates the state variables used by the specification
and separates retained state (carried across timesteps) from per-step
derived quantities. The canonical source for section references is
`docs/paper/sections/*.qmd`; anchors in these files are authoritative
when resolving cross-reference drift.

## State + Initialization (Canonical)

On cold start (no persisted state), initialize retained state as follows
(unless otherwise specified by knob priors):

-   **Global defaults:** `signals_processed = 0`, `u_uncertainty = 0`,
    `m_rate = 0`, `dt_ema = 0`, `recent_scores = []`,
    `last_rate_timestamp = now_ms()`, `last_retrieval_ts = 0`,
    `last_embedding = unset`, `x_pred_ema = unset`. (The oscillator
    phase `φ_t`, the mood vector, θ_dynamic, θ_target, hysteresis, the
    bias-corrected rate and the rate-reliability state were removed with
    their mechanisms; migrations 44, 57 and 58.)

-   **Accumulator defaults (per stream):** `μ_acc = 0_vector`,
    `c_t = 0_vector`, `drift_acc = 0`, `s_sum = 0`, `s_max = 0`,
    `n = 0`, `e_peak = 0_vector`, `emo_max = 0`, `arousal_sum = 0`,
    `acc_signals_window = []`, `t_start = 0`, `last_signal_ts = 0`,
    `eta_acc = 0`, `coherence_prev = 0`, `drift_accum = 0`,
    `drift_at_last_interrupt = 0`, `drift_acc_pacing = 0`,
    `x_last_check = unset` (μ_acc), `prev_x = unset` (μ_acc).

-   **Per-memory defaults (on insert):**
    `strength = memory_initial_strength(F,S,T)`, `drift_mag = 0`,
    `s_emotion_max = 0`, `s_arousal_avg = 0`, `boundary_score = 0`,
    `source_origin = "source"`. Lability, suppression and synaptic-tag
    fields were dropped by migrations 39, 45 and 46; the per-memory
    context and the source reliability, contradiction and last-verified
    fields by migration 56; `ambient_mood` by migration 57. Application
    provenance remains in the separate opaque `source_id`. SQLite
    defaults remain migration/backstop values for legacy rows. Evidence
    packets live in ordered `signals` rows, and the constructive-recall
    ledger starts empty until the initial reconstruction row is appended
    in `memory_reconstructions`.

-   **RLS defaults:** `w_* = w_bootstrap`, `P = diag(P_init(T))` with
    `P_init(T) = lerp(500, 2000, 1 − T)`, `blender_ready = false`,
    `blender_update_count = 0`.

-   **Knob-derived control parameters (initialized from priors):**
    `weight_relevance`, `attention_width`, `coverage_gain_floor`,
    `mismatch_weight`, `weight_novelty`, `weight_surprise`,
    `weight_valence`, `weight_arousal`, `emotion_gain`, `score_gain`.

-   **Buffers:** `signal_stream` (μ_acc stream), `score_stream`,
    `memory_stream`, `recent_memory_centroids` start empty.

-   **Retention history:** `retention_history` starts empty (populated
    after the first signal step).

-   **Accumulator state (per stream; retained across timesteps):**

    -   `{μ_acc, c_t, drift_acc, s_sum, s_max, n, e_peak, emo_max, arousal_sum, eta_acc, coherence_prev, acc_signals_window, t_start, last_signal_ts, drift_accum, drift_at_last_interrupt, drift_acc_pacing, x_last_check, prev_x}`

-   **Global state (retained across timesteps):**

    -   `{signals_processed, u_uncertainty, m_rate, dt_ema, recent_scores, last_rate_timestamp, last_retrieval_ts, last_embedding, x_pred_ema, weight_relevance, attention_width, coverage_gain_floor, mismatch_weight, weight_novelty, weight_surprise, weight_valence, weight_arousal, emotion_gain, score_gain, blender_ready, blender_update_count, blender_P}`

-   **Buffers (retained across timesteps; bounded by window rules):**

    -   `{signal_stream, score_stream, memory_stream, recent_memory_centroids, retention_history}`

-   **Recorded signal fields (per signal):**

    -   `{coherence_struct_t → SIGNALS.coherence, focus_spread_t → SIGNALS.focus_spread}`

-   **Recorded global fields:**

    -   `{u(t) → STATE.u_uncertainty}`

-   **Per-step derived scalars/vectors (ephemeral; recomputed each
    step):**

    -   `{signal_gap_s, coherence_curr, s_avg, S_window, boundary_score, max_signals, max_signal_flush, should_flush, write_memory, Δwrites, q_retrieval, ACh_t}`
        (NE_t and DA_t were removed with their modulators)

# Appendix B. Derived Signals: Definitions and Bounds

This appendix defines non-trivial derived signals used throughout the
specification. Unless otherwise stated, all derived scalars are clamped
to \[0, 1\].

**novelty:** A normalized dissimilarity-to-context signal computed on
the accumulator centroid.

    max_cos ← max_{c ∈ recent_context} cos(μ_acc, c)  # in [−1, 1]
    novelty_t ← clamp((1 − max_cos) / 2, 0, 1)

Fallback: if recent_context is empty, set novelty_t ← 1.

**μ_sim:** A normalized mean similarity to context.

    mean_cos ← mean_{c ∈ recent_context} cos(μ_acc, c)
    μ_sim ← clamp((mean_cos + 1) / 2, 0, 1)

Fallback: if recent_context is empty, set μ_sim ← 0.5.

**rarity_t:** A normalized rarity signal defined as
dissimilarity-to-context mean.

    rarity_t ← clamp(1 − μ_sim, 0, 1)

Fallback: inherits μ_sim fallback, so rarity_t defaults to 0.5 when
recent_context is empty.

**relevance_to_task(q, task_ctx):** A normalized relevance of embedding
q to a task context set.

    if |task_ctx| == 0: return 0.5
    return clamp((cos(q, mean(task_ctx)) + 1) / 2, 0, 1)

**novelty_to_set(q, S_set_embeddings):** A normalized novelty of q
relative to a set of embeddings.

    novelty_to_set(q, S_set_embeddings) ← 1 − redundancy(q, S_set_embeddings)

**ΔSSE:** A normalized improvement in reconstruction/prediction error
(utility proxy).

    ΔSSE ← clamp((SSE_prev − SSE_curr) / max(SSE_prev, ε), 0, 1)

Definitions:

    SSE_curr ← ‖μ_acc − x_pred_t‖^2          # prediction error at t (x_pred_t = x_pred_ema for EMA predictor)
    SSE_prev ← previous SSE_curr (t−1)     # or EWMA if smoothing is used

Fallback: if no prediction model is available, or if dimensions
mismatch, set ΔSSE ← 0.

**redundancy(a, S_set):** A normalized redundancy of item a w.r.t. a set
S_set.

    redundancy(a, S_set) ← max_{s ∈ S_set} clamp((cos(a, s) + 1) / 2, 0, 1)

Fallback: if S_set is empty, redundancy(a, S_set) ← 0.

**coverage_gain(candidate | included_set):** Incremental coverage
contribution of adding candidate.

    coverage_gain(candidate | included_set) ← 1 − redundancy(candidate, included_set)

# Appendix C. Edge Case Rules

These rules resolve empty-store and small-store cases to preserve
causality and avoid undefined metrics.

-   **memory_stream empty:** kNN returns empty; retrieval returns no
    candidates; graph traversal is skipped. Set `focus_spread_t = 0`.
-   **memory_stream small:** use
    `k_eff = min(k_neighbors(T), |memory_stream|)`. If `k_eff < 2`, set
    `focus_spread_t = 0`.
-   **graph absent or empty:** skip graph traversal; use vector search
    only.
-   **recent_context empty:** apply fallbacks defined in Appendix B
    (e.g., `novelty_t = 1`, `μ_sim = 0.5`, `rarity_t = 0.5`,
    `μ_ctx = 0_vector` so `map01(cos)=0.5`).
-   **recent_scores empty:** set `var_recent_norm ← 0` in uncertainty
    (<a href="#sec-math" class="quarto-xref">Section 4</a>).

# Appendix D. Main Loop Execution Order and Normative Invariants

This ordering is canonical and supersedes any other sequence described
elsewhere in the document. It mirrors the production
`BuildRootOperationSet` stage order: `CoreStage`, `StorageStage`,
`RetrievalStage`, then `FeedbackStage`. Probe mode runs only `CoreStage`
plus `RetrievalStage`.

Canonical single-step pseudocode (timestep t):

    # Main loop (single timestep t)
    now_s, now_ms, x_t ← read_inputs()

    # CoreStage
    initialize_embedded_centroids_and_priors()
    coherence_t ← compute_coherence()
    μ_acc ← update_accumulator_embedding(x_t)
    drift_accum ← update_cumulative_drift(x_t)
    focus_spread_t ← compute_focus_spread()
    prediction_error_t ← update_embedding_prediction_error()
    u_t ← update_uncertainty(...)
    update_focus_sensitivity_and_effective_focus()   # emotion projection
    metrics_t ← compute_structural_and_affective_metrics()
    update_neuromodulators_and_metric_weights()
    score_t ← compute_composite_score(metrics_t)
    update_accumulator_scores(score_t, μ_acc)
    update_signal_history(score_t)   # score window, dt_ema
    recent_context ← append_recent_context(μ_acc)
    boundary_score, should_flush ← detect_boundary(capacity_pressure, inactivity_boost)
    capacity_flush ← check_capacity_flush(n, max_signals, boundary_score)
    write_memory ← force_write OR should_flush   # every closed unit is stored
    Δwrites ← 1 if write_memory else 0

    # StorageStage
    stored_memory_id ← maybe_commit_memory_unit(write_memory)
    persist_signal_metrics(stored_memory_id)

    # RetrievalStage (q_retrieval is captured before any accumulator reset)
    q_retrieval ← l2_normalize(μ_acc)
    streaming_pacing_check()
    graph_retrieval(q_retrieval)
    update_rate_state(Δwrites)
    compute_mni_gate_decision()
    detect_memory_usage()   # usage counters only; reinforcement was removed

    # FeedbackStage
    record_memory_usage()
    update_working_memory()
    reset_accumulator_after_flush_or_interrupt()

The normative execution order for a single timestep t:

1.  **Read Inputs:** `x_t`, `now_s()`, `system_time_ms`.
2.  **Run CoreStage:** initialize priors, update accumulator/drift,
    compute uncertainty, adaptation, metrics, neuromodulators, composite
    score, signal history, boundary, capacity flush, and write gate.
3.  **Run StorageStage:** write the closed memory unit (every closed
    unit is stored), then persist signal metrics from the stored memory
    id.
4.  **Run RetrievalStage:** cache `q_retrieval ← μ_acc` before any
    reset, run streaming pacing and graph retrieval, update rate state
    from the current write decision, compute the MNI gate, and record
    memory usage.
5.  **Run FeedbackStage:** apply usage counters, working memory, and
    accumulator reset. A `force_consolidation` maintenance signal runs
    none of these stages; the processor only seals and recenters the
    sparse route.
6.  **Interrupt Abort:** If the MNI gate marked a pending abort, the
    next signal compares similarity to the selected memory against the
    current unit and resets only when the new signal aligns more with
    the selected memory.

Timing notes:

-   `now_s()` is captured at step 1 and reused for all Δt computations
    in this timestep.
-   The signal-history update in `CoreStage` uses the score computed
    earlier in the same stage.
-   Rate updates in `RetrievalStage` use Δwrites from the current step’s
    `write_memory` decision.
-   Accumulator resets happen in `FeedbackStage`, after storage,
    retrieval, MNI, memory-usage detection, and working-memory updates
    have consumed the current-unit query.

**Normative Invariants:** \* **Causality:** Step `k` uses only values
computed in steps `1` through `k` or retained from `t-1`. \* **Write
Atomicity:** A “write” is an atomic commit of a `should_flush` unit.
Partial units are never written. \* **Gap Consistency:** `now_s()` is
fixed at step 1. All `Δt` calculations use this fixed value.

# Appendix E. Optimization Ledger (Historical)

This appendix preserves, verbatim below the rule, the full optimization
ledger that was the manuscript’s performance section through engine v5.
It records micro-optimizations, `TRACE[...]` proof anchors and
experiments for every subsystem that existed at the time, including
subsystems the 2026-10 engine ablation later removed (retrieval-induced
inhibition, reconsolidation, emotional cascade, synaptic tagging,
clustering consolidation, graph expansion). It is a historical record,
not a description of the current engine.
<a href="#sec-optimization" class="quarto-xref">Section 17</a>
summarizes the surviving architecture. The ledger’s own first heading is
demoted to keep section identifiers unique; its body is otherwise
unchanged.

------------------------------------------------------------------------

## Performance Optimization (ledger as of engine v5)

This section documents the optimization state of the current hard-cut
Cortext runtime. The runtime no longer has internal decoder-backed
semantic operations, adapter dispatch, or mode-selected semantic replay,
so performance work now targets the embedding, graph, SQLite, and
hydration paths that remain in the production engine.

## Optimization Scope

The retained hot path is:

``` text
signal embedding
  -> core adaptive state
  -> memory storage
  -> graph retrieval
  -> constructive reconstruction when enabled
  -> feedback and working-memory maintenance
  -> sparse-route maintenance on explicit Consolidate()
```

The removed hot path was:

``` text
decoder semantic labeling
  -> adapter dispatch
  -> semantic fact and label materialization
  -> mode-selected semantic replay
```

Those removed costs are no longer runtime optimization targets. They
also no longer appear as dormant branches, fallback adapters, or hidden
configuration choices.

## Current Latency Discipline

The engine is optimized around three constraints:

-   **Bounded online outputs:** F/S/T-derived limits cap retrieval
    fanout, reconstruction history, consolidation breadth,
    working-memory size, graph updates, and written supersession edges.
    These caps do not by themselves bound exact candidate discovery as
    stored history grows.
-   **Current-state retrieval surfaces:** retrieval scores the current
    durable representative for each memory instead of scanning every
    historical embedding version.
-   **Hydration after selection:** payload reconstruction is performed
    after the ranked packet is selected, not while discovering
    candidates.

This keeps post-selection latency tied to selected memories and bounded
graph neighborhoods rather than source text length or hidden decoder
calls. Exact write-time and retrieval-seed discovery can still scale
with the durable embedding surface.

## Implemented Optimizations

The current branch includes the following implementation-level changes:

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr>
<th>Area</th>
<th>Optimization</th>
<th>Behavior contract</th>
</tr>
</thead>
<tbody>
<tr>
<td>SQLite execution</td>
<td>prepared-statement reuse in the store layer</td>
<td>same SQL semantics, lower repeated parse overhead</td>
</tr>
<tr>
<td>retrieval-induced inhibition</td>
<td>replace per-event recovery of every active row with a persistent
global log clock, indexed exact-threshold expiry, generation resets, and
a bounded connection-local SQLite active-epoch projection; publish the
projection only after the shared persistent commit and change its
active-row count only from the published membership delta; resolve a
legacy embedding-only loser to every memory sharing that embedding</td>
<td>effective strength, suppression, timestamp, <span
class="math inline">10<sup>−9</sup></span> freeze, restart, rollback,
caller SQL, eviction, and deterministic ranking remain exact;
publication failure rebuilds from persistent authority without replay;
retired generations do not consume the current active ceiling; Natural
and Durable use the same transaction and Durable adds only its
checkpoint barrier; no source-id or modality branch and no full-history
projection; ordinary structured competition remains memory-scoped, while
legacy embedding-only compatibility work may scale with shared-embedding
membership</td>
</tr>
<tr>
<td>vector ranking</td>
<td>bounded top-k selection and architecture SIMD kernels in the vector
path</td>
<td>same candidate contract; SIMD accepted after output checks</td>
</tr>
<tr>
<td>retrieval seed surface</td>
<td>current-memory embedding lookup; immutable normalized and
stored/Hadamard block summaries cached per private surface entry; exact
partial squared-distance rejection plus complementary sound 32-block
Cauchy–Schwarz bounds in the stored and Walsh–Hadamard bases before
exact cosine family checks; shared vectors retain all long-term metadata
alternatives; the private search cache follows the processor’s latest
reconstruction surface independently of persisted-current
materialization; cache loss uses the complete processor surface directly
or, only when that surface is incomplete, metadata-eligible,
distance-ordered, F/S/T-bounded SQL pages feeding one cross-page
cosine-family index that substitutes latest reconstructions before
accounting</td>
<td>latest reconstruction and eligible shared-embedding siblings remain
visible, byte-distinct near-duplicate boilerplate and stale base
families that converge after reconstruction cannot consume the
semantic-family candidate budget, sparse and dense orthogonal vector
families avoid broad exact cosine work, recovery does not return the
complete result at once, ephemeral recovery does not mutate the
registry, and source/modality metadata cannot influence rank; sqlite-vec
still scans linearly for every requested SQL page, the historical window
partition may internally sort and materialize the full eligible history,
and the knob-derived top-k bounds only rows returned to C++ and decoded
downstream rather than claiming sublinear search or bounded internal
sorter memory</td>
</tr>
<tr>
<td>graph expansion</td>
<td>removed in the 2026-10 engine ablation; with clustering
consolidation gone, only non-traversable <code>supersedes</code> edges
are written</td>
<td>retrieval ranks the family-collapsed seed set directly</td>
</tr>
<tr>
<td>supersession eligibility</td>
<td>removed in the 2026-10 engine ablation together with supersession
exclusion (<a href="#sec-experimental"
class="quarto-xref">Section 15</a>); the sidecar, its incremental
maintenance and the SQL <code>instr()</code> superseded-id filter are
gone</td>
<td>superseded targets compete as ordinary seeds;
<code>supersedes</code> edges still drive superseded-predecessor
hydration</td>
</tr>
<tr>
<td>reconstruction</td>
<td>knob-bounded retrieval-time reconstruction append</td>
<td>constructive recall remains non-decoder and can be disabled
cleanly</td>
</tr>
<tr>
<td>memory storage</td>
<td>avoid redundant object-store writes for single-signal memory
payloads</td>
<td>durable payload identity is preserved</td>
</tr>
<tr>
<td>supersession neighbor ranking</td>
<td>below the sparse cutover, or whenever the route is unpublished,
overflowing, or invalid, maintain private current/base embedding
surfaces and retain the exact historical-coverage proof and SQL
fallback; above the knob-derived cutover, reuse the SQLite HNSW
activation surface, inspect an <span
class="math inline">8<em>C</em></span> post-consolidation floor that
advances by <span
class="math inline"><em>R</em> = max (2, ⌊<em>B</em>/16⌋)</span> to a
<span class="math inline">9<em>C</em></span> ceiling, rerank at most
<span class="math inline"><em>A</em> = 2<em>C</em> + 2<em>B</em></span>
activated current centroids, then add exact candidates only from the
tracked historical/current population mismatches (memories whose
historical embedding differs from, or has no, current surface row)
instead of scanning the historical population; an ambiguous
memory-to-embedding lineage retains the SQLite fallback</td>
<td>the supersession candidate search is bounded by <span
class="math inline">9<em>C</em> + <em>A</em></span> plus the mismatch
count and no longer grows with the store (BEAM conv 7, 22.6k turns:
historical rows visited per turn 1.1k$<span
class="math inline">43<em>k</em><em>b</em><em>e</em><em>f</em><em>o</em><em>r</em><em>e</em>, 0<em>a</em><em>f</em><em>t</em><em>e</em><em>r</em>, <em>l</em><em>a</em><em>s</em><em>t</em> − <em>t</em><em>e</em><em>n</em><em>t</em><em>h</em><em>i</em><em>n</em><em>g</em><em>e</em><em>s</em><em>t</em>227</span><span
class="math inline">200<em>M</em><em>i</em><em>n</em><em>s</em><em>t</em><em>r</em><em>u</em><em>c</em><em>t</em><em>i</em><em>o</em><em>n</em><em>s</em>/<em>t</em><em>u</em><em>r</em><em>n</em>; 111.5<em>k</em> − <em>t</em><em>u</em><em>r</em><em>n</em><em>B</em><em>E</em><em>A</em><em>M</em><em>c</em><em>h</em><em>a</em><em>i</em><em>n</em> : 212<em>k</em></span><span
class="math inline">0<em>r</em><em>o</em><em>w</em><em>s</em>, <em>l</em><em>a</em><em>s</em><em>t</em> − <em>t</em><em>e</em><em>n</em><em>t</em><em>h</em>411</span>$277
M instructions/turn, total <span class="math inline">−22%</span>); edges
are now approximate above the cutover (<code>supersedes</code> Jaccard
against the exact scan 0.985 on conv 7 and 0.854 on the chain; packets
4/20 and 69/99 changed; gold-turn packet recall .088<span
class="math inline">=</span>.088 and .108<span
class="math inline">→</span>.106; Haiku sufficiency <span
class="math inline">−0.014 ± 0.027</span> and <span
class="math inline">−0.006 ± 0.019</span>, chain misleading <span
class="math inline">+0.030 ± 0.031</span>); the remaining ingest growth
is the saturated sparse-route search itself; timestamp, kind, duplicate
band, similarity threshold, edge limit, deterministic tie order,
rollback, source/modality independence, and public API shape remain
unchanged; write throughput and production-wide boundedness are
explicitly unclaimed</td>
</tr>
<tr>
<td>graph retrieval</td>
<td>score vector similarity first, restrict recency to a tie-break, and
reject an underfilled sparse seed proposal after operation-specific
eligibility filtering and again after exact semantic-family
collapse</td>
<td>recent unrelated rows and noisy graph neighbors cannot displace the
semantic seed surface; neither <code>ASSOCIATION</code> routing rows nor
duplicate activated family members can suppress the exact SQL
fallback</td>
</tr>
<tr>
<td>rollback snapshots</td>
<td>detach rebuildable durable caches, deep-copy the internal
execution-sidecar state only for full snapshots, move the private
accumulator map, and detach working-memory signal-record and
blob-reference vectors before copying remaining processor state; journal
active-source vector lengths for ordinary appends and lazily deep-back
up only the active source before accepted storage, flush reset, or
interrupt reset can mutate prior records; trim working-memory appends
and move an evicted slot’s record ownership into the journal; reserve a
full-map hook for explicit foreign-source accumulator-topology
mutation</td>
<td>non-boundary Natural ingestion does not recopy the growing pending
unit or source map; a one-source flush does not copy unrelated
open-source or working-memory history; failed writes and public no-store
queries restore durable and volatile state exactly; public state layout
and retention semantics are unchanged</td>
</tr>
<tr>
<td>emotional cascade metadata</td>
<td>removed in the 2026-10 engine ablation with emotional consolidation
and the cascade</td>
<td>—</td>
</tr>
<tr>
<td>synaptic tagging</td>
<td>removed in the 2026-10 engine ablation (its only consumer was the
consolidation score)</td>
<td>—</td>
</tr>
<tr>
<td>live replay ingress</td>
<td>internal replay path can skip per-event hydration when no probe
needs it</td>
<td>public processing APIs still hydrate normally</td>
</tr>
</tbody>
</table>

The important correctness point is that these are implementation changes
to the retained engine. They do not reintroduce the removed semantic
batch stack.

### Natural/Durable Write-Path Ownership

Natural retention was introduced after the earlier retrieval-latency
work, and the contemporaneous benchmark ingress was explicitly kept
Durable. The July 14 durable-ingestion optimization therefore
established the one-packet Durable profile, not the scaling of a long
open Natural unit. Source tracing shows that both retention values
already enter the same `FullRoot`; the competing cost was snapshot
ownership. Durable forces a boundary/write and resets the unit after
each call, while Natural can retain many embeddings and payloads. The
generic rollback copy detached durable search caches but still
deep-copied those pending records and completed working-memory record
lists on every packet.

The cutover retains one accumulator/write algorithm. Natural continues
to append and uses unchanged adaptive grouping. Durable only forces the
explicit flush/commit edge over the same pending unit, including its own
packet once. The private rollback journal makes ordinary Natural
snapshot work independent of pending-record count; an actual boundary
pays one exact active-unit backup/write cost. Deterministic 32-, 512-,
and 4,096-record accumulator plus matching working-memory regressions
verify unchanged payload/blob allocation identities and zero lazy record
backups on ordinary rollback. A failing-flush regression verifies one
three-record active-source backup restores cleared records plus mutated
payload/blob state while 4,096 unrelated-source and working-memory
records remain allocation-identical and add zero to the copy counter. A
working-memory eviction regression moves one erased slot into the
journal and restores it beside 4,096 untouched records with zero record
copies. Controlled failures after active-record detachment,
accumulator-map transfer, and working-memory detachment verify
exception-safe restoration. A direct 4,096-record working-memory append
regression verifies length trimming and payload allocation identity
without record copies. The journal is selected by a private marker on
the production operation root; arbitrary caller-supplied operation roots
retain complete automatic context snapshots, proven by hookless
destructive and nested different-source failure regressions. A public
integration regression verifies two Natural packets followed by one
Durable packet produce one three-signal memory and three ordered signal
rows. An explicitly marked escalation regression verifies that an exact
active-source backup remains authoritative when a later foreign-topology
mutation requests a full-map backup and the combined operation fails.
These are source-health and work-count proofs. They do not reproduce the
full 15,500-packet observation and do not turn the exact durable
candidate scan into an asymptotically flat store search.

The later active-epoch cutover targets the remaining
store-size-dependent retrieval-inhibition recovery directly. A complete
15,695-packet Natural replay made before the active-epoch ceilings
became knob-derived kept the active epoch within its then-fixed midpoint
harness ceilings: the observed high-water marks were 512 events, 1,859
distinct mutations, and 180,224 allocated bytes, with no over-limit
event. That evidence is retained as midpoint-only historical proof and
is not accepted as proof of the current F/S/T-derived contract. The
caller honored 127 consolidations. Across the audit’s first and last
five eligible windows, `Competition.rif_recovery_active_sql` changed
from 0.008629 to 0.009038 ms, a 1.047 ratio. The whole write path did
not plateau: mean process time rose from 2.265 to 5.507 ms and
throughput fell from 36.27 to 9.14 packets/s. This run therefore
verifies the lazy-RIF hotspot and the historical midpoint epoch
ceilings, while leaving graph retrieval, memory storage, and emotional
propagation as measured contributors. It is not a long-horizon or
production-wide boundedness claim.

The subsequent implementation change derives the active-epoch event
ceiling from *C*(*F*, *S*, *T*), the mutation ceiling from 64*C*, the
allocation ceiling from 131, 072*C* bytes, and every RIF
persistent/shadow statement batch from the same *B*(*F*, *S*, *T*) used
by sparse-route backfill. A compiled structural regression covers all 27
points in the {0, 0.5, 1}<sup>3</sup> grid and separately evaluates the
nine midpoint, all-low, all-high, and one-axis-low/high points used by
the production-shaped route ablation. It proves the formulas resolve to
256/16,384/32 MiB/64 rows, 512/32,768/64 MiB/128 rows, and 768/49,152/96
MiB/192 rows at the all-low, midpoint, and all-high corners. A separate
nine-point execution regression publishes *B* + 1 live active rows at
every named point, observes a maximum statement size of exactly the
resolved *B*, retains all *B* + 1 rows through a second statement, and
varies text, audio, image, shared, and opaque source metadata without
changing the schedule. The 129th midpoint live row is therefore
processed in a second statement; it must not be confused with the
historical backfill’s logical 129th completion probe, which is not read
or processed. The benchmark now records resolved values instead of
emitting fixed midpoint labels. Configured low, midpoint, and high
Natural and Durable regressions each reached their own resolved event
boundary, preserved `Required` at the logical limit-plus-one
observation, and reset after a successful consolidation; the same runs
varied opaque source and text, audio, and image labels without changing
the formula. One-event current-binary profile probes independently
emitted 256/16,384/32 MiB for all-low Natural and 768/49,152/96 MiB for
all-high Durable, with the recorded event limit equal to the
independently recorded route capacity. Non-finite knobs are rejected
before any capacity is converted to an integer, and the profile auditor
applies the same fail-closed rule. Long-horizon nine-point latency and
reset evidence is now complete over the full 15,695-packet corpus at
midpoint, both joint endpoints, and each one-axis endpoint. Every run
emitted its independently expected F/S/T-derived event, mutation, and
allocation limits; every run reached exactly its event limit without
exceeding any ceiling; and every run observed a successful reset. None
of the nine runs passed the whole-engine plateau or consolidation-cycle
contract, however.

<table>
<colgroup>
<col style="width: 13%" />
<col style="width: 18%" />
<col style="width: 18%" />
<col style="width: 18%" />
<col style="width: 18%" />
<col style="width: 13%" />
</colgroup>
<thead>
<tr>
<th>F/S/T point</th>
<th style="text-align: right;"><span
class="math inline"><em>C</em></span></th>
<th style="text-align: right;">Wall time (ms)</th>
<th style="text-align: right;">Mean process (ms)</th>
<th style="text-align: right;">Final/first process</th>
<th>Plateau</th>
</tr>
</thead>
<tbody>
<tr>
<td>midpoint</td>
<td style="text-align: right;">512</td>
<td style="text-align: right;">111,526</td>
<td style="text-align: right;">2.430</td>
<td style="text-align: right;">1.716</td>
<td>fail</td>
</tr>
<tr>
<td>all-low</td>
<td style="text-align: right;">256</td>
<td style="text-align: right;">119,572</td>
<td style="text-align: right;">4.020</td>
<td style="text-align: right;">1.963</td>
<td>fail</td>
</tr>
<tr>
<td>all-high</td>
<td style="text-align: right;">768</td>
<td style="text-align: right;">809,691</td>
<td style="text-align: right;">45.566</td>
<td style="text-align: right;">5.517</td>
<td>fail</td>
</tr>
<tr>
<td>focus-low</td>
<td style="text-align: right;">384</td>
<td style="text-align: right;">89,197</td>
<td style="text-align: right;">2.335</td>
<td style="text-align: right;">1.448</td>
<td>fail</td>
</tr>
<tr>
<td>focus-high</td>
<td style="text-align: right;">640</td>
<td style="text-align: right;">246,936</td>
<td style="text-align: right;">4.802</td>
<td style="text-align: right;">1.838</td>
<td>fail</td>
</tr>
<tr>
<td>sensitivity-low</td>
<td style="text-align: right;">448</td>
<td style="text-align: right;">74,894</td>
<td style="text-align: right;">1.565</td>
<td style="text-align: right;">1.554</td>
<td>fail</td>
</tr>
<tr>
<td>sensitivity-high</td>
<td style="text-align: right;">576</td>
<td style="text-align: right;">1,866,911</td>
<td style="text-align: right;">110.989</td>
<td style="text-align: right;">6.841</td>
<td>fail</td>
</tr>
<tr>
<td>stability-low</td>
<td style="text-align: right;">448</td>
<td style="text-align: right;">149,161</td>
<td style="text-align: right;">4.136</td>
<td style="text-align: right;">1.762</td>
<td>fail</td>
</tr>
<tr>
<td>stability-high</td>
<td style="text-align: right;">576</td>
<td style="text-align: right;">105,381</td>
<td style="text-align: right;">1.887</td>
<td style="text-align: right;">1.760</td>
<td>fail</td>
</tr>
</tbody>
</table>

Sensitivity was the dominant adverse axis: the high-sensitivity wall
time was 24.93 times the low-sensitivity wall time, and
`PropagateEmotionalCascade` accounted for 98.78 percent of its
early-to-late process growth. The all-high run attributed 95.10 percent
of growth to the same operation. At midpoint and lower sensitivity,
graph retrieval and memory storage remained the dominant contributors.
The content-addressed aggregate is
`active-epoch-knob-long-horizon-ablation-v1.json`. These text-ingress
runs prove the knob formulas, hard epoch ceilings, and reset mechanics
across the selected grid, but reject latency boundedness within every
envelope; modality and source agnosticism remain separately covered by
the fixed-embedding active-route regression rather than inferred from
this corpus.

The next retained active-epoch candidate removed redundant values from
the connection-local database rather than changing the persistent
recurrence. Profiling a copied mature 1,000-event slice first attributed
0.351847 ms of a 0.385052 ms publication to loading complete persistent
RIF rows. The retained identity ledger loads and publishes only active
`memory_id` values in the same knob-derived *B*-row statements; the
recovery clock, anchored suppression, recovery total, log factors,
expiry, and effective view remain authoritative only in persistent
SQLite. Strength-only refreshes do not publish when membership cannot
change. Natural and Durable still share the same write and publication
path, with Durable adding only its checkpoint.

Against the same copied database and eight fixed consolidations,
publication fell from 0.384426 to 0.103604 ms, a 73.05 percent reduction
that passes the preregistered 20 percent local gate. Mean process time
changed from 3.282725 to 3.243371 ms, total latency from 14.891846 to
15.254791 ms (+2.44 percent), and wall time from 16,096 to 16,368 ms
(+1.69 percent). All 1,000 public behavior rows, the canonical logical
database, and the raw SQLite database were exact; the observed midpoint
statement high-water was 128. A separate fresh 1,000-event screen
recorded 0.009047 ms publication and 0.684362 ms process, but used four
live-hint consolidations and is therefore only a non-regression screen,
not fixed-index equivalence. Removing the membership preflight and using
`RETURNING` was rejected after publication regressed to 0.335491 ms.
These copied-slice source-health results did not decide the full
15,695-event Natural plateau, Durable horizon and checkpoint proof, or
bounded restart. `TRACE[state:rif_active_identity_ledger]`.

The subsequent default-knob Natural replay resolved *B* = 125, processed
all 15,695 packets with an observed statement high-water of exactly 125,
and ran 31 consolidations in 82,994 ms. Mean process time was 1.944934
ms, mean total latency was 5.107905 ms, and active-epoch publication
averaged 0.019719 ms. This closes the full-Natural batch-bound check but
not the engine plateau: over the final six eligible windows process and
throughput ratios were 1.0072 and 0.9555, while transaction commit was
1.1129 and the exact graph-comparison work counter was 1.1304. The
identity ledger is therefore retained as a local repair and the next
hotspot is reranked; Durable and bounded-restart proof stay open.

The real 2,016-message Durable replay exercised that identical
default-knob path with four consolidations, the same derived *B* = 125
and the same observed 125-row high-water. Mean process and total latency
were 5.883704 and 11.471774 ms, wall time was 23,845 ms, and the added
checkpoint barrier averaged only 0.000158 ms. This supports the
one-path-plus-flush architecture, but the corpus contains only three
post-warmup 500-event windows against the required ten; it cannot
establish a Durable plateau, and same-index copied-store equivalence
remains separate proof.

That result nominates a bounded emotional-propagation experiment rather
than a larger retrieval envelope. The proposed `bounded-priority-prefix`
reuses the current-event retrieval capacity *C*, backfill batch *B*, and
activation target *A* = 2*C* + 2*B*. The candidate keeps a 5*C*-capacity
index of rows already satisfying the flashbulb, intensity, arousal, and
recency predicates, ordered by descending creation time and memory
identifier. It inspects that eligible prefix and uses at most 5*C**B*
comparisons or moves to select *B* sources in descending intensity and
ascending memory-identifier order. It may emit at most *B* source
memories and embeddings, visit 5*C* physical association edges, retain
5*C* + *B* frontier/visited and topology-footprint entries, consider *A*
activated identities, and enqueue *A* update statements. Source-index
maintenance is bounded independently by
(5*C*)<sup>2</sup> = 25*C*<sup>2</sup> comparisons or moves per event.

A statement count alone is not a fixed-work guarantee because an
embedding can be shared by arbitrarily many memory rows. The candidate
therefore admits only embedding identities with at most *B* members and
gives current-value member reads, SQL-affected rows, and cache-row
mutations separate 5*C* total budgets. An identity whose complete member
set does not fit every remaining budget is skipped. Member-vector size
detects the logical *B* + 1 overflow boundary in constant time; no
plus-one row is fetched or mutated. At *F* = *S* = *T* = 0.5, *C* = 512,
*B* = 128, and *A* = 1, 280: every 5*C* surface is 2,560, every *B*
source surface is 128, every *A* surface is 1,280, every 5*C* + *B*
surface is 2,688, source-index maintenance is 6,553,600, priority
selection is 327,680, and the logical overflow boundary is 129. All-low
resolves these families to 1,280/64/640/1,344/1,638,400/81,920/65;
all-high resolves them to 3,840/192/1,920/4,032/14,745,600/737,280/193.
Thus 128 and 129 are consequences of neutral F/S/T, not hidden
constants. The proposal has no durable backlog or second write path:
SQLite remains authoritative and the in-memory metadata/fanout view
remains rebuildable and rollback-equivalent. Neither modality nor source
identifier is a predicate or budget input.

This is a preregistered candidate, not a result or cutover. A
deterministic dense-graph regression must first fail on the current
unbounded traversal. The candidate must then prove all formulas over the
27-point structural knob grid, including source-index, selection,
emission, member-row, and cache work. Exact parity is required when the
unbounded event stays within every resolved source, edge, frontier,
identity, member, affected-row, and mutation limit. Above the envelope,
candidate and control run once on separate byte-identical pre-event
stores at identical consolidation indices. If *U*<sub>*c*</sub> and
*U*<sub>*b*</sub> are the control and bounded sets whose final minimum
intensity or bonus increased, the candidate requires
|*U*<sub>*b*</sub> \\ *U*<sub>*c*</sub>| = 0 and recall
|*U*<sub>*b*</sub> ∩ *U*<sub>*c*</sub>|/|*U*<sub>*c*</sub>| ≥ 0.95;
empty control and bounded sets pass, while an empty control with a
nonempty bounded set fails. For each metric, retained delta is the ratio
of the sum of positive bounded changes to the sum of positive control
changes over *U*<sub>*c*</sub> ∪ *U*<sub>*b*</sub>, must be at least
0.99, and passes a zero control denominator only when the bounded
numerator is also zero. Non-finite values and bounded per-identity
deltas exceeding control by more than 10<sup>−12</sup> fail. Downstream
gates allow at most one exact top-1 miss per 512-query run, require
identity recall at 16 of at least 0.998 and semantic coverage of at
least 0.95, and reject clustered top-1 misses.

Per-event cascade timing includes zero-work events. Early and late means
use the first and last 3,000 events; positive-over-zero and non-finite
ratios fail. The p99 is nearest-rank ⌈0.99*n*⌉ over all events and must
be at most one quarter of the same-knob, same-retention control, with
zero control passing only zero candidate. Sensitivity-high and all-high
must also hold late/early mean to 1.10. The preregistered nine tuples
are midpoint, both joint endpoints, and the low/high endpoint of each
one-axis change while the other knobs remain 0.5. Every tuple runs fresh
matched candidate/control pairs for both the exact 15,695-packet Natural
corpus and 2,016-message Durable corpus, with one selected algorithm per
event, control-derived fixed consolidation indices, and exact
source-tree, build, binary, corpus, database, profile, audit, raw, and
sanitized digests. During that experiment the private selector accepted
only `control` or `bounded`, defaulted to `control`, and rejected any
other value before mutation. The candidate could have cut over only by
removing that selector and the unbounded implementation, leaving one
Natural/Durable path with Durable adding only its checkpoint. The
midpoint rejection instead removed the candidate; the measured envelope
and disposition are retained here.

The rejected implementation packet exercised that preregistered shape
behind a private selector without claiming cutover. The red dense-graph
regression visited 2,561 edges against the neutral 5*C* = 2, 560 ceiling
on the prior path. The experimental implementation kept the eligible
recency prefix at 5*C*, performed deterministic bounded top-*B*
selection, traverses sources sequentially with first-source-wins
identity ownership, skips shared embeddings at the logical *B* + 1
boundary, and updates accepted member rows with one row-addressed
`UPDATE ... RETURNING` statement. `RETURNING` is required for exact
accounting through caller-supplied transaction wrappers: a later
compatibility statement would otherwise overwrite SQLite’s
connection-local `changes()` value. The same statement was enclosed by a
narrow managed-mutation scope so the caller-SQL compatibility layer
could distinguish the cascade’s cache-maintained write from arbitrary
caller SQL. Arbitrary same-event caller SQL invalidated every
database-derived surface and the bounded cascade failed closed; the next
event rebuilt from committed SQLite before traversal.

One hooks-on Release binary then passed 807 assertions in 13 fixed-work
tests and 907 assertions in 32 complete emotional-cascade tests. The
fixed-work set includes every one of the 27 structural F/S/T points,
exact *B* mutation, logical *B* + 1 overflow, source prefix and priority
work, dense edge saturation, below-envelope candidate/control parity,
invalid and mid-event selector behavior, caller-SQL invalidation and
rebuild, engine-owned rollback, process restart, and an explicit
consolidation event. A separate `FullRoot` regression passed 5,552
assertions while varying two then four opaque source identifiers,
text/audio/image labels, and both Natural and Durable retention; every
cascade counter remained within the same F/S/T-derived ceiling, and
Durable alone reported the existing post-commit checkpoint. Those
focused results established the structural and lifecycle seams only. The
full-corpus midpoint below then failed the preregistered timing and
identity-quality gates, so the selector, managed-mutation seam, and
candidate-specific counters were removed. The structural results remain
historical experiment evidence, not production source health.
`TRACE[state:emotional_cascade_fixed_work_candidate]`
`TRACE[rejected:EmotionalCascade.knob-derived-bounded-priority-prefix]`

The fixed-time midpoint screen rejected this candidate before the
nine-point matrix. A same-binary Natural control/candidate pair used
timestamps 1, 700, 000, 000, 000 + 1000*i* and the control’s 118
successful consolidation indices for both fresh stores. Every candidate
hard-work counter remained within its neutral-knob limit: source
candidates and executions reached 37, physical edges reached
5*C* = 2, 560, frontier and topology entries reached 1,888 and 476,
activated identities reached *A* = 1, 280, and member reads,
SQL-affected rows, and cache mutations each reached 133. Nevertheless,
candidate p99 was 0.397625 ms versus 0.693875 ms for control, a 0.573
ratio against the preregistered 0.25 maximum. Its late/early mean ratio
was 2.324. The control and candidate profiles are content-addressed by
`8ef80929db664afd8ac87c256033d7ae3c9ac88036edaddf5f270940c79e52ba` and
`dd0cdd0801b7656dd8835528977b8d3e925b15d2b8dca398c14bb7d819eacd55`; the
fixed schedule digest is
`901fac42a38125e525d6ca44b63312248b36b30b35d8eb3156ff5d4dc7fbcd8f`.

Three implementation-only iterations retained every formula and semantic
boundary. First, traversal scratch was reserved from 5*C* + *B* and
reused across sources, while a fixed indexed `embedding_id` update
replaced variable-length SQL construction and required the exact
returned memory identifiers to match the cache before mutation. This
produced p99 0.395666 ms. Second, the candidate persisted the
already-budgeted 5*C* + *B* expandable topology footprint and reused a
fixed point only when a new edge was provably outside that footprint;
active traversals fell from 6,710 to 1,369 and mean cascade time fell
from 0.0644 to 0.0260 ms, but p99 rose to 0.475541 ms because the
nearest-rank percentile now landed inside the remaining full traversals.
Third, a generation-stamped visited map allocated each event’s unique
memory identities once instead of reallocating hash nodes per source.
Mean cascade time fell again to 0.023903 ms and p99 to 0.432000 ms,
while traversal alone remained 0.223292 ms at p99. These tuning profiles
have digests
`d06e1042784afbe760ff2e2917541bf07628adc7d9055561372fa513aa6274b8`,
`ca4039abeaf873acb500b9279dfb1dacd4fb99f57c24c25751c5d7f1edd36bef`, and
`5ee53d12f0fc98c8e746553500be2f12802a62a5d38de681d84672580d55d916`. They
are tuning diagnostics rather than admissible performance pairs because
their comparison control predates the corresponding binary changes.

The final tuning profile independently failed the downstream
public-retrieval gate, so an additional exact-binary timing pair could
not rescue the candidate. Over 512 deterministic controls and four
opaque source identifiers, exact top-1 was 0.998047, identity recall at
16 was 0.980760, semantic coverage was 0.975084, and tie order remained
deterministic. The content-addressed audit digest is
`da402733380f8cf7f2eda5289a4f95f1a35b55fcd27ede4695499d8acd6b4973`.
Exact top-1 1.0 and recall 0.998 were mandatory, so
`bounded-priority-prefix` is terminally rejected without running the
remaining knob/corpus matrix or claiming direct bounded/control delta
fidelity. The structural formulas remain useful design evidence: neutral
128 and logical 129 were always derived from *B*(*F*, *S*, *T*), no
plus-one row was read or mutated, and the latest focused tests passed
824 assertions in 15 fixed-work cases, 924 in 34 emotional-cascade
cases, and 5,552 in the mixed source/modality FullRoot case. Those
source-health results do not override the measured rejection.

The observation-only bounded activation graph was then implemented
behind a default-off private flag and evaluated directly in compiled C++
over 35,496 256-dimensional embeddings with 512 age-stratified queries.
At F=S=T=0.5 it held 20 roots, 640 leaves, 24 diverse representatives
per leaf, and eight neighbors per leaf. Normal update work reached 163
comparisons against a fixed 196 ceiling; internal consolidation reached
28,462 against 29,360; and recall reached 996 against 1,196. Mean exact
top-16 identifier recall was 0.0562, duplicate-equivalent recall 0.1190,
best cosine 0.9445, and exact-neighbor semantic coverage 0.9545. These
results closely reproduce the separate prototype observation while
measuring the code that is actually wired to the processor lifecycle.
Deterministic common-pipeline tests produce identical candidate order,
state digest, and work for two versus four sources, text/audio/image
labels, and Natural versus Durable retention. The graph still loses
exact identifier recall and requires an O(N) restart rebuild, so it
remains diagnostic evidence rather than a retrieval cutover.

After the nested-transaction lifecycle repair, the fixed allocation was
19,180,880 bytes at F=S=T=0.5, including a journal sized by the existing
consolidation interval. An immediate RAII scope owns each prepared
journal entry, so exceptions during object-store or rollback-snapshot
setup clear the root journal even before the main processing guard is
active. A repaired 15,695-packet Natural replay published all 15,695
events without a bound or generation failure and preserved the current
flag-off behavior and logical-database digests. Its mean process cost
was 2.1077 ms versus 1.9115 ms in the paired flag-off artifact, a 10.26%
increase; end-to-end mean and wall time were 0.65% lower within run
noise. The repaired 2,016-message Durable replay likewise preserved
current flag-off behavior and database digests, with mean process cost
up 6.16% and end-to-end mean up 5.49%. These are observation costs, not
evidence that the present graph satisfies the retrieval-shape objective.

For subsequent candidates, the optimization objective ranks a bounded,
repeatable retrieval envelope ahead of maximum write throughput. A
modest measured write regression is acceptable when consolidation
produces stable peaks and troughs, a material reset, and a more
cycle-symmetric retrieval sawtooth. Write cost remains a reported
tradeoff, and this preference does not substitute for replay evidence or
authorize a semantic cutover.

The cutover audit makes that objective decidable. It measures retrieval
work as candidate plus exact-comparison counts rather than wall time,
divides the Natural replay at successful public-consolidation
boundaries, resamples each complete material cycle to 50 equal-progress
bins, and normalizes each cycle by the difference between its
first-50-event post-consolidation mean and its last-50-event
pre-consolidation mean. A cycle whose latter mean is lower is an inverse
ramp and fails; a zero range is represented by an all-zero profile and
also fails because every accepted cycle must exhibit the same material
positive ramp required of the process-time envelope. Retrieval-work
peaks and following troughs are independently subject to the same
non-rising half, Theil–Sen, and bootstrap trend bounds as the process
envelope, preventing geometrically declining flat cycles from
masquerading as consolidation resets. Each cycle also limits accumulated
negative variation to 0.25 of its anchored ramp and its maximum event
work to 1.25 times the trailing mean, while every event must stay within
the normal-comparison ceiling derived from F/S/T. These independent
shape and amplitude constraints prevent a candidate-derived median
template from accepting repeatable spikes. At least ten material cycles
are required; at least 80% must reset to no more than 90% of their
trailing pre-consolidation work; the nearest-rank 95th-percentile
cycle-to-median-template mean absolute error must be at most 0.20; and
the final-five versus prior-five template error must be at most 0.10.
Natural cutover cannot use flat-envelope mode to bypass these cycle
gates; the already-flat Durable control remains eligible for its
separate flat-envelope contract. The separate 512-query, top-16 corpus
gate requires mean exact-identifier recall of at least 0.998, at most
one exact top-1 miss, and mean exact-neighbor semantic coverage of at
least 0.95. Across the prescribed nine-run matrix, aggregate top-1 must
remain at least 0.999 and two misses may not share a modality, opaque
source, memory-age quartile, or exact knob point. The candidate’s
complete ordered top-16 lists must repeat byte-for-byte under the same
seeds, while all seven fixed identity-and-rank probes must still match
the exact control. This permits only the explicitly approved bounded
approximation; it does not permit clustered top-1 drift,
nondeterministic ties, or semantic substitution below the exact-recall
floor. The summary is recomputed from all 512 per-query records rather
than trusted as a label. The candidate must bind to a separate control
artifact whose exact content digest was approved before the audit;
deterministic query indices, embedding identities, exact ranked
controls, provenance, and the seven fixed probe positions are bound
across those artifacts. Candidate ranks are separately content-addressed
across two complete runs rather than being required to equal the exact
control outside the approved lower-rank allowance. The approved control
must identify the current public retrieval path, and the candidate must
also preserve the independently recorded digest of the pre-existing
seven public identity-and-rank probes; the pre- and post-restart probe
digests at every retained-history fraction must equal each other, and
the full-history fraction must additionally equal the candidate/control
fixed-probe digest. Each smaller fraction uses probes drawn only from
its retained prefix, so the gate tests restart equality without
referring to absent history; the benchmark’s exhaustive
embedding-neighbor mode can propose a control artifact but cannot label
itself approved. Hashed source provenance, modality, and history
position come from the queried embedding’s authoritative SQLite signal
rows; together they must cover early, middle, and late history, text,
audio, image, and at least four opaque sources. Restart acceptance
likewise derives the fixed row-visit ceiling from F/S/T and checks
measured retained-row and visited-row series from a production-shaped
persistent restart rather than trusting pass/fail labels. The three
retained counts are fixed at 25%, 50%, and 100% of the bound corpus,
visited-row counts must be nonzero and within the F/S/T capacity, and a
ranked probe digest before restart must match the restored probe with at
least top-16 candidate state. The current synthetic full-history
fallback measurement is explicitly ineligible. These are
candidate-selection thresholds, not evidence that the current
observation-only graph passes them.

The preliminary embedding-router feasibility selected a two-graph
packed-route hybrid evaluated over 34,456 normalized, 256-dimensional
embeddings. Each sealed embedding retained 512 route entries; each new
active-epoch embedding proposed at most 64 reciprocal updates; each
graph used a 16,384-comparison construction ceiling and a
1,280-comparison query ceiling. Across 512 deterministic top-16 queries,
exact top-1 was 1.0, mean exact-identifier recall was 0.998413, and mean
exact-neighbor semantic coverage was 0.999961. Thirteen of 8,192
exact-neighbor slots were absent, spread across ten queries, while all
seven fixed identity-and-rank probes remained exact. The complete
candidate rank stream repeated with the same content digest in two runs.
Mixed-input evidence included four opaque sources and text, audio, and
image labels after encoding; the router consumed only embedding values
and embedding identity.

The selected benchmark snapshot restarted from 1,024 anchor rows and
four packed SQLite blobs. At 25%, 50%, and 100% retained history, all
fixed prefix-local probe results were identical before and after
restart; the maximum restart visit count was 1,029 rows while the full
snapshot retained 34,456 embeddings. Active-epoch SQLite allocation
peaked at 708,608 bytes and reciprocal mutations peaked at 11,967, below
the historical midpoint 32,768 ceiling. Ten modeled material cycles had
zero normalized cycle-template error and zero late-template error, with
post-consolidation retrieval work 0.888889 of the preceding level. This
establishes a bounded and repeatable embedding-router direction, not the
production-shaped quality gate, live cycle, or latency proof: the cycle
result is derived from the fixed-work benchmark model and must still be
reproduced at public consolidation boundaries in the shared engine path.

Several variants were rejected before selecting that point. A
reciprocal-route only design reached 0.979858 mean exact recall; adding
a third graph did not improve recall at equal per-graph work; reducing
construction to 4,096 comparisons per graph reached 0.995972; and an
8,192-comparison construction with 64 reciprocal proposals reached
0.997803. All retained exact top-1, but all fell below the approved
0.998 lower-rank recall floor. The chosen point therefore spends
additional bounded construction work to remove all but 13 lower-rank
misses on that feasibility population, rather than weakening the
top-result or semantic gates.

An even simpler anchor-only route was also rejected on the corrected
3,172 current-memory surface. With four selected routes per query, 512
anchors reached mean exact recall 0.98425 and exact top-1 0.97656; 1,024
anchors reached 0.99194 and 0.98633 respectively. Adding exact
reciprocal node-route hops reached only approximately 0.982 recall.
These were private one-off randomized held-out feasibility probes and
their raw output was not retained, so they are recorded as negative
design evidence rather than reproducible acceptance proof. The result
rules out using anchors as the retrieval candidate set by themselves;
anchors remain useful only as bounded route entry points into the
independently constructed sparse graphs.
`TRACE[state:anchor_only_route_feasibility]`.

A control audit then found that the 34,456 identities came from the
complete `embeddings` table. The public seed path ranks eligible
long-term/current memory surfaces, not signal-only embedding ids, and
additionally applies timestamp, supersession, current-surface, and
family filters. Labeling the exhaustive embedding-neighbor artifact
`current-public-retrieval` was therefore invalid. The corresponding
production-shaped pass is revoked; the result remains only
router-feasibility evidence.

The corrected current long-term surface contains 3,172 memory
identities, a 10.86-fold smaller and semantically different population.
Over the same 512 deterministic top-16 queries, the two-graph route
reached exact top-1 1.0, exact recall-at-16 1.0, and semantic coverage
1.0 with no missing query. Maximum construction work was 6,344
comparisons. Ten 128-memory-node incremental feasibility epochs used at
most 184,320 active SQLite bytes; mean graph insert, route construction,
and reciprocal-update times were 0.073002, 0.044416, and 0.013539
seconds per epoch. At 25%, 50%, and 100% retained surfaces, restart
visited 798, 1,029, and 1,029 rows and preserved all prefix-local
probes. These results retain the sparse-route design but do not close
the public control: the next quality artifact must compare against the
runtime’s exact eligibility, supersession, current-surface, and family
behavior. The active sawtooth remains signal-event based, so the
128-memory-node epochs also cannot substitute for ten live 512-signal
consolidation cycles. This correction is
`TRACE[state:retrieval_quality_control_correction]`.

To close that gap without another approximate population model, the
benchmark now captures the actual public retrieval ranks during Natural
ingestion. Queries are sampled evenly from retrieval-active events, not
from every event: the latter includes legitimate no-retrieval decisions
and would make a routing quality denominator vacuous. A 100-event smoke
contained 27 retrieval-active events; an eight-query sample covered
events 6 through 96 and retained nonempty direct, graph-expanded, and
hydrated rank lists for every query. In a separate paired 200-event run,
enabling the capture preserved behavior SHA-256
`40833b57d441b8d7e704787768e8157bacdd8a2525616af8893bbf2d164a87bb` and
logical-database SHA-256
`e83dabd67109258d9609d82c26ed01c227045c6ae1f7786ef4b15bb296824ff8`
exactly. The complete 15,695-packet run then sampled 512 controls from
5,390 retrieval-active events, spanning events 6 through 15,691. Direct
controls contained 1–16 identities and the hydrated public result
contained 1–12; the fail-closed audit accepted the control population.
The corpus itself is one opaque source and text-only, so it cannot
satisfy the independent four-source text/audio/image implementation
gate. `TRACE[state:public_retrieval_control_capture_experiment]` is
exact public-control evidence, not candidate acceptance.

Three standalone HNSW variants then attempted to reconstruct
point-in-time candidate routes from the final database. All used two
graphs with *M* = 64, construction effort 200, query effort 1,280, and a
route cap of 512. Indexing every retained surface version produced
apparent identity coverage 0.983770, top-1 coverage 0.914063, and
73.8887 current candidates on average. Replacing each memory with one
current node raised occupancy to 406.641 but produced 0.983155 and
0.912109. Keeping the original surface plus one replaceable current
reconstruction produced 231.445 candidates and 0.983893/0.914063. An
exhaustive dual-surface top-512 oracle produced exactly the third
variant’s apparent misses, while every one of the 8,133 ranked control
slots was present in retained history at its sampled query (memory
identities repeat across queries). The mismatch is therefore not HNSW
approximation or eviction.

The quality labels are invalidated because the final database is not an
exact time-travel log of the retrieval surface: reconsolidation updates
`memories.embedding_id`, and the retained reconstruction/current tables
do not encode every intermediate live processor surface at every earlier
query. Even substituting `original_centroid` cannot reconstruct that
mutation order. These runs remain useful negative evidence about the
post-run evaluation method, not evidence for accepting or rejecting the
sparse algorithm. The next valid experiment must capture surface
upserts/removals live and replay that stream against the unchanged
public controls. Four-source text/audio/image evidence, live cycles, and
production cutover remain pending.
`TRACE[state:public_hnsw_route_prototype_experiment]`.

The live mutation capture first passed a 100-event source-health run: 48
ordered upserts were recorded, every vector had 256 dimensions, 27
events activated retrieval, and eight sampled public controls remained
valid. A paired 200-event run then produced identical behavior SHA-256
`25ed7beeb14a7fc1e4201c632c324c1d45a4d6d1f31e2d9e10799738cdb42c5e` and
identical logical-database SHA-256
`1625c3b6a85b585c4e55a8cc83c2871c2b7a6ab8730f788a21b2d9f41f268113` with
capture off and on. This proves observer neutrality only; the full live
mutation population was therefore still required. Using the live stream
as a conservative current-vector route, without attempting to replay
final-table supersession state backward in time, the two-graph candidate
covered every public seed identity and every top-1 identity in both the
eight-query/100-event and sixteen-query/200-event smokes. Candidate
occupancy grew from a mean 9.75 (maximum 20) to 21.9375 (maximum 41).
Exact timestamp, supersession, family, and final-rank decisions remain
downstream engine work; the preliminary exact coverage does not yet
establish full-run quality, bounded cycles, restart, or latency.
`TRACE[state:live_surface_mutation_capture_experiment]`.

The complete live run recorded 9,175 ordered 256-dimensional upserts
across 15,695 packets; this corpus produced no surface removal. It
retained the same 5,390 retrieval-active events and 512 sampled queries
as the prior control. Capture and control had identical full-run
behavior digest
`ed1146a0ed9b2a85eb2ff7c4b511f39357e6db9c363827d941d96163dc7a1463` and
logical-database digest
`522f0dfa9e33e2053903ae938a7bb934319ff836ba608250bd8cd42677b950ca`. The
live two-graph route covered all 8,133 ranked public seed slots and all
512 top-1 identities: exact identity coverage 1.0 and top-1 coverage
1.0. It returned 470.977 candidates on average and at most 512, with a
10.5183-second offline evaluation time. A complete repeat produced the
same per-query candidate-order digest
`2b3e30b0f413692cd8c5a77a15c464ebcc953685901be3904c1b19c96eaf22af` and
the same normalized result digest
`f8dafb51cdac5fafda8f93e1b2b07e9a5135a4f43066625cec92f7a19acebcb0`.
Exact downstream filters remain authoritative, so this is a
candidate-set quality pass, not a final-rank rewrite or a production
cutover. The corpus is still one opaque source and text-only;
four-source text/audio/image evidence, restart bounds, ten live cycles,
and integrated latency remain open.

The next experiment moved that exact candidate route into the real
retrieval operation and used a deterministic 2,500-packet Natural prefix
with four opaque source identities cycling across text, audio, and image
labels after the common encoder. It sampled 128 actual public controls
at *k* = 16. The exact control visited 1,078 current rows at the final
event and spent 1.83742 ms in cached seed discovery. The retained route
visited exactly 512 rows and spent 1.54363 ms, while all 128 ranked
identity lists, the complete behavior digest
`d5bf0c833a2b1677b2df21882145587b21ab03a44cd6b88935f0199d266c7dc2`, and
the logical-database digest
`05e7bf911fe848e1a7d7a8db77be7b7edcf289aab0055e965e901c80366dc06e`
remained exact. Total wall time moved from 48,176 to 49,828 ms
(1.03429x), and mean process time moved from 2.79103 to 2.85218 ms
(1.02191x). This is focused integrated evidence, not full-horizon
flatness, restart, or production cutover.

Two direct integrations were rejected before that result. Updating both
HNSW graphs on every surface mutation retained the 512-row route and
exact sampled ranks but raised wall time to 90,022 ms and mean process
time to 19.7725 ms. A sealed graph plus exact delta that rebuilt after
every one of the run’s 356 consolidations raised wall time to 93,969 ms.
The retained algorithm keeps the sealed graphs across consolidation,
journals changed row addresses in the exact delta, and reseals only when
512 distinct changes accumulate. This is the measured reason
consolidation is an epoch boundary rather than a command to rewrite the
entire index every time it is requested.
`TRACE[state:integrated_hnsw_sparse_route_experiment]`.

The same-event exact-control seam subsequently corrected the integrated
quality measurement. A complete 15,695-packet, four-source replay
cycling text, audio, and image labels sampled 512 retrieval-active
events while running the route and exact current-surface scan on the
same query. Exact top-1 was 1.0, exact identity recall at 16 was
0.999877451, and conservative semantic coverage was 0.999809451;
deterministic tie order and all control-population checks passed. The
exact scan is proof instrumentation only and its 1,160,884 ms wall time
is not a performance result. This satisfies the owner-authorized quality
tradeoff at this scope, but it does not establish bounded restart, cycle
symmetry, or default-on readiness.

A follow-up candidate changed consolidation from threshold-triggered
full resealing to deterministic incremental sealing of every successful
epoch’s distinct additions, updates, and tombstones into the retained
graphs. Focused delete, seal, reinsert, delta-reset, and same-route
tests passed. Two 2,500-packet runs completed all 356 public
consolidations in 59,526 and 60,427 ms wall time, compared with 48,176
ms for exact routing and 49,828 ms for the prior sealed-snapshot
candidate. In the first run, mean ordinary process time was 2.77838 ms
versus 2.79103 ms for control, mean reported total latency was 10.2603
versus 9.65932 ms (1.06221x), and measured consolidation time was
14,341.6 versus 13,907.5 ms (1.03121x). Because the first two wall
regressions of 1.23559x–1.25430x exceeded those fields, a third run
persisted the consolidation-context route timings instead of assigning
the difference to HNSW by inference. It completed in 54,165 ms (1.12432x
control), with mean ordinary process time 2.66766 ms (0.95580x), mean
reported total latency 9.88112 ms (1.02296x), and consolidation time
14,058.0 ms (1.01083x). Across 204 route-active consolidations,
incremental sealing consumed 381.006 ms total (1.86768 ms per active
seal) for 1,163 distinct delta identities. Reloading the authoritative
retrieval surface consumed 9,220.74 ms across all 356 consolidations,
showing that the existing full-surface reload—not the incremental HNSW
mutation—is the dominant named route-epoch cost. The incremental seal is
therefore retained as a default-off candidate within the current latency
tradeoff, while row-addressed SQLite route startup and removal of the
full-surface reload remain required before cutover.
`TRACE[state:integrated_hnsw_consolidation_seal_experiment]`.

### Historical pre-cutover knob-derived SQLite HNSW ablation

The first row-addressed route was moved from a fixed midpoint to the
complete F/S/T-derived surface defined in the implementation section. A
structural 3 × 3 × 3 test grid covered all 27 low, midpoint, and high
combinations with 603 assertions over capacity, backfill, expansion
breadth, bootstrap, visited nodes, cache size, maximum level, topology,
reciprocal updates, and construction/query effort. The neutral setting
intentionally remains the measured 512-candidate, 128-row-backfill
configuration; low and high settings now change the topology and every
work ceiling with it.

These nine runs describe the earlier 4*C* visited-node, 2*C*
search-effort, 8*C* cache candidate and its
exact-fallback-on-any-dirty-row policy. They remain retained as
evaluated history, but they are stale for the current source after the
quality-selected 8*C*/4*C*/16*C* route and bounded *B*-row dirty delta
were integrated. They cannot satisfy the final cutover matrix; the
current-source nine-point 2,000-event rerun is reported immediately
below, and the later 15,695-packet nine-point latency/reset ablation is
reported above.

Nine production-shaped 2,000-event Natural replays then used the
authorized private Claude-session sentence corpus. Each run generated
its own 512-query same-event exact `GraphRetrieve` control at *k* = 16,
used four opaque source identities, and cycled text, audio, and image
labels after the shared encoder. All 4,608 system-path queries passed:
exact top-1 and mean identity recall at 16 were 1.0 in every run,
deterministic tie order held, and the minimum conservative semantic
coverage was 0.999532. Four configurations actually queried the SQLite
sparse route and passed those gates. Sensitivity-low remained below its
activation threshold. Focus-low had a published hierarchy but a nonempty
durable journal at every sampled route opportunity, while
sensitivity-high, stability-high, and all-high remained unpublished; all
five therefore exercised exact fallback. Those runs establish system
fallback quality, not sparse-route quality. Every raw profile records
the resolved knob configuration rather than relying on the evaluator to
reconstruct it.

<table>
<colgroup>
<col style="width: 11%" />
<col style="width: 15%" />
<col style="width: 15%" />
<col style="width: 15%" />
<col style="width: 15%" />
<col style="width: 15%" />
<col style="width: 11%" />
</colgroup>
<thead>
<tr>
<th>Configuration <span
class="math inline">(<em>F</em>, <em>S</em>, <em>T</em>)</span></th>
<th style="text-align: right;"><span
class="math inline"><em>C</em></span> / <span
class="math inline"><em>B</em></span> / max level</th>
<th style="text-align: right;">neighbors / L0 / reciprocal</th>
<th style="text-align: right;">construction / query</th>
<th style="text-align: right;">max visited / restart</th>
<th style="text-align: right;">process / total ms</th>
<th>route state</th>
</tr>
</thead>
<tbody>
<tr>
<td>all-low (0,0,0)</td>
<td style="text-align: right;">256 / 64 / 8</td>
<td style="text-align: right;">32 / 64 / 4</td>
<td style="text-align: right;">100 / 640</td>
<td style="text-align: right;">776 / 1</td>
<td style="text-align: right;">25.532 / 33.749</td>
<td>active; 5 SQLite-route events; observed level 5</td>
</tr>
<tr>
<td>Focus-low (0,.5,.5)</td>
<td style="text-align: right;">384 / 96 / 9</td>
<td style="text-align: right;">48 / 96 / 6</td>
<td style="text-align: right;">150 / 960</td>
<td style="text-align: right;">0 / 0</td>
<td style="text-align: right;">19.010 / 25.449</td>
<td>dirty journal; exact fallback; hierarchy reached level 5</td>
</tr>
<tr>
<td>Sensitivity-low (.5,0,.5)</td>
<td style="text-align: right;">448 / 112 / 9</td>
<td style="text-align: right;">56 / 112 / 7</td>
<td style="text-align: right;">175 / 1,120</td>
<td style="text-align: right;">0 / 0</td>
<td style="text-align: right;">0.544 / 0.952</td>
<td>surface stayed below <span
class="math inline"><em>C</em></span></td>
</tr>
<tr>
<td>Stability-low (.5,.5,0)</td>
<td style="text-align: right;">448 / 112 / 9</td>
<td style="text-align: right;">56 / 112 / 7</td>
<td style="text-align: right;">175 / 1,120</td>
<td style="text-align: right;">906 / 1</td>
<td style="text-align: right;">19.443 / 23.669</td>
<td>active; 31 SQLite-route events; observed level 5</td>
</tr>
<tr>
<td>neutral (.5,.5,.5)</td>
<td style="text-align: right;">512 / 128 / 9</td>
<td style="text-align: right;">64 / 128 / 8</td>
<td style="text-align: right;">200 / 1,280</td>
<td style="text-align: right;">974 / 1</td>
<td style="text-align: right;">14.314 / 18.392</td>
<td>active; 30 SQLite-route events; observed level 5</td>
</tr>
<tr>
<td>Sensitivity-high (.5,1,.5)</td>
<td style="text-align: right;">576 / 144 / 10</td>
<td style="text-align: right;">72 / 144 / 9</td>
<td style="text-align: right;">225 / 1,440</td>
<td style="text-align: right;">0 / 0</td>
<td style="text-align: right;">59.718 / 62.364</td>
<td>build unpublished; exact fallback; observed level 4</td>
</tr>
<tr>
<td>Stability-high (.5,.5,1)</td>
<td style="text-align: right;">576 / 144 / 10</td>
<td style="text-align: right;">72 / 144 / 9</td>
<td style="text-align: right;">225 / 1,440</td>
<td style="text-align: right;">0 / 0</td>
<td style="text-align: right;">13.855 / 18.073</td>
<td>build unpublished; exact fallback; observed level 4</td>
</tr>
<tr>
<td>Focus-high (1,.5,.5)</td>
<td style="text-align: right;">640 / 160 / 10</td>
<td style="text-align: right;">80 / 160 / 10</td>
<td style="text-align: right;">250 / 1,600</td>
<td style="text-align: right;">968 / 1</td>
<td style="text-align: right;">12.800 / 14.656</td>
<td>active; 17 SQLite-route events; observed level 4</td>
</tr>
<tr>
<td>all-high (1,1,1)</td>
<td style="text-align: right;">768 / 192 / 10</td>
<td style="text-align: right;">96 / 192 / 12</td>
<td style="text-align: right;">300 / 1,920</td>
<td style="text-align: right;">0 / 0</td>
<td style="text-align: right;">51.722 / 52.864</td>
<td>build unpublished; exact fallback; observed level 4</td>
</tr>
</tbody>
</table>

The maximum observed backfill batches at the all-low, neutral, and
all-high points were exactly 64, 128, and 192 rows. Active routes never
exceeded their 4*C* visited-node limit and opened the route from one
metadata row. Zero visited rows do not claim zero retrieval work: those
configurations either remained below their activation threshold or had
not accumulated enough consolidation boundaries to publish the bounded
build, so the exact scan was both production candidate and control.

Four defects were rejected during this ablation rather than folded into
the reported pass. The first all-low run visited 1,042 rows against its
1,024-row ceiling because hot-cache hits bypassed the database-miss
budget; the fetcher now applies one ceiling to cached and uncached
nodes, and a warmed-cache regression reproduces the low-knob case.
Earlier generation-reset variants also discarded resumable progress, and
a later build marker could resume nodes but lost committed live
mutations. The retained design instead keeps an authoritative
dirty-identity journal across both active and unpublished generations.
The unpublished build was also copying every live write into an
unbounded in-memory delta even though exact fallback remained
authoritative. At sensitivity-high this exceeded the construction search
ceiling on the second batch. The retained path keeps those live
identities in SQLite, drains at most *B* dirty identities plus *B*
backfill rows per construction edge, and clears only the identities
actually sealed. A focused 400-dirty-identity regression covers both
active and unpublished drainage. Finally, the first persisted
implementation assigned every node to level zero; the retained route
deterministically assigns hierarchy levels capped by
⌈log<sub>2</sub>*C*⌉, bounds upper-level links by the knob-derived
neighbor limit, and requires observed upper-level rows before a corpus
slice may be called route-lifecycle exercised. The retained databases
additionally decode to nonempty upper-level adjacency on every exercised
route (236, 160, 125, and 90 upper-level nodes respectively), rather
than merely containing positive level labels. The nine reruns now record
zero backfill and search failures. They were rerun with benchmark
executable SHA-256
`a398ac2d0ce5188bb1708f811a30689d59412efcffc2c4c08587bb33c6f7c780`; the
four direct regression reports bind test executable SHA-256
`4ee4dd4c5d28583ffea8dc2ab6f43bc40572f2496d589923736b9fb242ad4513`. All
nine SQLite databases are retained in the private artifact root and
content-addressed beside their profiles and audits. The aggregate
artifact is content-bound by SHA-256
`8ef2012604c5efe6f547837bc8e8c0e625a691de52c051187858e43b344fe1a3`.

This matrix proves knob derivation, local work ceilings, bounded
live-dirty drainage, source/modality agnosticism, runtime hierarchy
construction, route lifecycle exercise on four configurations, exact
whole-system quality on all nine, and one-row sparse-route metadata open
at 2,000 events. It does not separately materialize route-active query
quality and does not prove bounded whole-engine restart; processor
startup still loads the complete current surface. It also does not prove
the full 15,695-event plateau, repeated sawtooth symmetry, or
production-wide boundedness. Whole-engine timing is also not an
HNSW-only ablation: changing F, S, or T changes retrieval admission,
consolidation frequency, and surface evolution.
`TRACE[state:sqlite_hnsw_knob_ablation]`.

### Consolidation-centroid and activation-effort follow-up

The first bounded-supersession full replay retained the static 6*C*
query effort. It completed 15,695 events and kept write-time
supersession candidates bounded, but 2,833 consolidations did not
flatten retrieval: mean SQLite sparse route visits rose with the
persisted graph. This rejected the claim that candidate capacity alone
made consolidation restorative.

A connection-local centroid recenter was then added. The first
5,000-event probe still consolidated 801 times because the detector
cooldown incorrectly used a backlog counter that can remain high when a
consolidation has no association mutations. Replacing that condition
with a post-acknowledgement observation count derived from *B* reduced
the same 5,000-event cadence to 35 consolidations, normally 128
observations apart at neutral knobs. That probe still did not lower
route work: 6*C* exceeded the reachable graph in those epochs, so
changing only the entry centroid could not change traversal breadth.
Both variants were therefore retained as rejected diagnostic steps
rather than reported as a performance fix.

The selected follow-up couples the centroid to the 2*C* → 6*C*
activation effort cycle described in the implementation section. A
1,000-event copied-late control at *F* = *S* = *T* = 0.5 exercised seven
successful resets to 2*C* = 1, 024. Its 512 independently captured
public controls across four opaque source identifiers and text, audio,
and image inputs retained exact top-1 1.0, recall at 16 of 1.0, semantic
coverage 1.0, and deterministic tie order. A separate 2,000-event
copied-late performance replay produced 15 complete large-store cycles.
The phase-edge sample is itself knob-derived as *m**a**x*(2, ⌊*B*/32⌋),
or four retrieval-active queries at neutral knobs. Eleven cycles had at
least a 1.10 pre-consolidation visited-row ramp and all 11 reset by at
least 10%; the median immediate reset ratio was 0.859. The first and
second halves of material-cycle peak work differed by 1.041, while
trough work differed by 0.992, both inside the 1.10 late-drift limit. A
fresh 5,000-event replay produced the intended 35-consolidation cadence
but remained below the 2*C* traversal envelope; it is warm-up evidence,
not a late-store sawtooth claim.

The subsequent 15,695-event full replay rejected queue effort as the
only work-control mechanism. It completed 117 consolidations in 927,365
ms. The recorded 2*C* → 6*C* waveform itself was repeatable (p95
normalized shape error 0.0118 and late-template error 0.00496), but
actual node visits continued to rise: only 13 of 23 material cycles
reset, and late/early peak and trough ratios were 1.167 and 1.159.
End-anchored process windows rose from about 2.21 to 3.24 ms, throughput
fell from 29.07 to 10.38 events/s, and the final 500-event tail reached
9.44 events/s. Normalized consolidation cost rose 2.50-fold per sealed
event and 2.16-fold per sealed mutation. HNSW queue effort limits the
candidate queue, not the number of row-addressed graph nodes fetched, so
a symmetric control waveform did not make actual work restorative.

The first actual-node candidate used the knob-derived ⌈11*C*/2⌉ → 12*C*
cycle, or 2,816 to 6,144 nodes in increments of 26 at neutral knobs. Its
copied-late quality probe passed exact top-1 at 1.0, recall at 16 at
0.999878, semantic coverage at 0.999947, and deterministic order. The
full 15,695-event replay cut wall time from 927,365 to 190,736 ms and
mean public latency from 57.542 to 11.068 ms, with no absolute or
dynamic node-budget violations. It nevertheless failed the
restorative-cycle gate: only 10 of 16 material cycles reset, late/early
peaks were 1.107, and troughs were 1.097. This floor was therefore
rejected as too shallow for the required symmetric sawtooth despite its
large runtime gain.

The selected lower-floor candidate uses 5*C* → 12*C* with step
⌈(12*C* − 5*C*)/*B*⌉. At neutral knobs this is 2,560 to 6,144 nodes in
increments of 28. A corrected 1,000-event copied-mature quality probe
enabled the exact HNSW control capture explicitly and exercised eight
simultaneous queue/node resets to 1,024/2,560. Across 512 controls it
kept exact top-1 at 1.0, recall at 16 at 0.999146, semantic coverage at
0.998680, and deterministic order over four opaque sources and text,
audio, and image. Actual visits ranged from 2,560 to 3,740, recorded
ceilings ranged from 2,560 to 6,144, and no query exceeded its current
ceiling. An earlier run without the exact-control capture flag was
classified candidate-pending rather than accepted.

Four follow-up variations were rejected rather than silently folded into
the selected result. Lowering the reset floor by one backfill batch to
5*C* − *B* = 4.75*C* reduced exact top-1 to 0.998047. Holding the 5*C*
floor while reducing its derived ramp increment from 28 to 27 also lost
exact top-1 on the copied mature store. Raising the queue floor
independently to 3*C*, 4*C*, or 5*C*, and holding it at 6*C*, reproduced
that same miss. Queue effort and the last quarter-batch of node budget
were therefore not the root cause. Inspection found a live target with
no incoming persisted links, and also found that the route’s
fetched-node cache was larger than the HNSW best queue but only the best
queue was exactly reranked. A connected traversal could never activate
that target regardless of queue tuning.

The retained repair keeps the HNSW traversal in the existing SQLite
route, then fills the unused portion of the current knob-derived
actual-node envelope from a deterministic indexed active-row slice that
starts at the current entry and wraps once. It exactly reranks every
fetched row, not only the HNSW best queue, before returning the existing
*C*-identity candidate set. The completion is bounded by the same
5*C* → 12*C* ceiling and does not branch on source or modality. In a
512-query copied-mature probe this exact bounded activation kept top-1
and recall at 16 at 1.0, semantic coverage at 0.999736, deterministic
ties, and zero work-bound violations.

A final source review found that seal-time node invalidation erased
cache-map rows without erasing their FIFO order entries. Repeated
seal/search cycles could therefore grow bookkeeping even while the node
map stayed at 24*C*. The retained correction clears the map and order
together after a successful seal. The focused SQLite-route suite
directly checks both structures against the knob-derived 24*C* capacity;
this is bounded-cache source proof, not a new full-corpus latency claim.

That expanding-envelope replay completed 117 consolidations in 175,166
ms, with mean process latency 3.493 ms and mean public latency 9.942 ms.
All 15 material retrieval cycles reset, late/early peak and trough
ratios were 1.074 and 1.002, the late template error was 0.00450, and
the maximum observed node work was 3,357 under the 6,144 hard ceiling.
The retrieval-specific restorative sawtooth passed under that binary.
The whole-engine plateau did not: final-suffix process latency was 1.165
times its leading suffix and throughput retained only 0.877 of its
leading rate. Consolidation process peak and trough ratios were 1.201
and 1.273. After live dirty reconciliation was corrected from a *B*
drain to a *C* drain, however, the expanding public envelope also became
an expensive construction envelope. The current-binary replay took
394,645 ms, including 65,562 ms in 118 consolidations and 58,825 ms in
SQLite route sealing. Public retrieval activated as many as
12*C* = 6, 144 rows, and each dirty root could repeat that work while
selecting graph neighbors. Exact top-1 remained 1.0, mean identity
recall at 16 was 0.998407, and semantic coverage was 0.997572. The
expanding envelope is retained as a rejected experiment, not as the
current algorithm or an engine-wide boundedness claim.

### Fixed activation and separate construction envelopes

The first fixed-envelope candidate separated retrieval work from graph
maintenance. Public retrieval always uses a 5*C* actual-node envelope
and 2*C* queue. Consolidation recenters the connection-local entry but
does not increase or reset work size; the zero ramp is an invariant
selecting the fixed F/S/T-derived envelope. Incremental sealing uses the
independently derived *C* + *B* node and 2*B* queue construction
envelope. Historical backfill remains *B* rows; *B* + 1 names the
logical boundary tested by checking the post-*B* in-memory iterator, not
an extra row read. Live dirty reconciliation remains *C* rows, with
*C* + 1 used only as its overflow probe.

The first neutral 2,500-packet current-binary probe used *C* = 512 and
*B* = 128: public work was fixed at 2,560 nodes and queue effort 1,024,
construction work was bounded at 640 nodes and queue effort 256, and 129
remained solely the logical historical-overflow boundary. Across 512
exact public controls, top-1 was 1.0, mean identity recall at 16 was
0.999876, semantic coverage was 0.999947, and all four opaque sources
plus text, audio, and image post-encoding inputs used the same route.
Eighteen consolidations spent 301.174 ms in SQLite route seals in total;
the maximum individual seal was 80.482 ms, down from the roughly
one-second late seals in the rejected current-binary expanding-envelope
run. This was only a quality and local-work probe. A full 15,695-packet
replay and new nine-point knob ablation were therefore required before
retention or plateau claims; those later runs are reported separately
and reject the whole-engine plateau.

The corresponding full replay rejected that first fixed-envelope
candidate. All mature route queries stayed at exactly the 2,560-row
ceiling and maximum seal time stayed below 114 ms, but exact top-1 fell
to 0.992188, mean identity recall at 16 to 0.975490, and semantic
coverage to 0.967325. Recall by query quarter was 1.0, 1.0, 0.980469,
and 0.921875, so the failure was age-related rather than an immediate 5C
coverage defect. Inspection found that an existing memory whose
consolidation centroid moved updated its embedding but retained its old
outbound neighbor list. The graph became semantically stale even though
its row count and every query remained bounded.

The relinking candidate recomputed outbound neighbors for both new and
changed active centroids under the same *C* + *B* node and 2*B* queue
construction envelope, preserves deterministic level, and applies at
most the derived max (2, ⌊*B*/16⌋) reciprocal updates per level. Its
focused restart regression proved a moved centroid changes persisted
links and remains the nearest result after reopen. The full replay
rejected it: exact top-1 fell to 0.980469, mean identity recall at 16 to
0.972794, semantic coverage to 0.964210, and maximum seal time rose to
140.669 ms. The source change and its regression were removed; the
result remains as negative evidence. The retained correctness repair
does not revive that relinker. A changed active embedding invalidates
the persisted generation, exact retrieval covers the interval, and the
already-bounded historical builder creates a coherent replacement. Entry
removal is cheaper: an indexed removal-count-plus-one query promotes the
highest-level surviving node with deterministic identity ties.

The dual-seed candidate tested activation entry ownership instead. The
prior recenter operation replaced the canonical maximum-level HNSW entry
with the nearest consolidation centroid, which was often a level-zero
node. Later queries therefore skipped global hierarchy descent and
crawled only the local graph. The dual-seed candidate always descends
from the persisted canonical entry, then adds the consolidation centroid
as a second level-zero frontier seed and as the deterministic envelope
pivot. Both seeds share the same fixed 2*C*/5*C* envelope, so this did
not double work. Its full replay still failed: exact top-1 was 0.984375,
mean identity recall at 16 was 0.974142, and semantic coverage was
0.968592. The source change was removed. A copied-late
duplicate-ingestion diagnostic is excluded from acceptance because
replaying already-present packets changed the authoritative population
and timestamps.

The 12*C*/6*C* candidate passed quality but saturated too late to
provide six complete mature 500-event windows, so it remains a
superseded experiment rather than plateau evidence. Reducing the fixed
public envelope to 10*C* nodes and a 5*C* queue moved saturation earlier
while construction remained independently bounded at *C* + *B* nodes and
2*B* queue effort. At neutral knobs these values are 5,120/2,560 for
public retrieval and 640/256 for construction. All are F/S/T-derived,
and the query-age increments remain zero.

The first 10*C*/5*C* full replay isolated five late operation slopes to
one cause: each supersession write or current-surface change invalidated
the whole target-eligibility sidecar, so later storage, reconsolidation,
and emotional cascade cache ensures rebuilt every target. The retained
repair recomputes only the supersession targets affected by the
committed edge or changed source. The next exact-binary replay removed
every late operation failure. A separate family-comparison counter still
had a noisy half ratio even though its event maximum was only 130. A
first *B* = 128 hard cap and a second *C* = 512 cap were both rejected
because the 430- and 600-near-duplicate pagination regressions could
consume the budget before the semantic target. The retained 2*C* cap
passes those regressions and provides a stronger per-event absolute
bound; half ratios remain reported, but bounded work below 2*C* is not
misclassified as store growth.

The final 15,695-packet Natural replay passed both quality and
whole-engine plateau gates. Exact top-1 was 1.0, mean identity recall at
16 was 0.999020, and semantic coverage was 0.998311 across 512 exact
public controls. The run took 333,475 ms overall, with mean process and
public total latencies of 8.885 and 16.617 ms. The accepted plateau
begins at event 11,695 and contains eight complete 500-event windows
while the authoritative store continues growing. Mean plateau process
time is 9.933 ms; late/early process and consolidation-inclusive
throughput ratios are 0.96177 and 0.97979, and the relative Theil–Sen
slope is -0.00768 per window. All operation and work-counter failures
are empty. Twenty-nine complete consolidation epochs form the intended
zero-amplitude fixed envelope, and the maximum 130 exact family checks
remain below the midpoint 2*C* = 1, 024 ceiling. Route sealing consumed
8,375 ms in total and no seal exceeded 97.562 ms.

### Historical C-identity fixed-envelope knob ablation

The prior 10*C*/5*C* candidate, which returned at most *C* identities
from the routing envelope, was rebuilt once and evaluated at all nine
prescribed F/S/T points. Each fresh SQLite store processed 2,000 corpus
packets and used its own 512-query exact public control at the same knob
values. The 4,608 controls covered four opaque source identifiers and
text, audio, and image post-encoding inputs. Exact top-1 and identity
recall at 16 were 1.0 in all nine runs; minimum semantic coverage was
0.999898. Midpoint and focus-low published and exercised the sparse
hierarchy. The other seven runs remained below threshold or ended with
bounded unpublished or dirty state and are therefore system-quality
exact fallbacks, not sparse-route quality claims.

<table>
<colgroup>
<col style="width: 8%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 8%" />
<col style="width: 11%" />
<col style="width: 11%" />
</colgroup>
<thead>
<tr>
<th>Point (F/S/T)</th>
<th style="text-align: right;">C/B</th>
<th style="text-align: right;">public nodes/queue</th>
<th style="text-align: right;">dirty limit/sentinel</th>
<th style="text-align: right;">backfill limit/sentinel</th>
<th style="text-align: right;">route events/resets</th>
<th>quality class</th>
<th style="text-align: right;">top-1 / recall@16 / semantic</th>
<th style="text-align: right;">process / total ms</th>
</tr>
</thead>
<tbody>
<tr>
<td>midpoint (.5/.5/.5)</td>
<td style="text-align: right;">512/128</td>
<td style="text-align: right;">5120/2560</td>
<td style="text-align: right;">512/513</td>
<td style="text-align: right;">128/129</td>
<td style="text-align: right;">15/1</td>
<td>route lifecycle exercised</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">13.749/17.930</td>
</tr>
<tr>
<td>all-low (0/0/0)</td>
<td style="text-align: right;">256/64</td>
<td style="text-align: right;">2560/1280</td>
<td style="text-align: right;">256/257</td>
<td style="text-align: right;">64/65</td>
<td style="text-align: right;">0/0</td>
<td>exact unpublished</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">22.853/27.654</td>
</tr>
<tr>
<td>all-high (1/1/1)</td>
<td style="text-align: right;">768/192</td>
<td style="text-align: right;">7680/3840</td>
<td style="text-align: right;">768/769</td>
<td style="text-align: right;">192/193</td>
<td style="text-align: right;">0/0</td>
<td>exact unpublished</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">48.580/49.771</td>
</tr>
<tr>
<td>focus-low (0/.5/.5)</td>
<td style="text-align: right;">384/96</td>
<td style="text-align: right;">3840/1920</td>
<td style="text-align: right;">384/385</td>
<td style="text-align: right;">96/97</td>
<td style="text-align: right;">232/5</td>
<td>route lifecycle exercised</td>
<td style="text-align: right;">1/1/0.999898</td>
<td style="text-align: right;">10.794/15.802</td>
</tr>
<tr>
<td>focus-high (1/.5/.5)</td>
<td style="text-align: right;">640/160</td>
<td style="text-align: right;">6400/3200</td>
<td style="text-align: right;">640/641</td>
<td style="text-align: right;">160/161</td>
<td style="text-align: right;">0/0</td>
<td>exact unpublished</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">12.363/14.390</td>
</tr>
<tr>
<td>sensitivity-low (.5/0/.5)</td>
<td style="text-align: right;">448/112</td>
<td style="text-align: right;">4480/2240</td>
<td style="text-align: right;">448/449</td>
<td style="text-align: right;">112/113</td>
<td style="text-align: right;">0/0</td>
<td>exact below threshold</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">0.429/0.823</td>
</tr>
<tr>
<td>sensitivity-high (.5/1/.5)</td>
<td style="text-align: right;">576/144</td>
<td style="text-align: right;">5760/2880</td>
<td style="text-align: right;">576/577</td>
<td style="text-align: right;">144/145</td>
<td style="text-align: right;">0/0</td>
<td>exact unpublished</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">57.418/60.169</td>
</tr>
<tr>
<td>stability-low (.5/.5/0)</td>
<td style="text-align: right;">448/112</td>
<td style="text-align: right;">4480/2240</td>
<td style="text-align: right;">448/449</td>
<td style="text-align: right;">112/113</td>
<td style="text-align: right;">0/0</td>
<td>exact unpublished</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">17.766/21.884</td>
</tr>
<tr>
<td>stability-high (.5/.5/1)</td>
<td style="text-align: right;">576/144</td>
<td style="text-align: right;">5760/2880</td>
<td style="text-align: right;">576/577</td>
<td style="text-align: right;">144/145</td>
<td style="text-align: right;">0/0</td>
<td>exact unpublished</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">13.264/17.515</td>
</tr>
</tbody>
</table>

The complete 3 × 3 × 3 structural grid passed 780 assertions, while the
bounded backfill/journal, route-metadata-open, and core-knob suites
passed 1,220, 566, and 393 assertions. At neutral knobs the historical
batch is *B* = 128; 129 is exclusively the derived logical *B* + 1
boundary. The implementation checks the post-*B* iterator without
reading or processing a 129th work row. The corresponding endpoint pairs
are 64/65 and 192/193. Public activation likewise scales from
2,560/1,280 nodes/queue through 5,120/2,560 to 7,680/3,840. Exact family
comparisons independently scale as 2*C*, or 512, 1,024, and 1,536 at
those same endpoints. The content-addressed aggregate digest is
`efdbd964cb255c44d4490814fde954380b683cd2e891ceae13723ba77b691313`. This
closed the earlier C-identity candidate’s knob-derivation and
sampled-quality gate. The later fixed-activation experiments below
supersede it: the *C*-identity route was fast enough for a plateau on
that binary, but did not retain exact long-history quality once the
activated working set was separated from the wider routing envelope.

### Fixed activated-identity follow-up

The retained route distinguishes two finite sets. SQLite routing may
inspect at most 5*C* graph identities so that a sparse hierarchy can
recover distant neighborhoods. Exact reranking then retains at most

*A* = 2*C* + 2*B*

identities for current-surface eligibility, family collapse, and final
ranking. At neutral knobs *C* = 512, *B* = 128, and *A* = 1, 280; at the
all-low and all-high endpoints *A* is 640 and 1,920. Construction
remains a separate *C* + *B* node and 2*B* queue operation. The
canonical maximum-level HNSW entry performs global descent, while the
nearest consolidation centroid is an additional level-zero seed and
deterministic fill pivot. Neither identity set, seed, nor limit depends
on modality or `source_id`.

The activated target was selected through a sequence of full-corpus and
focused negative experiments. The table reports exact top-1, mean
identity recall at 16, and semantic coverage. “Limited route” means the
routing-node ceiling was also set to the activated target rather than
retaining the wider 10*C* route.

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 28%" />
<col style="width: 28%" />
<col style="width: 21%" />
</colgroup>
<thead>
<tr>
<th>Candidate</th>
<th style="text-align: right;">Corpus / wall ms</th>
<th style="text-align: right;">top-1 / recall@16 / semantic</th>
<th>Disposition</th>
</tr>
</thead>
<tbody>
<tr>
<td><span class="math inline"><em>A</em> = 2<em>C</em></span></td>
<td style="text-align: right;">Durable / 27,006</td>
<td style="text-align: right;">0.994141 / 0.985797 / 0.978325</td>
<td>rejected: three sampled top-1 misses</td>
</tr>
<tr>
<td><span class="math inline"><em>A</em> = 3<em>C</em></span></td>
<td style="text-align: right;">Durable / 28,242</td>
<td style="text-align: right;">1 / 0.999753 / 0.999285</td>
<td>superseded: the 1,497 midpoint-near target saturated too late for
six mature windows</td>
</tr>
<tr>
<td><span
class="math inline"><em>A</em> = 2<em>C</em> + 2<em>B</em></span>,
limited route</td>
<td style="text-align: right;">Durable / 27,963</td>
<td style="text-align: right;">1 / 0.998518 / 0.995665</td>
<td>rejected for the shared path after Natural failed</td>
</tr>
<tr>
<td><span
class="math inline"><em>A</em> = 2<em>C</em> + 2<em>B</em></span>,
limited route</td>
<td style="text-align: right;">Natural / 115,461</td>
<td style="text-align: right;">0.964844 / 0.939345 / 0.926401</td>
<td>rejected: long-history neighborhoods were unreachable</td>
</tr>
<tr>
<td>canonical hierarchy plus consolidation seed, limited route</td>
<td style="text-align: right;">Natural / 114,909</td>
<td style="text-align: right;">0.980469 / 0.963583 / 0.950696</td>
<td>improved but rejected</td>
</tr>
<tr>
<td>fill the full target during traversal, limited route</td>
<td style="text-align: right;">Natural / 96,859</td>
<td style="text-align: right;">0.978516 / 0.965016 / 0.952839</td>
<td>rejected</td>
</tr>
<tr>
<td>several arbitrary highest-level seeds</td>
<td style="text-align: right;">Natural / 111,770</td>
<td style="text-align: right;">0.960938 / 0.938407 / 0.926448</td>
<td>rejected and removed</td>
</tr>
<tr>
<td>query-adaptive coarse highest-level seeds</td>
<td style="text-align: right;">Natural / 116,619</td>
<td style="text-align: right;">0.980469 / 0.952914 / 0.939977</td>
<td>rejected and removed</td>
</tr>
</tbody>
</table>

A direct node-envelope ablation then varied the routing ceiling without
changing *C*, *B*, *A*, construction, downstream eligibility, or exact
reranking. The 5,000-packet screen rejected *C* and *C* + *B*.
Full-horizon runs rejected 2*C* and *A* despite their lower wall times
because both violated the mandatory exact-top-1 and recall contracts.
5*C* was the smallest tested envelope to retain exact top-1 and the
accepted recall gate; 10*C* was exact but wider and slower. These are
route-envelope results, not whole-engine plateau passes.

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 28%" />
<col style="width: 28%" />
<col style="width: 21%" />
</colgroup>
<thead>
<tr>
<th>Routing ceiling</th>
<th style="text-align: right;">Events / wall ms</th>
<th style="text-align: right;">top-1 / recall@16 / semantic</th>
<th>Disposition</th>
</tr>
</thead>
<tbody>
<tr>
<td><span class="math inline"><em>C</em></span></td>
<td style="text-align: right;">5,000 / 28,807</td>
<td style="text-align: right;">0.890625 / 0.875265 / 0.864615</td>
<td>rejected</td>
</tr>
<tr>
<td><span class="math inline"><em>C</em> + <em>B</em></span></td>
<td style="text-align: right;">5,000 / 29,765</td>
<td style="text-align: right;">0.974609 / 0.970724 / 0.966273</td>
<td>rejected</td>
</tr>
<tr>
<td><span class="math inline">2<em>C</em></span></td>
<td style="text-align: right;">15,695 / 91,591</td>
<td style="text-align: right;">0.935547 / 0.907982 / 0.884337</td>
<td>rejected</td>
</tr>
<tr>
<td><span
class="math inline"><em>A</em> = 2<em>C</em> + 2<em>B</em></span></td>
<td style="text-align: right;">15,695 / 112,713</td>
<td style="text-align: right;">0.984375 / 0.971970 / 0.959175</td>
<td>rejected</td>
</tr>
<tr>
<td><span class="math inline">5<em>C</em></span></td>
<td style="text-align: right;">15,695 / 101,598</td>
<td style="text-align: right;">1 / 0.999507 / 0.999143</td>
<td>retained: smallest passing envelope</td>
</tr>
<tr>
<td><span class="math inline">10<em>C</em></span></td>
<td style="text-align: right;">15,695 / 103,534</td>
<td style="text-align: right;">1 / 1 / 1</td>
<td>superseded: wider than necessary</td>
</tr>
</tbody>
</table>

A denser graph candidate increased each node to *B* outgoing neighbors
and *B*/4 reciprocal updates. Its focused 600-node HNSW regression did
not finish within three minutes, so it was interrupted before any corpus
result and the retained *B*/2 neighbor and *B*/16 reciprocal formulas
were restored. Earlier integration attempts are also retained as
negative evidence: one full Durable run stayed on exact fallback because
backfill advanced only at consolidation; a lifecycle repair activated
the route but truncated candidates before operation-specific eligibility
and lost quality; a later exact-quality run let visited-node work grow
with the sub-ceiling store and failed the plateau audit. An apparent
15,695-event “Durable” replay that forced each Natural sentence to
Durable was invalidated because it changed the corpus unit and is not
used as performance evidence.

The selected implementation keeps the 5*C* routing ceiling and exact
rerank, then retains *A* = 2*C* + 2*B*. The final 15,695-packet Natural
replay at *F* = 0.45, *S* = *T* = 0.5 resolved *C* = 499, *B* = 125,
*A* = 1, 248, and a 2,495-row routing and queue ceiling. With no
experiment override, it completed in 100,805 ms with mean process and
public total latencies of 2.777 and 6.207 ms. Across 512 controls, exact
top-1 was 1.0, recall at 16 was 0.999507, semantic coverage was
0.999285, and ties were deterministic. The activated set never exceeded
*A*. This proves a fixed semantic activation set and a finite route
envelope, not a whole-engine plateau; the end-anchored plateau audit
still had no accepted suffix.

The corresponding 2,016-message Durable replay used the identical
algorithm and knobs, with Durable adding only its post-commit
checkpoint. With no experiment override, it completed in 29,767 ms with
mean process and public total latencies of 8.178 and 14.350 ms. All
three retrieval quality measures were 1.0. Only three post-warmup
500-event windows existed, below the required ten, so this is
insufficient horizon rather than a Durable plateau pass.

The new nine-point ablation rebuilt one binary and ran fresh
2,000-packet stores at midpoint, both joint endpoints, and each one-axis
endpoint. Every run used its own 512-query same-knob text control over
four opaque sources. A separate active-SQLite-route regression held
embeddings fixed and produced identical candidates and bounds while
varying memory and query labels across text, audio, image, shared
sources, and opaque sources. This split avoids fabricated media in the
corpus audit while proving that the post-encoding optimization has no
source or modality branch. All 4,608 whole-system controls achieved
exact top-1, recall at 16, and semantic coverage of 1.0. Five
configurations both traversed and successfully recentered the hierarchy.
Two traversed but are explicitly recenter-unproven; two remained exact
fallback. The retained audit does not separately materialize the
route-active query subset, so these are route-lifecycle classifications
paired with whole-system quality, not sparse-route-only quality claims.

<table>
<colgroup>
<col style="width: 13%" />
<col style="width: 18%" />
<col style="width: 18%" />
<col style="width: 18%" />
<col style="width: 13%" />
<col style="width: 18%" />
</colgroup>
<thead>
<tr>
<th>Point (F/S/T)</th>
<th style="text-align: right;">C/B/A</th>
<th style="text-align: right;">route events / recenters</th>
<th style="text-align: right;">max routed / activated</th>
<th>quality class</th>
<th style="text-align: right;">process / total ms</th>
</tr>
</thead>
<tbody>
<tr>
<td>midpoint (.5/.5/.5)</td>
<td style="text-align: right;">512/128/1280</td>
<td style="text-align: right;">93/0</td>
<td style="text-align: right;">573/578</td>
<td>traversed, recenter-unproven</td>
<td style="text-align: right;">1.522/7.519</td>
</tr>
<tr>
<td>all-low (0/0/0)</td>
<td style="text-align: right;">256/64/640</td>
<td style="text-align: right;">335/5</td>
<td style="text-align: right;">525/534</td>
<td>route lifecycle exercised</td>
<td style="text-align: right;">1.723/6.366</td>
</tr>
<tr>
<td>all-high (1/1/1)</td>
<td style="text-align: right;">768/192/1920</td>
<td style="text-align: right;">35/0</td>
<td style="text-align: right;">769/785</td>
<td>traversed, recenter-unproven</td>
<td style="text-align: right;">5.339/9.030</td>
</tr>
<tr>
<td>focus-low (0/.5/.5)</td>
<td style="text-align: right;">384/96/960</td>
<td style="text-align: right;">274/2</td>
<td style="text-align: right;">626/657</td>
<td>route lifecycle exercised</td>
<td style="text-align: right;">1.848/7.304</td>
</tr>
<tr>
<td>focus-high (1/.5/.5)</td>
<td style="text-align: right;">640/160/1600</td>
<td style="text-align: right;">48/1</td>
<td style="text-align: right;">643/671</td>
<td>route lifecycle exercised</td>
<td style="text-align: right;">1.601/7.757</td>
</tr>
<tr>
<td>sensitivity-low (.5/0/.5)</td>
<td style="text-align: right;">448/112/1120</td>
<td style="text-align: right;">0/0</td>
<td style="text-align: right;">0/0</td>
<td>exact below threshold</td>
<td style="text-align: right;">0.921/6.440</td>
</tr>
<tr>
<td>sensitivity-high (.5/1/.5)</td>
<td style="text-align: right;">576/144/1440</td>
<td style="text-align: right;">379/2</td>
<td style="text-align: right;">879/955</td>
<td>route lifecycle exercised</td>
<td style="text-align: right;">12.575/21.436</td>
</tr>
<tr>
<td>stability-low (.5/.5/0)</td>
<td style="text-align: right;">448/112/1120</td>
<td style="text-align: right;">470/4</td>
<td style="text-align: right;">892/920</td>
<td>route lifecycle exercised</td>
<td style="text-align: right;">3.054/9.256</td>
</tr>
<tr>
<td>stability-high (.5/.5/1)</td>
<td style="text-align: right;">576/144/1440</td>
<td style="text-align: right;">0/0</td>
<td style="text-align: right;">0/0</td>
<td>exact unpublished</td>
<td style="text-align: right;">1.309/7.771</td>
</tr>
</tbody>
</table>

The structural, backfill/journal, route-metadata-open, core-knob, and
active-route label-invariance suites passed 853, 1,234, 566, 393, and 23
assertions. All route and activation ceilings were respected; no search
or backfill failure occurred; route open read one metadata row.
Whole-engine startup remains *O*(history) because the processor still
hydrates every current memory. The neutral historical work batch remains
*B* = 128, while 129 is only the derived logical boundary checked
through post-*B* iterator state, not a fetched row. The
content-addressed aggregate digest for this pre-commit evidence snapshot
is `a6c8e94af01c1f84d1a1a25e83abc1a3f2cafe9d3880cd553d84dc4704ed66ec`.
Sensitivity-high remains the dominant whole-engine configuration at
43,418 ms wall and 12.575 ms mean process time; this reranks the
independent emotional cascade bound rather than weakening the fixed
retrieval envelope. Because the current Natural whole-engine plateau and
Durable horizon gates are open, these results establish the HNSW
cutover, knob derivation, exact sampled quality, and bounded activated
identity set without claiming the broader goal complete.

### Consolidation-persisted activation snapshot

The first attempt to make consolidation move the activated graph changed
only the route pivot. In a 13,000-packet neutral replay it reached exact
top-1 0.998047, recall at 16 of 0.993042, and semantic coverage of
0.998430. More importantly, none of 50 valid pre/post-recenter pairs
changed the activated identity set for the same consolidation query.
This rejected the hypothesis that replacing or supplementing an entry
pivot was sufficient: the canonical 5*C* completion dominated the pivot
while lower-rank quality fell below its 0.998 gate.

The retained design leaves that canonical HNSW route intact and gives
consolidation one separate, SQLite-persisted activation snapshot. On
each successful seal, the normalized consolidation embedding defines the
semantic center and the route records at most *A* = 2*C* + 2*B* nearest
activated identities. Ordinary queries still perform canonical
maximum-level descent; the snapshot’s identities are loaded through the
separate *A* row budget and become deterministic level-zero entry
points. Neighbor rows discovered from those entry points are still
charged to the canonical 5*C* traversal ceiling. Queries exactly rerank
the union while protecting the best 2*C* canonical identities; the
snapshot may change the remaining 2*B* identities in *A* = 2*C* + 2*B*.
Canonical rows, snapshot rows, and their sum are recorded independently
and fail above 5*C*, *A*, and 5*C* + *A*. The center, generation, first
identity, and packed identity list survive reopen in the single
route-metadata row. This is an internal SQLite migration and does not
add a public API or another Natural/Durable write path.

The neutral 13,000-packet screening replay used *C* = 512, *B* = 128,
and *A* = 1, 280. Canonical rows reached exactly 5*C* = 2, 560, snapshot
rows reached exactly *A*, and their sum reached but never exceeded
3,840. Across 512 public controls, exact top-1 was 1.0, recall at 16 was
0.999754, semantic coverage was 0.999789, and ties were deterministic.
Forty-six successful recenters were profiled; 35 snapshots reached their
full *A* capacity, 11 same-query pre/post pairs changed identity
membership, and no pair was invalid. This is the first retained result
in which consolidation demonstrably changes the fixed activated set
while preserving the canonical route. It was a screening predecessor,
not the final multi-entry implementation.

Three complete 15,695-event variants then isolated the remaining quality
loss. The canonical-only walk plus post-search snapshot returned top-1
0.998047, recall at 16 of 0.995213, and semantic coverage 0.994474.
Restoring one snapshot entry as a level-zero seed improved recall to
0.997054 but remained below the 0.998 floor. Protecting the best 2*C*
canonical identities produced perfect sampled top-1 but recall fell to
0.996072, showing that output-tail allocation could not recover
neighbors the walk never reached. The retained variant therefore makes
the complete persisted snapshot a bounded multi-entry surface before
traversal rather than only an after-the-fact rerank layer.

That retained variant processed all 15,695 events in 470,440 ms with 63
consolidations. Its 512 controls achieved exact top-1 1.0, recall at 16
of 0.999877, semantic coverage 0.999809, and deterministic ties. All 57
profiled recenter pairs were valid; 43 changed membership. In the final
3,000-event suffix, all 10 successful recenter edges changed the fixed
*A* = 1, 248 set and retained only 33.0–36.1 percent overlap. The same
suffix kept the canonical ceiling at 5*C* = 2, 495, the snapshot ceiling
at *A*, and the combined ceiling at 5*C* + *A* = 3, 743. This proves
consolidation changes activation locality while work remains fixed; it
does not prove the independent whole-process plateau.

The required nine production-shaped 4,000-event ablations then rebuilt
fresh stores at midpoint, both joint endpoints, and every one-axis
endpoint. Each run captured 512 same-event public controls over four
opaque sources. All nine traversed the SQLite hierarchy and subsequently
completed between three and 17 successful recenters; none passed through
an exact-fallback-only lifecycle. All 4,608 controls achieved exact
top-1 and recall at 16 of 1.0; minimum semantic coverage was 0.999695,
with zero miss fingerprints. The acceptance policy permits at most one
top-1 miss in each 512-query run (0.998047), requires at least 0.999
across all nine runs, and rejects two or more misses sharing a modality,
opaque source, memory-age quartile, or exact knob point. Thus the
contract allows a bounded approximation without describing it as exact,
while this measured matrix did not consume that allowance.

<table>
<colgroup>
<col style="width: 15%" />
<col style="width: 21%" />
<col style="width: 21%" />
<col style="width: 21%" />
<col style="width: 21%" />
</colgroup>
<thead>
<tr>
<th>Point <span
class="math inline">(<em>F</em>, <em>S</em>, <em>T</em>)</span></th>
<th style="text-align: right;"><span
class="math inline"><em>C</em>/<em>B</em>/<em>A</em></span></th>
<th style="text-align: right;">max canonical / snapshot / total
rows</th>
<th style="text-align: right;">recenters: valid / changed</th>
<th style="text-align: right;">process / total ms</th>
</tr>
</thead>
<tbody>
<tr>
<td>midpoint (.5/.5/.5)</td>
<td style="text-align: right;">512/128/1280</td>
<td style="text-align: right;">517 / 1096 / 1096</td>
<td style="text-align: right;">12 / 0</td>
<td style="text-align: right;">2.932 / 27.993</td>
</tr>
<tr>
<td>all-low (0/0/0)</td>
<td style="text-align: right;">256/64/640</td>
<td style="text-align: right;">405 / 640 / 1045</td>
<td style="text-align: right;">13 / 7</td>
<td style="text-align: right;">3.558 / 27.279</td>
</tr>
<tr>
<td>all-high (1/1/1)</td>
<td style="text-align: right;">768/192/1920</td>
<td style="text-align: right;">865 / 1415 / 1415</td>
<td style="text-align: right;">3 / 0</td>
<td style="text-align: right;">11.153 / 32.204</td>
</tr>
<tr>
<td>focus-low (0/.5/.5)</td>
<td style="text-align: right;">384/96/960</td>
<td style="text-align: right;">400 / 960 / 1304</td>
<td style="text-align: right;">12 / 6</td>
<td style="text-align: right;">3.666 / 26.646</td>
</tr>
<tr>
<td>focus-high (1/.5/.5)</td>
<td style="text-align: right;">640/160/1600</td>
<td style="text-align: right;">603 / 1267 / 1267</td>
<td style="text-align: right;">9 / 0</td>
<td style="text-align: right;">3.210 / 28.522</td>
</tr>
<tr>
<td>sensitivity-low (.5/0/.5)</td>
<td style="text-align: right;">448/112/1120</td>
<td style="text-align: right;">423 / 542 / 542</td>
<td style="text-align: right;">4 / 0</td>
<td style="text-align: right;">1.241 / 25.032</td>
</tr>
<tr>
<td>sensitivity-high (.5/1/.5)</td>
<td style="text-align: right;">576/144/1440</td>
<td style="text-align: right;">592 / 1440 / 1622</td>
<td style="text-align: right;">17 / 4</td>
<td style="text-align: right;">23.536 / 49.794</td>
</tr>
<tr>
<td>stability-low (.5/.5/0)</td>
<td style="text-align: right;">448/112/1120</td>
<td style="text-align: right;">698 / 1120 / 1818</td>
<td style="text-align: right;">13 / 7</td>
<td style="text-align: right;">5.886 / 29.842</td>
</tr>
<tr>
<td>stability-high (.5/.5/1)</td>
<td style="text-align: right;">576/144/1440</td>
<td style="text-align: right;">543 / 931 / 931</td>
<td style="text-align: right;">4 / 0</td>
<td style="text-align: right;">2.276 / 27.394</td>
</tr>
</tbody>
</table>

All 87 successful recenter pairs had explicit, valid overlap evidence
and zero failure codes; 24 pairs changed membership across four knob
runs. A zero in the changed column is not treated as a fabricated reset:
in those short runs the reachable published surface fit inside *A*, so
the fixed snapshot already covered it and overlap correctly remained
1.0. The full-horizon production-knob run above supplies the
mature-store movement proof.

The complete 27-point structural grid passed 985 assertions. The bounded
backfill/journal, restart plus centroid persistence, core-knob, and
active-route source/modality-invariance reports passed 1,234, 64, 393,
and 25 assertions with no failures. At neutral knobs the historical
batch remains *B* = 128 and 129 remains only the logical post-*B*
iterator boundary; no 129th historical row is fetched or processed.
Sensitivity-high remains an adverse whole-engine result rather than
being hidden by the aggregate. These runs prove formula derivation,
local work ceilings, persisted recenter behavior, label-agnostic
routing, and sampled quality. The full Natural plateau, the short
Durable horizon, whole-engine restart, and production-wide boundedness
remain separate gates. The content-addressed aggregate digest is
`f674c55915331279f99f7d4c9eb22e0ed9440b58acd6b3e2a529455219db93d1`.
`TRACE[state:sqlite_hnsw_knob_ablation]`.

### Activation-snapshot read-through reuse

(Superseded: the route generation is now resident in memory in full, see
the sparse-route paragraphs of the implementation section; the bounded
shadow cache described here was removed.)

The retained multi-entry route initially selected and decoded the same
persisted activation snapshot on every query. A read-through repair
places those generation-qualified node rows in the route’s existing
24*C* shadow cache. SQLite remains authoritative: reopen begins cold,
and every successful seal clears the cache before rows from the next
state may be reused. Query distances are still evaluated exactly, so
this changes row loading rather than the 5*C* + *A* semantic envelope. A
regression proves that the first query after reopen loads at most *A*
snapshot rows and an identical second query loads zero while returning
the same identities.

The matched 15,695-packet Natural replay completed in 194,736 ms, down
from 470,440 ms for the uncached activation-snapshot run. Mean process
time fell from 6.384 to 4.729 ms and mean end-to-end time from 29.563 to
12.115 ms. Only 13 events physically loaded snapshot rows; the maximum
load was the derived *A* = 1, 248, and all other routed events reused
the bounded cache. The 512-query public control retained exact top-1
1.0, recall at 16 of 0.999754, semantic coverage 0.999809, deterministic
ties, and four opaque sources. The late eight-window process ratio was
0.999920 and throughput ratio 1.021200, but the whole plateau remained
rejected by emotional-cascade growth, operation counters, and the
requirement for ten mature recenter edges. The different
wall-time-driven consolidation cadence is retained as production-like
temporal feedback, not described as behavior equivalence.

A fixed 2*C* + *A* actual-node ablation was then tested against the same
complete replay and public control. It is rejected: eight of 512 top
identities missed (top-1 0.984375), recall at 16 fell to 0.969452, and
semantic coverage was 0.959699. Its 203,096 ms wall time and 4.502 ms
mean process time cannot waive that loss. This result is why the
accepted 0.998 quality floor is useful rather than ceremonial; the lower
envelope is not retained.

### Exact emotional winner collapse

The next measured late-suffix contributor was the emotional cascade. Its
shared bit-mask breadth-first walk already visits graph adjacency once
per 64-source batch, but it materialized a source–embedding pair for
every source that reached an embedding. The later update loop then
discarded all but the first pair under the established first-source-wins
rule. The exact repair performs that winner selection inside the shared
traversal: for each reached embedding it retains the lowest source
ordinal and that source’s shortest positive depth, then materializes one
winning pair. Reachable-pair and physical edge-visit counters remain
separate from the smaller winning-pair counter so the optimization
cannot hide an expanding graph walk. A 70-source regression now proves
70 reachable pairs and 70 physical edge visits collapse to one executed
winner with the same final intensity. Full-horizon performance,
work-bound, and retrieval results remain required before retention.

The bounded source prefix uses the same intensity-descending, memory-id
ascending order as the SQL path. Because emotional consolidation can
lower an existing source and cascade propagation can raise one,
incremental insertion alone is insufficient: the next-best source may
currently be outside the prefix. The retained repair marks rank-changing
mutations dirty and rebuilds only the knob-derived prefix from a partial
SQLite index. Regressions cover a retained source falling below an
excluded source and an excluded source rising to the front; query-plan
proof confirms that refresh uses the index rather than a store scan. At
this experiment state the proof was specifically
intensity-before-truncation: the bounded cache still chose its global
flashbulb prefix before recency/arousal eligibility, whereas fallback
SQL filtered eligibility before `LIMIT`. The later exact-head edge
repair recorded below moves those predicates before truncation and
supersedes this selection gap; it does not retroactively turn this
earlier timing run into evidence for the repaired query.

The complete exact-winner replay kept top-1 at 1.0, recall at 16 at
0.999386, and semantic coverage at 0.999381, but it did not improve the
full run: wall time was 205,445 ms versus 194,736 ms for snapshot reuse
alone, and mean process time was 4.865 ms versus 4.729 ms. The new
physical counter also showed why: the walk still reached 6,786 adjacency
visits in one event. Exact winner collapse is therefore not promoted by
itself.

An experiment then combined winner collapse with a deterministic
physical-edge ceiling equal to the existing public node formula 5*C*. At
the production knobs the maximum was exactly 2,495 visits; 1,190 of
1,912 active cascade events reached the ceiling, and no 2,496th edge was
visited. Mean cascade time fell from 0.107 to 0.072 ms/event and the
replay completed in 198,222 ms. Its public control consumed the allowed
single miss: top-1 was 511/512 (0.998047), recall at 16 was 0.999141,
semantic coverage was 0.999285, and ties remained deterministic over
four opaque sources. The ten-window suffix beginning at event 10,695
passed whole-process ratio (1.018755), throughput ratio (1.012508),
relative slope (0.005399), and bootstrap upper slope (0.012176). It is
still an experiment rather than a cutover: the current source-selection
scan is not yet bounded by an F/S/T-derived prefix, operation incidence
remains visible, and one material consolidation edge failed the strict
raw post/trailing reset.

The follow-up bounded the source side as well. The maintained source
snapshot now admits at most 5*C* timestamp/identity-ordered candidates
and each cascade executes at most *B* sources, while physical adjacency
remains capped at 5*C* and updates at *A*. A dense all-low regression
proves the exact 1,280-source inspection boundary and the separate
64-source execution boundary; the 1,281st candidate is not materialized.
The audit also stopped double-counting the route’s detailed node and
snapshot counters on top of the already-total
`GraphRetrieve.rows_visited` value.

The corrected 15,695-packet release replay completed in 191,354 ms with
33 consolidations, mean process time 4.552 ms, and mean end-to-end time
11.903 ms. Its 512-query public control passed with top-1 1.0, identity
recall at 16 of 0.999386, semantic coverage 0.999667, deterministic
order, and four opaque sources. Observed maxima were 64 source
inspections and executions, 2,495 physical adjacency visits, 1,337
winning neighbors, and 73 updates, all below their production-derived
5*C* = 2, 495, *B* = 125, 5*C*, 5*C*, and *A* = 1, 248 ceilings. The
ten-window suffix beginning at event 10,695 passed whole-process ratio
(0.992191), throughput ratio (0.994754), relative slope (-0.002547),
bootstrap upper slope (0.001772), and every active-work counter. All ten
mature recenter observations were pair-valid, changed the activation
set, preserved the fixed envelope, and had overlap from 0.348558 to
0.690705.

This is still not a whole-goal cutover. Seven material consolidation
epochs were available, but the strict raw reset and normalized
consolidation-cost trend contracts failed, and several individually
timed bounded kernels varied by more than the current 1.05 component
ratio. The result therefore proves the quality and sparse-work envelope
of this experiment, not a completed Natural plateau, Durable
equivalence, or production-wide bound.

### Post-cutover shared activation distance reuse

The first complete Natural profile after the retained row-addressed HNSW
cutover still increased by 1.69146 ms between its first and final five
post-warmup windows. Graph retrieval contributed 0.97932 ms (57.90%) and
memory storage 0.52965 ms (31.31%), or 89.21% together. Both operations
call the same SQLite `SearchActivated` routine. The graph seed-cache
subsection alone increased by 0.66697 ms, while mean route-node work
increased 3.390x for graph retrieval and 3.460x for memory supersession.
These observations select query-distance materialization inside the
shared activation routine; they do not imply that HNSW routing itself or
the knob-derived activation target is unbounded.

A deterministic 1,300-node regression exposed duplicate arithmetic
directly: one activation made 2,761 exact query-to-node distance
evaluations. The retained query-local distance slot reduces that count
to no more than one per visited persisted node plus one per pending
updated embedding, without changing the 5*C* visit ceiling or
*A* = 2*C* + 2*B* outward activation target.

Two Release binaries built from the same source head then replayed the
same 500 events against byte-copies of the same mature Natural database.
The control removed only the query-local cached-distance early return;
the source delta, binaries, compiler, corpus, initial database, command
contract, profiles, and audits are content-addressed in the retained
evidence packet. Exact behavior, canonical logical-database, and
physical output-database digests matched. Mean process time fell from
4.10098 to 3.76532 ms (-8.18%), Graph retrieval from 2.02630 to 1.83392
ms (-9.49%), its seed-cache section from 1.77608 to 1.59049 ms
(-10.45%), and Memory storage from 0.87427 to 0.77841 ms (-10.96%).
End-to-end mean fell 5.28% and wall time fell 5.49%. A separate earlier
matched fresh 2,500-event pair reduced mean process 1.05%, Graph
retrieval 2.47%, seed-cache work 4.68%, and Memory storage 1.84%; its
wall-time difference was +1.01%, so it is reported as noise rather than
a fresh-store wall-time win. This is a measured local repair, not proof
of the required full-horizon Natural/Durable plateau or
consolidation-cycle shape.

The failed full replay also isolated a separate application-hydration
leak. The final six active working-memory slots contained 14,290 signal
rows, and because they lacked memory-level summary blobs, public
hydration replayed all 14,290 payloads on every packet. Mean hydration
increased from 3.07 ms in the early window to 96.16 ms in the final
window, dominating total latency. The then-retained fallback selected
only each memory’s newest *B* indexed signal rows while SQLite keeps the
complete history. On a copied mature database, a 500-event tail probe at
neutral knobs recorded the 128-row limit on every event and materialized
642–655 fallback rows total across working and retrieved memories. Mean
hydration was 5.395 ms, mean process time 2.900 ms, mean public latency
12.327 ms, end-to-end time 14.166 ms, and wall time 7,160 ms. Relative
to the rejected full replay’s final 500 events, hydration fell 94.4%
(96.162 to 5.395 ms) while six working slots remained active. The
focused regression test proves the newest 128 rows and chronological
output independently for opaque text, audio, and image sources. This
copied-tail probe is direct mature-store performance evidence, not yet a
full-horizon plateau or application-quality claim for omitted older
fallback payloads. A later review regression showed that applying *B*
before surface filtering could hide an older matching payload behind
newer nonmatching or malformed rows. The current implementation pages
newest-first until *B* payloads pass the public surface predicate or the
memory’s candidates are exhausted. A focused adversarial test places 128
newer mislabelled binary payloads ahead of an older valid text payload
and recovers the valid payload. The mature tail timing above is
therefore historical evidence for the pre-correction algorithm; no
post-correction hydration throughput result is claimed, and worst-case
fallback scan work is no longer claimed bounded by *B*.

### Consolidation-cost and seal-cadence ablation

A later full-horizon attribution pass corrected three observer and
routing details before changing the algorithm. Commit timing had
included a profiling-only vector identity lookup; identity resolution
now occurs when the committed mutation audit is consumed, while the
independent logical and physical audit streams remain intact. Mature
deterministic route completion now draws from the indexed
active-generation slice, so an inactive row cannot consume the fixed
envelope or hide a later active identity. The CMake and Zig build graphs
apply the same content-checked HNSW bounds patch; Zig patches a
build-local copy rather than mutating its shared package cache, and runs
Git’s content check and apply from inside that output so no host-native
output path crosses Git’s path parser. The next Windows run exposed a
distinct checkout boundary: CRLF conversion made the unified diff
corrupt to `git apply`, while an LF copy of the same patch applied
cleanly in a local adversarial reproduction. The exact patch now has a
repository `text eol=lf` rule. The patch is a declared build input, so a
patch-content or checkout-contract change invalidates the preparation
step. Local POSIX and CRLF-adversarial proof pass, while replacement
Windows CI remains required; the Zig package allowlist now carries the
`cmake` preparation and patch assets used by that build graph. The RIF
changed-row counter now records work performed by the current event
rather than the historical active population. Each correction has a
red-then-green regression and changes measurement, not public retrieval
semantics.

The same exact-head review exposed three compatibility and completeness
edges. First, migration 30’s empty active-signal ring could admit the
first new aggregate-linked signal before capturing the legacy exact
tail. The repaired first capacity check backfills the newest knob-sized
window; MemoryStorage does so before aggregate insertion, while the
auxiliary working-memory `Upsert` immediately overwrites its
just-inserted row with the supplied exact vector. The 150-plus-one
upgrade regression passes 155 assertions. Second, embedding-only legacy
RIF input had selected one arbitrary memory even when several memories
shared the embedding; the repaired compatibility path suppresses all
such siblings in stable order, and its focused regression passes 5
assertions. The ordinary structured path still supplies `memory_id` and
remains single-memory work; no fixed bound is claimed for the legacy
shared-embedding expansion. Third, `ASSOCIATION` rows could consume
sparse seed slots and then be filtered, leaving the result underfilled
while suppressing exact fallback. The repaired route rejects that
underfilled proposal and restores the SQL path; its 700-node regression
passes 4 assertions. Broader recent-context, competition, and SQLite
HNSW groups pass 309, 331, and 2,638 assertions respectively. These are
deterministic source-health and completeness experiments, not new
long-horizon performance or whole-engine boundedness evidence.

A later exact-head bot review found six additional edge cases in that
repair surface. They were reproduced and corrected together. Eviction
now removes a replacement from the retrieval surface before recomputing
supersession eligibility. Sparse retrieval rechecks requested
cardinality after exact semantic-family collapse, so duplicate activated
rows cannot certify a distinct-family underfill or suppress the complete
processor/SQL seed path. The exact-signal ring orders replacement by
signal timestamp with signal identity only as a stable tie, including
out-of-order writes and physical slot gaps. Emotional source refresh
applies the runtime recency, arousal, and intensity predicates before
the bounded intensity order and `LIMIT`. RIF suppression and recovery
leave `active_rows` unchanged until post-commit shadow publication
applies the membership delta exactly once. Finally, consolidation counts
and selects only rows whose generation matches the singleton recovery
clock, so retired rows do not consume the active-RIF ceiling. The six
focused regressions pass 36 assertions; the surrounding memory-storage,
emotional-cascade, active-RIF, bounded-consolidation, and eviction
groups pass 460, 149, 311, 110, and 40 assertions respectively. These
tests establish deterministic edge behavior on the repaired local
binary; they are not a replacement for exact-head CI, the historical
long-horizon quality matrix, or a new whole-engine boundedness claim.

The next exact-head rereview exposed three narrower ordering and
recovery edges. An exact-signal ring that becomes empty after migration
now backfills only legacy signal embeddings that can be distinguished
from their owning memory centroid, so surviving flattened rows cannot be
relabeled as exact. Sparse supersession applies self, kind, and
timestamp eligibility to the full activated proposal before its
candidate truncation, preserving an eligible duplicate placed
immediately behind a nearer ineligible prefix. Reactivating a persisted
HNSW node now restores it as the canonical entry when it ties the
current maximum level and has the lower memory id. These changes
preserve the existing SQLite authority, knob-derived envelopes,
source/modality neutrality, and Natural/Durable operation path; they add
no schema or public API. Focused regressions construct the emptied-ring
recovery, a larger-than-route supersession population with an ineligible
top-k prefix, and a same-level remove/reactivate tie. They are
deterministic correctness checks rather than a new throughput,
long-horizon quality, or engine-wide boundedness result.

The remaining consolidation scan was bounded by the same knobs. Let
*A* = 2*C* + 2*B* and *N* = max (8, ⌊*B*/2⌋). Score consolidation now
admits at most *A* low-effective-strength long-term identities after
merging two frontiers: at most *A* inactive identities from the indexed
materialized strength/time order and at most *A* active identities
ordered by their exact clock-derived RIF strength. A 64*C* + 1 sentinel
count fails closed before the active branch when the existing
active-epoch mutation ceiling would be exceeded. Both arms exclude
already-clustered memories before applying their *A* limits, so
clustered low-strength rows cannot consume the bounded slice and hide
the next eligible unclustered memory. The inactive arm uses a partial
strength/time/embedding index containing only unclustered long-term
rows; after at most 64*C* active-RIF exclusions it can therefore admit
*A* inactive rows without scanning a clustered history prefix. This
bounds the inactive index scan by *A* + 64*C*, the dynamic active
calculation by 64*C*, and the merged/downstream frontier by *A*, without
a history-sized sort. It also prevents partially recovered rows with
stale low materialized strength from excluding a genuinely weak row. The
scorer counts at most *N* association edges per admitted identity. A
sentinel row publishes the exact input cardinality even when no
candidate is selected. The formula, counter, and source/modality
invariants pass all 27 *F*, *S*, *T* ∈ {0, 0.5, 1}<sup>3</sup>
configurations across text, audio, image, and four opaque source labels.
At midpoint *C* = 512, *B* = 128, *A* = 1280, and *N* = 64; 129 occurs
only as the logical *B* + 1 completion probe.

Four ordinary SQLite-route seal cadences were evaluated. Sealing every
write raised full-replay wall time to 107,294 ms and did not remove
consolidation growth. Sealing at the search-expansion batch
*E* = max (8, ⌊*B*/4⌋) moved the sawtooth into ordinary writes. Sealing
at *N* = *B*/2 left late seed-rank and seal slopes. The retained
candidate seals at *B* − *E* (96 rows at neutral knobs and 94 at the
production replay’s *F* = 0.45, *S* = *T* = 0.5), which is derived at
every knob point rather than stored as a second threshold. Its
15,695-packet Natural replay completed 31 live consolidations and found
an accepted seven-window suffix beginning at event 12,195: no operation
or work-counter gate failed, six complete consolidation cycles were
present, fixed route work was 5*C* = 2495, and normalized consolidation
cost ratios were 0.984 per sealed event and 0.937 per sealed mutation.
This is the measured write-throughput-for-symmetry tradeoff selected for
further quality proof.

The stricter fixed-time, fixed-consolidation, four-opaque-source public
control then exposed a quality boundary not visible in the earlier
control. With the retained 5*C* node and 5*C* queue envelope, exact
top-1 was 0.990234 and identity recall@16 was 0.979902. Restoring the
older *B* seal cadence was slightly worse (0.984375 top-1), so *B* − *E*
did not cause the loss. Bounded node ablations kept the queue at 5*C*:
10*C* reached 0.998047 top-1 and 0.998897 recall; 12*C* reached 0.998047
and 0.999755; 16*C* reached 1.0 for top-1, recall, and semantic coverage
over all 512 queries. The 16*C* point is not retained: its 7,984-node
production budget exceeds the replay’s roughly 3,200 active memories, so
actual work has not saturated, continues growing with the store, and
fails the six-window plateau and fixed-envelope gates. Production
therefore remains at the only mature plateau candidate, 5*C*, while
exact top-1 remains an explicit unresolved cutover gate. These results
do not claim the overall goal complete.

Five fixed-5*C* moved-centroid topology repairs were then evaluated and
rejected on the same 15,695-packet, 512-query control. The first
attached the most recent *B* − *E* changed centroids to the canonical
hierarchy entry; it improved top-1/recall/semantic coverage from
0.990234/0.979902/0.975038 to 0.994141/0.983333/0.976229 but still
missed three top results. Retaining a full *B* identities on that entry
also missed three (0.994141/0.983088/0.975419). Distributing *B*
identities across *R* = max (2, ⌊*B*/16⌋) hierarchy hubs displaced too
much useful locality and fell to 0.986328/0.972426/0.964653; the
narrower two-hub 2*B* target reached only 0.990234/0.977083/0.968464.
Finally, preserving each moved centroid’s old level-zero neighborhood
while splicing only *R* newly proposed links and reciprocals reached
0.992188/0.978554/0.969131. Production knobs for these runs were
*C* = 499, *B* = 125, *E* = 31, and *R* = 7; neutral knobs would have
resolved to *C* = 512, *B* = 128, *E* = 32, and *R* = 8. No experiment
used 129 as a row count: it remains solely the neutral logical *B* + 1
completion boundary. The candidate source and its isolated
moved-centroid regression were removed after rejection. The profile
identities, in the order above, are
`41f4708064c9f0e9034a551c4fd356adfe8b73b4537a734a94920ee6d7e96858`,
`01464b04783eae0443707dc88187d05492c6f02e526d4e450a64c859ce58f5c0`,
`5b0eb95ee32d15f7a2a5ab2511cc5e2d1e72629f0b8d3fbbf968ceae77c2539e`,
`3f3a13a245331e1f9691bd0a7eaf0d6a0900f060bd3e9ed6626ba14117d4b8e8`, and
`1f3216ae0c41c7b105cbd27e299f11b30d0403f63401d2871c13ed98c0428ad5`.
Together these ablations reject bounded link sprinkling as the quality
repair; they do not reject a separately represented sparse activation
layer whose centroids are rebuilt at consolidation under a fixed
knob-derived capacity.

### Historical expanding-envelope knob ablation

The pre-fixed-envelope matrix regenerated all nine prescribed points
from empty SQLite stores on one rebuilt binary. Each run processed 2,000
corpus packets and compared 512 public retrievals at *k* = 16 with an
exact control generated under the same F/S/T values. Across 4,608
controls, exact top-1 and recall at 16 were 1.0; minimum semantic
coverage was 0.999898. The midpoint and focus-low runs published and
exercised the SQLite sparse route, including one and five successful
consolidation recenter resets respectively. The other seven are reported
as system-quality exact fallbacks, not misrepresented as sparse-route
quality: sensitivity-low remained below threshold, while the remaining
six ended with bounded unpublished or dirty hierarchy state.

<table style="width:100%;">
<colgroup>
<col style="width: 10%" />
<col style="width: 13%" />
<col style="width: 13%" />
<col style="width: 13%" />
<col style="width: 13%" />
<col style="width: 10%" />
<col style="width: 13%" />
<col style="width: 13%" />
</colgroup>
<thead>
<tr>
<th>Point (F/S/T)</th>
<th style="text-align: right;">C/B</th>
<th style="text-align: right;">dirty limit/sentinel</th>
<th style="text-align: right;">backfill limit/sentinel</th>
<th style="text-align: right;">route events/resets</th>
<th>quality class</th>
<th style="text-align: right;">top-1 / recall@16 / semantic</th>
<th style="text-align: right;">process / total ms</th>
</tr>
</thead>
<tbody>
<tr>
<td>midpoint (.5/.5/.5)</td>
<td style="text-align: right;">512/128</td>
<td style="text-align: right;">512/513</td>
<td style="text-align: right;">128/129</td>
<td style="text-align: right;">15/1</td>
<td>route lifecycle exercised</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">13.994/18.326</td>
</tr>
<tr>
<td>all-low (0/0/0)</td>
<td style="text-align: right;">256/64</td>
<td style="text-align: right;">256/257</td>
<td style="text-align: right;">64/65</td>
<td style="text-align: right;">0/0</td>
<td>exact unpublished</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">23.539/28.490</td>
</tr>
<tr>
<td>all-high (1/1/1)</td>
<td style="text-align: right;">768/192</td>
<td style="text-align: right;">768/769</td>
<td style="text-align: right;">192/193</td>
<td style="text-align: right;">0/0</td>
<td>exact unpublished</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">49.888/51.132</td>
</tr>
<tr>
<td>focus-low (0/.5/.5)</td>
<td style="text-align: right;">384/96</td>
<td style="text-align: right;">384/385</td>
<td style="text-align: right;">96/97</td>
<td style="text-align: right;">232/5</td>
<td>route lifecycle exercised</td>
<td style="text-align: right;">1/1/0.999898</td>
<td style="text-align: right;">10.964/16.135</td>
</tr>
<tr>
<td>focus-high (1/.5/.5)</td>
<td style="text-align: right;">640/160</td>
<td style="text-align: right;">640/641</td>
<td style="text-align: right;">160/161</td>
<td style="text-align: right;">0/0</td>
<td>exact unpublished</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">12.467/14.549</td>
</tr>
<tr>
<td>sensitivity-low (.5/0/.5)</td>
<td style="text-align: right;">448/112</td>
<td style="text-align: right;">448/449</td>
<td style="text-align: right;">112/113</td>
<td style="text-align: right;">0/0</td>
<td>exact below threshold</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">0.461/0.878</td>
</tr>
<tr>
<td>sensitivity-high (.5/1/.5)</td>
<td style="text-align: right;">576/144</td>
<td style="text-align: right;">576/577</td>
<td style="text-align: right;">144/145</td>
<td style="text-align: right;">0/0</td>
<td>exact unpublished</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">59.519/62.366</td>
</tr>
<tr>
<td>stability-low (.5/.5/0)</td>
<td style="text-align: right;">448/112</td>
<td style="text-align: right;">448/449</td>
<td style="text-align: right;">112/113</td>
<td style="text-align: right;">0/0</td>
<td>exact unpublished</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">18.284/22.522</td>
</tr>
<tr>
<td>stability-high (.5/.5/1)</td>
<td style="text-align: right;">576/144</td>
<td style="text-align: right;">576/577</td>
<td style="text-align: right;">144/145</td>
<td style="text-align: right;">0/0</td>
<td>exact unpublished</td>
<td style="text-align: right;">1/1/1</td>
<td style="text-align: right;">13.813/18.273</td>
</tr>
</tbody>
</table>

The matrix distinguishes the two ingestion surfaces that the earlier
pre-cutover experiment conflated. Historical backfill advances at most
*B* ordered rows; *B* + 1 is its logical overflow boundary, detected
from the post-*B* in-memory iterator rather than an extra row read. Live
dirty reconciliation drains at most *C* identities; *C* + 1 is only its
overflow probe. Thus neutral 129 is not a fetched row or work batch, and
neither 128 nor 129 is a hidden fixed value. The same formulas produce
64/65 and 192/193 at the endpoints, while live dirty limits become
256/257 and 768/769. The 27-point structural grid passed 752 assertions;
bounded backfill/journal, route-metadata-open, and core-knob suites
passed 1,220, 566, and 393 assertions, all without failures. The
aggregate digest is
`7313859052b25c87f6ddc103d6677c9a48572dbdf2b39333a19484a4ad75b892`.

The audit treated queue effort and the actual-node ceiling as separate
derived shape proofs, actual SQLite node rows as effect and bound proof,
and fallback hydration rows as a selected-context bound. This prevents
query-content variation from being mislabeled as engine drift while
still rejecting a perfectly shaped control that fails to reduce real
work. The matrix closed the expanding-envelope knob-derivation and
sampled-quality gate; it does not validate the replacement fixed
envelope, and it did not convert the still-failing whole-engine plateau
into a boundedness claim. The epoch audit also corrected one ownership
assumption: ongoing source-stream accumulator contents are observed
active state but are not owned or cleared by consolidation.
Working-memory pending signals and the three typed dirty-identity sets
remain the required post-commit reset counters.
`TRACE[state:sqlite_hnsw_activation_effort_cycle]`.

### Experiment and algorithm traceability inventory

The following inventory binds the task’s durable experiment state and
every task-owned algorithm group to this paper. The detailed corpus
sizes, parameters, measurements, and failure boundaries remain in the
adjacent prose and tables; these stable identities make omissions
detectable when the state, source diff, or generated manuscript changes.
A retained source-health result is not a production-wide boundedness
claim, and a benchmark, modeled, synthetic, or observation-only result
is never promoted to production-path proof.

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr>
<th>Trace identity</th>
<th>Recorded disposition and proof boundary</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>TRACE[state:baseline_results]</code></td>
<td>Immutable Natural and Durable controls; source-health baseline
only.</td>
</tr>
<tr>
<td><code>TRACE[state:post_hnsw_cost_attribution]</code></td>
<td>The first complete retained-route Natural profile attributes 89.21%
of remaining process growth to the two operations sharing SQLite
activation and selects query-distance materialization without claiming
whole-engine boundedness.</td>
</tr>
<tr>
<td><code>TRACE[state:post_hnsw_query_distance_reuse]</code></td>
<td>Retained query-local persisted-distance reuse; a source- and
binary-bound mature pair is behavior/database exact and improves the
local gate, while full-horizon Natural/Durable plateau proof remains
open.</td>
</tr>
<tr>
<td><code>TRACE[state:repair_results]</code></td>
<td>Ranked measured repairs and their exact behavior/database
equivalence evidence.</td>
</tr>
<tr>
<td><code>TRACE[state:plateau_profiler_smoke_proof]</code></td>
<td>Accepted instrumentation smoke proof; not plateau evidence.</td>
</tr>
<tr>
<td><code>TRACE[state:work_counter_activity_repair]</code></td>
<td>Retained explicit activity-marker repair over 100/200 Natural and
Durable events.</td>
</tr>
<tr>
<td><code>TRACE[state:graph_partial_distance_evaluation]</code></td>
<td>Retained local improvement; full-Natural flatness still failed.</td>
</tr>
<tr>
<td><code>TRACE[state:graph_eligibility_repair]</code></td>
<td>Retained exact eligibility repair; full-Natural flatness still
failed.</td>
</tr>
<tr>
<td><code>TRACE[state:graph_eligibility_db_proof]</code></td>
<td>Canonical database and fixed-probe equivalence proof for eligibility
routing.</td>
</tr>
<tr>
<td><code>TRACE[state:graph_supersession_eligibility_repair]</code></td>
<td>Superseded: supersession exclusion and its eligibility sidecar were
removed in the 2026-10 engine ablation.</td>
</tr>
<tr>
<td><code>TRACE[state:graph_family_vector_bound_candidate]</code></td>
<td>Rejected for fresh-process and graph regression.</td>
</tr>
<tr>
<td><code>TRACE[state:graph_family_projection_bound_candidate]</code></td>
<td>Rejected for copied-late process, graph, and comparison
regression.</td>
</tr>
<tr>
<td><code>TRACE[state:graph_family_precomputed_features_candidate]</code></td>
<td>Rejected for fresh/copied-late process and wall regression.</td>
</tr>
<tr>
<td><code>TRACE[state:supersession_metric_index_candidate]</code></td>
<td>Rejected by the fresh process and wall hard gate.</td>
</tr>
<tr>
<td><code>TRACE[state:competition_rif_returning_candidate]</code></td>
<td>Rejected by fresh process and RIF regression.</td>
</tr>
<tr>
<td><code>TRACE[state:competition_suppression_state_cache_candidate]</code></td>
<td>Rejected by fresh process, wall, and RIF hard gates.</td>
</tr>
<tr>
<td><code>TRACE[state:emotional_metadata_cache_repair]</code></td>
<td>Retained full-run improvement; the flatness audit reranked the next
contributor.</td>
</tr>
<tr>
<td><code>TRACE[state:emotional_cascade_fixed_point_repair]</code></td>
<td>Retained focused fixed-point repair pending full-run
verification.</td>
</tr>
<tr>
<td><code>TRACE[state:emotional_cascade_fixed_point_full_natural]</code></td>
<td>Retained hotspot improvement; the full Natural plateau still
failed.</td>
</tr>
<tr>
<td><code>TRACE[state:accepted_tree_natural_plateau_rerank]</code></td>
<td>End-anchored plateau failure reranked the next independent exact
candidate.</td>
</tr>
<tr>
<td><code>TRACE[state:emotional_cascade_fixed_work_candidate]</code></td>
<td>Terminally rejected bounded-priority experiment. The 27-point
structural grid and exact derived <span
class="math inline"><em>B</em>/<em>B</em> + 1</span> seams passed, but
the fixed-time Natural midpoint missed both the p99-ratio gate (0.573
versus at most 0.25) and public identity quality (top-1 0.998047;
recall@16 0.980760). Its selector was removed, and the nine-point matrix
was correctly skipped by fail-fast.</td>
</tr>
<tr>
<td><code>TRACE[state:shadow_sqlite_seam_proof]</code></td>
<td>Private seam/ownership evidence only.</td>
</tr>
<tr>
<td><code>TRACE[state:shadow_sqlite_scoped_evaluation]</code></td>
<td>Whole-operation shadow was rejected by numeric acceptance
gates.</td>
</tr>
<tr>
<td><code>TRACE[state:post_shadow_rerank]</code></td>
<td>Rejected shadow evidence reranked the measured exact
contributors.</td>
</tr>
<tr>
<td><code>TRACE[state:bounded_sparse_routing_observation]</code></td>
<td>Promising default-off observation; low exact recall and linear
restart forbid cutover.</td>
</tr>
<tr>
<td><code>TRACE[state:active_epoch_sqlite_evaluation]</code></td>
<td>Private active-epoch feasibility and lazy-RIF evidence; production
cutover remains false.</td>
</tr>
<tr>
<td><code>TRACE[state:rif_active_identity_ledger]</code></td>
<td>Retained identity-only, knob-batched active-epoch publication. A
matched copied-mature 1,000-event slice reduced the named hotspot 73.05
percent with exact public behavior and SQLite state; a full default-knob
Natural replay respected <span
class="math inline"><em>B</em> = 125</span> but rejected the
whole-engine plateau on commit and graph-work gates. Durable
checkpoint/horizon and bounded restart remain open.</td>
</tr>
<tr>
<td><code>TRACE[state:rif_epoch_consolidation_reset_candidate]</code></td>
<td>Retained empty post-consolidation in-memory RIF epoch reset. The
nine production-shaped knob points and full 27-point structural grid
preserve <span
class="math inline"><em>B</em> = round (64 + 64<em>F</em> + 32<em>S</em> + 32<em>T</em>)</span>;
at neutral knobs <span class="math inline"><em>B</em> = 128</span>,
while 129 is only the logical <span
class="math inline"><em>B</em> + 1</span> boundary. The named
publication hotspot stayed flat over 15,695 Natural packets, but
whole-engine plateau and retrieval-quality gates failed, so Durable
verification and production cutover remain open.</td>
</tr>
<tr>
<td><code>TRACE[state:post_emotional_priority_rejection_rerank]</code></td>
<td>The valid control reranked active-epoch publication after the
emotional candidate failed; its final six 500-event means rose from
0.347663 to 0.406671 ms while final-suffix whole-process and throughput
gates passed.</td>
</tr>
<tr>
<td><code>TRACE[state:bounded_route_hybrid_observation]</code></td>
<td>Embedding-router feasibility retained; its production-shaped quality
label was invalidated by the public-control population mismatch.</td>
</tr>
<tr>
<td><code>TRACE[state:retrieval_quality_control_correction]</code></td>
<td>Corrected 3,172-memory-surface feasibility passed exactly; runtime
public eligibility/family control remains pending.</td>
</tr>
<tr>
<td><code>TRACE[state:anchor_only_route_feasibility]</code></td>
<td>Rejected anchor-only candidate routing at both 512 and 1,024
anchors; raw one-off output was not retained and is not acceptance
proof.</td>
</tr>
<tr>
<td><code>TRACE[state:public_retrieval_control_capture_experiment]</code></td>
<td>Real GraphRetrieve control capture is observer-exact on the paired
smoke and validates 512 full-run public controls; the corpus is
one-source/text-only.</td>
</tr>
<tr>
<td><code>TRACE[state:public_hnsw_route_prototype_experiment]</code></td>
<td>Three standalone HNSW quality labels were invalidated because the
final database cannot reconstruct every point-in-time live surface; live
mutation capture is required.</td>
</tr>
<tr>
<td><code>TRACE[state:live_surface_mutation_capture_experiment]</code></td>
<td>Full live capture is behavior/database exact and the repeated
512-query route has exact identity/top-1 coverage;
cross-source/modality, restart, live-cycle, and latency gates
remain.</td>
</tr>
<tr>
<td><code>TRACE[state:integrated_hnsw_sparse_route_experiment]</code></td>
<td>Historical default-off sealed-HNSW plus exact-delta integration
established same-event quality before the row-addressed SQLite route was
retained; full-horizon plateau and cycle proof remain open.</td>
</tr>
<tr>
<td><code>TRACE[state:integrated_hnsw_consolidation_seal_experiment]</code></td>
<td>Historical incremental epoch sealing isolated 381 ms of seal work
versus 9,221 ms of pre-existing full-surface reload; the retained SQLite
route replaces full resealing, while full-horizon cycle proof remains
open.</td>
</tr>
<tr>
<td><code>TRACE[state:sqlite_hnsw_knob_ablation]</code></td>
<td>Production-default <span
class="math inline">8<em>C</em></span>-to-<span
class="math inline">9<em>C</em></span> plus consolidation-snapshot-<span
class="math inline"><em>A</em></span> matrix: the 27-point structural
grid and all nine sequential 4,000-event corpus runs passed. All 4,608
controls measured top-1, recall@16, and semantic coverage of 1.0 with
deterministic ties and zero miss clusters. Every short route was
traversed, but none reached the <span
class="math inline">8<em>C</em></span> maturity floor needed for a valid
pre/post activation pair, so all nine are explicitly
recenter-unevaluated. Separate backfill/journal, restart, core-knob, and
text/audio/image plus source-label invariance regressions pass. <span
class="math inline"><em>B</em>/<em>B</em> + 1</span>, <span
class="math inline"><em>C</em>/<em>C</em> + 1</span>, the <span
class="math inline">8<em>C</em></span> floor, <span
class="math inline"><em>R</em></span> step, <span
class="math inline">9<em>C</em></span> ceiling, snapshot <span
class="math inline"><em>A</em></span>, combined <span
class="math inline">9<em>C</em> + <em>A</em></span>, and
family-comparison <span class="math inline">2<em>C</em></span> bounds
are distinct and knob-derived.</td>
</tr>
<tr>
<td><code>TRACE[state:sqlite_hnsw_sparse_route_integration]</code></td>
<td>Retained row-addressed SQLite HNSW query and incremental seal in the
existing <code>GraphRetrieve</code> and <code>MemoryStorage</code>
paths; Natural and Durable share the same algorithm, and Durable adds
only its checkpoint. Current full-corpus retrieval quality and
activation bounds pass, while the Natural whole-engine plateau and
Durable horizon gates remain open.</td>
</tr>
<tr>
<td><code>TRACE[state:sqlite_hnsw_activation_effort_cycle]</code></td>
<td>Queue-effort-only, fixed-<span
class="math inline">5<em>C</em></span>, and dynamic <span
class="math inline">4<em>C</em></span> through <span
class="math inline">8<em>C</em></span> floor variants were rejected or
superseded. The retained candidate resets routing work to <span
class="math inline">8<em>C</em></span>, advances by <span
class="math inline"><em>R</em> = max (2, ⌊<em>B</em>/16⌋)</span> to
<span class="math inline">9<em>C</em></span>, keeps <span
class="math inline"><em>A</em> = 2<em>C</em> + 2<em>B</em></span>
downstream activation and <span
class="math inline">2<em>B</em>/(<em>C</em> + <em>B</em>)</span>
construction, passes the 30,380-packet mature-cycle gates at 511/512
top-1 and recall@16 0.999265, and passes canonical and nine-point
quality without an engine-wide plateau claim.</td>
</tr>
<tr>
<td><code>TRACE[state:sqlite_hnsw_production_cutover]</code></td>
<td>Retained the existing SQLite HNSW route as the production default at
an <span class="math inline">8<em>C</em></span> post-consolidation
floor, <span
class="math inline"><em>R</em> = max (2, ⌊<em>B</em>/16⌋)</span> step,
<span class="math inline">9<em>C</em></span> ceiling, separate <span
class="math inline"><em>A</em> = 2<em>C</em> + 2<em>B</em></span>
snapshot, and <span class="math inline">9<em>C</em> + <em>A</em></span>
total row ceiling. The nine-point matrix passed 4,608/4,608 top-1
controls, minimum recall@16 and semantic coverage 1.0, deterministic
ties, and source/modality-label invariance. Natural and Durable share
the operation path, with Durable adding only its checkpoint;
whole-engine reset, bounded restart, Durable plateau, release, and
deployment remain unclaimed.</td>
</tr>
<tr>
<td><code>TRACE[state:pr6_exact_head_ci_repair]</code></td>
<td>PR-monitor repair record: migrated active RIF rows are calibrated
before working-memory hydration; pinned HNSW speculative neighbor
prefetches are adjacency-bounded without disabling SIMD distance
kernels; empty adjacency snapshots skip a zero-byte copy whose vector
destination may be null; Zig applies the same patch to a packaged
build-local copy by running Git from inside that output rather than
passing a host-native output path; deterministic route fill excludes
inactive rows; entry removal promotes an indexed active hierarchy root,
reactivation restores a higher persisted root, overlapping activation
snapshots clear atomically, recenter rebuilds do not use prior snapshot
identities as explicit frontier seeds, and restart rejects legacy dead
canonical or snapshot identities; changed active embeddings invalidate
stale adjacency for bounded rebuild; consolidation merges exact
active-RIF and indexed inactive frontiers under the existing <span
class="math inline">64<em>C</em></span>/<span
class="math inline"><em>A</em></span> ceilings and filters clustered
memories through an eligible-only partial index before either bounded
arm; bounded emotional-source discovery preserves SQL intensity order
and refreshes rank-changing mutations through an indexed knob-sized
prefix; partial edge-limited cascade traversals do not become fixed
points; and cache-byte arithmetic uses explicit <code>std::size_t</code>
width for WebAssembly. The full media CTest baseline, focused
RIF/HNSW/consolidation regressions, build-local Zig proof, and
WebAssembly bundle build pass locally. The sandboxed macOS ASAN runtime
cannot execute because dynamic-shadow discovery or sanitizer-runtime
code signing fails before <code>main()</code>, so replacement exact-head
Linux sanitizer CI remains the required owner; the historical
4,608-query quality matrix is not relabeled as evidence from the
repaired binary.</td>
</tr>
<tr>
<td><code>TRACE[state:hnsw_fixed_6c_experiment]</code></td>
<td>Rejected knob-derived fixed-<span
class="math inline">6<em>C</em></span> routing envelope. All 27
structural F/S/T points and 931 assertions passed; neutral <span
class="math inline"><em>C</em> = 512</span> produced a 3,072-node
envelope while <span class="math inline"><em>B</em> = 128</span> and the
129 boundary remained distinct. An extra production-default
F=.45/S=.5/T=.5 pre-screen missed exact top-1, recall@16, plateau, and
wall-time gates, so fail-fast rejected the candidate before the exact
nine-point corpus matrix started and left production on fixed <span
class="math inline">5<em>C</em></span>.</td>
</tr>
<tr>
<td><code>TRACE[state:post_rif_empty_fixed_envelope_attribution]</code></td>
<td>Repaired proof classification for the retained zero-increment
fixed-<span class="math inline">5<em>C</em>/5<em>C</em></span> route.
Retrieval mode is independent of whole-engine process shape;
route-active rows fail closed on missing work, and queue, node ceiling,
actual visits, distance work, and activation are checked separately
across at least ten mature unchanged-envelope recenters. The production
suffix beginning at event 10,195 passes the classifier and work envelope
across 12 complete recenters, while activated-identity overlap,
process-cycle reset, exact full-horizon retrieval, Durable horizon, and
bounded restart remain open.</td>
</tr>
<tr>
<td><code>TRACE[state:write_gate_incidence_attribution_v2]</code></td>
<td>Deterministic range-bound observation-only 15,695-packet
production-knob attribution rejected boundary-to-write incidence as the
growth source: exact event ranges 0–2,499 and 13,195–15,694 contained
510 and 528 writes, while MemoryStorage time per write rose from 0.4833
to 2.8343 ms as the existing shared SQLite HNSW route warmed toward the
fixed <span class="math inline">5<em>C</em> = 2, 495</span> envelope.
Production <span
class="math inline"><em>C</em>/<em>B</em>/<em>A</em> = 499/125/1, 248</span>
and construction <span
class="math inline"><em>C</em> + <em>B</em>/2<em>B</em> = 624/250</span>
remain derived; neutral <span
class="math inline"><em>C</em>/<em>B</em>/<em>A</em> = 512/128/1, 280</span>
and 129 is only logical <span class="math inline"><em>B</em> + 1</span>.
Raw reset remains failed and unwaived; 27 structural and nine
production-shaped F/S/T ablations remain required.</td>
</tr>
<tr>
<td><code>TRACE[state:recenter_activation_overlap_natural_13000_v2]</code></td>
<td>Fail-closed same-query pre/post activation measurement rejected the
current connection-local HNSW recenter as a consolidation-driven
centroid reset. A deterministic builder accepts only explicitly
pair-valid, zero-failure observations; across 22 such successful
recenters in 13,000 Natural events, including 14 at the full <span
class="math inline"><em>A</em> = 1, 248</span> activation target and
later queries saturating the <span
class="math inline">5<em>C</em> = 2, 495</span> node envelope, all 22
retained exactly the same activated identities (minimum overlap 1.0).
This proves the route is integrated but its current recenter does not
move the activated set; raw reset, 27 structural and nine
production-shaped F/S/T ablations, modality/source-id invariance, and
whole-goal proof remain open.</td>
</tr>
<tr>
<td><code>TRACE[state:current_verification]</code></td>
<td>The no-selector production default passes canonical Natural quality
at 512/512 top-1 and nine-point quality at 4,608/4,608, with recall@16
and semantic coverage 1.0 throughout. The 2,016-message Durable replay
also passes 512/512 with identical route parameters and only its
checkpoint barrier, but its horizon remains insufficient for plateau
proof. Structural, backfill/journal, route-metadata-open, core-knob,
source/modality invariance, Python, and C++ focused gates pass.
Whole-engine raw process reset and bounded restart remain open and are
not part of the scoped retrieval-envelope claim.</td>
</tr>
<tr>
<td><code>TRACE[state:fixed5c_moved_centroid_reachability]</code></td>
<td>Five fixed-<span class="math inline">5<em>C</em></span> bounded
single-hierarchy moved-centroid link repairs were measured and rejected;
the result does not reject a separately represented
consolidation-rebuilt activation layer.</td>
</tr>
<tr>
<td><code>TRACE[state:bounded_fallback_hydration]</code></td>
<td>Retained newest-<span class="math inline"><em>B</em></span> indexed
signal fallback reduced copied mature-tail hydration from 96.162 to
5.395 ms and passed opaque text/audio/image recency tests; full-horizon
and application-quality proof remain open.</td>
</tr>
<tr>
<td><code>TRACE[state:packed_route_sqlite_representation_experiment]</code></td>
<td>Packed SQLite queries retained the exact route result, but
complete-snapshot rewrite was rejected for incremental
consolidation.</td>
</tr>
<tr>
<td><code>TRACE[state:row_addressed_route_sqlite_experiment]</code></td>
<td>Row-addressed SQLite made a 200-event seal incremental at fresh and
copied-late history; the representation is now integrated, with
full-horizon copied-late and cycle proof still pending.</td>
</tr>
<tr>
<td><code>TRACE[rejected:shared-shadow-sqlite-operation-view]</code></td>
<td>Rejected whole-history shadow projection.</td>
</tr>
<tr>
<td><code>TRACE[rejected:GraphRetrieve.seed_cache_family_compare.coordinate-block]</code></td>
<td>Rejected exact coordinate-block graph candidate.</td>
</tr>
<tr>
<td><code>TRACE[rejected:GraphRetrieve.seed_cache_family_compare.projection-bound]</code></td>
<td>Rejected projection-bound graph candidate.</td>
</tr>
<tr>
<td><code>TRACE[rejected:GraphRetrieve.seed_cache_family_compare.precomputed-features]</code></td>
<td>Rejected immutable precomputed-feature graph candidate.</td>
</tr>
<tr>
<td><code>TRACE[rejected:MemoryStorage.supersession_metric_index]</code></td>
<td>Rejected supersession metric-index candidate.</td>
</tr>
<tr>
<td><code>TRACE[rejected:Competition.rif_recovery_active_sql.update-returning]</code></td>
<td>Rejected update-returning RIF candidate.</td>
</tr>
<tr>
<td><code>TRACE[rejected:Competition.rif_recovery_active_sql.suppression-cache]</code></td>
<td>Rejected suppression-cache RIF candidate.</td>
</tr>
<tr>
<td><code>TRACE[rejected:RifActiveEpoch.row-batch-only-performance-cure]</code></td>
<td>The knob-derived batching contract was retained, but batching
complete RIF values reduced copied-mature publication only 3.99 percent
versus the required 20 percent and was rejected as the performance
cure.</td>
</tr>
</tbody>
</table>

The bounded-route search itself had five separately evaluated points
over the same 34,456-vector, 512-query top-16 benchmark:

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr>
<th>Trace identity</th>
<th>Result</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>TRACE[bounded-route-variant:reciprocal-route-only]</code></td>
<td>Rejected: exact recall at 16 was 0.979858.</td>
</tr>
<tr>
<td><code>TRACE[bounded-route-variant:three-sparse-graphs]</code></td>
<td>Rejected: no recall gain at equal per-graph work.</td>
</tr>
<tr>
<td><code>TRACE[bounded-route-variant:construction-budget-4096]</code></td>
<td>Rejected: exact recall at 16 was 0.995972.</td>
</tr>
<tr>
<td><code>TRACE[bounded-route-variant:construction-budget-8192-reciprocal-64]</code></td>
<td>Rejected: exact recall at 16 was 0.997803.</td>
</tr>
<tr>
<td><code>TRACE[bounded-route-variant:two-graphs-construction-16384-retrieval-1280-reciprocal-64]</code></td>
<td>Selected benchmark design: exact top-1 1.0 and recall at 16
0.998413; production proof pending.</td>
</tr>
</tbody>
</table>

The source diff is partitioned into the following algorithm and
experiment implementation groups. The traceability audit requires every
changed C/C++ path under `src/` or `include/`, every changed benchmark
C/C++ path, and every changed non-test experiment/audit tool to belong
to exactly one group, so an unrecorded algorithm or experiment
implementation fails closed.

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr>
<th>Trace identity</th>
<th>Algorithm surface</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>TRACE[algorithm:consolidation-envelope-and-detector]</code></td>
<td>Drift rearm, accumulator reset, and consolidation-throughput state
(shallow consolidation removed 2026-10).</td>
</tr>
<tr>
<td><code>TRACE[algorithm:shared-retention-and-rollback]</code></td>
<td>Shared Natural/Durable accumulation, signal-record rollback journal,
working memory, and processor transaction ownership.</td>
</tr>
<tr>
<td><code>TRACE[algorithm:lazy-rif-active-epoch]</code></td>
<td>Lazy RIF clock/generation state, active-epoch SQLite,
memory-strength integration, store compatibility, and migration.</td>
</tr>
<tr>
<td><code>TRACE[algorithm:bounded-graph-retrieval]</code></td>
<td>Bounded activation observation, graph retrieval, historical search
cache, evaluated embedding-family features, and explicitly enabled
public-seed tracing.</td>
</tr>
<tr>
<td><code>TRACE[algorithm:emotional-propagation-cache]</code></td>
<td>Emotional state, fixed-point propagation, and rebuildable metadata
cache.</td>
</tr>
<tr>
<td><code>TRACE[algorithm:memory-storage-and-graph-maintenance]</code></td>
<td>Supersession storage, association fanout, stability updates, and
usage detection (eviction removed 2026-10).</td>
</tr>
<tr>
<td><code>TRACE[algorithm:runtime-composition]</code></td>
<td>Engine pipeline composition and hydration/routing integration.</td>
</tr>
<tr>
<td><code>TRACE[algorithm:bounded-work-instrumentation]</code></td>
<td>Consolidation-epoch and mutation-ownership counters used by the
acceptance audit.</td>
</tr>
<tr>
<td><code>TRACE[algorithm:experiment-harness-and-audit]</code></td>
<td>Owner-authorized session extraction, Natural/Durable replay,
public-surface export and control capture, row-addressed SQLite
experiment, and fail-closed performance/quality audit
implementation.</td>
</tr>
</tbody>
</table>

Natural and Durable use one ingestion algorithm; Durable adds only the
post-commit flush/checkpoint barrier. The algorithm does not branch on
modality or source cardinality: `source_id` remains opaque provenance,
while text, audio, and image inputs converge after encoding. The final
traceability pass is rerun after production changes and the last
experiment, so this inventory cannot become a stale intermediate
snapshot.

A follow-up SQLite representation run separated fixed query work from
snapshot write amplification. The four-blob, 1,024-anchor snapshot
occupied 207,847,424 bytes for 34,456 embeddings. Restart touched 1,029
rows. The 512 top-16 route queries took 0.116659 seconds in total
(0.22785 ms/query), performed a mean 512.004 and maximum 513 exact
comparisons, and retained exact top-1 and recall-at-16 of 1.0. A
1,024-embedding fresh prefix built, exported, restarted, and checked in
0.44 seconds wall time. However, exporting the complete snapshot took
65.5011 seconds.
`TRACE[state:packed_route_sqlite_representation_experiment]` therefore
retains the row-addressed SQLite query seam but rejects rewriting the
store-sized packed blobs at consolidation. The production-shaped
candidate must append only the sealed epoch’s nodes, edges, and routes,
with bounded reciprocal updates; copied-late append, public-cycle, and
failure/ownership proof remain pending.

The row-addressed follow-up retained separate indexed node, graph-edge,
and route rows and changed only the sealed epoch. In a 200-event fresh
database, derived apply plus commit took 0.002320 seconds (0.01160
ms/event). At a 34,256-row copied-late prefix, inserting the same 200
nodes and two edge rows per node plus updating 3,544 unique reciprocal
route rows took 0.010586 seconds (0.05293 ms/event). The 4.56 late/fresh
ratio reflects the deliberately absent reciprocal population in the
empty control, but its absolute cost remained small and no store-sized
value was rewritten. An injected pre-commit failure restored exact table
counts; close and reopen retained exact committed counts; restart read
the dimension plus 1,024 anchors, or 1,025 rows total. The 512
fixed-route queries each compared exactly 512 embeddings and took
0.621792 seconds total (1.21444 ms/query). This is
`TRACE[state:row_addressed_route_sqlite_experiment]`: measured
representation evidence, not graph-construction, public-path latency,
post-authoritative-commit publication, Durable-barrier, or live-sawtooth
proof.

The profiler’s zero-counter guard was also corrected during this
evaluation. The initial 100-event Natural and Durable smoke audits
incorrectly treated a positive recovery-operation duration as evidence
that at least one suppressed row must exist. Recovery may legitimately
run while the active suppression population is empty. The audit now
pairs every required work count with the engine’s explicit boolean
activity marker (`candidate_activity`, `rows_visited_activity`, and the
corresponding graph, rollback, emotional, predictive, and supersession
markers) rather than with an inclusive operation timer. A zero count
beside a zero activity marker is valid; a zero count beside a positive
marker still fails as a vacuous placeholder. A regression preserves that
distinction. Rebuilt 100- and 200-event Natural and Durable profiles
then passed row schema, producer/count equality, activity,
consolidation-epoch, active-epoch-limit, and Durable-barrier validation.
These short runs validate instrumentation only; they are not plateau or
long-horizon performance proof.

On 2026-06-30, a full Meta MSC replay rerun verified the retrieval-cache
optimization on the same 9,130-turn slice used for the hosted
frontier-judge artifact. The run preserved native probe behavior
exactly: all non-timing probe fields, retrieved/working memory IDs,
retrieval counts, memory counts, and consolidation counts matched the
saved baseline. At 1,000-event progress checkpoints,
`GraphRetrieve.total` was 13.7-27.7 ms after optimization versus
31.4-323.5 ms in the baseline. Across the 9 judged probe turns,
`GraphRetrieve.total` was 21.8-28.6 ms after optimization versus
81.6-303.6 ms in the baseline. The verification artifact is
`docs/paper/artifacts/graph_profile/full_msc_verify_final/summary_slim.json`.

## Durable-Ingestion Scaling Audit

On 2026-07-14, a fixed 2,636-event private replay was used to profile
the sequential durable-ingestion path. Durable artifacts contain only
aggregate timings, counts, and cryptographic behavior/database digests;
no message text, identifiers, or local paths are published. The baseline
and every retained candidate produced the same event-by-event behavior
digest and the same canonical logical database digest, ending with 2,657
memories, 2,657 signals, and 57,643 associations.

The exact origin/main baseline averaged 1,022.552 ms of process time and
2,720,276 ms wall time. Its final-five/first-five 100-event-window
ratios were 9.808 for `MemoryStorage`, 9.821 for supersession-edge work,
9.318 for process latency, and 0.101 for sequential messages per second.

The retained Candidate 34 computes the typed supersession query norm
once and reuses the exact `double` across the candidate loop. It
produced the same event-by-event behavior digest
`4e728353eab989086481217cd140930875b112dcccb1fe372acdefc9b16c6bb8` and
canonical logical-database digest
`a2a538e619b9b10b1a5f2892993a168e7b0f8be31cd0eaa912937f9d26a2ddfe`. The
sanitized, content-bound aggregate supporting this retained candidate is
`artifacts/flat_storage_cost/candidate_34_summary.json`; raw profiles
and databases remain private proof inputs. The final current-source run,
after making the private pre-filter population mirror every runtime
reconstruction and working-memory embedding and repairing restart
hydration of the latest reconstruction, including base-only ablation
parity, averaged 8.560 ms process time, 14.551 ms total latency, and
39,338 ms wall time. It had the same behavior/database digests and
57,643/2,657/2,657 association/memory/signal counts. Its fixed gate
results were:

Earlier Candidate 34 timing samples omitted runtime reconstruction
embeddings from the private pre-filter mirror. Their behavior and
database digests remain valid, but their scan-cost measurements are
superseded by this corrected run.

<table>
<thead>
<tr>
<th>Gate metric</th>
<th style="text-align: right;">Candidate 34 observed</th>
<th style="text-align: right;">Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>MemoryStorage</code> final-five/first-five</td>
<td style="text-align: right;">2.063758</td>
<td style="text-align: right;">≤ 1.05</td>
</tr>
<tr>
<td><code>MemoryStorage</code> Theil–Sen slope (ms/message)</td>
<td style="text-align: right;">0.000684779</td>
<td style="text-align: right;">≤ 0.01</td>
</tr>
<tr>
<td><code>MemoryStorage</code> bootstrap 95% upper slope</td>
<td style="text-align: right;">0.000703932</td>
<td style="text-align: right;">≤ 0.02</td>
</tr>
<tr>
<td>supersession final-five/first-five</td>
<td style="text-align: right;">2.528664</td>
<td style="text-align: right;">≤ 1.05</td>
</tr>
<tr>
<td>supersession Theil–Sen slope (ms/message)</td>
<td style="text-align: right;">0.000686189</td>
<td style="text-align: right;">≤ 0.01</td>
</tr>
<tr>
<td>supersession bootstrap 95% upper slope</td>
<td style="text-align: right;">0.000694684</td>
<td style="text-align: right;">≤ 0.02</td>
</tr>
<tr>
<td>process final-five/first-five</td>
<td style="text-align: right;">1.739414</td>
<td style="text-align: right;">0.90–1.10</td>
</tr>
<tr>
<td>messages/s final-five/first-five</td>
<td style="text-align: right;">0.692262</td>
<td style="text-align: right;">0.90–1.10</td>
</tr>
</tbody>
</table>

The small absolute slopes pass their loose ceilings, but both storage
ratios and both throughput ratios fail. Candidate 34 is therefore a
material improvement over the baseline and earlier exact candidates,
**not** proof of flat storage cost or flat end-to-end throughput.

A separate current-source 1,000-event durable SQLite-profile sensitivity
run used 50-event audit windows. It matched the established 1,000-event
behavior and logical-database digests and ended with 19,845
associations, 1,021 memories, and 1,021 signals. The final
current-source run also failed the ratio gates: 1.633813 for
`MemoryStorage`, 2.070751 for supersession, 1.414776 for process
latency, and 0.844665 for messages/s. This confirms that the non-flat
conclusion is not an artifact of the realtime SQLite benchmark profile.

A later shared-path audit used the owner-authorized 15,695-packet
Natural corpus and its 2,016-message Durable form. The retained exact
historical coverage proof and batched emotional traversal produced a
Natural mean process time of 2.197 ms and wall time of 746,302 ms. Its
audited tail still failed: process final-five/first-five was 1.575951
and throughput was 0.331728. Graph retrieval contributed 0.473572 ms
(43.7% of the process delta), emotional cascade 0.146594 ms, and memory
storage 0.124507 ms. The corresponding Durable run used the identical
ingestion algorithm plus its checkpoint edge, averaged 10.148 ms process
time, and completed in 75,684 ms. It also remained non-flat: process
ratio 1.839862 and throughput failure, with graph retrieval contributing
3.289430 ms, commit 1.144678 ms, and the checkpoint edge 0.699095 ms of
the positive delta. The Natural behavior/logical-database digests are
`e4f843e017902bb04f10e6b57bafeb9e850d663f8dc3719970fa94ffd189f3de` and
`76ee7281189b3b4446361a125a6f64090599049d6c6f2d6a257da916eff10adf`; the
Durable digests are
`25762284607b1851ea89189e073332b78d3d8fa8126e6c038e3b396c174ff463` and
`1c067e1dde67e5d288656bc137b4206bcbd2c1a76c1c0e95a9391318b7ec5267`.
These are exact harness results, not production-wide boundedness
evidence.

The storage-cost audit contract was subsequently tightened to count
`rif_recovery_clock`, `rif_generation_resets`, and `rif_active_state` at
every store checkpoint, and to reject any Durable event reporting a
failed WAL checkpoint. The saved shared-path profiles above predate
those added fields, so their timings and digests remain historical
measurements but do not satisfy the current audit schema. A current-gate
replay is required before using them as cutover evidence.

Explicit consolidation required a separate lifecycle proof because force
persistence broadly prunes and rewrites embedding rows. Candidate 34 now
rebuilds the historical search registry from authoritative SQL only
after a successful force commit; a failed commit restores the
rolled-back registry. On the same 1,500-message prefix with forced
consolidations at messages 750 and 1,500, the exact pre-repair
SQL-fallback tree and the repaired tree produced the same event-behavior
digest
`a86d5aa3ed1ae945a794fd6b071f3d80bdada008a66cc2076313d7da404c6a5a` and
logical-database digest
`4751e7d7b6711d7e968ff7c31761150a22fab9735dbd4c961b3227f4bfa3bf81`.
Across the 750 messages after the midpoint consolidation, mean process
time was 8.098 ms with the rebuild versus 496.352 ms on the fallback;
mean `MemoryStorage` time was 1.117 versus 485.853 ms, and total replay
wall time was 19,586 versus 388,127 ms. A repaired full 2,636-message
midpoint run also completed two consolidations in 38,929 ms and retained
a 10.088 ms post-midpoint mean process time. The corresponding full
pre-repair comparator was intentionally stopped incomplete after 1,906
messages because the demonstrated fallback made it disproportionate; it
has no final digest or audit and is not counted as full baseline
evidence.

The fixed diminishing-returns audit then evaluated three distinct
behavior-exact, database-exact, warning-clean full-corpus changes:
caching target norms, retaining typed-container capacity, and
serializing the SQL query only when fallback was used. All three missed
every predeclared flatness-breakthrough threshold and were reverted.
Direct validated append and reusable assembly scratch also regressed at
their bounded exact gates. Chronological blocks, a sparse grid, a
matrix-vector kernel, persistent rank populations, rank refill,
cross-operation exact memoization, and previous-query triangle bounds
had already failed their exact gates or feasibility conditions.

At 2,636 events, the historical surface held 6,762 entries but 6,622
exact vector groups, so grouping removes only 140 evaluations (2.0704%).
The triangle-bound probe evaluated 99.975% of 256-dimensional candidates
on average over 1,000 events and 99.804% in its final 100. Under the
current observable candidate semantics, the retained scan remains
`O(Nd)` on this corpus. Another narrow constant-factor repair is not
expected to flatten it. Changing the candidate contract,
consolidation/tiering/eviction semantics, storage/index backend, public
processor-state ownership, or fixed proof gate requires separate
approval.

A later 15,695-packet natural-retention profile separated this retained
scan into static surface eligibility and dynamic retrieval work. In its
fixed late 200-packet window, the reference path evaluated a mean 38,074
embedding rows per retrieval although only 1,345 survived the existing
eligibility predicate. An incrementally maintained long-term-reference
index moved the unchanged timestamp, current-surface, and supersession
predicate ahead of distance evaluation. It reduced evaluated distance
rows to 3,211 (-91.6%), distance time from 4.459 ms to 0.376 ms
(-91.6%), and cached seed search from 13.561 ms to 9.575 ms (-29.4%). A
fixed-timestamp paired replay produced identical ordered behavior and
SQL-dump digests. The remaining family-comparison cost still grew with
the eligible surface. A subsequent exact rejection uses the identity
‖*a* − *b*‖<sup>2</sup> = ‖*a*‖<sup>2</sup> + ‖*b*‖<sup>2</sup> − 2*a* ⋅ *b*:
while accumulating squared coordinate differences, a partial sum already
above the maximum distance compatible with the duplicate threshold
soundly rejects that pair. Near-threshold pairs continue through the
existing Cauchy–Schwarz bounds and exact cosine check. In a corrected
fixed-clock, `--reuse-db` late-store pair, it reduced mean family
comparison from 1.757 to 0.798 ms, exact checks from 7,654 to 3.81 per
event, graph time from 2.321 to 1.349 ms, and process time from 4.974 to
4.417 ms. Event behavior and canonical logical-database digests matched
exactly. Total latency was unchanged because hydration dominated the
slice. A fresh 200-packet pair also matched behavior and logical
database state and showed no process or total-latency regression. This
is an exact constant-factor reduction, not proof of bounded retrieval
latency.

The retained partial-distance loop was subsequently evaluated in fixed
16-coordinate vectorized blocks. Computing through the end of a block
cannot invalidate rejection: squared coordinate contributions are
nonnegative, so a completed block above the threshold proves the
complete distance is also above it. On a fixed-clock copied-late
200-packet pair, the block form preserved the event behavior oracle
exactly while reducing mean family-bound comparison from 0.736 to 0.424
ms (-42.4%), cached graph seed work from 1.195 to 0.887 ms (-25.8%), and
process time from 6.912 to 6.741 ms (-2.5%). The fresh pair also
preserved the behavior oracle and improved process time. Total late
latency changed by about +1.1%, within the 10% regression gate. This
remains an exact constant-factor contributor repair; it does not
establish a flat tail or sublinear exact nearest-neighbor search.

The subsequent immutable-feature cache removes repeated construction of
the same normalized vector and the same stored/Hadamard block summaries.
In a fixed-clock copied-late 200-packet pair, it preserved both the
behavior and logical-database digests, reduced family-feature
construction from 0.216 to 0.049 ms per packet (-77.3%), and reduced
mean process time from 2.494 to 2.275 ms (-8.8%). The cache changes no
candidate population or rank and is replaced whenever its embedding
entry is replaced.

The emotional topology footprint was evaluated over all 15,695 packets.
It preserved the full behavior and logical-database digests, reduced
cascade traversal executions from 3,244 to 1,664, and lowered the late
five-window cascade mean from 0.235 to 0.144 ms. Mean process time
changed from 2.056 to 2.039 ms. The overall plateau gate still failed:
exact supersession selection and graph seed work remained store-size
dependent. These results support the two exact contributor repairs, not
a bounded-tail claim.

The final current-tree replay enabled both repairs by default and again
processed all 15,695 packets. It preserved the event-behavior digest
`1b5b29f3cd342b94b7438b90940c2da1e896f869777aba02cb2ce4790109e452` and
canonical logical-database digest
`551f5aabf8c9cf1cdddaaf4adbc2b83119474d46553610380f8001d1c51005e1`. Mean
process time was 2.116 ms and mean end-to-end time was 48.739 ms. The
current plateau audit still rejected the tail: final-five/first-five
ratios were 1.699 for `MemoryStorage`, 2.059 for supersession-edge work,
1.380 for process time, and 0.344 for throughput. The final artifact
therefore proves semantic equivalence for this replay and preserves the
measured exact constant-factor improvements, but it does not prove
bounded Natural writes or authorize a production-wide flat-storage
claim.

## Signal-centroid ring cutover and knob ablation

Commit profiling isolated a write-amplification defect that earlier
aggregate timings obscured. Every accepted signal inserted a new row
into the global sqlite-vec surface even though memory KNN later joined
through memories and discarded those signal-only rows. In a 2,000-event
control, accepted commits averaged 0.2240 ms and mutated 21.78
`embeddings_chunks` rows per accepted write. The retained centroid-ring
candidate reduced those figures to 0.1594 ms and 4.00 rows,
respectively; mean process time fell from 0.8614 to 0.7048 ms.

Two intermediate designs were rejected. Storing an exact vector inline
on the ever-growing `signals` table while repointing historical
working-memory rows to the newest centroid touched approximately 600
signal rows per accepted write in the diagnostic profile. Removing the
repoint avoided that direct *O*(*N*) mutation but still left an
unbounded inline-vector table. The retained design instead stores one
global centroid per memory and a persistent bounded exact-vector ring.
The original experiment used size *B*(*F*, *S*, *T*); restart review
later showed that `LoadRecentContext` can restore
*N*<sub>*c**t**x*</sub>(*T*) + *K*<sub>*c**t**x*</sub>(*T*) signals. The
current capacity is therefore
*Q* = max (*B*, *N*<sub>*c**t**x*</sub> + *K*<sub>*c**t**x*</sub>): 64
at the all-low point, 151 at neutral knobs, and 266 at the all-high
point. All 27 low/mid/high F/S/T combinations verify *Q* and the
complete restore-window inequality. Signal rows reference the centroid,
so the ring changes neither durable payload identity nor the public
retrieval surface.

The content-addressed benchmark binary, which predates the *Q*
restart-window correction and used the original *B* ring, then replayed
all 15,695 Natural packets at *F* = 0.45, *S* = *T* = 0.5, resolving
*B* = 125 and *A* = 1, 248. It processed 32 consolidations in 183,001 ms
with mean process and total times of 4.2730 and 11.4210 ms. The maximum
ring-table mutation count was 12 in any event. Emotional updates reached
133 against the hard *A* = 1, 248 enqueue limit, and the limit was never
saturated.

That run is not reported as a whole-engine plateau pass. No suffix
satisfied every combined consolidation-cycle, timing, and active-counter
gate. Its 512-query full-horizon control also measured exact top-1
0.994141, identity recall@16 0.993623, and semantic coverage 0.990139;
top-1 and recall therefore miss the current mandatory whole-goal
thresholds. The exact 2,016-message Durable replay completed four
consolidations in 30,128 ms with mean process and total times of 9.1500
and 14.5741 ms. Its retrieval controls remained 1.0, but neither
attribution nor the required plateau passed. These failures remain part
of the result rather than being replaced by the local ring improvement.

The current-binary nine-point Natural ablation used 2,000 events, 512
exact public GraphRetrieve controls, and four opaque source identifiers
per point. Every point passed exact top-1, exact identity recall at 16,
semantic coverage, and deterministic tie order at 1.0. A separate
active-route regression covers text, audio, and image labels plus shared
and opaque source identifiers, so the text-ingress corpus table is not
misrepresented as fabricated multimodal traffic.

<table>
<colgroup>
<col style="width: 15%" />
<col style="width: 21%" />
<col style="width: 21%" />
<col style="width: 21%" />
<col style="width: 21%" />
</colgroup>
<thead>
<tr>
<th>F/S/T point</th>
<th style="text-align: right;">ring <span
class="math inline"><em>B</em></span></th>
<th style="text-align: right;">max ring row mutations/event</th>
<th style="text-align: right;">wall (ms)</th>
<th style="text-align: right;">top-1 / recall@16 / semantic</th>
</tr>
</thead>
<tbody>
<tr>
<td>midpoint</td>
<td style="text-align: right;">128</td>
<td style="text-align: right;">12</td>
<td style="text-align: right;">14,072</td>
<td style="text-align: right;">1 / 1 / 1</td>
</tr>
<tr>
<td>all-low</td>
<td style="text-align: right;">64</td>
<td style="text-align: right;">8</td>
<td style="text-align: right;">12,039</td>
<td style="text-align: right;">1 / 1 / 1</td>
</tr>
<tr>
<td>all-high</td>
<td style="text-align: right;">192</td>
<td style="text-align: right;">4</td>
<td style="text-align: right;">15,619</td>
<td style="text-align: right;">1 / 1 / 1</td>
</tr>
<tr>
<td>focus-low</td>
<td style="text-align: right;">96</td>
<td style="text-align: right;">14</td>
<td style="text-align: right;">14,222</td>
<td style="text-align: right;">1 / 1 / 1</td>
</tr>
<tr>
<td>focus-high</td>
<td style="text-align: right;">160</td>
<td style="text-align: right;">7</td>
<td style="text-align: right;">14,997</td>
<td style="text-align: right;">1 / 1 / 1</td>
</tr>
<tr>
<td>sensitivity-low</td>
<td style="text-align: right;">112</td>
<td style="text-align: right;">12</td>
<td style="text-align: right;">12,464</td>
<td style="text-align: right;">1 / 1 / 1</td>
</tr>
<tr>
<td>sensitivity-high</td>
<td style="text-align: right;">144</td>
<td style="text-align: right;">6</td>
<td style="text-align: right;">39,900</td>
<td style="text-align: right;">1 / 1 / 1</td>
</tr>
<tr>
<td>stability-low</td>
<td style="text-align: right;">112</td>
<td style="text-align: right;">10</td>
<td style="text-align: right;">17,623</td>
<td style="text-align: right;">1 / 1 / 1</td>
</tr>
<tr>
<td>stability-high</td>
<td style="text-align: right;">144</td>
<td style="text-align: right;">14</td>
<td style="text-align: right;">14,558</td>
<td style="text-align: right;">1 / 1 / 1</td>
</tr>
</tbody>
</table>

The sensitivity-high latency outlier is retained as measured. The
aggregate `signal-centroid-ring-knob-ablation-v4.json` binds all
profiles and sanitized audits to benchmark binary SHA-256
`4ac8b7239545cdc1fcc7a0b51a0c131c6ef4ce3656ddbcbe6706c8f3baa76e83` and
has SHA-256
`d75f8244b84686b93578b644a7b4c2a9ecf0963bd98017f57dc75245204a7044`. Its
29-file source-code manifest has SHA-256
`41ea4060829c156230e5f40cd8f06d4197b00650b8a10a3e9aca06e6b7df2897`; the
scope is explicit so unrelated concurrent branch work cannot silently
change the evidence identity. The v4 builder fails closed unless every
point contains exactly 2,000 rows with unique event indices covering 0
through 1,999, 512 controls, four opaque source digests, all three
quality values at 1.0, deterministic ties, an observed ring-mutation
counter, and an audit digest bound to the private profile. Its capacity
check uses the production `std::lround` rule and includes a half-integer
regression. The plateau audit likewise treats
`MemoryStorage.insert_signal_rows` as elapsed time rather than inferring
diagnostics from a `_rows` suffix; a negative fixture grows only that
duration and must fail. This closes knob derivation, the named
nine-point ablation, public retrieval quality for those slices, and
source/modality-independent routing structure. It does not claim a
whole-engine plateau, a bounded emotional traversal, merge, release, or
deployment. `TRACE[state:signal_centroid_ring_cutover]`.

Post-signal-ring attribution exposed a distinct consolidation-edge
defect in the retained identity ledger. Across the first and final five
successful consolidations of the 15,695-packet Natural replay, mean
consolidation duration rose from 21.306208 to 53.983433 ms. The
disposable-ledger publication rose from 0.425858 to 6.475533 ms while
the mean persistent current-generation identity population rose from
226.0 to 3,113.2. Each SQL statement still obeyed the derived *B* = 125
limit, but the number of statements copied the complete population.
Batching therefore bounded statement size without bounding total
consolidation work.

The measured repair keeps persistent SQLite authoritative and changes
only successful epoch reset: the new connection-local SQLite epoch
begins with its clock and no copied identities, and later events publish
only their changed identities. The deterministic regression covers all
nine production-shaped F/S/T points, using persistent *B* + 1
populations and mixed modality and opaque source labels. It initially
failed at neutral knobs because 129 identities were copied, then passed
171 assertions after the repair; after the double-failure regression was
added, the complete active-epoch focused set passed 482 assertions in 15
cases. Neutral 128 remains the derived batch, and 129 remains the
logical *B* + 1 test population rather than a work batch.

Fresh blind review found that the first implementation did not carry the
empty-reset intent across consecutive failures. When both post-commit
publication and its immediate recovery failed, the next event could
rebuild the disposable RIF projection from all persistent active
identities. The first repair made the empty-reset requirement sticky and
preserved that sidecar across successful retrieval-surface reloads.
Round-two review then found that the reload exception handler still
erased it. That exact sequence was reproduced as a missing sidecar
before repair. The catch path now detaches and restores the RIF epoch
independently of the other disposable retrieval caches. A regression
with a neutral *B* + 1 = 129 mature persistent population injects
publication, recovery, and reload failures, then proves that the retry
observes an empty disposable epoch and publishes only one changed
identity; the durable 129 identities remain in persistent SQLite. This
is failure-path source-health evidence. The earlier 15,695-packet timing
run predates both review repairs and continues to support only the
successful-path local publication result.

The exact repaired binary then completed the full 15,695-packet Natural
replay with 38 live-hint consolidations in 219,074 ms. The selected
publication edge was flat: its first-five and final-five consolidation
means were 0.198092 and 0.196425 ms, and all 38 publications consumed
9.043001 ms. The local repair is therefore retained. It did not complete
the engine goal. No suffix passed the consolidation-cycle contract, and
the 512-query public control reached exact top-1 0.998047, exact
identity recall at 16 of 0.996094, and semantic coverage 0.993760, below
the mandatory 1.0 and 0.998 identity gates. Durable verification was not
run after the first Natural acceptance gate failed. Bounded restart,
recursive review, overall plateau, and production boundedness remain
open.

A smallest-next-envelope feasibility probe then raised only the private
HNSW node inspection bound from 5*C* to 6*C*; activation remained
*A* = 2*C* + 2*B* and every value remained F/S/T-derived. The complete
27-point structural grid passed 931 assertions, with neutral *C* = 512,
node budget 3,072, and *B* = 128; 129 remained only the logical *B* + 1
boundary. An additional production-default F=.45/S=.5/T=.5 Natural
pre-screen resolved *C* = 499 and 6C=2,994, processed all 15,695 packets
in 231,275 ms, and regressed wall time 5.57 percent relative to the
repaired 5C run. Exact top-1 remained 0.998047 with the same one miss;
identity recall at 16 improved from 0.996094 to 0.997070 but remained
below 0.998. No suffix passed the plateau or consolidation-cycle
contracts. The 6C candidate is therefore rejected before the exact
nine-point corpus matrix started. This is a complete 27-point formula
ablation, one additional default-knob pre-screen rejection, and 0/9
exact corpus-matrix points, not a nine-point performance pass. The
historical benchmark executable is bound by SHA-256
`86bafe48d4a75e8ebaf6d942fb92aca59e3078af189d4a8148a31846ee3e5a13`; its
unchanged experiment-owned selector, formula, harness, regression, and
audit source manifest is bound by SHA-256
`35c0de6f3f05dd3e0720db01c372fa32b58f572fe269610ed56a7a3ee26a5198`.

Re-attribution also exposed a proof-classification defect rather than
another HNSW algorithm change. The retained route declares zero
activation increments, but the audit previously chose a ramping
retrieval-cycle test whenever process time formed a sawtooth. The same
fixed 5*C* work therefore failed for having no material retrieval ramp.
A deterministic regression reproduced the mismatch, and the repaired
audit now selects fixed-envelope retrieval from the declared F/S/T
parameter vector independently of process mode. Over the production
replay suffix beginning at event 10,195, 12 complete successful
recenters kept both queue and node envelopes at 5*C* = 2, 495 with no
classifier or work-envelope failure. The historical profile did not
record activated-identity overlap, so it cannot prove the full
centroid-movement retrieval-cycle contract. The whole goal remains open:
overlap, process-cycle reset, operation/counter suffix gates, exact
top-1/recall, Durable horizon, and bounded restart are unchanged. This
repair adds no runtime value, route, modality branch, or source-id
branch; the existing 27 structural and nine production-shaped knob
contracts remain the required ablations.

The next process-reset attribution separated event incidence from
per-active-unit cost. Ten suffix epochs met the material-ramp
definition, and eight met the strict requirement that the following
50-event process mean be at most 0.9 of the trailing mean. The two
misses, epochs 28 and 35, still removed 90.65% and 66.40% of their
preceding process ramps. Their post windows contained two and six times
as many supersession candidates, respectively. Memory-storage time per
candidate improved by 28.46% in epoch 28 and changed by only 1.25% in
epoch 35. Epoch 28 also contained 3.69 times as many emotional neighbors
while time per neighbor improved by 7.57%; epoch 35 moved from no
trailing-neighbor activity to 472.48 neighbors per event after
consolidation. Thus these two absolute-latency misses do not identify a
store-size-dependent engine hotspot; they expose content-incidence
sensitivity in the strict reset audit. An activity-normalized diagnostic
now compares post to trailing windows separately for memory-storage
candidates and emotional-cascade neighbors. It uses ratios of window
sums, treats zero/zero activity as unevaluated, treats a one-sided zero
as incomparable, and deterministically ranks per-unit regressions before
activity-incidence explanations. It is diagnostic only: raw reset, peak,
trough, slope, height, throughput, work, and continued-store-growth
failures remain unchanged and cannot be waived.

On the same production suffix, all evaluable rows were
activity-incidence rather than per-unit regressions. Epoch 35 memory
storage was selected deterministically because its post window added
24.215 ms while candidate incidence rose sixfold; its unit-cost ratio
was 1.0125, below the existing 1.05 operation threshold. The result
continues to attribution of the memory-storage activity-generation path
while the strict process-reset failure remains. Nineteen focused
diagnostic tests and all 111 storage-audit tests pass, including
contiguous adjacent and cross-operation-identical window checks, finite
aggregate checks, and label-invariant arithmetic for text, audio, image,
shared, and opaque labels. The candidate introduces no runtime work
value. A content-addressed inventory evaluates 33 existing F/S/T-derived
formulas over all 27 structural points and retains the exact nine
production-shaped points; neutral *B* = 128, while 129 remains only the
logical *B* + 1 exclusion boundary and never a fetched or processed row.
No runtime algorithm, knob formula, public API, schema, modality
behavior, or source-id behavior changed in this audit-only
implementation.

Direct attribution of the selected memory-storage activity path
localized the remaining incidence jump one level earlier. In epoch 28
the trailing and post windows stored 6 and 12 memories; in epoch 35 they
stored 2 and 12. Each stored memory in both windows returned exactly 394
supersession candidates and visited the same F/S/T-derived 1,248-row
activation target. Thus the twofold and sixfold candidate totals are
exactly the twofold and sixfold counts of natural writes, not growth in
candidates per write. At epoch 35, sparse-route node rows per stored
memory changed by a factor of 1.0305, candidate-load time per stored
memory by 1.0144, and total memory-storage time per stored memory by
1.0125; epoch 28 improved on the corresponding time measures. The next
measured surface is therefore boundary-to-write incidence in
`ComputeWriteGate`, not another MemoryStorage candidate-search repair.
This is attribution, not authorization to alter memory semantics: the
raw reset miss remains failed and unwaived.

The replay used production knobs *F* = .45, *S* = .5, and *T* = .5,
deriving *C* = 499, *B* = 125, *A* = 1, 248, 5*C* = 2, 495, 2*C* = 998,
*C* + *B* = 624, and 2*B* = 250. Neutral `.5/.5/.5` remains a distinct
ablation point with 512/128/1, 280/2, 560/1, 024/640/256; 129 is only
its logical, unprocessed *B* + 1 boundary. The attribution adds no fixed
work value and uses neither modality nor source-id labels as a budget
input. The production corpus is text-only evidence; separate mixed-label
regressions retain responsibility for modality/source-id invariance.

### Write-gate incidence and existing-path HNSW attribution

An observation-only `ComputeWriteGate` profile then recorded every
decision input and exit class without changing write semantics. A
deterministic builder binds the first comparison to event indices
0–2,499 and the end-anchored final comparison to 13,195–15,694. The
first range contains 575 boundary events and 510 writes: 330 were
forced, 180 passed the score test, and 65 failed it. The final range
contains 573 boundary events and 528 writes: 332 were forced, 196
passed, and 45 failed. Mean scored margin changed from 0.0843 to 0.0895.
The gate therefore did not produce the measured storage-cost ramp:
boundaries changed by a factor of 0.997 and writes by 1.035.

The same run demonstrates where HNSW is already incorporated. Natural
and Durable share the existing SQLite-authoritative operation graph, and
`MemoryStorage` supersession uses the same persisted sparse route as
ordinary retrieval rather than a second durable write algorithm. During
finite route warm-up, sparse node rows per stored memory rose from 9.82
in the first range to 2,447.48 in the final range; exact current-row
comparisons reached the fixed *A* = 1, 248 target and mature routing
distance work reached the fixed 5*C* = 2, 495 envelope. Memory-storage
time per write consequently rose from 0.4833 ms to 2.8343 ms even though
write incidence stayed flat. The final seven end-anchored windows passed
process-half, throughput-half, relative-slope, height, operation,
work-counter, and continued-store-growth checks. Whole plateau
acceptance still failed because the strict absolute consolidation reset
classifier remained invalid; that raw failure is retained and unwaived.

A later correctness review found that the sparse current candidate set
had been treated as if it were an exact coverage certificate. That
shortcut could omit a true predecessor outside the activated subset and
suppress the historical fallback. The retained repair removes that
certificate: sparse current candidates remain bounded, but the
historical cache must provide its exact coverage/ranking proof or SQLite
fallback remains enabled. The focused large-population regression now
records 514 historical rows inspected rather than zero while retaining
the expected supersession edge. This restores supersession completeness
at the explicit cost of potentially history-sized write work; no revised
full replay or flat-write claim is made.

No new work constant was introduced. At the executed production knobs,
*C*/*B*/*A* = 499/125/1, 248, public routing is 5*C* = 2, 495,
construction remains *C* + *B* = 624 nodes and 2*B* = 250 queue effort,
and 126 is only the production logical *B* + 1 boundary. At neutral
`.5/.5/.5`, *B* = 128 and 129 remains only the logical, unprocessed
*B* + 1 probe. The existing 27-point structural grid and nine
production-shaped F/S/T corpus ablations remain the cutover requirement;
this single production-knob attribution neither replaces them nor claims
activated-identity overlap, bounded restart, Durable plateau, or overall
completion.

### Same-query recenter overlap

An experiment-only measurement now runs the same consolidation embedding
through the retained SQLite HNSW route immediately before and after each
successful recenter. A pair is valid only when both route queries
complete; the profile records a distinct pair-valid field and failure
code, and the deterministic builder rejects every invalid pair. Direct
consolidation-path regression covers both a valid equal-set result and
an invalid pre-query. Both searches retain the production-derived 5*C*
node ceiling and *A* = 2*C* + 2*B* activation target. Recenter timing
excludes both observer queries, the measurement is not part of the
accepted performance clock, and it does not change the production API or
schema.

In a regenerated 13,000-event Natural replay at *F*/*S*/*T* = .45/.5/.5,
22 recenters had explicitly valid, zero-failure paired measurements and
zero profiled pairs were invalid. Fourteen occurred after activation
reached the full *A* = 1, 248 target, and late pairs visited the
complete 5*C* = 2, 495 node envelope. Every pair nevertheless had
overlap 1.0: zero recenters changed even one activated identity for the
same embedding. Thus HNSW is already executed on the shared
Natural/Durable path, but moving the current connection-local entry is
not an effective consolidation-driven activation-centroid reset under
this route.

This is a rejected lifecycle behavior, not an accepted cycle. The next
candidate must change the sparse activated centroid while keeping the
result at *A*, query work at 5*C*, construction at *C* + *B* nodes and
2*B* effort, and all values derived from F/S/T. It must pass the 27
structural and nine exact production-shaped knob ablations and remain
modality/source-id agnostic. The raw process-reset gate remains
separate, failed, and unwaived. At neutral `.5/.5/.5`, *B* = 128; 129 is
only the logical unprocessed *B* + 1 boundary and is never a fetched row
or work budget.

### Saturated fixed-envelope quality boundary

The revised approximate-retrieval contract does not require an
impossible perfect score from every 512-query run. Each
production-shaped run may contain at most one exact top-1 miss
(511/512 = 0.998046875), the nine-run aggregate must remain at least
0.999, mean exact-identity recall@16 must remain at least 0.998,
semantic coverage must remain at least 0.95, deterministic tie order is
still exact, and misses may not cluster by modality, source identifier,
memory age, or knob setting. Structural bounds, rollback, provenance,
and ordering invariants remain 100% requirements.

A full-horizon saturation experiment then lowered only the private
routing-node envelope. At the production knobs
*F* = .45, *S* = *T* = .5, *C* = 499 and *B* = 125. The first point,
4*C* = 1, 996, saturated below the mature route population and met the
revised top-1 gate with 511/512 matches, but it was rejected because
identity recall@16 was 0.997791, below 0.998. Semantic coverage was
0.997571.

The next point added the already-derived reciprocal-update term
*R* = max (2, ⌊*B*/16⌋), yielding 4*C* + *R* = 2, 003. Matching both the
queue and actual-node ceilings at that value and retaining the separate
*A* = 2*C* + 2*B* = 1, 248 consolidation snapshot bounded total query
rows by 4*C* + *R* + *A* = 3, 251. Over 15,695 packets and 512 controls
it measured exact top-1 0.998047, identity recall@16 0.998650, semantic
coverage 0.998047, and deterministic tie order. At neutral knobs the
same formulas resolve to *C* = 512, *B* = 128, *R* = 8, a 2,056-row
route, and *A* = 1, 280; 129 remains only the logical, unprocessed
*B* + 1 boundary.

This is a quality-passing single production-knob experiment, not a
retained algorithm or a bounded-engine result. Its 15,695-packet run
took 206,060 ms, with mean process time 4.912 ms, mean end-to-end time
12.824 ms, and 33 consolidations. No audited suffix passed the whole
contract: emotional-cascade, commit, publication, and other operation
slopes remained; the raw post-consolidation process reset and
trough-stability gates remained failed; and the required nine-point
aggregate, mixed-modality/source regression, Durable run, and recursive
review have not been completed. The experiment therefore establishes
only that the 100% per-run top-1 requirement hid a viable saturated
quality point; it does not establish the storage-cost cure.

### Consolidation-reset routing-envelope experiments

The first dynamic experiment used the saturated quality point above as
its post-consolidation floor. Consolidation reset both queue effort and
actual-node ceiling to 4*C* + *R*, each retrieval-active query added
*R* = max (2, ⌊*B*/16⌋), and both values saturated at the retained 5*C*
ceiling. The separate activation snapshot remained bounded by
*A* = 2*C* + 2*B*, so total query-row work remained at most 5*C* + *A*.
At the production knobs this resolved to a 2,003-row floor, seven-row
steps, a 2,495-row peak, and a 3,743-row total ceiling; at neutral knobs
it resolves to 2,056, eight, 2,560, and 3,840 rows. No source-id or
modality condition participates in the formula.

The canonical 15,695-packet screen passed its 512-query quality control
with top-1 1.0, recall@16 0.998650, semantic coverage 0.998666, and
deterministic ties. It completed 34 consolidations in 204,822 ms.
However, the corpus ended while some route peaks were still warming, so
a second diagnostic appended one complete later owner-authorized Claude
session using the identical extraction rules. The combined input
contained 30,380 real packets, with no trimming, cycling, padding, or
synthesis. This diagnostic completed 72 consolidations in 438,059 ms.
Mature retrieval cycles reset on 30/30 evaluated edges and their peak
and trough half ratios were 0.9972 and 1.0003, but the normalized p95
shape error was 0.3074. More importantly, quality fell to 507/512 top-1
(0.990234), recall@16 0.977225, and semantic coverage 0.971084. All five
top-1 misses appeared in the late-history query quartile. The candidate
therefore fails the numeric and anti-clustering gates and is rejected;
the canonical corpus remains the acceptance corpus, and this
owner-authorized extension is only a maturity diagnostic.

The smallest follow-up was preregistered before measurement. It retained
the proven 5*C* canonical floor, reset to that floor after
consolidation, advanced queue effort and node ceiling by *R* per
retrieval-active query, and saturated at 6*C*. Total query-row work was
bounded by 6*C* + *A*. Production values were a 2,495-row floor,
seven-row step, 2,994-row peak, and 4,242-row total ceiling; neutral
values are 2,560, eight, 3,072, and 4,352 rows. The focused derivation
and reset regression passed 16 assertions, and all 116 storage-audit
fixtures passed after teaching the audit the new formula.

The same 30,380-packet diagnostic completed 65 consolidations in 388,336
ms, with 5.679 ms mean process time and 12.473 ms mean end-to-end time.
Exact top-1 was 512/512 and semantic coverage was 0.989234, but exact
identity recall@16 was only 0.992651. That is materially below the
preregistered 0.998 floor, so the candidate is rejected despite perfect
top-1. Its mature retrieval-cycle subcontract did pass: 24/24 cycles
reset, peak and trough half ratios were 1.0000 and 0.9985, normalized
p95 shape error was 0.1309, late-template error was 0.0137, and no
routing, queue, snapshot, combined-row, distance-coverage, or activation
bound was violated. Its shorter wall time than the lower-floor run is
not attributed to cheaper retrieval: the live time-based detector
requested seven fewer consolidations, and the run was not a
fixed-schedule throughput comparison. Neither dynamic envelope is
retained as a production algorithm, and neither result weakens the
quality thresholds. The sanitized decision record is
`artifacts/flat_storage_cost/dynamic-retrieval-sawtooth-maturity-v1.json`.

Because the 5C-to-6C candidate passed the complete mature-cycle shape
contract but failed only exact-neighbor recall, the next and only
preregistered routing point raises the floor by one *C* while preserving
the same one-*C* amplitude. It resets at 6*C*, advances by
*R* = max (2, ⌊*B*/16⌋), and saturates at 7*C*; the separate snapshot
remains *A*, so total query-row work is at most 7*C* + *A*. Production
values are a 2,994-row floor, seven-row step, 3,493-row peak, and
4,741-row total ceiling. Neutral values are 3,072, eight, 3,584, and
4,864 rows. The 30,380-packet result retained 511/512 top-1, semantic
coverage 0.993569, and deterministic ties, but recall@16 was 0.995833
and therefore failed the unchanged 0.998 gate. Its mature cycle passed
with 23/23 resets, peak/trough half ratios 1.0000/1.0003, p95 shape
error 0.0222, late-template error 0.0035, and no work-bound violation.
It is rejected.

The next floor ablation is preregistered at 7*C* to 8*C* with the same
*R* step and separate *A* snapshot. Production values are 3,493 floor,
seven step, 3,992 peak, and 5,240 total rows; neutral values are 3,584,
eight, 4,096, and 5,376. The maturity run achieved 511/512 top-1,
recall@16 0.997672, semantic coverage 0.996856, deterministic ties,
22/22 resets, p95 shape error 0.0230, and no work-bound violations.
Recall was 0.000328 below the unchanged gate, so the candidate is
rejected.

The final adjacent floor point is preregistered at 8*C* to 9*C*, again
stepped by *R* with the separate *A* snapshot. Production values are
3,992 floor, seven step, 4,491 peak, and 5,739 total rows; neutral
values are 4,096, eight, 4,608, and 5,888. The maturity run passed:
top-1 was 511/512, recall@16 was 0.999265, semantic coverage was
0.998571, ties were deterministic, and four opaque sources were
represented. In the suffix beginning at event 18,380, 19/20 material
cycles reset; peak and trough half ratios were 0.9991 and 0.9994, p95
shape error was 0.1908, late-template error was 0.0795, and no routing,
queue, snapshot, combined-row, distance-coverage, or activation bound
was violated. The canonical 15,695-packet acceptance run then passed
quality with 512/512 top-1, recall@16 1.0, semantic coverage 1.0,
deterministic ties, and four opaque sources. It completed 34
consolidations in 214,263 ms, with 5.233 ms mean process time and 13.334
ms mean end-to-end time. This shorter corpus never reached the derived
3,992-row floor: maximum actual retrieval work was 3,064 rows, so it
contained no mature post-floor cycles and cannot independently prove the
sawtooth shape. Together the runs establish canonical quality plus
extended-history retrieval maturity, not a whole-engine cure. The
canonical audit still failed the whole-engine plateau: the largest
first-to-last-five timing increases were `GraphRetrieve.seed_knn_cache`
(1.217 to 2.496 ms) and supersession current/candidate loading (about
0.48 to 1.40 ms). At that point the nine-point aggregate, Durable path,
hotspot attribution, and production cutover were still pending; the next
subsection records their resolution and remaining limits.

### Production-default 8C-to-9C cutover

The accepted adjacent envelope was moved into the existing SQLite HNSW
route as the production default rather than added as another retrieval
or write path. With hooks disabled, the route now resets queue effort
and actual-node ceiling to 8*C* after a successful consolidation
recenter, advances both by *R* = max (2, ⌊*B*/16⌋) per retrieval-active
query, and saturates at 9*C*. The independent consolidation snapshot
remains bounded by *A* = 2*C* + 2*B*, so total fetched rows remain at
most 9*C* + *A*. Restart opens at the 9*C* ceiling because the
in-process ramp position is not durable. Historical private selectors
remain only as reproducibility controls; a regression requires the
former selector-24 formula to resolve identically to the production
default. No source-id or modality label participates in any formula, and
Natural and Durable still share the operation graph; Durable adds only
its post-commit checkpoint.

A round-two rollback review found that the internal search used to build
a recenter snapshot could transiently advance the ramp before
persistence. The route now excludes recenter-building searches from the
*R* increment. A focused failure-injection regression aborts the
metadata update and proves that both effort and node budget remain
unchanged; successful recenter is still the only path that resets them
to 8*C*.

The no-selector 15,695-packet Natural replay at
*F* = .45, *S* = *T* = .5 resolved *C* = 499, *B* = 125, *A* = 1, 248,
*R* = 7, a 3,992-row floor, 4,491-row peak, and 5,739-row combined
ceiling. It completed 34 consolidations in 203,522 ms, with 4.921 ms
mean process time and 12.659 ms mean end-to-end time. Its 512 public
controls passed at 512/512 exact top-1, recall@16 1.0, semantic coverage
1.0, deterministic ties, and four opaque source identifiers. Timing
cannot be compared causally with the earlier selector run because
timestamps, live consolidation incidence, and build identity differ.

The production-default nine-point matrix replayed 4,000 packets
sequentially at midpoint, both joint endpoints, and each one-axis
endpoint. All 4,608 of 4,608 controls matched exact top-1; minimum
recall@16 and semantic coverage were both 1.0; tie order was
deterministic; and there were no misses to cluster by modality, opaque
source, memory age, or knob point. The focused 27-point structural grid
and bounded-backfill, metadata-open, core-knob, and text/audio/image
plus shared/opaque-source invariance regressions also passed. All nine
short routes were exercised, but none produced a valid mature
pre/post-activation pair: their active graphs remained below the 8C
floor. The aggregate therefore classifies them as quality-passing but
recenter-unevaluated rather than laundering warm-up into sawtooth
evidence. The 30,380-packet run above remains the mature-cycle proof.
Sensitivity-high took 140,392 ms versus 37,522–67,511 ms for the other
eight points and is retained as an adverse measured result, not hidden
by the perfect quality aggregate.

The mature 8C-to-9C profile was also split into two non-overlapping late
halves. While the persistent store continued to grow, mean process time
fell from 8.918 to 7.858 ms. `GraphRetrieve.seed_knn_cache` fell from
2.914 to 2.397 ms and its per-node unit ratio was 0.789; supersession
loading fell from 1.860 to 1.520 ms and its per-candidate unit ratio was
0.855. Emotional work rose slightly per edge, but its total fell from
0.486 to 0.430 ms. This is evidence that mature per-store work is not
continuing to grow under the bounded route. It does not convert the
still-failing raw process-time reset diagnostic into a pass:
consolidation changes retrieval locality and its work envelope, but it
cannot force every following event to perform the same amount of
content-dependent write work.

Finally, the real 2,016-message Durable corpus used the same no-selector
parameters and completed four consolidations in 79,481 ms, with 10.710
ms mean process time and 39.024 ms mean end-to-end time. Its 512
controls again passed at 512/512 top-1, recall@16 1.0, semantic coverage
1.0, deterministic ties, and four opaque sources. The corpus cannot
supply the required ten post-warmup 500-event windows, so this is
shared-path, checkpoint, and quality evidence; it is not a Durable
plateau claim. The production cutover is therefore scoped to the
knob-derived retrieval envelope and shared operation path, not to
bounded whole-engine restart, production-wide latency, release, or
deployment.

## Standard Eviction-Frontier Estimate (Historical)

Strength-based eviction and its storage gate were removed in the 2026-10
engine ablation
(<a href="#sec-reinforcement" class="quarto-xref">Section 13</a>); this
estimate is kept as history. The standard file-backed pressure gate is
500,000,000 bytes. Exact databases occupied 10,014,720 bytes at 500
messages, 18,169,856 bytes at 1,000, and 48,857,088 bytes at 2,636. The
full-span storage slope projects the gate at approximately 27,445
messages; the observed interval slopes give a planning range of roughly
26,700–30,500 messages.

Projecting Candidate 34’s corrected post-warmup trends to 27,445
messages gives 89.8 ms of engine process time (11.1 writes/s) and 103.7
ms sequential end-to-end latency (9.6 messages/s). This is a
pre-frontier estimate extrapolated about 10.4 times beyond the measured
corpus, not a direct run or post-eviction claim. Storage pressure only
permits eviction: a memory must also satisfy the consolidation-age and
strength predicates. The measured corpus has `last_consolidation_ts=0`,
so it would not evict at the pressure frontier as-is.

## Verification Gates

The optimization and cutover work is gated by:

``` bash
git diff --check

cmake -S . -B build/codex-hardcut-check \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build build/codex-hardcut-check -j

./build/codex-hardcut-check/tests/cortext_tests '~[cortext]~[aist]'

./build/codex-hardcut-check/tests/cortext_tests \
  '[operations][graph][retrieval],[operations][constructive_recall][graph],[state_persistence][working_memory][decay]'
```

Those hard-cutover commands historically passed **429 test cases** and
**2,489 assertions** in the broad filter and **5 test cases** and **16
assertions** in the targeted graph/reconstruction/state filter. The July
12 safety tree passed the complete default native suite with **509 test
cases** and **4,510 assertions**, plus Python validation, Go tests and
vet, Dart analysis and tests, the Node addon build, and the WebAssembly
bundle build. Candidate 34 separately completed a warning-clean native
build, passed the complete registered CTest suite (1/1), matched the
retained exact behavior and logical-database digests at 500 and 2,636
events, and matched the established 1,000-event digests under the
durable SQLite profile. After integration onto the current
runtime-safety branch, the combined tree passed the complete registered
CTest suite (1/1) and the focused rollback, cache, state-persistence,
memory-storage, graph-retrieval, graph-cache, and reconsolidation
groups. A fresh 500-event comparison against the exact pre-integration
branch head produced the same event-behavior digest, the same canonical
logical-database digest, and the same 9,091/521/521
association/memory/signal counts. Mean process latency was 9.492 ms on
the combined tree versus 47.846 ms on the control, and mean sequential
end-to-end latency was 17.297 versus 55.919 ms. The 500-event comparison
is an integration equivalence and bounded performance check, not a
flatness result; the full audit above remains the source of truth for
non-flat storage cost and throughput. The merged rollback path also has
an injected failed-`Flush()` regression: it asserts that working-memory
dirty state and the private historical-search registry are restored,
then performs a normal write before any successful flush to prove that
the exact accelerator remains available during the previously untested
post-failure interval.

## Remaining Bottlenecks

After the hard cutover, the expected bottlenecks are ordinary systems
costs:

-   SQLite row materialization and transaction bookkeeping;
-   vector blob decoding and copies;
-   sqlite-vec index work over the retained embedding surface;
-   object-store writes during durable memory creation; and
-   hydration when applications request full memory payloads.

These are the right bottlenecks for the product target. They are local,
profileable, and bounded by database size or selected context size. They
do not depend on external decoder latency, prompt shape, batching
behavior, or model server availability.

## Future Measurement

Future long-horizon quality and latency runs should treat this
repository as the production engine under test and keep any evaluator
outside the repository. Recommended reported metrics are:

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr>
<th>Metric</th>
<th>Reason</th>
</tr>
</thead>
<tbody>
<tr>
<td>mean and p95 process latency by corpus position</td>
<td>detects scaling leakage</td>
</tr>
<tr>
<td>retrieval candidate count and selected count</td>
<td>verifies bounded retrieval</td>
</tr>
<tr>
<td>reconstruction versions per memory</td>
<td>verifies bounded constructive recall</td>
</tr>
<tr>
<td>SQLite page count and WAL size</td>
<td>separates algorithm latency from storage pressure</td>
</tr>
<tr>
<td>context token count after hydration</td>
<td>tracks application-facing packet cost</td>
</tr>
<tr>
<td>frozen-probe recall or human adjudication</td>
<td>measures quality without adding evaluator code to the engine</td>
</tr>
</tbody>
</table>

The hard-cut branch is therefore suitable for continued product-quality
experiments, but the measured durable-ingestion path is explicitly
non-flat. Further asymptotic gains require an approved semantics,
lifecycle, backend, or state-ownership change rather than another
unproven narrow hot-loop rewrite or the restoration of deleted decoder
features.
