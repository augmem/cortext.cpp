# Cortext: A Three-Knob Adaptive Memory Architecture
Gabriel Willen
2026-07-01

# Abstract

We present Cortext, a biologically-inspired adaptive memory system
governed by three continuous control parameters: Focus (F), Sensitivity
(S), and Stability (T). Unlike traditional memory architectures that
employ discrete operational modes, Cortext achieves developmental
progression through parameter-derived rate modulation, allowing behavior
to emerge continuously from the interaction of knob settings and
experiential mass. The architecture integrates established cognitive
science principles—including Cowan’s working memory constraints and
Nader’s reconsolidation dynamics—into a unified computational framework.
We derive most system parameters from the three primary knobs through
principled mathematical transformations, while explicitly labeling the
small set of fixed invariants (e.g., controller gains), reducing
reliance on hard-coded constants. The production system is an
embedding-first graph memory engine: it does not ship an internal
decoder path, adapter registry, fact table, static taxonomy loader, or
processor-local shadow STM graph. The system demonstrates
self-calibrating priors that blend with evidence using
uncertainty-weighted Bayesian averaging, homeostatic threshold control
with effective sample size estimation, and graph-augmented retrieval
combining embedding similarity with durable association expansion.
Experimental analysis indicates the architecture maintains stable
operation across developmental phases while adapting write rates, decay
dynamics, and retrieval precision to environmental demands. This work
contributes a formally specified cognitive memory model suitable for
implementation in streaming AI systems requiring persistent,
context-aware memory.

**Keywords:** cognitive architecture, adaptive memory, working memory,
episodic memory, semantic memory, soft anchors, knowledge graphs,
homeostatic control

# Introduction

Memory systems in artificial intelligence face a fundamental tension
between plasticity and stability. Systems that learn rapidly risk
catastrophic interference, while those that maintain stable
representations may fail to capture novel patterns (McCloskey and Cohen
1989). Biological memory systems resolve this tension through
sophisticated regulatory mechanisms that modulate learning rates, decay
dynamics, and retrieval thresholds in response to environmental demands
and internal state (McClelland, McNaughton, and O’Reilly 1995).

This paper introduces Cortext, a cognitive memory architecture that
addresses this stability-plasticity dilemma through three continuous
control parameters that govern all system behavior. Rather than
implementing discrete operational modes or hard-coded phase transitions,
Cortext achieves developmental progression through the continuous
interaction of parameter settings with accumulated experience. The
architecture draws on established findings from cognitive psychology and
neuroscience, including working memory capacity limits (Cowan 2001),
memory reconsolidation (Nader, Schafe, and Le Doux 2000), serial
position effects (Murdock Jr 1962), and emotional modulation of memory
(McGaugh 2004).

The core contribution of this work is a formally specified memory
architecture in which:

1.  Most tuneable parameters derive from three primary knobs (Focus,
    Sensitivity, Stability) through explicit mathematical
    transformations, while fixed invariants (e.g., controller gains) are
    labeled and justified.
2.  System priors self-calibrate through uncertainty-weighted Bayesian
    blending with observed evidence.
3.  Developmental phases emerge from annealed safety bounds and
    experiential mass accumulation, not explicit mode switching.
4.  A knowledge graph layer enables shallow embedding consolidation and
    graph-augmented retrieval.
5.  Ingress-time Soft Anchor state preserves uncertain subject/entity
    continuity without forcing immediate hard binding.

The remainder of this paper is organized as follows. Section 2 reviews
relevant literature. Section 3 presents the mathematical foundations
including notation, helper functions, and knob-derived parameters.
Section 4 details the core algorithms for Focus, Sensitivity, and
Stability adaptation. Section 5 describes structural metrics and
composite scoring. Section 6 covers dynamic thresholding and homeostatic
control. Section 7 presents the reinforcement and decay dynamics.
Section 8 describes advanced cognitive processes including working
memory, Soft Anchor formation, pattern-separation/procedural sparse
keys, synaptic tagging, and emotional consolidation. Section 9 details
the consolidation and graph integration system. Section 10 presents the
interrupt gate for streaming integration. Section 11 reports
experimental results. Section 12 discusses implementation considerations
and computational complexity. Section 13 documents the performance
profiling and optimization of the reference implementation. Section 14
concludes with limitations and future directions. In the source tree
these correspond to `sections/1_...` through `sections/11_...`; the
rendered manuscript section numbering includes the introduction and
related-work sections before those files.

# Related Work

## Working Memory Models

The working memory component of Cortext draws conceptually on Cowan
(2001) embedded-processes model, which posits a capacity limit of
approximately 4±1 chunks for the focus of attention. This contrasts with
Miller (1956) earlier estimate of 7±2 items, which subsequent research
has shown conflates chunking with raw capacity (Cowan 2010). Cortext’s
production slots are larger, application-level memory units rather than
a direct cognitive-chunk implementation: the shipped F/S-derived
capacity is 15–27 slots, with 21 at neutral knobs, selected from the
product evaluation reported in
<a href="#sec-experimental" class="quarto-xref">Section 11</a>. The
biological result motivates boundedness and competition, not a claim
that the software reproduces human span numerically.

Baddeley (2000) multicomponent model informs our treatment of
maintenance and rehearsal processes, though we adopt a more unified
representational substrate based on distributed embeddings rather than
separate phonological and visuospatial stores. The episodic buffer
concept (Baddeley 2000) aligns with our approach to binding information
across modalities through shared vector spaces.

## Memory Consolidation

The consolidation mechanisms in Cortext reflect findings from the memory
reconsolidation literature (Nader, Schafe, and Le Doux 2000; Nader
2003). Reconsolidation theory posits that retrieved memories enter a
labile state during which they can be modified before restabilization.
Our architecture implements this through time-bounded lability windows
governed by the Stability parameter, with reconsolidation gain modulated
by both Sensitivity and contextual relevance.

The distinction between episodic and semantic memory (Tulving 1972)
motivates our two-tier storage approach: a streaming episodic buffer for
immediate experiences and a durable graph over consolidated embedding
structure. The current consolidation process transforms high-redundancy
episodic clusters into association centroid nodes linked to their
sources by durable graph edges; it does not synthesize payload summaries
or extract free-form semantic relations.

## Emotional Influences on Memory

McGaugh (2004) extensive work on emotional modulation of memory
consolidation informs our treatment of affect-gated encoding. The
architecture implements emotional intensity as a threshold modifier,
consistent with findings that arousal enhances memory through
amygdala-mediated modulation of hippocampal encoding (LaBar and Cabeza
2006). We adopt a dimensional model of emotion (Russell 1980) with
valence and arousal as primary axes, projected from categorical emotion
embeddings.

## Adaptive Control Systems

The homeostatic threshold controller draws on classical control theory,
specifically proportional-integral approaches to setpoint maintenance
(Åström and Murray 2008). The use of exponentially-weighted moving
averages for rate estimation follows standard practice in adaptive
systems, while our effective sample size calculation for reliability
estimation extends techniques from sequential Monte Carlo methods (Liu
and Chen 1998).

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
last_signal_ts, last_write_ts, drift_acc, eta_acc, coherence_prev,
emo_max, arousal_sum, drift_accum, drift_at_last_interrupt,
drift_acc_pacing, x_last_check. Global variables: signals_processed,
u_uncertainty, mood_vector, last_mood_ts, theta_dynamic, theta_target,
hysteresis, m_rate, rho_hat_prev, dt_ema, rate_ticks, reliability,
last_rate_timestamp, last_retrieval_ts, retention_ema,
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
(e.g., score_t, theta_dynamic_t). Retained state uses the bare name
(e.g., theta_dynamic). When a step proposes a new value, compute the
`_t` variant and assign the bare name at end of step (see Appendix D).

**Time deltas:** All Δt values, elapsed times (mem_elapsed, signal_gap,
idle_for), and time comparisons operate in seconds:

    Δt ← now_s() − to_s(last_rate_timestamp)   # seconds
    mem_elapsed ← now_s() − to_s(t_start)     # seconds
    signal_gap ← now_s() − to_s(last_signal_ts)  # seconds

**Time constants:** All time-related constants are specified with
explicit units (e.g., τ_min(T) in seconds). Gap scaling returns values
in seconds (gap_scale multiplies dt_ema to form gap_ref_s).

### No Fixed Behavioral Constants

Most behavioral thresholds, gains, and caps are derived from the three
knobs (F, S, T) and/or state. Fixed numeric values are permitted for
numerical stability floors (e.g., ε(F, S, T)), unit conversions, and
explicitly labeled resource/safety invariants such as storage-pressure
floors. This keeps the system’s qualitative behavior governed by the
three knobs while still allowing bounded implementation guards.

## Temporal Semantics

Cortext treats durable memories as **event-time observations**: a memory
is written when a coherent unit is detected, and its stored timestamp
records when that unit was observed. The engine no longer maintains a
separate structured fact layer with valid-time and transaction-time
intervals; temporal reasoning therefore operates over memory and signal
timestamps directly.

For an episodic memory `m_i`, the event timestamp is:

    t_event(m_i) = created_at(m_i)

Retrieval and consolidation consume this event time through recency
windows, association ordering, and explicit replay timestamps. The
accumulator’s temporal-context vector is used separately for
boundary/topic-shift state and is persisted with memories; as noted in
<a href="#sec-temporal-context" class="quarto-xref">Section 8.2</a>, it
is not yet a production retrieval-score term.

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

Cortext is governed by three continuous control parameters, each
representing a distinct dimension of cognitive regulation:

**Focus (F ∈ \[0, 1\]):** Perceptual selectivity and precision. Higher
Focus narrows attention, increases relevance weighting, and reduces
retrieval breadth. Focus modulates the trade-off between exploitation of
known-relevant information and exploration of potentially useful
context.

**Sensitivity (S ∈ \[0, 1\]):** Plasticity and affective gain. Higher
Sensitivity accelerates learning, increases emotional and novelty
responsiveness, and raises write-rate targets. Sensitivity governs how
readily the system captures novel information and responds to salient
stimuli.

**Stability (T ∈ \[0, 1\]):** Temporal persistence and inertia. Higher
Stability lengthens memory half-lives, widens hysteresis bands, slows
adaptive updates, and tightens safety bounds over time. Stability
controls the resistance to change and the preservation of established
knowledge.

A central design principle is that knobs set rates rather than modes.
Behavioral differences emerge continuously from parameter interactions;
there are no hard-coded phase transitions or discrete operational
states.

### Midpoint-Biased Knob Mapping

To make **F = S = T = 0.5** a neutral, all‑arounder operating point, we
apply a midpoint bias to Focus and Sensitivity before deriving
parameters. Endpoints remain fixed:

    F̃ = clamp(F − β_F × 4F(1 − F), 0, 1)
    S̃ = clamp(S − β_S × 4S(1 − S), 0, 1)
    β_F = β_S = 0.10

Unless explicitly stated otherwise, knob‑derived formulas in the paper
use **F̃** and **S̃** in place of the raw UI knobs. Stability (T) is not
biased. A small number of legacy/cold-start implementation priors still
consume raw UI knobs directly and are called out where they matter.

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
lookback for variance estimation and percentile calculation. The rate
window win_rate_s specifies the temporal horizon (in seconds) for
write-rate measurement.

### Half-Life and Decay

Memory half-life follows a log-scale mapping to span multiple orders of
magnitude:

    τ_min = 120.0 seconds (2 minutes)
    τ_max = 43200.0 seconds (12 hours)
    base_half_life(T) = exp(ln(τ_min) + T × ln(τ_max / τ_min))

This exponential mapping ensures that low Stability yields half-lives
near 2 minutes while high Stability approaches 12 hours, with smooth
interpolation across the range.

### Hysteresis and Rate Targets

    band_min = 0.02; band_max = 0.25
    base_band(T) = lerp(band_min, band_max, T)
    r_min = 0.2; r_max = 5.0  (writes per minute)
    base_rate(S) = lerp(r_min, r_max, S)

The hysteresis band prevents oscillation in threshold-crossing
decisions. Write-rate targets establish homeostatic setpoints for the
threshold controller.

These ranges are canonical knob maps for memory behavior: admission
thresholds, retrieval weights, graph expansion breadth, lifecycle
scoring, and consolidation policy are derived from F/S/T. Fixed values
may still appear as schema invariants, vector dimensions, parser
guardrails, numeric epsilons, migration defaults, and benchmark protocol
settings, but they are not additional user-facing memory knobs.

### Retrieval Breadth

Retrieval uses two F/S/T-derived breadths rather than one global top-k.
The candidate discovery pass stays broad so durable memories remain
reachable:

    seed_k(F) = round(lerp(96, 8, FocusBias(F)))

Before the vector-seed cap is applied, superseded targets and
near-duplicate embedding families are collapsed to one current
representative. Family lookup normalizes each embedding and partitions
its coordinates into 32 deterministic blocks. For a candidate `x` and
representative `y`, Cauchy–Schwarz gives the recall-safe upper bound

    cos(x, y) <= sum_b ||x_b||_2 ||y_b||_2

When that bound is below the cosine family threshold, the pair cannot be
a duplicate and the full dot product is skipped. All remaining pairs
receive an exact cosine check, so the screen cannot introduce a
false-negative partition while avoiding broad exact comparisons on
diverse vectors. The prompt-facing output is then narrowed after vector
relevance and bounded graph expansion:

    selected_k(F,T) = round(lerp(20, 8, FocusBias(F)) * lerp(1.08, 0.92, T))

Vector relevance is the primary rank. Event-time recency is restricted
to a bounded tie-break so an old semantic match cannot be displaced by a
recent but unrelated memory. Graph evidence contributes only a bounded
residual bonus:

    semantic(m) = map01(cos(q, embedding(m)))
    base(m) = semantic(m) + (1 - semantic(m)) * 10^-6 * temporal_rank(m)
    score(m) = base(m) + (1 - base(m)) * min(0.08, w_graph * graph(m))

New source-backed memories also receive a knob-derived source
reliability prior instead of relying on a fixed storage default:

    source_reliability_0(F,S,T)
      = clamp(0.70 + 0.06(F' - F'_0.5)
                   - 0.05(S' - S'_0.5)
                   + 0.08(T - 0.5), 0.50, 0.90)

This split lets low Focus explore many durable-memory seeds while
keeping the application prompt bounded by a compact final surface. The
intermediate graph breadth and score weights are likewise derived from
Focus, Sensitivity, and Stability in the implementation rather than
exposed as additional user settings. Neither `source_id` nor modality
participates in candidate discovery, family collapse, scoring, graph
expansion, or final selection. The stored `source_reliability` and
`source_contradiction_count` fields are not currently consumed by
production ranking;
<a href="#sec-advanced" class="quarto-xref">Section 8</a> distinguishes
that metadata from the proposed confidence gate.

### Experiential Mass and Maturity

The system tracks accumulated experience through a maturity function
that governs the annealing of safety bounds:

    τ_m(T) = lerp(10.0, 200.0, T)
    maturity(t) = 1 − exp(−count / τ_m(T))

where count is the total number of signals processed. This produces
asymptotic approach to unit maturity, with higher Stability slowing the
progression to reflect greater conservatism.

Safety bounds on the dynamic threshold anneal with maturity:

    T_min(t) = lerp(0.01, 0.05, maturity(t))
    T_max(t) = lerp(0.99, 0.95, maturity(t))
    max_ΔT_per_min(t) = lerp(0.30, 0.10, maturity(t))

Early operation permits wide threshold excursions; mature operation
constrains movement to a narrower band.

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
<a href="#sec-focus-spread" class="quarto-xref">Section 5.1.2</a> and is
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

# Core Adaptation Algorithms

This section presents the algorithms governing adaptation along each of
the three primary dimensions. Each algorithm consists of a prior
computation (executed at initialization) and a dynamic update (executed
per signal).

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

    rate_target = base_rate(S)  # writes/min
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
        p_c ← uniform(1/6)  # ensures downstream mood update is well-defined
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

### Threshold Modulation from Emotion

Emotional activation loosens write thresholds to capture salient
moments:

    κ_emo ← κ_base × S  # where κ_base = 0.10
    ΔThreshold_emotion_t ← −κ_emo × emotion_intensity_t ×
                            (0.5 + 0.5 × arousal_t)

The emotional state acts as a modulator for memory
encoding/consolidation, following McGaugh (2004). High emotion intensity
and arousal increase the likelihood of threshold crossings; valence is
recorded as a separate metric but does not enter this threshold delta
directly.

### Mood Integration

Distinct from instantaneous emotion, the mood state M_t ∈ ℝ⁶ maintains a
persistent background affective tone as a 6-dimensional vector (one
component per emotion category from <a href="#sec-emotional-projection"
class="quarto-xref">Section 4.2.2</a>):

    α_mood(S) = lerp(0.01, 0.20, S)  # reactivity
    half_life_mood(T) = lerp(30, 600, T)  # seconds
    Δt_mood ← now_s() − to_s(last_mood_ts)
    λ_mood(Δt_mood, T) ← exp(−ln(2) × Δt_mood / max(half_life_mood(T), ε))
    e_t ← p_c − (1/6)  # centered 6D vector (can be negative)
    M_t = λ_mood(Δt_mood, T) × M_{t−1} + α_mood(S) × e_t
    M_t ← clamp_elementwise(M_t, −1.0, 1.0)  # per-component
    last_mood_ts ← now_ms()  # update timestamp after mood update (ms)

Note: when all raw_cos_c ≤ 0, p_c is uniform and e_t = 0, so mood only
decays. Because e_t is centered around zero, M_t can have both positive
and negative components, reflecting sustained elevation or suppression
relative to baseline. The mood state provides a separate threshold bias
via its normalized magnitude:

    κ_mood ← κ_base × S
    m_norm ← ‖M_t‖ / √6  # max norm when all components at 1
    ΔThreshold_mood_t ← −κ_mood × clamp(m_norm, 0, 1)

## Stability-Driven Persistence

Stability governs temporal dynamics through half-life, decay rates, and
hysteresis.

### Stability Priors

Given Stability knob T ∈ \[0, 1\], initialize Stability control
variables:

    hysteresis = lerp(0.02, 0.25, T)
    half_life = base_half_life(T)
    rate_decay = lerp(0.60, 0.98, T)
    periphery_half_life = clamp(0.5 × half_life,
                                       τ_min, τ_max)
    drift_weight = 0.5 × (1 − T)

### Dynamic Stability Update

The stability update uses uncertainty-modulated learning rate and a
stability-derived retention window:

    α_min_T = 0.02; α_span_T = 0.18
    α_T(t) = α_min_T + (1 − T) × α_span_T × u(t)
    win_ret(T) = round(lerp(10, 50, T))  # retention history window size

At each signal event, compute retention statistics and adjust half-life:

    active_memories ← {m | strength(m) ≥ periphery_cutoff(T)}
    observed_retention ← mean_age(active_memories)
    retention_ema_t ← EWMA(retention_ema_{t−1},
                           observed_retention, α = α_T(t))

    retention_ema ← retention_ema_t

Definitions:

    age_s(m) ← now_s() − to_s(m.created_at)  # fallback: use start_ts if created_at unset
    mean_age(active_memories) ← mean_{m ∈ active_memories} age_s(m); if empty, return 0
    retention_history ← append(observed_retention); keep last win_ret(T) values

Compute z-score relative to recent retention history:

    last_win_ret ← tail(retention_history, win_ret(T))
    μ_ret ← mean(last_win_ret)
    σ_ret ← max(std(last_win_ret), 1.0)
    zscore_ret ← clamp((observed_retention − μ_ret) / σ_ret, −3, +3)

The target half-life incorporates feedback adjustment from the stability
feedback mechanism
(<a href="#sec-feedback" class="quarto-xref">Section 7.5</a>):

    stability_adj ← ΔHalfLife_adj_t if provided else 0
    target_half_life_t ← clamp(base_half_life(T) ×
                              (1 + 0.25 × zscore_ret + stability_adj),
                              τ_min, τ_max)
    half_life_t ← EWMA(half_life_{t−1}, target_half_life_t,
                       α = α_T(t))

    half_life ← half_life_t

## Neuromodulator Layer (Internal Mapping)

To keep the three-knob UI while enabling more biologically faithful
control, Cortext maps the knobs and internal signals to continuous
neuromodulator-like variables. These are **not** additional user knobs;
they are latent control signals derived from F/S/T and real-time stream
statistics.

We define three modulators:

-   **ACh-like (acetylcholine):** encode vs retrieve bias.
-   **NE-like (norepinephrine):** arousal + interrupt urgency.
-   **DA-like (dopamine):** reward prediction error + value learning.

Baseline levels come from knobs; phasic bursts come from current
signals:

    ACh_base(F,S,T) = clamp(0.15 + 0.55S + 0.25(1 − T) − 0.15F, 0, 1)
    NE_base(F,S,T)  = clamp(0.10 + 0.60S + 0.20(1 − T), 0, 1)
    DA_base(F,S,T)  = clamp(0.10 + 0.40F + 0.30T, 0, 1)

    ACh_t ← clamp(ACh_base + 0.35 × novelty_t − 0.20 × retrieval_pressure, 0, 1)
    NE_t  ← clamp(NE_base  + 0.50 × surprisal_t + 0.30 × arousal_t, 0, 1)
    DA_t  ← clamp(DA_base  + max(0, δ_reward_t), 0, 1)

Where `retrieval_pressure` is the normalized retrieval queue-depth
input, and `δ_reward_t` is a reward prediction error derived from
downstream outcome signals (see
<a href="#sec-structural-metrics" class="quarto-xref">Section 5</a>).
The production operation context does not currently populate queue
depth, so this term is zero in the shipped full pipeline; tests and
experiment hooks can supply it. The other phasic terms are live. These
modulators drive internal gates without introducing new user-facing
parameters:

    encode_bias ← ACh_t × (0.7 + 0.3S)
    retrieve_bias ← 1 − encode_bias
    write_threshold_scale ← 1 − 0.3 × NE_t
    reconsolidation_scale ← 1 + 0.4 × ACh_t
    retrieval_competition_scale ← 1 + 0.5 × NE_t
    value_update_gain ← 0.5 + 0.5 × DA_t

In the current implementation, these latent scales are applied at four
concrete downstream sites rather than introduced as separate runtime
controls:

-   `write_threshold_scale` multiplies the accumulator write-gate
    threshold, so high `NE_t` lowers the barrier for committing a
    boundary candidate.
-   `reconsolidation_scale` multiplies reconsolidation drift magnitude,
    so high `ACh_t` strengthens context-driven reconstruction updates.
-   `retrieval_competition_scale` multiplies lateral inhibition during
    retrieval competition, so high `NE_t` sharpens suppression among
    near-duplicate candidates.
-   `value_update_gain` scales procedural-store reward updates, so high
    `DA_t` increases reinforcement of recently used routines.

## Oscillatory Gating (No Discrete Modes)

To reduce interference without introducing hard modes, Cortext adds a
continuous oscillatory gate that gently alternates encode- vs
retrieve-bias over time.

    ω(F,S,T) = lerp(0.03, 0.12, S) × lerp(1.2, 0.8, T)  # rad/s
    φ_t ← (φ_{t−1} + ω × Δt) mod 2π
    osc_t ← 0.5 + 0.5 × sin(φ_t)

    encode_bias_t ← encode_bias × (0.6 + 0.4 × osc_t)
    retrieve_bias_t ← 1 − encode_bias_t

The oscillator is continuous and **never** switches modes; it merely
biases thresholds and competition in a time-varying, biologically
plausible manner.

## Meta-Learning Knob-to-Parameter Maps

All knob-derived ranges and coefficient tables (e.g., bootstrap
coefficients for RLS, lerp endpoints, base rates) are treated as
**species priors**. They remain the **only** user-visible controls, but
are made learnable at deployment or per-user:

    p(F,S,T) = lerp(a, b, κ(F,S,T))
    κ(F,S,T) = sigmoid(α_F F + α_S S + α_T T + β)

The coefficients `{α_F, α_S, α_T, β, a, b}` are updated from observed
outcomes (acceptance, uncertainty reduction, task reward), while the UI
still exposes only F/S/T. This preserves simplicity while allowing the
system to adapt its parameterization over time.

In the current branch, this generic learner is instantiated for three
live priors that already shape runtime behavior:

-   `attention_width_prior` (Focus family)
-   `rate_target_prior` / `base_rate_prior` (Sensitivity family)
-   `hysteresis_band_prior` (Stability family)

The implementation keeps the historical closed-form priors as the exact
cold-start species priors. When no learned row exists, Cortext uses the
legacy formulas directly. Once informative outcomes arrive, it seeds a
narrow learnable band around that legacy value in an internal
`meta_learning_coeffs` table and then updates:

    success_t = 0.30 × write_accept_t
              + 0.30 × retrieval_use_t
              + 0.25 × (1 − u_t)
              + 0.15 × clamp(0.5 + 0.5δ_reward_t, 0, 1)
    direction_t = 2 × success_t − 1

Only informative turns (`write_accept_t`, `retrieval_use_t`, or non-zero
reward error) trigger updates. The learner then nudges
`{α_F, α_S, α_T, β, a, b}` toward or away from the currently successful
dynamic state:

-   Focus observes the current `attention_width_t`
-   Sensitivity observes the realized write-rate estimate `ρ̂_t`
-   Stability observes the current hysteresis band

This makes the meta-learning claim operational without adding any new
public controls or changing the public API surface.

# Structural Metrics and Composite Scoring

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
<a href="#sec-drift-coherence" class="quarto-xref">Section 6.4.2</a>)
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
midpoint‑biased knobs F̃ and S̃ (see Section 1) for all knob‑derived
controls and Table 1 metric formulas.

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
<a href="#sec-boundary" class="quarto-xref">Section 6.4.3</a>.

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
utility outcomes**. This avoids circular fitting to relevance and turns
the blender into an adaptive cognitive policy. Initial weights derive
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

    α_out(T) = lerp(0.12, 0.03, T)
    outcome_pred_t ← EWMA(outcome_pred_{t−1}, outcome_t, α_out(T))
    δ_reward_t ← outcome_t − outcome_pred_t  # reward prediction error for DA-like modulation

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

# Dynamic Thresholding and Homeostatic Control

The write gate compares composite scores against an adaptive threshold
θ_dynamic (written as theta_dynamic when referring to the recorded
variable). This section details the threshold evolution algorithm
incorporating Bayesian prior-evidence blending and homeostatic rate
control.

**Knob note:** F and S here refer to midpoint‑biased values (F̃, S̃) from
Section 1.

## Prior-Evidence Blending

The threshold prior derives from knob settings:

    θ_prior(F, S, T) = lerp(0.10, 0.30, T) × (1 − 0.3S)

Observed evidence comes from the 90th percentile of recent scores:

    score_history_cap(T) = round(lerp(512, 1536, T))
    score_stream ← tail(score_stream, max(win_score(T), score_history_cap(T)))
    recent_scores ← tail(score_stream, win_score(T))
    if |recent_scores| == 0:
        observed_p90 ← θ_prior  # no evidence yet
    else:
        observed_p90 ← percentile(recent_scores, 90)

Prior and evidence masses weight the blend:

    ρ_prior ← prior_mass(T) = round(lerp(2, 32, T))
    ρ_obs ← u(t) × |recent_scores|

The target threshold blends prior and evidence:

    θ_target ← (ρ_prior × θ_prior + ρ_obs × observed_p90) /
                max(ε, ρ_prior + ρ_obs)

High Stability increases prior mass, making the system more resistant to
observed deviations. High uncertainty increases evidence mass, allowing
faster adaptation to volatile conditions.

## Homeostatic Rate Control

The controller maintains write rates near the target setpoint through
continuous-time estimation with effective sample size (ESS) reliability
weighting.

### Rate Estimation

    Δt ← now_s() − to_s(last_rate_timestamp)
    dt_min(T) = 10^(−4 + 2T)  # stability-derived minimum Δt (seconds)
    Δt ← max(Δt, dt_min(T))
    τ_dt(T) = lerp(0.5, 2.0, T)
    α_dt ← 1 − exp(−Δt / τ_dt(T))
    dt_ema ← (1 − α_dt) × dt_ema + α_dt × Δt
    dt_floor(T) = lerp(0.1, 0.5, T)  # stability-derived minimum cadence
    dt_base ← max(dt_ema, dt_floor(T))

The rate time constant scales with Stability:

    rate_tau_exponent(T) = 3T
    τ_rate ← max(2^rate_tau_exponent(T) × dt_base, dt_floor(T))
    α ← 1 − exp(−Δt / τ_rate)

Instantaneous rate estimation with bias correction. Δwrites is the
binary indicator (0 or 1) of whether a write occurred during the current
timestep:

    Δwrites ← 1 if write_memory else 0  # binary write event

Normative note (MUST): Δwrites is computed from the current step’s
write_memory decision. Rate state updates occur after the write decision
and affect subsequent timesteps.

    ρ_inst ← (Δwrites / Δt) × 60  # writes per minute
    m_rate_t ← (1 − α) × m_rate + α × ρ_inst
    denom ← max(1 − (1 − α)^(rate_ticks + 1), ε)
    ρ_hat_t ← m_rate_t / denom  # bias-corrected estimate for next step
    m_rate ← m_rate_t
    rho_hat_prev ← ρ_hat_t
    rate_ticks ← rate_ticks + 1
    last_rate_timestamp ← now_ms()  # update after rate computation

### Effective Sample Size

ESS estimates the effective number of independent samples in the EMA,
using a heuristic inspired by Liu and Chen (1998):

    β ← max(0, 1 − α)
    ESS_cap(T) = lerp(30, 120, T)
    ESS ← min((1 + β) / max(1 − β, ε), ESS_cap(T))
    reliability ← 1 − exp(−ESS × (1 − T))

High Stability dampens reliability, preventing aggressive corrections in
conservative regimes.

### Homeostatic Correction

The rate error drives threshold adjustment. The κ\_\* gains are derived
from the knobs to avoid fixed behavioral constants:

Normative note (MUST): rho_hat_prev is stored state entering the
timestep. After the write decision, the rate-state update computes
ρ_hat_t; the assignment rho_hat_prev ← ρ_hat_t occurs at end of step
(see Appendix D).

    rate_error ← tanh((rho_hat_prev − rate_target) /
                      max(rate_target, ε))
    κ_r(F,S,T) = lerp(0.06, 0.14, S) × lerp(1.1, 0.9, T)
    cap_homeo(F,S,T) = lerp(0.35, 0.15, T) × lerp(0.8, 1.2, S) × hysteresis
    Δθ_homeo ← clamp(reliability × κ_r(F,S,T) × (1 − T) ×
                      (1 − maturity(t)) × rate_error,
                      −cap_homeo(F,S,T), +cap_homeo(F,S,T))

The correction scales with reliability and is attenuated by both
Stability and maturity, ensuring conservative, mature systems make
minimal homeostatic adjustments.

### Sensitivity-Based Threshold Adjustment

Sensitivity modulates threshold based on recent score volatility:

    recent_scores ← tail(score_stream, win_score(T))
    if |recent_scores| < 2:
        σ_scores ← 0
    else:
        σ_scores ← std(recent_scores)
    κ_sens(F,S,T) = lerp(0.04, 0.12, S) × lerp(1.1, 0.9, T)
    cap_sens(F,S,T) = lerp(0.30, 0.10, T) × lerp(0.9, 1.1, S) × hysteresis
    σ_ref(S,T) = lerp(0.08, 0.14, S) × lerp(1.1, 0.9, T)
    Δθ_sens ← clamp(−κ_sens(F,S,T) × S × (σ_scores − σ_ref(S,T)),
                    −cap_sens(F,S,T), +cap_sens(F,S,T))

High score variance with high Sensitivity lowers threshold, capturing
more volatile signals.

### Precision-Based Threshold Adjustment

Focus-driven precision tightens threshold when structural coherence is
high:

    κ_prec(F,S,T) = lerp(0.04, 0.10, F) × lerp(1.1, 0.9, T)
    cap_prec(F,S,T) = lerp(0.25, 0.08, T) × lerp(0.9, 1.1, F) × hysteresis
    Δθ_prec ← clamp(κ_prec(F,S,T) × F × (coherence_struct_t − 0.5),
                    −cap_prec(F,S,T), +cap_prec(F,S,T))

High structural coherence with high Focus raises threshold, enforcing
stricter relevance filtering.

## Threshold Integration

All threshold deltas combine and pass through safety limiting. The
emotion and mood deltas from
<a href="#sec-emotional-threshold" class="quarto-xref">Section 4.2.3</a>
and <a href="#sec-mood" class="quarto-xref">Section 4.2.4</a> are
denoted Δθ_emo = ΔThreshold_emotion_t and Δθ_mood = ΔThreshold_mood_t:

    Δθ_total ← Δθ_sens + Δθ_homeo + Δθ_prec + Δθ_emo + Δθ_mood
    cap_total ← max_ΔT_per_min(t) × (Δt / 60.0)
    Δθ_limited ← clamp(Δθ_total, −cap_total, +cap_total)
    θ_dynamic_t ← clamp(EWMA(θ_dynamic_{t−1}, θ_target,
                            α = α_T(t)) + Δθ_limited,
                        T_min(t), T_max(t))
    theta_dynamic ← θ_dynamic_t

Hysteresis evolves toward the stability-derived base:

    hysteresis_t ← clamp(EWMA(hysteresis,
                         base_band(T), α = α_T(t)),
                    band_min, band_max)
    hysteresis ← hysteresis_t

## Write Pacing and Memory Accumulation

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
    last_write_ts ← 0  # ms; 0 means "no prior write" for refractory
    eta_acc ← 0  # drift EWMA state
    coherence_prev ← 0  # previous coherence (initialize to 0)
    acc_signals_window ← []  # ring buffer of recent embeddings for coherence

Accumulator state is live runtime staging state, not durable memory. If
the process restarts before a write boundary commits a unit, the
unfinished accumulator window may be lost; committed memories, signal
rows, and working memory remain the durability boundary.

Reset behavior: reset_accumulator() clears μ_acc, drift_acc, s_sum,
s_max, n, e_peak, emo_max, arousal_sum, eta_acc, coherence_prev, and
sets acc_signals_window ← \[\] (and refreshes t_start/last_signal_ts for
the next unit), but retains last_write_ts so the refractory term remains
well-defined across boundaries.

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
(<a href="#sec-struct-coherence" class="quarto-xref">Section 5.1.1</a>).
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

### Spike Bypass (Flashbulb Flush)

High-salience signals bypass accumulation and flush immediately,
capturing preceding context as a coherent memory unit:

    spike_margin(S,T) = lerp(0.2, 0.5, T) × lerp(1.2, 0.8, S)  # above θ_dynamic
    mem_tau(T) = win_mem_ctx(T)
    mem_maturity ← 1 − exp(−n / max(mem_tau(T), 1))  # n = signals in current unit
    coherence_scale ← 1 + (1 − T) × clamp(coherence_t, 0, 1)
    spike_margin_eff ← spike_margin(S,T) × (1 + (1 − mem_maturity)) × coherence_scale
    spike_bypass = score_t > (θ_dynamic + spike_margin_eff)
    force_write ← false

When spike_bypass triggers:

    if spike_bypass:
        should_flush = true   # force a boundary
        force_write = true   # bypass S_window > θ_memory

This keeps flashbulb moments intact while preventing early or highly
coherent windows from fragmenting into micro-memories. As the unit
matures, spike bypass becomes easier, and low coherence reduces the
effective margin.

The implementation also uses the same force-write path when the
accumulator reaches its maximum signal capacity. That capacity flush is
an operational safety valve, not an additional flashbulb criterion.

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

### Window Score and Refractory

Memory-level score combines peak and average with coverage bonus:

    s_avg ← s_sum / max(n, 1)
    α(F) = lerp(0.3, 0.7, F)  # peak vs avg weight
    coverage ← min(n / n_ctx(T), 1.0)  # memory completeness
    β(S) = lerp(0.05, 0.15, S)  # coverage weight
    S_window ← α(F) × s_max + (1 − α(F)) × s_avg + β(S) × coverage

Write refractory suppresses rapid successive writes:

    τ_write_refrac(T) = lerp(5, 30, T)  # seconds
    k_write_refrac = lerp(0.3, 0.1, T)
    Δt_write ← now_s() − to_s(last_write_ts)
    M_write_refrac ← 1.0 + k_write_refrac × exp(−Δt_write / τ_write_refrac(T))

Implementation note: `Δt_write` is treated as a signed elapsed duration
and clamped at zero. Backward event timestamps therefore receive the
maximum refractory multiplier instead of underflowing into an
effectively infinite elapsed interval.

The same defensive rule applies to boundary gaps and retrieval pacing:
non-increasing timestamps produce a zero elapsed interval. They cannot
create an unsigned wraparound boundary or bypass the adjacent-retrieval
limiter.

Final write decision:

    θ_memory ← θ_dynamic × M_write_refrac
    write_memory = force_write OR (should_flush AND (S_window > θ_memory))
    if write_memory: last_write_ts ← now_ms()  # update refractory timestamp (ms)

Normative rule (MUST): if should_flush is true, the current unit must be
finalized. If write_memory is false, discard the unit and
reset_accumulator() anyway (do not update last_write_ts). This prevents
perpetual should_flush states (time cap / drift cap / gap cap) while
never resetting.

Trace note: if a run trace reports both write_decision and stored,
interpret write_decision as the boolean gate outcome at the boundary
(the write_memory predicate above). Interpret stored as the eventual
recording outcome (after any final safety checks).

Representative embedding blends accumulator mean with peak:

### Representative Embedding

    ρ(F) = lerp(0.2, 0.6, F)  # mean vs peak blend
    e_rep ← l2_normalize(ρ(F) × μ_acc + (1 − ρ(F)) × e_peak)

    # Edge-case guard
    if |μ_acc| = 0: e_rep ← e_peak
    if |e_peak| = 0 OR dim(e_peak) ≠ dim(μ_acc): e_rep ← μ_acc

On write: store e_rep with metadata {n, s_max, s_avg, drift_acc,
mem_elapsed, s_emotion_max=emo_max, s_arousal_avg=arousal_sum / max(n,
1), boundary_score}. Store `c_t` as the temporal context for the memory.
Append e_rep to memory_stream and l2_normalize(μ_acc) to
recent_memory_centroids. Update `index_store` with the sparse key
(<a href="#sec-pattern-separation" class="quarto-xref">Section 8.14</a>).
Reset accumulator for next unit.

# Reinforcement and Decay Dynamics

## Memory Strength Model

Each memory maintains **multi-timescale traces** rather than a single
leaky bucket. A mixture of fast/medium/slow/ultra-slow traces yields a
power-law forgetting curve while preserving plasticity.

Let `f = FocusBias(F)`, `s = SensitivityBias(S)`, and let `f0` and `s0`
be their values at the neutral knobs. The active trace count is:

    N_traces(F,S,T) = clamp(round((2 + 2T) ×
                             (1 + 0.08(f-f0) + 0.06(s-s0))), 1, 4)

The familiar `0.10/0.50/2.00/8.00` half-life multipliers are the neutral
baseline, not universal constants. Production applies bounded F/S/T
policy scales:

    reactive = 1 + 0.14(s-s0) - 0.08(f-f0) + 0.06(T-0.5)
    stable   = 1 + 0.24(T-0.5) - 0.06(s-s0)
    τ_fast  = clamp(0.10 × reactive, 0.05, 0.20) × half_life
    τ_med   = clamp(0.50 × (1 + 0.12(T-0.5) - 0.06(f-f0)), 0.30, 0.80) × half_life
    τ_slow  = clamp(2.00 × stable, 1.20, 3.20) × half_life
    τ_ultra = clamp(8.00 × stable × (1 + 0.10(f-f0)), 4.00, 12.00) × half_life
    τ_list = [τ_fast, τ_med, τ_slow, τ_ultra]

    α_min_S = 0.05; α_span_S = 0.35
    α_S(t) = α_min_S + SensitivityBias(S) × α_span_S × u(t)
    used_flag(m) = 1 if m was retrieved and used in current step, else 0

Trace updates combine exponential decay, EWMA learning, and knob-scaled
reinforcement (for i in 1..N_traces):

    λ_i ← ln(2) / τ_list[i]
    reinforcement ← clamp((S_eff × used_flag(m) +
                           F_eff × clamp(influence_factor, 0, 1))
                           × serial_position_mult, 0, 1)
    increment_i ← (α_S(t) × used_flag(m) + reinforcement) / N_traces
    trace_i ← clamp(trace_i × exp(−λ_i × Δt) + increment_i, 0, 1)

The reinforcement term distributes S- and F-scaled retrieval feedback
uniformly across all active traces. This avoids front-loading
reinforcement into the fast trace (which decays quickly) and instead
strengthens the entire trace ensemble when a memory is retrieved and
used.

Trace coupling encourages long-lived knowledge without freezing
plasticity:

    coupling = clamp((0.05 + 0.10T) ×
                     (1 - 0.10(f-f0) + 0.12(s-s0)), 0.02, 0.20)
    trace_{i+1} ← clamp(trace_{i+1} + coupling × trace_i, 0, 1)

Combined strength uses a knob-shaped mixture that favors slow traces as
Stability increases:

    w_fast  = clamp((0.40 - 0.25T) ×
                    (1 - 0.08(f-f0) - 0.10(s-s0)), 0.05, 0.50)
    w_med   = clamp(0.25 × (1 - 0.04(f-f0) - 0.04(T-0.5)), 0.12, 0.35)
    w_slow  = clamp((0.20 + 0.15T) ×
                    (1 + 0.08(f-f0) + 0.06(s-s0)), 0.12, 0.45)
    w_ultra = clamp((0.15 + 0.10T) ×
                    (1 + 0.10(f-f0) + 0.06(s-s0)), 0.08, 0.35)
    w_raw ← [w_fast, w_med, w_slow, w_ultra]
    w ← normalize(w_raw[1..N_traces])
    strength_t ← clamp(Σ_i w_i × trace_i, 0, 1)

Δt is measured per memory using its last access timestamp:

    Δt(m) ← now_s() − to_s(m.last_access); if last_access is unset, use created_at

Memories falling below the periphery cutoff are candidates for eviction.
**Only LONG_TERM memories are evicted**; association and other
graph-scaffold rows are not evicted by the per-memory decay pass and are
pruned during explicit consolidation.

    periphery_cutoff(T) = lerp(0.03, 0.20, T)
    storage_budget_floor = max(
      min_db_bytes,
      db_avail_fraction × available_disk_bytes
    )

    if m.kind == LONG_TERM
       AND db_used_bytes ≥ storage_budget_floor
       AND strength_t < periphery_cutoff(T)
       AND m.created_at < last_consolidation_ts:
         evict(m)

Eviction removes the memory’s signal rows, reconstruction rows, graph
edges, and object payloads in the same transaction. The cleanup
candidate set includes the representative embedding and every embedding
owned by its signals and reconstructions; an embedding is deleted only
when no surviving row references it. This makes storage-pressure
eviction reclaim the dominant vector payloads without a later global
orphan sweep.

The v1 hard cut removed the structured fact-evidence floor. Retention
now depends on the trace strength, the storage-pressure gate, and the
consolidation watermark. Source provenance remains attached to episodic
rows, but it no longer creates a separate eviction exemption through
fact tables.

**Current implementation note:** the runtime now includes a storage gate
ahead of the usual strength-based eviction test. For file-backed stores,
eviction is skipped entirely until the on-disk database footprint
reaches a configurable floor. The default fixed floor is `500 MB`, and
an internal percentage-of-available-space floor can also be applied; the
active threshold is the maximum of the two. This keeps long chat
sessions from prematurely evicting weak but still useful traces simply
because they have decayed below `periphery_cutoff(T)` while the database
is still tiny. Deterministic file-backed sweeps in
<a href="#sec-experimental" class="quarto-xref">Section 11</a> showed
three useful regimes. Under the raw storage-growth sweep, budgets up to
`256 MB` still behaved like ordinary decay, `500 MB` was the first
tested budget that materially delayed eviction, and `1 GB` extended that
delay further. Under the synthetic human-sleep clock, that translated
into an interpretable retention boundary: `256 MB` was enough for
infant/toddler/preschool-style frequent consolidation, but it still
missed school-age, teen, adult, and older-adult nightly consolidation.
`500 MB` was the first tested floor that reliably preserved labels for
once-per-day adult-style consolidation, while `1 GB` added safety margin
rather than a new qualitative retention regime. In-memory stores keep
the historical behavior unless the gate is explicitly forced on for
studies.

Ablation studies in
<a href="#sec-experimental" class="quarto-xref">Section 11</a> confirm
that the retained components of this model contribute to selective
eviction behavior. Disabling coupling or reducing to a single trace
collapses the power-law forgetting curve into rapid indiscriminate
eviction, flashbulb bonuses provide the expected protective effect, the
storage gate prevents premature pruning of small databases, and uniform
reinforcement distribution makes S and F visibly effective in the
eviction dynamics.

## Contextual Gain and Per-Memory Stability (Definitions)

Contextual gain is a per-retrieval signal used by multiple feedback
loops:

    contextual_gain_t(m) ← cos(μ_acc, m.embedding)    # if m was retrieved and used (range [−1, 1])
    contextual_gain_t(m) ← undefined                # otherwise

When undefined, treat contextual_gain_t(m) as 0 in downstream updates.
Persisted per-memory gain is tracked as a smoothed value:

    L_cg = round(lerp(8, 32, T))
    α_cg = 2 / (L_cg + 1)
    contextual_gain(m) ← EWMA(contextual_gain(m), contextual_gain_t(m), α_cg)

Per-memory stability initializes at creation and is bounded:

    strength(m)_init = memory_initial_strength(F,S,T)
    stability(m)_init = memory_initial_stability(F,S,T)
    stability(m) ← clamp(stability(m), 0.0, 2.0)

Retrieved vs used:

    retrieved(m) = memory returned by retrieval (vector or graph expansion)
    used(m) = retrieved(m) AND injected into active context after gate decisions

Contextual gain is only measured for used memories.

## Influence-Weighted Updates

When contextual gain signals are available, influence factors enter the
reinforcement term that is distributed uniformly across traces (see
trace update above):

    influence_factor ← (used_count / max(retrieved_count, 1)) ×
                        clamp(contextual_gain(m), −1, +1)
    reinforcement ← clamp(S_eff × used_flag(m) +
                           F_eff × clamp(influence_factor, 0, 1), 0, 1)
                    × serial_position_mult

This replaces the earlier front-loaded boost that applied 60%/30%/10% to
fast/med/slow traces. Ablation studies showed the front-loaded
distribution had no measurable effect on eviction dynamics because its
contribution was absorbed by trace clamping. Uniform distribution
ensures all active traces receive meaningful reinforcement, improving
long-term retention for used memories.

## Reinforcement Graph Maintenance

The normal interrupt gate injects at most one selected memory into
active context per dispatch. Production retrieval therefore does not
synthesize a pairwise *reinforces* edge from its broader retrieved
packet: packet co-occurrence is not evidence that two memories were used
together. This also prevents repeated noisy packets from turning
boilerplate into durable graph hubs.

Existing *reinforces* edges remain valid association inputs for stores
and explicit graph workflows that already contain them. They are decayed
and pruned only during explicit consolidation cycles (triggered
externally via `Consolidate()` or `cortext_consolidate`), matching the
system’s idle-time consolidation policy. At retrieval time their weight
is degree-normalized before the bounded graph bonus is applied,
preventing high-degree hubs from dominating candidate rank.

## Causal Feedback Loop

The system tracks causal influence of retrieved memories on generation
quality through contextual gain—the semantic alignment (cosine
similarity) between the current signal embedding and the retrieved
memory embedding.

### Focus Feedback

    αF_base = 0.10; βF_base = 0.05
    weight_relevance_t ← weight_relevance
    attention_width_t ← attention_width
    for each used memory m:
        if contextual_gain(m) > 0:
            weight_relevance_t += αF_base × contextual_gain(m)
            attention_width_t *= (1 − βF_base)
        else:
            attention_width_t *= (1 + βF_base)
    # Drift-responsive dilation + restorative pull to prior
    dilation_force ← S × (1 − T) × drift_step_t
    restoring_force ← F × (attention_width_prior − attention_width_t)
    attention_width_t ← attention_width_t + dilation_force + restoring_force
    weight_relevance ← clamp(weight_relevance_t, 0, 1)
    attention_width_t ← clamp(attention_width_t,
                              attention_width_min, attention_width_max)
    attention_width ← attention_width_t

Positive contextual gain narrows attention and boosts relevance
weighting; negative gain widens attention to explore alternatives. The
drift_step_t term is the instantaneous drift signal from
<a href="#sec-structural-metrics" class="quarto-xref">Section 5</a>.

### Sensitivity Feedback

    η_base = 0.10
    weight_novelty_t ← weight_novelty
    for each used memory m:
        novelty_reward ← 1 − sim(m.embedding, recent_context)
        weight_novelty_t += η_base × (novelty_reward ×
                             contextual_gain(m) −
                             redundancy(m, recent_context))
    weight_novelty_t ← clamp(weight_novelty_t, 0, 1)
    weight_novelty ← weight_novelty_t

This rewards novelty that proves useful while penalizing redundant
retrievals.

### Stability Feedback

    γT_base = 0.05
    for each used memory m:
        if contextual_gain(m) > 0:
            stability_t(m) ← stability(m) + γT_base
        else:
            stability_t(m) ← stability(m) × (1 − γT_base)
        stability(m) ← clamp(stability_t(m), 0.0, 2.0)

The mean stability of used memories provides adjustment to the half-life
target:

    adj ← clamp(mean(stability(m_used)) − 1.0, −0.25, +0.25)
    ΔHalfLife_adj_t ← adj

This factor is consumed by the Stability update
(<a href="#sec-stability-update" class="quarto-xref">Section 4.3.2</a>),
avoiding conflicting adjustments between feedback mechanisms.

## Generation Influence Tracking

When generation embeddings are wired into the runtime, influence can
incorporate output trajectory. The current implementation keeps
generation-similarity and generation-drift terms neutral unless such
embeddings are available:

    Δḡ ← l2_normalize(ḡ_t) − l2_normalize(ḡ_{t−1})
    drift_mag_gen ← ‖Δḡ‖
    drift_contribution(m) ← (drift_mag_gen / 2) ×
                             max(0, cos(m.embedding, l2_normalize(Δḡ)))

The intended full influence blend is:

    λ₁ = 0.5; λ₂ = 0.4; λ₃ = 0.3
    influence(m) ← λ₁ × contextual_gain(m) +
                    λ₂ × cos(m.embedding, ḡ_t) −
                    λ₃ × drift_contribution(m)

Sustained influence accumulates over a stability-dependent horizon:

    L_sustain(T) = round(lerp(3, 5, T))
    sustained_influence ← EWMA(sustained_influence,
                               influence(m),
                               α = 2 / (L_sustain(T) + 1))

# Advanced Cognitive Processes

This section presents algorithms modeling higher-order cognitive
phenomena: working memory maintenance, soft anchor formation, retrieval
uncertainty, reconsolidation dynamics, and serial position effects.

**Knob note:** Unless explicitly stated otherwise, F and S in this
section refer to midpoint‑biased values (F̃, S̃) defined in Section 1.

## Working Memory Gates

Motivated by bounded working-memory models such as Cowan (2001), Cortext
maintains a limited number of active application-level memory units.
These slots are not asserted to be human cognitive chunks. Working
memory holds coherent memories as defined in
<a href="#sec-write-pacing" class="quarto-xref">Section 6.4</a>,
preserving the full content and signal sequence:

    base_capacity = round(3 × (lerp(8, 6, S̃) + lerp(−1, 1, F̃)))

This yields 15–27 slots and 21 at neutral knobs. High Sensitivity
reduces capacity (faster turnover), while high Focus increases the slot
budget. The operating point comes from the capacity study in
<a href="#sec-experimental" class="quarto-xref">Section 11</a> rather
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
    memory.context ← c_t  # temporal context state (@sec-temporal-context)
    memory.source_model ← {origin, reliability, contradiction_count, last_verified_ts}
    memory.evidence_packets ← ordered immutable signal rows
    memory.current_view ← latest reconstruction blob if present, else concatenated signal blobs
    memory.metadata ← {n, s_max, s_avg, drift_acc, mem_elapsed,
                       s_emotion_max, s_arousal_avg}

The emotional metrics (s_emotion_max, s_arousal_avg) are accumulated
during memory formation for use by Emotional Consolidation
(<a href="#sec-emotional-consolidation"
class="quarto-xref">Section 8.12</a>).

## Temporal Context State (Time-Cells Analogue)

Beyond the accumulator centroid μ_acc, Cortext maintains a slowly
drifting temporal context vector `c_t`. This provides ordered recall and
reduces topic-collision errors by enabling **context reinstatement**.

    α_c(S,T) = AccumulatorTemporalContextAlpha(S,T)
    c_t ← l2_normalize((1 − α_c(S,T)) × c_{t−1} + α_c(S,T) × μ_acc)

At neutral Sensitivity, `α_c` follows `lerp(0.06, 0.01, T)`; the
production policy applies a bounded Sensitivity adjustment. Each memory
stores `(e_rep, c_t)`. The current runtime uses `c_t` as the topic-shift
anchor in boundary detection and persists it with the memory. Production
graph retrieval does not currently add a separate `cos(c_t, m.context)`
score or reinstate `c_t` from a retrieved memory; it queries from the
current accumulator and recent-context surfaces instead. The following
context-score/reinstatement rule is therefore a proposed extension, not
shipped behavior:

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
    # last_ts changes only on chunking, rehearsal, or insertion

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
    rehearsal_rate = lerp(0.5, 2.0, S)
    memory_dedication_strength = lerp(0.3, 0.9, T)

Working memory gating evaluates coherent memories at accumulation
boundaries
(<a href="#sec-boundary" class="quarto-xref">Section 6.4.3</a>), not
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
    if |similar_memories| > 0:
        merge_into_chunk(similar_memories, memory)

**Implementation note:** durable ingress inputs preserve an ordered
working-memory trace independent of the literal `source_id` string.
`source_id` is still used for opaque provenance and exact same-source
grouping, but Cortext does not parse roles such as user, assistant,
speaker, or device from it. Internal maintenance work follows the same
rule: explicit consolidation is driven by a force flag on the signal,
not by a magic `source_id` value. Prompt hydration orders active slots
by insertion/start time rather than `last_access`; otherwise rehearsal
or retrieval touches can scramble the visible recent context even when
the underlying memories are correct.

## Two-Layer Graph Memory Surface

Cortext now separates live memory state into two graph-compatible
layers. The layers differ by latency and commitment rather than by
modality:

1.  **Working memory (WM):** the bounded active slot set used for
    immediate prompt reconstruction, overlap suppression, and
    recent-turn continuity. WM stores coherent memories and preserves
    their ordered evidence packets. It is the most volatile layer and is
    optimized for exact recent context rather than broad semantic
    search.
2.  **Long-term graph memory (LTM):** the persisted graph of long-term
    memories, association centroids, optional label nodes, and durable
    association edges. Durable memory nodes live in the `memories` table
    and durable edges live in `associations`. There is no
    processor-local STM shadow graph and no separate fact assertion
    store.

The intended control flow is:

    signal embedding
      -> WM active-slot update
      -> LTM retrieval from memories and association centroids
      -> explicit consolidation
      -> shallow association centroid insert/update
      -> durable LTM graph edge insert/update

This keeps the online loop fast and modality-agnostic. Consolidation is
a shallow, embedding-only commitment step: it clusters mature long-term
memories, creates association centroids, connects member memories with
`derived_from`, and optionally attaches already-existing label nodes by
embedding similarity.

## Soft Anchor Formation

Soft Anchor is an ingress-time uncertainty layer for subject, entity,
object, and event continuity. It is not a hard referent resolver and it
is not a retrieval-time reranker. At memory ingestion time, Cortext
compares the current memory signal against already-established
working-memory (WM), short-term memory (STM), and long-term memory (LTM)
anchor state. The output is a ranked, uncertainty-preserving set of
possible continuities. Retrieval may later consume these links as
context, but retrieval candidates do not create anchors.

The input signal model emits view vectors:

    x_sem  ← semantic_key(m)
    x_ent  ← entity_key(m)
    x_full ← full_key(m)

For an ES-AIST-style signal model these are the corrected slices:

    semantic_key = z[0:768]
    entity_key   = z[768:1536]
    full_key     = z[0:1536]

Production note: the shipped runtime first uses a dedicated
`soft_anchor_embedding` when the encoder exposes at least 1536
dimensions. When only the 256-dimensional retrieval embedding is
available, it uses a single-view fallback: semantic, entity, and full
centroids all reference the normalized available embedding, and entity
quality is lowered by an F/S/T-derived prior. ES-AIST slices are
therefore supported, but they are not required for the production path.

Each active soft anchor state stores separate centroids and support
statistics:

    SoftAnchorState {
      anchor_id
      status ∈ {provisional, active, durable, decayed, rejected}
      semantic_centroid, entity_centroid, full_centroid
      semantic_radius, entity_radius, full_radius
      memory_tier_mask
      source_histogram
      first_step, last_step, last_boundary_id
      anchor_strength
      support_count, contradiction_count
      recent_memory_ids
    }

Each memory may retain multiple possible links:

    SoftAnchorLink {
      memory_id
      anchor_id
      anchor_strength ∈ [0, 1]
      anchor_label ∈ {none, tentative, ambiguous, durable, decayed, rejected}
      evidence_kind
      memory_tier ∈ {WM, STM, LTM}
      score
      margin
      entropy
      support_count
      contradiction_count
      created_step
      updated_step
    }

### Candidate Set

The candidate set is built only from prior soft anchor state. The
current runtime iterates the bounded set of non-rejected
`SoftAnchorState` objects loaded into processor state; the WM/STM/LTM
split below describes the intended tiered candidate policy and the
metadata labels assigned to retained links, not a separate production
retrieval over three anchor indexes yet.

    C_WM  = active WM anchors
    C_STM = top STM anchors by state similarity, capped
    C_LTM = durable/provisional LTM anchors only when entity/full evidence is non-weak
    C     = bounded_union(C_WM, C_STM, C_LTM)

The caps are knob-derived:

    K_STM_pre = round(lerp(24, 8, F) × (0.75 + 0.50S))
    K_LTM_pre = round(lerp(16, 4, F) × (0.70 + 0.30S) × (0.80 + 0.20T))
    K_pre     = min(64, |C_WM| + K_STM_pre + K_LTM_pre)

No future labels, retrieved memories, target flags, candidate classes,
or gold actions enter the runtime candidate set.

### View Evidence

For each candidate anchor `a`, compute:

    s_sem(a)  = map01(cos(x_sem,  a.semantic_centroid))
    s_ent(a)  = map01(cos(x_ent,  a.entity_centroid))
    s_full(a) = map01(cos(x_full, a.full_centroid))

The view weights use the three knobs. Focus increases entity
specificity, Sensitivity preserves weak semantic continuity, and
Stability increases context/full-vector influence:

    w_sem_raw  = (0.45 - 0.15F + 0.10S) × q_sem
    w_ent_raw  = (0.25 + 0.35F) × q_ent
    w_full_raw = (0.20 + 0.25T + 0.05F) × q_full
    [w_sem, w_ent, w_full] = normalize([w_sem_raw, w_ent_raw, w_full_raw])

    view(a) = w_sem s_sem(a) + w_ent s_ent(a) + w_full s_full(a)

When quality estimates are unavailable, use `q_sem = q_full = 1` and an
F/S/T-derived weak entity-quality prior. At the neutral midpoint this
equals `q_ent = 0.5`. If the entity vector is missing or known-diffuse,
set `q_ent = 0`.

### Engine Evidence

Soft Anchor is an engine policy over signal evidence plus ingress state.
The real-anchor score combines view evidence with source continuity,
tier-adjusted recency, support history, boundary pressure, and
contradiction pressure. Generic or low-information suppression is
applied in the null/new hypothesis competition and hard-retention gates
rather than subtracted from every real anchor score:

    R(a) =
      w_view    view(a)
    + w_source  source_continuity(m, a)
    + w_recency recency_tier(m, a, T)
    + w_support support_prior(a, T)
    - w_boundary boundary_shift(m, a, F, T)
    - w_contra   contradiction_penalty(a, S)

The blend weights are normalized:

    [w_view, w_source, w_recency, w_support, w_boundary, w_contra] =
    normalize([
      0.55 + 0.20F,
      0.20 + 0.20F + 0.10T,
      0.20 + 0.20S + 0.10T,
      0.15 + 0.45T,
      0.25 + 0.35F + 0.15T,
      0.20 + 0.30S + 0.20F
    ])

Generic or low-information memories must not create durable continuity.
The generic suppressor is modality-general: it uses null centroids, low
entity mass, low information content, and diffuse candidate entropy.
Text shortcuts such as “ok”, “yeah”, “bye”, and filler are optional
extra evidence, not the anchor mechanism.

### Null and New Hypotheses

Soft Anchor competes every real anchor against two special hypotheses:

    H_none = no continuity; emit no retained link
    H_new  = start a new provisional anchor

Compute:

    best_real = max_a R(a), or 0 when C is empty
    H_real    = normalized_entropy({R(a)})
    info(m)   = clamp((q_sem + q_ent + q_full) / 3, 0, 1) × (1 - generic(m))

    score_none = SoftAnchorNoneScore(F,S,T,generic,H_real,
                                     1 - best_real, top_contra)

    score_new  = SoftAnchorNewScore(F,S,T,info,boundary_score,
                                    best_real,generic)

Then:

    β_anchor(F) = lerp(4.0, 14.0, F)
    P = softmaxβ([R(a_1), ..., R(a_n), score_none, score_new])

If `H_none` wins, the memory receives no active continuity link. If
`H_new` wins and the signal is informative, Cortext creates a
provisional soft anchor.

### Labels and Retention

Soft Anchor does not collapse uncertainty to top-1. It retains up to:

    K_keep(F,S,T) = clamp(round(lerp(8, 3, F) × lerp(1.10, 0.85, T) + 2S), 2, 10)

for real anchors satisfying posterior and score thresholds. The label is
a policy state, not a truth claim:

    none       H_none wins or no candidate passes retention
    tentative  one anchor has usable evidence but insufficient support
    ambiguous  multiple anchors survive with low margin or high entropy
    durable    repeated non-generic support or exceptional evidence passes
    decayed    prior link has weakened below active threshold
    rejected   contradiction, generic suppression, or correction invalidates link

Ambiguous links may be surfaced as possible continuity, not as a fact.
Durable assertions must not be formed from ambiguous links.

### Durable Promotion and Decay

Durable promotion is deliberately conservative:

    support_required(F,T) = clamp(ceil(2 + 1.5F + 2.0T), 2, 6)

A support event counts only if it is non-generic, boundary-compatible,
low-entropy, has sufficient margin, and is not a near duplicate of the
previous support event. Current durable promotion requires repeated
support plus low-entropy, high-margin evidence; one-shot exceptional
durable promotion remains a possible future policy rather than the
active runtime behavior.

Link strength decays independently from the anchor:

    strength_t = strength_0 × exp(-ln(2) × Δt / τ_link_tier(T))

with WM shortest, STM intermediate, and LTM longest. Higher Stability
increases half-life and support window length; it must not by itself
override boundary pressure or force cross-boundary durable binds.
Non-durable anchor states decay after a bounded active-TTL grace period
whose multiplier is also F/S/T-derived and remains `2x` at the neutral
midpoint.

### Knob Semantics

The expected monotonic behavior is:

    Focus ↑:
      fewer weak retained links
      higher margin requirement
      stronger entity/split pressure
      lower tentative recall acceptable

    Sensitivity ↑:
      more tentative links
      higher target-in-top-k recall
      more ambiguous context acceptable
      durable precision should not materially drop

    Stability ↑:
      slower decay
      longer support accumulation windows
      fewer label flips
      no cross-boundary durable overreach

Current checked-in evidence is limited to deterministic formation,
ambiguity, promotion, persistence, and hydration tests;
<a href="#sec-experimental" class="quarto-xref">Section 11</a> does not
report a human-subject anchoring benchmark. `UpdateSoftAnchor` runs at
ingress after memory storage and writes soft-anchor state and links by
default. Production retrieval does not rank from these links yet, and
application surfacing is limited to hydrated memory metadata and
experiment/benchmark utilities. The next open question is consumption:
formed anchors may become useful uncertain context only after replay and
manual-review experiments show that possible-continuity hints help more
often than they harm, without converting tentative or ambiguous evidence
into durable assertions.

## Retrieval Uncertainty

The current engine does not maintain a persistent higher-order
confidence state. Uncertainty is expressed directly through retrieval
scores, rank margins, coverage, candidate diversity, and source/evidence
metadata:

    rank_margin(q) = score_1(q) - score_2(q)
    retrieval_uncertainty(q) = 1 - clamp(rank_margin(q) / margin_ref(F,S,T), 0, 1)

This keeps uncertainty local to the retrieval event. Callers can use the
ranked candidate packet and evidence surfaces to decide whether to
answer, ask for clarification, or expose multiple possible memories
without the engine carrying a separate persistent confidence machine.

## Memory Reconsolidation

Following Nader, Schafe, and Le Doux (2000), retrieved memories enter a
labile state permitting modification:

    τ_labile = lerp(30, 300, T)  # seconds
    reconsolidation_gain = lerp(0.2, 0.02, T)
    lability_susceptibility = (1 − T) × (0.5 + 0.5 × S)

During the lability window, memories can drift toward current context:

    drift_magnitude = (1 − T) × S × lability ×
                       contextual_relevance

Reconsolidation effects propagate to semantically related memories with
decay:

    ripple_decay = lerp(0.5, 0.1, T)  # per semantic hop

## Stored Source Metadata and Supersession

Each memory stores opaque `source_id` provenance plus `source_origin`,
`source_reliability`, and `source_contradiction_count`. The current
writer uses the constant origin class `"source"`; the actual application
provenance remains in `source_id`. Reliability receives an F/S/T-derived
prior and supersession can increment contradiction history. These fields
are retained metadata, but the public hydrated memory surface does not
currently expose a computed source confidence and production graph
retrieval does not gate or rank on it.

The following confidence rule is a proposed consumption policy, not
shipped behavior:

    source_model = {origin, reliability, contradiction_count, last_verified_ts}
    source_prior(F, S, T) = SourceReliabilityPrior(F,S,T)
    freshness_weight = RetrievalSourceFreshnessWeight(F,S,T)
    freshness(m) = exp(−age(m) / RetrievalSourceFreshnessTauSeconds(F,S,T))
    source_confidence(m) ← clamp(source_prior(m.origin,F,S,T) ×
                                 (1 − contradiction_penalty(F,S,T) × contradiction_count) ×
                                 ((1 − freshness_weight) + freshness_weight × freshness(m)),
                                 0, 1)

If implemented, contradiction history would reduce computed confidence
and the threshold would gate injection into active context:

    if source_confidence(m) < lerp(0.15, 0.45, T):
        downrank_or_hold(m)

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
        contradiction_count(e_old) += 1

Retrieval treats `supersedes` as a directional family relation rather
than a traversable activation edge. Superseded targets are removed
before the vector seed cap, leaving the replacement eligible as the
family representative and preventing obsolete members from consuming
prompt slots. The number of written supersession edges is capped by
`SupersessionMaxEdges(F,S,T)`.

## Constructive Recall and Controlled Distortion

Constructive recall is now implemented as a two-layer memory view:

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

During retrieval:

    current_embedding ← latest(reconstruction.embedding, fallback = memory.embedding)
    score_retrieval(current_embedding, current_context)
    recon ← bounded_blend(current_embedding, query, temporal_context)
    append(memory_reconstructions,
           {embedding_id = recon.embedding,
            blob_id = current_blob,
            ts = now,
            uncertainty = inferred_retrieval_uncertainty,
            trigger = 'retrieval'})

Hydration resolves `memory.current_view` from the latest reconstruction
blob when one exists; otherwise it falls back to the immutable evidence
packets. Reconsolidation follows the same audit-trail rule: it appends a
new `trigger='reconsolidation'` reconstruction instead of overwriting
the evidence embedding stored on the base memory row. This gives the
current branch the controlled-distortion property the earlier draft
claimed, while preserving the original evidence vector for inspection
and replay.

## Retrieval Competition

Retrieved memories compete through lateral inhibition, modeling
retrieval-induced forgetting (Anderson, Bjork, and Bjork 1994):

    inhibition_radius = lerp(0.5, 0.85, F)
    winners_k = round(lerp(7, 3, F))
    suppression_per_retrieval = lerp(0.1, 0.01, T) ×
                                 (1 − winning_activation)
    recovery_time_RIF = lerp(300, 1800, T)  # seconds

High Focus produces narrow winner-take-all dynamics; low Focus permits
broader activation.

## Predictive Pre-activation

The system pre-activates memories predicted to be relevant based on the
EMA expectation state (`x_pred_ema`) and recent context:

    prediction_horizon = round(lerp(2, 8, F))
    pre_activation_decay = lerp(0.7, 0.3, T)
    prediction_conf_threshold = lerp(0.3, 0.7, F)
    surprise_sensitivity = S × lerp(2.0, 0.5, T)

When predictions fail (high surprise), the system increases the refresh
rate of pre-activation:

    update_rate_on_surprise = lerp(0.2, 0.02, T) × S

Current implementation note: the shipped pipeline updates and decays the
persisted `pre_activation` field, but production graph retrieval does
not read that field. The following bounded retrieval prior describes the
experimental ranking variant, not current runtime behavior:

    predictive_weight(F,T) = lerp(0.05, 0.20, FocusBias(F)) × lerp(1.0, 0.85, T)
    predictive_bonus = predictive_weight × pre_activation

In the current runtime, high-surprise updates refresh latent predictive
state without changing the next retrieval score. The historical ablation
in <a href="#sec-experimental" class="quarto-xref">Section 11</a>
evaluates the earlier research variant and must not be read as a
measurement of this hard-cutover implementation.

## Serial Position Effects

The architecture models primacy, recency, and distinctiveness effects
observed in human memory (Murdock Jr 1962):

    primacy_window = round(lerp(5, 2, F))
    primacy_bonus = lerp(1.2, 2.0, S)
    recency_window = round(lerp(7, 3, F))
    rehearsal_curve_depth = lerp(0.2, 0.6, S)

The von Restorff (isolation) effect enhances memory for distinctive
items (Hunt 1995):

    distinctiveness_threshold = lerp(0.6, 0.8, F)
    von_restorff_multiplier = lerp(1.5, 3.0, S)

Items in the middle region suffer interference:

    interference_zone = positions[primacy_window+1 : −recency_window]
    middle_suppression = lerp(0.8, 0.5, S) × (1 − F)

## Emotional Consolidation

High-emotion events trigger enhanced consolidation, following McGaugh
(2004) findings. As detailed in
<a href="#sec-consolidation" class="quarto-xref">Section 9</a>,
consolidation operates on stored memory metadata:

    θ_intensity = lerp(0.6, 0.8, 1 − S)
    θ_arousal = lerp(0.4, 0.2, S)
    # Consolidation uses stored memory emotional metadata
    trigger = (m.metadata.s_emotion_max ≥ θ_intensity) AND
               (m.metadata.s_arousal_avg ≥ θ_arousal)

Flashbulb memories receive extended half-life bonuses based on the
memory’s peak emotional intensity:

    flashbulb_threshold = lerp(0.97, 0.65, S)
    flashbulb_threshold_eff = flashbulb_threshold × (1 − 0.5 × s_emotion_max)
                               × (1 − 0.3 × s_arousal_avg)
    # Rate stabilizer only suppresses over-frequent flashbulbs; it does not lower
    # thresholds when the recent flashbulb rate is already below target.
    flashbulb_target = lerp(0.02, 0.06, S)
    flashbulb_rate_ewma ← EWMA(flashbulb, α = lerp(0.02, 0.12, S))
    rate_adjust = clamp(1 + lerp(1.0, 0.6, S) × max(0, flashbulb_rate_ewma − flashbulb_target),
                        1.0, 1.2)
    # Contextual percentile gate (embedding‑only, knob‑derived)
    emo_window = round(lerp(24, 96, S) × lerp(0.8, 1.2, T))
    percentile = lerp(0.95, 0.80, S)
    flashbulb_gain = lerp(1.0, 1.1, S)
    flashbulb_threshold_adj = max(flashbulb_threshold_eff × rate_adjust,
                                  Pctl_{percentile}(recent_emotion_intensity, emo_window))
    flashbulb_arousal = lerp(0.7, 0.5, S)
    flashbulb = ((s_arousal_avg ≥ flashbulb_arousal) AND
                 ((flashbulb_gain × s_emotion_max) ≥ flashbulb_threshold_adj))
    # Half-life bonus uses stored memory emotional peak
    emotional_half_life_bonus = exp(lerp(0, ln(3), S)) ×
                                 (1 + m.metadata.s_emotion_max)

The emotional metrics (s_emotion_max, s_arousal_avg) are accumulated
during memory formation and stored with the memory
(<a href="#sec-accumulator" class="quarto-xref">Section 6.4.1</a>).

    cascade_radius = round(lerp(1, 5, S))
    cascade_decay = lerp(0.7, 0.3, S)
    cascade_source_window_ms = 1000 × consolidation_interval_seconds(T)

Only flashbulb sources created within `cascade_source_window_ms` of the
current signal participate. The explicit conversion preserves the
intended five- to sixty-minute window while stored timestamps remain
milliseconds since epoch.

## Synaptic Tagging and Capture

Flashbulb events are extended with **synaptic tagging**: high surprise
or arousal tags nearby memories for preferential consolidation later,
even if they scored low at the time.

    tag_trigger ← (surprisal_t > lerp(0.6, 0.4, S)) OR (arousal_t > lerp(0.7, 0.5, S))
    tag_window = round(lerp(2, 8, S))  # memories before/after current time
    tag_decay_s = lerp(300, 3600, T)   # tag lifetime (seconds)

    if tag_trigger:
        for m in temporal_neighbors(now_s(), tag_window):
            m.tagged ← true
            m.tag_strength ← 1.0
            m.tag_expires_at ← now_s() + tag_decay_s

Tagged memories receive a consolidation bonus:

    if m.tagged and now_s() < m.tag_expires_at:
        score_consolidate(m) += lerp(0.10, 0.25, S) × m.tag_strength

## Pattern Separation Sparse Index

Pattern separation is a lightweight sparse-key side channel maintained
for experimental indexing and procedural-value updates without adding a
decoder, taxonomy, or fact layer. For an accumulator or stored memory
embedding `x`, Cortext selects the `k` largest-magnitude dimensions,
preserves each selected dimension’s sign, sorts the selected indices,
and serializes the result as a stable sparse key:

    k_sparse(F,S,T) = RetrievalSparseKeySize(F,S,T)
    top_k(x) = indices of the k_sparse largest |x_i|
    sparse_key(x) = join(sort({ i || sign(x_i) : i in top_k(x) }))

The write path stores each committed memory id under
`index_store[sparse_key(e_rep)]`. The procedural update lane reuses the
same keying function but stores values rather than membership lists:

    index_store[sparse_key(e_rep)] += memory_id
    procedural_store[sparse_key(μ_acc)][memory_id] = Q(proc_key, memory_id)

Thus `index_store` is a processor-local sparse content-addressed bucket,
while `procedural_store` is a processor-local sparse context-to-memory
value table. Empty embeddings or non-positive key sizes produce no key
and therefore do not update either store. Neither map is currently
persisted, and production graph retrieval does not consume
`index_store`; dense SQLite/vector discovery remains the active seed
path.

## Procedural Memory Lane (Habit/Skill Memory)

In addition to declarative memory, Cortext maintains experimental
procedural values for which previously successful routine memory could
be surfaced in a context, learned from repeated successful use.

    proc_key ← sparse_key(μ_acc)  # same sparsification as @sec-pattern-separation
    Q(proc_key, memory_id) ← routine value

Updates occur when a retrieved memory is used successfully and
downstream outcome signals are positive:

    Q ← Q + value_update_gain × δ_reward_t × (1 − Q)

The current implementation updates this processor-local table but does
not read it in production graph retrieval, so it does not yet add
proactive seeds or surface action tokens. The historical
`procedural_proactive` ablation in
<a href="#sec-experimental" class="quarto-xref">Section 11</a> evaluated
an earlier research-branch path and was an exact null; it is not
evidence that proactive procedural retrieval ships today.

# Consolidation and Graph Integration

Consolidation now means explicit embedding replay over stored memories.
It uses a maintenance-only transaction path beside the normal signal
pipeline, not a semantic batch processor or a synthetic input signal.
The current hard-cut implementation removes the previous internal
generative consolidation stack and keeps only shallow graph
consolidation.

## Complementary Learning Systems Split

Cortext keeps the Complementary Learning Systems split at the level that
matters for the product:

-   **Fast episodic write path:** coherent memories are written
    immediately with source payloads, embeddings, timestamps, and graph
    metadata.
-   **Bounded live state:** working memory and soft-anchor state support
    immediate continuity without scanning the whole store.
-   **Slow durable graph:** explicit consolidation clusters older
    long-term memories during observed throughput troughs and writes
    stable graph structure.

The split is functional rather than model-based. No decoder is required
to transfer state from the fast path to the durable graph.

## Consolidation Triggers

Consolidation is external-only. The runtime never schedules it by
itself. Callers invoke `Consolidate()` or process a maintenance signal
with `Signal::force_consolidation=true`. This keeps production
scheduling under the application’s control while still letting the
engine derive an ordered urgency state from the three knobs.

Ordinary retained and ephemeral signals use the normal operation chain.
A forced request dispatches only `EvaluateConsolidation`, its score
gate, the clusterer, shallow replay, and graph construction. Maintenance
therefore does not create an episode, signal, working-memory row,
long-term memory, blob, or searchable embedding, and it does not update
normal throughput observations. The single `consolidation_state` output
is an ordered hint for the caller; it does not schedule maintenance work
by itself.

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

consolidation_state =
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
or peak update. Elapsed wall or event time and arbitrary memory counts
do not enter the state decision; backlog is only a positive-work guard.
An explicit force request starts consolidation independently and its
output state is `None`. A committed forced request acknowledges the
active excursion by persisting `armed=false`, even when no coherent
cluster exists. The acknowledgment resets the next event-derived range
to the current rate: both floor and peak become that rate. This prevents
a historical spike from suppressing recommendations after a throughput
regime shift. Ordinary throughput observations establish a new peak and
rearm the classifier only after the observed relative range is material:
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

## Candidate Selection

Consolidation operates on stored memory representatives, not raw
signals. Candidates are restricted to eligible `LONG_TERM` source
memories that have not already been assigned to the current
consolidation cluster. The clustering threshold, minimum size, and batch
cap derive from F/S/T:

``` text
merge_threshold = MergeThreshold(F)
min_cluster_size = MinClusterSize(F)
max_clusters = ConsolidationMaxClusters(F,S,T)
```

An explicit request does not broaden eligibility and does not
manufacture a nearest-neighbor mini-cluster. If fewer than the
knob-derived minimum form a coherent density cluster at
`MergeThreshold(F)`, replay is a successful no-op. This makes force mean
“evaluate now,” not “merge something regardless of evidence.”

## Shallow Replay Product

For each cluster, shallow consolidation writes one centroid
`ASSOCIATION` memory and durable graph edges:

``` text
association.embedding = centroid(cluster)
association.kind = ASSOCIATION
association.source_id = association_<timestamp>_<counter>
association.label = ""

for source in cluster:
  source.cluster_id = cluster.cluster_id
  emit association --derived_from--> source
```

Centroids are averaged over embeddings that actually contribute to the
centroid dimension. Candidates with empty or incompatible embeddings are
excluded from cluster membership instead of diluting or reshaping the
accepted centroid vector. Downstream label matching derives its expected
dimension from the first non-empty centroid in the replay batch.

If durable `LABEL` memories already exist, the association centroid can
attach to the nearest labels:

``` text
max_labels = ShallowConsolidationMaxLabels(F,S,T)
threshold = ShallowConsolidationLabelMinSimilarity(F,S,T)
emit association --has_label--> label when cos(centroid, label) >= threshold
```

This is intentionally opportunistic. The current engine does not
synthesize labels from a decoder, does not attach a static taxonomy
database, and does not promote provisional shadow labels. Existing label
nodes are treated as graph evidence only.

## Graph Build

After shallow replay, `BuildGraphFromConsolidation` reinforces durable
graph structure among the cluster products and sources. The graph
builder remains bounded by the candidate/cluster cap and knob-derived
edge conditions and weights. The graph products are useful for later
retrieval because graph expansion can recover clustered source memories
through `derived_from`, co-occurrence, similarity, reinforcement, and
causal edges. Graph construction runs only when shallow replay persisted
a cluster and loads only the cluster identifiers created by that
transaction. Repeating replay without newly eligible coherent memories
therefore does not rebuild historical clusters, replace their
timestamps, or decay reinforcement edges.

Association centroids are derived navigation nodes rather than
independent source observations. Direct semantic vector seeds are
restricted to `LONG_TERM` memories; `ASSOCIATION` nodes remain
retrievable through incoming or outgoing graph expansion. Retrieval
collapses direct long-term families first and commits the knob-bounded
direct-anchor prefix before graph-expanded families can compete for the
remaining positions. The same family index is then seeded with those
protected representatives, so a near-duplicate derived centroid cannot
remove an already protected source. This preserves graph utility without
allowing derived nodes to consume the direct-anchor budget. When a
current embedding exists for a memory, retrieval also skips scoring its
redundant historical base embedding; the authoritative current surface
was already inserted first.

## Reconstruction Surface Discipline

Constructive reconsolidation appends reconstruction records rather than
rewriting every historical embedding reference. When an append succeeds,
the bounded `current_memory_embeddings` surface must advance to the
newest reconstruction embedding. Retrieval reads that current surface
first, falling back to the latest reconstruction only when the cached
current row is stale.

This keeps graph retrieval aligned with the live memory surface while
preserving older signal rows, shared base embeddings, and sibling
memories that still point at the original embedding. Non-constructive
fallback updates still fork shared embeddings before writing in place,
so a reconsolidated memory does not mutate unrelated memories that
happened to share the same vector row.

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
-   fact-aware retrieval and eviction policies.

The removal is a hard cut, not a compatibility shim. New databases no
longer create the removed tables, public bindings no longer expose the
removed mode fields, and replay reports no longer emit the removed debug
metrics.

## Latency Discipline

The consolidation target is bounded sleep-like replay. The work is
dominated by SQLite candidate reads, vector centroid formation,
association writes, and graph edge updates. There are no hidden decoder
calls in the current path, so replay latency should scale with the
number of selected memories and cluster edges, not with source text
length or multimodal payload size.

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

**Note:** As defined in Section 1, all knob symbols here use the
midpoint‑biased values F̃ and S̃ (T is unmodified). For affective gain, we
use a lighter bias ( \_{affect} = (S, -0.06) ) to preserve mid‑range
affect modulation.

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

When the interrupt gate allows a retrieval outside a flush/spike event,
we mark a **pending abort** for the current accumulator to avoid
persisting partial thoughts. On the next signal, we compare similarity
to the selected memory versus the current accumulator centroid. If the
new signal aligns more with the selected memory, we treat the interrupt
as **accepted** and drop the partial unit; otherwise we **resume** the
accumulator. This prevents “half‑utterances” from being stored when a
memory whisper redirects attention, while still allowing the speaker to
ignore the interrupt and continue seamlessly. This uses only embeddings
already present in the system (no new constants).

    if allow_interrupt AND NOT should_flush AND NOT spike_bypass:
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
(<a href="#sec-boundary" class="quarto-xref">Section 6.4.3</a>) also
trigger retrieval checks to ensure context updates align with natural
thought transitions. Ephemeral query ingress bypasses pacing throttles
for the current signal only; storage remains controlled by the
retention-aware write gate.

# Experimental Results

This section records the evidence for the current hard-cut Cortext
branch. Older adapter-era and internal-decoder experiments are preserved
in git history, not in the current manuscript, because those systems are
no longer part of the runtime being evaluated.

## Current Evaluation Scope

The evaluated system is the embedding-first runtime described in
<a href="#sec-implementation" class="quarto-xref">Section 12</a> and
<a href="#sec-consolidation" class="quarto-xref">Section 9</a>. It
contains:

-   online memory storage, boundary detection, graph construction,
    retrieval, reconsolidation, working memory, soft anchors, and
    eviction;
-   explicit shallow consolidation over stored embeddings and graph
    edges; and
-   C, C++, Python, Go, Dart, JavaScript, and WebAssembly entry points
    that expose only the retained processing, retrieval, consolidation,
    reset, and embedding APIs.

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
from vector and graph evidence</td>
</tr>
<tr>
<td>paper consistency</td>
<td>this section, <a href="#sec-consolidation"
class="quarto-xref">Section 9</a>, <a href="#sec-implementation"
class="quarto-xref">Section 12</a>, and <a href="#sec-optimization"
class="quarto-xref">Section 13</a> describe the current branch rather
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
memory metadata alternatives behind one stored vector so a superseded
sibling cannot hide an eligible sibling. Retrieval fallback executes
demand-driven ordered pages over the current and historical vector
surfaces. Every SQL page applies kind, timestamp, and supersession
eligibility before distance ordering and a knob-derived top-k page
limit; one C++ cosine-family index spans the pages until the ordinary
seed budget is full or the surface is exhausted. Processor hydration
records persisted-current currency separately from processor-surface
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

## Retrieval Behavior After Removal

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
temporal scoring but no longer reads `pre_activation`, so that
predictive verdict does not describe current runtime behavior. Removing
the metacognitive layer was mildly positive and had no long-horizon
story to defer to, which supported cutting it in v1. Removing daily
consolidation was also mildly positive at this horizon, but we deferred
that arm because consolidation-family verdicts require long-horizon
runs. The remaining removal arms were null or long-horizon deferred as
labeled.

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

The confirmation reverses the preliminary oscillator-null read. Removing
the encode/retrieve oscillator costs 0.49 sufficiency, the largest July
neuromodulator loss, and all three matched repetition deltas are
negative, so that per-signal fast dynamic stays. Synaptic tagging also
clears the 0.15 signal floor after the targeting fix with the most
stable July delta (rep SD 0.03), and is retained on both correctness and
short-horizon quality grounds. Neuromodulator effect scales and the
emotion/mood threshold cascade remain directionally negative but below
the signal floor in this confirmation; neither earns a hard cut from
this screen. Effect scales is not worth more short-horizon spend, and
the emotion/mood cascade remains deferred because its slow mood
integrator has a plausible long-horizon role.

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

The long-horizon run closes the deferred cut question for v1: no removal
improves the stack, so no additional mechanism earns a hard cut. The
table therefore uses two verdict tiers rather than treating all negative
mean deltas as equally resolved. The emotion/mood threshold cascade and
STM/LTM graph-label handoff are retained on measured harm. Emotion/mood
loses on all three paired repetitions, and handoff has the most stable
long-horizon loss (rep SD 0.05).

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

### Negative Results Summary

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
paired-repetition resolution. These verdicts do not remove the
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

## Experimental Interpretation

The hard-cut branch changes the research question. Earlier results asked
how much quality came from online memory, graph retrieval, semantic
extraction, daily decoder-backed consolidation, and external blind
judging. The current branch asks a narrower product question:

> How much long-horizon utility remains when Cortext is restricted to
> embedding-space memory, graph structure, shallow replay, and bounded
> reconstruction?

The answer is not fully settled by unit tests. The current evidence
proves that the v1 runtime compiles, passes its retained behavioral
suite, no longer depends on deleted decoder paths, structurally matches
the preserved replay binary, and recovers the historical roughly-15/31
blind-judge level on frozen one-year sparse probes while using
substantially fewer context tokens. Broader quality claims still require
replicated harnesses against frozen probes, with evaluator state
isolated from the production library and with the repository treated as
the production engine under test.

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
working-memory maintenance, and soft-anchor formation all operate over
embeddings, bounded processor state, and SQLite graph rows. Raw text,
audio, and image payloads are used to produce embeddings and to hydrate
returned memories for applications; hydrated payloads do not feed back
into online ranking.

The hard-cut runtime no longer contains an internal generative model
stack. There is no semantic batch stack, adapter registry, mode-selected
replay, attached static taxonomy, persistent confidence monitor,
bitemporal fact store, or short-term shadow graph. The remaining model
dependency in the core loop is the configured multimodal embedding path.

Retrieved memories are hydrated only after retrieval has selected a
compact ranked set. Internal routing nodes such as `LABEL` or
`ASSOCIATION` rows are not treated as user-facing payloads. When such a
row reaches the retrieval surface, hydration resolves bounded linked
source memories where possible so callers see concrete source-backed
context instead of empty routing artifacts. A linked source keeps its
own relevance and composite score when it was independently ranked.
Otherwise query-backed hydration computes relevance from that source’s
own embedding and computes its composite display score from that
relevance and the retained relationship weight. Scores therefore
describe the displayed source result rather than borrowing the internal
routing node’s score. The same scored memory object is serialized by the
C API and JavaScript addon. An internal hydration call without a query
and without an independent source trace leaves both optional scores
unset; it does not relabel a centroid score as source relevance.

Linked-source selection is query-aware before truncation. SQLite
computes cosine relevance for every eligible member of the bounded
linked graph set. Multiple graph routes to the same memory are grouped
by memory id before ranking, so they cannot consume several compact
slots. SQLite then combines source relevance with the F/S/T-derived
relationship weight, orders that complete unique set, and only then
applies the knob-derived compact limit. This prevents either duplicate
routes or a recent but weakly related fragment from occupying the
transfer cap ahead of an older verbatim-relevant source, while keeping
embedding decoding and application-side hydration bounded to the
selected rows.

## Durable Memory Surface

The current v1 runtime uses two durable graph layers plus one bounded
live layer:

-   **Working memory:** active coherent memories used for recent-turn
    continuity, overlap suppression, and prompt reconstruction.
-   **Long-term graph memory:** persisted `memories`, `signals`,
    `embeddings`, `associations`, reconstruction rows, eviction audit
    rows, and optional soft-anchor state.
-   **Soft anchors:** uncertainty-preserving continuity links formed at
    ingress and exposed as optional memory metadata.

Long-term graph memory is the durable substrate. `memories` stores
working, long-term, label, and association nodes; `associations` stores
graph edges such as `derived_from`, `has_label`, `similar_to`,
`co_occurs`, `causes`, and `reinforces`. Existing `LABEL` nodes can
still participate in shallow consolidation and hydration, but Cortext no
longer creates labels from an internal decoder or an attached static
taxonomy.

`source_id` is treated as opaque provenance and exact-equality grouping
only. Cortext does not infer application roles, speaker names,
modalities, maintenance commands, or consolidation behavior from
reserved source strings. Explicit maintenance is driven by API calls and
`Signal::force_consolidation`, not magic source identifiers.

SQLite parameters are checked at bind time, including exact
statement/argument counts, so `SQLITE_TOOBIG`, `SQLITE_RANGE`, and
allocation failures cannot be silently converted into NULL values.
`Process()` and `Flush()` both snapshot mutable processor state until
the database transaction commits; rollback restores working-memory dirty
flags and persisted-record cursors. Association writers invalidate the
fanout cache in constant time and rebuild it lazily on the next
traversal.

## Current Operation Pipeline

The production operation chain is:

``` text
CoreStage:
  embedded centroid init, priors, coherence, accumulation, drift,
  uncertainty, focus/sensitivity/mood, metrics, neuromodulators,
  adaptive metric weights, thresholding, recent context, boundary/write gate

StorageStage:
  memory storage, soft-anchor update, synaptic tagging, signal metrics

RetrievalStage:
  streaming pacing, graph-augmented retrieval, rate/MNI gating,
  memory-usage detection

FeedbackStage:
  retrieval competition, predictive pre-activation, reconsolidation,
  focus/sensitivity/stability feedback, influence feedback,
  serial-position effects, memory strength/eviction, emotional consolidation,
  working memory, accumulator reset, consolidation gate/cluster/shallow,
  graph build, emotional cascade, meta-learning
```

Probe mode intentionally runs only `CoreStage` and `RetrievalStage` so
replay probes can inspect retrieval without mutating the durable store.

## Consolidation Implementation

Consolidation is explicit. The runtime does not schedule it
automatically and does not expose mode variants. Callers invoke
`Consolidate()` or inject a signal with `force_consolidation=true`; a
signal dispatcher then runs only the knob-derived consolidation gate,
clusterer, shallow consolidation, and graph builder. The normal core,
storage, retrieval, feedback, episode, object-store, working-memory,
signal-counter, and rate-observation paths are not maintenance steps.

Shallow consolidation is embedding-only. It clusters eligible long-term
memories, writes an `ASSOCIATION` centroid node for each cluster,
assigns source memories to the cluster, and emits `derived_from` edges.
If durable `LABEL` nodes already exist, shallow consolidation can attach
the closest labels to the association centroid with F/S/T-derived limits
and similarity thresholds. It does not synthesize payload summaries,
extract relations, create fact assertions, call a decoder, or attach a
read-only external taxonomy.

This keeps replay close to the current product target: consolidation
behaves like low-latency sleep replay over stored embeddings and graph
edges, not a batch semantic extraction job.

## Retrieval Implementation

Production retrieval discovers a broad but bounded vector seed set from
durable memory embeddings. Before the seed cap, it removes directed
`supersedes` targets and collapses near-duplicate embedding families to
one representative. The family index uses complementary 32-block
Cauchy–Schwarz upper bounds in the stored coordinate system and a
normalized Walsh–Hadamard basis, followed by exact cosine checks. Each
upper bound is sound, so taking their minimum cannot exclude a pair
meeting the cosine threshold. The complementary bases avoid broad exact
pairwise work for both sparse coordinate-aligned vectors and dense
Hadamard-aligned vectors; arbitrary adversarial surfaces may still reach
the exact fallback. If the private vector cache is unavailable, durable
processing rebuilds it once from the complete persisted surface and
immediately applies the same family and supersession filters before
truncation. A failed rebuild installs a failure sentinel rather than
repeating the two complete surface scans on every request. A unique
cached vector retains every long-term memory metadata alternative that
shares its base embedding, and query-time eligibility chooses a
nonsuperseded sibling without duplicating vector storage. The fallback
executes a demand-driven sequence of ordered SQL pages over the current
and historical vector surfaces. Each page joins memory metadata and
applies kind, timestamp, and supersession eligibility before distance
ordering, excludes a historical row when that memory already has a
current reconstruction, collapses byte-identical historical vectors, and
uses a second application of the existing F/S/T-derived seed-search
breadth as its row bound. One in-memory cosine-family index spans every
page; paging stops when the ordinary semantic-family seed budget is full
or the eligible surface is exhausted. Processor hydration tracks
persisted-current-surface currency and processor-surface completeness
independently. The private current-vector search cache is built and
incrementally updated from the processor’s latest reconstruction
surface, even when a test or storage policy intentionally omits the
corresponding persisted-current write. A valid cache therefore performs
distance ordering and cosine-family accounting on latest vectors. If
that cache is unavailable while the processor surface is complete,
durable retrieval rebuilds a current-only cache from the processor
surface and ephemeral retrieval scans that surface directly without
mutating the registry. Only an incomplete processor surface falls back
to SQL; in that recovery case the query substitutes each latest
reconstruction before cross-page family accounting. Base families that
converge after reconstruction therefore consume one current-family slot
and cannot stop selection early. Current per-memory reconstructions
remain distinct rows and the corresponding base row for that same memory
is excluded. When constructive recall is disabled, both restart
hydration and cache-loss rebuild select the base embedding and ignore
reconstruction/current surfaces. This recovery path preserves an
eligible target behind 900 closer ineligible rows and preserves an
eligible sibling of a superseded shared embedding. A separate 600-member
byte-distinct cosine-near family requires two bounded pages and cannot
hide the next semantic target. The ordinary 430-family regression
remains one page and returns no more than the 477-row default page bound
rather than materializing all 1,331 eligible rows. The embedded
sqlite-vec implementation is brute-force, so every requested page scans
the persisted vector surfaces; the optimization bounds rows returned to
C++ and downstream embedding decoding, not the scan’s asymptotic order
or the historical window partition’s internal sort and materialization
work. Ephemeral retrieval uses the fallback without installing volatile
registry state. It then expands through retained structural graph edges
and returns a compact selected set. Vector similarity owns the rank;
recency is a tie-break and graph activation is a bounded residual bonus.
`source_id` and modality are metadata, not ranking inputs. Direct vector
seeds are source `LONG_TERM` memories; derived `ASSOCIATION` centroids
enter through graph expansion. Direct families are ranked and the
protected direct prefix is committed before expanded families are
admitted; expanded candidates are then deduplicated against those
protected representatives. When a current embedding exists, its
redundant historical base embedding is omitted from the historical seed
search because the authoritative surface was already searched. Current
writes are excluded by timestamp, and working-memory overlap filters
prevent the retrieval result from echoing the active memory tail.
Association edges are used as graph evidence; they do not bypass final
scoring or output caps. One dispatch exposes at most one actually used
retrieval candidate, so retrieval does not synthesize reinforcement pair
edges from the packet. Existing `reinforces` edges remain eligible for
decay, pruning, degree normalization, and graph evidence. Historical
public knob helpers for the removed packet-pair writer remain
source-compatible calibrations; only the step calibration still feeds
the production pruning floor.

The retrieval trace path records a ranking ledger for selected and
rejected candidates. Production graph retrieval populates composite
score, seed relevance, temporal score, and the base/spreading fields of
the activation ledger. The shared trace structure still contains
reserved processor, predictive, pre-activation, durable-source, and
evidence fields, but this path leaves them at zero; their presence is
not evidence that those terms participate in ranking. Deleted
label-graph and fact-layer fields are not emitted.

`Retention::Ephemeral` executes this read path inside a transaction,
assembles the retrieval result, rolls back the transaction and object
transaction, and restores the exact pre-query processor snapshot.
Retrieval-time reconstruction, usage reinforcement, reconsolidation, and
strength mutation are skipped for that retention policy, so querying
cannot change durable rank state.

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

## Soft Anchor Implementation Status

Soft Anchor
(<a href="#sec-soft-anchor" class="quarto-xref">Section 8.4</a>) is
wired into the engine as an always-on ingress formation path. It is not
a hard referent resolver and it is not a retrieval-time reranker. The
operation runs after memory storage, consumes the stored memory id and
current representative embedding, and compares the current signal only
against prior soft-anchor state.

When a model exposes a 1536-dimensional soft-anchor embedding, the
runtime uses the semantic/entity/full slices described in
<a href="#sec-soft-anchor" class="quarto-xref">Section 8.4</a>. When
only the standard 256-dimensional retrieval embedding is available, it
falls back to a single normalized view for all three centroids and
lowers entity quality with a knob-derived prior.

Schema migration 6 adds:

``` text
soft_anchors(anchor_id, status, semantic_centroid, entity_centroid,
             full_centroid, semantic_radius, entity_radius, full_radius,
             source_id, first_step, last_step, last_boundary_id,
             first_ts, last_ts, anchor_strength, support_count,
             contradiction_count, recent_memory_ids, updated_at)

soft_anchor_links(memory_id, anchor_id, anchor_strength, anchor_label,
                  evidence_kind, memory_tier, score, margin, entropy,
                  support_count, contradiction_count, created_step,
                  updated_step, created_at)
```

Hydrated memories expose up to three soft-anchor entries through
`Context::Memory::soft_anchors`, and the C API JSON mirrors those
entries under each returned memory. Production retrieval does not use
those links as ranking features, and no bundled chat renderer is part of
this claim.

## Computational Complexity

The dominant per-signal cost is exact candidate discovery over the
durable embedding and association surface. With `N` stored memories and
embedding dimension `d`, the current write path computes exact
current/base distances in `O(Nd)` and performs `O(N)` bounded-top-k
selection before writing a knob-capped number of supersession edges.
Graph expansion after seed discovery is bounded by F/S/T-derived row and
fanout limits. Working-memory and soft-anchor updates operate over
bounded live state and remain independent of total store size.

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

# Performance Optimization

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
  -> optional shallow consolidation on explicit request
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
<td>vector ranking</td>
<td>bounded top-k selection and architecture SIMD kernels in the vector
path</td>
<td>same candidate contract; SIMD accepted after output checks</td>
</tr>
<tr>
<td>retrieval seed surface</td>
<td>current-memory embedding lookup with pre-cap supersession filtering
and complementary sound 32-block Cauchy–Schwarz bounds in the stored and
Walsh–Hadamard bases followed by exact cosine family checks; shared
vectors retain all long-term metadata alternatives; the private search
cache follows the processor’s latest reconstruction surface
independently of persisted-current materialization; cache loss uses the
complete processor surface directly or, only when that surface is
incomplete, metadata-eligible, distance-ordered, F/S/T-bounded SQL pages
feeding one cross-page cosine-family index that substitutes latest
reconstructions before accounting</td>
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
<td>retained traversal types: <code>co_occurs</code>,
<code>similar_to</code>, <code>reinforces</code>, <code>causes</code>,
<code>derived_from</code>, <code>next_in_episode</code>,
<code>prev_in_episode</code>, and <code>within_same_event</code>;
<code>reinforces</code> is degree-normalized</td>
<td>graph activation is bounded and high-degree hubs cannot dominate;
directional <code>supersedes</code> relations define families but are
not activation edges</td>
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
<td>exact durable-neighbor ranking</td>
<td>maintain private current/base embedding surfaces in a process-wide
lifecycle registry, including signal-only rows that affect the
sqlite-vec population; rank into typed borrowed/owned candidates; reuse
the query norm across exact cosine checks</td>
<td>the registry follows processor teardown across sequential thread
migration and mirrors authoritative SQL write decisions; the same
candidate population, K, filters, tie order, cosine results, thresholds,
and supersession edges are retained; SQL remains the fallback and no
public header or API layout changes</td>
</tr>
<tr>
<td>graph retrieval</td>
<td>score vector similarity first, restrict recency to a tie-break, cap
graph bonus, and reserve final occupancy for direct semantic
anchors</td>
<td>recent unrelated rows and noisy graph neighbors cannot displace the
semantic seed surface</td>
</tr>
<tr>
<td>rollback snapshots</td>
<td>detach rebuildable durable caches before copying processor state;
ephemeral queries roll back their transaction and restore that exact
snapshot without rebuilding the unchanged search surface</td>
<td>generic failures and public no-store queries restore durable and
volatile processor state exactly</td>
</tr>
<tr>
<td>synaptic tagging</td>
<td>use indexed temporal candidate supersets with boundary ties before
applying the original exact score and order</td>
<td>the same nearest source-backed memories are tagged without a
whole-source group scan</td>
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

## Standard Eviction-Frontier Estimate

The standard file-backed pressure gate is 500,000,000 bytes. Exact
databases occupied 10,014,720 bytes at 500 messages, 18,169,856 bytes at
1,000, and 48,857,088 bytes at 2,636. The full-span storage slope
projects the gate at approximately 27,445 messages; the observed
interval slopes give a planning range of roughly 26,700–30,500 messages.

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

# Conclusion and Future Directions

We have presented Cortext, a three-knob adaptive memory architecture
that unifies working, episodic, and semantic memory processes under a
continuous control regime. By deriving most system parameters from
Focus, Sensitivity, and Stability, while fixing a small set of
invariants for controller stability, Cortext avoids the fragility of
hard-coded constants and enables robust operation across diverse
environments.

Key contributions include the formalization of knob-derived parameter
spaces, the integration of uncertainty-weighted Bayesian adaptation, and
the implementation of bio-inspired mechanisms like homeostatic rate
control and emotional consolidation.

Future work will focus on:

1.  **Replicated long-horizon evaluation:** Validating the architecture
    on frozen public corpora and larger stress runs.
2.  **Soft Anchor consumption:** Testing whether uncertain continuity
    hints improve downstream context without turning tentative evidence
    into hard assertions.
3.  **Hardware acceleration:** Optimizing vector search, graph
    traversal, and SQLite access paths for edge devices.

Cortext represents a step toward more organic, life-long learning
systems that adapt continuously to their experiences, moving beyond
static knowledge bases toward truly cognitive memory.

# References

Anderson, Michael C, Robert A Bjork, and Elizabeth L Bjork. 1994.
“Remembering Can Cause Forgetting: Retrieval Dynamics in Long-Term
Memory.” *Journal of Experimental Psychology: Learning, Memory, and
Cognition* 20 (5): 1063.

Åström, Karl Johan, and Richard M Murray. 2008. *Feedback Systems: An
Introduction for Scientists and Engineers*. Princeton university press.

Baddeley, Alan. 2000. “The Episodic Buffer: A New Component of Working
Memory?” *Trends in Cognitive Sciences* 4 (11): 417–23.

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

LaBar, Kevin S, and Roberto Cabeza. 2006. “Cognitive Neuroscience of
Emotional Memory.” *Nature Reviews Neuroscience* 7 (1): 54–64.

Liu, Jun S, and Rong Chen. 1998. “Sequential Monte Carlo Methods for
Dynamic Systems.” *Journal of the American Statistical Association* 93
(443): 1032–44.

McClelland, James L, Bruce L McNaughton, and Randall C O’Reilly. 1995.
“Why There Are Complementary Learning Systems in the Hippocampus and
Neocortex: Insights from the Successes and Failures of Connectionist
Models of Learning and Memory.” *Psychological Review* 102 (3): 419.

McCloskey, Michael, and Neal J Cohen. 1989. “Catastrophic Interference
in Connectionist Networks: The Sequential Learning Problem.” In *The
Psychology of Learning and Motivation*, 24:109–65. Elsevier.

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

Russell, James A. 1980. “A Circumplex Model of Affect.” *Journal of
Personality and Social Psychology* 39 (6): 1161.

Tulving, Endel. 1972. “Episodic and Semantic Memory.” In *Organization
of Memory*, 1:381–403. Academic Press.

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
    `mood_vector = 0_vector`, `last_mood_ts = now_ms()`,
    `theta_dynamic = theta_target = θ_prior(F,S,T)`,
    `hysteresis = lerp(0.02, 0.25, T)`, `half_life = base_half_life(T)`,
    `m_rate = 0`, `rho_hat_prev = 0`, `dt_ema = 0`, `rate_ticks = 0`,
    `reliability = 1`, `retention_ema = 0`,
    `last_rate_timestamp = now_ms()`, `last_retrieval_ts = 0`,
    `last_embedding = unset`, `x_pred_ema = unset`, `outcome_pred = 0`,
    `φ_t = 0`.

-   **Accumulator defaults (per stream):** `μ_acc = 0_vector`,
    `c_t = 0_vector`, `drift_acc = 0`, `s_sum = 0`, `s_max = 0`,
    `n = 0`, `e_peak = 0_vector`, `emo_max = 0`, `arousal_sum = 0`,
    `acc_signals_window = []`, `t_start = 0`, `last_signal_ts = 0`,
    `last_write_ts = 0`, `eta_acc = 0`, `coherence_prev = 0`,
    `drift_accum = 0`, `drift_at_last_interrupt = 0`,
    `drift_acc_pacing = 0`, `x_last_check = unset` (μ_acc),
    `prev_x = unset` (μ_acc).

-   **Per-memory defaults (on insert):**
    `strength = memory_initial_strength(F,S,T)`,
    `trace_* = memory_initial_trace_policy(F,S,T)`, `use_frequency = 0`,
    `stability = memory_initial_stability(F,S,T)`, `connectivity = 0`,
    `drift_mag = 0`, `influence = 0`, `sustained_influence = 0`,
    `contextual_gain = 0`, `redundancy = 0`, `pre_activation = 0`,
    `lability_state = 0`, `suppression_count = 0`, `suppression = 0`,
    `flashbulb = 0`, `s_emotion_max = 0`, `s_arousal_avg = 0`,
    `boundary_score = 0`, `tagged = false`, `tag_expires_at = 0`,
    `context = 0_vector`,
    `source_model = {origin: "source", reliability: source_reliability_0(F,S,T), contradiction_count: 0, last_verified_ts: 0}`.
    Application provenance remains in the separate opaque `source_id`.
    SQLite defaults remain migration/backstop values for legacy rows.
    Evidence packets live in ordered `signals` rows, and the
    constructive-recall ledger starts empty until the initial
    reconstruction row is appended in `memory_reconstructions`.

-   **RLS defaults:** `w_* = w_bootstrap`, `P = diag(P_init(T))` with
    `P_init(T) = lerp(500, 2000, 1 − T)`, `blender_ready = false`,
    `blender_update_count = 0`.

-   **Knob-derived control parameters (initialized from priors):**
    `weight_relevance`, `attention_width`, `coverage_gain_floor`,
    `mismatch_weight`, `weight_novelty`, `weight_surprise`,
    `weight_valence`, `weight_arousal`, `emotion_gain`, `score_gain`,
    `rate_target`, `rate_decay`, `periphery_half_life`,
    `salience_half_life`, `drift_weight`.

-   **Buffers:** `signal_stream` (μ_acc stream), `score_stream`,
    `memory_stream`, `recent_memory_centroids`, `index_store`,
    `procedural_store` start empty.

-   **Retention history:** `retention_history` starts empty (populated
    after the first signal step).

-   **Accumulator state (per stream; retained across timesteps):**

    -   `{μ_acc, c_t, drift_acc, s_sum, s_max, n, e_peak, emo_max, arousal_sum, eta_acc, coherence_prev, acc_signals_window, t_start, last_signal_ts, last_write_ts, drift_accum, drift_at_last_interrupt, drift_acc_pacing, x_last_check, prev_x}`

-   **Global state (retained across timesteps):**

    -   `{signals_processed, u_uncertainty, mood_vector, last_mood_ts, theta_dynamic, theta_target, hysteresis, half_life, m_rate, rho_hat_prev, dt_ema, rate_ticks, reliability, retention_ema, last_rate_timestamp, last_retrieval_ts, last_embedding, x_pred_ema, outcome_pred, φ_t, weight_relevance, attention_width, coverage_gain_floor, mismatch_weight, weight_novelty, weight_surprise, weight_valence, weight_arousal, emotion_gain, score_gain, rate_target, rate_decay, periphery_half_life, salience_half_life, drift_weight, blender_ready, blender_update_count, blender_P}`

-   **Buffers (retained across timesteps; bounded by window rules):**

    -   `{signal_stream, score_stream, memory_stream, recent_memory_centroids, index_store, procedural_store, retention_history}`

-   **Recorded signal fields (per signal):**

    -   `{coherence_struct_t → SIGNALS.coherence, focus_spread_t → SIGNALS.focus_spread}`

-   **Recorded global fields:**

    -   `{u(t) → STATE.u_uncertainty, M_t → STATE.mood_vector}`

-   **Per-step derived scalars/vectors (ephemeral; recomputed each
    step):**

    -   `{signal_gap_s, coherence_curr, s_avg, S_window, boundary_score, max_signals, max_signal_flush, should_flush, write_memory, Δwrites, q_retrieval, ACh_t, NE_t, DA_t}`

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
-   **recent_scores empty:** set `observed_p90 ← θ_prior` in threshold
    updates.

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
    update_focus_sensitivity_mood_and_effective_focus()
    metrics_t ← compute_structural_and_affective_metrics()
    update_neuromodulators_and_metric_weights()
    score_t ← compute_composite_score(metrics_t)
    update_accumulator_scores(score_t, μ_acc)
    update_precision_delta_and_threshold()
    recent_context ← append_recent_context(μ_acc)
    boundary_score, should_flush ← detect_boundary(capacity_pressure, inactivity_boost)
    spike_bypass ← check_spike_bypass(score_t, θ_dynamic, mem_maturity, coherence_t)
    write_memory ← compute_write_gate(force_write, should_flush, spike_bypass)
    Δwrites ← 1 if write_memory else 0

    # StorageStage
    stored_memory_id ← maybe_commit_memory_unit(write_memory)
    update_soft_anchor(stored_memory_id)
    apply_synaptic_tagging(stored_memory_id)
    persist_signal_metrics(stored_memory_id)

    # RetrievalStage (q_retrieval is captured before any accumulator reset)
    q_retrieval ← l2_normalize(μ_acc)
    streaming_pacing_check()
    graph_retrieval(q_retrieval)
    update_rate_state(Δwrites)
    compute_mni_gate_decision()
    detect_memory_usage_and_update_reinforcement()

    # FeedbackStage
    update_retrieval_competition()
    apply_predictive_preactivation()
    apply_reconsolidation()
    apply_focus_sensitivity_stability_feedback()
    apply_influence_feedback()
    apply_serial_position_and_strength_effects()
    run_emotional_consolidation()
    update_working_memory()
    reset_accumulator_after_flush_or_interrupt()
    if signal.force_consolidation:
        evaluate_consolidation()
        gate_cluster_and_shallow_replay()
        build_memory_graph()
    update_emotional_cascade()
    update_meta_learning()

The normative execution order for a single timestep t:

1.  **Read Inputs:** `x_t`, `now_s()`, `system_time_ms`.
2.  **Run CoreStage:** initialize priors, update accumulator/drift,
    compute uncertainty, adaptation, metrics, neuromodulators, composite
    score, threshold, boundary, spike bypass, and write gate.
3.  **Run StorageStage:** write the memory unit when the write gate
    opens, then update Soft Anchor, synaptic tagging, and signal metric
    persistence from the stored memory id.
4.  **Run RetrievalStage:** cache `q_retrieval ← μ_acc` before any
    reset, run streaming pacing and graph retrieval, update rate state
    from the current write decision, compute the MNI gate, and record
    memory usage/reinforcement.
5.  **Run FeedbackStage:** apply retrieval competition, predictive
    pre-activation, reconsolidation, F/S/T feedback, serial-position and
    strength updates, emotional consolidation, working memory,
    accumulator reset, optional forced consolidation, graph build,
    emotional cascade, and meta-learning.
6.  **Interrupt Abort:** If the MNI gate marked a pending abort, the
    next signal compares similarity to the selected memory against the
    current unit and resets only when the new signal aligns more with
    the selected memory.

Timing notes:

-   `now_s()` is captured at step 1 and reused for all Δt computations
    in this timestep.
-   Threshold updates in `CoreStage` use the score computed earlier in
    the same stage.
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
