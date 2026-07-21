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
by itself. In addition to the throughput classifier below, the private
active-epoch store raises `Required` when its event, mutation, or
allocated-byte safety ceiling is reached. That boundary is source- and
modality-independent, remains asserted if ignored, and is cleared only
after a successful forced-maintenance commit publishes a fresh epoch.

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

Natural and Durable retention enter the same operation chain and
accumulate the current signal before boundary evaluation. Natural leaves
boundary and write acceptance to the accumulator algorithms. Durable
adds an explicit boundary and accepted flush of that same pending unit;
it does not construct a second single-signal memory alongside the
accumulated unit. The Durable packet is the last signal in the flushed
unit exactly once.

### Lazy retrieval-inhibition state and the active epoch

Migration 28 keeps the existing `memories` columns but adds a private
SQLite recovery clock, generation resets, per-memory anchored inhibition
state, an expiry index, and a connection-local temporary effective-value
view. The persistent migration inventory is exactly the three state
tables plus the expiry index; it does not install durable views or
triggers. Retrieval competition advances one global log-recovery clock
per event, materializes at most one knob-derived row batch whose exact
suppression has crossed the existing 10<sup>−9</sup> active threshold,
and updates only newly suppressed losers. A full-recovery interval
advances a generation; older rows are materialized in the same
knob-derived batches and reset markers are pruned after their generation
has no remaining active rows. Internal reads that depend on strength,
suppression, or the suppression timestamp use the temporary effective
view. Compatibility handling conservatively materializes and rebuilds
the private representation around every caller-supplied SQL statement
because aliases, wildcards, views, subqueries, and triggers cannot be
classified safely from SQL text. A caller mutation also invalidates
every database-derived processor cache before the statement executes.

The engine’s structured competition records carry `memory_id`, so
ordinary retrieval-induced inhibition remains scoped to exactly one
losing memory. Legacy callers may instead provide only an embedding
identity. Because several memories may share that embedding,
compatibility resolution enumerates every matching memory in stable
identity order and applies the same suppression to each one. This
prevents one arbitrary sibling from escaping inhibition, but it also
means the embedding-only compatibility edge may perform work
proportional to the shared-embedding membership. It is not included in
the fixed memory-scoped operation claim.

On startup from schema 27, migration may anchor active rows that carry
different historical suppression timestamps to one recovery clock. The
processor calibrates those rows transactionally and rebuilds the
disposable active epoch before working-memory hydration reads the
effective-value view. This ordering prevents restart from seeding a
working slot with an uncalibrated strength and subsequently persisting
that stale value.

Consolidation candidate selection also consumes effective rather than
stale materialized RIF strength. It merges an indexed, at-most-*A*
inactive-memory frontier with an at-most-*A* exact active-RIF frontier,
then retains the lowest *A* combined identities for scoring. Before the
active branch is evaluated, the implementation counts at most 64*C* + 1
active rows and fails closed when the existing knob-derived active-epoch
mutation ceiling 64*C* has been exceeded. Consequently the indexed
inactive scan can skip at most 64*C* active rows, the active calculation
examines at most 64*C* rows, and downstream association work still sees
at most *A* identities. A recovered active row therefore cannot consume
a low-strength slot merely because its durable `memories.strength` has
not yet been materialized.

The authoritative signal mutation still commits once to persistent
SQLite. After that commit, a separate connection-local in-memory SQLite
database publishes only the current recovery clock and the identities
touched in the current engine epoch. Persistent SQLite may retain more
active inhibition identities than this disposable epoch projection.
Suppression, recovery-total, anchor, expiry, and effective-value fields
remain solely in persistent SQLite; no production consumer reads a
duplicated RIF value from the disposable identity ledger. Strength-only
changes that cannot alter active membership therefore do not republish
it. The ledger never mirrors durable memory, signal, embedding,
association, or payload history. A failed persistent commit restores the
pre-signal processor snapshot; a failed post-commit epoch publication
discards and rebuilds the disposable database from persistent authority
without replaying the committed signal. Natural returns after
publication. Durable follows the identical path and then executes only
the named WAL checkpoint barrier.

The active epoch records event, changed-row, allocated-byte, and maximum
statement-row measures. Its harness safety ceilings are derived from the
same clamped F/S/T route capacity and row batch

*C*(*F*, *S*, *T*) = round (256 + 256*F* + 128*S* + 128*T*),   *B*(*F*, *S*, *T*) = round (64 + 64*F* + 32*S* + 32*T*).

events are bounded by *C*, mutations by 64*C*, and allocated bytes by
131, 072*C*. Persistent active-row loads, in-memory SQLite publication,
recovery calibration, and retired-generation materialization use
statements of at most *B* rows. Thus all-low, midpoint, and all-high
resolve respectively to 256/16,384/32 MiB/64 rows, 512/32,768/64 MiB/128
rows, and 768/49,152/96 MiB/192 rows. The midpoint values are defaults,
not fixed production constants. Reaching an epoch ceiling raises the
existing `Required` consolidation hint before an honoring caller can
enter another event. Ignoring the hint neither starts another epoch nor
drops mutations. A live *B* + 1 mutation set is completed by a second
independently *B*-bounded statement; *B* is a statement bound, not a
history cap. This is distinct from historical sparse-route backfill,
whose logical *B* + 1 completion probe does not read or process the
additional historical row. Any epoch overflow observation is named as
the resolved limit plus one and is not an additional work-batch size. A
successful explicit consolidation commits persistent SQLite and then
starts a fresh in-memory SQLite epoch containing only the current clock.
It does not copy the persistent active-identity population into the new
epoch. The next ordinary event publishes only its changed identities. A
failed consolidation leaves the old epoch intact, while a publication
failure after persistent commit discards and recreates the same empty
epoch without replaying the durable mutation. The retry intent is sticky
until that empty epoch is published: even if both publication and
immediate recovery fail, the next processor event executes the empty
reset rather than a full persistent-history rebuild. Retrieval-surface
reloads preserve the RIF epoch sidecar across their independent cache
refresh, including the exception path after a failed reload, so
consolidation cannot erase that retry intent after the durable commit.
These are harness-scoped ceilings, not a claim that every production
workload is globally bounded. Non-finite F/S/T values fail before
integer capacity derivation; finite out-of-range values retain the
engine’s established clamping to \[0, 1\].

The row-batch contract is exercised structurally over all 27 points in
{0, 0.5, 1}<sup>3</sup> and through live *B* + 1 publication at nine
production-shaped points: midpoint, both joint endpoints, and each
one-axis endpoint. The latter varies text, audio, image, shared-source,
and opaque-source labels while requiring the same resolved batch and the
same two-statement completion. Thus 128 is only the neutral-knob result,
and 129 is only its derived *B* + 1 boundary.

The consolidation-reset regression uses the same nine knob points and
seeds a knob-derived *B* + 1 persistent active population with mixed
text, audio, image, shared-source, and opaque-source labels. At every
point the durable population survives consolidation, the new disposable
epoch contains zero copied identities, and the following event publishes
exactly its one changed identity. An additional failure-chain regression
seeds a neutral *B* + 1 = 129 mature durable population, injects
failures into publication, immediate recovery, and the following
retrieval-surface reload, and requires the next event to observe zero
copied rows before publishing exactly one changed identity. The
persistent 129-row population remains authoritative; it is never
converted into a 129-row statement. The reload exception was first
reproduced as a missing sidecar before the catch path was repaired. This
removes store-sized work from the successful-consolidation publication
edge. It does not yet bound initial processor startup: compatibility and
restart recovery may still reconstruct current-generation identity and
calibration state from persistent SQLite, so bounded whole-engine
restart remains an explicit open gate.

The same derivation owns the integrated SQLite HNSW maintenance and
consolidation frontiers. Public route inspection and queue effort reset
to 8*C* after consolidation, advance by *R* = max (2, ⌊*B*/16⌋) per
retrieval-active query, and saturate at 9*C*; downstream activated
identities are bounded by *A* = 2*C* + 2*B*; graph degree is
*N* = max (8, ⌊*B*/2⌋); search expansion is *E* = max (8, ⌊*B*/4⌋); and
ordinary route sealing occurs at *B* − *E*. Score consolidation reads at
most *A* indexed low-strength identities and at most *N* association
edges for each admitted identity. Natural and Durable execute these same
operations and mutations; Durable adds only the existing post-commit
flush/checkpoint barrier. None of these bounds branches on modality or
source identifier.

The pinned HNSW implementation retains its SIMD distance kernels, while
a content-checked local patch bounds every speculative neighbor prefetch
by the actual adjacency length. This prevents a sanitizer-visible
one-past-end read and skips the zero-byte adjacency copy whose
empty-vector destination may be null, without changing candidate
membership, adjacency contents, or ranking. Cache-byte estimation also
performs its knob-derived multiplication explicitly in `std::size_t`,
keeping the same capacity on native and WebAssembly targets instead of
relying on an implicit 64-to-32-bit narrowing conversion.

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
visited-node, maximum public queue, shadow-cache, construction-node, and
construction-queue limits are respectively 2*C*, 9*C*, 9*C*, 24*C*,
*C* + *B*, and 2*B*. The 9*C* quantity is a hard routing-inspection
ceiling, not the activated working set; the cache’s node map and FIFO
order are bounded together, and a successful seal clears both so
invalidated rows cannot leave accumulating order entries. After exact
reranking, no more than *A* identities may proceed into eligibility,
family collapse, graph expansion, and final ranking. Exact cosine family
checks have a separate 2*C* per-event ceiling. The breadth-first
adjacency fetch batch is *m**a**x*(8, ⌊*B*/4⌋). The larger query and
bounded cache coefficients were selected by copied-late quality probes:
2*C*/*C* and 3*C*/2*C* search candidates missed the exact-identity gate;
8*C*/4*C* with a 16*C* cache passed on the initial store but missed one
exact top-1 after the bounded supersession surface changed the live
graph. A later expanding 5*C* → 12*C* envelope restored exact sampled
quality but made both query and construction work grow with query age.
The retained candidate resets public route inspection to 8*C*, advances
it by *R* only until the 9*C* ceiling, fixes downstream activation at
*A* = 2*C* + 2*B*, and reserves the smaller *C* + *B*/2*B* envelope for
graph construction. These are knob-derived finite limits, not midpoint
constants.

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
eligibility-complete seed surface. Routing nodes such as `ASSOCIATION`
memories may consume activated slots and are then removed by the public
long-term seed predicate. If that filtering leaves fewer than the
requested candidate limit, the operation discards the underfilled sparse
proposal and executes the existing exact SQL seed path. Consequently the
fixed sparse envelope is not misrepresented as complete after
operation-specific filtering; the fallback may still perform store-sized
exact work.

Decoded activation-snapshot rows reuse the route’s existing
generation-qualified shadow-node cache rather than being selected and
decoded again for every query. The first query after reopen loads only
snapshot identities absent from that cache; later queries reuse those
immutable decoded rows while continuing to compute query distance
exactly. The shared cache remains bounded by 24*C*, and every successful
seal clears both its node map and FIFO order before any row from the
next authoritative SQLite state can be reused. A separate physical
snapshot-cache-miss counter distinguishes the fixed *A* rows evaluated
from the rows actually loaded from SQLite. This is a read-through
optimization of the one persisted route, not an in-memory authority or a
second write path, and it contains no modality or source-id predicate.

Emotional propagation uses the same first-source-wins semantics while
avoiding duplicate downstream work. Its breadth-first kernel carries up
to 64 source ordinals in one bit mask. For each reached embedding, the
kernel records only the lowest source ordinal and that source’s first
(therefore shortest) positive depth; after all batches, it emits one
winning pair per embedding in deterministic source/depth/identity order.
Separate counters record all source–embedding reachability pairs and all
physical adjacency visits, so winner collapse does not relabel unbounded
traversal as bounded work. The optimization is independent of modality
and source-id labels and does not change persistence or add a write
path.

The bounded-cascade experiment adds three independent F/S/T-derived
limits around that exact winner rule. Its maintained source-priority
snapshot holds at most the public routing budget 5*C*; one cascade
inspects no more than that snapshot and executes at most *B* eligible
sources. The shared traversal stops before physical adjacency visit
5*C* + 1, and the existing enqueue stage emits at most *A* = 2*C* + 2*B*
embedding updates. SQL fallback materializes at most the same 5*C*
source prefix. Inspection, accepted-source, physical-edge, winning-pair,
and update counts remain distinct, with their resolved limits recorded
on each profiled event. The snapshot ordering depends only on timestamp
and memory identity; no source-id or modality label participates. These
limits remain private experiment hooks until the complete quality,
plateau, restart, and knob-ablation contracts pass.

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
supersession, long-term kind, cosine-family collapse, graph expansion,
final scoring, and deterministic ties remain authoritative. This sparse
activation route fixes disconnected-node blindness without introducing a
second retrieval or write path; HNSW controls locality while the derived
envelope controls exact coverage and work.

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
identities from that row, resolves only that knob-bounded ordered slice,
and lazily fetches graph nodes under the 9*C* canonical plus *A*
snapshot public-query bounds; the route itself does not replay or mirror
complete history. This is not a bounded whole-engine restart claim:
`SignalProcessor` startup still materializes the complete current memory
surface before retrieval begins, so startup remains *O*(history) until
that separate hydration path is removed.

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

Normal writes and reconsolidations append only their memory identities
to the durable journal. A successful seal writes changed nodes and
knob-bounded reciprocal neighbors, advances graph metadata, and clears
only its staged root identities in the same SQLite transaction. Neighbor
proposal during this seal uses the *C* + *B* node and 2*B* queue
construction envelope, not the public 9*C* activation envelope. If an
existing active node’s normalized embedding moves beyond the fixed
10<sup>−6</sup> cosine-distance tolerance, the generation is invalidated
rather than pairing the new vector with stale adjacency. Exact retrieval
remains authoritative while the existing bounded backfill constructs a
replacement generation. The rejected incremental-relink experiment is
reported in the optimization section. A failed authoritative transaction
cannot leave a durable route mutation. An ordinary interrupted or failed
seal leaves the identities journaled for restaging. Intentional
moved-embedding invalidation instead abandons the incoherent generation
and clears its journal; the authoritative current surface feeds the
replacement bounded build while exact retrieval remains authoritative.
This gives consolidation a real route-epoch boundary without a
store-sized graph rewrite.

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

Emotional cascade metadata is hydrated into a private, rebuildable
processor cache keyed by memory id and base embedding id. The cache
preserves the SQL aggregation contract for shared embeddings by
retaining every member row and deriving current intensity and half-life
bonus with the same per-embedding minima. Its bounded flashbulb-source
prefix is ordered by descending emotional intensity with ascending
memory-id ties. Before each cascade, SQLite applies the event’s
intensity, arousal, and recent-window predicates and only then orders
and truncates to the knob-derived source limit. The private cache
therefore consumes the same eligible prefix as the SQL fallback: a stale
or low-arousal high-intensity row cannot hide a lower-intensity eligible
source. This is selection-equivalence evidence, not a claim that SQLite
can always locate that prefix in work proportional to the returned
limit; a history with many higher-intensity ineligible rows may require
additional indexed filtering. Emotional consolidation overwrites the
affected embedding family, cascade propagation applies the existing
maxima, storage and shallow consolidation append rows, and eviction
removes them. Journal-aware rollback discards and rehydrates this
database-derived cache, while arbitrary custom pipelines retain the
complete snapshot path. This changes execution ownership only; source
thresholds, tie order, traversal, decay, updates, and persisted rows are
unchanged. When multiple emotional sources require traversal, the
implementation batches them in source-order groups of at most 64 and
propagates a bit per source through the same breadth-first levels. Each
bit retains its source-specific radius, and reached embeddings are
emitted at the first depth observed for that source. Batches are
processed in original source order, so the existing first-source-wins
update rule is unchanged. This shares edge visits without using source
identity or modality as an optimization key. The cache also records the
memory nodes from which each source could still expand within its
radius. An incrementally inserted association can therefore advance the
fixed point without another traversal when neither endpoint lies in any
source’s expansion footprint. A topology rebuild invalidates the
footprint conservatively. Ordinary unconnected non-source inserts and
monotone increases to target emotional values likewise preserve the
fixed point; source changes and target-value decreases still re-arm it.
A traversal that reaches either its update ceiling or its physical
edge-visit ceiling is only a partial prefix and never records a fixed
point, so the next event revisits the unresolved graph tail.

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
one representative. The family index accumulates exact squared
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
family and supersession filters before truncation. A failed rebuild
installs a failure sentinel rather than repeating the two complete
surface scans on every request. A unique cached vector retains every
long-term memory metadata alternative that shares its base embedding,
and query-time eligibility chooses a nonsuperseded sibling without
duplicating vector storage. The cache separately indexes entries that
have at least one long-term metadata alternative. Retrieval applies the
existing timestamp, current-surface, and supersession predicates to that
subset before evaluating vector distance, so signal-only historical
embeddings do not enter the distance kernel. This is an execution-only
index: it does not change the eligible set, distance expression, tie
order, family collapse, or source and modality semantics. Alongside the
shared association fanout cache, an internal execution sidecar
maintains, for each supersession target, the earliest start timestamp of
an eligible replacement. Retrieval tests that timestamp directly instead
of rebuilding a target set by scanning every incoming edge. Memory
storage publishes its already-committed supersession decisions into the
fanout cache and recomputes only the affected target’s eligibility. A
current-surface change likewise recomputes only targets reached by that
memory’s outgoing `supersedes` edges. Missing target metadata
invalidates the cache and returns to authoritative SQLite rehydration.
Neither structure is part of the public processor layout. The strict
replacement-before-query predicate and base embedding identity remain
unchanged. The emotional-metadata portion of the sidecar is trusted only
by the engine-owned operation root that maintains it. A caller-supplied
root’s transaction view invalidates that sidecar before forwarding its
first SQL statement, so a custom transaction that updates emotional
columns directly cannot leave a valid-looking stale fixed point for a
later operation in the same transaction. Cache-only custom roots retain
their exact rollback snapshot. The fallback executes a demand-driven
sequence of ordered SQL pages over the current and historical vector
surfaces. Each page joins memory metadata and applies kind, timestamp,
and supersession eligibility before distance ordering, excludes a
historical row when that memory already has a current reconstruction,
collapses byte-identical historical vectors, and uses a second
application of the existing F/S/T-derived seed-search breadth as its row
bound. One in-memory cosine-family index spans every page; paging stops
when the ordinary semantic-family seed budget is full or the eligible
surface is exhausted. Processor hydration tracks
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
Rollback and flush recovery also rehydrate each current entry’s original
base ID from `memories.embedding_id`; they never infer it from the
reconstructed surface. When constructive recall is disabled, both
restart hydration and cache-loss rebuild select the base embedding and
ignore reconstruction/current surfaces. This recovery path preserves an
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
scoring or output caps. When eviction removes a replacement memory from
the processor surface, removal is published before its outgoing
`supersedes` targets are recomputed. The sidecar therefore cannot retain
an activation timestamp derived from a surface entry that has just been
evicted. One dispatch exposes at most one actually used retrieval
candidate, so retrieval does not synthesize reinforcement pair edges
from the packet. Existing `reinforces` edges remain eligible for decay,
pruning, degree normalization, and graph evidence. Historical public
knob helpers for the removed packet-pair writer remain source-compatible
calibrations; only the step calibration still feeds the production
pruning floor.

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
Graph expansion after seed discovery is bounded by F/S/T-derived row and
fanout limits. Working-memory and soft-anchor updates operate over
bounded live state and remain independent of total store size.

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

The RIF shadow epoch is a disposable SQLite projection of committed
persistent state. Suppression and recovery stage changed identities and
clock values while the persistent transaction is open, but they do not
pre-adjust the shadow’s `active_rows` counter. After the persistent
commit, publication compares the old and new shadow membership and
applies each insertion or removal exactly once. Consolidation’s
active-RIF count and both candidate arms join the singleton recovery
clock on the current generation; retired-generation rows may await
bounded materialization, but they neither consume the active work
ceiling nor enter the current active candidate frontier.

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
<td>current-memory embedding lookup with pre-cap supersession filtering;
immutable normalized and stored/Hadamard block summaries cached per
private surface entry; exact partial squared-distance rejection plus
complementary sound 32-block Cauchy–Schwarz bounds in the stored and
Walsh–Hadamard bases before exact cosine family checks; shared vectors
retain all long-term metadata alternatives; the private search cache
follows the processor’s latest reconstruction surface independently of
persisted-current materialization; cache loss uses the complete
processor surface directly or, only when that surface is incomplete,
metadata-eligible, distance-ordered, F/S/T-bounded SQL pages feeding one
cross-page cosine-family index that substitutes latest reconstructions
before accounting</td>
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
<td>supersession eligibility</td>
<td>derive target-to-earliest-replacement timestamps in an internal
execution sidecar beside the shared association fanout cache and
incrementally publish committed MemoryStorage supersession edges instead
of invalidating and rebuilding all fanout rows; remove an evicted
replacement surface before recomputing its targets</td>
<td>exact edge direction, base embedding ids, strict timestamp boundary,
source/modality independence, SQLite fallback, rollback rehydration, and
the public processor layout are retained; an evicted replacement cannot
leave a stale target timestamp</td>
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
activated current centroids, then require the historical cache’s exact
coverage/ranking proof or retain SQLite fallback rather than treating
the sparse result as complete</td>
<td>the current sparse-candidate query remains bounded by <span
class="math inline">9<em>C</em> + <em>A</em></span>, but exact
historical verification may remain history-sized; timestamp, kind,
duplicate band, similarity threshold, edge limit, deterministic tie
order, rollback, source/modality independence, and public API shape
remain unchanged; write throughput and production-wide boundedness are
explicitly unclaimed</td>
</tr>
<tr>
<td>graph retrieval</td>
<td>score vector similarity first, restrict recency to a tie-break, cap
graph bonus, reserve final occupancy for direct semantic anchors, and
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
<td>maintain a private rebuildable cache of per-memory emotional rows,
per-embedding shared-member minima, runtime-eligible flashbulb sources,
and the expandable topology footprint of the last fixed point; apply
intensity, arousal, and recent-window predicates before intensity-order
truncation; update the cache at the same successful storage,
consolidation, cascade, association, and eviction decisions and
rehydrate it after journal-aware rollback; caller-supplied roots
invalidate the sidecar before forwarding their first transaction
statement, while cache-only roots retain their exact snapshot; share
breadth-first edge visits across source-order 64-bit batches while
retaining a distinct visited bit and radius for every source</td>
<td>source eligibility and stable tie order, shared-embedding
<code>MIN</code>/<code>MAX</code> semantics, shortest depth per source,
first-source-wins ordering, custom raw-SQL updates within the current
transaction, update decisions, and durable database state match the
scalar path; high-intensity stale or low-arousal rows cannot hide
eligible sources, though locating the eligible prefix is not claimed to
require only returned-prefix work; an inserted edge outside every
source’s remaining-radius footprint advances the fixed point without a
graph walk, topology rebuilds invalidate conservatively, batches
continue for arbitrary source counts, and neither path keys behavior by
source id or modality; ordinary engine-owned snapshot work still copies
zero store-sized cache entries</td>
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

identities for current-surface eligibility, family collapse, graph
expansion, and final ranking. At neutral knobs *C* = 512, *B* = 128, and
*A* = 1, 280; at the all-low and all-high endpoints *A* is 640 and
1,920. Construction remains a separate *C* + *B* node and 2*B* queue
operation. The canonical maximum-level HNSW entry performs global
descent, while the nearest consolidation centroid is an additional
level-zero seed and deterministic fill pivot. Neither identity set,
seed, nor limit depends on modality or `source_id`.

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
<td>Retained focused repair; the next independent graph-family
comparison remained required.</td>
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
<td>Drift rearm, accumulator reset, shallow consolidation, and
consolidation-throughput state.</td>
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
<td>Supersession storage, association fanout, eviction,
predictive/stability updates, and usage detection.</td>
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
