# Cortext: A Three-Knob Adaptive Memory Architecture
Gabriel Willen, Cortext Team
2025-12-01

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
reliance on hard-coded constants. The system demonstrates
self-calibrating priors that blend with evidence using
uncertainty-weighted Bayesian averaging, homeostatic threshold control
with effective sample size estimation, and graph-augmented retrieval
combining embedding similarity with semantic extraction. Experimental
analysis indicates the architecture maintains stable operation across
developmental phases while adapting write rates, decay dynamics, and
retrieval precision to environmental demands. This work contributes a
formally specified cognitive memory model suitable for implementation in
streaming AI systems requiring persistent, context-aware memory.

**Keywords:** cognitive architecture, adaptive memory, working memory,
episodic memory, semantic memory, knowledge graphs, homeostatic control

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
4.  A knowledge graph layer enables semantic consolidation and
    graph-augmented retrieval.

The remainder of this paper is organized as follows. Section 2 reviews
relevant literature. Section 3 presents the mathematical foundations
including notation, helper functions, and knob-derived parameters.
Section 4 details the core algorithms for Focus, Sensitivity, and
Stability adaptation. Section 5 describes structural metrics and
composite scoring. Section 6 covers dynamic thresholding and homeostatic
control. Section 7 presents the reinforcement and decay dynamics.
Section 8 describes advanced cognitive processes including working
memory, metacognition, and emotional consolidation. Section 9 details
the consolidation and graph integration system. Section 10 presents the
interrupt gate for streaming integration. Section 11 reports preliminary
experimental results. Section 12 discusses implementation considerations
and computational complexity. Section 13 concludes with limitations and
future directions.

# Related Work

## Working Memory Models

The working memory component of Cortext draws primarily on Cowan (2001)
embedded-processes model, which posits a capacity limit of approximately
4±1 chunks for the focus of attention. This contrasts with Miller (1956)
earlier estimate of 7±2 items, which subsequent research has shown
conflates chunking with raw capacity (Cowan 2010). Our implementation
respects these empirically-derived constraints while allowing for
focus-dependent modulation within bounded ranges.

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
immediate experiences and a consolidated semantic graph for abstracted
knowledge. The consolidation process transforms high-redundancy episodic
clusters into summary nodes linked by typed semantic relations,
consistent with complementary learning systems theory (McClelland,
McNaughton, and O’Reilly 1995).

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
last_rate_timestamp, last_retrieval_ts, retention_ema, last_embedding,
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

All behavioral thresholds, gains, and caps are derived from the three
knobs (F, S, T) and/or state. Fixed numeric values are permitted only
for numerical stability floors (e.g., ε(F, S, T)) and unit conversions.
This keeps the system’s qualitative behavior entirely governed by the
three knobs, consistent with the human-memory objective.

## Temporal Semantics for Structured Facts

Cortext currently treats raw memories primarily as **event-time
observations**: a memory is written when a coherent unit is detected,
and its stored timestamp records when that unit was observed. The
phase-1 bitemporal fact layer adds a structured consolidation product
that distinguishes three related but non-identical temporal notions:

-   **Event time:** when the source experience was observed by the
    system.
-   **Valid time:** when an extracted fact was true in the world.
-   **Transaction time:** when Cortext learned, inserted, corrected, or
    superseded that fact.

For an episodic memory `m_i`, we define the event timestamp:

    t_event(m_i) = created_at(m_i)

For a structured fact assertion `f_j = (subject, predicate, object)`, we
define two half-open intervals:

    valid(f_j) = [t_v_start(f_j), t_v_end(f_j))
    tx(f_j)    = [t_tx_start(f_j), t_tx_end(f_j))

The valid interval models when the assertion held in the world. The
transaction interval models when the system regarded that assertion as
active knowledge.

This distinction supports three different query semantics:

    true_at(f_j, τ)     = 1 if τ ∈ valid(f_j), else 0
    known_at(f_j, τ)    = 1 if τ ∈ tx(f_j), else 0
    current(f_j, now)   = true_at(f_j, now) ∧ known_at(f_j, now)

This is the core of **bitemporal modeling**: the system can separately
answer “what was true then?” and “what did the system believe then?”
without collapsing historical corrections into a single timeless
statement.

In the current alpha, Cortext’s live behavior remains primarily
event-driven and recency-weighted. The bitemporal formulation above is
an additive structured-fact layer for evolving personal knowledge rather
than a replacement for the episodic memory stream.

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

Unless explicitly stated otherwise, all knob‑derived formulas use **F̃**
and **S̃** in place of the raw UI knobs. Stability (T) is not biased.

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

These ranges are canonical knob maps; no additional fixed thresholds are
introduced beyond the three knobs.

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
sadness, surprise}. categories inspired by Russell (1980) circumplex
model are projected onto the valence-arousal plane:coordinates:

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
confidence via geometric mean, providing a measure that is high only
when a single emotion dominates with high certainty.

### Threshold Modulation from Emotion

Emotional activation loosens write thresholds to capture salient
moments:

    κ_emo ← κ_base × S  # where κ_base = 0.10
    ΔThreshold_emotion_t ← −κ_emo × emotion_intensity_t ×
                            (0.5 + 0.5 × arousal_t)

The emotional state acts as a modulator for memory
encoding/consolidation, following McGaugh (2004). High arousal and
valence magnitude increase the likelihood of threshold crossings.

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

Where `retrieval_pressure` is the normalized retrieval queue depth, and
`δ_reward_t` is a reward prediction error derived from downstream
outcome signals (see
<a href="#sec-structural-metrics" class="quarto-xref">Section 5</a>).
These modulators drive internal gates without introducing new
user-facing parameters:

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

    τ_rate ← max(2^(3T) × dt_base, dt_floor(T))
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
(Zacks & Swallow, 2007).

This approach draws from EM-LLM (Fountas et al., 2024), which segments
token sequences into episodic events using surprise-based boundary
detection refined by graph-theoretic cohesion metrics. Their work shows
that combining prediction error signals with within-segment coherence
produces boundaries strongly correlated with human event perception. Our
adaptation uses EMA prediction error (surprisal_t) for surprise and
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
    drift_acc ← drift_acc + (drift_mag_t / 2)  # accumulate normalized instantaneous drift (drift_mag_t ∈ [0,2])
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
    ema_mean_x ← EWMA(ema_mean_x, x, α_local(T))
    ema_var_x  ← EWMA(ema_var_x, (x − ema_mean_x)^2, α_local(T))
    norm_gain(S,T) = clamp(lerp(1.6, 3.0, S) × lerp(1.05, 0.95, T), 1.0, 3.2)
    x̂ ← sigmoid(norm_gain × (x − ema_mean_x) / sqrt(ema_var_x + ε))  # warm-up uses raw x

    # Topic shift (embedding-only, temporal context anchor)
    topic_shift ← 1 − map01(cos(e_t, c_t))

    # Likelihood of a boundary given observations
    support ← clamp(0.5 × coĥ + 0.5 × topiĉ, 0, 1)
    support_gate ← lerp(0.45, 1.0, support)   # downweight isolated spikes
    gap_gate ← lerp(0.10, 0.50, support)
    support_boost ← lerp(1.0, 1.25, support)  # emphasize coherent/topic shifts
    [w_s, w_d, w_c, w_t, w_g] ← [0.34 + 0.14S, 0.18 + 0.05(1 − T),
                                0.36 + 0.16F, 0.30 + 0.22S, 0.01]
    [w_s, w_d, w_c, w_t, w_g] ← normalize([w_s×support_gate, w_d×support_gate,
                                          w_c×support_boost, w_t×support_boost, w_g×gap_gate])
    z_center(S,T) = clamp(lerp(0.44, 0.30, S) × lerp(1.05, 0.95, T), 0.26, 0.58)
    z_center_eff ← clamp(z_center × lerp(1.0, 0.75, support), 0.24, 0.58)
    z_t ← w_s×(surprisal̂ − z_center_eff) +
          w_d×(drift̂ − z_center_eff) +
          w_c×(coĥ − z_center_eff) +
          w_t×(topiĉ − z_center_eff) +
          w_g×(gap_score − z_center_eff)
    k_cp(S,T) = lerp(8, 22, S) × lerp(1.1, 0.9, T)
    k_cp_eff ← k_cp × lerp(0.9, 1.35, support)
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
    capacity ← max_mem_drift(S) × capacity_scale(T)
    pressure ← drift_acc × (1 + S)
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

### Inactivity Boost

We avoid hard timeouts for episode boundaries. Instead, when the stream
becomes quiet and natural indicators are weak, inactivity softly boosts
the boundary score:

    support_relax = exp(−lerp(0.6, 1.6, S) × lerp(1.15, 0.85, T) × gap_z⁺)
    gap_ratio = signal_gap_s / max(dt_ema, ε)
    gap_z_inact = log1p(max(gap_ratio − 1, 0))
    gap_score_inact = sigmoid(lerp(0.9, 1.8, S) × lerp(1.1, 0.9, T) × gap_z_inact)
    inactivity = gap_score_inact × (1 − support × support_relax)
    k_inact(S,T) = lerp(0.05, 0.25, S) × lerp(1.1, 0.9, T)
    gap_z⁺ = max(0, gap_z_inact)
    inactivity_scale = exp(lerp(0.6, 1.6, S) × lerp(1.1, 0.9, T) × gap_z⁺)
    boundary_score ← boundary_score + k_inact(S,T) × inactivity × inactivity_scale
    gap_force = sigmoid(lerp(2.0, 5.0, S) × lerp(1.1, 0.9, T) ×
                       (gap_z_inact − lerp(0.9, 0.4, S) × lerp(1.1, 0.9, T)))
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
(<a href="#sec-pattern-separation" class="quarto-xref">Section 9.2</a>).
Reset accumulator for next unit.

# Reinforcement and Decay Dynamics

## Memory Strength Model

Each memory maintains **multi-timescale traces** rather than a single
leaky bucket. A mixture of fast/medium/slow/ultra-slow traces yields a
power-law forgetting curve while preserving plasticity.

    N_traces(T) = 2 + round(2T)  # 2..4 coupled traces
    τ_fast  = 0.10 × half_life
    τ_med   = 0.50 × half_life
    τ_slow  = 2.00 × half_life
    τ_ultra = 8.00 × half_life
    τ_list = [τ_fast, τ_med, τ_slow, τ_ultra]

    α_min_S = 0.05; α_span_S = 0.35
    α_S(t) = α_min_S + SensitivityBias(S) × α_span_S × u(t)
    used_flag(m) = 1 if m was retrieved and used in current step, else 0

Trace updates combine exponential decay, EWMA learning, and knob-scaled
reinforcement (for i in 1..N_traces):

    λ_i ← ln(2) / τ_list[i]
    reinforcement ← clamp(S_eff × used_flag(m) +
                           F_eff × clamp(influence_factor, 0, 1), 0, 1)
                    × serial_position_mult
    increment_i ← (α_S(t) × used_flag(m) + reinforcement) / N_traces
    trace_i ← clamp(trace_i × exp(−λ_i × Δt) + increment_i, 0, 1)

The reinforcement term distributes S- and F-scaled retrieval feedback
uniformly across all active traces. This avoids front-loading
reinforcement into the fast trace (which decays quickly) and instead
strengthens the entire trace ensemble when a memory is retrieved and
used.

Trace coupling encourages long-lived knowledge without freezing
plasticity:

    coupling = 0.05 + 0.10T
    trace_{i+1} ← clamp(trace_{i+1} + coupling × trace_i, 0, 1)

Combined strength uses a knob-shaped mixture that favors slow traces as
Stability increases:

    w_raw ← [0.40 − 0.25T, 0.25, 0.20 + 0.15T, 0.15 + 0.10T]
    w ← normalize(w_raw[1..N_traces])
    strength_t ← clamp(Σ_i w_i × trace_i, 0, 1)

Δt is measured per memory using its last access timestamp:

    Δt(m) ← now_s() − to_s(m.last_access); if last_access is unset, use created_at

Memories falling below the periphery cutoff are candidates for eviction.
**Only LONG_TERM memories are evicted**; ASSOCIATION and LABEL nodes are
retained to stabilize summary/label retrieval and are pruned during
explicit consolidation.

    periphery_cutoff(T) = lerp(0.03, 0.20, T)
    fact_floor(T) = lerp(0, periphery_cutoff(T), T)
    has_fact_support(m) = evidence_count(m, active_facts) > 0
    storage_budget_floor = max(
      min_db_bytes,
      db_avail_fraction × available_disk_bytes
    )

    if m.kind == LONG_TERM
       AND db_used_bytes ≥ storage_budget_floor
       AND strength_t < periphery_cutoff(T)
       AND (NOT has_fact_support(m) OR strength_t < fact_floor(T)):
         evict(m)

The fact-evidence floor protects episodic memories that serve as
evidence for active (non-archived) structured facts. At low Stability,
the floor is zero and fact-linked memories receive no special treatment.
At high Stability, the floor approaches the periphery cutoff itself,
making fact-backed evidence nearly immune to eviction. This ensures that
the episodic substrate supporting durable semantic knowledge is not
discarded before the knowledge itself.

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
that each component of this model contributes to selective eviction
behavior. Disabling coupling or reducing to a single trace collapses the
power-law forgetting curve into rapid indiscriminate eviction, flashbulb
bonuses provide the expected protective effect, the fact-evidence floor
prevents loss of active fact support, and uniform reinforcement
distribution makes S and F visibly effective in the eviction dynamics.

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

    stability(m)_init = 1.0
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

Co-retrieved memories also form *reinforces* edges in the ASSOCIATIONS
graph. These edges are strengthened during retrieval/use and **only
decayed during explicit consolidation cycles** (triggered externally via
`cortext.consolidate()`), matching the system’s “idle-time
consolidation” policy. This prevents per-turn decay from erasing fresh
reinforcement and keeps runtime retrieval updates stable while still
allowing long-term pruning during consolidation.

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

When generation embeddings are available, influence incorporates output
trajectory:

    Δḡ ← l2_normalize(ḡ_t) − l2_normalize(ḡ_{t−1})
    drift_mag_gen ← ‖Δḡ‖
    drift_contribution(m) ← (drift_mag_gen / 2) ×
                             max(0, cos(m.embedding, l2_normalize(Δḡ)))

Total influence blends contextual gain, generation similarity, and drift
contribution:

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
phenomena: working memory maintenance, metacognitive monitoring,
reconsolidation dynamics, and serial position effects.

**Knob note:** Unless explicitly stated otherwise, F and S in this
section refer to midpoint‑biased values (F̃, S̃) defined in Section 1.

## Working Memory Gates

Following Cowan (2001) capacity constraints, working memory maintains a
limited number of active items. Working memory holds coherent memories
as defined in
<a href="#sec-write-pacing" class="quarto-xref">Section 6.4</a>,
preserving the full content and signal sequence:

    base_capacity = round(lerp(5, 3, S) + lerp(−1, 1, F))

This yields a range of approximately 2-6 memories, broadening the 4±1
chunk limit to accommodate task-dependent requirements. High Sensitivity
reduces capacity (faster turnover), while high Focus modulates breadth.

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
class="quarto-xref">Section 8.10</a>).

## Temporal Context State (Time-Cells Analogue)

Beyond the accumulator centroid μ_acc, Cortext maintains a slowly
drifting temporal context vector `c_t`. This provides ordered recall and
reduces topic-collision errors by enabling **context reinstatement**.

    α_c(T) = lerp(0.06, 0.01, T)  # slow drift at high Stability
    c_t ← l2_normalize((1 − α_c(T)) × c_{t−1} + α_c(T) × μ_acc + ξ_t)

where ξ_t is small isotropic noise (or a deterministic phase vector) to
prevent collapse. Each memory stores `(e_rep, c_t)`. Retrieval combines
content match and context match:

    w_ctx(F,S,T) = lerp(0.15, 0.45, F) × lerp(1.0, 0.85, S)
    association_boost(F,S,T) = lerp(0.015, 0.06, S) × lerp(1.0, 0.7, F) × lerp(1.0, 0.9, T)
    score_retrieval(m) ← (1 − w_ctx) × cos(q, m.embedding) +
                          w_ctx × cos(c_t, m.context) +
                          association_boost(F,S,T) × I[m.kind = ASSOCIATION]

Reinstatement: retrieving a memory updates `c_t` toward `m.context`,
improving ordered recall within the same episode.

### Maintenance Cost

Maintenance incurs cognitive cost:

    maintenance_cost_per_memory = lerp(0.05, 0.15, S)
    complexity_penalty = manifold_complexity × lerp(0.5, 1.5, S)

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
        [α, β, γ] ← normalize([lerp(0.55, 0.70, F),   # window score weight
                            lerp(0.20, 0.35, F),   # task relevance weight
                            lerp(0.10, 0.30, S)])   # novelty-to-WM weight
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

### Chunking at Memory Level

Chunking operates on memory embeddings to merge related content from the
same source:

    similar_memories ← {m ∈ active_memories | cos(m.embedding, memory.e_rep) > chunking_threshold AND
                                  m.source_id == memory.source_id}
    if |similar_memories| > 0:
        merge_into_chunk(similar_memories, memory)

**Implementation note (chat applications):** turn-structured chat
sources (`chat/user`, `chat/assistant`) bypass same-source chunk merging
and generic WM rejection so the active working-memory window remains a
FIFO sequence of recent turns for prompt reconstruction. Prompt
hydration must order active slots by insertion/start time rather than
`last_access`; otherwise rehearsal or retrieval touches can scramble the
conversational sequence even when the underlying memories are correct.

## Metacognitive Monitoring

The system implements feeling-of-knowing (FOK) and tip-of-tongue (TOT)
detection following Hart (1965) framework:

    FOK_threshold = lerp(0.2, 0.5, F)
    TOT_detection = (FOK > lerp(0.5, 0.8, F)) AND
                     (retrieval_strength < lerp(0.4, 0.2, F))

TOT occurs when metacognitive confidence is high but retrieval strength
is low—the characteristic experience of knowing one knows something but
being unable to access it.

Additional metacognitive parameters:

    confidence_decay_rate = lerp(0.01, 0.1, 1 − T)
    unknown_threshold = lerp(0.3, 0.1, F)
    strategy_switch_latency = lerp(500, 100, S)  # ms
    certainty_requirement = lerp(0.6, 0.9, T)
    metacognitive_sensitivity = F × (1 + 0.5 × S)

Current implementation note: these metacognitive states are no longer
diagnostic-only. The branch now maintains a persistent metacognitive
confidence state that decays between signals:

    Δt_meta = now_s − last_signal_s
    confidence_retained = confidence_{t−1} × exp(−confidence_decay_rate × Δt_meta)
    confidence_t = max(FOK_t, EWMA(confidence_retained, FOK_t, α_meta))

where `α_meta` is a bounded function of `metacognitive_sensitivity`.

That decayed confidence now affects two downstream policies:

-   a detected **TOT** state still arms a **next-turn retrieval recovery
    mode**, but the extra graph depth, candidate budget, and
    diversification relaxation are scaled by the retained confidence
    instead of being an all-or-nothing switch
-   a detected **unknown** state still arms a **next-turn caution
    mode**, and lower retained confidence makes the certainty cutoff
    stricter before a candidate can be returned

The switch duration still follows `strategy_switch_latency`, but the
behavioral effect is now a bounded retrieval-policy change rather than a
same-turn retry loop, and `confidence_decay_rate` is part of that
behavior rather than a paper-only parameter.

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

## Source Monitoring and Reality Constraints

Each memory carries a provenance model that tracks its origin,
reliability, and contradiction history. Retrieval returns **content +
source confidence**, enabling downstream gating and auditability.

    source_model = {origin, reliability, contradiction_count, last_verified_ts}
    source_prior(origin) = {'user': 0.8, 'assistant': 0.6, 'external': 0.7, 'system': 0.9}
    source_confidence(m) ← clamp(source_prior(m.origin) ×
                                 (1 − 0.15 × contradiction_count) ×
                                 (0.7 + 0.3 × freshness(m)), 0, 1)

Contradictions reduce `reliability`, and user corrections directly
update `contradiction_count`. Source confidence gates injection into
active context:

    if source_confidence(m) < lerp(0.15, 0.45, T):
        downrank_or_hold(m)

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

Current implementation note: predictive pre-activation is now consumed
directly by retrieval rather than only being written to the database.
`pre_activation` decays every turn via `pre_activation_decay(T)` and
contributes a bounded retrieval prior,

    predictive_weight(F,T) = lerp(0.05, 0.20, FocusBias(F)) × lerp(1.0, 0.85, T)
    predictive_bonus = predictive_weight × pre_activation

which is added to the normal retrieval score. High-surprise updates
therefore both refresh the latent predictive state and make that state
behaviorally visible at the next retrieval.

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
<a href="#sec-activity" class="quarto-xref">Section 9.3.1</a>,
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

## Procedural Memory Lane (Habit/Skill Memory)

In addition to declarative memory, Cortext maintains a procedural store
for **which previously successful routine memory to surface in a
context**, learned from repeated successful use.

    proc_key ← sparse_key(μ_acc)  # same sparsification as @sec-pattern-separation
    Q(proc_key, memory_id) ← routine value

Updates occur when a retrieved memory is used successfully and
downstream outcome signals are positive:

    Q ← Q + value_update_gain × δ_reward_t × (1 − Q)

Procedural retrieval runs in parallel with declarative retrieval. In the
current implementation, high-confidence routine memories are added as
**proactive retrieval seeds** for the current sparse context even before
a strong semantic match would have selected them; they still pass
through the same interrupt gate and final retrieval ranking. This is
intentionally narrower than a free-standing action policy: the
procedural lane surfaces likely next-step routines as memories, not
separate action tokens.

# Consolidation and Graph Integration

The consolidation system transforms episodic memories into semantic
structures through clustering, summarization, and knowledge graph
construction. Memory-level storage
(<a href="#sec-write-pacing" class="quarto-xref">Section 6.4</a>)
ensures each embedding represents a coherent unit.

## Complementary Learning Systems Split (Hippocampus vs Cortex)

Cortext implements a true **Complementary Learning Systems (CLS)**
split. The episodic stream remains intact, but a separate **sparse index
store** supports rapid binding and pattern completion (hippocampal
analogue). Consolidation then transfers structure into the semantic
graph and long-lived embeddings (cortical analogue) via replay.

-   **Episodic stream (hippocampus-like):** `memory_stream` of coherent
    units + full metadata.
-   **Index store (hippocampus-like):** sparse keys for fast binding,
    disambiguation, and pattern completion.
-   **Semantic graph + long-lived embeddings (cortex-like):**
    consolidated, slowly-changing semantic structures.

This split makes “episodic vs semantic” a functional distinction rather
than a storage convention.

## Pattern Separation and Completion

Each memory builds a **sparse key** for pattern separation. Retrieval
can then perform **pattern completion** from partial cues.

    proj ← W_sparse × e_rep          # learned projection
    key  ← topk_binary(proj, k = round(lerp(16, 64, F)))  # sparsify
    index_store[key].append(memory_id)

Pattern completion from a partial cue uses the sparse key to seed
retrieval:

    cue_key ← topk_binary(W_sparse × q_partial, k)
    seed_ids ← index_store.lookup(cue_key, radius = lerp(1, 3, S))

These seeds are merged with vector and graph candidates, improving
disambiguation when content similarity is ambiguous.

On write, every memory updates the index store:

    index_store[key].append(memory_id)

## Consolidation Triggers

Consolidation operates on stored memory representatives (e_rep from
<a href="#sec-rep-embedding" class="quarto-xref">Section 6.4.7</a>), not
individual signals. Candidates are restricted to **unclustered LONG_TERM
memories**, and deep mode additionally requires stored content blobs
(`blob_id` present) so summarization always has raw source text.
Previously generated **ASSOCIATION** summaries are treated as sealed
semantic products at the same consolidation level: they remain available
for retrieval and graph expansion, but they are not fed back into later
same-level deep summarization. **Consolidation is only initiated via an
explicit API call (`Consolidate()`).** The core runtime does not
auto-trigger consolidation.

External schedulers can still **derive a trigger policy** from the
knobs. One recommended heuristic is:

    should_consolidate = (memory_count > consolidation_threshold) OR
                          (m_rate < rate_target / 2) OR
                          (elapsed_time > consolidation_interval) OR
                          (tagged_memory_pending == true)

Definitions (knob-derived):

    consolidation_interval(T) = lerp(300, 3600, T)  # seconds (5 min → 1 hour)
    consolidation_threshold(T) = n_ctx(T) × win_score(T)  # memory-count trigger
    merge_threshold(F) = lerp(0.85, 0.95, F)  # similarity required for clustering

**Recommendation signal (external scheduling):** Each `Process*` call
returns two flags that notify the caller when consolidation is
**recommended** or **required** (without auto-triggering). These use the
same knob-derived thresholds:

    consolidation_recommended =
      (elapsed_time ≥ consolidation_interval) OR
      (memories_since_consolidation ≥ consolidation_threshold)
    consolidation_required =
      (elapsed_time ≥ consolidation_required_interval) OR
      (memories_since_consolidation ≥ consolidation_required_threshold)
    consolidation_required_interval(T) = consolidation_interval(T) × lerp(1.5, 2.5, T)
    consolidation_required_threshold(T) = consolidation_threshold(T) × lerp(1.5, 2.5, T)

The consolidation rate adapts to Stability and Sensitivity (write_rate
tracks memory writes, not signal writes):

    rate_consolidate = (1 / max(consolidation_interval, 1)) ×
                        (0.3 + 0.7T) × (1 − 0.5S)

When `Consolidate()` is invoked and no candidates fall below the
periphery cutoff, the forced path selects the lowest-scoring `w_ret(T)`
memories; if clustering still yields no groups, a fallback mini-cluster
is formed around the lowest-score item using its top cosine neighbors.
This keeps replay/summarization active while remaining fully
knob-derived.

### Activity-Aware Scheduling

If the caller wants idle-aware scheduling, the following knob-derived
gating can be applied **externally**. The is_accumulating_memory check
ensures consolidation doesn’t interrupt mid-memory accumulation:

    idle_required(T) = round(0.25 × win_rate_s(T))
    idle_for_s = now_s() − to_s(last_retrieval_ts)
    # Consolidation waits for memory completion, not just signal arrival
    should_start = (NOT is_accumulating_memory) AND
                    (retrieval_queue_depth == 0) AND
                    (idle_for_s ≥ idle_required(T))

Consolidation begins only after the current memory has been flushed
(<a href="#sec-boundary" class="quarto-xref">Section 6.4.3</a>) and the
idle period has elapsed. On retrieval events, consolidation pauses,
commits micro-batches, and resumes when idle.

**Implementation note (topical_chat_analysis):** consolidation is
**external-only**. The runtime does not schedule consolidation
automatically; the caller must invoke `Consolidate()` explicitly (e.g.,
at a known idle window).

## Consolidation Scoring

Each stored embedding represents a memory representative (e_rep from
<a href="#sec-rep-embedding" class="quarto-xref">Section 6.4.7</a>).
Each memory receives a consolidation score determining merge priority:

    score_consolidate(m) = weight_strength × strength(m) −
                            weight_redundancy × redundancy(m) +
                            weight_connectivity × connectivity(m) +
                            weight_stability × stability(m)

Weights derive from knobs:

    weight_strength = T; weight_redundancy = F; weight_connectivity = S; weight_stability = T

Low-scoring memories are marked for merging.

## Replay-Based Consolidation and Summaries

Consolidation proceeds via **generative replay** rather than one-shot
merges. Representative episodes are re-encoded and used to update
semantic structures, reducing drift and preserving detail.

**Summarization + labeling engine (deep mode):** Consolidation is the
only stage that invokes a generative model. Deep mode resolves a local
backend internally: **gemma-3n-e2b** through `third_party/litert_lm`,
**LFM2/LFM2.5** through `llama.cpp` GGUFs, or a **mixed** path that uses
Gemma for summarization and a Liquid GGUF for extraction. The Liquid
resolver now prefers `LFM2.5-350M-GGUF` for both summarization and
extraction, defaulting to the local `Q4_K_M` checkpoint when present and
accepting other bundled GGUF quantizations as fallbacks; if that asset
is absent, it falls back to the older pinned
`LFM2-2.6B-Transcript-Q4_K_M.gguf` summarizer and
`LFM2-1.2B-Extract-Q4_K_M.gguf` extractor. Auto resolution now prefers
the pure Liquid `LFM2.5/llama.cpp` path when those assets are present;
the mixed path and Gemma path are fallbacks. All other operations remain
embedding-only. **There is no extractive fallback**: if no deep backend
is available, or summarization fails, deep consolidation raises an error
rather than silently degrading. After the prompt-technique ablations in
[Section 9](#sec-extractor-prompt-ablation) and [Section
9](#sec-summarizer-prompt-ablation), the default local Liquid extraction
prompt is now a short **few-shot durable** prompt, and the default local
Liquid summarization prompt is now a **transcript few-shot** prompt that
feeds the small `LFM2.5-350M` model a chat transcript instead of
numbered excerpts. That combination keeps the local path fast while
materially reducing schema/meta leakage and last-excerpt bias.

## Shallow Consolidation (Embedding‑Only)

Shallow consolidation is an explicit, embedding‑only phase for fast
labeling and graph priming. It skips summarization/extraction and
instead:

1.  Creates a centroid **ASSOCIATION** node per cluster (no summary
    blob).
2.  Updates `cluster_id` for source memories and emits `derived_from`
    edges.
3.  Attaches existing **LABEL** nodes by embedding similarity.

Shallow labeling uses knob‑derived limits and thresholds:

    max_labels_per_cluster = round(lerp(2, 6, S̃) × lerp(1.0, 0.7, F̃))
    label_attach_threshold = clamp(lerp(0.30, 0.60, F̃) × lerp(1.0, 1.1, T), 0.15, 0.95)

For each cluster centroid, we compute cosine similarity to existing
label embeddings, retain candidates above the threshold, and attach the
top‑K labels by similarity.

To avoid a cold start, we optionally seed a **label bank** at
initialization: a static list of high‑frequency labels (objects, sounds,
relations, names) **pre‑embedded** with ImageBind and stored as
**LABEL** memories. Seeded labels receive a knob‑derived baseline
salience:

    salience_seed = clamp(lerp(0.35, 0.65, 0.5(F̃+S̃)) × lerp(1.0, 0.9, T), 0, 1)

This provides shallow consolidation with immediate label candidates
without invoking any generative model.

Label bank embeddings are generated offline (dataset → embeddings) and
saved as `data/label_bank/labels.jsonl` with a `metadata.json` manifest;
seeding reads the precomputed vectors and does not re‑encode at runtime.
We build the label list from **Hugging Face datasets** (FSD50K for
sounds, ConceptNet for common terms, and Gender‑by‑Name for names) using
`tools/label_bank_builder/build_label_bank.py`, then embed with
`tools/label_bank_generator`.

Replay batch:

    replay_batch ← sample(memory_stream, weights = strength(m) × (1 + tag_bonus(m)))
    for m in replay_batch:
        reencoded ← encode(fetch_blobs(m))  # re-encode episodic content
        update_semantic_structures(reencoded, m.context, m.source_model)

After replay, marked memories can still cluster via density-based
methods (e.g., DBSCAN) or k-means using embedding similarity:

    cluster_i = {m_j | cos(m_j, μ_i) > merge_threshold}
    μ_i = centroid(cluster_i)

Summary nodes replace clusters:

    summary.embedding = μ_i
    summary.blob = summarize(fetch_blobs(cluster_i))  # deep local LLM backend (Gemma/LiteRT-LM or LFM2/llama.cpp)
    summary.metadata.sources = [m.id for m in cluster_i]

**Current implementation note:** same-level deep summarization now
replays only raw episodic `LONG_TERM` memories that have not yet been
assigned to a consolidation cluster. This prevents repeated
summary-of-summary compression in long chat sessions while still
allowing later raw evidence to form new sibling summaries. Higher-level
semantic summarization, if desired, should be modeled as an explicit
separate layer rather than by recursively feeding first-level summary
blobs back into the same consolidation path.

To keep consolidation latency bounded, summarization input is capped by
knob‑derived limits. Source texts are ranked by similarity to the
cluster centroid and truncated:

    max_source_texts(F̃) = round(lerp(3, 8, F̃))
    max_text_chars(F̃)  = round(lerp(300, 900, F̃))
    max_total_chars(T)  = round(lerp(1200, 3600, T))
    source_texts = topk_by_cosine(cluster_texts, k=max_source_texts)
    truncate each text to max_text_chars and total to max_total_chars
    size decoder budget from capped input length; do not hard-trim stored summary text

## Semantic Extraction

For sufficiently large clusters, semantic extraction identifies labels
and relations (deep mode only):

    extraction_batch_size = round(lerp(8, 32, T))
    min_cluster_size = round(lerp(3, 10, F))
    label_frequency_threshold = round(lerp(5, 15, T))
    extraction_interval = lerp(300, 3600, T)  # 5 min → 1 hour
    max_extractions_per_cycle = round(lerp(20, 5, T))

Extraction uses structured prompting to identify labels/tags (people,
places, organizations, concepts) and relationships (co-occurrence,
implication, contradiction). Labels are stored as surface strings (no
type/category field) **and** persisted with their own embeddings so they
can participate in retrieval and graph expansion.

**Current implementation note:** the deployed runtime now persists
extracted labels directly from each extraction result, deduped by
normalized label key within the result and reused by key across the
store. The old knob-derived repetition gate and single-label fallback
were removed because they made durable label formation too dependent on
exact repeated strings and caused obvious chat failures like
identity/name loss and severe label undercounting. The online path is
still simple: label strings are inserted as `LABEL` memories, salience
is embedding-derived, and repeated mentions update `s_max` on the
existing label node. In parallel, we continue to study a more semantic
alternative through the **offline semantic label-classifier prototype**
in `scripts/build_wordnet_label_index.py`,
`scripts/build_name_priors.py`, `scripts/build_label_training_data.py`,
`scripts/train_label_classifier.py`, and
`scripts/eval_label_classifier.py`. That prototype trains two small
models over candidate spans:

-   a **span typing** head (`identity`, `person_entity`, `place`,
    `org_project`, `topic`, `state`, `none`), and
-   a **promotion** head (`durable`, `provisional`, `ignore`)

using WordNet-derived concept priors, multilingual given-name/surname
priors, sentence context, and optional external text embeddings. The
current study build sources those priors from the multilingual
`data/surnames` tables plus U.S. Census surnames. It is currently an
offline study artifact rather than part of the deployed consolidation
path.

Label salience is derived from embeddings rather than model guesses:

    e_label = encode(label_text)
    salience(label) = clamp((cos(e_label, summary.embedding) + 1) / 2, 0, 1)

Label embeddings are cached by normalized label key. If a label already
exists in the store (or is present in the in-memory cache), its
embedding is reused instead of re-encoding, reducing consolidation
latency without changing salience computation.

If a label already exists, its stored salience is updated with
`s_max = max(s_max, salience(label))`.

## Temporal Consolidation of Changing Facts

For long-horizon personal memory support, consolidation must avoid
flattening changing facts into one timeless summary. Living
arrangements, caregivers, appointments, medications, projects, and
routines can all change over time. A memory system that simply
overwrites older facts loses exactly the temporal nuance that matters
most in care-oriented settings.

Phase 1 adds **fact assertions with temporal semantics** as a
consolidation product alongside labels and relations:

    fact_assertion_j = (
      subject,
      predicate,
      object,
      valid = [t_v_start, t_v_end),
      tx    = [t_tx_start, t_tx_end),
      provenance = {source_memory_ids, source_summary_ids},
      confidence
    )

Under this model, consolidation remains **append-only** at the fact
layer. New evidence does not erase prior assertions. Instead:

1.  A new assertion is inserted with its own valid and transaction
    interval.
2.  If it conflicts with an earlier open-ended assertion on the same
    subject/predicate, the earlier assertion can be closed or
    superseded.
3.  Both the earlier and later assertions remain queryable through their
    intervals and provenance.

This yields a cleaner separation between:

-   what was true in the world at a given time,
-   when Cortext learned that truth,
-   and when Cortext later revised its belief.

For example, “caregiver = Sarah” may be valid through March, while
“caregiver = Emily” becomes valid in April but is only learned in May.
Bitemporal consolidation preserves both the world timeline and the
system-learning timeline, preventing late-arriving information from
rewriting the past.

This temporal structure is especially important for dementia-support
scenarios. Gentle reminders should be able to express not only the
current state (“Emily is your caregiver now”) but also historical
context (“Sarah used to visit on Tuesdays”) and uncertainty about when a
change was learned (“as of the latest update”). In practice, this means
summarization should verbalize temporal transitions rather than compress
them away. Retrieval-time use of these facts is a later phase; the
current phase is about preserving the right temporal ground truth and
provenance.

The next core requirement is lifecycle maintenance. Facts should remain
more durable than raw episodes, but they should not become an unbounded
append-only residue. The current implementation therefore treats
repeated confirmations as support updates rather than new open
assertions, tracks contradiction pressure and source diversity, decays
unsupported facts toward `weak` or `archived`, and only physically
deletes low-severity archived facts after an additional retention gate.
High-severity historical facts remain archived rather than deleted so
temporal reconstruction is preserved.

## Knowledge Graph Construction

The graph comprises two node types:

-   **Memory Nodes:** Summarized embeddings from merged clusters
-   **Label Nodes:** Extracted labels/tags derived from content blobs

Edge types capture relationships:

Edge weight convention (MUST): all association weights are normalized to
\[0, 1\]. For cosine-based edges, store weight01 = clamp((cos_sim + 1) /
2, 0, 1).

-   **co_occurs:** Shared context or temporal proximity
-   **implies:** Directional correlation in embedding drift
-   **causes:** Directional correlation in embedding drift
-   **contradicts:** Strong negative similarity or constraint violation
-   **reinforces:** Frequent joint retrieval
-   **derived_from:** Links summaries to source memories
-   **similar_to:** High cosine similarity (soft equivalence)
-   **has_label:** Attaches an extracted label/tag to a node
-   **next_in_episode:** Sequential link to next memory in the same
    episode
-   **prev_in_episode:** Sequential link to previous memory in the same
    episode
-   **within_same_event:** Link for memories within the same inferred
    boundary segment

### Edge Construction

Co-occurrence edges derive from embedding similarity:

    for (m_i, m_j) in cluster.sources:
        cos_sim ← cos(m_i.embedding, m_j.embedding)
        if cos_sim > lerp(0.85, 0.95, F):
            weight01 ← clamp((cos_sim + 1) / 2, 0, 1)
            create_edge(m_i, m_j, 'co_occurs', weight01)

Causal edges derive from temporal drift:

    temporal_order ← sort_by_timestamp(cluster.sources)
    for i in range(len(temporal_order) − 1):
        m_i, m_j ← temporal_order[i], temporal_order[i+1]
        drift_vec ← m_j.embedding − m_i.embedding
        drift_mag ← ‖drift_vec‖
        if drift_mag > lerp(0.15, 0.35, T):
            weight01 ← clamp(drift_mag / 2, 0, 1)  # drift_mag ∈ [0,2] when embeddings are L2-normalized
            create_edge(m_i, m_j, 'causes', weight01)

Sequential edges preserve episode order as first-class memory:

    for i in range(len(temporal_order) − 1):
        m_i, m_j ← temporal_order[i], temporal_order[i+1]
        gap_s ← to_s(m_j.created_at) − to_s(m_i.created_at)
        boundary_prob ← m_j.boundary_score  # stored posterior from @sec-boundary
        w_seq ← clamp(exp(−gap_s / lerp(10, 60, T)) × (1 − boundary_prob), 0, 1)
        create_edge(m_i, m_j, 'next_in_episode', w_seq)
        create_edge(m_j, m_i, 'prev_in_episode', w_seq)
        if boundary_prob < 0.3:
            create_edge(m_i, m_j, 'within_same_event', w_seq)

## Graph-Augmented Retrieval

Retrieval combines vector similarity, **index-store pattern
completion**, temporal context reinstatement, and graph expansion. The
query vector is the **current accumulator centroid** (μ_acc),
L2‑normalized for vector search.

    # Query and re-rank using current memory centroid + recent context
    μ_ctx ← mean(recent_context_embeddings)  # recent context window (embedding-only)
    λ_q(F) = lerp(0.25, 0.05, F)
    q ← l2_normalize((1 − λ_q) × μ_acc + λ_q × μ_ctx)  # cached pre-reset
    q_ctx ← c_t               # temporal context from @sec-temporal-context
    if |memory_stream| == 0:
        return []  # cold-start fallback (no retrieval candidates)

    kNN_size(F) = round(lerp(96, 8, F))
    results_vec ← topK(vector_search(q, k=kNN_size(F)))
    seed_vec ← [r.id for r in results_vec]
    summary_k(F̃,S̃,T) = round(lerp(2, 6, S̃) × lerp(1.0, 0.75, F̃) × lerp(1.0, 0.85, T))
    summary_seeds ← topK(summary_cache, k=summary_k)  # in‑memory label/association cache
    seed_idx ← index_store.lookup(sparse_key(q), radius = lerp(1, 3, S))
    seed_nodes ← union(seed_vec, summary_seeds, seed_idx)

    if |seed_nodes| == 0 OR graph is empty:
        return results_vec  # deterministic fallback, skip expansion

    expanded_nodes ← graph.traverse(seed_nodes,
                                    depth=graph_depth(T),
                                    min_edge_weight=min_edge_weight(F),
                                    edge_types={'semantic','next_in_episode','prev_in_episode','within_same_event'})
    combined ← union(seed_nodes, expanded_nodes)
    min_assoc(F̃,T) = round(lerp(3, 1, F̃) × lerp(1.0, 0.85, T))
    min_label(F̃,S̃) = round(lerp(0, 3, S̃) × lerp(1.0, 0.85, F̃))
    if min_assoc > 0:
        combined ← combined ∪ {assoc | edge_type='derived_from' AND target∈combined}
    if min_label > 0:
        combined ← combined ∪ {label | edge_type='has_label' AND source∈combined}
    min_edge_weight(F) = lerp(0.08, 0.40, F)
    dup_thresh = lerp(0.985, 0.95, F) × (0.98 + 0.02T)
    eligible ← {m ∈ combined | m.created_at < write_exclusion_ts}
    eligible ← {m ∈ eligible | max_{w∈WM} cos(m.embedding, w) < dup_thresh}
    eligible ← {m ∈ eligible | source_confidence(m) ≥ lerp(0.15, 0.45, T)}

    # Diversified re-rank (MMR-style) with context reinstatement
    w_rel_raw = lerp(0.65, 0.94, F) × lerp(1.0, 0.85, S) × lerp(1.0, 0.90, T)
    w_div_raw = lerp(0.35, 0.06, F) × lerp(0.75, 1.15, S) × lerp(0.85, 0.65, T)
    [w_rel, w_div] = normalize([w_rel_raw, w_div_raw])
    w_ctx = lerp(0.15, 0.35, F) × lerp(1.0, 0.85, S)
    w_proc = lerp(0.10, 0.25, S)
    w_emotion = lerp(0.0, 0.12, S_affect)
    affect_gain(S) = lerp(1.0, 2.4, S_affect)
    weights_aff_raw = [lerp(0.30, 0.55, S_affect), lerp(0.30, 0.55, S_affect), lerp(0.10, 0.20, S_affect)]
    [w_arousal, w_emotion_raw, w_salience] = normalize(weights_aff_raw)
    affect_drive = clamp(affect_gain(S) × (w_arousal × arousal + w_emotion_raw × emotion_intensity + w_salience × salience), 0, 1)
    weights_mem_raw = [lerp(0.60, 0.80, S), lerp(0.20, 0.40, S)]
    [w_mem_emotion, w_mem_arousal] = normalize(weights_mem_raw)

    selected ← []
    while |selected| < k AND |eligible| > 0:
        m* ← argmax_{m ∈ eligible} [
                w_rel × max(0, cos(q, m.embedding)) +
                w_ctx × max(0, cos(q_ctx, m.context)) +
                w_proc × max(0, proc_sim(m)) +
                w_emotion × affect_drive × (w_mem_emotion × emotional_intensity(m) + w_mem_arousal × s_arousal_avg(m)) −
                w_div × max_{s∈selected} max(0, cos(m.embedding, s.embedding))
             ]
        selected.append(m*)
        eligible.remove(m*)
    return selected

Graph expansion uses recursive traversal with depth limits to find
related context that pure vector search might miss. Using the memory
centroid maintains consistency between vector search and graph expansion
results.

`summary_cache` is an in‑memory list of LABEL/ASSOCIATION embeddings
seeded from the label bank and updated on consolidation. This avoids a
second sqlite‑vec pass while preserving embedding‑only retrieval.

For affective gain in retrieval, we use a lighter bias (S\_{affect} =
(S, -0.06)) to preserve mid‑range affect modulation.

`proc_sim(m)` denotes procedural similarity (habit/skill match) when a
procedural embedding is available; if no procedural cue is present,
treat it as 0 so retrieval remains purely declarative.

# Interrupt Gate and Streaming Integration

The interrupt gate controls when retrieved memories enter active context
during streaming generation. The gate balances novelty value against
disruption cost. The interrupt gate operates on memory-level context,
using centroids (μ_acc) rather than individual signal embeddings for
novelty and relevance computation. All thresholds and gains in this
section are derived from the three knobs (F, S, T); no fixed behavioral
constants are introduced.

**Note:** As defined in Section 1, all knob symbols here use the
midpoint‑biased values F̃ and S̃ (T is unmodified). For affective gain, we
use a lighter bias ( \_{affect} = (S, -0.06) ) to preserve mid‑range
affect modulation.

## Marginal Utility Computation

Novelty thresholds scale with knobs and refractory state:

    τ_novelty = lerp(0.10, 0.35, F) × (1 − 0.15S) × (1 + 0.3T)
    τ_mu = lerp(0.08, 0.18, F) × (1 − 0.4S) × (1 + 0.4T)
    retrieval_thresh(F) = lerp(0.12, 0.45, F)
    retrieval_thresh_interrupt(F,S) = retrieval_thresh(F) × (1 − 0.12S)
    affect_relax_coeff(S) = lerp(0.15, 0.55, S_affect)
    affect_gain(S) = lerp(1.0, 2.4, S_affect)
    weights_aff_raw = [lerp(0.30, 0.55, S_affect), lerp(0.30, 0.55, S_affect), lerp(0.10, 0.20, S_affect)]
    [w_arousal, w_emotion, w_salience] = normalize(weights_aff_raw)
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
        if sim_mem > sim_acc:        # accepted → drop partial unit
            reset_accumulator()
        pending_abort ← false

## Streaming Pacing

Streaming retrieval is gated by cumulative drift rate within the
accumulation unit. Retrieval checks trigger when drift exceeds threshold
or at boundaries. Retrieval uses q_retrieval (the accumulator centroid)
captured before any accumulator reset for this step. Retrieval returns a
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

    since_last_s ← if last_retrieval_ts == 0 then +∞ else (now_ms() − last_retrieval_ts) / 1000
    min_gap_s ← adjacent_window(F) × dt_ema
    adjacent_ok ← (since_last_s ≥ min_gap_s)
    force_check ← (drift_acc_pacing > max_wait_drift(F))

    # Retrieval triggered when drift exceeds threshold, at memory boundary, or when drift exceeds max_wait_drift.
    # Adjacent-window throttling is bypassed on boundaries and force_check.
    if (drift_acc_pacing > pacing_thresh(S) OR should_flush OR force_check) AND
       (adjacent_ok OR should_flush OR force_check):
        trigger_check(); x_last_check ← μ_acc; drift_acc_pacing ← 0; last_retrieval_ts ← now_ms()

High Sensitivity produces frequent checks triggered by small content
shifts; high Focus enforces strict drift limits. Memory boundaries
(<a href="#sec-boundary" class="quarto-xref">Section 6.4.3</a>) also
trigger retrieval checks to ensure context updates align with natural
thought transitions.

# Experimental Results

We present preliminary experimental results collected from live chat
sessions to validate the adaptive mechanisms.

This section mixes two experiment families:

-   **deterministic core benchmarks** in `examples/benchmark`, which are
    now reported from real `EmbeddingGemma/llama.cpp` reruns where
    noted; and
-   **topical-chat / corpus harness runs** in
    `examples/topical_chat_analysis`, where older historical logs used
    the then-active semantic encoder path and newer reruns explicitly
    record the resolved backend in `run.log`.

Going forward, rebuilt corpus ablations are being normalized onto a
study-only dataset stack that does **not** change the Cortext runtime
API or production config surface. The prep and scorer scripts live under
`scripts/` and keep the existing JSONL conversation contract unchanged:

-   `scripts/prepare_msc.py` for multi-session continuity and
    consolidation studies
-   `scripts/prepare_taskmaster.py` for procedural and sequential
    studies
-   `scripts/prepare_longmemeval.py` plus `scripts/score_longmemeval.py`
    for held-out long-memory QA
-   `scripts/generate_mechanism_eval_pack.py` plus
    `scripts/score_mechanism_eval.py` for contradiction-heavy synthetic
    mechanism probes

Wrapper scripts (`scripts/run_msc_study.sh`,
`scripts/run_taskmaster_study.sh`, `scripts/run_longmemeval_study.sh`,
and `scripts/run_mechanism_eval.sh`) call the existing harness with
explicit turn limits and existing experiment-only toggles. They do not
introduce new public runtime flags; their purpose is to keep future
rebuilt ablations reproducible and clearly separated from the historical
TopicalChat-era tables below.

For the topical-chat family, treat any subsection explicitly labeled
with an **Apr 4, 2026** EmbeddingGemma rerun note as the current source
of record. Unlabeled corpus tables remain historical results from
earlier encoder configurations.

An additional **Apr 4, 2026 EmbeddingGemma + default LFM2.5 addendum**
later in this section reports completed quick reruns on the rebuilt
branch where embeddings resolve to `EmbeddingGemma/llama.cpp` and auto
deep-backend selection now prefers `LFM2.5/llama.cpp`. Those addendum
tables are the current short-horizon reference points for the rebuilt
default stack.

At the time of the historical topical-chat runs, deep consolidation
summarization/labeling used the Gemma LiteRT path (`gemma-3n-e2b` via
`third_party/litert_lm`). The current implementation also supports an
LFM2 `llama.cpp` path for deep consolidation. Evaluation logs include
lexical overlap (token Jaccard) and **semantic overlap** (cosine under
the active text encoder backend) for retrieval and interrupt quality. A
multi-participant harness (`scripts/run_memory_harness.py`) interleaves
conversations to stress long-horizon recall under shared-memory load.

For reproducibility, the analysis runner supports **deterministic
synthetic timing** (`--deterministic`, `--seed`), which removes
wall‑clock variance while preserving cadence‑derived timing for boundary
and interrupt gating. Snapshot configs record the seed and synthetic
clock parameters.

To quantify consolidation utility, we also track:

-   **retrieval_summary_hit_rate:** share of retrieval turns containing
    consolidated summaries (ASSOCIATION nodes).
-   **summary_hit_overlap_mean:** semantic overlap (backend cosine) of
    the best summary hit on those turns.
-   **retrieval_summary_only_turn_rate:** share of retrieval turns where
    summaries are the only retrieved candidates.

For interrupt evaluation, we report precision/recall using the
knob-derived interrupt threshold (`retrieval_thresh_interrupt(F,S)`)
applied to best semantic overlap:

-   **interrupt_precision / interrupt_recall**
-   **false-positive / false-negative rates**
-   **interrupt_semantic_delta:** mean semantic overlap on interrupt
    turns minus non-interrupt turns
-   **interrupt_gate_fail\_\* rates:** per-gate failure rates among
    semantic-positive non-interrupt turns
-   **affect_drive_mean:** mean affect drive (arousal/emotion/salience
    composite)
-   **retrieval_emotion_bonus_mean:** mean emotion bonus applied during
    retrieval ranking

## Bitemporal Fact Evaluation

The current branch now evaluates the bitemporal fact layer
**core-first**. The source of record is the deterministic fact-store and
retrieval harnesses that measure whether Cortext preserves temporal
truth, temporal belief, support quality, and bounded retention under
delayed, conflicting, noisy, and repeated evidence traces. The chat
example is not treated as a separate fact-grounding surface; correctness
is established in the core before any downstream presentation layer is
considered.

The important distinction is that these are **assistive retrieval and
recall metrics**, not merely database-correctness checks. The goal is to
measure whether temporal fact modeling helps Cortext surface the right
current information, preserve older truths for reminiscence, and avoid
stale-fact intrusions when the user implicitly means “now.”

The relevant metrics are:

-   **current fact accuracy:** whether current-oriented retrieval
    surfaces the correct present-world fact
-   **historical fact accuracy:** whether `valid_at(t)` recovers what
    was true at world time `t` in internal deterministic evaluation
-   **belief-at-time accuracy:** whether `known_at(t)` reconstructs what
    Cortext would have believed at system time `t` in internal
    deterministic evaluation
-   **stale fact intrusion rate:** how often superseded facts appear
    first on present-oriented tasks
-   **retrieval precision/recall under temporal change:** whether
    fact-aware retrieval improves the supporting evidence surfaced
    around changing routines, caregivers, locations, and schedules
-   **routine continuity rate:** whether stable routine context remains
    available as background support when `Stability` is high
-   **provenance-grounded retrieval rate:** whether the top retrieved
    evidence remains directly linked to the matched fact rather than an
    unlinked semantic guess

These metrics should be reported with **severity weighting** rather than
uniform averaging. Mistakes about medication, caregiver, location,
schedule, and safety matter more than mistakes about preferences or
incidental details. Longitudinal reporting should also include stability
measures such as fact flip rate, stale resurfacing rate, and
time-to-correct after delayed evidence.

The recommended evaluation scenarios are controlled temporal-change
tasks: caregiver transitions, same-day schedule changes, location
changes, repeated routines with one-off exceptions, and old-home versus
current-home questions. Internal evaluation should still support three
query families: present-oriented (`current`), historical
(`valid_at(t)`), and historical-belief (`known_at(t)`). The primary
validation path is therefore the deterministic core harnesses rather
than prompt formatting or live-LLM output.

The companion retrieval benchmark remained green on the same branch:

-   benchmark: `examples/benchmark/cortext_bitemporal_retrieval_bench`
-   scenarios: current caregiver, `valid_at` location, `known_at`
    delayed move, provenance-linked schedule
-   result: **4 / 4 passed** (`accuracy = 1.0`)

**Deterministic phase-6 core robustness gates (Apr 1, 2026):**

-   benchmark: `examples/benchmark/cortext_bitemporal_robustness_bench`
-   scenario families: delayed evidence, conflicting low-confidence
    noise, delayed-plus-conflicting correction, repeated legitimate
    changes, repeated routine confirmations, routine exception and
    return
-   reported gates: weighted current accuracy, weighted historical
    accuracy, weighted belief-at-time accuracy, stale-fact intrusion
    rate, erroneous supersession count, unexpected flip count,
    time-to-correct, retrieval latency overhead
-   result: **6 / 6 passed**
-   aggregate: `weighted_current_accuracy = 1.0`,
    `weighted_historical_accuracy = 1.0`,
    `weighted_belief_accuracy = 1.0`,
    `weighted_stale_intrusion_rate = 0.0`,
    `erroneous_supersession_count = 0`, `unexpected_flip_count = 0`,
    `mean_time_to_correct_updates = 0.0`, `mean_retrieval_ms = 0.995`

These gates are intentionally core-only. They validate fact ingestion,
supersession policy, temporal querying, and fact-aware retrieval
directly, without depending on live chat prompting. This is the point at
which the branch can reasonably claim that current, historical, and
belief-at-time fact handling are stable enough for later ablation and
chat-facing integration work.

**Deterministic phase-8 expanded core ablation matrix (Apr 1, 2026):**

-   benchmark: `examples/benchmark/cortext_bitemporal_ablation_bench`
-   scenario families: caregiver transition, delayed location knowledge,
    conflicting caregiver noise, provenance-linked schedule retrieval,
    repeated legitimate location changes, routine exception, old-home vs
    current-home, updated preference, medication change, and a weak
    today-vs-usual schedule change
-   ablations: fact layer off, current-only collapse, explicit `current`
    / `valid_at` / `known_at` reporting, fact boost off/weak/strong,
    stale penalty off/moderate/strong, provenance any-fact-match,
    routine-biased vs recency-biased presets, and a full `3 x 3 x 3`
    F/S/T sweep on a representative subset
-   default result: **10 / 10 passed**, with
    `query_mode_current_accuracy = 1.0`,
    `query_mode_valid_at_accuracy = 1.0`,
    `query_mode_known_at_accuracy = 1.0`,
    `weighted_current_accuracy = 1.0`,
    `weighted_historical_accuracy = 1.0`,
    `weighted_belief_accuracy = 1.0`,
    `current_vs_usual_clarification_rate = 1.0`,
    `erroneous_supersession_count = 0`, and `unexpected_flip_count = 0`
-   key deltas:
    -   fact layer off dropped weighted current accuracy to **0.323**
        and raised weighted stale intrusion to **0.548**
    -   current-only collapse dropped weighted historical accuracy to
        **0.0** and weighted belief-at-time accuracy to **0.318**
    -   provenance relaxation (`any_fact_match`) raised unsupported top
        hits from **0.065** to **0.161**
    -   stale penalty off raised weighted stale intrusion from **0.065**
        to **0.161**
    -   strong fact boost outperformed boost-off on weighted current
        accuracy (`1.0` vs `0.710`)
    -   the routine-vs-recency probe separated as intended:
        routine-biased scoring preferred the stable routine, while
        recency-biased scoring and retrieval flipped to the current
        exception (`top_memory 100 -> 101`)

These expanded ablations matter because they move the claim from “the
default core works” to “the default core remains best under a broader
causal matrix.” The branch can now point to a deterministic core result:
bitemporal history is required for historical and belief-at-time
behavior, direct provenance links matter for support quality, stale
penalties materially reduce present-oriented intrusion, and the
routine-versus-recency tradeoff is now exposed explicitly in the scoring
path rather than left implicit in chat behavior.

**EmbeddingGemma rerun audit (Apr 4, 2026):**

-   log:
    `logs/embeddinggemma_ablations_20260404_015235/bitemporal_ablation.log`
-   encoder: `EmbeddingGemma/llama.cpp` via
    `models/llama_cpp/embeddinggemma-300M-Q8_0.gguf`
-   default result still passed **10 / 10** scenarios with
    `query_mode_current_accuracy = 1.0`,
    `query_mode_valid_at_accuracy = 1.0`,
    `query_mode_known_at_accuracy = 1.0`, and
    `unexpected_flip_count = 0`
-   however, several ablation-separation checks that were previously
    reported as causal deltas collapsed to **0** under real embeddings:
    provenance relaxation no longer raises unsupported top hits,
    stale-off vs stale-strong no longer separates stale intrusions, and
    the routine-vs-recency retrieval probe no longer flips top memory
    even though the score-gap signs still separate. Treat the older
    synthetic-encoder delta claims as superseded until those scenarios
    are retuned against the real encoder path.

**Deterministic phase-9 fact lifecycle gates (Apr 1, 2026):**

-   benchmark: `examples/benchmark/cortext_bitemporal_lifecycle_bench`
-   scenario families: support decay, archive-vs-delete,
    repeated-confirmation compression, evidence-loss resilience,
    severity-aware retention
-   default policy: archive-first; unsupported low-severity facts can be
    deleted after archival, while high-severity historical facts are
    retained
-   result: **5 / 5 passed** (`accuracy = 1.0`)

Phase 9 closes the remaining gap between the bitemporal fact layer and
the rest of Cortext’s self-maintaining memory system. Facts are still
more durable than individual episodic traces, but they no longer behave
as an unbounded append-only log. Repeated confirmations are compressed
into support statistics, unsupported facts decay to `weak` or
`archived`, and only low-severity archived facts are physically deleted
by default. The corresponding negative controls show why the default
matters: turning off support decay preserves stale low-value facts too
aggressively, disabling deletion prevents storage from shrinking,
disabling compression inflates support from repeated confirmations, and
disabling high-severity historical preservation breaks historical
recovery after evidence loss.

**Deterministic eviction ablation gates (Apr 3, 2026):**

-   benchmark: `examples/benchmark/cortext_eviction_ablation_bench`
-   scenario families: pure decay, selective reinforcement, flashbulb
    protection, recovery from near-eviction, long-horizon simulation (7
    days), fact-linked eviction, strength distribution, crowding
    pressure
-   ablations: trace count (1/2/3/4), coupling off/strong, reinforcement
    off/weak/strong, periphery cutoff low/high, half-life short/long,
    flashbulb off, equal weights, fact-evidence floor off, and a full
    `3 × 3 × 3` F/S/T sweep on a representative subset
-   default result (F=S=T=0.5): `mean_eviction_step = 220`,
    `selectivity_ratio = 100.0`, `flashbulb_advantage = 1.0`,
    `recovery_success_rate = 1.0`, `steady_state_count = 35`,
    `eviction_rate_per_hour = 0.79`, `fact_linked_eviction_rate = 0.0`
-   key deltas:
    -   single-trace collapsed mean eviction step from **220** to **12**
        (18× faster eviction, no power-law tail)
    -   coupling-off reduced mean eviction step from **220** to **27**
        and destroyed selectivity (used memories evicted alongside
        unused)
    -   four-trace extended mean eviction step to **500** with zero
        eviction over 500 simulated minutes
    -   half-life short (60 s) caused immediate eviction at step 1;
        half-life long (7200 s) prevented any eviction over the full run
    -   cutoff-high (0.20) reduced mean eviction step to **158**;
        cutoff-low (0.03) extended it to **368** with full recovery
        support
    -   flashbulb-off removed the survival advantage for emotionally
        marked memories (advantage dropped from **1.0** to **0.0**)
    -   equal weights produced mean eviction step **217**, close to
        default (**220**), confirming the retuned weight range now
        provides appropriate T-dependent balance
    -   fact-evidence floor off raised `fact_linked_eviction_rate` from
        **0.0** to **1.0** — without the floor, all fact-supporting
        episodic memories are evicted; with it, fact-linked memories
        survive as long as the supporting fact remains active
    -   F/S/T sweep: T dominates eviction dynamics — low T (0.2) causes
        eviction within 7–30 steps while high T (0.9) prevents eviction
        entirely; S and F have secondary effects through reinforcement
        scaling

**EmbeddingGemma rerun status (Apr 4, 2026):**

-   log:
    `logs/embeddinggemma_ablations_20260404_014916/eviction_ablation.log`
-   encoder: `EmbeddingGemma/llama.cpp` via
    `models/llama_cpp/embeddinggemma-300M-Q8_0.gguf`
-   status at write time: rerun started and confirmed to be on the real
    encoder path, but the full long-horizon matrix was still executing
    when this note was written because the benchmark spends most of its
    time in the seven-day decay / crowding loops rather than in
    embedding generation.

Three ablation-driven improvements were applied based on the initial
gate results:

1.  **Fact-evidence floor**
    (`fact_floor(T) = lerp(0, periphery_cutoff(T), T)`): Memories linked
    to active facts via `fact_evidence` are protected from eviction when
    their strength remains above the T-scaled floor. This closes the
    episodic-evidence gap where the eviction system would discard
    memories that serve as provenance for active structured facts. The
    `fact_floor_off` negative control confirms the protection is
    necessary: without it, fact-supporting evidence is evicted
    identically to unlinked noise.

2.  **Uniform reinforcement distribution**: The original front-loaded
    boost (60%/30%/10% to fast/med/slow traces) was replaced with
    uniform per-trace injection (`reinforcement / N_traces`). The
    original boost had no measurable effect on eviction dynamics because
    its contribution was absorbed by trace clamping at 1.0. Uniform
    distribution ensures all active traces receive meaningful S- and
    F-scaled reinforcement, improving long-term retention for used
    memories.

3.  **Widened T weight range** (`w_fast: 0.40−0.25T` replacing
    `0.55−0.20T`): The original weights gave the fast trace 50% of
    normalized weight at T=0.5, meaning half the strength score came
    from a trace that decays in minutes. The retuned range shifts from
    episodic-dominant (fast=40% at T=0) to semantic-dominant (fast=15%
    at T=1) with a balanced midpoint. Mean eviction step increased from
    172 to 220 at T=0.5.

**Deterministic memory system integration ablation gates (Apr 4,
2026):**

-   benchmark: `examples/benchmark/cortext_memory_system_ablation_bench`
-   studies: consolidation-eviction race, end-to-end fact emergence,
    memory-fact co-decay, retrieval-reinforcement feedback,
    cross-consolidation fact inheritance
-   result: **5 / 5 passed**
-   key findings:
    -   **Consolidation-eviction race**: Early consolidation (step 50)
        successfully extracts facts from all 30 candidate memories. Late
        consolidation (step 250) finds zero candidates — all evidence
        has already been evicted. This confirms that consolidation
        timing must outpace the decay cycle or facts will never emerge
        from episodic evidence.
    -   **End-to-end fact emergence**: The full pipeline (consolidation
        → clustering → extraction) produced **3 facts** across **3
        distinct predicates** (caregiver, location, medication) from **4
        clusters** of 20 seeded memories. Disabling extraction yielded 0
        facts with 4 summaries; disabling consolidation yielded 0 of
        both. Each layer is strictly necessary.
    -   **Memory-fact co-decay**: With the fact floor active,
        fact-linked memories survive at **100%** while unlinked controls
        evict to **0%** after 250 decay steps. Archiving the linked fact
        drops the floor: the memory then evicts in the next phase
        (**0%** survival). Without the floor, fact-linked and unlinked
        memories evict identically. This confirms the co-decay coupling:
        fact lifecycle governs evidence retention.
    -   **Retrieval-reinforcement feedback**: With fact boost,
        fact-linked memories accumulate a strength gap of **+0.30** over
        unlinked memories and dominate the top-5 retrieval positions
        (**100%**). Disabling the fact layer **reverses** the gap to
        **−0.30** (unlinked now dominate). This confirms the feedback
        loop is controlled by the fact layer, not runaway — toggling the
        fact layer flips the outcome completely.
    -   **Cross-consolidation fact inheritance**: Re-extracting the same
        fact (alice/caregiver/emily) from a new summary correctly merges
        into the existing fact with `confirmation_count` incrementing
        from 1 to **2** and zero duplicate facts. Extracting a
        conflicting fact (emily vs sarah) correctly supersedes the prior
        assertion. The fact deduplication and supersession logic handles
        re-consolidation without evidence inflation.

**EmbeddingGemma rerun audit (Apr 4, 2026):**

-   log:
    `logs/embeddinggemma_ablations_20260404_015134/memory_system_ablation.log`
-   encoder: `EmbeddingGemma/llama.cpp` via
    `models/llama_cpp/embeddinggemma-300M-Q8_0.gguf`
-   result: **5 / 5 passed**
-   the retrieval-reinforcement study still separates cleanly under the
    real encoder path: `strength_gap = +0.2996` with facts enabled and
    `strength_gap = -0.2996` with the fact layer disabled, with top-5
    fact fraction flipping from **1.0** to **0.0**.

## Exhaustive Operation-Family Ablation Matrix (Apr 4, 2026)

To answer the stricter removal question, we added a separate
**operation-family matrix** in `docs/operation-matrix.md` and ran its
full power set rather than pretending the atomic metric-site power set
was tractable. The matrix defines **12 independently removable non-fact
families**:

-   source confidence
-   predictive retrieval
-   constructive recall
-   procedural memory
-   metacognitive control
-   affect interrupt
-   affect retrieval
-   flashbulb consolidation
-   neuromodulation
-   meta-learning
-   reinforcement edges
-   sequential edges

This yields an exhaustive deterministic sweep of **`2^12 = 4096`
combinations**. The benchmark is
`examples/benchmark/cortext_full_operation_ablation_bench`, and the
recorded run is:

-   log:
    `logs/full_operation_ablation_20260404/full_operation_ablation.log`

The result is useful because it is not purely “everything on wins.” The
full mask and the best mask tie at **10.0 / 12.0**, and the smallest
best mask has **11** families enabled:

-   minimal best mask disabled only: `affect_interrupt`
-   therefore, on this deterministic claim-audit suite,
    **`affect_interrupt` is the first deterministic pruning candidate**

The marginal table from the exhaustive sweep was:

<table style="width:100%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 22%" />
<col style="width: 22%" />
<col style="width: 22%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th>family</th>
<th style="text-align: right;">mean marginal contribution</th>
<th style="text-align: right;">max score without family</th>
<th style="text-align: right;">essential for best score</th>
<th>verdict</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>source_confidence</code></td>
<td style="text-align: right;">0.1667</td>
<td style="text-align: right;">9.6667</td>
<td style="text-align: right;">1</td>
<td>narrow but required</td>
</tr>
<tr>
<td><code>predictive_retrieval</code></td>
<td style="text-align: right;">1.0000</td>
<td style="text-align: right;">9.0000</td>
<td style="text-align: right;">1</td>
<td>core</td>
</tr>
<tr>
<td><code>constructive_recall</code></td>
<td style="text-align: right;">1.0000</td>
<td style="text-align: right;">9.0000</td>
<td style="text-align: right;">1</td>
<td>core</td>
</tr>
<tr>
<td><code>procedural_memory</code></td>
<td style="text-align: right;">1.0000</td>
<td style="text-align: right;">9.0000</td>
<td style="text-align: right;">1</td>
<td>core</td>
</tr>
<tr>
<td><code>metacognitive_control</code></td>
<td style="text-align: right;">0.5000</td>
<td style="text-align: right;">9.3333</td>
<td style="text-align: right;">1</td>
<td>narrow but required</td>
</tr>
<tr>
<td><code>affect_interrupt</code></td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">10.0000</td>
<td style="text-align: right;">0</td>
<td>cut candidate</td>
</tr>
<tr>
<td><code>affect_retrieval</code></td>
<td style="text-align: right;">1.0000</td>
<td style="text-align: right;">9.0000</td>
<td style="text-align: right;">1</td>
<td>core</td>
</tr>
<tr>
<td><code>flashbulb_consolidation</code></td>
<td style="text-align: right;">1.0000</td>
<td style="text-align: right;">9.0000</td>
<td style="text-align: right;">1</td>
<td>core</td>
</tr>
<tr>
<td><code>neuromodulation</code></td>
<td style="text-align: right;">1.0000</td>
<td style="text-align: right;">9.0000</td>
<td style="text-align: right;">1</td>
<td>core</td>
</tr>
<tr>
<td><code>meta_learning</code></td>
<td style="text-align: right;">1.0000</td>
<td style="text-align: right;">9.0000</td>
<td style="text-align: right;">1</td>
<td>core</td>
</tr>
<tr>
<td><code>reinforcement_edges</code></td>
<td style="text-align: right;">1.0000</td>
<td style="text-align: right;">9.0000</td>
<td style="text-align: right;">1</td>
<td>core</td>
</tr>
<tr>
<td><code>sequential_edges</code></td>
<td style="text-align: right;">1.0000</td>
<td style="text-align: right;">9.0000</td>
<td style="text-align: right;">1</td>
<td>core</td>
</tr>
</tbody>
</table>

Two points matter.

First, the system no longer looks like arbitrary gears bolted together.
In this deterministic removal sweep, **11 of the 12 families were
necessary to preserve the best suite score**. Predictive retrieval,
constructive recall, procedural memory, affect retrieval, flashbulb
consolidation, neuromodulation, meta-learning, reinforcement edges, and
sequential edges all behaved like hard dependencies for at least one
claimed behavior in the suite.

Second, the sweep also found a real deterministic reduction candidate.
`affect_interrupt` was the **only** family with
`max_score_without_family == full_mask_score`. That does **not** prove
it never matters in live chat; the live-model confirmation pass later in
this section shows that the interrupt-only affect path still improves
recall on the rebuilt stack. So the correct interpretation is narrower:
on the current deterministic claim-audit suite, `affect_interrupt` is
removable without losing the best suite score, but it is **not** yet a
safe global deletion candidate.

The two narrow families were `source_confidence` and
`metacognitive_control`. `source_confidence` is the interesting case:
its standalone full-mask component stayed weak in this suite, but its
marginal contribution remained positive because it helps the
**metacognitive caution** pathway avoid weak unsupported returns. In
other words, it is currently an interaction-dependent mechanism rather
than a broad direct win.

### Boundary + Fact Extension Matrix (Apr 4, 2026)

The 4096-combination family sweep above intentionally excluded the
previously separate **boundary** and **bitemporal fact** subfamilies. We
therefore ran a second deterministic extension sweep over:

-   `boundary_surprisal`
-   `boundary_natural`
-   `fact_layer`
-   `fact_history`
-   `fact_stale_penalty`
-   `fact_provenance`

Extension matrix definition: `docs/operation-matrix.md`  
Extension log:
`logs/boundary_fact_extension_ablation_20260404/boundary_fact_extension_ablation.log`

This second sweep covers the full `2^6 = 64` power set for those omitted
families.

<table>
<thead>
<tr>
<th>family</th>
<th style="text-align: right;">mean marginal</th>
<th style="text-align: right;">max score without</th>
<th style="text-align: right;">essential for best</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>boundary_surprisal</code></td>
<td style="text-align: right;">0.000000</td>
<td style="text-align: right;">6.000000</td>
<td style="text-align: right;">0</td>
</tr>
<tr>
<td><code>boundary_natural</code></td>
<td style="text-align: right;">2.000000</td>
<td style="text-align: right;">4.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>fact_layer</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">5.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>fact_history</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">5.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>fact_stale_penalty</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">5.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>fact_provenance</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">5.000000</td>
<td style="text-align: right;">1</td>
</tr>
</tbody>
</table>

Best masks:

<table>
<colgroup>
<col style="width: 17%" />
<col style="width: 23%" />
<col style="width: 23%" />
<col style="width: 17%" />
<col style="width: 17%" />
</colgroup>
<thead>
<tr>
<th>rank</th>
<th style="text-align: right;">score</th>
<th style="text-align: right;">bits_on</th>
<th>enabled</th>
<th>disabled</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td style="text-align: right;">6.000000</td>
<td style="text-align: right;">5</td>
<td><code>boundary_natural,fact_layer,fact_history,fact_stale_penalty,fact_provenance</code></td>
<td><code>boundary_surprisal</code></td>
</tr>
<tr>
<td>2</td>
<td style="text-align: right;">6.000000</td>
<td style="text-align: right;">6</td>
<td><code>boundary_surprisal,boundary_natural,fact_layer,fact_history,fact_stale_penalty,fact_provenance</code></td>
<td><code>none</code></td>
</tr>
</tbody>
</table>

**Observations:** The extension sweep says the **fact layer is not
decorative**. Direct fact lifting, historical retrieval, stale-fact
suppression, and strict provenance all remain load-bearing in this
deterministic extension suite; disabling any one of those four lowers
the best achievable score from **6/6** to **5/6**. `boundary_natural` is
also clearly required, dropping the best score to **4/6** when removed.

`boundary_surprisal` is the only removable family inside this narrow
extension suite. That should not be over-read as a global deletion
verdict. The longer-horizon EmbeddingGemma boundary rerun reported later
in this section still shows a material `no_surprisal` change in
boundary-hit share, so the conservative reading is that surprisal is
redundant in this micro-suite but still active on naturalistic boundary
workloads.

### Live-Model Confirmation of Cut Candidates (Apr 4, 2026)

The deterministic family sweeps above identify **candidate** reductions,
not safe deletions. We therefore ran a compact live-model confirmation
pass on the rebuilt `EmbeddingGemma/llama.cpp` + default
`LFM2.5/llama.cpp` stack before treating any family as removable.

Runs:

-   TopicalChat interrupt-only affect:
    `logs/live_cut_confirmation_20260404/topical_affect_interrupt`
-   TopicalChat affect off:
    `logs/live_cut_confirmation_20260404/topical_affect_off`
-   EmpatheticDialogues interrupt-only affect:
    `logs/live_cut_confirmation_20260404/empathetic_affect_interrupt`
-   EmpatheticDialogues affect off:
    `logs/live_cut_confirmation_20260404/empathetic_affect_off`
-   TopicalChat boundary default:
    `logs/live_cut_confirmation_20260404/topical_boundary_default`
-   TopicalChat boundary no-surprisal:
    `logs/live_cut_confirmation_20260404/topical_boundary_no_surprisal`

**Affect interrupt confirmation**

<table>
<colgroup>
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
</colgroup>
<thead>
<tr>
<th>dataset</th>
<th>mode</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">precision</th>
<th style="text-align: right;">recall</th>
<th style="text-align: right;">fn_rate</th>
<th style="text-align: right;">retrieval_thresh_mean</th>
<th style="text-align: right;">boundary_mult_eff_mean</th>
<th style="text-align: right;">interrupt_semantic_overlap_mean</th>
</tr>
</thead>
<tbody>
<tr>
<td>TopicalChat 120 turns</td>
<td><code>interrupt</code></td>
<td style="text-align: right;">0.1583</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.950</td>
<td style="text-align: right;">0.050</td>
<td style="text-align: right;">0.23397</td>
<td style="text-align: right;">1.48048</td>
<td style="text-align: right;">0.79170</td>
</tr>
<tr>
<td>TopicalChat 120 turns</td>
<td><code>off</code></td>
<td style="text-align: right;">0.1417</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.850</td>
<td style="text-align: right;">0.150</td>
<td style="text-align: right;">0.23990</td>
<td style="text-align: right;">1.51800</td>
<td style="text-align: right;">0.78417</td>
</tr>
<tr>
<td>EmpatheticDialogues 360 turns</td>
<td><code>interrupt</code></td>
<td style="text-align: right;">0.1471</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.000</td>
<td style="text-align: right;">0.23514</td>
<td style="text-align: right;">1.48785</td>
<td style="text-align: right;">0.82955</td>
</tr>
<tr>
<td>EmpatheticDialogues 360 turns</td>
<td><code>off</code></td>
<td style="text-align: right;">0.1471</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.000</td>
<td style="text-align: right;">0.23990</td>
<td style="text-align: right;">1.51800</td>
<td style="text-align: right;">0.82490</td>
</tr>
</tbody>
</table>

**Observations:** `affect_interrupt` is **not** safe to remove. On
TopicalChat it raises interrupt recall by **10 points** (`0.85` →
`0.95`) while leaving precision unchanged, and it does so exactly
through the intended gate-relax path: the effective retrieval threshold
and boundary multiplier both fall. On EmpatheticDialogues the decision
surface is already saturated, but the same threshold relaxation remains
visible and semantic overlap is slightly higher with interrupt-only
affect enabled. The deterministic sweep was therefore correctly
identifying a **micro-suite pruning candidate**, not a live-stack
deletion candidate.

**Boundary surprisal confirmation**

For the boundary candidate we used the same proxy as the main boundary
study in this paper: the share of stored memories with
`boundary_score >= 0.5`, measured directly from each run DB.

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
<th>condition</th>
<th style="text-align: right;">memories</th>
<th style="text-align: right;">boundary_hits (score &gt;= 0.5)</th>
<th style="text-align: right;">rate</th>
<th style="text-align: right;">boundary_score_mean</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">interrupt_recall</th>
</tr>
</thead>
<tbody>
<tr>
<td>default</td>
<td style="text-align: right;">32</td>
<td style="text-align: right;">27</td>
<td style="text-align: right;">84.38%</td>
<td style="text-align: right;">0.2963</td>
<td style="text-align: right;">0.1833</td>
<td style="text-align: right;">0.9565</td>
</tr>
<tr>
<td><code>CORTEXT_BOUNDARY_DISABLE_SURPRISAL=1</code></td>
<td style="text-align: right;">40</td>
<td style="text-align: right;">37</td>
<td style="text-align: right;">92.50%</td>
<td style="text-align: right;">0.3519</td>
<td style="text-align: right;">0.2333</td>
<td style="text-align: right;">1.0000</td>
</tr>
</tbody>
</table>

**Observations:** `boundary_surprisal` is also **not** a safe global
cut. Disabling surprisal materially raises boundary-hit share on this
compact live run (**84.38%** → **92.50%**) and increases interrupt
frequency, which is directionally consistent with the longer-horizon
EmbeddingGemma boundary rerun reported elsewhere in this section. So the
deterministic extension sweep again identified a narrow micro-suite
redundancy, not a universal deletion candidate.

**Current pruning guidance:** the family sweeps remain valuable for
finding where to look first, but the live confirmation pass shows that
neither `affect_interrupt` nor `boundary_surprisal` is ready for
outright removal on the current rebuilt stack. The stronger current
conclusion is that the branch is more organic than decorative, and that
apparent deterministic redundancies must still survive a live-model
confirmation pass before any operation family is cut.

### Subcomponent Completion Matrix (Apr 4, 2026)

To close the matrix at the next level down, we ran a separate
deterministic subcomponent sweep over the currently separable
multi-toggle families:

-   boundary: `pressure`, `surprisal`, `natural`
-   flashbulb: `percentile`, `rate`, `arousal`
-   metacognitive control: `tot_recovery`, `unknown_caution`,
    `confidence_decay`
-   affect: `interrupt_path`, `retrieval_path`
-   neuromodulation: `write_scale`, `competition_scale`,
    `reconsolidation_scale`, `value_gain`

Benchmark: `examples/benchmark/cortext_subcomponent_matrix_bench`  
Log: `logs/flashbulb_rate_rewrite_20260404/subcomponent_matrix.log`

#### Boundary subcomponents

<table style="width:100%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 22%" />
<col style="width: 22%" />
<col style="width: 22%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th>subcomponent</th>
<th style="text-align: right;">mean marginal</th>
<th style="text-align: right;">max score without</th>
<th style="text-align: right;">essential for best</th>
<th>verdict</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pressure</code></td>
<td style="text-align: right;">0.000000</td>
<td style="text-align: right;">3.000000</td>
<td style="text-align: right;">0</td>
<td>micro-suite redundant</td>
</tr>
<tr>
<td><code>surprisal</code></td>
<td style="text-align: right;">0.000000</td>
<td style="text-align: right;">3.000000</td>
<td style="text-align: right;">0</td>
<td>micro-suite redundant</td>
</tr>
<tr>
<td><code>natural</code></td>
<td style="text-align: right;">3.000000</td>
<td style="text-align: right;">0.000000</td>
<td style="text-align: right;">1</td>
<td>core</td>
</tr>
</tbody>
</table>

Best masks:

<table>
<thead>
<tr>
<th>rank</th>
<th style="text-align: right;">score</th>
<th>enabled</th>
<th>disabled</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td style="text-align: right;">3.000000</td>
<td><code>natural</code></td>
<td><code>pressure,surprisal</code></td>
</tr>
<tr>
<td>2</td>
<td style="text-align: right;">3.000000</td>
<td><code>pressure,natural</code></td>
<td><code>surprisal</code></td>
</tr>
<tr>
<td>3</td>
<td style="text-align: right;">3.000000</td>
<td><code>surprisal,natural</code></td>
<td><code>pressure</code></td>
</tr>
<tr>
<td>4</td>
<td style="text-align: right;">3.000000</td>
<td><code>pressure,surprisal,natural</code></td>
<td><code>none</code></td>
</tr>
</tbody>
</table>

**Observations:** In the local calibration suite, `natural` is the only
truly load-bearing boundary subcomponent. Both `pressure` and
`surprisal` are redundant there. That still does **not** make either one
a safe global cut. `boundary_surprisal` already failed live-model
deletion confirmation above, and `boundary_pressure` has not yet been
subjected to the same live follow-up.

#### Flashbulb subcomponents

<table style="width:100%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 22%" />
<col style="width: 22%" />
<col style="width: 22%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th>subcomponent</th>
<th style="text-align: right;">mean marginal</th>
<th style="text-align: right;">max score without</th>
<th style="text-align: right;">essential for best</th>
<th>verdict</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>percentile</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">1</td>
<td>required</td>
</tr>
<tr>
<td><code>rate</code></td>
<td style="text-align: right;">0.000000</td>
<td style="text-align: right;">2.000000</td>
<td style="text-align: right;">0</td>
<td>neutral stabilizer</td>
</tr>
<tr>
<td><code>arousal</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">1</td>
<td>required</td>
</tr>
</tbody>
</table>

Best masks:

<table>
<thead>
<tr>
<th>rank</th>
<th style="text-align: right;">score</th>
<th>enabled</th>
<th>disabled</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td style="text-align: right;">2.000000</td>
<td><code>percentile,arousal</code></td>
<td><code>rate</code></td>
</tr>
<tr>
<td>2</td>
<td style="text-align: right;">2.000000</td>
<td><code>percentile,rate,arousal</code></td>
<td><code>none</code></td>
</tr>
<tr>
<td>3</td>
<td style="text-align: right;">1.000000</td>
<td><code>percentile</code></td>
<td><code>rate,arousal</code></td>
</tr>
</tbody>
</table>

**Observations:** After the flashbulb-rate rewrite, `percentile` and
`arousal` remain the load-bearing flashbulb paths in the deterministic
suite, while `rate` is now neutral instead of antagonistic. Keeping it
no longer hurts the best achievable score, but removing it still does
not improve anything. That matches the post-rewrite live EmbeddingGemma
flashbulb rerun later in this section, where baseline and `no_rate`
became identical on the current workload.

#### Metacognitive subcomponents

<table style="width:100%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 22%" />
<col style="width: 22%" />
<col style="width: 22%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th>subcomponent</th>
<th style="text-align: right;">mean marginal</th>
<th style="text-align: right;">max score without</th>
<th style="text-align: right;">essential for best</th>
<th>verdict</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>tot_recovery</code></td>
<td style="text-align: right;">0.500000</td>
<td style="text-align: right;">2.000000</td>
<td style="text-align: right;">1</td>
<td>required</td>
</tr>
<tr>
<td><code>unknown_caution</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">2.000000</td>
<td style="text-align: right;">1</td>
<td>required</td>
</tr>
<tr>
<td><code>confidence_decay</code></td>
<td style="text-align: right;">0.500000</td>
<td style="text-align: right;">2.000000</td>
<td style="text-align: right;">1</td>
<td>required</td>
</tr>
</tbody>
</table>

**Observations:** The metacognitive family remains internally coherent.
All three subpaths are required to preserve the best score in the
dedicated suite, with `unknown_caution` showing the strongest marginal
contribution.

#### Affect subcomponents

<table style="width:100%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 22%" />
<col style="width: 22%" />
<col style="width: 22%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th>subcomponent</th>
<th style="text-align: right;">mean marginal</th>
<th style="text-align: right;">max score without</th>
<th style="text-align: right;">essential for best</th>
<th>verdict</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>interrupt_path</code></td>
<td style="text-align: right;">0.000000</td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">0</td>
<td>micro-suite redundant</td>
</tr>
<tr>
<td><code>retrieval_path</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">0.000000</td>
<td style="text-align: right;">1</td>
<td>required</td>
</tr>
</tbody>
</table>

**Observations:** The local deterministic probe says `retrieval_path` is
the load-bearing affect subcomponent and `interrupt_path` is redundant
there. But the live-model confirmation above already showed
`affect_interrupt` improving interrupt recall on TopicalChat, so this
again has to be read as a micro-suite redundancy rather than a real
deletion verdict.

#### Neuromodulator subcomponents

<table style="width:100%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 22%" />
<col style="width: 22%" />
<col style="width: 22%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th>subcomponent</th>
<th style="text-align: right;">mean marginal</th>
<th style="text-align: right;">max score without</th>
<th style="text-align: right;">essential for best</th>
<th>verdict</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>write_scale</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">3.000000</td>
<td style="text-align: right;">1</td>
<td>required</td>
</tr>
<tr>
<td><code>competition_scale</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">3.000000</td>
<td style="text-align: right;">1</td>
<td>required</td>
</tr>
<tr>
<td><code>reconsolidation_scale</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">3.000000</td>
<td style="text-align: right;">1</td>
<td>required</td>
</tr>
<tr>
<td><code>value_gain</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">3.000000</td>
<td style="text-align: right;">1</td>
<td>required</td>
</tr>
</tbody>
</table>

**Observations:** Neuromodulation also remains internally coherent. All
four downstream scales are independently load-bearing in the
deterministic audit, so there is no evidence here for pruning any
neuromodulator subpath.

**Updated removal guidance:** after the family sweep, boundary/fact
extension, live-model confirmation, subcomponent completion pass, and
the flashbulb-rate rewrite, the branch still looks substantially
organic. `flashbulb_rate` is no longer behaving like a harmful term; it
now looks like a neutral stabilizer on the current deterministic and
live probes. The strongest “looks redundant locally but not yet safe to
delete globally” candidates are `affect_interrupt`,
`boundary_surprisal`, and possibly `boundary_pressure` pending a
live-model follow-up.

### Full Separable-Unit Matrix (Apr 4, 2026)

To finish the matrix before any surgery, we closed the audit at the
**separable-unit** level rather than stopping at operation families.

The current branch exposes **26 independently removable units** that
matter for this question:

-   **19 non-boundary/fact units**
-   **7 boundary/fact units**

A literal monolithic power set would be `2^26 = 67,108,864`
combinations. We did not materialize that as one giant benchmark because
the deterministic audit graph factors cleanly into a **non-boundary/fact
unit cluster** and a **boundary/fact unit cluster**. We therefore ran
both clusters exhaustively:

-   non-boundary/fact units:
    `logs/flashbulb_rate_rewrite_20260404/full_unit_ablation.log`
    -   benchmark: `examples/benchmark/cortext_full_unit_ablation_bench`
    -   exhaustive unit universe: `2^19 = 524,288`
-   boundary/fact units:
    `logs/full_boundary_fact_unit_ablation_20260404/full_boundary_fact_unit_ablation.log`
    -   benchmark:
        `examples/benchmark/cortext_full_boundary_fact_unit_ablation_bench`
    -   exhaustive unit universe: `2^7 = 128`

This is the completion criterion for the current matrix: every
**separable** unit is now covered by an exhaustive deterministic matrix
in its actual dependency cluster.

#### Non-boundary/fact 19-unit sweep

The 19-unit sweep covered:

-   `source_confidence`
-   `predictive_retrieval`
-   `constructive_recall`
-   `procedural_proactive`
-   `metacog_tot_recovery`
-   `metacog_unknown_caution`
-   `metacog_confidence_decay`
-   `affect_interrupt`
-   `affect_retrieval`
-   `flashbulb_percentile`
-   `flashbulb_rate`
-   `flashbulb_arousal`
-   `neuromod_write_scale`
-   `neuromod_competition_scale`
-   `neuromod_reconsolidation_scale`
-   `neuromod_value_gain`
-   `meta_learning`
-   `reinforcement_edges`
-   `sequential_edges`

Headline result:

<table>
<thead>
<tr>
<th>metric</th>
<th style="text-align: right;">value</th>
</tr>
</thead>
<tbody>
<tr>
<td>unit count</td>
<td style="text-align: right;">19</td>
</tr>
<tr>
<td>combinations</td>
<td style="text-align: right;">524288</td>
</tr>
<tr>
<td>full-mask score</td>
<td style="text-align: right;">15</td>
</tr>
<tr>
<td>best score</td>
<td style="text-align: right;">16</td>
</tr>
<tr>
<td>minimal best bits on</td>
<td style="text-align: right;">16</td>
</tr>
</tbody>
</table>

Minimal best disabled set:

-   `source_confidence`
-   `affect_interrupt`
-   `flashbulb_rate`

Per-unit summary:

<table>
<thead>
<tr>
<th>unit</th>
<th style="text-align: right;">mean marginal</th>
<th style="text-align: right;">max score without</th>
<th style="text-align: right;">essential for best</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>source_confidence</code></td>
<td style="text-align: right;">0.000000</td>
<td style="text-align: right;">16.000000</td>
<td style="text-align: right;">0</td>
</tr>
<tr>
<td><code>predictive_retrieval</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>constructive_recall</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>procedural_proactive</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>metacog_tot_recovery</code></td>
<td style="text-align: right;">0.500000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>metacog_unknown_caution</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>metacog_confidence_decay</code></td>
<td style="text-align: right;">0.500000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>affect_interrupt</code></td>
<td style="text-align: right;">0.000000</td>
<td style="text-align: right;">16.000000</td>
<td style="text-align: right;">0</td>
</tr>
<tr>
<td><code>affect_retrieval</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>flashbulb_percentile</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>flashbulb_rate</code></td>
<td style="text-align: right;">0.000000</td>
<td style="text-align: right;">16.000000</td>
<td style="text-align: right;">0</td>
</tr>
<tr>
<td><code>flashbulb_arousal</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>neuromod_write_scale</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>neuromod_competition_scale</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>neuromod_reconsolidation_scale</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>neuromod_value_gain</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>meta_learning</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>reinforcement_edges</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>sequential_edges</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">15.000000</td>
<td style="text-align: right;">1</td>
</tr>
</tbody>
</table>

**Observations:** This completion sweep tightened the removal story. The
older family-level uncertainty around `meta_learning`,
`reinforcement_edges`, and `sequential_edges` is gone in the
deterministic unit suite: all three are load-bearing once we stop hiding
them inside larger family masks. The flashbulb-rate rewrite also changed
the unit-level readout: `flashbulb_rate` is still non-essential for the
best score, but it is now neutral instead of negative.
`affect_interrupt` and `source_confidence` are also non-essential for
the best score in this clean unit suite, but both have stronger external
evidence elsewhere in the paper: `affect_interrupt` failed live deletion
confirmation, and `source_confidence` still blocks poisoned
low-confidence memories in the contradiction-heavy Ubuntu probe.

#### Boundary/fact 7-unit sweep

The final missing cluster adds `boundary_pressure` to the earlier
boundary/fact extension, yielding the full 7-unit boundary/fact unit
universe:

-   `boundary_pressure`
-   `boundary_surprisal`
-   `boundary_natural`
-   `fact_layer`
-   `fact_history`
-   `fact_stale_penalty`
-   `fact_provenance`

Headline result:

<table>
<thead>
<tr>
<th>metric</th>
<th style="text-align: right;">value</th>
</tr>
</thead>
<tbody>
<tr>
<td>unit count</td>
<td style="text-align: right;">7</td>
</tr>
<tr>
<td>combinations</td>
<td style="text-align: right;">128</td>
</tr>
<tr>
<td>full-mask score</td>
<td style="text-align: right;">7</td>
</tr>
<tr>
<td>best score</td>
<td style="text-align: right;">7</td>
</tr>
<tr>
<td>full-mask is best</td>
<td style="text-align: right;">1</td>
</tr>
</tbody>
</table>

Minimal best disabled set:

-   `boundary_pressure`
-   `boundary_surprisal`

Per-unit summary:

<table>
<thead>
<tr>
<th>unit</th>
<th style="text-align: right;">mean marginal</th>
<th style="text-align: right;">max score without</th>
<th style="text-align: right;">essential for best</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>boundary_pressure</code></td>
<td style="text-align: right;">0.000000</td>
<td style="text-align: right;">7.000000</td>
<td style="text-align: right;">0</td>
</tr>
<tr>
<td><code>boundary_surprisal</code></td>
<td style="text-align: right;">0.000000</td>
<td style="text-align: right;">7.000000</td>
<td style="text-align: right;">0</td>
</tr>
<tr>
<td><code>boundary_natural</code></td>
<td style="text-align: right;">3.000000</td>
<td style="text-align: right;">4.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>fact_layer</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">6.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>fact_history</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">6.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>fact_stale_penalty</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">6.000000</td>
<td style="text-align: right;">1</td>
</tr>
<tr>
<td><code>fact_provenance</code></td>
<td style="text-align: right;">1.000000</td>
<td style="text-align: right;">6.000000</td>
<td style="text-align: right;">1</td>
</tr>
</tbody>
</table>

**Observations:** The fact sublayer remains fully load-bearing at the
unit level. The only local redundancies are `boundary_pressure` and
`boundary_surprisal`, with `boundary_natural` doing the real work in
this deterministic boundary suite. As above, this is not enough by
itself to justify deletion: `boundary_surprisal` already failed
live-model deletion confirmation, so only `boundary_pressure` remains a
plausible boundary-side cut candidate.

**Final matrix reading:** the branch looks more organic than decorative
even at the completed separable-unit level. After the flashbulb-rate
rewrite, `flashbulb_rate` no longer looks harmful; it is simply
low-evidence on the current probes. The next plausible but still
unconfirmed target is `boundary_pressure`. Everything else either
remained load-bearing in the unit matrix or already survived a stronger
live-model confirmation pass.

## Threshold Adaptation

The dynamic threshold (θ_dynamic) successfully tracked score
distributions. In high-volatility inputs (within-accumulator drift_acc
\> 1.0), thresholds relaxed to ~0.15, while stable contexts tightened to
~0.27.

## Boundary Detection

Accumulator drift (drift_acc) aligned with semantic shifts. Conversation
turns with distinct topics triggered flushes (boundary_score \> 0.3)
while coherent continuations remained accumulated.

**Boundary calibration smoke (20 turns; F=S=T=0.5, consolidate=0):**

Run: `logs/topical_chat_snapshots/20260101_090407`

-   boundary_score_mean: **0.321** (p90 **0.917**)
-   boundary_score_pass_rate: **0.30**
-   boundary_at_rate: **0.30**

**Observation:** Increasing coherence/topic weighting further boosts
natural boundary crossings without increasing the boundary rate in the
short horizon. The long‑horizon run below will confirm whether
capacity/time are now edge cases rather than primary segmenters.

**Boundary tuning (long horizon; 156 turns, consolidate=0):**

Run: `logs/topical_chat_snapshots/20260101_090516_boundary`

-   boundary_at_rate: **0.192**
-   boundary_score_pass_rate: **0.122**
-   boundary_score_mean: **0.177**

Chunk‑level episode audit (first conversation, 2‑word cadence):

-   drift episodes: **19** (avg **4.42** chunks, p50 **3**, p90 **7**,
    range **3–8**)
-   capacity episodes: **10** (avg **6.00** chunks, p50 **5.5**, p90
    **8**, range **5–8**) **Observation:** Natural boundaries now
    dominate (19 drift vs 10 capacity). With time‑based flush removed,
    boundaries are driven by natural indicators plus capacity safety.

**Interrupt accept/ignore check (long horizon; F=S=T=0.5,
consolidate=0):**

Run: `logs/topical_chat_snapshots/20260101_113144` (156 turns;
dataset‑limited)

-   interrupt_abort_rate: **0.314** (16 / 51 interrupts)
-   interrupt_turn_rate: **0.327**
-   boundary_at_rate: **0.173**

**Observation:** The similarity‑based accept/ignore rule reduces
committed aborts substantially versus the prior novelty‑only heuristic,
while keeping interrupt frequency in the same band. This suggests many
interrupts are ignored (continued thought aligns more with μ_acc) rather
than treated as accepted shifts.

## Latency and Performance

End-to-end processing per turn is dominated by embedding + retrieval
rather than graph expansion. In the Dec 31, 2025 long-horizon sweep (156
turns, consolidation cycles=2), `perf_total_ms_mean` ranged
**0.606–5.05s** depending on knob settings; the default **F=S=T=0.5**
run averaged **0.97s/turn**, while **F=S=T=1.0** averaged
**0.61s/turn**. Graph expansion remains a small fraction of total time
after in-memory summary seeding.

## Long-Horizon Consolidation Ablation (720 Turns)

We ran a long-horizon stress test with **F=S=T=0.5**, `max_total=720`,
`max_turns=360`, and `max_conversations=10` (single-stream,
interleave=1) to isolate consolidation effects. Two conditions were
compared:

-   **No consolidation** (`consolidate_cycles=0`)
-   **Consolidation on** (`consolidate_cycles=2`, yielding 8
    consolidation runs; these historical runs used gemma-3n-e2b)

Key outcomes (Δ = consolidate − no-consolidate):

-   **Retrieval semantic overlap mean:** +0.0154 (0.5647 → 0.5801)
-   **Retrieval overlap mean:** +0.0011 (0.2264 → 0.2275)
-   **Interrupt recall:** +0.0124 (0.9543 → 0.9668)
-   **Interrupt precision:** −0.0026 (0.9947 → 0.9921)
-   **Retrieval turn rate:** −0.0056 (0.5528 → 0.5472)

Consolidation contribution metrics confirm summaries are being used (but
sparsely in this horizon):

-   **retrieval_summary_hit_rate:** 0.0051 (0.51% of retrieval turns
    include summaries)
-   **summary_hit_overlap_mean:** 0.606 (semantic overlap of best
    summary hit)
-   **consolidation_summary_count:** 8 (all from gemma-3n-e2b in this
    run)

Additional signals:

-   **memory_strength_mean:** 0.6800 → 0.6141 (Δ −0.066), suggesting
    consolidation redistributes strength across long-term entries rather
    than strictly amplifying it.
-   **retrieval_avg_candidates:** 55.17 → 55.67 (Δ +0.50), essentially
    unchanged.

Overall, consolidation increases semantic alignment and interrupt recall
without materially changing retrieval cadence, but summary usage remains
low at this horizon because only eight summaries were produced.

## Extraction Pipeline Validation (CPU-Only, 720 Turns)

We observed that LiteRT constrained decoding for extraction can fail on
GPU due to non‑host tensor buffers, so we forced **CPU extraction** and
re‑ran the **720‑turn** long‑horizon test (**F=S=T=0.5**,
`max_total=720`, `max_turns=720`, `max_conversations=999`,
`consolidate_cycles=4`). We compare against the same settings with
consolidation off, and against an earlier run where extraction silently
failed on GPU.

<table style="width:100%;">
<colgroup>
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">condition</th>
<th style="text-align: right;">consolidation_extraction_results</th>
<th style="text-align: right;">labels_seen</th>
<th style="text-align: right;">relations_seen</th>
<th style="text-align: right;">retrieval_summary_hit_rate</th>
<th style="text-align: right;">summary_hit_overlap_mean</th>
<th style="text-align: right;">retrieval_semantic_overlap_mean</th>
<th style="text-align: right;">interrupt_precision</th>
<th style="text-align: right;">interrupt_recall</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">no consolidation</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.000</td>
<td style="text-align: right;">0.56465</td>
<td style="text-align: right;">0.99471</td>
<td style="text-align: right;">0.95432</td>
</tr>
<tr>
<td style="text-align: right;">consolidation on (GPU extraction
failed)</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0.0176</td>
<td style="text-align: right;">0.645</td>
<td style="text-align: right;">0.56480</td>
<td style="text-align: right;">0.99208</td>
<td style="text-align: right;">0.95674</td>
</tr>
<tr>
<td style="text-align: right;">consolidation on (CPU extraction)</td>
<td style="text-align: right;">13</td>
<td style="text-align: right;">89</td>
<td style="text-align: right;">163</td>
<td style="text-align: right;">0.0427</td>
<td style="text-align: right;">0.585</td>
<td style="text-align: right;">0.56428</td>
<td style="text-align: right;">0.99474</td>
<td style="text-align: right;">0.95939</td>
</tr>
</tbody>
</table>

**Observations:** CPU extraction restores label/relation persistence (89
labels, 163 relations) and increases summary presence in retrieval
(~4.3% of retrieval turns) while keeping overall semantic alignment and
interrupt quality stable. The remaining gap to higher summary usage
appears driven by summary count and horizon length rather than
extraction errors.

## Association Boost Check (360 Turns)

We added a small **association boost** in retrieval ranking (see
<a href="#sec-advanced" class="quarto-xref">Section 8</a>) and re‑ran
the 360‑turn baseline with consolidation enabled (**F=S=T=0.5**,
`max_total=360`, `max_turns=360`, `max_conversations=6`). We compare
**before vs after** on identical settings:

<table>
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">variant</th>
<th style="text-align: right;">retrieval_summary_hit_rate</th>
<th style="text-align: right;">retrieval_association_turn_rate</th>
<th style="text-align: right;">summary_hit_overlap_mean</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">before (no boost)</td>
<td style="text-align: right;">0.0103</td>
<td style="text-align: right;">0.0103</td>
<td style="text-align: right;">0.664</td>
</tr>
<tr>
<td style="text-align: right;">after (boost)</td>
<td style="text-align: right;">0.0206</td>
<td style="text-align: right;">0.0206</td>
<td style="text-align: right;">0.644</td>
</tr>
</tbody>
</table>

**Observations:** The boost roughly **doubles** summary presence in
retrieval without affecting candidate volume (≈50 per turn). Summary
overlap remains high, indicating the boost favors consolidated
association nodes when they are relevant without degrading semantic
quality. The absolute hit rate remains low in this short horizon because
only four summaries were produced; longer horizons should show a larger
effect.

**Current-branch rerun (Mar 28, 2026; nondeterministic; stronger boost
probe):**

Runs:

-   current branch baseline:
    `logs/topical_chat_snapshots/assocboost_base_20260328`
-   stronger association boost (`0.015..0.06` → `0.025..0.09`):
    `logs/topical_chat_snapshots/assocboost_boosted_20260328`

<table style="width:100%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">variant</th>
<th style="text-align: right;">retrieval_summary_hit_rate</th>
<th style="text-align: right;">retrieval_association_turn_rate</th>
<th style="text-align: right;">summary_hit_overlap_mean</th>
<th style="text-align: right;">retrieval_semantic_overlap_mean</th>
<th style="text-align: right;">perf_total_ms_mean</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">current branch baseline</td>
<td style="text-align: right;">0.5648</td>
<td style="text-align: right;">0.5648</td>
<td style="text-align: right;">0.450</td>
<td style="text-align: right;">0.56297</td>
<td style="text-align: right;">946.12</td>
</tr>
<tr>
<td style="text-align: right;">stronger boost</td>
<td style="text-align: right;">0.5619</td>
<td style="text-align: right;">0.5619</td>
<td style="text-align: right;">0.458</td>
<td style="text-align: right;">0.56304</td>
<td style="text-align: right;">928.58</td>
</tr>
</tbody>
</table>

**Observations:** On the current branch, summary hits are already very
common (about **56%** of retrieval turns), far above the December
baseline that motivated the original boost. Pushing the association
bonus higher did **not** increase summary-hit rate; it stayed
flat-to-slightly-lower (`0.5648` → `0.5619`). The stronger boost
produced only tiny secondary differences: slightly higher summary-hit
overlap (`0.450` → `0.458`), negligible semantic-overlap change
(`0.56297` → `0.56304`), and about **1.9%** lower mean total time. This
does not justify further increasing the association bonus on the current
branch.

## Long-Horizon Knob Sweep (360 Max Turns)

We ran a long-horizon sweep using `scripts/run_topical_chat_sweep.sh`
with `max_turns=360`, `max_total=720`, and `consolidate_cycles=2` (no
periodic consolidation; consolidation runs once at the end). Each run
observed **156 turns** (single conversation), yielding **2 consolidation
runs** per config. The table reports retrieval/interrupt rates, label
candidate rate, and per‑turn performance.

Run metadata: **Dec 31, 2025**
(`logs/topical_chat_runs/long_horizon_20251231_112600`, git rev
`8ae72ef`). This run includes the **in‑memory summary cache** for
retrieval seeding.

<table>
<colgroup>
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">F</th>
<th style="text-align: right;">S</th>
<th style="text-align: right;">T</th>
<th style="text-align: right;">turns</th>
<th style="text-align: right;">cons_runs</th>
<th style="text-align: right;">retr_turn_rate</th>
<th style="text-align: right;">retr_avg_cands</th>
<th style="text-align: right;">label_rate</th>
<th style="text-align: right;">retr_sem_overlap</th>
<th style="text-align: right;">interrupt_rate</th>
<th style="text-align: right;">perf_ms</th>
<th style="text-align: right;">signals/sec</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">156</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">0.494</td>
<td style="text-align: right;">94.026</td>
<td style="text-align: right;">0.961</td>
<td style="text-align: right;">0.565</td>
<td style="text-align: right;">0.487</td>
<td style="text-align: right;">5050</td>
<td style="text-align: right;">0.133</td>
</tr>
<tr>
<td style="text-align: right;">0.25</td>
<td style="text-align: right;">0.25</td>
<td style="text-align: right;">0.25</td>
<td style="text-align: right;">156</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">0.494</td>
<td style="text-align: right;">80.844</td>
<td style="text-align: right;">0.740</td>
<td style="text-align: right;">0.595</td>
<td style="text-align: right;">0.474</td>
<td style="text-align: right;">2101</td>
<td style="text-align: right;">0.355</td>
</tr>
<tr>
<td style="text-align: right;">0.5</td>
<td style="text-align: right;">0.5</td>
<td style="text-align: right;">0.5</td>
<td style="text-align: right;">156</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">0.571</td>
<td style="text-align: right;">54.472</td>
<td style="text-align: right;">0.772</td>
<td style="text-align: right;">0.564</td>
<td style="text-align: right;">0.532</td>
<td style="text-align: right;">969</td>
<td style="text-align: right;">0.593</td>
</tr>
<tr>
<td style="text-align: right;">0.75</td>
<td style="text-align: right;">0.75</td>
<td style="text-align: right;">0.75</td>
<td style="text-align: right;">156</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">0.654</td>
<td style="text-align: right;">31.500</td>
<td style="text-align: right;">0.813</td>
<td style="text-align: right;">0.575</td>
<td style="text-align: right;">0.596</td>
<td style="text-align: right;">697</td>
<td style="text-align: right;">0.706</td>
</tr>
<tr>
<td style="text-align: right;">1</td>
<td style="text-align: right;">1</td>
<td style="text-align: right;">1</td>
<td style="text-align: right;">156</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">0.808</td>
<td style="text-align: right;">6.524</td>
<td style="text-align: right;">0.672</td>
<td style="text-align: right;">0.585</td>
<td style="text-align: right;">0.782</td>
<td style="text-align: right;">606</td>
<td style="text-align: right;">0.736</td>
</tr>
<tr>
<td style="text-align: right;">0.15</td>
<td style="text-align: right;">0.9</td>
<td style="text-align: right;">0.5</td>
<td style="text-align: right;">156</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">0.506</td>
<td style="text-align: right;">86.886</td>
<td style="text-align: right;">0.929</td>
<td style="text-align: right;">0.598</td>
<td style="text-align: right;">0.500</td>
<td style="text-align: right;">2921</td>
<td style="text-align: right;">0.274</td>
</tr>
</tbody>
</table>

**Observations:**

-   Higher **Focus/Sensitivity** compress candidate volume (**6.5** at
    F=1 vs **86.9** at F=0.15,S=0.9), improving throughput (0.61s/turn
    at F=1 vs 2.92s/turn at F=0.15,S=0.9).
-   Retrieval semantic overlap stays stable (~0.56–0.60), indicating
    selectivity gains do not degrade semantic alignment.
-   **Label candidate rate is high** (0.67–0.96) due to the label bank
    seeding, while association candidates remain absent in this sweep
    because consolidation only ran at the end (2 summaries total).

## Long-Horizon Harness Sweep (Jan 1, 2026)

We ran the multi‑participant harness (`scripts/run_memory_harness.py`)
with `max_turns=360`, `max_total=720`, and `consolidate_cycles=0` to
stress retrieval and interrupt behavior under long‑horizon load. The run
produced a baseline (**F=S=T=0.5**) plus a 3×3 sweep around mid‑range
knobs; individual runs are dataset‑limited at **120–156 turns**. Logs
and per‑run outputs: `logs/memory_harness/20260101_122657`.

We summarize the baseline and the best settings observed for **retrieval
rate** and **interrupt recall**:

<table>
<colgroup>
<col style="width: 8%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
</colgroup>
<thead>
<tr>
<th>config</th>
<th style="text-align: right;">turns</th>
<th style="text-align: right;">writes</th>
<th style="text-align: right;">retr_turn_rate</th>
<th style="text-align: right;">retr_avg_cands</th>
<th style="text-align: right;">interrupt_rate</th>
<th style="text-align: right;">precision</th>
<th style="text-align: right;">recall</th>
<th style="text-align: right;">mem_retrieved_mean</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>F=0.5, S=0.5, T=0.5</strong> (baseline)</td>
<td style="text-align: right;">156</td>
<td style="text-align: right;">29</td>
<td style="text-align: right;">0.3526</td>
<td style="text-align: right;">5.75</td>
<td style="text-align: right;">0.2051</td>
<td style="text-align: right;">0.9688</td>
<td style="text-align: right;">0.5741</td>
<td style="text-align: right;">10.90</td>
</tr>
<tr>
<td><strong>F=0.5, S=0.6, T=0.4</strong> (best retrieval rate)</td>
<td style="text-align: right;">120</td>
<td style="text-align: right;">23</td>
<td style="text-align: right;">0.4000</td>
<td style="text-align: right;">2.46</td>
<td style="text-align: right;">0.2417</td>
<td style="text-align: right;">0.9310</td>
<td style="text-align: right;">0.6000</td>
<td style="text-align: right;">5.13</td>
</tr>
<tr>
<td><strong>F=0.4, S=0.6, T=0.5</strong> (best interrupt recall)</td>
<td style="text-align: right;">120</td>
<td style="text-align: right;">23</td>
<td style="text-align: right;">0.3250</td>
<td style="text-align: right;">4.31</td>
<td style="text-align: right;">0.2583</td>
<td style="text-align: right;">1.0000</td>
<td style="text-align: right;">0.7949</td>
<td style="text-align: right;">7.30</td>
</tr>
</tbody>
</table>

**Observations:**

-   **Sensitivity≈0.6** consistently increases retrieval and interrupt
    activity relative to the baseline while keeping precision high.
-   Lower **Stability (T≈0.4)** favors higher retrieval rates (more
    surfacing), while mid‑range **T=0.5** improves interrupt recall
    without precision loss.
-   The baseline remains a strong **all‑rounder**, but targeted profiles
    outperform it on retrieval or interrupt recall depending on the
    objective.

## Flashbulb + Synaptic Tagging Check (EmpatheticDialogues)

We ran a targeted check on EmpatheticDialogues
(`data/empathetic_dialogues/valid.jsonl`) using the harness
(`scripts/run_memory_harness.py`). We compare baseline **F=S=T=0.5**
against lower/high Sensitivity to test flashbulb and synaptic‑tagging
behavior. Run logs and DBs: `logs/memory_harness/20260101_153306`
(pre‑tune), `logs/flashbulb_algo_tuned3_sweep_20260102_105932`
(post‑tune), and `logs/flashbulb_algo_tuned4_sweep_20260102_111607`
(rate‑stabilized).

<table>
<thead>
<tr>
<th>config</th>
<th style="text-align: right;">spike_bypass_true</th>
<th style="text-align: right;">flashbulb_count</th>
<th style="text-align: right;">tagged_memories</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>F=0.5, S=0.5, T=0.5</strong></td>
<td style="text-align: right;">4</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">90 / 109 (82.6%)</td>
</tr>
<tr>
<td><strong>F=0.5, S=0.4, T=0.5</strong></td>
<td style="text-align: right;">4</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">84 / 102 (82.4%)</td>
</tr>
<tr>
<td><strong>F=0.5, S=0.6, T=0.5</strong></td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">112 / 123 (91.1%)</td>
</tr>
</tbody>
</table>

Post‑tune flashbulb sweep (same dataset, `max_turns=160`,
`max_total=160`):

<table>
<thead>
<tr>
<th>config</th>
<th style="text-align: right;">flashbulb_count</th>
<th style="text-align: right;">memories</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>F=0.5, S=0.4, T=0.5</strong></td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">14</td>
</tr>
<tr>
<td><strong>F=0.5, S=0.5, T=0.5</strong></td>
<td style="text-align: right;">1</td>
<td style="text-align: right;">14</td>
</tr>
<tr>
<td><strong>F=0.5, S=0.6, T=0.5</strong></td>
<td style="text-align: right;">5</td>
<td style="text-align: right;">18</td>
</tr>
</tbody>
</table>

Aggressive tuning sweep (lowered θ_intensity/thresholds, stronger
gain/percentile blending):

<table>
<thead>
<tr>
<th>config</th>
<th style="text-align: right;">flashbulb_count</th>
<th style="text-align: right;">memories</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>F=0.5, S=0.4, T=0.5</strong></td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">14</td>
</tr>
<tr>
<td><strong>F=0.5, S=0.5, T=0.5</strong></td>
<td style="text-align: right;">4</td>
<td style="text-align: right;">14</td>
</tr>
<tr>
<td><strong>F=0.5, S=0.6, T=0.5</strong></td>
<td style="text-align: right;">6</td>
<td style="text-align: right;">18</td>
</tr>
</tbody>
</table>

**Observations:**

-   **Pre‑tune flashbulb markers did not trigger** (0 across cases).
-   **Post‑tune flashbulb markers trigger** and scale with Sensitivity
    (0 → 1 → 5 across S=0.4/0.5/0.6), bringing flashbulb rates into a
    usable range.
-   **Rate stabilizer kept counts stable** across the same sweep (no
    regression), confirming that the smoothing does not suppress
    flashbulb activation.
-   **Aggressive tuning shifted flashbulb rate** into a usable band at
    S=0.5 (≈29% of memories in this short run), and reduced the gap to
    S=0.6. This suggests the knobs can drive flashbulb frequency as
    intended, but we should verify with longer horizons to avoid
    over-triggering.

Post-rewrite long-horizon ablation rerun (EmpatheticDialogues, Apr 4,
2026; `EmbeddingGemma/llama.cpp`, 720/1440, F=0.5, S=0.8, T=0.2). Rerun
roots:

-   baseline: `logs/flashbulb_rate_rewrite_20260404/live_baseline`
-   `no_percentile`:
    `logs/flashbulb_rate_rewrite_20260404/live_no_percentile`
-   `no_rate`: `logs/flashbulb_rate_rewrite_20260404/live_no_rate`
-   `no_arousal`: `logs/flashbulb_rate_rewrite_20260404/live_no_arousal`

<table>
<thead>
<tr>
<th>ablation</th>
<th style="text-align: right;">memories</th>
<th style="text-align: right;">flashbulb</th>
<th style="text-align: right;">rate</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>baseline (rewritten rate controller)</strong></td>
<td style="text-align: right;">19</td>
<td style="text-align: right;">1</td>
<td style="text-align: right;">5.26%</td>
</tr>
<tr>
<td><strong>no_percentile</strong></td>
<td style="text-align: right;">18</td>
<td style="text-align: right;">8</td>
<td style="text-align: right;">44.44%</td>
</tr>
<tr>
<td><strong>no_arousal</strong></td>
<td style="text-align: right;">18</td>
<td style="text-align: right;">4</td>
<td style="text-align: right;">22.22%</td>
</tr>
<tr>
<td><strong>no_rate</strong></td>
<td style="text-align: right;">19</td>
<td style="text-align: right;">1</td>
<td style="text-align: right;">5.26%</td>
</tr>
</tbody>
</table>

**Ablation notes:**

-   Percentile gating remains the dominant suppressor under the real
    encoder; removing it raises flashbulb rate from **5.26%** to
    **44.44%**.
-   Arousal gating is still materially causal after the rewrite;
    removing it raises flashbulb rate to **22.22%**.
-   The rewritten rate controller is now behaviorally neutral on this
    live workload: baseline and `no_rate` are identical at **1/19**
    flashbulbs.
-   Treat the earlier pre-rewrite `EmbeddingGemma` flashbulb table as
    superseded. The old rate term was partly fighting the exception
    gate; after the rewrite, `rate` no longer suppresses
    percentile-qualified events, but it also does not add measurable
    lift on this corpus slice.

## Boundary Ablation (TopicalChat)

We reran the boundary ablation on TopicalChat
(`data/topical_chat/valid_freq.jsonl`) on **Apr 4, 2026** with
**F=S=T=0.5**, `max_turns=360`, `max_total=720`, using
`EmbeddingGemma/llama.cpp` (confirmed in `run.log` as
`text_encoder_backend=EmbeddingGemma/llama.cpp`). We measure the share
of memories with `boundary_score ≥ 0.5` as a proxy for boundary hits.
Rerun root: `logs/embeddinggemma_paper_boundary_20260404_seq`.

<table>
<thead>
<tr>
<th>ablation</th>
<th style="text-align: right;">memories</th>
<th style="text-align: right;">boundary_hits (score ≥ 0.5)</th>
<th style="text-align: right;">rate</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>baseline (EmbeddingGemma)</strong></td>
<td style="text-align: right;">45</td>
<td style="text-align: right;">28</td>
<td style="text-align: right;">62.22%</td>
</tr>
<tr>
<td><strong>no_time_capacity</strong></td>
<td style="text-align: right;">46</td>
<td style="text-align: right;">31</td>
<td style="text-align: right;">67.39%</td>
</tr>
<tr>
<td><strong>time_capacity_only</strong></td>
<td style="text-align: right;">21</td>
<td style="text-align: right;">9</td>
<td style="text-align: right;">42.86%</td>
</tr>
<tr>
<td><strong>no_surprisal</strong></td>
<td style="text-align: right;">53</td>
<td style="text-align: right;">43</td>
<td style="text-align: right;">81.13%</td>
</tr>
</tbody>
</table>

**Boundary notes:**

-   The older ImageBind-era `time_capacity_only = 0 / 0` result was not
    stable. Under EmbeddingGemma, time/capacity-only still produces
    nonzero flushes (**9 / 21**, **42.86%**).
-   Removing time/capacity does not reduce the boundary-hit share in
    this rerun; it rises slightly (**62.22%** → **67.39%**), so the
    earlier “changes little” conclusion still broadly holds, but with a
    higher baseline rate.
-   `no_surprisal` is no longer a minimal-delta ablation. Under the real
    encoder, the boundary-hit share rises materially to **81.13%**, so
    the older `37 / 22` claim should be treated as superseded.

## Boundary Inactivity Check

We replaced explicit inactivity flushes with a **soft inactivity boost**
to the boundary score (gap‑driven, support‑gated). Time gaps now raise
boundary probability without forcing a flush. A short cadence‑gap check
(`logs/topical_chat_snapshots/20260102_162117`, 5‑second gaps) did not
produce premature boundaries, indicating the boost remains a gentle
nudge. We repeated with longer gaps
(`logs/topical_chat_snapshots/20260102_163602`, 15‑second gaps, 20
turns) and observed **boundary_at_rate=0.25** and
**boundary_score_pass_rate=0.25**, still within the natural boundary
range. At 30‑second gaps (`logs/topical_chat_snapshots/20260102_163723`)
the rates remained **0.25**, indicating inactivity alone does not force
segmentation. With 5‑minute gaps
(`logs/topical_chat_snapshots/20260102_165112`, deterministic time
advance), **boundary_at_rate=0.30** and
**boundary_score_pass_rate=0.20**, still within the natural boundary
range. At 15‑minute gaps (`logs/topical_chat_snapshots/20260102_165927`)
the rates were unchanged (**0.30** / **0.20**), reinforcing that
inactivity boosts are soft and do not force segmentation on their own.
At 1‑hour gaps (`logs/topical_chat_snapshots/20260102_170229`) the rates
remained **0.30** / **0.20**. After switching to an exponential
inactivity boost, a 1‑hour gap run
(`logs/topical_chat_snapshots/20260102_170626`) still reported **0.30**
/ **0.20** at 20 turns, suggesting the boost needs a stronger scale or a
dedicated gap‑dominant term to force boundaries at extreme inactivity.
We then removed the cap and greatly increased the exponent; a follow‑up
1‑hour run (`logs/topical_chat_snapshots/20260102_170926`) still
reported **0.30** / **0.20**, indicating the suppression likely comes
from the support gate rather than scale alone. After switching to a
**dynamic cadence ratio** (gap measured vs `dt_ema`), a 1‑hour run
(`logs/topical_chat_snapshots/20260102_173006`) produced
**boundary_at_rate=0.40**, **boundary_score_pass_rate=0.00**, and
**boundary_score_mean=0.128**. This shows long inactivity now closes
episodes without requiring the main boundary score to cross its natural
threshold, preserving the “no fixed timeout” constraint while still
preventing unbounded accumulation. \* **Synaptic tagging is active** and
scales with Sensitivity (≈82% → 91% tagged), indicating frequent
surprisal/arousal triggers in this dataset. \* Spike‑bypass events are
rare (2–4 per run), consistent with flashbulb‑like flushes being
exceptional rather than common.

## Short-Horizon Smoke Check (120 Turns)

We ran a short smoke check at **F=S=T=0.5** with `max_turns=120`,
`max_total=120`, and `consolidate_cycles=2` to validate end‑to‑end
integration after the ASSOCIATION/LABEL retention fix. This produced **2
summaries** and **2 labels**, but summary/label candidates did not
appear in retrieval within this short horizon:

<table>
<colgroup>
<col style="width: 12%" />
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
<th style="text-align: right;">turns</th>
<th style="text-align: right;">summaries</th>
<th style="text-align: right;">labels</th>
<th style="text-align: right;">retr_turn_rate</th>
<th style="text-align: right;">retr_avg_cands</th>
<th style="text-align: right;">assoc_rate</th>
<th style="text-align: right;">label_rate</th>
<th style="text-align: right;">interrupt_rate</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">120</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">0.4917</td>
<td style="text-align: right;">13.93</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.4333</td>
</tr>
</tbody>
</table>

**Observation:** At 120 turns, summary/label production is too sparse to
materially affect retrieval. Longer horizons (see above) show non‑zero
association/label candidate rates once more summaries accumulate.

## Short-Horizon Consolidation Mode Check (20 Turns)

We ran a **20‑turn** sanity check at **F=S=T=0.5** to compare
**baseline**, **shallow**, and **deep** consolidation modes using the
topical‑chat pipeline (ImageBind embeddings; gemma‑3n‑e2b via LiteRT for
deep summarization). Consolidation was triggered twice
(`consolidate_every=20`, `consolidate_cycles=1`).

Runs (Dec 31, 2025):

-   baseline:
    `logs/topical_chat_snapshots/20251231_000002_baseline_small`
-   shallow: `logs/topical_chat_snapshots/20251231_000003_shallow_small`
-   deep: `logs/topical_chat_snapshots/20251231_000001_deep_small`

<table style="width:100%;">
<colgroup>
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">mode</th>
<th style="text-align: right;">cons_runs</th>
<th style="text-align: right;">assoc_created</th>
<th style="text-align: right;">labels_created</th>
<th style="text-align: right;">summaries</th>
<th style="text-align: right;">retr_turn_rate</th>
<th style="text-align: right;">retr_avg_cands</th>
<th style="text-align: right;">assoc_rate</th>
<th style="text-align: right;">label_rate</th>
<th style="text-align: right;">interrupt_rate</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">baseline</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0.300</td>
<td style="text-align: right;">2.67</td>
<td style="text-align: right;">0.000</td>
<td style="text-align: right;">0.000</td>
<td style="text-align: right;">0.200</td>
</tr>
<tr>
<td style="text-align: right;">shallow</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0.300</td>
<td style="text-align: right;">2.67</td>
<td style="text-align: right;">0.000</td>
<td style="text-align: right;">0.000</td>
<td style="text-align: right;">0.200</td>
</tr>
<tr>
<td style="text-align: right;">deep</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">0.300</td>
<td style="text-align: right;">2.67</td>
<td style="text-align: right;">0.000</td>
<td style="text-align: right;">0.000</td>
<td style="text-align: right;">0.200</td>
</tr>
</tbody>
</table>

**Observations:** Both consolidation modes create association nodes (and
deep additionally emits labels + summaries), but **20 turns is too
short** for these nodes to enter retrieval. Association/label candidate
rates remain **0**; longer horizons are required for retrieval to
surface consolidated memory.

## Verification Checks (Dec 31, 2025)

We ran targeted verification checks at **F=S=T=0.5** to confirm
end‑to‑end behavior after recent changes (summary cache, source
monitoring, sequential edges).

**Baseline vs idle consolidation (120 turns; consolidate_every=40,
deep):**

<table>
<colgroup>
<col style="width: 12%" />
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
<th style="text-align: right;">condition</th>
<th style="text-align: right;">cons_runs</th>
<th style="text-align: right;">summaries</th>
<th style="text-align: right;">labels_created</th>
<th style="text-align: right;">retr_sem_overlap</th>
<th style="text-align: right;">retr_label_rate</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">perf_total_ms</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">baseline</td>
<td style="text-align: right;">8</td>
<td style="text-align: right;">8</td>
<td style="text-align: right;">6479</td>
<td style="text-align: right;">0.5703</td>
<td style="text-align: right;">0.790</td>
<td style="text-align: right;">0.542</td>
<td style="text-align: right;">915</td>
</tr>
<tr>
<td style="text-align: right;">idle‑during (consolidate_during=1,
consolidate_idle=1)</td>
<td style="text-align: right;">8</td>
<td style="text-align: right;">8</td>
<td style="text-align: right;">6479</td>
<td style="text-align: right;">0.5704</td>
<td style="text-align: right;">0.790</td>
<td style="text-align: right;">0.542</td>
<td style="text-align: right;">908</td>
</tr>
</tbody>
</table>

**Observation:** Outputs are effectively identical, indicating the
idle/defer flags do not change behavior when `consolidate_every` already
governs consolidation timing.

**Idle vs end‑only consolidation (120 turns; consolidate_cycles=1, no
periodic consolidation):**

<table>
<colgroup>
<col style="width: 12%" />
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
<th style="text-align: right;">condition</th>
<th style="text-align: right;">cons_runs</th>
<th style="text-align: right;">summaries</th>
<th style="text-align: right;">labels_created</th>
<th style="text-align: right;">retr_sem_overlap</th>
<th style="text-align: right;">retr_label_rate</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">perf_total_ms</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">end‑only</td>
<td style="text-align: right;">1</td>
<td style="text-align: right;">1</td>
<td style="text-align: right;">6478</td>
<td style="text-align: right;">0.5602</td>
<td style="text-align: right;">0.830</td>
<td style="text-align: right;">0.525</td>
<td style="text-align: right;">917</td>
</tr>
<tr>
<td style="text-align: right;">idle‑during</td>
<td style="text-align: right;">1</td>
<td style="text-align: right;">1</td>
<td style="text-align: right;">6478</td>
<td style="text-align: right;">0.5602</td>
<td style="text-align: right;">0.835</td>
<td style="text-align: right;">0.525</td>
<td style="text-align: right;">916</td>
</tr>
</tbody>
</table>

**Observation:** With a single consolidation cycle, idle‑during matches
end‑only within noise; any differences in summary/label retrieval would
require longer horizons or more consolidation cycles.

**Current-branch rerun (Mar 28, 2026; 120 turns; nondeterministic;
consolidate_every=60):**

Runs:

-   end-only: `logs/topical_chat_snapshots/idle_endonly_nondet_20260328`
-   idle-during: `logs/topical_chat_snapshots/idle_idle_nondet_20260328`

<table>
<colgroup>
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 9%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">condition</th>
<th style="text-align: right;">cons_runs</th>
<th style="text-align: right;">summaries</th>
<th style="text-align: right;">labels_seen</th>
<th style="text-align: right;">relations_seen</th>
<th style="text-align: right;">retr_turn_rate</th>
<th style="text-align: right;">retr_avg_cands</th>
<th style="text-align: right;">retr_sem_overlap</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">interrupt_abort_rate</th>
<th style="text-align: right;">perf_total_ms</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">end-only</td>
<td style="text-align: right;">3</td>
<td style="text-align: right;">3</td>
<td style="text-align: right;">17</td>
<td style="text-align: right;">22</td>
<td style="text-align: right;">0.475</td>
<td style="text-align: right;">53.68</td>
<td style="text-align: right;">0.55472</td>
<td style="text-align: right;">0.4667</td>
<td style="text-align: right;">0.4107</td>
<td style="text-align: right;">842.76</td>
</tr>
<tr>
<td style="text-align: right;">idle-during</td>
<td style="text-align: right;">3</td>
<td style="text-align: right;">3</td>
<td style="text-align: right;">17</td>
<td style="text-align: right;">22</td>
<td style="text-align: right;">0.475</td>
<td style="text-align: right;">53.63</td>
<td style="text-align: right;">0.55472</td>
<td style="text-align: right;">0.4667</td>
<td style="text-align: right;">0.4107</td>
<td style="text-align: right;">805.75</td>
</tr>
</tbody>
</table>

**Observation:** The rerun reproduces the earlier conclusion: at this
horizon, idle consolidation is behaviorally indistinguishable from
end-only. The only measurable delta is wall-clock cost, with idle-during
about **4.4% faster** (`842.76ms` → `805.75ms` mean total time), which
is too small to claim a retrieval-quality benefit.

## Affect On/Off (Long Horizon, No Consolidation)

We compared affect gating at **F=S=T=0.5** over **156 turns** (single
conversation) with consolidation disabled to isolate interrupt gating +
retrieval effects.

<table style="width:100%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">affect_mode</th>
<th style="text-align: right;">retr_turn_rate</th>
<th style="text-align: right;">retr_sem_overlap</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">interrupt_sem_overlap</th>
<th style="text-align: right;">perf_total_ms</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">all</td>
<td style="text-align: right;">0.571</td>
<td style="text-align: right;">0.5638</td>
<td style="text-align: right;">0.532</td>
<td style="text-align: right;">0.5702</td>
<td style="text-align: right;">961</td>
</tr>
<tr>
<td style="text-align: right;">off</td>
<td style="text-align: right;">0.571</td>
<td style="text-align: right;">0.5649</td>
<td style="text-align: right;">0.487</td>
<td style="text-align: right;">0.5817</td>
<td style="text-align: right;">966</td>
</tr>
</tbody>
</table>

**Observation:** With affect off, interrupt rate drops (~4.5pp) while
semantic overlap stays within noise. At mid‑range knobs on this dataset,
affect primarily shifts interrupt gating frequency rather than semantic
match quality.

**Current-branch rerun (Mar 28, 2026; nondeterministic; no
consolidation):**

Runs:

-   affect `all`:
    `logs/topical_chat_snapshots/affect_all_nondet_20260328`
-   affect `off`:
    `logs/topical_chat_snapshots/affect_off_nondet_20260328`

<table style="width:100%;">
<colgroup>
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">affect_mode</th>
<th style="text-align: right;">retr_turn_rate</th>
<th style="text-align: right;">retr_sem_overlap</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">interrupt_abort_rate</th>
<th style="text-align: right;">interrupt_sem_overlap</th>
<th style="text-align: right;">retrieval_thresh_mean</th>
<th style="text-align: right;">boundary_mult_eff_mean</th>
<th style="text-align: right;">retrieval_emotion_bonus_mean</th>
<th style="text-align: right;">perf_total_ms</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">all</td>
<td style="text-align: right;">0.5256</td>
<td style="text-align: right;">0.56282</td>
<td style="text-align: right;">0.5128</td>
<td style="text-align: right;">0.4000</td>
<td style="text-align: right;">0.56167</td>
<td style="text-align: right;">0.19781</td>
<td style="text-align: right;">1.25165</td>
<td style="text-align: right;">0.01087</td>
<td style="text-align: right;">869.96</td>
</tr>
<tr>
<td style="text-align: right;">off</td>
<td style="text-align: right;">0.5128</td>
<td style="text-align: right;">0.57689</td>
<td style="text-align: right;">0.2564</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.58511</td>
<td style="text-align: right;">0.23990</td>
<td style="text-align: right;">1.51800</td>
<td style="text-align: right;">0.00000</td>
<td style="text-align: right;">788.92</td>
</tr>
</tbody>
</table>

**Observation:** On the current branch, affect now has a **large**
mixed-content gating effect. Compared with `off`, `affect_mode=all`
lowers the interrupt retrieval threshold (`0.23990` → `0.19781`) and
boundary multiplier (`1.518` → `1.252`), roughly **doubling interrupt
rate** (`0.2564` → `0.5128`) and reintroducing interrupt aborts
(`0.0000` → `0.4000`). Semantic quality also shifts: retrieval semantic
overlap drops by about **0.014** (`0.57689` → `0.56282`) and interrupt
semantic overlap drops by about **0.023** (`0.58511` → `0.56167`). This
is stronger than the earlier December result and suggests later affect /
interrupt-gate tuning made affect a dominant gating relaxer on this
benchmark rather than a small frequency nudge.

**Post-retune probe (Mar 28, 2026; lower interrupt affect relax
coefficient):**

To keep affect as a modulation term rather than a dominant gate relaxer,
we reduced `InterruptAffectRelaxCoeff(S)` from
`lerp(0.15, 0.55, S̃_affect)` to `lerp(0.08, 0.28, S̃_affect)` and reran
`affect_mode=all` on the same benchmark:

-   retuned affect `all`:
    `logs/topical_chat_snapshots/affect_all_relaxed_20260328`

<table style="width:100%;">
<colgroup>
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">affect_mode</th>
<th style="text-align: right;">retr_turn_rate</th>
<th style="text-align: right;">retr_sem_overlap</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">interrupt_abort_rate</th>
<th style="text-align: right;">interrupt_sem_overlap</th>
<th style="text-align: right;">retrieval_thresh_mean</th>
<th style="text-align: right;">boundary_mult_eff_mean</th>
<th style="text-align: right;">perf_total_ms</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">all (pre-retune)</td>
<td style="text-align: right;">0.5256</td>
<td style="text-align: right;">0.56282</td>
<td style="text-align: right;">0.5128</td>
<td style="text-align: right;">0.4000</td>
<td style="text-align: right;">0.56167</td>
<td style="text-align: right;">0.19781</td>
<td style="text-align: right;">1.25165</td>
<td style="text-align: right;">869.96</td>
</tr>
<tr>
<td style="text-align: right;">all (retuned)</td>
<td style="text-align: right;">0.5000</td>
<td style="text-align: right;">0.58003</td>
<td style="text-align: right;">0.3397</td>
<td style="text-align: right;">0.2264</td>
<td style="text-align: right;">0.58000</td>
<td style="text-align: right;">0.21970</td>
<td style="text-align: right;">1.39018</td>
<td style="text-align: right;">843.03</td>
</tr>
<tr>
<td style="text-align: right;">off</td>
<td style="text-align: right;">0.5128</td>
<td style="text-align: right;">0.57689</td>
<td style="text-align: right;">0.2564</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.58511</td>
<td style="text-align: right;">0.23990</td>
<td style="text-align: right;">1.51800</td>
<td style="text-align: right;">788.92</td>
</tr>
</tbody>
</table>

**Observation:** The reduced relax coefficient pulls affect much closer
to the non-affect baseline. Compared with the pre-retune `all` run,
interrupt rate drops from `0.5128` to `0.3397`, abort rate falls from
`0.4000` to `0.2264`, and the effective thresholds move back toward
`off` (`0.19781` → `0.21970`, `1.25165` → `1.39018`). Retrieval semantic
overlap also recovers (`0.56282` → `0.58003`), landing slightly above
the `off` run in this sample, while interrupt semantic overlap nearly
matches `off` (`0.58000` vs `0.58511`). This retune makes affect look
more like a bounded modulation layer again rather than a dominant
mixed-content interrupt trigger.

## EmpatheticDialogues Affect On/Off (No Consolidation)

We repeated the affect toggle on **EmpatheticDialogues (valid)** with
consolidation disabled to isolate gating effects on a more emotional
corpus. Settings: **F=S=T=0.5**, `max_turns=360`, `max_total=720`,
`max_conversations=6` (195 turns observed).

<table style="width:100%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">affect_mode</th>
<th style="text-align: right;">retr_turn_rate</th>
<th style="text-align: right;">retr_sem_overlap</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">interrupt_sem_overlap</th>
<th style="text-align: right;">perf_total_ms</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">all</td>
<td style="text-align: right;">0.518</td>
<td style="text-align: right;">0.5959</td>
<td style="text-align: right;">0.487</td>
<td style="text-align: right;">0.6026</td>
<td style="text-align: right;">1163</td>
</tr>
<tr>
<td style="text-align: right;">off</td>
<td style="text-align: right;">0.518</td>
<td style="text-align: right;">0.5959</td>
<td style="text-align: right;">0.482</td>
<td style="text-align: right;">0.6045</td>
<td style="text-align: right;">1156</td>
</tr>
</tbody>
</table>

**Observation:** On EmpatheticDialogues without consolidation, affect
has a small but consistent effect on interrupt frequency (~0.5pp) while
semantic overlap remains essentially unchanged. The gating effect is
visible but modest at mid‑range knobs.

## Source-Confidence Gating Check

Natural-corpus reruns remained inconclusive because the usual
TopicalChat / EmpatheticDialogues / Ubuntu snapshots rarely surface
memories whose computed `source_confidence` falls below the active
threshold. To verify that source monitoring is actually gating retrieval
end to end, we ran a **controlled probe** on the current branch:

-   start from the current Ubuntu validation DB at **F=S=0.5, T=1.0**
    (`theta_src = 0.45`)
-   copy the DB twice and clear `last_access`
-   downgrade the 128 previously retrieved memories to
    `source_reliability=0.2`, `source_contradiction_count=3`,
    `source_origin='external'`
-   with stale verification timestamps, these probe memories land at
    `source_confidence ≈ 0.077`
-   replay the same 80-turn Ubuntu slice once with gating enabled and
    once with `CORTEXT_DISABLE_SOURCE_CONF=1`

Runs:

-   strict probe:
    `logs/topical_chat_snapshots/sourceconf_probe_on_20260328`
-   disabled probe:
    `logs/topical_chat_snapshots/sourceconf_probe_off_20260328`

<table style="width:100%;">
<colgroup>
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">source_conf</th>
<th style="text-align: right;">downgraded_probe_reaccessed</th>
<th style="text-align: right;">low_conf_recent_count</th>
<th style="text-align: right;">retr_avg_candidates</th>
<th style="text-align: right;">retr_sem_overlap</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">perf_total_ms</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">on</td>
<td style="text-align: right;">0 / 128</td>
<td style="text-align: right;">71</td>
<td style="text-align: right;">16.95</td>
<td style="text-align: right;">0.58276</td>
<td style="text-align: right;">0.0375</td>
<td style="text-align: right;">710.87</td>
</tr>
<tr>
<td style="text-align: right;">off</td>
<td style="text-align: right;">113 / 128</td>
<td style="text-align: right;">183</td>
<td style="text-align: right;">56.90</td>
<td style="text-align: right;">0.69039</td>
<td style="text-align: right;">0.0500</td>
<td style="text-align: right;">1042.75</td>
</tr>
</tbody>
</table>

**Observation:** This controlled probe finally exposes the intended
behavior. With source-confidence gating enabled, **none** of the
deliberately downgraded memories are reaccessed. Disabling the gate
causes **113 / 128** of them to reappear. Candidate volume also jumps
sharply (`16.95` → `56.90`), which is consistent with the strict filter
removing a large low-confidence cohort before diversification. Some
low-confidence memories still appear in `recent_retrievals` even with
gating enabled (`71` rows below threshold), because graph retrieval
falls back to a relaxed pass when the strict filtered set is empty; the
important result is that the deliberately poisoned cohort stays out when
the strict gate is active.

## Scripted Chat Working-Memory Validation

The broad topical-chat ablations above do not exercise the exact
prompt-reconstruction path used by the interactive chat application. To
check whether working memory still behaves like recent chat history
under long runs, we added a **scripted 100-turn integration test** that
alternates user and assistant turns through the real high-level chat
processing path:

-   user turns commit through `ProcessTextAt(..., "chat/user", ...)`
-   assistant turns stream through the non-durable probe path, then
    finalize once as `chat/assistant`
-   after each user turn, prompt reconstruction is compared against the
    exact recent conversational tail
-   at the end of the run, the active working-memory rows and the final
    prompt are inspected directly

The initial failure looked like an algorithmic recall problem, but it
was not: late-run prompts drifted into malformed histories such as
`assistant 82, user 83, assistant 84, assistant 100`, even though the
intended recent tail was still alternating. Root cause analysis showed
two chat-specific policy bugs:

1.  full `chat/user` / `chat/assistant` turns were still passing through
    the generic working-memory admission and eviction logic rather than
    being preserved as prompt-bearing dialogue turns;
2.  prompt hydration was ordering active working-memory slots by
    `last_access`, so rehearsal or retrieval touches could reorder the
    visible chat history.

After fixing both issues, the same 100-turn scripted run again produced
the exact last four turns in order and the backing store ended with
exactly four active `WORKING` rows (`end_ts IS NULL`), each with
`n_signals = 1`.

**Observation:** This does **not** warrant a new topical-chat ablation.
The failure was not a missing long-horizon memory effect; it was a
product-path regression in chat-turn preservation and prompt ordering.
The correct follow-up is the scripted integration guard, not another
broad retrieval sweep.

## Recursive Summary Replay Revision

During live chat inspection on **Mar 30, 2026**, we observed that the
injected `<memories>` block could accumulate multiple near-duplicate
summary memories whose text had clearly been abstracted from earlier
summaries rather than directly from raw episodic turns. This looked like
repeated “summary of summaries” compression rather than fresh replay of
the original conversation evidence.

Root cause analysis showed that the previous policy admitted both
`LONG_TERM` and blob-backed `ASSOCIATION` memories into deep
consolidation. Because deep summaries are themselves stored as
`ASSOCIATION` memories with blobs, later consolidation cycles could
summarize prior summaries again.

We revised this policy on the current branch:

-   same-level consolidation candidates are now restricted to
    **unclustered `LONG_TERM` memories**;
-   existing `ASSOCIATION` summaries remain retrievable semantic
    products but are **sealed** with respect to same-level deep replay;
-   repeated deep consolidation with **no new raw evidence** should
    therefore create **no new summary nodes**.

This change preserves the ability to form additional sibling summaries
when fresh raw memories accumulate, while preventing repeated
summary-of-summary compression in the injected memory block. The correct
validation for this behavior is a focused chat-style E2E that repeats
deep consolidation on the same DB, not another broad topical-chat
retrieval ablation.

## Procedural + Sequential Link Ablation

We added toggles to disable the **procedural store** and **sequential
edges** and re‑ran a 156‑turn topical‑chat run with consolidation
enabled (`consolidate_every=40`, 4 runs). Settings were **F=S=T=0.5**.

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
<th style="text-align: right;">variant</th>
<th style="text-align: right;">retr_sem_overlap</th>
<th style="text-align: right;">retr_assoc_rate</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">perf_total_ms</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">baseline (procedural+sequential on)</td>
<td style="text-align: right;">0.5689</td>
<td style="text-align: right;">0.0241</td>
<td style="text-align: right;">0.5449</td>
<td style="text-align: right;">946</td>
</tr>
<tr>
<td style="text-align: right;">ablation (both off)</td>
<td style="text-align: right;">0.5688</td>
<td style="text-align: right;">0.0241</td>
<td style="text-align: right;">0.5449</td>
<td style="text-align: right;">954</td>
</tr>
</tbody>
</table>

**Observation:** At this horizon and dataset, disabling
procedural/sequence contributions does **not** materially change
retrieval or interrupt quality (Δ semantic overlap ≈ −0.00009). This
suggests either the effect is subtle at short horizons or the current
metrics are not sensitive to these pathways; longer horizons or targeted
tasks may be required to surface their impact.

We extended the comparison to a longer consolidation cadence (156 turns
observed, `consolidate_every=60`) with identical knobs:

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
<th style="text-align: right;">variant</th>
<th style="text-align: right;">retr_sem_overlap</th>
<th style="text-align: right;">retr_assoc_rate</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">perf_total_ms</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">baseline (procedural+sequential on)</td>
<td style="text-align: right;">0.5707</td>
<td style="text-align: right;">0.0156</td>
<td style="text-align: right;">0.5385</td>
<td style="text-align: right;">971</td>
</tr>
<tr>
<td style="text-align: right;">ablation (both off)</td>
<td style="text-align: right;">0.5704</td>
<td style="text-align: right;">0.0156</td>
<td style="text-align: right;">0.5385</td>
<td style="text-align: right;">969</td>
</tr>
</tbody>
</table>

**Observation:** Even with a longer cadence, deltas remain negligible (Δ
semantic overlap ≈ −0.00035). This further supports the need for longer
horizons or targeted sequential‑recall tasks to surface
procedural/sequence benefits.

**Affect on/off sanity (20 turns; deep, consolidate_every=20):**

<table style="width:100%;">
<colgroup>
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">affect_mode</th>
<th style="text-align: right;">retr_turn_rate</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">interrupt_precision</th>
<th style="text-align: right;">interrupt_recall</th>
<th style="text-align: right;">retrieval_emotion_bonus_mean</th>
<th style="text-align: right;">interrupt_gate_retrieval_thresh_mean</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">all</td>
<td style="text-align: right;">0.50</td>
<td style="text-align: right;">0.40</td>
<td style="text-align: right;">1.00</td>
<td style="text-align: right;">0.80</td>
<td style="text-align: right;">0.00185</td>
<td style="text-align: right;">0.223</td>
</tr>
<tr>
<td style="text-align: right;">off</td>
<td style="text-align: right;">0.50</td>
<td style="text-align: right;">0.40</td>
<td style="text-align: right;">1.00</td>
<td style="text-align: right;">0.80</td>
<td style="text-align: right;">0.00000</td>
<td style="text-align: right;">0.240</td>
</tr>
</tbody>
</table>

**Observation:** Affect pathways are active (bonus and lower interrupt
threshold when enabled) but do not materially change interrupt
precision/recall at this short horizon.

**Source monitoring + sequential edges (baseline DB):** 64 memories
store non‑default source metadata and sequential edges are present
(`next_in_episode`, `prev_in_episode`, `within_same_event`, 19 each),
confirming provenance tagging and ordered links are persisted during
consolidation.

## Long-Horizon Consolidation Mode Check (156 Turns)

We ran a **156‑turn** comparison at **F=S=T=0.5** to evaluate
**baseline**, **shallow**, and **deep** consolidation in the
topical‑chat pipeline. Consolidation ran **8 times**
(`consolidate_every=40`, `consolidate_cycles=2`), and deep used
gemma‑3n‑e2b via LiteRT for summarization/labeling. Runs are from **Dec
30, 2025**:

-   baseline:
    `logs/topical_chat_snapshots/20251230_202959_long_horizon/baseline`
-   shallow:
    `logs/topical_chat_snapshots/20251230_202959_long_horizon/shallow`
-   deep:
    `logs/topical_chat_snapshots/20251230_202959_long_horizon/deep`

<table style="width:100%;">
<colgroup>
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">mode</th>
<th style="text-align: right;">cons_runs</th>
<th style="text-align: right;">assoc_created</th>
<th style="text-align: right;">labels_created</th>
<th style="text-align: right;">summaries</th>
<th style="text-align: right;">retr_turn_rate</th>
<th style="text-align: right;">retr_avg_cands</th>
<th style="text-align: right;">retr_sem_overlap</th>
<th style="text-align: right;">assoc_rate</th>
<th style="text-align: right;">label_rate</th>
<th style="text-align: right;">assoc_turn_rate</th>
<th style="text-align: right;">label_turn_rate</th>
<th style="text-align: right;">interrupt_rate</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">baseline</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0.500</td>
<td style="text-align: right;">18.54</td>
<td style="text-align: right;">0.4746</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.4487</td>
</tr>
<tr>
<td style="text-align: right;">shallow</td>
<td style="text-align: right;">8</td>
<td style="text-align: right;">8</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0.526</td>
<td style="text-align: right;">25.18</td>
<td style="text-align: right;">0.4853</td>
<td style="text-align: right;">0.1269</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.8293</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.4808</td>
</tr>
<tr>
<td style="text-align: right;">deep</td>
<td style="text-align: right;">8</td>
<td style="text-align: right;">8</td>
<td style="text-align: right;">5</td>
<td style="text-align: right;">8</td>
<td style="text-align: right;">0.526</td>
<td style="text-align: right;">27.70</td>
<td style="text-align: right;">0.4868</td>
<td style="text-align: right;">0.1154</td>
<td style="text-align: right;">0.0766</td>
<td style="text-align: right;">0.8293</td>
<td style="text-align: right;">0.8293</td>
<td style="text-align: right;">0.4808</td>
</tr>
</tbody>
</table>

**Observations:** Shallow consolidation alone increases association hits
and retrieval breadth (candidate count) while slightly improving
semantic overlap. Deep consolidation adds labels and summaries that are
actively retrieved (label rate
**<sub>7.7%**,\ label\ turn\ rate\ **</sub>83%**) without degrading
interrupt precision. This indicates deep consolidation contributes
structured semantic signals beyond shallow associations.

## Reinforcement Ablation (Long Horizon)

We reran the reinforcement ablation on **Apr 4, 2026** under
`EmbeddingGemma/llama.cpp` by comparing **reinforcement on vs off** at
**F=S=T=0.5**, `max_total=360`, `max_turns=360`, `max_conversations=6`
(single stream, interleave=1). Consolidation was invoked externally
(`consolidate_cycles=2`). Rerun root:
`logs/embeddinggemma_paper_harness_ablation_fast_20260404`.

<table>
<colgroup>
<col style="width: 12%" />
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
<th style="text-align: right;">reinforcement_mode</th>
<th style="text-align: right;">reinforcement_edge_count</th>
<th style="text-align: right;">retrieval_avg_candidates</th>
<th style="text-align: right;">retrieval_semantic_overlap_mean</th>
<th style="text-align: right;">interrupt_precision</th>
<th style="text-align: right;">interrupt_recall</th>
<th style="text-align: right;">interrupt_fn_rate</th>
<th style="text-align: right;">memory_retrieved_count_mean</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">on</td>
<td style="text-align: right;">780</td>
<td style="text-align: right;">33.36</td>
<td style="text-align: right;">0.8351</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.8983</td>
<td style="text-align: right;">0.1017</td>
<td style="text-align: right;">21.52</td>
</tr>
<tr>
<td style="text-align: right;">off</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">32.26</td>
<td style="text-align: right;">0.8367</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.9138</td>
<td style="text-align: right;">0.0862</td>
<td style="text-align: right;">20.35</td>
</tr>
</tbody>
</table>

**Observations:** The older ImageBind-era reinforcement claim does
**not** reproduce on the real encoder path. Under EmbeddingGemma,
reinforcement changes candidate breadth only slightly (+1.10
candidates/turn) and does **not** improve interrupt quality: `off` is
marginally higher on both recall (**0.9138** vs **0.8983**) and semantic
overlap (**0.8367** vs **0.8351**). Treat the earlier “substantially
increases breadth and modestly improves recall” conclusion as superseded
for this benchmark.

## Affect Path Ablation (Long Horizon)

We reran the affect-path ablation on **Apr 4, 2026** under
`EmbeddingGemma/llama.cpp` by toggling affect usage in **interrupt
gating** vs **retrieval ranking** while holding **F=S=T=0.5** and
running **2 conversations × 360 turns** (`max_total=720`). Rerun root:
`logs/embeddinggemma_paper_harness_ablation_fast_20260404`. Four modes
were tested:

-   **all:** affect used in both interrupt gating and retrieval ranking
-   **interrupt:** affect used only in interrupt gating
-   **retrieval:** affect used only in retrieval ranking
-   **off:** affect disabled for both pathways

<table style="width:100%;">
<colgroup>
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">affect_mode</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">precision</th>
<th style="text-align: right;">recall</th>
<th style="text-align: right;">fp_rate</th>
<th style="text-align: right;">fn_rate</th>
<th style="text-align: right;">retrieval_emotion_bonus_mean</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">all</td>
<td style="text-align: right;">0.151</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.908</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.0923</td>
<td style="text-align: right;">0.00921</td>
</tr>
<tr>
<td style="text-align: right;">interrupt</td>
<td style="text-align: right;">0.151</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.894</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.1061</td>
<td style="text-align: right;">0</td>
</tr>
<tr>
<td style="text-align: right;">retrieval</td>
<td style="text-align: right;">0.153</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.882</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.1176</td>
<td style="text-align: right;">0.00924</td>
</tr>
<tr>
<td style="text-align: right;">off</td>
<td style="text-align: right;">0.153</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.882</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.1176</td>
<td style="text-align: right;">0</td>
</tr>
</tbody>
</table>

**Observations:** The older flat ImageBind-era table is superseded.
Under EmbeddingGemma, affect does produce a measurable but modest
interrupt-path effect at mid-range knobs: `all` improves recall over
`off` (**0.9077** vs **0.8824**) while keeping precision at **1.0** and
slightly lowering interrupt rate (**0.1509** vs **0.1535**). `retrieval`
matches `off` on interrupt metrics and differs only in the expected
emotion bonus, while `interrupt` sits between `all` and `off`. That
pattern indicates the gain comes primarily from affect in the interrupt
gate rather than from retrieval ranking alone.

We further repeated the ablation at **S=0.0** and **S=1.0** (F=T=0.5) to
test extremes:

<table style="width:100%;">
<colgroup>
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">S</th>
<th style="text-align: right;">affect_mode</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">precision</th>
<th style="text-align: right;">recall</th>
<th style="text-align: right;">fn_rate</th>
<th style="text-align: right;">retrieval_emotion_bonus_mean</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">0.0</td>
<td style="text-align: right;">all</td>
<td style="text-align: right;">0.486</td>
<td style="text-align: right;">0.995</td>
<td style="text-align: right;">0.879</td>
<td style="text-align: right;">0.121</td>
<td style="text-align: right;">0</td>
</tr>
<tr>
<td style="text-align: right;">0.0</td>
<td style="text-align: right;">off</td>
<td style="text-align: right;">0.486</td>
<td style="text-align: right;">0.995</td>
<td style="text-align: right;">0.879</td>
<td style="text-align: right;">0.121</td>
<td style="text-align: right;">0</td>
</tr>
<tr>
<td style="text-align: right;">1.0</td>
<td style="text-align: right;">all</td>
<td style="text-align: right;">0.512</td>
<td style="text-align: right;">0.985</td>
<td style="text-align: right;">0.938</td>
<td style="text-align: right;">0.0619</td>
<td style="text-align: right;">0.0102</td>
</tr>
<tr>
<td style="text-align: right;">1.0</td>
<td style="text-align: right;">off</td>
<td style="text-align: right;">0.491</td>
<td style="text-align: right;">0.984</td>
<td style="text-align: right;">0.900</td>
<td style="text-align: right;">0.100</td>
<td style="text-align: right;">0</td>
</tr>
</tbody>
</table>

**Observations:** At **S=0.0**, affect has no measurable effect (as
expected from the knob mapping). At **S=1.0**, enabling affect **raises
interrupt recall** (FN rate drops from 0.100 → 0.0619) at the cost of a
higher interrupt rate, confirming affect contributes primarily at high
sensitivity.

## EmpatheticDialogues Affect Check (Calibrated)

To test a more emotionally charged corpus, we converted
**EmpatheticDialogues (valid split)** into Cortext JSONL format and ran
**F=0.5, T=0.5** with **6 conversations** and `max_total=360` (195 turns
observed). We compared `affect_mode=all` vs `off` at **S=0.5** and
**S=1.0** (with consolidation cycles enabled) after increasing affect
gain and relax coefficients.

<table style="width:100%;">
<colgroup>
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">S</th>
<th style="text-align: right;">affect_mode</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">precision</th>
<th style="text-align: right;">recall</th>
<th style="text-align: right;">fn_rate</th>
<th style="text-align: right;">retrieval_emotion_bonus_mean</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">0.5</td>
<td style="text-align: right;">all</td>
<td style="text-align: right;">0.472</td>
<td style="text-align: right;">0.989</td>
<td style="text-align: right;">0.978</td>
<td style="text-align: right;">0.0215</td>
<td style="text-align: right;">0.00856</td>
</tr>
<tr>
<td style="text-align: right;">0.5</td>
<td style="text-align: right;">off</td>
<td style="text-align: right;">0.462</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.968</td>
<td style="text-align: right;">0.0323</td>
<td style="text-align: right;">0</td>
</tr>
<tr>
<td style="text-align: right;">1.0</td>
<td style="text-align: right;">all</td>
<td style="text-align: right;">0.462</td>
<td style="text-align: right;">0.967</td>
<td style="text-align: right;">0.888</td>
<td style="text-align: right;">0.112</td>
<td style="text-align: right;">0.0237</td>
</tr>
<tr>
<td style="text-align: right;">1.0</td>
<td style="text-align: right;">off</td>
<td style="text-align: right;">0.462</td>
<td style="text-align: right;">0.967</td>
<td style="text-align: right;">0.888</td>
<td style="text-align: right;">0.112</td>
<td style="text-align: right;">0</td>
</tr>
</tbody>
</table>

**Observations:** At **S=0.5**, affect now shifts gate behavior:
interrupt rate rises by ~1.0pp and recall improves by ~1.1pp, with a
~1.1pp precision trade‑off. The mean interrupt threshold drops from
~0.240 → 0.221 and boundary multiplier from ~1.52 → 1.40, confirming
measurable gating relaxation. At **S=1.0**, thresholds are substantially
relaxed but the interrupt decision remains saturated on this dataset;
only the retrieval emotion bonus changes.

## EmpatheticDialogues Affect Check (Long Horizon)

We repeated the EmpatheticDialogues check at longer horizon with **10
conversations** and `max_total=720` (371 turns observed), holding
**F=0.5, S=0.5, T=0.5** and comparing `affect_mode=all` vs `off`.
Consolidation ran **20** times with zero failures.

<table>
<colgroup>
<col style="width: 12%" />
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
<th style="text-align: right;">affect_mode</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">precision</th>
<th style="text-align: right;">recall</th>
<th style="text-align: right;">fn_rate</th>
<th style="text-align: right;">retrieval_emotion_bonus_mean</th>
<th style="text-align: right;">retrieval_thresh_mean</th>
<th style="text-align: right;">boundary_mult_eff_mean</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">all</td>
<td style="text-align: right;">0.493</td>
<td style="text-align: right;">0.989</td>
<td style="text-align: right;">0.978</td>
<td style="text-align: right;">0.0216</td>
<td style="text-align: right;">0.00841</td>
<td style="text-align: right;">0.220</td>
<td style="text-align: right;">1.394</td>
</tr>
<tr>
<td style="text-align: right;">off</td>
<td style="text-align: right;">0.477</td>
<td style="text-align: right;">0.994</td>
<td style="text-align: right;">0.951</td>
<td style="text-align: right;">0.0486</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0.240</td>
<td style="text-align: right;">1.518</td>
</tr>
</tbody>
</table>

**Observations:** Over a longer horizon, affect increases interrupt rate
by ~1.6pp and improves recall by ~2.7pp, with a ~0.5pp precision
trade‑off. The interrupt threshold and boundary multiplier relax (0.240
→ 0.220; 1.52 → 1.39), confirming affect contributes to gating and not
just retrieval ranking.

### Interrupt Ablation (Affect Modes, Short Horizon)

We ran a 120‑turn interrupt ablation with `affect_mode` toggles (no
consolidation). Runs:

-   all: `logs/topical_chat_snapshots/20260102_175025`
-   interrupt‑only: `logs/topical_chat_snapshots/20260102_180301`
-   retrieval‑only: `logs/topical_chat_snapshots/20260102_203459`
-   none: `logs/topical_chat_snapshots/20260102_203514`

<table style="width:100%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th>affect_mode</th>
<th>interrupt_turn_rate</th>
<th>interrupt_abort_rate</th>
<th>interrupt_semantic_overlap_mean</th>
<th>interrupt_context_semantic_overlap_mean</th>
<th>boundary_at_rate</th>
</tr>
</thead>
<tbody>
<tr>
<td>all</td>
<td>0.475</td>
<td>0.474</td>
<td>0.541</td>
<td>0.658</td>
<td>0.150</td>
</tr>
<tr>
<td>interrupt</td>
<td>0.458</td>
<td>0.400</td>
<td>0.534</td>
<td>0.661</td>
<td>0.158</td>
</tr>
<tr>
<td>retrieval</td>
<td>0.175</td>
<td>0.000</td>
<td>0.496</td>
<td>0.661</td>
<td>0.200</td>
</tr>
<tr>
<td>none</td>
<td>0.458</td>
<td>0.364</td>
<td>0.534</td>
<td>0.657</td>
<td>0.167</td>
</tr>
</tbody>
</table>

**Observations:** At **S=0.5**, affect‑only interrupts suppress gate
firing dramatically and eliminate aborts, but reduce interrupt rate.
Affect‑off mirrors baseline, indicating that non‑affect gating dominates
interrupt decisions in this dataset. This suggests affect modulation
primarily acts as a gain on the interrupt gate rather than a precision
enhancer at mid‑sensitivity.

### Procedural/Sequential Interrupt Ablation

We ablated procedural retrieval and sequential edges (no consolidation,
120 turns). Runs:

-   no‑procedural: `logs/topical_chat_snapshots/20260103_002742`
-   no‑sequential: `logs/topical_chat_snapshots/20260103_085920`
-   no‑procedural + no‑sequential:
    `logs/topical_chat_snapshots/20260103_092405`

<table style="width:100%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th>ablation</th>
<th>interrupt_turn_rate</th>
<th>interrupt_abort_rate</th>
<th>interrupt_semantic_overlap_mean</th>
<th>interrupt_context_semantic_overlap_mean</th>
<th>boundary_at_rate</th>
</tr>
</thead>
<tbody>
<tr>
<td>no‑procedural</td>
<td>0.442</td>
<td>0.358</td>
<td>0.534</td>
<td>0.665</td>
<td>0.158</td>
</tr>
<tr>
<td>no‑sequential</td>
<td>0.442</td>
<td>0.358</td>
<td>0.534</td>
<td>0.665</td>
<td>0.158</td>
</tr>
<tr>
<td>both off</td>
<td>0.442</td>
<td>0.358</td>
<td>0.534</td>
<td>0.665</td>
<td>0.158</td>
</tr>
</tbody>
</table>

**Observations:** On this dataset, disabling procedural retrieval and
sequential edges has **no measurable effect** on interrupt behavior at
120 turns. This suggests the interrupt gate is dominated by
semantic/label candidates in TopicalChat; procedural/sequence pathways
likely require tasks with action repetition or stronger temporal
dependencies to show impact.

### Source-Confidence Gating Ablation

We toggled source‑confidence filtering in graph retrieval using an
experiment‑only env flag (`CORTEXT_DISABLE_SOURCE_CONF=1`) to compare
strict vs disabled gating over 120 turns (no consolidation). Runs:

-   baseline (on): `logs/topical_chat_snapshots/20260103_102041`
-   disabled: `logs/topical_chat_snapshots/20260103_102058`

<table style="width:100%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th>source_conf</th>
<th>interrupt_turn_rate</th>
<th>interrupt_abort_rate</th>
<th>interrupt_semantic_overlap_mean</th>
<th>interrupt_context_semantic_overlap_mean</th>
<th>boundary_at_rate</th>
</tr>
</thead>
<tbody>
<tr>
<td>on</td>
<td>0.458</td>
<td>0.400</td>
<td>0.534</td>
<td>0.661</td>
<td>0.158</td>
</tr>
<tr>
<td>off</td>
<td>0.458</td>
<td>0.400</td>
<td>0.534</td>
<td>0.661</td>
<td>0.158</td>
</tr>
</tbody>
</table>

**Observations:** For TopicalChat at mid‑knob settings, disabling
source‑confidence gating has **no measurable effect** on interrupts or
overlap quality. This likely reflects high source reliability in the
dataset; a mixed‑provenance or contradiction‑heavy corpus is needed to
expose the benefit of source monitoring.

**Current-branch controlled probe (Mar 28, 2026):** We created exactly
that contradiction-heavy condition by downgrading a previously retrieved
Ubuntu-memory cohort to `source_confidence ≈ 0.077` at **T=1.0** and
replaying the same slice with gating on/off.

<table style="width:100%;">
<colgroup>
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
</colgroup>
<thead>
<tr>
<th>source_conf</th>
<th>downgraded_probe_reaccessed</th>
<th>low_conf_recent_count</th>
<th>retr_avg_candidates</th>
<th>retr_sem_overlap</th>
<th>interrupt_turn_rate</th>
<th>perf_total_ms</th>
</tr>
</thead>
<tbody>
<tr>
<td>on</td>
<td>0 / 128</td>
<td>71</td>
<td>16.95</td>
<td>0.58276</td>
<td>0.0375</td>
<td>710.87</td>
</tr>
<tr>
<td>off</td>
<td>113 / 128</td>
<td>183</td>
<td>56.90</td>
<td>0.69039</td>
<td>0.0500</td>
<td>1042.75</td>
</tr>
</tbody>
</table>

**Observations:** The old TopicalChat no-op result was a dataset
problem, not a dead feature. On the controlled Ubuntu probe,
source-confidence gating clearly blocks the poisoned cohort: **0 / 128**
downgraded memories are reaccessed with the gate on, versus **113 /
128** with the gate disabled.

## Apr 4, 2026 EmbeddingGemma + Default-LFM2.5 Addendum

We reran a compact topical-chat ablation batch on the rebuilt branch
where corpus `run.log` files now record
`text_encoder_backend=EmbeddingGemma/llama.cpp` and the rebuilt
integration tests validate auto deep-backend selection to
`LFM2.5/llama.cpp`. Rerun root:
`logs/embeddinggemma_remaining_historical_ablations_lfm25_default_20260404_final`.

Unless noted otherwise, these addendum checks use a **single 120-turn
TopicalChat slice** at **F=S=T=0.5**. They are intended as current-model
reference points for the rebuilt default stack, not as replacements for
the longer historical horizon sweeps above.

### Consolidation Quick Check (120 Turns)

<table style="width:100%;">
<colgroup>
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">condition</th>
<th style="text-align: right;">turns</th>
<th style="text-align: right;">cons_runs</th>
<th style="text-align: right;">summaries</th>
<th style="text-align: right;">extraction_results</th>
<th style="text-align: right;">labels_seen</th>
<th style="text-align: right;">retrieval_semantic_overlap_mean</th>
<th style="text-align: right;">interrupt_precision</th>
<th style="text-align: right;">interrupt_recall</th>
<th style="text-align: right;">retrieval_summary_hit_rate</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">off</td>
<td style="text-align: right;">120</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0</td>
<td style="text-align: right;">0.7791</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.750</td>
<td style="text-align: right;">0.0000</td>
</tr>
<tr>
<td style="text-align: right;">on</td>
<td style="text-align: right;">120</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;">1</td>
<td style="text-align: right;">1</td>
<td style="text-align: right;">12</td>
<td style="text-align: right;">0.7742</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.750</td>
<td style="text-align: right;">0.0000</td>
</tr>
</tbody>
</table>

**Observation:** On this short slice, consolidation does produce a real
summary/extraction event, but it does **not** surface into retrieval
often enough to move interrupt quality. The historical longer-horizon
consolidation gains should therefore be read as horizon-sensitive rather
than assumed at short horizons on the rebuilt stack.

### Affect Modes (120 Turns, No Consolidation)

<table>
<colgroup>
<col style="width: 12%" />
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
<th style="text-align: right;">affect_mode</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">interrupt_abort_rate</th>
<th style="text-align: right;">interrupt_semantic_overlap_mean</th>
<th
style="text-align: right;">interrupt_context_semantic_overlap_mean</th>
<th style="text-align: right;">retrieval_thresh_mean</th>
<th style="text-align: right;">boundary_mult_eff_mean</th>
<th style="text-align: right;">retrieval_emotion_bonus_mean</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">all</td>
<td style="text-align: right;">0.0500</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.7750</td>
<td style="text-align: right;">0.9082</td>
<td style="text-align: right;">0.2380</td>
<td style="text-align: right;">1.5057</td>
<td style="text-align: right;">0.00833</td>
</tr>
<tr>
<td style="text-align: right;">interrupt</td>
<td style="text-align: right;">0.0500</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.7750</td>
<td style="text-align: right;">0.9082</td>
<td style="text-align: right;">0.2380</td>
<td style="text-align: right;">1.5057</td>
<td style="text-align: right;">0.00000</td>
</tr>
<tr>
<td style="text-align: right;">retrieval</td>
<td style="text-align: right;">0.0500</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.7682</td>
<td style="text-align: right;">0.9081</td>
<td style="text-align: right;">0.2399</td>
<td style="text-align: right;">1.5180</td>
<td style="text-align: right;">0.00829</td>
</tr>
<tr>
<td style="text-align: right;">off</td>
<td style="text-align: right;">0.0500</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.7682</td>
<td style="text-align: right;">0.9081</td>
<td style="text-align: right;">0.2399</td>
<td style="text-align: right;">1.5180</td>
<td style="text-align: right;">0.00000</td>
</tr>
</tbody>
</table>

**Observation:** On the rebuilt default stack, the short-horizon affect
ablation collapses into two pairs. `all` matches `interrupt` on gate
outcomes, and `retrieval` matches `off`, with only the expected
path-local threshold shift / emotion bonus differences. That is a much
weaker result than the older large ImageBind-era gap.

### Procedural + Sequential Quick Check (Consolidation Cadence)

These quick reruns use one TopicalChat conversation with consolidation
enabled and explicit cadence overrides.

<table style="width:100%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">variant</th>
<th style="text-align: right;">consolidate_every</th>
<th style="text-align: right;">retrieval_avg_candidates</th>
<th style="text-align: right;">retrieval_association_candidate_rate</th>
<th style="text-align: right;">retrieval_semantic_overlap_mean</th>
<th style="text-align: right;">interrupt_turn_rate</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">baseline</td>
<td style="text-align: right;">40</td>
<td style="text-align: right;">11.75</td>
<td style="text-align: right;">0.0993</td>
<td style="text-align: right;">0.8001</td>
<td style="text-align: right;">0.1000</td>
</tr>
<tr>
<td style="text-align: right;">both off</td>
<td style="text-align: right;">40</td>
<td style="text-align: right;">11.79</td>
<td style="text-align: right;">0.0989</td>
<td style="text-align: right;">0.8017</td>
<td style="text-align: right;">0.0917</td>
</tr>
<tr>
<td style="text-align: right;">baseline</td>
<td style="text-align: right;">60</td>
<td style="text-align: right;">10.10</td>
<td style="text-align: right;">0.0660</td>
<td style="text-align: right;">0.8023</td>
<td style="text-align: right;">0.1000</td>
</tr>
<tr>
<td style="text-align: right;">both off</td>
<td style="text-align: right;">60</td>
<td style="text-align: right;">10.14</td>
<td style="text-align: right;">0.0657</td>
<td style="text-align: right;">0.8042</td>
<td style="text-align: right;">0.1000</td>
</tr>
</tbody>
</table>

**Observation:** The rebuilt-stack quick reruns again show no material
degradation from disabling procedural storage and sequential edges. If
anything, the tiny deltas slightly favor the ablation. The correct
interpretation remains that this pathway needs a stronger sequential
task to demonstrate value.

### Source-Confidence Quick Check (120 Turns)

<table style="width:100%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">source_conf</th>
<th style="text-align: right;">retrieval_avg_candidates</th>
<th style="text-align: right;">retrieval_semantic_overlap_mean</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">interrupt_abort_rate</th>
<th style="text-align: right;">interrupt_semantic_overlap_mean</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">on</td>
<td style="text-align: right;">1.3750</td>
<td style="text-align: right;">0.7742</td>
<td style="text-align: right;">0.0500</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.7682</td>
</tr>
<tr>
<td style="text-align: right;">off</td>
<td style="text-align: right;">1.4286</td>
<td style="text-align: right;">0.7842</td>
<td style="text-align: right;">0.0417</td>
<td style="text-align: right;">0.0000</td>
<td style="text-align: right;">0.7811</td>
</tr>
</tbody>
</table>

**Observation:** The clean TopicalChat slice still does not stress
source monitoring enough to produce a strong causal delta. The
contradiction-heavy Ubuntu controlled probe above remains the better
evidence that source-confidence filtering is functionally active.

### Synthetic Mechanism-Pack Smoke (Apr 4, 2026)

We also generated a small deterministic synthetic mechanism pack
(`data/mechanism_eval/valid.jsonl`) containing delayed corrections,
mixed provenance, conflicting low-confidence claims, routine exceptions,
current-vs-historical questions, and resumable routines. The
corresponding quick harness smoke lives under
`logs/mechanism_eval_20260404`.

<table style="width:100%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr>
<th style="text-align: right;">source_conf</th>
<th style="text-align: right;">turns</th>
<th style="text-align: right;">retrieval_avg_candidates</th>
<th style="text-align: right;">retrieval_semantic_overlap_mean</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">duration_s</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">on</td>
<td style="text-align: right;">19</td>
<td style="text-align: right;">1.0000</td>
<td style="text-align: right;">0.7853</td>
<td style="text-align: right;">0.0526</td>
<td style="text-align: right;">2.52</td>
</tr>
<tr>
<td style="text-align: right;">off</td>
<td style="text-align: right;">19</td>
<td style="text-align: right;">1.0000</td>
<td style="text-align: right;">0.7853</td>
<td style="text-align: right;">0.0526</td>
<td style="text-align: right;">2.02</td>
</tr>
</tbody>
</table>

**Observation:** Raw harness aggregates remain flat on this tiny pack.
That is expected: the pack is intended to be scored against explicit
answer keys (current, historical, belief-at-time, stale-first,
provenance class) using the external study scripts, not interpreted only
through interrupt/retrieval averages. The important implementation
result is that the rebuilt branch now has a reproducible
contradiction-heavy mechanism dataset and scorer path without changing
the public runtime API.

### Predictive + Metacognitive Activation Audit (Apr 4, 2026)

Two mechanisms were previously present in state but not convincingly
behavior-active in retrieval: predictive pre-activation and
metacognitive TOT/FOK control. We activated both without changing the
public API, then added deterministic on/off benches in
`examples/benchmark` plus targeted operation tests. A final paper-only
cleanup pass also made the previously paper-only `confidence_decay_rate`
behaviorally real: retained metacognitive confidence now decays between
signals, scales the strength of next-window TOT recovery, and tightens
unknown-caution cutoff behavior as confidence falls.

**Predictive deterministic ablation:**
`./build/examples/benchmark/cortext_predictive_ablation_bench`

<table>
<colgroup>
<col style="width: 27%" />
<col style="width: 36%" />
<col style="width: 36%" />
</colgroup>
<thead>
<tr>
<th>study</th>
<th style="text-align: right;">on/off result</th>
<th style="text-align: right;">key metric</th>
</tr>
</thead>
<tbody>
<tr>
<td>predictive ranking</td>
<td style="text-align: right;">on beats off</td>
<td
style="text-align: right;"><code>predictive_target_top1_on = 1</code>,
<code>predictive_target_top1_off = 0</code></td>
</tr>
<tr>
<td>transient decay</td>
<td style="text-align: right;">pass</td>
<td
style="text-align: right;"><code>preactivation_after_decay = 0.4</code></td>
</tr>
<tr>
<td>surprise refresh</td>
<td style="text-align: right;">pass</td>
<td
style="text-align: right;"><code>surprise_refresh_low = 0.0144</code>,
<code>surprise_refresh_high = 0.125963</code></td>
</tr>
</tbody>
</table>

**Metacognitive deterministic ablation:**
`./build/examples/benchmark/cortext_metacognitive_ablation_bench`

<table>
<colgroup>
<col style="width: 27%" />
<col style="width: 36%" />
<col style="width: 36%" />
</colgroup>
<thead>
<tr>
<th>study</th>
<th style="text-align: right;">on/off result</th>
<th style="text-align: right;">key metric</th>
</tr>
</thead>
<tbody>
<tr>
<td>TOT recovery</td>
<td style="text-align: right;">on beats off</td>
<td style="text-align: right;"><code>tot_recovery_hits_on = 1</code>,
<code>tot_recovery_hits_off = 0</code></td>
</tr>
<tr>
<td>unknown caution</td>
<td style="text-align: right;">on beats off</td>
<td style="text-align: right;"><code>unknown_empty_on = 1</code>,
<code>unknown_empty_off = 0</code></td>
</tr>
<tr>
<td>confidence decay</td>
<td style="text-align: right;">on beats off</td>
<td
style="text-align: right;"><code>confidence_decay_on = 0.326581</code>,
<code>confidence_decay_off = 0.8032</code></td>
</tr>
<tr>
<td>decayed TOT reach</td>
<td style="text-align: right;">decay suppresses stale deep recovery</td>
<td style="text-align: right;"><code>tot_decay_hit_on = 0</code>,
<code>tot_decay_hit_off = 1</code></td>
</tr>
<tr>
<td>mode expiry</td>
<td style="text-align: right;">pass</td>
<td style="text-align: right;"><code>mode_expiry_ok = 1</code></td>
</tr>
</tbody>
</table>

**Short live-model confidence-decay ablation:**
`scripts/run_memory_harness.py` on a 120-turn `TopicalChat` slice with
`EmbeddingGemma/llama.cpp`

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
<th>decay</th>
<th style="text-align: right;">retrieval_avg_candidates</th>
<th style="text-align: right;">retrieval_semantic_overlap_mean</th>
<th style="text-align: right;">interrupt_semantic_overlap_mean</th>
<th
style="text-align: right;">interrupt_context_semantic_overlap_mean</th>
<th style="text-align: right;">memory_retrieved_count_mean</th>
<th style="text-align: right;">interrupt_recall</th>
</tr>
</thead>
<tbody>
<tr>
<td>on</td>
<td style="text-align: right;">1.6875</td>
<td style="text-align: right;">0.772989</td>
<td style="text-align: right;">0.772989</td>
<td style="text-align: right;">0.903594</td>
<td style="text-align: right;">0.900000</td>
<td style="text-align: right;">1.0</td>
</tr>
<tr>
<td>off (<code>CORTEXT_DISABLE_METACOG_CONFIDENCE_DECAY=1</code>)</td>
<td style="text-align: right;">1.6250</td>
<td style="text-align: right;">0.769673</td>
<td style="text-align: right;">0.769673</td>
<td style="text-align: right;">0.901935</td>
<td style="text-align: right;">0.866667</td>
<td style="text-align: right;">1.0</td>
</tr>
</tbody>
</table>

Logs:

-   `logs/metacog_conf_decay_on_20260404`
-   `logs/metacog_conf_decay_off_20260404`

**Targeted test coverage:**
`./build/tests/cortext_tests "[operations][predictive]"` passed with
**21 assertions in 5 test cases**;
`./build/tests/cortext_tests "[operations][metacognitive]"` passed with
**31 assertions in 11 test cases**; and the retrieval-side metacognitive
integration slice
`./build/tests/cortext_tests "[operations][graph][metacognitive]"`
passed with **6 assertions in 3 test cases**.

**Observation:** These results are intentionally narrower than the
corpus reruns above. They do not claim that predictive pre-activation or
metacognition produce large gains on every naturalistic dataset. They do
establish the more basic scientific point that these gears are now real
causal parts of the retrieval system: predictive state can change
ranking, TOT can extend recovery reach on the next retrieval window,
unknown-caution can suppress low-confidence fallback into an empty
return when certainty is not met, and confidence decay now prevents
stale high-confidence states from overdriving recovery after long
delays. On a short clean `TopicalChat` slice the live-model effect is
modest but directionally positive.

### Constructive Recall Activation Audit (Apr 4, 2026)

Constructive recall had been one of the last places where the paper was
ahead of the branch: the manuscript described an
evidence-plus-reconstruction ledger, but the earlier implementation
still behaved like a single mutable embedding. We corrected that by
adding a versioned `memory_reconstructions` ledger, seeding an initial
reconstruction on write, making retrieval score against the latest
reconstruction when present, and making reconsolidation append a new
reconstruction instead of overwriting the base evidence embedding.

**Deterministic constructive-recall ablation:**
`./build/examples/benchmark/cortext_constructive_recall_ablation_bench`

<table>
<colgroup>
<col style="width: 27%" />
<col style="width: 36%" />
<col style="width: 36%" />
</colgroup>
<thead>
<tr>
<th>study</th>
<th style="text-align: right;">result</th>
<th style="text-align: right;">key metric</th>
</tr>
</thead>
<tbody>
<tr>
<td>initial ledger seeding</td>
<td style="text-align: right;">pass</td>
<td
style="text-align: right;"><code>constructive_initial_seeded = 1</code></td>
</tr>
<tr>
<td>retrieval reconstruction affects rank</td>
<td style="text-align: right;">on beats off</td>
<td
style="text-align: right;"><code>constructive_target_top1_on = 1</code>,
<code>constructive_target_top1_off = 0</code></td>
</tr>
<tr>
<td>retrieval appends uncertainty-tracked version</td>
<td style="text-align: right;">pass</td>
<td
style="text-align: right;"><code>constructive_recon_rows_on = 2</code>,
<code>constructive_recon_rows_off = 1</code>,
<code>constructive_latest_uncertainty = 0.425</code></td>
</tr>
<tr>
<td>reconsolidation preserves evidence row</td>
<td style="text-align: right;">pass</td>
<td
style="text-align: right;"><code>constructive_recon_preserves_evidence = 1</code>,
<code>constructive_recon_versions = 2</code></td>
</tr>
<tr>
<td>reconsolidation moves current view toward context</td>
<td style="text-align: right;">pass</td>
<td
style="text-align: right;"><code>constructive_recon_current_sim = 0.469205</code>,
<code>constructive_evidence_current_sim = 0.400006</code></td>
</tr>
</tbody>
</table>

**Targeted test coverage:**
`./build/tests/cortext_tests "[operations][constructive_recall]"` passed
with **29 assertions in 3 test cases**. Adjacent storage and migration
regression slices also remained green:
`./build/tests/cortext_tests "[operations][memory_storage]"` passed with
**33 assertions in 4 test cases**, and
`./build/tests/cortext_tests "[schema][migration]"` passed with **11
assertions in 3 test cases**.

**Observation:** This is the constructive-recall equivalent of the
predictive/TOT activation audit above. The result is not a claim of
dramatic corpus-level gains; it is a claim that the branch now actually
implements the algorithm the paper describes. The current memory view
can diverge from the original evidence through an uncertainty-tracked
reconstruction ledger, retrieval can prefer that reconstructed view when
it is more relevant, and reconsolidation no longer destroys the base
evidence embedding in place.

### Neuromodulator Downstream Activation Audit (Apr 4, 2026)

The neuromodulator layer had been partly real and partly aspirational:
`ACh`, `NE`, and `DA` already drove encode/retrieve bias, but the paper
also claimed four downstream effects that were not all behaviorally
wired. We closed that gap without changing the public API by applying
the latent neuromodulator scales inside the existing write-gate,
retrieval-competition, reconsolidation, and procedural-value-update
operations.

**Deterministic neuromodulator ablation:**
`./build/examples/benchmark/cortext_neuromodulator_ablation_bench`

<table>
<colgroup>
<col style="width: 27%" />
<col style="width: 36%" />
<col style="width: 36%" />
</colgroup>
<thead>
<tr>
<th>pathway</th>
<th style="text-align: right;">result</th>
<th style="text-align: right;">key metric</th>
</tr>
</thead>
<tbody>
<tr>
<td>NE write-threshold scaling</td>
<td style="text-align: right;">on beats off</td>
<td
style="text-align: right;"><code>neuromod_write_accept_on = 1</code>,
<code>neuromod_write_accept_off = 0</code></td>
</tr>
<tr>
<td>NE competition scaling</td>
<td style="text-align: right;">on beats off</td>
<td
style="text-align: right;"><code>neuromod_competition_loser_strength_on = 0.998705</code>,
<code>neuromod_competition_loser_strength_off = 0.999136</code></td>
</tr>
<tr>
<td>ACh reconsolidation scaling</td>
<td style="text-align: right;">on beats off</td>
<td
style="text-align: right;"><code>neuromod_recon_similarity_on = 0.832729</code>,
<code>neuromod_recon_similarity_off = 0.808152</code></td>
</tr>
<tr>
<td>DA value-update scaling</td>
<td style="text-align: right;">high beats low</td>
<td
style="text-align: right;"><code>neuromod_value_update_high_da = 0.8</code>,
<code>neuromod_value_update_low_da = 0.4</code>,
<code>neuromod_value_update_disabled = 0.8</code></td>
</tr>
</tbody>
</table>

The benchmark passed end to end: `neuromodulator_bench_passed = 1`.

**Targeted test coverage:** the downstream slices each have a direct
operation-level check.
`./build/tests/cortext_tests "[operations][write_gate][neuromod]"`,
`./build/tests/cortext_tests "[operations][competition][neuromod]"`,
`./build/tests/cortext_tests "[operations][recon][neuromod]"`, and
`./build/tests/cortext_tests "[operations][detect_memory_usage][neuromod]"`
all passed on the rebuilt branch.

**Observation:** This is an activation audit, not a broad
corpus-performance claim. The important scientific correction is
narrower: the paper can now honestly say that the latent neuromodulator
layer does more than set encode/retrieve bias. High `NE` now lowers the
write threshold and sharpens lateral inhibition, high `ACh` increases
reconstruction drift toward current context, and high `DA` strengthens
procedural reward updates.

### Procedural Retrieval Activation Audit (Apr 4, 2026)

The earlier manuscript language overclaimed the procedural lane as a
direct `Q(proc_key, action)` policy. The actual branch stores
`Q(proc_key, memory_id)`, so we tightened the implementation and the
wording together: procedural memory now acts as a **proactive retrieval
lane** that can surface a previously successful routine memory even when
it falls outside the normal semantic KNN seed set.

**Deterministic procedural ablation:**
`./build/examples/benchmark/cortext_procedural_ablation_bench`

<table>
<colgroup>
<col style="width: 27%" />
<col style="width: 36%" />
<col style="width: 36%" />
</colgroup>
<thead>
<tr>
<th>pathway</th>
<th style="text-align: right;">result</th>
<th style="text-align: right;">key metric</th>
</tr>
</thead>
<tbody>
<tr>
<td>proactive routine surfacing</td>
<td style="text-align: right;">on beats off</td>
<td
style="text-align: right;"><code>procedural_target_selected_on = 1</code>,
<code>procedural_target_selected_off = 0</code></td>
</tr>
<tr>
<td>routine ranking after surfacing</td>
<td style="text-align: right;">on beats off</td>
<td
style="text-align: right;"><code>procedural_target_top1_on = 1</code>,
<code>procedural_target_top1_off = 0</code></td>
</tr>
<tr>
<td>learned habit signal preserved</td>
<td style="text-align: right;">pass</td>
<td
style="text-align: right;"><code>procedural_target_proc_score_on = 1.0</code></td>
</tr>
</tbody>
</table>

The benchmark passed end to end: `procedural_bench_passed = 1`.

**Targeted test coverage:** the retrieval-side activation is covered by
`./build/tests/cortext_tests "[operations][graph][procedural]"`, and the
reward-update side remains covered by
`./build/tests/cortext_tests "[operations][detect_memory_usage][neuromod]"`.

**Observation:** This result explains why the older TopicalChat
procedural ablations above were hard to interpret. On generic chat
slices, the old pathway was mostly just a mild reranking term attached
to already-selected semantic candidates. The current branch now has a
true causal procedural lane: a strong learned routine can enter
retrieval proactively for the same sparse context and then compete
normally with other candidates. That is still narrower than a standalone
action policy, so the paper now describes it that way.

### Meta-Learning Activation Audit (Apr 4, 2026)

The paper had claimed that the knob-to-parameter maps were learnable,
but the branch still treated the relevant priors as fixed formulas. We
closed that gap by adding an internal `meta_learning_coeffs` table and
an end-of-turn learner that updates the generic sigmoid-plus-lerp map
for three live prior families:

-   Focus `attention_width_prior`
-   Sensitivity `rate_target_prior`
-   Stability `hysteresis_band_prior`

The implementation keeps the old formulas as exact cold-start priors and
only begins learning once real write/retrieval/reward outcomes exist.

**Deterministic meta-learning ablation:**
`./build/examples/benchmark/cortext_meta_learning_ablation_bench`

<table>
<colgroup>
<col style="width: 27%" />
<col style="width: 36%" />
<col style="width: 36%" />
</colgroup>
<thead>
<tr>
<th>pathway</th>
<th style="text-align: right;">result</th>
<th style="text-align: right;">key metric</th>
</tr>
</thead>
<tbody>
<tr>
<td>focus prior consolidation</td>
<td style="text-align: right;">on narrows vs off</td>
<td
style="text-align: right;"><code>meta_learning_attention_width_prior_on = 1.59894</code>,
<code>meta_learning_attention_width_prior_off = 1.72788</code></td>
</tr>
<tr>
<td>sensitivity rate-target learning</td>
<td style="text-align: right;">on raises vs off</td>
<td
style="text-align: right;"><code>meta_learning_rate_target_prior_on = 2.36446</code>,
<code>meta_learning_rate_target_prior_off = 2.12</code></td>
</tr>
<tr>
<td>stability hysteresis learning</td>
<td style="text-align: right;">on raises vs off</td>
<td
style="text-align: right;"><code>meta_learning_hysteresis_band_prior_on = 0.142751</code>,
<code>meta_learning_hysteresis_band_prior_off = 0.135</code></td>
</tr>
<tr>
<td>coefficient persistence</td>
<td style="text-align: right;">on writes rows, off does not</td>
<td
style="text-align: right;"><code>meta_learning_update_count_on = 24</code>,
<code>meta_learning_update_count_off = 0</code></td>
</tr>
</tbody>
</table>

**Targeted test coverage:**
`./build/tests/cortext_tests "[operations][meta_learning]"` passed with
**16 assertions in 3 test cases**.

**Live-model short ablation:** `logs/meta_learning_live_on_20260404_v3`
vs `logs/meta_learning_live_off_20260404_v3`

These runs use the rebuilt `topical_chat_analysis` binary, a single
**120-turn** TopicalChat slice at **F=S=T=0.5**, and
`EmbeddingGemma/llama.cpp` as recorded in `run.log`.

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
<th>meta_learning</th>
<th style="text-align: right;">retrieval_avg_candidates</th>
<th style="text-align: right;">retrieval_semantic_overlap_mean</th>
<th style="text-align: right;">interrupt_semantic_overlap_mean</th>
<th
style="text-align: right;">interrupt_context_semantic_overlap_mean</th>
<th style="text-align: right;">memory_retrieved_count_mean</th>
<th style="text-align: right;">interrupt_recall</th>
</tr>
</thead>
<tbody>
<tr>
<td>on</td>
<td style="text-align: right;">1.8000</td>
<td style="text-align: right;">0.796255</td>
<td style="text-align: right;">0.791804</td>
<td style="text-align: right;">0.902245</td>
<td style="text-align: right;">0.900000</td>
<td style="text-align: right;">0.933333</td>
</tr>
<tr>
<td>off</td>
<td style="text-align: right;">1.7333</td>
<td style="text-align: right;">0.792718</td>
<td style="text-align: right;">0.788015</td>
<td style="text-align: right;">0.900349</td>
<td style="text-align: right;">0.866667</td>
<td style="text-align: right;">0.933333</td>
</tr>
</tbody>
</table>

The learned rows after the `on` run show where adaptation actually
concentrated:

<table>
<colgroup>
<col style="width: 30%" />
<col style="width: 30%" />
<col style="width: 40%" />
</colgroup>
<thead>
<tr>
<th>family</th>
<th>learned row snapshot</th>
<th style="text-align: right;">update_count</th>
</tr>
</thead>
<tbody>
<tr>
<td>focus</td>
<td><code>alpha_f=5.99997</code>, <code>beta=-3.00007</code>,
<code>a=1.70857</code>, <code>b=1.73685</code></td>
<td style="text-align: right;">119</td>
</tr>
<tr>
<td>sensitivity</td>
<td><code>alpha_s=6.00342</code>, <code>beta=-2.99316</code>,
<code>a=2.59891</code>, <code>b=2.64975</code></td>
<td style="text-align: right;">119</td>
</tr>
<tr>
<td>stability</td>
<td><code>alpha_t=6.0</code>, <code>beta=-3.0</code>,
<code>a=0.13385</code>, <code>b=0.13615</code></td>
<td style="text-align: right;">119</td>
</tr>
</tbody>
</table>

**Observation:** The honest live-model result is modest, not dramatic.
On this short topical-chat slice, meta-learning nudges retrieval quality
in the expected direction and slightly increases candidate recovery, but
it does not change headline interrupt precision/recall. Most of the
short-horizon adaptation lands in the Sensitivity family
(`rate_target_prior` shifting upward from the legacy `2.12` baseline
toward `≈2.62`), while Focus narrows only slightly and Stability barely
moves. That is still a meaningful correction to the earlier paper claim:
the maps are now truly learnable and behaviorally active, but their
short-horizon impact is limited on clean chat data.

### Offline Semantic Label Classifier Prototype

Before the runtime labeling corrections below, the branch relied on an
overly aggressive repetition-gated label path. To test whether a more
semantic, memory-like label pathway is viable before fully replacing the
deployed path, we implemented an **offline prototype** that trains two
small classifiers over candidate spans:

-   **span typing:** `identity`, `person_entity`, `place`,
    `org_project`, `topic`, `state`, `none`
-   **promotion:** `durable`, `provisional`, `ignore`

The prototype uses:

-   `english-wordnet-2025.xml` for concept priors
-   multilingual given-name and surname priors built from
    `data/surnames/{forenames,surnames}.csv` plus U.S. Census surnames
    through `scripts/build_name_priors.py`
-   sentence/lexical context features
-   optional external text embeddings through the rebuilt
    `cortext_text_embedder`, which now resolves the repo’s preferred
    text encoder (`EmbeddingGemma/llama.cpp` on this machine)

#### Real-data study with multilingual priors (Apr 5, 2026)

Run root: `logs/label_classifier_realdata_20260405`

Training data:

-   multilingual name priors from the Kaggle
    `data/surnames/{forenames,surnames}.csv` tables, supplemented with
    U.S. Census surnames
-   real labeled entity spans from `WNUT17` (`train/dev/test`)
-   weakly labeled sentences mined from prepared `TopicalChat`,
    `EmpatheticDialogues`, and `Ubuntu Dialogue` exports
-   real encoder features via the rebuilt `cortext_text_embedder` on
    `EmbeddingGemma/llama.cpp`

The resulting multilingual prior file is large (`≈383 MB`) because this
run keeps the full aggregated name tables instead of pruning to a tiny
seed list. The train/validation split contained `16,284` / `3,815`
examples after mixing `WNUT17` with weak Cortext-style chat spans.

<table>
<thead>
<tr>
<th>head</th>
<th style="text-align: right;">macro_f1</th>
<th style="text-align: right;">supported_macro_f1</th>
<th style="text-align: right;">accuracy</th>
<th style="text-align: right;">candidate_recall</th>
</tr>
</thead>
<tbody>
<tr>
<td>span typing</td>
<td style="text-align: right;"><strong>0.302</strong></td>
<td style="text-align: right;"><strong>0.528</strong></td>
<td style="text-align: right;"><strong>0.724</strong></td>
<td style="text-align: right;"><strong>0.971</strong></td>
</tr>
<tr>
<td>promotion</td>
<td style="text-align: right;"><strong>0.487</strong></td>
<td style="text-align: right;"><strong>0.730</strong></td>
<td style="text-align: right;"><strong>0.776</strong></td>
<td style="text-align: right;"><strong>0.971</strong></td>
</tr>
</tbody>
</table>

The distinction between `macro_f1` and `supported_macro_f1` matters
here: `WNUT17` only labels entity-like spans plus negatives, so
`identity`, `topic`, and `state` have zero support in the held-out test
split. The supported score is therefore the honest headline metric for
the real-dataset run.

#### Secondary curated smoke (Apr 5, 2026)

Run root: `logs/label_classifier_realdata_20260405/eval_curated`

To preserve one small full-taxonomy check, we also evaluated the same
trained model on the curated span fixture in
`data/label_classifier/gold.jsonl`. This is no longer the
source-of-record experiment; it is a compact smoke that highlights
coverage gaps outside the WNUT entity taxonomy.

<table>
<thead>
<tr>
<th>head</th>
<th style="text-align: right;">macro_f1</th>
<th style="text-align: right;">supported_macro_f1</th>
<th style="text-align: right;">accuracy</th>
<th style="text-align: right;">candidate_recall</th>
</tr>
</thead>
<tbody>
<tr>
<td>span typing</td>
<td style="text-align: right;"><strong>0.341</strong></td>
<td style="text-align: right;"><strong>0.341</strong></td>
<td style="text-align: right;"><strong>0.433</strong></td>
<td style="text-align: right;"><strong>1.000</strong></td>
</tr>
<tr>
<td>promotion</td>
<td style="text-align: right;"><strong>0.439</strong></td>
<td style="text-align: right;"><strong>0.439</strong></td>
<td style="text-align: right;"><strong>0.633</strong></td>
<td style="text-align: right;"><strong>1.000</strong></td>
</tr>
</tbody>
</table>

**Observations:** The prototype has moved from “toy feasibility” to a
real offline experiment. On the held-out `WNUT17` test split, span
typing reaches `0.528` supported macro F1 and promotion reaches `0.730`,
with candidate recall still at `0.971`. That is strong enough to justify
replacing the current exact-string repetition story at the research
level. The curated smoke is now more useful as a failure probe than as a
headline metric: the same model underperforms on `state` and `topic`,
which tells us the current training mix is still too entity-heavy. So
the bottleneck is no longer whether a semantic label classifier can
train; it is whether we can broaden the real conversational supervision
enough to cover non-entity concepts before runtime integration.

### Online Label Persistence Ablation (Apr 5, 2026)

We also corrected the **deployed** label path itself. The previous
runtime policy applied a batch-level repetition threshold and, when
labels were too sparse to clear it, collapsed each extraction result to
a single highest-salience fallback label. This was the direct cause of
pathological under-labeling in long chats, including name loss and label
graphs with implausibly few nodes.

The current runtime now persists extracted labels directly from each
extraction result, deduped by normalized key within the result and
reused by key across the store. To measure the impact cleanly, we added
a deterministic ablation bench that re-enables the legacy gate through
an internal study-only toggle:

**Deterministic label-persistence ablation:**
`./build/examples/benchmark/cortext_label_persistence_ablation_bench`

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 28%" />
<col style="width: 28%" />
<col style="width: 21%" />
</colgroup>
<thead>
<tr>
<th>scenario</th>
<th style="text-align: right;">legacy gate</th>
<th style="text-align: right;">current runtime</th>
<th>result</th>
</tr>
</thead>
<tbody>
<tr>
<td>sparse three-label extraction</td>
<td style="text-align: right;"><code>1</code> label / <code>1</code>
<code>has_label</code> edge</td>
<td style="text-align: right;"><code>3</code> labels / <code>3</code>
<code>has_label</code> edges</td>
<td>current preserves all extracted labels</td>
</tr>
<tr>
<td>repeated single-label extraction</td>
<td style="text-align: right;"><code>1</code> label</td>
<td style="text-align: right;"><code>1</code> label</td>
<td>dedupe remains stable</td>
</tr>
</tbody>
</table>

Recorded metrics:

-   `label_sparse_legacy_count = 1`
-   `label_sparse_current_count = 3`
-   `label_sparse_legacy_edges = 1`
-   `label_sparse_current_edges = 3`
-   `label_repeated_legacy_count = 1`
-   `label_repeated_current_count = 1`

**Targeted test coverage:**
`./build/tests/cortext_tests "[operations][extraction][labels]"` passed
with **8 assertions in 2 test cases**.

**Observation:** This change is not about adding more decorative labels.
It removes a lossy gate that was erasing semantically important one-shot
entities before they could ever become part of the graph. The ablation
shows the intended behavior clearly: sparse extractions now survive
intact, while repeated identical labels still collapse to one durable
node as before.

### LFM2.5-350M Extractor Prompt Ablation

Once sparse labels were allowed through, the next failure mode was
obvious in the live chat DB: the fast local extractor was emitting junk
labels like `user`, `assistant`, `support`, `documents`, and even
schema-style tokens in contexts where the desired durable nodes were
things like `Gabriel`, `Chicago`, `SQLite migration`, `Cortext`,
`Sarah`, or `housing application`. Before paying the latency cost of a
larger extraction model, we first tested whether prompt technique alone
could recover better labels from the small `LFM2.5-350M` GGUF.

We compared five prompt styles on the live local extractor:

-   `baseline`: original recall-heavy zero-shot prompt
-   `durable`: zero-shot durable-node wording
-   `precision`: zero-shot precision-first wording
-   `fewshot_durable`: durable-node wording plus short conversational
    examples
-   `fewshot_precision`: precision-first wording plus short
    conversational examples

The benchmark uses three short multi-turn fixtures that mirror the
failure cases we saw in the chat DB:

-   interview-prep noise (`interview`, `confidence`)
-   identity plus work context (`Gabriel`, `Chicago`,
    `SQLite migration`, `Cortext`)
-   neighbor plus housing context (`Sarah`, `housing application`,
    `Logan Square`)

Each style is scored by desired-label hits, banned-label leakage, total
emitted labels, overflow beyond a compact durable-label budget, and
wall-clock latency.

**Deterministic prompt-technique ablation:**
`./build/examples/benchmark/cortext_extractor_prompt_ablation_bench`

Run log:
`logs/extractor_prompt_ablation_20260406/extractor_prompt_ablation.log`

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
<th>prompt style</th>
<th style="text-align: right;">desired hits</th>
<th style="text-align: right;">banned hits</th>
<th style="text-align: right;">labels emitted</th>
<th style="text-align: right;">latency (ms)</th>
<th style="text-align: right;">score</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>baseline</code></td>
<td style="text-align: right;"><code>6</code></td>
<td style="text-align: right;"><code>2</code></td>
<td style="text-align: right;"><code>13</code></td>
<td style="text-align: right;"><code>3116</code></td>
<td style="text-align: right;"><code>14</code></td>
</tr>
<tr>
<td><code>durable</code></td>
<td style="text-align: right;"><code>8</code></td>
<td style="text-align: right;"><code>7</code></td>
<td style="text-align: right;"><code>86</code></td>
<td style="text-align: right;"><code>9794</code></td>
<td style="text-align: right;"><code>-52</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td style="text-align: right;"><code>7</code></td>
<td style="text-align: right;"><code>1</code></td>
<td style="text-align: right;"><code>46</code></td>
<td style="text-align: right;"><code>4898</code></td>
<td style="text-align: right;"><code>-3</code></td>
</tr>
<tr>
<td><code>fewshot_durable</code></td>
<td style="text-align: right;"><code>8</code></td>
<td style="text-align: right;"><code>4</code></td>
<td style="text-align: right;"><code>16</code></td>
<td style="text-align: right;"><code>3856</code></td>
<td style="text-align: right;"><strong><code>16</code></strong></td>
</tr>
<tr>
<td><code>fewshot_precision</code></td>
<td style="text-align: right;"><code>5</code></td>
<td style="text-align: right;"><code>2</code></td>
<td style="text-align: right;"><code>8</code></td>
<td style="text-align: right;"><code>3073</code></td>
<td style="text-align: right;"><code>11</code></td>
</tr>
</tbody>
</table>

Representative outputs:

-   `baseline` interview fixture:
    `confidence, prep, support, interview, user, assistant`
-   `fewshot_durable` identity/work fixture:
    `Gabriel, Chicago, SQLite migration, Cortext`
-   `fewshot_precision` neighbor/housing fixture:
    `Sarah, housing application, Logan Square`

**Observation:** The zero-shot rewrites alone were not enough. `durable`
and `precision` both leaked their own instruction vocabulary back into
the labels, including prompt words like `schema`, `timestamps`,
`people`, and `projects`. The few-shot variants changed the behavior
qualitatively. `fewshot_durable` was the best overall compromise: it
recovered all four identity/work labels in the hardest fixture,
preserved the full `Sarah` / `housing application` / `Logan Square`
trio, and did so with only a modest latency increase over the old
baseline (`3856 ms` vs `3116 ms` across the full three-fixture run).
`fewshot_precision` was the leanest style, but it under-recalled the
identity/work fixture too aggressively. Based on this ablation, the
current branch now uses the **few-shot durable prompt as the deployed
default** for the local `LFM2.5-350M` extraction path, while still
allowing overrides through `CORTEXT_LFM2_EXTRACT_PROMPT_STYLE` for
further study.

### LFM2.5-350M Summarizer Prompt Ablation

Once extraction quality improved, the next weak point was the
**summarizer**: the fast local `LFM2.5-350M` summarization path was
still vulnerable to prompt echo, speaker-role leakage, and a strong bias
toward the **last excerpt** in a cluster. That showed up directly in
earlier deep-consolidation failures: summaries would sometimes repeat
prompt language, narrate the conversation mechanics, or ignore early
identity facts like `Gabriel` / `Chicago` in favor of the final turn.

Before paying the cost of a larger summarizer, we ran a prompt-technique
ablation on the live local `LFM2.5-350M` GGUF. We compared six prompt
styles:

-   `baseline`: original excerpt-based zero-shot prompt
-   `durable`: excerpt-based zero-shot durable-memory wording
-   `precision`: excerpt-based zero-shot precision-first wording
-   `fewshot_durable`: excerpt-based few-shot durable wording
-   `fewshot_precision`: excerpt-based few-shot precision wording
-   `transcript_fewshot`: few-shot prompt plus a **chat transcript**
    input format instead of numbered excerpts

The benchmark uses three short conversational fixtures that stress the
failure modes we care about:

-   identity plus work context (`Gabriel`, `Chicago`,
    `SQLite migration`, `Cortext`, `demo`)
-   project plus operations context (`Alice`, `Acme pilot`, `Seattle`,
    `SSO`, `audit exports`, `June`, `renewal forecast`)
-   neighbor plus housing context (`Sarah`, `housing application`,
    `Logan Square`, `pay stubs`, `Tuesday`)

Each style is scored by desired-fact hits, banned meta leakage (`User:`,
`Assistant:`, `Excerpt`, `Final summary`, prompt echo), and wall-clock
latency.

**Deterministic prompt-technique ablation:**
`./build/examples/benchmark/cortext_summarizer_prompt_ablation_bench`

Run log:
`logs/summarizer_prompt_ablation_20260406/summarizer_prompt_ablation.log`

<table>
<thead>
<tr>
<th>prompt style</th>
<th style="text-align: right;">desired hits</th>
<th style="text-align: right;">banned hits</th>
<th style="text-align: right;">latency (ms)</th>
<th style="text-align: right;">score</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>baseline</code></td>
<td style="text-align: right;"><code>7</code></td>
<td style="text-align: right;"><code>4</code></td>
<td style="text-align: right;"><code>2404</code></td>
<td style="text-align: right;"><code>2</code></td>
</tr>
<tr>
<td><code>durable</code></td>
<td style="text-align: right;"><code>1</code></td>
<td style="text-align: right;"><code>3</code></td>
<td style="text-align: right;"><code>1358</code></td>
<td style="text-align: right;"><code>-7</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td style="text-align: right;"><code>5</code></td>
<td style="text-align: right;"><code>2</code></td>
<td style="text-align: right;"><code>1216</code></td>
<td style="text-align: right;"><code>4</code></td>
</tr>
<tr>
<td><code>fewshot_durable</code></td>
<td style="text-align: right;"><code>1</code></td>
<td style="text-align: right;"><code>3</code></td>
<td style="text-align: right;"><code>2033</code></td>
<td style="text-align: right;"><code>-7</code></td>
</tr>
<tr>
<td><code>fewshot_precision</code></td>
<td style="text-align: right;"><code>6</code></td>
<td style="text-align: right;"><code>0</code></td>
<td style="text-align: right;"><code>1739</code></td>
<td style="text-align: right;"><code>12</code></td>
</tr>
<tr>
<td><code>transcript_fewshot</code></td>
<td style="text-align: right;"><strong><code>15</code></strong></td>
<td style="text-align: right;"><strong><code>0</code></strong></td>
<td style="text-align: right;"><code>1850</code></td>
<td style="text-align: right;"><strong><code>30</code></strong></td>
</tr>
</tbody>
</table>

Representative outputs:

-   `baseline` identity/work fixture:
    `User: I am debugging a SQLite migration for Cortext before the demo. Assistant: The migration is progressing well...`
-   `fewshot_precision` identity/work fixture:
    `Debugging a SQLite migration for Cortext before the demo.`
-   `transcript_fewshot` identity/work fixture:
    `Gabriel lives in Chicago and is debugging a SQLite migration for Cortext before the demo.`
-   `transcript_fewshot` project/ops fixture:
    `Alice is leading the Acme pilot in Seattle, and the customer needs SSO and audit exports before June.`

**Observation:** The biggest win did **not** come from adding more
wording. It came from changing the input shape. The excerpt-based
prompts remained last-turn-biased: even the better few-shot excerpt
styles still dropped early facts like `Gabriel` and `Chicago`, or
collapsed the whole cluster to the last blocker sentence. The
transcript-style prompt changed the behavior qualitatively. On the live
local `LFM2.5-350M` summarizer it recovered all five desired
identity/work facts, six of seven project/ops facts, and four of five
neighbor/housing facts, with **zero** banned meta hits and only a
moderate latency cost relative to the leanest zero-shot style (`1850 ms`
vs `1216 ms` across the three-fixture run). Based on this ablation, the
current branch now uses the **transcript few-shot prompt as the deployed
default** for the local `LFM2.5-350M` summarization path, while still
allowing overrides through `CORTEXT_LFM2_SUMMARY_PROMPT_STYLE` for
further study.

### Storage-Gated Eviction Ablation (Apr 5, 2026)

We also corrected a practical failure mode in the deployed eviction
policy. The previous runtime would evict weak `LONG_TERM` memories as
soon as they fell below `periphery_cutoff(T)`, even when the backing
chat database was still trivially small. In long user conversations this
produced premature forgetting: names and low-frequency autobiographical
details could decay out of the active store long before disk pressure
justified any deletion.

The current runtime now applies a storage gate before the usual
strength-based eviction query. For file-backed stores, eviction is
blocked until the database footprint reaches a configurable floor. The
floor can be a fixed byte budget, a percentage of currently available
disk space, or both; the active threshold is the maximum of those
values. The default deployed fixed floor is `500 MB`.

**Deterministic storage-budget ablation:**
`./build/examples/benchmark/cortext_storage_eviction_budget_bench`

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 28%" />
<col style="width: 28%" />
<col style="width: 21%" />
</colgroup>
<thead>
<tr>
<th>scenario</th>
<th style="text-align: right;">storage gate floor</th>
<th style="text-align: right;">surviving weak memory count</th>
<th>result</th>
</tr>
</thead>
<tbody>
<tr>
<td>high budget guard</td>
<td style="text-align: right;"><code>1 GiB</code></td>
<td style="text-align: right;"><code>1</code></td>
<td>weak memory is retained because storage budget is not yet full</td>
</tr>
<tr>
<td>zero budget</td>
<td style="text-align: right;"><code>0</code></td>
<td style="text-align: right;"><code>0</code></td>
<td>the same weak memory is evicted once the storage gate is
removed</td>
</tr>
</tbody>
</table>

Recorded metrics:

-   `storage_gate_high_budget_count = 1`
-   `storage_gate_blocks_early_eviction = 1`
-   `storage_gate_zero_budget_count = 0`
-   `storage_gate_allows_eviction_after_limit = 1`

**Targeted test coverage:**
`./build/tests/cortext_tests "[op18][memory_strength][storage_gate]"`
passed with **3 assertions in 2 test cases**.

**Observation:** This is a policy correction, not an attempt to make
eviction disappear. Strength and fact-floor rules still determine
*which* memories are eligible once storage pressure is real. The storage
gate simply prevents the engine from acting like a tiny chat database is
already full.

We then added a second deterministic sweep to answer the practical
sizing question: **how much storage budget is actually enough to delay
forgetting?** This sweep uses a file-backed SQLite store, grows the
database by `5 MB` per simulated step, and records both the
storage-threshold crossing step and the eventual eviction step for the
same weak memory.

**Deterministic storage-threshold survival sweep:**
`./build/examples/benchmark/cortext_storage_eviction_sweep_bench`

<table>
<thead>
<tr>
<th>fixed storage floor</th>
<th style="text-align: right;">threshold-cross step</th>
<th style="text-align: right;">eviction step</th>
<th>result</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>0 MB</code></td>
<td style="text-align: right;"><code>1</code></td>
<td style="text-align: right;"><code>62</code></td>
<td>pure decay-limited baseline</td>
</tr>
<tr>
<td><code>64 MB</code></td>
<td style="text-align: right;"><code>13</code></td>
<td style="text-align: right;"><code>62</code></td>
<td>no practical delay vs baseline</td>
</tr>
<tr>
<td><code>128 MB</code></td>
<td style="text-align: right;"><code>26</code></td>
<td style="text-align: right;"><code>62</code></td>
<td>still decay-limited</td>
</tr>
<tr>
<td><code>256 MB</code></td>
<td style="text-align: right;"><code>51</code></td>
<td style="text-align: right;"><code>62</code></td>
<td>still decay-limited</td>
</tr>
<tr>
<td><code>500 MB</code></td>
<td style="text-align: right;"><code>100</code></td>
<td style="text-align: right;"><code>100</code></td>
<td>first storage-limited regime</td>
</tr>
<tr>
<td><code>1 GB</code></td>
<td style="text-align: right;"><code>200</code></td>
<td style="text-align: right;"><code>200</code></td>
<td>larger-device storage-limited regime</td>
</tr>
<tr>
<td><code>2 GB</code></td>
<td style="text-align: right;"><code>400</code></td>
<td style="text-align: right;"><code>400</code></td>
<td>very large-device storage-limited regime</td>
</tr>
</tbody>
</table>

Recorded metrics:

-   `storage_sweep_0mb_eviction_step = 62`
-   `storage_sweep_64mb_eviction_step = 62`
-   `storage_sweep_128mb_eviction_step = 62`
-   `storage_sweep_256mb_eviction_step = 62`
-   `storage_sweep_500mb_eviction_step = 100`
-   `storage_sweep_500mb_cross_bytes = 501829632`
-   `storage_sweep_default_500mb_delays_vs_64mb = 1`
-   `storage_sweep_1gb_eviction_step = 200`
-   `storage_sweep_2gb_eviction_step = 400`
-   `storage_sweep_1gb_delays_vs_500mb = 1`
-   `storage_sweep_2gb_delays_vs_1gb = 1`

**Observation:** The useful takeaway is not that “more storage is always
better.” It is that there is a real **decay-limited regime** and a real
**storage-limited regime**. In this workload, any budget up to `256 MB`
still lets the memory die on its normal decay curve at step `62`,
because the database reaches those thresholds before the memory becomes
too weak. `500 MB` is the first tested budget that pushes the system
into the storage-limited regime, delaying eviction to step `100`; `1 GB`
and `2 GB` extend that same pattern to steps `200` and `400`. That is
why the default floor was raised well above the earlier `64 MiB`
setting: the smaller floor looked safer on paper than it actually was in
practice.

We then extended the study into an end-to-end steady-state chat regime,
where the relevant question is not just when a weak memory dies, but
whether it survives long enough for the next label-producing
consolidation window to fire. This benchmark uses the same file-backed
`5 MB/step` growth profile, keeps the memory in the
post-initial-consolidation pool, and measures whether a three-label
extraction (`Gabriel`, `Chicago`, `Cortext`) is created before the raw
memory is evicted.

**Deterministic storage × consolidation end-to-end ablation:**
`./build/examples/benchmark/cortext_storage_consolidation_e2e_bench`

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
<th>fixed storage floor</th>
<th style="text-align: right;">raw-memory eviction step</th>
<th style="text-align: right;">every <code>30</code> steps</th>
<th style="text-align: right;">every <code>60</code> steps</th>
<th style="text-align: right;">every <code>90</code> steps</th>
<th style="text-align: right;">every <code>120</code> steps</th>
<th style="text-align: right;">every <code>150</code> steps</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>0 MB</code></td>
<td style="text-align: right;"><code>62</code></td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>0</code> labels</td>
</tr>
<tr>
<td><code>64 MB</code></td>
<td style="text-align: right;"><code>62</code></td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>0</code> labels</td>
</tr>
<tr>
<td><code>128 MB</code></td>
<td style="text-align: right;"><code>62</code></td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>0</code> labels</td>
</tr>
<tr>
<td><code>256 MB</code></td>
<td style="text-align: right;"><code>62</code></td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>0</code> labels</td>
</tr>
<tr>
<td><code>500 MB</code></td>
<td style="text-align: right;"><code>100</code></td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>0</code> labels</td>
</tr>
<tr>
<td><code>1 GB</code></td>
<td style="text-align: right;"><code>200</code></td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>3</code> labels retained</td>
</tr>
<tr>
<td><code>2 GB</code></td>
<td style="text-align: right;"><code>400</code></td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>3</code> labels retained</td>
<td style="text-align: right;"><code>3</code> labels retained</td>
</tr>
</tbody>
</table>

Recorded checks:

-   `storage_cons_no_consolidation_creates_no_labels = 1`
-   `storage_cons_64mb_interval60_captures_labels = 1`
-   `storage_cons_64mb_interval90_misses_labels = 1`
-   `storage_cons_500mb_interval90_captures_labels = 1`
-   `storage_cons_500mb_interval90_delays_vs_64mb = 1`
-   `storage_cons_500mb_interval120_misses_labels = 1`
-   `storage_cons_1gb_interval120_captures_labels = 1`
-   `storage_cons_1gb_interval150_captures_labels = 1`
-   `storage_cons_2gb_interval150_captures_labels = 1`
-   `storage_cons_1gb_interval150_delays_vs_500mb = 1`

**Observation:** This is the strongest practical justification for the
higher default and for device-class scaling above it. Under small
storage budgets, the system still has enough time to consolidate on
short cadences (`30` or `60` steps), but it misses slower
label-formation windows because the raw memory is gone by step `62`.
With a `500 MB` floor, the same memory survives to step `100`, which is
enough for a `90`-step consolidation interval to capture and persist the
labels before the episodic trace is evicted. With `1 GB`, the survival
window extends to step `200`, which is enough for both `120`- and
`150`-step consolidation intervals. `2 GB` does not add new
label-capture wins on this task beyond `1 GB`; it extends the safety
margin further. So the current evidence suggests a useful plateau
structure: `500 MB` is a good default for ordinary chat retention, while
`1 GB` is the first meaningful “slow consolidation / large-memory” tier.

We then replaced the abstract fixed-step consolidation cadence with a
synthetic human sleep clock derived from representative age-group sleep
patterns. Each profile uses one consolidation opportunity per sleep
bout, where the number of bouts per 24 hours is `1 + naps/day`. The
benchmark uses a `24 h` episodic half-life, a synthetic file-backed
storage growth rate of `12 MB/hour`, and asks a more human-like
question: **does a weak autobiographical trace survive long enough to be
consolidated during the next realistic sleep cycle?**

**Deterministic storage × human sleep consolidation ablation:**
`./build/examples/benchmark/cortext_storage_human_sleep_consolidation_bench`

<table>
<colgroup>
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 15%" />
<col style="width: 15%" />
<col style="width: 15%" />
<col style="width: 15%" />
<col style="width: 15%" />
</colgroup>
<thead>
<tr>
<th>age group</th>
<th>sleep pattern used in the synthetic clock</th>
<th style="text-align: right;">effective consolidation cadence</th>
<th style="text-align: right;"><code>0 MB</code> floor</th>
<th style="text-align: right;"><code>256 MB</code> floor</th>
<th style="text-align: right;"><code>500 MB</code> floor</th>
<th style="text-align: right;"><code>1 GB</code> floor</th>
</tr>
</thead>
<tbody>
<tr>
<td>newborn</td>
<td><code>14–17 h</code>, <code>4–6</code> naps/day</td>
<td style="text-align: right;">every <code>4 h</code></td>
<td style="text-align: right;"><code>3</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
</tr>
<tr>
<td>infant</td>
<td><code>12–16 h</code>, <code>2–3</code> naps/day</td>
<td style="text-align: right;">every <code>7 h</code></td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
</tr>
<tr>
<td>toddler</td>
<td><code>11–14 h</code>, <code>1–2</code> naps/day</td>
<td style="text-align: right;">every <code>10 h</code></td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
</tr>
<tr>
<td>preschool</td>
<td><code>10–13 h</code>, <code>0–1</code> naps/day</td>
<td style="text-align: right;">every <code>16 h</code></td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
</tr>
<tr>
<td>school-age</td>
<td><code>9–12 h</code>, <code>0</code> naps/day</td>
<td style="text-align: right;">every <code>24 h</code></td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
</tr>
<tr>
<td>teen</td>
<td><code>8–10 h</code>, <code>0</code> naps/day</td>
<td style="text-align: right;">every <code>24 h</code></td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
</tr>
<tr>
<td>adult</td>
<td><code>7–9 h</code>, optional naps</td>
<td style="text-align: right;">every <code>24 h</code></td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
</tr>
<tr>
<td>older adult</td>
<td><code>7–8 h</code>, optional naps</td>
<td style="text-align: right;">every <code>24 h</code></td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>0</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
<td style="text-align: right;"><code>3</code> labels</td>
</tr>
</tbody>
</table>

Recorded checks:

-   `human_sleep_newborn_zero_budget_captures = 1`
-   `human_sleep_adult_zero_budget_misses = 1`
-   `human_sleep_adult_256mb_misses = 1`
-   `human_sleep_adult_500mb_captures = 1`
-   `human_sleep_adult_1gb_matches_500mb = 1`
-   `human_sleep_older_adult_500mb_captures = 1`

**Observation:** This synthetic-clock view is more interpretable than
the earlier abstract interval sweep. It shows that frequent sleep in
early life behaves like a natural consolidation rescue mechanism:
newborn-style sleep is frequent enough that even a zero storage floor
still captures the labels before the trace dies, while
infant/toddler/preschool profiles already benefit from modest storage
budgets such as `256 MB`. The adult-side boundary is sharper and more
important for real chat systems: once consolidation becomes essentially
once-per-day, `256 MB` is no longer enough, and `500 MB` is the first
tested floor that reliably keeps the trace alive until the nightly
consolidation window. `1 GB` still helps, but in this human-like
benchmark it acts as safety margin rather than unlocking a new retention
class. That is the clearest current justification for the deployed
`500 MB` default.

We also added a matching **estimated token-cost ablation** over the same
synthetic sleep schedules. This study uses the repository’s existing
character-to-token heuristic (`ceil(chars / 4)`) to estimate the prompt
and completion budget for each consolidation session over a `72 h`
synthetic clock with one autobiographical trace arriving per hour. It is
intentionally a cost proxy, not a billing measurement: the goal is to
compare how much more summarization/extraction budget frequent
sleep-style consolidation would consume relative to nightly
consolidation.

**Deterministic storage × human sleep token-cost ablation:**
`./build/examples/benchmark/cortext_storage_human_sleep_token_bench`

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
<th>age group</th>
<th style="text-align: right;">effective consolidation cadence</th>
<th style="text-align: right;">storage floor</th>
<th style="text-align: right;">consolidation events over
<code>72 h</code></th>
<th style="text-align: right;">estimated total tokens</th>
</tr>
</thead>
<tbody>
<tr>
<td>newborn</td>
<td style="text-align: right;">every <code>4 h</code></td>
<td style="text-align: right;"><code>500 MB</code></td>
<td style="text-align: right;"><code>18</code></td>
<td style="text-align: right;"><code>7090</code></td>
</tr>
<tr>
<td>infant</td>
<td style="text-align: right;">every <code>7 h</code></td>
<td style="text-align: right;"><code>500 MB</code></td>
<td style="text-align: right;"><code>10</code></td>
<td style="text-align: right;"><code>5199</code></td>
</tr>
<tr>
<td>toddler</td>
<td style="text-align: right;">every <code>10 h</code></td>
<td style="text-align: right;"><code>500 MB</code></td>
<td style="text-align: right;"><code>7</code></td>
<td style="text-align: right;"><code>4520</code></td>
</tr>
<tr>
<td>preschool</td>
<td style="text-align: right;">every <code>16 h</code></td>
<td style="text-align: right;"><code>500 MB</code></td>
<td style="text-align: right;"><code>4</code></td>
<td style="text-align: right;"><code>3590</code></td>
</tr>
<tr>
<td>adult</td>
<td style="text-align: right;">every <code>24 h</code></td>
<td style="text-align: right;"><code>0 MB</code></td>
<td style="text-align: right;"><code>3</code></td>
<td style="text-align: right;"><code>3700</code></td>
</tr>
<tr>
<td>adult</td>
<td style="text-align: right;">every <code>24 h</code></td>
<td style="text-align: right;"><code>500 MB</code></td>
<td style="text-align: right;"><code>3</code></td>
<td style="text-align: right;"><code>3700</code></td>
</tr>
<tr>
<td>adult</td>
<td style="text-align: right;">every <code>24 h</code></td>
<td style="text-align: right;"><code>1 GB</code></td>
<td style="text-align: right;"><code>3</code></td>
<td style="text-align: right;"><code>3700</code></td>
</tr>
</tbody>
</table>

Recorded checks:

-   `human_sleep_tokens_newborn_cost_exceeds_adult_500mb = 1`
-   `human_sleep_tokens_newborn_events_exceed_adult_500mb = 1`
-   `human_sleep_tokens_adult_500mb_cost_matches_0mb = 1`
-   `human_sleep_tokens_adult_1gb_matches_500mb_retention = 1`
-   `human_sleep_tokens_adult_1gb_matches_500mb_cost = 1`

**Observation:** In this steady hourly-trace workload, token cost is
dominated by **consolidation frequency**, not by the storage floor.
Newborn-style frequent sleep performs roughly `1.9×` the estimated token
work of adult nightly consolidation over the same `72 h` horizon (`7090`
vs `3700` tokens), because it runs far more consolidation sessions even
though each session is smaller. By contrast, once the workload is
already surviving into the nightly consolidation loop, raising the
storage floor from `0 MB` to `500 MB` or `1 GB` does not materially
change the estimated token budget in this proxy study. That does **not**
mean storage is irrelevant; the human-retention ablation above showed
that storage floor is still decisive for whether fragile traces survive
long enough to reach a nightly consolidation window at all. The joint
takeaway is that cadence drives cost, while storage floor determines
whether slower cadences remain viable.

### Cross-Session Surfacing vs Strength-Only Decay Ablation (Apr 6, 2026)

The remaining practical question was narrower than the storage-gate
policy itself: **if a memory is still physically present, does
per-memory strength decay by itself make Cortext stop surfacing it
between sessions?** The existing eviction studies did not answer that
directly, because they treated surfacing failure and physical deletion
as the same outcome. We therefore added a deterministic retrieval
ablation that holds the retrieval fixture constant while independently
toggling (a) whether strength decays and (b) whether eviction is allowed
to delete the trace.

**Deterministic cross-session surfacing ablation:**
`./build/examples/benchmark/cortext_decay_surface_ablation_bench`

Setup:

-   one target autobiographical episodic memory
-   one semantically nearby distractor and one irrelevant memory
-   two query forms: an exact-near query and a paraphrase-near query
-   three conditions:
    -   `decay_eviction_allowed`: effective half-life forced to `24 h`,
        storage floor `0`
    -   `decay_eviction_blocked`: same `24 h` half-life, but storage
        floor forced high enough to block deletion
    -   `frozen_eviction_blocked`: effectively frozen half-life (`3650`
        days), same blocked-eviction setting
-   fixed retrieval config: `F=S=T=0.5`

Key results:

<table>
<colgroup>
<col style="width: 8%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
</colgroup>
<thead>
<tr>
<th>condition</th>
<th style="text-align: right;">target exists at <code>24 h</code></th>
<th style="text-align: right;">target strength at <code>24 h</code></th>
<th style="text-align: right;">exact top-1 at <code>24 h</code></th>
<th style="text-align: right;">paraphrase top-1 at
<code>24 h</code></th>
<th style="text-align: right;">target exists at <code>168 h</code></th>
<th style="text-align: right;">target strength at
<code>168 h</code></th>
<th style="text-align: right;">exact top-1 at <code>168 h</code></th>
<th style="text-align: right;">paraphrase top-1 at
<code>168 h</code></th>
</tr>
</thead>
<tbody>
<tr>
<td><code>decay_eviction_allowed</code></td>
<td style="text-align: right;"><code>0</code></td>
<td style="text-align: right;"><code>0.000000</code></td>
<td style="text-align: right;"><code>0</code></td>
<td style="text-align: right;"><code>0</code></td>
<td style="text-align: right;"><code>0</code></td>
<td style="text-align: right;"><code>0.000000</code></td>
<td style="text-align: right;"><code>0</code></td>
<td style="text-align: right;"><code>0</code></td>
</tr>
<tr>
<td><code>decay_eviction_blocked</code></td>
<td style="text-align: right;"><code>1</code></td>
<td style="text-align: right;"><code>0.092343</code></td>
<td style="text-align: right;"><code>101</code></td>
<td style="text-align: right;"><code>101</code></td>
<td style="text-align: right;"><code>1</code></td>
<td style="text-align: right;"><code>0.006087</code></td>
<td style="text-align: right;"><code>101</code></td>
<td style="text-align: right;"><code>101</code></td>
</tr>
<tr>
<td><code>frozen_eviction_blocked</code></td>
<td style="text-align: right;"><code>1</code></td>
<td style="text-align: right;"><code>0.619835</code></td>
<td style="text-align: right;"><code>101</code></td>
<td style="text-align: right;"><code>101</code></td>
<td style="text-align: right;"><code>1</code></td>
<td style="text-align: right;"><code>0.615121</code></td>
<td style="text-align: right;"><code>101</code></td>
<td style="text-align: right;"><code>101</code></td>
</tr>
</tbody>
</table>

Full benchmark sweep (selected checkpoints):

-   `decay_eviction_allowed`: target still existed and surfaced at `6 h`
    (`strength=0.252589`, exact/paraphrase top-1=`101`), but was already
    gone by `24 h`
-   `decay_eviction_blocked`: target strength fell from `0.620625` at
    `0 h` to `0.006087` at `168 h`, while exact and paraphrase retrieval
    both remained top-1=`101` at every tested gap
    (`0, 1, 6, 24, 72, 168 h`)
-   `frozen_eviction_blocked`: target strength stayed effectively flat
    (`0.620625 -> 0.615121`) and produced the same top-1 retrieval order
    as the decaying-but-non-evicted condition

Recorded checks:

-   `surfacing_blocked_decay_exact_top1_all = 1`
-   `surfacing_blocked_decay_paraphrase_top1_all = 1`
-   `surfacing_frozen_exact_top1_all = 1`
-   `surfacing_frozen_paraphrase_top1_all = 1`
-   `surfacing_blocked_decay_matches_frozen_all = 1`
-   `surfacing_blocked_decay_strength_lt_frozen_168h = 1`
-   `surfacing_eviction_allowed_exists_at_6h = 1`
-   `surfacing_eviction_allowed_missing_at_24h = 1`
-   `surfacing_eviction_allowed_missing_at_168h = 1`

**Observation:** In the current retrieval stack, **strength-only decay
did not change surfacing once the raw trace was prevented from being
evicted**. The decaying-but-retained condition lost roughly `99%` of its
effective strength over `168 h` (`0.620625 -> 0.006087`), yet exact and
paraphrase queries returned the same top-1 target at every tested gap,
and the selected retrieval order matched the frozen-strength control
exactly. The only condition that stopped surfacing the target was the
control where eviction remained allowed, and there the failure mode was
physical disappearance by `24 h`, not retrieval demotion while still
present. This strongly suggests that the practical “old memory did not
come back” complaint should currently be investigated first as
**eviction, missing consolidation products, retrieval ranking, or query
mismatch**, not as a direct effect of retained-memory strength decay
inside `GraphAugmentedRetrieveCandidates`.

### Messy Chat Baseline (Ubuntu Dialogue Corpus)

We ran a long‑horizon baseline on the **Ubuntu Dialogue Corpus
(validation)** to stress the system with noisy, real IRC chat.
Configuration: `max_total=720`, `max_conversations=200`, **no
consolidation**, F=S=T=0.5.

Run: `logs/topical_chat_snapshots/20260103_114812`

-   interrupt_turn_rate: **0.503**
-   interrupt_abort_rate: **0.622**
-   interrupt_semantic_overlap_mean: **0.555**
-   interrupt_context_semantic_overlap_mean: **0.660**
-   boundary_at_rate: **0.103**

**Observations:** Compared to TopicalChat, Ubuntu’s noisy turns yield a
higher interrupt rate and substantially higher abort rate, indicating
the interrupt gate fires often but the acceptance/continuation logic
struggles with messy streams. This dataset is a better stress test for
interrupt gating and should be used for future robustness tuning.

### Ubuntu Interrupt Accept/Ignore Ablation

We disabled the interrupt accept/ignore comparator to test whether it
reduces abort churn on noisy chat. Configuration matches the Ubuntu
baseline above.

-   baseline: `logs/topical_chat_snapshots/20260103_114812`
-   accept/ignore disabled (`CORTEXT_DISABLE_INTERRUPT_ACCEPT=1`):
    `logs/topical_chat_snapshots/20260103_162953`

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
<th>mode</th>
<th>interrupt_turn_rate</th>
<th>interrupt_abort_rate</th>
<th>interrupt_semantic_overlap_mean</th>
<th>boundary_at_rate</th>
</tr>
</thead>
<tbody>
<tr>
<td>baseline</td>
<td>0.503</td>
<td>0.622</td>
<td>0.555</td>
<td>0.103</td>
</tr>
<tr>
<td>accept_off</td>
<td>0.511</td>
<td>0.649</td>
<td>0.557</td>
<td>0.094</td>
</tr>
</tbody>
</table>

**Observations:** Disabling accept/ignore **increases** abort rate on
Ubuntu, indicating the comparator helps stabilize interruption handling
under noisy conditions. The next step is to tune the acceptance margin
rather than remove the mechanism.

We then added a **knob‑derived acceptance margin** (requiring the next
signal to be closer to the selected memory by a small margin).
Configuration matches the Ubuntu baseline above.

-   margin tuned: `logs/topical_chat_snapshots/20260103_201348`
-   context‑aware comparator (regression):
    `logs/topical_chat_snapshots/20260104_182951`

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
<th>mode</th>
<th>interrupt_turn_rate</th>
<th>interrupt_abort_rate</th>
<th>interrupt_semantic_overlap_mean</th>
<th>boundary_at_rate</th>
</tr>
</thead>
<tbody>
<tr>
<td>baseline</td>
<td>0.503</td>
<td>0.622</td>
<td>0.555</td>
<td>0.103</td>
</tr>
<tr>
<td>accept_off</td>
<td>0.511</td>
<td>0.649</td>
<td>0.557</td>
<td>0.094</td>
</tr>
<tr>
<td>margin_tuned</td>
<td>0.507</td>
<td>0.611</td>
<td>0.551</td>
<td>0.107</td>
</tr>
<tr>
<td>context_aware</td>
<td>0.513</td>
<td>0.688</td>
<td>0.555</td>
<td>0.085</td>
</tr>
</tbody>
</table>

**Observations:** The margin reduces aborts (~1.1pp vs baseline) without
lowering interrupt rate, while the context‑aware comparator increases
aborts and is a regression. We keep the simple μ_acc comparator with a
small knob‑derived margin.

We also tested a **novelty‑weighted margin** (more permissive on
high‑novelty signals), which regressed abort rate:

-   novelty‑weighted margin:
    `logs/topical_chat_snapshots/20260104_203047`

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
<th>mode</th>
<th>interrupt_turn_rate</th>
<th>interrupt_abort_rate</th>
<th>interrupt_semantic_overlap_mean</th>
<th>boundary_at_rate</th>
</tr>
</thead>
<tbody>
<tr>
<td>novelty_margin</td>
<td>0.504</td>
<td>0.634</td>
<td>0.553</td>
<td>0.100</td>
</tr>
</tbody>
</table>

**Observations:** The novelty‑weighted margin increases abort rate on
Ubuntu, so we keep the fixed small margin.

We then tested **candidate diversity gating** (raising the interrupt
threshold when top‑K candidates were label‑only). This also regressed
abort rate:

-   diversity gating: `logs/topical_chat_snapshots/20260105_001810`

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
<th>mode</th>
<th>interrupt_turn_rate</th>
<th>interrupt_abort_rate</th>
<th>interrupt_semantic_overlap_mean</th>
<th>boundary_at_rate</th>
</tr>
</thead>
<tbody>
<tr>
<td>diversity_gate</td>
<td>0.488</td>
<td>0.675</td>
<td>0.553</td>
<td>0.094</td>
</tr>
</tbody>
</table>

**Observations:** Diversity‑gated thresholds increased aborts and
slightly reduced interrupt rate on Ubuntu, so we removed this mechanism.

## Affect-Gated Sensitivity Sweep (Long Horizon)

We evaluated the affect-gated interrupt + retrieval coupling by sweeping
Sensitivity while holding **F=0.5, T=0.5**, and running **2
conversations × 360 turns** (`max_total=720`, interleave=1). This
isolates how the affect drive and emotion bonus influence interrupt
quality.

<table>
<colgroup>
<col style="width: 12%" />
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
<th style="text-align: right;">S</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">precision</th>
<th style="text-align: right;">recall</th>
<th style="text-align: right;">fp_rate</th>
<th style="text-align: right;">fn_rate</th>
<th style="text-align: right;">affect_drive_mean</th>
<th style="text-align: right;">retrieval_emotion_bonus_mean</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">0.0</td>
<td style="text-align: right;">0.486</td>
<td style="text-align: right;">0.995</td>
<td style="text-align: right;">0.879</td>
<td style="text-align: right;">0.0053</td>
<td style="text-align: right;">0.121</td>
<td style="text-align: right;">0.236</td>
<td style="text-align: right;">0.0578</td>
</tr>
<tr>
<td style="text-align: right;">0.5</td>
<td style="text-align: right;">0.483</td>
<td style="text-align: right;">0.995</td>
<td style="text-align: right;">0.879</td>
<td style="text-align: right;">0.0053</td>
<td style="text-align: right;">0.121</td>
<td style="text-align: right;">0.246</td>
<td style="text-align: right;">0.0727</td>
</tr>
<tr>
<td style="text-align: right;">1.0</td>
<td style="text-align: right;">0.512</td>
<td style="text-align: right;">0.985</td>
<td style="text-align: right;">0.938</td>
<td style="text-align: right;">0.0150</td>
<td style="text-align: right;">0.0619</td>
<td style="text-align: right;">0.262</td>
<td style="text-align: right;">0.0847</td>
</tr>
</tbody>
</table>

**Observations:** Higher Sensitivity increases affect drive and the
retrieval emotion bonus, improving interrupt recall (fewer false
negatives) at the cost of a modest precision drop and higher interrupt
rate. The mid-point (S=0.5) preserves precision while providing moderate
affect influence.

## Focus Sweep (Long Horizon)

We swept Focus with **S=0.5, T=0.5**, using **2 conversations × 360
turns** (`max_total=720`). Focus primarily modulates retrieval breadth
and duplicate suppression, so we report candidate counts and interrupt
quality.

<table>
<colgroup>
<col style="width: 12%" />
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
<th style="text-align: right;">F</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">precision</th>
<th style="text-align: right;">recall</th>
<th style="text-align: right;">fp_rate</th>
<th style="text-align: right;">fn_rate</th>
<th style="text-align: right;">retrieval_avg_candidates</th>
<th style="text-align: right;">retrieval_semantic_overlap_mean</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">0.3</td>
<td style="text-align: right;">0.481</td>
<td style="text-align: right;">1.000</td>
<td style="text-align: right;">0.984</td>
<td style="text-align: right;">0.000</td>
<td style="text-align: right;">0.0157</td>
<td style="text-align: right;">32.54</td>
<td style="text-align: right;">0.547</td>
</tr>
<tr>
<td style="text-align: right;">0.5</td>
<td style="text-align: right;">0.483</td>
<td style="text-align: right;">0.995</td>
<td style="text-align: right;">0.879</td>
<td style="text-align: right;">0.0053</td>
<td style="text-align: right;">0.121</td>
<td style="text-align: right;">27.17</td>
<td style="text-align: right;">0.526</td>
</tr>
<tr>
<td style="text-align: right;">0.7</td>
<td style="text-align: right;">0.483</td>
<td style="text-align: right;">0.968</td>
<td style="text-align: right;">0.640</td>
<td style="text-align: right;">0.0317</td>
<td style="text-align: right;">0.360</td>
<td style="text-align: right;">26.14</td>
<td style="text-align: right;">0.535</td>
</tr>
</tbody>
</table>

**Observations:** Lower Focus (F=0.3) increases candidate breadth and
improves interrupt recall/precision; higher Focus narrows candidate sets
and increases misses (higher FN rate) despite slightly higher semantic
overlap. This confirms Focus governs selectivity at the cost of
interrupt coverage.

## Stability Sweep (Long Horizon)

We swept Stability with **F=0.5, S=0.5**, using **2 conversations × 360
turns** (`max_total=720`). Stability adjusts persistence and gating
strictness, so we track interrupt behavior and candidate volume.

<table>
<colgroup>
<col style="width: 12%" />
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
<th style="text-align: right;">T</th>
<th style="text-align: right;">interrupt_turn_rate</th>
<th style="text-align: right;">precision</th>
<th style="text-align: right;">recall</th>
<th style="text-align: right;">fp_rate</th>
<th style="text-align: right;">fn_rate</th>
<th style="text-align: right;">retrieval_avg_candidates</th>
<th style="text-align: right;">retrieval_semantic_overlap_mean</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">0.3</td>
<td style="text-align: right;">0.483</td>
<td style="text-align: right;">0.995</td>
<td style="text-align: right;">0.870</td>
<td style="text-align: right;">0.0053</td>
<td style="text-align: right;">0.1296</td>
<td style="text-align: right;">31.20</td>
<td style="text-align: right;">0.528</td>
</tr>
<tr>
<td style="text-align: right;">0.5</td>
<td style="text-align: right;">0.483</td>
<td style="text-align: right;">0.995</td>
<td style="text-align: right;">0.879</td>
<td style="text-align: right;">0.0053</td>
<td style="text-align: right;">0.1215</td>
<td style="text-align: right;">27.28</td>
<td style="text-align: right;">0.526</td>
</tr>
<tr>
<td style="text-align: right;">0.7</td>
<td style="text-align: right;">0.455</td>
<td style="text-align: right;">0.994</td>
<td style="text-align: right;">0.823</td>
<td style="text-align: right;">0.0056</td>
<td style="text-align: right;">0.1767</td>
<td style="text-align: right;">29.35</td>
<td style="text-align: right;">0.527</td>
</tr>
</tbody>
</table>

**Observations:** Higher Stability slightly reduces interrupt rate and
recall (more false negatives), consistent with stricter gating;
mid‑range T preserves recall while keeping precision stable. Candidate
counts remain in the same band, indicating Stability primarily affects
decision thresholds rather than retrieval breadth.

# Implementation Considerations

## Embedding-Only Operations

In the current alpha, all online operations (scoring, boundary
detection, retrieval, gating, graph updates) operate **only** on
embeddings. Raw text/audio/image content is used solely to generate
embeddings; the live loop never re-enters token space. This preserves
modality-agnostic behavior and prevents hidden heuristics from bypassing
knob control.

Retrieved memories are hydrated for inspection and evaluation only. When
a memory has no signal-level blobs (e.g., consolidation summaries), the
system loads the summary payload from `memories.blob_id` so summary
nodes can participate in analysis without affecting online computations.

## Model Stack (Local Inference)

-   **ImageBind embeddings:** image/text/audio inputs are embedded via
    ImageBind executed through ONNX (default path: `models/imagebind/`).
-   **Consolidation LLM:** summarization and labeling are the only
    generative steps and run only during consolidation. Cortext resolves
    a local backend internally: **gemma-3n-e2b** via
    `third_party/litert_lm`, **LFM2/LFM2.5** via `llama.cpp` GGUFs, or a
    **mixed** Gemma+Liquid path. The Gemma path auto-selects among
    available `.litertlm` variants under `models/gemma3n-e2b-litert/`.
    The Liquid path now prefers `LFM2.5-350M-GGUF` for both
    summarization and extraction, preferring `Q4_K_M` when present and
    otherwise accepting other shipped GGUF quantizations; env overrides
    still take precedence for exact paths. If the `LFM2.5-350M-GGUF`
    asset is absent, Cortext falls back to the older
    `LFM2-2.6B-Transcript-Q4_K_M.gguf` summarizer and
    `LFM2-1.2B-Extract-Q4_K_M.gguf` extractor. The mixed path uses Gemma
    for summarization and the preferred Liquid extractor path for
    labeling, but auto resolution now prefers the pure
    `LFM2.5/llama.cpp` path when the Liquid assets are present; mixed
    and Gemma are fallbacks. If no deep backend is available, deep
    consolidation raises an error rather than silently degrading. The
    summarization prompt remains role-aware for chat sources, favors
    direct fact-oriented memory notes over dialogue recaps, explicitly
    preserves named people, projects, and technologies, discourages
    speaker-role phrasing such as “the user” or “the assistant”, and
    preserves chronological ordering when selected chat excerpts are
    passed to the model. The Liquid GGUF path is text-only;
    audio-specific summarization/extraction methods remain unsupported
    there.
-   **Offline label-classifier prototype:** we now also ship a
    research-only Python pipeline that trains span typing / promotion
    models from WordNet, multilingual name priors, and sentence context.
    The current study path builds those priors from
    `data/surnames/{forenames,surnames}.csv` plus U.S. Census surnames.
    When external embeddings are enabled, the prototype reuses the
    repo’s preferred text encoder through the rebuilt
    `cortext_text_embedder`, so offline studies can run on
    `EmbeddingGemma/llama.cpp` without changing the online runtime path.

No other model runtimes are required for the online loop; all
operational decisions are embedding-space computations derived from
F/S/T.

## Alpha Scope and Structured-Fact Foundation

The current alpha stores temporality primarily at the episodic level:

-   episodic memories have event timestamps,
-   retrieval and decay are recency-aware,
-   consolidation produces summaries, labels, and graph structure.

This is sufficient for long-horizon continuity, but it is not enough on
its own for changing structured facts such as residence, caregiver,
routine, diagnosis, medication, or appointment state.

The current phase adds an additive **bitemporal structured-fact layer**
populated during consolidation rather than during the live loop. A
minimal fact record includes:

    fact_id
    subject
    predicate
    object_or_value
    valid_start, valid_end
    recorded_at, superseded_at
    confidence
    source_memory_ids
    source_summary_ids

This layer is additive, not a replacement for the existing episodic
graph. The division of responsibility is:

-   **episodic memories:** preserve raw experience and provenance,
-   **summaries and graph nodes:** provide semantic compression and
    associative retrieval,
-   **bitemporal facts:** answer changing factual questions with
    explicit world-time and system-time semantics.

Under this design, deep extraction in consolidation emits assertions
with valid-time and transaction-time boundaries. The fact store supports
three query modes:

-   current truth (`what is true now?`)
-   historical truth (`what was true at world time τ?`)
-   historical belief
    (`what would Cortext have answered at system time τ?`)

Crucially, this preserves the alpha architecture: the live control loop
remains embedding-only, while structured temporal reasoning is added as
a higher-level memory product derived from consolidation outputs. The
current branch uses those facts in the core retrieval and lifecycle path
rather than exposing a separate fact payload in the chat example.

The three user-facing knobs still shape the core behavior:

-   **Focus** increases fact-linked retrieval influence and strengthens
    stale-conflict suppression during ranking.
-   **Sensitivity** increases preference for newly confirmed caregiver,
    location, and schedule changes when current evidence conflicts with
    older routine summaries.
-   **Stability** contributes a lighter routine-preservation bias in the
    core retrieval path and broader memory dynamics.

Phase 6 hardens the **core** bitemporal layer before any additional chat
integration. Two changes were necessary in the retrieval pipeline.
First, current-mode ranking now penalizes stale same-pair evidence even
when an older fact has the same object string as the current winner (for
example, `Austin -> Denver -> Austin` no longer lets the old `Austin`
memory outrank the current one merely because it is semantically closer
to the query vector). Second, fact seeding now over-fetches from the
shared embedding table before temporal filtering so that current fact
embeddings are not crowded out by nearby non-fact memory embeddings. On
the ingestion side, conflicting low-confidence assertions are no longer
allowed to supersede stronger current facts; they are stored as
immediately superseded evidence until a sufficiently strong correction
arrives.

Phase 7 adds a second internal experiment layer on top of that hardened
core: a thread-local **retrieval ablation override** used only by
deterministic tests and benchmarks. It can disable the fact layer
entirely, collapse historical queries to current-only behavior, scale
fact boosts and stale penalties, relax provenance from direct evidence
links to unrestricted semantic fact matches, and apply a
routine-versus-recency preset in the core scoring path. Phase 8 extends
that layer with a broader deterministic matrix: explicit `current` /
`valid_at` / `known_at` reporting, a wider predicate/severity scenario
set, and a representative `3 x 3 x 3` F/S/T sweep. The key
implementation constraint remains unchanged: the override changes only
internal core behavior; no public headers, C API surfaces, or chat-path
controls are introduced.

Phase 9 adds the missing **fact lifecycle** layer. Fact assertions now
persist internal support traces such as support mass, source diversity,
contradiction mass, confirmation count, severity class, and lifecycle
state (`active`, `weak`, `archived`). Consolidation still inserts or
supersedes fact assertions, but it now also updates those support traces
and then runs a bounded lifecycle maintenance sweep. That sweep decays
unsupported facts, compresses repeated confirmations into support
statistics, archives low-support facts, and physically deletes only
low-severity archived facts by default. High-severity historical facts
are retained even after evidence loss so that deterministic `valid_at`
and `known_at` queries remain correct.

This leaves the architecture in the intended order: the fact store,
temporal queries, fact-aware retrieval, and lifecycle maintenance are
now benchmark-gated at the core level and causally ablated there, while
the chat example remains a consumer of the normal memory-return surface
rather than a separate fact-grounding path.

Lower-level temporal retrieval modes (`current`, `valid_at`, `known_at`)
remain implemented for deterministic tests and benchmarks. Explicit
user-facing temporal controls and any future chat-specific grounding or
reply templating remain later work.

## Computational Complexity

The primary per-stimulus cost is the k-nearest-neighbors (kNN) search
for metrics and retrieval. With `N` stored memories, naive search is
`O(N)`. We recommend maintaining an HNSW index (Malkov and Yashunin
2018) to achieve `O(log N)` retrieval.

Label/association seeds are selected from an in‑memory summary cache
(populated by the label bank and consolidation) to avoid an additional
vector search pass while keeping retrieval fully embedding‑based.

Recursive graph expansion adds overhead but is bounded by `depth` and
`branching_factor`, effectively constant-time relative to `N`. Total
complexity per stimulus is `O(log N)` with indexing.

## Memory Footprint

The system stores: \* `signal_stream`: `n_ctx` accumulator centroids
(256d float32 = 1KB each). \* `score_stream`: Scalar history. \*
`memory_stream`: `N` stored memories (embedding + metadata). \* `graph`:
Adjacency lists for graph edges.

For long-running instances, `memory_stream` and `graph` dominate.
Pruning low-strength memories
(<a href="#sec-memory-strength" class="quarto-xref">Section 7.1</a>) is
essential for bounding growth.

## Numerical Stability

Exponential decays and probability computations can suffer from
underflow. We use `log-space` arithmetic where appropriate and clamp all
potentially divergent values (e.g., division by small sums) with ε =
10⁻⁶.

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

Future work will focus on: 1. **Large-scale evaluation:** Validating the
architecture on long-horizon datasets (~1M+ steps). 2. **Multi-agent
dynamics:** Exploring how Cortext instances interact in collaborative
settings. 3. **Hardware acceleration:** Optimizing the kNN and graph
traversal kernels for edge devices.

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

Hart, J T. 1965. “Memory and the Feeling-of-Knowing Experience.”
*Journal of Educational Psychology* 56 (4): 208.

Hunt, R Reed. 1995. “The Subtlety of Distinctiveness: What von Restorff
Really Did.” *Psychonomic Bulletin & Review* 2 (1): 105–12.

LaBar, Kevin S, and Roberto Cabeza. 2006. “Cognitive Neuroscience of
Emotional Memory.” *Nature Reviews Neuroscience* 7 (1): 54–64.

Liu, Jun S, and Rong Chen. 1998. “Sequential Monte Carlo Methods for
Dynamic Systems.” *Journal of the American Statistical Association* 93
(443): 1032–44.

Malkov, Yu A, and Dmitry A Yashunin. 2018. “Efficient and Robust
Approximate Nearest Neighbor Search Using Hierarchical Navigable Small
World Graphs.” *IEEE Transactions on Pattern Analysis and Machine
Intelligence* 42 (4): 824–36.

McClelland, James L, Bruce L McNaughton, and Randall C O’Reilly. 1995.
“Why There Are Complementary Learning Systems in the Hippocampus and
Neocortex: Insights from the Successes and Failures of Connectionist
Models of Learning and Memory.” *Psychological Review* 102 (3): 419.

McCloskey, Michael, and Neal J Cohen. 1989. *Catastrophic Interference
in Connectionist Networks: The Sequential Learning Problem*. Vol. 24.
Elsevier.

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

Tulving, Endel. 1972. “Episodic and Semantic Memory.” *Organization of
Memory* 1: 381–403.

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

-   **Per-memory defaults (on insert):** `strength = 1`,
    `trace_fast = 0`, `trace_med = 0`, `trace_slow = 0`,
    `trace_ultra = 0`, `use_frequency = 0`, `stability = 1`,
    `connectivity = 0`, `drift_mag = 0`, `influence = 0`,
    `sustained_influence = 0`, `contextual_gain = 0`, `redundancy = 0`,
    `pre_activation = 0`, `lability_state = 0`, `suppression_count = 0`,
    `suppression = 0`, `flashbulb = 0`, `s_emotion_max = 0`,
    `s_arousal_avg = 0`, `boundary_score = 0`, `tagged = false`,
    `tag_expires_at = 0`, `context = 0_vector`,
    `source_model = {origin: 'unknown', reliability: 0.5, contradiction_count: 0, last_verified_ts: 0}`.
    Evidence packets live in ordered `signals` rows, and the
    constructive-recall ledger starts empty until the initial
    reconstruction row is appended in `memory_reconstructions`.

-   **RLS defaults:** `w_* = w_bootstrap`, `P = diag(1000)`,
    `blender_ready = false`, `blender_update_count = 0`.

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
elsewhere in the document.

Canonical single-step pseudocode (timestep t):

    # Main loop (single timestep t)
    now_s, now_ms, x_t ← read_inputs()
    update_accumulator_embedding(x_t)  # updates μ_acc for current unit
    recent_context ← tail(signal_stream, n_ctx(T))  # signal_stream stores prior μ_acc

    # Structural metrics + uncertainty
    coherence_struct_t, focus_spread_t, drift_mag_t, surprisal_t ← compute_structural_metrics(μ_acc)
    x_pred_ema ← update_prediction_ema(x_pred_ema, μ_acc, T)  # after surprisal_t
    u_t ← update_uncertainty(...)

    # Adaptation + scoring
    update_control_parameters(...)
    score_t ← compute_composite_score(...)
    score_stream.append(score_t)
    update_accumulator_scores(score_t, μ_acc)
    signal_stream.append(μ_acc)

    # Threshold updates
    θ_target ← prior_evidence_blend(...)
    Δθ_* ← compute_threshold_deltas(...)
    θ_dynamic ← update_theta_dynamic(...)
    hysteresis ← update_hysteresis(...)

    # Accumulator + boundary
    update_accumulator(...)
    boundary_score ← compute_boundary_score(...)
    spike_bypass ← check_spike_bypass(score_t, θ_dynamic, mem_maturity, coherence_t)
    should_flush ← boundary_decision(boundary_score, time_caps, spike_bypass)
    θ_memory ← θ_dynamic × M_write_refrac
    write_memory ← force_write OR (should_flush AND (S_window > θ_memory))
    Δwrites ← 1 if write_memory else 0  # computed immediately after write decision

    # Post-write and retrieval (q_retrieval captured before any reset)
    if write_memory: commit_memory_unit(); last_write_ts ← now_ms()
    q_retrieval ← l2_normalize(μ_acc)  # cache current-unit query
    streaming_pacing_check()
    graph_retrieval(q_retrieval)
    update_rate_state(Δwrites)
    if should_flush: reset_accumulator()  # regardless of write outcome

    # Interrupt gate (consumes retrieved candidates)
    interrupt_gate_check()
    if allow_interrupt and not should_flush and not spike_bypass:
        pending_abort ← true
        pending_mem ← selected_candidate_embedding

    # Next signal: accept if closer to selected memory, else resume.
    if pending_abort:
        if cos(x_t, pending_mem) > cos(x_t, μ_acc):
            reset_accumulator()
        pending_abort ← false

The normative execution order for a single timestep t:

1.  **Read Inputs:** `x_t`, `now_s()`, `system_time_ms`.
2.  **Update Accumulator (Embedding):** Update `μ_acc` with `x_t` (no
    score aggregation yet).
3.  **Prepare Context:** Retrieve
    `recent_context ← tail(signal_stream, n_ctx(T))` (prior μ_acc values
    only).
4.  **Compute Structural Metrics:** `coherence_struct`, `focus_spread`,
    `drift`, `surprisal` using `μ_acc`, then update `x_pred_ema` for the
    next step.
5.  **Update Uncertainty:** `u(t)`.
6.  **Compute Adaptation Dynamics:**
    -   Update `α_F(t)`, `α_T(t)`.
    -   Update `weight_relevance`, `half_life`.
    -   Update `emotion` and `mood` state.
7.  **Compute Composite Score:**
    -   Compute all 12 metrics.
    -   Update RLS weights `W_blend`.
    -   Compute `score_t`. Append to `score_stream`.
    -   Update accumulator score aggregates (`s_sum`, `s_max`, `e_peak`)
        using `score_t`.
    -   Append `μ_acc` to `signal_stream`.
8.  **Update Thresholds:**
    -   Compute `θ_target` (prior/evidence).
    -   Compute `Δθ` terms (homeo, sens, prec, emo, mood).
    -   Update `θ_dynamic`. Update `hysteresis`.
9.  **Execute Memory Accumulation:**
    -   Accumulator now holds `μ_acc`, `drift_acc`, `s_max`, etc.
    -   Compute `boundary_score` and `should_flush`.
    -   Check `spike_bypass` using the effective spike margin scaled by
        `mem_maturity` and `coherence_t`.
    -   Compute `S_window` and `θ_memory`.
    -   Decide `write_memory`.
10. **Post-Write Updates and Retrieval:**
    -   If `write_memory`: Write to `memory_stream`. Update
        `last_write_ts`.
    -   Cache `q_retrieval ← μ_acc` before any accumulator reset.
    -   Run streaming pacing and retrieval using `q_retrieval`.
    -   Update `rate_state` (homeostatic controller).
    -   If `should_flush` (regardless of write): Call
        `reset_accumulator()`.
11. **Run Interrupt Gate:** Check for streaming interrupt using
    retrieved candidates (already filtered by write‑exclusion and WM
    overlap during retrieval).
12. **Interrupt Abort (if allowed):** Mark a pending abort when an
    interrupt is permitted outside a flush/spike event. On the next
    signal, compare similarity to the selected memory vs current μ_acc;
    if the new signal aligns more with the selected memory, reset to
    drop partial utterances, otherwise resume.

Timing notes: \* `now_s()` is captured at step 1 and reused for all Δt
computations in this timestep. \* Threshold updates in step 7 use the
score computed in step 6. \* Rate updates in step 9 use Δwrites from the
current step’s `write_memory` decision.

**Normative Invariants:** \* **Causality:** Step `k` uses only values
computed in steps `1` through `k` or retained from `t-1`. \* **Write
Atomicity:** A “write” is an atomic commit of a `should_flush` unit.
Partial units are never written. \* **Gap Consistency:** `now_s()` is
fixed at step 1. All `Δt` calculations use this fixed value.
