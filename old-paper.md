**Cortext: A Three-Knob Adaptive Memory Architecture**

**for Continuous Cognitive Processing**

*Technical Report --- December 2025*

**Abstract**

We present Cortext, a biologically-inspired adaptive memory system governed by three continuous control parameters: Focus (F), Sensitivity (S), and Stability (T). Unlike traditional memory architectures that employ discrete operational modes, Cortext achieves developmental progression through parameter-derived rate modulation, allowing behavior to emerge continuously from the interaction of knob settings and experiential mass. The architecture integrates established cognitive science principles---including Cowan\'s working memory constraints and Nader\'s reconsolidation dynamics---into a unified computational framework. We derive all system parameters from the three primary knobs through principled mathematical transformations, reducing reliance on fixed constants. The system demonstrates self-calibrating priors that blend with evidence using uncertainty-weighted Bayesian averaging, homeostatic threshold control with effective sample size estimation, and graph-augmented retrieval combining embedding similarity with semantic extraction. Experimental analysis indicates the architecture maintains stable operation across developmental phases while adapting write rates, decay dynamics, and retrieval precision to environmental demands. This work contributes a formally specified cognitive memory model suitable for implementation in streaming AI systems requiring persistent, context-aware memory.

**Keywords:** cognitive architecture, adaptive memory, working memory, episodic memory, semantic memory, knowledge graphs, homeostatic control

1\. Introduction

Memory systems in artificial intelligence face a fundamental tension between plasticity and stability. Systems that learn rapidly risk catastrophic interference, while those that maintain stable representations may fail to capture novel patterns (McCloskey & Cohen, 1989). Biological memory systems resolve this tension through sophisticated regulatory mechanisms that modulate learning rates, decay dynamics, and retrieval thresholds in response to environmental demands and internal state (McClelland et al., 1995).

This paper introduces Cortext, a cognitive memory architecture that addresses this stability-plasticity dilemma through three continuous control parameters that govern all system behavior. Rather than implementing discrete operational modes or hard-coded phase transitions, Cortext achieves developmental progression through the continuous interaction of parameter settings with accumulated experience. The architecture draws on established findings from cognitive psychology and neuroscience, including working memory capacity limits (Cowan, 2001), memory reconsolidation (Nader et al., 2000), serial position effects (Murdock, 1962), and emotional modulation of memory (McGaugh, 2004).

The core contribution of this work is a formally specified memory architecture in which:

1.  All tuneable parameters derive from three primary knobs (Focus, Sensitivity, Stability) through explicit mathematical transformations, reducing reliance on fixed constants

2.  System priors self-calibrate through uncertainty-weighted Bayesian blending with observed evidence

3.  Developmental phases emerge from annealed safety bounds and experiential mass accumulation, not explicit mode switching

4.  A knowledge graph layer enables semantic consolidation and graph-augmented retrieval

The remainder of this paper is organized as follows. Section 2 reviews relevant literature. Section 3 presents the mathematical foundations including notation, helper functions, and knob-derived parameters. Section 4 details the core algorithms for Focus, Sensitivity, and Stability adaptation. Section 5 describes structural metrics and composite scoring. Section 6 covers dynamic thresholding and homeostatic control. Section 7 presents the reinforcement and decay dynamics. Section 8 describes advanced cognitive processes including working memory, metacognition, and emotional consolidation. Section 9 details the consolidation and graph integration system. Section 10 presents the interrupt gate for streaming integration. Section 11 reports preliminary experimental results. Section 12 discusses implementation considerations and computational complexity. Section 13 concludes with limitations and future directions.

2\. Related Work

2.1 Working Memory Models

The working memory component of Cortext draws primarily on Cowan\'s (2001) embedded-processes model, which posits a capacity limit of approximately 4±1 chunks for the focus of attention. This contrasts with Miller\'s (1956) earlier estimate of 7±2 items, which subsequent research has shown conflates chunking with raw capacity (Cowan, 2010). Our implementation respects these empirically-derived constraints while allowing for focus-dependent modulation within bounded ranges.

Baddeley\'s (2000) multicomponent model informs our treatment of maintenance and rehearsal processes, though we adopt a more unified representational substrate based on distributed embeddings rather than separate phonological and visuospatial stores. The episodic buffer concept (Baddeley, 2000) aligns with our approach to binding information across modalities through shared vector spaces.

2.2 Memory Consolidation

The consolidation mechanisms in Cortext reflect findings from the memory reconsolidation literature (Nader et al., 2000; Nader, 2003). Reconsolidation theory posits that retrieved memories enter a labile state during which they can be modified before restabilization. Our architecture implements this through time-bounded lability windows governed by the Stability parameter, with reconsolidation gain modulated by both Sensitivity and contextual relevance.

The distinction between episodic and semantic memory (Tulving, 1972) motivates our two-tier storage approach: a streaming episodic buffer for immediate experiences and a consolidated semantic graph for abstracted knowledge. The consolidation process transforms high-redundancy episodic clusters into summary nodes linked by typed semantic relations, consistent with complementary learning systems theory (McClelland et al., 1995).

2.3 Emotional Influences on Memory

McGaugh\'s (2004) extensive work on emotional modulation of memory consolidation informs our treatment of affect-gated encoding. The architecture implements emotional intensity as a threshold modifier, consistent with findings that arousal enhances memory through amygdala-mediated modulation of hippocampal encoding (LaBar & Cabeza, 2006). We adopt a dimensional model of emotion (Russell, 1980) with valence and arousal as primary axes, projected from categorical emotion embeddings.

2.4 Adaptive Control Systems

The homeostatic threshold controller draws on classical control theory, specifically proportional-integral approaches to setpoint maintenance (Åström & Murray, 2008). The use of exponentially-weighted moving averages for rate estimation follows standard practice in adaptive systems, while our effective sample size calculation for reliability estimation extends techniques from sequential Monte Carlo methods (Liu & Chen, 1998).

3\. Mathematical Foundations

3.1 Notation and Primitives

We establish the following notation used throughout this paper. Let ε = 10⁻⁶ denote a small constant for numerical stability. All knob values F, S, T lie in the closed interval \[0, 1\].

Core mathematical primitives:

lerp(a, b, x) = a + (b − a) × x

clamp(v, lo, hi) = max(lo, min(v, hi))

sigmoid(z) = 1 / (1 + exp(−z))

EWMA(prev, x, α) = (1 − α) × prev + α × x

Weight normalization and blending for combining multiple signals:

normalize(w) = w / max(sum(w), ε) \# w is a weight vector

Edge case: if sum(w) \< ε, return uniform weights (1/\|w\|) to avoid division by zero or invalid probability distributions.

blend(values, weights) = Σᵢ values\[i\] × weights\[i\]

Note: normalize() operates on weight vectors, not scalars. When used with scalar expressions like lerp(), collect the scalars into a vector first: normalize(\[lerp(\...), lerp(\...), \...\]). The blend() function assumes pre-normalized weights summing to 1.

For vectors, we define cosine similarity as cos(u, v) = u·v / (‖u‖ × ‖v‖), and safe L2 normalization as l2_normalize(v) = v / max(‖v‖, ε). Shannon entropy is computed in nats: H(p) = −Σᵢ pᵢ ln(pᵢ).

The temporal decay function follows exponential dynamics with configurable half-life:

decay(x, τ_half, Δt) = x × exp(−ln(2) × Δt / max(τ_half, τ_min))

where τ_min = 120 seconds provides a floor to prevent numerical instability from near-zero half-lives.

3.1.1 Units Convention

To ensure consistency and avoid unit mismatch errors, the following conventions apply throughout:

**Input timestamps:** All input timestamps (t) are specified in milliseconds since epoch, consistent with standard system time APIs.

**Internal time representation:** All internal time calculations operate in seconds. We define:

now_s() = system_time_ms / 1000 \# returns seconds

now_ms() = system_time_ms \# returns milliseconds

to_s(ts_ms) = ts_ms / 1000 \# ms → seconds conversion

**Stored timestamps:** All stored timestamps are milliseconds since Unix epoch.

**Naming contract (canonical):** Use the variable names below consistently throughout this document. Units: stored timestamps are integers in milliseconds (commonly suffixed \*\_ts, and also appearing as timestamp/created_at/last_rate_timestamp); derived time intervals in seconds use the \*\_s suffix. Accumulator variables: t_start, last_signal_ts, last_write_ts, drift_acc. Global variables: theta_dynamic, theta_target, hysteresis, m_rate, dt_ema, rate_ticks, reliability, last_rate_timestamp, last_retrieval_ts, drift_accum. Weight naming rule: weight\_\* and \*\_weight variables (e.g., weight_relevance, mismatch_weight, weight_surprise) are control parameters; w\_\* variables (e.g., w_relevance, w_mismatch, ... w_arousal) are composite-score blender weights.

**Time deltas:** All Δt values, elapsed times (mem_elapsed, signal_gap, idle_for), and time comparisons operate in seconds:

Δt ← now_s() − to_s(last_rate_timestamp) \# seconds

mem_elapsed ← now_s() − to_s(t_start) \# seconds

signal_gap ← now_s() − to_s(last_signal_ts) \# seconds

**Time constants:** All time-related constants are specified with explicit units (e.g., τ_min = 120 seconds, kRecencyTau = 60 seconds). Threshold limits like max_mem_time(T) and gap_threshold(T) return values in seconds.

3.2 The Three-Knob Philosophy

Cortext is governed by three continuous control parameters, each representing a distinct dimension of cognitive regulation:

**Focus (F ∈ \[0, 1\]):** Perceptual selectivity and precision. Higher Focus narrows attention, increases relevance weighting, and reduces retrieval breadth. Focus modulates the trade-off between exploitation of known-relevant information and exploration of potentially useful context.

**Sensitivity (S ∈ \[0, 1\]):** Plasticity and affective gain. Higher Sensitivity accelerates learning, increases emotional and novelty responsiveness, and raises write-rate targets. Sensitivity governs how readily the system captures novel information and responds to salient stimuli.

**Stability (T ∈ \[0, 1\]):** Temporal persistence and inertia. Higher Stability lengthens memory half-lives, widens hysteresis bands, slows adaptive updates, and tightens safety bounds over time. Stability controls the resistance to change and the preservation of established knowledge.

A central design principle is that knobs set rates rather than modes. Behavioral differences emerge continuously from parameter interactions; there are no hard-coded phase transitions or discrete operational states.

3.3 Knob-Derived Parameters

All system tunables derive from the three primary knobs. This section catalogs the key derivations.

3.3.1 Context Windows and Temporal Scales

n_ctx(T) = round(lerp(32, 256, T))

win_score(T) = round(lerp(20, 120, T))

win_rate_s(T) = round(lerp(60, 300, T))

The context window n_ctx determines how many recent items inform relevance computation. The scoring window win_score controls the lookback for variance estimation and percentile calculation. The rate window win_rate_s specifies the temporal horizon (in seconds) for write-rate measurement.

3.3.2 Half-Life and Decay

Memory half-life follows a log-scale mapping to span multiple orders of magnitude:

τ_min = 120.0 seconds (2 minutes)

τ_max = 43200.0 seconds (12 hours)

base_half_life(T) = exp(ln(τ_min) + T × ln(τ_max / τ_min))

This exponential mapping ensures that low Stability yields half-lives near 2 minutes while high Stability approaches 12 hours, with smooth interpolation across the range.

3.3.3 Hysteresis and Rate Targets

band_min = 0.02; band_max = 0.25

base_band(T) = lerp(band_min, band_max, T)

r_min = 0.2; r_max = 5.0 (writes per minute)

base_rate(S) = lerp(r_min, r_max, S)

The hysteresis band prevents oscillation in threshold-crossing decisions. Write-rate targets establish homeostatic setpoints for the threshold controller.

3.3.4 Experiential Mass and Maturity

The system tracks accumulated experience through a maturity function that governs the annealing of safety bounds:

τ_m(T) = lerp(10.0, 200.0, T)

maturity(t) = 1 − exp(−count / τ_m(T))

where count is the total number of signals processed. This produces asymptotic approach to unit maturity, with higher Stability slowing the progression to reflect greater conservatism.

Safety bounds on the dynamic threshold anneal with maturity:

T_min(t) = lerp(0.01, 0.05, maturity(t))

T_max(t) = lerp(0.99, 0.95, maturity(t))

max_ΔT_per_min(t) = lerp(0.30, 0.10, maturity(t))

Early operation permits wide threshold excursions; mature operation constrains movement to a narrower band.

3.4 Uncertainty Estimation

Uncertainty u(t) ∈ \[0, 1\] modulates learning rates and evidence weighting. The raw uncertainty estimate blends multiple signals:

var_score_max = 0.25

var_recent_norm = clamp(var(scores\[t−w:t\]) / var_score_max, 0, 1)

coherence_complement = 1 − coherence_struct_t \# uses structural coherence

When prediction error signals are available, novelty and surprisal are blended:

novelty_surprise = blend(\[novelty_t, surprisal_t\],

weights = normalize(\[S, 1 − T\]))

The final raw uncertainty combines these components with knob-derived weights:

weights_u = normalize(\[S, F, 1 − T, S × (1 − T)\])

u_raw(t) = clamp(blend(\[var_recent_norm, focus_spread,

coherence_complement, novelty_surprise\],

weights = weights_u), 0, 1)

Smoothed uncertainty applies EWMA with a stability-dependent rate:

α_u(T) = 0.10 + (1 − T) × 0.60

u(t) = EWMA(u(t−1), u_raw(t), α = α_u(T))

When structural metrics are unavailable, the fallback is u_raw(t) = 1 − maturity(t), ensuring high uncertainty during early operation.

4\. Core Adaptation Algorithms

This section presents the algorithms governing adaptation along each of the three primary dimensions. Each algorithm consists of a prior computation (executed at initialization) and a dynamic update (executed per signal).

4.1 Focus-Driven Selectivity

Focus governs perceptual selectivity through relevance weighting and attention width.

4.1.1 Focus Priors

Given Focus knob F ∈ \[0, 1\], initialize Focus control variables:

weight_relevance = sigmoid(2F − 1)

coverage_gain_floor = 0.3 + 0.7F

mismatch_weight = 1 − F

attention_width = lerp(π, 0.1π, F)

The attention width (in radians) controls the angular spread of the receptive field in embedding space. High Focus produces narrow attention (0.1π), while low Focus permits broad capture (π).

4.1.2 Dynamic Focus Update

At each signal event t with input embedding x_t:

recent_context ← tail(signals, n_ctx(T))

observed_cosine ← cos(x_t, mean(recent_context))

weight_relevance ← EWMA(weight_relevance,

map01(observed_cosine), α = α_F(t))

where map01(z) = clamp((z + 1) / 2, 0, 1) transforms cosine values from \[−1, 1\] to \[0, 1\].

The learning rate α_F(t) is modulated by uncertainty:

α_min_F = 0.05; α_span_F = 0.45

α_F(t) = α_min_F + F × α_span_F × u(t)

High uncertainty increases learning rate, allowing faster adaptation when the environment is volatile. The Focus knob scales the uncertainty responsiveness.

4.2 Sensitivity-Driven Plasticity

Sensitivity governs learning speed, emotional responsiveness, and novelty capture.

4.2.1 Sensitivity Priors

Given Sensitivity knob S ∈ \[0, 1\], initialize Sensitivity control variables:

rate_target = lerp(0.2, 5.0, S) × (0.5 + 1.5S) \# writes/min

weight_novelty = 0.3 + 0.7S

weight_surprise = 0.2 + 0.8S

weight_valence = 0.4 + 0.6S

weight_arousal = S

emotion_gain = exp(1.5S)

score_gain = exp(2S)

4.2.2 Emotional Projection

When emotion category centroids are available, the system projects input embeddings onto a discrete emotion space C = {anger, fear, joy, love, sadness, surprise}. Inspired by Russell\'s (1980) circumplex model, each category maps to valence and arousal coordinates:

v_map = {anger: −0.9, fear: −0.8, sadness: −0.9,

joy: +0.9, love: +0.8, surprise: 0.0}

a_map = {anger: +0.9, fear: +0.9, sadness: +0.3,

joy: +0.6, love: +0.5, surprise: +0.8}

The projection procedure:

raw_cos_c ← cos(x_t, centroids\[c\]) for each c ∈ C

if all raw_cos_c ≤ 0:

emotion_intensity_t ← 0; valence_t ← 0.5; arousal_t ← 0

else:

logits_c ← max(0, raw_cos_c)

β(S) = 4 + 8S \# softmax inverse temperature

p_c ← softmax(β(S) × logits_c)

peak ← max_c(p_c)

confidence ← 1 − H(p_c) / ln(6)

emotion_intensity_t ← sqrt(peak × confidence)

valence_t ← (Σ_c p_c × v_map\[c\] + 0.9) / 1.8

arousal_t ← clamp(Σ_c p_c × a_map\[c\], 0, 1)

The emotion intensity combines peak probability with distributional confidence via geometric mean, providing a measure that is high only when a single emotion dominates with high certainty.

4.2.3 Threshold Modulation from Emotion

Emotional activation loosens write thresholds to capture salient moments:

κ_emo ← κ_base × S \# where κ_base = 0.10

ΔThreshold_emotion_t ← −κ_emo × emotion_intensity_t ×

(0.5 + 0.5 × arousal_t)

This negative adjustment makes writing more likely during emotionally salient events, consistent with McGaugh\'s (2004) findings on arousal-enhanced encoding.

4.2.4 Mood Integration

Distinct from instantaneous emotion, the mood state M_t ∈ ℝ⁶ maintains a persistent background affective tone as a 6-dimensional vector (one component per emotion category from Section 4.2.2):

α_mood(S) = lerp(0.01, 0.20, S) \# reactivity

λ_mood(T) = lerp(0.90, 0.999, T) \# decay

e_t ← p_c for each c ∈ C \# 6D emotion probability vector

M_t = λ_mood(T) × M\_{t−1} + α_mood(S) × e_t

M_t ← clamp_elementwise(M_t, −1.0, 1.0) \# per-component

Note that this update does not enforce coefficient normalization; the explicit elementwise clamp handles potential accumulation. The mood state provides a separate threshold bias via its normalized magnitude:

κ_mood ← κ_base × S

m_norm ← ‖M_t‖ / √6 \# max norm when all components at 1

ΔThreshold_mood_t ← −κ_mood × clamp(m_norm, 0, 1)

4.3 Stability-Driven Persistence

Stability governs temporal dynamics through half-life, decay rates, and hysteresis.

4.3.1 Stability Priors

Given Stability knob T ∈ \[0, 1\], initialize Stability control variables:

hysteresis = lerp(0.02, 0.25, T)

half_life = base_half_life(T)

rate_decay = lerp(0.60, 0.98, T)

periphery_half_life = clamp(0.5 × half_life,

τ_min, τ_max)

drift_weight = 0.5 × (1 − T)

4.3.2 Dynamic Stability Update

The stability update uses uncertainty-modulated learning rate and a stability-derived retention window:

α_min_T = 0.02; α_span_T = 0.18

α_T(t) = α_min_T + (1 − T) × α_span_T × u(t)

win_ret(T) = round(lerp(10, 50, T)) \# retention history window size

At each signal event, compute retention statistics and adjust half-life:

active_memories ← {m \| strength(m) ≥ periphery_cutoff(T)}

observed_retention ← mean_age(active_memories)

retention_ema_t ← EWMA(retention_ema\_{t−1},

observed_retention, α = α_T(t))

Compute z-score relative to recent retention history:

last_win_ret ← tail(retention_history, win_ret(T))

μ_ret ← mean(last_win_ret)

σ_ret ← max(std(last_win_ret), 1.0)

zscore_ret ← clamp((observed_retention − μ_ret) / σ_ret, −3, +3)

The target half-life incorporates feedback adjustment from the stability feedback mechanism (Section 9.3):

stability_adj ← ΔHalfLife_adj_t if provided else 0

target_half_life_t ← clamp(base_half_life(T) ×

(1 + 0.25 × zscore_ret + stability_adj),

τ_min, τ_max)

half_life_t ← EWMA(half_life\_{t−1}, target_half_life_t,

α = α_T(t))

5\. Structural Metrics and Composite Scoring

5.1 Embedding-Derived Metrics

5.1.1 Structural Coherence

Structural coherence (coherence_struct) measures integration of the current signal with the broader context window. This metric is distinct from memory coherence (coherence_mem, defined in Section 6.4.2) which tracks within-memory similarity:

raw ← var(\[cos(x_t, c) for c in recent_context\])

coherence_struct_t ← 1 − clamp(raw, 0, 1) \# range \[0, 1\]

High structural coherence (low variance in similarities) indicates the signal fits consistently with context. The effective Focus is modulated: F_eff = F × (0.5 + 0.5 × coherence_struct_t).

5.1.2 Focus Spread

Focus spread quantifies the entropy of attention over nearest neighbors:

k ← k_neighbors(T) = round(lerp(8, 32, T))

p ← softmax(kNN_similarities)

focus_spread_t ← H(p) / ln(k)

Values near 1 indicate diffuse attention; values near 0 indicate concentrated attention. The effective Focus is further modulated: F_eff ← F_eff × (1 − focus_spread_t).

5.1.3 Trajectory Drift

Drift measures directional change in context centroids:

drift_vec_t ← l2_normalize(mean(ctx_t)) −

l2_normalize(mean(ctx\_{t−k}))

drift_mag_t ← ‖drift_vec_t‖

Since both centroids are unit-normalized, drift_mag_t ∈ \[0, 2\]. A threshold determines episode boundaries:

drift_threshold ← lerp(0.10, 0.35, T)

if drift_mag_t \> drift_threshold:

trigger_episode_boundary()

5.1.4 Embedding Prediction Error

We measure surprisal as the deviation of the current embedding from the predicted trajectory in latent space:

Δx_t = x_t − x\_{t−1}

Δx_trend_t = EWMA(Δx_trend\_{t−1}, Δx_t, α=0.1)

x_pred_t = x\_{t−1} + Δx_trend\_{t−1}

prediction_error_t = 1 − cos(x_pred_t, x_t)

This error is normalized to produce the surprisal signal:

err_max = 0.5

surprisal_t ← clamp(prediction_error_t / err_max, 0, 1)

This formulation captures purely kinematic surprise in the thought process

5.2 Composite Score Computation

The system computes 12 metrics that blend into a composite write score:

  ----------------------------------------------------------------------------
  **Metric**         **Knob**        **Expression**
  ------------------ --------------- -----------------------------------------
  Relevance          ↑F              cos(x, μ_ctx) × (0.5 + F)

  Mismatch           ↓F, ↑S          (1 − F) × S × novelty

  Surprise           ↑S, ↓T          surprisal_t × S × (1 − T)

  Rarity             ↑F, ↓T          (1 − μ_sim) × (0.5 + 0.5F) × (1 − 0.2T)

  Drift              ↓T              (drift_mag / 2) × (1 − T)

  Utility            ↑F, ↓S          ΔSSE × (0.5 + 0.5F) × (1 − 0.3S)

  Salience           F, S            (rarity + novelty) / 2 × (F + S) / 2

  Valence            S, ↓T           map01(Σ p_c × v_map\[c\])

  Arousal            S, ↓T           clamp(Σ p_c × a_map\[c\], 0, 1)

  Contradiction      ↑S, ↓F          max(0, S − F)

  Periphery          ↑T              (1 − relevance) × T

  Coverage           ↑F              F × relevance
  ----------------------------------------------------------------------------

Table 1: Metric definitions and knob dependencies. Arrows indicate direction of influence.

5.3 Metric Weight Blending

Metric weights adapt online using recursive least squares (RLS) to minimize prediction error between composite scores and observed outcomes. Initial weights derive from bootstrap coefficients:

Blender weights are maintained as variables w\_\* (do not confuse these with control weights like weight_relevance or mismatch_weight). We denote the 12-element blender weight vector as:

W_blend = \[w_relevance, w_mismatch, w_surprise, w_rarity, w_drift,

w_utility, w_salience, w_valence, w_arousal, w_contradiction,

w_periphery, w_coverage\]

The index i used below (e.g., w_bootstrap\[i\], w_rls\[i\]) follows this ordering.

w_bootstrap\[i\] ← sigmoid(c_F\[i\]×F + c_S\[i\]×S + c_T\[i\]×T + d_i)

RLS fitting updates coefficients with stability-dependent forgetting:

φ(T) = 0.90 + 0.09T

N ← round(lerp(64, 512, T)) \# fitting window

The fitted weights blend with bootstrap weights based on RLS confidence:

τ_rls ← lerp(20.0, 80.0, T)

confidence_rls ← 1 − exp(−t / τ_rls)

weight_i(t) ← (1 − confidence_rls) × w_bootstrap\[i\] +

confidence_rls × w_rls\[i\]

5.3.1 Score Normalization

Composite score computation requires careful normalization:

for each metric i:

m01\[i\] = map01(metric\[i\]) if signed else clamp(metric\[i\], 0, 1)

weight_sum ← Σ weights\[i\]

if weight_sum \< ε: return 0

weights_norm\[i\] ← weights\[i\] / weight_sum

score ← clamp(Σ weights_norm\[i\] × m01\[i\], 0, 1)

Weight normalization is critical: with 12 metrics and raw weights averaging \~0.6, the sum approaches 7.2. Without normalization, weighted sums would saturate and collapse variance.

6\. Dynamic Thresholding and Homeostatic Control

The write gate compares composite scores against an adaptive threshold θ_dynamic (written as theta_dynamic when referring to the recorded variable). This section details the threshold evolution algorithm incorporating Bayesian prior-evidence blending and homeostatic rate control.

6.1 Prior-Evidence Blending

The threshold prior derives from knob settings:

θ_prior(F, S, T) = lerp(0.10, 0.30, T) × (1 − 0.3S)

Observed evidence comes from the 90th percentile of recent scores:

w ← win_score(T)

observed_p90 ← percentile(scores\[t−w:t\], 90)

Prior and evidence masses weight the blend:

ρ_prior ← prior_mass(T) = round(lerp(2, 32, T))

ρ_obs ← u(t) × min(w, count)

The target threshold blends prior and evidence:

θ_target ← (ρ_prior × θ_prior + ρ_obs × observed_p90) /

max(ε, ρ_prior + ρ_obs)

High Stability increases prior mass, making the system more resistant to observed deviations. High uncertainty increases evidence mass, allowing faster adaptation to volatile conditions.

6.2 Homeostatic Rate Control

The controller maintains write rates near the target setpoint through continuous-time estimation with effective sample size (ESS) reliability weighting.

6.2.1 Rate Estimation

Δt ← now_s() − to_s(last_rate_timestamp)

Δt ← max(Δt, 10⁻³) \# minimum 1ms

α_dt ← 1 − exp(−Δt / 1.0)

dt_ema ← (1 − α_dt) × dt_ema + α_dt × Δt

dt_base ← max(dt_ema, 1.0)

The rate time constant scales with Stability:

τ_rate ← max(2\^(3T) × dt_base, 1.0)

α ← 1 − exp(−Δt / τ_rate)

Instantaneous rate estimation with bias correction. Δwrites is the binary indicator (0 or 1) of whether a write occurred during the current timestep:

Δwrites ← 1 if write_memory else 0 \# binary write event

ρ_inst ← (Δwrites / Δt) × 60 \# writes per minute

m_rate ← (1 − α) × m_rate + α × ρ_inst

denom ← max(1 − (1 − α)\^(rate_ticks + 1), ε)

ρ_hat ← m_rate / denom \# bias-corrected estimate

6.2.2 Effective Sample Size

ESS estimates the effective number of independent samples in the EMA, using a heuristic inspired by Liu & Chen (1998):

β ← max(0, 1 − α)

ESS ← min((1 + β) / max(1 − β, ε), 100)

reliability ← 1 − exp(−ESS × (1 − T))

High Stability dampens reliability, preventing aggressive corrections in conservative regimes.

6.2.3 Homeostatic Correction

The rate error drives threshold adjustment:

rate_error ← tanh((ρ_hat − rate_target_t) /

max(rate_target_t, ε))

κ_r = 0.10 \# rate error gain

cap_homeo ← 0.25 × hysteresis

Δθ_homeo ← clamp(reliability × κ_r × (1 − T) ×

(1 − maturity(t)) × rate_error,

−cap_homeo, +cap_homeo)

The correction scales with reliability and is attenuated by both Stability and maturity, ensuring conservative, mature systems make minimal homeostatic adjustments.

6.2.4 Sensitivity-Based Threshold Adjustment

Sensitivity modulates threshold based on recent score volatility:

σ_scores ← std(scores\[t−w:t\])

κ_sens = 0.08 \# sensitivity gain

cap_sens ← 0.20 × hysteresis

Δθ_sens ← clamp(−κ_sens × S × (σ_scores − 0.1),

−cap_sens, +cap_sens)

High score variance with high Sensitivity lowers threshold, capturing more volatile signals.

6.2.5 Precision-Based Threshold Adjustment

Focus-driven precision tightens threshold when structural coherence is high:

κ_prec = 0.06 \# precision gain

cap_prec ← 0.15 × hysteresis

Δθ_prec ← clamp(κ_prec × F × (coherence_struct_t − 0.5),

−cap_prec, +cap_prec)

High structural coherence with high Focus raises threshold, enforcing stricter relevance filtering.

6.3 Threshold Integration

All threshold deltas combine and pass through safety limiting. The emotion and mood deltas from Section 4.2.3-4.2.4 are denoted Δθ_emo = ΔThreshold_emotion_t and Δθ_mood = ΔThreshold_mood_t:

Δθ_total ← Δθ_sens + Δθ_homeo + Δθ_prec + Δθ_emo + Δθ_mood

cap_total ← max_ΔT_per_min(t) × (Δt / 60.0)

Δθ_limited ← clamp(Δθ_total, −cap_total, +cap_total)

θ_dynamic_t ← clamp(EWMA(θ_dynamic\_{t−1}, θ_target,

α = α_T(t)) + Δθ_limited,

T_min(t), T_max(t))

Hysteresis evolves toward the stability-derived base:

hysteresis ← clamp(EWMA(hysteresis,

base_band(T), α = α_T(t)),

band_min, band_max)

6.4 Write Pacing and Memory Accumulation

The write gate operates per-signal, but coherent \"thoughts\" often span multiple signals. This section introduces memory-level accumulation that groups signals into natural units before storage decisions, inspired by Event Segmentation Theory (Zacks & Swallow, 2007).

This approach draws from EM-LLM (Fountas et al., 2024), which segments token sequences into episodic events using surprise-based boundary detection refined by graph-theoretic cohesion metrics. Their work shows that combining prediction error signals with within-segment coherence produces boundaries strongly correlated with human event perception. Our adaptation uses embedding drift as a proxy for surprise and cosine similarity for cohesion, enabling modality-agnostic operation across text tokens, audio chunks, video frames, or any signal stream.

6.4.1 Memory Accumulator State

Each source stream maintains accumulator state:

μ_acc ← 256d running mean embedding

drift_acc ← accumulated drift within memory

s_sum ← sum of signal scores in memory

s_max ← max signal score in memory

n ← count of signals in memory

e_peak ← embedding of highest-scoring signal

t_start ← now_ms() \# timestamp of accumulation start (ms since epoch)

last_signal_ts ← now_ms() \# timestamp of previous signal (ms, for gap detection)

On each signal, update running statistics (note: n is the count before this signal):

μ_acc ← (n × μ_acc + x_t) / (n + 1)

n ← n + 1

drift_acc ← drift_acc + drift_mag_t

s_sum ← s_sum + score_t

if score_t \> s_max:

s_max ← score_t; e_peak ← x_t

last_signal_ts ← now_ms() \# update timestamp for next gap calculation (ms)

6.4.2 Hybrid Drift and Coherence Tracking

Boundary detection combines kinematic drift with semantic coherence:

ε_noise = 0.02 \# noise floor for drift

d_step ← max(drift_mag_t − ε_noise, 0)

eta_acc ← EWMA(eta_acc, d_step, α = lerp(0.3, 0.1, T))

Memory coherence tracks similarity within the current memory accumulation window (range \[−1, 1\] from mean cosine):

coherence_prev ← mean(\[cos(x_t, x_i) for x_i in current_window\])

Note: coherence_mem is distinct from coherence_struct (Section 5.1.1). The former tracks within-memory similarity using raw mean cosine, while the latter measures variance-based integration with broader context. This dual-signal approach mirrors EM-LLM\'s boundary detection mechanism. In their formulation, boundaries occur where surprise (token-level prediction error) exceeds a threshold and segment cohesion drops. Our drift spike approximates surprise via embedding-space velocity, while coherence_mem drop captures within-memory similarity degradation.

6.4.3 Natural Boundary Detection

Boundary score combines drift spike and coherence drop:

weight_drift_component = lerp(0.6, 0.4, T) \# weight on drift

weight_coh_component = 1 − weight_drift_component \# weight on coherence drop

drift_spike ← (d_step − eta_acc) / max(eta_acc, ε)

coh_drop ← max(0, coherence_prev − coherence_curr)

boundary_score ← weight_drift_component × sigmoid(drift_spike) +

weight_coh_component × coh_drop

Boundary threshold and limits:

b_thresh(F, S) = lerp(0.4, 0.7, F) × lerp(1.1, 0.9, S)

max_mem_time(T) = lerp(30, 120, T) \# seconds

max_mem_drift(S) = lerp(0.8, 2.0, S) \# cumulative drift cap

Trigger memory flush when:

mem_elapsed ← now_s() − to_s(t_start)

should_flush = (boundary_score \> b_thresh(F, S)) OR

(mem_elapsed \> max_mem_time(T)) OR

(drift_acc \> max_mem_drift(S)) OR

(signal_gap \> gap_threshold(T))

where signal_gap = now_s() − to_s(last_signal_ts) detects natural pauses (speech pauses, generation delays):

gap_threshold(T) = lerp(5, 30, T) \# seconds

6.4.4 Spike Bypass (Flashbulb Flush)

High-salience signals bypass accumulation and flush immediately, capturing preceding context as a coherent memory unit:

spike_margin(S) = lerp(0.3, 0.15, S) \# above θ_dynamic

spike_bypass = score_t \> (θ_dynamic + spike_margin(S))

When spike_bypass triggers:

if spike_bypass AND n \> 1:

commit_memory() \# includes spike signal

reset_accumulator()

This ensures flashbulb moments capture their surrounding context rather than creating isolated micro-memories.

6.4.5 Window Score and Refractory

Memory-level score combines peak and average with coverage bonus:

s_avg ← s_sum / max(n, 1)

α(F) = lerp(0.3, 0.7, F) \# peak vs avg weight

coverage ← min(n / n_ctx(T), 1.0) \# memory completeness

β(S) = lerp(0.05, 0.15, S) \# coverage weight

S_window ← α(F) × s_max + (1 − α(F)) × s_avg + β(S) × coverage

Write refractory suppresses rapid successive writes:

τ_write_refrac(T) = lerp(5, 30, T) \# seconds

k_write_refrac = lerp(0.3, 0.1, T)

Δt_write ← now_s() − to_s(last_write_ts)

M_write_refrac ← 1.0 + k_write_refrac × exp(−Δt_write / τ_write_refrac(T))

Final write decision:

θ_memory ← θ_dynamic × M_write_refrac

write_memory = (should_flush OR spike_bypass) AND (S_window \> θ_memory)

Trace note: if a run trace reports both write_decision and stored, interpret write_decision as the boolean gate outcome at the boundary (the write_memory predicate above). Interpret stored as the eventual recording outcome (after any final safety checks).

Representative embedding blends accumulator mean with peak:

ρ(F) = lerp(0.3, 0.7, F) \# mean vs peak blend

e_rep ← l2_normalize(ρ(F) × μ_acc + (1 − ρ(F)) × e_peak)

On write: store e_rep with metadata {n, s_max, s_avg, drift_acc, mem_elapsed}. Reset accumulator for next unit.

7\. Reinforcement and Decay Dynamics

7.1 Memory Strength Model

Each memory m maintains a strength value updated through use-frequency tracking and exponential decay. The update uses a sensitivity-modulated learning rate and a binary usage indicator:

α_min_S = 0.05; α_span_S = 0.35

α_S(t) = α_min_S + S × α_span_S × u(t)

used_flag(m) = 1 if m was retrieved and used in current step, else 0

use_frequency_t ← EWMA(use_frequency\_{t−1},

used_flag(m), α = α_S(t))

λ_t ← ln(2) / half_life_t

strength_t ← strength\_{t−1} × exp(−λ_t × Δt) + S × use_frequency_t

Memories falling below the periphery cutoff are candidates for eviction:

periphery_cutoff(T) = lerp(0.05, 0.25, T)

if strength_t \< periphery_cutoff(T): evict(m)

7.2 Influence-Weighted Updates

When contextual gain signals are available, influence factors modulate reinforcement:

influence_factor ← (used_count / max(retrieved_count, 1)) ×

clamp(contextual_gain(m), −1, +1)

strength_t ← strength\_{t−1} × exp(−λ_t × Δt) +

S × use_frequency_t + F × influence_factor

7.3 Causal Feedback Loop

The system tracks causal influence of retrieved memories on generation quality through contextual gain---the improvement in prediction accuracy attributable to including each memory in context.

7.3.1 Focus Feedback

αF_base = 0.10; βF_base = 0.05

for each used memory m:

if contextual_gain(m) \> 0:

weight_relevance += αF_base × contextual_gain(m)

attention_width_t \*= (1 − βF_base)

else:

attention_width_t \*= (1 + βF_base)

weight_relevance ← clamp(weight_relevance, 0, 1)

attention_width_t ← clamp(attention_width_t,

attention_width_min, attention_width_max)

Positive contextual gain narrows attention and boosts relevance weighting; negative gain widens attention to explore alternatives.

7.3.2 Sensitivity Feedback

η_base = 0.10

for each used memory m:

novelty_reward ← 1 − sim(m.embedding, recent_context)

weight_novelty_t += η_base × (novelty_reward ×

contextual_gain(m) −

redundancy(m, recent_context))

weight_novelty_t ← clamp(weight_novelty_t, 0, 1)

This rewards novelty that proves useful while penalizing redundant retrievals.

7.3.3 Stability Feedback

γT_base = 0.05

for each used memory m:

if contextual_gain(m) \> 0:

stability(m) += γT_base

else:

stability(m) \*= (1 − γT_base)

The mean stability of used memories provides adjustment to the half-life target:

adj ← clamp(mean(stability(m_used)) − 1.0, −0.25, +0.25)

ΔHalfLife_adj_t ← adj

This factor is consumed by the Stability update (Section 4.3.2), avoiding conflicting adjustments between feedback mechanisms.

7.4 Generation Influence Tracking

When generation embeddings are available, influence incorporates output trajectory:

Δḡ ← l2_normalize(ḡ_t) − l2_normalize(ḡ\_{t−1})

drift_mag_gen ← ‖Δḡ‖

drift_contribution(m) ← (drift_mag_gen / 2) ×

max(0, cos(m.embedding, l2_normalize(Δḡ)))

Total influence blends contextual gain, generation similarity, and drift contribution:

λ₁ = 0.5; λ₂ = 0.4; λ₃ = 0.3

influence(m) ← λ₁ × contextual_gain(m) +

λ₂ × cos(m.embedding, ḡ_t) −

λ₃ × drift_contribution(m)

Sustained influence accumulates over a stability-dependent horizon:

L_sustain(T) = round(lerp(3, 5, T))

sustained_influence ← EWMA(sustained_influence,

influence(m),

α = 2 / (L_sustain(T) + 1))

8\. Advanced Cognitive Processes

This section presents algorithms modeling higher-order cognitive phenomena: working memory maintenance, metacognitive monitoring, reconsolidation dynamics, and serial position effects.

8.1 Working Memory Gates

Following Cowan\'s (2001) capacity constraints, working memory maintains a limited number of active items. Working memory holds coherent memories as defined in Section 6.4, preserving the full content and signal sequence:

base_capacity = round(lerp(5, 3, S) + lerp(−1, 1, F))

This yields a range of approximately 2-6 memories, broadening the 4±1 chunk limit to accommodate task-dependent requirements. High Sensitivity reduces capacity (faster turnover), while high Focus modulates breadth.

8.1.1 Active Memory Structure

Each active memory holds a coherent memory with its full content and metadata:

memory.content ← concatenated signal blobs

memory.source_id ← source identifier (e.g., \'user\', \'assistant\')

memory.modality ← primary modality (\'text\', \'audio\', \'image\')

memory.blob_ids ← \[blob_1, blob_2, \..., blob_n\] \# blob refs

memory.embedding ← e_rep \# representative embedding (Section 6.4.5)

memory.signals ← \[x_1, x_2, \..., x_n\] \# ordered signal embeddings

memory.metadata ← {n, s_max, s_avg, drift_acc, mem_elapsed,

s_emotion_max, s_arousal_avg}

The emotional metrics (s_emotion_max, s_arousal_avg) are accumulated during memory formation for use by Emotional Consolidation (Section 8.7).

8.1.2 Maintenance Cost

Maintenance incurs cognitive cost:

maintenance_cost_per_memory = lerp(0.05, 0.15, S)

complexity_penalty = manifold_complexity × lerp(0.5, 1.5, S)

The manifold_complexity represents local variance in the embedding stream: 1 - mean(cos(window)).

8.1.3 Memory-Level Gating

Gating thresholds determine entry and chunking:

chunking_threshold = lerp(0.7, 0.9, F)

gate_threshold = lerp(0.1, 0.4, F)

rehearsal_rate = lerp(0.5, 2.0, S)

memory_dedication_strength = lerp(0.3, 0.9, T)

Working memory gating evaluates coherent memories at accumulation boundaries (Section 6.4.3), not individual signals:

on_memory_boundary:

memory_benefit ← α × S_window + β × relevance(μ_acc, task_context) +

γ × novelty(μ_acc, active_memories)

margin ← memory_benefit − gate_threshold

total_cost ← maintenance_cost_per_memory × \|active_memories\| + complexity_penalty

accept_memory = (margin ≥ total_cost)

Note that total_cost is computed from existing active memories only, not the prospective new memory. This avoids a bootstrap problem where empty working memory would require unreasonably high benefit scores to accept the first item.

8.1.4 Chunking at Memory Level

Chunking operates on memory embeddings to merge related content from the same source:

similar_memories ← {m ∈ active_memories \| cos(m.embedding, memory.e_rep) \> chunking_threshold AND

m.source_id == memory.source_id}

if \|similar_memories\| \> 0:

merge_into_chunk(similar_memories, memory)

8.2 Metacognitive Monitoring

The system implements feeling-of-knowing (FOK) and tip-of-tongue (TOT) detection following Hart\'s (1965) framework:

FOK_threshold = lerp(0.2, 0.5, F)

TOT_detection = (FOK \> lerp(0.5, 0.8, F)) AND

(retrieval_strength \< lerp(0.4, 0.2, F))

TOT occurs when metacognitive confidence is high but retrieval strength is low---the characteristic experience of knowing one knows something but being unable to access it.

Additional metacognitive parameters:

confidence_decay_rate = lerp(0.01, 0.1, 1 − T)

unknown_threshold = lerp(0.3, 0.1, F)

strategy_switch_latency = lerp(500, 100, S) \# ms

certainty_requirement = lerp(0.6, 0.9, T)

metacognitive_sensitivity = F × (1 + 0.5 × S)

8.3 Memory Reconsolidation

Following Nader et al. (2000), retrieved memories enter a labile state permitting modification:

τ_labile = lerp(30, 300, T) \# seconds

reconsolidation_gain = lerp(0.2, 0.02, T)

lability_susceptibility = (1 − T) × (0.5 + 0.5 × S)

During the lability window, memories can drift toward current context:

drift_magnitude = (1 − T) × S × lability ×

contextual_relevance

Reconsolidation effects propagate to semantically related memories with decay:

ripple_decay = lerp(0.5, 0.1, T) \# per semantic hop

8.4 Retrieval Competition

Retrieved memories compete through lateral inhibition, modeling retrieval-induced forgetting (Anderson et al., 1994):

inhibition_radius = lerp(0.5, 0.85, F)

winners_k = round(lerp(7, 3, F))

suppression_per_retrieval = lerp(0.1, 0.01, T) ×

(1 − winning_activation)

recovery_time_RIF = lerp(300, 1800, T) \# seconds

High Focus produces narrow winner-take-all dynamics; low Focus permits broader activation.

8.5 Predictive Pre-activation

The system pre-activates memories predicted to be relevant based on trajectory extrapolation:

prediction_horizon = round(lerp(2, 8, F))

pre_activation_decay = lerp(0.7, 0.3, T)

prediction_conf_threshold = lerp(0.3, 0.7, F)

surprise_sensitivity = S × lerp(2.0, 0.5, T)

When predictions fail (high surprise), the system updates its trajectory model:

update_rate_on_surprise = lerp(0.2, 0.02, T) × S

8.6 Serial Position Effects

The architecture models primacy, recency, and distinctiveness effects observed in human memory (Murdock, 1962):

primacy_window = round(lerp(5, 2, F))

primacy_bonus = lerp(1.2, 2.0, S)

recency_window = round(lerp(7, 3, F))

rehearsal_curve_depth = lerp(0.2, 0.6, S)

The von Restorff (isolation) effect enhances memory for distinctive items (Hunt, 1995):

distinctiveness_threshold = lerp(0.6, 0.8, F)

von_restorff_multiplier = lerp(1.5, 3.0, S)

Items in the middle region suffer interference:

interference_zone = positions\[primacy_window+1 : −recency_window\]

middle_suppression = lerp(0.8, 0.5, S) × (1 − F)

8.7 Emotional Consolidation

High-emotion events trigger enhanced consolidation, following McGaugh\'s (2004) findings. As detailed in Section 9.1.1, consolidation operates on stored memory metadata:

θ_intensity = lerp(0.6, 0.8, 1 − S)

θ_arousal = lerp(0.4, 0.2, S)

\# Consolidation uses stored memory emotional metadata

trigger = (m.metadata.s_emotion_max ≥ θ_intensity) AND

(m.metadata.s_arousal_avg ≥ θ_arousal)

Flashbulb memories receive extended half-life bonuses based on the memory\'s peak emotional intensity:

flashbulb_threshold = lerp(0.9, 0.4, S)

\# Half-life bonus uses stored memory emotional peak

emotional_half_life_bonus = exp(lerp(0, ln(3), S)) ×

(1 + m.metadata.s_emotion_max)

The emotional metrics (s_emotion_max, s_arousal_avg) are accumulated during memory formation and stored with the memory (Section 8.1.1).

cascade_radius = round(lerp(1, 5, S))

cascade_decay = lerp(0.7, 0.3, S)

9\. Consolidation and Graph Integration

The consolidation system transforms episodic memories into semantic structures through clustering, summarization, and knowledge graph construction. Memory-level storage (Section 6.4) ensures each embedding represents a coherent unit.

9.1 Consolidation Triggers

Consolidation operates on stored memory representatives (e_rep from Section 6.4.5), not individual signals. It activates under capacity, rate, or temporal conditions:

should_consolidate = (memory_count \> consolidation_threshold) OR

(m_rate \< rate_target_t / 2) OR

(elapsed_time \> consolidation_interval)

The consolidation rate adapts to Stability and Sensitivity (write_rate tracks memory writes, not signal writes):

rate_consolidate = (1 / max(consolidation_interval, 1)) ×

(0.3 + 0.7T) × (1 − 0.5S)

9.1.1 Activity-Aware Scheduling

Consolidation runs during idle periods and preempts for retrieval. The is_accumulating_memory check ensures consolidation doesn\'t interrupt mid-memory accumulation:

idle_required(T) = round(0.25 × win_rate_s(T))

idle_for_s = now_s() − to_s(last_retrieval_ts)

\# Consolidation waits for memory completion, not just signal arrival

should_start = (NOT is_accumulating_memory) AND

(retrieval_queue_depth == 0) AND

(idle_for_s ≥ idle_required(T))

Consolidation begins only after the current memory has been flushed (Section 6.4.3) and the idle period has elapsed. On retrieval events, consolidation pauses, commits micro-batches, and resumes when idle.

9.2 Consolidation Scoring

Each stored embedding represents a memory representative (e_rep from Section 6.4.5). Each memory receives a consolidation score determining merge priority:

score_consolidate(m) = weight_strength × strength(m) −

weight_redundancy × redundancy(m) +

weight_connectivity × connectivity(m) +

weight_stability × stability(m)

Weights derive from knobs:

weight_strength = T; weight_redundancy = F; weight_connectivity = S; weight_stability = T

Low-scoring memories are marked for merging.

9.3 Clustering and Summarization

Marked memories cluster via density-based methods (e.g., DBSCAN) or k-means using embedding similarity:

cluster_i = {m_j \| cos(m_j, μ_i) \> merge_threshold}

μ_i = centroid(cluster_i)

Summary nodes replace clusters:

summary.embedding = μ_i

summary.blob = summarize(fetch_blobs(cluster_i))

summary.metadata.sources = \[m.id for m in cluster_i\]

9.4 Semantic Extraction

For sufficiently large clusters, semantic extraction identifies entities and relations:

extraction_batch_size = round(lerp(8, 32, T))

min_cluster_size = round(lerp(3, 10, F))

entity_frequency_threshold = round(lerp(5, 15, T))

extraction_interval = lerp(300, 3600, T) \# 5 min → 1 hour

max_extractions_per_cycle = round(lerp(20, 5, T))

Extraction uses structured prompting to identify named entities (people, places, organizations, concepts) and relationships (co-occurrence, implication, contradiction).

9.5 Knowledge Graph Construction

The graph comprises three node types:

-   **Memory Nodes:** Summarized embeddings from merged clusters

-   **Entity Nodes:** Named entities extracted from content blobs

-   **Concept Nodes:** Emergent centroids representing recurrent topics

Edge types capture relationships:

-   **co_occurs:** Shared context or temporal proximity

-   **implies:** Directional correlation in embedding drift

-   **causes:** Directional correlation in embedding drift

-   **contradicts:** Strong negative similarity or constraint violation

-   **reinforces:** Frequent joint retrieval

-   **derived_from:** Links summaries to source memories

9.5.1 Edge Construction

Co-occurrence edges derive from embedding similarity:

for (m_i, m_j) in cluster.sources:

cos_sim ← cos(m_i.embedding, m_j.embedding)

if cos_sim \> lerp(0.85, 0.95, F):

create_edge(m_i, m_j, \'co_occurs\', cos_sim)

Causal edges derive from temporal drift:

temporal_order ← sort_by_timestamp(cluster.sources)

for i in range(len(temporal_order) − 1):

m_i, m_j ← temporal_order\[i\], temporal_order\[i+1\]

drift_vec ← m_j.embedding − m_i.embedding

drift_mag ← ‖drift_vec‖

if drift_mag \> lerp(0.15, 0.35, T):

create_edge(m_i, m_j, \'causes\', drift_mag)

9.6 Graph-Augmented Retrieval

Retrieval combines vector similarity with graph expansion. Both initial retrieval and re-ranking use the memory centroid (μ_acc from Section 6.4.1) as the query vector, ensuring retrieved memories are ranked by relevance to the overall context rather than momentary signal fluctuations:

\# Query and re-rank using current memory centroid

q ← μ_acc \# memory centroid from Section 6.4.1

results_vec ← topK(vector_search(q, k=kNN_size))

seed_nodes ← \[r.id for r in results_vec\]

expanded_nodes ← graph.traverse(seed_nodes, depth=graph_depth)

combined ← union(seed_nodes, expanded_nodes)

re_ranked ← sort_by(cos(q, embeddings(combined))) \# re-rank by memory centroid

Graph expansion uses recursive traversal with depth limits to find related context that pure vector search might miss. Using the memory centroid maintains consistency between vector search and graph expansion results.

10\. Interrupt Gate and Streaming Integration

The interrupt gate controls when retrieved memories enter active context during streaming generation. The gate balances novelty value against disruption cost. The interrupt gate operates on memory-level context, using centroids (μ_acc) rather than individual signal embeddings for novelty and relevance computation.

10.1 Marginal Utility Computation

Novelty thresholds scale with knobs and refractory state:

τ_novelty = lerp(0.10, 0.35, F) × (1 − 0.15S) × (1 + 0.3T)

τ_mu = lerp(0.08, 0.18, F) × (1 − 0.4S) × (1 + 0.4T)

retrieval_thresh(F) = lerp(0.25, 0.60, F)

Refractory dynamics suppress rapid successive interrupts:

Δ = cumulative_drift_since_last_interrupt

τ_refrac = lerp(24, 96, T) × lerp(1.4, 1.0, S)

k_refrac = lerp(0.20, 0.05, T) × lerp(0.8, 1.2, F)

M_refrac = 1.0 + k_refrac × exp(−Δ / τ_refrac)

Effective thresholds incorporate refractory pressure:

τ_novelty_eff = τ_novelty × M_refrac

τ_mu_eff = τ_mu × M_refrac

10.2 Marginal Utility Score

The marginal utility (MU) of a candidate memory combines four factors. Context comparisons use memory centroids rather than individual signal embeddings:

\# Context window contains recent memory centroids, not individual signals

ctx_window ← recent_memory_centroids \# deque of μ_acc values

ctx_centroid ← mean(ctx_window) \# centroid of recent memory centroids

weights_mu_raw = \[lerp(0.40, 0.60, F), \# coverage gain

lerp(0.35, 0.25, F), \# relevance

lerp(0.15, 0.25, S), \# redundancy penalty

lerp(0.15, 0.25, S)\] \# incoherence penalty

\[weight_cov, weight_rel, weight_red, weight_incoh\] = normalize(weights_mu_raw)

mu = weight_cov × coverage_gain(candidate \| included_set) +

weight_rel × cos(candidate, ctx_centroid) −

weight_red × redundancy(candidate, included_set) −

weight_incoh × (1 − coherence_struct_t) \# structural coherence penalty

10.3 Gate Decision Logic

Duplicate suppression threshold:

dup_thresh = lerp(0.88, 0.96, F) × (0.98 + 0.02T)

K = round(lerp(10, 6, F)) \# candidates to evaluate

10.3.1 Write Exclusion Filter

Memories stored during the current accumulation unit are excluded from interrupt consideration to prevent self-triggering. Using the accumulation start timestamp ensures all memories written within the current unit are excluded:

\# Exclude memories written during current accumulation to prevent self-triggering

write_exclusion_ts ← t_start \# start timestamp from Section 6.4.1

candidates_eligible ← {c ∈ candidates \| c.created_at \< write_exclusion_ts}

This filter is applied before novelty and marginal utility evaluation. All subsequent gate logic operates on candidates_eligible rather than the raw candidate set, preventing recursive triggering within a coherent thought unit.

Boundary-aware override permits lower-threshold interrupts at natural boundaries:

boundary_mult = lerp(1.3, 2.0, F) × lerp(1.1, 0.9, S)

The gate permits interrupt when:

embedding_novelty = 1 − max(cos(candidate, ctx_window))

allow_interrupt =

(max_relevance ≥ retrieval_thresh(F)) AND

(embedding_novelty ≥ τ_novelty_eff OR best_mu ≥ τ_mu_eff) AND

(max_semantic_overlap \< dup_thresh) AND

(at_drift_boundary OR best_mu ≥ boundary_mult × τ_mu_eff)

This logic suppresses low-drift interrupts unless the marginal utility substantially exceeds threshold, while permitting normal-threshold interrupts at natural transition points.

10.4 Streaming Pacing

Streaming retrieval is gated by cumulative drift rate within the accumulation unit. Retrieval checks trigger when drift exceeds threshold or at boundaries:

\# Pacing tracks drift within current memory formation

where cosine_dist(u, v) = 1 − cos(u, v).

drift_acc += cosine_dist(x_t, x\_{last_check})

pacing_thresh(S) = lerp(0.5, 0.1, S)

\# Retrieval triggered when drift exceeds threshold or at memory boundary

if drift_acc \> pacing_thresh(S) OR should_flush:

trigger_check()

max_wait_drift(F) = lerp(2.0, 0.5, F)

max_results(F) = round(lerp(64, 4, F))

adjacent_window(F) = round(lerp(8, 1, F))

High Sensitivity produces frequent checks triggered by small content shifts; high Focus enforces strict drift limits. Memory boundaries (Section 6.4.3) also trigger retrieval checks to ensure context updates align with natural thought transitions.

11\. Experimental Results

We present preliminary experimental results collected from live chat sessions to validate the adaptive mechanisms.

11.1 Threshold Adaptation

The dynamic threshold (θ_dynamic) successfully tracked score distributions. In high-volatility inputs (drift_accum \> 1.0), thresholds relaxed to \~0.15, while stable contexts tightened to \~0.27.

11.2 Boundary Detection

Accumulator drift (drift_acc) aligned with semantic shifts. Conversation turns with distinct topics triggered flushes (boundary_score \> 0.3) while coherent continuations remained accumulated.

11.3 Latency and Performance

End-to-end processing per token averaged \< 50ms. Graph expansion added \< 10ms overhead due to efficient kNN (k=32) and limited expansion depth (d=2).

12\. Implementation Considerations

12.1 Computational Complexity

Per-signal operations are dominated by embedding similarity computations. With n memories and d-dimensional embeddings:

-   Exact kNN: O(nd) per query

-   Approximate kNN (HNSW): O(d log n) per query

-   Composite scoring: O(1) per signal (fixed 12 metrics)

-   RLS weight update: O(m²) where m = 12 metrics

For stores exceeding 100,000 memories, approximate nearest neighbor indices (HNSW, IVF-PQ) become essential. A small exact cache covering the most recent n_ctx(T) items handles recency-biased queries efficiently.

12.2 Execution Cadence

Operations partition by frequency:

-   **Per-signal:** Focus/Sensitivity/Stability updates, threshold evolution, strength decay, interrupt gating

-   **Every K signals (K ≈ 3):** RLS weight fitting, heavy kNN computations, entropy estimation

-   **Per-episode boundary:** Batch writes, cache invalidation, episode ID rollover

-   **Periodic background:** Consolidation, ANN index maintenance, graph construction

12.3 State Representation

System variables partition into logical components for efficient resumption:

-   **Processor Variables:** Global parameter variables storing all evolving parameters (maturity, uncertainty, threshold, hysteresis, learning rates)

-   **Blender Weights:** 12-element weight vector and RLS covariance matrix

-   **Recent Context:** Rolling window of embedding vectors

-   **Recent Scores:** Rolling window for threshold adaptation

At initialization, the system can restore stored state and resume processing seamlessly.

13\. Discussion and Conclusion

13.1 Summary of Contributions

This paper has presented Cortext, a three-knob cognitive memory architecture that achieves adaptive behavior through continuous parameter modulation rather than discrete mode switching. The key contributions are:

5.  **Principled parameter derivation:** All system tunables trace to three primary knobs (Focus, Sensitivity, Stability) through explicit mathematical transformations, reducing reliance on fixed constants and providing interpretable control surfaces.

6.  **Self-calibrating priors:** The Bayesian prior-evidence blending mechanism allows the system to balance initial assumptions against accumulated experience, with the blend ratio itself governed by uncertainty estimation.

7.  **Homeostatic control:** The threshold controller maintains target write rates through continuous-time estimation with effective sample size reliability weighting, providing stable regulation across varying signal rates.

8.  **Cognitive fidelity:** The architecture incorporates established cognitive science findings---working memory capacity limits, reconsolidation dynamics, serial position effects, emotional modulation---within a computationally tractable framework.

9.  **Graph-augmented retrieval:** The consolidation system transforms episodic memories into semantic structures, enabling retrieval that combines embedding similarity with structural graph traversal.

13.2 Emergent Developmental Progression

A notable property of the architecture is that developmental phases emerge from parameter interactions rather than explicit programming:

-   **Early operation:** High uncertainty, light priors, wide safety bounds, rapid capture, permissive thresholds. The system behaves with high plasticity, quickly incorporating novel information.

-   **Intermediate operation:** Balanced learning, stabilizing weights, selective attention. The system becomes more discriminating while retaining adaptability.

-   **Mature operation:** Strong priors, narrow bounds, slow adaptation, high precision. The system exhibits expert-like behavior with reliable, stable retrieval.

These transitions arise naturally from the annealing of safety bounds (T_min, T_max, max_ΔT_per_min) and the accumulation of experiential mass (ρ_obs vs ρ_prior), without requiring explicit phase detection or switching logic.

13.3 Limitations

Several limitations warrant acknowledgment:

-   **Embedding dependence:** System behavior depends critically on embedding quality. Poor embeddings will produce poor coherence, novelty, and retrieval signals regardless of knob settings.

-   **Single-agent assumption:** The current architecture assumes single-user operation. Multi-agent or collaborative scenarios would require extensions for shared memory spaces and conflict resolution.

-   **Offline evaluation:** While the algorithms are fully specified, empirical validation on diverse task domains remains ongoing work.

-   **Extraction latency:** Semantic extraction for graph construction introduces latency during consolidation. The background scheduling mitigates but does not eliminate this cost.

13.4 Future Directions

Several directions merit further investigation:

-   **Meta-learning knob adaptation:** Learning optimal knob settings for specific task distributions through reinforcement or evolutionary optimization.

-   **Multimodal integration:** Extending the architecture to handle heterogeneous modalities (text, image, audio) through unified embedding spaces or modality-specific sub-systems.

-   **Distributed deployment:** Scaling the architecture across multiple nodes while maintaining consistency guarantees and low-latency retrieval.

-   **Prosthetic applications:** Adapting the architecture for assistive technology applications, particularly memory augmentation for individuals with cognitive impairment.

13.5 Conclusion

Cortext demonstrates that sophisticated adaptive memory behavior can emerge from a small set of principled control parameters. By grounding all system dynamics in three interpretable knobs---Focus, Sensitivity, and Stability---the architecture provides both theoretical clarity and practical tunability. The integration of cognitive science findings with modern embedding-based retrieval and knowledge graph construction offers a path toward AI systems with more human-like memory characteristics.

The formal specification provided here enables direct implementation while the modular design permits selective adoption of individual components. We hope this work contributes to the broader goal of building AI systems that learn and remember in ways that align with human cognitive architecture.

References

Anderson, M. C., Bjork, R. A., & Bjork, E. L. (1994). Remembering can cause forgetting: Retrieval dynamics in long-term memory. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 20*(5), 1063-1087.

Åström, K. J., & Murray, R. M. (2008). *Feedback systems: An introduction for scientists and engineers*. Princeton University Press.

Baddeley, A. (2000). The episodic buffer: A new component of working memory? *Trends in Cognitive Sciences, 4*(11), 417-423.

Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. *Behavioral and Brain Sciences, 24*(1), 87-114.

Cowan, N. (2010). The magical mystery four: How is working memory capacity limited, and why? *Current Directions in Psychological Science, 19*(1), 51-57.

Fountas, Z., et al. (2024). Event segmentation in large language models. *arXiv preprint arXiv:2407.03158*.

Hart, J. T. (1965). Memory and the feeling-of-knowing experience. *Journal of Educational Psychology, 56*(4), 208-216.

Hunt, R. R. (1995). The subtlety of distinctiveness: What von Restorff really did. *Psychonomic Bulletin & Review, 2*(1), 105-112.

LaBar, K. S., & Cabeza, R. (2006). Cognitive neuroscience of emotional memory. *Nature Reviews Neuroscience, 7*(1), 54-64.

Liu, J. S., & Chen, R. (1998). Sequential Monte Carlo methods for dynamic systems. *Journal of the American Statistical Association, 93*(443), 1032-1044.

McClelland, J. L., McNaughton, B. L., & O\'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review, 102*(3), 419-457.

McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation, 24*, 109-165.

McGaugh, J. L. (2004). The amygdala modulates the consolidation of memories of emotionally arousing experiences. *Annual Review of Neuroscience, 27*, 1-28.

Miller, G. A. (1956). The magical number seven, plus or minus two: Some limits on our capacity for processing information. *Psychological Review, 63*(2), 81-97.

Murdock, B. B. (1962). The serial position effect of free recall. *Journal of Experimental Psychology, 64*(5), 482-488.

Nader, K. (2003). Memory traces unbound. *Trends in Neurosciences, 26*(2), 65-72.

Nader, K., Schafe, G. E., & Le Doux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature, 406*(6797), 722-726.

Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology, 39*(6), 1161-1178.

Tulving, E. (1972). Episodic and semantic memory. In E. Tulving & W. Donaldson (Eds.), *Organization of memory* (pp. 381-403). Academic Press.
