**Cortext: A Three-Knob Adaptive Memory Architecture**

**Algorithmic Specification**

**Abstract**

This document details the complete algorithmic specification for
Cortext, a biologically-inspired adaptive memory system. It serves as a
technical reference for implementation, focusing purely on mathematical
definitions, signal processing flows, and adaptive control logic
governed by the three primary parameters: Focus (F), Sensitivity (S),
and Stability (T).

1\. Mathematical Foundations

1.1 Notation and Primitives

We establish the following notation used throughout this paper. Let ε =
10⁻⁶ denote a small constant for numerical stability. All knob values F,
S, T lie in the closed interval \[0, 1\].

Core mathematical primitives:

lerp(a, b, x) = a + (b − a) × x

clamp(v, lo, hi) = max(lo, min(v, hi))

sigmoid(z) = 1 / (1 + exp(−z))

EWMA(prev, x, α) = (1 − α) × prev + α × x

For vectors, we define cosine similarity as cos(u, v) = u·v / (‖u‖ ×
‖v‖), and safe L2 normalization as l2_normalize(v) = v / max(‖v‖, ε).
Shannon entropy is computed in nats: H(p) = −Σᵢ pᵢ ln(pᵢ).

The temporal decay function follows exponential dynamics with
configurable half-life:

decay(x, τ_half, Δt) = x × exp(−ln(2) × Δt / max(τ_half, τ_min))

where τ_min = 120 seconds provides a floor to prevent numerical
instability from near-zero half-lives.

1.2 The Three-Knob Philosophy

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

1.3 Knob-Derived Parameters

All system tunables derive from the three primary knobs. This section
catalogs the key derivations.

1.3.1 Context Windows and Temporal Scales

n_ctx(T) = round(lerp(32, 256, T))

w_score(T) = round(lerp(20, 120, T))

w_rate_seconds(T) = round(lerp(60, 300, T))

The context window n_ctx determines how many recent items inform
relevance computation. The scoring window w_score controls the lookback
for variance estimation and percentile calculation. The rate window
w_rate_seconds specifies the temporal horizon for write-rate
measurement.

1.3.2 Half-Life and Decay

Memory half-life follows a log-scale mapping to span multiple orders of
magnitude:

τ_min = 120.0 seconds (2 minutes)

τ_max = 43200.0 seconds (12 hours)

base_half_life(T) = exp(ln(τ_min) + T × ln(τ_max / τ_min))

This exponential mapping ensures that low Stability yields half-lives
near 2 minutes while high Stability approaches 12 hours, with smooth
interpolation across the range.

1.3.3 Hysteresis and Rate Targets

band_min = 0.02; band_max = 0.25

base_band(T) = lerp(band_min, band_max, T)

r_min = 0.2; r_max = 5.0 (writes per minute)

base_rate(S) = lerp(r_min, r_max, S)

The hysteresis band prevents oscillation in threshold-crossing
decisions. Write-rate targets establish homeostatic setpoints for the
threshold controller.

1.3.4 Experiential Mass and Maturity

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

1.4 Uncertainty Estimation

Uncertainty u(t) ∈ \[0, 1\] modulates learning rates and evidence
weighting. The raw uncertainty estimate blends multiple signals:

var_score_max = 0.25

var_recent_norm = clamp(var(scores\[t−w:t\]) / var_score_max, 0, 1)

coherence_complement = 1 − coherence_t

When prediction error signals are available, novelty and surprisal are
blended:

novelty_surprise = blend(\[novelty_t, surprisal_t\],

weights = normalize(\[S, 1 − T\]))

The final raw uncertainty combines these components with knob-derived
weights:

weights_u = normalize(\[S, F, 1 − T, S × (1 − T)\])

u_raw(t) = clamp(blend(\[var_recent_norm, focus_spread,

coherence_complement, novelty_surprise\],

weights = weights_u), 0, 1)

Smoothed uncertainty applies EWMA with a stability-dependent rate:

α_u(T) = 0.10 + (1 − T) × 0.60

u(t) = EWMA(u(t−1), u_raw(t), α = α_u(T))

When structural metrics are unavailable, the fallback is u_raw(t) = 1 −
maturity(t), ensuring high uncertainty during early operation.

2\. Core Adaptation Algorithms

This section presents the algorithms governing adaptation along each of
the three primary dimensions. Each algorithm consists of a prior
computation (executed at initialization) and a dynamic update (executed
per signal).

2.1 Focus-Driven Selectivity

Focus governs perceptual selectivity through relevance weighting and
attention width.

2.1.1 Focus Priors

Given Focus knob F ∈ \[0, 1\], compute initial priors:

weight_relevance_prior = sigmoid(2F − 1)

coverage_gain_floor = 0.3 + 0.7F

mismatch_weight_prior = 1 − F

attention_width_prior = lerp(π, 0.1π, F)

The attention width (in radians) controls the angular spread of the
receptive field in embedding space. High Focus produces narrow attention
(0.1π), while low Focus permits broad capture (π).

2.1.2 Dynamic Focus Update

At each signal event t with input embedding x_t:

recent_context ← tail(context_buffer, n_ctx(T))

observed_cosine ← cos(x_t, mean(recent_context))

weight_relevance_t ← EWMA(weight_relevance\_{t−1},

map01(observed_cosine), α = α_F(t))

where map01(z) = clamp((z + 1) / 2, 0, 1) transforms cosine values from
\[−1, 1\] to \[0, 1\].

The learning rate α_F(t) is modulated by uncertainty:

α_min_F = 0.05; α_span_F = 0.45

α_F(t) = α_min_F + F × α_span_F × u(t)

High uncertainty increases learning rate, allowing faster adaptation
when the environment is volatile. The Focus knob scales the uncertainty
responsiveness.

2.2 Sensitivity-Driven Plasticity

Sensitivity governs learning speed, emotional responsiveness, and
novelty capture.

2.2.1 Sensitivity Priors

Given Sensitivity knob S ∈ \[0, 1\], compute initial priors:

base_rate_prior = lerp(0.2, 5.0, S) \# writes/min

weight_novelty_prior = 0.3 + 0.7S

weight_surprise_prior = 0.2 + 0.8S

weight_valence_prior = 0.4 + 0.6S

weight_arousal_prior = S

emotion_gain_prior = exp(1.5S)

score_gain_prior = exp(2S)

rate_target_prior = base_rate_prior × (0.5 + 1.5S)

2.2.2 Emotional Projection

When emotion category centroids are available, the system projects input
embeddings onto a discrete emotion space C = {anger, fear, joy, love,
sadness, surprise}. Inspired by Russell\'s (1980) circumplex model, each
category maps to valence and arousal coordinates:

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

The emotion intensity combines peak probability with distributional
confidence via geometric mean, providing a measure that is high only
when a single emotion dominates with high certainty.

2.2.3 Threshold Modulation from Emotion

Emotional activation loosens write thresholds to capture salient
moments:

κ_emo ← κ_base × S \# where κ_base = 0.10

ΔThreshold_emotion_t ← −κ_emo × emotion_intensity_t ×

(0.5 + 0.5 × arousal_t)

This negative adjustment makes writing more likely during emotionally
salient events, consistent with McGaugh\'s (2004) findings on
arousal-enhanced encoding.

2.2.4 Mood Integration

Distinct from instantaneous emotion, the mood state M_t maintains a
persistent background affective tone:

α_mood(S) = lerp(0.01, 0.20, S) \# reactivity

λ_mood(T) = lerp(0.90, 0.999, T) \# decay

M_t = λ_mood(T) × M\_{t−1} + α_mood(S) × e_t

M_t ← clamp(M_t, −1.0, 1.0)

Note that this update does not enforce coefficient normalization; the
explicit clamp handles potential accumulation. The mood state provides a
separate threshold bias:

κ_mood ← κ_base × S

m_norm ← ‖M_t‖ / √6

ΔThreshold_mood_t ← −κ_mood × clamp(m_norm, 0, 1)

2.3 Stability-Driven Persistence

Stability governs temporal dynamics through half-life, decay rates, and
hysteresis.

2.3.1 Stability Priors

Given Stability knob T ∈ \[0, 1\], compute initial priors:

hysteresis_band_prior = lerp(0.02, 0.25, T)

half_life_prior = base_half_life(T)

rate_decay_prior = lerp(0.60, 0.98, T)

periphery_half_life_prior = clamp(0.5 × half_life_prior,

τ_min, τ_max)

drift_weight_prior = 0.5 × (1 − T)

2.3.2 Dynamic Stability Update

At each signal event, compute retention statistics and adjust half-life:

active_memories ← {m \| strength(m) ≥ periphery_cutoff(T)}

observed_retention ← mean_age(active_memories)

retention_ema_t ← EWMA(retention_ema\_{t−1},

observed_retention, α = α_T(t))

Compute z-score relative to recent retention history:

last_w_ret ← tail(retention_history, w_ret(T))

μ_ret ← mean(last_w_ret)

σ_ret ← max(std(last_w_ret), 1.0)

zscore_ret ← clamp((observed_retention − μ_ret) / σ_ret, −3, +3)

The target half-life incorporates feedback adjustment from the stability
feedback mechanism (Section 7.3):

stability_adj ← ΔHalfLife_adj_t if provided else 0

target_half_life_t ← clamp(base_half_life(T) ×

(1 + 0.25 × zscore_ret + stability_adj),

τ_min, τ_max)

half_life_t ← EWMA(half_life\_{t−1}, target_half_life_t,

α = α_T(t))

3\. Structural Metrics and Composite Scoring

3.1 Embedding-Derived Metrics

3.1.1 Coherence

Coherence measures integration of the current signal with context:

raw ← var(\[cos(x_t, c) for c in context_window\])

coherence_t ← 1 − clamp(raw, 0, 1)

High coherence (low variance in similarities) indicates the signal fits
consistently with context. The effective Focus is modulated: F_eff = F ×
(0.5 + 0.5 × coherence_t).

3.1.2 Focus Spread

Focus spread quantifies the entropy of attention over nearest neighbors:

k ← k_neighbors(T) = round(lerp(8, 32, T))

p ← softmax(kNN_similarities)

focus_spread_t ← H(p) / ln(k)

Values near 1 indicate diffuse attention; values near 0 indicate
concentrated attention. The effective Focus is further modulated: F_eff
← F_eff × (1 − focus_spread_t).

3.1.3 Trajectory Drift

Drift measures directional change in context centroids:

drift_vec_t ← l2_normalize(mean(ctx_t)) −

l2_normalize(mean(ctx\_{t−k}))

drift_mag_t ← ‖drift_vec_t‖

Since both centroids are unit-normalized, drift_mag_t ∈ \[0, 2\]. A
threshold determines episode boundaries:

drift_threshold ← lerp(0.10, 0.35, T)

if drift_mag_t \> drift_threshold:

trigger_episode_boundary()

3.1.4 Embedding Prediction Error

We measure surprisal as the deviation of the current embedding from the
predicted trajectory in latent space:

Δx_t = x_t − x\_{t−1}

Δx_trend_t = EWMA(Δx_trend\_{t−1}, Δx_t, α=0.1)

x_pred_t = x\_{t−1} + Δx_trend\_{t−1}

prediction_error_t = 1 − cos(x_pred_t, x_t)

This error is normalized to produce the surprisal signal:

err_max = 0.5

surprisal_t ← clamp(prediction_error_t / err_max, 0, 1)

This formulation captures purely kinematic surprise in the thought
process

3.2 Composite Score Computation

The system computes 12 metrics that blend into a composite write score:

  -----------------------------------------------------------------------
  **Metric**         **Knob**        **Expression**
  ------------------ --------------- ------------------------------------
  Relevance          ↑F              cos(x, μ_ctx) × (0.5 + F)

  Mismatch           ↓F, ↑S          (1 − F) × S × novelty

  Prediction Error   ↑S, ↓T          surprisal_t × S × (1 − T)

  Rarity             ↑F, ↓T          (1 − μ_sim) × (0.5 + 0.5F) × (1 −
                                     0.2T)

  Drift              ↓T              (drift_mag / 2) × (1 − T)

  Utility            ↑F, ↓S          ΔSSE × (0.5 + 0.5F) × (1 − 0.3S)

  Salience           F, S            (rarity + novelty) / 2 × (F + S) / 2

  Valence            S, ↓T           map01(Σ p_c × v_map\[c\])

  Arousal            S, ↓T           clamp(Σ p_c × a_map\[c\], 0, 1)

  Contradiction      ↑S, ↓F          max(0, S − F)

  Periphery          ↑T              (1 − relevance) × T

  Coverage           ↑F              F × relevance
  -----------------------------------------------------------------------

Table 1: Metric definitions and knob dependencies. Arrows indicate
direction of influence.

3.3 Metric Weight Blending

Metric weights adapt online using recursive least squares (RLS) to
minimize prediction error between composite scores and observed
outcomes. Initial weights derive from bootstrap coefficients:

w_bootstrap\[i\] ← sigmoid(c_F\[i\]×F + c_S\[i\]×S + c_T\[i\]×T + d_i)

RLS fitting updates coefficients with stability-dependent forgetting:

φ(T) = 0.90 + 0.09T

N ← round(lerp(64, 512, T)) \# fitting window

The fitted weights blend with bootstrap weights based on RLS confidence:

τ_rls ← lerp(20.0, 80.0, T)

confidence_rls ← 1 − exp(−t / τ_rls)

weight_i(t) ← (1 − confidence_rls) × w_bootstrap\[i\] +

confidence_rls × w_rls\[i\]

3.3.1 Score Normalization

Composite score computation requires careful normalization:

for each metric i:

m01\[i\] = map01(metric\[i\]) if signed else clamp(metric\[i\], 0, 1)

weight_sum ← Σ weights\[i\]

if weight_sum \< ε: return 0

weights_norm\[i\] ← weights\[i\] / weight_sum

score ← clamp(Σ weights_norm\[i\] × m01\[i\], 0, 1)

Weight normalization is critical: with 12 metrics and raw weights
averaging \~0.6, the sum approaches 7.2. Without normalization, weighted
sums would saturate and collapse variance.

4\. Dynamic Thresholding and Homeostatic Control

The write gate compares composite scores against an adaptive threshold
θ_dynamic. This section details the threshold evolution algorithm
incorporating Bayesian prior-evidence blending and homeostatic rate
control.

4.1 Prior-Evidence Blending

The threshold prior derives from knob settings:

θ_prior(F, S, T) = lerp(0.10, 0.30, T) × (1 − 0.3S)

Observed evidence comes from the 90th percentile of recent scores:

w ← w_score(T)

observed_p90 ← percentile(scores\[t−w:t\], 90)

Prior and evidence masses weight the blend:

ρ_prior ← prior_mass(T) = round(lerp(2, 32, T))

ρ_obs ← u(t) × min(w, count)

The target threshold blends prior and evidence:

θ_target ← (ρ_prior × θ_prior + ρ_obs × observed_p90) /

max(ε, ρ_prior + ρ_obs)

High Stability increases prior mass, making the system more resistant to
observed deviations. High uncertainty increases evidence mass, allowing
faster adaptation to volatile conditions.

4.2 Homeostatic Rate Control

The controller maintains write rates near the target setpoint through
continuous-time estimation with effective sample size (ESS) reliability
weighting.

4.2.1 Rate Estimation

Δt ← now − last_timestamp

Δt ← max(Δt, 10⁻³) \# minimum 1ms

α_dt ← 1 − exp(−Δt / 1.0)

dt_ema ← (1 − α_dt) × dt_ema + α_dt × Δt

dt_base ← max(dt_ema, 1.0)

The rate time constant scales with Stability:

τ_rate ← max(2\^(3T) × dt_base, 1.0)

α ← 1 − exp(−Δt / τ_rate)

Instantaneous rate estimation with bias correction:

ρ_inst ← (Δwrites / Δt) × 60 \# writes per minute

m_rate ← (1 − α) × m_rate + α × ρ_inst

denom ← max(1 − (1 − α)\^(rate_ticks + 1), ε)

ρ_hat ← m_rate / denom \# bias-corrected estimate

4.2.2 Effective Sample Size

ESS estimates the effective number of independent samples in the EMA,
using a heuristic inspired by Liu & Chen (1998):

β ← max(0, 1 − α)

ESS ← min((1 + β) / max(1 − β, ε), 100)

reliability ← 1 − exp(−ESS × (1 − T))

High Stability dampens reliability, preventing aggressive corrections in
conservative regimes.

4.2.3 Homeostatic Correction

The rate error drives threshold adjustment:

rate_error ← tanh((ρ_hat − rate_target_t) /

max(rate_target_t, ε))

κ_r = 0.10 \# rate error gain

cap_homeo ← 0.25 × hysteresis_t

Δθ_homeo ← clamp(reliability × κ_r × (1 − T) ×

(1 − maturity(t)) × rate_error,

−cap_homeo, +cap_homeo)

The correction scales with reliability and is attenuated by both
Stability and maturity, ensuring conservative, mature systems make
minimal homeostatic adjustments.

4.3 Threshold Integration

All threshold deltas combine and pass through safety limiting:

Δθ_total ← Δθ_sens + Δθ_homeo + Δθ_prec + Δθ_emo + Δθ_mood

cap_total ← max_ΔT_per_min(t) × (Δt / 60.0)

Δθ_limited ← clamp(Δθ_total, −cap_total, +cap_total)

θ_dynamic_t ← clamp(EWMA(θ_dynamic\_{t−1}, θ_target,

α = α_T(t)) + Δθ_limited,

T_min(t), T_max(t))

Hysteresis evolves toward the stability-derived base:

hysteresis_t ← clamp(EWMA(hysteresis\_{t−1},

base_band(T), α = α_T(t)),

band_min, band_max)

5\. Reinforcement and Decay Dynamics

5.1 Memory Strength Model

Each memory m maintains a strength value updated through use-frequency
tracking and exponential decay:

use_frequency_t ← EWMA(use_frequency\_{t−1},

used_flag(m), α = α_S(t))

λ_t ← ln(2) / half_life_t

strength_t ← strength\_{t−1} × exp(−λ_t × Δt) + S × use_frequency_t

Memories falling below the periphery cutoff are candidates for eviction:

periphery_cutoff(T) = lerp(0.05, 0.25, T)

if strength_t \< periphery_cutoff(T): evict(m)

5.2 Influence-Weighted Updates

When contextual gain signals are available, influence factors modulate
reinforcement:

influence_factor ← (used_count / max(retrieved_count, 1)) ×

clamp(contextual_gain(m), −1, +1)

strength_t ← strength\_{t−1} × exp(−λ_t × Δt) +

S × use_frequency_t + F × influence_factor

5.3 Causal Feedback Loop

The system tracks causal influence of retrieved memories on generation
quality through contextual gain---the improvement in prediction accuracy
attributable to including each memory in context.

5.3.1 Focus Feedback

αF_base = 0.10; βF_base = 0.05

for each used memory m:

if contextual_gain(m) \> 0:

weight_relevance_t += αF_base × contextual_gain(m)

attention_width_t \*= (1 − βF_base)

else:

attention_width_t \*= (1 + βF_base)

weight_relevance_t ← clamp(weight_relevance_t, 0, 1)

attention_width_t ← clamp(attention_width_t,

attention_width_min, attention_width_max)

Positive contextual gain narrows attention and boosts relevance
weighting; negative gain widens attention to explore alternatives.

5.3.2 Sensitivity Feedback

η_base = 0.10

for each used memory m:

novelty_reward ← 1 − sim(m.embedding, recent_context)

weight_novelty_t += η_base × (novelty_reward ×

contextual_gain(m) −

redundancy(m, recent_context))

weight_novelty_t ← clamp(weight_novelty_t, 0, 1)

This rewards novelty that proves useful while penalizing redundant
retrievals.

5.3.3 Stability Feedback

γT_base = 0.05

for each used memory m:

if contextual_gain(m) \> 0:

stability(m) += γT_base

else:

stability(m) \*= (1 − γT_base)

The mean stability of used memories provides adjustment to the half-life
target:

adj ← clamp(mean(stability(m_used)) − 1.0, −0.25, +0.25)

ΔHalfLife_adj_t ← adj

This factor is consumed by the Stability update (Section 4.3.2),
avoiding conflicting adjustments between feedback mechanisms.

5.4 Generation Influence Tracking

When generation embeddings are available, influence incorporates output
trajectory:

Δḡ ← l2_normalize(ḡ_t) − l2_normalize(ḡ\_{t−1})

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

6\. Advanced Cognitive Processes

This section presents algorithms modeling higher-order cognitive
phenomena: working memory maintenance, metacognitive monitoring,
reconsolidation dynamics, and serial position effects.

6.1 Working Memory Gates

Following Cowan\'s (2001) capacity constraints, working memory maintains
a limited number of active items:

base_capacity = round(lerp(5, 3, S) + lerp(−1, 1, F))

This yields a range of approximately 2-6 slots, broadening the 4±1 chunk
limit to accommodate task-dependent requirements. High Sensitivity
reduces capacity (faster turnover), while high Focus modulates breadth.

Maintenance incurs cognitive cost:

maintenance_cost_per_slot = lerp(0.05, 0.15, S)

complexity_penalty = manifold_complexity × lerp(0.5, 1.5, S)

The manifold_complexity represents local variance in the embedding
stream: 1 - mean(cos(window)).

Gating thresholds determine entry and chunking:

chunking_threshold = lerp(0.7, 0.9, F)

gate_threshold = lerp(0.1, 0.4, F)

rehearsal_rate = lerp(0.5, 2.0, S)

slot_dedication_strength = lerp(0.3, 0.9, T)

6.2 Metacognitive Monitoring

The system implements feeling-of-knowing (FOK) and tip-of-tongue (TOT)
detection following Hart\'s (1965) framework:

FOK_threshold = lerp(0.2, 0.5, F)

TOT_detection = (FOK \> lerp(0.5, 0.8, F)) AND

(retrieval_strength \< lerp(0.4, 0.2, F))

TOT occurs when metacognitive confidence is high but retrieval strength
is low---the characteristic experience of knowing one knows something
but being unable to access it.

Additional metacognitive parameters:

confidence_decay_rate = lerp(0.01, 0.1, 1 − T)

unknown_threshold = lerp(0.3, 0.1, F)

strategy_switch_latency = lerp(500, 100, S) \# ms

certainty_requirement = lerp(0.6, 0.9, T)

metacognitive_sensitivity = F × (1 + 0.5 × S)

6.3 Memory Reconsolidation

Following Nader et al. (2000), retrieved memories enter a labile state
permitting modification:

τ_labile = lerp(30, 300, T) \# seconds

reconsolidation_gain = lerp(0.2, 0.02, T)

lability_susceptibility = (1 − T) × (0.5 + 0.5 × S)

During the lability window, memories can drift toward current context:

drift_magnitude = (1 − T) × S × lability ×

contextual_relevance

Reconsolidation effects propagate to semantically related memories with
decay:

ripple_decay = lerp(0.5, 0.1, T) \# per semantic hop

6.4 Retrieval Competition

Retrieved memories compete through lateral inhibition, modeling
retrieval-induced forgetting (Anderson et al., 1994):

inhibition_radius = lerp(0.5, 0.85, F)

winners_k = round(lerp(7, 3, F))

suppression_per_retrieval = lerp(0.1, 0.01, T) ×

(1 − winning_activation)

recovery_time_RIF = lerp(300, 1800, T) \# seconds

High Focus produces narrow winner-take-all dynamics; low Focus permits
broader activation.

6.5 Predictive Pre-activation

The system pre-activates memories predicted to be relevant based on
trajectory extrapolation:

prediction_horizon = round(lerp(2, 8, F))

pre_activation_decay = lerp(0.7, 0.3, T)

prediction_conf_threshold = lerp(0.3, 0.7, F)

surprise_sensitivity = S × lerp(2.0, 0.5, T)

When predictions fail (high surprise), the system updates its trajectory
model:

update_rate_on_surprise = lerp(0.2, 0.02, T) × S

6.6 Serial Position Effects

The architecture models primacy, recency, and distinctiveness effects
observed in human memory (Murdock, 1962):

primacy_window = round(lerp(5, 2, F))

primacy_bonus = lerp(1.2, 2.0, S)

recency_window = round(lerp(7, 3, F))

rehearsal_curve_depth = lerp(0.2, 0.6, S)

The von Restorff (isolation) effect enhances memory for distinctive
items (Hunt, 1995):

distinctiveness_threshold = lerp(0.6, 0.8, F)

von_restorff_multiplier = lerp(1.5, 3.0, S)

Items in the middle region suffer interference:

interference_zone = positions\[primacy_window+1 : −recency_window\]

middle_suppression = lerp(0.8, 0.5, S) × (1 − F)

6.7 Emotional Consolidation

High-emotion events trigger enhanced consolidation, following McGaugh\'s
(2004) findings:

θ_intensity = lerp(0.6, 0.8, 1 − S)

θ_arousal = lerp(0.4, 0.2, S)

trigger = (emotion_intensity_t ≥ θ_intensity) AND

(arousal_t ≥ θ_arousal)

Flashbulb memories receive extended half-life bonuses:

flashbulb_threshold = lerp(0.9, 0.4, S)

emotional_half_life_bonus = exp(lerp(0, ln(3), S)) ×

(1 + emotion_intensity_t)

Emotional tags cascade to related memories with decay:

cascade_radius = round(lerp(1, 5, S))

cascade_decay = lerp(0.7, 0.3, S)

7\. Consolidation and Graph Integration

The consolidation system transforms episodic memories into semantic
structures through clustering, summarization, and knowledge graph
construction.

7.1 Consolidation Triggers

Consolidation activates under capacity, rate, or temporal conditions:

should_consolidate = (db_size \> consolidation_threshold) OR

(write_rate_t \< rate_target_t / 2) OR

(elapsed_time \> consolidation_interval)

The consolidation rate adapts to Stability and Sensitivity:

rate_consolidate = (1 / max(consolidation_interval, 1)) ×

(0.3 + 0.7T) × (1 − 0.5S)

7.1.1 Activity-Aware Scheduling

Consolidation runs during idle periods and preempts for retrieval:

idle_required(T) = round(0.25 × w_rate_seconds(T))

idle_for = now() − last_retrieval_ts

should_start = (NOT is_processing_signal) AND

(retrieval_queue_depth == 0) AND

(idle_for ≥ idle_required(T))

On retrieval events, consolidation pauses, commits micro-batches, and
resumes when idle.

7.2 Consolidation Scoring

Each memory receives a consolidation score determining merge priority:

score_consolidate(m) = w_s × strength(m) −

w_r × redundancy(m) +

w_c × connectivity(m) +

w_t × stability(m)

Weights derive from knobs:

w_s = T; w_r = F; w_c = S; w_t = T

Low-scoring memories are marked for merging.

7.3 Clustering and Summarization

Marked memories cluster via density-based methods (e.g., DBSCAN) or
k-means using embedding similarity:

cluster_i = {m_j \| cos(m_j, μ_i) \> merge_threshold}

μ_i = centroid(cluster_i)

Summary nodes replace clusters:

summary.embedding = μ_i

summary.text = summarize(fetch_blobs(cluster_i))

summary.metadata.sources = \[m.id for m in cluster_i\]

7.4 Semantic Extraction

For sufficiently large clusters, semantic extraction identifies entities
and relations:

extraction_batch_size = round(lerp(8, 32, T))

min_cluster_size = round(lerp(3, 10, F))

entity_frequency_threshold = round(lerp(5, 15, T))

extraction_interval = lerp(300, 3600, T) \# 5 min → 1 hour

max_extractions_per_cycle = round(lerp(20, 5, T))

Extraction uses structured prompting to identify named entities (people,
places, organizations, concepts) and relationships (co-occurrence,
implication, contradiction).

7.5 Knowledge Graph Construction

The graph comprises three node types:

-   **Memory Nodes:** Summarized embeddings from merged clusters

-   **Entity Nodes:** Named entities extracted from text

-   **Concept Nodes:** Emergent centroids representing recurrent topics

Edge types capture relationships:

-   **co_occurs_with:** Shared context or temporal proximity

-   **implies/causes:** Directional correlation in embedding drift

-   **contradicts:** Strong negative similarity or schema violation

-   **reinforces:** Frequent joint retrieval

-   **derived_from:** Links summaries to source memories

7.5.1 Edge Construction

Co-occurrence edges derive from embedding similarity:

for (m_i, m_j) in cluster.sources:

cos_sim ← cos(m_i.embedding, m_j.embedding)

if cos_sim \> lerp(0.85, 0.95, F):

create_edge(m_i, m_j, \'co_occurs_with\', cos_sim)

Causal edges derive from temporal drift:

temporal_order ← sort_by_timestamp(cluster.sources)

for i in range(len(temporal_order) − 1):

m_i, m_j ← temporal_order\[i\], temporal_order\[i+1\]

drift_vec ← m_j.embedding − m_i.embedding

drift_mag ← ‖drift_vec‖

if drift_mag \> lerp(0.15, 0.35, T):

create_edge(m_i, m_j, \'causes\', drift_mag)

7.6 Graph-Augmented Retrieval

Retrieval combines vector similarity with graph expansion:

results_vec ← topK(vector_search(x_t, k=kNN_size))

seed_nodes ← \[r.id for r in results_vec\]

expanded_nodes ← graph.traverse(seed_nodes, depth=graph_depth)

combined ← union(seed_nodes, expanded_nodes)

re_ranked ← sort_by(cos(x_t, embeddings(combined)))

Graph expansion uses recursive traversal with depth limits to find
related context that pure vector search might miss.

8\. Interrupt Gate and Streaming Integration

The interrupt gate controls when retrieved memories enter active context
during streaming generation. The gate balances novelty value against
disruption cost.

8.1 Marginal Utility Computation

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

8.2 Marginal Utility Score

The marginal utility (MU) of a candidate memory combines four factors:

w_cov = normalize(lerp(0.40, 0.60, F)) \# coverage gain

w_rel = normalize(lerp(0.35, 0.25, F)) \# relevance

w_red = normalize(lerp(0.15, 0.25, S)) \# redundancy penalty

w_coh = normalize(lerp(0.15, 0.25, S)) \# incoherence penalty

mu = w_cov × coverage_gain(candidate \| included_set) +

w_rel × cos(candidate, ctx_centroid) −

w_red × redundancy(candidate, included_set) −

w_coh × (1 − coherence_t)

8.3 Gate Decision Logic

Duplicate suppression threshold:

dup_thresh = lerp(0.88, 0.96, F) × (0.98 + 0.02T)

K = round(lerp(10, 6, F)) \# candidates to evaluate

Boundary-aware override permits lower-threshold interrupts at natural
boundaries:

boundary_mult = lerp(1.3, 2.0, F) × lerp(1.1, 0.9, S)

The gate permits interrupt when:

embedding_novelty = 1 − max(cos(candidate, ctx_window))

allow_interrupt =

(max_relevance ≥ retrieval_thresh(F)) AND

(embedding_novelty ≥ τ_novelty_eff OR best_mu ≥ τ_mu_eff) AND

(max_semantic_overlap \< dup_thresh) AND

(at_drift_boundary OR best_mu ≥ boundary_mult × τ_mu_eff)

This logic suppresses low-drift interrupts unless the marginal utility
substantially exceeds threshold, while permitting normal-threshold
interrupts at natural transition points.

8.4 Streaming Pacing

Streaming retrieval is gated by cumulative drift rate:

drift_accum += dist(x_t, x\_{last_check})

pacing_thresh(S) = lerp(0.5, 0.1, S)

if drift_accum \> pacing_thresh(S): trigger_check()

max_wait_drift(F) = lerp(2.0, 0.5, F)

max_results(F) = round(lerp(64, 4, F))

adjacent_window(F) = round(lerp(8, 1, F))

High Sensitivity produces frequent checks triggered by small content
shifts; high Focus enforces strict drift limits.
