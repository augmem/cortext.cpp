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

Cortext is governed by three continuous control parameters:

-   **Focus (F ∈ \[0, 1\]):** Perceptual selectivity and precision.

-   **Sensitivity (S ∈ \[0, 1\]):** Plasticity and affective gain.

-   **Stability (T ∈ \[0, 1\]):** Temporal persistence and inertia.

1.3 Knob-Derived Parameters

1.3.1 Context Windows and Temporal Scales

n_ctx(T) = round(lerp(32, 256, T))

w_score(T) = round(lerp(20, 120, T))

w_rate_seconds(T) = round(lerp(60, 300, T))

1.3.2 Half-Life and Decay

τ_min = 120.0 seconds (2 minutes)

τ_max = 43200.0 seconds (12 hours)

base_half_life(T) = exp(ln(τ_min) + T × ln(τ_max / τ_min))

1.3.3 Hysteresis and Rate Targets

band_min = 0.02; band_max = 0.25

base_band(T) = lerp(band_min, band_max, T)

r_min = 0.2; r_max = 5.0 (writes per minute)

base_rate(S) = lerp(r_min, r_max, S)

1.3.4 Experiential Mass and Maturity

τ_m(T) = lerp(10.0, 200.0, T)

maturity(t) = 1 − exp(−count / τ_m(T))

T_min(t) = lerp(0.01, 0.05, maturity(t))

T_max(t) = lerp(0.99, 0.95, maturity(t))

max_ΔT_per_min(t) = lerp(0.30, 0.10, maturity(t))

1.4 Uncertainty Estimation

var_score_max = 0.25

var_recent_norm = clamp(var(scores\[t−w:t\]) / var_score_max, 0, 1)

coherence_complement = 1 − coherence_t

novelty_surprise = blend(\[novelty_t, surprisal_t\],

weights = normalize(\[S, 1 − T\]))

weights_u = normalize(\[S, F, 1 − T, S × (1 − T)\])

u_raw(t) = clamp(blend(\[var_recent_norm, focus_spread,

coherence_complement, novelty_surprise\],

weights = weights_u), 0, 1)

α_u(T) = 0.10 + (1 − T) × 0.60

u(t) = EWMA(u(t−1), u_raw(t), α = α_u(T))

When structural metrics are unavailable, the fallback is u_raw(t) = 1 −
maturity(t).

2\. Core Adaptation Algorithms

2.1 Focus-Driven Selectivity

2.1.1 Focus Priors

weight_relevance_prior = sigmoid(2F − 1)

coverage_gain_floor = 0.3 + 0.7F

mismatch_weight_prior = 1 − F

attention_width_prior = lerp(π, 0.1π, F)

2.1.2 Dynamic Focus Update

recent_context ← tail(context_buffer, n_ctx(T))

observed_cosine ← cos(x_t, mean(recent_context))

weight_relevance_t ← EWMA(weight_relevance\_{t−1},

map01(observed_cosine), α = α_F(t))

α_min_F = 0.05; α_span_F = 0.45

α_F(t) = α_min_F + F × α_span_F × u(t)

2.2 Sensitivity-Driven Plasticity

2.2.1 Sensitivity Priors

base_rate_prior = lerp(0.2, 5.0, S) \# writes/min

weight_novelty_prior = 0.3 + 0.7S

weight_surprise_prior = 0.2 + 0.8S

weight_valence_prior = 0.4 + 0.6S

weight_arousal_prior = S

emotion_gain_prior = exp(1.5S)

score_gain_prior = exp(2S)

rate_target_prior = base_rate_prior × (0.5 + 1.5S)

2.2.2 Emotional Projection

v_map = {anger: −0.9, fear: −0.8, sadness: −0.9,

joy: +0.9, love: +0.8, surprise: 0.0}

a_map = {anger: +0.9, fear: +0.9, sadness: +0.3,

joy: +0.6, love: +0.5, surprise: +0.8}

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

2.2.3 Threshold Modulation from Emotion

κ_emo ← κ_base × S \# where κ_base = 0.10

ΔThreshold_emotion_t ← −κ_emo × emotion_intensity_t ×

(0.5 + 0.5 × arousal_t)

2.2.4 Mood Integration

α_mood(S) = lerp(0.01, 0.20, S) \# reactivity

λ_mood(T) = lerp(0.90, 0.999, T) \# decay

M_t = λ_mood(T) × M\_{t−1} + α_mood(S) × e_t

M_t ← clamp(M_t, −1.0, 1.0)

κ_mood ← κ_base × S

m_norm ← ‖M_t‖ / √6

ΔThreshold_mood_t ← −κ_mood × clamp(m_norm, 0, 1)

2.3 Stability-Driven Persistence

2.3.1 Stability Priors

hysteresis_band_prior = lerp(0.02, 0.25, T)

half_life_prior = base_half_life(T)

rate_decay_prior = lerp(0.60, 0.98, T)

periphery_half_life_prior = clamp(0.5 × half_life_prior,

τ_min, τ_max)

drift_weight_prior = 0.5 × (1 − T)

2.3.2 Dynamic Stability Update

active_memories ← {m \| strength(m) ≥ periphery_cutoff(T)}

observed_retention ← mean_age(active_memories)

retention_ema_t ← EWMA(retention_ema\_{t−1},

observed_retention, α = α_T(t))

last_w_ret ← tail(retention_history, w_ret(T))

μ_ret ← mean(last_w_ret)

σ_ret ← max(std(last_w_ret), 1.0)

zscore_ret ← clamp((observed_retention − μ_ret) / σ_ret, −3, +3)

stability_adj ← ΔHalfLife_adj_t if provided else 0

target_half_life_t ← clamp(base_half_life(T) ×

(1 + 0.25 × zscore_ret + stability_adj),

τ_min, τ_max)

half_life_t ← EWMA(half_life\_{t−1}, target_half_life_t,

α = α_T(t))

3\. Structural Metrics and Composite Scoring

3.1 Embedding-Derived Metrics

3.1.1 Coherence

raw ← var(\[cos(x_t, c) for c in context_window\])

coherence_t ← 1 − clamp(raw, 0, 1)

F_eff = F × (0.5 + 0.5 × coherence_t)

3.1.2 Focus Spread

k ← k_neighbors(T) = round(lerp(8, 32, T))

p ← softmax(kNN_similarities)

focus_spread_t ← H(p) / ln(k)

F_eff ← F_eff × (1 − focus_spread_t)

3.1.3 Trajectory Drift

drift_vec_t ← l2_normalize(mean(ctx_t)) −

l2_normalize(mean(ctx\_{t−k}))

drift_mag_t ← ‖drift_vec_t‖

drift_threshold ← lerp(0.10, 0.35, T)

if drift_mag_t \> drift_threshold:

trigger_episode_boundary()

3.1.4 Embedding Prediction Error

Δx_t = x_t − x\_{t−1}

Δx_trend_t = EWMA(Δx_trend\_{t−1}, Δx_t, α=0.1)

x_pred_t = x\_{t−1} + Δx_trend\_{t−1}

prediction_error_t = 1 − cos(x_pred_t, x_t)

err_max = 0.5

surprisal_t ← clamp(prediction_error_t / err_max, 0, 1)

3.2 Metric Weight Blending

w_bootstrap\[i\] ← sigmoid(c_F\[i\]×F + c_S\[i\]×S + c_T\[i\]×T + d_i)

φ(T) = 0.90 + 0.09T

N ← round(lerp(64, 512, T)) \# fitting window

τ_rls ← lerp(20.0, 80.0, T)

confidence_rls ← 1 − exp(−t / τ_rls)

weight_i(t) ← (1 − confidence_rls) × w_bootstrap\[i\] +

confidence_rls × w_rls\[i\]

3.3 Score Normalization

for each metric i:

m01\[i\] = map01(metric\[i\]) if signed else clamp(metric\[i\], 0, 1)

weight_sum ← Σ weights\[i\]

if weight_sum \< ε: return 0

weights_norm\[i\] ← weights\[i\] / weight_sum

score ← clamp(Σ weights_norm\[i\] × m01\[i\], 0, 1)

4\. Dynamic Thresholding and Homeostatic Control

4.1 Prior-Evidence Blending

θ_prior(F, S, T) = lerp(0.10, 0.30, T) × (1 − 0.3S)

w ← w_score(T)

observed_p90 ← percentile(scores\[t−w:t\], 90)

ρ_prior ← prior_mass(T) = round(lerp(2, 32, T))

ρ_obs ← u(t) × min(w, count)

θ_target ← (ρ_prior × θ_prior + ρ_obs × observed_p90) /

max(ε, ρ_prior + ρ_obs)

4.2 Homeostatic Rate Control

4.2.1 Rate Estimation

Δt ← now − last_timestamp

Δt ← max(Δt, 10⁻³) \# minimum 1ms

α_dt ← 1 − exp(−Δt / 1.0)

dt_ema ← (1 − α_dt) × dt_ema + α_dt × Δt

dt_base ← max(dt_ema, 1.0)

τ_rate ← max(2\^(3T) × dt_base, 1.0)

α ← 1 − exp(−Δt / τ_rate)

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

4.2.3 Homeostatic Correction

rate_error ← tanh((ρ_hat − rate_target_t) /

max(rate_target_t, ε))

κ_r = 0.10 \# rate error gain

cap_homeo ← 0.25 × hysteresis_t

Δθ_homeo ← clamp(reliability × κ_r × (1 − T) ×

(1 − maturity(t)) × rate_error,

−cap_homeo, +cap_homeo)

4.3 Threshold Integration

Δθ_total ← Δθ_sens + Δθ_homeo + Δθ_prec + Δθ_emo + Δθ_mood

cap_total ← max_ΔT_per_min(t) × (Δt / 60.0)

Δθ_limited ← clamp(Δθ_total, −cap_total, +cap_total)

θ_dynamic_t ← clamp(EWMA(θ_dynamic\_{t−1}, θ_target,

α = α_T(t)) + Δθ_limited,

T_min(t), T_max(t))

hysteresis_t ← clamp(EWMA(hysteresis\_{t−1},

base_band(T), α = α_T(t)),

band_min, band_max)

5\. Reinforcement and Decay Dynamics

5.1 Memory Strength Model

use_frequency_t ← EWMA(use_frequency\_{t−1},

used_flag(m), α = α_S(t))

λ_t ← ln(2) / half_life_t

strength_t ← strength\_{t−1} × exp(−λ_t × Δt) + S × use_frequency_t

periphery_cutoff(T) = lerp(0.05, 0.25, T)

if strength_t \< periphery_cutoff(T): evict(m)

5.2 Influence-Weighted Updates

influence_factor ← (used_count / max(retrieved_count, 1)) ×

clamp(contextual_gain(m), −1, +1)

strength_t ← strength\_{t−1} × exp(−λ_t × Δt) +

S × use_frequency_t + F × influence_factor

5.3 Causal Feedback Loop

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

5.3.2 Sensitivity Feedback

η_base = 0.10

for each used memory m:

novelty_reward ← 1 − sim(m.embedding, recent_context)

weight_novelty_t += η_base × (novelty_reward ×

contextual_gain(m) −

redundancy(m, recent_context))

weight_novelty_t ← clamp(weight_novelty_t, 0, 1)

5.3.3 Stability Feedback

γT_base = 0.05

for each used memory m:

if contextual_gain(m) \> 0:

stability(m) += γT_base

else:

stability(m) \*= (1 − γT_base)

adj ← clamp(mean(stability(m_used)) − 1.0, −0.25, +0.25)

ΔHalfLife_adj_t ← adj

5.4 Generation Influence Tracking

Δḡ ← l2_normalize(ḡ_t) − l2_normalize(ḡ\_{t−1})

drift_mag_gen ← ‖Δḡ‖

drift_contribution(m) ← (drift_mag_gen / 2) ×

max(0, cos(m.embedding, l2_normalize(Δḡ)))

λ₁ = 0.5; λ₂ = 0.4; λ₃ = 0.3

influence(m) ← λ₁ × contextual_gain(m) +

λ₂ × cos(m.embedding, ḡ_t) −

λ₃ × drift_contribution(m)

L_sustain(T) = round(lerp(3, 5, T))

sustained_influence ← EWMA(sustained_influence,

influence(m),

α = 2 / (L_sustain(T) + 1))

6\. Advanced Cognitive Processes

6.1 Working Memory Gates

base_capacity = round(lerp(5, 3, S) + lerp(−1, 1, F))

This yields a range of approximately 2-6 slots, broadening the 4±1 chunk
limit to accommodate task-dependent requirements. High Sensitivity
reduces capacity (faster turnover), while high Focus modulates breadth.

maintenance_cost_per_slot = lerp(0.05, 0.15, S)

complexity_penalty = manifold_complexity × lerp(0.5, 1.5, S)

chunking_threshold = lerp(0.7, 0.9, F)

gate_threshold = lerp(0.1, 0.4, F)

rehearsal_rate = lerp(0.5, 2.0, S)

slot_dedication_strength = lerp(0.3, 0.9, T)

6.2 Metacognitive Monitoring

FOK_threshold = lerp(0.2, 0.5, F)

TOT_detection = (FOK \> lerp(0.5, 0.8, F)) AND

(retrieval_strength \< lerp(0.4, 0.2, F))

confidence_decay_rate = lerp(0.01, 0.1, 1 − T)

unknown_threshold = lerp(0.3, 0.1, F)

strategy_switch_latency = lerp(500, 100, S) \# ms

certainty_requirement = lerp(0.6, 0.9, T)

metacognitive_sensitivity = F × (1 + 0.5 × S)

6.3 Memory Reconsolidation

τ_labile = lerp(30, 300, T) \# seconds

reconsolidation_gain = lerp(0.2, 0.02, T)

lability_susceptibility = (1 − T) × (0.5 + 0.5 × S)

drift_magnitude = (1 − T) × S × lability ×

contextual_relevance

ripple_decay = lerp(0.5, 0.1, T) \# per semantic hop

6.4 Retrieval Competition

inhibition_radius = lerp(0.5, 0.85, F)

winners_k = round(lerp(7, 3, F))

suppression_per_retrieval = lerp(0.1, 0.01, T) ×

(1 − winning_activation)

recovery_time_RIF = lerp(300, 1800, T) \# seconds

6.5 Predictive Pre-activation

prediction_horizon = round(lerp(2, 8, F))

pre_activation_decay = lerp(0.7, 0.3, T)

prediction_conf_threshold = lerp(0.3, 0.7, F)

surprise_sensitivity = S × lerp(2.0, 0.5, T)

update_rate_on_surprise = lerp(0.2, 0.02, T) × S

6.6 Serial Position Effects

primacy_window = round(lerp(5, 2, F))

primacy_bonus = lerp(1.2, 2.0, S)

recency_window = round(lerp(7, 3, F))

rehearsal_curve_depth = lerp(0.2, 0.6, S)

distinctiveness_threshold = lerp(0.6, 0.8, F)

von_restorff_multiplier = lerp(1.5, 3.0, S)

interference_zone = positions\[primacy_window+1 : −recency_window\]

middle_suppression = lerp(0.8, 0.5, S) × (1 − F)

6.7 Emotional Consolidation

θ_intensity = lerp(0.6, 0.8, 1 − S)

θ_arousal = lerp(0.4, 0.2, S)

trigger = (emotion_intensity_t ≥ θ_intensity) AND

(arousal_t ≥ θ_arousal)

flashbulb_threshold = lerp(0.9, 0.4, S)

emotional_half_life_bonus = exp(lerp(0, ln(3), S)) ×

(1 + emotion_intensity_t)

cascade_radius = round(lerp(1, 5, S))

cascade_decay = lerp(0.7, 0.3, S)

7\. Consolidation and Graph Integration

7.1 Consolidation Triggers

should_consolidate = (db_size \> consolidation_threshold) OR

(write_rate_t \< rate_target_t / 2) OR

(elapsed_time \> consolidation_interval)

rate_consolidate = (1 / max(consolidation_interval, 1)) ×

(0.3 + 0.7T) × (1 − 0.5S)

7.1.1 Activity-Aware Scheduling

idle_required(T) = round(0.25 × w_rate_seconds(T))

idle_for = now() − last_retrieval_ts

should_start = (NOT is_processing_signal) AND

(retrieval_queue_depth == 0) AND

(idle_for ≥ idle_required(T))

7.2 Consolidation Scoring

score_consolidate(m) = w_s × strength(m) −

w_r × redundancy(m) +

w_c × connectivity(m) +

w_t × stability(m)

w_s = T; w_r = F; w_c = S; w_t = T

7.3 Clustering and Summarization

cluster_i = {m_j \| cos(m_j, μ_i) \> merge_threshold}

μ_i = centroid(cluster_i)

summary.embedding = μ_i

summary.text = summarize(cluster_i.text)

summary.metadata.sources = \[m.id for m in cluster_i\]

7.4 Semantic Extraction

extraction_batch_size = round(lerp(8, 32, T))

min_cluster_size = round(lerp(3, 10, F))

entity_frequency_threshold = round(lerp(5, 15, T))

extraction_interval = lerp(300, 3600, T) \# 5 min → 1 hour

max_extractions_per_cycle = round(lerp(20, 5, T))

7.5 Knowledge Graph Construction

7.5.1 Edge Construction

for (m_i, m_j) in cluster.sources:

cos_sim ← cos(m_i.embedding, m_j.embedding)

if cos_sim \> lerp(0.85, 0.95, F):

create_edge(m_i, m_j, \'co_occurs_with\', cos_sim)

temporal_order ← sort_by_timestamp(cluster.sources)

for i in range(len(temporal_order) − 1):

m_i, m_j ← temporal_order\[i\], temporal_order\[i+1\]

drift_vec ← m_j.embedding − m_i.embedding

drift_mag ← ‖drift_vec‖

if drift_mag \> lerp(0.15, 0.35, T):

create_edge(m_i, m_j, \'causes\', drift_mag)

7.6 Graph-Augmented Retrieval

results_vec ← topK(vector_search(x_t, k=kNN_size))

seed_nodes ← \[r.id for r in results_vec\]

expanded_nodes ← graph.traverse(seed_nodes, depth=graph_depth)

combined ← union(seed_nodes, expanded_nodes)

re_ranked ← sort_by(cos(x_t, embeddings(combined)))

8\. Interrupt Gate and Streaming Integration

8.1 Marginal Utility Computation

τ_novelty = lerp(0.10, 0.35, F) × (1 − 0.15S) × (1 + 0.3T)

τ_mu = lerp(0.08, 0.18, F) × (1 − 0.4S) × (1 + 0.4T)

retrieval_thresh(F) = lerp(0.25, 0.60, F)

Δ = cumulative_drift_since_last_interrupt

τ_refrac = lerp(24, 96, T) × lerp(1.4, 1.0, S)

k_refrac = lerp(0.20, 0.05, T) × lerp(0.8, 1.2, F)

M_refrac = 1.0 + k_refrac × exp(−Δ / τ_refrac)

τ_novelty_eff = τ_novelty × M_refrac

τ_mu_eff = τ_mu × M_refrac

8.2 Marginal Utility Score

w_cov = normalize(lerp(0.40, 0.60, F)) \# coverage gain

w_rel = normalize(lerp(0.35, 0.25, F)) \# relevance

w_red = normalize(lerp(0.15, 0.25, S)) \# redundancy penalty

w_coh = normalize(lerp(0.15, 0.25, S)) \# incoherence penalty

mu = w_cov × coverage_gain(candidate \| included_set) +

w_rel × cos(candidate, ctx_centroid) −

w_red × redundancy(candidate, included_set) −

w_coh × (1 − coherence_t)

8.3 Gate Decision Logic

dup_thresh = lerp(0.88, 0.96, F) × (0.98 + 0.02T)

K = round(lerp(10, 6, F)) \# candidates to evaluate

boundary_mult = lerp(1.3, 2.0, F) × lerp(1.1, 0.9, S)

embedding_novelty = 1 − max(cos(candidate, ctx_window))

allow_interrupt =

(max_relevance ≥ retrieval_thresh(F)) AND

(embedding_novelty ≥ τ_novelty_eff OR best_mu ≥ τ_mu_eff) AND

(max_semantic_overlap \< dup_thresh) AND

(at_drift_boundary OR best_mu ≥ boundary_mult × τ_mu_eff)

8.4 Streaming Pacing

drift_accum += dist(x_t, x\_{last_check})

pacing_thresh(S) = lerp(0.5, 0.1, S)

if drift_accum \> pacing_thresh(S): trigger_check()

max_wait_drift(F) = lerp(2.0, 0.5, F)

max_results(F) = round(lerp(64, 4, F))

adjacent_window(F) = round(lerp(8, 1, F))
