## Cortext — Algorithms Only (3‑Knob Memory)

This document contains only the algorithmic definitions extracted from the plan. Narrative text, rationale, and non‑algorithm commentary are omitted.

***

### Module map updates

* New modules:
  * `cortext.algorithms.core` — clamp, lerp, sigmoid, ewma
  * `cortext.algorithms.predictive` — Alg 22 pre-activation helpers
  * `cortext.algorithms.emotion` — Alg 23 emotional cascade
  * `cortext.algorithms.influence` — Alg 19 influence + lambda tuning
  * `cortext.algorithms.serial` — Alg 26 serial-position helpers
* Moved derives: WM derives and metacog params now in `cortext.algorithms.working_memory`.
* Re-exports: legacy symbols remain in `cortext.adaptive` with DeprecationWarnings.

### Overview — Three‑Knob Philosophy

```
- Focus (F): perceptual selectivity and precision. Higher F narrows attention, raises relevance weighting, reduces breadth.
- Sensitivity (S): plasticity and affect gain. Higher S speeds learning, raises emotional and novelty gain, increases write‑rate responsiveness.
- Stability (T): temporal persistence and inertia. Higher T lengthens half‑lives, widens hysteresis, slows updates, tightens rails over time.
```

### Principles

```
- Knobs set rates, not phases. Behavior emerges continuously; no mode switches.
- All tunables are derived via lerp, log‑scale maps, or annealed bounds tied to experience.
- Uncertainty‑modulated updates adapt fast when signals disagree and slow when stable.
```

### Self‑Calibrating Priors & Experiential Mass

We blend knob‑derived priors with live evidence using experience‑weighted averaging:

value\_t = (ρ\_prior(T) × prior(F,S,T) + ρ\_obs(t) × observed\_t) / (ρ\_prior(T) + ρ\_obs(t))
ρ\_prior(T) = prior\_mass(T) = round(lerp(2, 32, T))              # stability → stronger prior
ρ\_obs(t)   = u(t) × min(w\_score(T), count)                      # uncertainty‑weighted evidence

Experiential mass increases with both sample count and uncertainty; early, high‑u(t) environments quickly reshape priors; later, low‑u(t) keeps the system stable. Safety rails (Tmin/Tmax, max\_ΔT\_per\_min) anneal with maturity(t).

### Boot Order (continuous from signal 1)

```
1.	Boot: Compute Section 0 priors, initialize windows and counters; set maturity(0)=0.
2.	Per signal:
- Run Alg 2/4/6 to update Focus/Sensitivity/Stability priors with observations.
- Compute structural signals Alg 10–13 and uncertainty u(t).
- Update composite metric weights Alg 7 (RLS throttled).
- Evolve dynamic threshold Alg 8 (apply Δ from 4 / 5.5 / emotion).
- Apply memory reinforcement/decay Alg 14/18; integrate feedback Alg 15–19.
- Check boundary Alg 12; finalize episode if triggered.
3.	Episode finalize: batch writes, refresh caches, roll episode id.
```

#### Boot Order Pseudocode

```python
# Initialization and per-signal loop (algorithmic)
def boot():
    # Priors and rails
    rho_prior = prior_mass(T)                 # §0.2
    maturity = 0.0
    init_rolling_windows(n_ctx(T), w_score(T), w_rate_seconds(T))
    init_threshold(T_prior(F,S,T), Tmin(0), Tmax(0))
    init_feedback_caches()

def on_signal(x_t):
    # 1) Update knob families from observations
    update_focus_priors(x_t)       # Alg 2
    update_sensitivity_priors(x_t) # Alg 4
    update_stability_priors(x_t)   # Alg 6

    # 2) Structural metrics and uncertainty
    compute_coherence(x_t)         # Alg 10
    compute_focus_spread(x_t)      # Alg 11
    compute_drift(x_t)             # Alg 12 (event-only)
    compute_logprob_surprisal()    # Alg 13 if available
    u_t = update_uncertainty()     # §0.4 → α schedules

    # 3) Composite weights (throttled)
    fit_metric_weights_RLS()       # Alg 7 (every K)

    # 4) Influence + reinforcement
    apply_influence_feedback()     # Alg 19
    update_memory_strengths()      # Alg 18 (fallback Alg 14)

    # 5) Precision modulation (batched)
    maybe_emit_precision_delta()   # §5.5 → ΔThreshold_precision_t

    # 6) Threshold evolution
    update_threshold()             # Alg 8 (fuses Δ from Alg 4, Sec 5.5, and homeo)

    # 7) Boundary handling
    if drift_exceeds_threshold():  # Alg 12
        finalize_episode()

```

### Execution Cadence

```
- Per‑signal (every event): 2, 4, 6, 8, 10, 11, 12, 13, 14/18, 19, 27
- Every K signals (K≈3; throttle for cost): 7 (RLS weight fit), heavy kNN in 11/16
- Per‑episode boundary: finalize/flush (12), DB batch writes, cache rebuilds
- Periodic (timer or batch): precision modulation (5.5), ANN index maintenance
- Streaming token pacing: checks every check_interval(S) tokens; boundary patience max_wait_tokens(F)
```

### Developmental Progression (Emergent Behavior)

Phases (emergent, not hard‑coded):
\- Initial‑like: rapid capture, high plasticity, permissive thresholds. High u(t), light priors, wide rails.
\- Teenager: balanced learning; thresholds and weights stabilize; exploratory but selective.
\- Expert: slow change, high precision; long half‑lives; narrow, reliable retrieval.

### Knob roles in transitions

```
- Focus (F): higher F accelerates selectivity; narrows attention width; reduces breadth (max_results, adjacent_window) and raises precision.
- Sensitivity (S): higher S speeds adaptation and emotional gain; lowers interrupt refractory; increases write‑rate responsiveness.
- Stability (T): higher T lengthens half‑lives, widens hysteresis, tightens rails (Tmin/Tmax), and reduces update rates via α_T(t).
```

Transitions arise from annealed rails (Tmin/Tmax, max\_ΔT\_per\_min) and experiential mass (ρ\_obs vs ρ\_prior)—no explicit phase switches.

### Performance Considerations (Scaling & Ops)

```
- ANN for large stores: when memory count > 100k, use approximate nearest neighbors (HNSW, IVFPQ/FAISS, or ANNOY). Maintain a small exact cache for the most‑recent context_window_size(T) region.
- kNN neighborhood caching: cache neighborhoods within sliding context windows; invalidate on episode boundaries (Alg 12) and when topic centroid shifts beyond the drift threshold.
- Throttle high‑cost updates: run entropy/kNN heavy steps (Alg 11/16) and RLS fitting (Alg 7) every K signals (K≈2–4), not every event.
- Batch writes: buffer DB updates and commit at episode boundaries (Alg 12). Prefer WAL mode and write‑coalescing to reduce fsyncs.
- Streaming pacing: use check_interval(S) to gate token‑level checks; cap patience with max_wait_tokens(F) to avoid mid‑sentence churn.
- Index hygiene: schedule periodic ANN maintenance (rebuild/prune) during low activity; shard by episode or topic for parallelism.
```

0\. Global Knob‑Derived Functions and Constants

Canonical definitions used across all algorithms. All values are knob‑derived or annealed from experience; there are no magic numbers.

0.1 Notation & Helpers

lerp(a,b,x) = a + (b − a) × x
clamp(v, lo, hi)
sigmoid(z) = 1 / (1 + e^(−z))
EWMA(prev, x, α) = (1 − α)×prev + α×x
cos(u,v) = cosine\_similarity(u,v)
normalize(v) = v / ||v||                  # L2 normalization (vectors)
normalize(x, \[a,b]) = (x − a) / (b − a)  # range normalization (scalars)
map01(z) = clamp((z + 1)/2, 0, 1)         # maps \[-1,1] → \[0,1] for signed metrics
ln = natural log; π = 3.14159…

# Coverage gain primitive (Algorithm 27)

# Returns how much candidate expands coverage around centroid given the included\_set of vectors.

# Implementation details live in adaptive.py; returns float in \[0,1].

compute\_coverage\_gain(centroid, candidate, included\_set) -> \[0,1]

0.2 Knob‑Derived Functions

# Context windows and sliding windows

n\_ctx(T)             = round(lerp(32, 256, T))
w\_score(T)           = round(lerp(20, 120, T))
w\_rate\_seconds(T)    = round(lerp(60, 300, T))

# Write‑rate bases (writes/min)

r\_min = 0.2; r\_max = 5.0
base\_rate\_prior(S)   = lerp(r\_min, r\_max, S)

# Hysteresis band

band\_min = 0.02; band\_max = 0.25
base\_band\_prior(T)   = lerp(band\_min, band\_max, T)

# Maturity (annealed safety rails)

τ\_m(T)               = lerp(10.0, 200.0, T)

# count = number of signals/events observed to date

maturity(t)          = 1 − exp(−count/τ\_m(T))
Tmin(t)              = lerp(0.01, 0.05, maturity(t))
Tmax(t)              = lerp(0.99, 0.95, maturity(t))
max\_ΔT\_per\_min(t)    = lerp(0.30, 0.10, maturity(t))

# Half‑life (log‑scale; seconds)

hl\_min = 120.0; hl\_max = 43200.0  # 2 min → 12 h
base\_half\_life\_prior(T) = exp( ln(hl\_min) + T × ln(hl\_max/hl\_min) )

# Attention gating

attention\_width\_min = 0.1π; attention\_width\_max = π

# kNN and entropy

k\_neighbors(T)       = round(lerp(8, 32, T))
entropy\_norm(k)      = ln(k)

# Retention z‑score window

w\_ret(T)             = round(lerp(20, 120, T))

# Threshold prior and adaptation rate

T\_prior(F,S,T)       = lerp(0.10, 0.30, T) × (1 − 0.3×S)
tau(T)               = lerp(5.0, 20.0, T)

# Periphery cutoff

periphery\_cutoff(T)  = lerp(0.05, 0.25, T)

# Prior/evidence mass for Bayesian blending

prior\_mass(T)        = round(lerp(2, 32, T))
ρ\_prior(T)           = prior\_mass(T)
ρ\_obs(t)             = u(t) × min(w\_score(T), count)

0.3 Global EWMA & Uncertainty Schedules

# Uncertainty smoothing schedule

α\_u\_min = 0.10; α\_u\_span = 0.60
α\_u(T)  = α\_u\_min + (1 − T) × α\_u\_span

# Raw uncertainty sources (see §0.4). When unavailable, use fallback below.

# u\_raw(t) ∈ \[0,1]

# Smoothed uncertainty

u(t) = EWMA(u(t−1), u\_raw(t), α = α\_u(T))

# EWMA learning‑rate floors scale with uncertainty

α\_min\_F = 0.05; α\_span\_F = 0.45
α\_min\_S = 0.05; α\_span\_S = 0.45
α\_min\_T = 0.05; α\_span\_T = 0.35

α\_F(t) = max(α\_min\_F × (1 + 0.5 × u(t)), α\_min\_F + F × α\_span\_F × u(t))
α\_S(t) = max(α\_min\_S × (1 + 0.5 × u(t)), α\_min\_S + S × α\_span\_S × u(t))
α\_T(t) = max(α\_min\_T × (1 + 0.5 × u(t)), α\_min\_T + (1 − T) × α\_span\_T × u(t))

0.4 Uncertainty Computation (u\_raw)

# Primary estimator (preferred)

u\_raw(t) = normalize\_weighted\_blend(\[
var\_recent\_scores,          # variance of recent composite scores over w\_score(T)
focus\_spread\_entropy,       # Algorithm 11 (normalized entropy of kNN sims)
coherence\_complement,       # 1 − coherence from Algorithm 10
novelty\_surprise\_spikes     # fusion of Algorithm 4 novelty + Algorithm 13 surprisal
], weights = normalize(\[F, S, 1 − T]))

# Fallback when structural metrics are unavailable

u\_raw(t) = 1 − maturity(t)

0.5 Global Constants & Gains

# Feedback gains (Algorithms 15–17, 19; 5.5; 27)

αF\_base = 0.10   # Alg 15
βF\_base = 0.05   # Alg 15
η\_base  = 0.10   # Alg 16
γT\_base = 0.05   # Alg 17
λ₁ = 0.5; λ₂ = 0.4; λ₃ = 0.3   # Alg 19 mixture weights
κ\_base  = 0.10   # Section 5.5, Alg 4 (emotion), Alg 27

# Homeostatic controller (Alg 8)

κ\_r = 0.10                      # rate‑error gain
sensitivity\_gain = 0.10         # ΔT\_sens default gain when Alg 4 not provided

# Sustained horizon (Alg 19)

L\_sustain\_min = 3; L\_sustain\_max = 5

# Retrieval & pacing defaults (Alg 27 and streaming)

R\_min = 4; R\_max = 64
adjacent\_min = 1; adjacent\_max = 8
wait\_min = 16; wait\_max = 128             # tokens
check\_min\_tokens = 8; check\_max\_tokens = 64

# Emotion projection maps & temperature (Algorithm 4)

### 0.6 Emotion Projection Maps (Algorithm 4)

C = {anger, fear, joy, love, sadness, surprise}
v\_map = { anger: −0.9, fear: −0.8, sadness: −0.9, joy: +0.9, love: +0.8, surprise: 0.0 }
a\_map = { anger: +0.9, fear: +0.9, sadness: +0.3, joy: +0.6, love: +0.5, surprise: +0.8 }
β(S) = 4 + 8S   # softmax temperature for emotion distribution

### 0.7 Operational Retrieval & Streaming Parameters

Retrieval breadth and locality (Focus)

max\_results(F)        = round(lerp(R\_max, R\_min, F))
adjacent\_window(F)    = round(lerp(adjacent\_max, adjacent\_min, F))

Streaming patience and check pacing

max\_wait\_tokens(F)    = round(lerp(wait\_max, wait\_min, F))
check\_interval(S)     = round(lerp(check\_max\_tokens, check\_min\_tokens, S))   # tokens

Aliases (clarity)

window\_size(T)         = w\_score(T)
context\_window\_size(T) = n\_ctx(T)
density\_k(T)           = k\_neighbors(T)

***

### Summary Tables

#### Metric‑Specific Adaptive Algorithms (Knob Dependencies)

```
Metric	Knob Dependence	Expression (normalized domains)
Relevance	↑F	cos(x, mean(ctx)) × (0.5 + F)
Mismatch	↓F, ↑S	(1 − F) × S × novelty(x)
Surprise	↑S, ↓T	entropy_spike × S × (1 − T)
Rarity	↑F, ↓T	(1 − mean_sim) × (0.5 + 0.5F) × (1 − 0.2T)
Drift	↓T	‖mean(ctx_t) − mean(ctx_{t−k})‖ × (1 − T)
Contradiction	S vs F	max(0, S − F)
Utility (ΔSSE)	↑F, ↓S	ΔSSE × (0.5 + 0.5F) × (1 − 0.3S)
Periphery	↑T	1 − decay(mean_sim, half_life_T)
Coverage	↑F	avg(max(0, cos(x,q) − top_db(q))) × F
Salience	F,S	(rarity + novelty_recent)/2 × (F + S)/2
Valence	S, ↓T	map_to_[0,1](Σ p_c v_map[c])
Arousal	S, ↓T	clamp(Σ p_c a_map[c], 0, 1)

```

#### T‑Related Variables Summary

```
Name	Meaning	Source	Cadence	Range/Clamp
T	Stability knob	User	static	[0,1]
α_T(t)	EWMA schedule	§0.3	per‑signal	[0.05,0.40]
base_half_life_prior(T)	Half‑life prior (sec)	§0.2	boot	[hl_min, hl_max]
half_life_t	Dynamic half‑life	Alg 6	per‑signal	[hl_min, hl_max]
w_score(T)	Score window	§0.2	static	[20,120]
w_ret(T)	Retention window	§0.2	static	[20,120]
Tmin(t),Tmax(t)	Annealed bounds	§0.2	per‑signal	widen early → tighten
max_ΔT_per_min(t)	Rate limiter	§0.2	per‑signal	0.30 → 0.10
T_dynamic_t	Write threshold	Alg 8	per‑signal	[Tmin,Tmax]
ΔThreshold_*	Threshold deltas	Algs 4/5.5/27	per‑signal	±max_ΔT_per_min
hysteresis_t	Hysteresis band	Alg 8	per‑signal	[band_min,band_max]
ρ_prior(T)	Prior mass	§0.2	boot	[2,32]
ρ_obs(t)	Evidence mass	§0.2	per‑signal	[0,w_score(T)]
u(t)	Uncertainty	§0.4	per‑signal	[0,1]
maturity(t)	Experience	§0.2	per‑signal	[0,1]

```

### Memory Table Enhancements — Schema

```
Column	Type	Purpose
retrieved_count	INTEGER	Retrieval frequency
used_count	INTEGER	Times used in active context
contextual_gain	REAL	Logprob improvement
mean_influence	REAL	Average impact across cycles
last_used	INTEGER	Timestamp of last positive influence
stability	REAL	Per‑memory stability multiplier
use_frequency	REAL	EWMA usage for the memory
episode_id	INTEGER	Episode id at last update

```

### Generation Trace Table — Schema

```
Column	Type	Purpose
embedding	BLOB	Output embedding vector
timestamp	INTEGER	Generation time
topic_id	INTEGER	Topic cluster id
semantic_drift	REAL	Δ direction vs previous output

```

### Signal Metrics Table — Schema

Per-signal metrics persisted for observability, debugging, and adaptive weight learning. Each row corresponds to a processed signal event.

```
Column	Type	Purpose
id	INTEGER	Primary key (autoincrement)
timestamp	INTEGER	Signal processing time
embedding_id	INTEGER	FK to memory (if written, else NULL)
relevance	REAL	cos(x, mean(ctx)) × (0.5 + F)
mismatch	REAL	(1 − F) × S × novelty
surprise	REAL	entropy_spike × S × (1 − T)
rarity	REAL	(1 − mean_sim) × (0.5 + 0.5F) × (1 − 0.2T)
drift	REAL	‖mean(ctx_t) − mean(ctx_{t−k})‖ × (1 − T)
contradiction	REAL	max(0, S − F)
utility	REAL	ΔSSE × (0.5 + 0.5F) × (1 − 0.3S)
periphery	REAL	1 − decay(mean_sim, half_life_T)
coverage	REAL	avg(max(0, cos(x,q) − top_db(q))) × F
salience	REAL	(rarity + novelty_recent)/2 × (F + S)/2
valence	REAL	Emotional valence mapped to [0,1]
arousal	REAL	Emotional arousal in [0,1]
composite_score	REAL	Weighted blend of above metrics
threshold_t	REAL	Dynamic threshold at time t
write_decision	INTEGER	1 if memory written, 0 if gated

```

**Usage:**
- Enables post-hoc analysis of write gate decisions
- Supports RLS weight fitting validation (Algorithm 7)
- Provides training data for adaptive threshold tuning
- Facilitates debugging of threshold/sensitivity interactions

### Processor State Table — Schema

Persisted adaptive state enabling algorithm resumption across restarts. Single-row table (id=1) storing all evolving parameters.

```
Column	Type	Purpose
id	INTEGER	Primary key (always 1)
signals_processed	INTEGER	Total signals seen (maturity denominator)
u_t	REAL	Current uncertainty estimate [0,1]

-- Focus State (Algorithms 1, 2, 15)
weight_relevance_prior	REAL	Prior from sigmoid(2F − 1)
coverage_gain_floor_prior	REAL	Prior: 0.3 + 0.7F
mismatch_weight_prior	REAL	Prior: 1 − F
attention_width_prior	REAL	Prior: lerp(0.1π, π, 1 − F)
weight_relevance	REAL	Dynamic EWMA value
attention_width	REAL	Dynamic attention width

-- Sensitivity State (Algorithms 3, 4, 16)
base_rate_prior	REAL	Prior: lerp(0.2, 5.0, S)
weight_novelty_prior	REAL	Prior: 0.3 + 0.7S
weight_surprise_prior	REAL	Prior: 0.2 + 0.8S
weight_valence_prior	REAL	Prior: 0.4 + 0.6S
weight_arousal_prior	REAL	Prior: S
weight_emotion_prior	REAL	Prior: 0.2 + 0.8S
emotion_gain_prior	REAL	Prior: exp(1.5S)
score_gain_prior	REAL	Prior: exp(2S)
rate_target_prior	REAL	Prior: base_rate × (0.5 + 1.5S)
rate_target	REAL	Dynamic rate target
weight_novelty	REAL	Dynamic novelty weight

-- Stability State (Algorithms 5, 6, 17)
hysteresis_band_prior	REAL	Prior hysteresis band
half_life_prior	REAL	Prior half-life (seconds)
rate_decay_prior	REAL	Prior rate decay factor
periphery_half_life_prior	REAL	Prior periphery half-life
salience_half_life_prior	REAL	Prior salience half-life
drift_weight_prior	REAL	Prior drift weight
half_life	REAL	Dynamic half-life
rate_decay	REAL	Dynamic rate decay
periphery_half_life	REAL	Dynamic periphery half-life
salience_half_life	REAL	Dynamic salience half-life

-- Threshold State (Algorithm 8)
T_dynamic	REAL	Current adaptive threshold
hysteresis	REAL	Current hysteresis band
dt_ema	REAL	Delta-time EMA for rate control
m_rate	REAL	Write rate EMA
rate_ticks	INTEGER	Tick counter for rate estimation

-- Influence State (Algorithm 19)
sustained_influence	REAL	Accumulated influence feedback

-- Timestamps
last_signal_timestamp	INTEGER	Last signal processing time
last_rate_timestamp	INTEGER	Last rate update time
last_consolidation_ts	INTEGER	Last consolidation time
last_retrieval_ts	INTEGER	Last retrieval time
updated_at	INTEGER	Row update timestamp

```

### Blender Weights Table — Schema

RLS-fitted metric weights for composite scoring (Algorithm 7). Separate table for the 12 metric weights.

```
Column	Type	Purpose
id	INTEGER	Primary key (always 1)
w_relevance	REAL	Weight for relevance metric
w_mismatch	REAL	Weight for mismatch metric
w_surprise	REAL	Weight for surprise metric
w_rarity	REAL	Weight for rarity metric
w_drift	REAL	Weight for drift metric
w_contradiction	REAL	Weight for contradiction metric
w_utility	REAL	Weight for utility metric
w_periphery	REAL	Weight for periphery metric
w_coverage	REAL	Weight for coverage metric
w_salience	REAL	Weight for salience metric
w_valence	REAL	Weight for valence metric
w_arousal	REAL	Weight for arousal metric
blender_ready	INTEGER	1 if RLS has converged
update_count	INTEGER	Number of RLS updates performed

```

### Blender Covariance Table — Schema

RLS covariance matrix P for online weight learning (Algorithm 7). Stored as flattened 12×12 matrix.

```
Column	Type	Purpose
id	INTEGER	Primary key (always 1)
P_matrix	BLOB	Flattened 12×12 covariance matrix (144 doubles)

```

### Recent Context Table — Schema

Rolling window of recent context embeddings for relevance and drift computation.

```
Column	Type	Purpose
id	INTEGER	Primary key (autoincrement)
embedding	BLOB	256d context embedding
timestamp	INTEGER	When this context was recorded
seq_order	INTEGER	Sequence position in window

```

### Recent Scores Table — Schema

Rolling window of recent composite scores for threshold adaptation.

```
Column	Type	Purpose
id	INTEGER	Primary key (autoincrement)
score	REAL	Composite score value
timestamp	INTEGER	When this score was recorded

```

**Resumption Protocol:**
1. On startup, load `processor_state` row (or initialize from knob priors if absent)
2. Load `blender_weights` and `blender_covariance` for RLS state
3. Load last N rows from `recent_context` and `recent_scores` based on w_score(T)
4. Resume processing with restored state — algorithms continue evolution seamlessly

### Feedback Summary

```
Class	Feedback Signal	Effect
Focus	contextual_gain>0	Narrows gating, boosts relevance precision
Sensitivity	novelty × gain	Accelerates valuable learning
Stability	Sustained gain	Lengthens half‑life of useful memories

```

⸻

1. Focus‑Driven Selectivity

Algorithm 1: Focus Priors and Continuous Update

Input: F ∈ \[0,1]
Output: relevance/gating priors

1. weight\_relevance\_prior   ← sigmoid(2F − 1)          # ∈ \[0,1]
2. coverage\_gain\_floor\_prior← 0.3 + 0.7F
3. mismatch\_weight\_prior    ← (1 − F)
4. attention\_width\_prior    ← lerp(attention\_width\_min, attention\_width\_max, 1 − F)  # radians

Algorithm 2: Dynamic Update per Signal (Focus)

At each signal event t:
recent\_context ← last n\_ctx(T) items
observed\_cosine ← cos(x\_t, mean(recent\_context))
weight\_relevance\_t ← EWMA(weight\_relevance\_{t−1}, observed\_cosine, α = α\_F(relevance))
attention\_width\_t ← clamp(attention\_width\_{t−1}, \[attention\_width\_min, attention\_width\_max])

***

2. Sensitivity‑Driven Plasticity

Algorithm 3: Sensitivity Priors

Input: S ∈ \[0,1]
Output: emotional gain, novelty weighting, and base write rate priors

0. base\_rate\_prior   ← base\_rate\_prior(S)  # see global conventions
1. weight\_novelty\_prior  ← 0.3 + 0.7S
2. weight\_surprise\_prior ← 0.2 + 0.8S
3. weight\_valence\_prior  ← 0.4 + 0.6S
4. weight\_arousal\_prior  ← S
5. weight\_emotion\_prior  ← 0.2 + 0.8S                 # strength for emotion\_intensity\_t as a modulator
6. emotion\_gain\_prior    ← exp(1.5S)
7. score\_gain\_prior      ← exp(2S)
8. rate\_target\_prior     ← base\_rate\_prior × (0.5 + 1.5S)

Algorithm 4: Dynamic Update per Signal (Sensitivity)

At each signal event t:
\# Discrete emotion projections (if centroids available)
if centroids available:
raw\_cos\_c ← cos(x\_t, centroids\[c])
if ∀c, raw\_cos\_c ≤ 0:
emotion\_intensity\_t ← 0
valence\_t ← 0.5
arousal\_t ← 0.0
else:
logits\_c ← max(0, raw\_cos\_c)
p\_c ← softmax(β(S) × logits\_c)
peak ← max\_c p\_c
confidence ← 1 − entropy(p\_c)/ln(6)
emotion\_intensity\_t ← sqrt(peak × confidence)
valence\_raw ← Σ\_c p\_c × v\_map\[c]
valence\_t ← (valence\_raw + 0.9)/1.8
arousal\_t ← clamp(Σ\_c p\_c × a\_map\[c], 0, 1)
else:
\# Fallback already produces valence\_t, arousal\_t; set emotion\_intensity\_t ← 0
emotion\_intensity\_t ← 0

```
novelty_t ← measure_novelty(x_t)
surprise_t ← entropy_spike(x_t)
w ← w_score(T)
gain_t ← exp(2S) × (1 + 0.1 × variance(scores_{t−w:t}))
rate_target_t ← EWMA(rate_target_{t−1}, observed_write_rate(w_rate_seconds), α = α_S(rate))

# Emotion loosens thresholds proportionally to intensity and arousal (capture salient moments)
κ_emo ← κ_base × S
ΔThreshold_emotion_t ← − κ_emo × emotion_intensity_t × (0.5 + 0.5 × arousal_t)

ΔThreshold_sensitivity_t ← (− gain_t × (S − 0.5) + ΔThreshold_emotion_t)
# This raw delta is passed to Algorithm 8 for final clamping
```

Parameter derivation for metric weights:
For each metric m (after sufficient history):
μ\_m ← mean(metric\_m over w\_score(T))
σ\_m ← std(metric\_m over w\_score(T))
base\_m ← normalize(μ\_m + σ\_m) → \[0,1]  # Target one std dev above mean
weight\_m\_t ← EWMA(weight\_m\_{t-1}, base\_m × (0.5 + 0.5S), α = α\_S(weight))

Algorithm 4b: The Mood Integrator (Tonic State)

Purpose: Maintain a persistent background state (M_t) distinct from instantaneous emotion (e_t).

Inputs:
- e_t: Instantaneous emotion vector (from Alg 4)
- M_{t-1}: Previous mood state (6d vector)
- S, T: Sensitivity and Stability knobs

Dynamics:
1. Reactivity (α_mood): α_mood(S) = lerp(0.01, 0.20, S)
2. Decay (λ_mood):     λ_mood(T) = lerp(0.90, 0.999, T)

Update Equation:
M_t = (λ_mood(T) × M_{t-1}) + (α_mood(S) × e_t)
M_t ← clamp(M_t, -1.0, 1.0)  # simple stability clamp

Output:
- M_t: Current mood vector
- ||M_t||: Mood magnitude used for threshold bias

Note on Storage (Dual-Signal):
When writing to memory, store both:
- event_emotion: e_t
- ambient_mood: M_t
This enables "Prosthetic Mode" attribution during retrieval.

***

3. Stability‑Driven Persistence

Algorithm 5: Stability Priors

Input: T ∈ \[0,1]
Output: decay, hysteresis, and persistence priors

1. hysteresis\_band\_prior  ← base\_band\_prior(T)
2. half\_life\_prior        ← base\_half\_life\_prior(T)
3. rate\_decay\_prior       ← lerp(0.60, 0.98, T)
4. periphery\_half\_life\_pr ← clamp(0.5 × half\_life\_prior, hl\_min, hl\_max)
5. salience\_half\_life\_pr  ← clamp(0.5 × half\_life\_prior, hl\_min, hl\_max)
6. drift\_weight\_prior     ← 0.5 × (1 − T)

Algorithm 6: Dynamic Update per Signal (Stability)

At each signal event t:
active\_memories ← { m | strength(m) ≥ periphery\_cutoff(T) }
observed\_retention ← avg\_age(active\_memories)
hysteresis\_t ← EWMA(hysteresis\_{t−1}, observed\_retention, α = α\_T(hysteresis))
last\_w\_ret ← last w\_ret(T) values of observed\_retention
μ\_ret ← mean(last\_w\_ret); σ\_ret ← max(std(last\_w\_ret), 1.0)
zscore\_ret ← clamp((observed\_retention − μ\_ret)/σ\_ret, −3, +3)
stability\_adj ← (ΔHalfLife\_adj\_t is provided) ? ΔHalfLife\_adj\_t : 0
target\_half\_life\_t ← clamp(base\_half\_life\_prior(T) × (1 + 0.25 × zscore\_ret + stability\_adj), hl\_min, hl\_max)
half\_life\_t ← EWMA(half\_life\_{t−1}, target\_half\_life\_t, α = α\_T(half\_life))
rate\_decay\_t ← clamp(rate\_decay\_{t−1}, \[0,1])

⸻

4. Composite Adaptation

Algorithm 7: Metric Weight Blending (Continuous Adaptation)

For each metric i:
weight\_i(t) ← sigmoid(a\_F\[i]×F + a\_S\[i]×S + a\_T\[i]×T + b\_i)
where coefficients (a\_F, a\_S, a\_T, b) are estimated via RLS
over the past N signals to minimize (observed\_score − predicted\_score)

Scheduling:
N ← round(lerp(64, 512, T))
Update via RLS with forgetting φ(T) = 0.90 + 0.09T (online)

Bootstrap + blend:
w\_bootstrap\[i] ← sigmoid(c\_F\[i]×F + c\_S\[i]×S + c\_T\[i]×T + d\_i)
w\_rls\[i]       ← sigmoid(a\_F\[i]×F + a\_S\[i]×S + a\_T\[i]×T + b\_i)
τ\_rls ← lerp(20.0, 80.0, T)
confidence\_rls ← 1 − exp(−t / τ\_rls)
weight\_i(t) ← (1 − confidence\_rls) × w\_bootstrap\[i] + confidence\_rls × w\_rls\[i]

Composite Score Calculation

Given:
metrics\_norm = {relevance: v₁, mismatch: v₂, ..., arousal: v₁₂}    # each vᵢ ∈ \[-1,1] or \[0,1]
weights\_0\_1   = {relevance: w₁, mismatch: w₂, ..., arousal: w₁₂}  # raw weights in \[0,1]

Preprocess (per metric definition):
m01\[i] = (metric is signed \[-1,1]) ? map01(metrics\_norm\[i]) : clamp(metrics\_norm\[i], 0, 1)

Compute:
weight\_sum ← Σ(wᵢ)
if weight\_sum < ε: return 0
weights\_normalized\[i] ← wᵢ / weight\_sum
score\_norm ← Σ(weights\_normalized\[i] × m01\[i])
score ← clamp(score\_norm, 0, 1)

Critical Implementation Note (Expanded)

* Always normalize weights to sum to 1.0 before averaging. Example: with 12 metrics and raw weights ≈0.6 each, Σw ≈ 7.2; without normalization, the weighted sum would saturate above 1.0 and collapse variance.
* Clamp inputs: each metric must be clipped to its normalized domain; convert signed values to \[0,1] via map01 before weighting; replace NaN/Inf with safe defaults (0).
* Non‑negativity: ensure all weights ≥0; if an adaptive fit proposes negatives, zero them and renormalize.
* Observability: emit weight\_sum and effective\_metric\_count each update to catch silent saturation.

⸻

5. Dynamic Thresholding

Algorithm 8: Adaptive Threshold Evolution (Unified, Bayesian Prior/Evidence, Homeostatic)

Input: scores\[0:t], knobs (F, S, T), optional ΔThreshold\_sensitivity\_t (Alg 4), optional ΔThreshold\_precision\_t (Sec. 5.5), optional ΔThreshold\_mood\_t (Alg 4b)
Output: T\_dynamic

1. w ← w\_score(T)
2. count ← len(scores)
3. T\_prior ← T\_prior(F, S, T)
4. observed\_p90 ← percentile(scores\_{t−w:t}, 90)
5. ρ\_prior ← prior\_mass(T);  ρ\_obs ← u(t) × min(w, count)
6. T\_target ← (ρ\_prior × T\_prior + ρ\_obs × observed\_p90) / max(1e−6, ρ\_prior + ρ\_obs)
7. T\_dynamic\_t ← EWMA(T\_dynamic\_{t−1}, T\_target, α = α\_T(t))
8. ΔT\_sens ← (ΔThreshold\_sensitivity\_t is provided) ? ΔThreshold\_sensitivity\_t : (S − 0.5) × sensitivity\_gain
9. Continuous‑time rate control (EMA + ESS):
   a) Δt ← now − last\_timestamp; last\_timestamp ← now; Δt ← max(Δt, 1e−3)
   b) α\_dt ← 1 − exp(−Δt/max(Δt,1e−3)); dt\_ema ← (1 − α\_dt)×dt\_ema + α\_dt×Δt; dt\_base ← max(dt\_ema, 1.0)
   c) τ\_rate ← max(2^(3T) × dt\_base, 1.0); α ← 1 − exp(−Δt/τ\_rate)
   d) ρ\_inst ← (Δwrites/Δt) × 60; m\_rate ← (1 − α)×m\_rate + α×ρ\_inst; denom ← max(1 − (1 − α)^(ticks+1), 1e−6); ρ\_hat ← m\_rate/denom
   e) β ← max(0, 1 − α); ESS ← min((1 + β)/max(1 − β, 1e−6), 100); reliability ← 1 − exp(−ESS × (1 − T))
   f) rate\_error ← tanh((ρ\_hat − rate\_target\_t) / max(rate\_target\_t, 1e−6))
   g) cap ← 0.25 × hysteresis\_t; ΔT\_homeo ← clamp(reliability × κ\_r × (1 − T) × (1 − maturity(t)) × rate\_error, −cap, +cap)
10. ΔT\_prec ← (ΔThreshold\_precision\_t is provided) ? ΔThreshold\_precision\_t : 0
11. ΔT\_emo ← (ΔThreshold\_emotion\_t is provided) ? ΔThreshold\_emotion\_t : 0
12. ΔT\_mood ← (ΔThreshold\_mood\_t is provided) ? ΔThreshold\_mood\_t : 0
13. ΔT ← ΔT\_sens + ΔT\_homeo + ΔT\_prec + ΔT\_emo + ΔT\_mood
14. ΔT\_limited ← clamp(ΔT, −max\_ΔT\_per\_min(t), +max\_ΔT\_per\_min(t))
15. T\_dynamic\_t ← clamp(T\_dynamic\_t + ΔT\_limited, Tmin(t), Tmax(t))
16. hysteresis\_t ← lerp(band\_min, band\_max, T)

Telemetry (diagnostics): Δt, dt\_base, τ\_rate, α, β, ESS, reliability, ρ\_inst, ρ\_hat, rate\_target\_t, rate\_error, κ\_r (gain), m\_gate = (1 − T) × (1 − maturity), cap, ΔT\_homeo, ΔT\_total, T\_prior, T\_dynamic\_t, hysteresis\_t

⸻

6. Embedding‑Derived Structural Metrics

Algorithm 10: Coherence / Integration

Input: x\_t, context\_window
Output: coherence\_t

1. raw ← var(cos(x\_t, context\_window))
2. coherence\_t ← 1 − clamp(raw, 0, 1)                # normalized to \[0,1]
3. Use as stabilizing term: F\_eff = F × (0.5 + 0.5×coherence\_t)

Algorithm 11: Contextual Entropy / Focus Spread

Input: kNN\_similarities
Output: focus\_spread\_t

1. k ← k\_neighbors(T)
2. p ← softmax(kNN\_similarities)
3. focus\_spread\_t ← entropy(p) / entropy\_norm(k)      # ∈ \[0,1]
4. Modifier: F\_eff ← F × (1 − focus\_spread\_t)

Algorithm 12: Trajectory Drift / Temporal Gradient

**Pseudocode — finalize\_episode() (Algorithm 12 extension)**

````python
def finalize_episode():
    db.begin_txn()
    db.write_episode_summary(current_episode)
    current_episode.id += 1
    retrieval_buffer.clear()
    novelty_cache.reset()
    # maintain recent_context to last n_ctx(T)
    recent_context = tail(recent_context, n_ctx(T))
    current_episode.boundary_ts = now()
    db.commit_txn()

Input: mean(ctx_t), mean(ctx_{t−k})
Output: drift_mag_t
	1.	drift_vec_t ← normalize(mean(ctx_t)) − normalize(mean(ctx_{t−k}))
	2.	drift_mag_t ← ||drift_vec_t||
	3.	drift_threshold ← lerp(0.10, 0.35, T)
	4.	Event: if drift_mag_t > drift_threshold → commit episodic boundary

### Algorithm 13: Logprob‑Derived Surprise

Input: token_logprobs[0:n]
Output: logprob_surprisal_t
	1.	mean_logp ← mean(−log(token_logprobs))
	2.	logprob_surprisal_t ← clamp(mean_logp / 5.0, 0, 1)
	3.	surprise_t ← 1 − (1 − drift_mag_t) × (1 − logprob_surprisal_t)

---

## 7. Reinforcement and Decay Dynamics

### Algorithm 14: Memory Strength Adjustment (base)

For each memory m at time t:
use_frequency_t ← EWMA(use_frequency_{t−1}, used_flag(m), α = α_S(use))
λ(T) ← ln(2) / base_half_life_prior(T)
strength_t ← strength_{t−1} + (S × use_frequency_t) − λ(T)
if strength_t < periphery_cutoff(T): evict(m)

---

## 8. Causal Retrieval Feedback Loop

### Algorithm 15: Focus Feedback Adjustment

for each used memory m:
if contextual_gain(m) > 0:
weight_relevance_t ← clamp(weight_relevance_t + αF × contextual_gain(m), 0, 1)
attention_width_t ← clamp(attention_width_t × (1 − βF), attention_width_min, attention_width_max)
else:
attention_width_t ← clamp(attention_width_t × (1 + βF), attention_width_min, attention_width_max)

### Algorithm 16: Sensitivity Feedback Adjustment

for each used memory m:
novelty_reward ← 1 − sim(m.embedding, recent_context)
weight_novelty_t ← clamp(weight_novelty_t + η × (novelty_reward × contextual_gain(m) − redundancy(m)), 0, 1)

### Algorithm 17: Stability Feedback Adjustment

for each used memory m:
if contextual_gain(m) > 0:
stability(m) ← stability(m) + γT
else:
stability(m) ← stability(m) × (1 − γT)

# Calculate stability adjustment factor
adj ← clamp(mean(stability(m_used)) − 1.0, −0.25, +0.25)

# Define a delta for the half-life target, to be used by Alg 6
# This stops Alg 17 from fighting with Alg 6's z-score logic
ΔHalfLife_adj_t ← adj  # (Passes the adjustment factor, e.g., -0.1 or +0.2)

### Algorithm 18: Influence‑Weighted Update

For each memory m:
denom ← max(1, retrieved_count)
influence_factor = (used_count / denom) × clamp(contextual_gain(m), -1, +1)
strength_t ← strength_{t−1} + (S × use_frequency_t) + (F × influence_factor) − λ(T)

### Adaptive Threshold Modulation (5.5)

retrieval_precision_t = used_count_total / max(1, retrieved_count_total)
if retrieval_precision_t < target_precision:
κ ← 0.10 × (1 − T)
ΔThreshold_precision_t ← − κ × (target_precision − retrieval_precision_t)
else:
ΔThreshold_precision_t ← 0

---

## 9. Generation Embedding Feedback

### Algorithm 19: Logprob + Embedding Influence Feedback

Δḡ ← ḡ_t − ḡ_{t−1}
Δḡ̂ ← normalize(Δḡ)
drift_contribution(m) = ||Δḡ|| × max(0, cos(m.embedding, Δḡ̂))

for each used memory m:
influence(m) = λ₁ * contextual_gain(m)
+ λ₂ * sim_gen(m)
− λ₃ * drift_contribution(m)

# apply to derived parameters (not knobs):
attention_width_t ← clamp(attention_width_t × (1 − 0.05 × influence(m)), attention_width_min, attention_width_max)
rate_target_t ← EWMA(rate_target_t, rate_target_t × (1 + 0.10 × influence(m)), α = α_S(rate))
L_sustain(T) = round(lerp(3, 5, T))
sustained_influence ← EWMA(sustained_influence, influence(m), α = 2/(L_sustain(T)+1))
hysteresis_t ← clamp(hysteresis_t × (1 + 0.05 × sustained_influence), band_min, band_max)

---

## 10. Advanced Cognitive Dynamics

### Algorithm 20: Memory Reconsolidation During Retrieval

τ_labile = lerp(30, 300, T) seconds
drift_magnitude = (1 − T) × S × lability × contextual_relevance
ripple_decay = lerp(0.5, 0.1, T)      # per semantic hop
reconsolidation_gain = lerp(0.2, 0.02, T)
lability_susceptibility = (1 − T) × (0.5 + 0.5 × S)

### Algorithm 21: Retrieval Competition & Interference

inhibition_radius = lerp(0.5, 0.85, F)
winners_k = round(lerp(7, 3, F))
suppression_per_retrieval = lerp(0.1, 0.01, T) × (1 − winning_activation)
recovery_time_RIF = lerp(60, 600, T) seconds
lateral_inhibition_strength = F × (1 + 0.5 × S)
competition_iterations = round(lerp(3, 10, F))

### Algorithm 22: Predictive Memory Pre‑activation

prediction_horizon = round(lerp(2, 8, F))
pre_activation_decay = lerp(0.7, 0.3, T)
prediction_conf_threshold = lerp(0.3, 0.7, F)
surprise_sensitivity = S × lerp(2.0, 0.5, T)
trajectory_samples = round(lerp(5, 1, F))
update_rate_on_surprise = lerp(0.2, 0.02, T) × S

### Algorithm 23: Emotional Consolidation Tags

Trigger:
trigger ← (emotion_intensity_t ≥ θ_intensity) AND (arousal_t ≥ θ_arousal)
θ_intensity = lerp(0.6, 0.8, 1 − S)
θ_arousal   = lerp(0.4, 0.2, S)

Knob‑derived parameters:
flashbulb_threshold = lerp(0.9, 0.4, S)
cascade_radius = round(lerp(1, 5, S))
cascade_decay = lerp(0.7, 0.3, S)
emotional_half_life_bonus = exp(lerp(0, ln(3), S)) × (1 + emotion_intensity_t)
detail_suppression = S × (1 − F) × 0.5
gist_components = round(lerp(5, 2, F))

Effective threshold:
flashbulb_threshold_eff = flashbulb_threshold × (1 − 0.5 × emotion_intensity_t) × (1 − 0.3 × arousal_t)

### Algorithm 24: Working Memory Gates & Maintenance

base_capacity = round(lerp(9, 5, S) + lerp(-2, 2, F))
maintenance_cost_per_slot = lerp(0.05, 0.15, S)
# entropy_signal is the SIGNAL complexity (e.g., token‑level entropy when available; 
# otherwise an embedding‑based proxy such as std over dimensions). 
# It is NOT WM‑state entropy.
complexity_penalty = entropy_signal × lerp(0.5, 1.5, S)
chunking_threshold = lerp(0.7, 0.9, F)
gate_threshold = lerp(0.1, 0.4, F)
rehearsal_rate = lerp(0.5, 2.0, S)
slot_dedication_strength = lerp(0.3, 0.9, T)

### Algorithm 25: Metacognitive Monitoring

FOK_threshold = lerp(0.2, 0.5, F)
TOT_detection = FOK > lerp(0.5, 0.8, F) AND retrieval < lerp(0.4, 0.2, F)
confidence_decay_rate = lerp(0.01, 0.1, 1 − T)
unknown_threshold = lerp(0.3, 0.1, F)
strategy_switch_latency = lerp(500, 100, S)   # ms
certainty_requirement = lerp(0.6, 0.9, T)
metacognitive_sensitivity = F × (1 + 0.5 × S)

### Algorithm 26: Serial Position Effects

primacy_window = round(lerp(5, 2, F))
primacy_bonus = lerp(1.2, 2.0, S)
recency_window = round(lerp(7, 3, F))
rehearsal_curve_depth = lerp(0.2, 0.6, S)
distinctiveness_threshold = lerp(0.6, 0.8, F)
von_restorff_multiplier = lerp(1.5, 3.0, S)
interference_zone = positions [primacy_window+1 : −recency_window]
middle_suppression = lerp(0.8, 0.5, S) × (1 − F)

Zone determination (for observability):
- if position ≤ primacy_window → "primacy"
- elif position ≥ N − recency_window + 1 → "recency"
- elif rarity > distinctiveness_threshold → "distinctive"
- else → "middle"

***

## 11. Interrupt Gate

### Algorithm 27: Marginal Utility Novelty (MNI) Gate — v2.1

**Knob‑Derived Parameters (no magic numbers):**

Novelty thresholds

tau_jaccard = lerp(0.10, 0.35, F) * (1 - 0.15S) * (1 + 0.3T)
tau_mu      = lerp(0.08, 0.18, F) * (1 - 0.4S)  * (1 + 0.4T)

Refractory (tokens since last interrupt)

Delta      = tokens_since_last_interrupt
tau_refrac = lerp(24, 96, T) * lerp(1.4, 1.0, S)
k_refrac   = lerp(0.20, 0.05, T) * lerp(0.8, 1.2, F)
M_refrac   = 1.0 + k_refrac * exp(-Delta / tau_refrac)

Effective thresholds under refractory pressure

tau_jaccard_eff = tau_jaccard * M_refrac
tau_mu_eff      = tau_mu      * M_refrac

Duplicate suppression and Top‑K

dup_thresh = lerp(0.88, 0.96, F) * (0.98 + 0.02*T)
K          = round(lerp(10, 6, F))

Marginal utility weights (normalized, knob‑derived)

w_cov_raw = lerp(0.40, 0.60, F)      # coverage gain
w_rel_raw = lerp(0.35, 0.25, F)      # relevance
w_red_raw = lerp(0.15, 0.25, S)      # redundancy penalty
w_coh_raw = lerp(0.15, 0.25, S)      # incoherence penalty
total_w = w_cov_raw + w_rel_raw + w_red_raw + w_coh_raw
w_cov = w_cov_raw / total_w
w_rel = w_rel_raw / total_w
w_red = w_red_raw / total_w
w_coh = w_coh_raw / total_w

**MU Computation (per new candidate):**

mu = (
w_cov * coverage_gain(candidate | included_set)
	+ w_rel * cos(candidate_vec, ctx_centroid)

	- w_red * redundancy(candidate_vec, included_vecs)
	- w_coh * coherence_penalty   # = 1 - coherence (Alg 10)
)

**Boundary‑Aware Override:**

boundary_mult = lerp(1.3, 2.0, F) * lerp(1.1, 0.9, S)

**Gate Decision:**

# Gate logic:
# 1. Must pass base relevance and duplication checks.
# 2. Must pass base novelty (jaccard or mu).
# 3. Must EITHER be at a boundary, OR pass a *stricter* mu threshold (boundary_mult)
# This suppresses mid-sentence interrupts that are only "medium" novel.
allow_interrupt = (
max_relevance >= retrieval_thresh and
(jaccard >= tau_jaccard_eff or best_mu >= tau_mu_eff) and
max_semantic_overlap < dup_thresh and
(at_boundary or best_mu >= boundary_mult * tau_mu_eff)
)

**Observability payload (on suppression):**

```
{
  "event": "INTERRUPT_SUPPRESSED",
  "reason": "duplicate | below_threshold | low_novelty | not_boundary",
  "jaccard": ,
  "best_mu": ,
  "max_semantic_overlap": ,
  "dup_thresh": ,
  "effective_threshold": ,
  "refractory_multiplier": ,
  "Delta_tokens": ,
  "active_context_size": ,
  "max_relevance": ,
  "boundary": ,
  "boundary_mult": ,
  "K": ,
  "weights": {"cov": , "rel": , "red": , "coh": }
}
````

On interrupt trigger, include:

```
{
  "active_context_size_before": ,
  "active_context_size_after": ,
  "suppression_count_since_last": 
}
```

**Minimal state:** `active_context_ids`, `last_interrupt_token`, `ctx_centroid`, `coherence_penalty`.

## 12. Consolidation and Graph Integration System

Status: Planned — Last Updated: November 2, 2025

***

### Overview

The Consolidation and Graph Integration System extends cortext with long‑term semantic organization. It periodically compresses redundant memories, merges related entries, and constructs a knowledge graph that captures persistent relationships, co‑occurrences, and causal patterns. The graph enhances recall precision, abstraction, and reasoning while preserving low‑latency stream operations. Consolidation runs in the background, triggered by capacity, write‑rate slowdown, or time‑based cadence, transforming episodic memories into semantic structures.

***

### 12.1 Consolidation Triggers

### Algorithm 28: Trigger Evaluation

```
Input: write_rate_t, db_size, consolidation_threshold, rate_target_t, elapsed_time, consolidation_interval, gemma_available

# Capacity or rate-based triggers (always run clustering)
if (db_size > consolidation_threshold)
   or (write_rate_t < rate_target_t / 2)
   or (elapsed_time > consolidation_interval):
    
    # Phase 1: Embedding-based clustering and archival (fast)
    clusters ← cluster_memories()
    summaries ← create_summary_nodes(clusters)
    archive_sources(clusters)
    build_embedding_edges(summaries)
    
    # Phase 2: Semantic extraction (async, if enabled)
    if gemma_available and extraction_enabled:
        for s in summaries:
            if should_extract(s):
                enqueue_extraction_job(s, gemma_3n_e2b)
    
    # Phase 3: Graph edge creation (completes asynchronously)
    # Extraction jobs populate graph_nodes and graph_edges tables

# Telemetry
emit_metric("consolidation.clusters_created", len(clusters))
emit_metric("consolidation.extractions_queued", len(extraction_jobs))
emit_metric("consolidation.duration_seconds", elapsed)
```

#### Algorithm 28b: Activity‑Aware Scheduling & Preemption

```
Input: last_retrieval_ts, tokens_in_flight, retrieval_queue_depth
Derived:
idle_required_seconds(T) = round(0.25 × w_rate_seconds(T))     # gate by stability

idle_for = now() − last_retrieval_ts
should_start = (tokens_in_flight == 0) and
               (retrieval_queue_depth == 0) and
               (idle_for ≥ idle_required_seconds(T))

if should_start:
    start_consolidation()
else:
    defer_until_idle()   # recheck periodically

# Preemption (hot‑path first)
on retrieval_event():
    pause_consolidation()
    commit_micro_batch()         # short txn only; avoid long locks
    release_db_resources()
    resume_when_idle()           # re‑evaluate should_start

# DB/IO contract:
# - Use WAL; prefer many short transactions
# - Yield between chunks; avoid table‑wide locks
# - Keep ANN index updates incremental
```

***

### 12.2 Memory Scoring for Consolidation

Each memory m is evaluated with:

* strength(m) — reinforcement‑based persistence
* redundancy(m) — similarity to nearby embeddings
* connectivity(m) — shared entities or relation overlap
* stability(m) — persistence over decay curve

### Algorithm 29: Consolidation Scoring

```
score_consolidate(m) = w_s×strength(m) − w_r×redundancy(m) + w_c×connectivity(m) + w_t×stability(m)

if score_consolidate(m) < consolidation_floor:
    mark_for_merge(m)
```

Knob‑derived weights (no magic numbers):

```
w_s = T
w_r = F
w_c = S
w_t = T
```

### 12.3.1 Semantic Extraction via Gemma

After clustering, extract interpretable entities and relations from consolidated text using Gemma 3n E2B Multimodal.

#### Algorithm 29c: Entity and Relation Extraction

```
Input: summary.text, cluster_i.text[]
Output: entities[], relations[]

1. context_window ← concatenate(cluster_i.text, max_tokens=2048)
2. prompt ← build_extraction_prompt(summary.text, context_window)
3. response ← gemma_3n_e2b.generate(prompt, max_tokens=512, temperature=0.3)
4. entities ← parse_entities(response)         # {name, type, salience}
5. relations ← parse_relations(response)       # {subject, predicate, object, confidence}
6. for e in entities:
       e.embedding_id ← embed_text(e.name) if is_frequent(e) else NULL
return entities, relations
```

Knob‑Derived Parameters:

```
extraction_batch_size = round(lerp(8, 32, T))                  # memories per batch
min_cluster_size_for_extraction = round(lerp(3, 10, F))        # skip small clusters at high F
entity_frequency_threshold = round(lerp(5, 15, T))             # vectorize if seen > threshold
```

Prompt Template:

```
Analyze this consolidated memory cluster and extract:
1. Named entities (people, places, organizations, concepts)
2. Relationships between entities (co-occurs, implies, contradicts)
3. Key themes or topics

Summary: {summary.text}
Context: {context_window}

Format as JSON:
{
  "entities": [{"name": "...", "type": "...", "salience": 0-1}],
  "relations": [{"subject": "...", "predicate": "...", "object": "...", "confidence": 0-1}]
}
```

Cost Optimization:

* Run extraction only when len(cluster\_i) ≥ min\_cluster\_size\_for\_extraction
* Batch multiple clusters per Gemma call (up to extraction\_batch\_size)
* Cache entity embeddings for high-frequency entities
* Skip extraction if T > 0.8 and cluster\_i already has entity metadata

***

### 12.3 Memory Merging and Clustering

Marked memories are grouped via density‑based clustering (DBSCAN) or k‑means using sqlite‑vec:

```
cluster_i = { m_j | cos(m_j, μ_i) > merge_threshold }
μ_i = centroid(cluster_i)
```

Create a summary memory and replace the cluster:

```
summary.embedding = μ_i
summary.text = summarize(cluster_i.text)
summary.metadata.sources = [m.id for m in cluster_i]
```

All source memories are cross‑referenced in the graph via derived\_from edges.

***

### 12.4 Knowledge Graph Construction

Node types:

* Entity Nodes: from named entities and topics extracted from text
* Memory Nodes: summarized embeddings from merged clusters
* Concept Nodes: emergent centroids representing recurrent topics

Edge types:

* co\_occurs\_with: shared context or time proximity
* implies / causes: directional correlation in embedding drift
* contradicts: strong negative cosine similarity or schema violation
* reinforces: frequent joint retrieval
* derived\_from: links summaries to source memories

```
**Node Storage Assignment:**
- Memory nodes → embeddings table (always vectorized)
- Summary nodes → embeddings table (centroids from Algorithm 29)
- Concept nodes → embeddings table (emergent topic centroids)
- Entity nodes → graph_nodes table (optional embedding_id reference)

This separation enables:
- Fast ANN queries over embedded content (memories, summaries, concepts)
- Clean graph traversal for structural queries (entity relationships)
- Optional entity vectorization when beneficial (e.g., frequently retrieved entities)
```

### Algorithm 30: Graph Construction (Hybrid Embeddings + Gemma)

```
Input: summary nodes, cluster sources, embeddings, gemma_3n_e2b
Output: populated graph_nodes, graph_edges

for each summary node s:
    # Semantic extraction via Gemma (Algorithm 29c)
    if len(s.sources) ≥ min_cluster_size_for_extraction:
        entities, relations ← extract_via_gemma(s, s.sources)
    else:
        entities ← []
        relations ← []
    
    # Create entity nodes and mention edges
    for e in entities:
        node_id ← ensure_node(e.name, type="Entity", label=e.name)
        if e.salience > 0.7 or is_frequent(e.name):
            node_id.embedding_id ← embed_text(e.name)
        create_edge(s, node_id, type="mentions", weight=e.salience)
    
    # Create relation edges from Gemma output
    for r in relations:
        subj ← ensure_node(r.subject, type="Entity")
        obj ← ensure_node(r.object, type="Entity")
        create_edge(subj, obj, type=r.predicate, weight=r.confidence)
    
    # Embedding-derived co-occurrence edges
    for (m_i, m_j) in s.sources:
        cos_sim ← cos(m_i.embedding, m_j.embedding)
        if cos_sim > lerp(0.85, 0.95, F):
            create_edge(m_i, m_j, type="co_occurs_with", weight=cos_sim)
    
    # Embedding-derived causal edges (drift detection)
    if len(s.sources) > 1:
        temporal_order ← sort_by_timestamp(s.sources)
        for i in range(len(temporal_order) - 1):
            m_i, m_j ← temporal_order[i], temporal_order[i+1]
            drift_vec ← normalize(m_j.embedding - m_i.embedding)
            drift_mag ← ||drift_vec||
            drift_threshold ← lerp(0.15, 0.35, T)
            if drift_mag > drift_threshold:
                create_edge(m_i, m_j, type="causes", weight=drift_mag)
    
    # Provenance edges to archived sources
    for m in s.sources:
        create_edge(s, m, type="derived_from", weight=m.original_strength)
```

Division of Labor:

Embeddings handle:

* Co-occurrence via cosine similarity
* Temporal causality via drift magnitude
* Concept clustering (Algorithm 29)

Gemma handles:

* Named entity recognition
* Semantic relation extraction (contradicts, reinforces, implies)
* Human-readable labels for clusters

***

### 12.5 Graph‑Augmented Retrieval

When processing a query or signal x\_t, vector lookup uses both direct similarity and graph expansion.

### Algorithm 31: Graph‑Augmented Retrieval

```
Input: x_t, kNN_size, graph_depth

1. results_vec ← topK(sqlite_vec.search(x_t, k=kNN_size))
   # Query embeddings table only (type IN ('memory', 'summary', 'concept'))

2. seed_nodes ← [r.id for r in results_vec]

3. expanded_nodes ← graph.traverse(seed_nodes, depth=graph_depth)
   # Recursive CTE over graph_edges table:
   # WITH RECURSIVE expand(id, depth) AS (
   #   SELECT source_id, 0 FROM graph_edges WHERE target_id IN seed_nodes
   #   UNION ALL
   #   SELECT e.source_id, ex.depth+1 
   #   FROM graph_edges e JOIN expand ex ON e.target_id = ex.id
   #   WHERE ex.depth < graph_depth
   # )

4. combined_ids ← union(seed_nodes ∪ expanded_nodes)

5. combined_context ← fetch_embeddings(combined_ids)
   # Join back to embeddings table; entities without embeddings are filtered

6. re_ranked ← sort_by(cos(x_t, combined_context))
return re_ranked
```

#### Schema: Graph‑Integrated Tables

##### Embeddings Table (Vector-Backed Nodes)

Extended embeddings table stores all nodes with vector representations (memory, summary, concept):

| Column         | Type             | Description                                 |
| -------------- | ---------------- | ------------------------------------------- |
| `id`           | TEXT PRIMARY KEY | Memory or node ID                           |
| `embedding`    | BLOB             | Vector representation                       |
| `type`         | TEXT             | (memory, summary, concept)                  |
| `strength`     | REAL             | Reinforcement score (used in consolidation) |
| `connectivity` | INTEGER          | Number of linked graph edges                |
| `drift_mag`    | REAL             | Temporal drift magnitude                    |
| `cluster_id`   | TEXT             | Consolidation cluster group ID              |
| `last_access`  | INTEGER          | Unix time of last retrieval                 |

Indices:

```sql
CREATE INDEX idx_embeddings_type ON embeddings(type);
CREATE INDEX idx_embeddings_connectivity ON embeddings(connectivity) 
  WHERE type IN ('summary', 'concept');
CREATE INDEX idx_embeddings_cluster ON embeddings(cluster_id) 
  WHERE cluster_id IS NOT NULL;
```

##### Graph Nodes Table (Entity References)

Stores entity nodes and other non-embedded graph elements:

| Column         | Type             | Description                       |
| -------------- | ---------------- | --------------------------------- |
| `id`           | TEXT PRIMARY KEY | Node identifier                   |
| `type`         | TEXT NOT NULL    | (entity, goal)                    |
| `label`        | TEXT             | Human-readable name               |
| `embedding_id` | TEXT             | FK to embeddings table (nullable) |
| `metadata`     | JSONB            | Additional attributes             |
| `created_at`   | INTEGER          | Unix time of creation             |

##### Graph Edges Table

Stores all relationships between nodes:

| Column                                              | Type    | Description                                |
| --------------------------------------------------- | ------- | ------------------------------------------ |
| `source_id`                                         | TEXT    | Origin node (any table)                    |
| `target_id`                                         | TEXT    | Destination node (any table)               |
| `edge_type`                                         | TEXT    | (co\_occurs\_with, implies, derived\_from) |
| `weight`                                            | REAL    | Relationship strength (default 1.0)        |
| `decay_rate`                                        | REAL    | Knob-derived via T                         |
| `last_reinforced`                                   | INTEGER | Unix time of last activation               |
| PRIMARY KEY (`source_id`, `target_id`, `edge_type`) |         |                                            |

Indices:

```sql
CREATE INDEX idx_edges_source ON graph_edges(source_id);
CREATE INDEX idx_edges_target ON graph_edges(target_id);
CREATE INDEX idx_edges_type ON graph_edges(edge_type);
```

**Storage Strategy:**

* **Memory & Summary nodes**: Always in embeddings table (require vectors for similarity)
* **Concept nodes**: In embeddings table (emergent centroids need vectors)
* **Entity nodes**: In graph\_nodes table (may reference embeddings via embedding\_id if vectorized)
* **All relationships**: In graph\_edges table

Optional:

* Add FTS5 virtual table for memory text (hybrid retrieval).
* Maintain edge\_cache table storing graph adjacency for faster expansions.

***

### 12.6 Adaptive Integration with cortext

| Knob            | Effect in Consolidation                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------- |
| Focus (F)       | Determines merge strictness (high F → conservative merges). Controls entity extraction threshold. |
| Sensitivity (S) | Influences how readily new graph nodes are created. Gates relation extraction confidence.         |
| Stability (T)   | Controls consolidation cadence, edge decay, and extraction batch size.                            |

### Algorithm 32: Adaptive Consolidation Rate

```
rate_consolidate_t = base_rate × (0.3 + 0.7T) × (1 − 0.5S)

# Gemma extraction throttling (background only, not on hot path)
extraction_enabled = (T > 0.2) and (len(cluster_i) ≥ min_cluster_size_for_extraction)
extraction_interval_seconds = lerp(300, 3600, T)  # 5 min → 1 hour

# Cost budget: max Gemma calls per consolidation cycle
max_extractions_per_cycle = round(lerp(20, 5, T))
```

Execution Model:

0. Wait for idle window (Algorithm 28b); preempt immediately on retrieval events
1. Cluster memories (fast, embeddings only)
2. Create summary nodes with centroid embeddings
3. Queue Gemma extraction jobs (async, rate-limited)
4. Build embedding-derived edges immediately (co-occurrence, drift)
5. Build Gemma-derived edges as extraction completes
6. Archive source memories after all edges created

At 30 tok/sec, extracting entities from a 512-token cluster summary takes \~17 seconds. With max\_extractions\_per\_cycle=20 at T=0, full consolidation completes in \~6 minutes (acceptable for background work).

***

### 12.7 Feedback into Scoring

Graph signals enrich scoring and context formation:

* Goal Alignment incorporates connected nodes of goal concepts.
* Coverage Gain expands over related nodes, improving recall diversity.
* Schema Violation edges log long‑term contradictions for reflection.

### Algorithm 33: Graph‑Enhanced Goal Alignment

```
Input: x_t, δ̂, graph

1. related_goals ← graph.expand(goal_nodes, depth=1)
2. goal_context ← mean(embeddings(related_goals))
3. goal_score_t ← cos(x_t, goal_context)
```

***

### 12.8 Lifecycle Summary

| Phase       | Operation               | Purpose                                         |
| ----------- | ----------------------- | ----------------------------------------------- |
| Stream      | Capture + Score + Store | Ingest short‑term experiences.                  |
| Consolidate | Merge + Graphify        | Compress experiences into structured knowledge. |
| Retrieve    | Recall + Expand         | Use episodic and semantic memory for reasoning. |

***

### 12.9 Integration Touchpoints with Algorithms 1–27

Core formulas for Algorithms 1–27 remain unchanged. Consolidation and the graph provide optional, knob‑consistent inputs and modifiers that enrich metrics, stabilize priors, and improve retrieval. Use these augmentations where available; fall back to the original signals otherwise.

Augmented signals (optional):

```
node_degree(v)            ∈ [0,∞)      # graph_edges out+in degree
clustering_coeff(v)       ∈ [0,1]      # local transitivity
contradiction_rate        ∈ [0,1]      # fraction of edges with type='contradicts' in active subgraph
active_subgraph_coherence ∈ [0,1]      # cohesion of expanded context neighborhood
precision_graph           ∈ [0,1]      # precision of graph-augmented retrieval (12.5)
```

Touchpoints:

* Algorithms 1–2 (Focus): tighten `attention_width_t` in dense subgraphs; boost `weight_relevance_t` near goal/summary hubs; use `active_subgraph_coherence` when available.
* Algorithms 3–4 (Sensitivity): raise novelty/surprise when new high‑confidence relations appear; gate `ΔThreshold_emotion_t` by relation confidence and entity salience.
* Algorithms 5–6 (Stability): lengthen `half_life_t` for summaries/concepts with high `node_degree`/`reinforces`; bound `rate_decay_t` using edge‑derived `decay_rate`.
* Algorithm 7 (Metric blending): include graph features in RLS inputs (degree, clustering, contradiction\_rate, connectivity).
* Algorithm 8 (Thresholding): compute precision from graph‑augmented retrieval; use it in `ΔT_prec`; improved stability from summaries reduces oscillations.
* Algorithms 10–11 (Coherence/Focus spread): compute over graph‑expanded context (entities+concepts) to refine `F_eff`.
* Algorithm 12 (Drift): measure drift via centroid motion of the active subgraph; trigger boundaries on structural topic shifts.
* Algorithm 13 (Surprise): incorporate spikes from new relations and `contradicts` edges.
* Algorithms 14, 18 (Strength/Influence): fold `connectivity` and provenance (`derived_from`) into `strength_t` updates; propagate attenuated gains to sources.
* Algorithms 15–17 (Feedback): apply contextual gains to neighbors via edge weights; incrementally increase attention precision and per‑memory stability.
* Algorithm 19 (Generation influence): consider similarity to concept nodes and drift along relation edges; use sustained influence to adjust `hysteresis_t`.
* Algorithm 21 (Competition): adapt `winners_k` to local cluster size; lateral inhibition within neighborhoods reduces interference.
* Algorithm 22 (Predictive): pre‑activate probable next nodes via bounded graph traversal (edge hops align to `prediction_horizon`).
* Algorithm 23 (Emotional tags): attach tags to summaries/entities; cascade along edges with decay.
* Algorithm 24 (Working memory): align chunking slots to concept nodes; complexity reflects entity/edge diversity.
* Algorithm 25 (Metacognition): increase FOK with consistent subgraph evidence; TOT when vectors are close but graph lacks support.
* Algorithm 26 (Serial position): distinctiveness includes graph centrality/outlier score.
* Algorithm 27 (Interrupt gate): compute `coverage_gain` over the active subgraph; duplicate suppression uses entity overlap.

***

### Algorithm 34: Mood-Congruent Retrieval Safety (Prosthetic Mode)

Purpose: Prevent feedback loops where negative mood biases retrieval, causing further negative mood.

1. Force Neutral Ranking:
   During retrieval, ensure `w_mood` is forced to 0.
   Score(m) = sim(q, m)  (purely semantic)

2. Focus Spike:
   If explicit query detected (e.g., "Who is this?"):
   F_temporary ← 1.0 (maximize precision and objectivity)

3. Context Display (Attribution Heuristic):
   If `||M_t|| > threshold` (High Mood) AND `e_t approx 0` (Neutral Event):
     Display: "Context: You met them while you were already feeling [Mood]."
   Else if `e_t > threshold`:
     Display: "Context: You reacted with [Emotion] to this specific event."
