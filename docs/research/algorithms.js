const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
    Header, Footer, AlignmentType, PageOrientation, LevelFormat,
    HeadingLevel, BorderStyle, WidthType, TabStopType,
    TabStopPosition, ShadingType, PageNumber, PageBreak } = require('docx');
const fs = require('fs');

// ==================== HELPER FUNCTIONS ====================

// Helper function to create a paragraph with text
const p = (text, options = {}) => new Paragraph({
    ...options,
    children: Array.isArray(text) ? text : [new TextRun(text)]
});

// Helper for bold text
const bold = (text) => new TextRun({ text, bold: true });

// Helper for italic text
const italic = (text) => new TextRun({ text, italics: true });

// Helper for inline code/math style
const code = (text) => new TextRun({ text, font: "Courier New", size: 20 });

// Helper for combined styles
const tr = (text, opts = {}) => new TextRun({ text, ...opts });

// ==================== STYLE DEFINITIONS ====================

// Table border style
const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

// Paragraph styles
const paragraphStyles = [
    {
        id: "Title", name: "Title", basedOn: "Normal",
        run: { size: 36, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 0, after: 240 }, alignment: AlignmentType.CENTER }
    },
    {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0 }
    },
    {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 280, after: 80 }, outlineLevel: 1 }
    },
    {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, italics: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 240, after: 60 }, outlineLevel: 2 }
    },
    {
        id: "Abstract", name: "Abstract", basedOn: "Normal",
        run: { size: 22, font: "Times New Roman" },
        paragraph: { spacing: { before: 0, after: 120 }, indent: { left: 720, right: 720 } }
    },
    {
        id: "Caption", name: "Caption", basedOn: "Normal",
        run: { size: 20, italics: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 60, after: 120 }, alignment: AlignmentType.CENTER }
    },
    {
        id: "Equation", name: "Equation", basedOn: "Normal",
        run: { size: 22, font: "Courier New" },
        paragraph: { spacing: { before: 120, after: 120 }, indent: { left: 720 } }
    }
];

// Numbering configuration
const numberingConfig = [
    {
        reference: "numbered-list",
        levels: [{
            level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
    },
    {
        reference: "bullet-list",
        levels: [{
            level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
    }
];

// ==================== SECTION CONTENT GENERATOR ====================

/**
 * Creates algorithm section content with configurable section numbering.
 * @param {number} offset - Section number offset (0 for algorithms.js, 2 for paper.js)
 * @returns {Object} Object containing arrays for each section
 */
const createAlgorithmSections = (offset = 0) => {
    const s = (n) => n + offset;  // Offset section numbers

    // Create metrics table (used in section 3/5)
    const metricsTable = new Table({
        columnWidths: [2500, 2000, 4860],
        rows: [
            new TableRow({
                tableHeader: true,
                children: [
                    new TableCell({
                        borders: cellBorders, shading: { fill: "E8E8E8", type: ShadingType.CLEAR },
                        children: [p([bold("Metric")], { alignment: AlignmentType.CENTER })]
                    }),
                    new TableCell({
                        borders: cellBorders, shading: { fill: "E8E8E8", type: ShadingType.CLEAR },
                        children: [p([bold("Knob")], { alignment: AlignmentType.CENTER })]
                    }),
                    new TableCell({
                        borders: cellBorders, shading: { fill: "E8E8E8", type: ShadingType.CLEAR },
                        children: [p([bold("Expression")], { alignment: AlignmentType.CENTER })]
                    })
                ]
            }),
            new TableRow({
                children: [
                    new TableCell({ borders: cellBorders, children: [p("Relevance")] }),
                    new TableCell({ borders: cellBorders, children: [p("↑F")] }),
                    new TableCell({ borders: cellBorders, children: [p([code("relevance_t = clamp(map01(cos(x_t, μ_ctx)) × (0.5 + 0.5F), 0, 1)")])] })
                ]
            }),
            new TableRow({
                children: [
                    new TableCell({ borders: cellBorders, children: [p("Mismatch")] }),
                    new TableCell({ borders: cellBorders, children: [p("↓F, ↑S")] }),
                    new TableCell({ borders: cellBorders, children: [p([code("(1 − F) × S × novelty_t")])] })
                ]
            }),
            new TableRow({
                children: [
                    new TableCell({ borders: cellBorders, children: [p("Surprise")] }),
                    new TableCell({ borders: cellBorders, children: [p("↑S, ↓T")] }),
                    new TableCell({ borders: cellBorders, children: [p([code("surprisal_t × S × (1 − T)")])] })
                ]
            }),
            new TableRow({
                children: [
                    new TableCell({ borders: cellBorders, children: [p("Rarity")] }),
                    new TableCell({ borders: cellBorders, children: [p("↑F, ↓T")] }),
                    new TableCell({ borders: cellBorders, children: [p([code("rarity_t × (0.5 + 0.5F) × (1 − 0.2T)")])] })
                ]
            }),
            new TableRow({
                children: [
                    new TableCell({ borders: cellBorders, children: [p("Drift")] }),
                    new TableCell({ borders: cellBorders, children: [p("↓T")] }),
                    new TableCell({ borders: cellBorders, children: [p([code("(drift_mag_t / 2) × (1 − T)")])] })
                ]
            }),
            new TableRow({
                children: [
                    new TableCell({ borders: cellBorders, children: [p("Utility")] }),
                    new TableCell({ borders: cellBorders, children: [p("↑F, ↓S")] }),
                    new TableCell({ borders: cellBorders, children: [p([code("ΔSSE × (0.5 + 0.5F) × (1 − 0.3S)")])] })
                ]
            }),
            new TableRow({
                children: [
                    new TableCell({ borders: cellBorders, children: [p("Salience")] }),
                    new TableCell({ borders: cellBorders, children: [p("F, S")] }),
                    new TableCell({ borders: cellBorders, children: [p([code("(rarity_t + novelty_t) / 2 × (F + S) / 2")])] })
                ]
            }),
            new TableRow({
                children: [
                    new TableCell({ borders: cellBorders, children: [p("Valence")] }),
                    new TableCell({ borders: cellBorders, children: [p("S, ↓T")] }),
                    new TableCell({ borders: cellBorders, children: [p([code("valence_t")])] })
                ]
            }),
            new TableRow({
                children: [
                    new TableCell({ borders: cellBorders, children: [p("Arousal")] }),
                    new TableCell({ borders: cellBorders, children: [p("S, ↓T")] }),
                    new TableCell({ borders: cellBorders, children: [p([code("arousal_t")])] })
                ]
            }),
            new TableRow({
                children: [
                    new TableCell({ borders: cellBorders, children: [p("Contradiction")] }),
                    new TableCell({ borders: cellBorders, children: [p("↑S, ↓F")] }),
                    new TableCell({ borders: cellBorders, children: [p([code("max(0, S − F)")])] })
                ]
            }),
            new TableRow({
                children: [
                    new TableCell({ borders: cellBorders, children: [p("Periphery")] }),
                    new TableCell({ borders: cellBorders, children: [p("↑T")] }),
                    new TableCell({ borders: cellBorders, children: [p([code("(1 − relevance_t) × T")])] })
                ]
            }),
            new TableRow({
                children: [
                    new TableCell({ borders: cellBorders, children: [p("Coverage")] }),
                    new TableCell({ borders: cellBorders, children: [p("↑F")] }),
                    new TableCell({ borders: cellBorders, children: [p([code("F × relevance_t")])] })
                ]
            })
        ]
    });

    return {
        // ==================== SECTION 1: MATHEMATICAL FOUNDATIONS ====================
        mathFoundations: [
            new Paragraph({ children: [new PageBreak()] }),
            p([tr(`${s(1)}. Mathematical Foundations`)], { heading: HeadingLevel.HEADING_1 }),

            p([tr(`${s(1)}.1 Notation and Primitives`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("We establish the following notation used throughout this paper. Let ε = 10⁻⁶ denote a small constant for numerical stability. All knob values F, S, T lie in the closed interval [0, 1].")]),

            p([tr("Core mathematical primitives:")]),

            p([code("lerp(a, b, x) = a + (b − a) × x")], { style: "Equation" }),
            p([code("clamp(v, lo, hi) = max(lo, min(v, hi))")], { style: "Equation" }),
            p([code("sigmoid(z) = 1 / (1 + exp(−z))")], { style: "Equation" }),
            p([code("EWMA(prev, x, α) = (1 − α) × prev + α × x")], { style: "Equation" }),

            p([tr("Weight normalization and blending for combining multiple signals:")]),

            p([code("normalize(w) = w / max(sum(w), ε) # w is a weight vector")], { style: "Equation" }),
            p([tr("Edge case: if sum(w) < ε, return uniform weights (1/|w|) to avoid division by zero or invalid probability distributions.")]),
            p([code("blend(values, weights) = Σᵢ values[i] × weights[i]")], { style: "Equation" }),

            p([tr("Note: normalize() operates on weight vectors, not scalars. When used with scalar expressions like lerp(), collect the scalars into a vector first: normalize([lerp(...), lerp(...), ...]). The blend() function assumes pre-normalized weights summing to 1.")]),

            p([tr("For vectors, we define cosine similarity as cos(u, v) = u·v / (‖u‖ × ‖v‖), and safe L2 normalization as l2_normalize(v) = v / max(‖v‖, ε). Shannon entropy is computed in nats: H(p) = −Σᵢ pᵢ ln(pᵢ).")]),

            p([tr("The temporal decay function follows exponential dynamics with configurable half-life:")]),

            p([code("decay(x, τ_half, Δt) = x × exp(−ln(2) × Δt / max(τ_half, τ_min))")], { style: "Equation" }),

            p([tr("where τ_min = 120 seconds provides a floor to prevent numerical instability from near-zero half-lives.")]),

            p([tr(`${s(1)}.1.1 Units Convention`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("To ensure consistency and avoid unit mismatch errors, the following conventions apply throughout:")]),

            p([bold("Input timestamps: "), tr("All input timestamps (t) are specified in milliseconds since epoch, consistent with standard system time APIs.")]),

            p([bold("Internal time representation: "), tr("All internal time calculations operate in seconds. We define:")]),

            p([code("now_s() = system_time_ms / 1000  # returns seconds")], { style: "Equation" }),
            p([code("now_ms() = system_time_ms         # returns milliseconds")], { style: "Equation" }),
            p([code("to_s(ts_ms) = ts_ms / 1000       # ms → seconds conversion")], { style: "Equation" }),

            p([bold("Stored timestamps: "), tr("All stored timestamps are milliseconds since Unix epoch.")]),

            p([bold("Naming contract (canonical): "), tr("Use the variable names below consistently throughout this document. Units: stored timestamps are integers in milliseconds (commonly suffixed *_ts, and also appearing as timestamp/created_at/last_rate_timestamp); derived time intervals in seconds use the *_s suffix. Accumulator variables: t_start, last_signal_ts, last_write_ts, drift_acc, eta_acc, coherence_prev, emo_max, arousal_sum, drift_accum, drift_at_last_interrupt, drift_acc_pacing, x_last_check. Global variables: u_uncertainty, mood_vector, last_mood_ts, theta_dynamic, theta_target, hysteresis, m_rate, dt_ema, rate_ticks, reliability, last_rate_timestamp, last_retrieval_ts. Weight naming rule: weight_* and *_weight variables (e.g., weight_relevance, mismatch_weight, weight_surprise) are control parameters; w_* variables (e.g., w_relevance, w_mismatch, … w_arousal) are composite-score blender weights.")]),

            p([bold("Time deltas: "), tr("All Δt values, elapsed times (mem_elapsed, signal_gap, idle_for), and time comparisons operate in seconds:")]),

            p([code("Δt ← now_s() − to_s(last_rate_timestamp)   # seconds")], { style: "Equation" }),
            p([code("mem_elapsed ← now_s() − to_s(t_start)     # seconds")], { style: "Equation" }),
            p([code("signal_gap ← now_s() − to_s(last_signal_ts)  # seconds")], { style: "Equation" }),

            p([bold("Time constants: "), tr("All time-related constants are specified with explicit units (e.g., τ_min = 120 seconds, kRecencyTau = 60 seconds). Threshold limits like max_mem_time(T) and gap_threshold(T) return values in seconds.")]),

            p([tr(`${s(1)}.1.2 Buffers`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("The specification uses distinct streaming buffers:")]),

            p([bold("signal_stream: "), tr("The raw per-signal embedding stream x_t used for scoring, uncertainty estimation, threshold/rate updates, and accumulator updates. Any context slices operate on signal_stream.")]),

            p([bold("score_stream: "), tr("The per-signal scalar score stream. At each step, append score_t to score_stream. Any score lookbacks operate on score_stream.")]),

            p([bold("memory_stream: "), tr("The stream of written memory representatives (e.g., e_rep/μ_acc for completed units) used for retrieval and interrupt-gate context. Sections that reference recent_memory_centroids operate on memory_stream.")]),

            p([bold("recent_memory_centroids: "), tr("A bounded deque of recent memory representatives used by the interrupt gate.")]),
            p([code("win_mem_ctx(T) = round(lerp(4, 32, T))  # max memories in interrupt context")], { style: "Equation" }),
            p([code("On successful write_memory: recent_memory_centroids.append(e_rep); recent_memory_centroids ← tail(recent_memory_centroids, win_mem_ctx(T))")], { style: "Equation" }),

            p([tr(`${s(1)}.1.3 Initialization (Normative)`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("To avoid cold-start artifacts in time-based dynamics, implementations MUST initialize timestamped and EWMA state as follows:")]),

            p([code("last_mood_ts ← now_ms()")], { style: "Equation" }),
            p([code("mood_vector ← 0_vector")], { style: "Equation" }),

            p([code("theta_target ← θ_prior(F, S, T)")], { style: "Equation" }),
            p([code("theta_dynamic ← theta_target")], { style: "Equation" }),
            p([code("hysteresis ← base_band(T)")], { style: "Equation" }),
            p([code("reliability ← 0")], { style: "Equation" }),
            p([code("last_retrieval_ts ← now_ms()")], { style: "Equation" }),

            p([code("last_rate_timestamp ← now_ms()")], { style: "Equation" }),
            p([code("dt_ema ← 1.0")], { style: "Equation" }),
            p([code("rate_ticks ← 0")], { style: "Equation" }),
            p([code("m_rate ← rate_target")], { style: "Equation" }),
            p([code("ρ_hat_prev ← rate_target")], { style: "Equation" }),

            p([tr("Per stream initialization: prev_x is unset; x_last_check is unset.")]),

            p([tr(`${s(1)}.2 The Three-Knob Philosophy`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Cortext is governed by three continuous control parameters, each representing a distinct dimension of cognitive regulation:")]),

            p([bold("Focus (F ∈ [0, 1]): "), tr("Perceptual selectivity and precision. Higher Focus narrows attention, increases relevance weighting, and reduces retrieval breadth. Focus modulates the trade-off between exploitation of known-relevant information and exploration of potentially useful context.")]),

            p([bold("Sensitivity (S ∈ [0, 1]): "), tr("Plasticity and affective gain. Higher Sensitivity accelerates learning, increases emotional and novelty responsiveness, and raises write-rate targets. Sensitivity governs how readily the system captures novel information and responds to salient stimuli.")]),

            p([bold("Stability (T ∈ [0, 1]): "), tr("Temporal persistence and inertia. Higher Stability lengthens memory half-lives, widens hysteresis bands, slows adaptive updates, and tightens safety bounds over time. Stability controls the resistance to change and the preservation of established knowledge.")]),

            p([tr("A central design principle is that knobs set rates rather than modes. Behavioral differences emerge continuously from parameter interactions; there are no hard-coded phase transitions or discrete operational states.")]),

            p([tr(`${s(1)}.3 Knob-Derived Parameters`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("All system tunables derive from the three primary knobs. This section catalogs the key derivations.")]),

            p([tr(`${s(1)}.3.1 Context Windows and Temporal Scales`)], { heading: HeadingLevel.HEADING_3 }),

            p([code("n_ctx(T) = round(lerp(32, 256, T))")], { style: "Equation" }),
            p([code("win_score(T) = round(lerp(20, 120, T))")], { style: "Equation" }),
            p([code("win_rate_s(T) = round(lerp(60, 300, T))")], { style: "Equation" }),

            p([tr("The context window n_ctx determines how many recent items inform relevance computation. The scoring window win_score controls the lookback for variance estimation and percentile calculation. The rate window win_rate_s specifies the temporal horizon (in seconds) for write-rate measurement.")]),

            p([tr(`${s(1)}.3.2 Half-Life and Decay`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Memory half-life follows a log-scale mapping to span multiple orders of magnitude:")]),

            p([code("τ_min = 120.0 seconds (2 minutes)")], { style: "Equation" }),
            p([code("τ_max = 43200.0 seconds (12 hours)")], { style: "Equation" }),
            p([code("base_half_life(T) = exp(ln(τ_min) + T × ln(τ_max / τ_min))")], { style: "Equation" }),

            p([tr("This exponential mapping ensures that low Stability yields half-lives near 2 minutes while high Stability approaches 12 hours, with smooth interpolation across the range.")]),

            p([tr(`${s(1)}.3.3 Hysteresis and Rate Targets`)], { heading: HeadingLevel.HEADING_3 }),

            p([code("band_min = 0.02; band_max = 0.25")], { style: "Equation" }),
            p([code("base_band(T) = lerp(band_min, band_max, T)")], { style: "Equation" }),
            p([code("r_min = 0.2; r_max = 5.0  (writes per minute)")], { style: "Equation" }),
            p([code("base_rate(S) = lerp(r_min, r_max, S)")], { style: "Equation" }),

            p([tr("The hysteresis band prevents oscillation in threshold-crossing decisions. Write-rate targets establish homeostatic setpoints for the threshold controller.")]),

            p([tr(`${s(1)}.3.4 Experiential Mass and Maturity`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("The system tracks accumulated experience through a maturity function that governs the annealing of safety bounds:")]),

            p([code("τ_m(T) = lerp(10.0, 200.0, T)")], { style: "Equation" }),
            p([code("maturity(t) = 1 − exp(−count / τ_m(T))")], { style: "Equation" }),

            p([tr("where count is the total number of signals processed. This produces asymptotic approach to unit maturity, with higher Stability slowing the progression to reflect greater conservatism.")]),

            p([tr("Safety bounds on the dynamic threshold anneal with maturity:")]),

            p([code("T_min(t) = lerp(0.01, 0.05, maturity(t))")], { style: "Equation" }),
            p([code("T_max(t) = lerp(0.99, 0.95, maturity(t))")], { style: "Equation" }),
            p([code("max_ΔT_per_min(t) = lerp(0.30, 0.10, maturity(t))")], { style: "Equation" }),

            p([tr("Early operation permits wide threshold excursions; mature operation constrains movement to a narrower band.")]),

            p([tr(`${s(1)}.4 Uncertainty Estimation`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Uncertainty u(t) ∈ [0, 1] modulates learning rates and evidence weighting. The raw uncertainty estimate blends multiple signals:")]),

            p([code("var_score_max = 0.25")], { style: "Equation" }),
            p([code("recent_scores ← tail(score_stream, win_score(T))")], { style: "Equation" }),
            p([code("if |recent_scores| < 2:")], { style: "Equation" }),
            p([code("    var_recent_norm ← 0")], { style: "Equation" }),
            p([code("else:")], { style: "Equation" }),
            p([code("    var_recent_norm ← clamp(var(recent_scores) / var_score_max, 0, 1)")], { style: "Equation" }),
            p([code("coherence_complement = 1 − coherence_struct_t  # uses structural coherence")], { style: "Equation" }),

            p([tr("When prediction error signals are available, novelty and surprisal are blended:")]),

            p([code("novelty_surprise = blend([novelty_t, surprisal_t],")], { style: "Equation" }),
            p([code("                        weights = normalize([S, 1 − T]))")], { style: "Equation" }),

            p([tr("The final raw uncertainty combines these components with knob-derived weights:")]),

            p([tr("Normative note (MUST): focus_spread_t is the per-step derived focus-spread metric from Section 3.1.2 and is available each step when computing u(t).")]),

            p([code("weights_u = normalize([S, F, 1 − T, S × (1 − T)])")], { style: "Equation" }),
            p([code("u_raw(t) = clamp(blend([var_recent_norm, focus_spread_t,")], { style: "Equation" }),
            p([code("                        coherence_complement, novelty_surprise],")], { style: "Equation" }),
            p([code("                       weights = weights_u), 0, 1)")], { style: "Equation" }),

            p([tr("Smoothed uncertainty applies EWMA with a stability-dependent rate:")]),

            p([code("α_u(T) = 0.10 + (1 − T) × 0.60")], { style: "Equation" }),
            p([code("u(t) = EWMA(u(t−1), u_raw(t), α = α_u(T))")], { style: "Equation" }),

            p([tr("When structural metrics are unavailable, the fallback is u_raw(t) = 1 − maturity(t), ensuring high uncertainty during early operation.")]),
        ],

        // ==================== SECTION 2: CORE ADAPTATION ALGORITHMS ====================
        coreAdaptation: [
            new Paragraph({ children: [new PageBreak()] }),
            p([tr(`${s(2)}. Core Adaptation Algorithms`)], { heading: HeadingLevel.HEADING_1 }),

            p([tr("This section presents the algorithms governing adaptation along each of the three primary dimensions. Each algorithm consists of a prior computation (executed at initialization) and a dynamic update (executed per signal).")]),

            p([tr(`${s(2)}.1 Focus-Driven Selectivity`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Focus governs perceptual selectivity through relevance weighting and attention width.")]),

            p([tr(`${s(2)}.1.1 Focus Priors`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Given Focus knob F ∈ [0, 1], initialize Focus control variables:")]),

            p([code("weight_relevance = sigmoid(2F − 1)")], { style: "Equation" }),
            p([code("coverage_gain_floor = 0.3 + 0.7F")], { style: "Equation" }),
            p([code("mismatch_weight = 1 − F")], { style: "Equation" }),
            p([code("attention_width = lerp(π, 0.1π, F)")], { style: "Equation" }),

            p([tr("The attention width (in radians) controls the angular spread of the receptive field in embedding space. High Focus produces narrow attention (0.1π), while low Focus permits broad capture (π).")]),

            p([tr(`${s(2)}.1.2 Dynamic Focus Update`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("At each signal event t with input embedding x_t:")]),

            p([code("recent_context ← tail(signal_stream, n_ctx(T))")], { style: "Equation" }),
            p([code("if |recent_context| == 0:")], { style: "Equation" }),
            p([code("    μ_ctx ← 0_vector; observed_cosine ← 0  # map01(0)=0.5")], { style: "Equation" }),
            p([code("else:")], { style: "Equation" }),
            p([code("    μ_ctx ← mean(recent_context)")], { style: "Equation" }),
            p([code("    observed_cosine ← cos(x_t, μ_ctx)")], { style: "Equation" }),
            p([code("weight_relevance ← EWMA(weight_relevance,")], { style: "Equation" }),
            p([code("                      map01(observed_cosine), α = α_F(t))")], { style: "Equation" }),

            p([tr("where map01(z) = clamp((z + 1) / 2, 0, 1) transforms cosine values from [−1, 1] to [0, 1].")]),

            p([tr("The learning rate α_F(t) is modulated by uncertainty:")]),

            p([code("α_min_F = 0.05; α_span_F = 0.45")], { style: "Equation" }),
            p([code("α_F(t) = α_min_F + F × α_span_F × u(t)")], { style: "Equation" }),

            p([tr("High uncertainty increases learning rate, allowing faster adaptation when the environment is volatile. The Focus knob scales the uncertainty responsiveness.")]),

            p([tr(`${s(2)}.2 Sensitivity-Driven Plasticity`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Sensitivity governs learning speed, emotional responsiveness, and novelty capture.")]),

            p([tr(`${s(2)}.2.1 Sensitivity Priors`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Given Sensitivity knob S ∈ [0, 1], initialize Sensitivity control variables:")]),

            p([code("rate_target = base_rate(S)  # writes/min")], { style: "Equation" }),
            p([code("weight_novelty = 0.3 + 0.7S")], { style: "Equation" }),
            p([code("weight_surprise = 0.2 + 0.8S")], { style: "Equation" }),
            p([code("weight_valence = 0.4 + 0.6S")], { style: "Equation" }),
            p([code("weight_arousal = S")], { style: "Equation" }),
            p([code("emotion_gain = exp(1.5S)")], { style: "Equation" }),
            p([code("score_gain = exp(2S)")], { style: "Equation" }),

            p([tr(`${s(2)}.2.2 Emotional Projection`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("When emotion category centroids are available, the system projects input embeddings onto a discrete emotion space C = {anger, fear, joy, love, sadness, surprise}. Inspired by Russell's (1980) circumplex model, each category maps to valence and arousal coordinates:")]),

            p([code("v_map = {anger: −0.9, fear: −0.8, sadness: −0.9,")], { style: "Equation" }),
            p([code("         joy: +0.9, love: +0.8, surprise: 0.0}")], { style: "Equation" }),
            p([code("a_map = {anger: +0.9, fear: +0.9, sadness: +0.3,")], { style: "Equation" }),
            p([code("         joy: +0.6, love: +0.5, surprise: +0.8}")], { style: "Equation" }),

            p([tr("The projection procedure:")]),

            p([code("raw_cos_c ← cos(x_t, centroids[c]) for each c ∈ C")], { style: "Equation" }),
            p([code("if all raw_cos_c ≤ 0:")], { style: "Equation" }),
            p([code("    emotion_intensity_t ← 0; valence_t ← 0.5; arousal_t ← 0")], { style: "Equation" }),
            p([code("else:")], { style: "Equation" }),
            p([code("    logits_c ← max(0, raw_cos_c)")], { style: "Equation" }),
            p([code("    β(S) = 4 + 8S  # softmax inverse temperature")], { style: "Equation" }),
            p([code("    p_c ← softmax(β(S) × logits_c)")], { style: "Equation" }),
            p([code("    peak ← max_c(p_c)")], { style: "Equation" }),
            p([code("    confidence ← 1 − H(p_c) / ln(6)")], { style: "Equation" }),
            p([code("    emotion_intensity_t ← sqrt(peak × confidence)")], { style: "Equation" }),
            p([code("    valence_t ← (Σ_c p_c × v_map[c] + 0.9) / 1.8")], { style: "Equation" }),
            p([code("    arousal_t ← clamp(Σ_c p_c × a_map[c], 0, 1)")], { style: "Equation" }),

            p([tr("The emotion intensity combines peak probability with distributional confidence via geometric mean, providing a measure that is high only when a single emotion dominates with high certainty.")]),

            p([tr(`${s(2)}.2.3 Threshold Modulation from Emotion`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Emotional activation loosens write thresholds to capture salient moments:")]),

            p([code("κ_emo ← κ_base × S  # where κ_base = 0.10")], { style: "Equation" }),
            p([code("ΔThreshold_emotion_t ← −κ_emo × emotion_intensity_t ×")], { style: "Equation" }),
            p([code("                        (0.5 + 0.5 × arousal_t)")], { style: "Equation" }),

            p([tr("This negative adjustment makes writing more likely during emotionally salient events, consistent with McGaugh's (2004) findings on arousal-enhanced encoding.")]),

            p([tr(`${s(2)}.2.4 Mood Integration`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr(`Distinct from instantaneous emotion, the mood state M_t ∈ ℝ⁶ maintains a persistent background affective tone as a 6-dimensional vector (one component per emotion category from Section ${s(2)}.2.2):`)]),

            p([code("α_mood(S) = lerp(0.01, 0.20, S)  # reactivity")], { style: "Equation" }),
            p([code("half_life_mood(T) = lerp(30, 600, T)  # seconds")], { style: "Equation" }),
            p([code("Δt_mood ← now_s() − to_s(last_mood_ts)")], { style: "Equation" }),
            p([code("λ_mood(Δt_mood, T) ← exp(−ln(2) × Δt_mood / max(half_life_mood(T), ε))")], { style: "Equation" }),
            p([code("e_t ← p_c − (1/6)  # centered 6D vector (can be negative)")], { style: "Equation" }),
            p([code("M_t = λ_mood(Δt_mood, T) × M_{t−1} + α_mood(S) × e_t")], { style: "Equation" }),
            p([code("M_t ← clamp_elementwise(M_t, −1.0, 1.0)  # per-component")], { style: "Equation" }),
            p([code("last_mood_ts ← now_ms()  # update timestamp after mood update (ms)")], { style: "Equation" }),

            p([tr("Because e_t is centered around zero, M_t can have both positive and negative components, reflecting sustained elevation or suppression relative to baseline. The mood state provides a separate threshold bias via its normalized magnitude:")]),

            p([code("κ_mood ← κ_base × S")], { style: "Equation" }),
            p([code("m_norm ← ‖M_t‖ / √6  # max norm when all components at 1")], { style: "Equation" }),
            p([code("ΔThreshold_mood_t ← −κ_mood × clamp(m_norm, 0, 1)")], { style: "Equation" }),

            p([tr(`${s(2)}.3 Stability-Driven Persistence`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Stability governs temporal dynamics through half-life, decay rates, and hysteresis.")]),

            p([tr(`${s(2)}.3.1 Stability Priors`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Given Stability knob T ∈ [0, 1], initialize Stability control variables:")]),

            p([code("hysteresis = lerp(0.02, 0.25, T)")], { style: "Equation" }),
            p([code("half_life = base_half_life(T)")], { style: "Equation" }),
            p([code("rate_decay = lerp(0.60, 0.98, T)")], { style: "Equation" }),
            p([code("periphery_half_life = clamp(0.5 × half_life,")], { style: "Equation" }),
            p([code("                                   τ_min, τ_max)")], { style: "Equation" }),
            p([code("drift_weight = 0.5 × (1 − T)")], { style: "Equation" }),

            p([tr(`${s(2)}.3.2 Dynamic Stability Update`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("The stability update uses uncertainty-modulated learning rate and a stability-derived retention window:")]),

            p([code("α_min_T = 0.02; α_span_T = 0.18")], { style: "Equation" }),
            p([code("α_T(t) = α_min_T + (1 − T) × α_span_T × u(t)")], { style: "Equation" }),
            p([code("win_ret(T) = round(lerp(10, 50, T))  # retention history window size")], { style: "Equation" }),

            p([tr("At each signal event, compute retention statistics and adjust half-life:")]),

            p([code("active_memories ← {m | strength(m) ≥ periphery_cutoff(T)}")], { style: "Equation" }),
            p([code("observed_retention ← mean_age(active_memories)")], { style: "Equation" }),
            p([code("retention_ema_t ← EWMA(retention_ema_{t−1},")], { style: "Equation" }),
            p([code("                       observed_retention, α = α_T(t))")], { style: "Equation" }),

            p([tr("Compute z-score relative to recent retention history:")]),

            p([code("last_win_ret ← tail(retention_history, win_ret(T))")], { style: "Equation" }),
            p([code("μ_ret ← mean(last_win_ret)")], { style: "Equation" }),
            p([code("σ_ret ← max(std(last_win_ret), 1.0)")], { style: "Equation" }),
            p([code("zscore_ret ← clamp((observed_retention − μ_ret) / σ_ret, −3, +3)")], { style: "Equation" }),

            p([tr(`The target half-life incorporates feedback adjustment from the stability feedback mechanism (Section ${s(7)}.3):`)]),

            p([code("stability_adj ← ΔHalfLife_adj_t if provided else 0")], { style: "Equation" }),
            p([code("target_half_life_t ← clamp(base_half_life(T) ×")], { style: "Equation" }),
            p([code("                          (1 + 0.25 × zscore_ret + stability_adj),")], { style: "Equation" }),
            p([code("                          τ_min, τ_max)")], { style: "Equation" }),
            p([code("half_life_t ← EWMA(half_life_{t−1}, target_half_life_t,")], { style: "Equation" }),
            p([code("                   α = α_T(t))")], { style: "Equation" }),
        ],

        // ==================== SECTION 3: STRUCTURAL METRICS ====================
        structuralMetrics: [
            new Paragraph({ children: [new PageBreak()] }),
            p([tr(`${s(3)}. Structural Metrics and Composite Scoring`)], { heading: HeadingLevel.HEADING_1 }),

            p([tr("Context definitions (used throughout Table 1 and structural metrics):")]),
            p([code("recent_context ← tail(signal_stream, n_ctx(T))")], { style: "Equation" }),
            p([code("if |recent_context| == 0:")], { style: "Equation" }),
            p([code("    μ_ctx ← 0_vector  # define cos(x_t, μ_ctx)=0 so map01(cos)=0.5")], { style: "Equation" }),
            p([code("else:")], { style: "Equation" }),
            p([code("    μ_ctx ← mean(recent_context)")], { style: "Equation" }),

            p([tr(`${s(3)}.1 Embedding-Derived Metrics`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr(`${s(3)}.1.1 Structural Coherence`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr(`Structural coherence (coherence_struct) measures integration of the current signal with the broader context window. This metric is distinct from memory coherence (coherence_mem, defined in Section ${s(4)}.4.2) which tracks within-memory similarity:`)]),

            p([code("if |recent_context| < 2:")], { style: "Equation" }),
            p([code("    coherence_struct_t ← 0.5  # neutral")], { style: "Equation" }),
            p([code("else:")], { style: "Equation" }),
            p([code("    raw ← var([cos(x_t, c) for c in recent_context])")], { style: "Equation" }),
            p([code("    coherence_struct_t ← 1 − clamp(raw, 0, 1)  # range [0, 1]")], { style: "Equation" }),

            p([tr("High structural coherence (low variance in similarities) indicates the signal fits consistently with context. The effective Focus is modulated: F_eff = F × (0.5 + 0.5 × coherence_struct_t).")]),

            p([tr(`${s(3)}.1.2 Focus Spread`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Focus spread quantifies the entropy of attention over nearest neighbors:")]),

            p([code("k ← k_neighbors(T) = round(lerp(8, 32, T))")], { style: "Equation" }),
            p([tr("Normative note (MUST): kNN_similarities MUST be computed by querying memory_stream with q = x_t and k = k_neighbors(T) (not recent_context).")]),
            p([code("p ← softmax(kNN_similarities)")], { style: "Equation" }),
            p([code("focus_spread_t ← H(p) / ln(k)")], { style: "Equation" }),

            p([tr("Values near 1 indicate diffuse attention; values near 0 indicate concentrated attention. The effective Focus is further modulated: F_eff ← F_eff × (1 − focus_spread_t).")]),

            p([tr(`${s(3)}.1.3 Trajectory Drift`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Drift measures directional change in context centroids:")]),

            p([code("k_ctx(T) = round(lerp(3, 10, T))  # step lag for drift")], { style: "Equation" }),
            p([code("ctx_t = signal_stream[t − n_ctx(T) + 1 : t]  # inclusive end")], { style: "Equation" }),
            p([code("ctx_{t−k} = signal_stream[t − k_ctx(T) − n_ctx(T) + 1 : t − k_ctx(T)]")], { style: "Equation" }),
            p([code("drift_vec_t ← l2_normalize(mean(ctx_t)) −")], { style: "Equation" }),
            p([code("              l2_normalize(mean(ctx_{t−k}))")], { style: "Equation" }),
            p([code("drift_mag_t ← ‖drift_vec_t‖")], { style: "Equation" }),

            p([tr("Since both centroids are unit-normalized, drift_mag_t ∈ [0, 2]. A threshold defines an informational drift-boundary signal:")]),

            p([code("drift_threshold ← lerp(0.10, 0.35, T)")], { style: "Equation" }),
            p([code("drift_boundary_t ← (drift_mag_t > drift_threshold)")], { style: "Equation" }),
            p([tr(`Normative note (MUST): drift_boundary_t is informational and MUST NOT trigger a memory flush on its own. Memory flush decisions are defined by should_flush in Section ${s(4)}.4.3.`)]),

            p([tr(`${s(3)}.1.4 Embedding Prediction Error`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("We measure surprisal as the deviation of the current embedding from the predicted trajectory in latent space:")]),

            p([code("if prev_x is unset:")], { style: "Equation" }),
            p([code("    surprisal_t ← 0")], { style: "Equation" }),
            p([code("    Δx_trend_t ← 0_vector")], { style: "Equation" }),
            p([code("else:")], { style: "Equation" }),
            p([code("    Δx_t = x_t − prev_x")], { style: "Equation" }),
            p([code("    Δx_trend_t = EWMA(Δx_trend_{t−1}, Δx_t, α=0.1)")], { style: "Equation" }),
            p([code("    x_pred_t = prev_x + Δx_trend_{t−1}")], { style: "Equation" }),
            p([code("    prediction_error_t = 1 − cos(x_pred_t, x_t)")], { style: "Equation" }),

            p([tr("This error is normalized to produce the surprisal signal:")]),

            p([code("err_max = 0.5")], { style: "Equation" }),
            p([code("    surprisal_t ← clamp(prediction_error_t / err_max, 0, 1)")], { style: "Equation" }),

            p([tr("This formulation captures purely kinematic surprise in the thought process")]),

            p([tr(`${s(3)}.2 Composite Score Computation`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The system computes 12 metrics that blend into a composite write score:")]),

            metricsTable,

            p([tr("Table 1: Metric definitions and knob dependencies. Arrows indicate direction of influence.")], { style: "Caption" }),

            p([tr(`${s(3)}.3 Metric Weight Blending`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Metric weights adapt online using recursive least squares (RLS) to minimize prediction error between composite scores and observed outcomes. Initial weights derive from bootstrap coefficients:")]),

            p([tr("Blender weights are maintained as variables w_* (do not confuse these with control weights like weight_relevance or mismatch_weight). We denote the 12-element blender weight vector as:")]),
            p([code("W_blend = [w_relevance, w_mismatch, w_surprise, w_rarity, w_drift,")], { style: "Equation" }),
            p([code("          w_utility, w_salience, w_valence, w_arousal, w_contradiction,")], { style: "Equation" }),
            p([code("          w_periphery, w_coverage]")], { style: "Equation" }),
            p([tr("The index i used below (e.g., w_bootstrap[i], w_rls[i]) follows this ordering.")]),

            p([code("w_bootstrap[i] ← sigmoid(c_F[i]×F + c_S[i]×S + c_T[i]×T + d_i)")], { style: "Equation" }),

            p([tr("RLS fitting updates coefficients with stability-dependent forgetting:")]),

            p([code("φ(T) = 0.90 + 0.09T")], { style: "Equation" }),
            p([code("N ← round(lerp(64, 512, T))  # fitting window")], { style: "Equation" }),

            p([tr("The fitted weights blend with bootstrap weights based on RLS confidence:")]),

            p([code("τ_rls ← lerp(20.0, 80.0, T)")], { style: "Equation" }),
            p([code("confidence_rls ← 1 − exp(−t / τ_rls)")], { style: "Equation" }),
            p([code("w_rls01[i] ← clamp(w_rls[i], 0, 1)  # constrain fitted weights to mixture range")], { style: "Equation" }),
            p([code("weight_i(t) ← clamp((1 − confidence_rls) × w_bootstrap[i] +")], { style: "Equation" }),
            p([code("                      confidence_rls × w_rls01[i], 0, 1)")], { style: "Equation" }),

            p([tr(`${s(3)}.3.1 Score Normalization`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Composite score computation requires careful normalization:")]),

            p([code("for each metric i:")], { style: "Equation" }),
            p([code("    m01[i] = clamp(metric[i], 0, 1)")], { style: "Equation" }),
            p([code("weight_sum ← Σ weights[i]")], { style: "Equation" }),
            p([code("if weight_sum < ε: return 0")], { style: "Equation" }),
            p([code("weights_norm[i] ← weights[i] / weight_sum")], { style: "Equation" }),
            p([code("score ← clamp(Σ weights_norm[i] × m01[i], 0, 1)")], { style: "Equation" }),

            p([tr("Invariant (MUST): all 12 Table 1 metric values are defined on [0, 1].")]),

            p([tr("Weight normalization is critical: with 12 metrics and raw weights averaging ~0.6, the sum approaches 7.2. Without normalization, weighted sums would saturate and collapse variance.")]),
        ],

        // ==================== SECTION 4: DYNAMIC THRESHOLDING ====================
        dynamicThresholding: [
            new Paragraph({ children: [new PageBreak()] }),
            p([tr(`${s(4)}. Dynamic Thresholding and Homeostatic Control`)], { heading: HeadingLevel.HEADING_1 }),

            p([tr("The write gate compares composite scores against an adaptive threshold θ_dynamic (written as theta_dynamic when referring to the recorded variable). This section details the threshold evolution algorithm incorporating Bayesian prior-evidence blending and homeostatic rate control.")]),

            p([tr(`${s(4)}.1 Prior-Evidence Blending`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The threshold prior derives from knob settings:")]),

            p([code("θ_prior(F, S, T) = lerp(0.10, 0.30, T) × (1 − 0.3S)")], { style: "Equation" }),

            p([tr("Observed evidence comes from the 90th percentile of recent scores:")]),

            p([code("recent_scores ← tail(score_stream, win_score(T))")], { style: "Equation" }),
            p([code("if |recent_scores| == 0:")], { style: "Equation" }),
            p([code("    observed_p90 ← θ_prior  # no evidence yet")], { style: "Equation" }),
            p([code("else:")], { style: "Equation" }),
            p([code("    observed_p90 ← percentile(recent_scores, 90)")], { style: "Equation" }),

            p([tr("Prior and evidence masses weight the blend:")]),

            p([code("ρ_prior ← prior_mass(T) = round(lerp(2, 32, T))")], { style: "Equation" }),
            p([code("ρ_obs ← u(t) × |recent_scores|")], { style: "Equation" }),

            p([tr("The target threshold blends prior and evidence:")]),

            p([code("θ_target ← (ρ_prior × θ_prior + ρ_obs × observed_p90) /")], { style: "Equation" }),
            p([code("            max(ε, ρ_prior + ρ_obs)")], { style: "Equation" }),

            p([tr("High Stability increases prior mass, making the system more resistant to observed deviations. High uncertainty increases evidence mass, allowing faster adaptation to volatile conditions.")]),

            p([tr(`${s(4)}.2 Homeostatic Rate Control`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The controller maintains write rates near the target setpoint through continuous-time estimation with effective sample size (ESS) reliability weighting.")]),

            p([tr(`${s(4)}.2.1 Rate Estimation`)], { heading: HeadingLevel.HEADING_3 }),

            p([code("Δt ← now_s() − to_s(last_rate_timestamp)")], { style: "Equation" }),
            p([code("Δt ← max(Δt, 10⁻³)  # minimum 1e-3 s (1 ms)")], { style: "Equation" }),
            p([code("α_dt ← 1 − exp(−Δt / 1.0)")], { style: "Equation" }),
            p([code("dt_ema ← (1 − α_dt) × dt_ema + α_dt × Δt")], { style: "Equation" }),
            p([code("dt_base ← max(dt_ema, 1.0)")], { style: "Equation" }),

            p([tr("The rate time constant scales with Stability:")]),

            p([code("τ_rate ← max(2^(3T) × dt_base, 1.0)")], { style: "Equation" }),
            p([code("α ← 1 − exp(−Δt / τ_rate)")], { style: "Equation" }),

            p([tr("Instantaneous rate estimation with bias correction. Δwrites is the binary indicator (0 or 1) of whether a write occurred during the current timestep:")]),

            p([code("Δwrites ← 1 if write_memory else 0  # binary write event")], { style: "Equation" }),
            p([tr("Normative note (MUST): Δwrites is computed from the current step's write_memory decision. Rate state updates occur after the write decision and affect subsequent timesteps.")]),
            p([code("ρ_inst ← (Δwrites / Δt) × 60  # writes per minute")], { style: "Equation" }),
            p([code("m_rate ← (1 − α) × m_rate + α × ρ_inst")], { style: "Equation" }),
            p([code("denom ← max(1 − (1 − α)^(rate_ticks + 1), ε)")], { style: "Equation" }),
            p([code("ρ_hat_next ← m_rate / denom  # bias-corrected estimate for next step")], { style: "Equation" }),
            p([code("rate_ticks ← rate_ticks + 1")], { style: "Equation" }),
            p([code("last_rate_timestamp ← now_ms()  # update after rate computation")], { style: "Equation" }),

            p([tr(`${s(4)}.2.2 Effective Sample Size`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("ESS estimates the effective number of independent samples in the EMA, using a heuristic inspired by Liu & Chen (1998):")]),

            p([code("β ← max(0, 1 − α)")], { style: "Equation" }),
            p([code("ESS ← min((1 + β) / max(1 − β, ε), 100)")], { style: "Equation" }),
            p([code("reliability ← 1 − exp(−ESS × (1 − T))")], { style: "Equation" }),

            p([tr("High Stability dampens reliability, preventing aggressive corrections in conservative regimes.")]),

            p([tr(`${s(4)}.2.3 Homeostatic Correction`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("The rate error drives threshold adjustment:")]),

            p([tr("Normative note (MUST): ρ_hat_prev is stored state entering the timestep. After the write decision, the rate-state update computes ρ_hat_next; the assignment ρ_hat_prev ← ρ_hat_next occurs at end of step (see Appendix C).")]),

            p([code("rate_error ← tanh((ρ_hat_prev − rate_target) /")], { style: "Equation" }),
            p([code("                  max(rate_target, ε))")], { style: "Equation" }),
            p([code("κ_r = 0.10  # rate error gain")], { style: "Equation" }),
            p([code("cap_homeo ← 0.25 × hysteresis")], { style: "Equation" }),
            p([code("Δθ_homeo ← clamp(reliability × κ_r × (1 − T) ×")], { style: "Equation" }),
            p([code("                  (1 − maturity(t)) × rate_error,")], { style: "Equation" }),
            p([code("                  −cap_homeo, +cap_homeo)")], { style: "Equation" }),

            p([tr("The correction scales with reliability and is attenuated by both Stability and maturity, ensuring conservative, mature systems make minimal homeostatic adjustments.")]),

            p([tr(`${s(4)}.2.4 Sensitivity-Based Threshold Adjustment`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Sensitivity modulates threshold based on recent score volatility:")]),

            p([code("recent_scores ← tail(score_stream, win_score(T))")], { style: "Equation" }),
            p([code("if |recent_scores| < 2:")], { style: "Equation" }),
            p([code("    σ_scores ← 0")], { style: "Equation" }),
            p([code("else:")], { style: "Equation" }),
            p([code("    σ_scores ← std(recent_scores)")], { style: "Equation" }),
            p([code("κ_sens = 0.08  # sensitivity gain")], { style: "Equation" }),
            p([code("cap_sens ← 0.20 × hysteresis")], { style: "Equation" }),
            p([code("Δθ_sens ← clamp(−κ_sens × S × (σ_scores − 0.1),")], { style: "Equation" }),
            p([code("                −cap_sens, +cap_sens)")], { style: "Equation" }),

            p([tr("High score variance with high Sensitivity lowers threshold, capturing more volatile signals.")]),

            p([tr(`${s(4)}.2.5 Precision-Based Threshold Adjustment`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Focus-driven precision tightens threshold when structural coherence is high:")]),

            p([code("κ_prec = 0.06  # precision gain")], { style: "Equation" }),
            p([code("cap_prec ← 0.15 × hysteresis")], { style: "Equation" }),
            p([code("Δθ_prec ← clamp(κ_prec × F × (coherence_struct_t − 0.5),")], { style: "Equation" }),
            p([code("                −cap_prec, +cap_prec)")], { style: "Equation" }),

            p([tr("High structural coherence with high Focus raises threshold, enforcing stricter relevance filtering.")]),

            p([tr(`${s(4)}.3 Threshold Integration`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr(`All threshold deltas combine and pass through safety limiting. The emotion and mood deltas from Section ${s(2)}.2.3-${s(2)}.2.4 are denoted Δθ_emo = ΔThreshold_emotion_t and Δθ_mood = ΔThreshold_mood_t:`)]),

            p([code("Δθ_total ← Δθ_sens + Δθ_homeo + Δθ_prec + Δθ_emo + Δθ_mood")], { style: "Equation" }),
            p([code("cap_total ← max_ΔT_per_min(t) × (Δt / 60.0)")], { style: "Equation" }),
            p([code("Δθ_limited ← clamp(Δθ_total, −cap_total, +cap_total)")], { style: "Equation" }),
            p([code("θ_dynamic_t ← clamp(EWMA(θ_dynamic_{t−1}, θ_target,")], { style: "Equation" }),
            p([code("                        α = α_T(t)) + Δθ_limited,")], { style: "Equation" }),
            p([code("                    T_min(t), T_max(t))")], { style: "Equation" }),

            p([tr("Hysteresis evolves toward the stability-derived base:")]),

            p([code("hysteresis ← clamp(EWMA(hysteresis,")], { style: "Equation" }),
            p([code("                     base_band(T), α = α_T(t)),")], { style: "Equation" }),
            p([code("                band_min, band_max)")], { style: "Equation" }),
        ],

        // ==================== SECTION 4.4: WRITE PACING AND MEMORY ACCUMULATION ====================
        writePacing: [
            p([tr(`${s(4)}.4 Write Pacing and Memory Accumulation`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The write gate operates per-signal, but coherent \"thoughts\" often span multiple signals. This section introduces memory-level accumulation that groups signals into natural units before storage decisions, inspired by Event Segmentation Theory (Zacks & Swallow, 2007).")]),

            p([tr("This approach draws from EM-LLM (Fountas et al., 2024), which segments token sequences into episodic events using surprise-based boundary detection refined by graph-theoretic cohesion metrics. Their work shows that combining prediction error signals with within-segment coherence produces boundaries strongly correlated with human event perception. Our adaptation uses embedding drift as a proxy for surprise and cosine similarity for cohesion, enabling modality-agnostic operation across text tokens, audio chunks, video frames, or any signal stream.")]),

            p([tr(`${s(4)}.4.1 Memory Accumulator State`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Each source stream maintains accumulator state:")]),

            p([code("μ_acc ← 256d running mean embedding")], { style: "Equation" }),
            p([code("drift_acc ← accumulated drift within memory")], { style: "Equation" }),
            p([code("s_sum ← sum of signal scores in memory")], { style: "Equation" }),
            p([code("s_max ← max signal score in memory")], { style: "Equation" }),
            p([code("n ← count of signals in memory")], { style: "Equation" }),
            p([code("e_peak ← embedding of highest-scoring signal")], { style: "Equation" }),
            p([code("emo_max ← 0  # max emotion_intensity_t within the unit")], { style: "Equation" }),
            p([code("arousal_sum ← 0  # sum of arousal_t within the unit (for avg)")], { style: "Equation" }),
            p([code("t_start ← now_ms()  # timestamp of accumulation start (ms since epoch)")], { style: "Equation" }),
            p([code("last_signal_ts ← now_ms()  # timestamp of previous signal (ms, for gap detection)")], { style: "Equation" }),
            p([code("last_write_ts ← 0  # ms; 0 means \"no prior write\" for refractory")], { style: "Equation" }),
            p([code("eta_acc ← 0  # drift EWMA state")], { style: "Equation" }),
            p([code("coherence_prev ← 0  # previous coherence (initialize to 0)")], { style: "Equation" }),
            p([code("acc_signals_window ← []  # ring buffer of recent embeddings for coherence")], { style: "Equation" }),

            p([tr("Reset behavior: reset_accumulator() clears μ_acc, drift_acc, s_sum, s_max, n, e_peak, emo_max, arousal_sum, eta_acc, coherence_prev, and sets acc_signals_window ← [] (and refreshes t_start/last_signal_ts for the next unit), but retains last_write_ts so the refractory term remains well-defined across boundaries.")]),
            p([code("reset_accumulator(): acc_signals_window ← []  # MUST clear coherence window at boundaries")], { style: "Equation" }),

            p([tr("On each signal, update running statistics (note: n is the count before this signal):")]),

            p([code("signal_gap_s ← now_s() − to_s(last_signal_ts)  # compute BEFORE updating last_signal_ts")], { style: "Equation" }),

            p([code("win_coh(T) = round(lerp(8, 32, T))  # coherence window size")], { style: "Equation" }),
            p([code("acc_signals_window ← tail(acc_signals_window, win_coh(T))")], { style: "Equation" }),

            p([code("μ_acc ← (n × μ_acc + x_t) / (n + 1)")], { style: "Equation" }),
            p([code("n ← n + 1")], { style: "Equation" }),
            p([code("drift_acc ← drift_acc + (drift_mag_t / 2)  # accumulate normalized context-centroid drift (drift_mag_t ∈ [0,2])")], { style: "Equation" }),
            p([code("s_sum ← s_sum + score_t")], { style: "Equation" }),
            p([code("emo_max ← max(emo_max, emotion_intensity_t)")], { style: "Equation" }),
            p([code("arousal_sum ← arousal_sum + arousal_t")], { style: "Equation" }),
            p([code("if score_t > s_max:")], { style: "Equation" }),
            p([code("    s_max ← score_t; e_peak ← x_t")], { style: "Equation" }),
            p([code("last_signal_ts ← now_ms()  # update timestamp for next gap calculation (ms)")], { style: "Equation" }),

            p([tr(`${s(4)}.4.2 Hybrid Drift and Coherence Tracking`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Boundary detection combines kinematic drift with semantic coherence:")]),

            p([code("ε_noise = 0.02  # noise floor for drift")], { style: "Equation" }),
            p([code("d_step ← max(drift_mag_t − ε_noise, 0)")], { style: "Equation" }),
            p([code("eta_prev ← eta_acc  # baseline before updating EWMA")], { style: "Equation" }),

            p([tr("Memory coherence tracks similarity within the current memory accumulation window (range [−1, 1] from mean cosine):")]),

            p([code("current_window ← acc_signals_window  # embeddings from the current unit before x_t")], { style: "Equation" }),
            p([code("if |current_window| == 0:")], { style: "Equation" }),
            p([code("    coherence_curr ← 1.0  # empty-window fallback")], { style: "Equation" }),
            p([code("else:")], { style: "Equation" }),
            p([code("    coherence_curr ← mean([cos(x_t, x_i) for x_i in current_window])")], { style: "Equation" }),
            p([code("# After computing coherence_curr, append x_t for the next step")], { style: "Equation" }),
            p([code("acc_signals_window.append(x_t); acc_signals_window ← tail(acc_signals_window, win_coh(T))")], { style: "Equation" }),
            p([code("# coherence_prev is the stored coherence value from the previous step")], { style: "Equation" }),

            p([tr(`Note: coherence_mem is distinct from coherence_struct (Section ${s(3)}.1.1). The former tracks within-memory similarity using raw mean cosine, while the latter measures variance-based integration with broader context. This dual-signal approach mirrors EM-LLM's boundary detection mechanism. In their formulation, boundaries occur where surprise (token-level prediction error) exceeds a threshold and segment cohesion drops. Our drift spike approximates surprise via embedding-space velocity, while coherence_mem drop captures within-memory similarity degradation.`)]),

            p([tr(`${s(4)}.4.3 Natural Boundary Detection`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Boundary score combines drift spike and coherence drop:")]),

            p([code("weight_drift_component = lerp(0.6, 0.4, T)  # weight on drift")], { style: "Equation" }),
            p([code("weight_coh_component = 1 − weight_drift_component  # weight on coherence drop")], { style: "Equation" }),
            p([code("ε0 = 0.01  # cold-start guard for eta_prev")], { style: "Equation" }),
            p([code("if eta_prev < ε0:")], { style: "Equation" }),
            p([code("    drift_spike ← 0")], { style: "Equation" }),
            p([code("else:")], { style: "Equation" }),
            p([code("    drift_spike ← (d_step − eta_prev) / max(eta_prev, ε)")], { style: "Equation" }),
            p([code("eta_acc ← EWMA(eta_prev, d_step, α = lerp(0.3, 0.1, T))")], { style: "Equation" }),
            p([code("coh_drop01 ← clamp((coherence_prev − coherence_curr) / 2, 0, 1)")], { style: "Equation" }),
            p([code("coherence_prev ← coherence_curr  # update for next step")], { style: "Equation" }),
            p([code("boundary_score ← weight_drift_component × sigmoid(drift_spike) +")], { style: "Equation" }),
            p([code("                  weight_coh_component × coh_drop01")], { style: "Equation" }),
            p([code("boundary_score ← clamp(boundary_score, 0, 1)")], { style: "Equation" }),

            p([tr("Boundary threshold and limits:")]),

            p([code("b_thresh(F, S) = lerp(0.4, 0.7, F) × lerp(1.1, 0.9, S)")], { style: "Equation" }),
            p([code("max_mem_time(T) = lerp(30, 120, T)  # seconds")], { style: "Equation" }),
            p([code("max_mem_drift(S) = lerp(0.8, 2.0, S)  # cumulative drift cap")], { style: "Equation" }),

            p([tr("Trigger memory flush when:")]),

            p([code("mem_elapsed ← now_s() − to_s(t_start)")], { style: "Equation" }),
            p([code("should_flush = (boundary_score > b_thresh(F, S)) OR")], { style: "Equation" }),
            p([code("               (mem_elapsed > max_mem_time(T)) OR")], { style: "Equation" }),
            p([code("               (drift_acc > max_mem_drift(S)) OR")], { style: "Equation" }),
            p([code("               (signal_gap_s > gap_threshold(T))")], { style: "Equation" }),

            p([tr("where signal_gap_s = now_s() − to_s(last_signal_ts) is computed at the start of signal processing (before last_signal_ts is updated), detecting natural pauses (speech pauses, generation delays):")]),

            p([code("gap_threshold(T) = lerp(5, 30, T)  # seconds")], { style: "Equation" }),

            p([tr(`${s(4)}.4.4 Spike Bypass (Flashbulb Flush)`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("High-salience signals bypass accumulation and flush immediately, capturing preceding context as a coherent memory unit:")]),

            p([code("spike_margin(S) = lerp(0.3, 0.15, S)  # above θ_dynamic")], { style: "Equation" }),
            p([code("spike_bypass = score_t > (θ_dynamic + spike_margin(S))")], { style: "Equation" }),
            p([code("force_write ← false")], { style: "Equation" }),

            p([tr("When spike_bypass triggers:")]),

            p([code("if spike_bypass:")], { style: "Equation" }),
            p([code("    should_flush = true   # force a boundary")], { style: "Equation" }),
            p([code("    force_write = true   # bypass S_window > θ_memory")], { style: "Equation" }),

            p([tr("This ensures flashbulb moments capture their surrounding context rather than creating isolated micro-memories.")]),

            p([tr(`${s(4)}.4.5 Window Score and Refractory`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Memory-level score combines peak and average with coverage bonus:")]),

            p([code("s_avg ← s_sum / max(n, 1)")], { style: "Equation" }),
            p([code("α(F) = lerp(0.3, 0.7, F)  # peak vs avg weight")], { style: "Equation" }),
            p([code("coverage ← min(n / n_ctx(T), 1.0)  # memory completeness")], { style: "Equation" }),
            p([code("β(S) = lerp(0.05, 0.15, S)  # coverage weight")], { style: "Equation" }),
            p([code("S_window ← α(F) × s_max + (1 − α(F)) × s_avg + β(S) × coverage")], { style: "Equation" }),

            p([tr("Write refractory suppresses rapid successive writes:")]),

            p([code("τ_write_refrac(T) = lerp(5, 30, T)  # seconds")], { style: "Equation" }),
            p([code("k_write_refrac = lerp(0.3, 0.1, T)")], { style: "Equation" }),
            p([code("Δt_write ← now_s() − to_s(last_write_ts)")], { style: "Equation" }),
            p([code("M_write_refrac ← 1.0 + k_write_refrac × exp(−Δt_write / τ_write_refrac(T))")], { style: "Equation" }),

            p([tr("Final write decision:")]),

            p([code("θ_memory ← θ_dynamic × M_write_refrac")], { style: "Equation" }),
            p([code("write_memory = force_write OR (should_flush AND (S_window > θ_memory))")], { style: "Equation" }),
            p([code("if write_memory: last_write_ts ← now_ms()  # update refractory timestamp (ms)")], { style: "Equation" }),
            p([tr("Normative rule (MUST): if should_flush is true, the current unit must be finalized. If write_memory is false, discard the unit and reset_accumulator() anyway (do not update last_write_ts). This prevents perpetual should_flush states (time cap / drift cap / gap cap) while never resetting.")]),

            p([tr("Trace note: if a run trace reports both write_decision and stored, interpret write_decision as the boolean gate outcome at the boundary (the write_memory predicate above). Interpret stored as the eventual recording outcome (after any final safety checks).")]),

            p([tr("Representative embedding blends accumulator mean with peak:")]),

            p([code("ρ(F) = lerp(0.3, 0.7, F)  # mean vs peak blend")], { style: "Equation" }),
            p([code("e_rep ← l2_normalize(ρ(F) × μ_acc + (1 − ρ(F)) × e_peak)")], { style: "Equation" }),

            p([tr("On write: store e_rep with metadata {n, s_max, s_avg, drift_acc, mem_elapsed, s_emotion_max=emo_max, s_arousal_avg=arousal_sum / max(n, 1)}. Append e_rep to memory_stream and recent_memory_centroids. Reset accumulator for next unit.")]),
        ],

        // ==================== SECTION 5: REINFORCEMENT AND DECAY ====================
        reinforcementDecay: [
            new Paragraph({ children: [new PageBreak()] }),
            p([tr(`${s(5)}. Reinforcement and Decay Dynamics`)], { heading: HeadingLevel.HEADING_1 }),

            p([tr(`${s(5)}.1 Memory Strength Model`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Each memory m maintains a strength value updated through use-frequency tracking and exponential decay. The update uses a sensitivity-modulated learning rate and a binary usage indicator:")]),

            p([code("α_min_S = 0.05; α_span_S = 0.35")], { style: "Equation" }),
            p([code("α_S(t) = α_min_S + S × α_span_S × u(t)")], { style: "Equation" }),
            p([code("used_flag(m) = 1 if m was retrieved and used in current step, else 0")], { style: "Equation" }),

            p([code("use_frequency_t ← EWMA(use_frequency_{t−1},")], { style: "Equation" }),
            p([code("                       used_flag(m), α = α_S(t))")], { style: "Equation" }),
            p([code("λ_t ← ln(2) / half_life_t")], { style: "Equation" }),
            p([code("strength_t ← clamp(strength_{t−1} × exp(−λ_t × Δt) +")], { style: "Equation" }),
            p([code("                   S × use_frequency_t, 0, 1)")], { style: "Equation" }),

            p([tr("Memories falling below the periphery cutoff are candidates for eviction:")]),

            p([code("periphery_cutoff(T) = lerp(0.05, 0.25, T)")], { style: "Equation" }),
            p([code("if strength_t < periphery_cutoff(T): evict(m)")], { style: "Equation" }),

            p([tr(`${s(5)}.2 Influence-Weighted Updates`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("When contextual gain signals are available, influence factors modulate reinforcement:")]),

            p([code("influence_factor ← (used_count / max(retrieved_count, 1)) ×")], { style: "Equation" }),
            p([code("                    clamp(contextual_gain(m), −1, +1)")], { style: "Equation" }),
            p([code("strength_t ← clamp(strength_{t−1} × exp(−λ_t × Δt) +")], { style: "Equation" }),
            p([code("                   S × use_frequency_t + F × influence_factor, 0, 1)")], { style: "Equation" }),

            p([tr(`${s(5)}.3 Causal Feedback Loop`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The system tracks causal influence of retrieved memories on generation quality through contextual gain—the improvement in prediction accuracy attributable to including each memory in context.")]),

            p([tr(`${s(5)}.3.1 Focus Feedback`)], { heading: HeadingLevel.HEADING_3 }),

            p([code("αF_base = 0.10; βF_base = 0.05")], { style: "Equation" }),
            p([code("for each used memory m:")], { style: "Equation" }),
            p([code("    if contextual_gain(m) > 0:")], { style: "Equation" }),
            p([code("        weight_relevance += αF_base × contextual_gain(m)")], { style: "Equation" }),
            p([code("        attention_width_t *= (1 − βF_base)")], { style: "Equation" }),
            p([code("    else:")], { style: "Equation" }),
            p([code("        attention_width_t *= (1 + βF_base)")], { style: "Equation" }),
            p([code("weight_relevance ← clamp(weight_relevance, 0, 1)")], { style: "Equation" }),
            p([code("attention_width_t ← clamp(attention_width_t,")], { style: "Equation" }),
            p([code("                          attention_width_min, attention_width_max)")], { style: "Equation" }),

            p([tr("Positive contextual gain narrows attention and boosts relevance weighting; negative gain widens attention to explore alternatives.")]),

            p([tr(`${s(5)}.3.2 Sensitivity Feedback`)], { heading: HeadingLevel.HEADING_3 }),

            p([code("η_base = 0.10")], { style: "Equation" }),
            p([code("for each used memory m:")], { style: "Equation" }),
            p([code("    novelty_reward ← 1 − sim(m.embedding, recent_context)")], { style: "Equation" }),
            p([code("    weight_novelty_t += η_base × (novelty_reward ×")], { style: "Equation" }),
            p([code("                         contextual_gain(m) −")], { style: "Equation" }),
            p([code("                         redundancy(m, recent_context))")], { style: "Equation" }),
            p([code("weight_novelty_t ← clamp(weight_novelty_t, 0, 1)")], { style: "Equation" }),

            p([tr("This rewards novelty that proves useful while penalizing redundant retrievals.")]),

            p([tr(`${s(5)}.3.3 Stability Feedback`)], { heading: HeadingLevel.HEADING_3 }),

            p([code("γT_base = 0.05")], { style: "Equation" }),
            p([code("for each used memory m:")], { style: "Equation" }),
            p([code("    if contextual_gain(m) > 0:")], { style: "Equation" }),
            p([code("        stability(m) += γT_base")], { style: "Equation" }),
            p([code("    else:")], { style: "Equation" }),
            p([code("        stability(m) *= (1 − γT_base)")], { style: "Equation" }),

            p([tr("The mean stability of used memories provides adjustment to the half-life target:")]),

            p([code("adj ← clamp(mean(stability(m_used)) − 1.0, −0.25, +0.25)")], { style: "Equation" }),
            p([code("ΔHalfLife_adj_t ← adj")], { style: "Equation" }),

            p([tr(`This factor is consumed by the Stability update (Section ${s(2)}.3.2), avoiding conflicting adjustments between feedback mechanisms.`)]),

            p([tr(`${s(5)}.4 Generation Influence Tracking`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("When generation embeddings are available, influence incorporates output trajectory:")]),

            p([code("Δḡ ← l2_normalize(ḡ_t) − l2_normalize(ḡ_{t−1})")], { style: "Equation" }),
            p([code("drift_mag_gen ← ‖Δḡ‖")], { style: "Equation" }),
            p([code("drift_contribution(m) ← (drift_mag_gen / 2) ×")], { style: "Equation" }),
            p([code("                         max(0, cos(m.embedding, l2_normalize(Δḡ)))")], { style: "Equation" }),

            p([tr("Total influence blends contextual gain, generation similarity, and drift contribution:")]),

            p([code("λ₁ = 0.5; λ₂ = 0.4; λ₃ = 0.3")], { style: "Equation" }),
            p([code("influence(m) ← λ₁ × contextual_gain(m) +")], { style: "Equation" }),
            p([code("                λ₂ × cos(m.embedding, ḡ_t) −")], { style: "Equation" }),
            p([code("                λ₃ × drift_contribution(m)")], { style: "Equation" }),

            p([tr("Sustained influence accumulates over a stability-dependent horizon:")]),

            p([code("L_sustain(T) = round(lerp(3, 5, T))")], { style: "Equation" }),
            p([code("sustained_influence ← EWMA(sustained_influence,")], { style: "Equation" }),
            p([code("                           influence(m),")], { style: "Equation" }),
            p([code("                           α = 2 / (L_sustain(T) + 1))")], { style: "Equation" }),
        ],

        // ==================== SECTION 6: ADVANCED COGNITIVE PROCESSES ====================
        advancedCognitive: [
            new Paragraph({ children: [new PageBreak()] }),
            p([tr(`${s(6)}. Advanced Cognitive Processes`)], { heading: HeadingLevel.HEADING_1 }),

            p([tr("This section presents algorithms modeling higher-order cognitive phenomena: working memory maintenance, metacognitive monitoring, reconsolidation dynamics, and serial position effects.")]),

            p([tr(`${s(6)}.1 Working Memory Gates`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr(`Following Cowan's (2001) capacity constraints, working memory maintains a limited number of active items. Working memory holds coherent memories as defined in Section ${s(4)}.4, preserving the full content and signal sequence:`)]),

            p([code("base_capacity = round(lerp(5, 3, S) + lerp(−1, 1, F))")], { style: "Equation" }),

            p([tr("This yields a range of approximately 2-6 memories, broadening the 4±1 chunk limit to accommodate task-dependent requirements. High Sensitivity reduces capacity (faster turnover), while high Focus modulates breadth.")]),

            p([tr(`${s(6)}.1.1 Active Memory Structure`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Each active memory holds a coherent memory with its full content and metadata:")]),

            p([code("memory.content ← concatenated signal blobs")], { style: "Equation" }),
            p([code("memory.source_id ← source identifier (e.g., 'user', 'assistant')")], { style: "Equation" }),
            p([code("memory.modality ← primary modality ('text', 'audio', 'image')")], { style: "Equation" }),
            p([code("memory.blob_ids ← [blob_1, blob_2, ..., blob_n]  # blob refs")], { style: "Equation" }),
            p([code(`memory.embedding ← e_rep  # representative embedding (Section ${s(4)}.4.5)`)]),
            p([code("memory.signals ← [x_1, x_2, ..., x_n]  # ordered signal embeddings")], { style: "Equation" }),
            p([code("memory.metadata ← {n, s_max, s_avg, drift_acc, mem_elapsed,")], { style: "Equation" }),
            p([code("                   s_emotion_max, s_arousal_avg}")], { style: "Equation" }),

            p([tr(`The emotional metrics (s_emotion_max, s_arousal_avg) are accumulated during memory formation for use by Emotional Consolidation (Section ${s(6)}.7).`)]),

            p([tr(`${s(6)}.1.2 Maintenance Cost`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Maintenance incurs cognitive cost:")]),

            p([code("maintenance_cost_per_memory = lerp(0.05, 0.15, S)")], { style: "Equation" }),
            p([code("complexity_penalty = manifold_complexity × lerp(0.5, 1.5, S)")], { style: "Equation" }),

            p([tr("The manifold_complexity is defined as a normalized local variability proxy (see Appendix B): manifold_complexity ← clamp((1 − mean_cos_window) / 2, 0, 1).")]),

            p([tr(`${s(6)}.1.3 Memory-Level Gating`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Gating thresholds determine entry and chunking:")]),

            p([code("chunking_threshold = lerp(0.7, 0.9, F)")], { style: "Equation" }),
            p([code("gate_threshold = lerp(0.1, 0.4, F)")], { style: "Equation" }),
            p([code("rehearsal_rate = lerp(0.5, 2.0, S)")], { style: "Equation" }),
            p([code("memory_dedication_strength = lerp(0.3, 0.9, T)")], { style: "Equation" }),

            p([tr(`Working memory gating evaluates coherent memories at accumulation boundaries (Section ${s(4)}.4.3), not individual signals:`)]),

            p([code("on_memory_boundary:")], { style: "Equation" }),
            p([code("    [α, β, γ] ← normalize([lerp(0.55, 0.70, F),   # window score weight")], { style: "Equation" }),
            p([code("                        lerp(0.20, 0.35, F),   # task relevance weight")], { style: "Equation" }),
            p([code("                        lerp(0.10, 0.30, S)])   # novelty-to-WM weight")], { style: "Equation" }),
            p([code("    memory_benefit ← α × S_window + β × relevance_to_task(μ_acc, task_context) +")], { style: "Equation" }),
            p([code("                       γ × novelty_to_set(μ_acc, {m.embedding | m ∈ active_memories})")], { style: "Equation" }),
            p([code("    margin ← memory_benefit − gate_threshold")], { style: "Equation" }),
            p([code("    k ← |active_memories|")], { style: "Equation" }),
            p([code("    C ← max(base_capacity, 1)")], { style: "Equation" }),
            p([code("    p_cap ← 3")], { style: "Equation" }),
            p([code("    capacity_pressure(k, C) ← 1 + max(0, (k − C) / C)^p_cap")], { style: "Equation" }),
            p([code("    base_cost ← maintenance_cost_per_memory × k + complexity_penalty")], { style: "Equation" }),
            p([code("    total_cost ← base_cost × capacity_pressure(k, C)")], { style: "Equation" }),
            p([code("    accept_memory = (margin ≥ total_cost)")], { style: "Equation" }),

            p([tr("Note that total_cost is computed from existing active memories only (k is the current count), so k=0 yields no bootstrap penalty. Capacity pressure activates only when k > C.")]),

            p([tr(`${s(6)}.1.4 Chunking at Memory Level`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Chunking operates on memory embeddings to merge related content from the same source:")]),

            p([code("similar_memories ← {m ∈ active_memories | cos(m.embedding, memory.e_rep) > chunking_threshold AND")], { style: "Equation" }),
            p([code("                              m.source_id == memory.source_id}")], { style: "Equation" }),
            p([code("if |similar_memories| > 0:")], { style: "Equation" }),
            p([code("    merge_into_chunk(similar_memories, memory)")], { style: "Equation" }),

            p([tr(`${s(6)}.2 Metacognitive Monitoring`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The system implements feeling-of-knowing (FOK) and tip-of-tongue (TOT) detection following Hart's (1965) framework:")]),

            p([code("FOK_threshold = lerp(0.2, 0.5, F)")], { style: "Equation" }),
            p([code("TOT_detection = (FOK > lerp(0.5, 0.8, F)) AND")], { style: "Equation" }),
            p([code("                 (retrieval_strength < lerp(0.4, 0.2, F))")], { style: "Equation" }),

            p([tr("TOT occurs when metacognitive confidence is high but retrieval strength is low—the characteristic experience of knowing one knows something but being unable to access it.")]),

            p([tr("Additional metacognitive parameters:")]),

            p([code("confidence_decay_rate = lerp(0.01, 0.1, 1 − T)")], { style: "Equation" }),
            p([code("unknown_threshold = lerp(0.3, 0.1, F)")], { style: "Equation" }),
            p([code("strategy_switch_latency = lerp(500, 100, S)  # ms")], { style: "Equation" }),
            p([code("certainty_requirement = lerp(0.6, 0.9, T)")], { style: "Equation" }),
            p([code("metacognitive_sensitivity = F × (1 + 0.5 × S)")], { style: "Equation" }),

            p([tr(`${s(6)}.3 Memory Reconsolidation`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Following Nader et al. (2000), retrieved memories enter a labile state permitting modification:")]),

            p([code("τ_labile = lerp(30, 300, T)  # seconds")], { style: "Equation" }),
            p([code("reconsolidation_gain = lerp(0.2, 0.02, T)")], { style: "Equation" }),
            p([code("lability_susceptibility = (1 − T) × (0.5 + 0.5 × S)")], { style: "Equation" }),

            p([tr("During the lability window, memories can drift toward current context:")]),

            p([code("drift_magnitude = (1 − T) × S × lability ×")], { style: "Equation" }),
            p([code("                   contextual_relevance")], { style: "Equation" }),

            p([tr("Reconsolidation effects propagate to semantically related memories with decay:")]),

            p([code("ripple_decay = lerp(0.5, 0.1, T)  # per semantic hop")], { style: "Equation" }),

            p([tr(`${s(6)}.4 Retrieval Competition`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Retrieved memories compete through lateral inhibition, modeling retrieval-induced forgetting (Anderson et al., 1994):")]),

            p([code("inhibition_radius = lerp(0.5, 0.85, F)")], { style: "Equation" }),
            p([code("winners_k = round(lerp(7, 3, F))")], { style: "Equation" }),
            p([code("suppression_per_retrieval = lerp(0.1, 0.01, T) ×")], { style: "Equation" }),
            p([code("                             (1 − winning_activation)")], { style: "Equation" }),
            p([code("recovery_time_RIF = lerp(300, 1800, T)  # seconds")], { style: "Equation" }),

            p([tr("High Focus produces narrow winner-take-all dynamics; low Focus permits broader activation.")]),

            p([tr(`${s(6)}.5 Predictive Pre-activation`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The system pre-activates memories predicted to be relevant based on trajectory extrapolation:")]),

            p([code("prediction_horizon = round(lerp(2, 8, F))")], { style: "Equation" }),
            p([code("pre_activation_decay = lerp(0.7, 0.3, T)")], { style: "Equation" }),
            p([code("prediction_conf_threshold = lerp(0.3, 0.7, F)")], { style: "Equation" }),
            p([code("surprise_sensitivity = S × lerp(2.0, 0.5, T)")], { style: "Equation" }),

            p([tr("When predictions fail (high surprise), the system updates its trajectory model:")]),

            p([code("update_rate_on_surprise = lerp(0.2, 0.02, T) × S")], { style: "Equation" }),

            p([tr(`${s(6)}.6 Serial Position Effects`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The architecture models primacy, recency, and distinctiveness effects observed in human memory (Murdock, 1962):")]),

            p([code("primacy_window = round(lerp(5, 2, F))")], { style: "Equation" }),
            p([code("primacy_bonus = lerp(1.2, 2.0, S)")], { style: "Equation" }),
            p([code("recency_window = round(lerp(7, 3, F))")], { style: "Equation" }),
            p([code("rehearsal_curve_depth = lerp(0.2, 0.6, S)")], { style: "Equation" }),

            p([tr("The von Restorff (isolation) effect enhances memory for distinctive items (Hunt, 1995):")]),

            p([code("distinctiveness_threshold = lerp(0.6, 0.8, F)")], { style: "Equation" }),
            p([code("von_restorff_multiplier = lerp(1.5, 3.0, S)")], { style: "Equation" }),

            p([tr("Items in the middle region suffer interference:")]),

            p([code("interference_zone = positions[primacy_window+1 : −recency_window]")], { style: "Equation" }),
            p([code("middle_suppression = lerp(0.8, 0.5, S) × (1 − F)")], { style: "Equation" }),

            p([tr(`${s(6)}.7 Emotional Consolidation`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr(`High-emotion events trigger enhanced consolidation, following McGaugh's (2004) findings. As detailed in Section ${s(7)}.1.1, consolidation operates on stored memory metadata:`)]),

            p([code("θ_intensity = lerp(0.6, 0.8, 1 − S)")], { style: "Equation" }),
            p([code("θ_arousal = lerp(0.4, 0.2, S)")], { style: "Equation" }),
            p([code("# Consolidation uses stored memory emotional metadata")], { style: "Equation" }),
            p([code("trigger = (m.metadata.s_emotion_max ≥ θ_intensity) AND")], { style: "Equation" }),
            p([code("           (m.metadata.s_arousal_avg ≥ θ_arousal)")], { style: "Equation" }),

            p([tr("Flashbulb memories receive extended half-life bonuses based on the memory's peak emotional intensity:")]),

            p([code("flashbulb_threshold = lerp(0.9, 0.4, S)")], { style: "Equation" }),
            p([code("# Half-life bonus uses stored memory emotional peak")], { style: "Equation" }),
            p([code("emotional_half_life_bonus = exp(lerp(0, ln(3), S)) ×")], { style: "Equation" }),
            p([code("                             (1 + m.metadata.s_emotion_max)")], { style: "Equation" }),

            p([tr(`The emotional metrics (s_emotion_max, s_arousal_avg) are accumulated during memory formation and stored with the memory (Section ${s(6)}.1.1).`)]),

            p([code("cascade_radius = round(lerp(1, 5, S))")], { style: "Equation" }),
            p([code("cascade_decay = lerp(0.7, 0.3, S)")], { style: "Equation" }),
        ],

        // ==================== SECTION 7: CONSOLIDATION AND GRAPH ====================
        consolidationGraph: [
            new Paragraph({ children: [new PageBreak()] }),
            p([tr(`${s(7)}. Consolidation and Graph Integration`)], { heading: HeadingLevel.HEADING_1 }),

            p([tr(`The consolidation system transforms episodic memories into semantic structures through clustering, summarization, and knowledge graph construction. Memory-level storage (Section ${s(4)}.4) ensures each embedding represents a coherent unit.`)]),

            p([tr(`${s(7)}.1 Consolidation Triggers`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr(`Consolidation operates on stored memory representatives (e_rep from Section ${s(4)}.4.5), not individual signals. It activates under capacity, rate, or temporal conditions:`)]),

            p([code("should_consolidate = (memory_count > consolidation_threshold) OR")], { style: "Equation" }),
            p([code("                      (m_rate < rate_target / 2) OR")], { style: "Equation" }),
            p([code("                      (elapsed_time > consolidation_interval)")], { style: "Equation" }),

            p([tr("The consolidation rate adapts to Stability and Sensitivity (write_rate tracks memory writes, not signal writes):")]),

            p([code("rate_consolidate = (1 / max(consolidation_interval, 1)) ×")], { style: "Equation" }),
            p([code("                    (0.3 + 0.7T) × (1 − 0.5S)")], { style: "Equation" }),

            p([tr(`${s(7)}.1.1 Activity-Aware Scheduling`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Consolidation runs during idle periods and preempts for retrieval. The is_accumulating_memory check ensures consolidation doesn't interrupt mid-memory accumulation:")]),

            p([code("idle_required(T) = round(0.25 × win_rate_s(T))")], { style: "Equation" }),
            p([code("idle_for_s = now_s() − to_s(last_retrieval_ts)")], { style: "Equation" }),
            p([tr("Normative rule (MUST): on any retrieval attempt (including interrupt injection), set last_retrieval_ts ← now_ms().")]),
            p([code("# Consolidation waits for memory completion, not just signal arrival")], { style: "Equation" }),
            p([code("should_start = (NOT is_accumulating_memory) AND")], { style: "Equation" }),
            p([code("                (retrieval_queue_depth == 0) AND")], { style: "Equation" }),
            p([code("                (idle_for_s ≥ idle_required(T))")], { style: "Equation" }),

            p([tr(`Consolidation begins only after the current memory has been flushed (Section ${s(4)}.4.3) and the idle period has elapsed. On retrieval events, consolidation pauses, commits micro-batches, and resumes when idle.`)]),

            p([tr(`${s(7)}.2 Consolidation Scoring`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr(`Each stored embedding represents a memory representative (e_rep from Section ${s(4)}.4.5). Each memory receives a consolidation score determining merge priority:`)]),

            p([code("score_consolidate(m) = weight_strength × strength(m) −")], { style: "Equation" }),
            p([code("                        weight_redundancy × redundancy(m) +")], { style: "Equation" }),
            p([code("                        weight_connectivity × connectivity(m) +")], { style: "Equation" }),
            p([code("                        weight_stability × stability(m)")], { style: "Equation" }),

            p([tr("Weights derive from knobs:")]),

            p([code("weight_strength = T; weight_redundancy = F; weight_connectivity = S; weight_stability = T")], { style: "Equation" }),

            p([tr("Low-scoring memories are marked for merging.")]),

            p([tr(`${s(7)}.3 Clustering and Summarization`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Marked memories cluster via density-based methods (e.g., DBSCAN) or k-means using embedding similarity:")]),

            p([code("cluster_i = {m_j | cos(m_j, μ_i) > merge_threshold}")], { style: "Equation" }),
            p([code("μ_i = centroid(cluster_i)")], { style: "Equation" }),

            p([tr("Summary nodes replace clusters:")]),

            p([code("summary.embedding = μ_i")], { style: "Equation" }),
            p([code("summary.blob = summarize(fetch_blobs(cluster_i))")], { style: "Equation" }),
            p([code("summary.metadata.sources = [m.id for m in cluster_i]")], { style: "Equation" }),

            p([tr(`${s(7)}.4 Semantic Extraction`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("For sufficiently large clusters, semantic extraction identifies entities and relations:")]),

            p([code("extraction_batch_size = round(lerp(8, 32, T))")], { style: "Equation" }),
            p([code("min_cluster_size = round(lerp(3, 10, F))")], { style: "Equation" }),
            p([code("entity_frequency_threshold = round(lerp(5, 15, T))")], { style: "Equation" }),
            p([code("extraction_interval = lerp(300, 3600, T)  # 5 min → 1 hour")], { style: "Equation" }),
            p([code("max_extractions_per_cycle = round(lerp(20, 5, T))")], { style: "Equation" }),

            p([tr("Extraction uses structured prompting to identify named entities (people, places, organizations, concepts) and relationships (co-occurrence, implication, contradiction).")]),

            p([tr(`${s(7)}.5 Knowledge Graph Construction`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The graph comprises three node types:")]),

            p([bold("Memory Nodes: "), tr("Summarized embeddings from merged clusters")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("Entity Nodes: "), tr("Named entities extracted from content blobs")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("Concept Nodes: "), tr("Emergent centroids representing recurrent topics")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([tr("Edge types capture relationships:")]),

            p([tr("Edge weight convention (MUST): all association weights are normalized to [0, 1]. For cosine-based edges, store weight01 = clamp((cos_sim + 1) / 2, 0, 1).")]),

            p([bold("co_occurs: "), tr("Shared context or temporal proximity")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("implies: "), tr("Directional correlation in embedding drift")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("causes: "), tr("Directional correlation in embedding drift")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("contradicts: "), tr("Strong negative similarity or constraint violation")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("reinforces: "), tr("Frequent joint retrieval")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("derived_from: "), tr("Links summaries to source memories")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("similar_to: "), tr("High cosine similarity (soft equivalence)")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("has_label: "), tr("Attaches an extracted label/tag to a node")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([tr(`${s(7)}.5.1 Edge Construction`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Co-occurrence edges derive from embedding similarity:")]),

            p([code("for (m_i, m_j) in cluster.sources:")], { style: "Equation" }),
            p([code("    cos_sim ← cos(m_i.embedding, m_j.embedding)")], { style: "Equation" }),
            p([code("    if cos_sim > lerp(0.85, 0.95, F):")], { style: "Equation" }),
            p([code("        weight01 ← clamp((cos_sim + 1) / 2, 0, 1)")], { style: "Equation" }),
            p([code("        create_edge(m_i, m_j, 'co_occurs', weight01)")], { style: "Equation" }),

            p([tr("Causal edges derive from temporal drift:")]),

            p([code("temporal_order ← sort_by_timestamp(cluster.sources)")], { style: "Equation" }),
            p([code("for i in range(len(temporal_order) − 1):")], { style: "Equation" }),
            p([code("    m_i, m_j ← temporal_order[i], temporal_order[i+1]")], { style: "Equation" }),
            p([code("    drift_vec ← m_j.embedding − m_i.embedding")], { style: "Equation" }),
            p([code("    drift_mag ← ‖drift_vec‖")], { style: "Equation" }),
            p([code("    if drift_mag > lerp(0.15, 0.35, T):")], { style: "Equation" }),
            p([code("        weight01 ← clamp(drift_mag / 2, 0, 1)  # drift_mag ∈ [0,2] when embeddings are L2-normalized")], { style: "Equation" }),
            p([code("        create_edge(m_i, m_j, 'causes', weight01)")], { style: "Equation" }),

            p([tr(`${s(7)}.6 Graph-Augmented Retrieval`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr(`Retrieval combines vector similarity with graph expansion. Both initial retrieval and re-ranking use the memory centroid (μ_acc from Section ${s(4)}.4.1) as the query vector, ensuring retrieved memories are ranked by relevance to the overall context rather than momentary signal fluctuations:`)]),

            p([code("# Query and re-rank using current memory centroid")], { style: "Equation" }),
            p([code(`q ← μ_acc  # memory centroid from Section ${s(4)}.4.1`)]),
            p([code("results_vec ← topK(vector_search(q, k=kNN_size(F)))")], { style: "Equation" }),
            p([code("seed_nodes ← [r.id for r in results_vec]")], { style: "Equation" }),
            p([code("expanded_nodes ← graph.traverse(seed_nodes, depth=graph_depth(T), min_edge_weight=min_edge_weight(F))")], { style: "Equation" }),
            p([code("combined ← union(seed_nodes, expanded_nodes)")], { style: "Equation" }),
            p([code("re_ranked ← sort_by(cos(q, embeddings(combined)))  # re-rank by memory centroid")], { style: "Equation" }),

            p([tr("Graph expansion uses recursive traversal with depth limits to find related context that pure vector search might miss. Using the memory centroid maintains consistency between vector search and graph expansion results.")]),
        ],

        // ==================== SECTION 8: INTERRUPT GATE ====================
        interruptGate: [
            new Paragraph({ children: [new PageBreak()] }),
            p([tr(`${s(8)}. Interrupt Gate and Streaming Integration`)], { heading: HeadingLevel.HEADING_1 }),

            p([tr("The interrupt gate controls when retrieved memories enter active context during streaming generation. The gate balances novelty value against disruption cost. The interrupt gate operates on memory-level context, using centroids (μ_acc) rather than individual signal embeddings for novelty and relevance computation.")]),

            p([tr(`${s(8)}.1 Marginal Utility Computation`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Novelty thresholds scale with knobs and refractory state:")]),

            p([code("τ_novelty = lerp(0.10, 0.35, F) × (1 − 0.15S) × (1 + 0.3T)")], { style: "Equation" }),
            p([code("τ_mu = lerp(0.08, 0.18, F) × (1 − 0.4S) × (1 + 0.4T)")], { style: "Equation" }),
            p([code("retrieval_thresh(F) = lerp(0.25, 0.60, F)")], { style: "Equation" }),

            p([tr("Refractory dynamics suppress rapid successive interrupts:")]),

            p([tr("Interrupt state (per stream):")]),
            p([code("prev_x is unset on the first signal of a stream; set prev_x ← x_t after processing each signal")], { style: "Equation" }),
            p([code("first_step ← (prev_x is unset for this stream)")], { style: "Equation" }),
            p([code("if first_step:")], { style: "Equation" }),
            p([code("    # cold start: do not update drift_accum")], { style: "Equation" }),
            p([code("    drift_accum ← drift_accum")], { style: "Equation" }),
            p([code("else:")], { style: "Equation" }),
            p([code("    drift_accum ← drift_accum + cosine_dist(x_t, prev_x)  # cumulative drift")], { style: "Equation" }),
            p([code("Δ ← drift_accum − drift_at_last_interrupt  # cumulative drift since last interrupt")], { style: "Equation" }),
            p([code("τ_refrac = lerp(24, 96, T) × lerp(1.4, 1.0, S)")], { style: "Equation" }),
            p([code("k_refrac = lerp(0.20, 0.05, T) × lerp(0.8, 1.2, F)")], { style: "Equation" }),
            p([code("M_refrac = 1.0 + k_refrac × exp(−Δ / τ_refrac)")], { style: "Equation" }),
            p([tr("On interrupt: set drift_at_last_interrupt ← drift_accum (resetting Δ to 0 for subsequent signals).")]),

            p([tr("Effective thresholds incorporate refractory pressure:")]),

            p([code("τ_novelty_eff = τ_novelty × M_refrac")], { style: "Equation" }),
            p([code("τ_mu_eff = τ_mu × M_refrac")], { style: "Equation" }),

            p([tr(`${s(8)}.2 Marginal Utility Score`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The marginal utility (MU) of a candidate memory combines four factors. Context comparisons use memory centroids rather than individual signal embeddings:")]),

            p([code("# Context window contains recent memory centroids, not individual signals")], { style: "Equation" }),
            p([code("ctx_window ← recent_memory_centroids  # bounded deque of recent memory representatives (e_rep)")], { style: "Equation" }),
            p([code("included_set ← {embedding(m) | m already injected into the current context window}")], { style: "Equation" }),
            p([tr("Fallback: if included_set is empty, treat redundancy(·, included_set) = 0 and overlap_star = −1.")]),
            p([code("if |ctx_window| == 0:")], { style: "Equation" }),
            p([code("    ctx_centroid ← μ_acc  # fallback: use current memory centroid")], { style: "Equation" }),
            p([code("else:")], { style: "Equation" }),
            p([code("    ctx_centroid ← mean(ctx_window)  # centroid of recent memory centroids")], { style: "Equation" }),

            p([code("weights_mu_raw = [lerp(0.40, 0.60, F),   # coverage gain")], { style: "Equation" }),
            p([code("         lerp(0.35, 0.25, F),   # relevance")], { style: "Equation" }),
            p([code("         lerp(0.15, 0.25, S),   # redundancy penalty")], { style: "Equation" }),
            p([code("         lerp(0.15, 0.25, S)]   # incoherence penalty")], { style: "Equation" }),
            p([code("[weight_cov, weight_rel, weight_red, weight_incoh] = normalize(weights_mu_raw)")], { style: "Equation" }),

            p([code("mu = weight_cov × coverage_gain(candidate | included_set) +")], { style: "Equation" }),
            p([code("      weight_rel × cos(candidate, ctx_centroid) −")], { style: "Equation" }),
            p([code("      weight_red × redundancy(candidate, included_set) −")], { style: "Equation" }),
            p([code("      weight_incoh × (1 − coherence_struct_t)  # structural coherence penalty")], { style: "Equation" }),

            p([tr(`${s(8)}.3 Gate Decision Logic`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Duplicate suppression threshold:")]),

            p([code("dup_thresh = lerp(0.88, 0.96, F) × (0.98 + 0.02T)")], { style: "Equation" }),
            p([code("K = round(lerp(10, 6, F))  # candidates to evaluate")], { style: "Equation" }),

            p([tr(`${s(8)}.3.1 Write Exclusion Filter`)], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Memories stored during the current accumulation unit are excluded from interrupt consideration to prevent self-triggering. Using the accumulation start timestamp ensures all memories written within the current unit are excluded:")]),

            p([code("# Exclude memories written during current accumulation to prevent self-triggering")], { style: "Equation" }),
            p([code(`write_exclusion_ts ← t_start  # start timestamp from Section ${s(4)}.4.1`)]),
            p([code("candidates_eligible ← {c ∈ candidates | c.created_at < write_exclusion_ts}")], { style: "Equation" }),

            p([tr("This filter is applied before novelty and marginal utility evaluation. All subsequent gate logic operates on candidates_eligible rather than the raw candidate set, preventing recursive triggering within a coherent thought unit.")]),

            p([tr("Boundary-aware override permits lower-threshold interrupts at natural boundaries:")]),

            p([code("boundary_mult = lerp(1.3, 2.0, F) × lerp(1.1, 0.9, S)")], { style: "Equation" }),

            p([tr("The gate permits interrupt when:")]),

            p([code("at_drift_boundary = should_flush  # boundary signal from the current accumulator")], { style: "Equation" }),
            p([code("candidate_star = argmax_{c ∈ candidates_eligible} mu(c)")], { style: "Equation" }),
            p([code("mu_star = mu(candidate_star)")], { style: "Equation" }),
            p([code("rel_star = cos(candidate_star, ctx_centroid)")], { style: "Equation" }),
            p([code("if |included_set| == 0:")], { style: "Equation" }),
            p([code("    overlap_star = −1.0")], { style: "Equation" }),
            p([code("else:")], { style: "Equation" }),
            p([code("    overlap_star = max_{y ∈ included_set} cos(candidate_star, y)")], { style: "Equation" }),
            p([code("if |ctx_window| == 0:")], { style: "Equation" }),
            p([code("    novelty_star = 1.0")], { style: "Equation" }),
            p([code("else:")], { style: "Equation" }),
            p([code("    max_cos = max_{c ∈ ctx_window} cos(candidate_star, c)  # in [−1, 1]")], { style: "Equation" }),
            p([code("    novelty_star = clamp((1 − max_cos) / 2, 0, 1)")], { style: "Equation" }),
            p([code("allow_interrupt = ")], { style: "Equation" }),
            p([code("    (rel_star ≥ retrieval_thresh(F)) AND")], { style: "Equation" }),
            p([code("    (novelty_star ≥ τ_novelty_eff OR mu_star ≥ τ_mu_eff) AND")], { style: "Equation" }),
            p([code("    (overlap_star < dup_thresh) AND")], { style: "Equation" }),
            p([code("    (at_drift_boundary OR mu_star ≥ boundary_mult × τ_mu_eff)")], { style: "Equation" }),

            p([tr("This logic suppresses low-drift interrupts unless the marginal utility substantially exceeds threshold, while permitting normal-threshold interrupts at natural transition points.")]),

            p([tr(`${s(8)}.4 Streaming Pacing`)], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Streaming retrieval is gated by cumulative drift rate within the accumulation unit. Retrieval checks trigger when drift exceeds threshold or at boundaries:")]),

            p([code("# Pacing tracks drift within current memory formation")], { style: "Equation" }),
            p([tr("where cosine_dist(u, v) = 1 − cos(u, v).")]),
            p([code("first_step ← (x_last_check is unset for this stream)  # MUST occur once per stream")], { style: "Equation" }),
            p([code("if first_step: x_last_check ← x_t; drift_acc_pacing ← 0")], { style: "Equation" }),
            p([code("drift_acc_pacing += cosine_dist(x_t, x_last_check)")], { style: "Equation" }),
            p([code("pacing_thresh(S) = lerp(0.5, 0.1, S)")], { style: "Equation" }),
            p([code("# Retrieval triggered when drift exceeds threshold or at memory boundary")], { style: "Equation" }),
            p([code("if drift_acc_pacing > pacing_thresh(S) OR should_flush:")], { style: "Equation" }),
            p([code("    trigger_check(); x_last_check ← x_t; drift_acc_pacing ← 0")], { style: "Equation" }),

            p([code("max_wait_drift(F) = lerp(2.0, 0.5, F)")], { style: "Equation" }),
            p([code("adjacent_window(F) = round(lerp(8, 1, F))")], { style: "Equation" }),

            p([tr(`High Sensitivity produces frequent checks triggered by small content shifts; high Focus enforces strict drift limits. Memory boundaries (Section ${s(4)}.4.3) also trigger retrieval checks to ensure context updates align with natural thought transitions.`)]),
        ],
        // ==================== SECTION 9: EXPERIMENTAL RESULTS ====================
        experimentalResults: [
            new Paragraph({ children: [new PageBreak()] }),
            p([tr(`${s(9)}. Experimental Results`)], { heading: HeadingLevel.HEADING_1 }),

            p([tr("We present preliminary experimental results collected from live chat sessions to validate the adaptive mechanisms.")]),

            p([tr(`${s(9)}.1 Threshold Adaptation`)], { heading: HeadingLevel.HEADING_2 }),
            p([tr("The dynamic threshold (θ_dynamic) successfully tracked score distributions. In high-volatility inputs (within-accumulator drift_acc > 1.0), thresholds relaxed to ~0.15, while stable contexts tightened to ~0.27.")]),

            p([tr(`${s(9)}.2 Boundary Detection`)], { heading: HeadingLevel.HEADING_2 }),
            p([tr("Accumulator drift (drift_acc) aligned with semantic shifts. Conversation turns with distinct topics triggered flushes (boundary_score > 0.3) while coherent continuations remained accumulated.")]),

            p([tr(`${s(9)}.3 Latency and Performance`)], { heading: HeadingLevel.HEADING_2 }),
            p([tr("End-to-end processing per token averaged < 50ms. Graph expansion added < 10ms overhead due to efficient kNN (k=32) and limited expansion depth (d=2).")]),
        ],

        // ==================== APPENDICES (NON-NUMERIC) ====================
        appendices: [
            new Paragraph({ children: [new PageBreak()] }),
            p([tr("Appendix A. State Variables Map")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("This appendix enumerates the state variables used by the specification and separates retained state (carried across timesteps) from per-step derived quantities.")]),

            p([bold("Accumulator state (per stream; retained across timesteps): ")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([code("{μ_acc, drift_acc, s_sum, s_max, n, e_peak, emo_max, arousal_sum, eta_acc, coherence_prev, acc_signals_window, t_start, last_signal_ts, last_write_ts, drift_accum, drift_at_last_interrupt, drift_acc_pacing, x_last_check, prev_x}")], { numbering: { reference: "bullet-list", level: 1 } }),

            p([bold("Global state (retained across timesteps): ")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([code("{u_uncertainty, mood_vector, last_mood_ts, theta_dynamic, theta_target, hysteresis, m_rate, dt_ema, rate_ticks, last_rate_timestamp, reliability, last_retrieval_ts}")], { numbering: { reference: "bullet-list", level: 1 } }),

            p([bold("Buffers (retained across timesteps; bounded by window rules): ")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([code("{signal_stream, score_stream, memory_stream, recent_memory_centroids}")], { numbering: { reference: "bullet-list", level: 1 } }),

            p([bold("Recorded signal fields (per signal): ")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([code("{coherence_struct_t → SIGNALS.coherence, focus_spread_t → SIGNALS.focus_spread}")], { numbering: { reference: "bullet-list", level: 1 } }),

            p([bold("Recorded global fields: ")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([code("{u(t) → STATE.u_uncertainty, M_t → STATE.mood_vector}")], { numbering: { reference: "bullet-list", level: 1 } }),

            p([bold("Per-step derived scalars (ephemeral; recomputed each step): ")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([code("{signal_gap_s, coherence_curr, s_avg, S_window, boundary_score, should_flush, write_memory, Δwrites}")], { numbering: { reference: "bullet-list", level: 1 } }),

            new Paragraph({ children: [new PageBreak()] }),
            p([tr("Appendix B. Derived Signals: Definitions and Bounds")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("This appendix defines non-trivial derived signals used throughout the specification. Unless otherwise stated, all derived scalars are clamped to [0, 1].")]),

            p([bold("novelty: "), tr("A normalized dissimilarity-to-context signal.")]),
            p([code("max_cos ← max_{c ∈ recent_context} cos(x_t, c)  # in [−1, 1]")], { style: "Equation" }),
            p([code("novelty_t ← clamp((1 − max_cos) / 2, 0, 1)")], { style: "Equation" }),
            p([tr("Fallback: if recent_context is empty, set novelty_t ← 1.")]),

            p([bold("μ_sim: "), tr("A normalized mean similarity to context.")]),
            p([code("mean_cos ← mean_{c ∈ recent_context} cos(x_t, c)")], { style: "Equation" }),
            p([code("μ_sim ← clamp((mean_cos + 1) / 2, 0, 1)")], { style: "Equation" }),
            p([tr("Fallback: if recent_context is empty, set μ_sim ← 0.5.")]),

            p([bold("rarity_t: "), tr("A normalized rarity signal defined as dissimilarity-to-context mean.")]),
            p([code("rarity_t ← clamp(1 − μ_sim, 0, 1)")], { style: "Equation" }),
            p([tr("Fallback: inherits μ_sim fallback, so rarity_t defaults to 0.5 when recent_context is empty.")]),

            p([bold("relevance_to_task(q, task_ctx): "), tr("A normalized relevance of embedding q to a task context set.")]),
            p([code("if |task_ctx| == 0: return 0.5")], { style: "Equation" }),
            p([code("return clamp((cos(q, mean(task_ctx)) + 1) / 2, 0, 1)")], { style: "Equation" }),

            p([bold("novelty_to_set(q, S_set_embeddings): "), tr("A normalized novelty of q relative to a set of embeddings.")]),
            p([code("novelty_to_set(q, S_set_embeddings) ← 1 − redundancy(q, S_set_embeddings)")], { style: "Equation" }),

            p([bold("ΔSSE: "), tr("A normalized improvement in reconstruction/prediction error (utility proxy).")]),
            p([code("ΔSSE ← clamp((SSE_prev − SSE_curr) / max(SSE_prev, ε), 0, 1)")], { style: "Equation" }),
            p([tr("Fallback: if SSE signals are unavailable, set ΔSSE ← 0.")]),

            p([bold("redundancy(a, S_set): "), tr("A normalized redundancy of item a w.r.t. a set S_set.")]),
            p([code("redundancy(a, S_set) ← max_{s ∈ S_set} clamp((cos(a, s) + 1) / 2, 0, 1)")], { style: "Equation" }),
            p([tr("Fallback: if S_set is empty, redundancy(a, S_set) ← 0.")]),

            p([bold("coverage_gain(candidate | included_set): "), tr("Incremental coverage contribution of adding candidate.")]),
            p([code("coverage_gain(candidate | included_set) ← 1 − redundancy(candidate, included_set)")], { style: "Equation" }),

            p([bold("contextual_gain(m): "), tr("A normalized marginal gain attributed to using memory m in context.")]),
            p([code("contextual_gain(m) ∈ [−1, +1]  # normalized gain/loss signal")], { style: "Equation" }),
            p([tr("Fallback: if marginal gain cannot be estimated, contextual_gain(m) ← 0.")]),

            p([bold("connectivity(m): "), tr("A normalized connectivity score (graph centrality proxy).")]),
            p([code("connectivity(m) ← clamp(degree(m) / deg_cap, 0, 1)")], { style: "Equation" }),
            p([tr("Fallback: if no graph structure is present, connectivity(m) ← 0.")]),

            p([bold("manifold_complexity: "), tr("A local embedding-stream complexity proxy used in working-memory cost.")]),
            p([code("manifold_complexity ← clamp((1 − mean_cos_window) / 2, 0, 1)")], { style: "Equation" }),
            p([tr("where mean_cos_window is the mean cosine similarity across adjacent embeddings in a short local window.")]),
            p([code("win_complex(T) = round(lerp(8, 32, T))")], { style: "Equation" }),
            p([code("window_signals ← tail(signal_stream, win_complex(T) + 1)")], { style: "Equation" }),
            p([code("if |window_signals| < 2:")], { style: "Equation" }),
            p([code("    mean_cos_window ← 1.0")], { style: "Equation" }),
            p([code("else:")], { style: "Equation" }),
            p([code("    mean_cos_window ← mean([cos(window_signals[i−1], window_signals[i]) for i = 1..|window_signals|−1])")], { style: "Equation" }),

            p([bold("kNN_similarities: "), tr("Vector of cosine similarities returned by nearest-neighbor search for the current query embedding.")]),
            p([code("kNN_similarities = [sim_1, …, sim_k] where sim_i ∈ [−1, 1]")], { style: "Equation" }),

            p([bold("kNN_size, graph_depth: "), tr("Discrete retrieval hyperparameters.")]),
            p([code("kNN_size(F) = round(lerp(64, 4, F))")], { style: "Equation" }),
            p([code("graph_depth(T) = round(lerp(1, 3, T))")], { style: "Equation" }),
            p([code("min_edge_weight(F) = lerp(0.70, 0.95, F)  # minimum association weight to traverse")], { style: "Equation" }),

            new Paragraph({ children: [new PageBreak()] }),
            p([tr("Appendix C. Main Loop and Normative Invariants")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("Main loop (per signal):")]),
            p([tr("1) Compute now_ms(), now_s(), and signal_gap_s (before updating last_signal_ts).")], { numbering: { reference: "numbered-list", level: 0 } }),
            p([tr("2) Update accumulator running statistics (μ_acc, drift_acc, s_sum, s_max, n, emo_max, arousal_sum, eta_acc, coherence_curr/coherence_prev).")], { numbering: { reference: "numbered-list", level: 0 } }),
            p([tr("3) Update uncertainty u(t), then update θ_dynamic and hysteresis.")], { numbering: { reference: "numbered-list", level: 0 } }),
            p([tr("4) Compute should_flush and spike_bypass/force_write; if boundary condition holds, compute S_window and decide write_memory.")], { numbering: { reference: "numbered-list", level: 0 } }),
            p([tr("5) Update rate state (m_rate, dt_ema, rate_ticks, last_rate_timestamp) using Δwrites derived from this step's write_memory. Compute ρ_hat_next and then set ρ_hat_prev ← ρ_hat_next.")], { numbering: { reference: "numbered-list", level: 0 } }),
            p([tr("6) If should_flush: finalize the unit. If write_memory: record the memory and update last_write_ts ← now_ms(); else discard. In both cases, reset_accumulator() for the next unit.")], { numbering: { reference: "numbered-list", level: 0 } }),

            p([tr("Normative invariants:")]),
            p([bold("MUST: "), tr("All stored timestamps are milliseconds since epoch; derived time deltas are seconds.")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("MUST: "), tr("All thresholds and composite scores are clamped to [0, 1].")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("MUST: "), tr("Strength is clamped to [0, 1] (bounded reinforcement).")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("MUST: "), tr("If weight_sum < ε during score normalization, return score = 0.")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("SHOULD: "), tr("All cosine-derived quantities that are interpreted as probabilities are map01/clamped consistently.")], { numbering: { reference: "bullet-list", level: 0 } }),
        ],
    };
};

// ==================== DOCUMENT GENERATION ====================

// Get algorithm sections with offset=0 for standalone algorithms.docx
const algorithmSections = createAlgorithmSections(0);

// Flatten all sections into children array
const allSectionChildren = [
    ...algorithmSections.mathFoundations,
    ...algorithmSections.coreAdaptation,
    ...algorithmSections.structuralMetrics,
    ...algorithmSections.dynamicThresholding,
    ...algorithmSections.writePacing,
    ...algorithmSections.reinforcementDecay,
    ...algorithmSections.advancedCognitive,
    ...algorithmSections.consolidationGraph,
    ...algorithmSections.interruptGate,
    ...algorithmSections.experimentalResults,
    ...algorithmSections.appendices,
];

// Create the document
const doc = new Document({
    styles: {
        default: {
            document: {
                run: { font: "Times New Roman", size: 24 }
            }
        },
        paragraphStyles: paragraphStyles
    },
    numbering: {
        config: numberingConfig
    },
    sections: [{
        properties: {
            page: {
                margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
                size: { orientation: PageOrientation.PORTRAIT }
            }
        },
        headers: {
            default: new Header({
                children: [new Paragraph({
                    alignment: AlignmentType.RIGHT,
                    children: [new TextRun({ text: "Cortext: Algorithms Specification", italics: true, size: 20 })]
                })]
            })
        },
        footers: {
            default: new Footer({
                children: [new Paragraph({
                    alignment: AlignmentType.CENTER,
                    children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })]
                })]
            })
        },
        children: [
            // ==================== TITLE ====================
            p([bold("Cortext: A Three-Knob Adaptive Memory Architecture")], { heading: HeadingLevel.TITLE }),
            p([tr("Algorithmic Specification", { bold: true, size: 32 })], { alignment: AlignmentType.CENTER, spacing: { after: 360 } }),

            // ==================== ABSTRACT ====================
            p([bold("Abstract")], { alignment: AlignmentType.CENTER, spacing: { before: 240, after: 120 } }),

            p([tr("This document details the complete algorithmic specification for Cortext, a biologically-inspired adaptive memory system. It serves as a technical reference for implementation, focusing purely on mathematical definitions, signal processing flows, and adaptive control logic governed by the three primary parameters: Focus (F), Sensitivity (S), and Stability (T).")],
                { style: "Abstract" }),

            // All algorithm sections
            ...allSectionChildren,
        ]
    }]
});

// Write the document
Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync("./algorithms.docx", buffer);
    console.log("Document created successfully: algorithms.docx");
}).catch(err => {
    console.error("Error creating document:", err);
});

// ==================== MODULE EXPORTS ====================

module.exports = {
    // Helper functions
    p, bold, italic, code, tr,

    // Styles
    tableBorder, cellBorders, paragraphStyles, numberingConfig,

    // Section generator
    createAlgorithmSections,

    // Pre-built sections for paper.js (with +2 offset)
    paperSections: createAlgorithmSections(2),
};
