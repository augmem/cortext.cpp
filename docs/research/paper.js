const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
    Header, Footer, AlignmentType, PageOrientation, LevelFormat,
    HeadingLevel, BorderStyle, WidthType, TabStopType,
    TabStopPosition, ShadingType, PageNumber, PageBreak } = require('docx');
const fs = require('fs');

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

// Table border style
const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

// Create the document
const doc = new Document({
    styles: {
        default: {
            document: {
                run: { font: "Times New Roman", size: 24 }
            }
        },
        paragraphStyles: [
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
        ]
    },
    numbering: {
        config: [
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
        ]
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
                    children: [new TextRun({ text: "Cortext: A Three-Knob Cognitive Memory Architecture", italics: true, size: 20 })]
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
            p([tr("for Continuous Cognitive Processing", { bold: true, size: 32 })], { alignment: AlignmentType.CENTER, spacing: { after: 360 } }),

            // Authors
            p([tr("Technical Report — December 2025", { italics: true })], { alignment: AlignmentType.CENTER, spacing: { after: 480 } }),

            // ==================== ABSTRACT ====================
            p([bold("Abstract")], { alignment: AlignmentType.CENTER, spacing: { before: 240, after: 120 } }),

            p([tr("We present Cortext, a biologically-inspired adaptive memory system governed by three continuous control parameters: Focus (F), Sensitivity (S), and Stability (T). Unlike traditional memory architectures that employ discrete operational modes, Cortext achieves developmental progression through parameter-derived rate modulation, allowing behavior to emerge continuously from the interaction of knob settings and experiential mass. The architecture integrates established cognitive science principles—including Cowan's working memory constraints and Nader's reconsolidation dynamics—into a unified computational framework. We derive all system parameters from the three primary knobs through principled mathematical transformations, reducing reliance on fixed constants. The system demonstrates self-calibrating priors that blend with evidence using uncertainty-weighted Bayesian averaging, homeostatic threshold control with effective sample size estimation, and graph-augmented retrieval combining embedding similarity with semantic extraction. Experimental analysis indicates the architecture maintains stable operation across developmental phases while adapting write rates, decay dynamics, and retrieval precision to environmental demands. This work contributes a formally specified cognitive memory model suitable for implementation in streaming AI systems requiring persistent, context-aware memory.")],
                { style: "Abstract" }),

            p([tr("Keywords: ", { bold: true }), tr("cognitive architecture, adaptive memory, working memory, episodic memory, semantic memory, knowledge graphs, homeostatic control")],
                { style: "Abstract", spacing: { after: 360 } }),

            // ==================== 1. INTRODUCTION ====================
            new Paragraph({ children: [new PageBreak()] }),
            p([tr("1. Introduction")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("Memory systems in artificial intelligence face a fundamental tension between plasticity and stability. Systems that learn rapidly risk catastrophic interference, while those that maintain stable representations may fail to capture novel patterns (McCloskey & Cohen, 1989). Biological memory systems resolve this tension through sophisticated regulatory mechanisms that modulate learning rates, decay dynamics, and retrieval thresholds in response to environmental demands and internal state (McClelland et al., 1995).")]),

            p([tr("This paper introduces Cortext, a cognitive memory architecture that addresses this stability-plasticity dilemma through three continuous control parameters that govern all system behavior. Rather than implementing discrete operational modes or hard-coded phase transitions, Cortext achieves developmental progression through the continuous interaction of parameter settings with accumulated experience. The architecture draws on established findings from cognitive psychology and neuroscience, including working memory capacity limits (Cowan, 2001), memory reconsolidation (Nader et al., 2000), serial position effects (Murdock, 1962), and emotional modulation of memory (McGaugh, 2004).")]),

            p([tr("The core contribution of this work is a formally specified memory architecture in which:")]),

            p([tr("All tuneable parameters derive from three primary knobs (Focus, Sensitivity, Stability) through explicit mathematical transformations, reducing reliance on fixed constants")],
                { numbering: { reference: "numbered-list", level: 0 } }),
            p([tr("System priors self-calibrate through uncertainty-weighted Bayesian blending with observed evidence")],
                { numbering: { reference: "numbered-list", level: 0 } }),
            p([tr("Developmental phases emerge from annealed safety bounds and experiential mass accumulation, not explicit mode switching")],
                { numbering: { reference: "numbered-list", level: 0 } }),
            p([tr("A knowledge graph layer enables semantic consolidation and graph-augmented retrieval")],
                { numbering: { reference: "numbered-list", level: 0 } }),

            p([tr("The remainder of this paper is organized as follows. Section 2 reviews relevant literature. Section 3 presents the mathematical foundations including notation, helper functions, and knob-derived parameters. Section 4 details the core algorithms for Focus, Sensitivity, and Stability adaptation. Section 5 describes structural metrics and composite scoring. Section 6 covers dynamic thresholding and homeostatic control. Section 7 presents the reinforcement and decay dynamics. Section 8 describes advanced cognitive processes including working memory, metacognition, and emotional consolidation. Section 9 details the consolidation and graph integration system. Section 10 discusses implementation considerations and computational complexity. Section 11 concludes with limitations and future directions.")]),

            // ==================== 2. RELATED WORK ====================
            p([tr("2. Related Work")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("2.1 Working Memory Models")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The working memory component of Cortext draws primarily on Cowan's (2001) embedded-processes model, which posits a capacity limit of approximately 4±1 chunks for the focus of attention. This contrasts with Miller's (1956) earlier estimate of 7±2 items, which subsequent research has shown conflates chunking with raw capacity (Cowan, 2010). Our implementation respects these empirically-derived constraints while allowing for focus-dependent modulation within bounded ranges.")]),

            p([tr("Baddeley's (2000) multicomponent model informs our treatment of maintenance and rehearsal processes, though we adopt a more unified representational substrate based on distributed embeddings rather than separate phonological and visuospatial stores. The episodic buffer concept (Baddeley, 2000) aligns with our approach to binding information across modalities through shared vector spaces.")]),

            p([tr("2.2 Memory Consolidation")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The consolidation mechanisms in Cortext reflect findings from the memory reconsolidation literature (Nader et al., 2000; Nader, 2003). Reconsolidation theory posits that retrieved memories enter a labile state during which they can be modified before restabilization. Our architecture implements this through time-bounded lability windows governed by the Stability parameter, with reconsolidation gain modulated by both Sensitivity and contextual relevance.")]),

            p([tr("The distinction between episodic and semantic memory (Tulving, 1972) motivates our two-tier storage approach: a streaming episodic buffer for immediate experiences and a consolidated semantic graph for abstracted knowledge. The consolidation process transforms high-redundancy episodic clusters into summary nodes linked by typed semantic relations, consistent with complementary learning systems theory (McClelland et al., 1995).")]),

            p([tr("2.3 Emotional Influences on Memory")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("McGaugh's (2004) extensive work on emotional modulation of memory consolidation informs our treatment of affect-gated encoding. The architecture implements emotional intensity as a threshold modifier, consistent with findings that arousal enhances memory through amygdala-mediated modulation of hippocampal encoding (LaBar & Cabeza, 2006). We adopt a dimensional model of emotion (Russell, 1980) with valence and arousal as primary axes, projected from categorical emotion embeddings.")]),

            p([tr("2.4 Adaptive Control Systems")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The homeostatic threshold controller draws on classical control theory, specifically proportional-integral approaches to setpoint maintenance (Åström & Murray, 2008). The use of exponentially-weighted moving averages for rate estimation follows standard practice in adaptive systems, while our effective sample size calculation for reliability estimation extends techniques from sequential Monte Carlo methods (Liu & Chen, 1998).")]),

            // ==================== 3. MATHEMATICAL FOUNDATIONS ====================
            new Paragraph({ children: [new PageBreak()] }),
            p([tr("3. Mathematical Foundations")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("3.1 Notation and Primitives")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("We establish the following notation used throughout this paper. Let ε = 10⁻⁶ denote a small constant for numerical stability. All knob values F, S, T lie in the closed interval [0, 1].")]),

            p([tr("Core mathematical primitives:")]),

            p([code("lerp(a, b, x) = a + (b − a) × x")], { style: "Equation" }),
            p([code("clamp(v, lo, hi) = max(lo, min(v, hi))")], { style: "Equation" }),
            p([code("sigmoid(z) = 1 / (1 + exp(−z))")], { style: "Equation" }),
            p([code("EWMA(prev, x, α) = (1 − α) × prev + α × x")], { style: "Equation" }),

            p([tr("For vectors, we define cosine similarity as cos(u, v) = u·v / (‖u‖ × ‖v‖), and safe L2 normalization as l2_normalize(v) = v / max(‖v‖, ε). Shannon entropy is computed in nats: H(p) = −Σᵢ pᵢ ln(pᵢ).")]),

            p([tr("The temporal decay function follows exponential dynamics with configurable half-life:")]),

            p([code("decay(x, τ_half, Δt) = x × exp(−ln(2) × Δt / max(τ_half, τ_min))")], { style: "Equation" }),

            p([tr("where τ_min = 120 seconds provides a floor to prevent numerical instability from near-zero half-lives.")]),

            p([tr("3.2 The Three-Knob Philosophy")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Cortext is governed by three continuous control parameters, each representing a distinct dimension of cognitive regulation:")]),

            p([bold("Focus (F ∈ [0, 1]): "), tr("Perceptual selectivity and precision. Higher Focus narrows attention, increases relevance weighting, and reduces retrieval breadth. Focus modulates the trade-off between exploitation of known-relevant information and exploration of potentially useful context.")]),

            p([bold("Sensitivity (S ∈ [0, 1]): "), tr("Plasticity and affective gain. Higher Sensitivity accelerates learning, increases emotional and novelty responsiveness, and raises write-rate targets. Sensitivity governs how readily the system captures novel information and responds to salient stimuli.")]),

            p([bold("Stability (T ∈ [0, 1]): "), tr("Temporal persistence and inertia. Higher Stability lengthens memory half-lives, widens hysteresis bands, slows adaptive updates, and tightens safety bounds over time. Stability controls the resistance to change and the preservation of established knowledge.")]),

            p([tr("A central design principle is that knobs set rates rather than modes. Behavioral differences emerge continuously from parameter interactions; there are no hard-coded phase transitions or discrete operational states.")]),

            p([tr("3.3 Knob-Derived Parameters")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("All system tunables derive from the three primary knobs. This section catalogs the key derivations.")]),

            p([tr("3.3.1 Context Windows and Temporal Scales")], { heading: HeadingLevel.HEADING_3 }),

            p([code("n_ctx(T) = round(lerp(32, 256, T))")], { style: "Equation" }),
            p([code("w_score(T) = round(lerp(20, 120, T))")], { style: "Equation" }),
            p([code("w_rate_seconds(T) = round(lerp(60, 300, T))")], { style: "Equation" }),

            p([tr("The context window n_ctx determines how many recent items inform relevance computation. The scoring window w_score controls the lookback for variance estimation and percentile calculation. The rate window w_rate_seconds specifies the temporal horizon for write-rate measurement.")]),

            p([tr("3.3.2 Half-Life and Decay")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Memory half-life follows a log-scale mapping to span multiple orders of magnitude:")]),

            p([code("τ_min = 120.0 seconds (2 minutes)")], { style: "Equation" }),
            p([code("τ_max = 43200.0 seconds (12 hours)")], { style: "Equation" }),
            p([code("base_half_life(T) = exp(ln(τ_min) + T × ln(τ_max / τ_min))")], { style: "Equation" }),

            p([tr("This exponential mapping ensures that low Stability yields half-lives near 2 minutes while high Stability approaches 12 hours, with smooth interpolation across the range.")]),

            p([tr("3.3.3 Hysteresis and Rate Targets")], { heading: HeadingLevel.HEADING_3 }),

            p([code("band_min = 0.02; band_max = 0.25")], { style: "Equation" }),
            p([code("base_band(T) = lerp(band_min, band_max, T)")], { style: "Equation" }),
            p([code("r_min = 0.2; r_max = 5.0  (writes per minute)")], { style: "Equation" }),
            p([code("base_rate(S) = lerp(r_min, r_max, S)")], { style: "Equation" }),

            p([tr("The hysteresis band prevents oscillation in threshold-crossing decisions. Write-rate targets establish homeostatic setpoints for the threshold controller.")]),

            p([tr("3.3.4 Experiential Mass and Maturity")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("The system tracks accumulated experience through a maturity function that governs the annealing of safety bounds:")]),

            p([code("τ_m(T) = lerp(10.0, 200.0, T)")], { style: "Equation" }),
            p([code("maturity(t) = 1 − exp(−count / τ_m(T))")], { style: "Equation" }),

            p([tr("where count is the total number of signals processed. This produces asymptotic approach to unit maturity, with higher Stability slowing the progression to reflect greater conservatism.")]),

            p([tr("Safety bounds on the dynamic threshold anneal with maturity:")]),

            p([code("T_min(t) = lerp(0.01, 0.05, maturity(t))")], { style: "Equation" }),
            p([code("T_max(t) = lerp(0.99, 0.95, maturity(t))")], { style: "Equation" }),
            p([code("max_ΔT_per_min(t) = lerp(0.30, 0.10, maturity(t))")], { style: "Equation" }),

            p([tr("Early operation permits wide threshold excursions; mature operation constrains movement to a narrower band.")]),

            p([tr("3.4 Uncertainty Estimation")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Uncertainty u(t) ∈ [0, 1] modulates learning rates and evidence weighting. The raw uncertainty estimate blends multiple signals:")]),

            p([code("var_score_max = 0.25")], { style: "Equation" }),
            p([code("var_recent_norm = clamp(var(scores[t−w:t]) / var_score_max, 0, 1)")], { style: "Equation" }),
            p([code("coherence_complement = 1 − coherence_t")], { style: "Equation" }),

            p([tr("When prediction error signals are available, novelty and surprisal are blended:")]),

            p([code("novelty_surprise = blend([novelty_t, surprisal_t],")], { style: "Equation" }),
            p([code("                        weights = normalize([S, 1 − T]))")], { style: "Equation" }),

            p([tr("The final raw uncertainty combines these components with knob-derived weights:")]),

            p([code("weights_u = normalize([S, F, 1 − T, S × (1 − T)])")], { style: "Equation" }),
            p([code("u_raw(t) = clamp(blend([var_recent_norm, focus_spread,")], { style: "Equation" }),
            p([code("                        coherence_complement, novelty_surprise],")], { style: "Equation" }),
            p([code("                       weights = weights_u), 0, 1)")], { style: "Equation" }),

            p([tr("Smoothed uncertainty applies EWMA with a stability-dependent rate:")]),

            p([code("α_u(T) = 0.10 + (1 − T) × 0.60")], { style: "Equation" }),
            p([code("u(t) = EWMA(u(t−1), u_raw(t), α = α_u(T))")], { style: "Equation" }),

            p([tr("When structural metrics are unavailable, the fallback is u_raw(t) = 1 − maturity(t), ensuring high uncertainty during early operation.")]),

            // ==================== 4. CORE ADAPTATION ALGORITHMS ====================
            new Paragraph({ children: [new PageBreak()] }),
            p([tr("4. Core Adaptation Algorithms")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("This section presents the algorithms governing adaptation along each of the three primary dimensions. Each algorithm consists of a prior computation (executed at initialization) and a dynamic update (executed per signal).")]),

            p([tr("4.1 Focus-Driven Selectivity")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Focus governs perceptual selectivity through relevance weighting and attention width.")]),

            p([tr("4.1.1 Focus Priors")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Given Focus knob F ∈ [0, 1], compute initial priors:")]),

            p([code("weight_relevance_prior = sigmoid(2F − 1)")], { style: "Equation" }),
            p([code("coverage_gain_floor = 0.3 + 0.7F")], { style: "Equation" }),
            p([code("mismatch_weight_prior = 1 − F")], { style: "Equation" }),
            p([code("attention_width_prior = lerp(π, 0.1π, F)")], { style: "Equation" }),

            p([tr("The attention width (in radians) controls the angular spread of the receptive field in embedding space. High Focus produces narrow attention (0.1π), while low Focus permits broad capture (π).")]),

            p([tr("4.1.2 Dynamic Focus Update")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("At each signal event t with input embedding x_t:")]),

            p([code("recent_context ← tail(context_buffer, n_ctx(T))")], { style: "Equation" }),
            p([code("observed_cosine ← cos(x_t, mean(recent_context))")], { style: "Equation" }),
            p([code("weight_relevance_t ← EWMA(weight_relevance_{t−1},")], { style: "Equation" }),
            p([code("                         map01(observed_cosine), α = α_F(t))")], { style: "Equation" }),

            p([tr("where map01(z) = clamp((z + 1) / 2, 0, 1) transforms cosine values from [−1, 1] to [0, 1].")]),

            p([tr("The learning rate α_F(t) is modulated by uncertainty:")]),

            p([code("α_min_F = 0.05; α_span_F = 0.45")], { style: "Equation" }),
            p([code("α_F(t) = α_min_F + F × α_span_F × u(t)")], { style: "Equation" }),

            p([tr("High uncertainty increases learning rate, allowing faster adaptation when the environment is volatile. The Focus knob scales the uncertainty responsiveness.")]),

            p([tr("4.2 Sensitivity-Driven Plasticity")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Sensitivity governs learning speed, emotional responsiveness, and novelty capture.")]),

            p([tr("4.2.1 Sensitivity Priors")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Given Sensitivity knob S ∈ [0, 1], compute initial priors:")]),

            p([code("base_rate_prior = lerp(0.2, 5.0, S)  # writes/min")], { style: "Equation" }),
            p([code("weight_novelty_prior = 0.3 + 0.7S")], { style: "Equation" }),
            p([code("weight_surprise_prior = 0.2 + 0.8S")], { style: "Equation" }),
            p([code("weight_valence_prior = 0.4 + 0.6S")], { style: "Equation" }),
            p([code("weight_arousal_prior = S")], { style: "Equation" }),
            p([code("emotion_gain_prior = exp(1.5S)")], { style: "Equation" }),
            p([code("score_gain_prior = exp(2S)")], { style: "Equation" }),
            p([code("rate_target_prior = base_rate_prior × (0.5 + 1.5S)")], { style: "Equation" }),

            p([tr("4.2.2 Emotional Projection")], { heading: HeadingLevel.HEADING_3 }),

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

            p([tr("4.2.3 Threshold Modulation from Emotion")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Emotional activation loosens write thresholds to capture salient moments:")]),

            p([code("κ_emo ← κ_base × S  # where κ_base = 0.10")], { style: "Equation" }),
            p([code("ΔThreshold_emotion_t ← −κ_emo × emotion_intensity_t ×")], { style: "Equation" }),
            p([code("                        (0.5 + 0.5 × arousal_t)")], { style: "Equation" }),

            p([tr("This negative adjustment makes writing more likely during emotionally salient events, consistent with McGaugh's (2004) findings on arousal-enhanced encoding.")]),

            p([tr("4.2.4 Mood Integration")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Distinct from instantaneous emotion, the mood state M_t maintains a persistent background affective tone:")]),

            p([code("α_mood(S) = lerp(0.01, 0.20, S)  # reactivity")], { style: "Equation" }),
            p([code("λ_mood(T) = lerp(0.90, 0.999, T)  # decay")], { style: "Equation" }),
            p([code("M_t = λ_mood(T) × M_{t−1} + α_mood(S) × e_t")], { style: "Equation" }),
            p([code("M_t ← clamp(M_t, −1.0, 1.0)")], { style: "Equation" }),

            p([tr("Note that this update does not enforce coefficient normalization; the explicit clamp handles potential accumulation. The mood state provides a separate threshold bias:")]),

            p([code("κ_mood ← κ_base × S")], { style: "Equation" }),
            p([code("m_norm ← ‖M_t‖ / √6")], { style: "Equation" }),
            p([code("ΔThreshold_mood_t ← −κ_mood × clamp(m_norm, 0, 1)")], { style: "Equation" }),

            p([tr("4.3 Stability-Driven Persistence")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Stability governs temporal dynamics through half-life, decay rates, and hysteresis.")]),

            p([tr("4.3.1 Stability Priors")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Given Stability knob T ∈ [0, 1], compute initial priors:")]),

            p([code("hysteresis_band_prior = lerp(0.02, 0.25, T)")], { style: "Equation" }),
            p([code("half_life_prior = base_half_life(T)")], { style: "Equation" }),
            p([code("rate_decay_prior = lerp(0.60, 0.98, T)")], { style: "Equation" }),
            p([code("periphery_half_life_prior = clamp(0.5 × half_life_prior,")], { style: "Equation" }),
            p([code("                                   τ_min, τ_max)")], { style: "Equation" }),
            p([code("drift_weight_prior = 0.5 × (1 − T)")], { style: "Equation" }),

            p([tr("4.3.2 Dynamic Stability Update")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("At each signal event, compute retention statistics and adjust half-life:")]),

            p([code("active_memories ← {m | strength(m) ≥ periphery_cutoff(T)}")], { style: "Equation" }),
            p([code("observed_retention ← mean_age(active_memories)")], { style: "Equation" }),
            p([code("retention_ema_t ← EWMA(retention_ema_{t−1},")], { style: "Equation" }),
            p([code("                       observed_retention, α = α_T(t))")], { style: "Equation" }),

            p([tr("Compute z-score relative to recent retention history:")]),

            p([code("last_w_ret ← tail(retention_history, w_ret(T))")], { style: "Equation" }),
            p([code("μ_ret ← mean(last_w_ret)")], { style: "Equation" }),
            p([code("σ_ret ← max(std(last_w_ret), 1.0)")], { style: "Equation" }),
            p([code("zscore_ret ← clamp((observed_retention − μ_ret) / σ_ret, −3, +3)")], { style: "Equation" }),

            p([tr("The target half-life incorporates feedback adjustment from the stability feedback mechanism (Section 7.3):")]),

            p([code("stability_adj ← ΔHalfLife_adj_t if provided else 0")], { style: "Equation" }),
            p([code("target_half_life_t ← clamp(base_half_life(T) ×")], { style: "Equation" }),
            p([code("                          (1 + 0.25 × zscore_ret + stability_adj),")], { style: "Equation" }),
            p([code("                          τ_min, τ_max)")], { style: "Equation" }),
            p([code("half_life_t ← EWMA(half_life_{t−1}, target_half_life_t,")], { style: "Equation" }),
            p([code("                   α = α_T(t))")], { style: "Equation" }),

            // ==================== 5. STRUCTURAL METRICS AND COMPOSITE SCORING ====================
            new Paragraph({ children: [new PageBreak()] }),
            p([tr("5. Structural Metrics and Composite Scoring")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("5.1 Embedding-Derived Metrics")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("5.1.1 Coherence")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Coherence measures integration of the current signal with context:")]),

            p([code("raw ← var([cos(x_t, c) for c in context_window])")], { style: "Equation" }),
            p([code("coherence_t ← 1 − clamp(raw, 0, 1)")], { style: "Equation" }),

            p([tr("High coherence (low variance in similarities) indicates the signal fits consistently with context. The effective Focus is modulated: F_eff = F × (0.5 + 0.5 × coherence_t).")]),

            p([tr("5.1.2 Focus Spread")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Focus spread quantifies the entropy of attention over nearest neighbors:")]),

            p([code("k ← k_neighbors(T) = round(lerp(8, 32, T))")], { style: "Equation" }),
            p([code("p ← softmax(kNN_similarities)")], { style: "Equation" }),
            p([code("focus_spread_t ← H(p) / ln(k)")], { style: "Equation" }),

            p([tr("Values near 1 indicate diffuse attention; values near 0 indicate concentrated attention. The effective Focus is further modulated: F_eff ← F_eff × (1 − focus_spread_t).")]),

            p([tr("5.1.3 Trajectory Drift")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Drift measures directional change in context centroids:")]),

            p([code("drift_vec_t ← l2_normalize(mean(ctx_t)) −")], { style: "Equation" }),
            p([code("              l2_normalize(mean(ctx_{t−k}))")], { style: "Equation" }),
            p([code("drift_mag_t ← ‖drift_vec_t‖")], { style: "Equation" }),

            p([tr("Since both centroids are unit-normalized, drift_mag_t ∈ [0, 2]. A threshold determines episode boundaries:")]),

            p([code("drift_threshold ← lerp(0.10, 0.35, T)")], { style: "Equation" }),
            p([code("if drift_mag_t > drift_threshold:")], { style: "Equation" }),
            p([code("    trigger_episode_boundary()")], { style: "Equation" }),

            p([tr("5.1.4 Embedding Prediction Error")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("We measure surprisal as the deviation of the current embedding from the predicted trajectory in latent space:")]),

            p([code("Δx_t = x_t − x_{t−1}")], { style: "Equation" }),
            p([code("Δx_trend_t = EWMA(Δx_trend_{t−1}, Δx_t, α=0.1)")], { style: "Equation" }),
            p([code("x_pred_t = x_{t−1} + Δx_trend_{t−1}")], { style: "Equation" }),
            p([code("prediction_error_t = 1 − cos(x_pred_t, x_t)")], { style: "Equation" }),

            p([tr("This error is normalized to produce the surprisal signal:")]),

            p([code("err_max = 0.5")], { style: "Equation" }),
            p([code("surprisal_t ← clamp(prediction_error_t / err_max, 0, 1)")], { style: "Equation" }),

            p([tr("This formulation captures purely kinematic surprise in the thought process")]),

            p([tr("5.2 Composite Score Computation")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The system computes 12 metrics that blend into a composite write score:")]),

            // Create metrics table
            new Table({
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
                            new TableCell({ borders: cellBorders, children: [p([code("cos(x, μ_ctx) × (0.5 + F)")])] })
                        ]
                    }),
                    new TableRow({
                        children: [
                            new TableCell({ borders: cellBorders, children: [p("Mismatch")] }),
                            new TableCell({ borders: cellBorders, children: [p("↓F, ↑S")] }),
                            new TableCell({ borders: cellBorders, children: [p([code("(1 − F) × S × novelty")])] })
                        ]
                    }),
                    new TableRow({
                        children: [
                            new TableCell({ borders: cellBorders, children: [p("Prediction Error")] }),
                            new TableCell({ borders: cellBorders, children: [p("↑S, ↓T")] }),
                            new TableCell({ borders: cellBorders, children: [p([code("surprisal_t × S × (1 − T)")])] })
                        ]
                    }),
                    new TableRow({
                        children: [
                            new TableCell({ borders: cellBorders, children: [p("Rarity")] }),
                            new TableCell({ borders: cellBorders, children: [p("↑F, ↓T")] }),
                            new TableCell({ borders: cellBorders, children: [p([code("(1 − μ_sim) × (0.5 + 0.5F) × (1 − 0.2T)")])] })
                        ]
                    }),
                    new TableRow({
                        children: [
                            new TableCell({ borders: cellBorders, children: [p("Drift")] }),
                            new TableCell({ borders: cellBorders, children: [p("↓T")] }),
                            new TableCell({ borders: cellBorders, children: [p([code("(drift_mag / 2) × (1 − T)")])] })
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
                            new TableCell({ borders: cellBorders, children: [p([code("(rarity + novelty) / 2 × (F + S) / 2")])] })
                        ]
                    }),
                    new TableRow({
                        children: [
                            new TableCell({ borders: cellBorders, children: [p("Valence")] }),
                            new TableCell({ borders: cellBorders, children: [p("S, ↓T")] }),
                            new TableCell({ borders: cellBorders, children: [p([code("map01(Σ p_c × v_map[c])")])] })
                        ]
                    }),
                    new TableRow({
                        children: [
                            new TableCell({ borders: cellBorders, children: [p("Arousal")] }),
                            new TableCell({ borders: cellBorders, children: [p("S, ↓T")] }),
                            new TableCell({ borders: cellBorders, children: [p([code("clamp(Σ p_c × a_map[c], 0, 1)")])] })
                        ]
                    })
                ]
            }),

            p([tr("Table 1: Metric definitions and knob dependencies. Arrows indicate direction of influence.")], { style: "Caption" }),

            p([tr("5.3 Metric Weight Blending")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Metric weights adapt online using recursive least squares (RLS) to minimize prediction error between composite scores and observed outcomes. Initial weights derive from bootstrap coefficients:")]),

            p([code("w_bootstrap[i] ← sigmoid(c_F[i]×F + c_S[i]×S + c_T[i]×T + d_i)")], { style: "Equation" }),

            p([tr("RLS fitting updates coefficients with stability-dependent forgetting:")]),

            p([code("φ(T) = 0.90 + 0.09T")], { style: "Equation" }),
            p([code("N ← round(lerp(64, 512, T))  # fitting window")], { style: "Equation" }),

            p([tr("The fitted weights blend with bootstrap weights based on RLS confidence:")]),

            p([code("τ_rls ← lerp(20.0, 80.0, T)")], { style: "Equation" }),
            p([code("confidence_rls ← 1 − exp(−t / τ_rls)")], { style: "Equation" }),
            p([code("weight_i(t) ← (1 − confidence_rls) × w_bootstrap[i] +")], { style: "Equation" }),
            p([code("               confidence_rls × w_rls[i]")], { style: "Equation" }),

            p([tr("5.3.1 Score Normalization")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Composite score computation requires careful normalization:")]),

            p([code("for each metric i:")], { style: "Equation" }),
            p([code("    m01[i] = map01(metric[i]) if signed else clamp(metric[i], 0, 1)")], { style: "Equation" }),
            p([code("weight_sum ← Σ weights[i]")], { style: "Equation" }),
            p([code("if weight_sum < ε: return 0")], { style: "Equation" }),
            p([code("weights_norm[i] ← weights[i] / weight_sum")], { style: "Equation" }),
            p([code("score ← clamp(Σ weights_norm[i] × m01[i], 0, 1)")], { style: "Equation" }),

            p([tr("Weight normalization is critical: with 12 metrics and raw weights averaging ~0.6, the sum approaches 7.2. Without normalization, weighted sums would saturate and collapse variance.")]),

            // ==================== 6. DYNAMIC THRESHOLDING ====================
            new Paragraph({ children: [new PageBreak()] }),
            p([tr("6. Dynamic Thresholding and Homeostatic Control")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("The write gate compares composite scores against an adaptive threshold θ_dynamic. This section details the threshold evolution algorithm incorporating Bayesian prior-evidence blending and homeostatic rate control.")]),

            p([tr("6.1 Prior-Evidence Blending")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The threshold prior derives from knob settings:")]),

            p([code("θ_prior(F, S, T) = lerp(0.10, 0.30, T) × (1 − 0.3S)")], { style: "Equation" }),

            p([tr("Observed evidence comes from the 90th percentile of recent scores:")]),

            p([code("w ← w_score(T)")], { style: "Equation" }),
            p([code("observed_p90 ← percentile(scores[t−w:t], 90)")], { style: "Equation" }),

            p([tr("Prior and evidence masses weight the blend:")]),

            p([code("ρ_prior ← prior_mass(T) = round(lerp(2, 32, T))")], { style: "Equation" }),
            p([code("ρ_obs ← u(t) × min(w, count)")], { style: "Equation" }),

            p([tr("The target threshold blends prior and evidence:")]),

            p([code("θ_target ← (ρ_prior × θ_prior + ρ_obs × observed_p90) /")], { style: "Equation" }),
            p([code("            max(ε, ρ_prior + ρ_obs)")], { style: "Equation" }),

            p([tr("High Stability increases prior mass, making the system more resistant to observed deviations. High uncertainty increases evidence mass, allowing faster adaptation to volatile conditions.")]),

            p([tr("6.2 Homeostatic Rate Control")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The controller maintains write rates near the target setpoint through continuous-time estimation with effective sample size (ESS) reliability weighting.")]),

            p([tr("6.2.1 Rate Estimation")], { heading: HeadingLevel.HEADING_3 }),

            p([code("Δt ← now − last_timestamp")], { style: "Equation" }),
            p([code("Δt ← max(Δt, 10⁻³)  # minimum 1ms")], { style: "Equation" }),
            p([code("α_dt ← 1 − exp(−Δt / 1.0)")], { style: "Equation" }),
            p([code("dt_ema ← (1 − α_dt) × dt_ema + α_dt × Δt")], { style: "Equation" }),
            p([code("dt_base ← max(dt_ema, 1.0)")], { style: "Equation" }),

            p([tr("The rate time constant scales with Stability:")]),

            p([code("τ_rate ← max(2^(3T) × dt_base, 1.0)")], { style: "Equation" }),
            p([code("α ← 1 − exp(−Δt / τ_rate)")], { style: "Equation" }),

            p([tr("Instantaneous rate estimation with bias correction:")]),

            p([code("ρ_inst ← (Δwrites / Δt) × 60  # writes per minute")], { style: "Equation" }),
            p([code("m_rate ← (1 − α) × m_rate + α × ρ_inst")], { style: "Equation" }),
            p([code("denom ← max(1 − (1 − α)^(rate_ticks + 1), ε)")], { style: "Equation" }),
            p([code("ρ_hat ← m_rate / denom  # bias-corrected estimate")], { style: "Equation" }),

            p([tr("6.2.2 Effective Sample Size")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("ESS estimates the effective number of independent samples in the EMA, using a heuristic inspired by Liu & Chen (1998):")]),

            p([code("β ← max(0, 1 − α)")], { style: "Equation" }),
            p([code("ESS ← min((1 + β) / max(1 − β, ε), 100)")], { style: "Equation" }),
            p([code("reliability ← 1 − exp(−ESS × (1 − T))")], { style: "Equation" }),

            p([tr("High Stability dampens reliability, preventing aggressive corrections in conservative regimes.")]),

            p([tr("6.2.3 Homeostatic Correction")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("The rate error drives threshold adjustment:")]),

            p([code("rate_error ← tanh((ρ_hat − rate_target_t) /")], { style: "Equation" }),
            p([code("                  max(rate_target_t, ε))")], { style: "Equation" }),
            p([code("κ_r = 0.10  # rate error gain")], { style: "Equation" }),
            p([code("cap_homeo ← 0.25 × hysteresis_t")], { style: "Equation" }),
            p([code("Δθ_homeo ← clamp(reliability × κ_r × (1 − T) ×")], { style: "Equation" }),
            p([code("                  (1 − maturity(t)) × rate_error,")], { style: "Equation" }),
            p([code("                  −cap_homeo, +cap_homeo)")], { style: "Equation" }),

            p([tr("The correction scales with reliability and is attenuated by both Stability and maturity, ensuring conservative, mature systems make minimal homeostatic adjustments.")]),

            p([tr("6.3 Threshold Integration")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("All threshold deltas combine and pass through safety limiting:")]),

            p([code("Δθ_total ← Δθ_sens + Δθ_homeo + Δθ_prec + Δθ_emo + Δθ_mood")], { style: "Equation" }),
            p([code("cap_total ← max_ΔT_per_min(t) × (Δt / 60.0)")], { style: "Equation" }),
            p([code("Δθ_limited ← clamp(Δθ_total, −cap_total, +cap_total)")], { style: "Equation" }),
            p([code("θ_dynamic_t ← clamp(EWMA(θ_dynamic_{t−1}, θ_target,")], { style: "Equation" }),
            p([code("                        α = α_T(t)) + Δθ_limited,")], { style: "Equation" }),
            p([code("                    T_min(t), T_max(t))")], { style: "Equation" }),

            p([tr("Hysteresis evolves toward the stability-derived base:")]),

            p([code("hysteresis_t ← clamp(EWMA(hysteresis_{t−1},")], { style: "Equation" }),
            p([code("                         base_band(T), α = α_T(t)),")], { style: "Equation" }),
            p([code("                    band_min, band_max)")], { style: "Equation" }),

            // ==================== 7. REINFORCEMENT AND DECAY ====================
            new Paragraph({ children: [new PageBreak()] }),
            p([tr("7. Reinforcement and Decay Dynamics")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("7.1 Memory Strength Model")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Each memory m maintains a strength value updated through use-frequency tracking and exponential decay:")]),

            p([code("use_frequency_t ← EWMA(use_frequency_{t−1},")], { style: "Equation" }),
            p([code("                       used_flag(m), α = α_S(t))")], { style: "Equation" }),
            p([code("λ_t ← ln(2) / half_life_t")], { style: "Equation" }),
            p([code("strength_t ← strength_{t−1} × exp(−λ_t × Δt) + S × use_frequency_t")], { style: "Equation" }),

            p([tr("Memories falling below the periphery cutoff are candidates for eviction:")]),

            p([code("periphery_cutoff(T) = lerp(0.05, 0.25, T)")], { style: "Equation" }),
            p([code("if strength_t < periphery_cutoff(T): evict(m)")], { style: "Equation" }),

            p([tr("7.2 Influence-Weighted Updates")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("When contextual gain signals are available, influence factors modulate reinforcement:")]),

            p([code("influence_factor ← (used_count / max(retrieved_count, 1)) ×")], { style: "Equation" }),
            p([code("                    clamp(contextual_gain(m), −1, +1)")], { style: "Equation" }),
            p([code("strength_t ← strength_{t−1} × exp(−λ_t × Δt) +")], { style: "Equation" }),
            p([code("              S × use_frequency_t + F × influence_factor")], { style: "Equation" }),

            p([tr("7.3 Causal Feedback Loop")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The system tracks causal influence of retrieved memories on generation quality through contextual gain—the improvement in prediction accuracy attributable to including each memory in context.")]),

            p([tr("7.3.1 Focus Feedback")], { heading: HeadingLevel.HEADING_3 }),

            p([code("αF_base = 0.10; βF_base = 0.05")], { style: "Equation" }),
            p([code("for each used memory m:")], { style: "Equation" }),
            p([code("    if contextual_gain(m) > 0:")], { style: "Equation" }),
            p([code("        weight_relevance_t += αF_base × contextual_gain(m)")], { style: "Equation" }),
            p([code("        attention_width_t *= (1 − βF_base)")], { style: "Equation" }),
            p([code("    else:")], { style: "Equation" }),
            p([code("        attention_width_t *= (1 + βF_base)")], { style: "Equation" }),
            p([code("weight_relevance_t ← clamp(weight_relevance_t, 0, 1)")], { style: "Equation" }),
            p([code("attention_width_t ← clamp(attention_width_t,")], { style: "Equation" }),
            p([code("                          attention_width_min, attention_width_max)")], { style: "Equation" }),

            p([tr("Positive contextual gain narrows attention and boosts relevance weighting; negative gain widens attention to explore alternatives.")]),

            p([tr("7.3.2 Sensitivity Feedback")], { heading: HeadingLevel.HEADING_3 }),

            p([code("η_base = 0.10")], { style: "Equation" }),
            p([code("for each used memory m:")], { style: "Equation" }),
            p([code("    novelty_reward ← 1 − sim(m.embedding, recent_context)")], { style: "Equation" }),
            p([code("    weight_novelty_t += η_base × (novelty_reward ×")], { style: "Equation" }),
            p([code("                         contextual_gain(m) −")], { style: "Equation" }),
            p([code("                         redundancy(m, recent_context))")], { style: "Equation" }),
            p([code("weight_novelty_t ← clamp(weight_novelty_t, 0, 1)")], { style: "Equation" }),

            p([tr("This rewards novelty that proves useful while penalizing redundant retrievals.")]),

            p([tr("7.3.3 Stability Feedback")], { heading: HeadingLevel.HEADING_3 }),

            p([code("γT_base = 0.05")], { style: "Equation" }),
            p([code("for each used memory m:")], { style: "Equation" }),
            p([code("    if contextual_gain(m) > 0:")], { style: "Equation" }),
            p([code("        stability(m) += γT_base")], { style: "Equation" }),
            p([code("    else:")], { style: "Equation" }),
            p([code("        stability(m) *= (1 − γT_base)")], { style: "Equation" }),

            p([tr("The mean stability of used memories provides adjustment to the half-life target:")]),

            p([code("adj ← clamp(mean(stability(m_used)) − 1.0, −0.25, +0.25)")], { style: "Equation" }),
            p([code("ΔHalfLife_adj_t ← adj")], { style: "Equation" }),

            p([tr("This factor is consumed by the Stability update (Section 4.3.2), avoiding conflicting adjustments between feedback mechanisms.")]),

            p([tr("7.4 Generation Influence Tracking")], { heading: HeadingLevel.HEADING_2 }),

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

            // ==================== 8. ADVANCED COGNITIVE PROCESSES ====================
            new Paragraph({ children: [new PageBreak()] }),
            p([tr("8. Advanced Cognitive Processes")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("This section presents algorithms modeling higher-order cognitive phenomena: working memory maintenance, metacognitive monitoring, reconsolidation dynamics, and serial position effects.")]),

            p([tr("8.1 Working Memory Gates")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Following Cowan's (2001) capacity constraints, working memory maintains a limited number of active items:")]),

            p([code("base_capacity = round(lerp(5, 3, S) + lerp(−1, 1, F))")], { style: "Equation" }),

            p([tr("This yields a range of approximately 2-6 slots, broadening the 4±1 chunk limit to accommodate task-dependent requirements. High Sensitivity reduces capacity (faster turnover), while high Focus modulates breadth.")]),

            p([tr("Maintenance incurs cognitive cost:")]),

            p([code("maintenance_cost_per_slot = lerp(0.05, 0.15, S)")], { style: "Equation" }),
            p([code("complexity_penalty = manifold_complexity × lerp(0.5, 1.5, S)")], { style: "Equation" }),

            p([tr("The manifold_complexity represents local variance in the embedding stream: 1 - mean(cos(window)).")]),

            p([tr("Gating thresholds determine entry and chunking:")]),

            p([code("chunking_threshold = lerp(0.7, 0.9, F)")], { style: "Equation" }),
            p([code("gate_threshold = lerp(0.1, 0.4, F)")], { style: "Equation" }),
            p([code("rehearsal_rate = lerp(0.5, 2.0, S)")], { style: "Equation" }),
            p([code("slot_dedication_strength = lerp(0.3, 0.9, T)")], { style: "Equation" }),

            p([tr("8.2 Metacognitive Monitoring")], { heading: HeadingLevel.HEADING_2 }),

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

            p([tr("8.3 Memory Reconsolidation")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Following Nader et al. (2000), retrieved memories enter a labile state permitting modification:")]),

            p([code("τ_labile = lerp(30, 300, T)  # seconds")], { style: "Equation" }),
            p([code("reconsolidation_gain = lerp(0.2, 0.02, T)")], { style: "Equation" }),
            p([code("lability_susceptibility = (1 − T) × (0.5 + 0.5 × S)")], { style: "Equation" }),

            p([tr("During the lability window, memories can drift toward current context:")]),

            p([code("drift_magnitude = (1 − T) × S × lability ×")], { style: "Equation" }),
            p([code("                   contextual_relevance")], { style: "Equation" }),

            p([tr("Reconsolidation effects propagate to semantically related memories with decay:")]),

            p([code("ripple_decay = lerp(0.5, 0.1, T)  # per semantic hop")], { style: "Equation" }),

            p([tr("8.4 Retrieval Competition")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Retrieved memories compete through lateral inhibition, modeling retrieval-induced forgetting (Anderson et al., 1994):")]),

            p([code("inhibition_radius = lerp(0.5, 0.85, F)")], { style: "Equation" }),
            p([code("winners_k = round(lerp(7, 3, F))")], { style: "Equation" }),
            p([code("suppression_per_retrieval = lerp(0.1, 0.01, T) ×")], { style: "Equation" }),
            p([code("                             (1 − winning_activation)")], { style: "Equation" }),
            p([code("recovery_time_RIF = lerp(300, 1800, T)  # seconds")], { style: "Equation" }),

            p([tr("High Focus produces narrow winner-take-all dynamics; low Focus permits broader activation.")]),

            p([tr("8.5 Predictive Pre-activation")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The system pre-activates memories predicted to be relevant based on trajectory extrapolation:")]),

            p([code("prediction_horizon = round(lerp(2, 8, F))")], { style: "Equation" }),
            p([code("pre_activation_decay = lerp(0.7, 0.3, T)")], { style: "Equation" }),
            p([code("prediction_conf_threshold = lerp(0.3, 0.7, F)")], { style: "Equation" }),
            p([code("surprise_sensitivity = S × lerp(2.0, 0.5, T)")], { style: "Equation" }),

            p([tr("When predictions fail (high surprise), the system updates its trajectory model:")]),

            p([code("update_rate_on_surprise = lerp(0.2, 0.02, T) × S")], { style: "Equation" }),

            p([tr("8.6 Serial Position Effects")], { heading: HeadingLevel.HEADING_2 }),

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

            p([tr("8.7 Emotional Consolidation")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("High-emotion events trigger enhanced consolidation, following McGaugh's (2004) findings:")]),

            p([code("θ_intensity = lerp(0.6, 0.8, 1 − S)")], { style: "Equation" }),
            p([code("θ_arousal = lerp(0.4, 0.2, S)")], { style: "Equation" }),
            p([code("trigger = (emotion_intensity_t ≥ θ_intensity) AND")], { style: "Equation" }),
            p([code("           (arousal_t ≥ θ_arousal)")], { style: "Equation" }),

            p([tr("Flashbulb memories receive extended half-life bonuses:")]),

            p([code("flashbulb_threshold = lerp(0.9, 0.4, S)")], { style: "Equation" }),
            p([code("emotional_half_life_bonus = exp(lerp(0, ln(3), S)) ×")], { style: "Equation" }),
            p([code("                             (1 + emotion_intensity_t)")], { style: "Equation" }),

            p([tr("Emotional tags cascade to related memories with decay:")]),

            p([code("cascade_radius = round(lerp(1, 5, S))")], { style: "Equation" }),
            p([code("cascade_decay = lerp(0.7, 0.3, S)")], { style: "Equation" }),

            // ==================== 9. CONSOLIDATION AND GRAPH INTEGRATION ====================
            new Paragraph({ children: [new PageBreak()] }),
            p([tr("9. Consolidation and Graph Integration")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("The consolidation system transforms episodic memories into semantic structures through clustering, summarization, and knowledge graph construction.")]),

            p([tr("9.1 Consolidation Triggers")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Consolidation activates under capacity, rate, or temporal conditions:")]),

            p([code("should_consolidate = (db_size > consolidation_threshold) OR")], { style: "Equation" }),
            p([code("                      (write_rate_t < rate_target_t / 2) OR")], { style: "Equation" }),
            p([code("                      (elapsed_time > consolidation_interval)")], { style: "Equation" }),

            p([tr("The consolidation rate adapts to Stability and Sensitivity:")]),

            p([code("rate_consolidate = (1 / max(consolidation_interval, 1)) ×")], { style: "Equation" }),
            p([code("                    (0.3 + 0.7T) × (1 − 0.5S)")], { style: "Equation" }),

            p([tr("9.1.1 Activity-Aware Scheduling")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Consolidation runs during idle periods and preempts for retrieval:")]),

            p([code("idle_required(T) = round(0.25 × w_rate_seconds(T))")], { style: "Equation" }),
            p([code("idle_for = now() − last_retrieval_ts")], { style: "Equation" }),
            p([code("should_start = (NOT is_processing_signal) AND")], { style: "Equation" }),
            p([code("                (retrieval_queue_depth == 0) AND")], { style: "Equation" }),
            p([code("                (idle_for ≥ idle_required(T))")], { style: "Equation" }),

            p([tr("On retrieval events, consolidation pauses, commits micro-batches, and resumes when idle.")]),

            p([tr("9.2 Consolidation Scoring")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Each memory receives a consolidation score determining merge priority:")]),

            p([code("score_consolidate(m) = w_s × strength(m) −")], { style: "Equation" }),
            p([code("                        w_r × redundancy(m) +")], { style: "Equation" }),
            p([code("                        w_c × connectivity(m) +")], { style: "Equation" }),
            p([code("                        w_t × stability(m)")], { style: "Equation" }),

            p([tr("Weights derive from knobs:")]),

            p([code("w_s = T; w_r = F; w_c = S; w_t = T")], { style: "Equation" }),

            p([tr("Low-scoring memories are marked for merging.")]),

            p([tr("9.3 Clustering and Summarization")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Marked memories cluster via density-based methods (e.g., DBSCAN) or k-means using embedding similarity:")]),

            p([code("cluster_i = {m_j | cos(m_j, μ_i) > merge_threshold}")], { style: "Equation" }),
            p([code("μ_i = centroid(cluster_i)")], { style: "Equation" }),

            p([tr("Summary nodes replace clusters:")]),

            p([code("summary.embedding = μ_i")], { style: "Equation" }),
            p([code("summary.text = summarize(fetch_blobs(cluster_i))")], { style: "Equation" }),
            p([code("summary.metadata.sources = [m.id for m in cluster_i]")], { style: "Equation" }),

            p([tr("9.4 Semantic Extraction")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("For sufficiently large clusters, semantic extraction identifies entities and relations:")]),

            p([code("extraction_batch_size = round(lerp(8, 32, T))")], { style: "Equation" }),
            p([code("min_cluster_size = round(lerp(3, 10, F))")], { style: "Equation" }),
            p([code("entity_frequency_threshold = round(lerp(5, 15, T))")], { style: "Equation" }),
            p([code("extraction_interval = lerp(300, 3600, T)  # 5 min → 1 hour")], { style: "Equation" }),
            p([code("max_extractions_per_cycle = round(lerp(20, 5, T))")], { style: "Equation" }),

            p([tr("Extraction uses structured prompting to identify named entities (people, places, organizations, concepts) and relationships (co-occurrence, implication, contradiction).")]),

            p([tr("9.5 Knowledge Graph Construction")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The graph comprises three node types:")]),

            p([bold("Memory Nodes: "), tr("Summarized embeddings from merged clusters")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("Entity Nodes: "), tr("Named entities extracted from text")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("Concept Nodes: "), tr("Emergent centroids representing recurrent topics")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([tr("Edge types capture relationships:")]),

            p([bold("co_occurs_with: "), tr("Shared context or temporal proximity")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("implies/causes: "), tr("Directional correlation in embedding drift")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("contradicts: "), tr("Strong negative similarity or schema violation")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("reinforces: "), tr("Frequent joint retrieval")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("derived_from: "), tr("Links summaries to source memories")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([tr("9.5.1 Edge Construction")], { heading: HeadingLevel.HEADING_3 }),

            p([tr("Co-occurrence edges derive from embedding similarity:")]),

            p([code("for (m_i, m_j) in cluster.sources:")], { style: "Equation" }),
            p([code("    cos_sim ← cos(m_i.embedding, m_j.embedding)")], { style: "Equation" }),
            p([code("    if cos_sim > lerp(0.85, 0.95, F):")], { style: "Equation" }),
            p([code("        create_edge(m_i, m_j, 'co_occurs_with', cos_sim)")], { style: "Equation" }),

            p([tr("Causal edges derive from temporal drift:")]),

            p([code("temporal_order ← sort_by_timestamp(cluster.sources)")], { style: "Equation" }),
            p([code("for i in range(len(temporal_order) − 1):")], { style: "Equation" }),
            p([code("    m_i, m_j ← temporal_order[i], temporal_order[i+1]")], { style: "Equation" }),
            p([code("    drift_vec ← m_j.embedding − m_i.embedding")], { style: "Equation" }),
            p([code("    drift_mag ← ‖drift_vec‖")], { style: "Equation" }),
            p([code("    if drift_mag > lerp(0.15, 0.35, T):")], { style: "Equation" }),
            p([code("        create_edge(m_i, m_j, 'causes', drift_mag)")], { style: "Equation" }),

            p([tr("9.6 Graph-Augmented Retrieval")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Retrieval combines vector similarity with graph expansion:")]),

            p([code("results_vec ← topK(vector_search(x_t, k=kNN_size))")], { style: "Equation" }),
            p([code("seed_nodes ← [r.id for r in results_vec]")], { style: "Equation" }),
            p([code("expanded_nodes ← graph.traverse(seed_nodes, depth=graph_depth)")], { style: "Equation" }),
            p([code("combined ← union(seed_nodes, expanded_nodes)")], { style: "Equation" }),
            p([code("re_ranked ← sort_by(cos(x_t, embeddings(combined)))")], { style: "Equation" }),

            p([tr("Graph expansion uses recursive traversal with depth limits to find related context that pure vector search might miss.")]),

            // ==================== 10. INTERRUPT GATE ====================
            new Paragraph({ children: [new PageBreak()] }),
            p([tr("10. Interrupt Gate and Streaming Integration")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("The interrupt gate controls when retrieved memories enter active context during streaming generation. The gate balances novelty value against disruption cost.")]),

            p([tr("10.1 Marginal Utility Computation")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Novelty thresholds scale with knobs and refractory state:")]),

            p([code("τ_novelty = lerp(0.10, 0.35, F) × (1 − 0.15S) × (1 + 0.3T)")], { style: "Equation" }),
            p([code("τ_mu = lerp(0.08, 0.18, F) × (1 − 0.4S) × (1 + 0.4T)")], { style: "Equation" }),
            p([code("retrieval_thresh(F) = lerp(0.25, 0.60, F)")], { style: "Equation" }),

            p([tr("Refractory dynamics suppress rapid successive interrupts:")]),

            p([code("Δ = cumulative_drift_since_last_interrupt")], { style: "Equation" }),
            p([code("τ_refrac = lerp(24, 96, T) × lerp(1.4, 1.0, S)")], { style: "Equation" }),
            p([code("k_refrac = lerp(0.20, 0.05, T) × lerp(0.8, 1.2, F)")], { style: "Equation" }),
            p([code("M_refrac = 1.0 + k_refrac × exp(−Δ / τ_refrac)")], { style: "Equation" }),

            p([tr("Effective thresholds incorporate refractory pressure:")]),

            p([code("τ_novelty_eff = τ_novelty × M_refrac")], { style: "Equation" }),
            p([code("τ_mu_eff = τ_mu × M_refrac")], { style: "Equation" }),

            p([tr("10.2 Marginal Utility Score")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The marginal utility (MU) of a candidate memory combines four factors:")]),

            p([code("w_cov = normalize(lerp(0.40, 0.60, F))  # coverage gain")], { style: "Equation" }),
            p([code("w_rel = normalize(lerp(0.35, 0.25, F))  # relevance")], { style: "Equation" }),
            p([code("w_red = normalize(lerp(0.15, 0.25, S))  # redundancy penalty")], { style: "Equation" }),
            p([code("w_coh = normalize(lerp(0.15, 0.25, S))  # incoherence penalty")], { style: "Equation" }),

            p([code("mu = w_cov × coverage_gain(candidate | included_set) +")], { style: "Equation" }),
            p([code("      w_rel × cos(candidate, ctx_centroid) −")], { style: "Equation" }),
            p([code("      w_red × redundancy(candidate, included_set) −")], { style: "Equation" }),
            p([code("      w_coh × (1 − coherence_t)")], { style: "Equation" }),

            p([tr("10.3 Gate Decision Logic")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Duplicate suppression threshold:")]),

            p([code("dup_thresh = lerp(0.88, 0.96, F) × (0.98 + 0.02T)")], { style: "Equation" }),
            p([code("K = round(lerp(10, 6, F))  # candidates to evaluate")], { style: "Equation" }),

            p([tr("Boundary-aware override permits lower-threshold interrupts at natural boundaries:")]),

            p([code("boundary_mult = lerp(1.3, 2.0, F) × lerp(1.1, 0.9, S)")], { style: "Equation" }),

            p([tr("The gate permits interrupt when:")]),

            p([code("embedding_novelty = 1 − max(cos(candidate, ctx_window))")], { style: "Equation" }),
            p([code("allow_interrupt = ")], { style: "Equation" }),
            p([code("    (max_relevance ≥ retrieval_thresh(F)) AND")], { style: "Equation" }),
            p([code("    (embedding_novelty ≥ τ_novelty_eff OR best_mu ≥ τ_mu_eff) AND")], { style: "Equation" }),
            p([code("    (max_semantic_overlap < dup_thresh) AND")], { style: "Equation" }),
            p([code("    (at_drift_boundary OR best_mu ≥ boundary_mult × τ_mu_eff)")], { style: "Equation" }),

            p([tr("This logic suppresses low-drift interrupts unless the marginal utility substantially exceeds threshold, while permitting normal-threshold interrupts at natural transition points.")]),

            p([tr("10.4 Streaming Pacing")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Streaming retrieval is gated by cumulative drift rate:")]),

            p([code("drift_accum += dist(x_t, x_{last_check})")], { style: "Equation" }),
            p([code("pacing_thresh(S) = lerp(0.5, 0.1, S)")], { style: "Equation" }),
            p([code("if drift_accum > pacing_thresh(S): trigger_check()")], { style: "Equation" }),

            p([code("max_wait_drift(F) = lerp(2.0, 0.5, F)")], { style: "Equation" }),
            p([code("max_results(F) = round(lerp(64, 4, F))")], { style: "Equation" }),
            p([code("adjacent_window(F) = round(lerp(8, 1, F))")], { style: "Equation" }),

            p([tr("High Sensitivity produces frequent checks triggered by small content shifts; high Focus enforces strict drift limits.")]),

            // ==================== 11. IMPLEMENTATION CONSIDERATIONS ====================
            new Paragraph({ children: [new PageBreak()] }),
            p([tr("11. Implementation Considerations")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("11.1 Computational Complexity")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Per-signal operations are dominated by embedding similarity computations. With n memories and d-dimensional embeddings:")]),

            p([tr("Exact kNN: O(nd) per query")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([tr("Approximate kNN (HNSW): O(d log n) per query")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([tr("Composite scoring: O(1) per signal (fixed 12 metrics)")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([tr("RLS weight update: O(m²) where m = 12 metrics")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([tr("For stores exceeding 100,000 memories, approximate nearest neighbor indices (HNSW, IVF-PQ) become essential. A small exact cache covering the most recent n_ctx(T) items handles recency-biased queries efficiently.")]),

            p([tr("11.2 Execution Cadence")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Operations partition by frequency:")]),

            p([bold("Per-signal: "), tr("Focus/Sensitivity/Stability updates, threshold evolution, strength decay, interrupt gating")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("Every K signals (K ≈ 3): "), tr("RLS weight fitting, heavy kNN computations, entropy estimation")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("Per-episode boundary: "), tr("Batch writes, cache invalidation, episode ID rollover")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("Periodic background: "), tr("Consolidation, ANN index maintenance, graph construction")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([tr("11.3 State Persistence")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("System state partitions across tables for efficient resumption:")]),

            p([bold("Processor State: "), tr("Single-row table storing all evolving parameters (maturity, uncertainty, threshold, hysteresis, learning rates)")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("Blender Weights: "), tr("12-element weight vector and RLS covariance matrix")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("Recent Context: "), tr("Rolling window of embedding vectors")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("Recent Scores: "), tr("Rolling window for threshold adaptation")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([tr("On startup, the system loads persisted state and resumes processing seamlessly.")]),

            p([tr("11.4 Database Considerations")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("The architecture assumes SQLite with vector extensions (sqlite-vec) for single-node deployment. Key optimizations:")]),

            p([tr("WAL mode for concurrent read/write access")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([tr("Write coalescing at episode boundaries to reduce fsync overhead")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([tr("Partial indices on high-cardinality columns (type, cluster_id)")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([tr("FTS5 virtual tables for hybrid text/vector retrieval")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([tr("For distributed deployment, the consolidation system's batch-oriented design maps naturally to periodic MapReduce-style jobs, while the streaming path remains latency-sensitive and benefits from in-memory caching.")]),

            // ==================== 12. CONCLUSION ====================
            new Paragraph({ children: [new PageBreak()] }),
            p([tr("12. Discussion and Conclusion")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("12.1 Summary of Contributions")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("This paper has presented Cortext, a three-knob cognitive memory architecture that achieves adaptive behavior through continuous parameter modulation rather than discrete mode switching. The key contributions are:")]),

            p([bold("Principled parameter derivation: "), tr("All system tunables trace to three primary knobs (Focus, Sensitivity, Stability) through explicit mathematical transformations, reducing reliance on fixed constants and providing interpretable control surfaces.")], { numbering: { reference: "numbered-list", level: 0 } }),

            p([bold("Self-calibrating priors: "), tr("The Bayesian prior-evidence blending mechanism allows the system to balance initial assumptions against accumulated experience, with the blend ratio itself governed by uncertainty estimation.")], { numbering: { reference: "numbered-list", level: 0 } }),

            p([bold("Homeostatic control: "), tr("The threshold controller maintains target write rates through continuous-time estimation with effective sample size reliability weighting, providing stable regulation across varying signal rates.")], { numbering: { reference: "numbered-list", level: 0 } }),

            p([bold("Cognitive fidelity: "), tr("The architecture incorporates established cognitive science findings—working memory capacity limits, reconsolidation dynamics, serial position effects, emotional modulation—within a computationally tractable framework.")], { numbering: { reference: "numbered-list", level: 0 } }),

            p([bold("Graph-augmented retrieval: "), tr("The consolidation system transforms episodic memories into semantic structures, enabling retrieval that combines embedding similarity with structural graph traversal.")], { numbering: { reference: "numbered-list", level: 0 } }),

            p([tr("12.2 Emergent Developmental Progression")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("A notable property of the architecture is that developmental phases emerge from parameter interactions rather than explicit programming:")]),

            p([bold("Early operation: "), tr("High uncertainty, light priors, wide safety bounds, rapid capture, permissive thresholds. The system behaves with high plasticity, quickly incorporating novel information.")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([bold("Intermediate operation: "), tr("Balanced learning, stabilizing weights, selective attention. The system becomes more discriminating while retaining adaptability.")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([bold("Mature operation: "), tr("Strong priors, narrow bounds, slow adaptation, high precision. The system exhibits expert-like behavior with reliable, stable retrieval.")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([tr("These transitions arise naturally from the annealing of safety bounds (T_min, T_max, max_ΔT_per_min) and the accumulation of experiential mass (ρ_obs vs ρ_prior), without requiring explicit phase detection or switching logic.")]),

            p([tr("12.3 Limitations")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Several limitations warrant acknowledgment:")]),

            p([bold("Embedding dependence: "), tr("System behavior depends critically on embedding quality. Poor embeddings will produce poor coherence, novelty, and retrieval signals regardless of knob settings.")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([bold("Single-agent assumption: "), tr("The current architecture assumes single-user operation. Multi-agent or collaborative scenarios would require extensions for shared memory spaces and conflict resolution.")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([bold("Offline evaluation: "), tr("While the algorithms are fully specified, empirical validation on diverse task domains remains ongoing work.")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([bold("Extraction latency: "), tr("Semantic extraction for graph construction introduces latency during consolidation. The background scheduling mitigates but does not eliminate this cost.")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([tr("12.4 Future Directions")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Several directions merit further investigation:")]),

            p([bold("Meta-learning knob adaptation: "), tr("Learning optimal knob settings for specific task distributions through reinforcement or evolutionary optimization.")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([bold("Multimodal integration: "), tr("Extending the architecture to handle heterogeneous modalities (text, image, audio) through unified embedding spaces or modality-specific sub-systems.")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([bold("Distributed deployment: "), tr("Scaling the architecture across multiple nodes while maintaining consistency guarantees and low-latency retrieval.")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([bold("Prosthetic applications: "), tr("Adapting the architecture for assistive technology applications, particularly memory augmentation for individuals with cognitive impairment.")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([tr("12.5 Conclusion")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("Cortext demonstrates that sophisticated adaptive memory behavior can emerge from a small set of principled control parameters. By grounding all system dynamics in three interpretable knobs—Focus, Sensitivity, and Stability—the architecture provides both theoretical clarity and practical tunability. The integration of cognitive science findings with modern embedding-based retrieval and knowledge graph construction offers a path toward AI systems with more human-like memory characteristics.")]),

            p([tr("The formal specification provided here enables direct implementation while the modular design permits selective adoption of individual components. We hope this work contributes to the broader goal of building AI systems that learn and remember in ways that align with human cognitive architecture.")]),

            // ==================== REFERENCES ====================
            new Paragraph({ children: [new PageBreak()] }),
            p([tr("References")], { heading: HeadingLevel.HEADING_1 }),

            p([tr("Anderson, M. C., Bjork, R. A., & Bjork, E. L. (1994). Remembering can cause forgetting: Retrieval dynamics in long-term memory. "), italic("Journal of Experimental Psychology: Learning, Memory, and Cognition, 20"), tr("(5), 1063-1087.")], { spacing: { after: 120 } }),

            p([tr("Åström, K. J., & Murray, R. M. (2008). "), italic("Feedback systems: An introduction for scientists and engineers"), tr(". Princeton University Press.")], { spacing: { after: 120 } }),

            p([tr("Baddeley, A. (2000). The episodic buffer: A new component of working memory? "), italic("Trends in Cognitive Sciences, 4"), tr("(11), 417-423.")], { spacing: { after: 120 } }),

            p([tr("Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. "), italic("Behavioral and Brain Sciences, 24"), tr("(1), 87-114.")], { spacing: { after: 120 } }),

            p([tr("Cowan, N. (2010). The magical mystery four: How is working memory capacity limited, and why? "), italic("Current Directions in Psychological Science, 19"), tr("(1), 51-57.")], { spacing: { after: 120 } }),

            p([tr("Hart, J. T. (1965). Memory and the feeling-of-knowing experience. "), italic("Journal of Educational Psychology, 56"), tr("(4), 208-216.")], { spacing: { after: 120 } }),

            p([tr("Hunt, R. R. (1995). The subtlety of distinctiveness: What von Restorff really did. "), italic("Psychonomic Bulletin & Review, 2"), tr("(1), 105-112.")], { spacing: { after: 120 } }),

            p([tr("LaBar, K. S., & Cabeza, R. (2006). Cognitive neuroscience of emotional memory. "), italic("Nature Reviews Neuroscience, 7"), tr("(1), 54-64.")], { spacing: { after: 120 } }),

            p([tr("Liu, J. S., & Chen, R. (1998). Sequential Monte Carlo methods for dynamic systems. "), italic("Journal of the American Statistical Association, 93"), tr("(443), 1032-1044.")], { spacing: { after: 120 } }),

            p([tr("McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. "), italic("Psychological Review, 102"), tr("(3), 419-457.")], { spacing: { after: 120 } }),

            p([tr("McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. "), italic("Psychology of Learning and Motivation, 24"), tr(", 109-165.")], { spacing: { after: 120 } }),

            p([tr("McGaugh, J. L. (2004). The amygdala modulates the consolidation of memories of emotionally arousing experiences. "), italic("Annual Review of Neuroscience, 27"), tr(", 1-28.")], { spacing: { after: 120 } }),

            p([tr("Miller, G. A. (1956). The magical number seven, plus or minus two: Some limits on our capacity for processing information. "), italic("Psychological Review, 63"), tr("(2), 81-97.")], { spacing: { after: 120 } }),

            p([tr("Murdock, B. B. (1962). The serial position effect of free recall. "), italic("Journal of Experimental Psychology, 64"), tr("(5), 482-488.")], { spacing: { after: 120 } }),

            p([tr("Nader, K. (2003). Memory traces unbound. "), italic("Trends in Neurosciences, 26"), tr("(2), 65-72.")], { spacing: { after: 120 } }),

            p([tr("Nader, K., Schafe, G. E., & Le Doux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. "), italic("Nature, 406"), tr("(6797), 722-726.")], { spacing: { after: 120 } }),

            p([tr("Russell, J. A. (1980). A circumplex model of affect. "), italic("Journal of Personality and Social Psychology, 39"), tr("(6), 1161-1178.")], { spacing: { after: 120 } }),

            p([tr("Tulving, E. (1972). Episodic and semantic memory. In E. Tulving & W. Donaldson (Eds.), "), italic("Organization of memory"), tr(" (pp. 381-403). Academic Press.")], { spacing: { after: 120 } })
        ]
    }]
});

// Write the document
Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync("./paper.docx", buffer);
    console.log("Document created successfully: cortext_paper.docx");
}).catch(err => {
    console.error("Error creating document:", err);
});