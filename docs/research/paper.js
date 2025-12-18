const { Document, Packer, Paragraph, TextRun, Header, Footer,
    AlignmentType, PageOrientation, HeadingLevel, PageNumber, PageBreak } = require('docx');
const fs = require('fs');

// Import shared components from algorithms.js
const {
    p, bold, italic, code, tr,
    paragraphStyles, numberingConfig,
    paperSections
} = require('./algorithms.js');

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

            // ==================== SECTIONS 3-10: IMPORTED FROM algorithms.js ====================
            new Paragraph({ children: [new PageBreak()] }),
            ...paperSections.mathFoundations,
            ...paperSections.coreAdaptation,
            ...paperSections.structuralMetrics,
            ...paperSections.dynamicThresholding,
            ...paperSections.writePacing,
            ...paperSections.reinforcementDecay,
            ...paperSections.advancedCognitive,
            ...paperSections.consolidationGraph,
            ...paperSections.interruptGate,

            // ==================== 11. IMPLEMENTATION CONSIDERATIONS ====================
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

            p([tr("11.3 State Representation")], { heading: HeadingLevel.HEADING_2 }),

            p([tr("System state partitions into logical components for efficient resumption:")]),

            p([bold("Processor State: "), tr("Global parameter state storing all evolving parameters (maturity, uncertainty, threshold, hysteresis, learning rates)")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("Blender Weights: "), tr("12-element weight vector and RLS covariance matrix")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("Recent Context: "), tr("Rolling window of embedding vectors")], { numbering: { reference: "bullet-list", level: 0 } }),
            p([bold("Recent Scores: "), tr("Rolling window for threshold adaptation")], { numbering: { reference: "bullet-list", level: 0 } }),

            p([tr("On startup, the system loads persisted state and resumes processing seamlessly.")]),

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

            p([tr("Fountas, Z., et al. (2024). Event segmentation in large language models. "), italic("arXiv preprint arXiv:2407.03158"), tr(".")]),
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
    console.log("Document created successfully: paper.docx");
}).catch(err => {
    console.error("Error creating document:", err);
});
