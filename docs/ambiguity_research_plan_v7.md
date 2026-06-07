# Cortext Ambiguity Retrieval Research Plan

## Next direction: Temporal-Context Candidate Inclusion

This document proposes the next experiment after the failed v6 Cue-Bound Episodic Reinstatement probe.

The decision is to stop testing anchor-only candidate inclusion. SC-Rerank proved that source continuity can improve ranking when the correct memory is already present. CBPC proved that bounded graph-based candidate inclusion can add missing candidates, but the source-tail and episode-edge path produced a remote-intrusion regression. CBER avoided that regression by becoming unreachable: it fired zero times across the benchmark and therefore did not address the missing-candidate failure.

The next research direction should test whether Cortext can recover missing candidates through temporal-context cueing, not source-tail fanout and not text-level reference detection. The proposed mechanism is Temporal-Context Candidate Inclusion, abbreviated TCCI.

TCCI is intended to answer one narrow question:

Can a bounded lookup over stored temporal context states propose the missing grounding memory when the current content embedding is weak, while source continuity, event-boundary state, and multi-context competition suppress remote intrusions?

This is a human-memory proposal, not an LLM-retrieval proposal. It treats Cortext’s stored temporal context vector as the computational analogue of contextual reinstatement in episodic memory. The core idea is that an underspecified current cue may fail as a content cue but still be embedded in an ongoing temporal, source, and episode context that can cue the relevant prior episode.

## Human-memory grounding

The relevant human-memory principle is not query expansion. It is cue-dependent episodic retrieval.

Encoding specificity says that a retrieval cue is effective when it overlaps with information encoded with the episode. For Cortext, the important implication is that the cue should not be limited to the current content embedding. The cue can include temporal context, source provenance, and episode state, because these are part of the episode as experienced.

The temporal context model gives a more operational analogue. It represents context as a gradually drifting state associated with studied items. At retrieval, the current state of context can activate recent or contiguous prior memories even when the item-level cue is weak. Cortext already stores a temporal context vector with memories and combines content match with context match during retrieval scoring. The missing-candidate problem arises because context match can only help after a candidate is reachable.

Event segmentation research supports using boundary state as a hard guard. Human long-term associative memory is stronger within event boundaries than across them. Therefore, context candidate inclusion should be easier inside the same ongoing episode and much harder after a topic shift or boundary crossing.

Source monitoring supports using source continuity as a provenance constraint, not as a source-tail fanout mechanism. Source should help decide whether a candidate belongs to the same stream of experience. It should not, by itself, propose arbitrary recent memories from that source.

Working-memory research supports boundedness. Humans do not keep an unbounded set of active referents. TCCI should propose at most one default internal candidate and at most two in a diagnostic sweep.

Pattern separation and completion support the paired requirement: partial cues can complete an episode, but similar lures must remain separated. Therefore, the experiment must measure not only recall lift but also lure intrusion, multiple-context ambiguity, and distractor capture.

## 1. Problem statement

TCCI targets a stricter missing-candidate failure than SC-Rerank:

A new signal is weak or underspecified by content embedding alone. The relevant grounding memory is absent from the baseline selected candidate set. However, the grounding memory belongs to the same recent source and ongoing episode, and its stored temporal context is close to the current temporal context state.

This is the case CBER did not reach. CBER depended on retrieved or working-memory anchors, so if the right memory never appeared in retrieval and no admissible anchor formed, the mechanism had no causal path.

TCCI should solve only the following subclass:

- baseline retrieval plus SC-Rerank does not include the target memory;
- the target memory has high temporal-context similarity to the current context state;
- source continuity is present;
- no strong event-boundary or topic-shift signal blocks continuity;
- the current content embedding is weak, flat, or ambiguous enough that content-only retrieval is underpowered;
- remote distractors exist and must remain suppressed.

TCCI is not intended to solve cases where the target is remote, source-discontinuous, across a clear boundary, not represented in memory, or only identifiable through text-specific reference words.

## 2. Mechanism hypothesis

Falsifiable hypothesis:

When baseline content retrieval fails to include the correct grounding memory, a bounded temporal-context lookup over stored memory context states will recover a measurable subset of missing candidates across modalities, provided that candidate proposal is gated by source continuity, event-boundary state, and multi-context competition. This will improve context-dependent reference hit@k without increasing remote intrusion, final returned count, or graph retrieval latency beyond the promotion gate.

The hypothesis predicts an intermediate causal path before final metric lift. If that path does not appear, the mechanism should be rejected even if aggregate metrics are noisy.

The mechanism is modality-general because it operates after all modalities have already been converted into Cortext’s common runtime state:

- current embedding;
- current temporal context vector;
- accumulator and uncertainty state;
- source id and source continuity;
- episode and boundary state;
- baseline retrieved candidates and score distribution;
- stored memory embeddings and stored memory temporal contexts;
- source confidence and memory metadata;
- graph edges;
- F/S/T-derived thresholds.

The mechanism must not read tokens, pronouns, transcripts, OCR text, modality labels, or durable semantic labels at runtime.

## 3. Proposed mechanism

TCCI should be tested as a post-initial-retrieval, pre-final-selection candidate inclusion probe inside `GraphAugmentedRetrieveCandidates`.

The runtime shape is:

1. Run baseline retrieval first.
2. Score the baseline candidate distribution for content underreach.
3. If underreach is present, test source continuity and boundary continuity.
4. If continuity is present, perform one bounded temporal-context candidate lookup over stored memory context states.
5. Filter proposed candidates through existing retrieval filters.
6. Apply multi-context competition. If several plausible contexts are tied, abstain.
7. Add at most one internal candidate by default. A max-two setting is diagnostic only.
8. Run the existing final selection and SC-Rerank over a fixed-size candidate set.
9. Return exactly the same final count as baseline.

The core distinction from rejected probes is important:

- It is not text reference detection.
- It is not recent-context centroid seeding.
- It is not a source-tail fanout.
- It is not episode-edge fanout by default.
- It does not broaden returned candidate count.
- It does not write durable labels, facts, or summaries.
- It does not add extraction or summarization.

### Content-underreach gate

The gate should identify situations where content retrieval looks underpowered, not merely low ranked.

Use only modality-general retrieval statistics:

- low top-1 content score;
- small top-1 versus top-2 margin;
- high retrieval-score entropy or flatness;
- high focus spread;
- low agreement among top candidates;
- current accumulator uncertainty;
- absence of a high-confidence same-source continuation among baseline candidates.

The experiment should include ablations that remove this gate. A useful gate should increase firing on missing-candidate positives and reduce firing on remote-intrusion negatives. If the gate does not separate those slices, it is not load-bearing.

### Continuity gate

Candidate inclusion should require continuity evidence independent of content similarity:

- source id continuity is present;
- current source has not changed abruptly;
- event-boundary state does not indicate a topic shift;
- current temporal context state is stable enough to use;
- working-memory context is not split among multiple active contexts.

Source continuity is necessary but not sufficient. CBPC already showed that source-tail and episode-edge inclusion can add missing candidates, but with intrusion risk. TCCI should treat source continuity as a guard around temporal-context lookup, not as a source-tail proposal path.

### Temporal-context lookup

The proposal path should query candidate memories by stored temporal context, not by current content embedding.

Conceptually:

- query: current temporal context vector, `c_t`;
- index: stored `memory.context` values;
- restriction: same source first, with a diagnostic cross-source condition only for negative controls;
- budget: top 1 by default, top 2 in a bounded diagnostic sweep;
- normal filters still apply: write exclusion, working-memory overlap, source confidence, duplicate suppression, memory kind eligibility, and graph safety.

The experiment may evaluate two low-latency variants:

1. In-memory context-neighbor cache. Store only recent memory IDs and their context vectors in temporary runtime state. This minimizes latency but may miss older same-episode targets.
2. Bounded context-index lookup. Add one bounded lookup over stored context vectors. This is more likely to recover missing candidates but must pass the latency gate.

The first promotion candidate should be the lower-latency variant if it has enough reachability. If it does not, test the bounded context-index lookup. Do not promote either variant if the gain comes from broad fanout.

### Multi-context competition

TCCI should abstain when temporal context is not specific.

Abstain if:

- the top two temporal-context candidates are near-tied but point to different episodes;
- the top candidates come from different active sources;
- the current boundary state indicates a topic shift;
- the proposed candidate’s content embedding is strongly inconsistent with the current embedding;
- a remote distractor has better temporal-context support than the same-source candidate;
- the source confidence of the proposed candidate is below the existing source-confidence gate.

This is the pattern-separation side of the mechanism. The goal is not to make recall maximally aggressive. The goal is to recover only when context is specific enough.

### F/S/T effects

F/S/T should affect thresholds, not create modes.

Focus should tighten inclusion. Higher Focus should require stronger context specificity, smaller candidate budgets, and larger separation between the top temporal-context candidate and the nearest lure.

Sensitivity should make the underreach gate more responsive. Higher Sensitivity may allow the mechanism to fire on weaker content evidence, but should not relax source-confidence or boundary-shift rejection.

Stability should lengthen the usable temporal-context horizon and strengthen boundary conservatism. Higher Stability can preserve longer same-episode context, but it should also make cross-boundary inclusion harder.

## 4. Observable causal path

Every run must log the causal chain below. Aggregate hit@k is not enough.

| Stage | Required observation | Failure meaning |
|---|---|---|
| 1. Content underreach detected | Baseline retrieval distribution is weak, flat, or ambiguous on the positive slice | If absent, the slice is not testing the target failure |
| 2. Continuity gate passes | Same-source and no-boundary evidence is present | If absent, TCCI should abstain |
| 3. Temporal-context cue forms | Current `c_t` is available and stable | If absent, the context-state path is unreachable |
| 4. Context candidate proposed | Target memory or target episode appears in the bounded context lookup | If absent, the mechanism cannot solve missing candidates |
| 5. Candidate passes filters | Candidate survives normal retrieval filters | If absent, failure is due to eligibility, not cueing |
| 6. Candidate enters internal pool | Candidate is added before final selection, without increasing final returned count | If absent, inclusion is not actually happening |
| 7. Candidate survives final selection | Candidate remains in top-k after final selection and SC-Rerank | If absent, ranking or selection is the next bottleneck |
| 8. Final metric improves | Reference hit@k or hit@1 improves on the target slice | If absent, the added candidate is not useful |

Promotion requires evidence at stages 4 through 8. A mechanism that only improves final metrics through candidate-count inflation fails. A mechanism that fires but never proposes the target also fails.

## 5. Required benchmark slice

The current 103-scenario benchmark is useful as a guard, but it is not enough. CBER firing zero times shows that the existing benchmark does not guarantee reachability for anchor-based mechanisms. The next benchmark must include an explicitly labeled TCCI-reachable slice.

A TCCI-reachable positive case must satisfy all of the following offline annotations:

- target absent from the baseline selected candidate set;
- target belongs to the same source or same continuous source stream;
- target is inside the same episode or adjacent no-boundary continuation;
- target has high stored-context similarity to the current temporal context;
- target has weak or insufficient current-content similarity, explaining why content retrieval missed it;
- at least one plausible distractor exists;
- runtime selection does not use the offline annotation.

Offline selection may use text metadata, transcripts, object labels, human annotation, or dataset labels to identify cases. Runtime evaluation must hide those features from the mechanism and expose only Cortext state.

### Positive scenario families

Context-dependent continuation:

A prior signal establishes a person, object, place, task, or event. A later signal is underspecified by content embedding but occurs inside the same source and episode context. Success means the target memory is absent from baseline but proposed by TCCI and survives final selection. Failure means no proposal, wrong-source proposal, or target proposed only by broad fanout.

Object/event continuation:

A visual or video frame continues a prior scene or object interaction but has sparse standalone semantic content. Success means the prior object/event memory is recovered through temporal context, not object-label parsing. Failure means recovery requires labels, OCR, or modality-specific visual branches.

Audio continuation:

A short tone, emotional vocalization, or low-information utterance continues a prior audio or audio-visual event. Success means source and temporal context recover the prior event. Failure means the mechanism only works for text dialogue.

Sensor or agent-state continuation:

A task state, location state, or agent loop continues a recent activity with weak current embedding. Success means same-place or same-task continuity is recovered through existing provenance and episode state. Failure means candidate inclusion collapses into generic recency.

Multiple plausible active contexts:

Two recent contexts are both plausible. Success means TCCI abstains unless temporal-context separation is clear. Failure means it chooses one simply because it is recent or same-source.

### Negative controls

Remote intrusion with distractors:

A weak cue occurs in a context that resembles an older memory, but source continuity or boundary continuity is absent. Success means no TCCI candidate is added. Failure means the old distractor is proposed.

Topic shift:

A low-information signal follows a clear boundary. Success means TCCI abstains. Failure means it drags the prior topic into the new episode.

Same-source lure:

A memory from the same source is recent but belongs to a different event. Success means temporal-context and boundary gates reject it. Failure means source continuity alone causes inclusion.

Continuous remote distractor:

A remote distractor has strong source continuity but weaker context match than the target. Success means the context cue separates target from distractor. Failure means source-tail behavior reappears.

Non-referential low-information input:

A short filler, silence-like embedding, generic sound, or ambiguous frame has no intended prior reference. Success means no candidate inclusion. Failure means the system invents a grounding memory.

Non-text guard:

Audio, image/video, or sensor cases must include both positive continuations and negative controls. Success means the mechanism fires on reachable non-text positives and abstains on non-text negatives. Failure means gains are text-only.

## 6. Experiment design

Run the experiments in four stages.

### Stage A: Reachability audit

Before changing retrieval output, run a shadow-only audit.

For every scenario, log:

- whether target is absent from baseline;
- whether current content retrieval is weak or flat;
- whether source continuity is present;
- whether boundary continuity is present;
- whether current temporal context is available;
- whether a bounded context lookup would propose the target;
- whether proposed candidates would pass normal filters;
- whether multiple-context competition would abstain.

Do not change returned candidates in Stage A.

Pass condition for moving to Stage B:

- at least 30 positive benchmark cases are target-absent and TCCI-reachable, or the dataset must be expanded;
- temporal-context lookup proposes the target or target episode in at least 25 percent of target-absent positives in shadow mode;
- shadow proposals on negative controls remain at or below 5 percent.

Reject or revise if this stage fires zero times. Do not repeat CBER’s failure mode.

### Stage B: Candidate proposal without graph expansion

Enable TCCI candidate inclusion, but do not expand from the context-proposed candidate through graph fanout. This isolates the temporal-context cue.

Pass condition:

- strict candidate inclusion increases over baseline on the reachable positive slice;
- no remote-intrusion guard receives a new target-intruding candidate;
- final returned count remains fixed.

If Stage B works, graph expansion may not be necessary.

### Stage C: Candidate proposal with one-hop episode support

Allow only one bounded episode-support operation after a context hit:

- same memory;
- same episode;
- `prev_in_episode`, `next_in_episode`, or `within_same_event` edge;
- no generic source-tail expansion;
- no label expansion;
- no associative-cue fanout unless the cue directly derives from the proposed memory.

Pass condition:

- additional strict inclusions exceed Stage B;
- intrusion remains unchanged;
- latency remains inside the gate.

If one-hop episode support adds intrusion, remove it and keep the pure temporal-context proposal.

### Stage D: Full final-selection integration

Run the best Stage B or C variant with final selection and SC-Rerank.

The mechanism succeeds only if the target not only appears internally but survives the fixed final candidate count.

## 7. Ablation matrix

Run all ablations on the same benchmark split and the same F/S/T profiles.

| Condition | Purpose | Expected result if TCCI is causal |
|---|---|---|
| Baseline retrieval + SC-Rerank | Current best safe path | No candidate inclusion for target-absent cases |
| Full TCCI | Main test | Strict inclusion and final hit@k lift on reachable positives |
| Temporal-context lookup disabled | Tests core cue | Inclusion lift collapses |
| Content-underreach gate disabled | Tests firing control | More negative-control firing, possible intrusion |
| Source-continuity gate disabled | Tests provenance guard | Remote or cross-source intrusions increase |
| Boundary gate disabled | Tests event segmentation guard | Topic-shift intrusions increase |
| Multi-context competition disabled | Tests pattern separation | Multiple-active-context errors increase |
| Current-embedding consistency veto disabled | Tests lure suppression | Intrusion may rise; if no effect, veto is not load-bearing |
| One-hop episode support disabled | Tests whether graph edges matter | If pure context works, little loss; if edge support matters, inclusion falls |
| Source-tail proposal only | Negative comparison to CBPC | Should not outperform TCCI without intrusion |
| Recent-centroid query blend | Negative comparison to rejected context seeding | Should not be the source of the gain |
| Final-count fixed disabled | Diagnostic only | Any gain here does not count for promotion |
| Max internal candidates = 1 | Default budget | Should preserve precision |
| Max internal candidates = 2 | Diagnostic budget | May raise inclusion; must not raise intrusion |
| F/S/T grid | Robustness | Effects should be stable and monotonic enough to explain |

### F/S/T sweeps

Use at least the following profiles:

- midpoint: F=0.5, S=0.5, T=0.5;
- high Focus: F=0.8, S=0.5, T=0.5;
- low Focus: F=0.3, S=0.5, T=0.5;
- high Sensitivity: F=0.5, S=0.8, T=0.5;
- low Sensitivity: F=0.5, S=0.3, T=0.5;
- high Stability: F=0.5, S=0.5, T=0.8;
- low Stability: F=0.5, S=0.5, T=0.3;
- full 3 x 3 x 3 grid on the smaller diagnostic slice.

Promotion should not depend on a single cherry-picked knob profile.

## 8. Metrics and promotion gates

Report metrics separately for:

- full benchmark;
- original 48 reference cases;
- original 48 remote-intrusion cases;
- original 7 non-text guards;
- new TCCI-reachable baseline-missing positives;
- new non-text continuation positives;
- new multiple-context negatives;
- new non-referential low-information negatives.

### Required intermediate metrics

- TCCI fire rate.
- Underreach-gate pass rate.
- Continuity-gate pass rate.
- Boundary-gate pass rate.
- Temporal-context proposal count.
- Target proposed count.
- Target proposed rate among baseline-missing positives.
- Proposed candidate pass-filter rate.
- Proposed candidate final-survival rate.
- Multi-context abstention rate.
- Negative-control proposal rate.
- Additional internal candidates per scenario.
- Final returned count.
- Mean and p95 graph retrieval latency.
- Mean and p95 TCCI overhead on fired cases.

### Promotion gates

Promote only if all gates pass.

Reachability gate:

- At least 30 target-absent, TCCI-reachable positive cases are present, or the benchmark is too small to decide.
- TCCI fires on at least 40 percent of reachable positives.
- TCCI fires on at most 5 percent of remote, topic-shift, same-source-lure, and non-referential negatives.
- Target proposal rate among target-absent positives improves by at least 10 absolute points over baseline.

Reference lift gate:

- On the TCCI-reachable baseline-missing slice, final hit@k improves by at least +0.05 absolute over baseline + SC-Rerank.
- On the same slice, strict candidate inclusion improves by at least +0.10 absolute.
- Overall reference hit@k must not regress.
- Overall hit@1 and hit@3 should improve or remain within 0.01 absolute of baseline + SC-Rerank.

Remote-intrusion gate:

- On the existing 48 remote-intrusion guards, allow zero additional remote intrusions. One extra intrusion is already about +0.0208 absolute and should fail the gate.
- On an expanded remote-intrusion set of at least 200 cases, absolute intrusion regression must be less than or equal to +0.005.
- Same-source lure intrusion must not increase.

Candidate-count gate:

- Final returned count must be identical to baseline for each scenario, not just equal on average.
- Internal proposed candidates must be bounded: max 1 in the promotion candidate, max 2 only in diagnostics.
- Any improvement observed only when final-count-fixed is disabled is invalid.

Precision gate:

- Top-k precision must improve or remain stable on the full benchmark.
- TCCI-proposed candidates must not reduce precision through low-value additions that replace useful baseline candidates.

Non-text gate:

- Non-text guard hit rate must not regress.
- TCCI must demonstrate at least some strict inclusion on reachable non-text continuation positives, or the mechanism remains text-corpus-only and should not be promoted.

Latency gate:

- Mean added overhead should be less than or equal to 0.5 ms on all scenarios.
- Fired-case p95 overhead should be less than or equal to 2 ms.
- Overall graph retrieval mean should not regress by more than 5 percent or 1 ms, whichever is smaller, relative to baseline + SC-Rerank on the same build.
- If the bounded context-index lookup cannot meet this gate, revise toward an in-memory context-neighbor cache.

F/S/T robustness gate:

- The mechanism should pass at midpoint settings.
- High Focus should be more conservative, not more intrusive.
- High Sensitivity may increase firing but must not increase intrusion.
- High Stability may increase context reach but must not cross clear event boundaries.

## 9. Interpreting mixed results

If context candidates are proposed but do not survive final selection, the cueing mechanism may be valid but the final selection policy is suppressing context candidates. Do not promote yet. Run a ranking-focused follow-up that leaves candidate inclusion unchanged.

If target proposal improves but final hit@k does not, inspect whether the added candidate replaces a useful baseline candidate. This would indicate candidate competition, not reachability, is the bottleneck.

If final hit@k improves but strict candidate inclusion does not, the lift is probably caused by reranking or noise. Do not credit TCCI.

If lift appears only when final returned count increases, reject. That is candidate inflation.

If lift appears only when source-tail proposal is enabled, reject. That repeats the CBPC intrusion-prone path.

If text cases improve but audio/image/sensor continuation cases do not, revise or reject. The runtime mechanism may be modality-general, but the empirical effect would still be text-local.

If the mechanism fires on positives and negatives at similar rates, reject the gate. The context cue is not discriminative enough.

If latency fails but candidate inclusion is clean, revise the data structure and rerun. Do not promote a slow path into realtime retrieval.

If Stage A cannot find enough TCCI-reachable positives, the next work item is dataset construction, not mechanism tuning.

## 10. Rejection gates

Reject TCCI if any of the following are observed:

- fires zero times on the reachable positive slice;
- proposes the target in fewer than 10 percent of target-absent reachable positives;
- improves reference metrics by less than +0.03 absolute on the reachable slice after adequate sample size;
- adds any extra intrusion on the 48-case remote guard;
- raises expanded remote intrusion by more than +0.005 absolute;
- improves only by increasing final returned count;
- improves only through source-tail or episode-edge fanout rather than temporal-context lookup;
- regresses non-text guards;
- requires text-token, transcript, OCR, or modality-specific runtime branches;
- writes durable labels, facts, summaries, or new extracted state;
- exceeds the latency gate;
- requires production environment toggles to behave correctly.

## 11. Dataset requirements

Do not validate on synthetic-only data. Synthetic fixtures are acceptable for unit reachability and edge-case debugging, but promotion requires real or real-derived data.

Use the existing `cortext_modality_agnostic_ambiguity_bench` as a guard, then add a TCCI-reachable extension.

Recommended sources:

- existing PersonaChat, TopicalChat, and Taskmaster-derived text cases for context-dependent dialogue continuation;
- Ego4D-derived egocentric video sequences for visual object/event continuation and episodic-memory-style past-context queries;
- RAVDESS-derived audio or audio-visual snippets for low-information audio continuations and non-text guards;
- internal real agent/sensor traces if available, especially same-task and same-location continuations;
- real distractor cases with same-source and cross-source lures.

Dataset selection may use offline text, transcripts, human annotations, object labels, or dataset metadata to identify target memories and distractors. The runtime mechanism must not receive those fields.

For every positive case, store offline labels for:

- target memory id;
- whether target is absent from baseline;
- target content similarity rank;
- target temporal-context rank;
- target source continuity;
- target episode relation;
- distractor ids;
- whether the current signal is non-text;
- expected success condition.

For every negative case, store offline labels for:

- lure memory id;
- lure source relation;
- lure episode relation;
- boundary shift status;
- whether the input is non-referential;
- expected abstention condition.

## 12. Implementation guidance for the experiment

Hook location:

Run TCCI after initial baseline vector retrieval and before final candidate selection inside `GraphAugmentedRetrieveCandidates`. SC-Rerank should remain after final selection, as in the current safe path.

State that may be read:

- current embedding;
- accumulator centroid;
- current temporal context vector `c_t`;
- accumulator uncertainty and focus spread;
- source id and recent source continuity;
- episode and boundary state;
- baseline retrieval candidates and scores;
- stored memory embeddings;
- stored memory temporal context values;
- memory source metadata and source confidence;
- working-memory slots;
- write-exclusion timestamp;
- graph edges for one-hop same-episode diagnostics;
- F/S/T knobs.

State that must not be read:

- text tokens;
- pronoun or deictic detectors;
- transcripts;
- OCR text;
- modality-specific labels;
- generated summaries;
- extracted entity labels;
- durable semantic facts;
- user-facing production toggles.

State that must not be written:

- durable labels;
- durable facts;
- durable summaries;
- new consolidation products;
- new memory graph edges;
- modified memory content;
- modified source metadata.

Allowed writes for experiments:

- benchmark trace logs;
- per-scenario diagnostic counters;
- temporary per-signal candidate lists;
- temporary per-run shadow audit records.

These are evaluation artifacts, not memory artifacts.

## 13. Recommended decision rule

Promote only if TCCI proves the full causal chain:

1. the benchmark contains target-absent reachable positives;
2. the content-underreach gate fires on those positives;
3. source and boundary gates suppress negatives;
4. temporal-context lookup proposes the missing target;
5. the target passes normal filters;
6. the target survives fixed-count final selection;
7. reference hit@k improves;
8. intrusion, returned count, non-text guards, and latency remain inside gates.

Revise if:

- temporal-context proposal works but final selection suppresses the candidate;
- non-text positives are under-sampled;
- latency is too high but the candidate path is clean;
- multi-context competition is too conservative but negative controls remain safe.

Reject if:

- it fires zero times like CBER;
- it repeats CBPC’s source-tail intrusion path;
- gains are caused by candidate-count inflation;
- gains are text-only;
- intrusion or latency regressions appear;
- the temporal-context cue is not causally load-bearing under ablation.

## 14. References for research grounding

Tulving, E., and Thomson, D. M. (1973). Encoding specificity and retrieval processes in episodic memory. Psychological Review, 80(5), 352-373. https://alicekim.ca/9.ESP73.pdf

Howard, M. W., and Kahana, M. J. (2002). A distributed representation of temporal context. Journal of Mathematical Psychology, 46, 269-299. https://memory.psych.upenn.edu/files/pubs/HowaKaha02.pdf

Ezzyat, Y., and Davachi, L. (2011). What constitutes an episode in episodic memory? Psychological Science, 22(2), 243-252. https://journals.sagepub.com/doi/10.1177/0956797610393742

Johnson, M. K., Hashtroudi, S., and Lindsay, D. S. (1993). Source monitoring. Psychological Bulletin, 114(1), 3-28. https://pubmed.ncbi.nlm.nih.gov/8346328/

Cowan, N. (2001). The magical number 4 in short-term memory: a reconsideration of mental storage capacity. Behavioral and Brain Sciences, 24(1), 87-114. https://pubmed.ncbi.nlm.nih.gov/11515286/

Yassa, M. A., and Stark, C. E. L. (2011). Pattern separation in the hippocampus. Trends in Neurosciences, 34(10), 515-525. https://pubmed.ncbi.nlm.nih.gov/21788086/

Ego4D Consortium. Ego4D dataset and episodic memory benchmark. https://ego4d-data.org/

Livingstone, S. R., and Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song. Zenodo. https://zenodo.org/records/1188976
