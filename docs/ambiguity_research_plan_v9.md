# Cortext Ambiguity Retrieval Research Plan v8

## Title

Cue-Diagnostic Pattern Completion for Modality-Agnostic Missing-Candidate Retrieval

## Decision from the TCCI pilot

Temporal-Context Candidate Inclusion should not be promoted as tested.

TCCI proved the important positive causal step: a stored temporal-context vector can propose a baseline-missing grounding memory using only modality-general state. That is the first successful evidence that the original missing-candidate problem is reachable without text-token logic or modality-specific branches.

The failure was not candidate proposal. The failure was source and episode discrimination. The same temporal-context carryover that recovered missing reference targets also raised remote intrusion. The next research direction should therefore not be “more temporal context” and not “broader graph expansion.” It should be a human-memory-inspired remote-intrusion guard around bounded pattern completion.

The proposed next experiment is **Cue-Diagnostic Pattern Completion**, abbreviated CDPC.

CDPC keeps the useful part of TCCI, namely bounded temporal-context candidate proposal, but adds a modality-general diagnosticity gate before any proposed candidate can enter final selection. The question is whether Cortext can distinguish a temporally reinstated target from a temporally plausible lure before the lure becomes a returned memory.

This is a research experiment only. Do not implement it as production behavior until the full verification chain passes.

---

## 1. Problem statement

The exact failure to solve is:

A current signal is low-information or underspecified. Baseline content retrieval misses the relevant grounding memory. Temporal context can propose that memory, but temporal context alone may also propose remote memories from a neighboring or recently active context. Cortext needs a way to complete from context only when the cue is diagnostic of one active episode rather than broadly compatible with multiple plausible episodes.

In human-memory terms, the problem is not merely recall. It is recall with source monitoring and lure suppression. Human memory benefits from contextual reinstatement, but false remembering can arise when a cue evokes contextual detail that is not diagnostic of the true source. The experiment should model this balance: pattern completion should be allowed only when pattern separation/source discrimination is strong enough.

The proposal is intended to solve these Cortext failures:

1. **Missing target, diagnostic context**: baseline retrieval misses the grounding memory, but the temporal-context candidate is the clear best match to the current source, recent episode, and graph neighborhood.

2. **Missing target, non-diagnostic context**: baseline misses the target, temporal context proposes something, but the proposed candidate is not sufficiently separable from remote lures. Correct behavior is abstention, not inclusion.

3. **Remote temporal carryover**: temporal context remains similar after a topic shift, source shift, or episode shift and proposes a stale candidate. Correct behavior is rejection before final ranking.

4. **Multiple active contexts**: two recent contexts remain plausible. Correct behavior is conservative ranking or abstention unless one candidate has a clear source/episode margin.

The mechanism does not try to detect pronouns, deixis, object labels, modality, OCR text, transcript structure, or named entities. It operates only after Cortext has converted the signal into embeddings, provenance, episode state, graph state, and retrieval candidates.

---

## 2. Mechanism hypothesis

### Falsifiable hypothesis

A bounded temporal-context candidate-inclusion path can improve context-dependent missing-candidate retrieval without increasing remote intrusion if and only if candidate inclusion is gated by **cue diagnosticity**: a proposed memory must be not only temporally compatible with the current state, but also separable from plausible remote lures under a source, boundary, graph-adjacency, and current-embedding cue bundle.

Operationally:

> If a temporal-context proposal has a positive diagnosticity margin over bounded lure candidates, then adding it to the fixed final candidate pool will improve missing-reference hit rate. If the margin is absent or ambiguous, abstaining will prevent the intrusion regression observed under TCCI.

This is falsifiable. It fails if the gate either fires zero times, blocks all useful TCCI inclusions, or still admits remote intrusions at a rate above SC-Rerank baseline.

### Human-memory grounding

CDPC is based on five human-memory ideas, translated into Cortext state.

First, encoding specificity says retrieval cues are useful when the cue at retrieval overlaps with the conditions under which the memory was encoded. In Cortext, temporal context, source id, boundary state, and graph neighborhood are treated as encoding-context traces, not as language tokens.

Second, source monitoring says recall is not only reactivation of content. It also requires attribution of a memory to the correct origin or context. In Cortext, source id, source continuity, source confidence, contradiction history, and recent source-local working-memory state are the available modality-general equivalents.

Third, event segmentation says boundaries structure memory. A cue that crosses a boundary should be less trusted than a cue inside a still-active event. In Cortext, boundary probability, next/prev/within-event graph edges, source-tail position, and episode adjacency provide the available boundary evidence.

Fourth, pattern completion and pattern separation are complementary. The same partial cue that recovers a target can also over-complete to a lure. CDPC treats temporal context as the completion cue and diagnosticity margin as the separation guard.

Fifth, temporal-context models explain why recent contextual states can retrieve contiguous or related memories. TCCI already demonstrated that this can work in Cortext. CDPC adds the missing discrimination step: context can cue recall, but it should not be enough by itself.

---

## 3. Proposed mechanism

### Name

Cue-Diagnostic Pattern Completion, or CDPC.

### Conceptual placement

CDPC should run as a research path inside `GraphAugmentedRetrieveCandidates`, after baseline vector/graph retrieval and before final candidate selection and SC-Rerank.

The mechanism has two separable stages:

1. **Bounded proposal**, inherited from TCCI:
   - use temporal context and bounded graph adjacency to propose at most one or two baseline-missing candidates;
   - no extra vector search;
   - no new extraction;
   - no durable writes;
   - no increased final returned count.

2. **Cue-diagnosticity gate**, new in v8:
   - compare each proposed candidate against a bounded lure set;
   - include the proposal only if it is source/episode/context diagnostic enough;
   - otherwise abstain.

### Allowed runtime state

CDPC may read:

- current embedding or accumulator centroid;
- current temporal context vector;
- stored memory temporal-context vectors;
- source id and source continuity state;
- source confidence and provenance metadata;
- working-memory slots;
- recent context embeddings;
- episode id, boundary probability, and boundary age;
- retrieved candidates from baseline retrieval;
- bounded graph edges, especially `next_in_episode`, `prev_in_episode`, `within_same_event`, `derived_from`, and `reinforces`;
- retrieval-score distribution over the already selected or already proposed set;
- F/S/T knobs.

CDPC must not read or use:

- token strings;
- pronouns or deictics;
- transcripts;
- OCR output;
- modality type branches;
- entity labels as online logic;
- summarization outputs as a new dependency;
- durable fact or label writes;
- production environment toggles;
- unbounded fanout;
- extra vector searches in the first candidate promotion test.

### Lure set

The key difference from TCCI is that every candidate proposal must be judged against plausible lures.

The lure set should be bounded and cheap. It should include:

- baseline returned candidates;
- other temporal-context proposals in the same bounded expansion pass;
- graph-neighbor proposals from the same anchors;
- candidates with high temporal-context similarity but weak source or episode support;
- candidates whose content score is close to the proposed target but whose source/episode state conflicts with the current source or event.

No additional vector search should be used in the first version. The lure set should be formed from candidates already available in baseline retrieval plus the bounded temporal/graph proposal path.

### Diagnosticity score

For each proposed candidate `m`, compute a diagnostic cue bundle conceptually, not as production code:

- **temporal-context support**: similarity between current temporal context and `m.context`;
- **source support**: same source, source continuity, source confidence, and absence of source contradiction pressure;
- **event support**: same episode, low boundary distance, or strong `next/prev/within_same_event` support from an active anchor;
- **working-memory support**: compatibility with current active memory slots without being a duplicate;
- **current-embedding compatibility**: current embedding is not strongly anti-compatible with `m.embedding`;
- **graph-local support**: bounded path from an active or recently retrieved anchor, not a broad semantic fanout;
- **competition penalty**: how many lures have comparable cue-bundle support.

The proposed candidate passes only when its cue-bundle score exceeds the best lure score by a margin. If the top two candidates are too close, or if the cue distribution is flat, CDPC abstains.

The diagnosticity score must not be used to increase final candidate count. It is a gate on candidate inclusion, not a broadening mechanism.

### Suggested gate semantics

The developer should treat the following as experimental semantics, not production code.

A proposed candidate is eligible only if all of the following are true:

1. Baseline retrieval under-reaches: the reference target is absent from baseline in the benchmark trace, or online metrics indicate weak/flat content retrieval.

2. Temporal-context proposal exists: TCCI-like proposal path identifies a bounded candidate not already in baseline.

3. Candidate is not contradicted by the current embedding: current embedding is weak, but not actively opposed to the candidate.

4. Candidate has at least two independent continuity supports:
   - temporal-context support;
   - source support;
   - event/boundary support;
   - graph-local support;
   - working-memory support.

5. Candidate beats the best lure by a knob-derived diagnosticity margin.

6. Candidate survives normal retrieval filters.

7. Final returned count remains exactly equal to baseline count.

This turns TCCI from “temporal context can add candidates” into “temporal context can add candidates only when it is source-diagnostic.”

### F/S/T effects

The F/S/T sweeps should test whether the mechanism behaves coherently under the existing Cortext control philosophy.

Focus should tighten inclusion. Higher F should require a larger diagnosticity margin and stronger current-embedding compatibility. Low F may allow a smaller margin, but should not disable lure competition.

Sensitivity should increase reach only when the cue is diagnostic. Higher S can lower the underreach trigger or allow one additional internal proposal, but it must not weaken remote-intrusion rejection once a lure set exists.

Stability should preserve continuity across longer ongoing episodes while respecting boundaries. Higher T can tolerate longer same-source temporal carryover inside the same event, but should strengthen rejection across known boundaries or source shifts.

The expected healthy pattern is: higher F reduces false inclusions, higher S raises reachable true proposals, and higher T improves same-episode continuity without resurrecting remote pre-boundary lures.

---

## 4. Observable causal path

The experiment must instrument the intermediate chain. Final hit-rate changes alone are not enough.

For every scenario, log the following counters:

1. **content-underreach gate passed**
   - baseline retrieval was weak, flat, or missing the target.

2. **temporal context available**
   - current temporal context and stored memory context vectors are present.

3. **bounded proposal formed**
   - at least one baseline-missing candidate was proposed before diagnosticity gating.

4. **target proposed**
   - for reference cases, the ground-truth target was among proposals.

5. **lure set formed**
   - at least one plausible alternative candidate was available for comparison.

6. **diagnosticity margin computed**
   - proposed candidate and best lure received comparable cue-bundle scores.

7. **diagnosticity gate passed**
   - proposed candidate exceeded the best lure by the required margin.

8. **normal filters passed**
   - source confidence, duplicate, write-exclusion, and other existing filters did not remove it.

9. **candidate entered final selection pool**
   - candidate was internally added before final fixed-count selection.

10. **candidate survived final selection**
    - candidate appeared in returned results without increasing returned count.

11. **candidate improved metric**
    - target appears in hit@k or improves hit@1/hit@3.

12. **intrusion blocked**
    - for negative controls, a candidate that would have been added under unguarded TCCI is rejected by diagnosticity.

Promotion is credible only if positive gains can be traced through steps 1 to 11, and intrusion control can be traced through step 12.

A result where the final metric improves but intermediate counters do not show target proposal and diagnosticity passage should be treated as confounded.

A result where intrusion does not regress only because CDPC never proposes candidates should be treated as another CBER-style failure.

---

## 5. Required benchmark slice

The existing 103-case ambiguity benchmark should remain as the general regression suite, but it is not sufficient by itself. TCCI fired only on a small number of cases after the harness was fixed, so the next benchmark must include a reachable diagnosticity slice.

Build two benchmark layers.

### Layer A: Current general regression suite

Keep the current 103 real or real-derived scenarios:

- 48 reference cases;
- 48 remote-intrusion guards;
- 7 non-text guards;
- same encoder path;
- same final returned-count accounting;
- same SC-Rerank baseline.

This layer answers whether CDPC breaks the existing benchmark.

### Layer B: reachable CDPC diagnosticity suite

Construct a new balanced slice where candidate inclusion is reachable and where lure discrimination is measurable.

The minimum useful size should be:

- at least 60 diagnostic reference cases;
- at least 60 matched remote-intrusion cases;
- at least 30 topic-shift cases;
- at least 30 multiple-active-context cases;
- at least 30 non-text or mixed-modality guard cases.

The slice should be real or real-derived. Synthetic cases may be used only as debugging fixtures, not as promotion evidence.

### Required scenario families

#### 1. Diagnostic context-dependent continuation

A weak signal follows an ongoing same-source episode. Baseline misses the target. Temporal context can propose the target. Lures are present but have weaker source/episode support.

Success: CDPC proposes, passes diagnosticity, returns the target, and does not increase returned count.

Failure: target is not proposed, target is rejected despite a clear margin, or a lure survives instead.

#### 2. Non-diagnostic continuation

A weak signal follows recent activity, but multiple active contexts are equally plausible.

Success: CDPC abstains or preserves SC-Rerank order without adding a candidate.

Failure: CDPC adds one of the plausible contexts without a diagnostic margin.

#### 3. Remote temporal carryover

A recent temporal context remains similar after a topic shift or source shift. TCCI would propose a stale candidate.

Success: CDPC forms a lure set and rejects the stale candidate at the diagnosticity gate.

Failure: stale candidate is included or returned.

#### 4. Same-source different-episode lure

The source id is continuous, but a boundary separates the current signal from the old candidate.

Success: boundary evidence suppresses inclusion unless graph-local evidence strongly reconnects the old episode.

Failure: same-source continuity alone admits the lure.

#### 5. Different-source same-content lure

The content embedding is similar to a remote memory from another source, but provenance conflicts with the current context.

Success: source diagnosticity blocks or downranks the lure.

Failure: content similarity overrides source mismatch.

#### 6. Multiple plausible active contexts

Two recent episodes remain active in working memory or source tails.

Success: CDPC includes a candidate only if one has a strong diagnosticity margin; otherwise abstains.

Failure: CDPC chooses arbitrarily or increases intrusions.

#### 7. Non-referential low-information input

The input is weak, short, noisy, or sparse, but it is not a continuation.

Success: CDPC does not add candidates.

Failure: CDPC treats generic low information as a completion request.

#### 8. Non-text continuation and guard cases

Use audio, video, image, or sensor-derived signals. Offline selection may use labels or annotations, but runtime mechanism must operate only on embeddings and Cortext state.

Success: CDPC behaves consistently with text-derived cases and does not require modality branches.

Failure: gains appear only in text, or non-text guards regress.

### Dataset guidance

Use the existing real text-derived sources only for part of the benchmark. The reachable diagnosticity slice should include non-text and mixed-modality sources.

Recommended sources:

- existing Cortext ambiguity benchmark cases from PersonaChat, TopicalChat, Taskmaster, and RAVDESS-derived audio guards;
- Ego4D episodic-memory tasks for first-person video continuity and object/event re-access;
- EPIC-KITCHENS or HD-EPIC for egocentric object/action continuation and same-place/same-task lures;
- Charades-Ego for first-person/third-person activity continuation and viewpoint-source separation;
- RAVDESS, CREMA-D, or similar real audio corpora for low-information audio affect/prosody guards;
- real Cortext session logs where consent and privacy handling permit offline benchmark construction.

Offline scenario selection may use transcripts, labels, annotations, or human marking to identify reference targets and lures. Runtime CDPC must never use those labels or text features.

---

## 6. Negative controls

Negative controls must be matched to the positive cases, not treated as unrelated noise.

### Remote intrusion controls

Construct cases where temporal context can propose a baseline-missing remote candidate, but that candidate belongs to the wrong source, wrong episode, or wrong side of a boundary.

Pass: unguarded TCCI would include the candidate, but CDPC blocks it.

Fail: CDPC includes the remote candidate.

### Topic shift controls

Current signal follows a boundary or topic shift. The prior topic remains temporally close.

Pass: CDPC detects non-diagnostic context and abstains.

Fail: CDPC resurrects the prior topic.

### Distractor controls

A lure is semantically close to the current signal and temporally close to the old context.

Pass: candidate inclusion requires source/episode margin, not content similarity alone.

Fail: lure beats the target because it is semantically closer.

### Multiple-active-context controls

Two recent episodes have comparable source and temporal support.

Pass: CDPC abstains unless one has a clear graph/source/boundary margin.

Fail: CDPC adds either one without margin.

### Non-text guards

Audio, video, image, or sensor cases are included where the correct behavior is no completion or conservative completion.

Pass: no increased false inclusion compared with SC-Rerank.

Fail: CDPC over-completes because weak embeddings are treated as ambiguity requests.

---

## 7. Ablation matrix

Run the ablations as named conditions. Every condition must keep final returned count fixed unless the condition is explicitly testing count inflation.

### Core comparisons

1. **Baseline retrieval**
   - existing retrieval path, no SC-Rerank, no TCCI, no CDPC.

2. **Baseline + SC-Rerank**
   - current best safe ranking baseline.

3. **Unguarded TCCI**
   - temporal-context candidate inclusion as previously tested, used as the recall-positive but intrusion-unsafe reference.

4. **Full CDPC**
   - bounded TCCI proposal plus cue-diagnosticity gate plus SC-Rerank.

### Causality ablations

5. **Diagnosticity gate disabled**
   - should revert toward unguarded TCCI behavior and reproduce intrusion regression.

6. **Lure-set comparison disabled**
   - candidate judged only by its own cue score.
   - should admit more remote intrusions if lure competition is load-bearing.

7. **Source support disabled**
   - diagnosticity ignores source id, source continuity, and source confidence.
   - should regress different-source and mixed-provenance controls.

8. **Boundary/event support disabled**
   - diagnosticity ignores boundary probability and episode adjacency.
   - should regress same-source different-episode lures.

9. **Graph-local support disabled**
   - diagnosticity ignores `next/prev/within_same_event` and bounded local edges.
   - should reduce true inclusions in object/event continuation cases.

10. **Current-embedding compatibility disabled**
    - should admit candidates that temporal context likes but current signal contradicts.

11. **Cue-entropy or flatness disabled**
    - should regress multiple-active-context cases.

12. **Working-memory support disabled**
    - should reduce correct continuation where the active slot anchors the target.

13. **Source-continuity only**
    - use source continuity as the sole guard.
    - expected to fail because prior pilots showed source continuity alone can be useful for reranking but not sufficient for safe inclusion.

14. **Temporal-context only**
    - use temporal context proposal without diagnosticity.
    - expected to reproduce TCCI recall/intrusion tradeoff.

15. **No SC-Rerank after inclusion**
    - verifies whether inclusion itself is sufficient or only works because SC-Rerank rescues order.

16. **Max-one versus max-two internal proposal budget**
    - final returned count fixed.
    - promotion should not depend on max-two unless the second proposal is strictly needed and does not increase intrusion.

17. **Candidate-count inflation control**
    - intentionally allow increased final returned count in a study-only condition.
    - if this is the only condition that improves hit@k, reject CDPC.

18. **Extra-vector-search control**
    - study-only upper bound, not promotion candidate.
    - if only this condition works, the proposed low-latency mechanism is not sufficient.

### F/S/T sweeps

Run the full mechanism and the key ablations over:

- F/S/T = 0.2, 0.5, 0.9 full grid for the reachable diagnosticity subset;
- at minimum, F/S/T = 0.5 for the full 103-case regression benchmark;
- targeted one-knob sweeps:
  - Focus sweep: F = 0.2, 0.5, 0.9 with S=T=0.5;
  - Sensitivity sweep: S = 0.2, 0.5, 0.9 with F=T=0.5;
  - Stability sweep: T = 0.2, 0.5, 0.9 with F=S=0.5.

Expected knob behavior:

- higher F should reduce inclusion rate and intrusion;
- higher S should increase proposal reach but not remove lure rejection;
- higher T should preserve same-episode continuity while suppressing cross-boundary carryover.

If knobs do not affect the expected intermediate counters, do not promote.

---

## 8. Metrics

Report metrics at both the full-suite level and the reachable-slice level.

### Primary metrics

- reference hit@k;
- reference hit@1;
- reference hit@3;
- reachable-slice reference hit@k;
- reachable-slice strict inclusion count;
- strict inclusions returned;
- remote-intrusion rate;
- remote-intrusion rate on cases where unguarded TCCI would fire;
- non-text guard hit/regression rate;
- top-k precision;
- returned count mean;
- graph retrieval mean latency;
- CDPC fired-case overhead;
- total graph path overhead;
- internal proposal count.

### Causal-chain metrics

For each scenario family, report:

- content-underreach passed;
- temporal context available;
- proposal formed;
- target proposed;
- lure set formed;
- diagnosticity pass;
- diagnosticity reject;
- normal filters passed;
- entered final pool;
- survived final selection;
- strict inclusion;
- strict inclusion returned;
- intrusion blocked.

### Diagnostic metrics

Report distributions, not only means:

- diagnosticity margin for true inclusions;
- diagnosticity margin for blocked lures;
- cue-bundle entropy or top-two gap;
- source-support contribution;
- boundary-support contribution;
- graph-local contribution;
- current-embedding compatibility contribution;
- number of lures considered;
- reason for abstention.

The diagnosticity margin should visibly separate true reference continuations from remote-intrusion guards. If the distributions overlap heavily, the mechanism is unlikely to be promotable even if one benchmark pass looks good.

---

## 9. Promotion gates

Promotion requires all gates below.

### Recall gate

Full-suite reference hit@k must improve over **Baseline + SC-Rerank** by at least +0.03, or the reachable diagnosticity slice must improve by at least +0.10 while the full suite remains non-regressed.

Reference hit@1 or hit@3 should improve or remain stable. A hit@k-only gain is not sufficient if it is caused by ordering degradation.

### Strict inclusion gate

CDPC must produce real strict inclusions.

Minimum acceptable evidence:

- at least 50% of unguarded TCCI true strict inclusions are retained, or
- at least 10 true strict inclusions on the new reachable diagnosticity slice.

A mechanism that fires zero times, proposes no baseline-missing targets, or only reranks already retrieved candidates is not solving this research target.

### Intrusion gate

Remote intrusion must not materially regress relative to **Baseline + SC-Rerank**.

Pass criteria:

- full-suite intrusion rate ≤ SC-Rerank intrusion rate + 0.01;
- reachable remote-intrusion slice ≤ SC-Rerank + 0.01;
- CDPC blocks at least 70% of unguarded TCCI false inclusions on cases where TCCI would have fired;
- no scenario family shows a concentrated intrusion regression above +0.03.

If unguarded TCCI improved recall but raised intrusion, CDPC must preserve a meaningful portion of recall while removing that intrusion regression.

### Candidate-count gate

Final returned count must remain fixed to baseline count.

Pass criteria:

- returned count mean equals baseline within measurement noise;
- no condition promoted because it returns more candidates;
- internal proposals remain bounded at max one by default, max two only in an ablation.

### Precision gate

Top-k precision must improve or remain stable.

Pass criteria:

- no precision regression greater than 0.01 on the full suite;
- no precision regression greater than 0.02 on any negative-control family;
- if hit@k improves while precision falls, reject.

### Non-text gate

Non-text guards must not regress.

Pass criteria:

- no increased false inclusion on audio/video/image/sensor cases;
- no reliance on modality labels or text-derived runtime state;
- at least one non-text continuation slice shows the same causal path if the slice is available.

If gains are text-only, reject.

### Latency gate

The mechanism must preserve realtime operation.

Pass criteria:

- no extra vector search in the promotion candidate;
- fired-case CDPC overhead ≤ 0.5 ms mean in the benchmark harness;
- graph retrieval mean latency ≤ SC-Rerank baseline + 5%;
- p95 graph latency should not regress by more than 10%;
- candidate proposal and lure scoring remain bounded.

### Ablation causality gate

At least three ablations must separate as predicted:

- disabling diagnosticity or lure-set comparison should increase intrusion;
- disabling source support should regress different-source controls;
- disabling boundary/event support should regress same-source different-episode controls;
- disabling graph-local support should reduce true continuation inclusions;
- disabling cue entropy/top-two margin should regress multiple-active-context controls.

If ablations do not show causal separation, do not promote even if headline metrics pass.

---

## 10. Rejection gates

Reject CDPC if any of the following occur.

1. **Zero-fire result**
   - CDPC never proposes or never includes candidates, matching the CBER failure mode.

2. **TCCI regression repeats**
   - reference recall improves, but intrusion rises above SC-Rerank +0.01.

3. **Fanout explains gain**
   - improvement appears only when final returned count increases, max-two budget is required without precision, or extra vector search is enabled.

4. **Text-only gain**
   - gains appear only in text-derived cases and fail non-text or mixed-modality guards.

5. **No diagnostic separation**
   - true targets and remote lures have overlapping diagnosticity margins and ablations cannot identify load-bearing source/boundary/lure components.

6. **Latency regression**
   - graph retrieval latency or p95 realtime performance meaningfully regresses.

7. **Source continuity alone explains result**
   - if the full mechanism reduces to SC-Rerank-like source continuity without missing-candidate inclusion, this is not a solution to the original problem.

8. **Unacceptable precision tradeoff**
   - hit@k improves by admitting more loosely related memories but top-k precision falls.

9. **Boundary-blind intrusion**
   - same-source different-episode controls regress because source continuity overwhelms event boundaries.

10. **No retained TCCI benefit**
    - CDPC blocks intrusion only by rejecting all temporal-context inclusions.

---

## 11. Mixed-result interpretation

### Recall improves, intrusion stable, but strict inclusions are few

Do not promote immediately. Increase the reachable diagnosticity slice and rerun. If the result remains stable with at least 10 true strict inclusions, promote may be considered.

### Recall improves only on reachable slice

Revise, not promote. This means the mechanism is promising but the general benchmark is underpowered or the trigger is too narrow.

### Intrusion improves, recall collapses

Revise or reject depending on whether any target proposals survive. If zero or near-zero true inclusions survive, reject.

### Non-text guards are neutral

Neutral is acceptable for promotion only if the benchmark includes at least some non-text reachable continuations. If non-text coverage is only negative guards, promotion should be limited to “no non-text regression” and the next study must add positive non-text continuations.

### Ablations do not separate

Reject. The experiment would not prove that cue diagnosticity caused the result.

### Strong results require extra vector search

Reject as a realtime candidate. Keep as an upper-bound study only.

---

## 12. Implementation guidance for the experiment

### Hook point

The experimental hook should be inside `GraphAugmentedRetrieveCandidates`.

Recommended sequence:

1. Run baseline vector/graph retrieval.
2. Record baseline eligible memory ids.
3. Run bounded TCCI-style temporal-context proposal in shadow.
4. Build the CDPC lure set from baseline candidates and bounded proposals.
5. Compute cue diagnosticity for proposals.
6. Add at most one passing candidate to the internal pool.
7. Run normal filters.
8. Run final fixed-count selection and SC-Rerank.
9. Log causal-chain counters.

This should be an internal experiment path, not the default engine path.

### State that may be read

The experiment may read:

- current embedding;
- accumulator centroid;
- current temporal context vector;
- stored memory context vectors;
- recent context embeddings;
- working-memory slots;
- source id;
- source continuity state;
- source confidence;
- created_at and last_access timestamps;
- episode/boundary state;
- boundary score and boundary age;
- graph edges and edge weights;
- baseline candidate ids and scores;
- retrieval-score distribution;
- F/S/T values.

### State that must not be written

The experiment must not write:

- durable labels;
- durable facts;
- summaries;
- new extraction outputs;
- new persistent source annotations;
- new production toggles;
- permanent graph edges;
- memory strength changes caused only by the experimental proposal.

Temporary per-signal instrumentation is allowed. If telemetry is persisted, it should be benchmark-only and clearly separated from memory state.

### Constraints

The experiment must keep:

- no text-token logic;
- no pronoun/deictic detection;
- no transcript parsing;
- no OCR;
- no modality branches;
- no extra vector search in the promotion candidate;
- no unbounded graph expansion;
- no increase in final returned count;
- no production environment toggle dependency;
- no new extraction or summarization model.

### Reporting format

Each run should report:

- benchmark name and commit/branch;
- encoder path;
- scenario count and family breakdown;
- baseline, SC-Rerank, unguarded TCCI, and full CDPC metrics;
- ablation table;
- F/S/T sweep table;
- causal-chain table;
- latency table;
- reason-coded abstention table;
- failure examples for at least five false negatives and five blocked intrusions.

---

## 13. Expected outcome

The most likely useful outcome is not a large full-suite hit@k jump. The expected gain is a narrower but safer result:

- TCCI proves that temporal context can find missing candidates.
- CDPC should prove that Cortext can decide when temporal-context completion is source-diagnostic.
- If successful, CDPC should retain some true TCCI inclusions while blocking most false TCCI inclusions.

This is the right next research direction because it addresses the demonstrated failure directly. The system already has a candidate-proposal path that can recover missing targets. What it lacks is a modality-general analogue of human source monitoring and pattern separation.

---

## 14. Research references for grounding

These references are included to guide the experiment design, not to justify any text-specific runtime behavior.

- Tulving, E., and Thomson, D. M. (1973). Encoding specificity and retrieval processes in episodic memory. This motivates requiring a retrieval cue to be diagnostic of the encoded episode, not merely similar.
  URL: https://wixtedlab.ucsd.edu/publications/Psych%20218/Tulving_Thompson_1973.pdf

- Johnson, M. K., Hashtroudi, S., and Lindsay, D. S. (1993). Source monitoring. This motivates source/provenance discrimination as part of recall.
  URL: https://pubmed.ncbi.nlm.nih.gov/8346328/

- Zacks, J. M., and Swallow, K. M. (2007). Event segmentation. This motivates boundary-sensitive gating.
  URL: https://pubmed.ncbi.nlm.nih.gov/22468032/

- Howard, M. W., and Kahana, M. J. (2002). A distributed representation of temporal context. This motivates temporal context as a recall cue while requiring downstream competition.
  URL: https://www.sciencedirect.com/science/article/pii/S0022249601913884

- Yassa, M. A., and Stark, C. E. L. (2011). Pattern separation in the hippocampus. This motivates lure discrimination as a necessary complement to pattern completion.
  URL: https://pubmed.ncbi.nlm.nih.gov/21788086/

- Herz, N., Bukala, B. R., Kragel, J. E., and Kahana, M. J. (2023). Hippocampal activity predicts contextual misattribution of false memories. This motivates treating contextual carryover as a possible source of false recall.
  URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10556612/

- Ego4D Episodic Memory benchmark. This is a recommended source of real egocentric video cases for non-text continuation and object/event re-access.
  URL: https://ego4d-data.org/docs/benchmarks/episodic-memory/

- EPIC-KITCHENS. This is a recommended source of real egocentric object/action continuation cases.
  URL: https://epic-kitchens.github.io/

- RAVDESS. This is a recommended source of real audio/affective guard cases.
  URL: https://www.kaggle.com/datasets/thbdh5765/audio-visual-database-of-emotional-speech-and-song
