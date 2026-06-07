# Soft Anchor Consumption Experiments

Soft Anchor formation is in the engine. Consumption is not. This document
defines every plausible way Cortext could consume formed anchors, what each
would mean in product behavior, and the experiments required before one of them
is accepted into the engine.

The principle is simple: no permanent policy flags. A consumption behavior is
either good enough to become the default product behavior, or it stays out.
Experiments may compare variants, but the output should be one accepted
consumption contract, not a menu of runtime modes.

## Current Evidence

Formation evidence is positive:

- chronological ingress can form Soft Anchor links before retrieval;
- the target is retained in top-k soft links on most reference cases;
- hard and durable false commits can be kept at zero on the repaired replay;
- Focus and Sensitivity move formation and consumption volume predictably.

Consumption evidence is not yet sufficient:

- conservative consumption surfaces too little useful context;
- high-recall consumption surfaces many useful references but also too many
  no-anchor hints;
- Stability is under-stressed by the current replay;
- no manual chat review has measured whether uncertain hints are actually useful
  to a human or LLM.

Therefore, the next work is not "make a flag." The next work is to find a
consumption shape that is useful enough, safe enough, and cheap enough to be in.

## Definitions

Soft Anchor formation:

- runs at ingress after memory storage;
- writes anchor states and soft links;
- does not alter retrieval ranking;
- does not surface context to the human or LLM.

Soft Anchor consumption:

- reads already-formed anchor state and soft links;
- may add uncertain context, suppress unsafe context, group memories, or guide
  clarification;
- must not infer anchors from retrieved candidates;
- must not turn tentative links into facts.

Consumption harm:

- a no-anchor turn receives subject/entity context that makes the assistant act
  as if a referent exists;
- a wrong-active candidate is surfaced without the true target also present;
- stale same-source context is surfaced as current;
- an ambiguous link is phrased as fact;
- prompt/context overhead makes normal turns worse;
- the user or LLM becomes more confused than with no anchor consumption.

Acceptable consumption error:

- uncertain top-k includes one wrong candidate and the right candidate;
- a weak reference gets no hint;
- the assistant asks a clarifying question instead of guessing;
- tentative context is shown as possible continuity only.

## Candidate Consumption Designs

### C0: No Consumption Baseline

Behavior: form anchors, do not surface or use them.

Purpose: baseline for user-visible value and latency.

Expected result: zero anchor-caused harm, zero anchor-caused benefit.

### C1: Silent Safety Consumer

Behavior: consume `none`, generic, contradiction, and ambiguity evidence only to
avoid overconfident memory use. Do not surface anchor candidates.

Example effect:

- if current turn has high `H_none`, avoid injecting nearby subject memories;
- if current link is ambiguous, avoid writing a durable fact from the turn;
- if generic turn, do not strengthen or surface anchors.

Why it might work: safety is easier than helpful surfacing.

Risk: invisible benefit may be hard to measure; could suppress useful context.

### C2: Possible Continuity Hint

Behavior: surface top-k candidate anchors as explicitly uncertain context.

Example prompt shape:

```text
Possible continuity, uncertain:
- A17, strength 0.46, recent: "Jared was showing me the pool."
- A41, strength 0.42, recent: "Alex came over after dinner."

Use this only as possible context. Do not assert identity unless the user
message makes it clear.
```

Why it might work: it avoids the humiliation of drawing a blank while preserving
uncertainty.

Risk: hints on no-anchor turns may distract the LLM.

### C3: Anchor Neighborhood Context

Behavior: when current ingress has a soft link to an anchor, include a compact
window of memories attached to that anchor, not just the anchor label.

Variants:

- latest memory from anchor;
- last 2-3 memories from anchor;
- one intro memory plus one recent memory;
- source-diverse neighborhood;
- multimodal summary placeholder when original item is image/audio.

Why it might work: a candidate anchor id alone is not useful. The LLM needs
enough evidence to decide.

Risk: context budget grows and stale memories may dominate.

### C4: Ambiguous Candidate Set

Behavior: if margin is low or entropy is high, surface multiple candidates as a
set. Do not choose top-1.

Example:

```text
Possible referents:
- Jared: mentioned in "I went to Jared's house yesterday"
- Alex: mentioned in "Alex stopped by later"

The current message may refer to one of these, or neither.
```

Why it might work: for human memory augmentation, two plausible references can
be better than none.

Risk: too many candidates creates noise.

### C5: Clarification Consumer

Behavior: when ambiguity is high and the current task can tolerate a question,
surface a clarifying action instead of context.

Example assistant behavior:

```text
Do you mean Jared or Alex?
```

Why it might work: it converts unsafe anchoring into an interaction.

Risk: annoying in low-stakes chat; bad for realtime speech if overused.

### C6: LLM Self-Selection Consumer

Behavior: provide structured candidates to the LLM and let the LLM decide whether
to use, ignore, or ask.

Example structured payload:

```json
{
  "soft_anchor_candidates": [
    {
      "label": "possible",
      "strength": 0.46,
      "evidence": ["Jared's house", "pool"],
      "instruction": "use only if current message supports it"
    }
  ]
}
```

Why it might work: LLMs can use uncertain evidence better than scalar rules.

Risk: LLM may over-trust the hint unless prompt wording is strict.

### C7: UI-Only Human Hint

Behavior: show possible continuity chips or expandable evidence to the human,
but do not put the hint into the LLM prompt.

Why it might work: humans can decide if the hint helps and can ignore it.

Risk: not useful for fully automated agent memory.

### C8: Retrieval Annotation Consumer

Behavior: retrieval results are unchanged, but returned memories are annotated
with anchor link evidence and grouped by anchor.

Example:

```text
Memory results:
- M12, possible same subject as current turn, strength 0.46
- M19, same anchor group as M12
```

Why it might work: it improves context organization without changing what is
retrieved.

Risk: annotations may still bias downstream generation.

### C9: Anchor-Aware Context Packing

Behavior: the final context pack uses anchors to remove duplicates, group
related memories, and preserve one representative per anchor.

Why it might work: consumption may add value by organizing context rather than
adding more context.

Risk: grouping errors could hide useful memories.

### C10: Anchor Neighborhood Expansion

Behavior: after ordinary retrieval selects a memory, add nearby memories from
the same pre-formed anchor.

Important distinction: this does not infer anchors from retrieval. It consumes
already-attached anchor links on retrieved memories.

Why it might work: ordinary retrieval finds one memory, anchor expansion supplies
continuity around it.

Risk: can pull stale same-anchor history into the current prompt.

### C11: Current-Turn Anchor Recall

Behavior: if the current user message forms a soft link to an existing anchor,
include the best evidence memory from that anchor even if ordinary retrieval
misses it.

Why it might work: this is the direct "he still means Jared" path.

Risk: no-anchor false hints are the central failure mode.

### C12: Durable-Only Fact Consumer

Behavior: only durable links can support durable fact formation or memory
consolidation. Tentative/ambiguous links may appear as context but never facts.

Why it might work: separates conversational help from long-term memory
corruption.

Risk: durable links are currently too rare to create visible benefit.

### C13: Decayed-Link Reminder

Behavior: decayed links can be surfaced only as weak reminders, never as current
referents.

Example:

```text
Older possibly related context: Jared was discussed earlier.
```

Why it might work: helps delayed references without claiming current identity.

Risk: stale/context-shift confusion.

### C14: Group Anchor Consumer

Behavior: preserve sets such as `we = user + Jared` or `they = Alex + Justin` as
uncertain group candidates.

Why it might work: many human references are group references, not single
entities.

Risk: current formation schema may not represent group composition strongly
enough.

### C15: Cross-Modal Evidence Consumer

Behavior: when an anchor has text, image, audio, or voice evidence, surface a
modality-neutral evidence note.

Example:

```text
Possible continuity: this may refer to the person shown in the pool image and
later mentioned as Jared.
```

Why it might work: this is the real Cortext target, not a text-only demo.

Risk: requires enough multimodal replay data for evaluation.

### C16: Correction-Aware Consumer

Behavior: user corrections demote or reject surfaced links and update future
consumption.

Example:

```text
User: I meant Alex, not Jared.
```

Why it might work: wrong guesses become recoverable learning events.

Risk: correction detection must not become another brittle text rule.

### C17: Ask-Or-Show Hybrid

Behavior: choose between no hint, possible context, or clarification based on
uncertainty and interaction state.

This is the likely final product shape:

- low uncertainty: surface one possible continuity hint;
- medium uncertainty: surface top-k possible candidates;
- high uncertainty and high value: ask;
- high uncertainty and low value: show nothing.

Risk: hardest to evaluate because value depends on task state.

## Experiment Program

### Experiment 1: Consumption Contract Replay

Question: which consumption shape improves useful target surfacing without
unacceptable no-anchor and wrong-active harm?

Dataset:

- repaired real replay;
- existing 421-case storage-time replay;
- delayed buckets 1, 2-4, 5-12;
- controls: no-anchor, wrong-active, stale, remote.

Variants:

- C0 no consumption;
- C1 silent safety;
- C2 top-1 possible continuity;
- C4 top-k ambiguous set;
- C10 anchor neighborhood expansion;
- C11 current-turn anchor recall;
- C17 ask-or-show hybrid.

Metrics:

- useful target surfaced;
- target surfaced in top-k;
- no-anchor hint surfaced;
- wrong-only surfaced;
- stale-only surfaced;
- useful/harmful ratio;
- mean/p95 context chars;
- mean/p95 added latency;
- F/S/T monotonicity.

Acceptance:

- useful/harmful ratio at least 5 on replay;
- no-anchor surfaced rate below 5 percent before manual review;
- wrong-only surfaced below 1 percent;
- p95 added context under a fixed budget;
- Focus and Sensitivity monotonicity preserved.

Outputs:

- `soft_anchor_consumption_contract_results.json`
- `soft_anchor_consumption_contract_summary.csv`
- `soft_anchor_consumption_contract_cases.csv`
- `soft_anchor_consumption_contract_failures.csv`

### Experiment 2: Manual Chat Usefulness Audit

Question: are uncertain hints actually helpful to humans or LLMs?

Dataset:

- 100 to 200 manually auditable chat turns;
- include ordinary chat, delayed references, no-anchor topic shifts,
  wrong-active cases, stale same-source cases, and low-information turns;
- include at least 25 user-like examples with "he", "she", "it", "we", "there",
  or equivalent weak references, but do not rely on those tokens at runtime.

Reviewer labels:

- useful hint;
- harmless extra hint;
- distracting hint;
- harmful wrong hint;
- should ask clarification;
- should show nothing;
- target missing from hint set;
- right target present but poorly explained.

Variants:

- no anchor context;
- possible continuity top-1;
- possible continuity top-3;
- evidence-rich top-2;
- clarification-only;
- ask-or-show hybrid.

Metrics:

- reviewer useful rate;
- reviewer harmful rate;
- useful/harmful ratio;
- "better than blank" rate;
- "would ask" agreement;
- average context length judged tolerable;
- cases where LLM answer improves;
- cases where LLM answer worsens.

Acceptance:

- useful/harmful ratio at least 5;
- harmful wrong hint at most 5 percent;
- no-anchor harmful hints at most 2 percent;
- at least 40 percent of weak-reference turns rated better than no hint;
- reviewers prefer some anchor consumption over no consumption on the target
  population.

Outputs:

- `soft_anchor_manual_review_cases.csv`
- `soft_anchor_manual_review_results.json`
- `soft_anchor_manual_review_examples.md`

### Experiment 3: LLM Answer Quality A/B

Question: does anchor context improve the assistant answer, not just offline
target matching?

Procedure:

For each audited chat turn, generate answers under:

- no anchor context;
- top-k possible continuity context;
- evidence-rich anchor context;
- ask-or-show hybrid.

Use deterministic generation settings. Blind-review the answers.

Metrics:

- answer uses correct referent;
- answer appropriately expresses uncertainty;
- answer asks a useful clarification;
- answer hallucinates identity;
- answer becomes verbose or awkward;
- user-rated preference if available.

Acceptance:

- correct-or-useful-clarification improves over no-anchor baseline;
- hallucinated identity does not increase materially;
- no-anchor turns do not get worse.

Outputs:

- `soft_anchor_llm_ab_answers.jsonl`
- `soft_anchor_llm_ab_review.csv`
- `soft_anchor_llm_ab_results.json`

### Experiment 4: Prompt Shape Ablation

Question: what wording makes uncertain anchors useful without over-claiming?

Prompt shapes:

- terse chip: `Possible continuity: Jared`;
- evidence note: `May refer to Jared: house, pool`;
- structured JSON candidates;
- negative instruction: `Do not assume identity`;
- confidence-free wording;
- numeric-strength wording;
- clarification suggestion wording.

Metrics:

- LLM over-commit rate;
- correct referent use;
- clarification rate;
- output verbosity;
- reviewer preference.

Acceptance:

- selected prompt shape must reduce over-commit versus numeric-strength wording;
- no use of `durable` or `confidence` as if it were truth probability;
- uncertain phrasing preserved in generated answer when evidence is ambiguous.

Outputs:

- `soft_anchor_prompt_shape_results.json`
- `soft_anchor_prompt_shape_examples.md`

### Experiment 5: Context Budget and Evidence Shape

Question: how much anchor evidence is enough?

Variants:

- anchor id only;
- latest memory only;
- intro memory only;
- intro plus latest;
- top 2 recent;
- top 3 source-diverse;
- compact summary generated at consolidation time;
- raw snippets capped at 80, 160, and 320 chars.

Metrics:

- target usefulness;
- harmful distraction;
- p95 added prompt chars;
- answer latency;
- reviewer preference.

Acceptance:

- evidence shape must outperform anchor id only;
- p95 context budget must be compatible with chat;
- no large raw context pack unless it gives a large usefulness lift.

Outputs:

- `soft_anchor_context_budget_results.json`
- `soft_anchor_context_budget_summary.csv`

### Experiment 6: Anchor Neighborhood Expansion

Question: should ordinary retrieval hits expand to same-anchor neighbors?

Variants:

- no expansion;
- expand one previous same-anchor memory;
- expand intro plus latest;
- expand only if current turn has a matching soft link;
- expand only if retrieved memory and current turn agree on anchor;
- expand with stale/boundary veto.

Metrics:

- answer usefulness;
- duplicate context;
- stale context injection;
- no-anchor regression;
- context budget.

Acceptance:

- expansion must improve answer quality on delayed references;
- stale same-source harmful injection must stay near zero;
- no-anchor turns must not receive expansion unless ordinary retrieval already
  surfaced relevant context.

Outputs:

- `soft_anchor_neighborhood_expansion_results.json`
- `soft_anchor_neighborhood_expansion_failures.csv`

### Experiment 7: Clarification Strategy

Question: when should Cortext help the assistant ask instead of hint?

Variants:

- never ask;
- ask on high entropy;
- ask on low margin;
- ask when top two candidate anchors have different entity evidence;
- ask only in interactive chat, not agent/tool mode;
- ask only when answer requires identity.

Metrics:

- useful clarification rate;
- unnecessary clarification rate;
- missed opportunity rate;
- user annoyance proxy;
- answer correctness after clarification if simulated.

Acceptance:

- clarification should be rare;
- when asked, reviewers should prefer asking over guessing in at least 70
  percent of cases;
- no increase in generic-turn interruptions.

Outputs:

- `soft_anchor_clarification_results.json`
- `soft_anchor_clarification_cases.csv`

### Experiment 8: F/S/T Consumption Semantics

Question: do the three knobs produce the right user experience?

Expected behavior:

- Focus up: fewer hints, fewer candidates, higher precision, more clarification
  instead of guessing;
- Sensitivity up: more tentative hints, more top-k candidates, higher weak
  reference recall;
- Stability up: longer availability of older anchor links, more decayed
  reminders, no cross-boundary overreach.

Dataset:

- repaired replay;
- long decay-heavy chat replay;
- manual review slice.

Metrics:

- target hint rate;
- no-anchor hint rate;
- wrong-only hint rate;
- mean candidate count;
- decayed-link use;
- stale overreach;
- Spearman or adjacent monotonicity checks.

Acceptance:

- Focus and Sensitivity monotonicity pass on replay and manual slice;
- Stability must move on long-horizon decay replay, not just configuration;
- high Stability cannot increase cross-boundary harmful hints.

Outputs:

- `soft_anchor_consumption_fst_results.json`
- `soft_anchor_consumption_fst_monotonicity.json`

### Experiment 9: Multimodal Consumption Audit

Question: does consumption remain useful when the anchor evidence is not text?

Cases:

- text current turn references image-introduced person/object/place;
- text current turn references audio/voice-introduced person;
- image current turn references text-introduced entity;
- audio current turn references prior image/text context;
- no-anchor multimodal topic shift.

Consumption variants:

- evidence note with modality labels;
- thumbnail/audio placeholder in UI-only mode;
- structured candidate set for LLM;
- no consumption baseline.

Metrics:

- cross-modal target surfaced;
- wrong-modal false hint;
- reviewer usefulness;
- context budget;
- latency.

Acceptance:

- at least one multimodal variant must outperform text-only baseline;
- no text-token dependency;
- no raw image/audio claim unless evidence exists.

Outputs:

- `soft_anchor_multimodal_consumption_results.json`
- `soft_anchor_multimodal_consumption_cases.csv`

### Experiment 10: Correction and Recovery

Question: if consumption guesses wrong, can the system recover naturally?

Procedure:

- simulate or collect user corrections;
- apply correction to soft anchor state;
- rerun later turns.

Variants:

- correction demotes link only;
- correction rejects link;
- correction splits anchor;
- correction creates relation between candidates;
- correction updates future prompt phrasing.

Metrics:

- repeated wrong hint rate;
- recovery within N turns;
- loss of useful target hints;
- audit trail correctness.

Acceptance:

- explicit correction must prevent repeated identical wrong hint;
- correction must not delete provenance;
- future useful hints should recover when evidence supports them.

Outputs:

- `soft_anchor_correction_recovery_results.json`
- `soft_anchor_correction_audit.csv`

### Experiment 11: Human UI Consumption

Question: should anchors be consumed by UI first, before LLM prompt injection?

UI variants:

- no UI;
- subtle possible-continuity chip;
- expandable evidence drawer;
- top-2 candidate selector;
- correction affordance;
- "not related" affordance.

Metrics:

- human selection/correction rate;
- ignored hint rate;
- harmful confusion reports;
- time to resolve referent;
- later system improvement after correction.

Acceptance:

- UI hint must be helpful without requiring interaction on every turn;
- correction affordance must be easy enough to use;
- no intrusive prompt for generic turns.

Outputs:

- `soft_anchor_ui_consumption_results.json`
- `soft_anchor_ui_events.csv`

### Experiment 12: Agent Tool Consumption

Question: how should non-chat agents consume anchors?

Variants:

- no anchor tool;
- read-only `possible_continuity` context field;
- explicit tool call to fetch anchor neighborhood;
- tool returns top-k candidates with evidence and uncertainty;
- tool can mark hint useful/not useful.

Metrics:

- task success;
- wrong-memory use;
- unnecessary tool calls;
- latency;
- traceability of memory use.

Acceptance:

- anchor tool improves tasks requiring prior subject continuity;
- wrong-memory use does not increase on no-anchor tasks;
- traces preserve uncertainty.

Outputs:

- `soft_anchor_agent_consumption_results.json`
- `soft_anchor_agent_traces.jsonl`

## Implementation Order for Experiments

1. Extend `cortext_ingress_anchor_formation_bench` or add
   `cortext_soft_anchor_consumption_bench` with replayed consumption adapters.
2. Add a context-pack renderer that emits the exact prompt/UI payload for each
   consumption variant.
3. Generate the manual review CSV from real chat turns.
4. Run offline replay metrics.
5. Run blind manual review.
6. Run LLM answer A/B.
7. Decide one default consumption behavior, or keep consumption out.

## Executable Replay Test

The first replay test is:

```bash
python3 tools/soft_anchor_consumption_contract_test.py \
  --input-dir build/soft_anchor_consumption_full_ablation_v2 \
  --output-dir build/soft_anchor_consumption_contract_test_packet
```

To sweep every stored formation policy:

```bash
python3 tools/soft_anchor_consumption_contract_test.py \
  --input-dir build/soft_anchor_consumption_full_ablation_v2 \
  --output-dir build/soft_anchor_consumption_contract_test_all_packet \
  --all-formation-policies
```

This test evaluates formed anchor candidates only. It does not run retrieval,
does not consume retrieved candidates, and does not use gold labels for runtime
selection. Labels are used only after selection for replay scoring.

The replay test reports two views:

- asserted-context scoring, where surfacing a no-anchor hint is considered
  harmful because the engine is acting as if a referent has been resolved;
- SoftAnchor context scoring, where surfaced candidates are optional
  full-sentence memory context and no-anchor surfaces are measured as
  context/noise load rather than automatic harm.

Under asserted-context scoring, no directly testable consumer passes. Under
SoftAnchor context scoring, the best possible-continuity consumer surfaces all
targets with zero wrong-only/stale-only reference contexts. It also emits
context on all no-anchor controls, so the remaining question is not anchor
correctness but whether the consumer can ignore irrelevant optional memory
context and whether the added context budget is acceptable.

## Minimum Product Acceptance

A consumption behavior can be accepted only if:

- it is on by default in experiments, not hidden behind a runtime mode;
- useful/harmful ratio is at least 5 on manual review;
- no-anchor harmful hints are below 2 percent on manual review;
- wrong-only hints are below 1 percent on replay;
- p95 added prompt context is within chat budget;
- p95 added latency is acceptable for realtime chat;
- Focus and Sensitivity behavior is monotonic and understandable;
- Stability behavior is validated on long-horizon decay replay;
- ambiguous context is phrased as possible continuity, not fact;
- durable fact formation never consumes tentative or ambiguous links.

If no candidate passes, consumption remains out. Formation still stays in.
