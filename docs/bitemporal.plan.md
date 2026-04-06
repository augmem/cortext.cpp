# Bitemporal Fact Modeling Plan

## Purpose

This document captures:

- what Cortext should add to the paper about bitemporal fact modeling
- why bitemporal facts matter for dementia support
- how bitemporal facts should influence retrieval and recall within the F/S/T knob system
- which ablation tests should be run before claiming retrieval or recall improvements

## Core idea

Bitemporal modeling stores two timelines for a fact:

- `valid time`: when the fact was true in the world
- `transaction time`: when Cortext learned, recorded, or revised the fact

This is meaningful because many dementia-related memory failures are not false facts. They are true facts from the wrong time.

Examples:

- someone who used to drive them is no longer the current driver
- an old home address is remembered as current
- a prior routine is recalled even though today is different

Cortext should be able to represent:

- what is true now
- what used to be true
- what Cortext would have believed at an earlier time

## Why this matters for dementia support

For dementia support, bitemporal facts improve:

- `orientation`: who is here, where they are, what happens next
- `safe correction`: old truths are treated as outdated rather than simply wrong
- `autobiographical continuity`: past facts remain available for recall instead of being overwritten
- `reduced frustration`: the system can redirect gently instead of bluntly contradicting
- `caregiver trust`: the system can justify why it believes a current fact

The main benefit is that the system can distinguish:

- `current truth`
- `historical truth`
- `historical belief`

This is directly useful for gentle nudges such as:

- "Emily is picking you up today"
- "You used to go there on Tuesdays, but today the appointment is at the clinic"
- "Sarah helped with that before; today it is Emily"

## Paper additions

### 1. Mathematical foundations

Add a subsection to `docs/paper/sections/1_math_foundations.qmd` defining:

- `event time` for episodic memories
- `valid time` for extracted structured facts
- `transaction time` for Cortext's learned belief state

Add simple notation for:

- episodic memory capture time
- fact valid interval
- fact transaction interval

Add explicit query semantics:

- `current`
- `valid_at(t)`
- `known_at(t)`

### 2. Consolidation

Add a subsection to `docs/paper/sections/7_consolidation.qmd` explaining that consolidation should not overwrite changing personal facts.

State that consolidation should:

- append new fact assertions
- preserve prior assertions
- mark supersession when a newer fact replaces an older one
- retain provenance to summary and episodic evidence

Connect this to dementia support scenarios:

- caregiver changes
- routine changes
- appointment changes
- location changes
- medication or care-plan changes

### 3. Experimental section

Add to `docs/paper/sections/9_experimental.qmd` a planned evaluation section covering:

- current fact accuracy
- historical fact accuracy
- belief-at-time accuracy
- stale fact intrusion rate
- correction quality for gentle nudges
- retrieval precision and recall under temporal change

The paper should be explicit that these are retrieval and assistance metrics, not only database correctness metrics.

### 4. Implementation section

Add to `docs/paper/sections/10_implementation.qmd` that:

- alpha Cortext remains centered on episodic memory, summaries, and embedding-based retrieval
- v1 adds a structured bitemporal fact layer on top of the existing memory system
- this layer is additive, not a replacement for episodic or summary memories
- the live loop may remain embedding-first while structured facts contribute retrieval signals, recall context, and nudge grounding

## Retrieval and recall design

The design rule is:

- fact storage should be objective
- fact influence should be behaviorally tuned

That means the fact table stores history faithfully, while the F/S/T knob system determines how strongly facts affect retrieval, recall, and nudging.

### Query modes

Retrieval and recall should use three temporal modes:

- `current`: what is true now
- `valid_at(t)`: what was true in the world at time `t`
- `known_at(t)`: what Cortext would have believed at time `t`

### Retrieval parameters that should matter

When facts contribute to retrieval or recall, ranking should be driven by:

- temporal mode
- fact criticality
- support strength
- current validity
- routine strength
- recency of confirmation
- subject or predicate specificity
- provenance strength

#### Temporal mode

- `current` should dominate orientation and task-support queries
- `valid_at(t)` should dominate historical recall
- `known_at(t)` should be used when reconstructing past belief state or explaining prior system behavior

#### Fact criticality

Suggested priority order:

- highest: medication, caregiver, location, schedule, safety
- medium: routine, household context
- lower: preferences, incidental details

#### Support strength

Support strength should include:

- extraction confidence
- number of confirmations
- diversity of evidence sources
- direct episodic evidence vs summary-only evidence

#### Current validity

For present-oriented queries:

- active current facts should receive a strong boost
- superseded facts should receive a strong penalty unless the query is explicitly historical

#### Routine strength

Repeated, stable facts should exert more retrieval influence because routines are grounding for people with dementia.

#### Recency of confirmation

Recent confirmations matter, but should not overwhelm a stable, well-supported fact unless the new change is strongly supported.

#### Specificity and provenance

Exact subject and predicate matches should matter heavily to avoid retrieving the wrong person, place, or activity.

Facts should boost memories that actually support them, not merely semantically similar memories.

## Dementia-oriented retrieval modes

### Orientation mode

Use when the user implicitly means "now", for example:

- who is here
- where are we going
- what is happening today
- who is helping me

Behavior:

- prefer `current` facts
- strongly penalize superseded facts
- inject a small number of high-confidence grounding facts
- prioritize safety- and schedule-critical facts

### Recall mode

Use when the user is reminiscing or explicitly asking about the past.

Behavior:

- prefer `valid_at(t)` plus temporally nearby episodic memories
- reduce penalty on older facts
- allow broader associative recall
- use current facts only as soft guardrails

## Mapping to the F/S/T knob system

Facts should be knob-neutral at rest and knob-shaped in use.

### Focus

`Focus` controls selectivity.

It should primarily affect:

- how narrowly facts bias retrieval
- how many facts are injected into working memory
- how strongly stale facts are penalized for current queries

High `Focus`:

- fewer facts
- stronger exact-match boost
- stronger stale-fact penalty

Low `Focus`:

- broader contextual recall
- more associative support
- weaker penalties on temporally nearby supporting memories

### Sensitivity

`Sensitivity` controls plasticity.

It should primarily affect:

- fact admission thresholds
- supersession thresholds
- how quickly new evidence changes the active fact state

High `Sensitivity`:

- faster updates from new evidence
- faster supersession
- quicker adaptation to today's changes

Low `Sensitivity`:

- more conservative updates
- less flip-flopping
- stronger protection for stable identity and routine facts

### Stability

`Stability` controls persistence.

It should primarily affect:

- how long fact-backed memories remain salient
- how strongly stable routines influence retrieval
- how much continuity appears in summaries and nudges

High `Stability`:

- durable routines stay behaviorally important
- repeated facts decay slowly in influence
- continuity is emphasized

Low `Stability`:

- retrieval leans more on fresh episodic evidence
- old fact-backed context loses influence faster

## Suggested default parameter posture for dementia support

These should not be treated as final constants, but as starting points:

- `identity facts`: low `Sensitivity`, high `Stability`
- `routine facts`: medium `Sensitivity`, high `Stability`
- `schedule or location today`: higher `Sensitivity`, medium `Stability`
- `caregiver today`: higher `Sensitivity`, highest criticality
- `preferences`: medium `Sensitivity`, medium `Stability`

Overall system posture for dementia support should likely favor:

- medium-high `Focus`
- medium-low `Sensitivity` globally, with higher values for schedule and location changes
- high `Stability`

## Retrieval scoring direction

When facts are later integrated into retrieval, the score should conceptually follow:

`memory_score = base_retrieval + fact_boost - stale_penalty`

Where `fact_boost` is strongest when:

- the fact is current
- the fact is safety- or orientation-critical
- the fact has strong support
- the fact exactly matches the subject and predicate implied by the query
- the candidate memory is direct evidence for that fact

## Planned ablation tests

The goal of ablations is to determine whether bitemporal facts improve assistive retrieval and recall rather than merely increasing structured metadata.

### A. Fact layer vs no fact layer

Compare:

- baseline episodic plus summary retrieval
- episodic plus summary plus fact retrieval signals

Measure:

- current fact recall accuracy
- historical recall accuracy
- orientation success
- stale fact intrusion rate
- latency overhead

### B. Bitemporal vs current-only facts

Compare:

- facts with only a single current canonical state
- facts with valid-time and transaction-time history

Measure:

- historical fact accuracy
- contradiction handling
- ability to answer `what was true then`
- ability to answer `what would Cortext have known then`

### C. Query mode ablation

Compare retrieval under:

- `current`
- `valid_at(t)`
- `known_at(t)`

Measure:

- accuracy for present-oriented prompts
- accuracy for historical prompts
- accuracy for past-belief prompts
- user-facing correction quality

### D. Fact boost ablation

Compare:

- no fact boost
- weak fact boost
- strong fact boost

Measure:

- retrieval precision
- retrieval recall
- stale fact suppression
- overconstraint failures where retrieval becomes too narrow

### E. Superseded fact penalty ablation

Compare:

- no penalty
- moderate penalty
- strong penalty

Measure:

- present-oriented confusion rate
- incorrect resurfacing of outdated caregivers, locations, schedules, or routines
- failure rate on explicitly historical queries

### F. Provenance ablation

Compare:

- fact boost from any fact match
- fact boost only when the candidate memory is provenance-linked evidence

Measure:

- precision of retrieved supporting memories
- hallucinated support rate
- nudge justification quality

### G. Routine vs recency ablation

Compare different weighting between:

- routine strength
- recency of confirmation

Measure:

- stability on durable routines
- adaptation to legitimate schedule changes
- flip-flop rate under noisy evidence

### H. F/S/T sensitivity analysis

Run sweeps across:

- low / medium / high `Focus`
- low / medium / high `Sensitivity`
- low / medium / high `Stability`

Measure:

- current fact accuracy
- historical fact accuracy
- retrieval precision and recall
- stale fact intrusion
- correction gentleness
- latency

This should validate that the fact system is being governed by the same three knobs rather than introducing a disconnected control surface.

### I. Delayed evidence ablation

Compare scenarios where the system learns the true state:

- immediately
- after a short delay
- after a long delay

Measure:

- `known_at(t)` accuracy
- time-to-correct-current-state
- stale fact intrusion during the delay window
- degradation in orientation-mode retrieval before correction

This is critical because assistive systems will often receive incomplete or late evidence from real wearable or caregiver pipelines.

### J. Conflicting and noisy evidence ablation

Compare:

- clean evidence only
- conflicting evidence from multiple summaries
- noisy extraction with false positives
- mixed delayed and conflicting evidence

Measure:

- fact flip-flop rate
- erroneous supersession rate
- recovery time after false updates
- current fact accuracy under noise

This tests whether the system remains stable instead of oscillating between competing facts.

## Fact lifecycle and pruning

The current bitemporal design is correctness-first and history-preserving, but it should not remain an unbounded append log forever. The long-term design should distinguish:

- `world invalid`: the fact is no longer true in the world
- `historically retained`: the fact is no longer current, but is still needed for `valid_at(t)` or `known_at(t)`
- `weakly supported`: the system has little recent or diverse evidence for the fact
- `unsupported`: the fact has lost enough support that it should no longer strongly influence behavior
- `archived`: the fact is retained only for historical reconstruction or offline analysis

Facts therefore should not be treated as immortal, but they also should not be deleted simply because their source episode decayed. The desired lifecycle is:

- episodic evidence may decay faster
- semantic facts should survive longer than individual episodes
- repeated confirmations should strengthen a stable fact rather than endlessly duplicating it
- unsupported low-value facts should eventually weaken, archive, or be pruned
- high-severity facts should retain longer history and stronger archival guarantees

### Proposed support model

Each fact should eventually track more than current-vs-superseded state. The lifecycle model should include:

- support mass
- support recency
- source diversity
- contradiction mass
- last confirmation timestamp
- last challenge timestamp
- retrieval/use frequency
- severity class

These variables should drive whether a fact remains:

- behaviorally active
- weak but retained
- archive-only
- eligible for deletion

### Biological modeling direction

The intended biological analogy is:

- episodic traces are fragile, detailed, and decay faster
- semantic facts are slower, more abstract, and survive repeated consolidation
- repeated similar episodes strengthen a semantic belief
- contradiction revises semantic state without instantly erasing historical truth

In practice this means:

- memories remain the raw evidence substrate
- facts are semantic consolidations over those memories
- fact pruning should depend on support, reuse, contradiction, and severity rather than age alone

### Proposed pruning policy direction

The future fact pruning/archival system should likely:

- keep all current active facts
- keep high-severity superseded facts much longer than low-severity ones
- compress repeated confirmations into support statistics rather than preserving endless duplicates
- archive old low-value facts once they are both unsupported and behaviorally irrelevant
- delete only facts that are low-severity, long-unused, weakly supported, and no longer needed for historical reconstruction

The critical rule is:

- loss of evidence should reduce fact support and influence
- loss of evidence should not automatically make a fact false

## Planned pruning tests and ablations

The fact lifecycle needs its own evaluation program, separate from the existing retrieval-only ablations.

### K. Fact support decay ablation

Compare facts under:

- stable repeated confirmation
- no new confirmation
- contradictory updates
- decayed or pruned episodic evidence

Measure:

- active fact survival
- support-score decay
- retrieval influence decay
- false invalidation rate

### L. Fact archive vs delete ablation

Compare policies:

- no pruning
- archive-only for unsupported facts
- archive plus low-severity deletion
- aggressive deletion

Measure:

- storage growth
- current fact accuracy
- `valid_at(t)` accuracy
- `known_at(t)` accuracy
- historical recall loss

### M. Repeated-confirmation compression ablation

Compare:

- append every confirmation
- merge confirmations into support counts
- merge confirmations into support counts plus archival snapshots

Measure:

- storage growth rate
- routine persistence quality
- retrieval stability
- loss of historical fidelity

### N. Evidence-loss resilience ablation

Compare scenarios where the underlying episodic evidence is:

- fully retained
- partially pruned
- heavily pruned after consolidation

Measure:

- whether high-support facts remain usable
- whether unsupported facts weaken appropriately
- whether high-severity facts remain reconstructable
- mismatch between fact confidence and evidence availability

### O. Severity-aware fact retention ablation

Compare retention policies across:

- medication, caregiver, location, schedule, safety
- routine and household context
- preferences and incidental details

Measure:

- storage saved by tiered retention
- harmful forgetting rate
- current-state accuracy under pruning
- historical reconstruction quality by severity class

## Additional metric requirements

The ablations above are not sufficient on their own unless the metrics reflect assistive importance.

### Severity-weighted metrics

Errors should be weighted by impact class rather than averaged uniformly.

Suggested classes:

- highest severity: medication, caregiver, location, schedule, safety
- medium severity: household context, routine, social context
- lower severity: preference, incidental detail

Report:

- weighted current fact accuracy
- weighted stale fact intrusion
- weighted correction failure rate

This prevents the system from appearing good overall while still making the most harmful mistakes.

### Longitudinal stability metrics

Because dementia support is a long-horizon use case, evaluation should include multi-day or multi-week temporal simulations.

Report:

- fact flip rate over time
- stale resurfacing rate
- time-to-correct-current-state
- persistence of stable routines
- regression after legitimate fact changes

The key question is not only whether the system answers correctly once, but whether it stays behaviorally stable over time.

### Assistive outcome metrics

Measure outcomes at the nudge and recall layer, not only raw retrieval ranking.

Report:

- orientation success after a nudge
- contradiction rate
- gentle correction rate
- outdated-first retrieval rate
- provenance-grounded explanation quality

These metrics should determine whether the retrieval behavior is actually helpful for a person with memory loss.

## Evaluation scenarios

The ablations should include scenarios with controlled temporal change, such as:

- caregiver transitions
- schedule changes on the same day
- location changes
- repeated routines with one-off exceptions
- old home vs current home
- old preference vs updated preference

Each scenario should support:

- a current query
- a historical query
- a past-belief query

Each scenario should also be tested under:

- immediate evidence
- delayed evidence
- conflicting evidence
- noisy extraction

## Core-first implementation order

The implementation order should prioritize proving the core fact system under
stress before any chat-path integration.

### 1. Core fact integrity

First complete and harden:

- fact assertion persistence
- supersession behavior
- `current` / `valid_at(t)` / `known_at(t)` semantics
- provenance linkage from facts to supporting memories

This stage should be treated as storage and query correctness only.

### 2. Core retrieval integration

Next integrate facts into retrieval and memory scoring:

- fact-aware candidate seeding
- provenance-linked memory boosts
- stale-fact penalties for present-oriented retrieval
- F/S/T-shaped retrieval influence

This stage should remain internal to retrieval and benchmark harnesses.

### 3. Core robustness and proof

Before touching chat behavior, the fact system should be stress-tested until it
is stable under:

- delayed evidence
- conflicting evidence
- noisy extraction
- repeated legitimate changes over long horizons

This stage should produce hard acceptance evidence for:

- stale fact intrusion
- fact flip-flop rate
- erroneous supersession rate
- time-to-correct-current-state
- weighted current fact accuracy
- longitudinal routine stability
- latency overhead

Only once these core metrics are consistently strong should the roadmap move on
to prompt grounding or reply behavior.

### 4. Chat-facing integration last

Chat-path changes should come only after the core is proven robust. That
includes:

- prompt grounding
- assistive reply shaping
- current-vs-usual phrasing
- nudge wording and explanation style

Those layers should consume an already-proven core rather than be used to mask
uncertainty in the fact system itself.

## Deferred chat-facing assistive evaluation scenarios

In addition to generic retrieval scenarios, the plan should include assistive tasks such as:

- current caregiver reminder
- today vs usual schedule clarification
- current location or destination cue
- routine reminder with one-off exception
- autobiographical recall of a past home, trip, or family event

For each scenario, evaluate whether Cortext:

- surfaces the correct current fact when needed
- avoids blunt contradiction
- preserves older truths for recall
- avoids resurfacing superseded facts in present-oriented nudges
- grounds the response in real supporting evidence

These scenarios should be run only after the core retrieval and robustness
gates above are satisfied.

## Success criteria

The bitemporal extension is successful if it improves:

- current orientation accuracy
- historical recall accuracy
- suppression of stale facts on present-oriented tasks
- gentle correction behavior
- provenance-grounded retrieval
- robustness under delayed and conflicting evidence
- longitudinal stability over repeated updates

without causing unacceptable regressions in:

- recall breadth
- latency
- stability under noisy evidence

## Next implementation implications

Once the paper and evaluation plan are in place, the next implementation phase should:

- harden fact-aware retrieval under delayed, conflicting, and noisy evidence
- add severity-weighted and longitudinal stability benchmarks as release gates
- prove `current` / `valid_at(t)` / `known_at(t)` behavior under repeated fact changes
- expose the effect of F/S/T on fact-driven retrieval in benchmarks
- defer all chat-path grounding and reply behavior until the core passes those gates
