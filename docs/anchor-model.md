# Anchor Model Spec Brainstorm

This document is a working design note for the next Cortext anchor model. It is
not a promotion plan for the current AAIT integration and not a retrieval gate
proposal.

Short-term memory is a system-level prerequisite for this model, not just a
model feature. The canonical STM architecture note is
[`docs/short-term-memory.md`](short-term-memory.md). This file describes how an
anchor controller should consume that layer once it exists.

## Current Conclusion

The anchor work has converged on one architectural lesson:

**Anchors should be formed during ingress, before retrieval, while stream order,
active state, and event continuity are still available.**

Retrieval-side methods repeatedly improved target reachability but failed safe
commitment:

- micro-anchor retrieval
- deterministic anchor slots
- single-step and temporal necessity
- hierarchical/episode attention
- feature forensics
- latent next-embedding rankers
- ColBERT-style retrieval interaction
- pooled, ColBERT, and feature-token ingress ledgers

The common failure was not finding the target. The common failure was deciding
when not to bind:

- no-anchor controls
- wrong-active candidates
- stale same-source candidates
- low-FPR recovery

The AAIT-86M-GGUF integration clarified the next step. Native runtime parity is
good, the candidate-track tensor contract is wired, and strict bind-head
consumer logic prevents no-anchor carryover. The remaining problem is that a
single-step candidate tensor pack is not enough to train robust anchor state.

The ES-AIST + WM/STM/LTM experiments sharpened this further. ES-AIST is useful
as a signal/proposal model, but scalar commit heads, set-aware top-k features,
and small nonlinear commit heads all failed source-held-out safe commitment.
An oracle audit showed that the repaired replay candidate pool already contains
the target for every reference case, so the remaining gap is not memory
availability. It is observable entity continuity.

The noisy entity-track sweep also showed the requirement is precision-first:
high target recall is not enough if false track matches leak into controls or
wrong-active candidates. A false track-match rate around `0.5%` was enough to
collapse zero-FPR recovery on the repaired replay headroom sweep.

The next architecture should add **short-term memory** as an explicit internal
layer. Working memory is the surfaced, curated set. Short-term memory is the
unsurfaced ingress trace that the anchor model can attend over. It should behave
like a shadow DOM for memory: ordered, local, dense, and not user-visible by
default.

## Design Goal

Train a small realtime ingress anchor model that maintains entity/event anchor
state over chronological episodes.

The model should answer:

> Given the current signal and the anchor state that existed before this signal,
> should this signal create, update, split, close, or abstain from anchors?

It should not answer:

> Which retrieved memory is closest to this query?

## Non-Goals

- Do not add another retrieval reranker.
- Do not let retrieval infer anchors after memory pooling.
- Do not rely on raw-token pronoun rules at runtime.
- Do not use label-only fields as runtime features.
- Do not promote any anchor behavior until no-anchor and wrong-active safety
  improve under delayed and source-held-out validation.

## Memory Stack

The anchor model needs three memory layers:

```text
working memory      surfaced, small, high-confidence context
short-term memory   unsurfaced recent ingress trace, dense anchor evidence
long-term memory    durable retrieval store
```

Working memory is what Cortext is willing to show. Short-term memory is what
the ingress model is allowed to attend to. Long-term memory is what retrieval
searches after anchors are already formed.

Short-term memory is not a retrieval result. It is an ordered buffer of recent
signals, memory writes, provisional evidence, weak observations, temporal
context, and anchor-relevant state that may never be surfaced.

## Short-Term Memory Layer

Initial runtime object:

```text
ShortTermMemoryItem {
  stm_id
  source_id
  step_id
  timestamp
  modality
  semantic_vector
  anchor_key
  temporal_context_vector
  evidence_vectors optional
  boundary_score
  drift
  coherence
  surprisal
  salience
  confidence
  anchor_link optional
  surfaced_to_working_memory
  committed_to_long_term
  ttl / expires_at
}
```

Required properties:

- append-only during a single ingress step;
- ordered by stream step;
- bounded by count and/or time;
- queryable by source, time window, and anchor link;
- not surfaced by default;
- available to benchmark and shadow model paths before retrieval.

First Cortext integration sketch:

```text
Signal ingress
  -> trimodal encode current signal
  -> append current signal to STM buffer
  -> run boundary/accumulator updates
  -> run anchor controller over current + WM + STM + anchors
  -> attach shadow anchor links
  -> normal working-memory/retrieval path unchanged
```

For strict chronology, the model snapshot for step `t` should use
`STM_before_t`. The current signal may be represented separately as
`current_signal`, then appended to STM after the anchor decision for future
steps. Benchmark code should export both:

```text
stm_before_step
current_signal
stm_after_step
```

Initial bounds:

```text
raw STM items: 64-128
compressed STM event summaries: 8-16
anchor-linked STM grace window: until anchor closes + short TTL
```

Expiration policy:

- expire ordinary STM by count/time;
- decay faster across strong boundaries;
- retain anchor-linked STM until the anchor has survived or closed;
- compact older STM into event summaries after anchor decisions settle.

Implementation phases before training:

1. Add an in-memory `ShortTermMemoryBuffer` in benchmark/shadow mode.
2. Populate it during chronological replay before retrieval.
3. Export STM snapshots alongside working-memory and anchor snapshots.
4. Audit that STM snapshots contain no future steps or labels.
5. Only then train a model that consumes STM.

## Runtime Architecture

The proposed architecture has four layers.

### 1. Trimodal Encoder

Input:

- text
- image
- audio

Output:

- `semantic_vector`
- `anchor_key`
- optional modality-specific evidence vectors

The semantic vector remains useful for retrieval. The anchor key is for ingress
anchor continuity.

### 2. Short-Term Memory Buffer

Ingress writes each encoded signal into STM before retrieval. STM stores signals
that may be useful for anchoring even if they are not surfaced into working
memory.

The buffer should expose:

```text
Append(signal_embedding, anchor_key, metadata)
GetRecent(max_items, source_filter optional)
GetAnchorLinked(anchor_id, max_items)
Expire(boundary_state, ttl)
Compact()
SnapshotBeforeStep(step_id)
```

For the first implementation this can be benchmark-only and in-memory. It does
not require a durable schema change until the model proves useful.

### 3. Ingress Anchor Controller

The anchor controller runs chronologically during ingress. It consumes the
current encoded signal and the anchor state available before the current step.
It should attend to both surfaced working memory and unsurfaced short-term
memory.

Runtime inputs:

```text
current_semantic: [D]
current_anchor_key: [K]
recent_context_vector: [D]
working_memory_semantic: [W, D]
working_memory_anchor_key: [W, K]
working_memory_features: [W, Fw]
working_memory_mask: [W]
short_term_semantic: [S, D]
short_term_anchor_key: [S, K]
short_term_temporal_context: [S, D]
short_term_features: [S, Fs]
short_term_mask: [S]
active_anchor_state: [A, D or K]
active_anchor_features: [A, Fa]
active_anchor_attached_memory_mask: [A, W + S] optional
active_anchor_mask: [A]
recent_closed_anchor_state: [C, D or K]
recent_closed_anchor_features: [C, Fc]
recent_closed_anchor_mask: [C]
modality_id
source_id or source bucket
time_delta
```

Initial working/short-term memory feature fields:

```text
age
salience
confidence
modality_mask_text
modality_mask_image
modality_mask_audio
last_seen_step
source_continuity
boundary_pressure
drift
coherence
surprisal
surfaced_flag
committed_flag
anchor_link_confidence
```

Initial active-anchor feature fields:

```text
age
salience
confidence
support_count
contradiction_count
last_seen_step
source_continuity
status
```

Features should remain compact and modality-general. Feature scales must be
documented in the dataset schema.

Outputs:

```text
anchor_action_logits:
  0 CREATE_ANCHOR
  1 UPDATE_EXISTING_ANCHOR
  2 SPLIT_ANCHOR
  3 CLOSE_ANCHOR
  4 ABSTAIN

bind_logits:
  active anchor slots 0..A-1
  abstain slot A

anchor_confidence
salience_delta
updated_anchor_key optional
close_target_logits optional
split_parent_logits optional
attention diagnostics optional
```

Strict consumer semantics:

- `CREATE_ANCHOR` creates a new anchor only.
- `UPDATE_EXISTING_ANCHOR` consumes an existing anchor only when bind logits
  choose a non-abstain valid candidate.
- `SPLIT_ANCHOR` creates a new anchor and records a parent/split relation if
  available.
- `CLOSE_ANCHOR` deactivates a selected stale anchor.
- `ABSTAIN` attaches nothing.

No fallback pooled cosine should override these outputs in the primary path.

Model shape:

```text
current token
working-memory tokens[W]
short-term-memory tokens[S]
active-anchor tokens[A]
closed-anchor tokens[C]
metadata tokens
```

Use typed embeddings for token roles:

```text
CURRENT
WORKING_MEMORY
SHORT_TERM_MEMORY
ACTIVE_ANCHOR
CLOSED_ANCHOR
TEMPORAL_CONTEXT
SOURCE
MODALITY
BOUNDARY
```

The model should bind over anchors, not directly over memories. Memories and
STM items are evidence. Anchors are state.

Initial target bounds:

```text
W = 8-16
S = 32-128
A = 8-16
C = 4-16
```

### 4. Retrieval Consumer

Retrieval consumes established anchors as context. It should not invent anchors.

Allowed retrieval-time use:

- prefer memories linked to the active anchor when the ingress ledger has
  already committed one;
- avoid stale/closed anchors;
- expose anchor state as a shadow diagnostic until promotion gates pass.

## Training Unit

The training unit should be an episode, not an independent row.

```json
{
  "episode_id": "string",
  "source_id": "string",
  "dataset": "string",
  "modality_mix": ["text", "image", "audio"],
  "steps": [
    {
      "step_id": 0,
      "timestamp": 0,
      "modality": "text",
      "runtime_input": {
        "current_semantic": [],
        "current_anchor_key": [],
        "recent_context_vector": [],
        "working_memory": [],
        "short_term_memory": [],
        "active_anchors": [],
        "recent_closed_anchors": [],
        "masks": {}
      },
      "labels": {
        "gold_action": "CREATE_ANCHOR",
        "gold_bind": "abstain",
        "gold_track_id": "track_1",
        "gold_entity_id": "entity_1",
        "gold_split_parent": null,
        "gold_close_target": null,
        "reference_type": "explicit_entity",
        "failure_type": "create_new_entity"
      }
    }
  ]
}
```

The `runtime_input` object is the only runtime surface. Labels are training and
audit fields only.

The episode file must also preserve enough source-of-truth structure to rebuild
runtime snapshots deterministically:

- chronological steps;
- working-memory membership before each step;
- short-term memory membership before each step;
- active and recently closed anchors before each step;
- anchor links created by earlier steps;
- source and dataset split keys.

Step tensor JSONL is a derived artifact. The episode JSONL is the source of
truth.

## Required Labels

Minimum labels:

- `gold_action`
- `gold_bind`
- `gold_track_id`
- `gold_split_parent`
- `gold_close_target`
- `reference_type`
- `failure_type`
- `track_active_before_step`
- `track_active_after_step`

Reference types:

- `explicit_entity`
- `weak_reference`
- `attribute_reference`
- `continuation`
- `topic_shift`
- `no_anchor`

Failure types:

- `no_anchor_no_candidates`
- `no_anchor_tempting_active`
- `wrong_active_more_recent`
- `wrong_active_higher_similarity`
- `wrong_active_same_source`
- `stale_same_source_close`
- `stale_semantically_close`
- `delayed_reference_2_4`
- `delayed_reference_5_12`
- `split_related_new_track`
- `remote_easy_negative`

## Dataset Plan

Do not train the next model until the STM architecture can be exported from
replay. The dataset must represent the model's actual runtime context:

```text
current signal
+ working-memory snapshot before step
+ short-term-memory snapshot before step
+ active-anchor snapshot before step
+ recently closed-anchor snapshot before step
```

The older candidate-track tensor pack is useful for black-box head repair, but
it is not enough for the next architecture because it does not expose the full
local evidence field.

### Tier 1: Synthetic V4 Episode Tracks

Purpose: teach the state machine cleanly.

Required families:

- text-only entity continuity
- cross-modal entity continuity
- no-anchor with tempting active anchors
- wrong-active more recent than target
- wrong-active higher cosine than target
- stale same-source semantically close
- split related-but-distinct track
- close stale/old anchors
- delayed references at distances 2-4 and 5-12
- attribute references
- media/object handoff (`it` refers to prior image/object)
- mixed person-object relations (`he had it`)
- relation false positives: wrong owner, wrong sender, wrong depicted object

Synthetic generation may use text templates offline, but runtime inputs remain
embeddings and compact metadata.

### Tier 2: GPT-Labeled Real Episodes

Purpose: teach realistic dialogue ambiguity and deployment distribution.

Use GPT-5.4-mini or stronger to label whole episodes, not isolated rows. The
teacher should produce track-level annotations:

- entity/event tracks
- step action
- bind target
- no-anchor abstentions
- split/close events
- rationale for audit only

The model never receives teacher rationale at runtime.

### Tier 3: Manual Gold Micro-Set

Purpose: final human-readable sanity gate.

Size target: 100-300 cases.

Must include:

- no-anchor controls
- wrong-active controls
- stale same-source controls
- delayed references
- split events
- close events
- cross-modal references if available

This should be used for evaluation, not bulk training.

## Entity / Object / Relation Contract

The motivating product behavior is not only a person-reference problem:

```text
User: Hey just got off the phone with Jared
Assistant: Great, what did he say?          -> he = Jared
User: He sent me this image of his dog      -> he = Jared, dog/image introduced
Assistant: Can you send it to me?           -> it = image/dog media object
User: Sure, here you go                     -> continuation = same image
Assistant: How long has he had it?          -> he = Jared, it = dog/image
```

This chain requires at least three kinds of anchor state:

```text
entity/person track      Jared
object/media track       dog / image
relation/event edge      owner(Jared, dog), sender(Jared, image), depicts(image, dog)
```

The multimodal chain headroom audit confirmed the hierarchy:

| upstream state | what it solves | what still fails |
|---|---|---|
| recency only | almost nothing safely | wrong-active dominates |
| entity/person tracks | `he` person references | `it`, image/object handoff, owner/object questions |
| entity + object/media tracks | image/object handoff | mixed `he had it` relation questions |
| entity + object/media + relation edges | full synthetic chain | depends on very low false relation rate |

Therefore, the next anchor model should not expose only one `anchor_key`.
It should expose either multiple geometry views or explicit proposal heads for:

```text
entity_key          same person/place/object identity
object_key          same physical/digital object or media artifact
event_key           same episode/scene/conversation segment
relation_key/head   owner/sender/depicts/attached-to/mentions links
```

For downstream engines, relation proposals should be optional but first-class:

```json
{
  "relation_type": "owner|sender|depicts|attached_to|mentions|same_event",
  "source_track": "track_person_jared",
  "target_track": "track_object_dog",
  "evidence_step": 12,
  "confidence": 0.0,
  "modality_evidence": ["text", "image", "audio"]
}
```

Runtime consumers must treat relation edges as high-precision evidence, not as
soft retrieval decorations. A false relation edge can be worse than a missing
edge because it creates a confident wrong-active bind.

## Training Objectives

Primary objective: maintain correct anchor ledger state over time.

Losses:

- action cross-entropy
- bind cross-entropy over active anchors + abstain
- no-anchor abstain loss
- wrong-active contrastive margin
- stale close/reject loss
- split-vs-update margin loss
- track consistency loss across rollout
- close-target loss
- STM-to-anchor evidence consistency loss
- optional low-weight next-embedding prediction

The next-embedding prediction loss should stay low weight. Earlier experiments
showed it improves target reachability but does not solve commitment.

## Rollout Training

Teacher-forced single-step training is not enough. The model should also be
trained/evaluated in rollout mode:

1. Start with an empty ledger.
2. Process episode steps chronologically.
3. Apply predicted actions to a shadow ledger.
4. Penalize downstream ledger divergence.

Rollout metrics:

- anchor merge error rate
- anchor split error rate
- no-anchor false carryover
- stale anchor survival
- wrong-active merge rate
- target track available before weak reference
- STM evidence used before surfacing
- STM source-continuity overbinding rate

## Short-Term Memory Validation

Before model training, validate the STM layer by itself.

Required audits:

- STM contains only prior steps before each current step.
- STM contains no labels, target flags, entity ids, or gold actions as runtime
  inputs.
- STM coverage includes weak observations not surfaced to working memory.
- STM retains recent wrong-active and stale evidence long enough for the anchor
  controller to reject it.
- STM expiration does not remove the target evidence before delayed references.
- STM does not simply mirror working memory.
- STM does not include retrieval candidates constructed after the current step.

Required metrics:

- mean/p50/p95 STM length;
- fraction of reference cases with target evidence in STM before current step;
- fraction of wrong-active cases with wrong-active evidence in STM before
  current step;
- fraction of no-anchor cases with tempting STM evidence;
- delayed-reference target-retention rate for distances 2-4 and 5-12;
- stale-evidence inactive/closed rate;
- STM/WM overlap rate.

Failure conditions:

- target evidence is usually absent from STM before weak references;
- no-anchor controls have no tempting STM evidence, making them too easy;
- STM contains future steps;
- STM becomes identical to working memory;
- STM is dominated by source continuity features without entity/event evidence.

## Validation Gates

Do not treat target AUC alone as success.

Required metrics:

- target anchor existed before current step
- target top-1/top-3
- no-anchor false-bind risk
- wrong-active rejection AUC
- stale rejection AUC
- split correctness
- close correctness
- zero-FPR recovery
- 5% FPR recovery
- delayed/prequential validation
- source-held-out validation
- dataset-held-out validation
- mean/p95 ingress overhead
- STM attention mass on target evidence
- STM attention mass on wrong-active/stale evidence
- STM ablation delta

Promotion criteria:

- no-anchor risk improves beyond chance;
- wrong-active rejection improves materially;
- low-FPR recovery becomes nonzero and beats retrieval-only baselines;
- target reachability does not collapse;
- source-held-out validation does not collapse;
- ingress overhead remains realtime-compatible.

## Export Layout

Recommended export:

```text
anchor_episode_v1/
  anchor_episodes_train.jsonl
  anchor_episodes_val.jsonl
  anchor_episodes_test.jsonl
  anchor_step_tensors_train.jsonl
  anchor_step_tensors_val.jsonl
  anchor_step_tensors_test.jsonl
  short_term_memory_snapshots_train.jsonl
  short_term_memory_snapshots_val.jsonl
  short_term_memory_snapshots_test.jsonl
  schema.json
  label_audit.json
  stm_audit.json
  split_audit.json
  failure_slices.json
  manual_gold_cases.jsonl
```

The episode files are the source of truth. Step tensors are derived artifacts.

## Open Questions

1. Should `anchor_key` be trained with an explicit same-track contrastive loss,
   or should the controller learn continuity entirely through action/bind loss?
2. Should `SPLIT_ANCHOR` bind a parent track, or should it always use the
   abstain bind slot plus an optional split-parent head?
3. Should `CLOSE_ANCHOR` be driven by the current signal, by passive decay, or
   both?
4. How much source continuity should be exposed? Too much source signal risks
   same-source continuity anchors rather than entity anchors.
5. Should candidate order be randomized at training time always, or should we
   mix random, recency, semantic, and adversarial orders?
6. Should the model have an internal recurrent state, or should the external
   ledger plus candidate states be sufficient?
7. What is the smallest manual gold set that catches no-anchor and wrong-active
   failures before expensive training?
8. Should STM be persisted durably, or remain an in-memory/benchmark layer until
   the anchor controller proves useful?
9. Should STM store raw evidence vectors per signal, compressed event summaries,
   or both?
10. How should STM represent audio/image evidence when the user-visible memory is
   text-only or summary-only?
11. Should retrieval ever read STM, or should STM be reserved for ingress,
   consolidation, and anchor formation?

## Recommended Next Step

Add a benchmark-only `ShortTermMemoryBuffer` first. It should populate during
chronological replay, export snapshots before each current ingress step, and
pass the STM validation audits above. After that, build `anchor_episode_v1` as
an episode-first dataset with STM snapshots included.

The current 5k tensor pack is useful as a black-box deployment-shape repair
scaffold, but it should not be the primary source of truth for the next model.
The next model should be trained from explicit entity/event track episodes and
exported into a working-memory + short-term-memory attention contract for
Cortext ingress.
