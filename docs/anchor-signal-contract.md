# Anchor Signal Contract

This document defines the next input surface required by the anchor
experiments. It is model- and engine-facing, but it is not a production
promotion plan.

The current evidence says Cortext does not need another retrieval reranker over
the same scalar ES/WM/STM/LTM features. It needs a high-precision signal that
can propose entity, object/media, event, and relation continuity during
ingress.

## Role

The signal model or upstream proposal system should answer:

> What entity, object/media, event, and relation evidence is present in this
> incoming signal?

It should not answer:

> Which long-term memory should retrieval return?

The engine owns the final soft-anchor policy. The signal model provides typed
evidence that the engine can attach to working memory, short-term memory,
active anchors, and later retrieval diagnostics.

## Runtime Inputs

The signal model should work from the current incoming modality content and
minimal modality-general metadata:

```text
current_text optional
current_image optional
current_audio optional
modality_id
timestamp / step delta
source bucket optional
```

It should not require Cortext-specific candidate ids, target flags, label
classes, gold actions, entity ids, or retrieval candidates.

## Runtime Outputs

Minimum output:

```json
{
  "semantic_key": [0.0],
  "entity_key": [0.0],
  "object_key": [0.0],
  "event_key": [0.0],
  "salience": 0.0,
  "boundary_score": 0.0,
  "novelty": 0.0
}
```

Proposal output:

```json
{
  "entity_proposals": [
    {
      "proposal_id": "local",
      "kind": "person|place|object|voice|organization|unknown",
      "key": [0.0],
      "confidence": 0.0,
      "span_or_region": null,
      "modality_evidence": ["text"]
    }
  ],
  "object_media_proposals": [
    {
      "proposal_id": "local",
      "kind": "image|video|audio_clip|document|physical_object|unknown",
      "key": [0.0],
      "confidence": 0.0,
      "span_or_region": null,
      "modality_evidence": ["image"]
    }
  ],
  "relation_proposals": [
    {
      "relation_type": "owner|sender|depicts|attached_to|mentions|same_event|unknown",
      "source_proposal_id": "local",
      "target_proposal_id": "local",
      "confidence": 0.0,
      "evidence_key": [0.0],
      "modality_evidence": ["text", "image"]
    }
  ]
}
```

Downstream engines may assign durable track ids after matching proposals to the
active ledger. The signal model should not need durable Cortext track ids to
run.

## Precision Requirement

The noisy entity-track headroom sweep sets the requirement:

- false-track matches around `0.5%` already drop zero-FPR recovery from
  `270 / 270` to `25 / 270` on repaired replay headroom;
- relation edges are even more sensitive because a false owner/sender/depicts
  relation can create a confident wrong-active bind.

Initial gates for a candidate signal artifact:

| gate | required direction |
|---|---|
| entity false proposal rate on no-anchor/wrong-active controls | < 0.5%, target <= 0.1% |
| object/media false proposal rate on no-anchor/wrong-active controls | < 0.5%, target <= 0.1% |
| relation false proposal rate on hard negatives | < 0.5%, target <= 0.1% |
| entity target recall on reference cases | >= 95% after calibration |
| object/media target recall on handoff cases | >= 95% after calibration |
| relation target recall on mixed `he had it` cases | >= 90% after calibration |
| source-held-out zero-FPR recovery with engine policy | nonzero and above ES scalar baselines |
| 5% FPR recovery with engine policy | materially above ES scalar baselines |

The false proposal rates are intentionally strict. The prior experiments showed
that recall without precision is not useful for anchor commitment.

## Evaluation Surfaces

The next artifact should be tested on these surfaces before any engine
promotion:

1. **Runtime smoke**
   - load model/proposal backend;
   - run text, image, and audio inputs if supported;
   - report output dimensions, finite scores, and latency.

2. **Repaired real replay**
   - use the existing 421-case repaired replay;
   - compare proposal-conditioned policy against ES/WM/STM/LTM baselines;
   - report no-anchor, wrong-active, stale, zero-FPR, and source-held-out.

3. **Multimodal chain headroom**
   - person reference;
   - media/object handoff;
   - mixed person-object relation;
   - cross-modal image/audio handoff;
   - no-anchor topic shift.

4. **Manual gold micro-set**
   - 100-300 auditable cases;
   - include hard false relation cases;
   - do not tune thresholds on this set.

5. **Ablations**
   - entity only;
   - entity + object/media;
   - entity + object/media + relation;
   - relation removed;
   - source/recency metadata removed;
   - modality-held-out where data exists.

## Required Artifacts

Recommended output directory:

```text
build/anchor_signal_contract_eval/
  anchor_signal_runtime_smoke.json
  anchor_signal_proposals.jsonl
  anchor_signal_replay_results.json
  anchor_signal_replay_cases.csv
  anchor_signal_chain_results.json
  anchor_signal_false_proposal_audit.csv
  anchor_signal_latency.json
  anchor_signal_failure_examples.csv
```

## Executable Contract Check

The scaffold evaluator lives at:

```bash
python3 tools/evaluate_anchor_signal_contract.py \
  --episodes 200 \
  --output-dir build/anchor_signal_contract_eval
```

The built-in proposal modes are headroom checks, not model evidence. They
verify that the gate logic distinguishes:

- entity-only proposals;
- entity + object/media proposals;
- entity + object/media + relation proposals;
- small false-proposal rates.

Future artifacts can be evaluated with:

```bash
python3 tools/evaluate_anchor_signal_contract.py \
  --cases-jsonl path/to/cases.jsonl \
  --proposals-jsonl path/to/proposals.jsonl \
  --output-dir build/anchor_signal_contract_eval_external
```

The evaluator writes:

- `anchor_signal_contract_eval_cases.jsonl`
- `anchor_signal_proposals_*.jsonl`
- `anchor_signal_contract_results.json`
- `anchor_signal_contract_summary.csv`
- `anchor_signal_false_proposal_audit.csv`
- `anchor_signal_failure_examples.csv`

## Ingress-Only Benchmark Guardrail

The benchmark target below exists specifically to prevent this work from
falling back into retrieval-time candidate scoring. It intentionally has no
non-model, oracle, or scaffold runtime mode. It requires ES-AIST GGUF and tests
the real model signal as an ingress input.

```bash
cmake --build build -j 8 --target cortext_ingress_anchor_formation_bench

CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH=1 \
./build/examples/benchmark/cortext_ingress_anchor_formation_bench \
  --models models \
  --teacher-model models/ES-AIST-81M-preview-GGUF/ES-AIST-81M_q8_0.gguf \
  --max-conversations 96 \
  --max-turns 96 \
  --max-cases 421 \
  --output-dir build/ingress_storage_attention_es_aist
```

The runtime surface is:

```text
current incoming signal
  -> ES-AIST semantic_key/entity_key/full_key
  -> attention over already-stored prior WM/STM/LTM memories
  -> shadow storage-time anchor link/create before current memory write
```

The benchmark does not score retrieved memories. Its audit file writes
`retrieved_candidate_count=0`, `uses_retrieved_candidates=0`, and
`runtime_policy_uses_labels=0` for each step. Evaluation labels are used only
to audit whether the ledger preserved the expected anchor identity; they are
not runtime inputs. The current full run is
`build/ingress_storage_attention_es_aist/ingress_storage_attention_results.json`:
loose storage attention selects `109 / 270` reference targets but abstains on
only `36 / 151` no-anchor controls, while high-precision attention abstains on
`151 / 151` controls but selects only `2 / 270` targets. Zero-FPR recovery
remains `0`, so this storage-time ES signal is not yet a safe anchor policy.
The same run also includes a model-free adaptive ledger. That ledger carries
support, confidence, stability, contradiction, provisional/active state, and
split pressure over time. It confirms that natural online state helps
continuity but not safe commitment: the loose adaptive policy selects
`266 / 270` targets with only `2 / 270` wrong-active selections, but no-anchor
abstention collapses to `0 / 151`; the ultra-precision adaptive policy
abstains on `140 / 151` controls but selects only `5 / 270` targets.

The soft-anchor view is more useful than the hard-commit view. In the same run,
the loose adaptive ledger has the target in tentative top-3 for `270 / 270`
references, and the high-precision ledger has the target in tentative top-3 for
`226 / 270` references. Manual review of real chat rows shows that some
benchmark no-anchor "false binds" are acceptable as tentative conversational
context, while generic matches such as "Ok" or "Bye" must not become durable
anchors.

The benchmark now uses two explicit fields for that distinction:

- `anchor_strength`: the continuous evidence value;
- `anchor_label`: the derived policy view (`none`, `tentative`, `ambiguous`,
  `durable`, `decayed`, or `rejected`).

The latest label pass confirms why the label must remain derived rather than
stored as ground truth. High-precision labels still mark `64 / 151` no-anchor
controls as durable, while ultra-precision cuts that to `3 / 151` by shifting
most selected links into `tentative` or `ambiguous`. The right engine contract
is therefore to persist strength/evidence and treat the label as recalibratable
policy.

The follow-up label/promotion ablation kept the ES signal fixed and changed only
the derived policy view. Conservative high-precision labels reduced no-anchor
durable binds to `45 / 151` without changing target reachability. A high-focus
knob profile reduced no-anchor durable binds further to `16 / 151`, but target
top-3 fell to `168 / 270`; ultra precision reduced no-anchor durable binds to
`3 / 151`, but hard target commits fell to `5 / 270`. This confirms that
current ES-AIST outputs are useful evidence for an engine soft-anchor policy, not a
standalone durable anchor decision.

## Engine Consumption Rule

Signal proposals are evidence, not final anchor actions.

The engine should:

1. match proposals against active WM/STM/LTM anchor state;
2. keep top-k tentative tracks when the evidence is ambiguous;
3. create or update provisional tracks only when proposal confidence and
   ledger consistency are sufficient;
4. promote a track to durable only after repeated support or high-confidence
   relation evidence;
5. treat relation edges as high-precision evidence;
6. abstain from durable binding when proposals are missing or ambiguous;
7. never let a retrieval candidate create an anchor after the fact.

## Soft Anchor v1 Synthesis

The research recommendation matches the experimental direction with one
important framing constraint: ES-AIST is not an anchor model. It is a signal
model. The engine should use it to accumulate evidence, preserve uncertainty,
and derive anchor labels from policy knobs.

The first implementation target should be a benchmark/shadow-only Soft Anchor
Ledger. It should not attempt to resolve a single referent at ingress. Instead,
it should keep a bounded ranked hypothesis set:

```text
SoftAnchorLink {
  memory_id
  anchor_id
  anchor_strength: float [0,1]
  anchor_label: none | tentative | ambiguous | durable | decayed | rejected
  evidence_kind
  memory_tier: WM | STM | LTM
  score
  margin
  entropy
  support_count
  contradiction_count
  created_step
  updated_step
}
```

`AnchorState` should remain an opaque latent continuity object owned by the
engine:

```text
AnchorState {
  anchor_id
  status: provisional | active | durable | decayed | rejected
  semantic_centroid
  entity_centroid
  full_centroid
  semantic_radius
  entity_radius
  source_histogram
  boundary_span
  wm_strength
  stm_strength
  ltm_strength
  support_count
  contradiction_count
  generic_support_count
  first_step
  last_step
  last_boundary_id
  recent_memory_ids
}
```

For each ingress memory, ES-AIST contributes only signal views:

```text
x_sem = semantic_key
x_ent = entity_key
x_full = full_key
```

The engine computes view evidence against active anchors:

```text
s_sem  = map01(cos(x_sem,  a.semantic_centroid))
s_ent  = map01(cos(x_ent,  a.entity_centroid))
s_full = map01(cos(x_full, a.full_centroid))
```

The view blend should be knob-derived:

```text
w_sem  increases with Sensitivity and lower Focus
w_ent  increases with Focus
w_full increases with Stability
```

A starting benchmark blend is:

```text
[w_sem, w_ent, w_full] =
normalize([
  0.35 + 0.20 * (1 - F) + 0.10 * S,
  0.30 + 0.35 * F,
  0.20 + 0.25 * T
])
```

The final link score should combine view evidence with causal ingress state:

```text
score(a) =
  view_score(a)
  + source_weight(F,T) * source_continuity(m,a)
  + recency_weight(S,T) * tier_decay_adjusted_recency(m,a,T)
  + support_weight(T) * support_prior(a,T)
  - boundary_weight(F,T) * boundary_cross_penalty(m,a,F,T)
  - contradiction_weight(S,T) * contradiction_penalty(a,S)
  - generic_weight * generic_turn_score(m)
```

Softmax temperature, retained hypothesis count, durable thresholds, entropy
tolerance, margin tolerance, and decay should all be functions of Focus,
Sensitivity, and Stability. The expected monotonicity is:

| knob change | expected soft-anchor behavior |
|---|---|
| Focus up | fewer tentative anchors, higher margin requirement, lower ambiguity tolerance, higher durable precision, lower tentative recall |
| Sensitivity up | more tentative links, higher target-in-top-k recall, more ambiguous context, durable precision should not materially drop |
| Stability up | slower decay, longer support accumulation, fewer label flips, no cross-boundary durable overreach |

The generic-turn suppressor is valid, but the core suppressor must be
modality-general. Text shortcuts for "ok", "yeah", "bye", "thanks", speaker
role terms, and filler are allowed as one optional source of generic evidence,
not as the whole mechanism. The modality-general suppressor should use null or
generic centroids, low entity mass, low information content, and high
uncertainty.

Durable promotion should be rare. A link can become durable only through:

1. repeated non-generic support with sufficient score, margin, entropy,
   boundary compatibility, and low contradiction;
2. exceptionally strong evidence with low entropy and no competing active
   anchor.

Even durable links are not durable facts. Fact formation remains a separate
conservative layer.

## Soft Anchor v1 Ablation Checklist

The current consolidated Soft Anchor artifact is:

```text
build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/
  ingress_storage_attention_results.json
  soft_anchor_ablation_summary.csv
```

It covers these policy pieces:

| ablation | current coverage | remaining action |
|---|---|---|
| no generic-turn suppressor | high-precision baseline and label runs | keep as control |
| generic-turn suppressor | action-gate ablation improves 5% FPR recovery to `8` but remains unsafe | extend to modality-general null/generic centroid score |
| no repeated-support requirement | loose/high-precision runs exist | keep as control |
| repeated-support promotion | update-gate ablation is too strong: `0 / 270` hard target commits | redesign as durable-promotion gate, not tentative-update gate |
| no boundary/event pressure | baseline runs exist | keep as control |
| boundary/event pressure | event-label guard had no effect | improve boundary feature before relying on it |
| WM only | explicit tier ablation: top-3 `242 / 270`, no-anchor durable `66 / 151` | not sufficient alone |
| WM + STM | explicit tier ablation equals high-precision baseline on this slice | keep as main short-horizon tier mix |
| WM + STM + LTM | full storage attention exists | keep as main path |
| semantic only | explicit view ablation: top-3 `247 / 270`, no-anchor durable `88 / 151` | use as recall view, not durable policy |
| entity only | explicit view ablation: top-3 `215 / 270`, no-anchor durable `43 / 151` | use as safety/durable guard |
| semantic + entity + full | current main path | keep as main path |
| no contradiction penalty | explicit counterfactual improved no-anchor durable to `42 / 151` | current contradiction formula is unreliable |
| no null/no-anchor hypothesis | sticky loose policy and high-precision baseline | keep as control |
| null/no-anchor hypothesis | explicit cutoff lowers hard commits but worsens durable-label quality | redesign as calibrated competing hypothesis |
| no entropy/margin labeling | explicit gate removal increases recall but worsens false binds | keep entropy/margin as safety features |

This checklist is not a request for another learned anchor controller. It is the
completion list for exhausting engine-side policy ablations over the existing
ES-AIST signal.

The main design result is that semantic and entity views have different roles.
Semantic view improves tentative recall but is sticky; entity view is weaker for
reachability but safer for durable commitment. A useful soft anchor should therefore
store separate view evidence and use entity evidence as a promotion guard
rather than collapsing everything into one scalar.

## Soft Anchor v1 Hypothesis Pass

The next benchmark pass implemented the actual hypothesis shape:

```text
build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/
  ingress_storage_attention_results.json
  soft_anchor_v1_update_sweep_summary.csv
  ingress_adaptive_anchor_soft_links.csv
```

Changes:

- explicit `H_none` and `H_new` hypotheses;
- top-k `SoftAnchorLink` output rows;
- `soft_update` for tentative state refresh without hard commitment;
- hard `update_existing` reserved for durable-safe commits;
- durable promotion kept conservative.

Key result:

| policy | target top-3 soft | hard wrong commits | no-anchor durable binds | soft updates | wrong-active distinct |
|---|---:|---:|---:|---:|---:|
| high precision baseline | 226 / 270 | 142 / 421 | 64 / 151 | 0 | 75 / 270 |
| soft anchor v1 | 261 / 270 | 0 / 421 | 0 / 151 | 1408 | 0 / 270 |
| v1 update very strict | 242 / 270 | 0 / 421 | 0 / 151 | 1135 | 51 / 270 |

Interpretation: v1 validates the uncertainty-preserving soft-anchor direction, but
not final subject anchoring. Default soft updates preserve useful continuity
without durable false binds, but they collapse wrong-active subjects into the
same continuity anchor. Stricter soft-update thresholds restore some
wrong-active separation at a recall cost. The next policy work should improve
new/split pressure, not relax durable promotion.

## F/S/T Sweep Result

The first full Soft Anchor knob sweep is:

```text
build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/
  ingress_storage_attention_results.json
  soft_anchor_fst_sweep_summary.csv
  soft_anchor_fst_monotonicity.json
```

All 27 grid points kept durable safety at zero hard/durable false binds. The
knob behavior after the Stability remap is:

| knob | validated behavior | failed or incomplete behavior |
|---|---|---|
| Focus | higher Focus lowers recall and improves wrong-active separation | ambiguous-label count is not yet monotonic |
| Sensitivity | higher Sensitivity improves target top-3 and soft-update volume | higher Sensitivity reduces wrong-active separation |
| Stability | configured TTL increases monotonically; soft-update and create counts are preserved across the short replay | needs a longer decay-heavy replay to prove high Stability preserves continuity after gaps |

Current mapping decision:

- Focus can be used as the precision/split knob.
- Sensitivity can be used as the tentative recall/soft volume knob.
- Stability should control decay, half-life, support-window length, and pruning
  separately from soft-update thresholds. It should not control durable
  promotion until a longer decay-heavy replay passes.

## Stop Conditions

Do not continue with engine policy tuning if:

- entity/object/relation proposals fail the false proposal gates;
- relation proposals cannot distinguish wrong owner/sender/depicted-object
  negatives;
- source-held-out zero-FPR remains zero after proposal-conditioned policy;
- proposal latency is incompatible with ingress.

In those cases, the next work is data/model training, not Cortext policy logic.
