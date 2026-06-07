# Anchor Experiment Audit

This audit summarizes the current state of the modality-agnostic anchor work.
It is not a promotion plan. Production retrieval remains unchanged.

## Objective

Build enough evidence to decide how Cortext should anchor incoming episodes by
entity/person/place/object/media/event continuity, including chains like:

```text
Jared -> he -> image of dog -> it -> he had it
```

The system must eventually support:

- person/entity references across turns and modalities;
- object/media references such as image or dog continuity;
- relation/event references such as owner/sender/depicts links;
- no-anchor abstention;
- wrong-active rejection;
- stale same-source rejection;
- low-FPR recovery under delayed and source-held-out validation.

## Completion Criteria for This Experiment Pass

This experiment pass is complete only if the available surfaces answer these
questions:

1. Can current retrieval-side scores solve the anchor problem?
2. Can ES-AIST embeddings be used as a standalone anchor policy?
3. Can ES-AIST + WM/STM/LTM attention propose useful candidates?
4. Can a safety/commit policy over those features produce low-FPR binds?
5. Is the repaired candidate pool missing the targets?
6. Is a precise entity/object/relation signal sufficient in headroom?
7. Can the current ES-AIST scalar signal be thresholded into that precise
   proposal signal?
8. What concrete input surface is required before another engine policy pass is
   worth running?

The answer is now complete for the current artifact set. This is not a
production promotion: the engine still lacks the required high-precision
signal.

## Prompt-to-Artifact Checklist

| requirement | evidence | status |
|---|---|---|
| Production retrieval must not change | all ES/STM/chain artifacts are shadow/headroom only; benchmark JSON records `production_retrieval_changed=false` where applicable | covered |
| Test whether ES-AIST is useful as signal, not standalone policy | `build/es_aist_contextual_anchor_shadow_sequence_mlp/es_aist_attention_retrieval_stage1.csv` | covered |
| Separate proposal from commit | `es_aist_safety_commit_stage2.csv`; two-stage docs in `docs/experiments.md` and paper section 9 | covered |
| Test scalar safety head | source-held-out single safety head: top-3 `0.1296`, zero-FPR `0` | covered, failed |
| Test multi-head safety | multi-head bind-only: top-3 `0.3593`, zero-FPR `1`, wrong-active commits `0.0519` | covered, unsafe |
| Test set-aware top-k safety | set-aware bind-only: top-3 `0.3333`, zero-FPR `1`, 5% FPR `4`, wrong-active commits `0.0296` | covered, insufficient |
| Test compact nonlinear top-k consumer | set-aware MLP bind-only: top-3 `0.4185`, zero-FPR `0`, wrong-active commits `0.0593` | covered, unsafe |
| Check if target memories are missing from candidate pool | `entity_track_oracle_headroom.json`: `270 / 270` references have target candidate, `0` controls contain target | covered |
| Quantify needed entity-track signal quality | `entity_track_signal_quality_sweep.json`: `0.5%` false-track rate drops zero-FPR recovery to `25 / 270` | covered |
| Test whether entity tracks alone solve product chain | `multimodal_anchor_chain_headroom.json`: entity-only safe target `0.4000`, relation top-1 `0.0000` | covered, failed |
| Test whether object/media tracks are needed | chain audit: entity+object/media safe target `0.8000`, media top-1 `1.0000`, relation top-1 `0.0000` | covered |
| Test whether relation/event edges are needed | chain audit: entity+object/media+relation safe target `1.0000`, wrong-active `0.0000`, relation top-1 `1.0000` | covered |
| Define executable signal contract gates | `build/anchor_signal_contract_eval/anchor_signal_contract_summary.csv` distinguishes entity-only, entity+object/media, relation-complete, and noisy proposal modes | covered |
| Test whether current ES-AIST scalar signals can satisfy proposal gates | `build/es_aist_signal_contract_threshold_audit/es_aist_signal_contract_threshold_results.json`: best `<0.005` false-candidate gate selects `8 / 270` targets; best `<=0.001` gate selects `1 / 270` | covered, failed |
| Add an ES-AIST ingress/storage benchmark that cannot use retrieved candidates | `build/ingress_storage_attention_es_aist/ingress_storage_attention_cases.csv` records `retrieved_candidate_count=0`, `uses_retrieved_candidates=0`, and `runtime_policy_uses_labels=0`; C++ target `cortext_ingress_anchor_formation_bench` now defaults to real chronological storage-time attention over prior memories | covered |
| Test storage-time ES-AIST + real WM/STM/LTM memory attention | `ingress_storage_attention_results.json`: loose policy selects `109 / 270` targets but abstains on only `36 / 151` no-anchor controls; high-precision policy abstains `151 / 151` controls but selects only `2 / 270` targets; zero-FPR recovery remains `0` | covered, failed |
| Test model-free online adaptive anchor state | `ingress_storage_attention_results.json` adaptive variants: evidence-loose selects `266 / 270` targets and has target in tentative top-3 for `270 / 270`, but hard-commits through all controls; ultra-precision cuts hard wrong commits to `14 / 421` but target hard binds fall to `5 / 270` | covered, soft signal useful |
| Manually audit soft-anchor behavior on real chat | `build/ingress_storage_attention_es_aist_soft_anchor/ingress_adaptive_anchor_manual_review_sample.csv`: 12 reviewed rows include acceptable tentative continuations, ambiguous-but-useful rows, and bad generic matches | covered |
| Add `anchor_strength` / `anchor_label` audit fields | `build/ingress_storage_attention_es_aist_anchor_strength_label/*`: candidates, cases, links, states, and manual audit rows include continuous strength and derived labels; high precision still emits `64 / 151` no-anchor durable binds, ultra precision emits `3 / 151` | covered, needs calibration |
| Sweep derived-label and knob-shaped promotion rules | `build/ingress_storage_attention_es_aist_label_ablation/*`: conservative high-precision label cuts no-anchor durable binds from `64 / 151` to `45 / 151` without changing reachability; high-focus cuts them to `16 / 151` but target top-3 drops to `168 / 270`; ultra precision cuts them to `3 / 151` but hard target commits fall to `5 / 270` | covered, supports soft evidence |
| Test Soft Anchor policy pieces directly | `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/*`: semantic-only improves top-3 to `247 / 270` but no-anchor durable rises to `88 / 151`; entity-only lowers no-anchor durable to `43 / 151` with top-3 `215 / 270`; generic action suppression improves 5% FPR recovery to `8`; repeated-support update gate blocks all hard target commits | covered, informs v1 design |
| Test Soft Anchor v1 hypotheses and soft updates | `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/*`: default v1 keeps target in soft top-3 for `261 / 270` with `0` hard/durable false binds, but collapses wrong-active distinct anchors to `0 / 270`; very-strict soft updates restore `51 / 270` wrong-active distinct anchors with top-3 `242 / 270` | covered, useful but not finished anchoring |
| Sweep Soft Anchor v1 across F/S/T | `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/*`: all 27 settings preserve zero hard/durable false binds; Focus improves wrong-active separation at recall cost; Sensitivity improves soft recall; repaired Stability now changes TTL/support windows without tightening soft-update acceptance | covered, partial monotonicity |
| Update paper evidence | `docs/paper/sections/9_experimental.qmd` and rendered `docs/paper/_manuscript/index.md` include latest ES, headroom, and chain sections | covered |
| Define the next required input surface | `docs/anchor-signal-contract.md` specifies runtime outputs, false-proposal gates, evaluation surfaces, and stop conditions | covered |

## Example-Chain Coverage

| chain step | required capability | evidence | status |
|---|---|---|---|
| `Jared` then `he` | person/entity continuity | real storage-time ES-AIST attention can reuse some targets (`109 / 270` under loose policy), but safe policies collapse recall | covered, unsafe |
| `He sent me this <image>` | person-to-media handoff | headroom audits still show object/media tracks are required; storage-time ES-AIST attention does not add explicit object/media proposals | covered, failed |
| `it` refers to image/object | object/media continuity | storage-time attention only stores ES semantic/entity/full keys and does not emit object/media tracks | covered, failed |
| `he had it` | person + object + relation edge | storage-time attention has no relation edge output; relation/event signal remains missing | covered, failed |
| no-anchor topic shift | abstain rather than bind | high-precision storage-time policy abstains `151 / 151` controls but recovers only `2 / 270` targets; loose policy recovers targets but false-binds most controls | covered, failed |
| wrong-active reference | reject recent wrong entity | loose policy selects wrong-active `32 / 270`; strict and high-precision reduce wrong-active selection by collapsing target reuse | covered, failed |
| ingress-time formation, not retrieval-time rescue | `ingress_storage_attention_cases.csv` proves zero retrieved candidates at runtime and `ingress_storage_attention_links.csv` records links created before future evaluation steps | covered |

## Current Findings

1. ES-AIST is useful as a proposal signal, especially when combined with
   WM/STM/LTM state, but it is not a safe commit policy.
2. The current scalar/top-k feature family is exhausted enough to stop tuning:
   logistic, multi-head, set-aware, and compact MLP consumers all fail
   source-held-out safe commitment.
3. The repaired replay candidate pool has complete target coverage. The target
   memory is present for every reference case, so the bottleneck is not long-term
   memory availability.
4. The missing signal is precision-first entity/object/relation continuity.
   High recall is not enough; tiny false-track leakage destroys zero-FPR
   recovery.
5. Entity tracks alone are insufficient for the motivating behavior. The engine
   also needs object/media tracks and relation/event edges.
6. The current ES-AIST/WM/STM/LTM scalar surface cannot be thresholded into the
   missing proposal contract. At the required false-candidate gate it recovers
   only `8 / 270` targets, and at the stricter target false gate it recovers
   `1 / 270`.
7. The new ES-AIST storage-time benchmark confirms the architectural boundary:
   anchoring must be formed at ingress from current signal plus already-stored
   WM/STM/LTM memory state. Retrieval candidates are not part of the runtime
   input.
8. Under that corrected storage contract, ES-AIST attention over real prior
   memories shows the same recall/safety tradeoff: loose policies recover some
   references but false-bind controls, while strict policies protect controls
   by refusing almost all reference binds. The missing piece is therefore not
   benchmark location; it is the ingress consumer signal/policy surface.
9. Model-free online adaptation improves continuity and should be interpreted
   as a soft anchor signal, not only as a hard commit policy. The loose soft anchor
   keeps targets alive and puts the target in tentative top-3 for `270 / 270`
   references, while manual review shows that some benchmark no-anchor "false
   binds" are useful as uncertain context. The unsafe part is durable hard
   commitment, especially generic "Ok"/"Bye" style matches.
10. `anchor_strength` and `anchor_label` are the right engine terms. The
   strength should be the persistent evidence value; the label should remain a
   derived/cached policy view. The first derivation is not calibrated enough for
   production because it still marks too many no-anchor controls as durable in
   high-recall policies.
11. Label/promotion ablations confirm that ES-AIST is a signal, not a policy.
   Better derived labels reduce wrong durable promotion without improving the
   underlying hard-commit tradeoff. Knob-shaped high-focus policies can make the
   soft anchor safer, but only by lowering recall. The useful engine contract is
   therefore uncertainty-preserving: store strength, expose tentative top-k
   anchors, and promote only after repeated support or stronger typed evidence.
12. Direct Soft Anchor ablations show why the engine should keep ES views
   separate. Semantic evidence is useful for tentative reachability but sticky;
   entity evidence is safer for durable promotion but loses recall. Generic
   action suppression helps but is not enough. Null/no-anchor and repeated
   support must be designed as calibrated hypotheses and durable-promotion gates,
   not as simple post-score cutoffs or update blockers.
13. Soft Anchor v1 is useful as an uncertainty-preserving soft anchor,
   not as a finished hard anchorer. The default v1 policy keeps `261 / 270`
   reference targets in top-3 soft links and produces zero hard/durable
   no-anchor false binds, but it collapses wrong-active subjects into one
   continuity anchor. Tightening soft-update thresholds restores some
   wrong-active separation while preserving durable safety, at a recall cost.
14. The full F/S/T sweep validates Focus and Sensitivity directionally and fixes
   the first Stability bug. Focus behaves like a precision/split knob,
   Sensitivity behaves like a tentative recall/volume knob, and Stability is now
   isolated to TTL/support-window behavior instead of soft-update acceptance. A
   longer decay-heavy replay is still needed before engine promotion.

## Missing Requirement

No current artifact provides a real high-precision modality-agnostic
entity/object/relation signal for durable commitment. ES-AIST plus online
WM/STM/LTM state is useful for tentative context proposals, but further hard
threshold sweeps over the same scalar features are not productive.

The next productive work requires one of:

- a model artifact trained to emit high-precision entity, object/media, event,
  and relation signals;
- upstream entity/object/voice/visual-track proposals;
- real or high-fidelity multimodal episode data with entity/object/relation
  labels for training and validation.

Until one of those exists, the anchoring product goal should be implemented as
uncertainty-preserving shadow state rather than promoted as durable production
retrieval behavior. The current experiment pass has exhausted the available
ES/WM/STM/LTM hard-threshold policy surfaces and should stop tuning for
perfection over the same inputs.

The concrete contract for that next artifact is
[`docs/anchor-signal-contract.md`](anchor-signal-contract.md).
