# Cortext Experiment Notes

## Local Nemotron Privacy Prune/Redact Pass

Command:

```bash
scripts/launch_nemotron_judge_server.sh

python3 tools/nemotron_privacy_prune_redact.py \
  --input-dir "/path/to/your/chat-export" \
  --out-dir build/chat_replay_privacy_redacted
```

Behavior:

- Uses the local OpenAI-compatible Nemotron endpoint served by `vllm-mlx`.
- Writes a redacted transcript and copies only media classified as safe into the output tree.
- Writes `privacy_manifest.json` with decisions, categories, confidence, hashes, and aggregate counts.
- Does not include raw private message text in the manifest.
- Defaults to non-destructive behavior. Private media is removed from the sanitized output, not deleted from the source tree.
- `--apply-delete` is available for explicit destructive source-media deletion after reviewing a dry run.
- Image and video-frame classification can use multimodal payloads; audio currently uses a conservative metadata policy because OpenAI-compatible audio payload support varies across `vllm-mlx` model builds.

Smoke:

```bash
python3 tools/nemotron_privacy_prune_redact.py \
  --input-dir build/privacy_prune_redact_smoke_input \
  --out-dir build/privacy_prune_redact_smoke_out \
  --max-messages 2 \
  --media-limit 2 \
  --batch-size 2
```

Result:

- Fake phone/address text was redacted from the output transcript.
- `privacy_manifest.json` did not contain the fake raw phone number.
- Both synthetic media files were pruned from the sanitized output.
- No source files were deleted.

## AAIT-86M-GGUF Native Runtime Parity and Ingress Anchor Shadow

Command:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --aait-ingress-anchor-shadow-only \
  --models=models \
  --output-dir build/aait_ingress_anchor_shadow
```

Status:

- `AAIT-86M_q8_0.gguf` resolves under `models/AAIT-86M-GGUF/`.
- Metadata inspection works and writes `build/aait_gguf_metadata.json`.
- The GGUF declares custom `triembed` architecture, 1280-dimensional semantic output, text/image/audio modality support, and anchor-head tensors for `anchor_key`, `anchor_action_logits`, `anchor_confidence`, `salience_delta`, and bind logits.
- The text model recorded in metadata is `MongoDB/mdbr-leaf-ir`, matching the current LEAF-IR/MDBR text behavior.
- Generic Homebrew llama.cpp cannot execute this GGUF: it rejects tensor index 54, `audio_encoder_efficientat.features.1.block.0.1.num_batches_tracked`, with length `66 >= 64`.
- Cortext now has a native benchmark-only `triembed` GGUF path for the text + anchor heads. It parses/dequantizes the needed GGUF tensors directly and prefers the local HuggingFace `MongoDB/mdbr-leaf-ir` `vocab.txt` at `models/mdbr-leaf-ir/vocab.txt`, falling back to GGUF tokenizer metadata only when that vocab is unavailable.
- No AAIT anchor outputs are synthesized from semantic vectors.

Runtime smoke:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --aait-runtime-smoke \
  --models=models \
  --output-dir build/aait_runtime_smoke
```

Smoke output:

- runtime: `native_gguf_triembed`
- semantic vector: 1280d, norm ~1.0
- anchor key: 128d, norm ~1.0
- action logits: 5
- confidence/salience: finite
- mean smoke inference: ~495 ms

Reference parity:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --aait-reference-parity \
  --models=models \
  --output-dir build/aait_reference_parity \
  --max-cases 25
```

Artifacts:

- `aait_reference_parity_results.json`
- `aait_reference_parity_cases.csv`
- `aait_reference_parity_failures.csv`
- `aait_tokenization_parity.json`
- `aait_action_logit_audit.csv`
- `aait_runtime_tensor_mapping.json`
- `aait_ingress_consumption_audit.json`
- `aait_reference_ingress_results.json`
- `aait_native_vs_reference_ingress_compare.json`
- `aait_latency_breakdown.json`
- `aait_cache_stats.json`

Parity result over 484 fixed/replay inputs:

- status: `parity_passed`
- semantic cosine mean/min: 0.9999589 / 0.9999216
- anchor-key cosine mean/min: 0.9999985 / 0.9999918
- action-logit MAE: 0.00142
- top-action agreement: 1.0000

The first parity smoke exposed the actual blocker: native tensor math matched the safetensors reference when forced to use native token IDs, but the old native tokenizer did not match the HuggingFace tokenizer and mapped many common words to `[UNK]`. After switching the native path to the local HF `vocab.txt`, both the HF-tokenizer and native-token reference paths passed parity. This means the current AAIT benchmark failure is not a GGUF execution bug.

Corrected 25-case shadow run:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --aait-ingress-anchor-shadow-only \
  --models=models \
  --output-dir build/aait_ingress_anchor_shadow_real_parity_passed \
  --max-cases 25
```

Results:

- AAIT inference ran: true.
- link-only target-vs-candidate AUC: 0.5000 because query-time non-UPDATE actions are now hard abstentions.
- AAIT ingress + retrieval diagnostic target-vs-candidate AUC: 0.8001.
- AAIT ingress + retrieval diagnostic target top-1/top-3: 0.1176 / 0.4706.
- AAIT ingress + retrieval diagnostic no-anchor false-bind AUC: 0.5012.
- AAIT ingress + retrieval diagnostic wrong-active rejection AUC: 0.1167 inverted / 0.8833 risk direction.
- AAIT ingress + retrieval diagnostic stale rejection AUC: 0.5405 inverted / 0.4595 risk direction.
- zero-FPR and 5% FPR recovery: 0 / 17.
- target anchor existed before query: 17 / 17 references.
- wrong-active had distinct anchor: 17 / 17 references.
- no-anchor had no valid active anchor: 8 / 8 controls after making only `UPDATE_EXISTING_ANCHOR` consumable at query time.
- link types: 390 `created`, 63 `update_without_active_anchor`.
- mean/p95 native AAIT inference overhead: 10.91 s / 20.54 s per replay case on this unoptimized per-candidate C++ path.
- mean/p95 ledger update overhead: 0.51 ms / 0.57 ms.
- cache hit rate: 0.756; 350 unique semantic encodes from 1434 semantic requests.

Candidate-track tensor parity:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --aait-candidate-contract-parity \
  --models=models \
  --output-dir build/aait_candidate_contract_parity_25 \
  --max-cases 25
```

Artifacts:

- `aait_candidate_contract_tensors.jsonl`
- `aait_candidate_contract_summary.json`
- `aait_candidate_contract_failures.csv`
- `aait_candidate_contract_reference_compare.json`
- `aait_candidate_contract_reference_cases.csv`
- `aait_candidate_feature_range_audit.json`
- `aait_action_bind_semantics_audit.json`
- `aait_ingress_consumer_contract_audit.json`
- `aait_candidate_contract_toy_sanity.json`
- `aait_candidate_contract_toy_cases.csv`
- `aait_candidate_contract_calibration.json`

Result:

- The parity harness exports the exact tensors passed to AAIT: current semantic, recent context vector, active anchor state, candidate-track semantic matrix, candidate features, candidate mask, candidate order, action logits, bind logits, anchor confidence, and salience delta.
- It then reruns those same tensors through the safetensors/PyTorch reference path.
- Native-vs-reference candidate tensor parity passed over 481 records: 478 replay tensor records plus 3 replay-derived toy fixtures.
- Top action agreement: 1.0000.
- Selected bind-candidate agreement: 1.0000.
- Action-logit MAE: 7.72e-05.
- Bind-logit MAE: 5.06e-05.
- Feature order is `[age, salience, confidence, mod_text, mod_image, mod_audio, last_seen_step]`.
- Candidate counts ranged from 0 to 8; candidate count mean was 2.91.
- Candidate masks used true = valid; no padded candidate was selected.
- `age` range: 1..20; `last_seen_step` range: 0..19; salience/confidence stayed in [0,1].
- UPDATE rate on reference current steps: 17 / 17.
- Action-head ABSTAIN rate on no-anchor current steps: 0 / 8.
- No-anchor current steps emitted UPDATE 8 / 8, but bind logits selected the abstain slot for all no-anchor current steps. The strict consumer now treats this as no valid bind.
- Toy sanity is mixed: the replay-derived no-anchor fixture still emits UPDATE at the action head, but the bind head abstains. The create and update fixtures pass.

Interpretation: native GGUF and safetensors/PyTorch agree on the full candidate-track tensor contract. The remaining no-anchor issue is not native runtime parity. It is action/bind-head interpretation and possibly training/objective semantics: action UPDATE alone is insufficient evidence of commitment when the bind head selects abstain.

Black-box wrong-active candidate-track pass:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --aait-ingress-anchor-shadow-only \
  --models=models \
  --output-dir build/aait_ingress_anchor_black_box_25 \
  --max-cases 25
```

New artifacts:

- `aait_wrong_active_tensor_audit.csv`
- `aait_candidate_tensor_summary.json`
- `aait_candidate_order_ablation_results.json`
- `aait_candidate_feature_ablation_results.json`
- `aait_ingress_consumer_audit.json`
- `aait_bind_head_analysis.json`
- `aait_ingress_anchor_wrong_active_repaired_results.json`
- `aait_wrong_active_failure_examples.csv`
- `aait_deployment_tensor_pack_train.jsonl`
- `aait_deployment_tensor_pack_val.jsonl`
- `aait_deployment_tensor_pack_test.jsonl`
- `aait_deployment_tensor_pack_schema.json`
- `aait_deployment_tensor_pack_summary.json`

Result:

- Selected format: `candidate_track_tensor_contract`.
- String-format input ablations are no longer rerun by default; set `CORTEXT_AAIT_RUN_STRING_INPUT_ABLATIONS=1` to repeat them. This pass treats AAIT as a black-box ingress model and focuses on candidate construction, ordering, masking, feature scaling, and strict consumer logic.
- In older artifact fields, `query_*` means the held-out current ingress signal/current ingress step, not a production retrieval query.
- Strict consumer semantics were tightened: `UPDATE_EXISTING_ANCHOR` can consume an existing anchor only when bind logits select a non-abstain candidate.
- Prior-memory UPDATE now uses the bind-selected candidate anchor when bind logits are available. If the bind head selects the abstain slot, the prior memory starts a new anchor instead of merging into the best cosine anchor. This prevents hidden wrong-active/target merges during ledger construction.
- Target anchors existed before the current ingress step in 17 / 17 references.
- Wrong-active anchors were distinct from target anchors in 17 / 17 references.
- No-anchor controls had no valid active anchor: 8 / 8.
- Current-step action head emitted `CREATE_ANCHOR` on all 17 references and all 8 no-anchor controls under the repaired prior ledger.
- Current-step bind head selected the abstain slot on all 17 references and all 8 no-anchor controls.
- Target and wrong-active anchors were both present in the current candidate set for 17 / 17 references, so the failure is not missing-candidate reachability.
- Candidate count was exactly 8 on this slice. Current Cortext order is most-recent-first, and the ablations explicitly tested alternate orders.
- Candidate feature ranges in the current candidate set: age 1..12, salience 0.857..0.904, confidence 0.912..0.929, last_seen_step 5..19.
- Candidate-order ablations (`current_order`, fixed random, reversed, semantic-similarity, confidence, and recency order) did not change the outcome: UPDATE rate 0 / 17, bind-target selection 0 / 17, bind-wrong-active selection 0 / 17, bind-abstain 17 / 17.
- Candidate-feature ablations (current features, normalized age/last_seen, inverted age, confidence fixed to 1, salience fixed to 1, removed salience/confidence, metadata removed/semantic-only, and mask-only sanity) also did not change the outcome.
- Link-only target AUC: 0.5000; link-only zero-FPR and 5% FPR recovery: 0 / 17.
- Link-only no-anchor false-bind AUC: 0.5000.
- Link-only wrong-active rejection AUC: 0.5000.
- Link-only stale rejection AUC: 0.5000.
- AAIT ingress + retrieval diagnostic target AUC: 0.8001; top-1/top-3: 0.1176 / 0.4706.
- AAIT ingress + retrieval diagnostic no-anchor false-bind AUC: 0.5012.
- AAIT ingress + retrieval diagnostic wrong-active rejection AUC: 0.1167.
- AAIT ingress + retrieval diagnostic stale rejection AUC: 0.5405.
- AAIT ingress + retrieval diagnostic zero-FPR and 5% FPR recovery: 0 / 17.
- mean/p95 native AAIT inference overhead in this focused pass: 9.16 s / 16.49 s per replay case; ledger update remains sub-millisecond. This path still performs many native text encodes while reconstructing prior candidate tracks.

Interpretation: the original string-only failure was an input-contract bug, and the no-anchor carryover bug was a consumer bug. The wrong-active merge bug was a prior-ledger construction bug and is now fixed structurally. The remaining failure is different: with a correct strict consumer and distinct wrong-active anchors, the current reference event chooses CREATE plus bind-abstain, not UPDATE plus target bind. Reordering candidates, changing simple feature scales, removing metadata, or using a mask-only sanity input does not repair it. Because no Cortext-side ablation fixed the issue, the benchmark exported a black-box deployment tensor pack with exact model input tensors and labels split into train/val/test. The pack does not include raw text, candidate classes, target flags, track ids, or gold actions as runtime features. AAIT remains shadow-only and not promotion-ready.

Large deployment tensor pack:

```bash
python3 tools/generate_aait_large_tensor_pack.py \
  --input-dir build/aait_ingress_anchor_black_box_25 \
  --output-dir build/aait_ingress_anchor_black_box_large \
  --rows 5000
```

Artifacts:

- `aait_ingress_anchor_black_box_train.jsonl`
- `aait_ingress_anchor_black_box_val.jsonl`
- `aait_ingress_anchor_black_box_test.jsonl`
- `aait_ingress_anchor_black_box_schema.json`
- `aait_ingress_anchor_black_box_summary.json`
- `aait_ingress_anchor_black_box_split_audit.json`
- `aait_ingress_anchor_black_box_target_index_distribution.json`
- `aait_ingress_anchor_black_box_failure_slices.json`

Large-pack audit:

- row count: 5000.
- split counts: 4000 train / 500 val / 500 test.
- source-held-out split: train = `HuggingFaceH4/ultrachat_200k`, val = `bentrevett/schema_guided_dialog`, test = `pietrolesci/multiwoz_all_versions`.
- action distribution: 1000 each for `CREATE_ANCHOR`, `UPDATE_EXISTING_ANCHOR`, `SPLIT_ANCHOR`, `CLOSE_ANCHOR`, and `ABSTAIN`.
- bind labels: 3000 abstain labels plus 2000 candidate-bind labels.
- target index histogram for candidate-bind labels: slot 0 = 167, slot 1 = 262, slot 2 = 262, slot 3 = 262, slot 4 = 262, slot 5 = 262, slot 6 = 262, slot 7 = 261.
- candidate counts cover 0, 1, 2, 3, 4, 5, 6, and 8 valid slots.
- candidate order variants are balanced across fixed-seed random, recency sorted, semantic sorted, reversed, adversarial target position, and current order.
- duplicate runtime tensor signatures across splits: 0.
- runtime input fields are only `current_semantic`, `recent_context_vector`, `active_anchor_state`, `candidate_semantic_matrix`, `candidate_feature_matrix`, and `candidate_mask`.
- no production retrieval behavior changed; no retrieval candidates are included.

Label audit:

```bash
python3 tools/audit_aait_tensor_pack_labels.py \
  --input-dir build/aait_ingress_anchor_black_box_large \
  --output-dir build/aait_ingress_anchor_black_box_label_audit \
  --gpt-rows 0 \
  --notify
```

Artifacts:

- `aait_ingress_anchor_black_box_label_audit.json`
- `aait_ingress_anchor_black_box_label_disagreements.jsonl`

Result:

- deterministic label-invariant audit rows: 5000.
- deterministic failed rows: 0.
- deterministic failure counts: none.
- status: `passed_structural_label_audit`.
- GPT-5.4-mini audit was not run in this shell because `OPENAI_API_KEY` was not present.

The GPT validation path is implemented but cost-capped by explicit sample size.
For example:

```bash
python3 tools/audit_aait_tensor_pack_labels.py \
  --input-dir build/aait_ingress_anchor_black_box_large \
  --output-dir build/aait_ingress_anchor_black_box_label_audit_gpt \
  --gpt-rows 240 \
  --batch-size 10 \
  --model gpt-5.4-mini-2026-03-17 \
  --notify
```

GPT sample-audit result using `gpt-5.4-mini-2026-03-17`:

- reviewed rows: 240 / 240 requested.
- disagreements: 0.
- severity counts: 238 `ok`, 2 `warning`.
- usage: 140,753 total tokens.
- status: `passed_gpt_sample_audit_with_warnings`.
- result path: `build/aait_ingress_anchor_black_box_label_audit_gpt/aait_ingress_anchor_black_box_gpt_label_audit.jsonl`.

The OpenAI review sees compact tensor-derived candidate statistics and
procedural labels, not raw source text or entity-track ground truth. It can
validate the generated label contract and catch contradictions, but it cannot
prove true real-world referent semantics from embeddings alone. The two warnings
were both valid-label/underdetermined-intent cases in hard update slices where
the compact stats supported the intended lower-ranked target pattern.

Runtime controls:

- `CORTEXT_AAIT_MODEL_PATH`
- `CORTEXT_AAIT_ENABLE=1`
- `CORTEXT_AAIT_SHADOW_ONLY=1`
- `CORTEXT_AAIT_RUNTIME=gguf` (`native` and `native_gguf` are aliases)
- `CORTEXT_AAIT_USE_SEMANTIC_FOR_RETRIEVAL=0` by default
- `CORTEXT_AAIT_N_GPU_LAYERS`
- `CORTEXT_AAIT_THREADS`
- `CORTEXT_AAIT_CONTEXT_LENGTH`
- `CORTEXT_AAIT_TOKENIZER_GGUF_PATH`

Action logit order:

1. `CREATE_ANCHOR`
2. `UPDATE_EXISTING_ANCHOR`
3. `SPLIT_ANCHOR`
4. `CLOSE_ANCHOR`
5. `ABSTAIN`

Interpretation:

This is now an executable integration result with reference parity and the candidate-track tensor contract wired into Cortext. The previous string-only 25-case AAIT failure is not evidence against the GGUF runtime. Native execution matches the safetensors reference. The latest black-box pass shows that no-anchor protection and wrong-active anchor separation can be achieved in the strict consumer/ledger, but the current-step bind decision still abstains on all reference cases under Cortext-reconstructed candidate tracks. The path is not promotion-ready. AAIT remains disabled by default and shadow-only.

## ESS-AIST-81M Preview GGUF Runtime Smoke

Model repo: `augmem/ESS-AIST-81M-preview-GGUF`.

Local asset:

- `models/ESS-AIST-81M-preview-GGUF/ESS-AIST-81M_q8_0.gguf`

Benchmark commands:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --ess-aist-runtime-smoke \
  --models=models \
  --output-dir build/ess_aist_runtime_smoke
```

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --ess-aist-anchor-slice-shadow-only \
  --models=models \
  --output-dir build/ess_aist_anchor_slice_shadow_25 \
  --max-cases 25
```

Runtime result:

- native triembed GGUF shape-smoke execution works for the preview q8_0 checkpoint.
- the optimized kernel-backed path now runs the text encoder as a full GGML graph using the custom GGUF parser: token embedding lookup, six BERT layers, attention, layer norms, mean pooling, projection, and L2 normalization all stay inside backend graph execution.
- the verified backend on the local machine was `MTL0 (Apple M3 Max)`.
- output vector dimension is 1536.
- slices are semantic `[0:512]`, subject `[512:1024]`, and event `[1024:1536]`.
- normalized semantic, subject, event, prefix-512, prefix-1024, and prefix-1536 views are emitted.
- no AAIT anchor-head tensors are present, so this is an embedding-slice model, not an ingress-anchor model.
- single-sample smoke latency with the old Eigen fallback was 446.66 ms after model load.
- single-sample smoke latency with the incremental ggml/Metal dense-kernel path was 216.47 ms after model load.
- full-graph parity against the old per-linear debug path passed on the Jared reference sentence: cosine `1.000000`, L2 diff `0.000589`, max absolute diff `0.000056`, both norms `1.000000`. The old per-linear debug path took `145.25 ms`; the full graph first measured call took `10.62 ms`.
- sequential fair cache-miss text comparison using `cortext_text_encoder_bench --vary-text`
  on the same Jared reference sentence, with 10 warmup rows and 100 measured rows:
  - regular LEAF-IR / `models/llama_cpp/mdbr-leaf-ir-q8_0.gguf`: 256d,
    mean `3.24 ms`, `309.02` embeddings/s.
  - ESS-AIST / `ESS-AIST-81M_q8_0.gguf` full GGML graph: 1536d,
    mean `2.28 ms`, `438.60` embeddings/s.
- the runtime smoke now reports `kernel_ops_granularity = full_text_graph`,
  `slice_benchmark_ready = true`, and single-sample inference `7.07 ms`.
- the 25-case slice shadow run completed without debug overrides in `10.66 s`
  with `1596` uncached semantic requests. The full graph was used for the run.
- on that tiny 25-case slice, ESS-AIST is still diagnostic only: repaired-real
  hierarchical attention over the subject slice reached target AUC `0.9870`,
  top-1 `0.8824`, and top-3 `1.0000`, but no-anchor AUC stayed near chance
  (`0.4787`) and wrong-active rejection remained poor (`0.1144`). This is not
  anchor promotion evidence.
- production retrieval remains unchanged.

Artifacts:

- `build/ess_aist_runtime_smoke_full_graph_q8/ess_aist_runtime_smoke.json`
- `build/ess_aist_runtime_smoke_full_graph_q8/ess_aist_runtime_latency.json`
- `build/ess_aist_runtime_smoke_full_graph_q8/ess_aist_model_metadata.json`
- `build/ess_aist_runtime_smoke_full_graph_q8/ess_aist_gguf_tensor_name_audit.json`
- `build/ess_aist_anchor_slice_shadow_full_graph_25/ess_aist_slice_results.json`
- `build/ess_aist_anchor_slice_shadow_full_graph_25/ess_aist_slice_summary.csv`

Integration notes:

- the native GGUF loader now accepts embedding-only triembed models instead of requiring anchor-head tensors.
- q8_0 is the preferred auto-discovered checkpoint; q5_1 remains a fallback/debug asset.
- q5_1 dequantization was added only so old preview artifacts still inspect/run.
- the text projection runner now discovers available residual projection blocks instead of assuming the AAIT anchor-head projection depth.
- `--ess-aist-anchor-slice-shadow-only` is wired as a benchmark-only slice comparison over repaired real replay and hard-synthetic v3 and now requires the full-text GGML graph by default. The per-linear ggml path can be forced only by setting `CORTEXT_AAIT_DISABLE_FULL_GGML_GRAPH=1` together with `CORTEXT_ESS_AIST_ALLOW_INCREMENTAL_GGML_KERNELS=1` for debug profiling. The native Eigen fallback can be forced only with `CORTEXT_ESS_AIST_ALLOW_EIGEN_FALLBACK=1` for debug shape checks.
- direct llama.cpp kernel-backed loading was probed and failed before architecture execution because the installed loader still rejects long GGUF tensor names (`tensor name 60 is too long: 64 >= 64`). Probe artifacts are `build/ess_aist_runtime_smoke_q8/ess_aist_llama_cpp_kernel_probe.json` and `build/ess_aist_runtime_smoke_q8/ess_aist_llama_cpp_kernel_probe.log`.

Interpretation:

ESS-AIST now has a real high-performance GGUF text path. The previous 50x slowdown was an implementation artifact from dispatching per-linear debug graphs. The full-graph path is fast enough for benchmark sweeps and comparable to, or faster than on this local run, the regular LEAF-IR q8 path while producing a 1536d semantic/subject/event vector. The first slice run shows the same pattern as earlier anchor work: target reachability can improve sharply, but no-anchor and wrong-active commitment do not improve just because a subject slice exists. The correct next experiment is a larger ESS-AIST slice sweep and downstream consumer ablation that treats no-anchor and wrong-active rejection as primary metrics, not target AUC alone.

### ESS-AIST ingress-native shadow ledger

After the slice/rerank diagnostic, we added a separate ingress-native benchmark:

```bash
./build/examples/benchmark/cortext_anchor_replay_bench \
  --ess-aist-ingress-anchor-shadow-only \
  --models=models \
  --output-dir build/ess_aist_ingress_anchor_shadow_full
```

This benchmark does not score retrieved candidates at runtime. For each repaired
case it replays the source conversation chronologically up to the held-out
current ingress step, encodes each incoming signal once with ESS-AIST, and
updates separate shadow ledgers from:

- `semantic_key = z[0:512]`
- `subject_key = z[512:1024]`
- `event_key = z[1024:1536]`

The candidate pool and labels are used only after replay to audit whether the
pre-current ledger had the right subject/event state. Runtime inputs are the
current signal and prior ledger state only; no retrieval candidates, target
flags, candidate classes, track ids, or gold actions are passed into the
ledger.

Full repaired-surface result:

- cases: `270` references and `151` no-anchor controls
- ESS-AIST encodes: `5220` uncached view-cache entries
- runtime: `16.25 s` total, `kernel_ops_granularity = full_text_graph`
- target subject track existed before the current ingress step: `1.000`
- primary subject+event-agreement policy selected safe target: `0.000`
- primary no-anchor abstention: `0.020`
- primary wrong-active distinct-before-current rate: `0.000`
- zero-FPR recovery: `0 / 270`
- 5 percent FPR recovery: `0 / 270`

The stricter policy sweeps did not fix the failure. The best no-anchor
abstention came from `subject_event_very_strict` at `0.278`, but safe target
selection was only `0.007`, wrong-active distinct-before-current was only
`0.026`, and low-FPR recovery was still `0 / 270`. The common failure is now
unambiguous: target and wrong-active turns usually attach to the same ingress
subject track. Counting that as target recovery would be wrong, so the benchmark
now requires the selected target track to be distinct from the wrong-active
track.

Artifacts:

- `build/ess_aist_ingress_anchor_shadow_full/ess_aist_ingress_anchor_results.json`
- `build/ess_aist_ingress_anchor_shadow_full/ess_aist_ingress_anchor_cases.csv`
- `build/ess_aist_ingress_anchor_shadow_full/ess_aist_ingress_anchor_links.csv`
- `build/ess_aist_ingress_anchor_shadow_full/ess_aist_ingress_anchor_states.csv`
- `build/ess_aist_ingress_anchor_shadow_full/ess_aist_ingress_anchor_failure_examples.csv`
- `build/ess_aist_ingress_anchor_shadow_full/ess_aist_ingress_anchor_latency.csv`

Interpretation: evaluating ESS-AIST at ingress was the right correction, but the
preview checkpoint still does not solve anchoring as a pure vector-similarity
ledger. The subject/event blocks preserve broad stream continuity, not
human-level subject identity under wrong-active competition. This does not
invalidate the ESS-AIT vector contract, but it means the next checkpoint needs
explicit training/evaluation for subject-track splitting, no-match abstention,
wrong-active separation, and stale event closure. A downstream Cortext consumer
should not promote these slices as an anchor mechanism without those metrics.

### ESS-AIST v9 checkpoint smoke and ingress replay

The requested `v9` revision was not exposed as a Hugging Face branch or tag for
`augmem/ESS-AIST-81M-preview-GGUF`; the repo only advertised `main`. The current
`main` manifest, however, points at the v9 checkpoint:
`ess_aist_full_v9_subjectfix_l4k/best_model.pt`. We downloaded that state into a
separate local directory so it is not confused with the earlier preview asset:

- local model: `models/ESS-AIST-81M-preview-GGUF-v9/ESS-AIST-81M_q8_0.gguf`
- old q8 SHA-256:
  `a9c79593801aa28823934aad09ec3efec5684b3802e294d004dbc71d93ca69b1`
- v9 q8 SHA-256:
  `de05d0787e0fe87e56e69d714f3a99ee42a2db58ea1f31440842ff9c565579a8`

The checkpoint metadata shows the intended subject/event improvements:

| view | subject same/different AUC | event same/different AUC | same-subject different-event AUC |
|---|---:|---:|---:|
| semantic key | 0.9563 | 0.8275 | 0.6695 |
| subject key | 0.9881 | 0.8855 | 0.7381 |
| event key | 0.9551 | 0.8193 | 0.6807 |
| prefix 1024 | 0.9815 | 0.8613 | 0.7009 |
| prefix 1536 | 0.9779 | 0.8518 | 0.6938 |

Runtime smoke:

```bash
CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH=1 ./build/examples/benchmark/cortext_anchor_replay_bench \
  --ess-aist-runtime-smoke \
  --models=models \
  --teacher-model models/ESS-AIST-81M-preview-GGUF-v9/ESS-AIST-81M_q8_0.gguf \
  --output-dir build/ess_aist_v9_runtime_smoke_q8
```

- status: `ok`
- backend: `MTL0 (Apple M3 Max)`
- graph granularity: `full_text_graph`
- output: 1536d with 512d semantic, subject, and event slices
- single-sample inference: `12.37 ms`
- production retrieval changed: `false`

Ingress-native replay:

```bash
CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH=1 ./build/examples/benchmark/cortext_anchor_replay_bench \
  --ess-aist-ingress-anchor-shadow-only \
  --models=models \
  --teacher-model models/ESS-AIST-81M-preview-GGUF-v9/ESS-AIST-81M_q8_0.gguf \
  --output-dir build/ess_aist_v9_ingress_anchor_shadow_full
```

The v9 full repaired-surface run covered `270` references and `151` no-anchor
controls, used `5220` uncached view-cache entries, and completed in `17.58 s`
with the Metal full-text graph.

| policy | safe target selected | no-anchor abstain | wrong-active distinct | selected wrong-active | zero-FPR | 5% FPR |
|---|---:|---:|---:|---:|---:|---:|
| subject_only_loose | 0.000 | 0.000 | 0.000 | 1.000 | 0/270 | 0/270 |
| subject_only_strict | 0.000 | 0.000 | 0.000 | 0.989 | 0/270 | 0/270 |
| event_only | 0.000 | 0.000 | 0.000 | 0.996 | 0/270 | 0/270 |
| subject_event_agreement | 0.000 | 0.000 | 0.000 | 1.000 | 0/270 | 0/270 |
| subject_event_high_split | 0.022 | 0.099 | 0.070 | 0.863 | 0/270 | 0/270 |
| subject_event_very_strict | 0.026 | 0.437 | 0.289 | 0.489 | 0/270 | 2/270 |

Compared with the earlier preview run, v9 materially improves the strictest
split/abstain behavior: wrong-active distinct-before-current rises from `0.026`
to `0.289`, and no-anchor abstention rises from `0.278` to `0.437`. It still is
not an anchor mechanism: safe target selection is only `0.026`, zero-FPR
recovery remains `0`, and 5 percent FPR recovery is only `2 / 270`.

Slice/rerank diagnostic:

```bash
CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH=1 ./build/examples/benchmark/cortext_anchor_replay_bench \
  --ess-aist-anchor-slice-shadow-only \
  --models=models \
  --teacher-model models/ESS-AIST-81M-preview-GGUF-v9/ESS-AIST-81M_q8_0.gguf \
  --output-dir build/ess_aist_v9_anchor_slice_shadow_full
```

Best repaired-real diagnostic rows:

| score family | best view | target AUC | top-1 | top-3 | no-anchor AUC | wrong-active rejection AUC | zero-FPR | 5% FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline retrieval | subject key | 0.5943 | 0.1259 | 0.2778 | 0.5212 | 0.4047 | 0 | 0 |
| micro-anchor | subject key | 0.6928 | 0.1556 | 0.4074 | 0.5015 | 0.1981 | 0 | 2 |
| hierarchical attention | semantic key | 0.7394 | 0.2667 | 0.4593 | 0.5151 | 0.2097 | 0 | 0 |
| latent ranker only | subject key | 0.8231 | 0.3370 | 0.6222 | 0.5017 | 0.1491 | 0 | 0 |

Interpretation: v9 confirms that the model-side subject/event training improved
offline subject and event geometry, and it improves the ingress split/abstain
curve relative to the first preview. The repaired replay still fails the
commitment gate. Target reachability is decent in the slice diagnostic, but
no-anchor and wrong-active remain near the old failure surface. ESS-AIST v9
therefore stays benchmark-only; production retrieval is unchanged and the slices
should not be promoted as anchors.

### ESS-AIST v9 ingress subject/event attention

We then tested whether a transformer-like ingress attention consumer over the
frozen v9 subject and event slices could avoid a special anchor head. This mode
is still shadow-only and retrieval-free:

```bash
CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH=1 ./build/examples/benchmark/cortext_anchor_replay_bench \
  --ess-aist-ingress-attention-shadow-only \
  --models=models \
  --teacher-model models/ESS-AIST-81M-preview-GGUF-v9/ESS-AIST-81M_q8_0.gguf \
  --output-dir build/ess_aist_v9_ingress_attention_shadow_full
```

The consumer keeps ESS-AIST frozen. During chronological replay it creates
causal prior-track tokens from subject and event slices, then scores each
incoming signal by attention over prior subject/event state. The ablation
includes:

- attention over subject tracks and event tracks separately;
- cross-event agreement penalties;
- entropy/margin split gates;
- joint subject+event keys, including event-heavy and ultra-split variants.

It still does not use retrieval candidates, target ids, candidate classes,
track labels, or future labels at runtime.

Full repaired-surface result:

- cases: `270` references and `151` no-anchor controls
- runtime: `21.86 s` total
- backend: `MTL0 (Apple M3 Max)`
- graph granularity: `full_text_graph`
- ESS-AIST view-cache entries: `5220`
- production retrieval changed: `false`

| policy | safe target selected | no-anchor abstain | wrong-active distinct | selected wrong-active | zero-FPR | 5% FPR |
|---|---:|---:|---:|---:|---:|---:|
| attention_loose | 0.000 | 0.000 | 0.000 | 1.000 | 0/270 | 0/270 |
| attention_cross_split | 0.000 | 0.000 | 0.000 | 1.000 | 0/270 | 0/270 |
| attention_very_strict | 0.004 | 0.040 | 0.030 | 0.937 | 0/270 | 0/270 |
| joint_attention_cross_split | 0.000 | 0.000 | 0.000 | 0.993 | 0/270 | 0/270 |
| joint_attention_very_strict | 0.004 | 0.060 | 0.044 | 0.911 | 0/270 | 0/270 |
| joint_attention_ultra_split | 0.000 | 0.993 | 1.000 | 0.004 | 0/270 | 0/270 |

Interpretation: causal attention over frozen subject/event slices exposes the
same tradeoff as cosine ledgering. Loose attention preserves reachability but
merges into wrong-active. Ultra-strict joint subject/event attention can split
wrong-active and abstain on nearly all no-anchor controls, but it loses all
safe reference binding. That is useful evidence: the failure is not simply that
the previous consumer lacked a transformer-style attention readout. With this
checkpoint, a hand-built attention consumer over frozen slices cannot get both
referent continuity and abstention. Avoiding a special anchor model likely
requires either a learned ingress attention adapter over these slices or a
future ESS-AIST checkpoint whose subject/event geometry is explicitly trained
for bind/no-bind under wrong-active competition.

### ES-AIST-81M preview GGUF

Model repo: `augmem/ES-AIST-81M-preview-GGUF`.

The model card describes a different contract than ESS-AIST v9. ES-AIST is not a
3 x 512 semantic/subject/event slice model. The release metrics use:

- `semantic_768_key = z[0:768]`
- `entity_key = z[768:1536]`
- `full_key = z[0:1536]`

We first ran the old 3 x 512 ESS split and then discarded those numbers as a
contract mismatch. The benchmark was patched to expose the ES 768/768 views and
to use `entity_key` as the ingress identity key. In the ingress consumers the
third track view is `full_key`, not an independently trained event block.

Local asset:

- `models/ES-AIST-81M-preview-GGUF/ES-AIST-81M_q8_0.gguf`
- q8 SHA-256:
  `3c77c5aa277c1041c325723d47221970686ec0194b0aa59f3617a7f1192a5699`

Release metadata:

- source checkpoint:
  `es_aist_full_v13_anchor_memory_eventboost_er125_bs4096_nw0_l4b/best_model.pt`
- `entity_key` same/different entity AUC: `0.9953`
- `entity_key` same-topic/different-entity rejection AUC: `0.9953`
- `entity_key` same-entity/different-event rejection AUC: `0.8001`
- `entity_key` stale same-source rejection AUC: `0.9241`
- `entity_key` wrong-active rejection AUC: `0.8799`
- weak-reference entity candidate R@1: `1.0000`
- anchor-memory entity candidate R@1: `0.9647`

Runtime smoke:

```bash
CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH=1 ./build/examples/benchmark/cortext_anchor_replay_bench \
  --ess-aist-runtime-smoke \
  --models=models \
  --teacher-model models/ES-AIST-81M-preview-GGUF/ES-AIST-81M_q8_0.gguf \
  --output-dir build/es_aist_runtime_smoke_q8_contract
```

- status: `ok`
- backend: `MTL0 (Apple M3 Max)`
- graph granularity: `full_text_graph`
- output dimension: `1536`
- `semantic_768_key`: 768d, finite, norm ~1.0
- `entity_key`: 768d, finite, norm ~1.0
- inference: `7.08 ms`
- production retrieval changed: `false`

Engine runtime integration update:

- ES-AIST q8 is now an executable native GGUF encoder for text, image, and
  audio. The path uses GGUF tensors through GGML/Metal/CPU backend kernels; no
  Python, safetensors helper, or Eigen fallback is used for normal execution.
- Text runs the full LEAF-IR-style graph and returns the corrected 1536d ES
  vector. Image runs the native image encoder/projection graph after 384 x 384
  RGB ImageNet-style preprocessing. Audio runs the native audio
  encoder/projection graph after converting public 16 kHz mono PCM ingress into
  the model's 32 kHz, 128 x 1000 log-mel input.
- When the local ES-AIST artifact is present, Cortext resolves it as the default
  trimodal signal encoder unless AAIT is explicitly enabled. The existing
  256-dimensional storage/retrieval schema is preserved by using a normalized
  compact retrieval view, while the full normalized 1536d vector is attached to
  each ingress `Signal` for Soft Anchor formation.
- GGML backend loading is now process-wide and idempotent, avoiding repeated
  backend registration across ES-AIST, AAIT, LFM, and EmbeddingGemma paths.
- Internal consolidation signals no longer insert or evict working-memory chat
  slots; this keeps consolidation from perturbing the visible chat tail while it
  still drives consolidation jobs.

Verification:

```bash
cmake --build build -j 8 --target cortext_tests
./build/tests/cortext_tests "ES-AIST GGUF executes native text image and audio kernels" -s
./build/tests/cortext_tests "AAIT encoder load executes native triembed inference" -s
./build/tests/cortext_tests "Integration: scripted chat preserves turn-shaped working memory" -s
./build/tests/cortext_tests "Integration: scripted chat consolidation preserves prompt shape and graph integrity" -s
cmake --build build -j 8 --target cortext_chat
```

The ES-AIST integration test confirms `architecture=triembed`, output dimension
`1536`, native runtime availability, corrected `semantic_768_key` and
`entity_key` slices, and finite unit-norm embeddings for text, image, and audio.
The verbose run shows GGML/Metal kernels for text attention/projection,
image im2col/convolution/pooling/projection, and audio convolution/activation
/pooling/projection. The chat demo target rebuilds against the same engine path.

Correct-contract ingress ledger:

```bash
CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH=1 ./build/examples/benchmark/cortext_anchor_replay_bench \
  --ess-aist-ingress-anchor-shadow-only \
  --models=models \
  --teacher-model models/ES-AIST-81M-preview-GGUF/ES-AIST-81M_q8_0.gguf \
  --output-dir build/es_aist_ingress_anchor_shadow_full_contract
```

| policy | safe target selected | no-anchor abstain | wrong-active distinct | selected wrong-active | zero-FPR | 5% FPR |
|---|---:|---:|---:|---:|---:|---:|
| subject_only_loose | 0.015 | 0.007 | 0.022 | 0.978 | 0/270 | 0/270 |
| subject_high_split | 0.093 | 0.219 | 0.337 | 0.570 | 0/270 | 0/270 |
| subject_event_high_split | 0.089 | 0.238 | 0.337 | 0.567 | 0/270 | 0/270 |
| subject_event_very_strict | 0.048 | 0.517 | 0.581 | 0.333 | 0/270 | 0/270 |

Correct-contract ingress attention:

```bash
CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH=1 ./build/examples/benchmark/cortext_anchor_replay_bench \
  --ess-aist-ingress-attention-shadow-only \
  --models=models \
  --teacher-model models/ES-AIST-81M-preview-GGUF/ES-AIST-81M_q8_0.gguf \
  --output-dir build/es_aist_ingress_attention_shadow_full_contract
```

| policy | safe target selected | no-anchor abstain | wrong-active distinct | selected wrong-active | zero-FPR | 5% FPR |
|---|---:|---:|---:|---:|---:|---:|
| attention_very_strict | 0.033 | 0.318 | 0.304 | 0.593 | 0/270 | 0/270 |
| joint_attention_very_strict | 0.033 | 0.159 | 0.152 | 0.737 | 0/270 | 0/270 |
| joint_attention_ultra_split | 0.000 | 0.993 | 1.000 | 0.004 | 0/270 | 0/270 |

Correct-contract slice diagnostic:

```bash
CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH=1 ./build/examples/benchmark/cortext_anchor_replay_bench \
  --ess-aist-anchor-slice-shadow-only \
  --models=models \
  --teacher-model models/ES-AIST-81M-preview-GGUF/ES-AIST-81M_q8_0.gguf \
  --output-dir build/es_aist_anchor_slice_shadow_full_contract
```

Best repaired-real rows:

| score family | best view | target AUC | top-1 | top-3 | no-anchor AUC | wrong-active rejection AUC | zero-FPR | 5% FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline retrieval | semantic 768 | 0.5528 | 0.1074 | 0.2667 | 0.5401 | 0.4278 | 0 | 0 |
| micro-anchor | semantic 768 | 0.6740 | 0.1963 | 0.4444 | 0.5096 | 0.2364 | 0 | 2 |
| hierarchical attention | semantic 768 | 0.7219 | 0.2000 | 0.4444 | 0.5187 | 0.2263 | 0 | 0 |
| latent ranker only | semantic 768 | 0.7975 | 0.3037 | 0.6037 | 0.5138 | 0.1844 | 0 | 0 |

Interpretation: ES-AIST's entity geometry is better on the model-side eval and
does improve strict ingress split behavior relative to loose policies. It still
does not transfer into safe Cortext anchoring on repaired replay. The best plain
ingress ledger policy selected the safe target in only `0.093` of references;
the strictest useful policy abstained on `0.517` no-anchor controls and split
wrong-active in `0.581` of references, but selected the safe target in only
`0.048` and had zero low-FPR recovery. Attention over the corrected entity/full
views did not solve the tradeoff. These direct anchor-policy benchmarks remain
diagnostic only: ES-AIST is useful as a trimodal signal encoder, but it is not a
standalone hard anchor resolver, and retrieval ranking is not changed by the
anchor experiments.

#### ES-AIST as a signal for WM/STM/LTM engine policy

The direct ingress-ledger runs still treated ES-AIST too much like a standalone
anchor policy. The corrected interpretation is that ES-AIST emits semantic and
entity signals; Cortext owns the bind/abstain policy using WM, STM, LTM,
recency, active state, and thresholds.

New benchmark mode:

```bash
CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH=1 ./build/examples/benchmark/cortext_anchor_replay_bench \
  --es-aist-contextual-anchor-shadow-only \
  --models=models \
  --teacher-model models/ES-AIST-81M-preview-GGUF/ES-AIST-81M_q8_0.gguf \
  --output-dir build/es_aist_contextual_anchor_shadow_full_contract
```

Outputs:

- `es_aist_contextual_anchor_results.json`
- `es_aist_contextual_anchor_summary.csv`
- `es_aist_contextual_anchor_cases.csv`
- `es_aist_contextual_anchor_candidate_features.csv`
- `es_aist_contextual_anchor_failure_examples.csv`

This surface builds per-candidate features from:

- `semantic_key = z[0:768]`
- `entity_key = z[768:1536]`
- `full_key = z[0:1536]`
- working-memory prior turns
- STM prior turns
- LTM candidate memories
- WM/STM/LTM attention support, entropy, margins, recency, and source flags

Run size and runtime:

- repaired replay cases: `421`
- candidate feature rows: `7696`
- backend: `MTL0 (Apple M3 Max)`
- elapsed: `25.81 s`
- production retrieval changed: `false`

| policy | target AUC | top-1 | top-3 | no-anchor AUC | wrong-active rejection AUC | stale rejection AUC | zero-FPR | 5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic signal only | 0.5528 | 0.1074 | 0.2667 | 0.5401 | 0.4278 | 0.4833 | 0 | 0 |
| entity signal only | 0.5463 | 0.1148 | 0.2630 | 0.5338 | 0.4301 | 0.4812 | 0 | 0 |
| full signal only | 0.5505 | 0.1074 | 0.2593 | 0.5375 | 0.4283 | 0.4819 | 0 | 0 |
| entity + STM context | 0.5762 | 0.1407 | 0.2926 | 0.5248 | 0.4034 | 0.4773 | 0 | 0 |
| WM/STM/LTM attention | 0.6185 | 0.1481 | 0.3519 | 0.5268 | 0.3574 | 0.4704 | 0 | 1 |
| learned context policy, all-fit | 0.8579 | 0.3630 | 0.6741 | 0.4895 | 0.0219 | 0.6443 | 3 | 10 |
| learned context policy, delayed prequential | 0.8581 | 0.3667 | 0.6481 | 0.5001 | 0.0216 | 0.7424 | 2 | 7 |
| learned context policy, source-held-out | 0.7837 | 0.2444 | 0.6519 | 0.4860 | 0.0149 | 0.5958 | 0 | 0 |

Interpretation: ES-AIST is useful as an entity/candidate signal, but the current
engine-side policy is not good enough. WM/STM/LTM attention improves target
reachability over raw ES slices, and a lightweight learned policy reaches
source-held-out target AUC `0.7837` and top-3 `0.6519`. But wrong-active
rejection collapses (`0.0149` source-held-out), no-anchor remains near chance,
and low-FPR recovery is still zero.

The policy-vs-signal audit for the source-held-out learned policy found:

- `66` references where the target was top-3 and separated from stale/wrong
  candidates, but the calibrated policy still abstained.
- `204` references where the signal itself remained ambiguous or wrong-active
  scored above target.
- `7` controls remained high-scoring false-bind risks.

So the next experiment should not be another direct threshold. It should be a
small engine policy trained with explicit wrong-active/no-anchor objectives over
ES signal + causal WM/STM/LTM state.

#### Two-stage ES attention retrieval + safety commit

The contextual benchmark now also separates attention retrieval from commit
safety. This matches the intended engine shape:

1. Attention over ES semantic/entity/full signals and WM/STM/LTM banks proposes
   top-k candidate anchors.
2. A separate safety head decides bind vs abstain/split/create from top-k
   margins, tier agreement, recency, source continuity, active competition, and
   no-anchor pressure.

Additional outputs:

- `es_aist_attention_retrieval_stage1.csv`
- `es_aist_safety_commit_stage2.csv`
- `es_aist_two_stage_attention_safety_examples.csv`

Stage 1:

| policy | top-1 | top-3 | top-5 | selected wrong-active | selected stale | selected remote |
|---|---:|---:|---:|---:|---:|---:|
| semantic key | 0.1074 | 0.2667 | 0.3741 | 0.3563 | 0.1401 | 0.1259 |
| entity key | 0.1148 | 0.2630 | 0.3704 | 0.3515 | 0.1283 | 0.1283 |
| full key | 0.1074 | 0.2593 | 0.3852 | 0.3682 | 0.1425 | 0.1354 |
| WM/STM/LTM attention | 0.1481 | 0.3519 | 0.5000 | 0.5416 | 0.1093 | 0.0214 |

Stage 2:

| policy | safe target selected | no-anchor abstain | selected wrong-active | target AUC | top-3 | wrong-active rejection AUC | zero-FPR | 5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all-fit top-5 safety | 0.0037 | 0.9536 | 0.0000 | 0.6187 | 0.2481 | 0.3883 | 0 | 3 |
| delayed-prequential top-5 safety | 0.0296 | 0.9536 | 0.0185 | 0.6437 | 0.4000 | 0.3362 | 2 | 8 |
| source-held-out top-5 safety | 0.0000 | 0.9536 | 0.0037 | 0.5833 | 0.1296 | 0.3817 | 0 | 0 |

Interpretation: the proposed two-stage architecture is the right direction, but
the first safety head is too conservative and not source-stable. Stage 1 proves
ES + WM/STM/LTM attention is useful for candidate recall. Stage 2 proves a
safety layer can suppress no-anchor/wrong-active commits, but the current
training objective/calibration over-suppresses valid binds. The next iteration
should improve the commit head, not abandon attention retrieval.

#### Multi-head ES safety commit ablation

We then replaced the single generic safety head with four explicit heads over
the same top-5 attention proposal rows:

- bind target
- no-anchor risk
- wrong-active risk
- stale risk

The benchmark remained shadow-only and used the same q8 ES-AIST artifact and
Metal full graph:

```bash
CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH=1 ./build/examples/benchmark/cortext_anchor_replay_bench \
  --es-aist-contextual-anchor-shadow-only \
  --models=models \
  --teacher-model models/ES-AIST-81M-preview-GGUF/ES-AIST-81M_q8_0.gguf \
  --output-dir build/es_aist_contextual_anchor_shadow_set_aware_fast
```

Additional output:

- `es_aist_multi_head_safety_examples.csv`
- `es_aist_set_aware_safety_examples.csv`

Source-held-out results:

| safety policy | safe target selected | no-anchor abstain | selected wrong-active | top-3 | wrong-active rejection AUC | zero-FPR | 5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| single safety head | 0.0000 | 0.9536 | 0.0037 | 0.1296 | 0.3817 | 0 | 0 |
| multi-head bind only | 0.0037 | 0.9536 | 0.0519 | 0.3593 | 0.3269 | 1 | 2 |
| set-aware multi-head bind only | 0.0148 | 0.9536 | 0.0296 | 0.3333 | 0.3328 | 1 | 4 |
| multi-head recall-biased | 0.0000 | 0.9536 | 0.0000 | 0.2741 | 0.3411 | 0 | 0 |
| multi-head balanced | 0.0000 | 0.9536 | 0.0000 | 0.2148 | 0.3560 | 0 | 0 |
| set-aware multi-head balanced | 0.0000 | 0.9536 | 0.0000 | 0.1926 | 0.3553 | 0 | 0 |
| multi-head guarded | 0.0000 | 0.9536 | 0.0000 | 0.1852 | 0.3657 | 0 | 0 |
| set-aware multi-head guarded | 0.0000 | 0.9536 | 0.0000 | 0.1593 | 0.3734 | 0 | 0 |

Interpretation: the separated heads expose the tradeoff more clearly but do not
solve it. The bind-only head carries more target signal than the original
single safety head, reaching source-held-out top-3 `0.3593` and nonzero low-FPR
recovery (`1` at zero FPR, `2` at 5% FPR). However, that recovery comes with a
wrong-active commit rate of `0.0519`. We then added a set-aware top-k feature
surface: each candidate row is augmented with mean, max, standard deviation,
and candidate-minus-summary features over the full top-5 proposal set. This
reduces bind-only wrong-active commits to `0.0296` and improves 5% FPR recovery
to `4`, but guarded variants still recover zero safe binds. The optimized
set-aware run used `240` safety features and completed in `70.69 s`; this is
acceptable as a benchmark diagnostic but too expensive to carry as-is into a
realtime policy.

This supports the engine framing: ES-AIST + WM/STM/LTM attention is a useful
proposal layer, but the current scalar commit policy is not good enough. A
sequence-aware top-k surface helps a little, but not enough. The next useful
experiment should use an actual compact top-k item consumer rather than
hand-expanded summary features.

#### Compact sequence-MLP top-k consumer

We then tested a compact nonlinear consumer over the same top-5 proposal set.
The model uses four small deterministic MLP heads, one each for bind,
no-anchor, wrong-active, and stale risk. It is still benchmark-only and uses
only the same runtime-compatible ES/WM/STM/LTM feature rows:

```bash
CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH=1 ./build/examples/benchmark/cortext_anchor_replay_bench \
  --es-aist-contextual-anchor-shadow-only \
  --models=models \
  --teacher-model models/ES-AIST-81M-preview-GGUF/ES-AIST-81M_q8_0.gguf \
  --output-dir build/es_aist_contextual_anchor_shadow_sequence_mlp
```

Additional output:

- `es_aist_sequence_mlp_safety_examples.csv`

The MLP used `10` hidden units and `60` training epochs per head, evaluated on
delayed-prequential and source-held-out splits. The source-held-out result did
not improve safe commitment:

| policy | safe target selected | no-anchor abstain | selected wrong-active | top-3 | zero-FPR | 5% FPR |
|---|---:|---:|---:|---:|---:|---:|
| set-aware logistic bind-only | 0.0148 | 0.9536 | 0.0296 | 0.3333 | 1 | 4 |
| MLP bind-only | 0.0000 | 0.9536 | 0.0333 | 0.2148 | 0 | 0 |
| set-aware MLP bind-only | 0.0074 | 0.9536 | 0.0593 | 0.4185 | 0 | 2 |
| set-aware MLP balanced | 0.0000 | 0.9536 | 0.0519 | 0.1741 | 0 | 0 |
| set-aware MLP guarded | 0.0000 | 0.9536 | 0.0481 | 0.1778 | 0 | 0 |

Delayed-prequential looked better, but the source-held-out collapse is decisive
for promotion: set-aware MLP bind-only reached top-3 `0.4593`, zero-FPR
recovery `4`, and 5% FPR recovery `10`, but the same policy lost zero-FPR
recovery on source-held-out and selected wrong-active candidates at `0.0593`.
The run completed in `204.22 s`, making this too slow as a benchmark sweep and
not worth pursuing without a stronger signal or supervision surface.

Interpretation: nonlinear capacity over the current scalar ES/WM/STM/LTM
features does not fix commitment. The useful signal is still mostly candidate
proposal/reachability, while safe commitment remains unstable under
source-held-out transfer. At this point the scalar/top-k feature-family is
exhausted enough to stop: the next anchor experiment should change the input
surface, likely toward item-level/entity-track supervision or a model that sees
ordered multimodal entity evidence directly.

#### Entity-track oracle headroom audit

Finally, we added an offline headroom audit over the same repaired replay
candidate pool. This is not a runtime policy and does not use labels as runtime
features. It asks only whether an upstream entity/track observation would have
enough candidate coverage to solve the replay, or whether the target memories
are absent from the pool.

Artifacts:

- `build/es_aist_contextual_anchor_shadow_sequence_mlp/entity_track_oracle_headroom.json`
- `build/es_aist_contextual_anchor_shadow_sequence_mlp/entity_track_oracle_headroom.csv`

Results:

| metric | value |
|---|---:|
| cases | 421 |
| references | 270 |
| controls | 151 |
| references with target in candidate pool | 270 |
| references missing target candidate | 0 |
| controls with target label present | 0 |
| oracle safe target selected | 1.0000 |
| oracle no-anchor abstain | 1.0000 |
| oracle zero-FPR recovery | 270 / 270 |

Interpretation: the candidate pool has enough coverage if a real entity/track
signal exists. The target is present for every reference case, and controls do
not contain target-labeled candidates. Current ES/WM/STM/LTM attention reaches
only top-5 `0.5000` and over-selects wrong-active candidates, so the problem is
not LTM availability. The missing component is observable entity continuity:
either upstream entity/object/voice/track proposals or an embedding model whose
entity signal is strong enough for the engine to form and maintain those tracks.

#### Noisy entity-track signal quality sweep

We then quantified how accurate that missing entity/track signal must be. This
is still an offline headroom sweep, not a runtime policy. It starts from the
existing `WM/STM/LTM attention` score and injects a synthetic entity-track boost
with controlled:

- true target track recall;
- false track-match rate on non-target candidates;
- signal strength.

Command:

```bash
python3 tools/entity_track_signal_headroom.py \
  --artifact-dir build/es_aist_contextual_anchor_shadow_sequence_mlp
```

Artifacts:

- `entity_track_signal_quality_sweep.csv`
- `entity_track_signal_quality_sweep.json`

The baseline row reproduces the current attention proposal layer:

| mode | top-1 | top-3 | top-5 | zero-FPR | 5% FPR |
|---|---:|---:|---:|---:|---:|
| WM/STM/LTM attention | 0.1481 | 0.3519 | 0.5000 | 0 | 1 |

With zero false track matches and full target-track recall, the required signal
strength breakpoints were:

| target recovery | zero-FPR strength | 5% FPR strength |
|---:|---:|---:|
| >= 25% | 0.30 | 0.20 |
| >= 50% | 0.30 | 0.30 |
| >= 80% | 0.45 | 0.45 |
| >= 95% | 0.60 | 0.45 |

The precision requirement is sharp. At target recall `1.0` and signal strength
`0.60`, increasing false track matches from `0.0` to only `0.005` dropped
zero-FPR recovery from `270 / 270` to `25 / 270`; at `0.01`, recovery was
`10 / 270`.

| false track rate | zero-FPR recovery | 5% FPR recovery |
|---:|---:|---:|
| 0.000 | 270 | 270 |
| 0.005 | 25 | 90 |
| 0.010 | 10 | 38 |
| 0.020 | 10 | 27 |
| 0.050 | 7 | 22 |
| 0.100 | 0 | 10 |

Interpretation: entity-aware anchoring is not mainly a recall problem once the
candidate pool is available. It is a precision problem. The engine needs an
upstream entity/object/voice/track signal with very low false-track rate, or a
commit policy with strong independent evidence that a proposed track match is
valid. Otherwise even small false-track leakage destroys zero-FPR behavior.

#### Multimodal anchor-chain headroom

The repaired replay surface tests candidate commitment, but the motivating
product behavior is a chain: person reference, media/object handoff, and a
later mixed person/object question such as "How long has he had it?" We added a
deterministic synthetic headroom audit for that structure:

```bash
python3 tools/multimodal_anchor_chain_headroom.py \
  --episodes 200 \
  --output-dir build/multimodal_anchor_chain_headroom
```

Artifacts:

- `multimodal_anchor_chain_cases.csv`
- `multimodal_anchor_chain_headroom.csv`
- `multimodal_anchor_chain_headroom.json`

The benchmark generates `1200` cases over six families:

- person pronoun: "what did he say?"
- media/object reference: "send it"
- owner/object relation: "how long has he had it?"
- wrong-active person
- cross-modal person-to-media image handoff
- no-anchor topic shift

Candidate units include person tracks, media/object tracks, and relation
candidates such as `owner(Jared, dog)`. The key comparison is what upstream
state the engine would need:

| policy | safe target selected | selected wrong-active | relation top-1 | media top-1 | zero-FPR safe recovery |
|---|---:|---:|---:|---:|---:|
| recency only | 0.2000 | 0.8000 | 0.0000 | 0.0000 | 200 / 1000 |
| entity only | 0.4000 | 0.6000 | 0.0000 | 0.0000 | 400 / 1000 |
| entity + object/media | 0.8000 | 0.2000 | 0.0000 | 1.0000 | 800 / 1000 |
| entity + object/media + relation | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1000 / 1000 |

Interpretation: entity tracks alone are not enough for the motivating chain.
They handle the "he" part but not "it", image/object continuity, or
owner/object questions. Entity + object/media tracks solve media handoff, but
still fail the mixed "he had it" relation case. The engine needs relation or
event-edge state that can represent owner/sender/depicts/attached-to style
links between people, media, and objects. This should become an explicit
requirement for the next modality-agnostic anchor model or upstream proposal
interface.

#### Anchor experiment audit

We consolidated the current evidence in
[`docs/anchor-experiment-audit.md`](anchor-experiment-audit.md). The audit maps
the product objective to concrete artifacts and gates.

Current status:

- ES-AIST + WM/STM/LTM attention is useful as a proposal layer.
- Single scalar, multi-head, set-aware, and small MLP commit consumers all fail
  source-held-out safe commitment.
- The repaired replay candidate pool is not the blocker: `270 / 270` reference
  cases have the target candidate present.
- The missing signal is precision-first entity/object/media/relation
  continuity.
- Entity-only is insufficient for the motivating chain; object/media tracks and
  relation/event edges are required.

Conclusion: additional policy sweeps over the current scalar ES/WM/STM/LTM
feature surface are not productive. The next experiment requires a new input
surface: either a model that emits high-precision entity/object/relation
signals, upstream multimodal track proposals, or real/high-fidelity
entity-object-relation episode labels.

We defined that next input surface in
[`docs/anchor-signal-contract.md`](anchor-signal-contract.md). The contract
requires typed entity, object/media, event, and relation proposals; strict false
proposal gates; repaired replay evaluation; multimodal chain evaluation; and
manual-gold validation before any further engine policy promotion.

#### Executable anchor-signal contract evaluator

We converted the anchor-signal contract into an executable scaffold evaluator:

```bash
python3 tools/evaluate_anchor_signal_contract.py \
  --episodes 200 \
  --output-dir build/anchor_signal_contract_eval
```

Artifacts:

- `anchor_signal_contract_eval_cases.jsonl`
- `anchor_signal_proposals_empty.jsonl`
- `anchor_signal_proposals_entity_only.jsonl`
- `anchor_signal_proposals_entity_object.jsonl`
- `anchor_signal_proposals_entity_object_relation.jsonl`
- `anchor_signal_proposals_entity_object_relation_noisy_0.005.jsonl`
- `anchor_signal_proposals_entity_object_relation_noisy_0.01.jsonl`
- `anchor_signal_contract_results.json`
- `anchor_signal_contract_summary.csv`
- `anchor_signal_false_proposal_audit.csv`
- `anchor_signal_failure_examples.csv`

This evaluator is not model evidence. The built-in proposal modes are
headroom/scaffold checks so future model or upstream proposal artifacts can be
judged against the same gates. External cases and proposals can be passed with
`--cases-jsonl` and `--proposals-jsonl`.

The synthetic contract surface uses `1200` cases over the same six chain
families: person reference, cross-modal media handoff, object reference,
person/object relation, wrong-active reference, and no-anchor topic shift.

| proposal mode | required gates | target gates | entity recall | object/media recall | relation recall | entity false / case | object false / case | relation false / case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| empty | fail | fail | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| entity only | fail | fail | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| entity + object/media | fail | fail | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| entity + object/media + relation | pass | pass | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| entity + object/media + relation, noisy 0.005 | pass | fail | 1.0000 | 1.0000 | 1.0000 | 0.0025 | 0.0033 | 0.0042 |
| entity + object/media + relation, noisy 0.01 | fail | fail | 1.0000 | 1.0000 | 1.0000 | 0.0083 | 0.0092 | 0.0092 |

Interpretation: the executable contract reproduces the architectural
requirement in a form we can hand to future model artifacts. Entity proposals
alone cannot cover object/media handoff or mixed relation questions.
Entity+object/media still misses `owner/sender/depicts` style relation
evidence. A complete relation scaffold passes, but a tiny false-proposal rate
already fails the stricter target gates. This is consistent with the repaired
replay headroom result: anchor signals must be precision-first, not just high
recall.

#### Existing ES-AIST signal contract threshold audit

We then tested whether the current ES-AIST + WM/STM/LTM scalar feature surface
could itself be thresholded into a high-precision signal proposal source:

```bash
python3 tools/es_aist_signal_contract_threshold_audit.py \
  --artifact-dir build/es_aist_contextual_anchor_shadow_sequence_mlp \
  --output-dir build/es_aist_signal_contract_threshold_audit
```

Artifacts:

- `es_aist_signal_contract_threshold_audit.csv`
- `es_aist_signal_contract_threshold_results.json`
- `es_aist_signal_contract_failure_examples.csv`

The audit evaluates semantic, entity, full, margin, context-only, and
stage-1-attention score views over the `421` repaired replay cases. It asks how
many target candidates can be selected while false candidate proposals remain
under the contract gates.

Best rows:

| gate | best view | false candidate rate | target recall | selected targets |
|---|---|---:|---:|---:|
| required false gate `<0.005` | full margin | 0.00498 | 0.0296 | 8 / 270 |
| strict target false gate `<=0.001` | entity margin | 0.00094 | 0.0037 | 1 / 270 |

At the same required false gate, the `stage1_attention` view selected only
`1 / 270` targets. At 5% false-candidate rate it selected `27 / 270` targets,
which is still weak and far outside the precision requirement.

Interpretation: the existing ES-AIST/WM/STM/LTM scalar scores cannot simply be
thresholded into the missing signal contract. They remain useful as a proposal
or ranking surface, but they do not provide a high-precision entity/object/
relation signal. This closes the "can the current signal surface do it with
better thresholding?" branch.

#### ES-AIST storage-time WM/STM/LTM attention benchmark

The earlier chronological ingress benchmark was still too synthetic: it used a
small scripted chain instead of the real conversation streams where storage
decisions happen. We replaced the default path in
`cortext_ingress_anchor_formation_bench` with a real storage-time shadow test.
This benchmark is not a retrieval reranker and does not consume retrieved
candidates.

Exact runtime process:

1. Load real conversation turns from PersonaChat, TopicalChat, and Taskmaster.
2. Replay each conversation chronologically.
3. At turn `t`, encode only the current incoming signal with
   `ES-AIST-81M_q8_0.gguf`.
4. Before writing the current memory, attend over already-stored prior memories:
   WM is age `<=4`, STM is age `5..32`, and LTM is age `>32`.
5. The attention score combines ES semantic/entity/full cosines, recency, and
   small tier biases. If score and margin pass, the shadow ledger updates the
   selected existing anchor; otherwise it creates a new shadow anchor for the
   current memory.
6. Only after that storage-time action does the benchmark apply evaluation
   labels to score target reuse, no-anchor safety, wrong-active selection, and
   stale selection.

The output contract records `retrieved_candidate_count=0`,
`uses_retrieved_candidates=0`, and `runtime_policy_uses_labels=0` for every
row. The final run also verified zero duplicate current-ingress steps per
policy.

Build and run:

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

Artifacts:

- `ingress_storage_attention_cases.csv`
- `ingress_storage_attention_links.csv`
- `ingress_storage_attention_failure_examples.csv`
- `ingress_storage_attention_results.json`
- `ingress_storage_attention_model.json`

Model/runtime check:

| field | value |
|---|---:|
| model | `models/ES-AIST-81M-preview-GGUF/ES-AIST-81M_q8_0.gguf` |
| SHA-256 | `3c77c5aa277c1041c325723d47221970686ec0194b0aa59f3617a7f1192a5699` |
| backend | `MTL0 (Apple M3 Max)` |
| kernel ops | `true` |
| full text graph ops | `true` |
| semantic key | `z[0:768]` |
| entity key | `z[768:1536]` |
| full key | `z[0:1536]` |
| conversations | `71` |
| evaluated ingress steps | `421` |
| unique text encodes | `1,496` |
| stored memory updates | `1,530` per policy |
| mean encode latency | `2.49 ms` |
| p95 encode latency | `3.93 ms` |

| storage-time policy | selected target | no-anchor abstain | selected wrong-active | selected stale | zero-FPR recovery | 5% FPR recovery |
|---|---:|---:|---:|---:|---:|---:|
| loose | 109 / 270 | 36 / 151 | 32 / 270 | 30 / 270 | 0 | 6 |
| mid | 23 / 270 | 103 / 151 | 15 / 270 | 5 / 270 | 0 | 2 |
| STM weighted | 7 / 270 | 132 / 151 | 6 / 270 | 0 / 270 | 0 | 4 |
| strict | 4 / 270 | 141 / 151 | 3 / 270 | 0 / 270 | 0 | 2 |
| high precision | 2 / 270 | 151 / 151 | 0 / 270 | 0 / 270 | 0 | 0 |

Interpretation: this is now the right architectural test. ES-AIST is used as a
storage-time signal over real prior WM/STM/LTM memory state, before retrieval,
with production retrieval unchanged. The result is still negative for replacing
a dedicated anchor model with the current hand-built attention policy. Loose
attention recovers targets but false-binds no-anchor and wrong-active cases.
Strict/high-precision attention protects no-anchor controls but collapses target
reuse and still has zero zero-FPR recovery. The bottleneck is therefore not
benchmark location anymore; it is the missing commit/safety policy or a stronger
model-side entity/event/object signal.

#### Model-free adaptive ingress ledger

Because a fixed trained commit model would not naturally adapt to a user's
evolving memory stream, we then added a model-free adaptive ledger in the same
benchmark. This variant still uses ES-AIST only as an ingress signal, but the
ledger state evolves online:

- every stored memory creates or updates a provisional/active anchor;
- anchors carry support count, confidence, stability, contradiction count, and
  status;
- update decisions are relative over active anchors, using attention entropy,
  score margin, confidence, support, recency tier, and entity-vs-event split
  pressure;
- unsafe or ambiguous evidence creates/splits a new anchor instead of consuming
  an existing one.

Additional artifacts:

- `ingress_adaptive_anchor_cases.csv`
- `ingress_adaptive_anchor_candidates.csv`
- `ingress_adaptive_anchor_links.csv`
- `ingress_adaptive_anchor_states.csv`
- `ingress_adaptive_anchor_failure_examples.csv`
- `ingress_adaptive_anchor_manual_audit.csv`
- `ingress_adaptive_anchor_manual_review_sample.csv`

| adaptive policy | selected target | no-anchor abstain | selected wrong-active | selected stale | anchor states | zero-FPR recovery | 5% FPR recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| evidence loose | 266 / 270 | 0 / 151 | 2 / 270 | 2 / 270 | 89 | 0 | 6 |
| evidence mid | 252 / 270 | 2 / 151 | 5 / 270 | 4 / 270 | 146 | 0 | 3 |
| hysteresis | 237 / 270 | 10 / 151 | 11 / 270 | 9 / 270 | 246 | 0 | 2 |
| split guard | 247 / 270 | 2 / 151 | 8 / 270 | 8 / 270 | 193 | 0 | 2 |
| high precision | 165 / 270 | 32 / 151 | 19 / 270 | 10 / 270 | 550 | 0 | 3 |
| ultra precision | 5 / 270 | 140 / 151 | 3 / 270 | 0 / 270 | 1,456 | 0 | 1 |

The same run now reports a soft-anchor view, because hard bind/abstain is too
strict for actual conversational anchoring. A wrong or ambiguous reference can
still be useful when it is surfaced as a tentative candidate rather than
written as a durable fact.

| adaptive policy | hard selected target | tentative target top-3 | useful uncertain references | hard wrong commits | no-anchor hard binds |
|---|---:|---:|---:|---:|---:|
| evidence loose | 266 / 270 | 270 / 270 | 4 / 270 | 153 / 421 | 151 / 151 |
| evidence mid | 252 / 270 | 268 / 270 | 16 / 270 | 156 / 421 | 149 / 151 |
| hysteresis | 237 / 270 | 261 / 270 | 24 / 270 | 156 / 421 | 141 / 151 |
| split guard | 247 / 270 | 268 / 270 | 21 / 270 | 159 / 421 | 149 / 151 |
| high precision | 165 / 270 | 226 / 270 | 61 / 270 | 142 / 421 | 119 / 151 |
| ultra precision | 5 / 270 | 118 / 270 | 113 / 270 | 14 / 421 | 11 / 151 |

Interpretation: the model-free online ledger does produce the expected
continuity behavior. Loose adaptive evidence keeps the correct target anchor in
`266 / 270` references with only `2 / 270` wrong-active selections. But it does
that by becoming too sticky: no-anchor abstention collapses to `0 / 151`.
Increasing hysteresis, split pressure, and precision thresholds gradually
restores no-anchor safety, but the tradeoff returns; ultra precision abstains
on `140 / 151` controls while recovering only `5 / 270` targets. This tells us
natural online state evolution helps continuity but still does not expose a
reliable hard no-anchor/split signal from ES-AIST plus scalar WM/STM/LTM state
alone.

Manual audit changed the interpretation. In the high-precision policy, several
benchmark no-anchor "false binds" are acceptable as tentative conversational
context: for example, "that is terrible i am sorry . why do they hate you ?"
points to the prior family-hate turn, and "they will try to acquire it . lol"
points to the prior bank/tin-market turn. These should not become durable
identity facts, but they are not useless. The manual sample contains `12` real
chat rows: `2` acceptable hard binds, `3` acceptable tentative-only references,
`2` acceptable no-hard-bind cases, `2` ambiguous-but-useful rows, and `3` bad
or generic anchor signals. The right engine contract is therefore soft ingress
anchoring:

- keep top-k tentative anchors with scores and memory-tier evidence;
- allow ambiguous references to be returned as uncertain context;
- require repeated support before durable consolidation;
- treat bad generic matches such as "Ok"/"Bye" as tentative at most;
- score future benchmarks by useful uncertain references and wrong durable
  commits, not by zero-FPR alone.

#### Anchor strength and label pass

We then made that contract explicit in the benchmark artifacts. Each adaptive
candidate, case, link, and final anchor state now includes:

- `anchor_strength`: continuous evidence in `[0, 1]`;
- `anchor_label`: a derived policy label:
  - `none`
  - `tentative`
  - `ambiguous`
  - `durable`
  - `decayed`
  - `rejected`

The labels are derived from score, attention, margin, entropy, support count,
contradiction count, confidence, and decay state. They are not runtime labels
or gold supervision. This pass wrote:

- `build/ingress_storage_attention_es_aist_anchor_strength_label/ingress_adaptive_anchor_cases.csv`
- `build/ingress_storage_attention_es_aist_anchor_strength_label/ingress_adaptive_anchor_candidates.csv`
- `build/ingress_storage_attention_es_aist_anchor_strength_label/ingress_adaptive_anchor_links.csv`
- `build/ingress_storage_attention_es_aist_anchor_strength_label/ingress_adaptive_anchor_states.csv`
- `build/ingress_storage_attention_es_aist_anchor_strength_label/ingress_adaptive_anchor_manual_audit.csv`
- `build/ingress_storage_attention_es_aist_anchor_strength_label/ingress_adaptive_anchor_manual_review_sample.csv`

| adaptive policy | target top-3 | hard wrong commits | durable selections | ambiguous selections | tentative selections | no-anchor durable binds |
|---|---:|---:|---:|---:|---:|---:|
| evidence loose | 270 / 270 | 153 / 421 | 379 | 29 | 13 | 138 / 151 |
| high precision | 226 / 270 | 142 / 421 | 158 | 175 | 88 | 64 / 151 |
| ultra precision | 118 / 270 | 14 / 421 | 9 | 186 | 226 | 3 / 151 |

Interpretation: the terms are right, and the audit is now legible. The current
label derivation is still too willing to call sticky no-anchor matches durable
under loose/high-precision policies. Ultra precision shows the opposite corner:
very few durable false binds, many tentative/ambiguous links, and much lower
hard target recovery. This supports the engine design but not the current
thresholds as production policy. `anchor_strength` should be the stored
continuous value; `anchor_label` should be a derived/cached view that can be
recalibrated without rewriting evidence.

#### Anchor promotion label ablation

We then swept the derived-label and knob-shaped promotion rules without changing
the underlying storage-time ES-AIST signal or the production retrieval path.
This pass kept ES-AIST as a signal model only: the engine used the
`semantic_key`, `entity_key`, and `full_key` together with already-stored
WM/STM/LTM state, online support, contradiction, entropy, margin, recency, and
source continuity. No retrieved candidates or labels were runtime inputs.

The run wrote:

- `build/ingress_storage_attention_es_aist_label_ablation/ingress_storage_attention_results.json`
- `build/ingress_storage_attention_es_aist_label_ablation/ingress_adaptive_anchor_cases.csv`
- `build/ingress_storage_attention_es_aist_label_ablation/ingress_adaptive_anchor_candidates.csv`
- `build/ingress_storage_attention_es_aist_label_ablation/ingress_adaptive_anchor_links.csv`
- `build/ingress_storage_attention_es_aist_label_ablation/ingress_adaptive_anchor_states.csv`
- `build/ingress_storage_attention_es_aist_label_ablation/ingress_adaptive_anchor_manual_audit.csv`
- `build/ingress_storage_attention_es_aist_label_ablation/ingress_adaptive_anchor_manual_review_sample.csv`

| adaptive policy | target top-3 | hard target commits | hard wrong commits | no-anchor hard binds | no-anchor durable binds | durable | ambiguous | tentative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| evidence loose | 270 / 270 | 266 / 270 | 153 / 421 | 151 / 151 | 138 / 151 | 379 | 29 | 13 |
| high precision | 226 / 270 | 165 / 270 | 142 / 421 | 119 / 151 | 64 / 151 | 158 | 175 | 88 |
| high precision + generic guard | 226 / 270 | 165 / 270 | 142 / 421 | 119 / 151 | 57 / 151 | 144 | 175 | 102 |
| high precision + margin guard | 226 / 270 | 165 / 270 | 142 / 421 | 119 / 151 | 53 / 151 | 130 | 175 | 116 |
| high precision + conservative label | 226 / 270 | 165 / 270 | 142 / 421 | 119 / 151 | 45 / 151 | 114 | 175 | 132 |
| knob: high focus | 168 / 270 | 66 / 270 | 87 / 421 | 65 / 151 | 16 / 151 | 37 | 177 | 207 |
| ultra precision | 118 / 270 | 5 / 270 | 14 / 421 | 11 / 151 | 3 / 151 | 9 | 186 | 226 |

Interpretation: the signal model is not the policy, and the label ablation makes
that boundary explicit. Label-only guards improve durable safety without
changing candidate reachability: the conservative high-precision label reduces
no-anchor durable binds from `64 / 151` to `45 / 151` while keeping the same
target top-3 (`226 / 270`) and hard target commits (`165 / 270`). That is useful
but not sufficient. The knob-shaped high-focus corner reduces no-anchor durable
binds to `16 / 151` and hard wrong commits to `87 / 421`, but target top-3 drops
to `168 / 270`. Ultra precision is the safe extreme (`3 / 151` no-anchor
durable binds), but it recovers almost no hard targets. This means the engine
should persist `anchor_strength` and top-k tentative evidence, derive
`anchor_label` from knobs, and promote to durable only after repeated support or
stronger typed evidence. It should not ask ES-AIST to make final anchor
commitments.

#### Soft Anchor policy ablation

After the research synthesis, we converted the Soft Anchor checklist into
actual benchmark policy toggles rather than label-only variants. The new pass
adds:

- ES view weights: semantic-only, entity-only, full-only, semantic+entity, and
  combined;
- memory-tier inclusion: WM-only, WM+STM, and STM+LTM without WM;
- generic-turn suppression as an action gate;
- null/no-anchor hypothesis as an action gate;
- repeated-support update gating;
- contradiction-penalty removal;
- entropy/margin gate removal.

The run wrote:

- `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/ingress_storage_attention_results.json`
- `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/soft_anchor_ablation_summary.csv`
- `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/ingress_adaptive_anchor_cases.csv`
- `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/ingress_adaptive_anchor_candidates.csv`
- `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/ingress_adaptive_anchor_links.csv`
- `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/ingress_adaptive_anchor_states.csv`

| adaptive policy | target top-3 | hard target commits | hard wrong commits | no-anchor hard binds | no-anchor durable binds | no-anchor abstain | 5% FPR recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| high precision baseline | 226 / 270 | 165 / 270 | 142 / 421 | 119 / 151 | 64 / 151 | 32 / 151 | 3 |
| generic action suppressor | 218 / 270 | 149 / 270 | 135 / 421 | 104 / 151 | 52 / 151 | 47 / 151 | 8 |
| no contradiction penalty | 226 / 270 | 163 / 270 | 130 / 421 | 106 / 151 | 42 / 151 | 45 / 151 | 4 |
| no entropy/margin gate | 239 / 270 | 189 / 270 | 151 / 421 | 129 / 151 | 77 / 151 | 22 / 151 | 4 |
| null hypothesis | 176 / 270 | 154 / 270 | 120 / 421 | 99 / 151 | 69 / 151 | 52 / 151 | 3 |
| null + generic suppressor | 170 / 270 | 145 / 270 | 113 / 421 | 91 / 151 | 66 / 151 | 60 / 151 | 9 |
| repeated-support update gate | 132 / 270 | 0 / 270 | 0 / 421 | 0 / 151 | 0 / 151 | 151 / 151 | 0 |
| semantic-only view | 247 / 270 | 214 / 270 | 154 / 421 | 134 / 151 | 88 / 151 | 17 / 151 | 7 |
| entity-only view | 215 / 270 | 139 / 270 | 141 / 421 | 108 / 151 | 43 / 151 | 43 / 151 | 6 |
| full-only view | 230 / 270 | 181 / 270 | 149 / 421 | 124 / 151 | 65 / 151 | 27 / 151 | 4 |
| semantic+entity view | 225 / 270 | 169 / 270 | 152 / 421 | 122 / 151 | 65 / 151 | 29 / 151 | 4 |
| WM only | 242 / 270 | 167 / 270 | 143 / 421 | 120 / 151 | 66 / 151 | 31 / 151 | 3 |
| STM+LTM, no WM | 21 / 270 | 1 / 270 | 39 / 421 | 36 / 151 | 3 / 151 | 115 / 151 | 0 |

Interpretation: the ablation reinforces the soft-anchor shape. Semantic evidence is
the recall view: semantic-only keeps the target in top-3 for `247 / 270` and
hard-binds `214 / 270` references, but it also produces `88 / 151` no-anchor
durable binds. Entity evidence is the safer view: entity-only reduces
no-anchor durable binds to `43 / 151` and increases no-anchor abstention, but it
loses target recall. Full/context evidence behaves between those two. The
engine should therefore not collapse the views into one score; it should use
semantic evidence for tentative reachability and entity evidence as a durable
safety guard.

The action-level suppressors helped but did not solve commitment. Generic
suppression improves both no-anchor hard binds and 5% FPR recovery (`8`
references), but it is still far from promotion-ready. The null/no-anchor
hypothesis lowers hard wrong commits but worsens durable-label quality under the
current thresholds, so it needs calibration as a competing hypothesis rather
than a simple post-score cutoff. Requiring support before any update is too
strong: it achieves perfect safety by preventing all hard target commits. The
right repeated-support design is not "block all updates until support exists";
it is "allow tentative updates, but promote durable only after independent
support." Removing the current contradiction penalty unexpectedly improves
durable safety, which means the existing contradiction formula is not yet a
reliable signal and should remain experimental.

#### Soft Anchor v1 hypothesis pass

We then implemented the first bounded multi-hypothesis soft anchor pass in the same
benchmark. This version adds explicit `H_none` and `H_new` hypotheses, stores
top-k `SoftAnchorLink` rows, separates soft tentative refresh from hard
`update_existing`, and keeps durable promotion conservative. The run wrote:

- `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/ingress_storage_attention_results.json`
- `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/soft_anchor_v1_update_sweep_summary.csv`
- `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/ingress_adaptive_anchor_soft_links.csv`

| adaptive policy | target top-3 soft | hard target commits | hard wrong commits | no-anchor hard binds | no-anchor durable binds | no-anchor abstain | soft updates | distinct wrong-active anchors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high precision baseline | 226 / 270 | 165 / 270 | 142 / 421 | 119 / 151 | 64 / 151 | 32 / 151 | 0 | 75 / 270 |
| soft anchor v1 | 261 / 270 | 0 / 270 | 0 / 421 | 0 / 151 | 0 / 151 | 151 / 151 | 1408 | 0 / 270 |
| v1 high focus | 261 / 270 | 0 / 270 | 0 / 421 | 0 / 151 | 0 / 151 | 151 / 151 | 1408 | 0 / 270 |
| v1 high sensitivity | 261 / 270 | 0 / 270 | 0 / 421 | 0 / 151 | 0 / 151 | 151 / 151 | 1408 | 0 / 270 |
| v1 update strict | 261 / 270 | 0 / 270 | 0 / 421 | 0 / 151 | 0 / 151 | 151 / 151 | 1367 | 7 / 270 |
| v1 update very strict | 242 / 270 | 0 / 270 | 0 / 421 | 0 / 151 | 0 / 151 | 151 / 151 | 1135 | 51 / 270 |
| v1 no generic | 270 / 270 | 0 / 270 | 0 / 421 | 0 / 151 | 0 / 151 | 151 / 151 | 1459 | 0 / 270 |

Interpretation: the v1 soft anchor is the first engine-side policy that preserves a
large soft target set while eliminating hard and durable false binds on this
slice. The default v1 keeps the target in the retained top-3 for `261 / 270`
references and emits no hard commits, so every no-anchor control is protected
from durable false binding. However, the default soft-update threshold collapses
most of each conversation into one continuity anchor: wrong-active distinct
anchors fall to `0 / 270`. Tightening soft-update thresholds restores some
anchor separation (`51 / 270` distinct wrong-active anchors in the very-strict
variant) while still avoiding hard/durable false binds, but target top-3 drops
to `242 / 270`.

This is acceptable only as a **soft anchor evidence layer**, not as finished human
subject anchoring. ES-AIST plus WM/STM/LTM can preserve useful uncertain
continuity without unsafe durable commits, but the current engine policy still
does not reliably split people/subjects. The next implementation should keep
soft links and conservative durable promotion, then improve split/new-anchor
pressure with better boundary, source, and entity-specific evidence instead of
turning soft links back into hard retrieval-time binds.

#### Soft Anchor F/S/T sweep

We then generated a full `3 x 3 x 3` grid over Soft Anchor v1:

- `Focus`: `0.25`, `0.50`, `0.75`
- `Sensitivity`: `0.25`, `0.50`, `0.75`
- `Stability`: `0.25`, `0.50`, `0.75`

The benchmark wrote:

- `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/ingress_storage_attention_results.json`
- `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/soft_anchor_fst_sweep_summary.csv`
- `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/soft_anchor_fst_monotonicity.json`
- `build/ingress_storage_attention_es_aist_soft_anchor_fst_sweep/ingress_adaptive_anchor_soft_links.csv`

All 27 Soft Anchor settings preserved durable safety on this slice:
`0` hard wrong commits, `0` no-anchor hard binds, `0` no-anchor durable binds,
and `0` durable retained soft links. The knobs therefore changed soft-context
behavior without reintroducing unsafe hard commitment.

Mean behavior by Focus:

| Focus | target top-3 soft | retained soft links | ambiguous labels | wrong-active distinct | creates | soft updates |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 261.0 | 1408.0 | 93.3 | 0.0 | 71.0 | 1408.0 |
| 0.50 | 261.0 | 1415.3 | 28.7 | 0.0 | 71.7 | 1406.7 |
| 0.75 | 255.7 | 1418.3 | 224.3 | 21.3 | 169.7 | 1305.3 |

Mean behavior by Sensitivity:

| Sensitivity | target top-3 soft | retained soft links | ambiguous labels | wrong-active distinct | creates | soft updates |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 256.0 | 1397.7 | 169.0 | 14.7 | 144.0 | 1332.0 |
| 0.50 | 260.7 | 1428.3 | 113.0 | 5.3 | 93.3 | 1384.3 |
| 0.75 | 261.0 | 1415.7 | 64.3 | 1.3 | 75.0 | 1403.7 |

Mean behavior by Stability:

| Stability | target top-3 soft | retained soft links | ambiguous labels | wrong-active distinct | creates | soft updates | active TTL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 259.2 | 1413.9 | 115.4 | 7.1 | 104.1 | 1373.3 | 26.0 |
| 0.50 | 259.2 | 1413.9 | 115.4 | 7.1 | 104.1 | 1373.3 | 36.0 |
| 0.75 | 259.2 | 1413.9 | 115.4 | 7.1 | 104.1 | 1373.3 | 46.0 |

Monotonicity checks:

| expected relation | adjacent checks passed |
|---|---:|
| Focus up lowers target top-3 recall or keeps it flat | 18 / 18 |
| Focus up improves wrong-active distinct-anchor separation | 18 / 18 |
| Sensitivity up improves target top-3 recall | 18 / 18 |
| Sensitivity up increases retained soft links | 12 / 18 |
| Stability up increases configured active TTL | 18 / 18 |
| Stability up increases or preserves actual soft updates | 18 / 18 |
| Stability up lowers or preserves create count | 18 / 18 |

Interpretation: Focus and Sensitivity are usable but not final. Focus behaves
like a precision/split knob: high Focus lowers soft recall but improves
wrong-active separation. Sensitivity behaves like a recall knob: high
Sensitivity recovers more targets and accepts more soft updates, but it reduces
wrong-active separation. Stability is now isolated from the soft-update
acceptance thresholds: increasing Stability raises configured TTL/support
window length without changing short-horizon soft acceptance behavior. On this
short replay that appears as flat soft-update/create counts rather than more
continuity, which is acceptable for the first engine mapping and avoids the
previous bug where high Stability became a stricter commit filter.

The implementation recommendation is therefore:

- keep Focus as the main precision/split control;
- keep Sensitivity as the tentative recall/control-volume control;
- map Stability only to decay, link half-life, support windows, and pruning in
  the first engine implementation;
- keep durable promotion disabled until a longer decay-heavy replay validates
  that high Stability preserves useful soft continuity without cross-boundary
  overreach.

#### Soft Anchor consumption ablation

The earlier Soft Anchor runs measured formation: did chronological ingress retain
the target as a soft hypothesis without making unsafe hard commits? We then added
a separate consumption pass that asks what a chat/human/LLM consumer would
actually surface from the formed soft links.

The run wrote:

- `build/soft_anchor_consumption_full_ablation_v2/soft_anchor_consumption_results.json`
- `build/soft_anchor_consumption_full_ablation_v2/soft_anchor_consumption_ablation_summary.csv`
- `build/soft_anchor_consumption_full_ablation_v2/soft_anchor_consumption_knob_sweep_summary.csv`
- `build/soft_anchor_consumption_full_ablation_v2/soft_anchor_consumption_cases.csv`
- `build/soft_anchor_consumption_full_ablation_v2/soft_anchor_consumption_surfaces.csv`
- `build/soft_anchor_consumption_full_ablation_v2/soft_anchor_consumption_failure_examples.csv`

| consumer policy | useful target surfaced | no-anchor surfaced | wrong-only surfaced | harmful contexts | useful / harmful | mean links | p95 chars |
|---|---:|---:|---:|---:|---:|---:|---:|
| no consumption | 0 / 270 | 0 / 151 | 0 | 0 | 0.00 | 0.000 | 0 |
| default | 15 / 270 | 8 / 151 | 0 | 8 | 1.88 | 0.055 | 69 |
| high recall top-k | 252 / 270 | 139 / 151 | 5 | 148 | 1.70 | 0.950 | 208 |
| low-F / mid-S | 29 / 270 | 16 / 151 | 0 | 16 | 1.81 | 0.107 | 101 |
| low-F / high-S | 155 / 270 | 77 / 151 | 1 | 78 | 1.99 | 0.553 | 176 |
| high-F / mid-S | 2 / 270 | 0 / 151 | 0 | 0 | 2.00 | 0.005 | 0 |
| durable only | 0 / 270 | 0 / 151 | 0 | 0 | 0.00 | 0.000 | 0 |

Interpretation: formation and consumption are not the same result. Soft Anchor
formation is valuable because it preserves useful uncertainty. Consumption is a
separate UX policy that is not ready to default on. High-recall consumption gives
the human/LLM many chances to recover the referent (`252 / 270`), but it also
surfaces too much context on no-anchor controls (`139 / 151`). Conservative
consumption is safe but mostly silent (`15 / 270`). This supports keeping
formation in the engine while leaving surfaced consumption out until a manually
reviewed chat slice shows the hints are useful more often than harmful.

Knob effects in the consumption pass:

- Focus behaves as the precision/volume knob: mean surfaced links, harmful
  context rate, and no-anchor surfacing all decreased in `18 / 18` adjacent
  Focus checks.
- Sensitivity behaves as the recall/volume knob: target surfacing and mean
  surfaced links increased in `18 / 18` adjacent Sensitivity checks.
- Stability is still under-stressed by this replay. Target surfacing changed in
  only `1 / 18` adjacent Stability checks, mean links changed in `2 / 18`, and
  no LTM surfaces were produced. Stability should remain mapped to decay,
  support windows, and pruning, but it needs a longer decay-heavy replay before
  we claim behavioral validation.

#### Soft Anchor consumption experiment plan

The current conclusion is "formation is in, consumption is out." To move
consumption in, we wrote the design and experiment plan in
`docs/soft-anchor-consumption-experiments.md`. The plan treats consumption as a
single product behavior to earn, not as a permanent runtime switch.

Candidate consumption shapes to test:

- silent safety consumer: use anchor uncertainty only to avoid overconfident
  memory use;
- possible-continuity hint: surface uncertain top-k anchor candidates;
- anchor-neighborhood context: include compact evidence memories from a formed
  anchor;
- ambiguous candidate set: preserve multiple plausible referents instead of
  forcing top-1;
- clarification consumer: ask when ambiguity is high and the interaction can
  tolerate a question;
- LLM self-selection consumer: pass structured uncertain candidates to the LLM;
- UI-only human hint: show chips/evidence to the human without prompt injection;
- retrieval annotation: annotate returned memories with pre-formed anchor links
  without changing ranking;
- context packing: group or deduplicate context by anchor;
- neighborhood expansion: expand ordinary retrieval hits with same-anchor
  neighbors;
- current-turn anchor recall: include evidence from the current turn's formed
  anchor when ordinary retrieval misses it;
- durable-only fact consumer: allow only durable links into fact formation;
- decayed-link reminder: surface old links only as older possible context;
- group-anchor consumer: handle `we`-style entity sets;
- cross-modal evidence consumer: consume text/image/audio/voice anchor evidence;
- correction-aware consumer: learn from user corrections;
- ask-or-show hybrid: decide among no hint, possible context, and clarification.

The planned experiments are:

1. replay consumption contract over repaired real replay;
2. manual chat usefulness audit;
3. LLM answer-quality A/B;
4. prompt-shape ablation;
5. context-budget and evidence-shape sweep;
6. anchor-neighborhood expansion;
7. clarification strategy;
8. F/S/T consumption semantics with long Stability stress;
9. multimodal consumption audit;
10. correction and recovery;
11. human UI consumption;
12. agent/tool consumption.

Minimum acceptance before consumption moves in:

- useful/harmful ratio at least `5` on manual review;
- no-anchor harmful hints below `2%` on manual review;
- wrong-only hints below `1%` on replay;
- p95 added context and latency stay within chat budget;
- Focus and Sensitivity remain monotonic;
- Stability is validated on long-horizon decay replay;
- ambiguous context is phrased as possible continuity, never fact;
- durable fact formation never consumes tentative or ambiguous links.

#### Soft Anchor consumption contract replay test

We then turned the consumption plan into an executable replay test:

```bash
python3 tools/soft_anchor_consumption_contract_test.py \
  --input-dir build/soft_anchor_consumption_full_ablation_v2 \
  --output-dir build/soft_anchor_consumption_contract_test_packet

python3 tools/soft_anchor_consumption_contract_test.py \
  --input-dir build/soft_anchor_consumption_full_ablation_v2 \
  --output-dir build/soft_anchor_consumption_contract_test_all_packet \
  --all-formation-policies
```

The test reads `ingress_adaptive_anchor_candidates.csv` and evaluates formed
Soft Anchor links as candidate context. It does not rerun retrieval, does not
use retrieved candidates, and does not use labels for runtime selection. Labels
are used only after a policy selects links, to score whether the surfaced hint
would have helped or harmed the replay case.

Artifacts:

- `build/soft_anchor_consumption_contract_test_packet/soft_anchor_consumption_contract_results.json`
- `build/soft_anchor_consumption_contract_test_packet/soft_anchor_consumption_contract_summary.csv`
- `build/soft_anchor_consumption_contract_test_packet/soft_anchor_consumption_contract_cases.csv`
- `build/soft_anchor_consumption_contract_test_packet/soft_anchor_consumption_contract_failures.csv`
- `build/soft_anchor_consumption_contract_test_all_packet/soft_anchor_consumption_contract_results.json`
- `build/soft_anchor_consumption_contract_test_all_packet/soft_anchor_consumption_contract_summary.csv`
- `build/soft_anchor_consumption_contract_test_all_packet/soft_anchor_consumption_contract_cases.csv`
- `build/soft_anchor_consumption_contract_test_all_packet/soft_anchor_consumption_contract_failures.csv`

The test covers the directly replay-testable consumers from the plan: no
consumption, silent safety, possible top-1 hints, loose top-1 hints, anchor
neighborhood hints, ambiguous top-3 sets, clarification, structured LLM
self-selection, UI-only top-3, current-turn recall, durable-only fact
consumption, decayed reminders, and ask-or-show hybrid. Six planned consumers
were recorded as not testable from this artifact alone: retrieval annotation,
context packing, retrieval neighborhood expansion, group anchors, cross-modal
evidence, and correction-aware recovery. Those require retrieval-result traces,
context-pack traces, group composition metadata, multimodal rows, or correction
events.

The first scoring pass was too conservative: it treated every surfaced
no-anchor hint as harmful, which is only true if the consumer asserts the hint
as a resolved referent. That is not the intended consumption contract. The
intended surface is SoftAnchor context on retrieved memories: full memory text
plus uncertainty and possibly multiple candidate continuities. The human, LLM,
or downstream agent decides which candidate is usable.

We therefore split the metric into two views:

- **asserted-context view:** no-anchor surfacing is harmful because the system is
  acting as if a referent exists;
- **SoftAnchor context view:** no-anchor surfacing is measured as context/noise
  load, while harm is wrong-only or stale-only evidence on reference cases.

After this correction, directly testable SoftAnchor context consumers do pass the
replay criterion.

| run | best SoftAnchor context variant | target surfaced | wrong-only contexts | optional contexts on no-anchor controls | p95 prompt chars |
|---|---|---:|---:|---:|---:|
| curated formation policies | C2 possible top-1 over `adaptive_soft_anchor_v1_no_generic` | 270 / 270 | 0 | 151 / 151 | 208 |
| all formation policies | C2 possible top-1 over `adaptive_soft_anchor_v1_no_generic` | 270 / 270 | 0 | 151 / 151 | 208 |

The strict asserted-context view still has `0` passes, which is expected: Soft
Anchor should not be consumed as a fact or hard bind. But as SoftAnchor context,
the same formed links are useful: the best variant surfaces all targets with no
wrong-only/stale-only reference contexts in the replay. The cost is that it also
surfaces optional context for every no-anchor control. That is not a correctness
failure if the context is phrased and rendered as optional memory support, but it
is a UX and context-budget question.

Interpretation: Soft Anchor consumption should be implemented as uncertain
retrieved-memory context, not as an automatic resolved anchor. When a memory is
retrieved, attaching the top candidate sentences and strengths is valid; the
consumer can choose, ignore, or ask. The next tests should therefore focus on
SoftAnchor context shape, context budget, and manual/LLM usefulness rather than
trying to make no-anchor context emission zero.

#### Soft Anchor engine wiring

Soft Anchor is now wired into the engine as an always-on ingress formation
operation. The path runs after `MemoryStorage`, consumes the stored memory id
and representative embedding, compares only against prior `SoftAnchorState`,
and writes `soft_anchors` and `soft_anchor_links`. Retrieval does not read those
rows for ranking, so production retrieval ranking remains unchanged. Retrieved
and working memories now hydrate up to three `SoftAnchor` entries from those
links and expose them on `Context::Memory::soft_anchors` and C API JSON
`soft_anchors`. The product decision is explicit: formation is in and the
memory-return surface carries optional SoftAnchor context; hard anchor binding
and retrieval reranking remain out. The demo chat mirrors those entries in the
memory event UI and includes them as nested `<soft_anchor>` elements under each
retrieved `<memory>` in the prompt snapshot, with instructions that they are
optional continuity likelihoods rather than resolved facts.

Verification:

```bash
cmake --build build -j 8 --target cortext
cmake --build build -j 8 --target cortext_tests
cmake --build build -j 8 --target cortext_chat
./build/tests/cortext_tests "[cortext][hydration][soft_anchor]"
./build/tests/cortext_tests "[operations][soft_anchor]"
./build/tests/cortext_tests "[operations]"
./build/tests/cortext_tests "ES-AIST GGUF executes native text image and audio kernels" -s
ctest --test-dir build -R cortext_tests --output-on-failure
```

Result:

- focused Soft Anchor tests passed: default formation, first ES-style signal
  creation, and continued-link update;
- broader operations slice passed: `892` assertions across `218` cases;
- ES-AIST native multimodal GGUF execution passed for text, image, and audio,
  with corrected 768d semantic/entity slices and unit-norm 1536d outputs;
- full `ctest --test-dir build -R cortext_tests --output-on-failure` passed in
  `133.41 s`;
- the chat demo target rebuilt against the same engine path.

#### Direct webcam and microphone chat ingress

The chat demo now includes a `Webcam` tab for direct camera and microphone
ingress. This is deliberately not the voice-chat path: it does not run
speech-to-text, diarization, speaker labeling, or reply generation. On macOS it
uses AVFoundation to capture camera frames as RGB images and microphone samples
as mono float PCM, then sends them directly through the normal Cortext
multimodal APIs:

- camera frames call `Cortext::ProcessImage`;
- microphone chunks call `Cortext::ProcessAudio`;
- video defaults to `1` FPS, can use `CORTEXT_CHAT_WEBCAM_FPS` as the startup
  default, and can be adjusted/persisted from the chat Settings tab;
- audio uses `16 kHz` mono chunks with a `2 s` chunk duration;
- capture callbacks enqueue at most eight pending media items so capture cannot
  grow unbounded if model ingest falls behind;
- database-explorer refresh is throttled to once per second from the media
  ingest worker so frame ingest does not turn into UI refresh churn.

Production retrieval behavior is unchanged. The chat example simply gives the
engine live image/audio signals from the webcam/microphone instead of converting
them to text first.

Verification:

```bash
cmake --build build -j 8 --target cortext_chat
```

Result: `cortext_chat` rebuilt successfully. The build emitted only existing
OpenTelemetry deprecation warnings from vendored headers.

#### ES-AIST direct media throughput

We extended `cortext_text_encoder_bench` so it can measure text, image, and
audio through the same `Encoder` interface used by chat. The benchmark was run
with native GGUF/GGML kernels on an Apple M3 Max through the Metal backend.

Commands:

```bash
CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH=1 \
CORTEXT_ESS_AIST_MODEL_PATH=models/ESS-AIST-81M-preview-GGUF-v9/ESS-AIST-81M_q8_0.gguf \
./build/examples/benchmark/cortext_text_encoder_bench \
  --encoder ess-aist --models-dir models --modality image \
  --image-width 640 --image-height 480 --iterations 30 --warmup 5

CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH=1 \
./build/examples/benchmark/cortext_text_encoder_bench \
  --encoder ess-aist --models-dir models --modality image \
  --image-width 640 --image-height 480 --iterations 20 --warmup 3
```

Results:

| model | modality | input | mean ms | embeddings/sec |
|---|---|---:|---:|---:|
| ES-AIST q8 default | text | short varied text | 2.20 | 454.60 |
| ES-AIST q8 default, old generic GGML image graph | image | 640x480 RGB | 303.58 | 3.29 |
| ES-AIST q8 default, native HWC image lane | image | 640x480 RGB | 83.14 | 12.03 |
| ES-AIST q8 native HWC image lane, CPU backend | image | 640x480 RGB | 82.24 | 12.16 |
| ES-AIST q8 native HWC image lane, CPU backend, 2 frame workers | image | 640x480 RGB | 42.16 | 23.72 |
| ES-AIST q8 native HWC image lane, CPU backend, 3 frame workers | image | 640x480 RGB | 27.27 | 36.67 |
| ES-AIST q8 native HWC image lane, CPU backend, 4 frame workers | image | 640x480 RGB | 22.16 | 45.13 |
| ES-AIST q8 old full-image GGML graph, CPU backend | image | 640x480 RGB | 313.49 | 3.19 |
| ES-AIST q8 default | audio, before preprocessing optimization | 2 s / 16 kHz mono | 488.31 | 2.05 chunks/s |
| ES-AIST q8 default | audio, after sparse mel + silent-tail skip | 2 s / 16 kHz mono | 157.66 | 6.34 chunks/s |
| ES-AIST q8 CPU backend | audio, after sparse mel + silent-tail skip | 2 s / 16 kHz mono | 336.67 | 2.97 chunks/s |
| ESS-AIST v9 q8 | text | short varied text | 2.23 | 448.62 |
| ESS-AIST v9 q8 | image | 640x480 RGB | 306.52 | 3.26 |
| ESS-AIST v9 q8 | image | 320x240 RGB | 297.13 | 3.37 |
| ESS-AIST v9 q8 | audio | 2 s / 16 kHz mono | 667.79 | 1.50 chunks/s |

Optimization result: the audio preprocessing path was wasting work by computing
the full 10-second mel surface for every 2-second chat chunk, including hundreds
of guaranteed-silent tail frames. The GGUF runtime now uses a cached sparse mel
filterbank and skips FFT/filterbank work for silent padded tail frames. The
default ES-AIST q8 audio path improved from `488.31 ms` to `157.66 ms` per
2-second chunk (`2.05` to `6.34` chunks/s), while still reporting
`full_audio_graph`.

Interpretation: for direct webcam ingest, the image encoder is now the limiting
path. On this machine, encoder-only 640x480 throughput is about `3.3 FPS`; the
full chat path also pays capture conversion, memory processing, storage, and UI
costs, so a practical default near `1 FPS` is justified. `2-3 FPS` is the
reasonable live range for continuous direct image memory ingest. Lowering the
synthetic image input from `640x480` to `320x240` barely changed throughput,
which points to the image convolution GGML graph/kernel as the bottleneck rather
than CPU input copy or resize. Higher settings should be treated as
burst/diagnostic values where the bounded queue may drop older frames if ingest
falls behind.

Follow-up comparison: the adjacent `emel.cpp` embedding generator benchmark on
this same host reports the TE-75M q8 image lane at about `88.0 ms` per image
(`~11.4 FPS`) with `prepare_ns ~= 4.4 ms` and `encode_ns ~= 83.6 ms`. That path
uses a native Mobilenet/EfficientAT execution lane over GGUF-loaded tensors:
weights are bound once, convolution kernels are prepacked, HWC scratch buffers
are persistent, and pointwise/depthwise/BN/ReLU kernels are fused in the native
lane. A Cortext experiment that only folded GGML graph batch-normalization into
conv weights did not materially improve image throughput (`~312 ms` per image)
and regressed audio, so it was not retained. We then added the first ES-AIST
native HWC image lane: RGB resize/normalize feeds a MobileNetV4-style native
image stack over GGUF-loaded tensors, while the existing ES-AIST projection and
L2 normalization preserve the 1536d signal contract. Dense convolution and
pointwise stages now use Accelerate-backed row-major GEMMs on Apple. In Release,
the CPU backend improved from `313.49 ms` to `87.68 ms` per image (`3.19` to
`11.41 FPS`), essentially matching the emel single-thread target. Removing the
redundant HWC-to-row copy from every 1x1 convolution and moving convolution
outputs directly into HWC tensor storage raised the CPU backend to `82.43 ms`
per image (`12.13 FPS`). Moving the image projection onto the host tensor path
kept the lane CPU-first even when Metal is available for text/audio and measured
`82.24 ms` (`12.16 FPS`), exceeding the emel reference class while staying
CPU-first. With default backend selection the same native lane measured `83.14
ms` (`12.03 FPS`). CPU audio remains realtime-compatible at `336.67 ms` per
`2 s` chunk (`2.97` chunks/s, about `5.9x` realtime). The remaining gap is code
quality rather than model contract: the current native lane still allocates
temporary im2col buffers per block, while emel uses persistent scratch and
packed kernels. The next optimization should reuse scratch buffers and
pack/fuse depthwise/BN/ReLU instead of returning to generic GGML convolution
graphs.

Frame-level parallelism gets the benchmark past the 30 FPS webcam target without
changing the vector contract. The release benchmark now supports
`--parallelism`, which creates one encoder instance per worker and measures
wall-clock throughput across independent frames. On the same CPU-backed native
HWC lane, `--parallelism 3` reached `36.67 FPS` and `--parallelism 4` reached
`45.13 FPS`. This is throughput, not lower single-frame latency: serial latency
remains about `80-85 ms`. The chat demo can now request up to `30 FPS`, but the
engine write path still needs single-writer staging if we want full Cortext
ingestion, storage, and UI accounting to sustain that rate without dropping old
frames.

#### Downloaded video through Cortext audio and visual paths

We then downloaded a real public sample video and ran decoded visual/audio
signals through the Cortext C API rather than only timing synthetic encoder
inputs.

Source:

```text
https://download.samplelib.com/mp4/sample-10s.mp4
```

Decoded input:

| stream | decoded shape |
|---|---:|
| video | 303 RGB frames, 640x480, 30 FPS, 10.10 s |
| audio | 163,469 float32 mono samples, 16 kHz, 10.2168 s |

Artifacts:

```text
build/video_media_perf/samplelib_10s.mp4
build/video_media_perf/source_ffprobe.json
build/video_media_perf/video_640x480_30fps.rgb
build/video_media_perf/audio_16k_mono_f32le.raw
build/video_media_perf/cortext_video_media_perf.json
build/video_media_perf/cortext_video_media_perf_10fps.json
```

Full 30 FPS visual replay used one Cortext instance and processed all video
frames serially through `ProcessImage`, then audio chunks through
`ProcessAudio`:

| path | calls | media seconds | mean total ms | p95 total ms | effective throughput | realtime multiple |
|---|---:|---:|---:|---:|---:|---:|
| visual `ProcessImage` at 30 FPS input | 303 | 10.10 | 83.74 | 88.76 | 11.88 FPS | 0.398x |
| audio `ProcessAudio`, 2 s chunks | 6 | 10.2168 | 344.47 | 368.55 | 2.90 chunks/s | 4.94x |
| combined serial pass | 309 | 10.2168 | - | - | - | 0.370x |

Interpretation: audio is comfortably realtime. Serial full-engine visual ingest
does not sustain 30 FPS because each frame still pays about `79 ms` encode time
plus about `5 ms` processing/storage/hydration overhead. This matches the
single-frame encoder ceiling and means 30 FPS requires either frame-parallel
encoding with a single-writer ingest stage or deliberate frame dropping.

We also replayed the same decoded video at a 10 FPS visual sample rate plus the
same audio chunks through one Cortext instance:

| path | calls | media seconds | mean total ms | p95 total ms | realtime multiple |
|---|---:|---:|---:|---:|---:|
| visual `ProcessImage` at 10 FPS sampling | 101 | 10.10 | 79.91 | 82.56 | 1.25x |
| audio `ProcessAudio`, 2 s chunks | 6 | 10.2168 | 343.51 | 367.85 | 4.96x |
| combined serial pass | 107 | 10.2168 | - | - | 0.998x |

That puts the current full Cortext serial path at roughly the edge of realtime
for `10 FPS` video plus audio. The encoder-only parallel benchmark can exceed
30 FPS, but the end-to-end engine needs a bounded multi-worker pre-encode stage
and single-writer memory commit path before continuous 30 FPS visual ingest is
honest.

#### SignalFilter parallel modality benchmark

We added a benchmark-only `cortext_signal_filter_bench` target to test the
general ingress idea as `SignalFilter`, with one modality-specific evidence
adapter per modality and one shared adaptive policy. The benchmark supports:

```bash
./build-release-static/examples/benchmark/cortext_signal_filter_bench \
  --modality all \
  --frames build/video_media_perf/video_640x480_30fps.rgb \
  --audio build/video_media_perf/audio_16k_mono_f32le.raw \
  --text-lines build/video_media_perf/sample_text_lines.txt \
  --width 640 --height 480 --channels 3 --fps 30 \
  --sample-rate 16000 --chunk-ms 1000 --text-step-seconds 1 \
  --output-dir build/signal_filter_bench/samplelib_all_parallel \
  --process-accepted --process-policy adaptive
```

In `--modality all`, the image, audio, and text filters run concurrently. With
`--process-accepted`, each modality then uses its own benchmark Cortext instance
and database so the benchmark measures parallel modality pressure without
claiming production single-writer semantics.

Filter results on the same downloaded video frames/audio plus a 120-line text
stream were:

| modality | items | adaptive accepted | accepted/sec | max gap |
|---|---:|---:|---:|---:|
| image | 303 | 79 | 7.82 | 0.367 s |
| audio | 11 | 7 | 0.685 | 2.000 s |
| text | 120 | 108 | 0.900 | 2.000 s |

Parallel accepted-signal Cortext processing measured:

| modality | processed | mean total ms | p95 total ms | realtime multiple |
|---|---:|---:|---:|---:|
| image | 79 | 95.37 | 102.81 | 1.33x |
| audio | 7 | 354.62 | 394.26 | 4.12x |
| text | 108 | 13.11 | 29.24 | 84.59x |

Interpretation: the image `SignalFilter` is already the useful one for video
rate control, cutting `303` candidate frames to `79` accepted frames and keeping
accepted visual processing above realtime. With 1-second audio chunks, adaptive
filtering accepts `7/11` chunks and keeps accepted audio processing at `4.12x`
realtime, so 1 second is a practical realtime hearing default. The current text
filter removes some near-redundant lines but is intentionally conservative;
text is cheap enough that its filter should focus on low-information/generic
suppression rather than aggressive throughput reduction.

#### Audio SignalFilter salience ablation

We then extended `cortext_signal_filter_bench` with an audio-specific salience
audit on the same real downloaded video audio. The benchmark still uses one
shared `SignalFilter` policy, but the audio adapter now exports auditable
salience events derived from waveform features: short-window RMS deltas,
max-local energy change, positive velocity/onset, and current energy. These are
not semantic labels. They are a benchmark target for the user claim that
continuous hearing should process focus-grabbing sound changes while adapting
to ambient volume.

Artifacts:

- `build/audio_signal_filter_real_video/audio_signal_filter_ablation_summary.json`
- `build/audio_signal_filter_real_video/audio_signal_filter_ablation_summary.csv`
- per-run row exports under `build/audio_signal_filter_real_video/chunk_*`

Chunk-size sweep on `build/video_media_perf/audio_16k_mono_f32le.raw`:

| chunk | policy | events | event recall | event precision | ambient accept rate | accepted/sec |
|---:|---|---:|---:|---:|---:|---:|
| 250 ms | all | 4 | 1.00 | 0.488 | 1.000 | 4.00 |
| 250 ms | fixed | 4 | 0.50 | 0.429 | 0.190 | 0.68 |
| 250 ms | adaptive | 4 | 1.00 | 0.538 | 0.286 | 1.27 |
| 500 ms | all | 2 | 1.00 | 0.429 | 1.000 | 2.00 |
| 500 ms | fixed | 2 | 1.00 | 0.333 | 0.333 | 0.57 |
| 500 ms | adaptive | 2 | 1.00 | 0.600 | 0.333 | 0.95 |
| 1000 ms | all | 2 | 1.00 | 0.727 | 1.000 | 1.00 |
| 1000 ms | fixed | 2 | 1.00 | 0.667 | 0.667 | 0.55 |
| 1000 ms | adaptive | 2 | 1.00 | 0.714 | 0.667 | 0.64 |

The 250 ms adaptive path is the best realtime-hearing candidate in this slice:
it retains all four detected acoustic events while cutting candidate chunks
from `41` to `13`. Accepted-signal Cortext processing for that policy averaged
`124.51 ms` total with p95 `143.17 ms` over `13` chunks, or `6.30x` realtime for
the accepted audio itself. The coarser 1-second path is cheaper and still
realtime, but it is less faithful to human-like auditory focus because each
decision covers a full second.

Focus/Sensitivity sweep at 250 ms showed the knobs act in the expected
direction. Higher Sensitivity retained more chunks and preserved event recall.
Higher Focus with low Sensitivity became too strict and dropped recall to
`0.75`; with medium/high Sensitivity it retained full event recall at lower
ambient acceptance than the all-pass baseline:

| Focus | Sensitivity | accepted | event recall | event precision | ambient accept rate |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.2 | 12 | 1.00 | 0.583 | 0.238 |
| 0.2 | 0.8 | 20 | 1.00 | 0.600 | 0.381 |
| 0.5 | 0.2 | 10 | 0.75 | 0.500 | 0.238 |
| 0.5 | 0.5 | 13 | 1.00 | 0.538 | 0.286 |
| 0.5 | 0.8 | 17 | 1.00 | 0.647 | 0.286 |
| 0.8 | 0.2 | 9 | 0.75 | 0.444 | 0.238 |
| 0.8 | 0.8 | 15 | 1.00 | 0.600 | 0.286 |

Interpretation: audio filtering is working as an adaptive salience gate, not as
a semantic event detector. For always-on hearing, the viable default is likely
250 ms windows with adaptive gating and a Sensitivity floor near the midpoint;
this gives sub-second focus while avoiding full-rate encoding. Fixed thresholds
are less robust because they miss half the detected events at 250 ms.

Implementation follow-up: the validated adaptive audio policy and visual delta
policy are now wired into Cortext as a core `SignalFilter` component rather than
chat-side special cases. `Cortext::ProcessAudio` and `Cortext::ProcessImage`
evaluate the filter before encoding/storage and return a normal `Context` with
`signal_filter_*` diagnostics when a chunk/frame is skipped. The component is
deliberately general: it owns modality-specific adapters, audio and image are
enabled by default, and text currently registers as a pass-through slot until a
text-side filter is promoted. The image adapter uses a bounded `32 x 24` luma
grid with fixed per-cell samples, so the filter cost is bounded and does not
scale with every input pixel.

#### Real-media multimodal episode ablation

We then added `cortext_real_multimodal_episode_bench` and
`tools/prepare_real_multimodal_episode_assets.py` to run the same question with
actual media. The asset script downloads public media from Wikimedia Commons,
converts images to `384x384` RGB and audio to `16 kHz` mono float32 PCM, and
writes SHA-256/provenance metadata. The benchmark calls the real Cortext API:
`ProcessImage`, `ProcessAudio`, and `ProcessText`. It does not construct
embeddings directly and does not set forced boundaries.

Artifacts:

- `build/real_multimodal_episode_assets/real_multimodal_episode_assets_manifest.json`
- `build/real_multimodal_episode_concat_audit/real_multimodal_episode_summary.json`
- `build/real_multimodal_episode_concat_audit/real_multimodal_episode_cases.json`
- `build/real_multimodal_episode_concat_audit/real_multimodal_episode_cases.csv`
- `build/real_multimodal_episode_concat_audit/real_multimodal_episode_groups.csv`
- `build/real_multimodal_episode_bench_no_signal_filter/real_multimodal_episode_summary.json`

Media:

- dog image: Wikimedia `File:Golden_Retriever.jpg`, public domain;
- dog-name audio: Wikimedia `File:En-us-bailey.ogg`, CC BY-SA/GFDL;
- car-crash image: Wikimedia `File:Car_crash_1.jpg`, public domain;
- crash audio: Wikimedia `File:En-us-crash.ogg`, CC BY-SA/GFDL.

Results:

| scenario | dog image+name fused | dog image+audio fused | all dog modalities fused | event split success | mixed-event memory |
|---|---:|---:|---:|---:|---:|
| dog image + text + Bailey audio | 1 | 0 | 0 | n/a | 0 |
| dog image/text/audio then car image/text/audio | 1 | 0 | 0 | 0 | 1 |
| dog image/audio then car image/audio | 1 | 1 | 0 | 0 | 1 |

The latest manual/programmatic audit writes one row per formed memory in
`real_multimodal_episode_groups.csv`. Programmatically, only **1 / 3** cases
passed all checks. The storage-byte contract did pass: `mixed_memory_blob_violations=0`
and `audio_only_concat_violations=0`. Mixed-modal memories are not serialized as
one invalid byte stream; image/audio/text payloads remain inspectable through
their signal rows, and audio-only concatenation remains coherent.

The episode grouping contract did not pass. Two real sequences still mixed
offline event labels inside one memory:

- `real_dog_name_then_car_crash`: memory `3` grouped `dog_name_audio` with
  `car_crash_image` and `car_crash_text`.
- `real_dog_name_audio_then_car_crash_audio`: memory `1` grouped `dog_image`,
  `dog_name_audio`, and `car_crash_image`.

Interpretation: modalities can be grouped during episodes, and audio-only bytes
can be concatenated safely, but the current real encoder + accumulator/write
timing does **not** yet reliably separate natural multimodal episodes. The
failure is not that the database cannot represent mixed modalities; it is that
the engine still lets a same-source audio/visual accumulator cross a real event
shift. The next implementation target is natural multimodal write timing:
same-episode audio arriving after a text write must either attach back to the
current episode or close cleanly before a subsequent visual event.

Signal-filter ablation: the benchmark path explicitly uses no signal filter
(`signal_filter_used=false`, `signal_filter_policy=none`) and processes every
real input through `ProcessImage`, `ProcessAudio`, or `ProcessText`. The
no-filter rerun reproduced the same full-sequence failure: `6/6` events were
processed, dog image+audio fusion remained `0`, event split success remained
`0`, and a mixed dog/car memory remained. So the failure is not caused by
frame/audio/text filtering.

#### Audio/video accumulator duration audit

We added `cortext_audio_video_accumulator_bench` to isolate why the chat DB
view can make audio memories look like one-second clips. The benchmark records
memory-level span, `n_signals`, per-signal payload duration, and memory-level
audio duration when the memory contains only audio PCM.

Command:

```bash
./build/examples/benchmark/cortext_audio_video_accumulator_bench \
  --output-dir build/audio_video_accumulator_bench \
  --audit-db examples/chat/chat_memory.db
```

Artifacts:

- `build/audio_video_accumulator_bench/audio_video_accumulator_summary.json`
- `build/audio_video_accumulator_bench/audio_video_memory_rows.csv`

Controlled results:

| scenario | memories | expected signal chunk | observed memory shape |
|---|---:|---:|---|
| `audio_10x_1s_final_flush` | 4 | 1.0 s | three 3-signal / 3.0 s audio memories plus one 1.0 s tail |
| `audio_5x_1s_force_each` | 5 | 1.0 s | five 1-signal / 1.0 s audio memories |
| `audio_chunk_sweep_250ms` | 2 | 0.25 s | one 3-signal / 0.75 s memory plus one 0.25 s tail |
| `audio_chunk_sweep_500ms` | 2 | 0.5 s | one 3-signal / 1.5 s memory plus one 0.5 s tail |
| `audio_chunk_sweep_1000ms` | 2 | 1.0 s | one 3-signal / 3.0 s memory plus one 1.0 s tail |
| `audio_chunk_sweep_2000ms` | 2 | 2.0 s | one 3-signal / 6.0 s memory plus one 2.0 s tail |

Interpretation: audio signals are one chunk each because webcam capture is
configured for 1-second audio chunks. The accumulator does not intrinsically
limit a memory to one second: when not forced every chunk, memory storage
persists multiple audio signals and the audio-only memory blob is the
concatenated PCM for those signals. The default boundary/write path still tends
to flush after about three audio chunks in this controlled run, so audio
memories are naturally short unless the accumulator/write policy is changed.

The default chat DB audit matched the same pattern: audio payloads attached to
memories were one-second signal chunks, while memories often contained multiple
signals and mixed image/audio rows. Therefore the immediate UI issue is that
the DB view exposes per-signal audio playback, which makes each playable row
one second. The deeper engine issue is separate: multimodal accumulation needs
a media-aware consumer that can assemble/play the memory's audio signal set and
a future write-timing experiment for whether audio/video episodes should span
longer than the current natural boundary cadence.

Follow-up implementation: mixed media memories now remain grouped by their
`signals` rows instead of receiving a single invalid concatenated memory blob.
Memory-level blobs are only written for coherent byte streams: text, audio-only
PCM, or a single payload. The chat DB view now shows memory-level media counts
and plays combined memory audio by concatenating all audio signals in serial
order, while image payloads remain lazy-loaded by button.

#### Ephemeral processing retention

We added an explicit processing retention contract for no-storage live queries.
`Retention::Durable` is the default for all public `ProcessText`,
`ProcessTextAt`, `ProcessAudio`, and `ProcessImage` calls. Callers must pass
`Retention::Ephemeral` explicitly when they want the signal to update live
processing state and retrieve context without creating a durable memory.

The write gate treats ephemeral signals as a hard no-store decision before
flush, spike-bypass, or force-write logic. `MemoryStorage` also has a defensive
retention check, and the accumulator avoids adding ephemeral payload records to
future durable writes. In the chat demo, Playground text/audio/video/image
captures use `Ephemeral`; the normal webcam combined stream remains durable.

This gives the UI the behavior needed for a no-storage playground: WM/STM/LTM
state can be exercised and cross-modal retrieval can be inspected, while the
database is not polluted by every probe typed or captured during manual testing.
