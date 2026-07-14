# Paper Artifacts

This directory tracks the small aggregate JSON artifacts cited by the README
and manuscript. The full `eval_runs/` and `build/graph_profile/` trees are
local run outputs and are intentionally ignored because they include databases,
prompt inputs, logs, and large replay summaries.

Included files:

| Artifact | Source run | Notes |
|---|---|---|
| `release_eval_20260628_gemma4_vllm/current_sparse_1y_system_ggml_20260628T0713Z/judge_vllm_gemma4_12b_awq_131k_rep3.json` | One-year sparse replay, Gemma4-12B-AWQ judge | Aggregate judge JSON |
| `msc_frontier_late_200dlg_gpt55_20260630T053427Z/judge_openai_gpt55_four_system_clean.json` | Hosted MSC frontier-judge probe | Aggregate judge JSON |
| `msc_frontier_late_200dlg_gpt55_20260630T053427Z/judge_openai_gpt55_four_system_clean.rows.jsonl` | Hosted MSC frontier-judge probe | Per-judgment rows |
| `msc_rag_ablation_128k_gpt55_20260630T_actual/judge_openai_gpt55_rag_ablation_128k.json` | 128k RAG ablation | Aggregate judge JSON |
| `msc_rag_ablation_128k_gpt55_20260630T_actual/judge_openai_gpt55_rag_ablation_128k.json.rows.jsonl` | 128k RAG ablation | Per-judgment rows |
| `msc_gemma4_temporal_baseline_20260701T200553Z/no_compaction_judge/judge_vllm_gemma4_12b_awq_ctx262k_nocomp_rep1.json` | Temporal-score local A/B baseline | Aggregate judge JSON |
| `msc_gemma4_temporal_fix_20260701T204155Z/no_compaction_judge/judge_vllm_gemma4_12b_awq_ctx262k_nocomp_rep1.json` | Temporal-score local A/B fix | Aggregate judge JSON |
| `replay_v8_context_blowout/judge_gemma4_12b_local_context128_prioronly.json` | Long-horizon context-blowout stress run | Aggregate judge JSON |
| `graph_profile/full_msc_verify_final/summary_slim.json` | Full MSC graph-profile verification | Slim tracked summary with packet text and large local-only fields removed |
| `neuromodulator_mechanism_sweep_20260706T232135Z/mechanism_sweep_summary.json` | July neuromodulator single-repetition sweep | Aggregate arm summary |
| `neuromodulator_mechanism_confirm_20260707T002126Z/mechanism_sweep_summary.json` | July neuromodulator three-repetition confirmation | Aggregate arm summary |
| `long_horizon_mechanism_sweep_20260707T022225Z/mechanism_sweep_summary.json` | Long-horizon deferred-mechanism sweep | Aggregate arm summary |

The aggregate JSON files preserve their original run metadata, including
historical `eval_runs/` paths for local databases, input directories, and row
checkpoints. Those heavy local inputs are not tracked. Claims tied only to an
untracked local path are labeled as such in the manuscript; the files listed
above are the checkout-auditable aggregate evidence.
