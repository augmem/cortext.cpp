# Cognitive Architecture Mechanism Ablation

Date: 2026-06-11

## Scope

This pass tested six small mechanisms inspired by cognitive architectures after
the ACT-R/evidence-confidence work. It does not import OpenCog, LIDA, Soar, ONA,
CLARION, Sigma, NARS, or any runtime. The implementation is limited to internal
helper functions in `src/operations/cognitive_mechanisms.hpp`, focused unit
tests, and a benchmark target:

```bash
examples/benchmark/cortext_cognitive_mechanism_ablation_bench
```

No public API, C API, or schema changes were made. The mechanisms are disabled
by default and named behind internal environment flags:

| mechanism | flag |
|---|---|
| OpenCog-style attention ledger | `CORTEXT_ENABLE_COG_ATTENTION_LEDGER` |
| LIDA-style packet competition | `CORTEXT_ENABLE_COG_PACKET_COMPETITION` |
| Soar-style cue rarity / negative cues | `CORTEXT_ENABLE_COG_CUE_RARITY` |
| ONA-style usefulness rank | `CORTEXT_ENABLE_COG_USEFULNESS` |
| CLARION-style explicit/implicit lanes | `CORTEXT_ENABLE_COG_EVIDENCE_LANES` |
| Sigma-style factor fusion | `CORTEXT_ENABLE_COG_FACTOR_FUSION` |

These flags are intentionally not wired into production ranking in this pass.
The benchmark uses them as on/off ablation gates, so the disabled side remains
the current embedding-only behavior.

## Validation

Build:

```bash
cmake --build build --target cortext_tests cortext_cognitive_mechanism_ablation_bench -j
```

Focused unit tests:

```bash
./build/tests/cortext_tests "[operations][cognitive-mechanisms]"
```

Result: all tests passed, `14` assertions in `6` test cases.

Real-encoder ablation:

```bash
CORTEXT_AIST_MODEL_PATH=/Users/gabrielwillen/VSCode/cortext/models/AIST-87M-GGUF/AIST-87M_q8_0.gguf \
  ./build/examples/benchmark/cortext_cognitive_mechanism_ablation_bench \
  --models-dir /Users/gabrielwillen/VSCode/cortext/models
```

Resolved encoder:

```text
encoder_backend=AIST-87M-GGUF
model=/Users/gabrielwillen/VSCode/cortext/models/AIST-87M-GGUF/AIST-87M_q8_0.gguf
```

| study | source | off winner | on winner | off target | off comparison | on target | on comparison | verdict |
|---|---|---|---|---:|---:|---:|---:|---|
| `opencog_attention_ledger` | OpenCog | comparison | target | 0.706926 | 0.932638 | 0.747612 | 0.304568 | keep gated |
| `lida_packet_competition` | LIDA | comparison | target | 0.706224 | 0.801292 | 0.636614 | 0.801292 | keep gated |
| `soar_cue_rarity_negative` | Soar | comparison | target | 0.756858 | 0.911577 | 1.000000 | 0.664162 | keep gated |
| `ona_usefulness_rank` | ONA | comparison | target | 0.834012 | 0.951573 | 1.000000 | 0.951573 | keep gated |
| `clarion_explicit_implicit_lanes` | CLARION | comparison | target | 0.732446 | 0.859891 | 0.766382 | 0.342163 | keep gated |
| `sigma_factor_fusion` | Sigma | comparison | target | 0.916532 | 0.928370 | 0.836923 | 0.572766 | keep gated |

Summary: `6/6 passed`.

Regression checks:

```bash
./build/tests/cortext_tests "[operations][graph]"
./build/tests/cortext_tests "[operations][actr]"
CORTEXT_AIST_MODEL_PATH=/Users/gabrielwillen/VSCode/cortext/models/AIST-87M-GGUF/AIST-87M_q8_0.gguf \
  ./build/examples/benchmark/cortext_actr_retrieval_ablation_bench \
  --models-dir /Users/gabrielwillen/VSCode/cortext/models
```

Results: graph retrieval tests passed, ACT-R tests passed, and the ACT-R
real-encoder ablation remained `6/6 passed`.

## Verdicts

| source | mechanism | decision | reason |
|---|---|---|---|
| OpenCog | Split transient activation, durable importance, persistence intent, and evidence confidence before recombining. | keep gated | It flipped a high lexical/semantic distractor to the durable evidence target under the real encoder. Useful as a bounded ledger shape, not a replacement ranker. |
| LIDA | Bounded packet proposal competition with refractory suppression. | keep gated | Refractory suppression moved selection away from a recently repeated packet without changing the production retrieval path. |
| Soar | Cue rarity plus negative-cue penalties. | keep gated | Rare positive support and explicit negative evidence beat a generic high-semantic distractor. Needs corpus-level cue statistics before production use. |
| ONA | Usefulness-ranked admission/forgetting pressure from retrieval, selection, feedback, and age. | keep gated | Historical usefulness repaired the ranking in the fixture. It should remain bounded because the current run saturates the target score. |
| CLARION | Explicit fact confidence and implicit semantic evidence as separate lanes before fusion. | keep gated | Source-backed explicit evidence overcame a generic semantic match, while unsupported explicit evidence remains downweighted in unit coverage. |
| Sigma | Small product-of-experts fusion over semantic/source/quality factors. | keep gated | Balanced support beat a spiky high-semantic/low-quality candidate. This is a candidate scoring primitive, not a graph rewrite. |

No mechanism failed this ablation. None is promoted directly into production
ranking by this branch because the benchmark is a focused two-candidate causal
test, not a corpus sweep. The next valid promotion step is to wire one mechanism
at a time into an existing Cortext ranking or admission surface behind its flag,
then rerun broader retrieval and long-horizon harnesses with unchanged public
APIs.
