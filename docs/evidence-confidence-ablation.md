# Evidence-Weighted Confidence Ablation

Date: 2026-06-11

## Scope

This branch adopts only the evidence-confidence/revision mechanism suggested by
`docs/cognitive-architecture-survey.md`. It does not import ACT-R, NARS,
OpenCog, or another cognitive runtime, and it does not introduce public API, C
API, or storage-schema changes.

The mechanism is internal and gated by `CORTEXT_ENABLE_EVIDENCE_CONFIDENCE`.
With the flag unset, duplicate fact confidence keeps the previous
`max(existing, incoming)` behavior, graph evidence packets carry zero evidence
confidence, and retrieval rank/score behavior is preserved.

## Mechanism

`src/operations/evidence_confidence.hpp` represents evidence as:

- agreement/frequency
- confidence and evidence weight
- source/evidence stamp sets
- contradiction mass and dampening
- projected confidence under time decay

Revision combines independent, non-overlapping evidence stamps. Overlapping
stamps are treated as duplicates or choices, so repeated evidence does not
inflate confidence. Conflicting evidence dampens confidence instead of blindly
raising support.

The first integration points are intentionally narrow:

- fact confidence revision in `ProcessExtractionResults`
- fact-backed packet/member confidence annotations in graph retrieval
- `evidence_blend` reconstruction source confidence when packet confidence is
  available

## Validation

Targeted unit coverage:

```bash
./build/tests/cortext_tests "[operations][evidence-confidence]"
./build/tests/cortext_tests "[operations][facts]"
./build/tests/cortext_tests "[operations][graph]"
./build/tests/cortext_tests "[operations][actr]"
```

The tests cover monotonic confidence growth from independent support, duplicate
no-inflation, contradiction dampening, time projection/decay, disabled fact
behavior, enabled fact confidence revision, and rank-preserving packet
annotation.

Real-encoder ablation:

```bash
CORTEXT_AIST_MODEL_PATH=/Users/gabrielwillen/VSCode/cortext/models/AIST-87M-GGUF/AIST-87M_q8_0.gguf \
  ./build/examples/benchmark/cortext_actr_retrieval_ablation_bench \
  --models-dir /Users/gabrielwillen/VSCode/cortext/models
```

Result: `summary=6/6 passed`.

The evidence-weighted-confidence study preserved target/comparison ranks and
scores while adding useful packet confidence:

| metric | off | on |
|---|---:|---:|
| target rank | 1 | 1 |
| comparison rank | 2 | 2 |
| target score | 0.993295 | 0.993295 |
| comparison score | 0.961908 | 0.961908 |
| packet confidence | 0.000000 | 0.834525 |
| reconstruction source confidence | 0.274780 | 0.834525 |

## Decision

Keep the mechanism behind the environment flag. The effect is not a broad
retrieval-ranker win and should not be promoted as one: it deliberately leaves
rank and score unchanged. It is useful as packet-quality metadata and as a
source-confidence repair for fact-backed evidence packets, where the real
encoder ablation shows a material source-confidence improvement without adding
ranker weight or new architecture.
