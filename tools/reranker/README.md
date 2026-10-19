# Late-interaction reranker (training and export)

`models/AIST-87M-GGUF/AIST-87M-LI.gguf` is a 0.2M-parameter learned reranker over the frozen
AIST-87M encoder. Queries use it to reorder the head of the candidate pool (top 64) before the
packet is cut. The engine side is `src/models/aist_late_interaction.{hpp,cpp}`; the query step is
`RerankLateInteraction` in `src/operations/graph_retrieval.cpp`.

## Model

Every item becomes a set of unit vectors in one 64-d late-interaction space. An item is a query or
a stored signal, in any modality. Its vectors come from two sources:

- **Position features.** These are AIST text token hidden states (384) and image patch features
  (12x12x960). Each tower has its own linear map. A stored signal keeps 16 spherical k-means codes
  of its positions as int8 (`codes.py`; table `memory_signal_li_codes`, about 1 KB per signal). A
  query keeps every position, with a learned gate weight on each.
- **Global vector.** This is the AIST shared-space retrieval vector, which the engine already
  stores as the 256-d view. One map serves every modality.

A member's score is the gate-weighted mean, over the query's vectors, of each vector's best cosine
to the member's vectors (MaxSim). A memory scores as a soft maximum over its first 8 members, plus
a rank prior. A memory none of whose members has codes keeps its slot. Such memories include
towers without a trained map (audio today) and signals stored before codes existed.

## Pipeline (local MPS; outputs under /Volumes/Build/cortext/reranker)

1. Build the pools.
   - `build_data.py` reads Cortext pool exports from the headroom lane: Eval1, DolphinBench, and
     BEAM 1M (pool order, memory members, gold units).
   - `build_haystack.py` builds engine-free pools for LongMemEval-S (MIT) and MemLens (CC-BY-4.0,
     images). It excludes the evaluation questions.
2. `featurize.py` extracts AIST features with `aist_torch.py`, a PyTorch mirror of the GGUF towers.
   The mirror's text vectors reach cosine 1.0000 against the engine. Its image values match the
   engine's pinned values.
3. `teacher.py` scores every (question, member) pair with the cross-encoder
   cross-encoder/ms-marco-MiniLM-L6-v2 (Apache-2.0). An image member is scored through its BLIP
   caption.
4. `codes.py` computes the fixed k-means codes (K=16).
5. `li.py` trains the model. The loss is listwise KL to the teacher (T=2) plus multi-positive
   listwise cross-entropy on the gold memories. Eval1 LoCoMo conversations (CC-BY-NC) are never
   trained on.
6. `export_gguf.py` writes `models/AIST-87M-GGUF/AIST-87M-LI.gguf` (local only); then
   `scripts/shard_model.py` shards it into `chunks/AIST-87M-LI.gguf.part-*` and updates
   `models/manifest.json`, the same way the AIST model ships. Commit the parts and the manifest.
7. `parity_fixture.py` writes `tests/fixtures/late_interaction/parity.{json,bin}` for the C++
   parity test.

`packet_recall.py` compares gold-turn recall between two arms of in-engine packets, with a paired
bootstrap over conversations.

## Splits

| Data | Split |
|---|---|
| Eval1 | 30% of conversations held out (sha1 hash) |
| DolphinBench | persona held out (riley) |
| LongMemEval, MemLens | 20% of questions held out |
| BEAM 1M | 30% of conversations held out |
| MemLens image evaluation set (108 questions) | excluded entirely |
