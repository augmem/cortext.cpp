# Standing-directive head (data, training, offline evaluation)

This folder holds a small learned head over frozen AIST-87M features. It decides whether a stored signal states
a **lasting directive or preference**, and it was built to replace the English marker rule `IsStandingDirective`
in `src/operations/standing_memory_internal.cpp`. This phase covers data, training and offline evaluation only.
No engine file is touched. The model card is `MODEL_CARD.md`.

## Conclusion: the rules stay

`IsStandingDirective` stays as the standing-memory detector. This work is closed, and no engine change follows
from it.

**Why.** Multilingual coverage was the only reason to replace the English marker rules, and multilingual
support is not a current concern. On real English streams the rules beat every learned head:

| Evidence (real replay streams, about 1.04M turns) | Rules | Head (attn h8) |
|---|---|---|
| Turns flagged | 2,193 (0.21%) | 48,723 (4.7%) |
| Judged lasting, 25 sampled disagreements each (gpt-6.1-sol) | **15 / 25** | 1 / 25 |
| Judged lasting, head's top-confidence flags (score ≥ 6, 25 sampled) | n/a | 1 / 25 |
| BEAM instruction_following gold recall (127 turns) | **0.98** | 0.62 |
| BEAM preference_following gold recall (125 turns) | **0.37** | 0.06 |

The head wins only on the synthetic held-out test:

| Synthetic held-out test, English (1,138 rows) | P | R | F1 |
|---|---|---|---|
| Rules | 0.79 | 0.26 | 0.39 |
| Head (attn h8) | 0.86 | 0.77 | 0.81 |

That gain does not transfer. The synthetic negatives miss the register of real chat turns, which is where the
head's false positives come from. For ja/zh/ko/hi/ar, no head on the current encoder can work at all, because
the engine tokenizer maps every byte ≥ 0x80 to `[UNK]`.

The full tables are in sections 1–5 below. The engine findings (tokenizer `[UNK]`, locale dependence) stand on
their own, and the tokenizer is being fixed separately. The FTS5 recommendation is to keep porter.

### How to resume if multilingual support ever matters

1. **Wait for the tokenizer fix and re-embed.** Every head variant then needs new features: the attention heads
   (token states change), the pooled heads (vectors change), the stream features (`streams/feat_*`), the
   modality probe and the parity fixtures. `rules.py` and `fts5_study.py` do not depend on the encoder.
2. **Add real-chat data.** oasst1 is already downloaded but unused. It sits at
   `/Volumes/Build/cortext/directive-head/data/oasst1`, revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`,
   Apache-2.0. Then run `sh run_oasst.sh`. That script labels user turns with the gpt-6.1-sol judge, which
   costs CPA calls; splits by conversation; merges with the synthetic set; then retrains, re-scores the
   streams, evaluates against the rules, judges, and exports.
3. **Bar for replacing the rules.** On a judged sample, head-only precision must be ≥ rule-only precision. BEAM
   instruction_following gold recall must be ≥ 0.95. The stream flag rate must be at most a few times the
   rules'. The integration plan below applies only once that bar is met.

## Pipeline

All outputs go under `/Volumes/Build/cortext/directive-head`. The torch env is
`/Volumes/Build/cortext/reranker/venv`.

| Step | Script | Output |
|---|---|---|
| 1. Generate | `gen.py --out gen --workers 48` (prompts and categories in `spec.py`) | `gen/calls.jsonl`: 3,499 cells with 52,320 messages (one of 3,500 planned cells timed out and was dropped) |
| 2. Build splits | `build_data.py --gen gen --out data` | `data/rows.jsonl`: 52,261 rows (train 34,505 / val 6,358 / test 11,398) |
| 3. Featurize | `featurize.py --in data/rows.jsonl --out data/feat_engine --tokens` (`--tok unicode` for the counterfactual) | `sem.npy`, `tok.npy`, `tok_off.npy` |
| 4. Train | `train.py --data data --out runs4/pooled --arms lr256,mlp256,mlp1280`; `train.py ... --arms attn --tokens --attn-heads 8` | `*.pt`, `summary.json` |
| 5. Streams | `streams.py --hill .claude/hillclimb --out streams`; `featurize.py --in streams/texts.jsonl --out streams/feat_attn3 --max-bytes 480 --head runs4/attn_h8_pw1.0/attn.pt` | 1,043,352 turns (1,041,044 unique texts), `head_scores.npy`, `beam_gold.jsonl` |
| 6. Modality + parity | `LC_ALL=C modality_probe.py --out modality` (engine binding: `embed_image`, `embed_audio`, `embed_text`) | `image.npy`, `audio.npy`, `speech.npy`, `parity_*.json` |
| 7. Evaluate | `evaluate.py --data data --runs runs4/pooled --arms "...,attn=runs4/attn_h8_pw1.0/attn.pt" --arm attn --streams streams --stream-feat feat_attn3 --modality modality --out runs4/streams/eval.json` | `eval.json`, `disagreements.jsonl`, `rule_flags.jsonl` |
| 8. Judge | `judge.py --dis runs4/streams/disagreements.jsonl --n 50`; `--bands runs4/streams`; `--audit data/rows.jsonl` | `judged*.jsonl` and summaries |
| 9. Export | `export_gguf.py --ckpt runs4/attn_h8_pw1.0/attn.pt --out AIST-87M-DH.gguf --fixture dh_parity.json --data data` | GGUF and a parity fixture |
| 10. Latency | `cc -O3 -ffast-math -mcpu=native bench_head.c -lm && ./a.out attn 56 8` | ns per call |
| 11. FTS5 | `fts5_study.py --data .claude/hillclimb/scorecard/data --out runs/fts5.json` | hit@k for both tokenizers |

`rules.py` is a byte-exact Python port of `IsStandingDirective` (origin/main 55793178). Run against the engine
function compiled from its exact source lines, it gave 0 mismatches on 200,000 random stream texts.

## Data

- **Source.** The data is synthetic, generated with `gabe-codex-{0,1}/gpt-6.1-sol` (reasoning low) through
  `.claude/hillclimb/cpa_judge/cpa_backend.py`. The key is read at call time and is never printed.
- **Shape.** There are 22 categories, 74 templates, 40 topic areas and 10 languages (en es fr de pt ja zh ko hi
  ar), with about 5,200 rows per language and 37% positives.
- **Splits.** Splits are by template and topic, never by random row:
  - 21 templates are test-only.
  - 8 topics are test-only and 4 are val-only.
  - Duplicates across splits are kept in the held-out split only.
  - Train rows that are near-duplicates of held-out rows are removed (char TF-IDF cosine ≥ 0.9).
- **Stream prefixes.** 60% of rows carry a synthetic timestamp or speaker prefix of the kind the streams use.
- **Excluded benchmarks.** No LongMemEval, LoCoMo, BEAM, MemLens or DolphinBench data was used for training,
  thresholds or selection.
- **Rounds.**
  1. Round 1 had 15 categories.
  2. Round 2 added realistic requests, assistant replies, sharing and long positives, after the first stream
     run fired on 13% of turns.
  3. Round 3 added opinions, long single-task prompts and preferences stated inline, after round 2 still
     fired on 6.6% of turns.
- **Label audit.** gpt-6.1-sol agreed with the generator label on 188 of 200 round-1 test rows (0.94). This
  is a consistency check, not independent ground truth.

## 1. Held-out test (synthetic, 11,398 rows)

Cells show P / R / F1 / PR-AUC at each arm's threshold. The heads are trained on round-3 data and use the final
calibration.

| Slice | n (pos) | engine rule | brief rule | mlp256 | mlp1280 | **attn h8** | attn h8, Unicode tokenizer (counterfactual) |
|---|---|---|---|---|---|---|---|
| all | 11398 (4680) | 0.79 / 0.03 / 0.05 / n/a | 0.59 / 0.02 / 0.03 / n/a | 0.82 / 0.02 / 0.04 / 0.59 | 0.87 / 0.09 / 0.16 / 0.62 | 0.85 / 0.27 / 0.41 / 0.71 | 0.84 / 0.32 / 0.46 / 0.75 |
| en | 1138 (468) | 0.79 / 0.26 / 0.39 / n/a | 0.59 / 0.17 / 0.27 / n/a | 0.76 / 0.07 / 0.13 / 0.67 | 0.85 / 0.29 / 0.43 / 0.74 | **0.86 / 0.77 / 0.81 / 0.91** | 0.87 / 0.65 / 0.75 / 0.89 |
| es | 1140 (468) | 0 / 0 / 0 | 0 / 0 / 0 | 0.67 / 0.03 / 0.07 / 0.66 | 0.85 / 0.15 / 0.25 / 0.69 | 0.83 / 0.51 / 0.63 / 0.82 | 0.85 / 0.49 / 0.62 / 0.83 |
| fr | 1140 (468) | 0 / 0 / 0 | 0 / 0 / 0 | 0.93 / 0.06 / 0.10 / 0.70 | 0.90 / 0.22 / 0.35 / 0.74 | 0.87 / 0.47 / 0.61 / 0.83 | 0.86 / 0.49 / 0.63 / 0.82 |
| de | 1140 (468) | 0 / 0 / 0 | 0 / 0 / 0 | 0.90 / 0.02 / 0.04 / 0.61 | 0.85 / 0.12 / 0.21 / 0.67 | 0.82 / 0.43 / 0.57 / 0.77 | 0.82 / 0.32 / 0.47 / 0.76 |
| pt | 1140 (468) | 0 / 0 / 0 | 0 / 0 / 0 | 0.95 / 0.04 / 0.08 / 0.69 | 0.91 / 0.13 / 0.23 / 0.72 | 0.86 / 0.51 / 0.64 / 0.83 | 0.88 / 0.47 / 0.61 / 0.83 |
| ja | 1140 (468) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0.46 | 0 / 0 / 0 / 0.44 | 0 / 0 / 0 / 0.47 | 0.79 / 0.30 / 0.43 / 0.68 |
| zh | 1140 (468) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0.41 | 0 / 0 / 0 / 0.42 | 0 / 0 / 0 / 0.44 | 0.71 / 0.13 / 0.22 / 0.57 |
| ko | 1140 (468) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0.47 | 0 / 0 / 0 / 0.47 | 1.00 / 0.00 / 0.00 / 0.51 | 0.78 / 0.10 / 0.17 / 0.66 |
| hi | 1140 (468) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0.44 | 0 / 0 / 0 / 0.44 | 0 / 0 / 0 / 0.44 | 0.84 / 0.11 / 0.20 / 0.70 |
| ar | 1140 (468) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0.46 | 0 / 0 / 0 / 0.45 | 0 / 0 / 0 / 0.49 | 0.74 / 0.10 / 0.18 / 0.63 |
| held-out template | 5039 (2160) | 0.47 / 0.01 / 0.02 | 0.20 / 0.00 / 0.01 | 0.78 / 0.02 / 0.04 / 0.58 | 0.87 / 0.09 / 0.16 / 0.61 | 0.80 / 0.27 / 0.40 / 0.67 | 0.79 / 0.31 / 0.45 / 0.72 |
| held-out topic | 6359 (2520) | 0.89 / 0.04 / 0.08 | 0.80 / 0.03 / 0.05 | 0.85 / 0.02 / 0.05 / 0.59 | 0.87 / 0.09 / 0.16 / 0.63 | 0.90 / 0.27 / 0.41 / 0.74 | 0.87 / 0.32 / 0.47 / 0.78 |

- **PR-AUC near 0.45 is chance.** The base rate is 0.41, so a PR-AUC around 0.45 for ja/zh/ko/hi/ar under the
  engine tokenizer means no signal.
- **The rules ignore three of the brief's examples.** Neither rule fires on "Under no circumstances …", and
  both pass the "Always sunny …" and "Never mind …" idioms. Across all test languages, the rule's flag rate is 0.06
  on `pos_marker`, 0.01 on `pos_negation` and 0.01 on `neg_idiom`. The flag rate for every category is in
  `runs4/eval_test.json`.
- **`brief_rule` is not on main.** It is the variant described in the task: utterance-initial markers plus
  ≥ 3 content tokens. `HasDirectiveClausePrefix`, `ExtractDistinctStems` and `IsStandingMemoryText` do not
  exist on origin/main. What main has is `IsStandingDirective`, a marker list checked at the start of every
  clause, plus a 480-byte gate.

## 2. Real benchmark streams (evaluation only)

All replay turns were used: LME-S, LME-M100, LoCoMo, BEAM 100K/1M/10M and the three DolphinBench personas. The
head scores text of ≤ 480 bytes, the same gate the rule uses, at threshold 1.163.

| Stream | Turns | Rule flags | Head flags | Both | Rule only | Head only |
|---|---|---|---|---|---|---|
| lme_s | 246,738 | 167 | 14,599 | 41 | 126 | 14,558 |
| lme_m100 | 488,135 | 329 | 28,663 | 78 | 251 | 28,585 |
| locomo | 5,882 | 41 | 216 | 3 | 38 | 213 |
| beam100k | 5,732 | 120 | 158 | 67 | 53 | 91 |
| beam1m | 74,630 | 525 | 1,589 | 233 | 292 | 1,356 |
| beam10m | 208,696 | 1,010 | 3,110 | 531 | 479 | 2,579 |
| dolphin alex / morgan / riley | 13,539 | 1 | 388 | 0 | 1 | 388 |
| **all** | **1,043,352** | **2,193** | **48,723** | **953** | 1,240 | 47,770 |

Cohen's κ between rule and head is 0.034.

**Judge.** These were graded by gpt-6.1-sol (reasoning low) on the question "is this a lasting directive or
preference", using the samples in `judged.jsonl` and `bands.jsonl`.

| Sample (round-robin over streams) | Judged lasting |
|---|---|
| 25 rule-only disagreements | **15 / 25** |
| 25 head-only disagreements | **1 / 25** |
| 25 rule flags the head scores < 0 (a would-be veto) | 19 / 25 |
| 25 rule flags the head scores ≥ 0 | 21 / 25 |
| 25 head-only flags with score ≥ 6 (top 471) | 1 / 25 |

- **The high-confidence flags are ordinary requests.** The head's most confident real-stream flags look like
  "Can you recommend …", "I'll make sure to …" and "hmm, how do I …". The synthetic negatives never captured
  that register.
- **The head does not work as a veto on the rule.** Rule flags it would drop are judged lasting at nearly the
  same rate as the ones it keeps.

**BEAM gold.** The BEAM gold turns are the user turns the questions cite.

| Category | Gold turns | Rule recall | Head recall | Either |
|---|---|---|---|---|
| instruction_following | 127 | **0.98** | 0.62 | 0.98 |
| preference_following | 125 | 0.37 | 0.06 | 0.38 |

BEAM instructions are almost all literally "Always … when I ask about …", which is exactly the rule's pattern.
BEAM preference gold turns are mostly long technical turns that only imply a preference.

## 3. Latency and size

| Head | Parameters | GGUF bytes | CPU per call |
|---|---|---|---|
| attn h8 (text token states) | 12,297 | 49,920 | 21 µs at 56 tokens (median gated stream turn), 52 µs at 124 tokens (p99) |
| attn h4 | 6,149 | about 25 KB | 11 µs at 56 tokens |
| mlp256 (stored 256-d vector) | 16,513 | 66,656 | 1.0 µs |
| lr256 | 257 | about 1 KB | 0.14 µs |

- **How it was timed.** Each call is one thread of scalar C (`bench_head.c`, `-O3 -ffast-math -mcpu=native`) on
  this M-series host at load average 7–11. Every head fits the 0.1 ms budget. The AIST text encode the engine
  already does per signal costs milliseconds, so the head adds well under 1%.
- **Parity.** The exported GGUF matches torch within 1.4e-6 (`dh_parity.json`). The PyTorch mirror matches the
  engine `embed_text` at cosine ≥ 0.99938 on 400 texts in all ten languages under `LC_ALL=C`.

## 4. Calibration

Each threshold is picked on English validation rows: the smallest threshold whose precision is ≥ the rule's
(0.874) **and** whose false-positive rate is ≤ the rule's (0.0365). Matching FPR as well as precision keeps
the head at least as precise as the rule at any base rate, and real streams have a far lower base rate than the
balanced validation split.

- **Validation, English:** the attn h8 head reaches P 0.92, R 0.65, against the rule's P 0.87, R 0.39.
- **Test, English:** P 0.86, R 0.77, against the rule's P 0.79, R 0.26.

The calibration holds on synthetic data and breaks on real data:

- **The rule fires far less on real turns.** It fires on 0.21% of real turns but on 3.65% of synthetic
  negatives.
- **The head fires more.** It fires on 9.6% of the real turns it scores (≤ 480 bytes), against 3.6% of
  synthetic negatives.

The synthetic hard negatives are much harder than real turns for the rule and much easier for the head, so
matching precision and FPR there does not carry over.

## 5. Modality

- **The proposed attention head is text-only.** It reads the text tower's token states, so image and audio
  signals have no input and are never scored. It cannot fire on them. This is the same convention the
  late-interaction reranker uses for towers without a map.
- **The pooled global-vector head (`mlp256`) is modality-agnostic but does fire on non-text inputs.** On engine
  embeddings it flags 0.7% of 600 MemLens images, 2.3% of 600 salt audio clips and 5.2% of 600 salt speech
  clips at its threshold. Its p99 score on speech (3.8) is above the threshold (2.1). If a global head ships,
  train it with non-text negatives, or gate it to the text tower.

## Findings (engine, not fixed here)

1. **The AIST tokenizer is blind to non-Latin scripts.**
   - `BasicTokens` in `src/models/aist_gguf_encoder.cpp` emits every byte ≥ 0x80 as its own one-byte token,
     and none of them is in the vocabulary, so each becomes `[UNK]`.
   - "常に日本語で答えてください。" ("always answer in Japanese") and "今日は晴れです。おはよう！！" ("it's sunny
     today, good morning!!") embed at cosine **1.0000** in the engine.
   - Accented Latin words split into fragments. All retrieval for ja/zh/ko/hi/ar text runs on byte length and
     ASCII alone.
   - A counterfactual BERT-style Unicode tokenizer over the same vocabulary lifts ja F1 from 0.00 to 0.43, and
     the five non-Latin languages to F1 0.17–0.43, with no other change. Fixing this changes every text
     embedding, so it needs a new embedding-model pin and a re-embed.
2. **Text embeddings depend on the process locale.**
   - The tokenizer calls `std::isalnum`, `std::isspace` and `std::tolower` on raw bytes. Under a UTF-8
     `LC_CTYPE` (Python sets it, so the Python binding gets it), bytes ≥ 0x80 count as letters.
   - The same accented text therefore embeds differently in the Python binding and in a C++ runner: cosine
     0.878 for "Búscame alojamiento cerca del aeropuerto de Bogotá.", with a minimum of 0.895 and a mean of
     0.994 over 400 dataset texts. Under `LC_ALL=C` the minimum is 0.99938.
   - A store written through one binding and queried through another can drift. The fix is to classify bytes
     explicitly instead of through `<cctype>`.
3. **`TermsOf` / `RankByTerms` (BM25 inside the standing set) are ASCII- and English-centric.** They use the
   English stopword list from `memory_text_index_internal.cpp`. A multilingual detector would still rank
   members mostly through the dense channel.

## Integration plan (once a head passes the real-stream bar)

**Bar.** On a judged sample, head-only precision must be ≥ rule-only precision. BEAM instruction_following gold
recall must be ≥ 0.95. Stream flag rate must be at most a few times the rule's.

1. **Where it plugs in.**
   - Write path: `memory_storage.cpp` around line 1469, where `IsStandingDirective(text)` gates the
     `standing_signals` insert. Backfill: `RunPendingBackfill` in `memory_text_index_internal.cpp` around line 550.
   - The attn head needs the text tower's last hidden states. The encoder already produces them, and LI codes
     are computed from the same positions. The encode step should therefore emit a `standing_logit` next to
     the LI codes, and the storage gate becomes `standing_logit >= dh.threshold`.
   - A global-vector head would run directly on the 256-d vector `memory_storage` already holds.
2. **How it ships.**
   - `models/AIST-87M-GGUF/AIST-87M-DH.gguf`, sharded by `scripts/shard_model.py` into `chunks/`, listed in
     `models/manifest.json`, and resolved beside the AIST model like `ResolveLateInteractionModelPath`.
   - It is required with the encoder. The metadata keys are documented in `export_gguf.py`.
3. **Per-signal cost.** About 21 µs median and 52 µs p99 (attn h8), or about 1 µs (mlp256), against
   milliseconds for the encode.
4. **How `IsStandingDirective` is replaced.**
   - Cut over in one change: delete the marker list and `IsStandingDirective`, and keep `kCapacity`,
     `kPacketMembers`, `TermsOf` and `RankByTerms`, which rank members and do not detect them.
   - Keep `kMaxTextBytes` as `dh.max_text_bytes`.
   - Non-text towers are not scored for attn. For a global head they are scored only if it was trained with
     non-text negatives.
5. **Migration and backfill of existing stores.**
   - Add a schema migration that queues `standing_signals` in `cortext_pending_backfills`, the same mechanism
     migration 53 uses, and records a `standing_detector` id (the DH GGUF hash) in store metadata. A detector
     change re-queues the rebuild.
   - The rebuild clears `standing_signals` and re-inserts the newest `kCapacity` matches. That is cheap for a
     global head, because it reads `memory_signal_vectors` with no re-encode.
   - The attn head has to re-encode stored text of ≤ 480 bytes from the object store, at a few ms per
     candidate. That is bounded, but on a 250k-turn store it means minutes. It could run lazily: rebuild from
     the newest signals backward and stop once `kCapacity` members are found, so the cost is proportional to
     how far back the 256th directive sits, not to the store size.
   - Stores written before the locale fix (finding 2) should be re-embedded before any detector that depends
     on tokens is backfilled.

## FTS5 tokenizer: keep `porter unicode61 remove_diacritics 2`

This compares BM25 hit@k of gold turns using the engine's query words (`fts5_study.py`, one FTS5 row per turn,
QueryWords ≤ 16, df cut at max(64, 5%), OR query, bm25).

| | Questions | porter hit@1 / 5 / 10 / 20 | unicode61 (no porter) hit@1 / 5 / 10 / 20 | Δ hit@10 (95% CI) |
|---|---|---|---|---|
| LoCoMo (evidence turns) | 1,533 | 0.353 / 0.601 / 0.678 / 0.740 | 0.323 / 0.562 / 0.628 / 0.699 | **−0.050** (−0.067, −0.034) |
| LongMemEval-S (answer-session turns) | 470 | 0.885 / 0.955 / 0.977 / 0.989 | 0.870 / 0.947 / 0.974 / 0.983 | −0.002 (−0.013, +0.009) |

- **Dropping porter costs recall.** It loses 3–5 points of recall at every k on LoCoMo (significant) and
  0.2–1.5 points on LME-S (not significant).
- **No measured gain to offset it.** The only possible upside of removing porter is on non-English Latin-script
  text, where porter applies English suffix rules. These streams contain no such text to measure, and
  `remove_diacritics 2` is already on in both arms.
- **Recommendation:** keep the current tokenizer. Revisit only with a multilingual retrieval set.

## Next steps

None while multilingual support is out of scope. See "How to resume" above.
