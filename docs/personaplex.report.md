# PersonaPlex Memory Eval Report

Date: 2026-06-23

## Summary

PersonaPlex evals show that Cortext is useful for preserving person-level memory beyond the live Moshi context window, but current retrieval is still brittle for adjacent-turn details.

The strongest result is profile memory: Cortext kept seeded facts about Maya at 5/5 through 30 minutes. These facts included her full name, address, cat, planning availability, coffee order, and nickname rule.

The main weakness is not broad forgetting. It is local retrieval composition: Cortext often retrieves the right topic turn but misses the neighboring turn that contains the exact answer. Examples:

- Retrieves “sunrise ceramics class starts at 7:15” but misses the adjacent “Lina Torres is teaching.”
- Retrieves “found an old film camera” but misses the follow-up “silver Olympus.”
- Retrieves “Marcus painting the hallway / warmer than gray” but misses the follow-up “fog pearl.”
- Retrieves “room over the bakery” but misses the follow-up bakery name “Morning Bell.”

Native consolidation every 5 minutes did not reliably fix this. In a same-question replay, profile facts stayed perfect, but adjacent-detail misses remained and one 20-minute run got worse.

## Eval Setup

The current PersonaPlex memory eval is a natural two-friend conversation between Maya and Jordan.

Each run:

- Generates a casual friend-chat transcript.
- Seeds profile facts naturally inside the conversation.
- Ingests text into Cortext only.
- Does not use `/api/chat`, voice generation, guard routing, System0/System1, or PersonaPlex consolidation paths.
- Uses Gemini to ask natural recall questions after the conversation.
- Uses Gemini to judge Cortext memory retrieval against the full transcript oracle.

The live PersonaPlex Moshi window is approximately 4 minutes:

- `context = 3000`
- frame rate = `12.5`
- `3000 / 12.5 = 240 seconds`

The comparison eval therefore clips answerer context to 4 minutes for the live-window condition while leaving the judge on the full transcript oracle.

## What Worked

### Profile Memory

Cortext retained seeded person facts across all tested durations.

| Duration | Profile fact recall |
|---:|---:|
| 5m | 5/5 |
| 10m | 5/5 |
| 15m | 5/5 |
| 20m | 5/5 |
| 25m | 5/5 |
| 30m | 5/5 |

This is the clearest positive signal. Cortext is doing the important “know the person” job even after those details fall outside the live model window.

### Grounding

Grounding stayed strong. The failure mode was usually omission, not fabrication.

Across the evaluated runs, Cortext generally did not invent unsupported details. When it missed, it retrieved nearby context or admitted uncertainty downstream rather than producing false memories.

### Token Value vs Full-Context Replay

Against a full-context baseline, Cortext provides the intended token benefit.

| Duration | Full-context baseline tokens | Cortext tokens | Token change |
|---:|---:|---:|---:|
| 5m | 8,399 | 9,248 | 10.1% more |
| 10m | 14,403 | 9,171 | 36.3% fewer |
| 15m | 20,180 | 8,874 | 56.0% fewer |
| 20m | 25,740 | 9,073 | 64.8% fewer |
| 25m | 31,726 | 8,888 | 72.0% fewer |
| 30m | 36,632 | 8,679 | 76.3% fewer |

The crossover happens after 5 minutes. At 30 minutes, Cortext uses 76.3% fewer input tokens than replaying the full transcript while preserving profile facts.

## Where Cortext Falls Short

### 1. Adjacent-Turn Detail Retrieval

The dominant miss pattern is retrieving the right topic but not enough local context around it.

Examples from the eval:

| Question target | Retrieved context | Missing adjacent detail |
|---|---|---|
| Ceramics teacher | sunrise ceramics class / 7:15 Saturdays | Lina Torres |
| Camera model | old film camera / 2021 roll | silver Olympus |
| Paint color | Marcus hallway / warmer than gray | fog pearl |
| Bakery name | Harbor House room over bakery | Morning Bell |
| Playlist details | playlist title | first song |

This suggests retrieval is not bundling local conversational neighborhoods. The exact answer often appears one turn after the retrieved topic anchor.

### 2. Multi-Part Questions Are Fragile

When a question asks for two linked details, Cortext often retrieves the first detail but misses the second.

Observed examples:

- Playlist name retrieved, first song missed.
- Camera existence retrieved, model/color missed.
- Class schedule retrieved, teacher missed.

The retrieval layer needs better support for “same topic, adjacent clarification” instead of treating each line as an isolated memory candidate.

### 3. Consolidation Every 5 Minutes Did Not Reliably Help

Native `Cortext.consolidate()` was exposed through PersonaPlex sidecar as `POST /consolidate` and run every 5 simulated minutes.

Fresh consolidated runs looked better in some samples, but a fair same-question replay did not show reliable improvement.

| Duration | No consolidation | Same questions + 5m consolidation |
|---:|---|---|
| 5m | pass, recall 5/5 | pass, recall 5/5 |
| 10m | pass, recall 5/5 | pass, recall 5/5 |
| 15m | pass, recall 4/5 | pass, recall 4/5 |
| 20m | pass, recall 4/5 | fail, recall 3/5 |
| 25m | pass, recall 4/5 | pass, recall 4/5 |
| 30m | pass, recall 5/5 | pass, recall 5/5 |

Profile facts stayed 5/5 with consolidation, but the adjacent-detail issue remained.

Conclusion: this is probably not solved by simply running consolidation more often. It needs retrieval-time or storage-time neighborhood handling.

### 4. Retrieved Context Contains Distractors

Several retrieval outputs included unrelated but high-scoring snippets. This did not usually break grounding, but it increases prompt cost and can crowd out the exact answer turn.

Example pattern:

- Correct target memory appears alongside unrelated recurring snippets.
- A broad topic match ranks above the exact answer detail.
- Proper nouns in adjacent turns can be ranked below generic topic turns.

This points to ranking and packing, not just recall availability.

## Recommended Fixes

### 1. Adjacent-Turn Expansion

When a retrieved memory comes from a conversational transcript source like:

`run_id/turn-0080/caller`

retrieval should optionally include nearby turns from the same run:

- previous caller/model turn
- next caller/model turn
- same speaker’s next clarification

A simple window of `turn - 2` through `turn + 2` would likely fix many observed misses.

This should be done after retrieval candidate selection but before context packing, with deduplication and budget limits.

### 2. Topic Bundle Packing

If two adjacent turns share a topic, store or retrieve them as a bundle:

- setup turn: “I signed up for sunrise ceramics class.”
- clarification turn: “Lina Torres is teaching.”

The model needs both to answer natural recall questions.

Candidate approach:

- detect sequential semantic continuity during ingest,
- store a lightweight neighborhood id or topic id,
- at retrieval time, pull siblings from the same neighborhood when one member scores highly.

### 3. Proper-Noun Boosting

Many misses are exact names or titles:

- Lina Torres
- Morning Bell
- Fog pearl
- Silver Olympus

Ranking should boost candidate turns containing capitalized named entities or rare proper-noun-like phrases when the query asks “who,” “which,” “what was the name,” “what was it called,” or contains correction language.

### 4. Multi-Part Query Expansion

For questions with conjunctions or correction structure, split retrieval into subqueries.

Examples:

- “Which camera was it, not the Canon?” should query both `camera closet 2021` and `not Canon model`.
- “What playlist and first song?” should query `playlist title` and `first song`.

Merge and rerank the combined candidate set before packing.

### 5. Context Packing Should Prefer Answer Specificity

The current retrieval can include general topic lines before exact answer lines. Packing should reserve slots for:

- exact lexical matches,
- proper nouns,
- numeric/time details,
- adjacent clarifications,
- correction targets such as “not X, Y.”

This may matter more than increasing `item_limit`, because simply adding more items can add distractors.

## What Not To Overclaim

The eval is synthetic. It is useful for regression and architectural diagnosis, but it is not proof of human-grade memory.

The current positive claim should be narrow:

Cortext preserves seeded person-level facts outside the live PersonaPlex context window and can reduce token cost substantially versus full transcript replay.

The current limitation should also be explicit:

Cortext does not yet reliably retrieve exact adjacent details from conversational neighborhoods.

## Relevant Artifacts

PersonaPlex repo artifacts:

- `/shared/personaplex/eval/ramp/latest.md`
- `/shared/personaplex/eval/friend_clipped_chat_compare/latest.md`
- `/shared/personaplex/eval/friend_full_baseline_vs_cortext/latest.md`
- `/shared/personaplex/eval/ramp_consolidate_5m/latest.md`
- `/shared/personaplex/eval/ramp_consolidate_5m_replay/latest.md`

Code changes used for the eval:

- `/shared/personaplex/memory_ramp_eval.py`
- `/shared/personaplex/normal_chat_compare_eval.py`
- `/shared/personaplex/cortext_sidecar.py`
