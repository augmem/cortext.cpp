# Memory Eval Harness

This repo has adapters for the current long-term memory eval set:

- LongMemEval-S
- LongMemEval-V2 Small
- LoCoMo
- LoCoMo-Plus
- BEAM 100K, with optional 500K/1M preparation
- MemoryAgentBench

The harness normalizes each benchmark into ignored local files under
`data/memory_evals/` and writes run artifacts under `logs/memory_evals/`.
Raw downloaded data is kept under `data/raw/memory_evals/`.

## Run

Build the public-API runner:

```bash
cmake -S . -B build -DCORTEXT_BUILD_TOOLS=ON
cmake --build build --target cortext_memory_eval_runner -j
```

Run the bounded smoke profile for every adapter:

```bash
scripts/run_memory_evals.py --profile smoke --benchmarks all
```

By default this also writes deterministic packet-answer predictions and scores
them. Disable answer scoring with `--answer-mode none`.

Prepare and run the uncapped profile:

```bash
scripts/run_memory_evals.py --profile full --benchmarks all
```

Run a larger bounded slice without switching to the uncapped profile:

```bash
scripts/run_memory_evals.py \
  --profile smoke \
  --benchmarks all \
  --prepare-limit-episodes 2 \
  --prepare-limit-queries-per-episode 4 \
  --prepare-limit-turns-per-episode 64 \
  --prepare-limit-lme-v2-trajectories 2 \
  --prepare-limit-lme-v2-states 4
```

Include the larger BEAM tiers:

```bash
scripts/run_memory_evals.py --profile full --benchmarks all --include-large
```

Use an OpenAI-compatible Chat Completions model for answer generation:

```bash
OPENAI_API_KEY=... scripts/run_memory_evals.py \
  --profile smoke \
  --benchmarks all \
  --answer-mode openai \
  --answer-model gpt-5-mini
```

Use any local or remote CLI as a judge:

```bash
scripts/run_memory_evals.py \
  --profile smoke \
  --benchmarks all \
  --answer-mode harness \
  --judge-label codex \
  --judge-command 'python3 scripts/judges/codex_judge.py {prompt_file}'
```

The command judge contract is intentionally small: the harness sends the judge
prompt on stdin and expects the final answer on stdout. If a CLI prefers files,
the command may reference `{prompt_file}` for the plain-text prompt or
`{input_json}` for a JSON object containing the prompt, question, snippets, and
query identifiers. Those placeholders are expanded to temporary files for each
query. `--judge-label` keeps outputs separate, for example
`answers_harness_codex`, `answers_harness_grok`, or `answers_harness_agy`.
`--answer-mode command` is an equivalent alias when the external process is not
a model judge.

You can also configure it from the environment:

```bash
CORTEXT_EVAL_JUDGE_LABEL=grok \
CORTEXT_EVAL_JUDGE_COMMAND='grok < {prompt_file}' \
scripts/run_memory_evals.py --profile smoke --benchmarks all --answer-mode harness
```

For CLIs that do not read stdin directly, use a tiny wrapper that reads stdin,
calls the tool, and prints only the final answer. The scorer treats any stderr
output as diagnostic text and fails the row if the command exits non-zero.

Run all configured judges against one retrieval pass:

```bash
scripts/run_memory_eval_judges.py \
  --profile smoke \
  --benchmarks all \
  --max-episodes 1 \
  --max-queries 1 \
  --judges all
```

If `--run` is supplied, the judge coordinator reuses an existing
`scripts/run_memory_evals.py --answer-mode none` output directory instead of
running retrieval again:

```bash
scripts/run_memory_eval_judges.py \
  --run logs/memory_evals/command_mode_smoke_20260708T032500Z \
  --judges codex,grok,agy
```

The built-in adapters are:

- `scripts/judges/codex_judge.py`: uses `codex exec`, low reasoning effort by
  default, and `--output-last-message` so stdout contains only the answer.
- `scripts/judges/grok_judge.py`: uses `grok --prompt-file` with memory and web
  search disabled.
- `scripts/judges/agy_judge.py`: uses `CORTEXT_AGY_COMMAND` if set, or falls
  back to `agy run --stdin` / `antigravity run --stdin` when one is installed.

Useful adapter overrides:

- `CORTEXT_JUDGE_TIMEOUT_S`
- `CORTEXT_CODEX_BIN`, `CORTEXT_CODEX_MODEL`, `CORTEXT_CODEX_EFFORT`,
  `CORTEXT_CODEX_ARGS`
- `CORTEXT_GROK_BIN`, `CORTEXT_GROK_MODEL`, `CORTEXT_GROK_ARGS`
- `CORTEXT_AGY_COMMAND`, `CORTEXT_AGY_BIN`

## What It Measures

`cortext_memory_eval_runner` ingests benchmark histories with
`Retention::Durable`, then asks benchmark questions with
`Retention::Ephemeral`. This exercises the public "query without polluting"
path: retrieval is expected, but the query itself is not written as a durable
memory.

The runner reports retrieval-packet coverage:

- `answer_hit_rate`: any gold answer string appears in the top-k retrieved
  memory packet.
- `evidence_hit_rate`: any provided evidence phrase appears in the top-k
  retrieved memory packet.
- `abstention_empty_rate`: abstention questions return no retrieved memory.

`scripts/score_memory_eval_answers.py` then produces answer predictions from
those packets. `--answer-mode packet` is local and deterministic: it treats the
retrieved packet as the answer, so it is best understood as an extractive
upper-bound/check that scoring is wired. `--answer-mode openai` sends the
retrieved packet and question to `/chat/completions` and scores the generated
answer with the same normalized exact/substring scorer. `--answer-mode command`
and `--answer-mode harness` run an external judge command per query with the
same prompt and scoring path, so Codex, Grok, Antigravity, or another harness
can be compared without changing the eval code.

Rows without a reference answer and without an abstention label are emitted as
predictions but marked `scored=false`; this applies to the LoCoMo-Plus cognitive
sample until a constraint-consistency judge is supplied.

The normalized schema is modality-aware. The runner processes:

- `modality=text` with `text`.
- `modality=image` with raw RGB/RGBA `data_base64`, `width`, `height`, and
  `channels`.
- `modality=audio` with `samples` or little-endian float32
  `audio_f32_base64`.

Image/audio records that only contain external paths, such as LME-V2 screenshot
references when screenshot archives have not been materialized, are preserved
and counted as skipped rather than silently coerced into text.
