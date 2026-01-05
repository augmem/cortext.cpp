# Repository Guidelines

## Project Structure & Module Organization
- `src/` + `include/`: C++ core engine and public headers.
- `tests/`: Catch2 unit/integration tests (`cortext_tests` target).
- `examples/`: runnable demos and analysis tooling (e.g., `examples/topical_chat_analysis`).
- `scripts/` + `tools/`: experiment harnesses and generators (e.g., `scripts/run_memory_harness.py`).
- `docs/paper/sections/`: manuscript source; `docs/paper/_manuscript/` is generated output.
- `models/` + `third_party/`: runtime assets (ImageBind/ONNX, LiteRT, sqlite extensions).

## Build, Test, and Development Commands
- Configure/build:
  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
  cmake --build build -j
  ```
- Run tests:
  ```bash
  ctest --test-dir build -R cortext_tests --output-on-failure
  ```
- Run topical chat analysis:
  ```bash
  ./build/examples/topical_chat_analysis/cortext_topical_chat_analysis --help
  ```
- Long-horizon harness example:
  ```bash
  python scripts/run_memory_harness.py --max-conversations 2 --max-turns 360 --max-total 720 --no-multi
  ```

## Coding Style & Naming Conventions
- Match local style in the file you touch (e.g., 2-space indentation, braces on new lines).
- Prefer existing naming patterns (PascalCase helpers, lower_snake locals) over introducing new conventions.
- Pre-release rule: breaking changes are fine, but do not leave unused/deprecated code behind.

## Testing Guidelines
- Add tests when changing algorithms or thresholds (interrupts, retrieval, consolidation).
- Use `examples/topical_chat_analysis` for end-to-end validation before large sweeps.
- Keep outputs deterministic where possible; prefer fixed seeds when adding new metrics.
- Run long sweeps with `nohup` (or equivalent) so they survive terminal/session disconnects.
- Use `scripts/notify-codex.sh` only for long-running scripts; for short runs, watch the output directly.
- Do not use `sleep` to wait/poll background commands; prefer `scripts/notify-codex.sh` for long runs or manual tailing for short runs.
- For long-running scripts, append `&& scripts/notify-codex.sh cortext {{what to do next when the command finishes}}` to send a completion ping. Use an action-oriented message (e.g., “... done — review boundary alignment and update docs”), not just “done”.

## Docs & Experiment Reporting
- Always update `docs/paper/sections/` when algorithms change or experiment results are produced.
- Regenerate the manuscript:
  ```bash
  QUARTO_DISABLE_GIT=1 QUARTO_DISABLE_GITHUB=1 quarto render docs/paper
  ```
- `docs/paper/_manuscript/index.md` is the generated source of truth for the compiled paper.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and descriptive (e.g., “Add interrupt precision/recall metrics”).
- PRs should include a brief summary, test commands run, and any updated experiment logs/paths.

## Repository-Specific Notes
- Behavior should derive from the three knobs (F/S/T) wherever possible.
- Consolidation/labeling uses gemma-3n-e2b via LiteRT; embeddings use ImageBind/ONNX.
- Do not modify the public API surface (public headers in `include/`, C API) without explicit approval.
