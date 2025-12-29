# Repository Guidelines

## Project Structure & Module Organization
- `src/` and `include/`: C++17 core library and public headers (main Cortext engine).
- `tests/`: Catch2-based unit/integration tests (target: `cortext_tests`).
- `examples/`: runnable demos and analysis tooling (e.g., `examples/topical_chat_analysis`).
- `scripts/`: experiment harnesses and utilities (e.g., `scripts/run_memory_harness.py`).
- `docs/paper/`: manuscript source in `sections/` and generated output in `_manuscript/`.
- `models/`, `third_party/`: runtime assets (ImageBind, LiteRT, sqlite extensions).

## Build, Test, and Development Commands
- Configure/build:
  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
  cmake --build build -j
  ```
- Run tests (project suite only):
  ```bash
  ctest --test-dir build -R cortext_tests --output-on-failure
  ```
- Run topical chat analysis:
  ```bash
  ./build/examples/topical_chat_analysis/cortext_topical_chat_analysis --help
  ```
- Run harness (long horizon example):
  ```bash
  python scripts/run_memory_harness.py --max-conversations 2 --max-turns 360 --max-total 720 --no-multi
  ```

## Coding Style & Naming Conventions
- Follow existing C++ style: 2‑space indentation, braces on new lines, and whitespace around parentheses (see `src/operations/*.cpp`).
- Use existing naming patterns within the file you modify (PascalCase for helpers, lower_snake for locals).
- Avoid introducing unused or deprecated code — this repo is pre‑release and prefers breaking changes without regressions.

## Testing Guidelines
- Tests use Catch2 via the `cortext_tests` target.
- Add/extend tests when changing algorithms or thresholds, especially for interrupts and retrieval.
- Validate end‑to‑end behavior with `examples/topical_chat_analysis` before large sweeps.

## Docs & Experiment Reporting
- When algorithms change or new results are produced, update `docs/paper/sections/*` and regenerate:
  ```bash
  QUARTO_DISABLE_GIT=1 QUARTO_DISABLE_GITHUB=1 quarto render docs/paper
  ```
- The manuscript source of truth is `docs/paper/_manuscript/index.md` (generated from sections).

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and descriptive (e.g., “Add interrupt precision/recall metrics”).
- PRs should include: summary, test commands run, and any updated experiment outputs/log paths.

## Agent-Specific Notes
- All behavior should derive from the three knobs where possible.
- Consolidation/labeling must use gemma-3n-e2b via LiteRT; embeddings use ImageBind/ONNX.
