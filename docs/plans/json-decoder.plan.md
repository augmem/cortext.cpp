# Plan: Session API + JSON-Constrained Label Extraction

## Goal
Migrate label extraction from LiteRT-LM Conversation API to Session API and enforce
JSON-schema-constrained output for labels/relations.

## Scope
- Label extraction path only (Gemma/phi4 extractors).
- Schema-constrained decoding for JSON payloads (`labels`, `relations`), with labels as string arrays.
- Tests run through `cortext_tests` binary.
- Treat as a breaking change; no backwards-compatibility shims.

## Out of Scope (for this plan)
- Summarization path.
- Graph construction logic beyond ensuring payload conformity.
- Changes outside the label/relation extraction schema.

---

## Phase 0: Inventory + Decisions
- [x] Map current extraction flow, inputs, and parsing paths (Conversation API).
- [x] List Session API capabilities and required data (model, tokenizer, runtime config).
- [x] Decide constraint approach:
  - [x] Evaluate adapting `old/json_decoder.cpp` into a constraint.
  - [x] Evaluate adapting `old/constrained_sampler.cpp` into a constraint.
  - [x] Evaluate `llguidance` C sample integration (used by Chromium + LiteRT-LM).
  - [x] Evaluate implementing a new constraint for the schema subset used.
- [x] Define JSON schema subset to enforce (types, required fields, min items, enums).
- [x] Document failure modes and fallback strategy (retry, error, empty).
- [x] Update this plan with Phase 0 findings (requirements, constraints, risks).

## Phase 0 Findings (2025-12-19)
### Current extraction flow
- Consolidation builds `ExtractionRequest` (summary + sources), then
  `ProcessExtractionResults` invokes `Extractor::ExtractFromText` or external callback.
- Gemma extractor uses Session API; Phi-4 uses OGA + llguidance JSON schema guidance.
  `ProcessExtractionResults` now consumes `ExtractionResult::labels` with label-only payloads.

### Session API capabilities + constraints
- Session API supports constraints via `DecodeConfig::SetConstraint(Constraint*)`;
  constraints are token-level bitmaps. No built-in JSON schema constraint provider
  in-tree (only `FakeConstraint` exists). `ConstrainedDecodingOptions` lists
  only `UNSPECIFIED_LIBRARY`.
- Gemma3 Conversation `CreateConstraint(...)` returns `nullptr`, so current
  “constrained decoding” in Conversation is effectively disabled for Gemma.
- Session applies prompt templates by default; for strict JSON output we likely
  need to disable templates or supply a minimal, deterministic template.

### Constraint approach evaluation
- `old/json_decoder.cpp` provides a streaming JSON schema parser and “allowed tokens”
  list, but it is string-based and not wired to LiteRT-LM token IDs.
- `old/constrained_sampler.cpp` masks logits using a GemmaTokenizer-specific mapping.
  It would need porting to LiteRT-LM’s `Tokenizer` and `Constraint` bitmap interface.
- `llguidance` C++ sample exists, but its README notes JSON schema support is TODO;
  it currently expects a grammar and requires custom tokenization + logit masking.
- Recommendation: implement a LiteRT-LM `Constraint` wrapper around the existing
  streaming JSON parser, using LiteRT-LM’s tokenizer to map allowed strings to token IDs.
  Do not maintain parallel constraint paths.

### JSON schema subset to enforce
- Root: object with `labels` (array of strings) and `relations` (array of objects).
- Labels: array of strings (surface labels only; no type/category fields).
- Relation fields: `subject` (string, required), `predicate` (string, required),
  `object` (string, required), `confidence` (number, optional).
- No `oneOf`, enums, nested unions, or polymorphic types needed for this path.

### Failure modes + fallback
- Model may emit non-JSON, partial JSON, or extra trailing text if unconstrained.
- Constraint end-state resets automatically; ensure a stop condition (stop token or
  explicit decode cutoff) to avoid multiple JSON objects in one generation.
- Planned handling: retry N times when output invalid; on final failure, return empty
  result or surface error depending on caller policy (to be decided in Phase 3).

### Breaking change posture
- This migration is explicitly breaking; we will remove Conversation API extraction
  and legacy payload handling without compatibility layers.

## Phase 1: Session API Extraction (Unconstrained)
- [x] Implement Session-based extraction path (replace Conversation API path).
- [x] Build a deterministic prompt formatter for labels/relations.
- [x] Add integration test that exercises Session extraction end-to-end (no constraint yet).
- [x] Verify label extraction succeeds in `cortext_tests` with model path lookup.

## Phase 2: Constraint Implementation
- [x] Implement `Constraint` adapter that enforces the JSON schema subset.
- [x] Integrate streaming validation (token-by-token) for schema compliance.
- [x] Support tool-call or raw JSON output (choose one canonical format).
- [x] Add focused unit tests for constraint acceptance/rejection cases.

## Phase 3: Wire Constraint into Session Extraction
- [x] Attach constraint via Session API (`DecodeConfig::SetConstraint`).
- [x] Remove Conversation API dependency for label extraction.
- [x] Ensure schema-required fields are always emitted (labels array minItems=1).
- [x] Add retry or fallback behavior when decoding fails.

## Phase 4: Integration Tests + Robustness
- [x] Add true integration tests that force label extraction with constraints enabled.
- [x] Add regression test for malformed/non-JSON output (should be corrected or retried).
- [x] Update existing extractor tests to assert strict schema conformance.

## Phase 5: Cleanup + Documentation
- [x] Remove unused Conversation API paths for extraction.
- [x] Update docs/audit.md with completed migration details and evidence.
- [x] Capture migration notes in `docs/plans/json-decoder.plan.md` completion log.

---

## Notes
- Prefer minimal schema surface area to maximize decoding stability.
- Ensure all tests run through `./cortext_tests` (not via CTest).
- Keep extraction payloads using `labels` only (no `entities`).

## Completion Log
- 2025-12-20: Phase 0–5 complete. Extraction uses Session API with JSON-schema constrained decoding (Gemma constraint adapter; Phi-4 llguidance). Default extraction schema requires `labels` with `minItems=1`; prompts instruct model to always emit at least one label. Gemma extraction retries constrained decode on invalid/empty label output. Constraint tests cover required labels and relation required fields; Gemma integration test asserts non-empty labels/relations. No Conversation API paths remain in extraction (summarizer still uses Conversation). Evidence: `cmake --build build --target cortext_tests`, `./build/tests/cortext_tests -s "[constraint]"`, `./build/tests/cortext_tests -s "[extractor][gemma][integration]"`.
