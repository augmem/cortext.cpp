import assert from "node:assert/strict";
import { createRequire } from "node:module";
import fs from "node:fs";

const declarations = fs.readFileSync (
  new URL ("./index.d.ts", import.meta.url), "utf8");

assert.match (
  declarations,
  /export type ConsolidationState = "none" \| "recommended" \| "required";/);
assert.match (declarations, /consolidation_state: ConsolidationState;/);
assert.doesNotMatch (declarations, /consolidation_recommended/);
assert.doesNotMatch (declarations, /consolidation_required/);

const require = createRequire (import.meta.url);
const { Cortext } = require ("./index.js");
const engine = new Cortext (
  { focus: 0.5, sensitivity: 0.5, stability: 0.5 }, ":memory:");
const runtime = JSON.parse (engine.processTextJson (
  "JavaScript addon consolidation-state contract probe.",
  "contract/runtime",
  { retention: "ephemeral" }));

assert.equal (runtime.consolidation_state, "none");
assert.equal (
  Object.prototype.hasOwnProperty.call (runtime, "consolidation_recommended"),
  false);
assert.equal (
  Object.prototype.hasOwnProperty.call (runtime, "consolidation_required"),
  false);
