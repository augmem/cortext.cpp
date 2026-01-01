Notes

- Push summary relevance further: add a small boost for ASSOCIATION nodes in retrieval ranking and re-run the 360-turn ablation.

Verification TODO
- Compare consolidation during idle vs end-only and measure association/summary retrieval impact.
- Validate interrupt gating under mixed content with affect on/off (longer horizon).
- Confirm source monitoring confidence is surfaced and gates injection as intended.
- Validate procedural store + sequential links influence retrieval (ablation/metrics).
  - Deferred: add a targeted sequential‑recall benchmark to surface procedural/sequence effects.
