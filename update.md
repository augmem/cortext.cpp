  Section 6.1: Working Memory Gates

  - Added subsection 6.1.1 Slot Content Structure defining what a WM slot contains:
    - slot.content (concatenated signal texts)
    - slot.embedding (e_rep from Section 4.4.5)
    - slot.signals (ordered signal embeddings)
    - slot.metadata with emotional metrics (s_emotion_max, s_arousal_avg)
  - Added subsection 6.1.2 Maintenance Cost
  - Added subsection 6.1.3 Segment-Level Gating with on_segment_boundary evaluation
  - Added subsection 6.1.4 Chunking at Segment Level

  Section 6.7: Emotional Consolidation

  - Changed emotion_intensity_t → m.metadata.s_emotion_max
  - Changed arousal_t → m.metadata.s_arousal_avg
  - Added explanatory text about asynchronous consolidation using stored segment metadata

  Section 7: Consolidation and Graph Integration

  - 7.1: Added segment-awareness note, changed write_rate_t → segment_write_rate_t
  - 7.1.1: Changed is_processing_signal → is_accumulating_segment
  - 7.2: Added note about segment representatives (e_rep)
  - 7.6: Changed query vector x_t → μ_seg for both initial search AND re-ranking

  Section 8: Interrupt Gate and Streaming Integration

  - Added cross-reference to segment-level context in introduction
  - 8.2: Added explicit ctx_window ← recent_segment_centroids and ctx_centroid definitions
  - 8.3.1: Changed current_signal.start_timestamp → t_seg_start
  - 8.4: Added segment_flush_required trigger condition