# Cross-Modal Embedding Experiments: Findings

## Models Tested

| Model | Params | Modalities | Architecture | Embedding Dim |
|-------|--------|-----------|--------------|---------------|
| Qwen3.5-0.8B | 853M | Text + Image + (Video) | Gated DeltaNet VL | 1024 |
| PE-AV-Small | 847M | Audio + Video + Text | Contrastive multi-tower | 1024 |

## Qwen3.5-0.8B: Hidden State Embeddings

**Method**: Extract hidden states from the multimodal transformer, mean-pool across sequence.

### Best Configurations (Flickr30k, n=200)

| Config | T→I R@1 | T→I R@10 | I→T R@1 | I→T R@10 |
|--------|---------|----------|---------|----------|
| 0.5*L2 + 0.5*L22 | **0.460** | **0.870** | 0.415 | 0.800 |
| all_layers_avg | 0.390 | 0.840 | **0.560** | **0.850** |
| L22 alone | 0.310 | 0.745 | 0.430 | 0.800 |
| L24 alone | 0.150 | 0.670 | 0.525 | 0.870 |

### Key Findings

1. **Mean pooling is the ONLY viable strategy** — last_token, first_token, max all produce near-random results
2. **Layer blend (0.5*L2 + 0.5*L22) is optimal for T→I** — early layers capture visual content, late layers capture semantic alignment
3. **All-layer average is optimal for I→T** — spreading across all layers captures the full representation
4. **The model genuinely aligns text and image in hidden space** without contrastive training
5. **Isotropy**: embeddings are anisotropic (avg cos ~0.9), effective rank ~50-85 out of 1024

### CLIP ViT-B/32 Reference (Flickr30k 1K)

- T→I R@1 ≈ 0.31, I→T R@1 ≈ 0.52
- Qwen3.5 blended **beats CLIP on T→I** (0.46 vs 0.31) with comparable I→T

## PE-AV-Small: Contrastive Audio-Video-Text Encoder

**Method**: Purpose-built contrastive model with separate projection heads per modality pair.

### Results (Flickr30k, n=200)

| Direction | R@1 | R@5 | R@10 |
|-----------|-----|-----|------|
| Text→Image (single-frame) | **0.825** | 0.960 | 0.975 |
| Image→Text (single-frame) | 0.010 | 0.055 | 0.175 |
| Audio→Text (synthetic) | 0.000 | 0.020 | 0.055 |

### Key Findings

1. **Image embeddings are degenerate for single frames** — avg pairwise cosine = 0.984 (near-collapsed)
2. **T→I R@1=0.825 is misleadingly high** — text embeddings are distinctive enough to find the right image among near-identical representations, but the reverse fails completely
3. **Audio pipeline works** — embeddings are differentiated (intra-audio cos=0.716), all modalities share 1024-dim space
4. **Two separate text heads**: video-aligned and audio-aligned text embeddings differ (matched cos=0.677), meaning the text encoder produces modality-specific projections
5. **Designed for video, not images** — temporal encoder needs multiple frames for discriminative output

## Architecture Implications for Cortext

### Recommended Approach

Use **both models** in a complementary configuration:

1. **Qwen3.5-0.8B** for text + image embedding:
   - Layer blend (0.5*L2 + 0.5*L22), mean-pooled → 1024d
   - Strong cross-modal alignment without contrastive training
   - Also provides text generation / extraction capability (dual use)

2. **PE-AV-Small** for audio embedding:
   - Purpose-built audio encoder with DAC frontend → 1024d
   - Trained on real audio-text correspondence
   - Audio→text retrieval quality depends on real speech/sounds (not testable with synthetic tones)

3. **Alignment**: Both produce 1024d vectors. A small learned linear projection (or even just L2-norm + cosine) may be sufficient to align the two spaces for cross-modal retrieval across all three modalities.

### Alternative: Single Model

**Qwen2.5-Omni-3B** (~6B params) handles text + image + audio in a shared 2048d hidden space, but:
- 7x larger than either model above
- Requires quantization for MPS
- Hidden state quality for retrieval is unknown (would need its own experiment)

### Integration Path

- Replace the legacy multimodal encoder with Qwen3.5-0.8B hidden states (better image+text alignment, smaller)
- Add PE-AV audio encoder alongside (or keep sherpa-onnx ASR → text → Qwen3.5 embed pipeline)
- Both models can run on MPS with float16
