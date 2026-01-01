## MonoidEmbed: a block‑compiled, codebook‑selective, prefix‑scannable state machine for byte embeddings

### What is novel here

Mamba’s selectivity makes the layer efficient only if you run a hardware‑aware scan recurrence, because you cannot use the time‑invariant convolution fast path when parameters depend on input. ([arXiv][1])
MonoidEmbed takes that idea one step further for CPU streaming:

1. **Turn each token’s effect into a tiny affine “monoid element”** (diagonal affine transform) chosen by a codebook (LUT), avoiding per‑token dense projections. This is philosophically aligned with table‑lookup inference work (replace matmul with LUT). ([ACM Digital Library][2])

2. **Compile an entire microblock of tokens into one composed monoid transform** and apply it to the persistent state once per block. This slashes state read/write traffic and eliminates per‑token framework overhead. The associativity is exactly the prefix‑scan property emphasized in recent “prefix scannable models / sequential‑parallel duality” work.

3. **Exploit 3 cores with a hard real‑time pipeline**: one core does ingress + emission, two cores do the composed‑transform build and application on disjoint channel tiles (CPU‑agnostic vector kernels).

The key “push the limits” move is not just LUTs. It is **block‑compiling the recurrence so the persistent state is not touched per token**, only per microblock, while still being exact for that recurrence.

---

# 1) Root‑cause: why a single‑core Mamba‑2 streaming stepper lands near ~4 KB/s

On CPU, the usual streaming Mamba‑2 path hits several stacked bottlenecks:

1. **Selectivity forces recurrent scan**, not a single FFT/conv style kernel. Mamba explicitly notes that input‑dependent parameters break the time‑invariant efficiency path and require a scan/recurrent algorithm. ([arXiv][1])

2. **Too many per‑token small ops**: even if each linear is optimized, the step function is a long chain of small kernels (proj, gates, elementwise, state update, pooling, head). Dispatch and synchronization overhead matters when you do it thousands of times per second.

3. **State traffic and cache misses**: deep SSM stacks touch a lot of state per token. If that state (plus weights touched per step) does not fit in L2, you pay recurring L2/L3 bandwidth and latency costs.

4. **Quantization overhead**: INT8 helps only if the whole hot path stays in a friendly layout and you do not pay repeated requantization or scale handling. SmoothQuant exists because activations are hard to quantize and outliers force expensive handling. ([arXiv][3])

So even a strong Mamba‑2 kernel can end up “death by a thousand cuts” at byte granularity.

---

# 2) The MonoidEmbed architecture

## 2.1 Streaming interface

* Input stream: raw bytes, and optionally TEMPEST tokens.
* Output: Matryoshka embeddings, canonical 512, usable prefixes 256/128, INT4 output acceptable.

## 2.2 Tokenization that preserves streaming and boosts throughput

You need to decouple “bytes in” from “model steps”.

**Ingress rule** (bounded latency):

* Default: **microblock budget of 256 bytes** (configurable down to 64 bytes for interactive text).
* Raw fallback: pack bytes into cheap tokens (e.g., 8 bytes per token).
* Known formats: TEMPEST emits structured tokens. For JPEG‑like inputs, compressed‑domain / partial decode paths are well established as a speed lever. ([NeurIPS Papers][4])

The architecture does not require TEMPEST to function, but TEMPEST multiplies throughput when available.

## 2.3 State representation

Persistent state is tiny and cache resident:

* Total state dimension `D = 512` in `int16` (1 KB).
* Split into 8 tiles of 64 dims for cache and SIMD friendliness.

We also maintain a stable streaming “reported embedding accumulator”:

* `p ∈ int16[512]` as EMA of state‑derived vector (or directly of state).

## 2.4 Core update rule: codebook‑selective diagonal affine recurrence

Each token is mapped to a discrete code `c` in `[0, C)` (C=256 is a good default).

For each code we store two vectors (per tile):

* `a[c] ∈ int8[512]` (decay, Q1.7 fixed point, magnitude constrained)
* `b[c] ∈ int16[512]` (injection)

Per token recurrence (conceptual, not how we implement it):

* `s ← a[c] ⊙ s + b[c]`

This is intentionally prefix‑scannable (associative) as an affine transform. That is the lever that lets us block‑compile.

This echoes the general scan framing used in SSMs like Mamba, and the broader “prefix scannable models” view: state updates computable by associative aggregation. ([arXiv][1])

## 2.5 Block compilation: compose the microblock into one transform

For a microblock of K token codes `c0..cK-1`, define monoid elements:

* `m_t = (a[c_t], b[c_t])`

Composition (diagonal affine):

* `(a2, b2) ⊗ (a1, b1) = (a2 ⊙ a1, b2 + a2 ⊙ b1)`

This is associative.

So the whole block becomes:

* `(A_blk, B_blk) = m_{K-1} ⊗ ... ⊗ m_0`
* `s ← A_blk ⊙ s + B_blk`

Important: we **do not touch `s` inside the microblock**. We only update the running composed transform `(A,B)` in registers.

That is the big structural win vs “step the model per byte”.

## 2.6 Cheap nonlinearity and cross‑tile mixing (block‑rate)

A purely diagonal affine machine can be too weak, so we add a block‑rate nonlinear mixer that is still CPU‑cheap.

After applying the block transform:

1. `s ← sat16(s)`
2. Apply a **grouped 64×64 int8 mixing per tile** every microblock:

   * For each tile t: `s_t ← s_t + W_t * q8(s_t)`
     where `W_t` is int8 64×64 and `q8` is a cheap per‑tile quantizer.
3. Apply a **lookup nonlinearity** per element: `s ← lut_tanh(s)` (256‑entry LUT per byte of magnitude).

This gives nonlinearity and within‑tile interactions without heavy GEMMs.

## 2.7 Emission

When the application requests an embedding (or every N microblocks), compute:

* `e512 = Head(q8(s))`

Head is a standard int8 matmul:

* Option A: `512 → 512` (single layer)
* Option B: `512 → 1536 → 512` if you can afford it at emission cadence

Use SmoothQuant on the head so it stays W8A8 reliably. ([arXiv][3])
Use oneDNN or XNNPACK depending on platform, both are designed for high‑efficiency CPU inference. ([GitHub][5])

Finally pack to INT4, Matryoshka prefixes are just slices.

---

# 3) Exact step function pseudocode

### Data structures (persistent)

```c
const int D = 512;
const int TILE = 64;
const int NT = D / TILE;     // 8
const int C = 256;           // codebook size

// Persistent
int16 s[D];                  // recurrent state
int16 p[D];                  // EMA pooled state (optional)

// Codebooks (hot, cache-resident)
int8  a[C][D];               // decay vectors (Q1.7)
int16 b[C][D];               // injection vectors

// Block mixer
int8  W[NT][TILE][TILE];     // 8 tiles, each 64x64
int8  tanh_lut[256];         // magnitude -> approx tanh
```

### Microblock processing (core hot path)

```c
// Process K token codes (K chosen so bytes per block stays bounded)
void update_block(uint8 codes[K]) {
  // Accumulators for composed affine transform
  // Start with identity: A = 1, B = 0
  int16 A[D];  set_to_one_q1_15(A);     // Q1.15 identity
  int16 B[D];  set_to_zero(B);

  // Compose transforms for the block, without touching s
  for (int t = 0; t < K; t++) {
    uint8 c = codes[t];

    // For each tile: B = b[c] + a[c] * B;  A = a[c] * A
    for (int tile = 0; tile < NT; tile++) {
      // vectorized over 64 dims
      vec_i16 Bv = load_i16(B + tile*TILE);
      vec_i16 Av = load_i16(A + tile*TILE);

      vec_i8  av8 = load_i8(a[c] + tile*TILE);
      vec_i16 av  = widen_q1_15(av8);           // Q1.7 -> Q1.15
      vec_i16 bv  = load_i16(b[c] + tile*TILE);

      Bv = bv + mul_q1_15(av, Bv);
      Av =      mul_q1_15(av, Av);

      store_i16(B + tile*TILE, Bv);
      store_i16(A + tile*TILE, Av);
    }
  }

  // Apply compiled block transform once
  for (int tile = 0; tile < NT; tile++) {
    vec_i16 sv = load_i16(s + tile*TILE);
    vec_i16 Av = load_i16(A + tile*TILE);
    vec_i16 Bv = load_i16(B + tile*TILE);

    sv = mul_q1_15(Av, sv) + Bv;
    store_i16(s + tile*TILE, sv);
  }

  // Block mixer: per tile 64x64 + LUT tanh
  for (int tile = 0; tile < NT; tile++) {
    int8 q[TILE] = quantize_tile_to_int8(s + tile*TILE);

    int16 delta[TILE] = matmul_i8_i8_to_i16(W[tile], q);   // 64x64
    add_i16_inplace(s + tile*TILE, delta);

    apply_tanh_lut_i16_inplace(s + tile*TILE, tanh_lut);
  }

  // Optional: update EMA pooled vector p at block rate
  ema_i16(p, s);
}
```

This is intentionally CPU agnostic:

* Only needs “vector ops over TILE” and “int8 64×64 microkernel”.
* TILE can be 32 for smaller SIMD, 128 for wider SIMD.
* You can implement vector ops with portable SIMD (Highway / std::simd) and provide ISA specializations.

---

# 4) Concrete kernel plan (no handwaving)

## 4.1 Hot kernel: `compose_block_affine`

Goal: keep A/B accumulators in L1 and registers, stream LUT rows.

Layout:

* `a` stored as `a[c][tile][lane]` contiguous in `lane`.
* `b` stored similarly.

Blocking:

* Unroll over tokens: process 2 codes per iteration to reduce loop overhead.
* Prefetch next `a[c_next]` and `b[c_next]` when reading `c`.

SIMD:

* Operations are multiply, add, widen, saturate. All vectorizable.
* No branches in the inner loop.

Quantization:

* `a[c]` stored Q1.7, widened to Q1.15 for multiply with `A,B,s`.
* `B` in int16, `A` in Q1.15.

Why this wins:

* Persistent state `s` is read and written once per microblock, not per token.
* If you pick K so the block is 128 to 512 bytes, the amortized state traffic per byte drops by 16× to 64×.

## 4.2 Block mixer kernel

Use a fixed microkernel shape: **64×64 int8 matmul** per tile.

Implementation:

* On x86, map to a hand microkernel (AVX2/AVX‑512 dpbusd if available).
* On ARM, use NEON dotprod if available.
* Fallback is a scalar microkernel.

Because it is always 64×64, you can optimize it aggressively without needing a big GEMM library.

If you do want to use a library:

* oneDNN has explicit int8 matmul support and encourages “create primitive once, reuse, weights prepacked” which is exactly what we want.

## 4.3 Head kernel

Emission cadence is low, so you can use oneDNN or XNNPACK safely.

* XNNPACK is explicitly cross‑platform (ARM, x86, WASM, RISC‑V), which matches your CPU‑agnostic requirement. ([GitHub][5])

Apply SmoothQuant to keep W8A8 stable for the head. ([arXiv][3])

---

# 5) 3‑core scheduling plan (development ready)

Pin threads:

* Core0: ingress + emission
* Core1: update_block tiles 0..3
* Core2: update_block tiles 4..7

Streaming queue:

* SPSC ring from core0 to (core1, core2): microblocks carry only:

  * `codes[K]` (K bytes)
  * `block_id` and timestamp
    No large token tensors. Bounded memory.

Sync:

* Core1 and Core2 process the same microblock ID on different tiles.
* Use a very small barrier per microblock:

  * after compose, before applying to `s` (each touches disjoint tiles so barrier is minimal)
* Emission reads `s` via double‑buffer snapshot:

  * core1/core2 write into `s_next`
  * swap pointers atomically at block boundary

Why this scales:

* The hot loop is pure SIMD math and LUT streaming.
* Two cores split the state dimension cleanly, minimal contention.
* Core0 stays light if it only produces codes and occasionally runs head.

---

# 6) Throughput estimate and explicit speedup decomposition

Let baseline be 4 KB/s on 1 core.

### Worst case: raw bytes, no TEMPEST

Assume:

* 8 bytes per token
* microblock size 256 bytes (K=32 codes)

**Algorithmic speedup on 1 core**

* Baseline touches state per byte step and runs heavy projections.
* MonoidEmbed touches state once per 256 bytes and uses LUT affine composition.

Conservative estimate:

* 1 core MonoidEmbed: **80 to 160 KB/s** (20× to 40×)

Where it comes from:

* 8× fewer “token steps” (bytepack)
* 3× to 6× cheaper step (no per‑token matmuls, no deep state traffic)
* plus elimination of per‑op overhead by fusing

**Scaling to 3 cores**

* Parallelizable fraction is the block update and mixer, about 85% of runtime.
* With two worker cores on disjoint tiles, you get ~1.7× to ~2.0× on that portion.
* Core0 overlaps ingress and head, so it should not limit steady‑state throughput.

Expected:

* 3 cores MonoidEmbed: **200 to 350 KB/s** (50× to 87× vs baseline)

### Typical case: TEMPEST tokens available

If TEMPEST reduces steps by 5× to 50× depending on format, you multiply the above by that reduction. Compressed‑domain inference is a proven lever (JPEG‑domain work shows large wins by avoiding full decode). ([NeurIPS Papers][4])

Even a conservative 8× token reduction puts you well above 1 MB/s.

---

# 7) Quality preservation plan

You will not get quality “for free” from this recurrence. You preserve it by training it as a student:

1. **Geometric distillation to the Gemma 3n teacher embedding** (cosine + MSE on normalized embeddings).
2. **Matryoshka losses**: apply the same teacher alignment at 128, 256, 512 prefixes.
3. **Quantization aware training**

   * fake quant on `a`, `b`, and on state `s` passthrough
   * saturation penalty (discourage frequent clipping)
4. **Stability constraints**

   * enforce `|a[c]| ≤ 1` by parameterization
   * penalize `E[log ||s||]` drift over long unrolled streams

Optional cheap corrector (still bounded):

* Every 2 to 4 seconds, run a small windowed encoder on last W bytes and do EMA on the reported embedding only. This matches your spec idea but keeps the fast path dominant.

---

# 8) Validation plan

Microbenchmarks first:

* `compose_block_affine`: cycles per token per tile, L1 miss rate
* `apply_block`: cycles per tile
* mixer 64×64: cycles per tile

End to end:

* KB/s sustained for 1 hour, raw and TEMPEST
* p50/p99 update latency per 256 byte microblock
* p50/p99 embedding emission latency (512d)
* memory (RSS, persistent state, queue)

Quality gates:

* cosine similarity to teacher embeddings on held‑out multimodal streams
* drift over hours: cosine between embeddings at time t and teacher windowed sample
* prefix quality at 128 and 256 dims

---

# 9) Implementation roadmap and top risks

## Roadmap

1. C++ reference scalar implementation with exact fixed point formats.
2. Add portable SIMD (Highway or std::simd), keep scalar fallback.
3. Add 3‑thread scheduler and ring buffers.
4. Add the 64×64 tile microkernel (one per ISA).
5. Integrate SmoothQuant and int8 head via oneDNN or XNNPACK. ([GitHub][5])
6. Train student with distillation + QAT, export codebooks and head.

## Risks and de‑risk

1. **Quality ceiling** (diagonal block recurrence too weak)
   De‑risk: increase tile mixer strength (64×64), add a second mixer every N blocks, or add a small cross‑tile permutation each block.

2. **State drift under int16**
   De‑risk: decay constraints, saturation penalty, long‑stream training, optional periodic embedding re‑anchoring.

3. **Table lookup bandwidth**
   LUT‑based inference papers note hardware can be unfriendly to naive table lookups. ([ResearchGate][6])
   De‑risk: keep codebooks small, contiguous, prefetch, tile‑locality, avoid random access.

---

If you want to push this further after an initial prototype: the next “edge” is to add a second monoid channel that operates directly on TEMPEST structural events (DCT block types, Huffman run lengths, MDCT band energies), so the same block compiler covers both semantic content and format signatures with essentially the same kernels. That is where you get the 100× range reliably across modalities.

[1]: https://arxiv.org/pdf/2312.00752?utm_source=chatgpt.com "Mamba: Linear-Time Sequence Modeling with Selective ..."
[2]: https://dl.acm.org/doi/10.1145/3665314.3670804?utm_source=chatgpt.com "LUTIN: Efficient Neural Network Inference with Table Lookup"
[3]: https://arxiv.org/pdf/2211.10438?utm_source=chatgpt.com "SmoothQuant"
[4]: https://papers.neurips.cc/paper/2018/file/7af6266cc52234b5aa339b16695f7fc4-Paper.pdf?utm_source=chatgpt.com "Faster Neural Networks Straight from JPEG"
[5]: https://github.com/google/XNNPACK?utm_source=chatgpt.com "google/XNNPACK: High-efficiency floating-point neural ..."
[6]: https://www.researchgate.net/publication/368333493_LUT-NN_Towards_Unified_Neural_Network_Inference_by_Table_Lookup?utm_source=chatgpt.com "Towards Unified Neural Network Inference by Table Lookup"
