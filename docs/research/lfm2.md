# **Operational Blueprint for Deploying ExecuTorch and Quantized LFM2-Audio-1.5B on Raspberry Pi 5**

## **Executive Summary**

The proliferation of multimodal artificial intelligence has necessitated a migration of inference workloads from centralized data centers to the network edge, driven by requirements for latency reduction, privacy preservation, and bandwidth optimization. This report articulates a comprehensive engineering strategy for deploying Liquid AI’s LFM2-Audio-1.5B—a hybrid generative model combining linear operators, convolutions, and attention mechanisms—on the Raspberry Pi 5 embedded platform. Specifically, the analysis focuses on the extraction of high-dimensional vector embeddings from both audio and textual modalities using ExecuTorch, PyTorch’s static runtime environment tailored for constrained systems.

The Raspberry Pi 5, equipped with the Broadcom BCM2712 System-on-Chip (SoC) and quad-core Arm Cortex-A76 processors, represents a significant leap in edge processing capabilities. However, deploying a 1.5 billion parameter model on a device with limited memory (4GB or 8GB) and thermal constraints requires rigorous optimization. This report details the utilization of 4-bit weight-only quantization with 8-bit dynamic activations (8da4w) to compress the model footprint to approximately 1.2 GB, ensuring residency within the device's physical RAM while maintaining inference fidelity. Furthermore, it explicates the software architecture required to interface with the ExecuTorch C++ runtime, bypassing the Python interpreter to maximize throughput via the XNNPACK backend.

By leveraging a custom graph export pipeline that exposes internal hidden states rather than generation logits, this deployment strategy transforms the LFM2-Audio-1.5B from a generative chatbot into a robust semantic encoder. This capability enables downstream applications such as offline semantic search, acoustic anomaly detection, and privacy-centric voice analysis, all executed locally on commodity embedded hardware.

## **1\. Architectural Analysis of LFM2-Audio-1.5B**

To effectively map the LFM2-Audio-1.5B model onto the Raspberry Pi 5’s hardware resources, one must first dissect its heterogenous architecture. Unlike standard Transformer models that rely exclusively on self-attention, the LFM2 family integrates concepts from dynamic systems and signal processing, creating unique challenges and opportunities for edge deployment.

### **1.1 The Hybrid Backbone: Convergence of Linear and Attention Mechanisms**

The core of the LFM2-Audio-1.5B model is its hybrid backbone, which diverges from the ubiquitous Transformer architecture found in models like Llama or GPT. The architecture is designed to mitigate the quadratic computational complexity ($O(N^2)$) associated with global self-attention, which is particularly punitive for long sequences typical of audio processing.

Gated Linear Operators and Short Convolutions:  
A defining characteristic of the LFM2 architecture is its extensive use of gated short convolutions and linear recurrence layers.1 These layers function as efficient token mixers, allowing the model to capture local dependencies—such as phonemic transitions in speech or syntactic structures in text—without the heavy memory bandwidth cost of full attention matrices. For the Raspberry Pi 5, this is advantageous; the Cortex-A76’s NEON vector units can process these convolutional operations with high arithmetic intensity, minimizing pipeline stalls associated with memory access.  
Grouped Query Attention (GQA):  
While linear operators handle local context, the model employs Grouped Query Attention (GQA) to manage long-range dependencies and global context.2 GQA reduces the memory bandwidth required for loading Key (K) and Value (V) caches by sharing KV heads across multiple Query (Q) heads. In a quantized execution environment, GQA significantly lowers the memory transfer overhead during the decoding phase, which is often the bottleneck on embedded CPUs with 128-bit memory buses like the BCM2712.

### **1.2 The Audio Pipeline: FastConformer and Mimi**

LFM2-Audio-1.5B is an end-to-end multimodal model, but its internal processing of audio involves distinct stages that must be replicated in the ExecuTorch graph.

FastConformer Encoder:  
The model utilizes a FastConformer encoder, specifically initialized from the canary-180m-flash checkpoint.3 Standard Conformer models combine Convolutional Neural Networks (CNNs) and Transformers. The "Fast" variant introduces aggressive subsampling (typically 4x or 8x) early in the network layers. This downsampling reduces the sequence length of the audio representation before it reaches the computationally expensive transformer blocks. For an edge device, this downsampling is critical; processing 10 seconds of audio at a high sample rate would generate thousands of tokens, overwhelming the CPU. FastConformer compresses this temporal resolution, allowing the RPi 5 to maintain real-time throughput.  
Mimi Tokenizer:  
The output of the audio encoder interfaces with an RQ-Transformer (Residual Vector Quantization) that maps continuous representations to discrete tokens using the Mimi codebook.3 Mimi is a neural audio codec that utilizes 8 parallel codebooks to capture high-fidelity audio information at a 24kHz sampling rate.5 However, for the purpose of embedding extraction, the discrete tokens are often less valuable than the continuous vector representations generated by the FastConformer or the hidden states of the LFM2 backbone immediately prior to tokenization. This report focuses on extracting these continuous vectors to preserve rich semantic and acoustic information.

### **1.3 Text Modality and Tokenization**

The text pathway of LFM2-Audio utilizes a vocabulary size of 65,536 tokens.3 While not explicitly identified as a specific library in the provided research snippets, the vocabulary size and architecture suggest a Byte-Pair Encoding (BPE) tokenizer similar to those used in Llama 3 or TikToken.6 Correctly implementing this tokenizer in C++ is a prerequisite for generating text embeddings, as the model expects integer token IDs as input. The ExecuTorch ecosystem provides extension/llm/tokenizer libraries that support these formats, which must be configured to match the tokenizer.json or tokenizer.model file distributed with LFM2.

## **2\. Hardware and Software Ecosystem**

The successful deployment of high-performance AI on embedded systems requires a deep understanding of the target hardware's capabilities and constraints. This section analyzes the Raspberry Pi 5 platform and the ExecuTorch software stack.

### **2.1 Raspberry Pi 5 Hardware Specifications**

The Raspberry Pi 5 introduces the Broadcom BCM2712 SoC, which fundamentally alters the performance landscape for single-board computers.

| Component | Specification | Implication for LFM2 Deployment |
| :---- | :---- | :---- |
| **SoC** | Broadcom BCM2712 | 16nm process technology allowing higher clock speeds and thermal density. |
| **CPU** | Quad-core Arm Cortex-A76 @ 2.4GHz | The Cortex-A76 is an out-of-order superscalar core with dual 128-bit NEON execution pipes. It offers \~2-3x the integer and floating-point performance of the Cortex-A72 found in the RPi 4\. |
| **RAM** | 4GB or 8GB LPDDR4X-4267 | High memory bandwidth is crucial for transformer inference. The 4GB model leaves little headroom for a 1.5B parameter model (approx 3GB in FP16), making quantization mandatory. |
| **PCIe** | PCIe 2.0 x1 Interface | Allows for fast NVMe storage, reducing model load times compared to MicroSD cards. |
| **Cooling** | Active Cooler Required | Sustained inference utilizes all four cores at 100% load. Without the official active cooler, the CPU will throttle within seconds, degrading inference latency. |

NEON Optimization:  
The ARMv8.2-A architecture of the Cortex-A76 supports half-precision floating-point (FP16) arithmetic and dot-product instructions (UDOT/SDOT) for 8-bit integers. ExecuTorch’s XNNPACK backend is explicitly optimized to leverage these instructions, allowing 4-bit quantized weights to be unpacked and computed against 8-bit activations with high efficiency.

### **2.2 ExecuTorch Runtime Architecture**

ExecuTorch is designed to address the fragmentation and bloat of edge AI deployment. Unlike LibTorch (PyTorch C++), which includes autograd, backward passes, and a JIT compiler, ExecuTorch adopts an "Ahead-of-Time" (AOT) compilation philosophy.

**The Execution Lifecycle:**

1. **Export (Host):** The Python model is traced into a graph (EXIR \- ExecuTorch Intermediate Representation).  
2. **Compilation (Host):** The graph is optimized (operator fusion, dead code elimination) and memory-planned. A "Memory Plan" is generated, calculating exactly how much RAM is needed for activation tensors at every step of execution.  
3. **Serialization (Host):** The graph and weights are serialized into a flatbuffer file (.pte).  
4. **Execution (Device):** The C++ runtime loads the .pte file. It allocates a single static memory arena based on the Memory Plan. This eliminates dynamic memory allocation (malloc/free) overhead during inference, preventing memory fragmentation and ensuring deterministic latency—a critical requirement for real-time audio processing.

Backend Delegation:  
ExecuTorch supports "delegates," which allow subgraphs to be offloaded to specialized hardware or optimized libraries. For the Raspberry Pi 5, the XNNPACK delegate is the primary acceleration engine.7 XNNPACK provides highly tuned implementations of matrix multiplications and convolutions for ARM64 CPUs, handling the packing and tile-based execution strategies necessary to keep the processor's pipelines fed.

## **3\. Methodology: Model Transformation and Export**

The standard export pipelines for LFM2 models typically target text generation (logits output). To satisfy the requirement of **embedding extraction**, we must engineer a custom export workflow that intercepts the model's internal states.

### **3.1 Designing the Embedding Extraction Wrappers**

The LFM2-Audio-1.5B model is a monolith in its original PyTorch definition. To provide flexibility and manage memory usage on the RPi 5, it is strategically advantageous to export the model as two distinct ExecuTorch modules: the **Audio Encoder** and the **Backbone**.

Wrapper 1: The Backbone Embedder  
The objective is to capture the output of the final transformer/hybrid block before it enters the vocabulary projection layer.

* **Input:** Token IDs (for text) or Continuous Embeddings (from the audio encoder).  
* **Transformation:** The wrapper class must modify the forward method. Standard LFM2 models return a CausalLMOutput object. The wrapper must invoke the underlying model with output\_hidden\_states=True and return the last\_hidden\_state.  
* **Pooling Strategy:** The raw output is a sequence of vectors \`\`. For sentence or audio segment embeddings, the wrapper should implement a pooling operation—either Mean Pooling (averaging across the sequence dimension) or EOS Pooling (taking the vector corresponding to the End-Of-Sequence token). Given the hybrid nature of LFM2, mean pooling often yields more robust representations for retrieval tasks.

Wrapper 2: The Audio Encoder  
The FastConformer encoder takes audio features and outputs a sequence of acoustic vectors.

* **Input:** Mel-Spectrogram tensor \`\`.  
* **Transformation:** The wrapper encapsulates the FastConformer module. It is essential to ensure that the input dimensions are dynamic (to handle varying audio lengths) or padded to a fixed context length supported by the RPi 5's memory plan (e.g., 10 seconds of audio).  
* **Output:** Acoustic embeddings \`\`.

### **3.2 Graph Capture and Lowering**

Once the wrappers are defined in Python, the export process involves capturing the computational graph.

1. **Tracing:** Using torch.export.capture\_pre\_autograd\_graph, we trace the execution flow using dummy inputs. This produces a Core ATen graph.  
2. **Operator Decomposition:** Some specialized operators in LFM2 (like the gated linear units) may not have direct equivalents in the standard ATen operator set used by ExecuTorch. The export script must register custom implementations or rely on torch.export to decompose these into primitive operations (add, mul, sigmoid) that XNNPACK can accelerate.  
3. **Memory Planning:** During the lowering to ExecuTorch, the memory planner analyzes the liveness of every tensor. For a 1.5B model, the activation memory can be substantial. The planner allows us to visualize peak memory usage and adjust the batch size (strictly 1 for RPi 5\) or context length (e.g., 2048 tokens) to fit within the 4GB/8GB envelope.

### **3.3 Audio Preprocessing Export**

FastConformer expects a specific spectral representation of audio. Implementing the Short-Time Fourier Transform (STFT) and Mel-filterbank conversion in raw C++ is error-prone and slow. A superior approach is to export a **preprocessing module** as a separate .pte file using executorch.extension.audio.

* **Module:** executorch.extension.audio.mel\_spectrogram.9  
* **Configuration:**  
  * Sample Rate: 48,000 Hz.3  
  * n\_fft: 1024 (Typical for 48k audio to capture sufficient frequency resolution).  
  * hop\_length: 480 (10ms hop).  
  * n\_mels: 80\.  
* **Output:** This module will accept a 1D float tensor (waveform) and output a 2D tensor (Mel-spectrogram) ready for the Audio Encoder.

## **4\. Quantization Strategy**

Running the model in FP32 (32-bit floating point) or even BF16 (16-bit Brain Float) is infeasible on the Raspberry Pi 5 due to memory bandwidth and capacity limitations. We will employ **Quantization-Aware Training (QAT)** or **Post-Training Quantization (PTQ)** techniques compatible with the XNNPACK backend.

### **4.1 The 8da4w Scheme**

The research snippets highlight the use of **8da4w** (8-bit dynamic activations, 4-bit weights) as the optimal scheme for LFM2 models on ExecuTorch.10

* **4-bit Weights (Groupwise):** The weights of linear layers are compressed to 4 bits. To maintain accuracy, weights are grouped (e.g., blocks of 32 or 128), and a scaling factor is stored for each group. This reduces the model size by approximately 8x compared to FP32, bringing the 1.5B parameter model down to \~0.8 \- 1.0 GB.  
* **8-bit Dynamic Activations:** Activation tensors (the data flowing through the network) are quantized to 8 bits dynamically at runtime. For every token, the range \[min, max\] of the activations is computed, and the values are scaled to the \[-128, 127\] range. This allows the CPU to use integer dot-product instructions (SDOT), which are significantly faster than floating-point math on the Cortex-A76.

### **4.2 The Calibration Process**

Although 8da4w is "dynamic" for activations, calibrating the model is crucial to determine the optimal scaling factors for the weights and to identify layers that are sensitive to quantization.

1. **Calibration Dataset:** We must feed a representative dataset (e.g., Librispeech samples for audio, C4 dataset snippets for text) through the unquantized model.  
2. **Observer Insertion:** The prepare\_pt2e function inserts "observer" nodes into the graph that record the distribution of values at each layer.  
3. **Quantization Conversion:** Based on the observed statistics, the convert\_pt2e function replaces floating-point operators with their quantized counterparts (e.g., quantized\_linear).  
4. **XNNPACK Targeting:** The XNNPACKQuantizer is specifically configured to map these quantized operators to the fbgemm or qnnpack schemas that the XNNPACK delegate recognizes.11

## **5\. Implementation: The C++ Runtime Application**

The culmination of the deployment is the C++ application running on the Raspberry Pi. This application acts as the orchestrator, managing data ingestion, preprocessing, inference, and result extraction.

### **5.1 Application Architecture**

The application is structured around the executorch::extension::Module API, which provides a high-level interface for loading .pte files.

**Component Diagram:**

1. **Input Interface:** Handles reading WAV files (using libsndfile or dr\_wav) and text strings.  
2. **Tokenizer:** A C++ BPE implementation (using sentencepiece or tiktoken-cpp) converts text to token IDs.  
3. **ExecuTorch Runner:** A wrapper class managing three Module instances: Preprocessor, AudioEncoder, and Backbone.  
4. **Output Handler:** Extracts tensors, performs pooling, and serializes the resulting embeddings (e.g., to JSON or binary).

### **5.2 Multimodal Runner Integration**

The research highlights the MultimodalRunner class in ExecuTorch 12, designed to unify text, vision, and audio. While a convenient abstraction, for *embedding extraction* specifically, a custom runner logic is often required to access intermediate tensors. However, we can adapt the principles of MultimodalRunner for our embedded C++ code.

**Key C++ Logic Flow:**

C++

// Pseudocode representation of the embedding pipeline  
\#**include** \<executorch/extension/module/module.h\>  
\#**include** \<executorch/extension/data\_loader/file\_data\_loader.h\>

// 1\. Initialize Modules  
auto backend \= executorch::extension::XnnpackBackend();  
Module audio\_encoder("audio\_encoder\_q4.pte");  
Module backbone("backbone\_q4.pte");  
Module preprocessor("preprocessor.pte");

// 2\. Process Audio Input  
std::vector\<float\> waveform \= load\_wav("input.wav");  
auto mel\_spec \= preprocessor.forward(from\_blob(waveform));

// 3\. Generate Acoustic Embeddings  
// The Audio Encoder produces embeddings representing the sound  
auto acoustic\_embeddings \= audio\_encoder.forward(mel\_spec);

// 4\. Generate Semantic Embeddings (Contextualized)  
// Feed acoustic embeddings into the Backbone to get high-level representations  
// Note: The Backbone exported must accept embeddings, not just token IDs  
auto final\_embeddings \= backbone.forward(acoustic\_embeddings);

// 5\. Post-Processing  
// Extract the tensor data pointer and perform normalization  
float\* embedding\_data \= final\_embeddings.data\_ptr\<float\>();  
normalize\_vector(embedding\_data, dimension);

### **5.3 Memory Management and Threading**

Static Memory Allocation:  
ExecuTorch’s static memory planning means the exact RAM usage is known at compile time. On the RPi 5, we must ensure that the operating system does not OOM (Out of Memory) kill the process.

* **Swap Space:** It is recommended to increase the swap file size to 4GB during the *development/compilation* phase, as linking large C++ binaries can consume significant RAM.  
* **Runtime RAM:** During inference, the quantized model should occupy \~1.2GB. This leaves ample room on a 4GB RPi 5 for the OS and input buffers.

Threading Model:  
The XNNPACK backend manages its own thread pool. The Cortex-A76 has 4 physical cores.

* **Configuration:** We configure XNNPACK to use 4 threads (xnnpack\_backend.set\_num\_threads(4)).  
* **Affinity:** Using taskset on Linux, we can pin the inference process to the specific CPU cores to avoid context switching and cache thrashing.

### **5.4 Tokenizer Implementation Details**

Since the text pipeline requires tokenization in C++, and Python libraries are unavailable in the pure C++ runtime, we must integrate a C++ tokenizer library.

* **SentencePiece:** If LFM2 uses a .model file (Protobuf), we link against libsentencepiece.  
* **TikToken/HuggingFace:** If LFM2 uses tokenizer.json, we use the executorch::extension::llm::tokenizer library. This library implements the BPE merge logic defined in the JSON file. We load the tokenizer.json and encode strings into std::vector\<long\> token IDs, which are then wrapped in an ExecuTorch tensor.

## **6\. Build and Deployment Plan**

### **6.1 Phase 1: Host Preparation (Cross-Compilation Environment)**

1. **Toolchain:** Install clang-15 and cmake.  
2. **SDK Generation:** Build the ExecuTorch libraries (libexecutorch.a, libextension\_module.a, libxnnpack\_backend.a) on the host. While cross-compilation is possible, building *directly* on the RPi 5 is often simpler for avoiding ABI mismatches, provided the RPi has sufficient swap space. Given the complexity, this plan assumes building on the RPi 5 itself to ensure perfect library compatibility.

### **6.2 Phase 2: Model Export (Python on Host)**

1. Develop export\_lfm2.py utilizing liquid-audio and executorch.  
2. Implement BackboneWrapper and AudioEncoderWrapper.  
3. Run quantization calibration using XNNPACKQuantizer.  
4. Generate lfm2\_audio\_enc\_8da4w.pte and lfm2\_backbone\_8da4w.pte.  
5. Generate preprocessor.pte with 48kHz configurations.  
6. Validate .pte files using the Python ExecuTorch runtime to ensure embedding outputs match the FP32 model (cosine similarity \> 0.99).

### **6.3 Phase 3: Runtime Development (C++ on RPi 5\)**

1. **Transfer:** Copy the .pte files and tokenizer.json to the RPi.  
2. **Source Code:** Write main.cpp implementing the logic described in Section 5.2.  
3. **CMake Configuration:**  
   CMake  
   cmake\_minimum\_required(VERSION 3.19)  
   project(lfm2\_embedder)  
   find\_package(executorch REQUIRED) \# Assuming installed/built on device  
   add\_executable(embedder main.cpp)  
   target\_link\_libraries(embedder executorch\_module executorch\_xnnpack\_backend extension\_data\_loader)

4. **Compilation:** Build with \-O3 \-mcpu=cortex-a76 to enable NEON auto-vectorization for any non-library code.

### **6.4 Phase 4: System Integration**

1. **Thermal Management:** Install the Active Cooler. Configure the fan profile in /boot/config.txt to aggressive cooling.  
2. **CPU Governor:** Set the CPU governor to performance to prevent clock downscaling during idle periods between audio chunks.  
   Bash  
   sudo cpupower frequency-set \-g performance

3. **Testing:** Run the embedder on a reference audio clip. Measure latency using ETDump. Verify that the embedding vector is consistent with the host validation.

## **7\. Performance Verification and Benchmarking**

To validate the deployment, a structured benchmarking regime is defined.

| Metric | Target | Verification Method |
| :---- | :---- | :---- |
| **Model Size** | \< 1.5 GB | Check file size of .pte binaries and runtime RAM usage via htop. |
| **Latency (Audio)** | \< 200ms per 5s chunk | Time the forward() call of AudioEncoder \+ Backbone. |
| **Latency (Text)** | \< 50ms per query | Time the forward() call of the Backbone for text tokens. |
| **Accuracy** | \> 0.98 Cosine Similarity | Compare RPi embeddings against FP16 Host embeddings on a standard dataset. |
| **Thermals** | \< 80°C | Monitor vcgencmd measure\_temp during sustained 10-minute inference load. |

Profiling with ETDump:  
ExecuTorch includes a profiling tool called ETDump. By enabling it in the build, the runtime generates a trace file. This trace can be visualized to see exactly how much time is spent in each operator (e.g., convolution, linear). This allows identifying if any operators are falling back to the generic CPU implementation instead of the optimized XNNPACK implementation, which would severely degrade performance.

## **8\. Conclusion**

This report has outlined a vertically integrated strategy for deploying LFM2-Audio-1.5B on the Raspberry Pi 5\. By bypassing the Python interpreter and utilizing a static C++ runtime with 8da4w quantization, the limitations of the embedded hardware are effectively mitigated. The transformation of the generative model into an embedding extractor unlocks powerful semantic capabilities at the edge, ensuring data privacy and operational independence from cloud infrastructure. The rigorous adherence to the ExecuTorch compilation path, specifically leveraging the XNNPACK delegate for the Cortex-A76 architecture, ensures that the deployment is not only functional but highly optimized for the specific instruction set capabilities of the Raspberry Pi 5\.

#### **Works cited**

1. LFM2-Audio: An End-to-End Audio Foundation Model \- Liquid AI, accessed December 23, 2025, [https://www.liquid.ai/blog/lfm2-audio-an-end-to-end-audio-foundation-model](https://www.liquid.ai/blog/lfm2-audio-an-end-to-end-audio-foundation-model)  
2. \[2511.23404\] LFM2 Technical Report \- arXiv, accessed December 23, 2025, [https://www.arxiv.org/abs/2511.23404](https://www.arxiv.org/abs/2511.23404)  
3. LiquidAI/LFM2-Audio-1.5B · Hugging Face, accessed December 23, 2025, [https://huggingface.co/LiquidAI/LFM2-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B)  
4. Neural audio codecs: how to get audio into LLMs \- Kyutai, accessed December 23, 2025, [https://kyutai.org/codec-explainer](https://kyutai.org/codec-explainer)  
5. Liquid4All/liquid-audio: Liquid Audio \- Speech-to-Speech ... \- GitHub, accessed December 23, 2025, [https://github.com/Liquid4All/liquid-audio](https://github.com/Liquid4All/liquid-audio)  
6. Llama3 \- Hugging Face, accessed December 23, 2025, [https://huggingface.co/docs/transformers/en/model\_doc/llama3](https://huggingface.co/docs/transformers/en/model_doc/llama3)  
7. 3D positional embeddings \- Harold Benoit, accessed December 23, 2025, [https://haroldbenoit.com/notes/ml/llms/multi-modality/3d-positional-embeddings](https://haroldbenoit.com/notes/ml/llms/multi-modality/3d-positional-embeddings)  
8. Part 2: Sing this song in another language, translating Machine Learning Pipelines to Android \- Arm Developer, accessed December 23, 2025, [https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/translating-machine-learning-pipelines-to-android](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/translating-machine-learning-pipelines-to-android)  
9. executorch/examples/models/whisper/README.md at main \- GitHub, accessed December 23, 2025, [https://github.com/pytorch/executorch/blob/main/examples/models/whisper/README.md](https://github.com/pytorch/executorch/blob/main/examples/models/whisper/README.md)  
10. Introducing LFM2: The Fastest On-Device Foundation Models on the Market | Liquid AI, accessed December 23, 2025, [https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)  
11. Category Page: Model Compression \- OpenVINO™ Blog, accessed December 23, 2025, [https://blog.openvino.ai/category/model-compression](https://blog.openvino.ai/category/model-compression)  
12. ExecuTorch \- On-Device AI Inference Powered by PyTorch, accessed December 23, 2025, [https://executorch.ai/](https://executorch.ai/)