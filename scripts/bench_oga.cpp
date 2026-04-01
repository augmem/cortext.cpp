/**
 * Benchmark onnxruntime-genai (OGA) inference in C++.
 * Tests Int4 quantized models with internal KV cache management.
 *
 * Build:
 *   g++ -std=c++17 -O3 bench_oga.cpp -o bench_oga \
 *       -I/path/to/oga/include -L/path/to/oga/lib \
 *       -lonnxruntime-genai
 *
 * Usage:
 *   ./bench_oga --model models/phi2-oga-test --max-tokens 50
 */

#include <chrono>
#include <iostream>
#include <string>
#include <cstdlib>

#include "ort_genai.h"

struct BenchmarkResult {
    std::string model;
    int threads;
    double tps;
    double ms_per_token;
    double prefill_ms;
    double decode_ms;
    int tokens;
    double load_s;
};

BenchmarkResult benchmark_oga(
    const std::string& model_path,
    const std::string& prompt,
    int max_tokens = 50,
    int num_threads = 1
) {
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "OGA C++ Benchmark: " << model_path << "\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "Prompt: " << prompt.substr(0, 50) << "...\n";
    std::cout << "Max tokens: " << max_tokens << "\n";
    std::cout << std::string(60, '=') << "\n";

    // Load model
    auto t_load_start = Clock::now();
    auto model = OgaModel::Create(model_path.c_str());
    auto tokenizer = OgaTokenizer::Create(*model);
    auto load_time = std::chrono::duration<double>(Clock::now() - t_load_start).count();
    std::cout << "Model load: " << load_time << "s\n";

    // Create generator params
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", max_tokens + 100);
    params->SetSearchOptionBool("do_sample", false);  // Greedy

    // Encode prompt
    auto sequences = OgaSequences::Create();
    tokenizer->Encode(prompt.c_str(), *sequences);

    // Create generator and append input tokens
    auto generator = OgaGenerator::Create(*model, *params);
    generator->AppendTokenSequences(*sequences);

    // Prefill (first token)
    auto t_prefill_start = Clock::now();
    generator->GenerateNextToken();
    double prefill_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - t_prefill_start).count();
    std::cout << "Prefill: " << prefill_ms << "ms\n";

    // Decode loop
    int tokens_generated = 1;  // First token from prefill
    auto t_decode_start = Clock::now();

    while (!generator->IsDone() && tokens_generated < max_tokens) {
        generator->GenerateNextToken();
        tokens_generated++;
    }

    double decode_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - t_decode_start).count();

    // Calculate metrics
    double tps = 0, ms_per_token = 0;
    if (tokens_generated > 0 && decode_ms > 0) {
        tps = (tokens_generated * 1000.0) / decode_ms;
        ms_per_token = decode_ms / tokens_generated;
    }

    // Decode output
    size_t seq_len = generator->GetSequenceCount(0);
    const int32_t* seq_data = generator->GetSequenceData(0);
    auto output_text = tokenizer->Decode(seq_data, seq_len);

    std::cout << "\nResults:\n";
    std::cout << "  Tokens generated: " << tokens_generated << "\n";
    std::cout << "  Decode time: " << decode_ms << "ms\n";
    std::cout << "  Throughput: " << tps << " tokens/sec\n";
    std::cout << "  Avg per token: " << ms_per_token << "ms\n";
    std::cout << "\nOutput: " << std::string(output_text.p_).substr(0, 200) << "...\n";

    return {
        model_path,
        num_threads,
        tps,
        ms_per_token,
        prefill_ms,
        decode_ms,
        tokens_generated,
        load_time
    };
}

int main(int argc, char* argv[]) {
    std::string model_path = "models/phi2-oga-test";
    std::string prompt = "Explain quantum computing in simple terms.";
    int max_tokens = 50;
    int threads = 1;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "  --model PATH      Path to OGA model directory\n";
            std::cout << "  --prompt TEXT     Input prompt\n";
            std::cout << "  --max-tokens N    Max tokens to generate\n";
            std::cout << "  --threads N       Number of threads\n";
            return 0;
        }
    }

    // Set thread count
    setenv("OMP_NUM_THREADS", std::to_string(threads).c_str(), 1);
    setenv("MKL_NUM_THREADS", std::to_string(threads).c_str(), 1);

    auto result = benchmark_oga(model_path, prompt, max_tokens, threads);

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "SUMMARY\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Model: " << result.model << "\n";
    std::cout << "Throughput: " << result.tps << " tokens/sec (single-thread)\n";
    std::cout << "Prefill: " << result.prefill_ms << "ms\n";
    std::cout << "ms/token: " << result.ms_per_token << "\n";

    return 0;
}
