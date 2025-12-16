// Minimal benchmark to isolate ONNX Runtime performance
// Compile: g++ -std=c++17 -O2 -I../include -I../third_party/onnxruntime/include minimal_bench.cpp -L../build -lcortext -lonnxruntime -o minimal_bench

#include <chrono>
#include <iostream>
#include <filesystem>

// Just test the raw session.Run vs IOBinding performance

int main() {
    std::cout << "This is a placeholder - run the Python benchmark instead\n";
    std::cout << "The issue is likely in the KV cache PastInputs() copying\n";
    return 0;
}
