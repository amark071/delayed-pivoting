#include "delayed_dgetrf_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using DeviceGetrf = DelayedGetrfCudaStatus (*)(
    int, int, double *, int, int *, int *, int &, cudaStream_t);

std::size_t index(const int row, const int col, const int lda) {
  return static_cast<std::size_t>(row) +
         static_cast<std::size_t>(col) * static_cast<std::size_t>(lda);
}

bool parse_positive(const char *text, int &value) {
  char *end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

bool benchmark(const std::string &name, const DeviceGetrf getrf, const int n,
               const int repeats, const double *d_original, double *d_work,
               const std::size_t bytes, const cudaStream_t stream) {
  std::vector<int> ipiv(static_cast<std::size_t>(n));
  std::vector<int> jpiv(static_cast<std::size_t>(n));
  double best_seconds = std::numeric_limits<double>::infinity();

  for (int repeat = 0; repeat < repeats; ++repeat) {
    if (cudaMemcpyAsync(d_work, d_original, bytes, cudaMemcpyDeviceToDevice,
                        stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      std::cerr << "device copy failed\n";
      return false;
    }

    int info = -1;
    const auto begin = std::chrono::steady_clock::now();
    const DelayedGetrfCudaStatus status =
        getrf(n, n, d_work, n, ipiv.data(), jpiv.data(), info, stream);
    const auto end = std::chrono::steady_clock::now();
    if (status != DelayedGetrfCudaStatus::success || info != n) {
      std::cerr << name << ": " << delayed_getrf_cuda_status_string(status)
                << ", info=" << info << '\n';
      return false;
    }
    best_seconds = std::min(
        best_seconds, std::chrono::duration<double>(end - begin).count());
  }

  const double flops = (2.0 / 3.0) * static_cast<double>(n) * n * n;
  std::cout << std::left << std::setw(26) << name << std::right
            << " best=" << std::setw(10) << best_seconds << " s, "
            << std::setw(10) << flops / best_seconds / 1.0e9 << " GFLOP/s\n";
  return true;
}

} // namespace

int main(int argc, char **argv) {
  int n = 2048;
  int repeats = 3;
  if ((argc > 1 && !parse_positive(argv[1], n)) ||
      (argc > 2 && !parse_positive(argv[2], repeats)) || argc > 3) {
    std::cerr << "usage: delayed_lu_cuda_benchmark [n] [repeats]\n";
    return 2;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "没有可用的 NVIDIA GPU，跳过 benchmark。\n";
    return 77;
  }

  const std::size_t elements = static_cast<std::size_t>(n) * n;
  const std::size_t bytes = elements * sizeof(double);
  std::vector<double> matrix(elements);
  std::mt19937_64 generator(20260807ULL);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  for (double &value : matrix) {
    value = distribution(generator);
  }
  for (int k = 0; k < n; ++k) {
    matrix[index(k, k, n)] += static_cast<double>(n);
  }

  double *d_original = nullptr;
  double *d_work = nullptr;
  cudaStream_t stream = nullptr;
  if (cudaMalloc(&d_original, bytes) != cudaSuccess ||
      cudaMalloc(&d_work, bytes) != cudaSuccess ||
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
      cudaMemcpyAsync(d_original, matrix.data(), bytes, cudaMemcpyHostToDevice,
                      stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    std::cerr << "CUDA initialization failed\n";
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    if (d_work != nullptr) {
      cudaFree(d_work);
    }
    if (d_original != nullptr) {
      cudaFree(d_original);
    }
    return 1;
  }

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "matrix=" << n << "x" << n << ", repeats=" << repeats
            << ", input already resident on device\n";
  bool passed = benchmark("GPU-native", delayed_getrf_cuda_native_device, n,
                          repeats, d_original, d_work, bytes, stream);
  passed &= benchmark("CPU-panel/GPU-update",
                      delayed_getrf_cuda_hybrid_device, n, repeats, d_original,
                      d_work, bytes, stream);

  cudaStreamDestroy(stream);
  cudaFree(d_work);
  cudaFree(d_original);
  return passed ? 0 : 1;
}
