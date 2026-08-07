#include "delayed_dgetrf_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

using HostGetrf = DelayedGetrfCudaStatus (*)(
    int, int, double *, int, int *, int *, int &, cudaStream_t);

std::size_t index(const int row, const int col, const int lda) {
  return static_cast<std::size_t>(row) +
         static_cast<std::size_t>(col) * static_cast<std::size_t>(lda);
}

// 对紧凑存储的 [L | U/Schur] 做一个小规模 CPU 重建，仅用于示例和 CTest。
double reconstruction_error(const std::vector<double> &original,
                            const std::vector<double> &factor, const int m,
                            const int n, const int lda,
                            const std::vector<int> &ipiv,
                            const std::vector<int> &jpiv,
                            const int eliminated) {
  std::vector<double> lower(static_cast<std::size_t>(m) * m, 0.0);
  std::vector<double> upper(static_cast<std::size_t>(m) * n, 0.0);
  for (int row = 0; row < m; ++row) {
    lower[index(row, row, m)] = 1.0;
  }
  for (int k = 0; k < eliminated; ++k) {
    for (int row = k + 1; row < m; ++row) {
      lower[index(row, k, m)] = factor[index(row, k, lda)];
    }
    for (int col = k; col < n; ++col) {
      upper[index(k, col, m)] = factor[index(k, col, lda)];
    }
  }
  // 未消去的右下角 Schur 补等价于 Lbar 的单位块乘 Ubar 的这些行。
  for (int row = eliminated; row < m; ++row) {
    for (int col = eliminated; col < n; ++col) {
      upper[index(row, col, m)] = factor[index(row, col, lda)];
    }
  }

  std::vector<double> product(static_cast<std::size_t>(m) * n, 0.0);
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < m; ++row) {
      double value = 0.0;
      for (int k = 0; k < m; ++k) {
        value += lower[index(row, k, m)] * upper[index(k, col, m)];
      }
      product[index(row, col, m)] = value;
    }
  }

  std::vector<int> row_order(static_cast<std::size_t>(m));
  std::vector<int> col_order(static_cast<std::size_t>(n));
  std::iota(row_order.begin(), row_order.end(), 0);
  std::iota(col_order.begin(), col_order.end(), 0);
  for (int k = 0; k < eliminated; ++k) {
    std::swap(row_order[k], row_order[ipiv[k]]);
    std::swap(col_order[k], col_order[jpiv[k]]);
  }

  double max_error = 0.0;
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < m; ++row) {
      const double expected =
          original[index(row_order[row], col_order[col], lda)];
      max_error =
          std::max(max_error, std::abs(product[index(row, col, m)] - expected));
    }
  }
  return max_error;
}

bool run_case(const std::string &implementation, const HostGetrf getrf,
              const std::string &name, const int m, const int n,
              const std::vector<double> &input, const int expected_info) {
  const int lda = m;
  const int pivots = std::min(m, n);
  std::vector<double> factor = input;
  std::vector<int> ipiv(static_cast<std::size_t>(pivots));
  std::vector<int> jpiv(static_cast<std::size_t>(pivots));
  int info = -1;

  const DelayedGetrfCudaStatus status = getrf(
      m, n, factor.data(), lda, ipiv.data(), jpiv.data(), info, nullptr);
  if (status != DelayedGetrfCudaStatus::success) {
    std::cerr << implementation << " / " << name << ": "
              << delayed_getrf_cuda_status_string(status) << '\n';
    return false;
  }
  const double error = reconstruction_error(input, factor, m, n, lda, ipiv,
                                            jpiv, info);
  double scale = 1.0;
  for (const double value : input) {
    scale = std::max(scale, std::abs(value));
  }
  const double tolerance = 1.0e-11 * scale * std::max(m, n);
  const bool passed = info == expected_info && error <= tolerance;
  std::cout << implementation << " / " << name << ": info=" << info
            << ", max_error=" << error << " -> "
            << (passed ? "PASS" : "FAIL") << '\n';
  return passed;
}

bool run_suite(const std::string &implementation, const HostGetrf getrf) {
  bool passed = true;

  const std::vector<double> dense{
      8.0, 2.0, 1.0, 0.0, // column 0
      1.0, 9.0, 1.0, 1.0, // column 1
      2.0, 1.0, 7.0, 1.0, // column 2
      0.0, 1.0, 2.0, 6.0  // column 3
  };
  passed &= run_case(implementation, getrf, "dense square", 4, 4, dense, 4);

  std::vector<double> delayed(36, 0.0);
  delayed[index(2, 2, 6)] = 7.0;
  delayed[index(4, 4, 6)] = -5.0;
  delayed[index(5, 5, 6)] = 3.0;
  passed &=
      run_case(implementation, getrf, "delayed rank-3", 6, 6, delayed, 3);

  const std::vector<double> rectangular{
      5.0, 1.0, 0.0, 1.0, 0.0, // column 0
      1.0, 6.0, 1.0, 0.0, 1.0, // column 1
      0.0, 1.0, 7.0, 1.0, 0.0  // column 2
  };
  passed &= run_case(implementation, getrf, "rectangular 5x3", 5, 3,
                     rectangular, 3);

  std::vector<double> far_pivot(70 * 70, 0.0);
  far_pivot[index(0, 0, 70)] = 100.0;
  far_pivot[index(69, 69, 70)] = 5.0;
  passed &= run_case(implementation, getrf, "mid-panel far pivot", 70, 70,
                     far_pivot, 2);
  return passed;
}

} // namespace

int main() {
  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    std::cout << "没有可用的 NVIDIA GPU，跳过运行时测试。\n";
    return 77;
  }

  bool passed = run_suite("GPU-native", delayed_getrf_cuda_native_host);
  passed &= run_suite("CPU-panel/GPU-update",
                      delayed_getrf_cuda_hybrid_host);

  return passed ? 0 : 1;
}
