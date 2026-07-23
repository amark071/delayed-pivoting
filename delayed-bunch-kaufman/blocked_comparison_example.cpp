#include "delayed_sytrf.hpp"
#include "delayed_sytrf_blocked.hpp"
#include "lapack_dsytrf.hpp"

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

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

using Factorization =
    void (*)(int, double *, int, int *, int *, int &);

struct TimingResult {
  std::string name;
  double seconds;
  int info;
};

std::size_t index(const int row, const int col, const int lda) {
  return static_cast<std::size_t>(row) +
         static_cast<std::size_t>(col) * static_cast<std::size_t>(lda);
}

double symmetric_lower_value(const std::vector<double> &a, const int lda,
                             const int row, const int col) {
  return row >= col ? a[index(row, col, lda)] : a[index(col, row, lda)];
}

// 以下辅助函数把原地存储的 L、1x1/2x2 D，以及延迟后缀中的 Schur 补
// 展开为稠密矩阵，再通过两次 dgemm 验证
//
//   P^T*A_original*P = L*Dbar*L^T。
//
// Dbar 同时包含已接受的块对角 D 和延迟后缀 Schur 补，因此该验证既适用于
// info==n 的完整分解，也适用于 info<n 的部分分解。
bool d_row_is_nonzero(const std::vector<double> &factor,
                      const std::vector<int> &piv_size, const int n,
                      const int lda, const int factor_limit, const int row) {
  if (row >= factor_limit) {
    for (int col = factor_limit; col < n; ++col) {
      if (symmetric_lower_value(factor, lda, row, col) != 0.0) {
        return true;
      }
    }
    return false;
  }

  if (factor[index(row, row, lda)] != 0.0) {
    return true;
  }
  if (piv_size[static_cast<std::size_t>(row)] == 2) {
    return factor[index(row + 1, row, lda)] != 0.0;
  }
  return row > 0 && piv_size[static_cast<std::size_t>(row - 1)] == 2 &&
         factor[index(row, row - 1, lda)] != 0.0;
}

double d_entry(const std::vector<double> &factor,
               const std::vector<int> &piv_size, const int lda,
               const int factor_limit, const int row, const int col) {
  if (row >= factor_limit && col >= factor_limit) {
    return symmetric_lower_value(factor, lda, row, col);
  }
  if (row >= factor_limit || col >= factor_limit) {
    return 0.0;
  }
  if (row == col) {
    return factor[index(row, row, lda)];
  }

  const int first = std::min(row, col);
  const int second = std::max(row, col);
  return second == first + 1 &&
                 piv_size[static_cast<std::size_t>(first)] == 2
             ? factor[index(second, first, lda)]
             : 0.0;
}

int first_multiplier_row(const std::vector<int> &piv_size, const int col) {
  if (piv_size[static_cast<std::size_t>(col)] == 2) {
    return col + 2;
  }
  if (col > 0 && piv_size[static_cast<std::size_t>(col - 1)] == 2) {
    return col + 1;
  }
  return col + 1;
}

double factorization_error(const std::vector<double> &original,
                           const std::vector<double> &factor,
                           const std::vector<int> &perm,
                           const std::vector<int> &piv_size, const int n,
                           const int lda, const int factor_limit) {
  std::vector<int> active_d_rows;
  for (int row = 0; row < n; ++row) {
    if (d_row_is_nonzero(factor, piv_size, n, lda, factor_limit, row)) {
      active_d_rows.push_back(row);
    }
  }

  const int inner = static_cast<int>(active_d_rows.size());
  std::vector<double> l(static_cast<std::size_t>(n) * inner, 0.0);
  std::vector<double> d(static_cast<std::size_t>(inner) * inner, 0.0);

  for (int compressed_col = 0; compressed_col < inner; ++compressed_col) {
    const int factor_col =
        active_d_rows[static_cast<std::size_t>(compressed_col)];
    l[index(factor_col, compressed_col, n)] = 1.0;
    if (factor_col < factor_limit) {
      for (int row = first_multiplier_row(piv_size, factor_col); row < n;
           ++row) {
        l[index(row, compressed_col, n)] =
            factor[index(row, factor_col, lda)];
      }
    }

    for (int compressed_row = 0; compressed_row < inner; ++compressed_row) {
      const int factor_row =
          active_d_rows[static_cast<std::size_t>(compressed_row)];
      d[index(compressed_row, compressed_col, inner)] =
          d_entry(factor, piv_size, lda, factor_limit, factor_row, factor_col);
    }
  }

  std::vector<double> ld(static_cast<std::size_t>(n) * inner, 0.0);
  std::vector<double> reconstructed(static_cast<std::size_t>(n) * n, 0.0);
  if (inner > 0) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, inner, inner,
                1.0, l.data(), n, d.data(), inner, 0.0, ld.data(), n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, inner, 1.0,
                ld.data(), n, l.data(), n, 0.0, reconstructed.data(), n);
  }

  double max_error = 0.0;
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      const double expected =
          original[index(perm[static_cast<std::size_t>(row)],
                         perm[static_cast<std::size_t>(col)], lda)];
      max_error =
          std::max(max_error,
                   std::abs(reconstructed[index(row, col, n)] - expected));
    }
  }
  return max_error;
}

// 构造一个稠密、非定且包含稳定 1x1/2x2 主元混合的矩阵。
// 小的非对角背景值保证数据确实是稠密的；每隔 16 个位置放置一个
// [0 2;2 0] 块，其余对角元取 +/-4，从而同时覆盖两种主元路径。
std::vector<double> make_mixed_matrix(const int n) {
  std::vector<double> a(static_cast<std::size_t>(n) * n, 0.0);
  for (int col = 0; col < n; ++col) {
    for (int row = col + 1; row < n; ++row) {
      const std::size_t key =
          (static_cast<std::size_t>(row + 1) * 11400714819323198485ull) ^
          (static_cast<std::size_t>(col + 3) * 14029467366897019727ull);
      const double value =
          static_cast<double>(static_cast<int>(key % 2001u) - 1000) * 1.0e-6;
      a[index(row, col, n)] = value;
      a[index(col, row, n)] = value;
    }
  }

  for (int k = 0; k < n;) {
    if (k % 16 == 0 && k + 1 < n) {
      a[index(k, k, n)] = 0.0;
      a[index(k + 1, k + 1, n)] = 0.0;
      a[index(k + 1, k, n)] = 2.0;
      a[index(k, k + 1, n)] = 2.0;
      k += 2;
    } else {
      a[index(k, k, n)] = k % 2 == 0 ? 4.0 : -4.0;
      ++k;
    }
  }
  return a;
}

bool check_one(const std::vector<double> &original, const int n,
               const double tolerance, const int expected_info) {
  for (const auto algorithm :
       {Factorization(delayed_sytrf), Factorization(delayed_sytrf_blocked)}) {
    std::vector<double> factor = original;
    std::vector<int> perm(static_cast<std::size_t>(n));
    std::vector<int> piv_size(static_cast<std::size_t>(n));
    int info = -1;
    algorithm(n, factor.data(), n, perm.data(), piv_size.data(), info);
    if (info != expected_info) {
      std::cerr << "unexpected info: got " << info << ", expected "
                << expected_info << '\n';
      return false;
    }
    const double error = factorization_error(original, factor, perm, piv_size,
                                             n, n, info);
    if (!(error <= tolerance)) {
      std::cerr << "factorization residual failed: " << error << '\n';
      return false;
    }
  }
  return true;
}

bool run_correctness_tests() {
  // 随机小矩阵用于高覆盖率检查主元交换和混合 1x1/2x2 路径。
  constexpr int random_n = 12;
  std::mt19937_64 generator(20260722);
  std::uniform_real_distribution<double> distribution(-2.0, 2.0);
  for (int trial = 0; trial < 120; ++trial) {
    std::vector<double> a(static_cast<std::size_t>(random_n) * random_n, 0.0);
    for (int col = 0; col < random_n; ++col) {
      for (int row = col; row < random_n; ++row) {
        double value = distribution(generator);
        if (row == col) {
          value *= 0.1;
        }
        a[index(row, col, random_n)] = value;
        a[index(col, row, random_n)] = value;
      }
    }
    if (!check_one(a, random_n, 1.0e-9, random_n)) {
      std::cerr << "random correctness trial " << trial << " failed\n";
      return false;
    }
  }

  // n>64 强制产生多个面板；混合矩阵还会覆盖 2x2 主元恰好靠近面板边界。
  constexpr int cross_panel_n = 160;
  const std::vector<double> cross_panel = make_mixed_matrix(cross_panel_n);
  if (!check_one(cross_panel, cross_panel_n, 1.0e-9, cross_panel_n)) {
    std::cerr << "cross-panel mixed-pivot test failed\n";
    return false;
  }

  // 在面板中部制造两个完全为零的节点。它们会被移动到延迟后缀，而被换入
  // 的活动节点在 W 中已有累计贡献，因此可以检查 A/perm/W 同步交换是否正确。
  constexpr int delayed_n = 96;
  std::vector<double> delayed = make_mixed_matrix(delayed_n);
  for (const int zero_node : {5, 37}) {
    for (int j = 0; j < delayed_n; ++j) {
      delayed[index(zero_node, j, delayed_n)] = 0.0;
      delayed[index(j, zero_node, delayed_n)] = 0.0;
    }
  }
  if (!check_one(delayed, delayed_n, 1.0e-9, delayed_n - 2)) {
    std::cerr << "delayed-suffix panel test failed\n";
    return false;
  }

  return true;
}

TimingResult time_factorization(const std::string &name,
                                const Factorization algorithm,
                                const std::vector<double> &original,
                                const int n, const int repeats) {
  double best = std::numeric_limits<double>::infinity();
  int observed_info = -1;
  for (int repeat = 0; repeat < repeats; ++repeat) {
    std::vector<double> factor = original;
    std::vector<int> perm(static_cast<std::size_t>(n));
    std::vector<int> piv_size(static_cast<std::size_t>(n));
    int info = -1;
    const auto begin = std::chrono::steady_clock::now();
    algorithm(n, factor.data(), n, perm.data(), piv_size.data(), info);
    const auto end = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double>(end - begin).count());
    observed_info = info;
  }
  return {name, best, observed_info};
}

} // namespace

int main(const int argc, char **argv) {
  int benchmark_n = 2048;
  if (argc > 1) {
    benchmark_n = std::atoi(argv[1]);
  }
  if (benchmark_n < 2) {
    std::cerr << "benchmark size must be at least 2\n";
    return 1;
  }

  if (!run_correctness_tests()) {
    return 2;
  }
  std::cout << "correctness: PASS (120 random, cross-panel, delayed suffix)\n";

  // 小矩阵预热用于触发 BLAS 运行库初始化，避免初始化时间只落在第一个算法。
  const std::vector<double> warmup = make_mixed_matrix(128);
  (void)time_factorization("warmup", delayed_sytrf_blocked, warmup, 128, 1);

  const std::vector<double> original = make_mixed_matrix(benchmark_n);
  const int repeats = benchmark_n <= 1024 ? 3 : 2;
  const TimingResult unblocked =
      time_factorization("unblocked delayed BK", delayed_sytrf, original,
                         benchmark_n, repeats);
  const TimingResult blocked =
      time_factorization("blocked delayed BK", delayed_sytrf_blocked, original,
                         benchmark_n, repeats);
  const TimingResult lapack =
      time_factorization("LAPACK DSYTRF", lapack_sytrf, original, benchmark_n,
                         repeats);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "\nperformance matrix: " << benchmark_n << " x " << benchmark_n
            << " (dense mixed 1x1/2x2 pivots)\n";
  for (const TimingResult *result : {&unblocked, &blocked, &lapack}) {
    std::cout << result->name << ": " << result->seconds
              << " s, info=" << result->info << '\n';
  }
  std::cout << "blocked speedup over unblocked: "
            << unblocked.seconds / blocked.seconds << "x\n";
  std::cout << "blocked / LAPACK time ratio: "
            << blocked.seconds / lapack.seconds << "x\n";

  const bool passed = unblocked.info == benchmark_n &&
                      blocked.info == benchmark_n && lapack.info == benchmark_n;
  std::cout << "performance-case pivot completion: "
            << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 3;
}
