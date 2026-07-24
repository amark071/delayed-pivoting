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
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace {

// 三个实现都使用同一接口，因此可以通过函数指针共用计时与验证代码。
using Factorization =
    void (*)(int, double *, int, int *, int *, int &);

struct Result {
  std::string name;
  double factor_seconds;
  double verification_seconds;
  double max_error;
  int info;
  bool verified;
};

std::size_t index(const int row, const int col, const int lda) {
  return static_cast<std::size_t>(row) +
         static_cast<std::size_t>(col) * static_cast<std::size_t>(lda);
}

// 因子只保证下三角有效；读取逻辑对称矩阵时统一映射到下三角。
double symmetric_lower_value(const std::vector<double> &a, const int lda,
                             const int row, const int col) {
  return row >= col ? a[index(row, col, lda)] : a[index(col, row, lda)];
}

// 判断 D 的某一行/列是否非零。只保留非零的 D 行，可以避免为数值秩很低
// 的 4096 阶算例额外建立两个完整的 4096x4096 中间矩阵。
bool d_row_is_nonzero(const std::vector<double> &factor,
                      const std::vector<int> &piv_size, const int n,
                      const int lda, const int factor_limit,
                      const bool has_delayed_schur, const int row) {
  if (has_delayed_schur && row >= factor_limit) {
    for (int col = factor_limit; col < n; ++col) {
      if (symmetric_lower_value(factor, lda, row, col) != 0.0) {
        return true;
      }
    }
    return false;
  }

  if (row >= factor_limit) {
    return false;
  }
  if (factor[index(row, row, lda)] != 0.0) {
    return true;
  }
  if (piv_size[row] == 2) {
    return factor[index(row + 1, row, lda)] != 0.0;
  }
  return row > 0 && piv_size[row - 1] == 2 &&
         factor[index(row, row - 1, lda)] != 0.0;
}

// 从紧凑因子存储中读取 D(row,col)。延迟算法的右下角保留未消去的
// Schur 补，验证时把它看作 D 的最后一个对称块。
double d_entry(const std::vector<double> &factor,
               const std::vector<int> &piv_size, const int lda,
               const int factor_limit, const bool has_delayed_schur,
               const int row, const int col) {
  if (has_delayed_schur && row >= factor_limit && col >= factor_limit) {
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
  return second == first + 1 && piv_size[first] == 2
             ? factor[index(second, first, lda)]
             : 0.0;
}

// 1x1 主元的乘子从下一行开始；2x2 主元的两列乘子都从该块之后开始。
int first_multiplier_row(const std::vector<int> &piv_size, const int col) {
  if (piv_size[col] == 2) {
    return col + 2;
  }
  if (col > 0 && piv_size[col - 1] == 2) {
    return col + 1;
  }
  return col + 1;
}

// 用两次 BLAS dgemm 计算 L*D 和 (L*D)*L^T，然后与 P^T*A*P 比较。
double blas_factorization_error(const std::vector<double> &original,
                                const std::vector<double> &factor,
                                const std::vector<int> &perm,
                                const std::vector<int> &piv_size, const int n,
                                const int lda, const int factor_limit,
                                const bool has_delayed_schur) {
  std::vector<int> active_d_rows;
  active_d_rows.reserve(static_cast<std::size_t>(n));
  for (int row = 0; row < n; ++row) {
    if (d_row_is_nonzero(factor, piv_size, n, lda, factor_limit,
                         has_delayed_schur, row)) {
      active_d_rows.push_back(row);
    }
  }

  const int inner = static_cast<int>(active_d_rows.size());
  std::vector<double> l(static_cast<std::size_t>(n) * inner, 0.0);
  std::vector<double> d(static_cast<std::size_t>(inner) * inner, 0.0);

  for (int compressed_col = 0; compressed_col < inner; ++compressed_col) {
    const int factor_col = active_d_rows[compressed_col];
    l[index(factor_col, compressed_col, n)] = 1.0;
    if (factor_col < factor_limit) {
      for (int row = first_multiplier_row(piv_size, factor_col); row < n;
           ++row) {
        l[index(row, compressed_col, n)] =
            factor[index(row, factor_col, lda)];
      }
    }

    for (int compressed_row = 0; compressed_row < inner; ++compressed_row) {
      const int factor_row = active_d_rows[compressed_row];
      d[index(compressed_row, compressed_col, inner)] =
          d_entry(factor, piv_size, lda, factor_limit, has_delayed_schur,
                  factor_row, factor_col);
    }
  }

  std::vector<double> ld(static_cast<std::size_t>(n) * inner, 0.0);
  std::vector<double> reconstructed(
      static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
  if (inner > 0) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, inner, inner,
                1.0, l.data(), n, d.data(), inner, 0.0, ld.data(), n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, inner, 1.0,
                ld.data(), n, l.data(), n, 0.0, reconstructed.data(), n);
  }

  double max_error = 0.0;
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      const double expected = original[index(perm[row], perm[col], lda)];
      max_error =
          std::max(max_error,
                   std::abs(reconstructed[index(row, col, n)] - expected));
    }
  }
  return max_error;
}

Result run_delayed(const std::string &name, const Factorization factorization,
                   const std::vector<double> &original, const int n,
                   const int lda, const double tolerance) {
  std::vector<double> factor = original;
  std::vector<int> perm(static_cast<std::size_t>(n));
  std::vector<int> piv_size(static_cast<std::size_t>(n));
  int info = -1;

  const auto factor_begin = std::chrono::steady_clock::now();
  factorization(n, factor.data(), lda, perm.data(), piv_size.data(), info);
  const auto factor_end = std::chrono::steady_clock::now();

  const auto verification_begin = std::chrono::steady_clock::now();
  const double max_error =
      info >= 0 ? blas_factorization_error(original, factor, perm, piv_size, n,
                                           lda, info, true)
                : std::numeric_limits<double>::infinity();
  const auto verification_end = std::chrono::steady_clock::now();

  return {name,
          std::chrono::duration<double>(factor_end - factor_begin).count(),
          std::chrono::duration<double>(verification_end - verification_begin)
              .count(),
          max_error,
          info,
          max_error <= tolerance};
}

Result run_lapack(const std::vector<double> &original, const int n,
                  const int lda, const double tolerance) {
  std::vector<double> factor = original;
  std::vector<int> perm(static_cast<std::size_t>(n));
  std::vector<int> piv_size(static_cast<std::size_t>(n));
  int info = -1;

  const auto factor_begin = std::chrono::steady_clock::now();
  lapack_sytrf(n, factor.data(), lda, perm.data(), piv_size.data(), info);
  const auto factor_end = std::chrono::steady_clock::now();

  const auto verification_begin = std::chrono::steady_clock::now();
  const double max_error =
      info >= 0 ? blas_factorization_error(original, factor, perm, piv_size, n,
                                           lda, n, false)
                : std::numeric_limits<double>::infinity();
  const auto verification_end = std::chrono::steady_clock::now();

  return {"LAPACK DSYTRF",
          std::chrono::duration<double>(factor_end - factor_begin).count(),
          std::chrono::duration<double>(verification_end - verification_begin)
              .count(),
          max_error,
          info,
          max_error <= tolerance};
}

void print_result(const Result &result, const int n,
                  const bool delayed_semantics) {
  std::cout << "\n[" << result.name << "]\n";
  std::cout << "分解耗时：" << result.factor_seconds << " 秒\n";
  std::cout << "验证耗时：" << result.verification_seconds << " 秒\n";
  if (delayed_semantics) {
    std::cout << "延迟起点（0-based）：" << result.info << '\n';
    std::cout << "延迟节点数：" << n - result.info << '\n';
  } else if (result.info == n) {
    std::cout << "首个奇异 D 块位置：无\n";
  } else {
    std::cout << "首个奇异 D 块位置（0-based）：" << result.info << '\n';
    std::cout << "说明：LAPACK 会继续分解，该位置不表示延迟后缀\n";
  }
  std::cout << "BLAS 重建验证："
            << (result.verified ? "通过" : "失败") << '\n';
  std::cout << "最大绝对误差：" << result.max_error << '\n';
}

} // namespace

int main() {
  constexpr int n = 4096;
  constexpr int lda = n;
  constexpr int rank = 8;
  constexpr std::size_t elements =
      static_cast<std::size_t>(lda) * static_cast<std::size_t>(n);

  try {
    // 三个实现共用这一份原始矩阵：
    // 位置 2、3 构成一个可接受的 2x2 不定主元块，位置 4 至 9 是六个
    // 1x1 主元，其余节点全零并会被延迟。
    std::vector<double> original(elements, 0.0);
    original[index(3, 2, lda)] = 2.0;
    original[index(2, 3, lda)] = 2.0;
    for (int k = 0; k < rank - 2; ++k) {
      const int position = k + 4;
      original[index(position, position, lda)] =
          static_cast<double>(rank - 2 - k);
    }

    const double tolerance =
        100.0 * std::numeric_limits<double>::epsilon() * rank;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "矩阵大小：" << n << " x " << n << '\n';
    std::cout << "数值秩：" << rank
              << "（六个 1x1 主元和一个 2x2 主元）\n";

    const Result unblocked =
        run_delayed("非分块延迟 Bunch-Kaufman", delayed_sytrf, original, n,
                    lda, tolerance);
    print_result(unblocked, n, true);

    const Result blocked =
        run_delayed("分块延迟 Bunch-Kaufman", delayed_sytrf_blocked, original,
                    n, lda, tolerance);
    print_result(blocked, n, true);

    const Result lapack = run_lapack(original, n, lda, tolerance);
    print_result(lapack, n, false);

    if (blocked.factor_seconds > 0.0) {
      std::cout << "\n非分块/分块分解耗时比："
                << unblocked.factor_seconds / blocked.factor_seconds << '\n';
    }
    if (lapack.factor_seconds > 0.0) {
      std::cout << "分块延迟/LAPACK 分解耗时比："
                << blocked.factor_seconds / lapack.factor_seconds << '\n';
    }

    const bool passed =
        unblocked.verified && blocked.verified && lapack.verified &&
        unblocked.info == rank && blocked.info == rank && lapack.info == 0;
    std::cout << "\n总体比较结果：" << (passed ? "通过" : "失败") << '\n';
    return passed ? 0 : 2;
  } catch (const std::bad_alloc &) {
    std::cerr << "堆内存不足，无法完成 4096x4096 算例\n";
    return 3;
  }
}
