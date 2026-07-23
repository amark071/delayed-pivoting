#include "delayed_sytrf.hpp"
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
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

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

double symmetric_lower_value(const std::vector<double> &a, const int lda,
                             const int row, const int col) {
  return row >= col ? a[index(row, col, lda)] : a[index(col, row, lda)];
}

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

int first_multiplier_row(const std::vector<int> &piv_size, const int col) {
  if (piv_size[col] == 2) {
    return col + 2;
  }
  if (col > 0 && piv_size[col - 1] == 2) {
    return col + 1;
  }
  return col + 1;
}

double blas_factorization_error(const std::vector<double> &original,
                                const std::vector<double> &factor,
                                const std::vector<int> &perm,
                                const std::vector<int> &piv_size, const int n,
                                const int lda, const int factor_limit,
                                const bool has_delayed_schur) {
  std::vector<int> active_d_rows;
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

Result run_delayed(const std::vector<double> &original, const int n,
                   const int lda, const double tolerance) {
  std::vector<double> factor = original;
  std::vector<int> perm(n), piv_size(n);
  int info = -1;

  const auto factor_begin = std::chrono::steady_clock::now();
  delayed_sytrf(n, factor.data(), lda, perm.data(), piv_size.data(), info);
  const auto factor_end = std::chrono::steady_clock::now();

  const auto verification_begin = std::chrono::steady_clock::now();
  const double max_error =
      info >= 0 ? blas_factorization_error(original, factor, perm, piv_size, n,
                                           lda, info, true)
                : std::numeric_limits<double>::infinity();
  const auto verification_end = std::chrono::steady_clock::now();

  return {"delayed Bunch-Kaufman",
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
  std::vector<int> perm(n), piv_size(n);
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
  std::cout << "factorization time: " << result.factor_seconds << " s\n";
  std::cout << "verification time: " << result.verification_seconds << " s\n";
  if (delayed_semantics) {
    std::cout << "first delayed position (0-based): " << result.info << '\n';
    std::cout << "delayed node count: " << n - result.info << '\n';
  } else if (result.info == n) {
    std::cout << "first singular D position (0-based): none\n";
  } else {
    std::cout << "first singular D position (0-based): " << result.info
              << '\n';
    std::cout << "note: LAPACK continues; this is not a delayed suffix\n";
  }
  std::cout << "BLAS verification: "
            << (result.verified ? "PASS" : "FAIL") << '\n';
  std::cout << "max absolute error: " << result.max_error << '\n';
}

} // namespace

int main() {
  constexpr int n = 4096;
  constexpr int lda = n;
  constexpr int rank = 8;
  constexpr std::size_t elements =
      static_cast<std::size_t>(lda) * static_cast<std::size_t>(n);

  try {
    // Exercise LAPACK's nontrivial interchange and 2x2 IPIV encodings before
    // running the large timing comparison.
    {
      constexpr int test_n = 8;
      std::mt19937_64 generator(20260722);
      std::uniform_real_distribution<double> distribution(-2.0, 2.0);
      for (int trial = 0; trial < 50; ++trial) {
        std::vector<double> test_original(
            static_cast<std::size_t>(test_n) * test_n, 0.0);
        for (int col = 0; col < test_n; ++col) {
          for (int row = col; row < test_n; ++row) {
            double value = distribution(generator);
            if (row == col) {
              value *= 0.1;
            }
            test_original[index(row, col, test_n)] = value;
            test_original[index(col, row, test_n)] = value;
          }
        }

        std::vector<double> test_factor = test_original;
        std::vector<int> test_perm(test_n), test_piv_size(test_n);
        int test_info = -1;
        lapack_sytrf(test_n, test_factor.data(), test_n, test_perm.data(),
                     test_piv_size.data(), test_info);
        const double test_error = blas_factorization_error(
            test_original, test_factor, test_perm, test_piv_size, test_n,
            test_n, test_n, false);
        if (test_info != test_n || test_error > 1e-10) {
          std::cerr << "LAPACK pivot conversion self-test failed: "
                    << test_error << '\n';
          return 1;
        }
      }
    }

    std::vector<double> original(elements, 0.0);
    // One indefinite 2x2 block at positions 2:3 plus six 1x1 pivots.
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
    std::cout << "matrix size: " << n << " x " << n << '\n';
    std::cout << "numerical rank: " << rank
              << " (six 1x1 pivots plus one 2x2 pivot)\n";

    const Result delayed = run_delayed(original, n, lda, tolerance);
    print_result(delayed, n, true);

    const Result lapack = run_lapack(original, n, lda, tolerance);
    print_result(lapack, n, false);

    const bool passed = delayed.verified && lapack.verified &&
                        delayed.info == rank && lapack.info == 0;
    std::cout << "\ncomparison result: " << (passed ? "PASS" : "FAIL")
              << '\n';
    return passed ? 0 : 2;
  } catch (const std::bad_alloc &) {
    std::cerr << "not enough heap memory for the 4096x4096 comparison\n";
    return 3;
  }
}
