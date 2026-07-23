#include "delayed_dgetrf.hpp"
#include "delayed_dgetrf_column.hpp"
#include "lapack_dgetrf.hpp"

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
#include <string>
#include <vector>

namespace {

using DelayedGetrf = void (*)(int, int, double *, int, int *, int *, int &);

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

double blas_reconstruction_error(const std::vector<double> &original,
                                 const std::vector<double> &factor,
                                 const int n, const int lda,
                                 const std::vector<int> &ipiv,
                                 const std::vector<int> *jpiv,
                                 const int eliminated,
                                 const bool delayed_factorization) {
  // Lbar*Ubar can omit every identically zero Ubar row and its corresponding
  // Lbar column.  This is an exact compression of the inner dimension, not a
  // sparse approximation.  The actual matrix product is still done by dgemm.
  std::vector<int> product_rows;
  for (int k = 0; k < n; ++k) {
    const int first_col =
        delayed_factorization && k >= eliminated ? eliminated : k;
    bool nonzero = false;
    for (int col = first_col; col < n; ++col) {
      if (factor[index(k, col, lda)] != 0.0) {
        nonzero = true;
        break;
      }
    }
    if (nonzero) {
      product_rows.push_back(k);
    }
  }

  const int inner = static_cast<int>(product_rows.size());
  std::vector<double> left(static_cast<std::size_t>(n) * inner, 0.0);
  std::vector<double> right(static_cast<std::size_t>(inner) * n, 0.0);

  for (int compressed_k = 0; compressed_k < inner; ++compressed_k) {
    const int factor_k = product_rows[compressed_k];

    // For delayed rows, Lbar contains the identity block below INFO.
    left[index(factor_k, compressed_k, n)] = 1.0;
    if (!delayed_factorization || factor_k < eliminated) {
      for (int row = factor_k + 1; row < n; ++row) {
        left[index(row, compressed_k, n)] =
            factor[index(row, factor_k, lda)];
      }
    }

    // Rows above INFO contain U; rows below INFO contain the Schur block S.
    const int first_col = delayed_factorization && factor_k >= eliminated
                              ? eliminated
                              : factor_k;
    for (int col = first_col; col < n; ++col) {
      right[index(compressed_k, col, inner)] =
          factor[index(factor_k, col, lda)];
    }
  }

  std::vector<double> reconstructed(
      static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
  if (inner > 0) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, inner, 1.0,
                left.data(), n, right.data(), inner, 0.0,
                reconstructed.data(), n);
  }

  std::vector<int> row_order(static_cast<std::size_t>(n));
  std::vector<int> col_order(static_cast<std::size_t>(n));
  std::iota(row_order.begin(), row_order.end(), 0);
  std::iota(col_order.begin(), col_order.end(), 0);
  const int swap_count = delayed_factorization ? eliminated : n;
  for (int k = 0; k < swap_count; ++k) {
    std::swap(row_order[k], row_order[ipiv[k]]);
    if (jpiv != nullptr) {
      std::swap(col_order[k], col_order[(*jpiv)[k]]);
    }
  }

  double max_error = 0.0;
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      const double expected =
          original[index(row_order[row], col_order[col], lda)];
      max_error = std::max(
          max_error,
          std::abs(reconstructed[index(row, col, n)] - expected));
    }
  }
  return max_error;
}

Result run_delayed(const std::string &name, const DelayedGetrf getrf,
                   const std::vector<double> &original, const int n,
                   const int lda, const double tolerance) {
  std::vector<double> factor = original;
  std::vector<int> ipiv(static_cast<std::size_t>(n));
  std::vector<int> jpiv(static_cast<std::size_t>(n));
  int info = -1;

  const auto factor_begin = std::chrono::steady_clock::now();
  getrf(n, n, factor.data(), lda, ipiv.data(), jpiv.data(), info);
  const auto factor_end = std::chrono::steady_clock::now();

  const auto verification_begin = std::chrono::steady_clock::now();
  const double max_error =
      info >= 0 ? blas_reconstruction_error(original, factor, n, lda, ipiv,
                                            &jpiv, info, true)
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
  std::vector<int> ipiv(static_cast<std::size_t>(n));
  int info = -1;

  const auto factor_begin = std::chrono::steady_clock::now();
  lapack_getrf(n, n, factor.data(), lda, ipiv.data(), info);
  const auto factor_end = std::chrono::steady_clock::now();

  const auto verification_begin = std::chrono::steady_clock::now();
  const double max_error =
      info >= 0 ? blas_reconstruction_error(original, factor, n, lda, ipiv,
                                            nullptr, n, false)
                : std::numeric_limits<double>::infinity();
  const auto verification_end = std::chrono::steady_clock::now();

  return {"LAPACK DGETRF",
          std::chrono::duration<double>(factor_end - factor_begin).count(),
          std::chrono::duration<double>(verification_end - verification_begin)
              .count(),
          max_error,
          info,
          max_error <= tolerance};
}

void print_delayed_result(const Result &result, const int n) {
  std::cout << "\n[" << result.name << "]\n";
  std::cout << "factorization time: " << result.factor_seconds << " s\n";
  std::cout << "verification time: " << result.verification_seconds << " s\n";
  std::cout << "first delayed position (0-based): " << result.info << '\n';
  std::cout << "delayed pivot count: " << n - result.info << '\n';
  std::cout << "verification P^T*Lbar*Ubar*Q^T == A: "
            << (result.verified ? "PASS" : "FAIL") << '\n';
  std::cout << "max absolute error: " << result.max_error << '\n';
}

void print_lapack_result(const Result &result, const int n) {
  std::cout << "\n[" << result.name << "]\n";
  std::cout << "factorization time: " << result.factor_seconds << " s\n";
  std::cout << "verification time: " << result.verification_seconds << " s\n";
  if (result.info == n) {
    std::cout << "first zero U diagonal (0-based): none\n";
  } else {
    std::cout << "first zero U diagonal (0-based): " << result.info << '\n';
  }
  std::cout << "note: LAPACK continues after a singular pivot; this is not a "
               "delayed suffix\n";
  std::cout << "verification P^T*L*U == A: "
            << (result.verified ? "PASS" : "FAIL") << '\n';
  std::cout << "max absolute error: " << result.max_error << '\n';
}

} // namespace

int main() {
  constexpr int n = 4096;
  constexpr int lda = n;
  constexpr int nonzeros = 8;
  constexpr std::size_t matrix_elements =
      static_cast<std::size_t>(lda) * static_cast<std::size_t>(n);

  try {
    std::vector<double> original(matrix_elements, 0.0);
    for (int k = 0; k < nonzeros; ++k) {
      const int position = k + 2;
      original[index(position, position, lda)] =
          static_cast<double>(nonzeros - k);
    }

    const double tolerance =
        100.0 * std::numeric_limits<double>::epsilon() * nonzeros;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "matrix size: " << n << " x " << n << '\n';
    std::cout << "nonzero entries / numerical rank: " << nonzeros << '\n';

    const Result complete =
        run_delayed("global-maximum fallback", delayed_getrf, original, n,
                    lda, tolerance);
    print_delayed_result(complete, n);

    const Result column =
        run_delayed("first-nonzero-column fallback", delayed_getrf_column,
                    original, n, lda, tolerance);
    print_delayed_result(column, n);

    const Result lapack = run_lapack(original, n, lda, tolerance);
    print_lapack_result(lapack, n);

    const bool passed = complete.verified && column.verified &&
                        lapack.verified && complete.info == nonzeros &&
                        column.info == nonzeros && lapack.info == 0;
    std::cout << "\ncomparison result: " << (passed ? "PASS" : "FAIL")
              << '\n';
    return passed ? 0 : 2;
  } catch (const std::bad_alloc &) {
    std::cerr << "not enough heap memory: BLAS reconstruction needs about "
                 "385 MiB for this rank-8 4096x4096 example\n";
    return 3;
  }
}
