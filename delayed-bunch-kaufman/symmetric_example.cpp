#include "delayed_sytrf.hpp"

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <new>
#include <random>
#include <vector>

namespace {

std::size_t index(const int row, const int col, const int lda) {
  return static_cast<std::size_t>(row) +
         static_cast<std::size_t>(col) * static_cast<std::size_t>(lda);
}

double factorization_residual(const std::vector<double> &original,
                              const std::vector<double> &factor,
                              const std::vector<int> &perm,
                              const std::vector<int> &piv_size, const int lda,
                              const int eliminated) {
  if (eliminated == 0) {
    return 0.0;
  }

  const std::size_t factor_elements =
      static_cast<std::size_t>(eliminated) * eliminated;
  std::vector<double> l(factor_elements, 0.0);
  std::vector<double> d(factor_elements, 0.0);
  for (int i = 0; i < eliminated; ++i) {
    l[index(i, i, eliminated)] = 1.0;
  }

  for (int k = 0; k < eliminated; ++k) {
    if (piv_size[static_cast<std::size_t>(k)] == 1) {
      d[index(k, k, eliminated)] = factor[index(k, k, lda)];
      for (int row = k + 1; row < eliminated; ++row) {
        l[index(row, k, eliminated)] = factor[index(row, k, lda)];
      }
    } else if (piv_size[static_cast<std::size_t>(k)] == 2) {
      d[index(k, k, eliminated)] = factor[index(k, k, lda)];
      d[index(k + 1, k, eliminated)] = factor[index(k + 1, k, lda)];
      d[index(k, k + 1, eliminated)] = factor[index(k + 1, k, lda)];
      d[index(k + 1, k + 1, eliminated)] =
          factor[index(k + 1, k + 1, lda)];
      for (int row = k + 2; row < eliminated; ++row) {
        l[index(row, k, eliminated)] = factor[index(row, k, lda)];
        l[index(row, k + 1, eliminated)] =
            factor[index(row, k + 1, lda)];
      }
      ++k;
    }
  }

  std::vector<double> ld(factor_elements, 0.0);
  std::vector<double> reconstructed(factor_elements, 0.0);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eliminated,
              eliminated, eliminated, 1.0, l.data(), eliminated, d.data(),
              eliminated, 0.0, ld.data(), eliminated);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, eliminated, eliminated,
              eliminated, 1.0, ld.data(), eliminated, l.data(), eliminated,
              0.0, reconstructed.data(), eliminated);

  double max_error = 0.0;
  for (int col = 0; col < eliminated; ++col) {
    for (int row = 0; row < eliminated; ++row) {
      const double expected = original[index(perm[row], perm[col], lda)];
      max_error =
          std::max(max_error,
                   std::abs(expected -
                            reconstructed[index(row, col, eliminated)]));
    }
  }
  return max_error;
}

} // namespace

int main() {
  {
    constexpr int n = 4, lda = n;
    // diag(0,2,0,3): nodes 0 and 2 are delayed without overwriting either.
    std::vector<double> a(static_cast<std::size_t>(n) * n, 0.0);
    a[index(1, 1, lda)] = 2.0;
    a[index(3, 3, lda)] = 3.0;
    std::vector<int> perm(n), piv_size(n);
    int info = -1;
    delayed_sytrf(n, a.data(), lda, perm.data(), piv_size.data(), info);
    if (info < 0)
      return 1;
    if (info != 2 || perm[2] != 2 || perm[3] != 0)
      return 2;
    std::cout << "delayed range: [" << info << ", " << n << ")\n";
  }

  {
    constexpr int n = 2, lda = n;
    const std::vector<double> original = {0.0, 1.0, 1.0, 0.0};
    std::vector<double> factor = original;
    std::vector<int> perm(n), piv_size(n);
    int info = -1;
    delayed_sytrf(n, factor.data(), lda, perm.data(), piv_size.data(), info);
    if (info < 0)
      return 3;
    if (info != 2 || piv_size[0] != 2 || piv_size[1] != 0)
      return 4;
    const double error =
        factorization_residual(original, factor, perm, piv_size, lda, info);
    std::cout << "max |P^T*A*P-L*D*L^T|: " << error << '\n';
    if (error > 1e-12)
      return 5;
  }

  {
    constexpr int n = 3, lda = n;
    const std::vector<double> original = {0.0, 2.0, 0.5, 2.0, 0.0,
                                          1.0, 0.5, 1.0, 3.0};
    std::vector<double> factor = original;
    std::vector<int> perm(n), piv_size(n);
    int info = -1;
    delayed_sytrf(n, factor.data(), lda, perm.data(), piv_size.data(), info);
    if (info < 0)
      return 6;
    if (info != n || piv_size[0] != 2 || piv_size[1] != 0)
      return 7;
    const double error =
        factorization_residual(original, factor, perm, piv_size, lda, info);
    std::cout << "2x2-update residual: " << error << '\n';
    if (error > 1e-12)
      return 8;
  }

  {
    constexpr int n = 3, lda = n;
    const std::vector<double> original = {4.0, 1.0, 2.0, 1.0, 5.0,
                                          0.5, 2.0, 0.5, 6.0};
    std::vector<double> factor = original;
    std::vector<int> perm(n), piv_size(n);
    int info = -1;
    delayed_sytrf(n, factor.data(), lda, perm.data(), piv_size.data(), info);
    if (info < 0)
      return 9;
    if (info != n)
      return 10;
    const double error =
        factorization_residual(original, factor, perm, piv_size, lda, info);
    std::cout << "1x1-pivot residual: " << error << '\n';
    if (error > 1e-12)
      return 11;
  }

  {
    constexpr int n = 6, lda = n;
    std::mt19937_64 generator(20260721);
    std::uniform_real_distribution<double> distribution(-2.0, 2.0);
    bool observed_2x2 = false;
    for (int trial = 0; trial < 100; ++trial) {
      std::vector<double> original(static_cast<std::size_t>(n) * n, 0.0);
      for (int col = 0; col < n; ++col) {
        for (int row = col; row < n; ++row) {
          double value = distribution(generator);
          if (row == col)
            value *= 0.1;
          original[index(row, col, lda)] = value;
          original[index(col, row, lda)] = value;
        }
      }
      std::vector<double> factor = original;
      std::vector<int> perm(n), piv_size(n);
      int info = -1;
      delayed_sytrf(n, factor.data(), lda, perm.data(), piv_size.data(), info);
      if (info < 0)
        return 12;
      if (info != n)
        return 13;
      observed_2x2 = observed_2x2 ||
                     std::find(piv_size.begin(), piv_size.end(), 2) !=
                         piv_size.end();
      const double error =
          factorization_residual(original, factor, perm, piv_size, lda, info);
      if (error > 1e-10)
        return 14;
    }
    if (!observed_2x2)
      return 15;
    std::cout << "100 deterministic symmetric residual tests passed\n";
  }

  {
    constexpr int n = 1200, lda = n;
    const std::size_t matrix_elements =
        static_cast<std::size_t>(lda) * static_cast<std::size_t>(n);
    try {
      std::vector<double> a(matrix_elements, 0.0);
      std::vector<int> perm(n), piv_size(n);
      a[0] = 1.0;
      int info = -1;
      delayed_sytrf(n, a.data(), lda, perm.data(), piv_size.data(), info);
      if (info != 1)
        return 16;
      std::cout << "1200x1200 heap-backed smoke test passed\n";
    } catch (const std::bad_alloc &) {
      std::cerr << "not enough heap memory for the 1200x1200 smoke test\n";
      return 17;
    }
  }
  return 0;
}
