#include "delayed_sytrf_blocked.hpp"

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

// 与非分块实现使用相同的 Bunch--Kaufman 判据和 2x2 稳定性容差。
constexpr double kBunchKaufmanAlpha =
    (1.0 + 4.1231056256176605498) / 8.0; // (1 + sqrt(17)) / 8
constexpr double kDeterminantTolerance =
    64.0 * std::numeric_limits<double>::epsilon();

constexpr int kBlockSize = 64;

// 对称矩阵只用访问下三角
double symmetric_lower_value(const double *a, const int lda, const int row,
                             const int col) {
  return row >= col ? a[row + col * lda] : a[col + row * lda];
}

// 只访问下三角，完成对称 x/y 交换。
void symmetric_swap_lower(const int n, double *a, const int lda, int x, int y,
                          int *perm) {
  if (x == y) {
    return;
  }
  if (x > y) {
    std::swap(x, y);
  }

  if (x > 0) {
    cblas_dswap(x, a + x, lda, a + y, lda);
  }

  const int middle = y - x - 1;
  if (middle > 0) {
    cblas_dswap(middle, a + (x + 1) + x * lda, 1,
                a + y + (x + 1) * lda, lda);
  }
  std::swap(a[x + x * lda], a[y + y * lda]);

  const int tail = n - y - 1;
  if (tail > 0) {
    cblas_dswap(tail, a + (y + 1) + x * lda, 1,
                a + (y + 1) + y * lda, 1);
  }
  std::swap(perm[x], perm[y]);
}

// 面板内的尾部 Schur 补尚未写回 A，而由
//
//   S = A_stored - L_panel * D_panel * L_panel^T
//     = A_stored - L_panel * W_panel^T,
//   W_panel = L_panel * D_panel
//
// 隐式表示。
// 若交换两个活动节点，不仅要对称交换 A 和 perm，还必须交换L_panel 与 W_panel 中对应的两行。
// A 中已有的 L 行由 symmetric_swap_lower 一并交换；这里额外交换独立工作矩阵 W 的两行。
void swap_active_nodes(const int n, double *a, const int lda, const int x,
                       const int y, int *perm, double *work, const int ldw,
                       const int panel_width) {
  if (x == y) {
    return;
  }
  symmetric_swap_lower(n, a, lda, x, y, perm);
  if (panel_width > 0) {
    cblas_dswap(panel_width, work + x, ldw, work + y, ldw);
  }
}

// 更新当前 Schur 补的第 col 列，从对角元一直计算到第 n-1 行。
//
// A(col:n-1,col) 是面板开始时的基准值，尚未减去已接受面板列的贡献。
// dgemv 一次完成：
//
//   column <- column - L(col:n-1,:) * W(col,:)^T.
void materialize_current_column(const int n, const double *a, const int lda,
                                const int panel_begin,
                                const int panel_width, const double *work,
                                const int ldw, const int col,
                                double *column) {
  for (int row = col; row < n; ++row) {
    column[row] = a[row + col * lda];
  }
  if (panel_width > 0) {
    cblas_dgemv(CblasColMajor, CblasNoTrans, n - col, panel_width, -1.0,
                a + col + panel_begin * lda, lda, work + col, ldw, 1.0,
                column + col, 1);
  }
}

// 更新活动区间 [active_begin,active_end) 中指定 row 的逻辑对称行。
// 基准值通过下三角对称读取；随后一次 dgemv 减去面板累计贡献：
//
//   row_values <- row_values - L(active_begin:active_end,:)*W(row,:)^T.
//
// Bunch--Kaufman 的候选行最大值必须在当前 Schur 补上计算，不能直接扫描尚未写回的 A_stored。
void materialize_active_row(const double *a, const int lda,
                            const int panel_begin, const int panel_width,
                            const double *work, const int ldw,
                            const int active_begin, const int active_end,
                            const int row, double *row_values) {
  for (int col = active_begin; col < active_end; ++col) {
    row_values[col] = symmetric_lower_value(a, lda, row, col);
  }
  if (panel_width > 0) {
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                active_end - active_begin, panel_width, -1.0,
                a + active_begin + panel_begin * lda, lda, work + row, ldw,
                1.0, row_values + active_begin, 1);
  }
}

// 判断一个2x2的矩阵是否稳定
bool stable_2x2(const double a, const double b, const double c) {
  const double scale = std::max({std::abs(a), std::abs(b), std::abs(c)});
  if (!(scale > 0.0)) {
    return false;
  }

  const double as = a / scale;
  const double bs = b / scale;
  const double cs = c / scale;
  const double determinant_scaled = as * cs - bs * bs;
  const double norm_scaled =
      std::max(std::abs(as) + std::abs(bs), std::abs(bs) + std::abs(cs));
  return std::abs(determinant_scaled) >
         kDeterminantTolerance * norm_scaled * norm_scaled;
}

// 将一个完整面板 [begin,end) 的累计贡献一次性写回尾部下三角。
// 令 W21=L21*D11，则目标更新为：
//
//    A22 <- A22 - L21*D11*L21^T = A22 - W21*L21^T.
//
// 由于 D11 对称，W21*L21^T 与 L21*W21^T 数学上相同，所以可以利用 dsyr2k 只更新下三角：
//
//    A22 <- A22 - 0.5*(W21*L21^T + L21*W21^T).
//
// 更新范围一直到 n，而不是 active_end；延迟节点虽然不再参与当前主元搜索，
// 仍必须接收 Schur 补，才能作为正确的 contribution block 继续向上层传递。
void finish_panel(const int n, double *a, const int lda, const int begin,
                  const int end, const double *work, const int ldw) {
  const int width = end - begin;
  const int trailing = n - end;
  if (width <= 0 || trailing <= 0) {
    return;
  }

  cblas_dsyr2k(CblasColMajor, CblasLower, CblasNoTrans, trailing, width,
               -0.5, a + end + begin * lda, lda, work + end, ldw, 1.0,
               a + end + end * lda, lda);
}

} // namespace

void delayed_sytrf_blocked(const int n, double *a, const int lda, int *perm,
                           int *piv_size, int &info) {
  info = -1;
  if (n < 0 || lda < std::max(1, n) ||
      (n > 0 && (a == nullptr || perm == nullptr || piv_size == nullptr))) {
    return;
  }

  for (int i = 0; i < n; ++i) {
    perm[i] = i;
    piv_size[i] = 0;
  }

  // W 采用 n×kBlockSize 列主序堆存储，避免大矩阵时占用线程栈。
  // 三个长度 n 的临时向量分别用于当前列、2x2 的第二列和候选行更新。
  const int ldw = std::max(1, n);
  std::vector<double> work(static_cast<std::size_t>(ldw) * kBlockSize, 0.0);
  std::vector<double> column0(static_cast<std::size_t>(n), 0.0);
  std::vector<double> column1(static_cast<std::size_t>(n), 0.0);
  std::vector<double> row_values(static_cast<std::size_t>(n), 0.0);

  int panel_begin = 0;
  int active_end = n;

  while (panel_begin < active_end) {
    int k = panel_begin;
    int panel_width = 0;

    while (k < active_end && panel_width < kBlockSize) {
      // 当前列必须先更新；列最大值、对角元以及后续乘子都复用该结果。
      materialize_current_column(n, a, lda, panel_begin, panel_width,
                                 work.data(), ldw, k, column0.data());

      const int active_size = active_end - k;
      const double abs_akk = std::abs(column0[static_cast<std::size_t>(k)]);
      int candidate_row = k;
      double column_max = 0.0;
      if (active_size > 1) {
        const int relative = static_cast<int>(cblas_idamax(
            active_size - 1, column0.data() + k + 1, 1));
        candidate_row = k + 1 + relative;
        column_max =
            std::abs(column0[static_cast<std::size_t>(candidate_row)]);
      }

      bool use_1x1 = false;
      bool use_2x2 = false;
      int interchange = k;

      if (!(std::max(abs_akk, column_max) > 0.0)) {
        // 当前节点对应的活动列为零，无法组成 1x1 或 2x2 主元，稍后延迟。
      } else if (abs_akk >= kBunchKaufmanAlpha * column_max) {
        use_1x1 = true;
      } else {
        materialize_active_row(a, lda, panel_begin, panel_width, work.data(),
                               ldw, k, active_end, candidate_row,
                               row_values.data());
        double row_max = 0.0;
        for (int col = k; col < active_end; ++col) {
          if (col != candidate_row) {
            row_max =
                std::max(row_max,
                         std::abs(row_values[static_cast<std::size_t>(col)]));
          }
        }
        const double abs_app =
            std::abs(row_values[static_cast<std::size_t>(candidate_row)]);

        if (abs_akk >=
            kBunchKaufmanAlpha * column_max * (column_max / row_max)) {
          use_1x1 = true;
        } else if (abs_app >= kBunchKaufmanAlpha * row_max) {
          use_1x1 = true;
          interchange = candidate_row;
        } else {
          const double d00 = column0[static_cast<std::size_t>(k)];
          const double d10 =
              column0[static_cast<std::size_t>(candidate_row)];
          const double d11 =
              row_values[static_cast<std::size_t>(candidate_row)];
          if (stable_2x2(d00, d10, d11)) {
            use_2x2 = true;
            interchange = candidate_row;
          }
        }
      }

      if (!use_1x1 && !use_2x2) {
        // 把当前节点插入延迟后缀前端。W 的对应行必须与 A 同步交换，否则 下一次惰性更新会把其他节点的累计贡献减到新位置上。
        swap_active_nodes(n, a, lda, k, active_end - 1, perm, work.data(),
                          ldw, panel_width);
        --active_end;
        // k 不增加：交换到当前位置的新候选节点仍需检查。
        continue;
      }

      if (use_2x2 && panel_width + 2 > kBlockSize) {
        // 2x2 D 块不能跨越面板边界。当前面板提前结束，更新尾部后再从 k
        // 开始下一面板；这与 LAPACK DLASYF 可能返回 NB-1 列的原因相同。
        break;
      }

      if (use_1x1) {
        if (interchange != k) {
          swap_active_nodes(n, a, lda, k, interchange, perm, work.data(),
                            ldw, panel_width);
          // 交换改变了第 k 列，必须在新顺序下重新更新。
          materialize_current_column(n, a, lda, panel_begin, panel_width,
                                     work.data(), ldw, k, column0.data());
        }

        const double d = column0[static_cast<std::size_t>(k)];
        a[k + k * lda] = d;
        piv_size[k] = 1;

        // 对 1x1 块，L(:,k)=B(:,k)/d，而 W(:,local)=L(:,k)*d=B(:,k)。
        // 因此 W 可以直接保存更新列除法前的值。
        work[k + panel_width * ldw] = d;
        for (int row = k + 1; row < n; ++row) {
          const double b = column0[static_cast<std::size_t>(row)];
          a[row + k * lda] = b / d;
          work[row + panel_width * ldw] = b;
        }

        ++k;
        ++panel_width;
        continue;
      }

      // 2x2 主元由节点 k 与 interchange 组成，把后者移动到 k+1。
      swap_active_nodes(n, a, lda, k + 1, interchange, perm, work.data(),
                        ldw, panel_width);

      // 两列都必须在写入 D 和 L 之前更新；提前覆盖其中一列会破坏另一列
      // 仍需使用的面板基准数据。
      materialize_current_column(n, a, lda, panel_begin, panel_width,
                                 work.data(), ldw, k, column0.data());
      materialize_current_column(n, a, lda, panel_begin, panel_width,
                                 work.data(), ldw, k + 1, column1.data());

      const double d00 = column0[static_cast<std::size_t>(k)];
      const double d10 = column0[static_cast<std::size_t>(k + 1)];
      const double d11 = column1[static_cast<std::size_t>(k + 1)];
      a[k + k * lda] = d00;
      a[(k + 1) + k * lda] = d10;
      a[(k + 1) + (k + 1) * lda] = d11;
      piv_size[k] = 2;
      piv_size[k + 1] = 0;

      const double scale =
          std::max({std::abs(d00), std::abs(d10), std::abs(d11)});
      const double d00_scaled = d00 / scale;
      const double d10_scaled = d10 / scale;
      const double d11_scaled = d11 / scale;
      const double determinant_scaled =
          d00_scaled * d11_scaled - d10_scaled * d10_scaled;
      const double inverse_scaled_determinant =
          1.0 / (scale * determinant_scaled);

      // 2x2 块在 L 的块对角位置是 I2，因此相应的两行 W 就是 D 的两行。
      work[k + panel_width * ldw] = d00;
      work[(k + 1) + panel_width * ldw] = d10;
      work[k + (panel_width + 1) * ldw] = d10;
      work[(k + 1) + (panel_width + 1) * ldw] = d11;

      for (int row = k + 2; row < n; ++row) {
        const double b0 = column0[static_cast<std::size_t>(row)];
        const double b1 = column1[static_cast<std::size_t>(row)];
        a[row + k * lda] =
            (d11_scaled * b0 - d10_scaled * b1) *
            inverse_scaled_determinant;
        a[row + (k + 1) * lda] =
            (d00_scaled * b1 - d10_scaled * b0) *
            inverse_scaled_determinant;
        // W=[L0 L1]*D 恰好等于分解前更新出的两列 B。
        work[row + panel_width * ldw] = b0;
        work[row + (panel_width + 1) * ldw] = b1;
      }

      k += 2;
      panel_width += 2;
    }

    // 此时 [panel_begin,k) 已形成混合 1x1/2x2 面板。一次 BLAS-3 调用
    // 将累计 Schur 补写回 [k,n)，随后 W 可以被下一面板复用。
    finish_panel(n, a, lda, panel_begin, k, work.data(), ldw);
    panel_begin = k;
  }

  info = active_end;
}
