#include "delayed_dgetrf_column.hpp"

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <algorithm>
#include <cmath>

namespace {

// 与全主元版本相同：面板内用 BLAS-2，面板完成后用 BLAS-3 更新尾部。
constexpr int block_size = 64;

// 当当前列为零时，记录从其他列中找到的可用列主元。
// row、col 均为矩阵中的 0-based 绝对下标。
struct ColumnCandidate {
  int row;
  int col;
  double abs_value;
};

// 当前第 k 列已经确认全零，因此从剩余列的最右端向左查找第一列非零列。
//
// 这就是与全主元版本的核心区别：这里不比较所有候选列的最大值，找到第一列可用列就停止；在该列内部仍调用 idamax 采用列主元法。
// 因而延迟列通常被交换到右端，而正常主元仍保持部分主元选取的数值稳定性。
ColumnCandidate find_nonzero_column(const int m, const int n, const double *a,
                                    const int lda, const int k) {
  for (int j = n - 1; j > k; --j) {
    // 搜索候选列 A(k:m-1,j)。CBLAS 返回相对于 a(k,j) 的 0-based 下标。
    const int relative_row =
        static_cast<int>(cblas_idamax(m - k, a + k + j * lda, 1));
    const int row = k + relative_row;
    const double value = std::abs(a[row + j * lda]);
    if (value > 0.0) {
      // 惰性停止：不再检查更左侧列，也不做全局最大值比较。
      return {row, j, value};
    }
  }
  // 剩余每一列都为零，活动子矩阵已无可用主元。
  return {k, k, 0.0};
}

// 完全一致于全主元版本：完成面板A12,A22的块更新。
void finish_panel(const int m, const int n, double *a, const int lda,
                  const int begin, const int end) {
  const int width = end - begin;
  const int cols_right = n - end;
  if (width <= 0 || cols_right <= 0) {
    return;
  }

  // L11 为隐式单位下三角矩阵，故使用 CblasUnit。
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
              CblasUnit, width, cols_right, 1.0,
              a + begin + begin * lda, lda, a + begin + end * lda, lda);

  const int rows_below = m - end;
  if (rows_below > 0) {
    // C = -A*B+C，即右下角 Schur 补更新。
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rows_below,
                cols_right, width, -1.0, a + end + begin * lda, lda,
                a + begin + end * lda, lda, 1.0,
                a + end + end * lda, lda);
  }
}

// 完全一致于全主元版本：在面板分解到一半时，第 k 列突然找不到主元，需要检查其他列。
void materialize_partial_panel(const int m, const int n, double *a,
                               const int lda, const int begin, const int k,
                               const int panel_end) {
  const int width = k - begin;
  const int far_cols = n - panel_end;
  if (width <= 0 || far_cols <= 0) {
    return;
  }

  // U_far <- inv(L_prefix) * A_far。
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
              CblasUnit, width, far_cols, 1.0,
              a + begin + begin * lda, lda,
              a + begin + panel_end * lda, lda);

  const int active_rows = m - k;
  if (active_rows > 0) {
    // A_active_far <- A_active_far - L_active_prefix * U_far。
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, active_rows,
                far_cols, width, -1.0, a + k + begin * lda, lda,
                a + begin + panel_end * lda, lda, 1.0,
                a + k + panel_end * lda, lda);
  }
}

} // namespace

void delayed_getrf_column(const int m, const int n, double *a, const int lda,
                          int *ipiv, int *jpiv, int &info) {
  // 非法参数统一返回 -1；本接口使用引用输出 info，因此函数本身为 void。
  info = -1;
  if (m < 0 || n < 0 || lda < std::max(1, m)) {
    return;
  }

  // 矩形矩阵的主元步数上限。
  const int pivots = std::min(m, n);
  if ((m > 0 && n > 0 && a == nullptr) ||
      (pivots > 0 && (ipiv == nullptr || jpiv == nullptr))) {
    return;
  }
  // 正常完成时 info=pivots；若活动子矩阵全零，则 info=k，表示延迟从 k 开始。
  info = pivots;
  // ipiv/jpiv 保存每一步实际发生的 0-based 行、列交换，初始为恒等交换。
  for (int j = 0; j < pivots; ++j) {
    ipiv[j] = j;
    jpiv[j] = j;
  }

  // 当前块面板的起点；positioned_pivot 表示重启面板时 (k,k) 主元已经就位。
  int panel_begin = 0;
  bool positioned_pivot = false;
  while (panel_begin < pivots) {
    // 面板右端不超过可分解主元总数。
    const int panel_end = std::min(pivots, panel_begin + block_size);
    bool restart_panel = false;

    for (int k = panel_begin; k < panel_end; ++k) {
      if (positioned_pivot) {
        // 该主元是在上一个短面板更新后选出并交换到位的，不可再次搜索或交换。
        positioned_pivot = false;
      } else {
        // 无条件先在当前列 A(k:m-1,k) 内做列主元搜索。
        const int relative_pivot =
            static_cast<int>(cblas_idamax(m - k, a + k + k * lda, 1));
        const int column_pivot_row = k + relative_pivot;
        int pivot_row = column_pivot_row;
        int pivot_col = k;
        // 该列最大绝对值为零（或比较遇到 NaN）时，当前列不可用。
        const double alpha = std::abs(a[column_pivot_row + k * lda]);
        const bool zero_current_column = !(alpha > 0.0);

        if (zero_current_column) {
          if (k > panel_begin) {
            // 当前列出现在面板中部，先把远端列更新到当前 Schur 补状态。
            materialize_partial_panel(m, n, a, lda, panel_begin, k,
                                      panel_end);
          }

          const ColumnCandidate column_pivot =
              find_nonzero_column(m, n, a, lda, k);
          if (!(column_pivot.abs_value > 0.0)) {
            // 没有任何剩余非零列，后续主元全部延迟，k 为延迟起点。
            info = k;
            return;
          }
          pivot_row = column_pivot.row;
          pivot_col = column_pivot.col;
        }

        // 记录第 k 步的交换目标，而不是最终排列向量。
        ipiv[k] = pivot_row;
        jpiv[k] = pivot_col;
        if (pivot_row != k) {
          // 交换完整行（跨全部 n 列），包括先前已经形成的 L 部分。
          cblas_dswap(n, a + k, lda, a + pivot_row, lda);
        }
        if (pivot_col != k) {
          // 交换完整列（跨全部 m 行），包括先前已经形成的 U 部分。
          cblas_dswap(m, a + k * lda, 1, a + pivot_col * lda, 1);
        }

        if (zero_current_column && k > panel_begin) {
          // 原面板被截断为 [panel_begin,k)。从 k 重启一个新面板，并保留
          // 刚刚执行完的行列交换，避免对同一主元重复搜索。
          panel_begin = k;
          positioned_pivot = true;
          restart_panel = true;
          break;
        }
      }

      const int rows_below = m - k - 1;
      if (rows_below > 0) {
        // 用 dscal 计算当前列的 L 乘子：L(:,k)=A(:,k)/U(k,k)。
        cblas_dscal(rows_below, 1.0 / a[k + k * lda],
                    a + (k + 1) + k * lda, 1);
      }

      const int panel_cols_right = panel_end - k - 1;
      if (rows_below > 0 && panel_cols_right > 0) {
        // 只更新面板内尚未分解的列：A_panel -= L(:,k)*U(k,:)；
        // 面板右侧的大块更新留到 finish_panel 中交给 dgemm。
        cblas_dger(CblasColMajor, rows_below, panel_cols_right, -1.0,
                   a + (k + 1) + k * lda, 1,
                   a + k + (k + 1) * lda, lda,
                   a + (k + 1) + (k + 1) * lda, lda);
      }
    }

    if (restart_panel) {
      // panel_begin 已设为 k，重新进入 while 后会建立新的面板边界。
      continue;
    }

    // 当前面板完整结束，批量形成右侧 U 块行并更新尾部矩阵。
    finish_panel(m, n, a, lda, panel_begin, panel_end);
    panel_begin = panel_end;
  }
}
