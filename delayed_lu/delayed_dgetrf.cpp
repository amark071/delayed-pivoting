#include "delayed_dgetrf.hpp"

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <algorithm>
#include <cmath>

namespace {

// 面板宽度。面板内部使用 BLAS-2 的逐列更新，面板完成后再用 BLAS-3 一次性更新右下角子矩阵。
constexpr int block_size = 64;

// 一个主元候选同时包含它在当前矩阵中的绝对行、列下标及绝对值。
struct PivotCandidate {
  int row;
  int col;
  double abs_value;
};

// 在活动子矩阵 A(k:m-1, k:n-1) 中寻找绝对值最大的元素。
//
// cblas_idamax 一次只能在一个向量中找最大绝对值元素的位置，因此这里按列扫描：
//   1. 对每一列调用 idamax，得到该列活动部分的最大元素；
//   2. 再比较各列的最大值，得到整个活动子矩阵的最大值。
//
// 注意：该函数只在“当前列已经确定为全零”时调用。正常情况下只做当前列的列主元搜索，不会提前扫描整个活动子矩阵。
PivotCandidate find_active_max(const int m, const int n, const double *a,
                               const int lda, const int k) {
  PivotCandidate candidate{k, k, 0.0};
  for (int j = k; j < n; ++j) {
    // idamax 返回相对于向量首地址 a(k,j) 的 0-based 下标。
    const int relative_row =
        static_cast<int>(cblas_idamax(m - k, a + k + j * lda, 1));
    // 转成矩阵中的绝对行下标。
    const int row = k + relative_row;
    const double value = std::abs(a[row + j * lda]);
    if (value > candidate.abs_value) {
      candidate = {row, j, value};
    }
  }
  return candidate;
}

// 完成面板A12,A22的块更新。
// 调用本函数之前，面板内各列已经通过 dger 完成分解，矩阵可按块理解为：
//
//          |----------|----------------------|
//          |   L11    |        A12           |
//          |   L21    |        A22           |
//          |----------|----------------------|
//
// 其中 L11 是单位下三角矩阵，L21 已经计算完成。下面用两次 BLAS-3 调用：
//   U12 = inv(L11) * A12
//   A22 = A22 - L21 * U12
// 相比每选一个主元就更新整个尾部矩阵，这种写法能显著提高大矩阵上的性能。
void finish_panel(const int m, const int n, double *a, const int lda,
                  const int begin, const int end) {
  const int width = end - begin;
  const int cols_right = n - end;
  if (width <= 0 || cols_right <= 0) {
    return;
  }

  // dtrsm 解三角方程 L11 * U12 = A12。
  // CblasUnit 表示 L11 的对角线不从 A 中读取，而是按 1 处理；这是因为
  // LU 的 L 对角线是隐式单位对角线，A 的对角位置实际存放的是 U 的对角元。
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
              CblasUnit, width, cols_right, 1.0,
              a + begin + begin * lda, lda, a + begin + end * lda, lda);

  const int rows_below = m - end;
  if (rows_below > 0) {
    // dgemm 完成 Schur 补更新：A22 <- A22 - L21 * U12。
    // alpha=-1、beta=1 对应 C = -A*B + C。
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rows_below,
                cols_right, width, -1.0, a + end + begin * lda, lda,
                a + begin + end * lda, lda, 1.0,
                a + end + end * lda, lda);
  }
}

// 在面板分解到一半时，第 k 列突然找不到主元，需要检查其他列。

// 此时面板前缀 [begin,k) 已经完成消去，但整个活动子矩阵并非全部是最新值。
// [begin, k)       已完成的面板前缀
// [k, panel_end)   已被 dger 更新，是最新值
// [panel_end, n)   尚未应用当前面板更新，是旧值
// 如果直接在整个活动子矩阵中寻找主元，那么 [panel_end,n) 中还是旧数据，可能选出错误主元。因此必须先将这部分更新
//
//                   begin       k       panel_end          n
//                     |---------|-----------|---------------|
// begin               | L/U 前缀| 面板内数据 | 远端 U 块行      |
// k                   | L 乘子  | 当前活动块 | 远端活动块       |
//                     |---------|-----------|---------------|
//
void materialize_partial_panel(const int m, const int n, double *a,
                               const int lda, const int begin, const int k,
                               const int panel_end) {
  const int width = k - begin;
  const int far_cols = n - panel_end;
  if (width <= 0 || far_cols <= 0) {
    return;
  }

  // 先求远端块行中的 U：U_far <- inv(L_prefix) * A_far。
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
              CblasUnit, width, far_cols, 1.0,
              a + begin + begin * lda, lda,
              a + begin + panel_end * lda, lda);

  const int active_rows = m - k;
  if (active_rows > 0) {
    // 再把已接受前缀的贡献从活动行中减掉，使 A(k:m-1,panel_end:n-1)
    // 成为真正的当前 Schur 补，随后才能进行全局主元搜索。
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, active_rows,
                far_cols, width, -1.0, a + k + begin * lda, lda,
                a + begin + panel_end * lda, lda, 1.0,
                a + k + panel_end * lda, lda);
  }
}

} // namespace

void delayed_getrf(const int m, const int n, double *a, const int lda,
                   int *ipiv, int *jpiv, int &info) {
  // info=-1 统一表示输入参数非法；合法输入会在下面改写为其他值。
  info = -1;
  if (m < 0 || n < 0 || lda < std::max(1, m)) {
    return;
  }

  // 矩形 m×n 矩阵最多只能消去 min(m,n) 个主元。
  const int pivots = std::min(m, n);
  if ((m > 0 && n > 0 && a == nullptr) ||
      (pivots > 0 && (ipiv == nullptr || jpiv == nullptr))) {
    return;
  }
  // 默认认为全部主元都能完成。若从第 k 个主元起无法继续，则把 info 改为 k。
  // 因而正常完成时 info=min(m,n)，延迟发生时 [info,pivots) 是延迟主元区间。
  info = pivots;
  // ipiv[k]/jpiv[k] 记录第 k 步实际执行的行交换和列交换目标。
  // 初始化为恒等交换，方便空矩阵以及尚未发生交换的步骤保持明确语义。
  for (int j = 0; j < pivots; ++j) {
    ipiv[j] = j;
    jpiv[j] = j;
  }

  // panel_begin 是当前面板的起始主元下标。
  int panel_begin = 0;
  // 若短面板中途遇到零列，我们会先选好全主元并把它交换到 (k,k)，然后
  // 重启面板。该标志告诉新面板：第一个主元已经就位，不要重复选主元和交换。
  bool positioned_pivot = false;
  while (panel_begin < pivots) {
    // 最后一个面板可能不足 block_size，因此必须截断到 pivots。
    const int panel_end = std::min(pivots, panel_begin + block_size);
    // 在 for 循环内部不能直接修改循环边界来重启面板，使用标志退出后重进。
    bool restart_panel = false;

    for (int k = panel_begin; k < panel_end; ++k) {
      if (positioned_pivot) {
        // 前一个短面板已经完成完成，并把选出的全主元移动到了 (k,k)；对应的ipiv[k]、jpiv[k] 也已经记录，因此这里只清除标志并直接进入消去。
        positioned_pivot = false;
      } else {
        // 第一步始终只检查当前列 A(k:m-1,k)，采用列主元法。
        const int relative_pivot =
            static_cast<int>(cblas_idamax(m - k, a + k + k * lda, 1));
        const int column_pivot_row = k + relative_pivot;
        int pivot_row = column_pivot_row;
        int pivot_col = k;
        // alpha 是当前列最大元素的绝对值。若 alpha==0，则这一列活动部分全零。
        // 使用 !(alpha>0) 还能把 NaN 当作不可接受的主元，避免继续传播非法数值。
        const double alpha = std::abs(a[column_pivot_row + k * lda]);
        const bool zero_current_column = !(alpha > 0.0);

        // 严格遵守惰性选择顺序：只有当前列不含可用主元时，才扫描活动子矩阵。
        // 绝大多数正常列不会触发昂贵的全矩阵扫描。
        if (zero_current_column) {
          if (k > panel_begin) {
            // 当前零列出现在面板中间。远端列尚未收到本面板前缀的更新，必须先更新短面板，确保下面的全局搜索基于当前 Schur 补。
            materialize_partial_panel(m, n, a, lda, panel_begin, k, panel_end);
          }

          const PivotCandidate global_pivot = find_active_max(m, n, a, lda, k);
          if (!(global_pivot.abs_value > 0.0)) {
            // 整个活动子矩阵都没有非零元素，后续主元全部延迟；k 即延迟起点。
            info = k;
            return;
          }
          pivot_row = global_pivot.row;
          pivot_col = global_pivot.col;
        }

        // 交换采用“交换历史”而不是最终排列：第 k 步把 pivot_row 行换到 k，
        // 把 pivot_col 列换到 k。恢复 P、Q 时应按这些记录依次重放交换。
        ipiv[k] = pivot_row;
        jpiv[k] = pivot_col;
        if (pivot_row != k) {
          // 交换整行，长度为 n、步长为 lda。必须包含左侧已经形成的 L，才能让此前的消去因子与新的行排列保持一致。
          cblas_dswap(n, a + k, lda, a + pivot_row, lda);
        }
        if (pivot_col != k) {
          // 交换整列，长度为 m、步长为 1。必须包含上方已经形成的 U，才能让列置换 Q 与已分解部分保持一致。
          cblas_dswap(m, a + k * lda, 1, a + pivot_col * lda, 1);
        }

        if (zero_current_column && k > panel_begin) {
          // [panel_begin,k) 已经成为一个完整的短面板。若继续沿用原 panel_end，
          // 后续块更新会重复或遗漏；因此从 k 创建新面板，并复用刚放到位的主元。
          panel_begin = k;
          positioned_pivot = true;
          restart_panel = true;
          break;
        }
      }

      const int rows_below = m - k - 1;
      if (rows_below > 0) {
        // 计算 L(k+1:m-1,k) = A(k+1:m-1,k) / U(k,k)。
        // 乘以倒数可由一次 dscal 完成；L 的单位对角线不显式写入矩阵。
        const double inverse_pivot = 1.0 / a[k + k * lda];
        cblas_dscal(rows_below, inverse_pivot,
                    a + (k + 1) + k * lda, 1);
      }

      // 面板内部做秩 1 更新：
      // A(k+1:m-1,k+1:panel_end-1) -= L(:,k) * U(k,:)。
      // 此处只更新面板剩余列；更大的右侧尾部累计到 finish_panel 中用 dgemm 更新。
      const int panel_cols_right = panel_end - k - 1;
      if (rows_below > 0 && panel_cols_right > 0) {
        cblas_dger(CblasColMajor, rows_below, panel_cols_right, -1.0,
                   a + (k + 1) + k * lda, 1,
                   a + k + (k + 1) * lda, lda,
                   a + (k + 1) + (k + 1) * lda, lda);
      }
    }

    if (restart_panel) {
      // panel_begin 已在内部改为 k，下一轮 while 会重新计算新的 panel_end。
      continue;
    }

    // 当前面板正常完成：形成其 U 块行并更新右下角尾部矩阵。
    finish_panel(m, n, a, lda, panel_begin, panel_end);
    panel_begin = panel_end;
  }
}
