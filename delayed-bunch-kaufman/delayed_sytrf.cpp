#include "delayed_sytrf.hpp"

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

// Bunch--Kaufman 对角主元判据中的经典常数 alpha=(1+sqrt(17))/8。
// 该阈值用于在数值稳定性和 1x1 主元数量之间取得平衡。
constexpr double kBunchKaufmanAlpha =
    (1.0 + 4.1231056256176605498) / 8.0;
// 判断 2x2 主元块是否接近奇异时使用的相对容差。乘以 64 是给浮点舍入
// 留出安全余量；检查过程会先缩放矩阵块，因此该容差与数据绝对量级无关。
constexpr double kDeterminantTolerance =
    64.0 * std::numeric_limits<double>::epsilon();

// 读取逻辑上的对称元素 A(row,col)。本实现只把下三角视为有效存储：
// 若请求上三角元素，就读取与其对称的下三角位置 A(col,row)。
// 因此调用者不需要初始化上三角，算法也不会依赖上三角中的旧数据。
double symmetric_lower_value(const double *a, const int lda, const int row,
                             const int col) {
  return row >= col ? a[row + col * lda] : a[col + row * lda];
}

// 在“只存下三角”的布局中执行对称交换
void symmetric_swap_lower(const int n, double *a, const int lda, int x, int y,
                          int *perm) {
  if (x == y) {
    // 恒等置换，不需要改矩阵和 perm。
    return;
  }
  if (x > y) {
    // 后续指针公式假定 x<y，交换形参不会改变置换本身的含义。
    std::swap(x, y);
  }

  if (x > 0) {
    // 区段 1：两段都是“固定行、跨列”，列主序下步长均为 lda。
    cblas_dswap(x, a + x, lda, a + y, lda);
  }
  const int middle = y - x - 1;
  if (middle > 0) {
    // 区段 2：左向量位于第 x 列，内存连续；右向量是第 y 行跨列，
    // 列主序下相邻元素相隔 lda。
    cblas_dswap(middle, a + (x + 1) + x * lda, 1,
                a + y + (x + 1) * lda, lda);
  }
  // 区段 3：交换两个对角元。A(y,x) 不需要交换。
  std::swap(a[x + x * lda], a[y + y * lda]);
  const int tail = n - y - 1;
  if (tail > 0) {
    // 区段 4：两段分别位于第 x、y 列，均为连续内存。
    cblas_dswap(tail, a + (y + 1) + x * lda, 1,
                a + (y + 1) + y * lda, 1);
  }
  // perm[pos] 表示当前位置 pos 上来自原矩阵的哪个节点，因此矩阵位置交换时
  // 直接交换相应的 perm 元素。
  std::swap(perm[x], perm[y]);
}

// 求活动区间 [active_begin,active_end) 内指定行的最大非对角元素绝对值。
// Bunch--Kaufman 判据需要同时比较当前列最大值和候选行最大值。由于上三角未维护，每次读取都通过 symmetric_lower_value 映射到下三角存储。
double active_row_max_excluding_diagonal(const int active_begin,
                                         const int active_end, const double *a,
                                         const int lda, const int row) {
  double result = 0.0;
  for (int col = active_begin; col < active_end; ++col) {
    if (col == row) {
      // 行最大值的定义不包含对角元 A(row,row)。
      continue;
    }
    result =
        std::max(result, std::abs(symmetric_lower_value(a, lda, row, col)));
  }
  return result;
}

// 判断对称块 D=[a b; b c] 是否适合作为可逆的 2x2 主元。
//
// 直接计算 a*c-b*b 可能上溢或下溢，因此先用块中最大绝对值 scale 归一化，
// 再在 O(1) 量级上计算行列式。最后把缩放后的行列式与矩阵范数平方乘机器精度的阈值比较；过于接近奇异的块不接受，而是让当前节点进入延迟区。
bool stable_2x2(const double a, const double b, const double c) {
  const double scale = std::max({std::abs(a), std::abs(b), std::abs(c)});
  if (!(scale > 0.0)) {
    // 全零块以及含 NaN 导致比较失败的块都不能作为主元。
    return false;
  }

  // 缩放后 |as|、|bs|、|cs| 均不超过 1，行列式计算更加安全。
  const double as = a / scale;
  const double bs = b / scale;
  const double cs = c / scale;
  const double determinant_scaled = as * cs - bs * bs;
  const double norm_scaled =
      std::max(std::abs(as) + std::abs(bs), std::abs(bs) + std::abs(cs));
  return std::abs(determinant_scaled) >
         kDeterminantTolerance * norm_scaled * norm_scaled;
}

void factor_1x1(const int n, double *a, const int lda, const int k) {
  // trailing 是当前 1x1 主元下方仍需更新的节点数。
  const int trailing = n - k - 1;
  if (trailing <= 0) {
    return;
  }

  // 分块形式为 [d, b^T; b, C]。分解后需要保存 l=b/d，并形成
  // Schur 补 C <- C-l*d*l^T。
  const double d = a[k + k * lda];
  double *const l = a + (k + 1) + k * lda;
  // dscal 原地把主元列下方的 b 改写为 L 的乘子 l=b/d。
  cblas_dscal(trailing, 1.0 / d, l, 1);
  // dsyr 做对称秩 1 更新 C <- C-d*l*l^T；CblasLower 保证只更新下三角，
  // 与本算法“上三角不保证有效”的存储约定一致。
  cblas_dsyr(CblasColMajor, CblasLower, trailing, -d, l, 1,
             a + (k + 1) + (k + 1) * lda, lda);
}

void factor_2x2(const int n, double *a, const int lda, const int k) {
  // 2x2 主元占据 k 和 k+1，尾部从 k+2 开始。
  const int trailing = n - k - 2;
  if (trailing <= 0) {
    return;
  }

  // D=[d00 d10; d10 d11]。下三角位置 A(k+1,k) 保存 D 的非对角元。
  const double d00 = a[k + k * lda];
  const double d10 = a[(k + 1) + k * lda];
  const double d11 = a[(k + 1) + (k + 1) * lda];
  // 与稳定性检查一致，先缩放 D，再求逆，避免直接形成 d00*d11-d10*d10
  // 时溢出或下溢。stable_2x2 已保证 scale 和 determinant_scaled 可用。
  const double scale = std::max({std::abs(d00), std::abs(d10), std::abs(d11)});
  const double d00_scaled = d00 / scale;
  const double d10_scaled = d10 / scale;
  const double d11_scaled = d11 / scale;
  const double determinant_scaled =
      d00_scaled * d11_scaled - d10_scaled * d10_scaled;
  const double inverse_scaled_determinant = 1.0 / (scale * determinant_scaled);

  // 此时两列在尾部保存原始 B。需要原地计算 L=B*inv(D)。
  double *const l0 = a + (k + 2) + k * lda;
  double *const l1 = a + (k + 2) + (k + 1) * lda;

  for (int i = 0; i < trailing; ++i) {
    // 先保存 b0、b1，避免写回 l0[i] 后破坏计算 l1[i] 所需的原值。
    const double b0 = l0[i];
    const double b1 = l1[i];
    l0[i] = (d11_scaled * b0 - d10_scaled * b1) *
            inverse_scaled_determinant;
    l1[i] = (d00_scaled * b1 - d10_scaled * b0) *
            inverse_scaled_determinant;
  }

  // 展开 C <- C-L*D*L^T：
  //   C <- C - d00*l0*l0^T
  //          - d10*(l0*l1^T+l1*l0^T)
  //          - d11*l1*l1^T
  // 两次 dsyr 加一次 dsyr2 只更新下三角，不需要额外工作向量，也比两次
  // 非对称 dger 更符合这里的存储形式。
  double *const c = a + (k + 2) + (k + 2) * lda;
  cblas_dsyr(CblasColMajor, CblasLower, trailing, -d00, l0, 1, c, lda);
  cblas_dsyr2(CblasColMajor, CblasLower, trailing, -d10, l0, 1, l1, 1, c,
              lda);
  cblas_dsyr(CblasColMajor, CblasLower, trailing, -d11, l1, 1, c, lda);
}

} // namespace

void delayed_sytrf(const int n, double *a, const int lda, int *perm,
                   int *piv_size, int &info) {
  // 非法维数、lda 或空指针统一返回 info=-1。
  info = -1;
  if (n < 0 || lda < std::max(1, n) || (n > 0 && (!a || !perm || !piv_size))) {
    return;
  }

  // 初始排列为恒等排列；piv_size=0 表示该位置尚未被选为已接受主元。
  for (int i = 0; i < n; ++i) {
    perm[i] = i;
    piv_size[i] = 0;
  }

  // 在任意时刻，下标区间具有如下含义：
  //   [0,k)          已完成分解的 1x1/2x2 主元；
  //   [k,active_end) 尚可参与主元选择的活动节点；
  //   [active_end,n) 已延迟节点，顺序固定且不能再被覆盖。
  int k = 0;
  int active_end = n;
  while (k < active_end) {
    // 只在候选区中选主元，绝不能把已延迟后缀重新纳入搜索。
    const int active_size = active_end - k;
    const double abs_akk = std::abs(a[k + k * lda]);

    int candidate_row = k;
    double column_max = 0.0;
    if (active_size > 1) {
      // 在当前列对角线下方 A(k+1:active_end-1,k) 中寻找最大绝对值。
      // CBLAS idamax 返回相对起点的 0-based 下标。
      const int relative = static_cast<int>(
          cblas_idamax(active_size - 1, a + (k + 1) + k * lda, 1));
      candidate_row = k + 1 + relative;
      column_max = std::abs(a[candidate_row + k * lda]);
    }

    // 三个状态变量描述本轮判据的结果：选 1x1、选 2x2，或两者都不选而延迟。
    bool use_1x1 = false;
    bool use_2x2 = false;
    int interchange = k;

    if (!(std::max(abs_akk, column_max) > 0.0)) {
      // 当前节点的对角元和活动列都为零，包含节点 k 的 1x1/2x2 主元均不可用。
      // 保持两个 use 标志为 false，稍后把节点 k 移入延迟区。
    } else if (abs_akk >= kBunchKaufmanAlpha * column_max) {
      // 判据 1：当前对角元相对本列足够大，直接使用 A(k,k) 作为 1x1 主元。
      use_1x1 = true;
    } else {
      // 当前对角元不够大，需要检查列最大元素所在的候选行。
      const double row_max = active_row_max_excluding_diagonal(
          k, active_end, a, lda, candidate_row);
      const double abs_app = std::abs(a[candidate_row + candidate_row * lda]);

      if (abs_akk >= kBunchKaufmanAlpha * column_max * (column_max / row_max)) {
        // 判据 2：结合候选行尺度后，A(k,k) 仍可接受为 1x1 主元。
        use_1x1 = true;
      } else if (abs_app >= kBunchKaufmanAlpha * row_max) {
        // 判据 3：候选行的对角元 A(p,p) 足够大。先把节点 p 对称交换到 k，
        // 再以新的 A(k,k) 作为 1x1 主元。
        use_1x1 = true;
        interchange = candidate_row;
      } else {
        const double d00 = a[k + k * lda];
        const double d10 = a[candidate_row + k * lda];
        const double d11 = a[candidate_row + candidate_row * lda];
        if (stable_2x2(d00, d10, d11)) {
          // 两个 1x1 候选都不合适时，尝试由节点 k 和 p 组成 2x2 主元块。
          // 仅当该块不接近奇异时接受，否则延迟当前节点。
          use_2x2 = true;
          interchange = candidate_row;
        }
      }
    }

    if (!use_1x1 && !use_2x2) {
      // 把当前节点 k 插入延迟后缀的最前端，即交换到 active_end-1，然后将
      // active_end 左移。不能每次都交换到固定的 n-1：那里可能已经保存较早
      // 延迟的节点，再次交换会把它换回活动区，破坏延迟后缀。
      symmetric_swap_lower(n, a, lda, k, active_end - 1, perm);
      --active_end;
      // k 不增加：交换到位置 k 的新节点还没有检查，下一轮继续处理它。
      continue;
    }

    if (use_1x1) {
      // 把选定节点放到 k，记录 1x1 块，随后计算 L 乘子和 Schur 补。
      symmetric_swap_lower(n, a, lda, k, interchange, perm);
      piv_size[k] = 1;
      factor_1x1(n, a, lda, k);
      ++k;
      continue;
    }

    // 2x2 块的第一个节点已经在 k，只需把候选节点交换到 k+1。
    // piv_size[k]=2 标记块起点，piv_size[k+1]=0 标记它是同一块的第二位置。
    symmetric_swap_lower(n, a, lda, k + 1, interchange, perm);
    piv_size[k] = 2;
    piv_size[k + 1] = 0;
    factor_2x2(n, a, lda, k);
    k += 2;
  }

  // 循环结束时 k==active_end：[0,info) 已完成 LDL^T 分解，
  // [info,n) 是无法接受 1x1/2x2 主元而被延迟的节点区间。
  info = active_end;
}
