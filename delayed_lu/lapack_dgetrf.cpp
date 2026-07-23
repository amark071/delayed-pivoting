#include "lapack_dgetrf.hpp"

#if defined(__APPLE__)
// macOS Accelerate 在新 LAPACK 接口下使用 __LAPACK_int；不能默认它永远
// 与 C++ 的 int 完全相同，因此下面单独定义 LapackInt 并做显式转换。
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
using LapackInt = __LAPACK_int;
#else
// Linux/其他平台通常通过 Fortran 符号 dgetrf_ 链接 LAPACK。
using LapackInt = int;
extern "C" void dgetrf_(const LapackInt *m, const LapackInt *n, double *a,
                         const LapackInt *lda, LapackInt *ipiv,
                         LapackInt *info);
#endif

#include <algorithm>
#include <cstddef>
#include <vector>

void lapack_getrf(const int m, const int n, double *a, const int lda,
                  int *ipiv, int &info) {
  // 与两个自定义 delayed LU 保持相同的错误约定：-1 表示参数非法。
  info = -1;
  if (m < 0 || n < 0 || lda < std::max(1, m)) {
    return;
  }

  // DGETRF 对 m×n 矩阵产生 min(m,n) 条主元记录。
  const int pivots = std::min(m, n);
  if ((m > 0 && n > 0 && a == nullptr) || (pivots > 0 && ipiv == nullptr)) {
    return;
  }

  // 在接口边界统一转换整数类型，避免直接把 int* 传给 ILP64 LAPACK。
  const LapackInt lapack_m = static_cast<LapackInt>(m);
  const LapackInt lapack_n = static_cast<LapackInt>(n);
  const LapackInt lapack_lda = static_cast<LapackInt>(lda);
  LapackInt lapack_info = 0;
  // LAPACK 原生 IPIV 是 1-based 且元素类型为 LapackInt，先用临时数组接收。
  std::vector<LapackInt> lapack_ipiv(static_cast<std::size_t>(pivots));

  // 原地分解：返回后 A 的严格下三角部分保存 L 的乘子（L 对角线隐式为 1），
  // 上三角部分保存 U。DGETRF 仅做行交换，因此没有 jpiv。
  dgetrf_(&lapack_m, &lapack_n, a, &lapack_lda, lapack_ipiv.data(),
          &lapack_info);

  // Fortran LAPACK 的 IPIV[k] 取值范围从 1 开始；本项目公开接口严格使用
  // C++ 风格 0-based 下标，所以每一项减 1。其含义仍是“第 k 步与哪一行交换”。
  for (int k = 0; k < pivots; ++k) {
    ipiv[k] = static_cast<int>(lapack_ipiv[static_cast<std::size_t>(k)] - 1);
  }

  if (lapack_info == 0) {
    // 为便于和 delayed LU 对比，成功时也返回 pivots，表示没有延迟区间。
    info = pivots;
  } else if (lapack_info > 0) {
    // 原生 LAPACK INFO=i>0 表示 U(i,i) 恰好为零，其中 i 是 1-based。
    // 转为 0-based 后得到 i-1。注意 LAPACK 仍会继续完成分解；这里的 info
    // 只是第一个零对角元的位置，并不表示它真的采用了“延迟后缀”算法。
    info = static_cast<int>(lapack_info - 1);
  }
}
