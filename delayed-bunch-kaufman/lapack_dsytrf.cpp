#include "lapack_dsytrf.hpp"

#if defined(__APPLE__)
// Apple Accelerate 的新版 LAPACK 可能采用与 int 不同的整数 ABI，统一使用
// __LAPACK_int 接收原生参数和 IPIV，再显式转换到公开接口的 int。
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
using LapackInt = __LAPACK_int;
#else
// 非 Apple 平台按常见 Fortran 符号名称链接 DSYTRF 与 DSYCONV。
using LapackInt = int;
extern "C" void dsytrf_(const char *uplo, const LapackInt *n, double *a,
                         const LapackInt *lda, LapackInt *ipiv, double *work,
                         const LapackInt *lwork, LapackInt *info);
extern "C" void dsyconv_(const char *uplo, const char *way,
                          const LapackInt *n, double *a,
                          const LapackInt *lda, const LapackInt *ipiv,
                          double *e, LapackInt *info);
#endif

#include <algorithm>
#include <cstddef>
#include <vector>

void lapack_sytrf(const int n, double *a, const int lda, int *perm,
                  int *piv_size, int &info) {
  // 与 delayed_sytrf 保持一致：非法输入用 info=-1 表示。
  info = -1;
  if (n < 0 || lda < std::max(1, n) ||
      (n > 0 && (a == nullptr || perm == nullptr || piv_size == nullptr))) {
    return;
  }

  // 输出先初始化为恒等排列和“未标记块”。后面根据 LAPACK IPIV 重放交换。
  for (int k = 0; k < n; ++k) {
    perm[k] = k;
    piv_size[k] = 0;
  }
  if (n == 0) {
    // 0 阶矩阵无需调用 LAPACK；info==n==0 表示正常完成。
    info = 0;
    return;
  }

  // 只使用并覆盖 A 的下三角，与自定义实现的存储约定完全一致。
  const char uplo = 'L';
  // 所有传入 Fortran LAPACK 的整数都先转换到 LapackInt。
  const LapackInt lapack_n = static_cast<LapackInt>(n);
  const LapackInt lapack_lda = static_cast<LapackInt>(lda);
  // LAPACK 的 IPIV 使用 1-based 特殊编码，且整数宽度取决于所链接的库。
  std::vector<LapackInt> lapack_ipiv(static_cast<std::size_t>(n));
  LapackInt lapack_info = 0;
  // 第一次以 LWORK=-1 调用是标准工作区查询：不做实际分解，只把推荐的
  // 工作区长度写入 work_query。使用推荐值能让 LAPACK 选择其高效分块路径。
  LapackInt lwork = -1;
  double work_query = 0.0;
  dsytrf_(&uplo, &lapack_n, a, &lapack_lda, lapack_ipiv.data(), &work_query,
          &lwork, &lapack_info);
  if (lapack_info < 0) {
    // LAPACK 报告参数错误；公开接口保留初始化时的 info=-1。
    return;
  }

  // 分配查询得到的工作区，然后第二次调用执行真正的原地分解。
  lwork = std::max<LapackInt>(1, static_cast<LapackInt>(work_query));
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dsytrf_(&uplo, &lapack_n, a, &lapack_lda, lapack_ipiv.data(), work.data(),
          &lwork, &lapack_info);
  if (lapack_info < 0) {
    return;
  }

  // DSYTRF 的下三角输出把置换信息与三角乘子交织在一起，不能直接按自定义
  // delayed_sytrf 的显式 L/D 布局解释。调用 DSYCONV(WAY='C') 后：
  //   1. 与 IPIV 对应的交换被应用到 L 的乘子行；
  //   2. 每个 2x2 D 块的非对角元被暂时分离到数组 E。
  // 这样矩阵中的 L 才能使用与自定义实现相同的方式进行验证和比较。
  const char convert = 'C';
  std::vector<double> d_off_diagonal(static_cast<std::size_t>(n), 0.0);
  LapackInt conversion_info = 0;
  dsyconv_(&uplo, &convert, &lapack_n, a, &lapack_lda, lapack_ipiv.data(),
           d_off_diagonal.data(), &conversion_info);
  if (conversion_info < 0) {
    return;
  }
  // 自定义布局要求 2x2 D 的非对角元仍保存在 A(block+1,block)。因此扫描
  // 原生 IPIV，遇到负数编码的 2x2 块时，把 DSYCONV 分离到 E 的值放回去。
  for (int block = 0; block + 1 < n;) {
    if (lapack_ipiv[static_cast<std::size_t>(block)] < 0) {
      a[(block + 1) + block * lda] =
          d_off_diagonal[static_cast<std::size_t>(block)];
      // 一个 2x2 块占两个连续位置，直接跨过第二项。
      block += 2;
    } else {
      ++block;
    }
  }

  // 把 LAPACK 针对下三角分解的 1-based IPIV 编码翻译成公开的 0-based
  // perm+piv_size 表示，并按顺序重放对称交换：
  //   IPIV[k] > 0：1x1 块，第 k 个位置与 IPIV[k]-1 交换；
  //   IPIV[k] < 0：k、k+1 构成 2x2 块，位置 k+1 与 -IPIV[k]-1 交换。
  int k = 0;
  while (k < n) {
    const LapackInt encoded = lapack_ipiv[static_cast<std::size_t>(k)];
    if (encoded > 0) {
      // 1-based 转 0-based。
      const int interchange = static_cast<int>(encoded - 1);
      std::swap(perm[k], perm[interchange]);
      piv_size[k] = 1;
      ++k;
    } else {
      // 下三角 DSYTRF 的 2x2 编码为一对相同负值；交换作用在块的第二位置。
      const int interchange = static_cast<int>(-encoded - 1);
      std::swap(perm[k + 1], perm[interchange]);
      piv_size[k] = 2;
      piv_size[k + 1] = 0;
      k += 2;
    }
  }

  // 为了和 delayed_sytrf 的接口形式统一，LAPACK 正常完成时返回 info=n。
  // 若原生 INFO=i>0，则 D 的第 i 行/列对应块奇异，这里转为 0-based 的 i-1。
  // LAPACK 仍会继续分解后续位置，所以该值不是“延迟后缀”的起点。
  info = lapack_info == 0 ? n : static_cast<int>(lapack_info - 1);
}
