#pragma once

// LAPACK DGETRF 的轻量 C++ 包装器。A 为列主序；原生 LAPACK IPIV 的1-based 行交换下标会被转换为本接口的 0-based 下标。
// 注意：LAPACK 报告奇异位置后仍会继续分解，因此这里 info<min(m,n) 并不表示 delayed_getrf 意义下的延迟后缀；

void lapack_getrf(int m, int n, double *a, int lda, int *ipiv, int &info);
