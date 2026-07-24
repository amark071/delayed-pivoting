#pragma once

// LAPACK DSYTRF(UPLO='L') 的 0-based C++ 包装器。A 为列主序且仅使用下三角。
// 包装器额外调用 DSYCONV，把 LAPACK 交织的置换/乘子存储转换为与 delayed_sytrf 相同的显式 L、D、perm 和 piv_size 表示，便于直接比较。
//
// info==n 表示 LAPACK 未报告奇异 D 块；info 位于 [0,n) 时，它是转换为 0-based 的第一个奇异位置。
// LAPACK 会继续处理后续位置，因此该 info 不代表 delayed_sytrf 意义下的延迟后缀。参数非法时 info=-1。
void lapack_sytrf(int n, double *a, int lda, int *perm, int *piv_size, int &info);
