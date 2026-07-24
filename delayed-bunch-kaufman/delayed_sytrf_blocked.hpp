#pragma once

/**
 @brief 采用Bunch-Kaufman的选主元方法对未知格式的对称稠密矩阵进行 LDL^T 分解，并采取延迟主元策略将待延迟的主元延迟到右下角。
  @param n 矩阵的(行)列数。
  @param a 矩阵的指针。
  @param lda 矩阵的 leading dimension。
  @param perm 矩阵的行列交换记录。
  @param piv_size 记录分解的主元情况，值为1时代表当前位置为1x1主元，值为2时则代表2x2主元。
  @param info 输出参数，表示分解的状态。若 info = k，则表示从第 k 个主元开始无法继续分解，后续主元被延迟。
 @return 
  @note 输入矩阵a是列主序存储的，且在分解过程中会被修改为LDL^T分解的结果，注意到由于对称性的性质，我们只会读取矩阵的下半部分，也只会存放下半部分，因为对角块存在2x2主元，所以会占据部分上三角部分。
  @note 为获得更好性能我们在这里使用了 CBLAS 库来进行矩阵运算，确保在编译时链接相应的 BLAS 库。
  @note 参数非法时 void 函数立即返回，并设置 info=-1。 
 */
void delayed_sytrf_blocked(int n, double *a, int lda, int *perm, int *piv_size, int &info);
