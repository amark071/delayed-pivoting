#pragma once

/**
 @brief 采用列主元的方法对未知格式的稠密矩阵进行 LU 分解，并采取延迟主元策略将待延迟的主元延迟到右下角。
  @param m 矩阵的行数。
  @param n 矩阵的列数。
  @param a 矩阵的指针。
  @param lda 矩阵的 leading dimension。
  @param ipiv 行交换记录。
  @param jpiv 列交换记录。
  @param info 输出参数，表示分解的状态。若 info = k，则表示从第 k 个主元开始无法继续分解，后续主元被延迟。
 @return 
  @note 输入矩阵a是列主序存储的，且在分解过程中会被修改为LU分解的结果。
  @note 正常情况下每一步只在当前列做部分主元搜索；若当前列活动部分全零，则从最右侧剩余列向左扫描，把遇到的第一列非零列交换到当前位置，而不是在整个活动子矩阵中寻找绝对值最大的全主元。
  @note 为获得更好性能我们在这里使用了 CBLAS 库来进行矩阵运算，确保在编译时链接相应的 BLAS 库。
  @note 参数非法时 void 函数立即返回，并设置 info=-1。 
 */

void delayed_getrf_column(int m, int n, double *a, int lda, int *ipiv, int *jpiv, int &info);
