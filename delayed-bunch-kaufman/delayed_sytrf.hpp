#pragma once

// 对 n×n 实对称矩阵执行带延迟主元的 Bunch--Kaufman LDL^T 分解。
// A 使用列主序，lda 是相邻两列首地址之间的元素数。算法只读取和覆盖下三角；
// 上三角无需初始化，也不会与下三角同步。
//
// 返回后，[0,info) 中原地保存 P^T*A_original*P=L*D*L^T：L 为单位下三角，
// D 由 1x1 和 2x2 对角块组成；[info,n) 是未能选出稳定主元的延迟节点。
//
// perm[pos]：当前位置 pos 对应原矩阵中的哪个节点，采用 0-based 下标。
// piv_size[pos]：1 表示 1x1 主元；2 表示 2x2 主元的首位置；0 表示 2x2
// 主元的第二位置或延迟位置。
//
// n、lda 或指针非法时设置 info=-1 并立即返回；正常无延迟时 info=n。
void delayed_sytrf(int n, double *a, int lda, int *perm, int *piv_size,
                   int &info);
