#pragma once

// delayed_sytrf 的面板化版本。
//
// 接口、列主序下三角存储、0-based perm/piv_size 以及 info 的含义均与
// delayed_sytrf 完全相同。区别在于该版本把多个 1x1/2x2 主元累计为面板，
// 使用工作矩阵 W=L_panel*D_panel 表示尚未写回的 Schur 补贡献，并在面板
// 结束时通过 BLAS-3 一次性更新尾部下三角矩阵。
void delayed_sytrf_blocked(int n, double *a, int lda, int *perm,
                           int *piv_size, int &info);
