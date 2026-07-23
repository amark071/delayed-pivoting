# 非对称稠密矩阵的延迟主元 LU

## 接口和存储

```cpp
// 当前列失败后选活动子块的全局最大元
delayed_getrf(m, n, A, lda, ipiv, jpiv, info);

// 方案一：找到第一个可用列，每个被检查列都使用列主元法
delayed_getrf_column(m, n, A, lda, ipiv, jpiv, info);

// 直接调用 LAPACK DGETRF，无列交换、无延迟后缀
lapack_getrf(m, n, A, lda, ipiv, info);
```

接口采用 C++ 参数习惯。`A` 必须是列主序，元素
`A(i,j)` 位于 `A[i + j*lda]`。调用后，已消去的左上部分按 LAPACK
`DGETRF` 的方式原位保存单位下三角 `L`（对角线不存）和上三角 `U`。
`ipiv` 和 `jpiv` 分别记录每步交换的行号和列号，全部为 **0-based**。

- `info` 是延迟区间的 0-based 起点；
- `[0, info)` 已完成消元，`[info, min(m,n))` 为延迟区；
- `info == min(m,n)` 表示没有延迟主元；
- 参数或指针非法时函数直接返回，并令 `info == -1`。

上述 `info` 语义适用于两个延迟 LU 函数。使用半开区间是因为
0-based 接口中，若继续令 `info==0` 表示成功，就无法表达“从第 0 个
主元开始延迟”。

两个延迟算法在第 `k` 步都先用 `cblas_idamax` 在
`A(k:m-1,k)` 中找当前列绝对值最大的元素 `alpha`。若
`alpha > 0`，交换它所在行和第 `k` 行后直接消元。两者只在当前列
全零时出现差别。

- `delayed_getrf` 搜索整个活动子块，选取绝对值全局最大的元素，
  因此是全主元回退版本。
- `delayed_getrf_column` 从活动列末尾向前逐列调用 `idamax`，找到
  第一个非零列就停止，将它交换到第 `k` 列。换入列的行主元仍是
  该列的最大元，不计算活动子块的全局最大元。

若对应策略找不到任何非零主元，函数令 `info == k` 并延迟
`[k,min(m,n))`。两个延迟版本的分解关系都是 `P*A*Q=L*U`。

`lapack_getrf` 只是一个 0-based C++ 包装。LAPACK `DGETRF` 只做行部分主元，
不会在零列时搜索其他列，也不会把未分解部分整体延迟。其
`info < min(m,n)` 表示第一个零 `U` 对角的 0-based 位置，但 LAPACK 仍会
继续处理后续列；因此这个 `info` 不是延迟区间。

## BLAS 调用与前置工作

延迟实现使用 CBLAS 接口：`cblas_idamax` 定位主元，
`cblas_dswap` 交换行列，`cblas_dscal` 计算乘子。实现使用 64 列的
分块 LU：面板内部的小更新用 `cblas_dger`，面板完成后用
`cblas_dtrsm` 计算块行，再用 BLAS-3 `cblas_dgemm` 完成
`A22 -= L21*U12`。稠密大矩阵的主要计算因此能在高速缓存中复用
数据，并使用 BLAS 的多线程矩阵乘法内核。

分块不会改变主元流程。面板中每一步仍先用 `idamax` 检查当前
列；仅当列主元为零时，才先落实该面板尚未写回的块更新，然后
按各自策略搜索。这保证搜索看到的是当前 Schur 补。

“直接分解”使用的是 LAPACK `DGETRF`，不是单独的 BLAS 函数。BLAS 提供
向量、矩阵计算内核；`DGETRF` 属于 LAPACK，内部再调用 BLAS。

macOS 已自带 Accelerate，无需额外安装：

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Linux 推荐 OpenBLAS。Debian/Ubuntu 可先安装：

```sh
sudo apt install cmake g++ libopenblas-dev
```

然后使用同样的 CMake 命令。手工编译时：

```sh
# macOS
c++ -std=c++17 delayed_dgetrf.cpp delayed_dgetrf_column.cpp \
  lapack_dgetrf.cpp example.cpp -framework Accelerate -o example

# Linux + OpenBLAS
c++ -std=c++17 delayed_dgetrf.cpp delayed_dgetrf_column.cpp \
  lapack_dgetrf.cpp example.cpp -llapack -lopenblas -o example
```

不要混用 LP64 与 ILP64：当前接口的整数是 32 位 `int`，应链接 LP64
BLAS。若整个工程使用 MKL ILP64，参数类型应统一改成 64 位整数。

## 大矩阵内存

不要在函数或 `main` 的局部栈上声明大矩阵：

```cpp
// 错误示例：1000x1000 个 double 约为 8 MB，容易直接栈溢出。
std::array<double, 1000 * 1000> A;
```

使用 `std::vector` 把矩阵放到堆上，并在乘法前转换到 `std::size_t`：

```cpp
const std::size_t elements =
    static_cast<std::size_t>(lda) * static_cast<std::size_t>(n);
std::vector<double> A(elements, 0.0);
std::vector<int> ipiv(std::min(m, n));
std::vector<int> jpiv(std::min(m, n));

delayed_getrf(m, n, A.data(), lda, ipiv.data(), jpiv.data(), info);
```

打印整个大矩阵和用三重循环显式计算 `L*U` 也非常昂贵。大规模实验应只
输出 `info`、残差或少量主元信息，并用 BLAS `dgemm` 做矩阵乘法验证。

当前 `example.cpp` 使用一个 4096x4096、包含 8 个非零元的秩 8 矩阵。
矩阵及当前分解副本共约占用 256 MiB 堆内存；验证时加上重建
矩阵和压缩因子，该秩 8 例子的峰值约为 385 MiB。两个延迟版本预期
`info == 8`、延迟区为 `[8,4096)`；LAPACK 预期报告第 0 个对角为首个
零主元。示例对三种实现分别输出分解时间、主元信息、验证时间和
最大绝对误差。

有延迟主元时，尾部存储的是尚未分解的 Schur 补 `S`，验证时不能只把
前部当作普通完整 `L*U`。示例按下式恢复置换后的完整矩阵：

```text
[L11  0] [U11 U12]
[L21  I] [ 0    S]
```

本实现的置换约定为 `P*A*Q = Lbar*Ubar`，所以原矩阵恢复式为
`A = P^T*Lbar*Ubar*Q^T`。示例先删去 `Ubar` 中恒为零的行及
`Lbar` 中对应的列，这只压缩乘法的内维度，不改变乘积。真正的
矩阵乘法统一由 `cblas_dgemm` 完成；C++ 循环只负责根据 `ipiv/jpiv`
生成置换映射，并扫描 BLAS 计算结果得到最大绝对误差。LAPACK 路径的
`L*U` 验证也调用同一个 `cblas_dgemm` 内核。
