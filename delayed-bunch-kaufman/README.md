# 对称不定矩阵的延迟主元 Bunch–Kaufman 分解

这是一个面向列主序实对称稠密矩阵的 C++17 实现。算法只读取并覆盖矩阵的下三角，使用 Bunch–Kaufman 规则选择 1x1 或 2x2 主元，并在两种主元都不能接受时把当前节点移入延迟后缀。分解结果满足

```text
P^T * A * P = L * D * L^T
```

其中 `P` 由 `perm` 表示，`D` 由 1x1 和 2x2 对角块组成。

## 接口与存储

```cpp
void delayed_sytrf(int n, double* A, int lda,
                   int* perm, int* piv_size, int& info);

void delayed_sytrf_blocked(int n, double* A, int lda,
                           int* perm, int* piv_size, int& info);

void lapack_sytrf(int n, double* A, int lda,
                  int* perm, int* piv_size, int& info);
```

全部公开下标均为 0-based。`A(i,j)` 位于 `A[i+j*lda]`，调用者只需要初始化 `i>=j` 的下三角；上三角不会被算法读取，也不会随下三角同步更新。

- `perm[pos]` 表示分解后位置 `pos` 上的节点来自原矩阵的哪个位置。
- `piv_size[pos] == 1` 表示当前位置是一个 1x1 主元。
- `piv_size[pos] == 2` 表示当前位置是一个 2x2 主元的起点。
- `piv_size[pos] == 0` 表示当前位置是 2x2 主元的第二列，或者属于延迟区域。
- 对两个延迟算法，`[0,info)` 已被消去，`[info,n)` 是延迟后缀。
- 对两个延迟算法，`info == n` 表示没有节点被延迟，`info == -1` 表示参数非法。

`delayed_sytrf` 与 `delayed_sytrf_blocked` 的接口、紧凑因子格式以及输出含义完全相同，因此上层程序可以直接替换二者进行比较。

`lapack_sytrf` 是 LAPACK `DSYTRF(UPLO='L')` 的 C++ 包装器。对这个包装器，`info == n` 表示 LAPACK 没有报告奇异 `D` 块；`info < n` 表示转换为 0-based 后的首个奇异位置。LAPACK 在该位置之后仍会继续分解，所以这个 `info` 不是延迟后缀的起点。包装器还调用 `DSYCONV`，把 LAPACK 交织存放的置换、乘子和 2x2 块非对角元转换为与两个延迟算法相同的显式 `L`、`D`、`perm` 和 `piv_size` 格式。

## 延迟节点的处理

分解过程中维护三个连续区间：

```text
[0,k)             已消去区域
[k,active_end)    当前仍可选作主元的活动区域
[active_end,n)    已延迟区域
```

如果位置 `k` 既不能组成稳定的 1x1 主元，也不能组成稳定的 2x2 主元，算法将它与 `active_end-1` 对称交换，然后令 `active_end` 减一。这样，新延迟节点总是插在已有延迟后缀的前面，不会每次都与 `n-1` 交换而破坏之前已经延迟的节点。

最终的右下角 `[info,n)` 仍保存对应的 Schur 补。验证完整分解时，可以把这个未消去部分看作 `D` 的最后一个对称块。

## 两个延迟实现

### 非分块实现

`delayed_sytrf` 是结构直接、便于检查的非分块版本，只存储和更新下三角：

- 1x1 主元使用 `cblas_dsyr` 完成对称秩一更新；
- 2x2 主元使用两次 `cblas_dsyr` 和一次 `cblas_dsyr2` 完成更新；
- 2x2 乘子直接在原矩阵中计算，不为每个主元重复分配临时向量；
- 对称交换只移动下三角保存的 Schur 补和已经计算好的 `L` 行；
- 候选行最大值通过下三角的对称读取获得。

### 分块实现

`delayed_sytrf_blocked` 以最多 64 列组成一个面板，并保证面板边界不会拆开 2x2 主元。面板内部尚未写回的 Schur 补由

```text
S = A_stored - L_panel * W_panel^T
W_panel = L_panel * D_panel
```

隐式表示。需要当前主元列或候选行时，通过 `dgemv` 按需物化；面板完成后，再用一次下三角 `dsyr2k` 把整个面板的贡献写回尾矩阵。发生主元交换或延迟时，算法同步交换 `A`、`perm` 和工作矩阵中的对应行。

面板更新覆盖活动候选和延迟节点，所以右下角的延迟后缀始终保持为正确的 Schur 补。工作矩阵使用堆内存，并在每个面板结束后清空和复用。

## 统一示例

对称目录只保留一个 `example.cpp`。它建立一份 `4096x4096`、数值秩为 8 的对称不定矩阵：

- 位置 2、3 构成一个 2x2 不定主元；
- 位置 4 至 9 构成六个 1x1 主元；
- 其余节点为零，两个延迟算法应在位置 8 开始延迟。

示例把同一份原矩阵分别复制给以下三个实现：

1. 非分块延迟 Bunch–Kaufman；
2. 分块延迟 Bunch–Kaufman；
3. LAPACK `DSYTRF` 包装器。

程序分别输出三者的分解耗时、验证耗时、`info` 含义和最大绝对误差。验证过程使用两次 BLAS `dgemm` 计算 `L*D*L^T`，然后与置换后的原矩阵 `P^T*A*P` 比较。分解计时和验证计时彼此独立。

这个低秩算例主要用于同时检查延迟语义、1x1/2x2 紧凑存储和三种实现的基本耗时，不代表一般稠密满秩矩阵上的最终性能排序。

## 编译与运行

macOS 使用系统自带的 Accelerate：

```sh
c++ -O3 -DNDEBUG -std=c++17 delayed_sytrf.cpp \
  delayed_sytrf_blocked.cpp lapack_dsytrf.cpp example.cpp \
  -framework Accelerate -o delayed_sytrf_example
./delayed_sytrf_example
```

Linux 需要 CBLAS 和 LAPACK，例如 Debian/Ubuntu 可以安装 OpenBLAS：

```sh
sudo apt install cmake g++ libopenblas-dev
c++ -O3 -DNDEBUG -std=c++17 delayed_sytrf.cpp \
  delayed_sytrf_blocked.cpp lapack_dsytrf.cpp example.cpp \
  -llapack -lopenblas -o delayed_sytrf_example
./delayed_sytrf_example
```

也可以使用 CMake：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## 大矩阵的内存分配

不要把大型稠密矩阵定义为局部 C 数组或 `std::array`。一个 `1000x1000` 的 `double` 矩阵约占 8 MB，已经接近很多系统的默认线程栈上限。矩阵和主元元数据应使用 `std::vector` 在堆上分配：

```cpp
const std::size_t elements =
    static_cast<std::size_t>(lda) * static_cast<std::size_t>(n);
std::vector<double> A(elements, 0.0);
std::vector<int> perm(static_cast<std::size_t>(n));
std::vector<int> piv_size(static_cast<std::size_t>(n));

delayed_sytrf(n, A.data(), lda, perm.data(), piv_size.data(), info);
```

在乘法前先转换为 `std::size_t`，还可以避免用 `int` 计算分配大小时发生有符号整数溢出。
