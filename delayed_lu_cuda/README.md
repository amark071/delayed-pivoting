# CUDA 版延迟主元 LU

本目录是 `delayed_lu/delayed_dgetrf.cpp` 的 NVIDIA CUDA 实现，保持相同的列主序紧凑存储、0-based 交换历史和 `info` 语义：正常完成时 `info=min(m,n)`；若活动子矩阵从第 `k` 步起全零，则 `info=k`，右下角保留尚未消去的 Schur 补。

## 算法映射

- 当前列主元搜索：CUDA 两级归约核；
- 当前列全零时的活动子矩阵全主元搜索：CUDA 两级归约核；
- 行/列交换与面板秩 1 更新：`cublasDswap`、`cublasDger`；
- 面板完成后的块行求解和 Schur 补更新：`cublasDtrsm`、`cublasDgemm`；
- L 乘子缩放：直接读取设备端 `A(k,k)` 的 CUDA kernel，省去每步额外的主元值 D2H 同步。

面板宽度仍为 64。若零列出现在面板中部，实现会先用 `DTRSM+DGEMM` 物化远端列，再搜索当前 Schur 补，并从该位置重启面板；这部分控制流与 CPU 版一致。

归约按活动子矩阵的列主序位置处理相等绝对值，稳定选择最靠前的候选；NaN 不作为可用主元。每个主元仍有一次 GPU 到 CPU 的候选同步，因为下一步的行/列交换和“是否进入全局回退”依赖该结果。因此性能收益主要来自大矩阵的 `DGEMM` 尾部更新；很小的矩阵通常不适合用这个接口。

## 接口

```cpp
// d_a 在设备端；ipiv/jpiv 在主机端。适合实际 GPU 工作流。
auto status = delayed_getrf_cuda_device(
    m, n, d_a, lda, ipiv, jpiv, info, stream);

// a 在主机端；便利接口会自动分配显存并执行 H2D/D2H。
auto status = delayed_getrf_cuda_host(
    m, n, a, lda, ipiv, jpiv, info, stream);
```

两个函数返回前都会同步传入的 stream。`status != success` 表示运行环境或参数错误，此时 `info=-1`；`status == success` 时才按上述延迟主元语义解释 `info`。

## 构建

要求 NVIDIA GPU、CUDA Toolkit（含 cuBLAS）和支持 CUDA 的 CMake。可单独构建：

```sh
cmake -S delayed_lu_cuda -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

请把 `80` 改成目标 GPU 的 compute capability，例如 75、86 或 90。也可以从仓库根目录构建；根 `CMakeLists.txt` 会检测 CUDA 编译器，存在时自动加入本目录，不存在时跳过，不影响原 CPU 版本。

`delayed_lu_cuda_example` 覆盖普通方阵、带延迟后缀的秩亏方阵、矩形矩阵，以及面板中途物化远端列的路径，并在 CPU 上重建 `P*A*Q` 检查误差。若仅有 CUDA Toolkit 而运行机器没有 GPU，CTest 会把运行时测试标记为跳过。

## 适用范围与限制

- 这是双精度稠密 LU，仅支持 NVIDIA CUDA/cuBLAS；Apple Silicon 和 AMD GPU 不能运行。
- 与 CPU 原型一样，主元接受条件是绝对值严格大于 0，没有额外阈值。若希望做数值阈值延迟，应把判断改为相对于列范数或矩阵范数的容差。
- cuSOLVER 的标准 `getrf` 更适合普通部分主元 LU，但它不提供这里的“零列时全主元回退并形成延迟后缀”语义，不能直接替换本实现。
