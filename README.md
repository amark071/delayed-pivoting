# 延迟主元分解

这个仓库保存使用 C++17、BLAS/LAPACK 或 CUDA/cuBLAS 实现的延迟主元稠密分解：

- [`delayed_lu`](delayed_lu)：非对称矩阵的分块延迟主元 LU，包含全主元回退、零列后移并继续采用列主元，以及 LAPACK `DGETRF` 对照实现。
- [`delayed_lu_cuda`](delayed_lu_cuda)：`delayed_dgetrf.cpp` 的 NVIDIA CUDA 版本，保留全主元回退和延迟后缀语义，并提供设备内存与主机内存接口。
- [`delayed-bunch-kaufman`](delayed-bunch-kaufman)：对称不定矩阵的延迟主元 Bunch--Kaufman，包含非分块实现、面板化实现和 LAPACK `DSYTRF` 对照实现。

三个子目录保持各自独立的接口、示例、测试与说明文档，也可以从仓库根目录统一构建。

## 统一构建

macOS 使用系统自带的 Accelerate：

```sh
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

Linux 需要提供 CBLAS 和 LAPACK：

```sh
sudo apt install cmake g++ libopenblas-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

根构建会自动检测 CUDA 编译器；检测到时同时构建 `delayed_lu_cuda`，否则安全跳过。也可用 `-DDELAYED_PIVOTING_BUILD_CUDA=OFF` 明确关闭。CUDA 子目录的独立构建方法见其 README。

如果希望直接输出对比结果： 
```sh 
./build/delayed_lu/delayed_lu_example 
./build/delayed-bunch-kaufman/delayed_sytrf_example 
```

## 目录结构

```text
delayed-pivoting/
├── delayed_lu/
│   ├── delayed_dgetrf.cpp
│   ├── delayed_dgetrf_column.cpp
│   ├── lapack_dgetrf.cpp
│   └── example.cpp
├── delayed_lu_cuda/
│   ├── delayed_dgetrf_cuda.cu
│   ├── delayed_dgetrf_cuda.hpp
│   └── example.cu
└── delayed-bunch-kaufman/
    ├── delayed_sytrf.cpp
    ├── delayed_sytrf_blocked.cpp
    ├── lapack_dsytrf.cpp
    └── example.cpp
```

矩阵统一采用列主序，公开下标和置换记录统一采用 0-based。具体接口、延迟区间语义、紧凑因子存储和验证方式见两个子目录中的 `README.md`。
