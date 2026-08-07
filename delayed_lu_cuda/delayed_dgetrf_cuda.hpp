#pragma once

#include <cuda_runtime_api.h>

// CUDA 版本同时报告参数、CUDA runtime 和 cuBLAS 三类错误。
// info 仍只描述 LU/延迟主元语义；status != success 时 info == -1。
enum class DelayedGetrfCudaStatus {
  success = 0,
  invalid_argument,
  allocation_failed,
  cuda_failure,
  cublas_failure,
};

/**
 * @brief GPU-native 版本：主元归约和数值更新均在 GPU 上执行。
 *
 * d_a 必须指向设备端列主序矩阵；ipiv、jpiv 位于主机端。函数在 stream
 * 上执行，但返回前会同步该 stream，因此矩阵和交换记录在返回时均可用。
 *
 * 正常完成时 info == min(m,n)；若从第 k 个主元起需要延迟，则 info == k。
 * ipiv/jpiv 保存 0-based 的逐步交换历史，分解满足 P*A*Q = L*U。
 */
DelayedGetrfCudaStatus delayed_getrf_cuda_device(
    int m, int n, double *d_a, int lda, int *ipiv, int *jpiv, int &info,
    cudaStream_t stream = nullptr);

/**
 * @brief 主机内存便利接口。
 *
 * 本函数分配临时显存，将 lda*n 个 double 复制到 GPU，调用设备接口，再把
 * 原位 LU/Schur 补结果复制回 a。重复分解或矩阵本来就在 GPU 上时，应直接
 * 使用 delayed_getrf_cuda_device 以避免 PCIe 传输和重复分配。
 */
DelayedGetrfCudaStatus delayed_getrf_cuda_host(
    int m, int n, double *a, int lda, int *ipiv, int *jpiv, int &info,
    cudaStream_t stream = nullptr);

// 显式 native 名称。上面两个无后缀接口为向后兼容别名。
DelayedGetrfCudaStatus delayed_getrf_cuda_native_device(
    int m, int n, double *d_a, int lda, int *ipiv, int *jpiv, int &info,
    cudaStream_t stream = nullptr);

DelayedGetrfCudaStatus delayed_getrf_cuda_native_host(
    int m, int n, double *a, int lda, int *ipiv, int *jpiv, int &info,
    cudaStream_t stream = nullptr);

/**
 * @brief Hybrid 版本：CPU 分解 pinned-memory 面板，GPU 更新尾部矩阵。
 *
 * 正常面板使用双 CUDA stream 做 look-ahead：GPU 更新下一面板并回传 CPU
 * 的同时，在另一 stream 更新更远的尾部列。若 CPU 遇到零列，活动子矩阵
 * 的全主元搜索仍在 GPU 上执行，不会把整个 Schur 补复制回主机。
 */
DelayedGetrfCudaStatus delayed_getrf_cuda_hybrid_device(
    int m, int n, double *d_a, int lda, int *ipiv, int *jpiv, int &info,
    cudaStream_t stream = nullptr);

DelayedGetrfCudaStatus delayed_getrf_cuda_hybrid_host(
    int m, int n, double *a, int lda, int *ipiv, int *jpiv, int &info,
    cudaStream_t stream = nullptr);

const char *delayed_getrf_cuda_status_string(DelayedGetrfCudaStatus status);
