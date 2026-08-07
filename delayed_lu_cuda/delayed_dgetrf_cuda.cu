#include "delayed_dgetrf_cuda.hpp"
#include "delayed_dgetrf_cuda_internal.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace delayed_lu_cuda_internal {

__device__ PivotCandidate better_candidate(const PivotCandidate lhs,
                                           const PivotCandidate rhs) {
  if (rhs.abs_value > lhs.abs_value ||
      (rhs.abs_value == lhs.abs_value && rhs.order < lhs.order)) {
    return rhs;
  }
  return lhs;
}

// 既可搜索当前列（active_cols=1），也可搜索整个活动子矩阵。
// 每个 thread block 输出一个局部候选，随后由第二个 kernel 完成最终归约。
__global__ void search_active_kernel(const double *a, const int lda,
                                     const int k, const int active_rows,
                                     const int active_cols,
                                     PivotCandidate *block_candidates) {
  __shared__ PivotCandidate shared[reduction_threads];

  const unsigned long long count =
      static_cast<unsigned long long>(active_rows) *
      static_cast<unsigned long long>(active_cols);
  const unsigned long long first =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;

  PivotCandidate best{k, k, 0.0, 0ULL};
  for (unsigned long long logical = first; logical < count;
       logical += stride) {
    const int row = k + static_cast<int>(logical % active_rows);
    const int col = k + static_cast<int>(logical / active_rows);
    const double magnitude =
        fabs(a[static_cast<std::size_t>(row) +
               static_cast<std::size_t>(col) * lda]);
    // 对 NaN，两个比较均为 false，因此它不会成为可接受主元。
    const PivotCandidate current{row, col, magnitude, logical};
    best = better_candidate(best, current);
  }

  shared[threadIdx.x] = best;
  __syncthreads();
  for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] = better_candidate(
          shared[threadIdx.x], shared[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    block_candidates[blockIdx.x] = shared[0];
  }
}

__global__ void finish_search_kernel(const PivotCandidate *block_candidates,
                                     const int candidate_count, const int k,
                                     PivotCandidate *result) {
  __shared__ PivotCandidate shared[reduction_threads];
  PivotCandidate best{k, k, 0.0, 0ULL};
  for (int i = threadIdx.x; i < candidate_count; i += blockDim.x) {
    best = better_candidate(best, block_candidates[i]);
  }
  shared[threadIdx.x] = best;
  __syncthreads();
  for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] = better_candidate(
          shared[threadIdx.x], shared[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *result = shared[0];
  }
}

// 主元值已经通过同一 stream 上的搜索和交换保证非零。让 kernel 直接读取
// A(k,k)，可以避免仅为把倒数传给 cublasDscal 而增加一次 D2H 同步。
__global__ void scale_pivot_column_kernel(double *a, const int lda,
                                          const int k, const int rows_below) {
  const double inverse_pivot =
      1.0 / a[static_cast<std::size_t>(k) + static_cast<std::size_t>(k) * lda];
  for (int relative = blockIdx.x * blockDim.x + threadIdx.x;
       relative < rows_below; relative += blockDim.x * gridDim.x) {
    const int row = k + 1 + relative;
    a[static_cast<std::size_t>(row) + static_cast<std::size_t>(k) * lda] *=
        inverse_pivot;
  }
}

class DeviceAllocation {
public:
  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation &) = delete;
  DeviceAllocation &operator=(const DeviceAllocation &) = delete;

  ~DeviceAllocation() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  cudaError_t allocate(const std::size_t bytes) {
    return cudaMalloc(&pointer_, bytes);
  }

  template <class T> T *as() { return static_cast<T *>(pointer_); }

private:
  void *pointer_ = nullptr;
};

class CublasHandle {
public:
  CublasHandle() = default;
  CublasHandle(const CublasHandle &) = delete;
  CublasHandle &operator=(const CublasHandle &) = delete;

  ~CublasHandle() {
    if (handle_ != nullptr) {
      cublasDestroy(handle_);
    }
  }

  cublasStatus_t create() { return cublasCreate(&handle_); }
  cublasHandle_t get() const { return handle_; }

private:
  cublasHandle_t handle_ = nullptr;
};

double *element(double *a, const int lda, const int row, const int col) {
  return a + static_cast<std::size_t>(row) +
         static_cast<std::size_t>(col) * static_cast<std::size_t>(lda);
}

int search_block_count(const unsigned long long value_count) {
  const unsigned long long required =
      (value_count + values_per_search_block - 1) / values_per_search_block;
  return static_cast<int>(
      std::max(1ULL, std::min<unsigned long long>(max_search_blocks, required)));
}

DelayedGetrfCudaStatus search_pivot(
    const double *d_a, const int lda, const int k, const int active_rows,
    const int active_cols, PivotCandidate *d_block_candidates,
    PivotCandidate *d_result, PivotCandidate &host_result,
    const cudaStream_t stream) {
  const unsigned long long value_count =
      static_cast<unsigned long long>(active_rows) *
      static_cast<unsigned long long>(active_cols);
  const int blocks = search_block_count(value_count);
  search_active_kernel<<<blocks, reduction_threads, 0, stream>>>(
      d_a, lda, k, active_rows, active_cols, d_block_candidates);
  if (cudaGetLastError() != cudaSuccess) {
    return DelayedGetrfCudaStatus::cuda_failure;
  }
  finish_search_kernel<<<1, reduction_threads, 0, stream>>>(
      d_block_candidates, blocks, k, d_result);
  if (cudaGetLastError() != cudaSuccess) {
    return DelayedGetrfCudaStatus::cuda_failure;
  }
  if (cudaMemcpyAsync(&host_result, d_result, sizeof(PivotCandidate),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    return DelayedGetrfCudaStatus::cuda_failure;
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return DelayedGetrfCudaStatus::cuda_failure;
  }
  return DelayedGetrfCudaStatus::success;
}

DelayedGetrfCudaStatus finish_panel(const cublasHandle_t handle, const int m,
                                    const int n, double *d_a, const int lda,
                                    const int begin, const int end) {
  const int width = end - begin;
  const int cols_right = n - end;
  if (width <= 0 || cols_right <= 0) {
    return DelayedGetrfCudaStatus::success;
  }

  const double one = 1.0;
  cublasStatus_t cublas_status = cublasDtrsm(
      handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
      CUBLAS_DIAG_UNIT, width, cols_right, &one,
      element(d_a, lda, begin, begin), lda, element(d_a, lda, begin, end),
      lda);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    return DelayedGetrfCudaStatus::cublas_failure;
  }

  const int rows_below = m - end;
  if (rows_below > 0) {
    const double minus_one = -1.0;
    cublas_status = cublasDgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, rows_below, cols_right, width,
        &minus_one, element(d_a, lda, end, begin), lda,
        element(d_a, lda, begin, end), lda, &one,
        element(d_a, lda, end, end), lda);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      return DelayedGetrfCudaStatus::cublas_failure;
    }
  }
  return DelayedGetrfCudaStatus::success;
}

DelayedGetrfCudaStatus materialize_partial_panel(
    const cublasHandle_t handle, const int m, const int n, double *d_a,
    const int lda, const int begin, const int k, const int panel_end) {
  const int width = k - begin;
  const int far_cols = n - panel_end;
  if (width <= 0 || far_cols <= 0) {
    return DelayedGetrfCudaStatus::success;
  }

  const double one = 1.0;
  cublasStatus_t cublas_status = cublasDtrsm(
      handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
      CUBLAS_DIAG_UNIT, width, far_cols, &one,
      element(d_a, lda, begin, begin), lda,
      element(d_a, lda, begin, panel_end), lda);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    return DelayedGetrfCudaStatus::cublas_failure;
  }

  const int active_rows = m - k;
  if (active_rows > 0) {
    const double minus_one = -1.0;
    cublas_status = cublasDgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, active_rows, far_cols, width,
        &minus_one, element(d_a, lda, k, begin), lda,
        element(d_a, lda, begin, panel_end), lda, &one,
        element(d_a, lda, k, panel_end), lda);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      return DelayedGetrfCudaStatus::cublas_failure;
    }
  }
  return DelayedGetrfCudaStatus::success;
}

DelayedGetrfCudaStatus factorize(
    const cublasHandle_t handle, const int m, const int n, double *d_a,
    const int lda, int *ipiv, int *jpiv, int &info,
    PivotCandidate *d_block_candidates, PivotCandidate *d_result,
    const cudaStream_t stream) {
  const int pivots = std::min(m, n);
  int panel_begin = 0;
  bool positioned_pivot = false;

  while (panel_begin < pivots) {
    const int panel_end = std::min(pivots, panel_begin + block_size);
    bool restart_panel = false;

    for (int k = panel_begin; k < panel_end; ++k) {
      if (positioned_pivot) {
        positioned_pivot = false;
      } else {
        PivotCandidate pivot{k, k, 0.0, 0ULL};
        DelayedGetrfCudaStatus status = search_pivot(
            d_a, lda, k, m - k, 1, d_block_candidates, d_result, pivot,
            stream);
        if (status != DelayedGetrfCudaStatus::success) {
          return status;
        }

        int pivot_row = pivot.row;
        int pivot_col = k;
        const bool zero_current_column = !(pivot.abs_value > 0.0);

        if (zero_current_column) {
          if (k > panel_begin) {
            status = materialize_partial_panel(handle, m, n, d_a, lda,
                                               panel_begin, k, panel_end);
            if (status != DelayedGetrfCudaStatus::success) {
              return status;
            }
          }

          status = search_pivot(d_a, lda, k, m - k, n - k,
                                d_block_candidates, d_result, pivot, stream);
          if (status != DelayedGetrfCudaStatus::success) {
            return status;
          }
          if (!(pivot.abs_value > 0.0)) {
            info = k;
            return DelayedGetrfCudaStatus::success;
          }
          pivot_row = pivot.row;
          pivot_col = pivot.col;
        }

        ipiv[k] = pivot_row;
        jpiv[k] = pivot_col;
        if (pivot_row != k) {
          if (cublasDswap(handle, n, element(d_a, lda, k, 0), lda,
                          element(d_a, lda, pivot_row, 0), lda) !=
              CUBLAS_STATUS_SUCCESS) {
            return DelayedGetrfCudaStatus::cublas_failure;
          }
        }
        if (pivot_col != k) {
          if (cublasDswap(handle, m, element(d_a, lda, 0, k), 1,
                          element(d_a, lda, 0, pivot_col), 1) !=
              CUBLAS_STATUS_SUCCESS) {
            return DelayedGetrfCudaStatus::cublas_failure;
          }
        }

        if (zero_current_column && k > panel_begin) {
          panel_begin = k;
          positioned_pivot = true;
          restart_panel = true;
          break;
        }
      }

      const int rows_below = m - k - 1;
      if (rows_below > 0) {
        const int blocks = std::min(
            max_search_blocks,
            (rows_below + reduction_threads - 1) / reduction_threads);
        scale_pivot_column_kernel<<<blocks, reduction_threads, 0, stream>>>(
            d_a, lda, k, rows_below);
        if (cudaGetLastError() != cudaSuccess) {
          return DelayedGetrfCudaStatus::cuda_failure;
        }
      }

      const int panel_cols_right = panel_end - k - 1;
      if (rows_below > 0 && panel_cols_right > 0) {
        const double minus_one = -1.0;
        if (cublasDger(handle, rows_below, panel_cols_right, &minus_one,
                       element(d_a, lda, k + 1, k), 1,
                       element(d_a, lda, k, k + 1), lda,
                       element(d_a, lda, k + 1, k + 1), lda) !=
            CUBLAS_STATUS_SUCCESS) {
          return DelayedGetrfCudaStatus::cublas_failure;
        }
      }
    }

    if (restart_panel) {
      continue;
    }

    const DelayedGetrfCudaStatus status =
        finish_panel(handle, m, n, d_a, lda, panel_begin, panel_end);
    if (status != DelayedGetrfCudaStatus::success) {
      return status;
    }
    panel_begin = panel_end;
  }

  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return DelayedGetrfCudaStatus::cuda_failure;
  }
  return DelayedGetrfCudaStatus::success;
}

bool matrix_byte_count(const int lda, const int n, std::size_t &bytes) {
  const std::size_t columns = static_cast<std::size_t>(n);
  const std::size_t leading = static_cast<std::size_t>(lda);
  if (columns != 0 &&
      leading > std::numeric_limits<std::size_t>::max() / columns) {
    return false;
  }
  const std::size_t elements = leading * columns;
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    return false;
  }
  bytes = elements * sizeof(double);
  return true;
}

bool valid_arguments(const int m, const int n, const double *a,
                     const int lda, const int *ipiv, const int *jpiv) {
  if (m < 0 || n < 0 || lda < std::max(1, m)) {
    return false;
  }
  const int pivots = std::min(m, n);
  if ((m > 0 && n > 0 && a == nullptr) ||
      (pivots > 0 && (ipiv == nullptr || jpiv == nullptr))) {
    return false;
  }
  std::size_t ignored = 0;
  return matrix_byte_count(lda, n, ignored);
}

} // namespace delayed_lu_cuda_internal

using namespace delayed_lu_cuda_internal;

DelayedGetrfCudaStatus delayed_getrf_cuda_device(
    const int m, const int n, double *d_a, const int lda, int *ipiv,
    int *jpiv, int &info, const cudaStream_t stream) {
  info = -1;
  if (!valid_arguments(m, n, d_a, lda, ipiv, jpiv)) {
    return DelayedGetrfCudaStatus::invalid_argument;
  }

  const int pivots = std::min(m, n);
  info = pivots;
  for (int k = 0; k < pivots; ++k) {
    ipiv[k] = k;
    jpiv[k] = k;
  }
  if (pivots == 0) {
    return DelayedGetrfCudaStatus::success;
  }

  DeviceAllocation block_candidates;
  DeviceAllocation result;
  if (block_candidates.allocate(max_search_blocks * sizeof(PivotCandidate)) !=
          cudaSuccess ||
      result.allocate(sizeof(PivotCandidate)) != cudaSuccess) {
    info = -1;
    return DelayedGetrfCudaStatus::allocation_failed;
  }

  CublasHandle handle;
  if (handle.create() != CUBLAS_STATUS_SUCCESS ||
      cublasSetStream(handle.get(), stream) != CUBLAS_STATUS_SUCCESS ||
      cublasSetPointerMode(handle.get(), CUBLAS_POINTER_MODE_HOST) !=
          CUBLAS_STATUS_SUCCESS) {
    info = -1;
    return DelayedGetrfCudaStatus::cublas_failure;
  }

  const DelayedGetrfCudaStatus status = factorize(
      handle.get(), m, n, d_a, lda, ipiv, jpiv, info,
      block_candidates.as<PivotCandidate>(), result.as<PivotCandidate>(),
      stream);
  if (status != DelayedGetrfCudaStatus::success) {
    info = -1;
  }
  return status;
}

DelayedGetrfCudaStatus delayed_getrf_cuda_host(
    const int m, const int n, double *a, const int lda, int *ipiv, int *jpiv,
    int &info, const cudaStream_t stream) {
  info = -1;
  if (!valid_arguments(m, n, a, lda, ipiv, jpiv)) {
    return DelayedGetrfCudaStatus::invalid_argument;
  }
  const int pivots = std::min(m, n);
  if (pivots == 0) {
    info = 0;
    return DelayedGetrfCudaStatus::success;
  }

  std::size_t bytes = 0;
  if (!matrix_byte_count(lda, n, bytes)) {
    return DelayedGetrfCudaStatus::invalid_argument;
  }
  DeviceAllocation matrix;
  if (matrix.allocate(bytes) != cudaSuccess) {
    return DelayedGetrfCudaStatus::allocation_failed;
  }
  double *d_a = matrix.as<double>();
  if (cudaMemcpyAsync(d_a, a, bytes, cudaMemcpyHostToDevice, stream) !=
      cudaSuccess) {
    return DelayedGetrfCudaStatus::cuda_failure;
  }

  const DelayedGetrfCudaStatus status = delayed_getrf_cuda_device(
      m, n, d_a, lda, ipiv, jpiv, info, stream);
  if (status != DelayedGetrfCudaStatus::success) {
    return status;
  }
  if (cudaMemcpyAsync(a, d_a, bytes, cudaMemcpyDeviceToHost, stream) !=
          cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    info = -1;
    return DelayedGetrfCudaStatus::cuda_failure;
  }
  return DelayedGetrfCudaStatus::success;
}

DelayedGetrfCudaStatus delayed_getrf_cuda_native_device(
    const int m, const int n, double *d_a, const int lda, int *ipiv,
    int *jpiv, int &info, const cudaStream_t stream) {
  return delayed_getrf_cuda_device(m, n, d_a, lda, ipiv, jpiv, info, stream);
}

DelayedGetrfCudaStatus delayed_getrf_cuda_native_host(
    const int m, const int n, double *a, const int lda, int *ipiv, int *jpiv,
    int &info, const cudaStream_t stream) {
  return delayed_getrf_cuda_host(m, n, a, lda, ipiv, jpiv, info, stream);
}

const char *delayed_getrf_cuda_status_string(
    const DelayedGetrfCudaStatus status) {
  switch (status) {
  case DelayedGetrfCudaStatus::success:
    return "success";
  case DelayedGetrfCudaStatus::invalid_argument:
    return "invalid argument";
  case DelayedGetrfCudaStatus::allocation_failed:
    return "CUDA allocation failed";
  case DelayedGetrfCudaStatus::cuda_failure:
    return "CUDA runtime failure";
  case DelayedGetrfCudaStatus::cublas_failure:
    return "cuBLAS failure";
  }
  return "unknown status";
}
