#include "delayed_dgetrf_cuda.hpp"
#include "delayed_dgetrf_cuda_internal.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace {

using delayed_lu_cuda_internal::PivotCandidate;
using delayed_lu_cuda_internal::block_size;
using delayed_lu_cuda_internal::materialize_partial_panel;
using delayed_lu_cuda_internal::max_search_blocks;
using delayed_lu_cuda_internal::search_pivot;

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

class PinnedAllocation {
public:
  PinnedAllocation() = default;
  PinnedAllocation(const PinnedAllocation &) = delete;
  PinnedAllocation &operator=(const PinnedAllocation &) = delete;
  ~PinnedAllocation() {
    if (pointer_ != nullptr) {
      cudaFreeHost(pointer_);
    }
  }

  cudaError_t allocate(const std::size_t bytes) {
    return cudaMallocHost(&pointer_, bytes);
  }
  template <class T> T *as() { return static_cast<T *>(pointer_); }

private:
  void *pointer_ = nullptr;
};

class OwnedStream {
public:
  OwnedStream() = default;
  OwnedStream(const OwnedStream &) = delete;
  OwnedStream &operator=(const OwnedStream &) = delete;
  ~OwnedStream() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  cudaError_t create() {
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  }
  cudaStream_t get() const { return stream_; }

private:
  cudaStream_t stream_ = nullptr;
};

class OwnedEvent {
public:
  OwnedEvent() = default;
  OwnedEvent(const OwnedEvent &) = delete;
  OwnedEvent &operator=(const OwnedEvent &) = delete;
  ~OwnedEvent() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }

  cudaError_t create() {
    return cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
  }
  cudaEvent_t get() const { return event_; }

private:
  cudaEvent_t event_ = nullptr;
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

  cublasStatus_t create(const cudaStream_t stream) {
    cublasStatus_t status = cublasCreate(&handle_);
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasSetStream(handle_, stream);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST);
    }
    return status;
  }
  cublasHandle_t get() const { return handle_; }

private:
  cublasHandle_t handle_ = nullptr;
};

double *element(double *a, const int lda, const int row, const int col) {
  return a + static_cast<std::size_t>(row) +
         static_cast<std::size_t>(col) * static_cast<std::size_t>(lda);
}

DelayedGetrfCudaStatus copy_panel_to_host(
    const double *d_a, const int lda, const int begin, const int end,
    const int m, double *host_panel, const int host_lda,
    const cudaStream_t stream) {
  const std::size_t column_bytes =
      static_cast<std::size_t>(m - begin) * sizeof(double);
  const std::size_t source_pitch = static_cast<std::size_t>(lda) * sizeof(double);
  const std::size_t host_pitch =
      static_cast<std::size_t>(host_lda) * sizeof(double);
  const cudaError_t status = cudaMemcpy2DAsync(
      host_panel, host_pitch,
      d_a + static_cast<std::size_t>(begin) +
          static_cast<std::size_t>(begin) * lda,
      source_pitch, column_bytes, end - begin, cudaMemcpyDeviceToHost, stream);
  return status == cudaSuccess ? DelayedGetrfCudaStatus::success
                               : DelayedGetrfCudaStatus::cuda_failure;
}

DelayedGetrfCudaStatus copy_panel_to_device(
    const double *host_panel, const int host_lda, const int begin,
    const int end, const int m, double *d_a, const int lda,
    const cudaStream_t stream) {
  const std::size_t column_bytes =
      static_cast<std::size_t>(m - begin) * sizeof(double);
  const std::size_t destination_pitch =
      static_cast<std::size_t>(lda) * sizeof(double);
  const std::size_t host_pitch =
      static_cast<std::size_t>(host_lda) * sizeof(double);
  const cudaError_t status = cudaMemcpy2DAsync(
      element(d_a, lda, begin, begin), destination_pitch, host_panel,
      host_pitch, column_bytes, end - begin, cudaMemcpyHostToDevice, stream);
  return status == cudaSuccess ? DelayedGetrfCudaStatus::success
                               : DelayedGetrfCudaStatus::cuda_failure;
}

PivotCandidate find_host_column_pivot(const double *panel,
                                      const int panel_lda,
                                      const int panel_begin, const int k,
                                      const int m) {
  const int local_col = k - panel_begin;
  PivotCandidate candidate{k, k, 0.0, 0ULL};
  for (int row = k; row < m; ++row) {
    const int local_row = row - panel_begin;
    const double magnitude =
        std::abs(panel[static_cast<std::size_t>(local_row) +
                       static_cast<std::size_t>(local_col) * panel_lda]);
    if (magnitude > candidate.abs_value) {
      candidate = {row, k, magnitude,
                   static_cast<unsigned long long>(row - k)};
    }
  }
  return candidate;
}

void swap_host_panel_rows(double *panel, const int panel_lda,
                          const int panel_width, const int first,
                          const int second) {
  if (first == second) {
    return;
  }
  for (int col = 0; col < panel_width; ++col) {
    std::swap(panel[static_cast<std::size_t>(first) +
                    static_cast<std::size_t>(col) * panel_lda],
              panel[static_cast<std::size_t>(second) +
                    static_cast<std::size_t>(col) * panel_lda]);
  }
}

void eliminate_host_panel_column(double *panel, const int panel_lda,
                                 const int panel_width,
                                 const int local_k) {
  const double inverse_pivot =
      1.0 / panel[static_cast<std::size_t>(local_k) +
                  static_cast<std::size_t>(local_k) * panel_lda];
  for (int row = local_k + 1; row < panel_lda; ++row) {
    panel[static_cast<std::size_t>(row) +
          static_cast<std::size_t>(local_k) * panel_lda] *= inverse_pivot;
  }

  for (int col = local_k + 1; col < panel_width; ++col) {
    const double upper =
        panel[static_cast<std::size_t>(local_k) +
              static_cast<std::size_t>(col) * panel_lda];
    for (int row = local_k + 1; row < panel_lda; ++row) {
      panel[static_cast<std::size_t>(row) +
            static_cast<std::size_t>(col) * panel_lda] -=
          panel[static_cast<std::size_t>(row) +
                static_cast<std::size_t>(local_k) * panel_lda] *
          upper;
    }
  }
}

DelayedGetrfCudaStatus apply_row_swaps(const cublasHandle_t handle,
                                       const int n, double *d_a,
                                       const int lda, const int begin,
                                       const int end, const int *ipiv) {
  for (int k = begin; k < end; ++k) {
    if (ipiv[k] != k &&
        cublasDswap(handle, n, element(d_a, lda, k, 0), lda,
                    element(d_a, lda, ipiv[k], 0), lda) !=
            CUBLAS_STATUS_SUCCESS) {
      return DelayedGetrfCudaStatus::cublas_failure;
    }
  }
  return DelayedGetrfCudaStatus::success;
}

DelayedGetrfCudaStatus flush_host_panel(
    const cublasHandle_t panel_handle, const int m, const int n, double *d_a,
    const int lda, const int panel_begin, const int panel_end,
    const int accepted_end, const int *ipiv, const double *host_panel,
    const int host_lda, const cudaStream_t panel_stream) {
  DelayedGetrfCudaStatus status = apply_row_swaps(
      panel_handle, n, d_a, lda, panel_begin, accepted_end, ipiv);
  if (status != DelayedGetrfCudaStatus::success) {
    return status;
  }
  return copy_panel_to_device(host_panel, host_lda, panel_begin, panel_end, m,
                              d_a, lda, panel_stream);
}

DelayedGetrfCudaStatus schedule_trailing_update_and_prefetch(
    const cublasHandle_t panel_handle, const cublasHandle_t update_handle,
    const cudaStream_t panel_stream, const cudaStream_t update_stream,
    const cudaEvent_t trsm_done, const int m, const int n, double *d_a,
    const int lda, const int pivots, const int begin, const int end,
    double *host_panel, bool &prefetched) {
  prefetched = false;
  const int width = end - begin;
  const int cols_right = n - end;
  if (width <= 0) {
    return DelayedGetrfCudaStatus::success;
  }

  const double one = 1.0;
  if (cols_right > 0 &&
      cublasDtrsm(panel_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                  CUBLAS_OP_N, CUBLAS_DIAG_UNIT, width, cols_right, &one,
                  element(d_a, lda, begin, begin), lda,
                  element(d_a, lda, begin, end), lda) !=
          CUBLAS_STATUS_SUCCESS) {
    return DelayedGetrfCudaStatus::cublas_failure;
  }

  const int next_end = std::min(pivots, end + block_size);
  const int next_cols = next_end - end;
  const int rows_below = m - end;
  if (rows_below > 0 && cols_right > 0) {
    if (cudaEventRecord(trsm_done, panel_stream) != cudaSuccess) {
      return DelayedGetrfCudaStatus::cuda_failure;
    }
    const double minus_one = -1.0;
    if (next_cols > 0 &&
        cublasDgemm(panel_handle, CUBLAS_OP_N, CUBLAS_OP_N, rows_below,
                    next_cols, width, &minus_one,
                    element(d_a, lda, end, begin), lda,
                    element(d_a, lda, begin, end), lda, &one,
                    element(d_a, lda, end, end), lda) !=
            CUBLAS_STATUS_SUCCESS) {
      return DelayedGetrfCudaStatus::cublas_failure;
    }

    const int far_cols = n - next_end;
    if (far_cols > 0) {
      if (cudaStreamWaitEvent(update_stream, trsm_done, 0) != cudaSuccess) {
        return DelayedGetrfCudaStatus::cuda_failure;
      }
      if (cublasDgemm(update_handle, CUBLAS_OP_N, CUBLAS_OP_N, rows_below,
                      far_cols, width, &minus_one,
                      element(d_a, lda, end, begin), lda,
                      element(d_a, lda, begin, next_end), lda, &one,
                      element(d_a, lda, end, next_end), lda) !=
          CUBLAS_STATUS_SUCCESS) {
        return DelayedGetrfCudaStatus::cublas_failure;
      }
    }
  }

  if (next_cols > 0) {
    const DelayedGetrfCudaStatus status = copy_panel_to_host(
        d_a, lda, end, next_end, m, host_panel, m - end, panel_stream);
    if (status != DelayedGetrfCudaStatus::success) {
      return status;
    }
    prefetched = true;
  }
  return DelayedGetrfCudaStatus::success;
}

DelayedGetrfCudaStatus factorize_hybrid(
    const cublasHandle_t panel_handle, const cublasHandle_t update_handle,
    const cudaStream_t panel_stream, const cudaStream_t update_stream,
    const cudaEvent_t trsm_done, const int m, const int n, double *d_a,
    const int lda, int *ipiv, int *jpiv, int &info, double *host_panel,
    PivotCandidate *d_block_candidates, PivotCandidate *d_result) {
  const int pivots = std::min(m, n);
  int panel_begin = 0;
  bool positioned_pivot = false;
  bool prefetched = false;

  while (panel_begin < pivots) {
    const int panel_end = std::min(pivots, panel_begin + block_size);
    const int panel_lda = m - panel_begin;
    const int panel_width = panel_end - panel_begin;
    if (!prefetched) {
      const DelayedGetrfCudaStatus status = copy_panel_to_host(
          d_a, lda, panel_begin, panel_end, m, host_panel, panel_lda,
          panel_stream);
      if (status != DelayedGetrfCudaStatus::success) {
        return status;
      }
    }
    if (cudaStreamSynchronize(panel_stream) != cudaSuccess) {
      return DelayedGetrfCudaStatus::cuda_failure;
    }
    prefetched = false;
    bool restart_panel = false;

    for (int k = panel_begin; k < panel_end; ++k) {
      if (positioned_pivot) {
        positioned_pivot = false;
      } else {
        const PivotCandidate column_pivot = find_host_column_pivot(
            host_panel, panel_lda, panel_begin, k, m);
        if (!(column_pivot.abs_value > 0.0)) {
          // far update 与 CPU 面板分解并行；全局搜索前必须先完成它。
          if (cudaStreamSynchronize(update_stream) != cudaSuccess) {
            return DelayedGetrfCudaStatus::cuda_failure;
          }
          DelayedGetrfCudaStatus status = flush_host_panel(
              panel_handle, m, n, d_a, lda, panel_begin, panel_end, k, ipiv,
              host_panel, panel_lda, panel_stream);
          if (status != DelayedGetrfCudaStatus::success) {
            return status;
          }
          if (k > panel_begin) {
            status = materialize_partial_panel(panel_handle, m, n, d_a, lda,
                                               panel_begin, k, panel_end);
            if (status != DelayedGetrfCudaStatus::success) {
              return status;
            }
          }

          PivotCandidate global_pivot{k, k, 0.0, 0ULL};
          status = search_pivot(d_a, lda, k, m - k, n - k,
                                d_block_candidates, d_result, global_pivot,
                                panel_stream);
          if (status != DelayedGetrfCudaStatus::success) {
            return status;
          }
          if (!(global_pivot.abs_value > 0.0)) {
            info = k;
            return DelayedGetrfCudaStatus::success;
          }

          ipiv[k] = global_pivot.row;
          jpiv[k] = global_pivot.col;
          if (global_pivot.row != k &&
              cublasDswap(panel_handle, n, element(d_a, lda, k, 0), lda,
                          element(d_a, lda, global_pivot.row, 0), lda) !=
                  CUBLAS_STATUS_SUCCESS) {
            return DelayedGetrfCudaStatus::cublas_failure;
          }
          if (global_pivot.col != k &&
              cublasDswap(panel_handle, m, element(d_a, lda, 0, k), 1,
                          element(d_a, lda, 0, global_pivot.col), 1) !=
                  CUBLAS_STATUS_SUCCESS) {
            return DelayedGetrfCudaStatus::cublas_failure;
          }

          panel_begin = k;
          const int restarted_end =
              std::min(pivots, panel_begin + block_size);
          status = copy_panel_to_host(d_a, lda, panel_begin, restarted_end, m,
                                      host_panel, m - panel_begin,
                                      panel_stream);
          if (status != DelayedGetrfCudaStatus::success) {
            return status;
          }
          positioned_pivot = true;
          prefetched = true;
          restart_panel = true;
          break;
        }

        ipiv[k] = column_pivot.row;
        jpiv[k] = k;
        swap_host_panel_rows(host_panel, panel_lda, panel_width,
                             k - panel_begin,
                             column_pivot.row - panel_begin);
      }

      eliminate_host_panel_column(host_panel, panel_lda, panel_width,
                                  k - panel_begin);
    }

    if (restart_panel) {
      continue;
    }

    // 等待上一面板的远端 DGEMM，然后才可对完整 GPU 行执行本面板交换。
    if (cudaStreamSynchronize(update_stream) != cudaSuccess) {
      return DelayedGetrfCudaStatus::cuda_failure;
    }
    DelayedGetrfCudaStatus status = flush_host_panel(
        panel_handle, m, n, d_a, lda, panel_begin, panel_end, panel_end, ipiv,
        host_panel, panel_lda, panel_stream);
    if (status != DelayedGetrfCudaStatus::success) {
      return status;
    }
    status = schedule_trailing_update_and_prefetch(
        panel_handle, update_handle, panel_stream, update_stream, trsm_done, m,
        n, d_a, lda, pivots, panel_begin, panel_end, host_panel, prefetched);
    if (status != DelayedGetrfCudaStatus::success) {
      return status;
    }
    panel_begin = panel_end;
  }

  if (cudaStreamSynchronize(panel_stream) != cudaSuccess ||
      cudaStreamSynchronize(update_stream) != cudaSuccess) {
    return DelayedGetrfCudaStatus::cuda_failure;
  }
  return DelayedGetrfCudaStatus::success;
}

bool panel_byte_count(const int m, const int pivots, std::size_t &bytes) {
  const std::size_t rows = static_cast<std::size_t>(m);
  const std::size_t columns =
      static_cast<std::size_t>(std::min(block_size, pivots));
  if (columns != 0 &&
      rows > std::numeric_limits<std::size_t>::max() / columns) {
    return false;
  }
  const std::size_t elements = rows * columns;
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    return false;
  }
  bytes = elements * sizeof(double);
  return true;
}

} // namespace

DelayedGetrfCudaStatus delayed_getrf_cuda_hybrid_device(
    const int m, const int n, double *d_a, const int lda, int *ipiv,
    int *jpiv, int &info, const cudaStream_t stream) {
  info = -1;
  if (!delayed_lu_cuda_internal::valid_arguments(m, n, d_a, lda, ipiv,
                                                  jpiv)) {
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

  std::size_t panel_bytes = 0;
  if (!panel_byte_count(m, pivots, panel_bytes)) {
    return DelayedGetrfCudaStatus::invalid_argument;
  }

  PinnedAllocation host_panel;
  DeviceAllocation block_candidates;
  DeviceAllocation result;
  OwnedStream update_stream;
  OwnedEvent trsm_done;
  if (host_panel.allocate(panel_bytes) != cudaSuccess ||
      block_candidates.allocate(max_search_blocks * sizeof(PivotCandidate)) !=
          cudaSuccess ||
      result.allocate(sizeof(PivotCandidate)) != cudaSuccess) {
    info = -1;
    return DelayedGetrfCudaStatus::allocation_failed;
  }
  if (update_stream.create() != cudaSuccess ||
      trsm_done.create() != cudaSuccess) {
    info = -1;
    return DelayedGetrfCudaStatus::cuda_failure;
  }

  CublasHandle panel_handle;
  CublasHandle update_handle;
  if (panel_handle.create(stream) != CUBLAS_STATUS_SUCCESS ||
      update_handle.create(update_stream.get()) != CUBLAS_STATUS_SUCCESS) {
    info = -1;
    return DelayedGetrfCudaStatus::cublas_failure;
  }

  const DelayedGetrfCudaStatus status = factorize_hybrid(
      panel_handle.get(), update_handle.get(), stream, update_stream.get(),
      trsm_done.get(), m, n, d_a, lda, ipiv, jpiv, info,
      host_panel.as<double>(), block_candidates.as<PivotCandidate>(),
      result.as<PivotCandidate>());
  if (status != DelayedGetrfCudaStatus::success) {
    info = -1;
  }
  return status;
}

DelayedGetrfCudaStatus delayed_getrf_cuda_hybrid_host(
    const int m, const int n, double *a, const int lda, int *ipiv, int *jpiv,
    int &info, const cudaStream_t stream) {
  info = -1;
  if (!delayed_lu_cuda_internal::valid_arguments(m, n, a, lda, ipiv, jpiv)) {
    return DelayedGetrfCudaStatus::invalid_argument;
  }
  const int pivots = std::min(m, n);
  if (pivots == 0) {
    info = 0;
    return DelayedGetrfCudaStatus::success;
  }

  std::size_t bytes = 0;
  if (!delayed_lu_cuda_internal::matrix_byte_count(lda, n, bytes)) {
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

  const DelayedGetrfCudaStatus status = delayed_getrf_cuda_hybrid_device(
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
