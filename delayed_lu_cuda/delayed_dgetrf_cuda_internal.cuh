#pragma once

#include "delayed_dgetrf_cuda.hpp"

#include <cublas_v2.h>

#include <cstddef>

namespace delayed_lu_cuda_internal {

inline constexpr int block_size = 64;
inline constexpr int reduction_threads = 256;
inline constexpr int max_search_blocks = 1024;
inline constexpr int values_per_search_block = reduction_threads * 8;

struct PivotCandidate {
  int row;
  int col;
  double abs_value;
  unsigned long long order;
};

DelayedGetrfCudaStatus search_pivot(
    const double *d_a, int lda, int k, int active_rows, int active_cols,
    PivotCandidate *d_block_candidates, PivotCandidate *d_result,
    PivotCandidate &host_result, cudaStream_t stream);

DelayedGetrfCudaStatus materialize_partial_panel(
    cublasHandle_t handle, int m, int n, double *d_a, int lda, int begin,
    int k, int panel_end);

bool matrix_byte_count(int lda, int n, std::size_t &bytes);

bool valid_arguments(int m, int n, const double *a, int lda, const int *ipiv,
                     const int *jpiv);

} // namespace delayed_lu_cuda_internal
