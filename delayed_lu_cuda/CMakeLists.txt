cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(delayed_lu_cuda LANGUAGES CXX CUDA)
  enable_testing()
endif()

find_package(CUDAToolkit REQUIRED)

add_library(delayed_lu_cuda
  delayed_dgetrf_cuda.cu
  delayed_dgetrf_cuda_hybrid.cu
)
target_include_directories(delayed_lu_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(delayed_lu_cuda PUBLIC cxx_std_17 cuda_std_17)
target_link_libraries(delayed_lu_cuda PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(delayed_lu_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

add_executable(delayed_lu_cuda_example example.cu)
target_link_libraries(delayed_lu_cuda_example PRIVATE delayed_lu_cuda)

add_executable(delayed_lu_cuda_benchmark benchmark.cu)
target_link_libraries(delayed_lu_cuda_benchmark PRIVATE delayed_lu_cuda)

add_test(NAME delayed_lu_cuda_example COMMAND delayed_lu_cuda_example)
set_tests_properties(delayed_lu_cuda_example PROPERTIES SKIP_RETURN_CODE 77)
