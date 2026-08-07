cmake_minimum_required(VERSION 3.18)
project(delayed_pivoting LANGUAGES CXX)

enable_testing()

add_subdirectory(delayed_lu)
add_subdirectory(delayed-bunch-kaufman)

option(DELAYED_PIVOTING_BUILD_CUDA
       "Build the CUDA delayed LU implementation when a CUDA compiler exists"
       ON)
if(DELAYED_PIVOTING_BUILD_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_subdirectory(delayed_lu_cuda)
  else()
    message(STATUS "CUDA compiler not found; skipping delayed_lu_cuda")
  endif()
endif()
