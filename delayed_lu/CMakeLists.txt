cmake_minimum_required(VERSION 3.18)
project(delayed_lu LANGUAGES CXX)

add_library(delayed_lu
  delayed_dgetrf.cpp
  delayed_dgetrf_column.cpp
  lapack_dgetrf.cpp
)
target_include_directories(delayed_lu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(delayed_lu PUBLIC cxx_std_17)

if(APPLE)
  target_link_libraries(delayed_lu PUBLIC "-framework Accelerate")
else()
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  include(CheckIncludeFileCXX)
  check_include_file_cxx(cblas.h HAVE_CBLAS_H)
  if(NOT HAVE_CBLAS_H)
    message(FATAL_ERROR "cblas.h not found; install libopenblas-dev (Debian/Ubuntu) or openblas-devel (Fedora/RHEL)")
  endif()
  target_link_libraries(delayed_lu PUBLIC BLAS::BLAS LAPACK::LAPACK)
endif()

add_executable(delayed_lu_example example.cpp)
target_link_libraries(delayed_lu_example PRIVATE delayed_lu)

enable_testing()
add_test(NAME delayed_lu_example COMMAND delayed_lu_example)
