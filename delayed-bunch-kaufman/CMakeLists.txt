cmake_minimum_required(VERSION 3.18)
project(delayed_bunch_kaufman LANGUAGES CXX)

add_library(delayed_bunch_kaufman
  delayed_sytrf.cpp
  delayed_sytrf_blocked.cpp
  lapack_dsytrf.cpp
)
target_include_directories(delayed_bunch_kaufman PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(delayed_bunch_kaufman PUBLIC cxx_std_17)

if(APPLE)
  target_link_libraries(delayed_bunch_kaufman PUBLIC "-framework Accelerate")
else()
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  include(CheckIncludeFileCXX)
  check_include_file_cxx(cblas.h HAVE_CBLAS_H)
  if(NOT HAVE_CBLAS_H)
    message(FATAL_ERROR "cblas.h not found; install libopenblas-dev or openblas-devel")
  endif()
  target_link_libraries(delayed_bunch_kaufman PUBLIC BLAS::BLAS LAPACK::LAPACK)
endif()

add_executable(delayed_sytrf_example example.cpp)
target_link_libraries(delayed_sytrf_example PRIVATE delayed_bunch_kaufman)

enable_testing()
add_test(NAME delayed_sytrf_example COMMAND delayed_sytrf_example)
