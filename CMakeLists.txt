cmake_minimum_required(VERSION 3.18)
project(delayed_pivoting LANGUAGES CXX)

enable_testing()

add_subdirectory(delayed_lu)
add_subdirectory(delayed-bunch-kaufman)
