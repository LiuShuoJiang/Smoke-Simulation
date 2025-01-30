# Copyright (c) 2025. Shuojiang Liu.

cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(Smoke_Simulation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

#set_target_properties(Smoke_Simulation PROPERTIES
#                      CUDA_SEPARABLE_COMPILATION ON)

include(cmake/CPM.cmake)

add_subdirectory(src)
