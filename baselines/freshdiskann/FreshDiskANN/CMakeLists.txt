# FreshDiskANN library (libdiskann_s.a) for the NAVIS baselines; Linux only.
cmake_minimum_required(VERSION 3.16)
project(diskann CXX)

set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -DNDEBUG -march=native -mtune=native -ftree-vectorize")

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include_directories(include include/tsl)
# Linux AIO for disk I/O; Intel MKL (ILP64) for PQ and distance kernels.
add_definitions(-DUSE_AIO -DMKL_ILP64 -DUSE_ACCELERATED_PQ -DUSE_AVX2)
add_compile_options(-g -O3 -m64 -march=native -Wall -Wfatal-errors -funroll-loops
    -fopenmp -fopenmp-simd
    -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free)

add_subdirectory(src)
