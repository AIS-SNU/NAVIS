cmake_minimum_required(VERSION 3.16)
project(pipeann CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# liburing install prefix (include/, lib/liburing.a), built by baselines/build.sh.
set(LIBURING_DIR "" CACHE PATH "liburing 2.10 install prefix")
if(NOT EXISTS "${LIBURING_DIR}/lib/liburing.a")
    message(FATAL_ERROR "LIBURING_DIR must point to a liburing install (run baselines/build.sh)")
endif()

include_directories(${PROJECT_SOURCE_DIR}/include ${LIBURING_DIR}/include)

add_compile_options(
    -fopenmp -fopenmp-simd
    -Wall -Wextra -Wfatal-errors
    -Wno-class-memaccess -Wno-unused-variable -Wno-unused-parameter -Wno-unused-but-set-variable
    -g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops)

# OdinANN configuration used in the NAVIS paper: background I/O thread and
# delta pruning (OdinANN's optimizations), overlapped init, dynamic pipe width.
add_definitions(-DNDEBUG -DBG_IO_THREAD -DDELTA_PRUNING -DOVERLAP_INIT -DDYN_PIPE_WIDTH)

link_libraries(${LIBURING_DIR}/lib/liburing.a jemalloc glog)

# AVX-512 / AVX2 distance kernels, chosen from the compiler and the host CPU.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
try_run(RUN_RESULT COMPILE_RESULT "${CMAKE_BINARY_DIR}"
        SOURCES "${PROJECT_SOURCE_DIR}/tests/test_cpu.cpp")
if(COMPILER_SUPPORTS_AVX512F AND COMPILE_RESULT AND RUN_RESULT EQUAL 3)
    add_definitions(-DUSE_AVX512)
elseif(COMPILER_SUPPORTS_AVX2 AND COMPILE_RESULT AND RUN_RESULT EQUAL 1)
    add_definitions(-DUSE_AVX2)
endif()

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(BLAS REQUIRED)
link_libraries(${BLAS_LIBRARIES})

add_subdirectory(src)
add_subdirectory(tests)
