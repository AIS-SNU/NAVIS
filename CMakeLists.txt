cmake_minimum_required(VERSION 3.24)
project(navis LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
find_library(JEMALLOC_LIBRARY jemalloc REQUIRED)

# ===========================================================================
# Third-party dependencies (fetched at pinned versions)
# ===========================================================================
include(FetchContent)
include(ExternalProject)

# Header-only libraries: SOURCE_SUBDIR points at a non-existent directory so
# only the sources are fetched, without configuring their CMake projects.
FetchContent_Declare(robin_map
    URL https://github.com/Tessil/robin-map/archive/refs/tags/v1.3.0.tar.gz
    URL_HASH SHA256=a8424ad3b0affd4c57ed26f0f3d8a29604f0e1f2ef2089f497f614b1c94c7236
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR _headers_only)
FetchContent_Declare(libcuckoo
    URL https://github.com/efficient/libcuckoo/archive/864c388b77281985e10c911ae9d268183a00ff13.tar.gz
    URL_HASH SHA256=865308565ecff9acdf1115489773781d877035b995ad01fb088194da74ccb54d
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR _headers_only)
FetchContent_MakeAvailable(robin_map libcuckoo)

# liburing 2.10 (autotools), linked statically.
set(LIBURING_PREFIX ${CMAKE_BINARY_DIR}/_deps/liburing-install)
ExternalProject_Add(liburing_ep
    URL https://github.com/axboe/liburing/archive/refs/tags/liburing-2.10.tar.gz
    URL_HASH SHA256=0a687616a6886cd82b746b79c4e33dc40b8d7c0c6e24d0f6f3fd7cf41886bf53
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    BUILD_IN_SOURCE TRUE
    CONFIGURE_COMMAND ./configure --prefix=${LIBURING_PREFIX}
    BUILD_COMMAND make -C src
    INSTALL_COMMAND make -C src install
    BUILD_BYPRODUCTS ${LIBURING_PREFIX}/lib/liburing.a)
file(MAKE_DIRECTORY ${LIBURING_PREFIX}/include)
add_library(uring STATIC IMPORTED)
set_target_properties(uring PROPERTIES
    IMPORTED_LOCATION ${LIBURING_PREFIX}/lib/liburing.a
    INTERFACE_INCLUDE_DIRECTORIES ${LIBURING_PREFIX}/include)
add_dependencies(uring liburing_ep)

# ===========================================================================
# NAVIS library
# ===========================================================================
file(GLOB_RECURSE NAVIS_SOURCES src/*.cpp src/*.c)
add_library(navis_lib STATIC ${NAVIS_SOURCES})
target_include_directories(navis_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${robin_map_SOURCE_DIR}/include
    ${libcuckoo_SOURCE_DIR})
target_compile_options(navis_lib PRIVATE
    -O3 -march=native -mtune=native
    -fopenmp-simd -ftree-vectorize -funroll-loops
    -fno-strict-aliasing
    -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare)
target_link_libraries(navis_lib PUBLIC
    uring OpenMP::OpenMP_CXX ${JEMALLOC_LIBRARY} pthread dl rt)

# ===========================================================================
# Tools
# ===========================================================================
foreach(tool navis_build navis_bench)
    add_executable(${tool} tools/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE navis_lib)
    target_compile_options(${tool} PRIVATE -O3 -march=native -fno-strict-aliasing)
endforeach()

# navis_convert: data preparation (subset, ground truth). Exact k-NN ground
# truth uses BLAS sgemm; OpenBLAS is preferred, Intel MKL is the fallback.
set(BLA_VENDOR OpenBLAS)
find_package(BLAS QUIET)
if(NOT BLAS_FOUND)
    set(BLA_VENDOR Intel10_64lp)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        set(NAVIS_BLAS_MKL ON)
        find_path(MKL_INCLUDE_DIR mkl.h PATH_SUFFIXES mkl REQUIRED)
    endif()
endif()
if(NOT BLAS_FOUND)
    message(FATAL_ERROR "navis_convert needs OpenBLAS (apt install libopenblas-dev) or Intel MKL")
endif()

add_executable(navis_convert tools/navis_convert.cpp)
target_compile_options(navis_convert PRIVATE -O3 -march=native)
# BLAS first: with MKL, OpenMP calls resolve to the same runtime MKL uses.
target_link_libraries(navis_convert PRIVATE BLAS::BLAS navis_lib)
if(NAVIS_BLAS_MKL)
    target_compile_definitions(navis_convert PRIVATE NAVIS_BLAS_MKL)
    target_include_directories(navis_convert PRIVATE ${MKL_INCLUDE_DIR})
endif()

# ===========================================================================
# Tests
# ===========================================================================
enable_testing()

file(GLOB TEST_SOURCES tests/test_*.cpp)
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE navis_lib)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

add_test(NAME test_batch_gt
         COMMAND bash ${CMAKE_SOURCE_DIR}/tests/tools/test_batch_gt.sh $<TARGET_FILE:navis_convert>)
