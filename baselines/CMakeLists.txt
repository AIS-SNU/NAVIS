# Baseline tools: NAVIS -> baseline index converters, and the FreshDiskANN
# benchmark driver. Built by baselines/build.sh.
cmake_minimum_required(VERSION 3.16)
project(navis_baselines CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)
set(TOOL_FLAGS -O3 -march=native -fno-strict-aliasing)

# Reader for the NAVIS on-disk index (_disk.edge, _disk.vec, _indirection.bin).
add_library(navis_layout_reader STATIC common/navis_layout_reader.cpp)
target_include_directories(navis_layout_reader PUBLIC common)
target_compile_options(navis_layout_reader PRIVATE ${TOOL_FLAGS})

foreach(tool odinann/navis_to_odinann freshdiskann/navis_to_freshdiskann)
    get_filename_component(name ${tool} NAME)
    add_executable(${name} ${tool}.cpp)
    target_link_libraries(${name} PRIVATE navis_layout_reader OpenMP::OpenMP_CXX)
    target_compile_options(${name} PRIVATE ${TOOL_FLAGS})
endforeach()

# FreshDiskANN driver, linked against the FreshDiskANN static library.
set(FRESHDISKANN_LIB "" CACHE FILEPATH "FreshDiskANN libdiskann_s.a")
if(FRESHDISKANN_LIB)
    set(FRESHDISKANN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/freshdiskann/FreshDiskANN)
    add_library(bench_harness STATIC common/harness/bench_harness.cpp)
    target_include_directories(bench_harness PUBLIC common/harness)
    target_compile_options(bench_harness PRIVATE ${TOOL_FLAGS})
    target_link_libraries(bench_harness PUBLIC pthread)

    add_executable(bench_freshdiskann freshdiskann/bench_freshdiskann.cpp)
    target_include_directories(bench_freshdiskann PRIVATE
        ${FRESHDISKANN_SRC}/include ${FRESHDISKANN_SRC}/include/v2)
    target_compile_options(bench_freshdiskann PRIVATE ${TOOL_FLAGS} -fopenmp
        -DUSE_AIO -DUSE_ACCELERATED_PQ -DUSE_AVX2 -DMKL_ILP64)
    target_link_options(bench_freshdiskann PRIVATE -fopenmp)
    target_link_libraries(bench_freshdiskann PRIVATE bench_harness ${FRESHDISKANN_LIB}
        pthread aio dl glog mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5)
endif()
