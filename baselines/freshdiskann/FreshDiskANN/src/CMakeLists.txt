add_library(${PROJECT_NAME}_s STATIC
    ann_exception.cpp aux_utils.cpp distance.cpp index.cpp linux_aligned_file_reader.cpp
    logger.cpp math_utils.cpp memory_mapper.cpp partition_and_pq.cpp pq_flash_index.cpp
    utils.cpp v2/graph_delta.cpp v2/index_merger.cpp v2/merge_insert.cpp)
