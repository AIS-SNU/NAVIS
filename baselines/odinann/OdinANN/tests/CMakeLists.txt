# Concurrent search + insertion (insert workload).
add_executable(test_insert_search test_insert_search.cpp)
target_link_libraries(test_insert_search ${PROJECT_NAME})

# Concurrent search + insertion + deletion with periodic merges (delete workload).
add_executable(overall_performance overall_performance.cpp)
target_link_libraries(overall_performance ${PROJECT_NAME})
