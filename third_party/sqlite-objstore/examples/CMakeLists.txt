add_executable(objstore_example_file_metadata file_metadata_example.c)
target_link_libraries(objstore_example_file_metadata PRIVATE objstore SQLite::SQLite3)

add_executable(objstore_example_cache cache_example.c)
target_link_libraries(objstore_example_cache PRIVATE objstore SQLite::SQLite3)

if(BUILD_TESTING)
    add_test(NAME example_file_metadata COMMAND objstore_example_file_metadata)
    add_test(NAME example_cache COMMAND objstore_example_cache)
endif()

