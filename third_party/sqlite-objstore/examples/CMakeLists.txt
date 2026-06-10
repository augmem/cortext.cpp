add_executable(objstore_example_file_metadata file_metadata_example.c)
target_link_libraries(objstore_example_file_metadata PRIVATE objstore)

add_executable(objstore_example_cache cache_example.c)
target_link_libraries(objstore_example_cache PRIVATE objstore)

add_executable(objstore_example_orphan_sweep orphan_sweep.c)
target_include_directories(objstore_example_orphan_sweep
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(objstore_example_orphan_sweep PRIVATE objstore)

if(BUILD_TESTING)
    add_test(NAME example_file_metadata COMMAND objstore_example_file_metadata)
    add_test(NAME example_cache COMMAND objstore_example_cache)
endif()
