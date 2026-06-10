add_executable(objstore_file_bench
    file_backend_bench.c
)

target_include_directories(objstore_file_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(objstore_file_bench
    PRIVATE
        objstore
)

if(UNIX)
    target_link_libraries(objstore_file_bench PRIVATE m)
endif()
