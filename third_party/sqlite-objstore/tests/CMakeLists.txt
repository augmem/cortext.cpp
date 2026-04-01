if(NOT BUILD_TESTING)
    return()
endif()

include(FetchContent)

FetchContent_Declare(
    unity
    GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
    GIT_TAG v2.6.0
)

FetchContent_MakeAvailable(unity)

add_library(objstore_unity STATIC ${unity_SOURCE_DIR}/src/unity.c)
target_include_directories(objstore_unity PUBLIC ${unity_SOURCE_DIR}/src)

add_executable(objstore_tests
    test_harness.c
    core_api_test.c
    vtab_crud_test.c
    scalar_function_test.c
    sqlite_helper_test.c
    file_backend_vtab_test.c
    txn_log_test.c
    backend_sqlite_test.c
    backend_file_test.c
    backend_vfs_test.c
    object_manager_test.c
)

target_include_directories(objstore_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(objstore_tests
    PRIVATE
        objstore
        objstore_unity
)

add_executable(objstore_wasm_matrix
    wasm_matrix_runner.c
)

target_include_directories(objstore_wasm_matrix
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(objstore_wasm_matrix
    PRIVATE
        objstore
)

if(OBJSTORE_ENABLE_PERF_TESTS)
    add_library(objstore_perf_common STATIC
        perf/perf_common.c
    )
    target_include_directories(objstore_perf_common
        PUBLIC
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/perf
    )
    target_link_libraries(objstore_perf_common
        PUBLIC
            objstore
    )

    add_executable(objstore_perf_micro
        perf/bench_micro.c
    )
    target_link_libraries(objstore_perf_micro PRIVATE objstore_perf_common)

    add_executable(objstore_perf_stress
        perf/stress_mix.c
    )
    target_link_libraries(objstore_perf_stress PRIVATE objstore_perf_common)

    add_executable(objstore_perf_stream_1gb
        perf/stream_1gb.c
    )
    target_link_libraries(objstore_perf_stream_1gb PRIVATE objstore_perf_common objstore)

    add_test(NAME perf_micro_sqlite
             COMMAND objstore_perf_micro
                     --backend sqlite
                     --ops 64
                     --payload 4096)
    set_tests_properties(perf_micro_sqlite PROPERTIES LABELS perf)

    add_test(NAME perf_micro_file
             COMMAND objstore_perf_micro
                     --backend file
                     --ops 32
                     --payload 32768)
    set_tests_properties(perf_micro_file PROPERTIES LABELS perf)

    add_test(NAME perf_stress_sqlite
             COMMAND objstore_perf_stress
                     --backend sqlite
                     --ops 512
                     --min 1024
                     --max 8192
                     --txn-batch 32)
    set_tests_properties(perf_stress_sqlite PROPERTIES LABELS perf)

    add_test(NAME perf_stream_1gb_file
             COMMAND objstore_perf_stream_1gb
                     --backend file
                     --bytes 1073741824)
    set_tests_properties(perf_stream_1gb_file PROPERTIES LABELS perf)
endif()

if(NOT WIN32)
    add_executable(objstore_crash_kill
        crash/kill_during_commit.c
    )
    target_link_libraries(objstore_crash_kill
        PRIVATE
            objstore
            SQLite::SQLite3
    )
    add_test(NAME crash_kill_during_commit COMMAND objstore_crash_kill)
    set_tests_properties(crash_kill_during_commit PROPERTIES LABELS crash)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "WASI")
    find_program(OBJSTORE_WASMTIME_EXECUTABLE NAMES wasmtime)
    if(OBJSTORE_WASMTIME_EXECUTABLE)
        add_test(NAME objstore_tests
                 COMMAND ${OBJSTORE_WASMTIME_EXECUTABLE} --dir=. $<TARGET_FILE:objstore_tests>)
    else()
        message(WARNING "wasmtime not found; WASI objstore_tests will be skipped")
    endif()
else()
    add_test(NAME objstore_tests COMMAND objstore_tests)
endif()

if(OBJSTORE_ENABLE_WASM_MATRIX)
    set(OBJSTORE_WASM_MATRIX_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/wasm)
    set(OBJSTORE_WASM_MATRIX_FIXTURES
        crud_basic
        streaming_large
        txn_commit_rollback
        idempotent_put
    )

    foreach(fixture IN LISTS OBJSTORE_WASM_MATRIX_FIXTURES)
        add_test(
            NAME wasm_matrix_${fixture}
            COMMAND objstore_wasm_matrix
                    --bundle-root ${OBJSTORE_WASM_MATRIX_ROOT}
                    --fixture sql/matrix/${fixture}.sql
                    --fixture-id ${fixture}
        )
    endforeach()
endif()

if(OBJSTORE_ENABLE_WASM_CROSS_LANG)
    add_test(NAME wasm_cross_language
             COMMAND ${CMAKE_SOURCE_DIR}/scripts/run-wasm-cross-lang.sh
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
