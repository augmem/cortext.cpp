cmake_minimum_required(VERSION 3.28)

project(sqlite_objstore VERSION 0.1.0 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(OBJSTORE_ENABLE_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
option(OBJSTORE_ENABLE_ASAN "Enable AddressSanitizer instrumentation" OFF)
option(OBJSTORE_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer instrumentation" OFF)
option(OBJSTORE_ENABLE_SANITIZERS "Deprecated: enable both ASan and UBSan" OFF)
option(OBJSTORE_ENABLE_COVERAGE "Enable LLVM coverage instrumentation" OFF)
option(OBJSTORE_ENABLE_CLANG_TIDY "Run clang-tidy during builds" OFF)
option(OBJSTORE_ENABLE_CPPCHECK "Run cppcheck during builds" OFF)
option(OBJSTORE_ENABLE_FUZZERS "Build libFuzzer-based hardening targets" OFF)
option(OBJSTORE_BUILD_SHARED "Build the shared sqlite-objstore library" ON)
option(OBJSTORE_BUILD_STATIC "Build the static sqlite-objstore library" ON)

if(NOT OBJSTORE_BUILD_SHARED AND NOT OBJSTORE_BUILD_STATIC)
    message(FATAL_ERROR "Enable at least one of OBJSTORE_BUILD_SHARED or OBJSTORE_BUILD_STATIC")
endif()

include(GNUInstallDirs)
include(StaticAnalysis)
objstore_configure_static_analysis()

option(OBJSTORE_USE_SYSTEM_SQLITE "Link against SQLite provided by the host system" OFF)
set(OBJSTORE_SQLITE_RELEASE_YEAR "2025" CACHE STRING "Year used to form the SQLite download URL")
set(OBJSTORE_SQLITE_RELEASE_ID "3480000" CACHE STRING "Numeric release identifier (e.g., 3.48.0 => 3480000)")

if(OBJSTORE_USE_SYSTEM_SQLITE)
    find_package(SQLite3 REQUIRED)
else()
    include(FetchContent)

    set(_objstore_sqlite_url
        "https://www.sqlite.org/${OBJSTORE_SQLITE_RELEASE_YEAR}/sqlite-amalgamation-${OBJSTORE_SQLITE_RELEASE_ID}.zip")

    FetchContent_Declare(
        sqlite_amalgamation
        URL ${_objstore_sqlite_url}
    )
    FetchContent_MakeAvailable(sqlite_amalgamation)

    add_library(sqlite3_amalgamation STATIC ${sqlite_amalgamation_SOURCE_DIR}/sqlite3.c)
    target_include_directories(sqlite3_amalgamation
        PUBLIC
            $<BUILD_INTERFACE:${sqlite_amalgamation_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/sqlite3>
    )
    target_compile_features(sqlite3_amalgamation PRIVATE c_std_17)
    target_compile_definitions(sqlite3_amalgamation
        PUBLIC
            SQLITE_THREADSAFE=1
            SQLITE_OMIT_LOAD_EXTENSION=1
    )

    add_library(SQLite::SQLite3 ALIAS sqlite3_amalgamation)

    install(TARGETS sqlite3_amalgamation
            EXPORT objstoreTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    install(FILES
            ${sqlite_amalgamation_SOURCE_DIR}/sqlite3.h
            ${sqlite_amalgamation_SOURCE_DIR}/sqlite3ext.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sqlite3)
endif()

option(OBJSTORE_BACKEND_FILE "Build the native file-backed backend" ON)
option(OBJSTORE_BACKEND_SQLITE "Build the SQLite fallback backend" ON)
option(OBJSTORE_BACKEND_OPFS "Build the OPFS backend stubs" ON)
option(OBJSTORE_BACKEND_VFS "Build the VFS backend stubs" ON)
option(OBJSTORE_ENABLE_WASM_CROSS_LANG "Run the WASM cross-language harness via ctest" OFF)
option(OBJSTORE_ENABLE_PERF_TESTS "Build perf/stress harness binaries and tests" OFF)
option(OBJSTORE_ENABLE_WASM_MATRIX "Enable wasm SQL matrix fixture tests" OFF)

if(CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID MATCHES "GNU")
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wall>)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wextra>)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wpedantic>)

    if(OBJSTORE_ENABLE_WARNINGS_AS_ERRORS)
        add_compile_options($<$<COMPILE_LANGUAGE:C>:-Werror>)
    endif()
endif()

set(_objstore_enabled_sanitizers "")

if(OBJSTORE_ENABLE_SANITIZERS)
    message(DEPRECATION
        "OBJSTORE_ENABLE_SANITIZERS is deprecated. "
        "Use OBJSTORE_ENABLE_ASAN/OBJSTORE_ENABLE_UBSAN instead.")
    set(OBJSTORE_ENABLE_ASAN ON CACHE BOOL "" FORCE)
    set(OBJSTORE_ENABLE_UBSAN ON CACHE BOOL "" FORCE)
endif()

if(OBJSTORE_ENABLE_ASAN)
    list(APPEND _objstore_enabled_sanitizers "address")
endif()

if(OBJSTORE_ENABLE_UBSAN)
    list(APPEND _objstore_enabled_sanitizers "undefined")
endif()

if(_objstore_enabled_sanitizers)
    list(REMOVE_DUPLICATES _objstore_enabled_sanitizers)
    string(REPLACE ";" "," _objstore_sanitizer_flags "${_objstore_enabled_sanitizers}")
    foreach(lang IN ITEMS C CXX)
        add_compile_options($<$<COMPILE_LANGUAGE:${lang}>:-fsanitize=${_objstore_sanitizer_flags}>)
        add_compile_options($<$<COMPILE_LANGUAGE:${lang}>:-fno-omit-frame-pointer>)
        if(OBJSTORE_ENABLE_ASAN)
            if(CMAKE_${lang}_COMPILER_ID MATCHES "Clang|AppleClang")
                add_compile_options($<$<COMPILE_LANGUAGE:${lang}>:-fsanitize-address-use-after-scope>)
            endif()
        endif()
    endforeach()
    add_link_options(-fsanitize=${_objstore_sanitizer_flags})
endif()

if(OBJSTORE_ENABLE_COVERAGE)
    foreach(lang IN ITEMS C CXX)
        add_compile_options($<$<COMPILE_LANGUAGE:${lang}>:-fprofile-instr-generate>)
        add_compile_options($<$<COMPILE_LANGUAGE:${lang}>:-fcoverage-mapping>)
        add_compile_options($<$<COMPILE_LANGUAGE:${lang}>:-g>)
        add_compile_options($<$<COMPILE_LANGUAGE:${lang}>:-O0>)
    endforeach()
    add_link_options(-fprofile-instr-generate)
endif()

include(CTest)

option(OBJSTORE_BUILD_BENCHMARKS "Build benchmark tooling" ON)

add_subdirectory(src)
add_subdirectory(examples)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(OBJSTORE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

include(CMakePackageConfigHelpers)

configure_package_config_file(
    cmake/ObjstoreConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/objstoreConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/objstore
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/objstoreConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(EXPORT objstoreTargets
        NAMESPACE objstore::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/objstore)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/objstoreConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/objstoreConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/objstore
)

find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    doxygen_add_docs(objstore_doc
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/docs/metadata-patterns.md
        ${PROJECT_SOURCE_DIR}/docs/transactions.md
        ${PROJECT_SOURCE_DIR}/docs/getting-started.md
    )
endif()
