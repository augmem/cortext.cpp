set(OBJSTORE_SOURCES
    objstore.c
    backend_registry.c
    backend_fs_common.c
    backend_portable.c
    objstore_vtab.c
    objstore_txn.c
    object_manager.c
    blake3_hash.c
)

set(OBJSTORE_BLAKE3_SOURCES
    ${PROJECT_SOURCE_DIR}/third_party/blake3/blake3.c
    ${PROJECT_SOURCE_DIR}/third_party/blake3/blake3_dispatch.c
    ${PROJECT_SOURCE_DIR}/third_party/blake3/blake3_portable.c
)

set(OBJSTORE_BLAKE3_DEFINES "")

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" OBJSTORE_SYSTEM_PROCESSOR)

set(OBJSTORE_TARGET_IS_X86 OFF)
if(OBJSTORE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|i[3-6]86")
    set(OBJSTORE_TARGET_IS_X86 ON)
endif()

set(OBJSTORE_TARGET_IS_AARCH64 OFF)
if(OBJSTORE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(OBJSTORE_TARGET_IS_AARCH64 ON)
endif()

if(OBJSTORE_TARGET_IS_X86)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
        list(APPEND OBJSTORE_BLAKE3_SOURCES
            ${PROJECT_SOURCE_DIR}/third_party/blake3/blake3_sse2.c
            ${PROJECT_SOURCE_DIR}/third_party/blake3/blake3_sse41.c
            ${PROJECT_SOURCE_DIR}/third_party/blake3/blake3_avx2.c
            ${PROJECT_SOURCE_DIR}/third_party/blake3/blake3_avx512.c
        )
        set_source_files_properties(${PROJECT_SOURCE_DIR}/third_party/blake3/blake3_sse2.c PROPERTIES
            COMPILE_OPTIONS "-msse2")
        set_source_files_properties(${PROJECT_SOURCE_DIR}/third_party/blake3/blake3_sse41.c PROPERTIES
            COMPILE_OPTIONS "-msse4.1")
        set_source_files_properties(${PROJECT_SOURCE_DIR}/third_party/blake3/blake3_avx2.c PROPERTIES
            COMPILE_OPTIONS "-mavx2")
        set_source_files_properties(${PROJECT_SOURCE_DIR}/third_party/blake3/blake3_avx512.c PROPERTIES
            COMPILE_OPTIONS "-mavx512f;-mavx512vl")
    else()
        list(APPEND OBJSTORE_BLAKE3_DEFINES
            BLAKE3_NO_SSE2=1
            BLAKE3_NO_SSE41=1
            BLAKE3_NO_AVX2=1
            BLAKE3_NO_AVX512=1
        )
    endif()
endif()

if(OBJSTORE_TARGET_IS_AARCH64)
    list(APPEND OBJSTORE_BLAKE3_SOURCES
        ${PROJECT_SOURCE_DIR}/third_party/blake3/blake3_neon.c
)
endif()

list(APPEND OBJSTORE_SOURCES ${OBJSTORE_BLAKE3_SOURCES})

if(OBJSTORE_BACKEND_SQLITE)
    list(APPEND OBJSTORE_SOURCES backend_sqlite.c)
endif()

if(OBJSTORE_BACKEND_FILE)
    list(APPEND OBJSTORE_SOURCES backend_file.c)
endif()

if(OBJSTORE_BACKEND_OPFS)
    list(APPEND OBJSTORE_SOURCES backend_opfs.c)
endif()

if(OBJSTORE_BACKEND_VFS)
    list(APPEND OBJSTORE_SOURCES backend_vfs.c)
endif()

if(WIN32)
    list(APPEND OBJSTORE_SOURCES win_dirent.c)
endif()

function(_objstore_configure_library target)
    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/blake3>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/third_party/blake3>
    )
    target_compile_features(${target} PUBLIC c_std_17)
    target_link_libraries(${target}
        PUBLIC
            SQLite::SQLite3
    )
    target_compile_definitions(${target}
        PRIVATE
            ${OBJSTORE_BLAKE3_DEFINES}
            $<$<BOOL:${OBJSTORE_BACKEND_FILE}>:OBJSTORE_HAVE_BACKEND_FILE=1>
            $<$<BOOL:${OBJSTORE_BACKEND_SQLITE}>:OBJSTORE_HAVE_BACKEND_SQLITE=1>
            $<$<BOOL:${OBJSTORE_BACKEND_OPFS}>:OBJSTORE_HAVE_BACKEND_OPFS=1>
            $<$<BOOL:${OBJSTORE_BACKEND_VFS}>:OBJSTORE_HAVE_BACKEND_VFS=1>
    )
endfunction()

add_library(objstore INTERFACE)
add_library(objstore::objstore ALIAS objstore)

set(_objstore_install_targets)

if(OBJSTORE_BUILD_SHARED)
    add_library(objstore_shared SHARED ${OBJSTORE_SOURCES})
    _objstore_configure_library(objstore_shared)
    set_target_properties(objstore_shared
        PROPERTIES
            OUTPUT_NAME objstore
            VERSION ${PROJECT_VERSION}
            SOVERSION ${PROJECT_VERSION_MAJOR}
            EXPORT_NAME objstore_shared
    )
    target_link_libraries(objstore INTERFACE objstore_shared)
    add_library(objstore::objstore_shared ALIAS objstore_shared)
    list(APPEND _objstore_install_targets objstore_shared)
endif()

if(OBJSTORE_BUILD_STATIC)
    add_library(objstore_static STATIC ${OBJSTORE_SOURCES})
    _objstore_configure_library(objstore_static)
    set_target_properties(objstore_static
        PROPERTIES
            OUTPUT_NAME objstore
            EXPORT_NAME objstore_static
    )
    if(NOT OBJSTORE_BUILD_SHARED)
        target_link_libraries(objstore INTERFACE objstore_static)
    endif()
    add_library(objstore::objstore_static ALIAS objstore_static)
    list(APPEND _objstore_install_targets objstore_static)
endif()

if(NOT OBJSTORE_BUILD_SHARED AND NOT OBJSTORE_BUILD_STATIC)
    message(FATAL_ERROR "At least one objstore library target must be enabled")
endif()

install(TARGETS ${_objstore_install_targets}
        EXPORT objstoreTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(TARGETS objstore
        EXPORT objstoreTargets)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${PROJECT_SOURCE_DIR}/third_party/blake3/blake3.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/third_party/blake3)
