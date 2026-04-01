# CMakeLists.txt for building SQLite from third_party

cmake_minimum_required(VERSION 3.10)

# Generate sqlite3.h from template
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sqlite.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/sqlite3.h
    COPYONLY
)

# Collect all SQLite core source files
file(GLOB SQLITE_SOURCES "src/*.c")

# Remove test files and other non-core files
list(FILTER SQLITE_SOURCES EXCLUDE REGEX "test[0-9]*\\.c$")
list(FILTER SQLITE_SOURCES EXCLUDE REGEX "shell\\.c$")
list(FILTER SQLITE_SOURCES EXCLUDE REGEX "tclsqlite\\.c$")
list(FILTER SQLITE_SOURCES EXCLUDE REGEX "fuzzcheck\\.c$")
list(FILTER SQLITE_SOURCES EXCLUDE REGEX "mptest\\.c$")
list(FILTER SQLITE_SOURCES EXCLUDE REGEX "ossshell\\.c$")
list(FILTER SQLITE_SOURCES EXCLUDE REGEX "speedtest1\\.c$")
list(FILTER SQLITE_SOURCES EXCLUDE REGEX "threadtest[0-9]*\\.c$")
list(FILTER SQLITE_SOURCES EXCLUDE REGEX "wordcount\\.c$")

# Create SQLite static library
add_library(sqlite3 STATIC ${SQLITE_SOURCES})

# SQLite is C code, but we'll let CMake figure out the linking

# Set include directories
target_include_directories(sqlite3
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    INTERFACE
        $<INSTALL_INTERFACE:include>
)

# SQLite compile definitions
target_compile_definitions(sqlite3 PRIVATE
    SQLITE_THREADSAFE=1
    SQLITE_ENABLE_FTS5
    SQLITE_ENABLE_JSON1
    SQLITE_ENABLE_RTREE
)

# Suppress warnings for third-party code
if(MSVC)
    target_compile_options(sqlite3 PRIVATE /W0)
else()
    target_compile_options(sqlite3 PRIVATE -w)
endif()

# Install headers
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sqlite3.h
    DESTINATION include
)
install(TARGETS sqlite3
    ARCHIVE DESTINATION lib
)
