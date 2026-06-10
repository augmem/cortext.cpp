cmake_minimum_required(VERSION 3.28)

project(objstore_downstream_smoke LANGUAGES C)

find_package(objstore CONFIG REQUIRED)

enable_testing()

add_executable(objstore_downstream_smoke_default main.c)
target_link_libraries(objstore_downstream_smoke_default PRIVATE objstore::objstore)

if(TARGET objstore::objstore_static)
    add_executable(objstore_downstream_smoke_static main.c)
    target_link_libraries(objstore_downstream_smoke_static PRIVATE objstore::objstore_static)
    add_test(NAME objstore_downstream_smoke_static COMMAND objstore_downstream_smoke_static)
endif()
