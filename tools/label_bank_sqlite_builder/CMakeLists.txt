add_executable(cortext_label_bank_sqlite_builder
  main.cpp
)

target_link_libraries(cortext_label_bank_sqlite_builder PRIVATE cortext nlohmann_json::nlohmann_json)
target_compile_features(cortext_label_bank_sqlite_builder PRIVATE cxx_std_20)
target_include_directories(cortext_label_bank_sqlite_builder PRIVATE ${CMAKE_SOURCE_DIR}/src)

if (MSVC)
  target_compile_options(cortext_label_bank_sqlite_builder PRIVATE /W4)
else ()
  target_compile_options(cortext_label_bank_sqlite_builder PRIVATE -Wall -Wextra -Wpedantic)
endif ()
