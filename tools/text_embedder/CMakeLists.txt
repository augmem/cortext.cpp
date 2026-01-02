add_executable(cortext_text_embedder
  main.cpp
)

target_link_libraries(cortext_text_embedder PRIVATE cortext nlohmann_json::nlohmann_json)
target_compile_features(cortext_text_embedder PRIVATE cxx_std_20)

if (MSVC)
  target_compile_options(cortext_text_embedder PRIVATE /W4)
else ()
  target_compile_options(cortext_text_embedder PRIVATE -Wall -Wextra -Wpedantic)
endif ()
