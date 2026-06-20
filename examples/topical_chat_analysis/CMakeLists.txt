# Topical-Chat integration analysis (full pipeline via Cortext::ProcessText)

add_executable(cortext_topical_chat_analysis
  main.cpp
)

target_link_libraries(cortext_topical_chat_analysis PRIVATE
  cortext::cortext
  nlohmann_json::nlohmann_json
)

if(NOT CORTEXT_DISABLE_OPENTELEMETRY)
  target_link_libraries(cortext_topical_chat_analysis PRIVATE
    opentelemetry-cpp::api
    opentelemetry-cpp::logs
  )
else()
  target_compile_definitions(cortext_topical_chat_analysis PRIVATE
    CORTEXT_DISABLE_OPENTELEMETRY=1
  )
endif()

target_compile_features(cortext_topical_chat_analysis PRIVATE cxx_std_20)
target_include_directories(cortext_topical_chat_analysis PRIVATE
  ${PROJECT_SOURCE_DIR}/src
)

if(MSVC)
  target_compile_options(cortext_topical_chat_analysis PRIVATE /W4)
else()
  target_compile_options(cortext_topical_chat_analysis PRIVATE -Wall -Wextra -Wpedantic)
endif()
