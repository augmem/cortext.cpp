# Public-API memory evaluation runner.

add_executable(cortext_memory_eval_runner
  main.cpp
)

target_link_libraries(cortext_memory_eval_runner PRIVATE
  cortext::cortext
  nlohmann_json::nlohmann_json
)

target_compile_features(cortext_memory_eval_runner PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(cortext_memory_eval_runner PRIVATE /W4)
else()
  target_compile_options(cortext_memory_eval_runner PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS cortext_memory_eval_runner
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT cortext
)
