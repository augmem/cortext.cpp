# Command-line surface over the public Cortext API.

add_executable(cortext_cli
  main.cpp
)

target_link_libraries(cortext_cli PRIVATE
  cortext::cortext
)

target_compile_features(cortext_cli PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(cortext_cli PRIVATE /W4)
else()
  target_compile_options(cortext_cli PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS cortext_cli
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT cortext
)
