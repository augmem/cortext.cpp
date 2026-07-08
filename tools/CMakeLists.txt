add_subdirectory(aist_audio_embedder)
add_subdirectory(cli)
add_subdirectory(memory_eval_runner)
add_subdirectory(text_embedder)

add_custom_target(cortext_tools
  DEPENDS cortext_aist_audio_embedder
          cortext_cli
          cortext_memory_eval_runner
          cortext_text_embedder
)
