add_subdirectory(label_bank_generator)
add_subdirectory(label_bank_sqlite_builder)
add_subdirectory(aist_audio_embedder)
add_subdirectory(text_embedder)

add_custom_target(cortext_tools
  DEPENDS cortext_label_bank_generator
          cortext_label_bank_sqlite_builder
          cortext_aist_audio_embedder
          cortext_text_embedder
)
