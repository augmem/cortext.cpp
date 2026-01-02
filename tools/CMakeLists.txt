add_subdirectory(label_bank_generator)
add_subdirectory(text_embedder)

add_custom_target(cortext_tools
  DEPENDS cortext_label_bank_generator
          cortext_text_embedder
)
