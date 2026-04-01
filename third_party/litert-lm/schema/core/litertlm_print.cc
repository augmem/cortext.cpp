// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstddef>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"
#include "schema/core/litertlm_utils.h"

namespace litert {
namespace lm {
namespace schema {

// --- ANSI Escape Code Definitions ---
const char ANSI_BOLD[] = "\033[1m";
const char ANSI_RESET[] = "\033[0m";

// --- Indentation Constants ---
const int INDENT_SPACES = 2;

// --- Formatting Helper Functions ---
void PrintHorizontalLine(std::ostream& os, char corner_left, char horizontal,
                         char corner_right, int width) {
  os << corner_left;
  for (int i = 0; i < width - 2; ++i) {  // Width - 2 for corners
    os << horizontal;
  }
  os << corner_right << "\n";
}

// Function to produce ASCII boxing of text.
void PrintBoxedTitle(std::ostream& os, const std::string& title,
                     int box_width = 50) {
  // Pass char literals for pure ASCII boxing
  PrintHorizontalLine(os, '+', '-', '+', box_width);  // Top border: +---+
  int padding_left = (box_width - 2 - title.length()) / 2;
  int padding_right = box_width - 2 - title.length() - padding_left;
  os << "|" << std::string(padding_left, ' ') << title
     << std::string(padding_right, ' ') << "|\n";     // Middle: |   Title   |
  PrintHorizontalLine(os, '+', '-', '+', box_width);  // Bottom border: +---+
}

// Helper function to print KeyValuePair data with specified indentation.
void PrintKeyValuePair(const KeyValuePair* kvp, std::ostream& output_stream,
                       int indent_level) {
  std::string indent_str(indent_level * INDENT_SPACES, ' ');

  if (!kvp) {
    output_stream << indent_str << "KeyValuePair: nullptr\n";
    return;
  }
  // Apply ANSI_BOLD before "Key" and "Value", and ANSI_RESET after each.
  output_stream << indent_str << ANSI_BOLD << "Key" << ANSI_RESET << ": "
                << kvp->key()->str() << ", ";

  switch (kvp->value_type()) {
    case VData::VData_StringValue: {
      output_stream << ANSI_BOLD << "Value" << ANSI_RESET << " (String): "
                    << kvp->value_as_StringValue()->value()->c_str() << "\n";
      break;
    }
    case VData::VData_Int32: {
      output_stream << ANSI_BOLD << "Value" << ANSI_RESET
                    << " (Int32): " << kvp->value_as_Int32()->value() << "\n";
      break;
    }
    case VData::VData_Float32: {
      output_stream << ANSI_BOLD << "Value" << ANSI_RESET
                    << " (Float): " << kvp->value_as_Float32()->value() << "\n";
      break;
    }
    case VData::VData_Bool: {
      output_stream << ANSI_BOLD << "Value" << ANSI_RESET
                    << " (Bool): " << kvp->value_as_Bool()->value() << "\n";
      break;
    }
    case VData::VData_UInt64: {
      output_stream << ANSI_BOLD << "Value" << ANSI_RESET
                    << " (Uint64): " << kvp->value_as_UInt64()->value() << "\n";
      break;
    }
    default:
      output_stream << ANSI_BOLD << "Value" << ANSI_RESET
                    << " (Unknown Type)\n";
      break;
  }
}

// Main function to process and print the LiteRT-LM file information
absl::Status ProcessLiteRTLMFile(const std::string& litertlm_file,
                                 std::ostream& output_stream) {
  LitertlmHeader header;
  absl::Status status = ReadHeaderFromLiteRTLM(litertlm_file, &header);

  if (!status.ok()) {
    return status;
  }

  // Print version information
  output_stream << "LiteRT-LM Version: " << header.major_version << "."
                << header.minor_version << "." << header.patch_version
                << "\n\n";

  if (header.metadata == nullptr) {
    ABSL_LOG(ERROR) << "header metadata is null ";
    return absl::InvalidArgumentError("header metadata is null");
  }

  auto system_metadata = header.metadata->system_metadata();

  if (system_metadata == nullptr) {
    ABSL_LOG(ERROR) << "system metadata is null ";
    return absl::InvalidArgumentError("system metadata is null");
  }

  // Print system metadata with boxing
  PrintBoxedTitle(output_stream, "System Metadata");
  auto entries = system_metadata->entries();

  if (entries && entries->size() > 0) {
    // No "Iterating over..." line, directly print entries
    for (size_t i = 0; i < entries->size(); ++i) {
      const KeyValuePair* entry = entries->Get(i);
      PrintKeyValuePair(entry, output_stream, 1);
    }
  } else {
    output_stream << std::string(INDENT_SPACES, ' ')
                  << "SystemMetadata has no entries.\n";
  }
  output_stream << "\n";  // Add a newline after system metadata block

  // Print section information with boxing
  auto section_metadata_obj = header.metadata->section_metadata();
  auto section_objects = section_metadata_obj->objects();
  PrintBoxedTitle(output_stream,
                  "Sections (" + std::to_string(section_objects->size()) + ")");
  output_stream << "\n";  // Add a newline after the sections title

  if (section_objects->size() == 0) {
    output_stream << std::string(INDENT_SPACES, ' ') << "<None>\n";
  } else {
    for (size_t i = 0; i < section_objects->size(); ++i) {
      auto sec_obj = section_objects->Get(i);
      output_stream << ANSI_BOLD << "Section " << i << ":" << ANSI_RESET
                    << "\n";
      output_stream << std::string(INDENT_SPACES, ' ') << "Items:\n";

      const auto& items = sec_obj->items();
      if (items && items->size() > 0) {
        for (size_t j = 0; j < items->size(); ++j) {
          const KeyValuePair* item = items->Get(j);
          PrintKeyValuePair(item, output_stream, 2);
        }
      }

      output_stream << std::string(INDENT_SPACES, ' ')
                    << "Begin Offset: " << sec_obj->begin_offset() << "\n";
      output_stream << std::string(INDENT_SPACES, ' ')
                    << "End Offset:   " << sec_obj->end_offset() << "\n";
      output_stream << std::string(INDENT_SPACES, ' ')
                    << "Data Type:    "
                    << AnySectionDataTypeToString(sec_obj->data_type())
                    << "\n";
      if (sec_obj->data_type() ==
          AnySectionDataType::AnySectionDataType_LlmMetadataProto) {
        output_stream << std::string(INDENT_SPACES, ' ')
                      << "<<<<<<<< start of LlmMetadata\n";
        using litert::lm::proto::LlmMetadata;
        LlmMetadata llm_metadata;
        absl::Status result =
            ReadLlmMetadataFromSection(litertlm_file, i, &llm_metadata);
        std::istringstream inputStream(llm_metadata.DebugString());
        std::string line;
        while (std::getline(inputStream, line)) {
          output_stream << std::string(INDENT_SPACES * 2, ' ') << line << '\n';
        }
        output_stream << std::string(INDENT_SPACES, ' ')
                      << ">>>>>>>> end of LlmMetadata\n";
      }
      output_stream
          << "\n";  // Add a newline after each section for better separation
    }
  }

  return absl::OkStatus();
}

}  // namespace schema
}  // namespace lm
}  // namespace litert
