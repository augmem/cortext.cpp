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

// This tool is used to create a LiteRT-LM file from a set of input files
// (tokenizer, tflite model, llm parameters), and metadata.
//
// Example usage:
//
// bazel run
// //third_party/odml/litert_lm/schema/cc:litertlm_export_main \
//   -- --tokenizer_file=/path/to/tokenizer.model \
//   --tflite_file=/path/to/model.tflite \
//   --llm_metadata=/path/to/llm_metadata.pb
//   --output_path=/path/to/output.litertlm \
//   --section_metadata="tokenizer:key1=value1,key2=value2;\
//     tflite:key3=123,key4=true:llm_metadata=key5=abc"
//
// (Also accepts `--llm_metadata_text' instead if text proto is preferred)
//  (or --llm_metadata_text for a text proto) \
// NB: This tool is deprecated and will be replaced with litertlm-writer.

#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/proto/llm_metadata.pb.h"
#include "schema/core/litertlm_export.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_section.h"
#include "google/protobuf/text_format.h"  // from @com_google_protobuf

// Section names used in the section_metadata flag.
constexpr char kTokenizerSectionName[] = "tokenizer";
constexpr char kTfliteSectionName[] = "tflite";
constexpr char kLlmMetadataSectionName[] = "llm_metadata";

ABSL_FLAG(std::string, tokenizer_file, "",
          "The path to the file that contains the SP tokenizer.");

ABSL_FLAG(std::string, hf_tokenizer_json_file, "",
          "The path to the file that contains the HF tokenizer JSON config.");

ABSL_FLAG(std::string, tflite_file, "", "The path to the TFLite model file.");

ABSL_FLAG(
    std::string, llm_metadata, "",
    "The path to the file containing the LlmMetadata proto (binary format).");

ABSL_FLAG(std::string, binary_data, "",
          "The path to a file containing binary data.");

ABSL_FLAG(std::string, llm_metadata_text, "",
          "The path to the file containing the LlmMetadata text proto.");

ABSL_FLAG(std::string, output_path, "",
          "The path for the output LiteRT-LM file.");

// Flag to handle key-value pairs.  Example usage:
// --section_metadata="tokenizer:key1=value1,key2=value2;tflite:key3=123,key4=true"
// TODO(b/416130396): Get rid of this method of metadata creation.
ABSL_FLAG(std::string, section_metadata, "",
          "Metadata for sections in the format "
          "'section_name:key1=value1,key2=value2;...'. "
          "Supported value types: int32, int64, uint32, uint64, bool, float, "
          "string.");

namespace {

using ::litert::lm::proto::LlmMetadata;
using ::litert::lm::schema::AnySectionDataType;
using ::litert::lm::schema::AnySectionDataType_GenericBinaryData;
using ::litert::lm::schema::AnySectionDataType_HF_Tokenizer_Zlib;
using ::litert::lm::schema::AnySectionDataType_LlmMetadataProto;
using ::litert::lm::schema::AnySectionDataType_SP_Tokenizer;
using ::litert::lm::schema::AnySectionDataType_TFLiteModel;
using ::litert::lm::schema::CreateKeyValuePair;
using ::litert::lm::schema::CreateStringValue;
using ::litert::lm::schema::FileBackedSectionStream;
using ::litert::lm::schema::KVPair;
using ::litert::lm::schema::MakeLiteRTLMFromSections;
using ::litert::lm::schema::ProtoBufSectionStream;
using ::litert::lm::schema::SectionStreamBase;
using ::litert::lm::schema::ZlibBackendedSectionStream;

// Helper function to parse a single key-value pair.
absl::Status ParseKeyValuePair(absl::string_view kv_str, std::string& key,
                               std::string& value) {
  std::vector<std::string> parts = absl::StrSplit(kv_str, '=');
  if (parts.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid key-value pair: ", kv_str));
  }
  key = parts[0];
  value = parts[1];
  return absl::OkStatus();
}

// Helper function to convert string value to the correct type.
KVPair ConvertKeyValue(flatbuffers::FlatBufferBuilder& builder,
                       const std::string& key, const std::string& value_str) {
  int32_t int32_value;
  int64_t int64_value;
  uint32_t uint32_value;
  uint64_t uint64_value;
  float float_value;
  bool bool_value;

  if (absl::SimpleAtoi(value_str, &int32_value)) {
    return CreateKeyValuePair(builder, key, int32_value);
  } else if (absl::SimpleAtoi(value_str, &int64_value)) {
    return CreateKeyValuePair(builder, key, int64_value);
  } else if (absl::SimpleAtoi(value_str, &uint32_value)) {
    return CreateKeyValuePair(builder, key, uint32_value);
  } else if (absl::SimpleAtoi(value_str, &uint64_value)) {
    return CreateKeyValuePair(builder, key, uint64_value);
  } else if (absl::SimpleAtof(value_str, &float_value)) {
    return CreateKeyValuePair(builder, key, float_value);
  } else if (value_str == "true" || value_str == "false") {
    bool_value = (value_str == "true");
    return CreateKeyValuePair(builder, key, bool_value);
  } else {
    // Default to string.
    return CreateKeyValuePair(builder, key, value_str);
  }
}

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string tokenizer_file = absl::GetFlag(FLAGS_tokenizer_file);
  std::string hf_tokenizer_json_file =
      absl::GetFlag(FLAGS_hf_tokenizer_json_file);
  std::string tflite_file = absl::GetFlag(FLAGS_tflite_file);
  std::string output_path = absl::GetFlag(FLAGS_output_path);
  std::string llm_metadata_file = absl::GetFlag(FLAGS_llm_metadata);
  std::string llm_metadata_text_file = absl::GetFlag(FLAGS_llm_metadata_text);
  std::string section_metadata_str = absl::GetFlag(FLAGS_section_metadata);
  std::string binary_data = absl::GetFlag(FLAGS_binary_data);

  ABSL_LOG(INFO) << "tokenizer file is " << tokenizer_file << "\n";
  ABSL_LOG(INFO) << "hf_tokenizer_json_file is " << hf_tokenizer_json_file
                 << "\n";
  ABSL_LOG(INFO) << "tflite file is " << tflite_file << "\n";
  ABSL_LOG(INFO) << "output_path is " << output_path << "\n";
  ABSL_LOG(INFO) << "llm_metadata file is " << llm_metadata_file << "\n";
  ABSL_LOG(INFO) << "llm_metadata_text file is " << llm_metadata_text_file
                 << "\n";
  ABSL_LOG(INFO) << "section_metadata is " << section_metadata_str << "\n";
  ABSL_LOG(INFO) << "binary_data file is " << binary_data << "\n";

  // Enforce that at least one input file flag is specified.
  if (tokenizer_file.empty() && hf_tokenizer_json_file.empty() &&
      tflite_file.empty() && llm_metadata_file.empty() &&
      llm_metadata_text_file.empty()) {
    return absl::InvalidArgumentError(
        "At least one of --tokenizer_file, --tflite_file, --llm_metadata, or "
        "--llm_metadata_text must be provided.");
  }

  // Enforce that only one of --llm_metadata or --llm_metadata_text is
  // specified.
  if (!llm_metadata_file.empty() && !llm_metadata_text_file.empty()) {
    return absl::InvalidArgumentError(
        "Only one of --llm_metadata or --llm_metadata_text can be specified.");
  }

  std::vector<std::unique_ptr<SectionStreamBase>> sections;
  std::vector<AnySectionDataType> section_types;
  std::vector<std::vector<KVPair>> section_items_list;

  if (!tokenizer_file.empty()) {
    std::unique_ptr<SectionStreamBase> fbs =
        std::make_unique<FileBackedSectionStream>(tokenizer_file);
    sections.push_back(std::move(fbs));
    // We only support 1 Tokenizer type right now, SentencePiece.
    section_types.push_back(AnySectionDataType_SP_Tokenizer);
    section_items_list.push_back(
        {});  // Add an empty vector, to be populated later
  }

  if (!hf_tokenizer_json_file.empty()) {
    std::unique_ptr<SectionStreamBase> base_stream =
        std::make_unique<FileBackedSectionStream>(hf_tokenizer_json_file);
    std::unique_ptr<SectionStreamBase> compressed_stream =
        std::make_unique<ZlibBackendedSectionStream>(std::move(base_stream));
    sections.push_back(std::move(compressed_stream));
    section_types.push_back(AnySectionDataType_HF_Tokenizer_Zlib);
    section_items_list.push_back(
        {});  // Add an empty vector, to be populated later
  }

  if (!tflite_file.empty()) {
    std::unique_ptr<SectionStreamBase> fbs =
        std::make_unique<FileBackedSectionStream>(tflite_file);
    sections.push_back(std::move(fbs));
    section_types.push_back(AnySectionDataType_TFLiteModel);
    section_items_list.push_back(
        {});  // Add an empty vector, to be populated later
  }

  if (!llm_metadata_file.empty() || !llm_metadata_text_file.empty()) {
    LlmMetadata llm_metadata_proto;

    if (!llm_metadata_file.empty()) {
      // Read binary proto from file.
      std::ifstream ifs(llm_metadata_file, std::ios::binary);
      if (!ifs.is_open()) {
        return absl::NotFoundError(absl::StrCat(
            "Could not open llm_metadata file: ", llm_metadata_file));
      }
      std::string proto_str((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());

      if (!llm_metadata_proto.ParseFromString(proto_str)) {
        return absl::InvalidArgumentError(
            "Failed to parse LlmMetadata protobuf from binary file.");
      }
#if !defined(__ANDROID__) && !defined(OS_IOS)
    } else {  // llm_metadata_text_file is not empty
              // Read text proto from file.
      std::ifstream ifs(llm_metadata_text_file, std::ios::binary);
      if (!ifs.is_open()) {
        return absl::NotFoundError(absl::StrCat(
            "Could not open llm_metadata text file: ", llm_metadata_text_file));
      }
      std::string proto_text_str((std::istreambuf_iterator<char>(ifs)),
                                 std::istreambuf_iterator<char>());

      if (!google::protobuf::TextFormat::ParseFromString(proto_text_str,
                                               &llm_metadata_proto)) {
        return absl::InvalidArgumentError(
            "Failed to parse LlmMetadata protobuf from text file.");
      }
#endif  // !defined(__ANDROID__) && !defined(OS_IOS)
    }

    // Create a ProtoBufSectionStream with the parsed proto.
    std::unique_ptr<SectionStreamBase> pbs =
        std::make_unique<ProtoBufSectionStream<LlmMetadata>>(
            llm_metadata_proto);
    sections.push_back(std::move(pbs));
    section_types.push_back(AnySectionDataType_LlmMetadataProto);
    section_items_list.push_back(
        {});  // Add an empty vector, to be populated later
  }

  if (!binary_data.empty()) {
    std::unique_ptr<SectionStreamBase> fbs =
        std::make_unique<FileBackedSectionStream>(binary_data);
    sections.push_back(std::move(fbs));
    section_types.push_back(AnySectionDataType_GenericBinaryData);
    section_items_list.push_back(
        {});  // Add an empty vector, to be populated later
  }

  // Parse section metadata.
  flatbuffers::FlatBufferBuilder builder;
  std::map<std::string, std::vector<KVPair>> section_metadata_map;
  if (!section_metadata_str.empty()) {
    std::vector<std::string> section_parts =
        absl::StrSplit(section_metadata_str, ';');
    for (const auto& section_part : section_parts) {
      std::vector<std::string> parts = absl::StrSplit(section_part, ':');
      if (parts.size() != 2) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid section metadata format: ", section_part));
      }
      std::string section_name = parts[0];
      std::vector<std::string> kv_pairs = absl::StrSplit(parts[1], ',');
      for (const auto& kv_str : kv_pairs) {
        std::string key, value_str;
        absl::Status parsed_status =
            (ParseKeyValuePair(kv_str, key, value_str));
        if (!parsed_status.ok()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Failed to parse key-value pair: ", kv_str));
        }
        absl::StatusOr<KVPair> value = ConvertKeyValue(builder, key, value_str);
        if (!value.ok()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Failed to convert key-value pair: ", kv_str));
        }
        section_metadata_map[section_name].push_back(value.value());
      }
    }
  }

  // Assign parsed metadata to the correct section_items_list entries.
  // This logic assumes the order of sections matches the order of
  // tokenizer_file, tflite_file, and llm_metadata_file processing.
  int section_index = 0;
  if (!tokenizer_file.empty() &&
      section_metadata_map.count(kTokenizerSectionName)) {
    section_items_list[section_index] =
        section_metadata_map[kTokenizerSectionName];
    ++section_index;
  }
  if (!tflite_file.empty() && section_metadata_map.count(kTfliteSectionName)) {
    section_items_list[section_index] =
        section_metadata_map[kTfliteSectionName];
    ++section_index;
  }
  // Check for either llm_metadata or llm_metadata_text
  if ((!llm_metadata_file.empty() || !llm_metadata_text_file.empty()) &&
      section_metadata_map.count(kLlmMetadataSectionName)) {
    section_items_list[section_index] =
        section_metadata_map[kLlmMetadataSectionName];
    ++section_index;
  }
  std::vector<KVPair> system_meta = {
      CreateKeyValuePair(
          builder, std::string("arch"),
          CreateStringValue(builder, builder.CreateString(std::string("all")))),
      CreateKeyValuePair(builder, std::string("version"),
                         CreateStringValue(builder, builder.CreateString(
                                                        std::string("0.1"))))};

  absl::Status result =
      MakeLiteRTLMFromSections(builder, sections, section_types, system_meta,
                               section_items_list, output_path);

  return result;
}

}  // namespace

int main(int argc, char** argv) {
  ABSL_CHECK_OK(MainHelper(argc, argv));
  return 0;
}
