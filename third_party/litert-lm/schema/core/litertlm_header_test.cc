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

#include "schema/core/litertlm_header.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "schema/core/litertlm_header_schema_generated.h"

namespace litert {
namespace lm {
namespace schema {
namespace {

// constexpr values for test data
constexpr int32_t kVersion = 123;
constexpr float kAccuracy = 0.987f;
constexpr bool kSFT = true;
constexpr uint64_t kParamCount = 305;
constexpr char kModelName[] = "Gemma3-4B";
constexpr size_t kSection1BeginOffset = 16 * 1024;
constexpr size_t kSection1EndOffset = 16 * 1024 + 2048;
constexpr size_t kSection2BeginOffset = 2 * 16 * 1024;
constexpr size_t kSection2EndOffset = 2 * 16 * 1024 + 48;

// Helper functions to extract and assert values.
void AssertStringValue(const KeyValuePair* pair, const char* key,
                       const char* expected_value) {
  ASSERT_TRUE(!!pair);
  ASSERT_STREQ(pair->key()->c_str(), key);
  ASSERT_EQ(pair->value_type(), VData::VData_StringValue);
  ASSERT_STREQ(pair->value_as_StringValue()->value()->c_str(), expected_value);
}

void AssertIntValue(const KeyValuePair* pair, const char* key,
                    int32_t expected_value) {
  ASSERT_TRUE(!!pair);
  ASSERT_STREQ(pair->key()->c_str(), key);
  ASSERT_EQ(pair->value_type(), VData::VData_Int32);
  ASSERT_EQ(pair->value_as_Int32()->value(), expected_value);
}

void AssertFloatValue(const KeyValuePair* pair, const char* key,
                      float expected_value) {
  ASSERT_TRUE(!!pair);
  ASSERT_STREQ(pair->key()->c_str(), key);
  ASSERT_EQ(pair->value_type(), VData::VData_Float32);
  ASSERT_FLOAT_EQ(pair->value_as_Float32()->value(),
                  expected_value);  // Use ASSERT_FLOAT_EQ for floats
}

void AssertBoolValue(const KeyValuePair* pair, const char* key,
                     bool expected_value) {
  ASSERT_TRUE(!!pair);
  ASSERT_STREQ(pair->key()->c_str(), key);
  ASSERT_EQ(pair->value_type(), VData::VData_Bool);
  ASSERT_EQ(pair->value_as_Bool()->value(), expected_value);
}

void AssertUInt64Value(const KeyValuePair* pair, const char* key,
                       uint64_t expected_value) {
  ASSERT_TRUE(!!pair);
  ASSERT_STREQ(pair->key()->c_str(), key);
  ASSERT_EQ(pair->value_type(), VData::VData_UInt64);
  ASSERT_EQ(pair->value_as_UInt64()->value(), expected_value);
}

TEST(LitertlmHeaderTest, RoundTripHeader) {
  flatbuffers::FlatBufferBuilder builder;

  auto value1_str = builder.CreateString(kModelName);
  flatbuffers::Offset<StringValue> string_data =
      CreateStringValue(builder, value1_str);
  auto kvp1 = CreateKeyValuePair(builder, "model_name", string_data);

  auto kvp2 = CreateKeyValuePair(builder, "version", kVersion);
  auto kvp3 = CreateKeyValuePair(builder, "accuracy", kAccuracy);
  auto kvp4 = CreateKeyValuePair(builder, "SFT", kSFT);
  auto kvp5 = CreateKeyValuePair(builder, "param_count", kParamCount);

  std::vector<flatbuffers::Offset<KeyValuePair>> kvp_vector = {kvp1, kvp2, kvp3,
                                                               kvp4, kvp5};
  auto system_metadata_offset =
      CreateSystemMetadata(builder, builder.CreateVector(kvp_vector));

  std::vector<flatbuffers::Offset<KeyValuePair>> section1_items_vector = {
      kvp1, kvp2, kvp3, kvp4, kvp5};
  auto section_object1 = CreateSectionObject(
      builder, builder.CreateVector(section1_items_vector),
      kSection1BeginOffset, kSection1EndOffset, AnySectionDataType_TFLiteModel);

  // Section metadata is optional. Create this section with no metadata.
  auto section_object2 =
      CreateSectionObject(builder, 0, kSection2BeginOffset, kSection2EndOffset,
                          AnySectionDataType_SP_Tokenizer);

  // All Section Object data.
  std::vector<flatbuffers::Offset<SectionObject>> section_objects_vector = {
      section_object1, section_object2};
  auto section_metadata_offset = CreateSectionMetadata(
      builder, builder.CreateVector(section_objects_vector));

  // Finish created LiteRTLMMetaData.
  auto root_offset = CreateLiteRTLMMetaData(builder, system_metadata_offset,
                                            section_metadata_offset);
  builder.Finish(root_offset);

  uint8_t* buffer = builder.GetBufferPointer();
  size_t size = builder.GetSize();

  // TODO(talumbau) Use in-memory files or std::tmpfile here.
  auto header_path = std::filesystem::path(testing::TempDir()) / "header.bin";
  std::ofstream output_file(header_path, std::ios::binary);
  output_file.write(reinterpret_cast<const char*>(buffer), size);

  std::ofstream::pos_type bytes_written = output_file.tellp();
  ASSERT_GT(bytes_written, 0);

  output_file.close();
  ABSL_LOG(INFO) << "Successfully wrote metadata to " << header_path;

  // Now read and verify that we wrote the correct data.
  std::ifstream input_stream(header_path, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(input_stream.is_open());

  size_t buffer_size = input_stream.tellg();
  input_stream.seekg(0, std::ios::beg);
  std::vector<char> file_buffer(buffer_size);
  ASSERT_TRUE(!!(input_stream.read(file_buffer.data(), buffer_size)));
  ABSL_LOG(INFO) << "Successfully read metadata to " << header_path;

  input_stream.close();

  // Get a pointer to the buffer.
  const uint8_t* buffer_ptr =
      reinterpret_cast<const uint8_t*>(file_buffer.data());

  // Get a pointer to the root object.
  auto metadata = GetLiteRTLMMetaData(buffer_ptr);

  // Access the SystemMetadata.
  auto system_metadata = metadata->system_metadata();
  ASSERT_TRUE(!!system_metadata);

  auto entries = system_metadata->entries();
  ASSERT_TRUE(!!entries);  // Ensure entries is not null

  // Extract and assert the values using the helper functions.
  ASSERT_EQ(entries->size(), 5);  // Check the number of key-value pairs.

  AssertStringValue(entries->Get(0), "model_name", kModelName);
  AssertIntValue(entries->Get(1), "version", kVersion);
  AssertFloatValue(entries->Get(2), "accuracy", kAccuracy);
  AssertBoolValue(entries->Get(3), "SFT", kSFT);
  AssertUInt64Value(entries->Get(4), "param_count", kParamCount);

  auto section_metadata_obj = metadata->section_metadata();
  ASSERT_TRUE(!!section_metadata_obj);
  auto section_objects = section_metadata_obj->objects();
  ASSERT_TRUE(!!section_objects);

  ASSERT_EQ(section_objects->size(), 2);

  auto s_object1 = section_objects->Get(0);
  ASSERT_TRUE(!!s_object1);
  ASSERT_EQ(s_object1->begin_offset(), kSection1BeginOffset);
  ASSERT_EQ(s_object1->end_offset(), kSection1EndOffset);
  ASSERT_EQ(s_object1->data_type(), AnySectionDataType_TFLiteModel);

  auto s_object2 = section_objects->Get(1);
  ASSERT_TRUE(!!s_object2);
  ASSERT_EQ(s_object2->begin_offset(), kSection2BeginOffset);
  ASSERT_EQ(s_object2->end_offset(), kSection2EndOffset);
  ASSERT_EQ(s_object2->data_type(), AnySectionDataType_SP_Tokenizer);
}

}  // namespace
}  // namespace schema
}  // namespace lm
}  // namespace litert
