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

#include "runtime/components/huggingface_tokenizer.h"

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/debugging/leak_check.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "include/tokenizers_cpp.h"  // from @tokenizers_cpp

namespace litert::lm {

// Replacement character (U+FFFD) in UTF-8.
// This character is used to represent incomplete BPE sequences (see below
// https://github.com/huggingface/tokenizers/blob/76abe0f77d409aec1687ead442cedaa0a8c058e8/tokenizers/src/decoders/byte_fallback.rs#L25)
const char kReplacementCharacter[] = "\xef\xbf\xbd";

// Checks if the decoded string ends with the replacement character, which
// indicates that the set of token IDs passed to the tokenizer is part of a BPE
// sequence and needs more tokens to be decoded.
static bool has_bpe_suffix(const std::string& decoded) {
  return decoded.ends_with(kReplacementCharacter);
}

absl::StatusOr<std::unique_ptr<HuggingFaceTokenizer>>
HuggingFaceTokenizer::CreateFromFile(absl::string_view json_path) {
  ASSIGN_OR_RETURN(auto memory_mapped_file,  // NOLINT
                   MemoryMappedFile::Create(json_path));
  std::string json_data(memory_mapped_file->length(), '\0');
  memcpy(json_data.data(), memory_mapped_file->data(),
         memory_mapped_file->length());
  return CreateFromJson(json_data);
}

absl::StatusOr<std::unique_ptr<HuggingFaceTokenizer>>
HuggingFaceTokenizer::CreateFromJson(std::string json) {
  auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(json);
  if (!tokenizer) {
    return absl::InvalidArgumentError("Failed to create tokenizer from JSON.");
  }
  return absl::WrapUnique(new HuggingFaceTokenizer(std::move(tokenizer)));
}

// Encodes the given text into a TensorBuffer of token ids.
absl::StatusOr<std::vector<int>> HuggingFaceTokenizer::TextToTokenIds(
    absl::string_view text) {
  {
    // Disable leak check as Google's default leak checker does not properly
    // support Rust's lazy_static initialization.
    // TODO(b/379364190) - Remove this once the leak checker is fixed.
    absl::LeakCheckDisabler disabler;
    return tokenizer_->Encode(std::string{text});
  }
}

absl::StatusOr<int> HuggingFaceTokenizer::TokenToId(absl::string_view token) {
  return tokenizer_->TokenToId(std::string{token});
}

// Decodes the given TensorBuffer of token ids into a vector of strings.
absl::StatusOr<std::string> HuggingFaceTokenizer::TokenIdsToText(
    const std::vector<int>& token_ids) {
  {
    absl::LeakCheckDisabler disabler;
    // Disable leak check as Google's default leak checker does not properly
    // support Rust's lazy_static initialization.
    // TODO(b/379364190) - Remove this once the leak checker is fixed.
    std::string decoded = tokenizer_->Decode(token_ids);
    if (has_bpe_suffix(decoded)) {
      return absl::DataLossError(
          "The set of token IDs passed to the tokenizer is part of a BPE "
          "sequence and needs more tokens to be decoded.");
    } else {
      return decoded;
    }
  }
}

}  // namespace litert::lm
