// Copyright 2026 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CHANNEL_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CHANNEL_UTIL_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/conversation/io_types.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// Extracts channel content from responses and removes it from the responses
// in-place. Returns a map from channel name to extracted content.
// Args:
//   channels: The list of channels to extract.
//   responses: The responses to extract channel content from.
//   open_channel_name: The name of the open channel, if any.
//     Open channels are usually due to a channel being started in the prefill
//     text.
// Returns:
//   A map from channel name to extracted content.
absl::StatusOr<absl::flat_hash_map<std::string, std::string>>
ExtractChannelContent(
    const std::vector<Channel>& channels, Responses& responses,
    const std::optional<std::string>& open_channel_name = std::nullopt);

// Inserts extracted channel content into the assistant message under the
// "channels" key.
void InsertChannelContentIntoMessage(
    const absl::flat_hash_map<std::string, std::string>& channel_content,
    Message& assistant_message);

// Detects if there is an open channel at the end of the text.
// A channel is open if the start tag is present but the end tag is not.
// If multiple channels are open, the one with the most recent start tag is
// returned.
std::optional<std::string> GetOpenChannelName(
    absl::string_view text, const std::vector<Channel>& channels);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CHANNEL_UTIL_H_
