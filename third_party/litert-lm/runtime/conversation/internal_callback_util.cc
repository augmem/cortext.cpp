// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may
// may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/conversation/internal_callback_util.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {
namespace {

// Returns the number of overlapping characters between the suffix of string
// `a` and the prefix of string `b`.
size_t SuffixPrefixOverlap(absl::string_view a, absl::string_view b) {
  if (a.empty() || b.empty()) {
    return 0;
  }

  size_t max_overlap = std::min(a.length(), b.length());

  for (size_t len = max_overlap; len > 0; --len) {
    if (a.substr(a.length() - len) == b.substr(0, len)) {
      return len;
    }
  }

  return 0;
};

void SendMessage(
    absl::AnyInvocable<void(absl::StatusOr<Message>)>& user_callback,
    absl::string_view text, const ModelDataProcessor& model_data_processor,
    DataProcessorArguments processor_args) {
  if (text.empty()) {
    return;
  }
  auto message = model_data_processor.ToMessage(
      Responses(TaskState::kProcessing, {std::string(text)}), processor_args);
  if (!message.ok()) {
    user_callback(message.status());
    return;
  }
  user_callback(std::move(message.value()));
}

void SendCompleteMessage(
    absl::AnyInvocable<void(absl::StatusOr<Message>)>& user_callback,
    absl::string_view accumulated_response_text,
    const ModelDataProcessor& model_data_processor,
    DataProcessorArguments processor_args, int cursor,
    absl::AnyInvocable<void(Message)>& complete_message_callback) {
  if (cursor < accumulated_response_text.size()) {
    SendMessage(user_callback, accumulated_response_text.substr(cursor),
                model_data_processor, processor_args);
  }
  const auto& complete_message = model_data_processor.ToMessage(
      Responses(TaskState::kProcessing,
                {std::string(accumulated_response_text)}),
      processor_args);
  if (!complete_message.ok()) {
    user_callback(complete_message.status());
    return;
  }
  if (complete_message_callback) {
    complete_message_callback(complete_message.value());
  }
  user_callback(Message(JsonMessage()));
}

}  // namespace

absl::AnyInvocable<void(absl::StatusOr<Responses>)> CreateInternalCallback(
    const ModelDataProcessor& model_data_processor,
    const DataProcessorArguments processor_args,
    absl::AnyInvocable<void(absl::StatusOr<Message>)> user_callback,
    absl::AnyInvocable<void()> cancel_callback,
    absl::AnyInvocable<void(Message)> complete_message_callback) {
  return [&model_data_processor, processor_args,
          user_callback = std::move(user_callback),
          cancel_callback = std::move(cancel_callback),
          complete_message_callback = std::move(complete_message_callback),
          accumulated_response_text = std::string(), cursor = 0,
          inside_tool_call =
              false](absl::StatusOr<Responses> responses) mutable {
    if (!responses.ok()) {
      // If the error is due to cancellation, then we should trigger the cancel
      // callback for removing the last message from the history.
      if (cancel_callback && absl::IsCancelled(responses.status())) {
        cancel_callback();
      }
      user_callback(responses.status());
      return;
    }
    // If there are no more new responses, it means the model has finished
    // generating content, trigger the complete message callback and return an
    // OK status to indicate the inference is done.
    if (responses->GetTaskState() == TaskState::kDone ||
        responses->GetTaskState() == TaskState::kMaxNumTokensReached) {
      SendCompleteMessage(user_callback, accumulated_response_text,
                          model_data_processor, processor_args, cursor,
                          complete_message_callback);
      return;
    }
    // Else, add the new response text to the accumulated text and process the
    // response text.(Which sends to the user callback accordingly.)
    if (responses->GetTaskState() == TaskState::kProcessing) {
      // If there are no new responses, it is just a state update and we can
      // return early.
      if (responses->GetTexts().empty()) {
        return;
      }

      accumulated_response_text += responses->GetTexts()[0];

      absl::string_view code_fence_start =
          model_data_processor.CodeFenceStart();
      absl::string_view code_fence_end = model_data_processor.CodeFenceEnd();

      while (cursor < accumulated_response_text.size()) {
        if (!inside_tool_call) {
          size_t code_fence_start_pos =
              accumulated_response_text.find(code_fence_start, cursor);
          if (!code_fence_start.empty() &&
              code_fence_start_pos != std::string::npos) {
            // The text from the cursor up to the code fence is normal text.
            SendMessage(user_callback,
                        absl::string_view(accumulated_response_text)
                            .substr(cursor, code_fence_start_pos - cursor),
                        model_data_processor, processor_args);

            // Move cursor up to code_fence_start.
            cursor = code_fence_start_pos;
            inside_tool_call = true;
          } else {
            // code_fence_start not found, but we still need to check
            // if there's a partial match at the end of the string.
            size_t overlap = SuffixPrefixOverlap(
                accumulated_response_text.substr(cursor), code_fence_start);

            if (overlap > 0) {
              // There's a partial match of the code fence at the end of the
              // string.
              size_t possible_start_pos =
                  accumulated_response_text.size() - overlap;

              // Call the callback with text up to the potential start of the
              // code fence.
              SendMessage(user_callback,
                          accumulated_response_text.substr(
                              cursor, possible_start_pos - cursor),
                          model_data_processor, processor_args);

              // Move cursor up to potential start of code fence.
              cursor = possible_start_pos;
              break;
            } else {
              // Remaining string is text.
              SendMessage(user_callback,
                          accumulated_response_text.substr(cursor),
                          model_data_processor, processor_args);

              cursor = accumulated_response_text.size();
            }
          }
        }

        if (inside_tool_call) {
          // Look for code fence end.
          size_t code_fence_end_pos = accumulated_response_text.find(
              code_fence_end, cursor + code_fence_start.size());
          if (code_fence_end_pos != std::string::npos) {
            SendMessage(user_callback,
                        accumulated_response_text.substr(
                            cursor, code_fence_end_pos + code_fence_end.size() -
                                        cursor),
                        model_data_processor, processor_args);

            // Move cursor to end of tool code block.
            cursor = code_fence_end_pos + code_fence_end.size();
            inside_tool_call = false;
          } else {
            // We're inside a tool call but the code fence end has not been
            // found. Break for the next token.
            break;
          }
        }
      }
    }
  };
}

}  // namespace litert::lm
