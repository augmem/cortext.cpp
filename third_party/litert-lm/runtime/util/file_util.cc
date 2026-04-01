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

#include "runtime/util/file_util.h"

#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

namespace {

#if defined(_WIN32)
constexpr char kPathSeparator = '\\';
#else
constexpr char kPathSeparator = '/';
#endif

std::pair<absl::string_view, absl::string_view> SplitPath(
    absl::string_view path) {
  absl::string_view::size_type pos = path.find_last_of(kPathSeparator);

  // Handle the case with no '/' or '\' in 'path'.
  if (pos == absl::string_view::npos)
    return std::make_pair(path.substr(0, 0), path);

  // Handle the case with a single leading '/' or '\' in 'path'.
  if (pos == 0)
    return std::make_pair(path.substr(0, 1), absl::ClippedSubstr(path, 1));

  return std::make_pair(path.substr(0, pos + 1),
                        absl::ClippedSubstr(path, pos + 1));
}

}  // namespace

// 40% of the time in JoinPath() is from calls with 2 arguments, so we
// specialize that case.
absl::StatusOr<std::string> JoinPath(absl::string_view path1,
                                     absl::string_view path2) {
  if (path1.empty()) return absl::InvalidArgumentError("Empty path1.");
  if (path2.empty()) return absl::InvalidArgumentError("Empty path2.");
  if (path1.back() == kPathSeparator) {
    if (path2.front() == kPathSeparator)
      return absl::StrCat(path1, absl::ClippedSubstr(path2, 1));
  } else {
    if (path2.front() != kPathSeparator)
      return absl::StrCat(path1, std::string(1, kPathSeparator), path2);
  }
  return absl::StrCat(path1, path2);
}

absl::string_view Basename(absl::string_view path) {
  return SplitPath(path).second;
}

absl::string_view Dirname(absl::string_view path) {
  return SplitPath(path).first;
}

}  // namespace litert::lm
