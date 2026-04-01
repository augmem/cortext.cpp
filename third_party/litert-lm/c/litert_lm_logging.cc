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

#include "c/litert_lm_logging.h"

#include <cstdarg>
#include <cstdio>

#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"  // from @litert

extern "C" {

void litert_lm_log(int severity, const char* file, int line, const char* format,
                   ...) {
  va_list ap;
  va_start(ap, format);
  // A reasonable buffer size for log messages.
  char buf[1024];
  vsnprintf(buf, sizeof(buf), format, ap);
  va_end(ap);
  ABSL_LOG(LEVEL(severity)) << buf;
}

void litert_lm_set_min_log_level(int level) {
  absl::SetMinLogLevel(static_cast<absl::LogSeverityAtLeast>(level));
  LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(),
                             static_cast<LiteRtLogSeverity>(level));
}

}  // extern "C"
