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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_PRINT_H_
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_PRINT_H_

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "schema/core/litertlm_header_schema_generated.h"

namespace litert {

namespace lm {

namespace schema {

// Utility function to provide a string name for AnySectionDataType enum
// values.
std::string AnySectionDataTypeToString(AnySectionDataType value);

// Useful class that works around the lack of std::spanstream in C++23.
// It *should* be possible to make an inputstream from a known buffer
// (i.e. some pointer plus some length). MemoryStreamBuf provides the
// streambuf interface based on just that knowledge.
// NB: Delete this and replace with std::spanstream when C++23 is allowed.
class MemoryStreamBuf : public std::streambuf {
 public:
  // Constructor: Takes a pointer to the start of the buffer and its length
  MemoryStreamBuf(char* buffer, std::size_t length)
      : _buffer_start(buffer), _buffer_end(buffer + length) {
    // Set the buffer pointers for reading
    setg(_buffer_start, _buffer_start, _buffer_end);
  }

 protected:
  // Override underflow to handle reading past the current buffer (not typically
  // needed for a fixed memory buffer, but good practice for completeness)
  virtual int_type underflow() override {
    // If we've reached the end of the buffer, return EOF
    if (gptr() == egptr()) {
      return traits_type::eof();
    }
    // Otherwise, return the current character
    return traits_type::to_int_type(*gptr());
  }

  // Override seekoff to enable tellg() and seekg()
  virtual pos_type seekoff(
      off_type off, std::ios_base::seekdir way,
      std::ios_base::openmode which = std::ios_base::in |
                                      std::ios_base::out) override {
    // We only support input operations for this streambuf
    if (!(which & std::ios_base::in)) {
      return pos_type(off_type(-1));
    }

    char* new_gptr = nullptr;
    switch (way) {
      case std::ios_base::beg:
        new_gptr = _buffer_start + off;
        break;
      case std::ios_base::cur:
        new_gptr = gptr() + off;
        break;
      case std::ios_base::end:
        new_gptr = _buffer_end + off;
        break;
      default:
        return pos_type(off_type(-1));  // Invalid seekdir
    }

    // Check if the new position is within the valid range
    if (new_gptr < _buffer_start || new_gptr > _buffer_end) {
      return pos_type(off_type(-1));  // Invalid position
    }

    // Update the current get pointer
    setg(_buffer_start, new_gptr, _buffer_end);

    // Return the new absolute position (offset from beginning)
    return new_gptr - _buffer_start;
  }

 private:
  char* _buffer_start;
  char* _buffer_end;
};

}  // end namespace schema
}  // end namespace lm
}  // end namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_PRINT_H_
