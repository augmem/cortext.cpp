# Copyright 2025 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for inspecting the contents of a LiteRT-LM file."""

import os
import struct
from typing import IO, Optional

from google.protobuf import text_format

from litert_lm.runtime.proto import llm_metadata_pb2
from litert_lm.schema.core import litertlm_header_schema_py_generated as schema
from litert_lm.schema.py import litertlm_core

# --- ANSI Escape Code Definitions ---
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"
# --- Indentation Constants ---
INDENT_SPACES = 2


def print_boxed_title(
    output_stream: IO[str], title: str, box_width: int = 50
) -> None:
  """Prints a title surrounded by an ASCII box.

  Args:
    output_stream: The stream to write the output to.
    title: The title to print.
    box_width: The width of the box.
  """
  top_bottom = "+" + "-" * (box_width - 2) + "+"
  padding_left = (box_width - 2 - len(title)) // 2
  padding_right = box_width - 2 - len(title) - padding_left
  middle = "|" + " " * padding_left + title + " " * padding_right + "|"
  output_stream.write(f"{top_bottom}\n{middle}\n{top_bottom}\n")


def print_key_value_pair(
    kvp: schema.KeyValuePair, output_stream: IO[str], indent_level: int
) -> None:
  """Prints a formatted KeyValuePair.

  Args:
    kvp: The KeyValuePair object.
    output_stream: The stream to write the output to.
    indent_level: The indentation level.
  """
  indent_str = " " * (indent_level * INDENT_SPACES)
  if not kvp:
    output_stream.write(f"{indent_str}KeyValuePair: nullptr\n")
    return

  use_color = hasattr(output_stream, "isatty") and output_stream.isatty()
  bold = ANSI_BOLD if use_color else ""
  reset = ANSI_RESET if use_color else ""

  key_bytes = kvp.Key()
  key = key_bytes.decode("utf-8") if key_bytes is not None else None
  output_stream.write(f"{indent_str}{bold}Key{reset}: {key}, ")

  value_type = kvp.ValueType()
  union_table = kvp.Value()

  if union_table is None:
    output_stream.write(f"{bold}Value{reset}: <null>\n")
    return

  if value_type == schema.VData.StringValue:
    value_obj = schema.StringValue()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    value_bytes = value_obj.Value()
    value = value_bytes.decode("utf-8") if value_bytes is not None else None
    output_stream.write(f"{bold}Value{reset} (String): {value}\n")
  elif value_type == schema.VData.UInt8:
    value_obj = schema.UInt8()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (UInt8): {value_obj.Value()}\n")
  elif value_type == schema.VData.Int8:
    value_obj = schema.Int8()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (Int8): {value_obj.Value()}\n")
  elif value_type == schema.VData.UInt16:
    value_obj = schema.UInt16()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (UInt16): {value_obj.Value()}\n")
  elif value_type == schema.VData.Int16:
    value_obj = schema.Int16()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (Int16): {value_obj.Value()}\n")
  elif value_type == schema.VData.UInt32:
    value_obj = schema.UInt32()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (UInt32): {value_obj.Value()}\n")
  elif value_type == schema.VData.Int32:
    value_obj = schema.Int32()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (Int32): {value_obj.Value()}\n")
  elif value_type == schema.VData.UInt64:
    value_obj = schema.UInt64()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (UInt64): {value_obj.Value()}\n")
  elif value_type == schema.VData.Int64:
    value_obj = schema.Int64()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (Int64): {value_obj.Value()}\n")
  elif value_type == schema.VData.Double:
    value_obj = schema.Double()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(
        f"{bold}Value{reset} (Double): {value_obj.Value():.4f}\n"
    )
  elif value_type == schema.VData.Bool:
    value_obj = schema.Bool()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(
        f"{bold}Value{reset} (Bool): {bool(value_obj.Value())}\n"
    )
  else:
    output_stream.write(f"{bold}Value{reset} (Unknown Type)\n")


def read_litertlm_header(
    file_path: str, output_stream: IO[str]
) -> schema.LiteRTLMMetaData:
  """Reads the header of a LiteRT-LM file and returns the metadata.

  Args:
    file_path: The path to the LiteRT-LM file.
    output_stream: The stream to write version info to.

  Returns:
    The LiteRTLMMetaData object.

  Raises:
    ValueError: If the file has an invalid magic number.
  """
  with litertlm_core.open_file(file_path, "rb") as file_stream:
    magic = file_stream.read(8)
    if magic != b"LITERTLM":
      raise ValueError(f"Invalid magic number: {magic}")

    major, minor, patch = struct.unpack("<III", file_stream.read(12))
    output_stream.write(f"LiteRT-LM Version: {major}.{minor}.{patch}\n\n")

    file_stream.seek(litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET)
    header_end_offset = struct.unpack("<Q", file_stream.read(8))[0]

    file_stream.seek(litertlm_core.HEADER_BEGIN_BYTE_OFFSET)
    header_data = file_stream.read(
        header_end_offset - litertlm_core.HEADER_BEGIN_BYTE_OFFSET
    )

    metadata = schema.LiteRTLMMetaData.GetRootAs(header_data, 0)
    return metadata


def _get_model_type(section_object: schema.SectionObject) -> Optional[str]:
  """Extracts model_type from section items."""
  for j in range(section_object.ItemsLength()):
    item = section_object.Items(j)
    if item is None:
      continue
    key_bytes = item.Key()
    key = key_bytes.decode("utf-8") if key_bytes is not None else None
    if key == "model_type":
      value_type = item.ValueType()
      union_table = item.Value()
      if not (
          union_table
          and union_table.Bytes
          and union_table.Pos
          and value_type == schema.VData.StringValue
      ):
        continue
      value_obj = schema.StringValue()
      value_obj.Init(union_table.Bytes, union_table.Pos)
      value_bytes = value_obj.Value()
      return value_bytes.decode("utf-8") if value_bytes else None
  return None


def _get_tflite_model_filename(
    section_object: schema.SectionObject, section_index: int
) -> str:
  """Constructs a filename for a TFLiteModel section."""
  model_type = _get_model_type(section_object)
  file_name = f"Section{section_index}_TFLiteModel"
  if model_type:
    file_name += f"_{model_type}"
  return f"{file_name}.tflite"


def _get_tflite_weight_filename(
    section_object: schema.SectionObject, section_index: int
) -> str:
  """Constructs a filename for a TFLite weight section."""
  model_type = _get_model_type(section_object)
  file_name = f"Section{section_index}_TFLiteWeights"
  if model_type:
    file_name += f"_{model_type}"
  return f"{file_name}.weight"


def _dump_llm_metadata_proto(
    file_stream: IO[bytes],
    section_object: schema.SectionObject,
    dump_files_dir: str | None,
    output_stream: IO[str],
) -> None:
  """Dumps LlmMetadataProto section content."""
  file_stream.seek(section_object.BeginOffset())
  proto_data = file_stream.read(
      section_object.EndOffset() - section_object.BeginOffset()
  )
  llm_metadata = llm_metadata_pb2.LlmMetadata()
  llm_metadata.ParseFromString(proto_data)
  output_stream.write(f"{' ' * INDENT_SPACES}<<<<<<<< start of LlmMetadata\n")
  debug_str = text_format.MessageToString(llm_metadata)
  for line in debug_str.splitlines():
    output_stream.write(f"{' ' * (INDENT_SPACES * 2)}{line}\n")
  output_stream.write(f"{' ' * INDENT_SPACES}>>>>>>>> end of LlmMetadata\n")

  if dump_files_dir:
    file_path = os.path.join(dump_files_dir, "LlmMetadataProto.pbtext")
    with litertlm_core.open_file(file_path, "w") as f_out:
      f_out.write(debug_str)
    output_stream.write(
        f"{' ' * INDENT_SPACES}LlmMetadataProto dumped to: {file_path}\n"
    )


def _dump_section_content(
    file_stream: IO[bytes],
    section_object: schema.SectionObject,
    section_index: int,
    dump_files_dir: Optional[str],
    output_stream: IO[str],
    get_filename_fn,
) -> None:
  """Helper to dump section content to a file."""
  if dump_files_dir:
    file_name = get_filename_fn(section_object, section_index)
    file_path = os.path.join(dump_files_dir, file_name)
    file_stream.seek(section_object.BeginOffset())
    with litertlm_core.open_file(file_path, "wb") as f_out:
      f_out.write(
          file_stream.read(
              section_object.EndOffset() - section_object.BeginOffset()
          )
      )
    output_stream.write(
        f"{' ' * INDENT_SPACES}{file_name} dumped to: {file_path}\n"
    )


def _dump_tflite_model(
    file_stream: IO[bytes],
    section_object: schema.SectionObject,
    section_index: int,
    dump_files_dir: Optional[str],
    output_stream: IO[str],
) -> None:
  """Dumps TFLiteModel section content."""
  _dump_section_content(
      file_stream,
      section_object,
      section_index,
      dump_files_dir,
      output_stream,
      _get_tflite_model_filename,
  )


def _dump_tflite_weight(
    file_stream: IO[bytes],
    section_object: schema.SectionObject,
    section_index: int,
    dump_files_dir: Optional[str],
    output_stream: IO[str],
) -> None:
  """Dumps TFLite weight section content."""
  _dump_section_content(
      file_stream,
      section_object,
      section_index,
      dump_files_dir,
      output_stream,
      _get_tflite_weight_filename,
  )


def _get_generic_section_file_extension(data_type_str: str) -> str:
  """Returns the file extension for a generic section based on its data type."""
  if data_type_str == "SP_Tokenizer":
    return ".spiece"
  elif data_type_str == "HF_Tokenizer_Zlib":
    return ".json"
  else:
    return ".bin"


def _dump_generic_section(
    file_stream: IO[bytes],
    section_object: schema.SectionObject,
    section_index: int,
    dump_files_dir: Optional[str],
    output_stream: IO[str],
) -> None:
  """Dumps generic section content."""
  if dump_files_dir:
    data_type_str = litertlm_core.any_section_data_type_to_string(
        section_object.DataType()
    )
    file_extension = _get_generic_section_file_extension(data_type_str)
    file_name = f"Section{section_index}_{data_type_str}{file_extension}"
    file_path = os.path.join(dump_files_dir, file_name)
    file_stream.seek(section_object.BeginOffset())
    with litertlm_core.open_file(file_path, "wb") as f_out:
      f_out.write(
          file_stream.read(
              section_object.EndOffset() - section_object.BeginOffset()
          )
      )
    output_stream.write(
        f"{' ' * INDENT_SPACES}Section{section_index}_{data_type_str} dumped"
        f" to: {file_path}\n"
    )


def peek_litertlm_file(
    litertlm_path: str, dump_files_dir: Optional[str], output_stream: IO[str]
) -> None:
  """Reads and prints information from a LiteRT-LM file.

  Args:
    litertlm_path: The path to the LiteRT-LM file.
    dump_files_dir: Optional directory to dump section contents.
    output_stream: The stream to write the output to.
  """
  metadata = read_litertlm_header(litertlm_path, output_stream)
  with litertlm_core.open_file(litertlm_path, "rb") as file_stream:

    # Print System Metadata
    system_metadata = metadata.SystemMetadata()
    print_boxed_title(output_stream, "System Metadata")
    if system_metadata and system_metadata.EntriesLength() > 0:
      for i in range(system_metadata.EntriesLength()):
        print_key_value_pair(system_metadata.Entries(i), output_stream, 1)
    else:
      output_stream.write(" " * INDENT_SPACES + "No system metadata entries.\n")
    output_stream.write("\n")

    # Print Section Metadata
    section_metadata = metadata.SectionMetadata()
    num_sections = section_metadata.ObjectsLength() if section_metadata else 0
    print_boxed_title(output_stream, f"Sections ({num_sections})")

    if dump_files_dir:
      os.makedirs(dump_files_dir, exist_ok=True)

    if num_sections == 0 or section_metadata is None:
      output_stream.write(" " * INDENT_SPACES + "<None>\n")
    else:
      use_color = hasattr(output_stream, "isatty") and output_stream.isatty()
      bold = ANSI_BOLD if use_color else ""
      reset = ANSI_RESET if use_color else ""
      for i in range(num_sections):
        section_object = section_metadata.Objects(i)
        output_stream.write(f"\n{bold}Section {i}:{reset}\n")
        output_stream.write(" " * INDENT_SPACES + "Items:\n")
        if section_object is None:
          output_stream.write(" " * INDENT_SPACES + "<None>\n")
          continue

        # Print the items in the section.
        if section_object.ItemsLength() > 0:
          for j in range(section_object.ItemsLength()):
            print_key_value_pair(section_object.Items(j), output_stream, 2)
        else:
          output_stream.write(" " * (2 * INDENT_SPACES) + "<None>\n")

        output_stream.write(
            f"{' ' * INDENT_SPACES}Begin Offset:"
            f" {section_object.BeginOffset()}\n"
        )
        output_stream.write(
            f"{' ' * INDENT_SPACES}End Offset:   {section_object.EndOffset()}\n"
        )
        output_stream.write(
            f"{' ' * INDENT_SPACES}Data Type:    "
            f"{litertlm_core.any_section_data_type_to_string(section_object.DataType())}\n"
        )

        data_type = section_object.DataType()
        if data_type == schema.AnySectionDataType.LlmMetadataProto:
          _dump_llm_metadata_proto(
              file_stream, section_object, dump_files_dir, output_stream
          )
        elif data_type == schema.AnySectionDataType.TFLiteModel:
          _dump_tflite_model(
              file_stream, section_object, i, dump_files_dir, output_stream
          )
        elif data_type == schema.AnySectionDataType.TFLiteWeights:
          _dump_tflite_weight(
              file_stream, section_object, i, dump_files_dir, output_stream
          )
        else:
          _dump_generic_section(
              file_stream, section_object, i, dump_files_dir, output_stream
          )

        output_stream.write("\n")
