#include "cortext/models/aist_gguf_encoder.hpp"
#include "llama_cpp_support.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <Eigen/Dense>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#endif

#if defined(CORTEXT_ENABLE_LLAMA_CPP)
#include <ggml.h>
#include CORTEXT_GGML_BACKEND_HEADER_PATH
#endif

namespace cortext
{

namespace
{

enum class GgufValueType : std::uint32_t
{
  Uint8 = 0,
  Int8 = 1,
  Uint16 = 2,
  Int16 = 3,
  Uint32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  Uint64 = 10,
  Int64 = 11,
  Float64 = 12,
};

std::string
GetEnvOrDefault (const char *name, const std::string &fallback = {})
{
  const char *value = std::getenv (name);
  if (value == nullptr || *value == '\0')
    {
      return fallback;
    }
  return value;
}

int
GetEnvInt (const char *name, int fallback)
{
  const std::string value = GetEnvOrDefault (name);
  if (value.empty ())
    {
      return fallback;
    }
  try
    {
      return std::stoi (value);
    }
  catch (...)
    {
      return fallback;
    }
}

bool
GetEnvBool (const char *name, bool fallback)
{
  const std::string value = GetEnvOrDefault (name);
  if (value.empty ())
    {
      return fallback;
    }
  std::string lower (value);
  std::transform (lower.begin (), lower.end (), lower.begin (),
                  [] (unsigned char ch) {
                    return static_cast<char> (std::tolower (ch));
                  });
  return lower == "1" || lower == "true" || lower == "yes"
         || lower == "on";
}

template <typename T>
T
ReadScalar (std::istream &in)
{
  T value{};
  in.read (reinterpret_cast<char *> (&value), sizeof (T));
  if (!in)
    {
      throw std::runtime_error ("AIST GGUF inspect failed while reading scalar");
    }
  return value;
}

std::string
ReadString (std::istream &in)
{
  const std::uint64_t size = ReadScalar<std::uint64_t> (in);
  if (size > static_cast<std::uint64_t> (std::numeric_limits<int>::max ()))
    {
      throw std::runtime_error ("AIST GGUF string is unreasonably large");
    }
  std::string value (static_cast<std::size_t> (size), '\0');
  if (size > 0)
    {
      in.read (value.data (), static_cast<std::streamsize> (size));
      if (!in)
        {
          throw std::runtime_error (
              "AIST GGUF inspect failed while reading string");
        }
    }
  return value;
}

void
SkipBytes (std::istream &in, std::uint64_t count)
{
  if (count == 0)
    {
      return;
    }
  if (count > static_cast<std::uint64_t> (std::numeric_limits<int>::max ()))
    {
      throw std::runtime_error ("AIST GGUF value is unreasonably large");
    }
  in.seekg (static_cast<std::streamoff> (count), std::ios::cur);
  if (!in)
    {
      throw std::runtime_error ("AIST GGUF inspect failed while skipping bytes");
    }
}

std::uint64_t
ScalarSize (GgufValueType type)
{
  switch (type)
    {
    case GgufValueType::Uint8:
    case GgufValueType::Int8:
    case GgufValueType::Bool:
      return 1;
    case GgufValueType::Uint16:
    case GgufValueType::Int16:
      return 2;
    case GgufValueType::Uint32:
    case GgufValueType::Int32:
    case GgufValueType::Float32:
      return 4;
    case GgufValueType::Uint64:
    case GgufValueType::Int64:
    case GgufValueType::Float64:
      return 8;
    case GgufValueType::String:
    case GgufValueType::Array:
      return 0;
    }
  return 0;
}

void
SkipValue (std::istream &in, GgufValueType type)
{
  if (type == GgufValueType::String)
    {
      (void)ReadString (in);
      return;
    }
  if (type == GgufValueType::Array)
    {
      const auto element_type = static_cast<GgufValueType> (
          ReadScalar<std::uint32_t> (in));
      const std::uint64_t count = ReadScalar<std::uint64_t> (in);
      if (element_type == GgufValueType::String)
        {
          for (std::uint64_t i = 0; i < count; ++i)
            {
              (void)ReadString (in);
            }
          return;
        }
      if (element_type == GgufValueType::Array)
        {
          throw std::runtime_error ("AIST GGUF nested arrays are unsupported");
        }
      SkipBytes (in, ScalarSize (element_type) * count);
      return;
    }
  SkipBytes (in, ScalarSize (type));
}

std::string
ReadMetadataStringValue (std::istream &in, GgufValueType type)
{
  if (type == GgufValueType::String)
    {
      return ReadString (in);
    }
  if (type == GgufValueType::Uint32)
    {
      return std::to_string (ReadScalar<std::uint32_t> (in));
    }
  if (type == GgufValueType::Uint64)
    {
      return std::to_string (ReadScalar<std::uint64_t> (in));
    }
  if (type == GgufValueType::Int32)
    {
      return std::to_string (ReadScalar<std::int32_t> (in));
    }
  if (type == GgufValueType::Float32)
    {
      return std::to_string (ReadScalar<float> (in));
    }
  if (type == GgufValueType::Bool)
    {
      return ReadScalar<std::uint8_t> (in) == 0 ? "false" : "true";
    }
  if (type == GgufValueType::Array)
    {
      const auto element_type = static_cast<GgufValueType> (
          ReadScalar<std::uint32_t> (in));
      const std::uint64_t count = ReadScalar<std::uint64_t> (in);
      std::string value = "[";
      for (std::uint64_t i = 0; i < count; ++i)
        {
          if (i != 0)
            {
              value += ", ";
            }
          if (element_type == GgufValueType::String)
            {
              value += ReadString (in);
            }
          else if (element_type == GgufValueType::Uint32)
            {
              value += std::to_string (ReadScalar<std::uint32_t> (in));
            }
          else if (element_type == GgufValueType::Uint64)
            {
              value += std::to_string (ReadScalar<std::uint64_t> (in));
            }
          else
            {
              SkipBytes (in, ScalarSize (element_type));
              value += "?";
            }
        }
      value += "]";
      return value;
    }
  SkipValue (in, type);
  return {};
}

bool
ContainsTensor (const std::vector<std::string> &names, const std::string &name)
{
  return std::find (names.begin (), names.end (), name) != names.end ();
}

void
AppendIfPresent (std::vector<std::string> &values, const std::string &value)
{
  if (std::find (values.begin (), values.end (), value) == values.end ())
    {
      values.push_back (value);
    }
}

std::string
RuntimeUnavailableReason ()
{
  return "aist_runtime_backend_unavailable: AIST native triembed runtime was "
         "not initialized";
}

std::uint64_t
AlignOffset (std::uint64_t value, std::uint64_t alignment)
{
  if (alignment == 0)
    {
      return value;
    }
  const std::uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

std::uint64_t
Product (const std::vector<std::uint64_t> &shape)
{
  if (shape.empty ())
    {
      return 1;
    }
  std::uint64_t out = 1;
  for (std::uint64_t dim : shape)
    {
      out *= dim;
    }
  return out;
}

std::uint64_t
TensorByteSize (std::uint32_t ggml_type, std::uint64_t element_count)
{
  switch (ggml_type)
    {
    case 0: // F32
      return element_count * 4;
    case 1: // F16
      return element_count * 2;
    case 7: // Q5_1: d/m halfs + 32 5-bit values per block.
      return ((element_count + 31) / 32) * 24;
    case 8: // Q8_0: half scale + 32 int8 values per block.
      return ((element_count + 31) / 32) * 34;
    default:
      throw std::runtime_error ("AIST unsupported GGML tensor type: "
                                + std::to_string (ggml_type));
    }
}

float
HalfToFloat (std::uint16_t value)
{
  const std::uint32_t sign = (static_cast<std::uint32_t> (value & 0x8000U))
                             << 16U;
  std::uint32_t exponent = (value >> 10U) & 0x1FU;
  std::uint32_t mantissa = value & 0x03FFU;

  std::uint32_t bits = 0;
  if (exponent == 0)
    {
      if (mantissa == 0)
        {
          bits = sign;
        }
      else
        {
          exponent = 1;
          while ((mantissa & 0x0400U) == 0)
            {
              mantissa <<= 1U;
              --exponent;
            }
          mantissa &= 0x03FFU;
          bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
        }
    }
  else if (exponent == 0x1FU)
    {
      bits = sign | 0x7F800000U | (mantissa << 13U);
    }
  else
    {
      bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }

  float out = 0.0F;
  std::memcpy (&out, &bits, sizeof (out));
  return out;
}

std::uint16_t
ReadLe16 (const std::vector<std::uint8_t> &bytes, std::size_t offset)
{
  return static_cast<std::uint16_t> (bytes[offset])
         | (static_cast<std::uint16_t> (bytes[offset + 1]) << 8U);
}

float
ReadLeFloat32 (const std::vector<std::uint8_t> &bytes, std::size_t offset)
{
  float out = 0.0F;
  std::uint32_t raw = static_cast<std::uint32_t> (bytes[offset])
                      | (static_cast<std::uint32_t> (bytes[offset + 1]) << 8U)
                      | (static_cast<std::uint32_t> (bytes[offset + 2]) << 16U)
                      | (static_cast<std::uint32_t> (bytes[offset + 3]) << 24U);
  std::memcpy (&out, &raw, sizeof (out));
  return out;
}

std::vector<float>
DequantizeTensorBytes (std::uint32_t ggml_type,
                       const std::vector<std::uint8_t> &bytes,
                       std::uint64_t element_count)
{
  std::vector<float> out (static_cast<std::size_t> (element_count), 0.0F);
  if (ggml_type == 0)
    {
      for (std::uint64_t i = 0; i < element_count; ++i)
        {
          out[static_cast<std::size_t> (i)] = ReadLeFloat32 (
              bytes, static_cast<std::size_t> (i * 4));
        }
      return out;
    }
  if (ggml_type == 1)
    {
      for (std::uint64_t i = 0; i < element_count; ++i)
        {
          out[static_cast<std::size_t> (i)] = HalfToFloat (
              ReadLe16 (bytes, static_cast<std::size_t> (i * 2)));
        }
      return out;
    }
  if (ggml_type == 8)
    {
      std::size_t src = 0;
      std::uint64_t dst = 0;
      while (dst < element_count)
        {
          const float scale = HalfToFloat (ReadLe16 (bytes, src));
          src += 2;
          for (int i = 0; i < 32 && dst < element_count; ++i, ++dst)
            {
              const auto q = static_cast<std::int8_t> (bytes[src++]);
              out[static_cast<std::size_t> (dst)]
                  = scale * static_cast<float> (q);
            }
          if (dst >= element_count)
            {
              break;
            }
        }
      return out;
    }
  if (ggml_type == 7)
    {
      std::size_t src = 0;
      std::uint64_t block_start = 0;
      while (block_start < element_count)
        {
          const float scale = HalfToFloat (ReadLe16 (bytes, src));
          const float min_value = HalfToFloat (ReadLe16 (bytes, src + 2));
          src += 4;
          const std::uint32_t qh =
              static_cast<std::uint32_t> (bytes[src])
              | (static_cast<std::uint32_t> (bytes[src + 1]) << 8U)
              | (static_cast<std::uint32_t> (bytes[src + 2]) << 16U)
              | (static_cast<std::uint32_t> (bytes[src + 3]) << 24U);
          src += 4;
          const std::size_t qs_offset = src;
          src += 16;
          for (int j = 0; j < 16 && block_start + j < element_count; ++j)
            {
              const auto qh0 = static_cast<std::uint8_t> (
                  ((qh >> static_cast<std::uint32_t> (j)) << 4U) & 0x10U);
              const int q = (bytes[qs_offset + static_cast<std::size_t> (j)]
                             & 0x0FU)
                            | qh0;
              out[static_cast<std::size_t> (block_start + j)]
                  = static_cast<float> (q) * scale + min_value;
            }
          for (int j = 0; j < 16 && block_start + 16 + j < element_count; ++j)
            {
              const auto qh1 = static_cast<std::uint8_t> (
                  (qh >> static_cast<std::uint32_t> (j + 12)) & 0x10U);
              const int q = (bytes[qs_offset + static_cast<std::size_t> (j)]
                             >> 4U)
                            | qh1;
              out[static_cast<std::size_t> (block_start + 16 + j)]
                  = static_cast<float> (q) * scale + min_value;
            }
          block_start += 32;
        }
      return out;
    }
  throw std::runtime_error ("AIST unsupported GGML tensor type: "
                            + std::to_string (ggml_type));
}

bool
StartsWithAny (const std::string &value,
               const std::vector<std::string> &prefixes)
{
  for (const auto &prefix : prefixes)
    {
      if (value.rfind (prefix, 0) == 0)
        {
          return true;
        }
    }
  return false;
}

struct RuntimeTensor
{
  std::vector<std::size_t> shape;
  std::vector<float> data;

  std::size_t
  Rows () const
  {
    return shape.empty () ? data.size () : shape[0];
  }

  std::size_t
  Cols () const
  {
    return shape.size () < 2 ? 1 : shape[1];
  }
};

using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>;
using ConstMatrixMap = Eigen::Map<const RowMatrix>;

constexpr float kPi = 3.14159265358979323846F;

void ApplyGeluInPlace (std::vector<float> &values);

void
FftInPlace (std::vector<std::complex<float>> &a)
{
  const std::size_t n = a.size ();
  for (std::size_t i = 1, j = 0; i < n; ++i)
    {
      std::size_t bit = n >> 1U;
      for (; (j & bit) != 0; bit >>= 1U)
        {
          j ^= bit;
        }
      j ^= bit;
      if (i < j)
        {
          std::swap (a[i], a[j]);
        }
    }

  for (std::size_t len = 2; len <= n; len <<= 1U)
    {
      const float angle = -2.0F * kPi / static_cast<float> (len);
      const std::complex<float> wlen (std::cos (angle), std::sin (angle));
      for (std::size_t i = 0; i < n; i += len)
        {
          std::complex<float> w (1.0F, 0.0F);
          for (std::size_t j = 0; j < len / 2; ++j)
            {
              const auto u = a[i + j];
              const auto v = a[i + j + len / 2] * w;
              a[i + j] = u + v;
              a[i + j + len / 2] = u - v;
              w *= wlen;
            }
        }
    }
}

float
HtkMel (float hz)
{
  return 2595.0F * std::log10 (1.0F + hz / 700.0F);
}

float
HtkMelInv (float mel)
{
  return 700.0F * (std::pow (10.0F, mel / 2595.0F) - 1.0F);
}

std::vector<std::vector<float>>
BuildMelFilterbank (int sample_rate, int n_fft, int num_mel_bins,
                    float f_min, float f_max)
{
  const int n_freq = n_fft / 2 + 1;
  const float mel_min = HtkMel (f_min);
  const float mel_max = HtkMel (f_max);
  std::vector<float> mel_points (static_cast<std::size_t> (num_mel_bins + 2));
  for (int i = 0; i < num_mel_bins + 2; ++i)
    {
      mel_points[static_cast<std::size_t> (i)]
          = mel_min
            + (mel_max - mel_min) * static_cast<float> (i)
                  / static_cast<float> (num_mel_bins + 1);
    }

  std::vector<int> bins (static_cast<std::size_t> (num_mel_bins + 2));
  for (int i = 0; i < num_mel_bins + 2; ++i)
    {
      const float hz = HtkMelInv (mel_points[static_cast<std::size_t> (i)]);
      int bin = static_cast<int> (
          std::floor ((static_cast<float> (n_fft) + 1.0F) * hz
                      / static_cast<float> (sample_rate)));
      bins[static_cast<std::size_t> (i)] = std::clamp (bin, 0, n_freq - 1);
    }

  std::vector<std::vector<float>> fbanks (
      static_cast<std::size_t> (num_mel_bins),
      std::vector<float> (static_cast<std::size_t> (n_freq), 0.0F));
  for (int m = 0; m < num_mel_bins; ++m)
    {
      const int left = bins[static_cast<std::size_t> (m)];
      const int center = bins[static_cast<std::size_t> (m + 1)];
      const int right = bins[static_cast<std::size_t> (m + 2)];
      if (center == left || right == center)
        {
          continue;
        }
      for (int k = left; k < center; ++k)
        {
          fbanks[static_cast<std::size_t> (m)][static_cast<std::size_t> (k)]
              = static_cast<float> (k - left)
                / static_cast<float> (center - left);
        }
      for (int k = center; k < right; ++k)
        {
          fbanks[static_cast<std::size_t> (m)][static_cast<std::size_t> (k)]
              = static_cast<float> (right - k)
                / static_cast<float> (right - center);
        }
    }
  return fbanks;
}

std::vector<std::vector<std::pair<int, float>>>
BuildSparseMelFilterbank (int sample_rate, int n_fft, int num_mel_bins,
                          float f_min, float f_max)
{
  const auto dense = BuildMelFilterbank (sample_rate, n_fft, num_mel_bins,
                                         f_min, f_max);
  std::vector<std::vector<std::pair<int, float>>> sparse;
  sparse.reserve (dense.size ());
  for (const auto &row : dense)
    {
      std::vector<std::pair<int, float>> entries;
      entries.reserve (row.size ());
      for (std::size_t i = 0; i < row.size (); ++i)
        {
          if (row[i] != 0.0F)
            {
              entries.emplace_back (static_cast<int> (i), row[i]);
            }
        }
      sparse.push_back (std::move (entries));
    }
  return sparse;
}

float
GetPixelClamped (const std::uint8_t *data, int width, int height,
                 int channels, int x, int y, int c)
{
  x = std::clamp (x, 0, width - 1);
  y = std::clamp (y, 0, height - 1);
  c = std::clamp (c, 0, channels - 1);
  return static_cast<float> (data[(y * width + x) * channels + c]);
}

float
CubicWeight (float x)
{
  constexpr float a = -0.5F;
  x = std::fabs (x);
  if (x < 1.0F)
    {
      return (a + 2.0F) * x * x * x - (a + 3.0F) * x * x + 1.0F;
    }
  if (x < 2.0F)
    {
      return a * x * x * x - 5.0F * a * x * x + 8.0F * a * x - 4.0F * a;
    }
  return 0.0F;
}

std::vector<float>
PreprocessAistImageNHWC (const std::uint8_t *data, int width, int height,
                           int channels)
{
  if (data == nullptr || width <= 0 || height <= 0 || channels < 3)
    {
      throw std::runtime_error ("EncodeImage: invalid input image");
    }

  constexpr int kTarget = 384;
  std::vector<float> out (
      static_cast<std::size_t> (kTarget * kTarget * 3), 0.0F);
  const float scale_x = static_cast<float> (width) / static_cast<float> (kTarget);
  const float scale_y = static_cast<float> (height) / static_cast<float> (kTarget);
  const float mean[3] = { 0.485F, 0.456F, 0.406F };
  const float stdv[3] = { 0.229F, 0.224F, 0.225F };

  for (int y = 0; y < kTarget; ++y)
    {
      const float fy = (static_cast<float> (y) + 0.5F) * scale_y - 0.5F;
      const int iy = static_cast<int> (std::floor (fy));
      for (int x = 0; x < kTarget; ++x)
        {
          const float fx = (static_cast<float> (x) + 0.5F) * scale_x - 0.5F;
          const int ix = static_cast<int> (std::floor (fx));
          for (int c = 0; c < 3; ++c)
            {
              float acc = 0.0F;
              float wsum = 0.0F;
              for (int yy = -1; yy <= 2; ++yy)
                {
                  const float wy = CubicWeight (fy - static_cast<float> (iy + yy));
                  for (int xx = -1; xx <= 2; ++xx)
                    {
                      const float wx = CubicWeight (
                          fx - static_cast<float> (ix + xx));
                      const float w = wy * wx;
                      acc += w * GetPixelClamped (data, width, height,
                                                  channels, ix + xx,
                                                  iy + yy, c);
                      wsum += w;
                    }
                }
              if (wsum != 0.0F)
                {
                  acc /= wsum;
                }
              float value = std::clamp (acc / 255.0F, 0.0F, 1.0F);
              value = (value - mean[c]) / stdv[c];
              out[static_cast<std::size_t> (x + kTarget * (y + kTarget * c))]
                  = value;
            }
        }
    }
  return out;
}

std::vector<float>
PreprocessAistImageHWC (const std::uint8_t *data, int width, int height,
                          int channels)
{
  if (data == nullptr || width <= 0 || height <= 0 || channels < 3)
    {
      throw std::runtime_error ("EncodeImage: invalid input image");
    }

  constexpr int kTarget = 384;
  std::vector<float> out (
      static_cast<std::size_t> (kTarget * kTarget * 3), 0.0F);
  const float scale_x = static_cast<float> (width) / static_cast<float> (kTarget);
  const float scale_y = static_cast<float> (height) / static_cast<float> (kTarget);
  const float mean[3] = { 0.485F, 0.456F, 0.406F };
  const float stdv[3] = { 0.229F, 0.224F, 0.225F };

  for (int y = 0; y < kTarget; ++y)
    {
      const float fy = (static_cast<float> (y) + 0.5F) * scale_y - 0.5F;
      const int iy = static_cast<int> (std::floor (fy));
      for (int x = 0; x < kTarget; ++x)
        {
          const float fx = (static_cast<float> (x) + 0.5F) * scale_x - 0.5F;
          const int ix = static_cast<int> (std::floor (fx));
          for (int c = 0; c < 3; ++c)
            {
              float acc = 0.0F;
              float wsum = 0.0F;
              for (int yy = -1; yy <= 2; ++yy)
                {
                  const float wy = CubicWeight (fy - static_cast<float> (iy + yy));
                  for (int xx = -1; xx <= 2; ++xx)
                    {
                      const float wx = CubicWeight (
                          fx - static_cast<float> (ix + xx));
                      const float w = wy * wx;
                      acc += w * GetPixelClamped (data, width, height,
                                                  channels, ix + xx,
                                                  iy + yy, c);
                      wsum += w;
                    }
                }
              if (wsum != 0.0F)
                {
                  acc /= wsum;
                }
              float value = std::clamp (acc / 255.0F, 0.0F, 1.0F);
              value = (value - mean[c]) / stdv[c];
              out[static_cast<std::size_t> ((y * kTarget + x) * 3 + c)]
                  = value;
            }
        }
    }
  return out;
}

std::vector<float>
PreprocessAistAudioNHWC (const float *pcm, std::size_t num_samples)
{
  if (pcm == nullptr)
    {
      throw std::runtime_error ("EncodeAudio: pcm is null");
    }
  if (num_samples == 0)
    {
      throw std::runtime_error ("EncodeAudio: num_samples is 0");
    }

  constexpr int kSampleRate = 32000;
  constexpr int kSourceRate = 16000;
  constexpr int kClipSamples = kSampleRate * 10;
  constexpr int kWinLength = 800;
  constexpr int kHop = 320;
  constexpr int kNfft = 1024;
  constexpr int kMelBins = 128;
  constexpr int kFrames = 1000;
  constexpr float kFmin = 0.0F;
  constexpr float kFmax = 15000.0F;

  std::vector<float> audio (static_cast<std::size_t> (kClipSamples), 0.0F);
  const std::size_t source_needed = std::min (
      num_samples, static_cast<std::size_t> (kClipSamples / 2));
  for (std::size_t i = 0; i < source_needed; ++i)
    {
      const float current = pcm[i];
      const float next = (i + 1 < num_samples) ? pcm[i + 1] : current;
      const std::size_t dst = i * 2;
      audio[dst] = current;
      if (dst + 1 < audio.size ())
        {
          audio[dst + 1] = 0.5F * (current + next);
        }
    }
  (void)kSourceRate;

  std::vector<float> preemph (audio.size () - 1, 0.0F);
  for (std::size_t i = 0; i < preemph.size (); ++i)
    {
      preemph[i] = audio[i + 1] - 0.97F * audio[i];
    }

  std::vector<float> padded (preemph.size () + kNfft, 0.0F);
  const int pad = kNfft / 2;
  for (int i = 0; i < pad; ++i)
    {
      const int src = std::clamp (pad - i, 0,
                                  static_cast<int> (preemph.size ()) - 1);
      padded[static_cast<std::size_t> (i)] = preemph[static_cast<std::size_t> (src)];
    }
  std::copy (preemph.begin (), preemph.end (),
             padded.begin () + static_cast<std::ptrdiff_t> (pad));
  for (int i = 0; i < pad; ++i)
    {
      const int src = std::clamp (
          static_cast<int> (preemph.size ()) - 2 - i, 0,
          static_cast<int> (preemph.size ()) - 1);
      padded[static_cast<std::size_t> (pad + preemph.size () + i)]
          = preemph[static_cast<std::size_t> (src)];
    }

  static const std::vector<float> window = [] {
    std::vector<float> values (static_cast<std::size_t> (kWinLength));
    for (int i = 0; i < kWinLength; ++i)
      {
        values[static_cast<std::size_t> (i)]
            = 0.5F
              - 0.5F * std::cos (2.0F * kPi * static_cast<float> (i)
                                 / static_cast<float> (kWinLength - 1));
      }
    return values;
  } ();

  static const std::vector<std::vector<std::pair<int, float>>> mel_filterbank
      = BuildSparseMelFilterbank (kSampleRate, kNfft, kMelBins, kFmin, kFmax);
  const std::size_t active_preemph_end = std::min (
      preemph.size (), source_needed * static_cast<std::size_t> (2));
  const std::size_t active_padded_end =
      static_cast<std::size_t> (pad) + active_preemph_end;
  const float silence_log_mel =
      static_cast<float> ((std::log (1.0e-5) + 4.5) / 5.0);
  std::vector<float> out (
      static_cast<std::size_t> (kFrames * kMelBins), 0.0F);
  std::vector<std::complex<float>> fft (static_cast<std::size_t> (kNfft));
  std::vector<float> power (static_cast<std::size_t> (kNfft / 2 + 1));
  for (int frame = 0; frame < kFrames; ++frame)
    {
      const std::size_t start = static_cast<std::size_t> (frame * kHop);
      if (start >= active_padded_end)
        {
          for (int mel = 0; mel < kMelBins; ++mel)
            {
              out[static_cast<std::size_t> (frame + kFrames * mel)]
                  = silence_log_mel;
            }
          continue;
        }
      std::fill (fft.begin (), fft.end (), std::complex<float> (0.0F, 0.0F));
      for (int i = 0; i < kWinLength; ++i)
        {
          const std::size_t idx = start + static_cast<std::size_t> (i);
          if (idx < padded.size ())
            {
              fft[static_cast<std::size_t> (i)]
                  = std::complex<float> (
                      padded[idx] * window[static_cast<std::size_t> (i)],
                      0.0F);
            }
        }
      FftInPlace (fft);
      for (int k = 0; k <= kNfft / 2; ++k)
        {
          const auto value = fft[static_cast<std::size_t> (k)];
          power[static_cast<std::size_t> (k)] = value.real () * value.real ()
                                                + value.imag () * value.imag ();
        }
      for (int mel = 0; mel < kMelBins; ++mel)
        {
          double energy = 0.0;
          const auto &weights =
              mel_filterbank[static_cast<std::size_t> (mel)];
          for (const auto &[bin, weight] : weights)
            {
              energy += static_cast<double> (weight)
                        * static_cast<double> (
                            power[static_cast<std::size_t> (bin)]);
            }
          const float log_mel = static_cast<float> (std::log (energy + 1.0e-5));
          out[static_cast<std::size_t> (frame + kFrames * mel)]
              = (log_mel + 4.5F) / 5.0F;
        }
    }
  return out;
}

class RuntimeTensorStore
{
public:
  void
  Load (const std::filesystem::path &model_path,
        const std::vector<std::string> &prefixes)
  {
    std::ifstream in (model_path, std::ios::binary);
    if (!in)
      {
        throw std::runtime_error ("AIST native runtime model not found: "
                                  + model_path.string ());
      }

    char magic[4]{};
    in.read (magic, sizeof (magic));
    if (!in || std::memcmp (magic, "GGUF", 4) != 0)
      {
        throw std::runtime_error ("AIST native runtime expected GGUF: "
                                  + model_path.string ());
      }
    const std::uint32_t version = ReadScalar<std::uint32_t> (in);
    if (version < 2 || version > 3)
      {
        throw std::runtime_error ("Unsupported AIST GGUF version: "
                                  + std::to_string (version));
      }
    const std::uint64_t tensor_count = ReadScalar<std::uint64_t> (in);
    const std::uint64_t kv_count = ReadScalar<std::uint64_t> (in);

    std::uint64_t alignment = 32;
    for (std::uint64_t i = 0; i < kv_count; ++i)
      {
        const std::string key = ReadString (in);
        const auto type = static_cast<GgufValueType> (
            ReadScalar<std::uint32_t> (in));
        if (key == "general.alignment" && type == GgufValueType::Uint32)
          {
            alignment = ReadScalar<std::uint32_t> (in);
          }
        else
          {
            SkipValue (in, type);
          }
      }

    struct TensorRecord
    {
      std::string name;
      std::vector<std::uint64_t> shape;
      std::uint32_t type = 0;
      std::uint64_t offset = 0;
    };

    std::vector<TensorRecord> records;
    records.reserve (static_cast<std::size_t> (tensor_count));
    for (std::uint64_t i = 0; i < tensor_count; ++i)
      {
        TensorRecord record;
        record.name = ReadString (in);
        const std::uint32_t n_dims = ReadScalar<std::uint32_t> (in);
        record.shape.reserve (n_dims);
        for (std::uint32_t dim = 0; dim < n_dims; ++dim)
          {
            record.shape.push_back (ReadScalar<std::uint64_t> (in));
          }
        record.type = ReadScalar<std::uint32_t> (in);
        record.offset = ReadScalar<std::uint64_t> (in);
        if (StartsWithAny (record.name, prefixes))
          {
            records.push_back (std::move (record));
          }
      }

    const auto raw_data_start = static_cast<std::uint64_t> (in.tellg ());
    const std::uint64_t data_start = AlignOffset (raw_data_start, alignment);
    for (const auto &record : records)
      {
        const std::uint64_t count = Product (record.shape);
        const std::uint64_t byte_count = TensorByteSize (record.type, count);
        std::vector<std::uint8_t> bytes (static_cast<std::size_t> (byte_count));
        in.seekg (static_cast<std::streamoff> (data_start + record.offset),
                  std::ios::beg);
        in.read (reinterpret_cast<char *> (bytes.data ()),
                 static_cast<std::streamsize> (bytes.size ()));
        if (!in)
          {
            throw std::runtime_error ("AIST native runtime failed to read tensor "
                                      + record.name);
          }

        RuntimeTensor tensor;
        tensor.shape.reserve (record.shape.size ());
        for (auto it = record.shape.rbegin (); it != record.shape.rend (); ++it)
          {
            tensor.shape.push_back (static_cast<std::size_t> (*it));
          }
        tensor.data = DequantizeTensorBytes (record.type, bytes, count);
        tensors_.emplace (record.name, std::move (tensor));
      }
  }

  const RuntimeTensor &
  Required (const std::string &name) const
  {
    const auto it = tensors_.find (name);
    if (it == tensors_.end ())
      {
        throw std::runtime_error ("AIST native runtime missing tensor: " + name);
      }
    return it->second;
  }

  bool
  Contains (const std::string &name) const
  {
    return tensors_.find (name) != tensors_.end ();
  }

private:
  std::unordered_map<std::string, RuntimeTensor> tensors_;
};

struct HwcTensor
{
  int height = 0;
  int width = 0;
  int channels = 0;
  std::vector<float> data;

  HwcTensor () = default;
  HwcTensor (int h, int w, int c)
      : height (h), width (w), channels (c),
        data (static_cast<std::size_t> (h * w * c), 0.0F)
  {
  }

  float &
  operator() (int y, int x, int c)
  {
    return data[static_cast<std::size_t> ((y * width + x) * channels + c)];
  }

  float
  operator() (int y, int x, int c) const
  {
    return data[static_cast<std::size_t> ((y * width + x) * channels + c)];
  }
};

struct FoldedBatchNorm
{
  std::vector<float> scale;
  std::vector<float> bias;
};

FoldedBatchNorm
LoadFoldedBatchNorm (const RuntimeTensorStore &store,
                     const std::string &prefix)
{
  const RuntimeTensor &weight = store.Required (prefix + ".weight");
  const RuntimeTensor &bias = store.Required (prefix + ".bias");
  const RuntimeTensor &mean = store.Required (prefix + ".running_mean");
  const RuntimeTensor &var = store.Required (prefix + ".running_var");
  const std::size_t count = var.data.size ();
  if (weight.data.size () != count || bias.data.size () != count
      || mean.data.size () != count)
    {
      throw std::runtime_error ("AIST image batchnorm mismatch for "
                                + prefix);
    }
  FoldedBatchNorm out;
  out.scale.resize (count);
  out.bias.resize (count);
  for (std::size_t i = 0; i < count; ++i)
    {
      const float inv_std = 1.0F / std::sqrt (var.data[i] + 1.0e-5F);
      out.scale[i] = weight.data[i] * inv_std;
      out.bias[i] = bias.data[i] - mean.data[i] * out.scale[i];
    }
  return out;
}

std::vector<float>
MulRowsByWeightTranspose (const float *rows, int row_count,
                          int col_count, const RuntimeTensor &weight,
                          int out_count)
{
    std::vector<float> out (
        static_cast<std::size_t> (row_count * out_count), 0.0F);
#if defined(__APPLE__)
  cblas_sgemm (CblasRowMajor, CblasNoTrans, CblasTrans, row_count,
               out_count, col_count, 1.0F, rows, col_count,
               weight.data.data (), col_count, 0.0F, out.data (),
               out_count);
#else
  ConstMatrixMap x (rows, static_cast<Eigen::Index> (row_count),
                    static_cast<Eigen::Index> (col_count));
  ConstMatrixMap w (weight.data.data (),
                    static_cast<Eigen::Index> (out_count),
                    static_cast<Eigen::Index> (col_count));
  RowMatrix y = x * w.transpose ();
  std::copy (y.data (), y.data () + y.size (), out.begin ());
#endif
  return out;
}

std::vector<float>
LinearHostVector (const RuntimeTensorStore &store,
                  const std::vector<float> &input,
                  const std::string &prefix)
{
  const RuntimeTensor &weight = store.Required (prefix + ".weight");
  const RuntimeTensor &bias = store.Required (prefix + ".bias");
  const int rows = static_cast<int> (weight.Rows ());
  const int cols = static_cast<int> (weight.Cols ());
  if (static_cast<int> (input.size ()) != cols
      || static_cast<int> (bias.data.size ()) != rows)
    {
      throw std::runtime_error ("AIST host linear mismatch for " + prefix);
    }
  std::vector<float> out = MulRowsByWeightTranspose (
      input.data (), 1, cols, weight, rows);
  for (int i = 0; i < rows; ++i)
    {
      out[static_cast<std::size_t> (i)] += bias.data[static_cast<std::size_t> (i)];
    }
  return out;
}

std::vector<float>
LayerNormHostVector (const RuntimeTensorStore &store,
                     const std::vector<float> &input,
                     const std::string &prefix, float eps)
{
  const RuntimeTensor &weight = store.Required (prefix + ".weight");
  const RuntimeTensor &bias = store.Required (prefix + ".bias");
  if (weight.data.size () != input.size () || bias.data.size () != input.size ())
    {
      throw std::runtime_error ("AIST host layernorm mismatch for "
                                + prefix);
    }
  double mean = 0.0;
  for (float value : input)
    {
      mean += value;
    }
  mean /= static_cast<double> (input.size ());
  double variance = 0.0;
  for (float value : input)
    {
      const double centered = static_cast<double> (value) - mean;
      variance += centered * centered;
    }
  variance /= static_cast<double> (input.size ());
  const double inv = 1.0 / std::sqrt (variance + static_cast<double> (eps));
  std::vector<float> out (input.size ());
  for (std::size_t i = 0; i < input.size (); ++i)
    {
      out[i] = static_cast<float> (
          ((static_cast<double> (input[i]) - mean) * inv)
              * static_cast<double> (weight.data[i])
          + static_cast<double> (bias.data[i]));
    }
  return out;
}

std::vector<float>
ProjectionBlockHostVector (const RuntimeTensorStore &store,
                           const std::vector<float> &input,
                           const std::string &prefix)
{
  std::vector<float> x = LinearHostVector (store, input, prefix + ".expand.0");
  ApplyGeluInPlace (x);
  x = LayerNormHostVector (store, x, prefix + ".expand.2", 1e-5F);
  for (int block = 0;; ++block)
    {
      const std::string block_prefix =
          prefix + ".residual_blocks." + std::to_string (block);
      if (!store.Contains (block_prefix + ".0.weight"))
        {
          break;
        }
      std::vector<float> y = LinearHostVector (store, x, block_prefix + ".0");
      ApplyGeluInPlace (y);
      y = LayerNormHostVector (store, y, block_prefix + ".2", 1e-5F);
      for (std::size_t i = 0; i < x.size (); ++i)
        {
          x[i] += y[i];
        }
    }
  return LinearHostVector (store, x, prefix + ".project");
}

std::vector<float>
MulRowsByWeightTranspose (const std::vector<float> &rows, int row_count,
                          int col_count, const RuntimeTensor &weight,
                          int out_count)
{
  return MulRowsByWeightTranspose (rows.data (), row_count, col_count,
                                   weight, out_count);
}

void
ApplyBatchNormAndActivationInPlace (HwcTensor &x, const FoldedBatchNorm &bn,
                                    bool relu)
{
  if (bn.scale.size () != static_cast<std::size_t> (x.channels)
      || bn.bias.size () != static_cast<std::size_t> (x.channels))
    {
      throw std::runtime_error ("AIST image batchnorm channel mismatch");
    }
  for (int y = 0; y < x.height; ++y)
    {
      for (int px = 0; px < x.width; ++px)
        {
          for (int c = 0; c < x.channels; ++c)
            {
              float value = x (y, px, c) * bn.scale[static_cast<std::size_t> (c)]
                            + bn.bias[static_cast<std::size_t> (c)];
              if (relu && value < 0.0F)
                {
                  value = 0.0F;
                }
              x (y, px, c) = value;
            }
        }
    }
}

HwcTensor
StandardConv2dHwc (const HwcTensor &input, const RuntimeTensor &weight,
                   int stride, int pad)
{
  if (weight.shape.size () != 4)
    {
      throw std::runtime_error ("AIST image conv weight must be 4D");
    }
  const int out_channels = static_cast<int> (weight.shape[0]);
  const int in_channels = static_cast<int> (weight.shape[1]);
  const int kh = static_cast<int> (weight.shape[2]);
  const int kw = static_cast<int> (weight.shape[3]);
  if (in_channels != input.channels)
    {
      throw std::runtime_error ("AIST image conv input channel mismatch");
    }
  const int out_h = (input.height + 2 * pad - kh) / stride + 1;
  const int out_w = (input.width + 2 * pad - kw) / stride + 1;
  const int row_count = out_h * out_w;
  const int col_count = in_channels * kh * kw;
  std::vector<float> patches (
      static_cast<std::size_t> (row_count * col_count), 0.0F);
  for (int oy = 0; oy < out_h; ++oy)
    {
      for (int ox = 0; ox < out_w; ++ox)
        {
          const int row = oy * out_w + ox;
          int col = 0;
          for (int ic = 0; ic < in_channels; ++ic)
            {
              for (int ky = 0; ky < kh; ++ky)
                {
                  const int iy = oy * stride + ky - pad;
                  for (int kx = 0; kx < kw; ++kx)
                    {
                      const int ix = ox * stride + kx - pad;
                      float value = 0.0F;
                      if (iy >= 0 && iy < input.height && ix >= 0
                          && ix < input.width)
                        {
                          value = input (iy, ix, ic);
                        }
                      patches[static_cast<std::size_t> (row * col_count
                                                        + col++)]
                          = value;
                    }
                }
            }
        }
    }
  std::vector<float> out_rows = MulRowsByWeightTranspose (
      patches, row_count, col_count, weight, out_channels);
  HwcTensor out (out_h, out_w, out_channels);
  out.data = std::move (out_rows);
  return out;
}

HwcTensor
DepthwiseConv2dHwc (const HwcTensor &input, const RuntimeTensor &weight,
                    int stride, int pad)
{
  if (weight.shape.size () != 4)
    {
      throw std::runtime_error ("AIST image depthwise weight must be 4D");
    }
  const int channels = static_cast<int> (weight.shape[0]);
  const int multiplier = static_cast<int> (weight.shape[1]);
  const int kh = static_cast<int> (weight.shape[2]);
  const int kw = static_cast<int> (weight.shape[3]);
  if (channels != input.channels || multiplier != 1)
    {
      throw std::runtime_error (
          "AIST image depthwise channel mismatch");
    }
  const int out_h = (input.height + 2 * pad - kh) / stride + 1;
  const int out_w = (input.width + 2 * pad - kw) / stride + 1;
  HwcTensor out (out_h, out_w, channels);
  for (int oy = 0; oy < out_h; ++oy)
    {
      for (int ox = 0; ox < out_w; ++ox)
        {
          for (int c = 0; c < channels; ++c)
            {
              float sum = 0.0F;
              for (int ky = 0; ky < kh; ++ky)
                {
                  const int iy = oy * stride + ky - pad;
                  if (iy < 0 || iy >= input.height)
                    {
                      continue;
                    }
                  for (int kx = 0; kx < kw; ++kx)
                    {
                      const int ix = ox * stride + kx - pad;
                      if (ix < 0 || ix >= input.width)
                        {
                          continue;
                        }
                      const auto w_idx = static_cast<std::size_t> (
                          ((c * multiplier) * kh + ky) * kw + kx);
                      sum += input (iy, ix, c) * weight.data[w_idx];
                    }
                }
              out (oy, ox, c) = sum;
            }
        }
    }
  return out;
}

HwcTensor
PointwiseConv2dHwc (const HwcTensor &input, const RuntimeTensor &weight)
{
  if (weight.shape.size () != 4 || weight.shape[2] != 1 || weight.shape[3] != 1)
    {
      throw std::runtime_error ("AIST image pointwise weight must be 1x1");
    }
  const int out_channels = static_cast<int> (weight.shape[0]);
  const int in_channels = static_cast<int> (weight.shape[1]);
  if (in_channels != input.channels)
    {
      throw std::runtime_error (
          "AIST image pointwise input channel mismatch");
    }
  const int row_count = input.height * input.width;
  std::vector<float> out_rows = MulRowsByWeightTranspose (
      input.data.data (), row_count, in_channels, weight, out_channels);
  HwcTensor out (input.height, input.width, out_channels);
  out.data = std::move (out_rows);
  return out;
}

HwcTensor
ImageConvBnActNative (const RuntimeTensorStore &store, const HwcTensor &input,
                      const std::string &conv_prefix,
                      const std::string &bn_prefix, int stride, int pad,
                      bool depthwise, bool relu)
{
  const RuntimeTensor &weight = store.Required (conv_prefix + ".weight");
  HwcTensor out = depthwise ? DepthwiseConv2dHwc (input, weight, stride, pad)
                            : (weight.shape[2] == 1 && weight.shape[3] == 1
                                   ? PointwiseConv2dHwc (input, weight)
                                   : StandardConv2dHwc (input, weight, stride,
                                                        pad));
  ApplyBatchNormAndActivationInPlace (out,
                                      LoadFoldedBatchNorm (store, bn_prefix),
                                      relu);
  return out;
}

void
AddResidualInPlace (HwcTensor &x, const HwcTensor &residual)
{
  if (x.height != residual.height || x.width != residual.width
      || x.channels != residual.channels || x.data.size () != residual.data.size ())
    {
      return;
    }
  for (std::size_t i = 0; i < x.data.size (); ++i)
    {
      x.data[i] += residual.data[i];
    }
}

HwcTensor
ImageEdgeBlockNative (const RuntimeTensorStore &store, const HwcTensor &input,
                      const std::string &prefix)
{
  HwcTensor x = ImageConvBnActNative (store, input, prefix + ".conv_exp",
                                      prefix + ".bn1", 2, 1, false, true);
  return ImageConvBnActNative (store, x, prefix + ".conv_pwl",
                               prefix + ".bn2", 1, 0, false, false);
}

HwcTensor
ImageUniversalBlockNative (const RuntimeTensorStore &store,
                           const HwcTensor &input,
                           const std::string &prefix, int dw_mid_stride)
{
  const HwcTensor residual = input;
  HwcTensor x = input;
  if (store.Contains (prefix + ".dw_start.conv.weight"))
    {
      const RuntimeTensor &weight = store.Required (
          prefix + ".dw_start.conv.weight");
      const int kernel = static_cast<int> (weight.shape[2]);
      x = ImageConvBnActNative (store, x, prefix + ".dw_start.conv",
                                prefix + ".dw_start.bn", 1, kernel / 2,
                                true, false);
    }
  x = ImageConvBnActNative (store, x, prefix + ".pw_exp.conv",
                            prefix + ".pw_exp.bn", 1, 0, false, true);
  if (store.Contains (prefix + ".dw_mid.conv.weight"))
    {
      const RuntimeTensor &weight = store.Required (
          prefix + ".dw_mid.conv.weight");
      const int kernel = static_cast<int> (weight.shape[2]);
      x = ImageConvBnActNative (store, x, prefix + ".dw_mid.conv",
                                prefix + ".dw_mid.bn", dw_mid_stride,
                                kernel / 2, true, true);
    }
  x = ImageConvBnActNative (store, x, prefix + ".pw_proj.conv",
                            prefix + ".pw_proj.bn", 1, 0, false, false);
  AddResidualInPlace (x, residual);
  return x;
}

std::vector<float>
AveragePoolChannels (const HwcTensor &x)
{
  std::vector<float> out (static_cast<std::size_t> (x.channels), 0.0F);
  const float inv = 1.0F / static_cast<float> (x.height * x.width);
  for (int y = 0; y < x.height; ++y)
    {
      for (int px = 0; px < x.width; ++px)
        {
          for (int c = 0; c < x.channels; ++c)
            {
              out[static_cast<std::size_t> (c)] += x (y, px, c) * inv;
            }
        }
    }
  return out;
}

std::vector<float>
EncodeAistImageFeaturesNative (const RuntimeTensorStore &store,
                                 const std::vector<float> &input_hwc)
{
  if (input_hwc.size () != static_cast<std::size_t> (384 * 384 * 3))
    {
      throw std::runtime_error ("AIST native image input must be 384x384x3");
    }
  HwcTensor x (384, 384, 3);
  x.data = input_hwc;
  x = ImageConvBnActNative (store, x, "image_encoder.conv_stem",
                            "image_encoder.bn1", 2, 1, false, true);
  x = ImageEdgeBlockNative (store, x, "image_encoder.blocks.0.0");
  const std::vector<std::pair<std::string, int>> blocks = {
    { "image_encoder.blocks.1.0", 2 },
    { "image_encoder.blocks.1.1", 1 },
    { "image_encoder.blocks.2.0", 2 },
    { "image_encoder.blocks.2.1", 1 },
    { "image_encoder.blocks.2.2", 1 },
    { "image_encoder.blocks.2.3", 1 },
    { "image_encoder.blocks.2.4", 1 },
    { "image_encoder.blocks.2.5", 1 },
    { "image_encoder.blocks.2.6", 1 },
    { "image_encoder.blocks.2.7", 1 },
    { "image_encoder.blocks.3.0", 2 },
    { "image_encoder.blocks.3.1", 1 },
    { "image_encoder.blocks.3.2", 1 },
    { "image_encoder.blocks.3.3", 1 },
    { "image_encoder.blocks.3.4", 1 },
    { "image_encoder.blocks.3.5", 1 },
    { "image_encoder.blocks.3.6", 1 },
    { "image_encoder.blocks.3.7", 1 },
    { "image_encoder.blocks.3.8", 1 },
    { "image_encoder.blocks.3.9", 1 },
    { "image_encoder.blocks.3.10", 1 },
  };
  for (const auto &block : blocks)
    {
      x = ImageUniversalBlockNative (store, x, block.first, block.second);
    }
  x = ImageConvBnActNative (store, x, "image_encoder.blocks.4.0.conv",
                            "image_encoder.blocks.4.0.bn1", 1, 0, false,
                            true);
  std::vector<float> pooled = AveragePoolChannels (x);
  HwcTensor pooled_tensor (1, 1, static_cast<int> (pooled.size ()));
  pooled_tensor.data = std::move (pooled);
  x = PointwiseConv2dHwc (pooled_tensor,
                          store.Required ("image_encoder.conv_head.weight"));
  ApplyBatchNormAndActivationInPlace (
      x, LoadFoldedBatchNorm (store, "image_encoder.norm_head"), true);
  return x.data;
}

#if defined(CORTEXT_ENABLE_LLAMA_CPP)
enum class GgmlLinearLoadStatus
{
  Loaded,
  Disabled,
  Unavailable,
};

enum ggml_type
ToGgmlType (std::uint32_t type)
{
  switch (type)
    {
    case 0:
      return GGML_TYPE_F32;
    case 1:
      return GGML_TYPE_F16;
    case 7:
      return GGML_TYPE_Q5_1;
    case 8:
      return GGML_TYPE_Q8_0;
    default:
      throw std::runtime_error ("AIST ggml kernel unsupported tensor type: "
                                + std::to_string (type));
    }
}

struct RawGgufTensor
{
  std::string name;
  std::vector<std::uint64_t> shape;
  std::uint32_t type = 0;
  std::vector<std::uint8_t> bytes;
};

std::vector<RawGgufTensor>
LoadRawGgufTensors (const std::filesystem::path &model_path,
                    const std::vector<std::string> &prefixes)
{
  std::ifstream in (model_path, std::ios::binary);
  if (!in)
    {
      throw std::runtime_error ("AIST ggml kernel model not found: "
                                + model_path.string ());
    }

  char magic[4]{};
  in.read (magic, sizeof (magic));
  if (!in || std::memcmp (magic, "GGUF", 4) != 0)
    {
      throw std::runtime_error ("AIST ggml kernel expected GGUF: "
                                + model_path.string ());
    }
  const std::uint32_t version = ReadScalar<std::uint32_t> (in);
  if (version < 2 || version > 3)
    {
      throw std::runtime_error ("Unsupported AIST GGUF version: "
                                + std::to_string (version));
    }
  const std::uint64_t tensor_count = ReadScalar<std::uint64_t> (in);
  const std::uint64_t kv_count = ReadScalar<std::uint64_t> (in);

  std::uint64_t alignment = 32;
  for (std::uint64_t i = 0; i < kv_count; ++i)
    {
      const std::string key = ReadString (in);
      const auto type = static_cast<GgufValueType> (
          ReadScalar<std::uint32_t> (in));
      if (key == "general.alignment" && type == GgufValueType::Uint32)
        {
          alignment = ReadScalar<std::uint32_t> (in);
        }
      else
        {
          SkipValue (in, type);
        }
    }

  struct TensorRecord
  {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::uint32_t type = 0;
    std::uint64_t offset = 0;
  };

  std::vector<TensorRecord> records;
  records.reserve (static_cast<std::size_t> (tensor_count));
  for (std::uint64_t i = 0; i < tensor_count; ++i)
    {
      TensorRecord record;
      record.name = ReadString (in);
      const std::uint32_t n_dims = ReadScalar<std::uint32_t> (in);
      record.shape.reserve (n_dims);
      for (std::uint32_t dim = 0; dim < n_dims; ++dim)
        {
          record.shape.push_back (ReadScalar<std::uint64_t> (in));
        }
      record.type = ReadScalar<std::uint32_t> (in);
      record.offset = ReadScalar<std::uint64_t> (in);
      if (StartsWithAny (record.name, prefixes))
        {
          records.push_back (std::move (record));
        }
    }

  const auto raw_data_start = static_cast<std::uint64_t> (in.tellg ());
  const std::uint64_t data_start = AlignOffset (raw_data_start, alignment);
  std::vector<RawGgufTensor> out;
  out.reserve (records.size ());
  for (const auto &record : records)
    {
      const std::uint64_t count = Product (record.shape);
      const std::uint64_t byte_count = TensorByteSize (record.type, count);
      RawGgufTensor tensor;
      tensor.name = record.name;
      tensor.shape = record.shape;
      tensor.type = record.type;
      tensor.bytes.resize (static_cast<std::size_t> (byte_count));
      in.seekg (static_cast<std::streamoff> (data_start + record.offset),
                std::ios::beg);
      in.read (reinterpret_cast<char *> (tensor.bytes.data ()),
               static_cast<std::streamsize> (tensor.bytes.size ()));
      if (!in)
        {
          throw std::runtime_error ("AIST ggml kernel failed to read tensor "
                                    + record.name);
        }
      out.push_back (std::move (tensor));
    }
  return out;
}

class GgmlLinearKernel
{
public:
  ~GgmlLinearKernel ()
  {
    image_plan_.reset ();
    audio_plan_.reset ();
    plans_.clear ();
    full_text_plans_.clear ();
    if (weights_buffer_ != nullptr)
      {
        ggml_backend_buffer_free (weights_buffer_);
      }
    if (weights_ctx_ != nullptr)
      {
        ggml_free (weights_ctx_);
      }
    if (backend_ != nullptr)
      {
        ggml_backend_free (backend_);
      }
  }

  GgmlLinearLoadStatus
  Load (const std::filesystem::path &model_path,
        const std::vector<std::string> &prefixes, int threads,
        std::string &error)
  {
    if (GetEnvBool ("CORTEXT_AIST_DISABLE_GGML_KERNELS", false))
      {
        error = "disabled by CORTEXT_AIST_DISABLE_GGML_KERNELS";
        return GgmlLinearLoadStatus::Disabled;
      }

    try
      {
        raw_tensors_ = LoadRawGgufTensors (model_path, prefixes);
        if (raw_tensors_.empty ())
          {
            error = "no tensors matched GGUF kernel prefixes";
            return GgmlLinearLoadStatus::Unavailable;
          }

        internal::LoadGgmlBackendsOnce ();
        std::string backend_error;
        if (!InitBestBackend (threads, backend_error))
          {
            error = backend_error.empty () ? "no ggml backend device available"
                                           : backend_error;
            return GgmlLinearLoadStatus::Unavailable;
          }

        const std::size_t metadata_bytes =
            raw_tensors_.size () * ggml_tensor_overhead () + 1024 * 1024;
        ggml_init_params params{};
        params.mem_size = metadata_bytes;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        weights_ctx_ = ggml_init (params);
        if (weights_ctx_ == nullptr)
          {
            throw std::runtime_error ("ggml_init failed for AIST weights");
          }

        for (const auto &raw : raw_tensors_)
          {
            struct ggml_tensor *tensor = NewWeightTensor (raw);
            if (tensor == nullptr)
              {
                throw std::runtime_error ("ggml tensor allocation failed for "
                                          + raw.name);
              }
            weights_.emplace (raw.name, tensor);
            if (raw.name.size () >= 5
                && raw.name.compare (raw.name.size () - 5, 5, ".bias") == 0)
              {
                auto values = DequantizeTensorBytes (raw.type, raw.bytes,
                                                     Product (raw.shape));
                biases_.emplace (raw.name, values);
                host_vectors_.emplace (raw.name, std::move (values));
              }
            else if (raw.shape.size () <= 1)
              {
                host_vectors_.emplace (
                    raw.name, DequantizeTensorBytes (raw.type, raw.bytes,
                                                     Product (raw.shape)));
              }
          }
        CreateBatchNormConstants ();

        weights_buffer_ = ggml_backend_alloc_ctx_tensors (weights_ctx_,
                                                          backend_);
        if (weights_buffer_ == nullptr)
          {
            throw std::runtime_error (
                "ggml backend failed to allocate AIST weights");
          }
        ggml_backend_buffer_set_usage (weights_buffer_,
                                       GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto &raw : raw_tensors_)
          {
            struct ggml_tensor *tensor = weights_.at (raw.name);
            ggml_backend_tensor_set (tensor, raw.bytes.data (), 0,
                                     raw.bytes.size ());
          }
        for (const auto &constant : folded_constants_)
          {
            struct ggml_tensor *tensor = weights_.at (constant.first);
            ggml_backend_tensor_set (
                tensor, constant.second.data (), 0,
                sizeof (float) * constant.second.size ());
          }

        RowMatrix self_test = RowMatrix::Zero (1, 384);
        (void)LinearRows (self_test, "text_encoder.2.linear");
        raw_tensors_.clear ();
        raw_tensors_.shrink_to_fit ();
        return GgmlLinearLoadStatus::Loaded;
      }
    catch (const std::exception &ex)
      {
        error = ex.what ();
        image_plan_.reset ();
        audio_plan_.reset ();
        full_text_plans_.clear ();
        plans_.clear ();
        weights_.clear ();
        biases_.clear ();
        host_vectors_.clear ();
        folded_constants_.clear ();
        raw_tensors_.clear ();
        if (weights_buffer_ != nullptr)
          {
            ggml_backend_buffer_free (weights_buffer_);
            weights_buffer_ = nullptr;
          }
        if (weights_ctx_ != nullptr)
          {
            ggml_free (weights_ctx_);
            weights_ctx_ = nullptr;
          }
        if (backend_ != nullptr)
          {
            ggml_backend_free (backend_);
            backend_ = nullptr;
          }
        backend_name_.clear ();
        return GgmlLinearLoadStatus::Unavailable;
      }
  }

  bool
  IsReady () const
  {
    return backend_ != nullptr && weights_ctx_ != nullptr
           && weights_buffer_ != nullptr;
  }

  bool
  UsesFullTextGraph () const
  {
    return full_text_graph_used_;
  }

  bool
  UsesFullImageGraph () const
  {
    return full_image_graph_used_;
  }

  bool
  UsesFullAudioGraph () const
  {
    return full_audio_graph_used_;
  }

  const std::string &
  FullTextGraphError () const
  {
    return full_text_graph_error_;
  }

  const std::string &
  BackendName () const
  {
    return backend_name_;
  }

  std::vector<float>
  Linear (const std::vector<float> &input, const std::string &prefix)
  {
    RowMatrix rows (1, static_cast<Eigen::Index> (input.size ()));
    for (Eigen::Index col = 0; col < rows.cols (); ++col)
      {
        rows (0, col) = input[static_cast<std::size_t> (col)];
      }
    RowMatrix out = LinearRows (rows, prefix);
    return std::vector<float> (out.data (), out.data () + out.size ());
  }

  RowMatrix
  LinearRows (const RowMatrix &input, const std::string &prefix)
  {
    if (!IsReady ())
      {
        throw std::runtime_error ("AIST ggml linear kernel is not loaded");
      }
    struct ggml_tensor *weight = Required (prefix + ".weight");
    const auto bias_it = biases_.find (prefix + ".bias");
    if (bias_it == biases_.end ())
      {
        throw std::runtime_error ("AIST ggml missing bias: " + prefix
                                  + ".bias");
      }
    const Eigen::Index cols = static_cast<Eigen::Index> (weight->ne[0]);
    const Eigen::Index rows = static_cast<Eigen::Index> (weight->ne[1]);
    if (input.cols () != cols)
      {
        throw std::runtime_error ("AIST ggml linear input mismatch for "
                                  + prefix);
      }
    if (static_cast<Eigen::Index> (bias_it->second.size ()) != rows)
      {
        throw std::runtime_error ("AIST ggml linear bias mismatch for "
                                  + prefix);
      }

    LinearPlan &plan = GetOrCreatePlan (prefix, weight, input.rows ());
    ggml_backend_tensor_set (plan.input, input.data (), 0,
                             sizeof (float)
                                 * static_cast<std::size_t> (input.size ()));
    const enum ggml_status status = ggml_backend_graph_compute (
        backend_, plan.graph);
    if (status != GGML_STATUS_SUCCESS)
      {
        throw std::runtime_error ("AIST ggml graph compute failed for "
                                  + prefix + ": "
                                  + std::to_string (static_cast<int> (
                                      status)));
      }
    RowMatrix out (input.rows (), rows);
    ggml_backend_tensor_get (plan.output, out.data (), 0,
                             sizeof (float)
                                 * static_cast<std::size_t> (out.size ()));
    for (Eigen::Index row = 0; row < out.rows (); ++row)
      {
        for (Eigen::Index col = 0; col < out.cols (); ++col)
          {
            out (row, col) += bias_it->second[static_cast<std::size_t> (col)];
          }
      }
    return out;
  }

  std::vector<float>
  EncodeTextTokensFullGraph (const std::vector<int> &ids)
  {
    if (!IsReady ())
      {
        throw std::runtime_error ("AIST ggml full graph kernel is not loaded");
      }
    if (ids.empty ())
      {
        return {};
      }
    FullTextPlan &plan = GetOrCreateFullTextPlan (
        static_cast<Eigen::Index> (ids.size ()));
    std::vector<std::int32_t> token_ids;
    token_ids.reserve (ids.size ());
    for (int id : ids)
      {
        token_ids.push_back (static_cast<std::int32_t> (id));
      }
    ggml_backend_tensor_set (plan.token_ids, token_ids.data (), 0,
                             sizeof (std::int32_t) * token_ids.size ());
    const enum ggml_status status = ggml_backend_graph_compute (
        backend_, plan.graph);
    if (status != GGML_STATUS_SUCCESS)
      {
        throw std::runtime_error ("AIST ggml full text graph compute failed: "
                                  + std::to_string (static_cast<int> (
                                      status)));
      }
    std::vector<float> out (
        static_cast<std::size_t> (ggml_nelements (plan.output)), 0.0F);
    ggml_backend_tensor_get (plan.output, out.data (), 0,
                             sizeof (float) * out.size ());
    full_text_graph_used_ = true;
    return out;
  }

  std::vector<float>
  EncodeImageFullGraph (const std::vector<float> &nhwc)
  {
    if (!IsReady ())
      {
        throw std::runtime_error ("AIST ggml image graph kernel is not loaded");
      }
    if (nhwc.size () != static_cast<std::size_t> (384 * 384 * 3))
      {
        throw std::runtime_error ("AIST image input must be 384x384x3");
      }
    ModalityPlan &plan = GetOrCreateImagePlan ();
    ggml_backend_tensor_set (plan.input, nhwc.data (), 0,
                             sizeof (float) * nhwc.size ());
    const enum ggml_status status = ggml_backend_graph_compute (
        backend_, plan.graph);
    if (status != GGML_STATUS_SUCCESS)
      {
        throw std::runtime_error ("AIST ggml image graph compute failed: "
                                  + std::to_string (static_cast<int> (
                                      status)));
      }
    std::vector<float> out (
        static_cast<std::size_t> (ggml_nelements (plan.output)), 0.0F);
    ggml_backend_tensor_get (plan.output, out.data (), 0,
                             sizeof (float) * out.size ());
    full_image_graph_used_ = true;
    return out;
  }

  std::vector<float>
  EncodeAudioFullGraph (const std::vector<float> &nhwc)
  {
    if (!IsReady ())
      {
        throw std::runtime_error ("AIST ggml audio graph kernel is not loaded");
      }
    if (nhwc.size () != static_cast<std::size_t> (1000 * 128))
      {
        throw std::runtime_error ("AIST audio input must be 1000x128x1");
      }
    ModalityPlan &plan = GetOrCreateAudioPlan ();
    ggml_backend_tensor_set (plan.input, nhwc.data (), 0,
                             sizeof (float) * nhwc.size ());
    const enum ggml_status status = ggml_backend_graph_compute (
        backend_, plan.graph);
    if (status != GGML_STATUS_SUCCESS)
      {
        throw std::runtime_error ("AIST ggml audio graph compute failed: "
                                  + std::to_string (static_cast<int> (
                                      status)));
      }
    std::vector<float> out (
        static_cast<std::size_t> (ggml_nelements (plan.output)), 0.0F);
    ggml_backend_tensor_get (plan.output, out.data (), 0,
                             sizeof (float) * out.size ());
    full_audio_graph_used_ = true;
    return out;
  }

private:
  struct ggml_tensor *
  NewWeightTensor (const RawGgufTensor &raw)
  {
    const enum ggml_type type = ToGgmlType (raw.type);
    switch (raw.shape.size ())
      {
      case 0:
        return ggml_new_tensor_1d (weights_ctx_, type, 1);
      case 1:
        return ggml_new_tensor_1d (
            weights_ctx_, type, static_cast<int64_t> (raw.shape[0]));
      case 2:
        return ggml_new_tensor_2d (
            weights_ctx_, type, static_cast<int64_t> (raw.shape[0]),
            static_cast<int64_t> (raw.shape[1]));
      case 3:
        return ggml_new_tensor_3d (
            weights_ctx_, type, static_cast<int64_t> (raw.shape[0]),
            static_cast<int64_t> (raw.shape[1]),
            static_cast<int64_t> (raw.shape[2]));
      case 4:
        return ggml_new_tensor_4d (
            weights_ctx_, type, static_cast<int64_t> (raw.shape[0]),
            static_cast<int64_t> (raw.shape[1]),
            static_cast<int64_t> (raw.shape[2]),
            static_cast<int64_t> (raw.shape[3]));
      default:
        throw std::runtime_error ("AIST ggml unsupported tensor rank for "
                                  + raw.name);
      }
  }

  struct ggml_tensor *
  Required (const std::string &name) const
  {
    const auto found = weights_.find (name);
    if (found == weights_.end ())
      {
        throw std::runtime_error ("AIST ggml missing tensor: " + name);
      }
    return found->second;
  }

  struct LinearPlan
  {
    ~LinearPlan ()
    {
      if (buffer != nullptr)
        {
          ggml_backend_buffer_free (buffer);
        }
      if (ctx != nullptr)
        {
          ggml_free (ctx);
        }
    }

    struct ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    struct ggml_tensor *input = nullptr;
    struct ggml_tensor *output = nullptr;
    struct ggml_cgraph *graph = nullptr;
  };

  struct FullTextPlan
  {
    ~FullTextPlan ()
    {
      if (buffer != nullptr)
        {
          ggml_backend_buffer_free (buffer);
        }
      if (ctx != nullptr)
        {
          ggml_free (ctx);
        }
    }

    struct ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    struct ggml_tensor *token_ids = nullptr;
    struct ggml_tensor *position_ids = nullptr;
    struct ggml_tensor *token_type_ids = nullptr;
    struct ggml_tensor *mean_weights = nullptr;
    struct ggml_tensor *output = nullptr;
    struct ggml_cgraph *graph = nullptr;
  };

  struct ModalityPlan
  {
    ~ModalityPlan ()
    {
      if (buffer != nullptr)
        {
          ggml_backend_buffer_free (buffer);
        }
      if (ctx != nullptr)
        {
          ggml_free (ctx);
        }
    }

    struct ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    struct ggml_tensor *input = nullptr;
    struct ggml_tensor *output = nullptr;
    struct ggml_cgraph *graph = nullptr;
  };

  struct ggml_tensor *
  Weight (const std::string &name) const
  {
    return Required (name);
  }

  bool
  HasWeight (const std::string &name) const
  {
    return weights_.find (name) != weights_.end ();
  }

  void
  CreateBatchNormConstants ()
  {
    std::vector<std::string> prefixes;
    for (const auto &item : host_vectors_)
      {
        const std::string suffix = ".running_var";
        const std::string &name = item.first;
        if (name.size () > suffix.size ()
            && name.compare (name.size () - suffix.size (), suffix.size (),
                             suffix)
                   == 0)
          {
            prefixes.push_back (name.substr (0, name.size () - suffix.size ()));
          }
      }
    for (const auto &prefix : prefixes)
      {
        const auto weight_it = host_vectors_.find (prefix + ".weight");
        const auto bias_it = host_vectors_.find (prefix + ".bias");
        const auto mean_it = host_vectors_.find (prefix + ".running_mean");
        const auto var_it = host_vectors_.find (prefix + ".running_var");
        if (weight_it == host_vectors_.end () || bias_it == host_vectors_.end ()
            || mean_it == host_vectors_.end () || var_it == host_vectors_.end ())
          {
            continue;
          }
        const std::size_t count = var_it->second.size ();
        if (weight_it->second.size () != count || bias_it->second.size () != count
            || mean_it->second.size () != count)
          {
            continue;
          }
        std::vector<float> scale (count, 1.0F);
        std::vector<float> bias (count, 0.0F);
        for (std::size_t i = 0; i < count; ++i)
          {
            const float inv_std = 1.0F / std::sqrt (var_it->second[i] + 1.0e-5F);
            scale[i] = weight_it->second[i] * inv_std;
            bias[i] = bias_it->second[i] - mean_it->second[i] * scale[i];
          }
        const std::string scale_name = prefix + ".__fold_scale";
        const std::string bias_name = prefix + ".__fold_bias";
        weights_.emplace (
            scale_name,
            ggml_new_tensor_1d (weights_ctx_, GGML_TYPE_F32,
                                static_cast<int64_t> (count)));
        weights_.emplace (
            bias_name,
            ggml_new_tensor_1d (weights_ctx_, GGML_TYPE_F32,
                                static_cast<int64_t> (count)));
        folded_constants_.emplace_back (scale_name, std::move (scale));
        folded_constants_.emplace_back (bias_name, std::move (bias));
      }
  }

  struct ggml_tensor *
  LinearOp (struct ggml_context *ctx, struct ggml_tensor *input,
            const std::string &prefix)
  {
    struct ggml_tensor *weight = Weight (prefix + ".weight");
    struct ggml_tensor *bias = Weight (prefix + ".bias");
    struct ggml_tensor *out = ggml_mul_mat (ctx, weight, input);
    return ggml_add (ctx, out, ggml_repeat (ctx, bias, out));
  }

  struct ggml_tensor *
  LinearMaybeLoraOp (struct ggml_context *ctx, struct ggml_tensor *input,
                     const std::string &prefix)
  {
    if (!HasWeight (prefix + ".base.weight"))
      {
        return LinearOp (ctx, input, prefix);
      }
    struct ggml_tensor *base = LinearOp (ctx, input, prefix + ".base");
    if (HasWeight (prefix + ".lora_a.weight")
        && HasWeight (prefix + ".lora_b.weight"))
      {
        struct ggml_tensor *a = LinearOpNoBias (ctx, input,
                                                prefix + ".lora_a");
        struct ggml_tensor *b = LinearOpNoBias (ctx, a, prefix + ".lora_b");
        base = ggml_add (ctx, base, b);
      }
    return base;
  }

  struct ggml_tensor *
  LinearOpNoBias (struct ggml_context *ctx, struct ggml_tensor *input,
                  const std::string &prefix)
  {
    struct ggml_tensor *weight = Weight (prefix + ".weight");
    return ggml_mul_mat (ctx, weight, input);
  }

  struct ggml_tensor *
  BatchNormOp (struct ggml_context *ctx, struct ggml_tensor *input,
               const std::string &prefix)
  {
    struct ggml_tensor *scale = ggml_reshape_4d (
        ctx, Weight (prefix + ".__fold_scale"), 1, 1, input->ne[2], 1);
    scale = ggml_repeat_4d (ctx, scale, input->ne[0], input->ne[1],
                            input->ne[2], input->ne[3]);
    struct ggml_tensor *bias = ggml_reshape_4d (
        ctx, Weight (prefix + ".__fold_bias"), 1, 1, input->ne[2], 1);
    bias = ggml_repeat_4d (ctx, bias, input->ne[0], input->ne[1],
                           input->ne[2], input->ne[3]);
    return ggml_add (ctx, ggml_mul (ctx, input, scale), bias);
  }

  enum class Activation
  {
    None,
    Relu,
    HardSwish,
  };

  struct ggml_tensor *
  ApplyActivation (struct ggml_context *ctx, struct ggml_tensor *input,
                   Activation activation)
  {
    switch (activation)
      {
      case Activation::None:
        return input;
      case Activation::Relu:
        return ggml_relu (ctx, input);
      case Activation::HardSwish:
        return ggml_hardswish (ctx, input);
      }
    return input;
  }

  struct ggml_tensor *
  Conv2dOp (struct ggml_context *ctx, struct ggml_tensor *input,
            const std::string &prefix, int stride, int pad, bool depthwise)
  {
    struct ggml_tensor *weight = Weight (prefix + ".weight");
    if (depthwise)
      {
        return ggml_conv_2d_dw (ctx, weight, input, stride, stride, pad, pad,
                                1, 1);
      }
    return ggml_conv_2d (ctx, weight, input, stride, stride, pad, pad, 1, 1);
  }

  struct ggml_tensor *
  Conv2dMaybeLoraOp (struct ggml_context *ctx, struct ggml_tensor *input,
                     const std::string &prefix, int stride, int pad,
                     bool depthwise)
  {
    if (!HasWeight (prefix + ".base.weight"))
      {
        return Conv2dOp (ctx, input, prefix, stride, pad, depthwise);
      }
    struct ggml_tensor *base = Conv2dRawWeightOp (
        ctx, input, prefix + ".base.weight", stride, pad, depthwise);
    if (HasWeight (prefix + ".lora_a.weight")
        && HasWeight (prefix + ".lora_b.weight"))
      {
        struct ggml_tensor *a = Conv2dRawWeightOp (
            ctx, input, prefix + ".lora_a.weight", 1, 0, false);
        struct ggml_tensor *b = Conv2dRawWeightOp (
            ctx, a, prefix + ".lora_b.weight", stride, pad, false);
        base = ggml_add (ctx, base, b);
      }
    return base;
  }

  struct ggml_tensor *
  Conv2dRawWeightOp (struct ggml_context *ctx, struct ggml_tensor *input,
                     const std::string &weight_name, int stride, int pad,
                     bool depthwise)
  {
    struct ggml_tensor *weight = Weight (weight_name);
    if (depthwise)
      {
        return ggml_conv_2d_dw (ctx, weight, input, stride, stride, pad, pad,
                                1, 1);
      }
    return ggml_conv_2d (ctx, weight, input, stride, stride, pad, pad, 1, 1);
  }

  struct ggml_tensor *
  ConvBnAct (struct ggml_context *ctx, struct ggml_tensor *input,
             const std::string &conv_prefix, const std::string &bn_prefix,
             int stride, int pad, bool depthwise, Activation activation)
  {
    struct ggml_tensor *x = Conv2dMaybeLoraOp (ctx, input, conv_prefix, stride,
                                               pad, depthwise);
    x = BatchNormOp (ctx, x, bn_prefix);
    return ApplyActivation (ctx, x, activation);
  }

  struct ggml_tensor *
  LayerNormOp (struct ggml_context *ctx, struct ggml_tensor *input,
               const std::string &prefix, float eps)
  {
    struct ggml_tensor *weight = Weight (prefix + ".weight");
    struct ggml_tensor *bias = Weight (prefix + ".bias");
    struct ggml_tensor *out = ggml_norm (ctx, input, eps);
    out = ggml_mul (ctx, out, ggml_repeat (ctx, weight, out));
    return ggml_add (ctx, out, ggml_repeat (ctx, bias, out));
  }

  struct ggml_tensor *
  ProjectionBlockOp (struct ggml_context *ctx, struct ggml_tensor *input,
                     const std::string &prefix)
  {
    struct ggml_tensor *x = LinearOp (ctx, input, prefix + ".expand.0");
    x = ggml_gelu_erf (ctx, x);
    x = LayerNormOp (ctx, x, prefix + ".expand.2", 1e-5F);
    for (int block = 0;; ++block)
      {
        const std::string block_prefix =
            prefix + ".residual_blocks." + std::to_string (block);
        if (weights_.find (block_prefix + ".0.weight") == weights_.end ())
          {
            break;
          }
        struct ggml_tensor *y = LinearOp (ctx, x, block_prefix + ".0");
        y = ggml_gelu_erf (ctx, y);
        y = LayerNormOp (ctx, y, block_prefix + ".2", 1e-5F);
        x = ggml_add (ctx, x, y);
      }
    return LinearOp (ctx, x, prefix + ".project");
  }

  struct ggml_tensor *
  ImageEdgeBlock (struct ggml_context *ctx, struct ggml_tensor *input,
                  const std::string &prefix)
  {
    struct ggml_tensor *x = ConvBnAct (ctx, input, prefix + ".conv_exp",
                                       prefix + ".bn1", 2, 1, false,
                                       Activation::Relu);
    return ConvBnAct (ctx, x, prefix + ".conv_pwl", prefix + ".bn2", 1, 0,
                      false, Activation::None);
  }

  struct ggml_tensor *
  ImageUniversalBlock (struct ggml_context *ctx, struct ggml_tensor *input,
                       const std::string &prefix, int dw_mid_stride)
  {
    struct ggml_tensor *residual = input;
    struct ggml_tensor *x = input;
    if (HasWeight (prefix + ".dw_start.conv.weight"))
      {
        const int kernel = static_cast<int> (
            Weight (prefix + ".dw_start.conv.weight")->ne[0]);
        x = ConvBnAct (ctx, x, prefix + ".dw_start.conv",
                       prefix + ".dw_start.bn", 1, kernel / 2, true,
                       Activation::None);
      }
    x = ConvBnAct (ctx, x, prefix + ".pw_exp.conv", prefix + ".pw_exp.bn",
                   1, 0, false, Activation::Relu);
    if (HasWeight (prefix + ".dw_mid.conv.weight"))
      {
        const int kernel = static_cast<int> (
            Weight (prefix + ".dw_mid.conv.weight")->ne[0]);
        x = ConvBnAct (ctx, x, prefix + ".dw_mid.conv", prefix + ".dw_mid.bn",
                       dw_mid_stride, kernel / 2, true, Activation::Relu);
      }
    x = ConvBnAct (ctx, x, prefix + ".pw_proj.conv", prefix + ".pw_proj.bn",
                   1, 0, false, Activation::None);
    if (x->ne[0] == residual->ne[0] && x->ne[1] == residual->ne[1]
        && x->ne[2] == residual->ne[2] && x->ne[3] == residual->ne[3])
      {
        x = ggml_add (ctx, x, residual);
      }
    return x;
  }

  struct ggml_tensor *
  ImageEncoderGraph (struct ggml_context *ctx, struct ggml_tensor *input)
  {
    struct ggml_tensor *x = ConvBnAct (ctx, input, "image_encoder.conv_stem",
                                       "image_encoder.bn1", 2, 1, false,
                                       Activation::Relu);
    x = ImageEdgeBlock (ctx, x, "image_encoder.blocks.0.0");
    const std::vector<std::pair<std::string, int>> blocks = {
      { "image_encoder.blocks.1.0", 2 },
      { "image_encoder.blocks.1.1", 1 },
      { "image_encoder.blocks.2.0", 2 },
      { "image_encoder.blocks.2.1", 1 },
      { "image_encoder.blocks.2.2", 1 },
      { "image_encoder.blocks.2.3", 1 },
      { "image_encoder.blocks.2.4", 1 },
      { "image_encoder.blocks.2.5", 1 },
      { "image_encoder.blocks.2.6", 1 },
      { "image_encoder.blocks.2.7", 1 },
      { "image_encoder.blocks.3.0", 2 },
      { "image_encoder.blocks.3.1", 1 },
      { "image_encoder.blocks.3.2", 1 },
      { "image_encoder.blocks.3.3", 1 },
      { "image_encoder.blocks.3.4", 1 },
      { "image_encoder.blocks.3.5", 1 },
      { "image_encoder.blocks.3.6", 1 },
      { "image_encoder.blocks.3.7", 1 },
      { "image_encoder.blocks.3.8", 1 },
      { "image_encoder.blocks.3.9", 1 },
      { "image_encoder.blocks.3.10", 1 },
    };
    for (const auto &block : blocks)
      {
        x = ImageUniversalBlock (ctx, x, block.first, block.second);
      }
    x = ConvBnAct (ctx, x, "image_encoder.blocks.4.0.conv",
                   "image_encoder.blocks.4.0.bn1", 1, 0, false,
                   Activation::Relu);
    x = ggml_pool_2d (ctx, x, GGML_OP_POOL_AVG,
                      static_cast<int> (x->ne[0]),
                      static_cast<int> (x->ne[1]),
                      static_cast<int> (x->ne[0]),
                      static_cast<int> (x->ne[1]), 0.0F, 0.0F);
    x = Conv2dMaybeLoraOp (ctx, x, "image_encoder.conv_head", 1, 0, false);
    x = BatchNormOp (ctx, x, "image_encoder.norm_head");
    x = ggml_relu (ctx, x);
    x = ggml_cont_2d (ctx, x, 1280, 1);
    x = ProjectionBlockOp (ctx, x, "image_projection");
    return ggml_l2_norm (ctx, x, 1e-12F);
  }

  Activation
  AudioBlockActivation (int index) const
  {
    return index >= 7 ? Activation::HardSwish : Activation::Relu;
  }

  struct ggml_tensor *
  AudioSqueezeExcite (struct ggml_context *ctx, struct ggml_tensor *input,
                      const std::string &prefix)
  {
    struct ggml_tensor *pooled = ggml_pool_2d (
        ctx, input, GGML_OP_POOL_AVG, static_cast<int> (input->ne[0]),
        static_cast<int> (input->ne[1]), static_cast<int> (input->ne[0]),
        static_cast<int> (input->ne[1]), 0.0F, 0.0F);
    pooled = ggml_cont_2d (ctx, pooled, input->ne[2], 1);
    struct ggml_tensor *x = LinearMaybeLoraOp (ctx, pooled, prefix + ".fc1");
    x = ggml_relu (ctx, x);
    x = LinearMaybeLoraOp (ctx, x, prefix + ".fc2");
    x = ggml_sigmoid (ctx, x);
    x = ggml_reshape_4d (ctx, x, 1, 1, input->ne[2], 1);
    x = ggml_repeat_4d (ctx, x, input->ne[0], input->ne[1], input->ne[2],
                        input->ne[3]);
    return ggml_mul (ctx, input, x);
  }

  struct ggml_tensor *
  AudioInvertedBlock (struct ggml_context *ctx, struct ggml_tensor *input,
                      int index, int stride, bool use_se)
  {
    const std::string prefix = "audio_encoder.features." + std::to_string (index)
                               + ".block";
    const Activation activation = AudioBlockActivation (index);
    struct ggml_tensor *residual = input;
    struct ggml_tensor *x = input;
    int next = 0;
    if (HasWeight (prefix + ".0.0.weight")
        && Weight (prefix + ".0.0.weight")->ne[2] == 1)
      {
        const int kernel = static_cast<int> (Weight (prefix + ".0.0.weight")->ne[0]);
        x = ConvBnAct (ctx, x, prefix + ".0.0", prefix + ".0.1", stride,
                       kernel / 2, true, activation);
        next = 1;
      }
    else
      {
        x = ConvBnAct (ctx, x, prefix + ".0.0", prefix + ".0.1", 1, 0,
                       false, activation);
        next = 1;
        const int kernel = static_cast<int> (Weight (prefix + ".1.0.weight")->ne[0]);
        x = ConvBnAct (ctx, x, prefix + ".1.0", prefix + ".1.1", stride,
                       kernel / 2, true, activation);
        next = 2;
      }
    if (use_se)
      {
        x = AudioSqueezeExcite (ctx, x,
                                prefix + "." + std::to_string (next)
                                    + ".conc_se_layers.0");
        ++next;
      }
    x = ConvBnAct (ctx, x, prefix + "." + std::to_string (next) + ".0",
                   prefix + "." + std::to_string (next) + ".1", 1, 0, false,
                   Activation::None);
    if (x->ne[0] == residual->ne[0] && x->ne[1] == residual->ne[1]
        && x->ne[2] == residual->ne[2] && x->ne[3] == residual->ne[3])
      {
        x = ggml_add (ctx, x, residual);
      }
    return x;
  }

  struct ggml_tensor *
  AudioEncoderGraph (struct ggml_context *ctx, struct ggml_tensor *input)
  {
    struct ggml_tensor *x = ConvBnAct (ctx, input, "audio_encoder.features.0.0",
                                       "audio_encoder.features.0.1", 2, 1,
                                       false, Activation::HardSwish);
    const std::vector<int> strides = {
      0, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1,
    };
    const std::unordered_set<int> se_blocks = { 4, 5, 6, 11, 12, 13, 14, 15 };
    for (int i = 1; i <= 15; ++i)
      {
        x = AudioInvertedBlock (ctx, x, i, strides[static_cast<std::size_t> (i)],
                                se_blocks.count (i) != 0);
      }
    x = ConvBnAct (ctx, x, "audio_encoder.features.16.0",
                   "audio_encoder.features.16.1", 1, 0, false,
                   Activation::HardSwish);
    x = ggml_pool_2d (ctx, x, GGML_OP_POOL_AVG,
                      static_cast<int> (x->ne[0]),
                      static_cast<int> (x->ne[1]),
                      static_cast<int> (x->ne[0]),
                      static_cast<int> (x->ne[1]), 0.0F, 0.0F);
    x = ggml_cont_2d (ctx, x, 1920, 1);
    x = LinearOp (ctx, x, "audio_encoder.classifier.2");
    x = ggml_hardswish (ctx, x);
    x = LinearOp (ctx, x, "audio_encoder.classifier.5");
    x = ProjectionBlockOp (ctx, x, "audio_projection");
    return ggml_l2_norm (ctx, x, 1e-12F);
  }

  struct ggml_tensor *
  BertAttentionContextOp (struct ggml_context *ctx, struct ggml_tensor *q,
                          struct ggml_tensor *k, struct ggml_tensor *v,
                          Eigen::Index seq)
  {
    constexpr int kHeads = 12;
    constexpr int kHeadDim = 32;
    q = ggml_permute (ctx,
                      ggml_reshape_3d (ctx, q, kHeadDim, kHeads, seq),
                      0, 2, 1, 3);
    k = ggml_permute (ctx,
                      ggml_reshape_3d (ctx, k, kHeadDim, kHeads, seq),
                      0, 2, 1, 3);
    v = ggml_permute (ctx,
                      ggml_reshape_3d (ctx, v, kHeadDim, kHeads, seq),
                      0, 2, 1, 3);
    struct ggml_tensor *context = ggml_flash_attn_ext (
        ctx, q, k, v, nullptr,
        1.0F / std::sqrt (static_cast<float> (kHeadDim)), 0.0F, 0.0F);
    return ggml_cont_2d (ctx, context, kHeads * kHeadDim, seq);
  }

  FullTextPlan &
  GetOrCreateFullTextPlan (Eigen::Index seq)
  {
    const auto found = full_text_plans_.find (seq);
    if (found != full_text_plans_.end ())
      {
        return *found->second;
      }

    ggml_init_params params{};
    params.mem_size = 64 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    std::unique_ptr<FullTextPlan> plan = std::make_unique<FullTextPlan> ();
    plan->ctx = ggml_init (params);
    if (plan->ctx == nullptr)
      {
        throw std::runtime_error ("ggml_init failed for AIST full text graph");
      }

    struct ggml_context *ctx = plan->ctx;
    plan->token_ids = ggml_new_tensor_1d (ctx, GGML_TYPE_I32, seq);
    plan->position_ids = ggml_new_tensor_1d (ctx, GGML_TYPE_I32, seq);
    plan->token_type_ids = ggml_new_tensor_1d (ctx, GGML_TYPE_I32, seq);
    plan->mean_weights = ggml_new_tensor_2d (ctx, GGML_TYPE_F32, seq, 1);

    struct ggml_tensor *word = Weight (
        "text_encoder.0.model.embeddings.word_embeddings.weight");
    struct ggml_tensor *position = Weight (
        "text_encoder.0.model.embeddings.position_embeddings.weight");
    struct ggml_tensor *token_type = Weight (
        "text_encoder.0.model.embeddings.token_type_embeddings.weight");
    struct ggml_tensor *x = ggml_add (
        ctx,
        ggml_add (ctx, ggml_get_rows (ctx, word, plan->token_ids),
                  ggml_get_rows (ctx, position, plan->position_ids)),
        ggml_get_rows (ctx, token_type, plan->token_type_ids));
    x = LayerNormOp (ctx, x, "text_encoder.0.model.embeddings.LayerNorm",
                     1e-12F);

    for (int layer = 0; layer < 6; ++layer)
      {
        const std::string prefix = "text_encoder.0.model.encoder.layer."
                                   + std::to_string (layer);
        struct ggml_tensor *residual = x;
        struct ggml_tensor *q = LinearOp (
            ctx, x, prefix + ".attention.self.query");
        struct ggml_tensor *k = LinearOp (
            ctx, x, prefix + ".attention.self.key");
        struct ggml_tensor *v = LinearOp (
            ctx, x, prefix + ".attention.self.value");
        struct ggml_tensor *context = BertAttentionContextOp (
            ctx, q, k, v, seq);
        struct ggml_tensor *attn_out = LinearOp (
            ctx, context, prefix + ".attention.output.dense");
        x = LayerNormOp (ctx, ggml_add (ctx, residual, attn_out),
                         prefix + ".attention.output.LayerNorm", 1e-12F);

        residual = x;
        struct ggml_tensor *intermediate = LinearOp (
            ctx, x, prefix + ".intermediate.dense");
        intermediate = ggml_gelu_erf (ctx, intermediate);
        struct ggml_tensor *ffn = LinearOp (
            ctx, intermediate, prefix + ".output.dense");
        x = LayerNormOp (ctx, ggml_add (ctx, residual, ffn),
                         prefix + ".output.LayerNorm", 1e-12F);
      }

    struct ggml_tensor *x_t = ggml_cont (ctx, ggml_transpose (ctx, x));
    struct ggml_tensor *pooled = ggml_mul_mat (ctx, x_t, plan->mean_weights);
    struct ggml_tensor *features = LinearOp (ctx, pooled,
                                             "text_encoder.2.linear");
    struct ggml_tensor *semantic = ProjectionBlockOp (ctx, features,
                                                      "text_projection");
    plan->output = ggml_l2_norm (ctx, semantic, 1e-12F);

    plan->graph = ggml_new_graph_custom (ctx, 4096, false);
    ggml_build_forward_expand (plan->graph, plan->output);
    plan->buffer = ggml_backend_alloc_ctx_tensors (ctx, backend_);
    if (plan->buffer == nullptr)
      {
        throw std::runtime_error (
            "ggml backend failed to allocate AIST full text graph tensors");
      }

    std::vector<std::int32_t> positions (static_cast<std::size_t> (seq));
    std::vector<std::int32_t> token_types (static_cast<std::size_t> (seq),
                                           0);
    std::vector<float> mean (static_cast<std::size_t> (seq),
                             1.0F / static_cast<float> (seq));
    for (Eigen::Index i = 0; i < seq; ++i)
      {
        positions[static_cast<std::size_t> (i)] =
            static_cast<std::int32_t> (i);
      }
    ggml_backend_tensor_set (plan->position_ids, positions.data (), 0,
                             sizeof (std::int32_t) * positions.size ());
    ggml_backend_tensor_set (plan->token_type_ids, token_types.data (), 0,
                             sizeof (std::int32_t) * token_types.size ());
    ggml_backend_tensor_set (plan->mean_weights, mean.data (), 0,
                             sizeof (float) * mean.size ());

    auto inserted = full_text_plans_.emplace (seq, std::move (plan));
    return *inserted.first->second;
  }

  ModalityPlan &
  GetOrCreateImagePlan ()
  {
    if (image_plan_)
      {
        return *image_plan_;
      }
    ggml_init_params params{};
    params.mem_size = 256 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    image_plan_ = std::make_unique<ModalityPlan> ();
    image_plan_->ctx = ggml_init (params);
    if (image_plan_->ctx == nullptr)
      {
        throw std::runtime_error ("ggml_init failed for AIST image graph");
      }
    struct ggml_context *ctx = image_plan_->ctx;
    image_plan_->input = ggml_new_tensor_4d (ctx, GGML_TYPE_F32, 384, 384,
                                             3, 1);
    image_plan_->output = ImageEncoderGraph (ctx, image_plan_->input);
    image_plan_->graph = ggml_new_graph_custom (ctx, 8192, false);
    ggml_build_forward_expand (image_plan_->graph, image_plan_->output);
    image_plan_->buffer = ggml_backend_alloc_ctx_tensors (ctx, backend_);
    if (image_plan_->buffer == nullptr)
      {
        throw std::runtime_error (
            "ggml backend failed to allocate AIST image graph tensors");
      }
    return *image_plan_;
  }

  ModalityPlan &
  GetOrCreateAudioPlan ()
  {
    if (audio_plan_)
      {
        return *audio_plan_;
      }
    ggml_init_params params{};
    params.mem_size = 256 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    audio_plan_ = std::make_unique<ModalityPlan> ();
    audio_plan_->ctx = ggml_init (params);
    if (audio_plan_->ctx == nullptr)
      {
        throw std::runtime_error ("ggml_init failed for AIST audio graph");
      }
    struct ggml_context *ctx = audio_plan_->ctx;
    audio_plan_->input = ggml_new_tensor_4d (ctx, GGML_TYPE_F32, 1000, 128,
                                             1, 1);
    audio_plan_->output = AudioEncoderGraph (ctx, audio_plan_->input);
    audio_plan_->graph = ggml_new_graph_custom (ctx, 8192, false);
    ggml_build_forward_expand (audio_plan_->graph, audio_plan_->output);
    audio_plan_->buffer = ggml_backend_alloc_ctx_tensors (ctx, backend_);
    if (audio_plan_->buffer == nullptr)
      {
        throw std::runtime_error (
            "ggml backend failed to allocate AIST audio graph tensors");
      }
    return *audio_plan_;
  }

  LinearPlan &
  GetOrCreatePlan (const std::string &prefix, struct ggml_tensor *weight,
                   Eigen::Index input_rows)
  {
    const std::string key = prefix + "#" + std::to_string (input_rows);
    const auto found = plans_.find (key);
    if (found != plans_.end ())
      {
        return *found->second;
      }

    ggml_init_params params{};
    params.mem_size = 2 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    std::unique_ptr<LinearPlan> plan = std::make_unique<LinearPlan> ();
    plan->ctx = ggml_init (params);
    if (plan->ctx == nullptr)
      {
        throw std::runtime_error ("ggml_init failed for AIST graph");
      }

    plan->input = ggml_new_tensor_2d (
        plan->ctx, GGML_TYPE_F32, weight->ne[0], input_rows);
    plan->output = ggml_mul_mat (plan->ctx, weight, plan->input);
    plan->graph = ggml_new_graph_custom (plan->ctx, 16, false);
    ggml_build_forward_expand (plan->graph, plan->output);
    plan->buffer = ggml_backend_alloc_ctx_tensors (plan->ctx, backend_);
    if (plan->buffer == nullptr)
      {
        throw std::runtime_error (
            "ggml backend failed to allocate AIST graph tensors");
      }

    auto inserted = plans_.emplace (key, std::move (plan));
    return *inserted.first->second;
  }

  bool
  InitBestBackend (int threads, std::string &error)
  {
    const std::string requested = GetEnvOrDefault (
        "CORTEXT_AIST_GGML_BACKEND");
    std::vector<ggml_backend_dev_t> candidates;
    const auto append_matching = [&] (auto predicate) {
      const std::size_t count = ggml_backend_dev_count ();
      for (std::size_t i = 0; i < count; ++i)
        {
          ggml_backend_dev_t dev = ggml_backend_dev_get (i);
          if (dev != nullptr && predicate (dev)
              && std::find (candidates.begin (), candidates.end (), dev)
                     == candidates.end ())
            {
              candidates.push_back (dev);
            }
        }
    };

    if (!requested.empty ())
      {
        std::string lower_requested = requested;
        std::transform (lower_requested.begin (), lower_requested.end (),
                        lower_requested.begin (), [] (unsigned char ch) {
                          return static_cast<char> (std::tolower (ch));
                        });
        append_matching ([&] (ggml_backend_dev_t dev) {
          std::string name = ggml_backend_dev_name (dev);
          std::string desc = ggml_backend_dev_description (dev);
          std::transform (name.begin (), name.end (), name.begin (),
                          [] (unsigned char ch) {
                            return static_cast<char> (std::tolower (ch));
                          });
          std::transform (desc.begin (), desc.end (), desc.begin (),
                          [] (unsigned char ch) {
                            return static_cast<char> (std::tolower (ch));
                          });
          return name.find (lower_requested) != std::string::npos
                 || desc.find (lower_requested) != std::string::npos;
        });
      }

    append_matching ([] (ggml_backend_dev_t dev) {
      const enum ggml_backend_dev_type type = ggml_backend_dev_type (dev);
      return type == GGML_BACKEND_DEVICE_TYPE_GPU
             || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
    });
    append_matching ([] (ggml_backend_dev_t dev) {
      return ggml_backend_dev_type (dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL;
    });
    append_matching ([] (ggml_backend_dev_t dev) {
      return ggml_backend_dev_type (dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    });

    std::string last_error;
    for (ggml_backend_dev_t dev : candidates)
      {
        ggml_backend_t backend = ggml_backend_dev_init (dev, nullptr);
        if (backend == nullptr)
          {
            last_error = std::string ("failed to initialize ")
                         + ggml_backend_dev_name (dev);
            continue;
          }
        backend_ = backend;
        backend_name_ = std::string (ggml_backend_dev_name (dev)) + " ("
                        + ggml_backend_dev_description (dev) + ")";
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg (dev);
        if (reg != nullptr && threads > 0)
          {
            auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t> (
                ggml_backend_reg_get_proc_address (
                    reg, "ggml_backend_set_n_threads"));
            if (set_threads != nullptr)
              {
                set_threads (backend_, threads);
              }
          }
        return true;
      }

    error = last_error;
    return false;
  }

  std::vector<RawGgufTensor> raw_tensors_;
  std::unordered_map<std::string, struct ggml_tensor *> weights_;
  std::unordered_map<std::string, std::vector<float>> biases_;
  std::unordered_map<std::string, std::vector<float>> host_vectors_;
  std::vector<std::pair<std::string, std::vector<float>>> folded_constants_;
  std::unordered_map<std::string, std::unique_ptr<LinearPlan>> plans_;
  std::unordered_map<Eigen::Index, std::unique_ptr<FullTextPlan>>
      full_text_plans_;
  std::unique_ptr<ModalityPlan> image_plan_;
  std::unique_ptr<ModalityPlan> audio_plan_;
  struct ggml_context *weights_ctx_ = nullptr;
  ggml_backend_t backend_ = nullptr;
  ggml_backend_buffer_t weights_buffer_ = nullptr;
  std::string backend_name_;
  mutable bool full_text_graph_used_ = false;
  mutable bool full_image_graph_used_ = false;
  mutable bool full_audio_graph_used_ = false;
  std::string full_text_graph_error_;
};
#else
enum class GgmlLinearLoadStatus
{
  Loaded,
  Disabled,
  Unavailable,
};

class GgmlLinearKernel
{
public:
  GgmlLinearLoadStatus
  Load (const std::filesystem::path &, const std::vector<std::string> &, int,
        std::string &error)
  {
    error = "ggml kernel runtime was not compiled into this build";
    return GgmlLinearLoadStatus::Unavailable;
  }

  bool
  IsReady () const
  {
    return false;
  }

	  const std::string &
	  BackendName () const
	  {
	    static const std::string none = "not_used";
	    return none;
	  }

	  bool
	  UsesFullTextGraph () const
	  {
	    return false;
	  }

	  bool
	  UsesFullImageGraph () const
	  {
	    return false;
	  }

	  bool
	  UsesFullAudioGraph () const
	  {
	    return false;
	  }

	  const std::string &
	  FullTextGraphError () const
	  {
	    static const std::string error
	        = "ggml kernel runtime was not compiled into this build";
	    return error;
	  }

	  std::vector<float>
	  Linear (const std::vector<float> &, const std::string &)
	  {
	    throw std::runtime_error ("ggml kernel runtime unavailable");
	  }

	  std::vector<float>
	  EncodeTextTokensFullGraph (const std::vector<int> &)
	  {
	    throw std::runtime_error ("ggml kernel runtime unavailable");
	  }

	  std::vector<float>
	  EncodeImageFullGraph (const std::vector<float> &)
	  {
	    throw std::runtime_error ("ggml kernel runtime unavailable");
	  }

	  std::vector<float>
	  EncodeAudioFullGraph (const std::vector<float> &)
	  {
	    throw std::runtime_error ("ggml kernel runtime unavailable");
	  }

	  RowMatrix
	  LinearRows (const RowMatrix &, const std::string &)
  {
    throw std::runtime_error ("ggml kernel runtime unavailable");
  }
};
#endif

class BertWordpieceTokenizer
{
public:
  void
  Load (const std::filesystem::path &path)
  {
    if (path.filename () == "vocab.txt" || path.extension () == ".txt")
      {
        LoadFromVocabFile (path);
      }
    else
      {
        LoadFromGguf (path);
      }
  }

private:
  void
  LoadFromVocabFile (const std::filesystem::path &path)
  {
    std::ifstream in (path);
    if (!in)
      {
        throw std::runtime_error ("AIST tokenizer vocab not found: "
                                  + path.string ());
      }
    tokens_.clear ();
    vocabulary_.clear ();
    std::string line;
    int id = 0;
    while (std::getline (in, line))
      {
        if (!line.empty () && line.back () == '\r')
          {
            line.pop_back ();
          }
        vocabulary_[line] = id;
        tokens_.push_back (line);
        if (line == "[PAD]")
          {
            pad_id_ = id;
          }
        else if (line == "[UNK]")
          {
            unk_id_ = id;
          }
        else if (line == "[CLS]")
          {
            cls_id_ = id;
          }
        else if (line == "[SEP]")
          {
            sep_id_ = id;
          }
        ++id;
      }
    if (tokens_.empty ())
      {
        throw std::runtime_error ("AIST tokenizer vocab is empty: "
                                  + path.string ());
      }
  }

  void
  LoadFromGguf (const std::filesystem::path &path)
  {
    std::ifstream in (path, std::ios::binary);
    if (!in)
      {
        throw std::runtime_error ("AIST tokenizer GGUF not found: "
                                  + path.string ());
      }
    char magic[4]{};
    in.read (magic, sizeof (magic));
    if (!in || std::memcmp (magic, "GGUF", 4) != 0)
      {
        throw std::runtime_error ("AIST tokenizer path is not GGUF: "
                                  + path.string ());
      }
    const std::uint32_t version = ReadScalar<std::uint32_t> (in);
    if (version < 2 || version > 3)
      {
        throw std::runtime_error ("Unsupported tokenizer GGUF version: "
                                  + std::to_string (version));
      }
    (void)ReadScalar<std::uint64_t> (in);
    const std::uint64_t kv_count = ReadScalar<std::uint64_t> (in);
    for (std::uint64_t i = 0; i < kv_count; ++i)
      {
        const std::string key = ReadString (in);
        const auto type = static_cast<GgufValueType> (
            ReadScalar<std::uint32_t> (in));
        if (key == "tokenizer.ggml.tokens" && type == GgufValueType::Array)
          {
            const auto element_type = static_cast<GgufValueType> (
                ReadScalar<std::uint32_t> (in));
            const std::uint64_t count = ReadScalar<std::uint64_t> (in);
            if (element_type != GgufValueType::String)
              {
                throw std::runtime_error (
                    "AIST tokenizer token array is not string typed");
              }
            tokens_.clear ();
            tokens_.reserve (static_cast<std::size_t> (count));
            for (std::uint64_t idx = 0; idx < count; ++idx)
              {
                std::string token = ReadString (in);
                vocabulary_[token] = static_cast<int> (idx);
                tokens_.push_back (std::move (token));
              }
          }
        else if (key == "tokenizer.ggml.bos_token_id"
                 && type == GgufValueType::Uint32)
          {
            cls_id_ = static_cast<int> (ReadScalar<std::uint32_t> (in));
          }
        else if ((key == "tokenizer.ggml.eos_token_id"
                  || key == "tokenizer.ggml.seperator_token_id")
                 && type == GgufValueType::Uint32)
          {
            sep_id_ = static_cast<int> (ReadScalar<std::uint32_t> (in));
          }
        else if (key == "tokenizer.ggml.unknown_token_id"
                 && type == GgufValueType::Uint32)
          {
            unk_id_ = static_cast<int> (ReadScalar<std::uint32_t> (in));
          }
        else if (key == "tokenizer.ggml.padding_token_id"
                 && type == GgufValueType::Uint32)
          {
            pad_id_ = static_cast<int> (ReadScalar<std::uint32_t> (in));
          }
        else
          {
            SkipValue (in, type);
          }
      }

    if (tokens_.empty ())
      {
        throw std::runtime_error ("AIST tokenizer GGUF has no token list");
      }
  }

public:
  std::vector<int>
  Encode (const std::string &text, int max_length) const
  {
    std::vector<int> ids;
    ids.reserve (static_cast<std::size_t> (std::max (2, max_length)));
    ids.push_back (cls_id_);
    for (const std::string &token : BasicTokens (text))
      {
        for (int id : WordPieces (token))
          {
            if (static_cast<int> (ids.size ()) >= max_length - 1)
              {
                ids.push_back (sep_id_);
                return ids;
              }
            ids.push_back (id);
          }
      }
    ids.push_back (sep_id_);
    return ids;
  }

private:
  static std::vector<std::string>
  BasicTokens (const std::string &text)
  {
    std::vector<std::string> out;
    std::string current;
    const auto flush = [&] () {
      if (!current.empty ())
        {
          out.push_back (current);
          current.clear ();
        }
    };
    for (unsigned char ch : text)
      {
        if (std::isalnum (ch) != 0)
          {
            current.push_back (
                static_cast<char> (std::tolower (static_cast<int> (ch))));
          }
        else if (std::isspace (ch) != 0)
          {
            flush ();
          }
        else
          {
            flush ();
            out.emplace_back (1, static_cast<char> (ch));
          }
      }
    flush ();
    return out;
  }

  std::vector<int>
  WordPieces (const std::string &token) const
  {
    if (token.size () > 100)
      {
        return { unk_id_ };
      }
    std::vector<int> ids;
    std::size_t start = 0;
    while (start < token.size ())
      {
        std::size_t end = token.size ();
        int best = -1;
        while (start < end)
          {
            std::string piece = token.substr (start, end - start);
            if (start != 0)
              {
                piece = "##" + piece;
              }
            const auto it = vocabulary_.find (piece);
            if (it != vocabulary_.end ())
              {
                best = it->second;
                break;
              }
            --end;
          }
        if (best < 0)
          {
            return { unk_id_ };
          }
        ids.push_back (best);
        start = end;
      }
    return ids.empty () ? std::vector<int>{ unk_id_ } : ids;
  }

  std::vector<std::string> tokens_;
  std::unordered_map<std::string, int> vocabulary_;
  int pad_id_ = 0;
  int unk_id_ = 100;
  int cls_id_ = 101;
  int sep_id_ = 102;
};

std::filesystem::path
ResolveAistTokenizerPath (const std::filesystem::path &model_path,
                          const std::string &override_path)
{
  const std::string env_path = GetEnvOrDefault ("CORTEXT_AIST_TOKENIZER_GGUF_PATH");
  if (!override_path.empty ())
    {
      return override_path;
    }
  if (!env_path.empty ())
    {
      return env_path;
    }
  const auto model_dir = model_path.parent_path ();
  const std::vector<std::filesystem::path> candidates{
    model_dir / "vocab.txt",
    model_dir.parent_path () / "mdbr-leaf-ir" / "vocab.txt",
    std::filesystem::path ("models") / "mdbr-leaf-ir" / "vocab.txt",
    model_dir / "mdbr-leaf-ir-q8_0.gguf",
    model_dir.parent_path () / "llama_cpp" / "mdbr-leaf-ir-q8_0.gguf",
    std::filesystem::path ("models") / "llama_cpp" / "mdbr-leaf-ir-q8_0.gguf",
  };
  for (const auto &candidate : candidates)
    {
      if (std::filesystem::exists (candidate))
        {
          return candidate;
        }
    }
  return candidates.front ();
}

float
Gelu (float value)
{
  return 0.5F * value
         * (1.0F + std::erf (value / static_cast<float> (std::sqrt (2.0))));
}

void
ApplyGeluInPlace (std::vector<float> &values)
{
  for (float &value : values)
    {
      value = Gelu (value);
    }
}

void
L2Normalize (std::vector<float> &values)
{
  NormalizeL2InPlace (values);
}

} // namespace

class AistGgufEncoder::NativeRuntime
{
public:
  void
  Load (const AistGgufConfig &config)
  {
    context_length_ = std::max (8, config.context_length);
    std::vector<std::string> prefixes{
      "text_encoder.",  "text_projection.",  "image_encoder.",
      "image_projection.", "audio_encoder.", "audio_projection.",
    };
    tensors_.Load (config.model_path, prefixes);
    std::string kernel_error;
    kernel_ = std::make_unique<GgmlLinearKernel> ();
    const GgmlLinearLoadStatus kernel_status = kernel_->Load (
        config.model_path, prefixes, config.threads, kernel_error);
    if (kernel_status != GgmlLinearLoadStatus::Loaded)
      {
        kernel_.reset ();
        kernel_error_ = kernel_error;
      }
    tokenizer_path_ = ResolveAistTokenizerPath (config.model_path,
                                                config.tokenizer_gguf_path);
    tokenizer_.Load (tokenizer_path_);
  }

  const std::filesystem::path &
  tokenizer_path () const
  {
    return tokenizer_path_;
  }

  std::vector<float>
  Encode (const std::string &text)
  {
    return EncodeSemantic (text);
  }

  std::vector<float>
  EncodeImage (const std::uint8_t *data, int width, int height, int channels)
  {
    native_image_used_ = false;
    if (!GetEnvBool ("CORTEXT_AIST_DISABLE_NATIVE_IMAGE_KERNELS", false))
      {
        std::vector<float> input = PreprocessAistImageHWC (
            data, width, height, channels);
        std::vector<float> features = EncodeAistImageFeaturesNative (
            tensors_, input);
        std::vector<float> semantic = ProjectionBlockHostVector (
            tensors_, features, "image_projection");
        NormalizeL2InPlace (semantic);
        native_image_used_ = true;
        return semantic;
      }

    if (kernel_ == nullptr || !kernel_->IsReady ())
      {
        throw std::runtime_error (
            "triembed image runtime requires compiled ggml kernel ops");
      }
    std::vector<float> input = PreprocessAistImageNHWC (data, width, height,
                                                          channels);
    std::vector<float> semantic = kernel_->EncodeImageFullGraph (input);
    NormalizeL2InPlace (semantic);
    return semantic;
  }

  std::vector<float>
  EncodeAudio (const float *pcm, std::size_t num_samples)
  {
    if (kernel_ == nullptr || !kernel_->IsReady ())
      {
        throw std::runtime_error (
            "triembed audio runtime requires compiled ggml kernel ops");
      }
    std::vector<float> input = PreprocessAistAudioNHWC (pcm, num_samples);
    std::vector<float> semantic = kernel_->EncodeAudioFullGraph (input);
    NormalizeL2InPlace (semantic);
    return semantic;
  }

  std::vector<int>
  Tokenize (const std::string &text) const
  {
    return tokenizer_.Encode (text, context_length_);
  }

  AistRuntimeStats
  Stats () const
  {
    return stats_;
  }

  bool
  UsesKernelOps () const
  {
    return kernel_ != nullptr && kernel_->IsReady ();
  }

  bool
  UsesFullTextGraphOps () const
  {
    return kernel_ != nullptr && kernel_->IsReady ()
           && kernel_->UsesFullTextGraph ();
  }

  std::string
  KernelBackendName () const
  {
    if (kernel_ != nullptr && kernel_->IsReady ())
      {
        return kernel_->BackendName ();
      }
    return kernel_error_.empty () ? "not_used" : "not_used: " + kernel_error_;
  }

  std::string
  KernelOpsGranularity () const
  {
    if (kernel_ == nullptr || !kernel_->IsReady ())
      {
        return native_image_used_ ? "native_image_hwc" : "none";
      }
    if (native_image_used_)
      {
        return "native_image_hwc";
      }
    if (GetEnvBool ("CORTEXT_AIST_DISABLE_FULL_GGML_GRAPH", false))
      {
        return "per_linear_graph_debug";
      }
    if (kernel_->UsesFullTextGraph ())
      {
        return "full_text_graph";
      }
    if (kernel_->UsesFullImageGraph ())
      {
        return "full_image_graph";
      }
    if (kernel_->UsesFullAudioGraph ())
      {
        return "full_audio_graph";
      }
    if (!full_graph_error_.empty ())
      {
        return "per_linear_graph_after_full_text_graph_failure";
      }
    return "full_text_graph_pending";
  }

  std::string
  FullTextGraphError () const
  {
    if (!full_graph_error_.empty ())
      {
        return full_graph_error_;
      }
    if (kernel_ != nullptr && kernel_->IsReady ())
      {
        return kernel_->FullTextGraphError ();
      }
    return {};
  }

private:
  using ConstMatrixMap = Eigen::Map<const RowMatrix>;
  using ConstVectorMap = Eigen::Map<const Eigen::VectorXf>;

  const RuntimeTensor &
  T (const std::string &name) const
  {
    return tensors_.Required (name);
  }

  std::vector<float>
  Linear (const std::vector<float> &input, const std::string &prefix) const
  {
    if (kernel_ != nullptr && kernel_->IsReady ())
      {
        return kernel_->Linear (input, prefix);
      }
    const RuntimeTensor &weight = T (prefix + ".weight");
    const RuntimeTensor &bias = T (prefix + ".bias");
    const Eigen::Index rows = static_cast<Eigen::Index> (weight.Rows ());
    const Eigen::Index cols = static_cast<Eigen::Index> (weight.Cols ());
    if (static_cast<std::size_t> (cols) != input.size ())
      {
        throw std::runtime_error ("AIST linear input mismatch for " + prefix);
      }
    ConstMatrixMap w (weight.data.data (), rows, cols);
    ConstVectorMap x (input.data (), cols);
    ConstVectorMap b (bias.data.data (), rows);
    Eigen::VectorXf y = (w * x) + b;
    return std::vector<float> (y.data (), y.data () + y.size ());
  }

  RowMatrix
  LinearRows (const RowMatrix &input, const std::string &prefix) const
  {
    if (kernel_ != nullptr && kernel_->IsReady ())
      {
        return kernel_->LinearRows (input, prefix);
      }
    const RuntimeTensor &weight = T (prefix + ".weight");
    const RuntimeTensor &bias = T (prefix + ".bias");
    const Eigen::Index rows = static_cast<Eigen::Index> (weight.Rows ());
    const Eigen::Index cols = static_cast<Eigen::Index> (weight.Cols ());
    if (input.cols () != cols)
      {
        throw std::runtime_error ("AIST row-linear input mismatch for "
                                  + prefix);
      }
    ConstMatrixMap w (weight.data.data (), rows, cols);
    ConstVectorMap b (bias.data.data (), rows);
    RowMatrix out = input * w.transpose ();
    out.rowwise () += b.transpose ();
    return out;
  }

  std::vector<float>
  LayerNorm (const std::vector<float> &input, const std::string &prefix,
             float eps) const
  {
    const RuntimeTensor &weight = T (prefix + ".weight");
    const RuntimeTensor &bias = T (prefix + ".bias");
    if (weight.data.size () != input.size () || bias.data.size () != input.size ())
      {
        throw std::runtime_error ("AIST layernorm shape mismatch for "
                                  + prefix);
      }
    double mean = 0.0;
    for (float value : input)
      {
        mean += value;
      }
    mean /= static_cast<double> (input.size ());
    double variance = 0.0;
    for (float value : input)
      {
        const double centered = static_cast<double> (value) - mean;
        variance += centered * centered;
      }
    variance /= static_cast<double> (input.size ());
    const double inv = 1.0 / std::sqrt (variance + static_cast<double> (eps));
    std::vector<float> out (input.size ());
    for (std::size_t i = 0; i < input.size (); ++i)
      {
        out[i] = static_cast<float> (
            ((static_cast<double> (input[i]) - mean) * inv)
                * static_cast<double> (weight.data[i])
            + static_cast<double> (bias.data[i]));
      }
    return out;
  }

  void
  LayerNormRowsInPlace (RowMatrix &matrix, const std::string &prefix,
                        float eps) const
  {
    const RuntimeTensor &weight = T (prefix + ".weight");
    const RuntimeTensor &bias = T (prefix + ".bias");
    if (static_cast<std::size_t> (matrix.cols ()) != weight.data.size ()
        || weight.data.size () != bias.data.size ())
      {
        throw std::runtime_error ("AIST row layernorm shape mismatch for "
                                  + prefix);
      }
    for (Eigen::Index row = 0; row < matrix.rows (); ++row)
      {
        double mean = 0.0;
        for (Eigen::Index col = 0; col < matrix.cols (); ++col)
          {
            mean += matrix (row, col);
          }
        mean /= static_cast<double> (matrix.cols ());
        double variance = 0.0;
        for (Eigen::Index col = 0; col < matrix.cols (); ++col)
          {
            const double centered = static_cast<double> (matrix (row, col))
                                    - mean;
            variance += centered * centered;
          }
        variance /= static_cast<double> (matrix.cols ());
        const double inv = 1.0 / std::sqrt (variance + eps);
        for (Eigen::Index col = 0; col < matrix.cols (); ++col)
          {
            const auto idx = static_cast<std::size_t> (col);
            matrix (row, col) = static_cast<float> (
                ((static_cast<double> (matrix (row, col)) - mean) * inv)
                    * static_cast<double> (weight.data[idx])
                + static_cast<double> (bias.data[idx]));
          }
      }
  }

  RowMatrix
  EmbedTokens (const std::vector<int> &ids) const
  {
    const RuntimeTensor &word = T (
        "text_encoder.0.model.embeddings.word_embeddings.weight");
    const RuntimeTensor &position = T (
        "text_encoder.0.model.embeddings.position_embeddings.weight");
    const RuntimeTensor &token_type = T (
        "text_encoder.0.model.embeddings.token_type_embeddings.weight");
    const Eigen::Index seq = static_cast<Eigen::Index> (ids.size ());
    const Eigen::Index hidden = static_cast<Eigen::Index> (word.Cols ());
    RowMatrix out (seq, hidden);
    for (Eigen::Index row = 0; row < seq; ++row)
      {
        const int token_id = std::clamp (
            ids[static_cast<std::size_t> (row)], 0,
            static_cast<int> (word.Rows ()) - 1);
        for (Eigen::Index col = 0; col < hidden; ++col)
          {
            const auto w_idx = static_cast<std::size_t> (token_id) * word.Cols ()
                               + static_cast<std::size_t> (col);
            const auto p_idx = static_cast<std::size_t> (row) * position.Cols ()
                               + static_cast<std::size_t> (col);
            const auto tt_idx = static_cast<std::size_t> (col);
            out (row, col) = word.data[w_idx] + position.data[p_idx]
                             + token_type.data[tt_idx];
          }
      }
    LayerNormRowsInPlace (out,
                          "text_encoder.0.model.embeddings.LayerNorm",
                          1e-12F);
    return out;
  }

  RowMatrix
  AttentionContext (const RowMatrix &q, const RowMatrix &k,
                    const RowMatrix &v) const
  {
    constexpr int kHeads = 12;
    constexpr int kHeadDim = 32;
    const Eigen::Index seq = q.rows ();
    RowMatrix out = RowMatrix::Zero (seq, q.cols ());
    std::vector<float> scores (static_cast<std::size_t> (seq));
    const float scale = 1.0F / std::sqrt (static_cast<float> (kHeadDim));
    for (int head = 0; head < kHeads; ++head)
      {
        const Eigen::Index base = static_cast<Eigen::Index> (head * kHeadDim);
        for (Eigen::Index i = 0; i < seq; ++i)
          {
            float max_score = -std::numeric_limits<float>::infinity ();
            for (Eigen::Index j = 0; j < seq; ++j)
              {
                float dot = 0.0F;
                for (int d = 0; d < kHeadDim; ++d)
                  {
                    dot += q (i, base + d) * k (j, base + d);
                  }
                const float score = dot * scale;
                scores[static_cast<std::size_t> (j)] = score;
                max_score = std::max (max_score, score);
              }
            float denom = 0.0F;
            for (Eigen::Index j = 0; j < seq; ++j)
              {
                const float value = std::exp (
                    scores[static_cast<std::size_t> (j)] - max_score);
                scores[static_cast<std::size_t> (j)] = value;
                denom += value;
              }
            if (denom <= 0.0F)
              {
                continue;
              }
            for (int d = 0; d < kHeadDim; ++d)
              {
                float value = 0.0F;
                for (Eigen::Index j = 0; j < seq; ++j)
                  {
                    value += (scores[static_cast<std::size_t> (j)] / denom)
                             * v (j, base + d);
                  }
                out (i, base + d) = value;
              }
          }
      }
    return out;
  }

  std::vector<float>
  EncodeSemantic (const std::string &text)
  {
    ++stats_.semantic_requests;
    const auto cache_it = semantic_cache_.find (text);
    if (cache_it != semantic_cache_.end ())
      {
        ++stats_.semantic_cache_hits;
        return cache_it->second;
      }
    ++stats_.semantic_cache_misses;

    const std::vector<int> ids = tokenizer_.Encode (text, context_length_);
    if (kernel_ != nullptr && kernel_->IsReady ()
        && !GetEnvBool ("CORTEXT_AIST_DISABLE_FULL_GGML_GRAPH", false))
      {
        try
          {
            std::vector<float> semantic = kernel_->EncodeTextTokensFullGraph (
                ids);
            NormalizeL2InPlace (semantic);
            auto inserted = semantic_cache_.emplace (text,
                                                     std::move (semantic));
            return inserted.first->second;
          }
        catch (const std::exception &ex)
          {
            if (GetEnvBool ("CORTEXT_AIST_REQUIRE_FULL_GGML_GRAPH", false))
              {
                throw;
              }
            full_graph_error_ = ex.what ();
          }
      }
    RowMatrix x = EmbedTokens (ids);
    for (int layer = 0; layer < 6; ++layer)
      {
        const std::string prefix = "text_encoder.0.model.encoder.layer."
                                   + std::to_string (layer);
        RowMatrix residual = x;
        RowMatrix q = LinearRows (x, prefix + ".attention.self.query");
        RowMatrix k = LinearRows (x, prefix + ".attention.self.key");
        RowMatrix v = LinearRows (x, prefix + ".attention.self.value");
        RowMatrix context = AttentionContext (q, k, v);
        RowMatrix attn_out = LinearRows (
            context, prefix + ".attention.output.dense");
        x = residual + attn_out;
        LayerNormRowsInPlace (x, prefix + ".attention.output.LayerNorm",
                              1e-12F);

        residual = x;
        RowMatrix intermediate = LinearRows (x, prefix + ".intermediate.dense");
        for (Eigen::Index row = 0; row < intermediate.rows (); ++row)
          {
            for (Eigen::Index col = 0; col < intermediate.cols (); ++col)
              {
                intermediate (row, col) = Gelu (intermediate (row, col));
              }
          }
        RowMatrix ffn = LinearRows (intermediate, prefix + ".output.dense");
        x = residual + ffn;
        LayerNormRowsInPlace (x, prefix + ".output.LayerNorm", 1e-12F);
      }

    Eigen::VectorXf pooled = x.colwise ().mean ();
    std::vector<float> pooled_raw (static_cast<std::size_t> (pooled.size ()));
    for (Eigen::Index i = 0; i < pooled.size (); ++i)
      {
        pooled_raw[static_cast<std::size_t> (i)] = pooled (i);
      }
    std::vector<float> text_features = Linear (pooled_raw,
                                               "text_encoder.2.linear");
    std::vector<float> semantic = ProjectionBlock (text_features,
                                                   "text_projection");
    L2Normalize (semantic);
    auto inserted = semantic_cache_.emplace (text, std::move (semantic));
    return inserted.first->second;
  }

  std::vector<float>
  ProjectionBlock (const std::vector<float> &input,
                   const std::string &prefix) const
  {
    std::vector<float> x = Linear (input, prefix + ".expand.0");
    ApplyGeluInPlace (x);
    x = LayerNorm (x, prefix + ".expand.2", 1e-5F);
    for (int block = 0;; ++block)
      {
        const std::string block_prefix =
            prefix + ".residual_blocks." + std::to_string (block);
        if (!tensors_.Contains (block_prefix + ".0.weight"))
          {
            break;
          }
        std::vector<float> y = Linear (
            x, block_prefix + ".0");
        ApplyGeluInPlace (y);
        y = LayerNorm (y, block_prefix + ".2", 1e-5F);
        for (std::size_t i = 0; i < x.size (); ++i)
          {
            x[i] += y[i];
          }
      }
    return Linear (x, prefix + ".project");
  }

  std::vector<float>
  EmbeddingLookup (const std::string &name, int id) const
  {
    const RuntimeTensor &table = T (name);
    const int safe_id = std::clamp (id, 0, static_cast<int> (table.Rows ()) - 1);
    const std::size_t cols = table.Cols ();
    const std::size_t begin = static_cast<std::size_t> (safe_id) * cols;
    return std::vector<float> (table.data.begin ()
                                   + static_cast<std::ptrdiff_t> (begin),
                               table.data.begin ()
                                   + static_cast<std::ptrdiff_t> (begin + cols));
  }

  RuntimeTensorStore tensors_;
  BertWordpieceTokenizer tokenizer_;
  std::filesystem::path tokenizer_path_;
  int context_length_ = 512;
  std::unique_ptr<GgmlLinearKernel> kernel_;
  std::string kernel_error_;
  std::string full_graph_error_;
  bool native_image_used_ = false;
  std::unordered_map<std::string, std::vector<float>> semantic_cache_;
  AistRuntimeStats stats_;
};

std::optional<std::filesystem::path>
ResolveAistGgufModelPath (const std::filesystem::path &models_dir,
                          const std::filesystem::path &override_path)
{
  const std::string env_path = GetEnvOrDefault ("CORTEXT_AIST_MODEL_PATH");
  if (!override_path.empty ())
    {
      if (std::filesystem::exists (override_path))
        {
          return override_path;
        }
      throw std::runtime_error ("AIST model override does not exist: "
                                + override_path.string ());
    }
  if (!env_path.empty ())
    {
      std::filesystem::path path (env_path);
      if (std::filesystem::exists (path))
        {
          return path;
        }
      throw std::runtime_error ("CORTEXT_AIST_MODEL_PATH does not exist: "
                                + env_path);
    }

  std::vector<std::filesystem::path> search_bases{ models_dir };
  if (models_dir.has_parent_path ())
    {
      search_bases.push_back (models_dir.parent_path ());
    }
  std::vector<std::filesystem::path> roots;
  for (const auto &base : search_bases)
    {
      roots.push_back (base / "AIST-87M-GGUF");
      roots.push_back (base / "aist-87m-gguf");
      roots.push_back (base);
    }
  for (const auto &root : roots)
    {
      if (!std::filesystem::exists (root))
        {
          continue;
        }
      if (std::filesystem::is_regular_file (root)
          && root.extension () == ".gguf")
        {
          return root;
        }
      if (std::filesystem::is_directory (root))
        {
          const std::vector<std::filesystem::path> preferred_files = {
            root / "AIST-87M_q8_0.gguf",
            root / "AIST-87M_q5_1.gguf",
          };
          for (const auto &candidate : preferred_files)
            {
              if (std::filesystem::exists (candidate))
                {
                  return candidate;
                }
            }
          std::vector<std::filesystem::path> matches;
          for (const auto &entry : std::filesystem::directory_iterator (root))
            {
              const bool is_file_or_link =
                  entry.is_regular_file ()
                  || std::filesystem::is_symlink (entry.symlink_status ());
              const std::string filename = entry.path ().filename ().string ();
              if (is_file_or_link && entry.path ().extension () == ".gguf"
                  && filename.find ("AIST-87M") != std::string::npos)
                {
                  matches.push_back (entry.path ());
                }
            }
          std::sort (matches.begin (), matches.end (),
                     [] (const auto &lhs, const auto &rhs) {
                       const std::string left = lhs.filename ().string ();
                       const std::string right = rhs.filename ().string ();
                       const bool left_q8 = left.find ("q8_0")
                                            != std::string::npos;
                       const bool right_q8 = right.find ("q8_0")
                                             != std::string::npos;
                       if (left_q8 != right_q8)
                         {
                           return left_q8;
                         }
                       return left < right;
                     });
          if (!matches.empty ())
            {
              return matches.front ();
            }
        }
    }
  return std::nullopt;
}

AistModelInfo
InspectAistGgufModel (const std::filesystem::path &model_path)
{
  std::ifstream in (model_path, std::ios::binary);
  if (!in)
    {
      throw std::runtime_error ("AIST GGUF model not found: "
                                + model_path.string ());
    }

  char magic[4]{};
  in.read (magic, sizeof (magic));
  if (!in || std::memcmp (magic, "GGUF", 4) != 0)
    {
      throw std::runtime_error ("AIST model is not a GGUF file: "
                                + model_path.string ());
    }
  const std::uint32_t version = ReadScalar<std::uint32_t> (in);
  if (version < 2 || version > 3)
    {
      throw std::runtime_error ("Unsupported AIST GGUF version: "
                                + std::to_string (version));
    }
  const std::uint64_t tensor_count = ReadScalar<std::uint64_t> (in);
  const std::uint64_t kv_count = ReadScalar<std::uint64_t> (in);

  AistModelInfo info;
  info.model_path = model_path.string ();
  std::unordered_set<std::string> key_set;
  for (std::uint64_t i = 0; i < kv_count; ++i)
    {
      const std::string key = ReadString (in);
      const auto type = static_cast<GgufValueType> (
          ReadScalar<std::uint32_t> (in));
      const std::string value = ReadMetadataStringValue (in, type);
      info.metadata_keys.push_back (key);
      key_set.insert (key);
      if (key == "general.architecture")
        {
          info.architecture = value;
        }
      else if (key == "triembed.embed_dim"
               || key == "triembed.shared_embedding_dim")
        {
          try
            {
              info.n_embd = std::stoi (value);
              info.n_embd_out = info.n_embd;
            }
          catch (...)
            {
            }
        }
      else if (key == "triembed.text_model")
        {
          AppendIfPresent (info.supported_modalities, "text");
        }
      else if (key == "triembed.image_encoder_name")
        {
          AppendIfPresent (info.supported_modalities, "image");
        }
      else if (key == "triembed.audio_encoder_name")
        {
          AppendIfPresent (info.supported_modalities, "audio");
        }
    }

  info.tensors.reserve (static_cast<std::size_t> (tensor_count));
  info.tensor_names.reserve (static_cast<std::size_t> (tensor_count));
  for (std::uint64_t i = 0; i < tensor_count; ++i)
    {
      AistTensorInfo tensor;
      tensor.name = ReadString (in);
      const std::uint32_t n_dims = ReadScalar<std::uint32_t> (in);
      tensor.shape.reserve (n_dims);
      for (std::uint32_t dim = 0; dim < n_dims; ++dim)
        {
          tensor.shape.push_back (ReadScalar<std::uint64_t> (in));
        }
      tensor.ggml_type = ReadScalar<std::uint32_t> (in);
      tensor.offset = ReadScalar<std::uint64_t> (in);
      info.tensor_names.push_back (tensor.name);
      info.tensors.push_back (std::move (tensor));
    }

  info.has_semantic_vector
      = ContainsTensor (info.tensor_names, "text_projection.project.weight")
        || key_set.count ("triembed.embed_dim") > 0;
  info.pooling_type = "model-defined";
  info.runtime_available = info.architecture == "triembed"
                           && info.has_semantic_vector;
  info.runtime_status = info.runtime_available
                            ? "triembed_embedding_runtime_available"
                            : "triembed_embedding_outputs_unavailable";
  info.runtime_error = info.runtime_available ? "" : RuntimeUnavailableReason ();
  return info;
}

void
NormalizeL2InPlace (std::vector<float> &values)
{
  double sum_sq = 0.0;
  for (float value : values)
    {
      sum_sq += static_cast<double> (value) * static_cast<double> (value);
    }
  if (sum_sq <= 0.0)
    {
      return;
    }
  const double inv_norm = 1.0 / std::sqrt (sum_sq);
  for (float &value : values)
    {
      value = static_cast<float> (static_cast<double> (value) * inv_norm);
    }
}

std::vector<float>
TruncateAistMatryoshka (const std::vector<float> &values, std::size_t dim)
{
  std::vector<float> out = values;
  if (out.size () > dim)
    {
      out.resize (dim);
    }
  NormalizeL2InPlace (out);
  return out;
}

std::vector<float>
SliceAistEmbedding (const std::vector<float> &values, std::size_t begin,
                    std::size_t end)
{
  if (begin >= end || begin >= values.size ())
    {
      return {};
    }
  end = std::min (end, values.size ());
  std::vector<float> out (values.begin ()
                              + static_cast<std::ptrdiff_t> (begin),
                          values.begin ()
                              + static_cast<std::ptrdiff_t> (end));
  NormalizeL2InPlace (out);
  return out;
}

AistGgufEncoder::AistGgufEncoder () = default;

AistGgufEncoder::AistGgufEncoder (AistGgufConfig config)
{
  (void)Load (config);
}

AistGgufEncoder::~AistGgufEncoder () = default;

bool
AistGgufEncoder::Load (const AistGgufConfig &config)
{
  config_ = config;
  if (config_.threads <= 0)
    {
      config_.threads = GetEnvInt ("CORTEXT_AIST_THREADS", 0);
    }
  config_.n_gpu_layers = GetEnvInt ("CORTEXT_AIST_N_GPU_LAYERS",
                                    config_.n_gpu_layers);
  config_.context_length = GetEnvInt ("CORTEXT_AIST_CONTEXT_LENGTH",
                                      config_.context_length);
  config_.shadow_only = GetEnvBool ("CORTEXT_AIST_SHADOW_ONLY",
                                    config_.shadow_only);
  config_.use_semantic_for_retrieval = GetEnvBool (
      "CORTEXT_AIST_USE_SEMANTIC_FOR_RETRIEVAL",
      config_.use_semantic_for_retrieval);
  const std::string tokenizer_override = GetEnvOrDefault (
      "CORTEXT_AIST_TOKENIZER_GGUF_PATH");
  if (!tokenizer_override.empty ())
    {
      config_.tokenizer_gguf_path = tokenizer_override;
    }
  const std::string runtime_override = GetEnvOrDefault (
      "CORTEXT_AIST_RUNTIME");
  if (!runtime_override.empty ())
    {
      config_.runtime = runtime_override;
    }
  info_ = InspectAistGgufModel (config_.model_path);
  if (!info_.runtime_available)
    {
      loaded_ = true;
      return loaded_;
    }
  if (!(config_.runtime == "gguf" || config_.runtime == "native"
        || config_.runtime == "native_gguf"))
    {
      info_.runtime_available = false;
      info_.runtime_status = "aist_runtime_unavailable";
      info_.runtime_error = "AIST runtime '" + config_.runtime
                            + "' is not built in this Cortext binary; use "
                              "CORTEXT_AIST_RUNTIME=gguf";
      loaded_ = true;
      return loaded_;
    }
  try
    {
      runtime_ = std::make_unique<NativeRuntime> ();
      runtime_->Load (config_);
      info_.runtime_available = true;
      const std::string base_status = "triembed_embedding_runtime_loaded";
      info_.runtime_status = runtime_->UsesKernelOps ()
                                  ? base_status + "_with_ggml_kernel_ops"
                                  : base_status + "_with_eigen_fallback";
      info_.runtime_error.clear ();
    }
  catch (const std::exception &ex)
    {
      runtime_.reset ();
      info_.runtime_available = false;
      info_.runtime_status = "aist_native_triembed_runtime_load_failed";
      info_.runtime_error = ex.what ();
    }
  loaded_ = true;
  return loaded_;
}

const AistModelInfo &
AistGgufEncoder::Inspect () const
{
  return info_;
}

bool
AistGgufEncoder::IsLoaded () const
{
  return loaded_;
}

bool
AistGgufEncoder::IsRuntimeAvailable () const
{
  return loaded_ && info_.runtime_available;
}

bool
AistGgufEncoder::UsesKernelOps () const
{
  return runtime_ != nullptr && runtime_->UsesKernelOps ();
}

bool
AistGgufEncoder::UsesFullTextGraphOps () const
{
  return runtime_ != nullptr && runtime_->UsesFullTextGraphOps ();
}

std::string
AistGgufEncoder::KernelOpsBackend () const
{
  if (runtime_ == nullptr)
    {
      return "not_used";
    }
  return runtime_->KernelBackendName ();
}

std::string
AistGgufEncoder::KernelOpsGranularity () const
{
  if (runtime_ == nullptr)
    {
      return "none";
    }
  return runtime_->KernelOpsGranularity ();
}

std::string
AistGgufEncoder::FullTextGraphError () const
{
  if (runtime_ == nullptr)
    {
      return {};
    }
  return runtime_->FullTextGraphError ();
}

std::vector<int>
AistGgufEncoder::DebugTokenizeText (const std::string &text) const
{
  if (!runtime_)
    {
      return {};
    }
  return runtime_->Tokenize (text);
}

AistRuntimeStats
AistGgufEncoder::DebugRuntimeStats () const
{
  if (!runtime_)
    {
      return {};
    }
  return runtime_->Stats ();
}

void
AistGgufEncoder::EncodeText (const std::string &text,
                             std::vector<float> &out_embedding)
{
  if (!loaded_)
    {
      throw std::runtime_error ("AIST GGUF encoder is not loaded");
    }
  if (!runtime_)
    {
      throw std::runtime_error (info_.runtime_error.empty ()
                                    ? RuntimeUnavailableReason ()
                                    : info_.runtime_error);
    }
  out_embedding = runtime_->Encode (text);
  if (out_embedding.empty ())
    {
      throw std::runtime_error ("AIST semantic_vector output unavailable");
    }
}

void
AistGgufEncoder::EncodeAudio (const float *pcm, std::size_t num_samples,
                              std::vector<float> &out_embedding)
{
  if (!loaded_)
    {
      throw std::runtime_error ("AIST GGUF encoder is not loaded");
    }
  if (!runtime_ || !info_.runtime_available)
    {
      throw std::runtime_error (info_.runtime_error.empty ()
                                    ? RuntimeUnavailableReason ()
                                    : info_.runtime_error);
    }
  out_embedding = runtime_->EncodeAudio (pcm, num_samples);
}

void
AistGgufEncoder::EncodeImage (const std::uint8_t *data, int width,
                              int height, int channels,
                              std::vector<float> &out_embedding)
{
  if (!loaded_)
    {
      throw std::runtime_error ("AIST GGUF encoder is not loaded");
    }
  if (!runtime_ || !info_.runtime_available)
    {
      throw std::runtime_error (info_.runtime_error.empty ()
                                    ? RuntimeUnavailableReason ()
                                    : info_.runtime_error);
    }
  out_embedding = runtime_->EncodeImage (data, width, height, channels);
}

} // namespace cortext
