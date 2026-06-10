#include "cortext/models/embedding_model_pin.hpp"

#include <objstore/blake3.h>

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace cortext::models
{

namespace
{

std::string
HashFileBlake3Hex16 (const std::filesystem::path &path)
{
  std::FILE *f = std::fopen (path.string ().c_str (), "rb");
  if (f == nullptr)
    {
      throw std::runtime_error (
          "embedding-model pin: cannot open model file: "
          + path.string ());
    }
  objstore_blake3 ctx;
  objstore_blake3_init (&ctx);
  std::vector<unsigned char> buffer (1 << 20);
  size_t n = 0;
  while ((n = std::fread (buffer.data (), 1, buffer.size (), f)) > 0)
    {
      objstore_blake3_update (&ctx, buffer.data (), n);
    }
  const bool read_error = std::ferror (f) != 0;
  std::fclose (f);
  if (read_error)
    {
      throw std::runtime_error (
          "embedding-model pin: read failed for model file: "
          + path.string ());
    }
  objstore_id id{};
  objstore_blake3_final (&ctx, &id);
  static const char *kHex = "0123456789abcdef";
  std::string hex;
  hex.reserve (16);
  for (int i = 0; i < 8; ++i)
    {
      const unsigned char byte = id.bytes[i];
      hex.push_back (kHex[byte >> 4]);
      hex.push_back (kHex[byte & 0x0F]);
    }
  return hex;
}

} // namespace

std::string
ComputeEmbeddingModelPin (const std::string &backend_name,
                                  const std::filesystem::path &model_path,
                                  int dim)
{
  const std::string hash = model_path.empty ()
                               ? "none"
                               : HashFileBlake3Hex16 (model_path);
  return backend_name + ":b3-" + hash + ":dim" + std::to_string (dim);
}

} // namespace cortext::models
