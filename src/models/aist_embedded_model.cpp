#include "aist_embedded_model.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shlobj.h>
#endif

#if defined(CORTEXT_HAS_EMBEDDED_AIST_MODEL) && CORTEXT_HAS_EMBEDDED_AIST_MODEL
#include "aist_embedded_meta.h"
#include "aist_embedded_table.inc"
#endif

namespace cortext
{
namespace
{

// Helpers below only run when Git AIST shards are linked into the library.
// Keep them out of embed-off builds so -Werror=unused-function stays clean.
#if defined(CORTEXT_HAS_EMBEDDED_AIST_MODEL) && CORTEXT_HAS_EMBEDDED_AIST_MODEL

class Sha256
{
public:
  Sha256 () { Reset (); }

  void
  Reset ()
  {
    state_[0] = 0x6a09e667u;
    state_[1] = 0xbb67ae85u;
    state_[2] = 0x3c6ef372u;
    state_[3] = 0xa54ff53au;
    state_[4] = 0x510e527fu;
    state_[5] = 0x9b05688cu;
    state_[6] = 0x1f83d9abu;
    state_[7] = 0x5be0cd19u;
    bitlen_ = 0;
    datalen_ = 0;
  }

  void
  Update (const unsigned char *bytes, std::size_t len)
  {
    for (std::size_t i = 0; i < len; ++i)
      {
        data_[datalen_++] = bytes[i];
        if (datalen_ == 64)
          {
            Transform (state_, data_);
            bitlen_ += 512;
            datalen_ = 0;
          }
      }
  }

  std::string
  FinalHex ()
  {
    unsigned char hash[32];
    Final (hash);
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out (64, '0');
    for (int i = 0; i < 32; ++i)
      {
        out[static_cast<std::size_t> (i) * 2] = kHex[(hash[i] >> 4) & 0xf];
        out[static_cast<std::size_t> (i) * 2 + 1] = kHex[hash[i] & 0xf];
      }
    return out;
  }

private:
  static std::uint32_t
  Rotr (std::uint32_t x, std::uint32_t n)
  {
    return (x >> n) | (x << (32 - n));
  }

  static void
  Transform (std::uint32_t state[8], const std::uint8_t data[64])
  {
    static constexpr std::uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };
    std::uint32_t m[64];
    for (int i = 0; i < 16; ++i)
      {
        m[i] = (static_cast<std::uint32_t> (data[i * 4]) << 24)
               | (static_cast<std::uint32_t> (data[i * 4 + 1]) << 16)
               | (static_cast<std::uint32_t> (data[i * 4 + 2]) << 8)
               | (static_cast<std::uint32_t> (data[i * 4 + 3]));
      }
    for (int i = 16; i < 64; ++i)
      {
        const std::uint32_t s0
            = Rotr (m[i - 15], 7) ^ Rotr (m[i - 15], 18) ^ (m[i - 15] >> 3);
        const std::uint32_t s1
            = Rotr (m[i - 2], 17) ^ Rotr (m[i - 2], 19) ^ (m[i - 2] >> 10);
        m[i] = m[i - 16] + s0 + m[i - 7] + s1;
      }
    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (int i = 0; i < 64; ++i)
      {
        const std::uint32_t S1 = Rotr (e, 6) ^ Rotr (e, 11) ^ Rotr (e, 25);
        const std::uint32_t ch = (e & f) ^ ((~e) & g);
        const std::uint32_t temp1 = h + S1 + ch + k[i] + m[i];
        const std::uint32_t S0 = Rotr (a, 2) ^ Rotr (a, 13) ^ Rotr (a, 22);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = S0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
      }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  void
  Final (unsigned char hash[32])
  {
    std::size_t i = datalen_;
    if (datalen_ < 56)
      {
        data_[i++] = 0x80;
        while (i < 56)
          {
            data_[i++] = 0x00;
          }
      }
    else
      {
        data_[i++] = 0x80;
        while (i < 64)
          {
            data_[i++] = 0x00;
          }
        Transform (state_, data_);
        std::memset (data_, 0, 56);
      }
    bitlen_ += static_cast<std::uint64_t> (datalen_) * 8ull;
    data_[63] = static_cast<std::uint8_t> (bitlen_);
    data_[62] = static_cast<std::uint8_t> (bitlen_ >> 8);
    data_[61] = static_cast<std::uint8_t> (bitlen_ >> 16);
    data_[60] = static_cast<std::uint8_t> (bitlen_ >> 24);
    data_[59] = static_cast<std::uint8_t> (bitlen_ >> 32);
    data_[58] = static_cast<std::uint8_t> (bitlen_ >> 40);
    data_[57] = static_cast<std::uint8_t> (bitlen_ >> 48);
    data_[56] = static_cast<std::uint8_t> (bitlen_ >> 56);
    Transform (state_, data_);
    for (i = 0; i < 4; ++i)
      {
        hash[i] = (state_[0] >> (24 - i * 8)) & 0xff;
        hash[i + 4] = (state_[1] >> (24 - i * 8)) & 0xff;
        hash[i + 8] = (state_[2] >> (24 - i * 8)) & 0xff;
        hash[i + 12] = (state_[3] >> (24 - i * 8)) & 0xff;
        hash[i + 16] = (state_[4] >> (24 - i * 8)) & 0xff;
        hash[i + 20] = (state_[5] >> (24 - i * 8)) & 0xff;
        hash[i + 24] = (state_[6] >> (24 - i * 8)) & 0xff;
        hash[i + 28] = (state_[7] >> (24 - i * 8)) & 0xff;
      }
  }

  std::uint32_t state_[8]{};
  std::uint64_t bitlen_ = 0;
  std::uint8_t data_[64]{};
  std::size_t datalen_ = 0;
};

std::string
Sha256Hex (const unsigned char *data, std::size_t size)
{
  Sha256 ctx;
  ctx.Update (data, size);
  return ctx.FinalHex ();
}

std::filesystem::path
DefaultModelCacheRoot ()
{
  if (const char *override_env = std::getenv ("CORTEXT_MODEL_CACHE_DIR"))
    {
      if (override_env[0] != '\0')
        {
          return std::filesystem::path (override_env);
        }
    }
#if defined(_WIN32)
  char base[MAX_PATH] = {};
  if (SUCCEEDED (SHGetFolderPathA (nullptr, CSIDL_LOCAL_APPDATA, nullptr,
                                   SHGFP_TYPE_CURRENT, base)))
    {
      return std::filesystem::path (base) / "augmem" / "cortext" / "models";
    }
  return std::filesystem::path (".") / "augmem" / "cortext" / "models";
#elif defined(__APPLE__)
  const char *home = std::getenv ("HOME");
  if (home != nullptr && home[0] != '\0')
    {
      return std::filesystem::path (home) / "Library" / "Caches" / "augmem"
             / "cortext" / "models";
    }
  return std::filesystem::path ("/tmp") / "augmem" / "cortext" / "models";
#else
  if (const char *xdg = std::getenv ("XDG_CACHE_HOME"))
    {
      if (xdg[0] != '\0')
        {
          return std::filesystem::path (xdg) / "augmem" / "cortext" / "models";
        }
    }
  const char *home = std::getenv ("HOME");
  if (home != nullptr && home[0] != '\0')
    {
      return std::filesystem::path (home) / ".cache" / "augmem" / "cortext"
             / "models";
    }
  return std::filesystem::path ("/tmp") / "augmem" / "cortext" / "models";
#endif
}

std::filesystem::path
SidecarShaPath (const std::filesystem::path &asset_path)
{
  return asset_path.parent_path ()
         / (asset_path.filename ().string () + ".cortext-sha256");
}

void
WriteSidecarSha (const std::filesystem::path &asset_path,
                 const std::string &sha)
{
  const auto side = SidecarShaPath (asset_path);
  std::ofstream out (side, std::ios::trunc);
  if (out)
    {
      out << sha << '\n';
    }
}

bool
FileMatchesStreaming (const std::filesystem::path &path,
                      const std::string &expected_sha,
                      std::uint64_t expected_size)
{
  std::error_code ec;
  if (!std::filesystem::is_regular_file (path, ec))
    {
      return false;
    }
  if (static_cast<std::uint64_t> (std::filesystem::file_size (path, ec))
      != expected_size)
    {
      return false;
    }
  // Fast path: sidecar written after a successful assemble/materialize.
  {
    std::ifstream side (SidecarShaPath (path));
    std::string recorded;
    if (side && std::getline (side, recorded) && recorded == expected_sha)
      {
        return true;
      }
  }
  std::ifstream in (path, std::ios::binary);
  if (!in)
    {
      return false;
    }
  Sha256 ctx;
  std::vector<unsigned char> chunk (1024 * 1024);
  std::uint64_t total = 0;
  while (in)
    {
      in.read (reinterpret_cast<char *> (chunk.data ()),
               static_cast<std::streamsize> (chunk.size ()));
      const auto n = in.gcount ();
      if (n > 0)
        {
          ctx.Update (chunk.data (), static_cast<std::size_t> (n));
          total += static_cast<std::uint64_t> (n);
        }
    }
  if (total != expected_size || ctx.FinalHex () != expected_sha)
    {
      return false;
    }
  WriteSidecarSha (path, expected_sha);
  return true;
}

std::filesystem::path
UniqueTempPath (const std::filesystem::path &dest)
{
  // Per-process/thread temp so concurrent first-use materialize cannot
  // race on a shared deterministic `.tmp` name.
  const auto stamp
      = std::chrono::high_resolution_clock::now ().time_since_epoch ().count ();
  const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id ());
  return dest.parent_path ()
         / (std::string (".") + dest.filename ().string () + ".tmp."
            + std::to_string (static_cast<unsigned long long> (stamp)) + "."
            + std::to_string (static_cast<unsigned long long> (tid)));
}

void
ReplaceFile (const std::filesystem::path &tmp,
             const std::filesystem::path &dest,
             const std::string &expected_sha,
             std::uint64_t expected_size)
{
  std::error_code ec;
  std::filesystem::rename (tmp, dest, ec);
  if (!ec)
    {
      return;
    }
  // On Windows (and some POSIX races) rename fails if dest exists. Only drop
  // our temp if dest is already the correct payload; otherwise replace the
  // stale/corrupt dest. Never promote a bad dest by writing a fresh sidecar.
  if (FileMatchesStreaming (dest, expected_sha, expected_size))
    {
      std::filesystem::remove (tmp, ec);
      return;
    }
  std::filesystem::remove (dest, ec);
  std::filesystem::rename (tmp, dest, ec);
  if (ec)
    {
      std::filesystem::remove (tmp, ec);
      throw std::runtime_error ("failed to replace embedded AIST cache file: "
                                + dest.string () + " (" + ec.message () + ")");
    }
}

void
WriteAtomically (const std::filesystem::path &dest, const unsigned char *data,
                 std::size_t size, const std::string &expected_sha)
{
  if (Sha256Hex (data, size) != expected_sha)
    {
      throw std::runtime_error (
          "embedded AIST asset digest mismatch before materialize");
    }
  std::filesystem::create_directories (dest.parent_path ());
  const auto tmp = UniqueTempPath (dest);
  {
    std::ofstream out (tmp, std::ios::binary | std::ios::trunc);
    if (!out)
      {
        throw std::runtime_error ("failed to open temp path for embedded AIST: "
                                  + tmp.string ());
      }
    out.write (reinterpret_cast<const char *> (data),
               static_cast<std::streamsize> (size));
    out.flush ();
    if (!out)
      {
        std::error_code ec;
        std::filesystem::remove (tmp, ec);
        throw std::runtime_error ("failed writing embedded AIST asset: "
                                  + tmp.string ());
      }
  }
  ReplaceFile (tmp, dest, expected_sha, static_cast<std::uint64_t> (size));
  WriteSidecarSha (dest, expected_sha);
}

/// Concatenate linked shards into dest; verify per-shard and whole digests.
void
AssembleShardsAtomically (const std::filesystem::path &dest)
{
  std::filesystem::create_directories (dest.parent_path ());
  const auto tmp = UniqueTempPath (dest);
  Sha256 whole;
  std::uint64_t total = 0;
  {
    std::ofstream out (tmp, std::ios::binary | std::ios::trunc);
    if (!out)
      {
        throw std::runtime_error (
            "failed to open temp path for embedded AIST reassembly: "
            + tmp.string ());
      }
    for (std::size_t i = 0; i < CORTEXT_EMBEDDED_AIST_SHARD_COUNT; ++i)
      {
        const unsigned char *data = CortextEmbeddedAistShardData (i);
        const std::size_t size
            = static_cast<std::size_t> (kCortextEmbeddedAistShardSizes[i]);
        if (data == nullptr || size == 0)
          {
            std::error_code ec;
            std::filesystem::remove (tmp, ec);
            throw std::runtime_error ("missing embedded AIST shard payload");
          }
        if (Sha256Hex (data, size) != kCortextEmbeddedAistShardSha256[i])
          {
            std::error_code ec;
            std::filesystem::remove (tmp, ec);
            throw std::runtime_error (
                "embedded AIST shard digest mismatch at index "
                + std::to_string (i));
          }
        out.write (reinterpret_cast<const char *> (data),
                   static_cast<std::streamsize> (size));
        if (!out)
          {
            std::error_code ec;
            std::filesystem::remove (tmp, ec);
            throw std::runtime_error (
                "failed writing embedded AIST shard to temp file");
          }
        whole.Update (data, size);
        total += static_cast<std::uint64_t> (size);
      }
    out.flush ();
  }
  if (total != CORTEXT_EMBEDDED_AIST_GGUF_SIZE
      || whole.FinalHex () != CORTEXT_EMBEDDED_AIST_GGUF_SHA256)
    {
      std::error_code ec;
      std::filesystem::remove (tmp, ec);
      throw std::runtime_error (
          "assembled embedded AIST model digest/size mismatch");
    }
  ReplaceFile (tmp, dest, CORTEXT_EMBEDDED_AIST_GGUF_SHA256,
               CORTEXT_EMBEDDED_AIST_GGUF_SIZE);
  WriteSidecarSha (dest, CORTEXT_EMBEDDED_AIST_GGUF_SHA256);
}

bool
EmbeddedShardsPresent ()
{
  if (CORTEXT_EMBEDDED_AIST_SHARD_COUNT == 0)
    {
      return false;
    }
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < CORTEXT_EMBEDDED_AIST_SHARD_COUNT; ++i)
    {
      if (CortextEmbeddedAistShardData (i) == nullptr)
        {
          return false;
        }
      total += kCortextEmbeddedAistShardSizes[i];
    }
  return total == CORTEXT_EMBEDDED_AIST_GGUF_SIZE
         && CORTEXT_EMBEDDED_AIST_VOCAB_SIZE > 0;
}

#endif

} // namespace

bool
HasEmbeddedAistModel ()
{
#if defined(CORTEXT_HAS_EMBEDDED_AIST_MODEL) && CORTEXT_HAS_EMBEDDED_AIST_MODEL
  return EmbeddedShardsPresent ();
#else
  return false;
#endif
}

std::optional<std::filesystem::path>
MaterializeEmbeddedAistModel ()
{
#if !(defined(CORTEXT_HAS_EMBEDDED_AIST_MODEL) && CORTEXT_HAS_EMBEDDED_AIST_MODEL)
  return std::nullopt;
#else
  if (!HasEmbeddedAistModel ())
    {
      return std::nullopt;
    }
  const auto cache = DefaultModelCacheRoot ();
  const auto model_path = cache / "AIST-87M-GGUF" / "AIST-87M_q8_0.gguf";
  const auto vocab_path = cache / "mdbr-leaf-ir" / "vocab.txt";

  // Assemble linked shards → full .gguf in the cache (once).
  if (!FileMatchesStreaming (model_path, CORTEXT_EMBEDDED_AIST_GGUF_SHA256,
                             CORTEXT_EMBEDDED_AIST_GGUF_SIZE))
    {
      AssembleShardsAtomically (model_path);
    }
  if (!FileMatchesStreaming (vocab_path, CORTEXT_EMBEDDED_AIST_VOCAB_SHA256,
                             CORTEXT_EMBEDDED_AIST_VOCAB_SIZE))
    {
      WriteAtomically (vocab_path, &cortext_embedded_aist_vocab_start,
                       static_cast<std::size_t> (CORTEXT_EMBEDDED_AIST_VOCAB_SIZE),
                       CORTEXT_EMBEDDED_AIST_VOCAB_SHA256);
    }
  return model_path;
#endif
}

std::optional<std::filesystem::path>
MaterializeEmbeddedAistVocab ()
{
  if (auto model = MaterializeEmbeddedAistModel ())
    {
      return model->parent_path ().parent_path () / "mdbr-leaf-ir" / "vocab.txt";
    }
  return std::nullopt;
}

} // namespace cortext
