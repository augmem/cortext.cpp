/// @file
/// @brief ImageBind encoder implementation (tries GPU, falls back to CPU).
///
/// This module requires ONNX Runtime, filesystem I/O for loading BPE vocabulary
/// and model files, and zlib for decompressing the vocabulary. It is completely
/// disabled when CORTEXT_ENABLE_IMAGEBIND_ORT is not defined, and all methods
/// throw std::runtime_error to indicate unavailability.
///
/// WASM builds should not link this module or should use stub encoders instead.
#include "cortext/encoder/imagebind.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <complex>
#include <mutex>
#include <vector>

#if defined(CORTEXT_ENABLE_IMAGEBIND_ORT)
#include <onnxruntime/onnxruntime_cxx_api.h>
#if defined(__APPLE__)
#include <onnxruntime/coreml_provider_factory.h>
#endif
#include <zlib.h>

#include "cortext/core/thread_config.hpp"

#endif

namespace cortext
{

namespace
{

#if defined(CORTEXT_ENABLE_IMAGEBIND_ORT)

constexpr float kPi = 3.14159265358979323846f;

inline std::string
Trim (std::string s)
{
  auto is_space = [] (unsigned char c) { return std::isspace (c) != 0; };
  while (!s.empty () && is_space (static_cast<unsigned char> (s.front ())))
    s.erase (s.begin ());
  while (!s.empty () && is_space (static_cast<unsigned char> (s.back ())))
    s.pop_back ();
  return s;
}

inline std::string
WhitespaceCleanLower (std::string s)
{
  std::string out;
  out.reserve (s.size ());
  bool prev_space = true;
  for (unsigned char c : s)
    {
      if (std::isspace (c))
        {
          if (!prev_space)
            out.push_back (' ');
          prev_space = true;
          continue;
        }
      prev_space = false;
      out.push_back (static_cast<char> (std::tolower (c)));
    }
  return Trim (out);
}

inline std::string
Utf8FromCodepoint (uint32_t cp)
{
  std::string out;
  if (cp <= 0x7F)
    {
      out.push_back (static_cast<char> (cp));
      return out;
    }
  if (cp <= 0x7FF)
    {
      out.push_back (static_cast<char> (0xC0 | ((cp >> 6) & 0x1F)));
      out.push_back (static_cast<char> (0x80 | (cp & 0x3F)));
      return out;
    }
  if (cp <= 0xFFFF)
    {
      out.push_back (static_cast<char> (0xE0 | ((cp >> 12) & 0x0F)));
      out.push_back (static_cast<char> (0x80 | ((cp >> 6) & 0x3F)));
      out.push_back (static_cast<char> (0x80 | (cp & 0x3F)));
      return out;
    }
  out.push_back (static_cast<char> (0xF0 | ((cp >> 18) & 0x07)));
  out.push_back (static_cast<char> (0x80 | ((cp >> 12) & 0x3F)));
  out.push_back (static_cast<char> (0x80 | ((cp >> 6) & 0x3F)));
  out.push_back (static_cast<char> (0x80 | (cp & 0x3F)));
  return out;
}

inline std::vector<std::string>
Utf8Split (const std::string &s)
{
  std::vector<std::string> out;
  for (std::size_t i = 0; i < s.size ();)
    {
      const unsigned char c = static_cast<unsigned char> (s[i]);
      std::size_t len = 1;
      if ((c & 0x80) == 0x00)
        len = 1;
      else if ((c & 0xE0) == 0xC0)
        len = 2;
      else if ((c & 0xF0) == 0xE0)
        len = 3;
      else if ((c & 0xF8) == 0xF0)
        len = 4;
      if (i + len > s.size ())
        len = 1;
      out.push_back (s.substr (i, len));
      i += len;
    }
  return out;
}

inline std::vector<std::string>
ReadGzipLines (const std::filesystem::path &path,
               std::size_t max_lines_including_header)
{
  std::vector<std::string> lines;
  gzFile f = gzopen (path.string ().c_str (), "rb");
  if (!f)
    throw std::runtime_error ("Failed to open gzip file: " + path.string ());

  std::string buf;
  buf.resize (1 << 15);
  while (lines.size () < max_lines_including_header)
    {
      char *res = gzgets (f, buf.data (), static_cast<int> (buf.size ()));
      if (!res)
        break;
      std::string line (res);
      while (!line.empty ()
             && (line.back () == '\n' || line.back () == '\r'))
        line.pop_back ();
      lines.push_back (std::move (line));
    }
  gzclose (f);
  return lines;
}

inline std::filesystem::path
FindBpePath (const std::filesystem::path &models_dir)
{
  namespace fs = std::filesystem;
  const std::vector<fs::path> direct = {
    models_dir / "bpe" / "bpe_simple_vocab_16e6.txt.gz",
    models_dir / "bpe_simple_vocab_16e6.txt.gz",
  };
  for (const auto &p : direct)
    {
      if (fs::exists (p))
        return p;
    }

  fs::path cur = fs::absolute (models_dir);
  while (!cur.empty ())
    {
      fs::path cand
          = cur / "poc" / "ImageBind" / "imagebind" / "bpe"
            / "bpe_simple_vocab_16e6.txt.gz";
      if (fs::exists (cand))
        return cand;
      const fs::path parent = cur.parent_path ();
      if (parent == cur)
        break;
      cur = parent;
    }

  throw std::runtime_error (
      "ImageBind BPE merges not found. Expected one of: "
      + (models_dir / "bpe" / "bpe_simple_vocab_16e6.txt.gz").string ()
      + " or " + (models_dir / "bpe_simple_vocab_16e6.txt.gz").string ()
      + " or a repo checkout containing "
        "poc/ImageBind/imagebind/bpe/bpe_simple_vocab_16e6.txt.gz");
}

struct PairHash
{
  std::size_t
  operator() (const std::pair<std::string, std::string> &p) const noexcept
  {
    const std::hash<std::string> h;
    return h (p.first) ^ (h (p.second) << 1);
  }
};

inline std::unordered_map<uint8_t, std::string>
BytesToUnicode ()
{
  std::vector<int> bs;
  bs.reserve (256);
  for (int i = static_cast<int> ('!'); i <= static_cast<int> ('~'); ++i)
    bs.push_back (i);
  for (int i = 0xA1; i <= 0xAC; ++i)
    bs.push_back (i);
  for (int i = 0xAE; i <= 0xFF; ++i)
    bs.push_back (i);

  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; ++b)
    {
      if (std::find (bs.begin (), bs.end (), b) == bs.end ())
        {
          bs.push_back (b);
          cs.push_back (256 + n);
          ++n;
        }
    }

  std::unordered_map<uint8_t, std::string> out;
  out.reserve (256);
  for (std::size_t i = 0; i < bs.size (); ++i)
    {
      out.emplace (static_cast<uint8_t> (bs[i]),
                   Utf8FromCodepoint (static_cast<uint32_t> (cs[i])));
    }
  return out;
}

inline std::unordered_set<std::pair<std::string, std::string>, PairHash>
GetPairs (const std::vector<std::string> &word)
{
  std::unordered_set<std::pair<std::string, std::string>, PairHash> pairs;
  if (word.size () < 2)
    return pairs;
  for (std::size_t i = 0; i + 1 < word.size (); ++i)
    pairs.insert ({ word[i], word[i + 1] });
  return pairs;
}

class SimpleTokenizer
{
public:
  explicit SimpleTokenizer (std::filesystem::path bpe_path,
                            int context_length = 77)
      : byte_encoder_ (BytesToUnicode ()),
        context_length_ (context_length)
  {
    // Match OpenAI CLIP vocab size 49408:
    // 256 bytes + 256 bytes</w> + 48894 merges + 2 special tokens.
    constexpr std::size_t kMaxMerges = 49152 - 256 - 2; // 48894
    const auto lines = ReadGzipLines (bpe_path, kMaxMerges + 1);
    if (lines.size () < 2)
      throw std::runtime_error ("BPE merges file appears empty: "
                                + bpe_path.string ());

    merges_.reserve (kMaxMerges);
    for (std::size_t i = 1; i < lines.size () && merges_.size () < kMaxMerges;
         ++i)
      {
        const std::string &line = lines[i];
        const std::size_t sp = line.find (' ');
        if (sp == std::string::npos)
          continue;
        merges_.push_back ({ line.substr (0, sp), line.substr (sp + 1) });
      }

    std::vector<std::string> vocab;
    vocab.reserve (256 * 2 + merges_.size () + 2);
    std::vector<std::string> base;
    base.reserve (256);
    for (int b = 0; b < 256; ++b)
      base.push_back (byte_encoder_.at (static_cast<uint8_t> (b)));
    for (const auto &v : base)
      vocab.push_back (v);
    for (const auto &v : base)
      vocab.push_back (v + "</w>");
    for (const auto &m : merges_)
      vocab.push_back (m.first + m.second);
    vocab.push_back ("<|startoftext|>");
    vocab.push_back ("<|endoftext|>");

    encoder_.reserve (vocab.size ());
    for (std::size_t i = 0; i < vocab.size (); ++i)
      encoder_[vocab[i]] = static_cast<int> (i);

    bpe_ranks_.reserve (merges_.size ());
    for (std::size_t i = 0; i < merges_.size (); ++i)
      bpe_ranks_[merges_[i]] = static_cast<int> (i);

    cache_["<|startoftext|>"] = "<|startoftext|>";
    cache_["<|endoftext|>"] = "<|endoftext|>";

    sot_id_ = GetIdOrThrow ("<|startoftext|>");
    eot_id_ = GetIdOrThrow ("<|endoftext|>");
  }

  std::vector<int64_t>
  Tokenize (const std::string &text) const
  {
    std::vector<int64_t> result (static_cast<std::size_t> (context_length_),
                                 0);
    std::vector<int> toks;
    toks.reserve (static_cast<std::size_t> (context_length_));
    toks.push_back (sot_id_);
    const auto enc = Encode (text);
    toks.insert (toks.end (), enc.begin (), enc.end ());
    if (static_cast<int> (toks.size ()) > context_length_ - 1)
      toks.resize (static_cast<std::size_t> (context_length_ - 1));
    toks.push_back (eot_id_);
    for (std::size_t i = 0; i < toks.size () && i < result.size (); ++i)
      result[i] = toks[i];
    return result;
  }

private:
  int
  GetIdOrThrow (const std::string &token) const
  {
    const auto it = encoder_.find (token);
    if (it == encoder_.end ())
      throw std::runtime_error ("Tokenizer missing token: " + token);
    return it->second;
  }

  std::vector<int>
  Encode (const std::string &text) const
  {
    static const std::regex pat (
        R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[A-Za-z]+|[0-9]|[^\sA-Za-z0-9]+)",
        std::regex::icase);

    const std::string cleaned = WhitespaceCleanLower (text);
    std::vector<int> bpe_tokens;
    std::sregex_iterator it (cleaned.begin (), cleaned.end (), pat);
    std::sregex_iterator end;
    for (; it != end; ++it)
      {
        const std::string tok = it->str ();
        std::string encoded;
        encoded.reserve (tok.size () * 2);
        for (unsigned char b : tok)
          encoded += byte_encoder_.at (static_cast<uint8_t> (b));

        const std::string bpe_out = Bpe (encoded);
        std::size_t start = 0;
        while (start < bpe_out.size ())
          {
            const std::size_t sp = bpe_out.find (' ', start);
            const std::size_t len
                = (sp == std::string::npos) ? std::string::npos : sp - start;
            const std::string piece = bpe_out.substr (start, len);
            if (!piece.empty ())
              {
                const auto eid = encoder_.find (piece);
                if (eid != encoder_.end ())
                  bpe_tokens.push_back (eid->second);
              }
            if (sp == std::string::npos)
              break;
            start = sp + 1;
          }
      }
    return bpe_tokens;
  }

  std::string
  Bpe (const std::string &token) const
  {
    auto c_it = cache_.find (token);
    if (c_it != cache_.end ())
      return c_it->second;

    const auto chars = Utf8Split (token);
    if (chars.empty ())
      return token;

    std::vector<std::string> word;
    word.reserve (chars.size ());
    for (std::size_t i = 0; i + 1 < chars.size (); ++i)
      word.push_back (chars[i]);
    word.push_back (chars.back () + "</w>");

    auto pairs = GetPairs (word);
    if (pairs.empty ())
      {
        cache_[token] = token + "</w>";
        return cache_[token];
      }

    while (true)
      {
        int best_rank = std::numeric_limits<int>::max ();
        std::pair<std::string, std::string> best;
        bool found = false;
        for (const auto &p : pairs)
          {
            const auto r_it = bpe_ranks_.find (p);
            if (r_it == bpe_ranks_.end ())
              continue;
            if (r_it->second < best_rank)
              {
                best_rank = r_it->second;
                best = p;
                found = true;
              }
          }
        if (!found)
          break;

        const std::string &first = best.first;
        const std::string &second = best.second;
        std::vector<std::string> new_word;
        new_word.reserve (word.size ());

        std::size_t i = 0;
        while (i < word.size ())
          {
            std::size_t j = i;
            for (; j < word.size (); ++j)
              {
                if (word[j] == first)
                  break;
              }
            for (std::size_t k = i; k < j; ++k)
              new_word.push_back (word[k]);
            i = j;
            if (i >= word.size ())
              break;

            if (word[i] == first && (i + 1) < word.size ()
                && word[i + 1] == second)
              {
                new_word.push_back (first + second);
                i += 2;
              }
            else
              {
                new_word.push_back (word[i]);
                i += 1;
              }
          }
        word = std::move (new_word);
        if (word.size () == 1)
          break;
        pairs = GetPairs (word);
      }

    std::string out;
    for (std::size_t i = 0; i < word.size (); ++i)
      {
        if (i)
          out.push_back (' ');
        out += word[i];
      }
    cache_[token] = out;
    return out;
  }

  std::unordered_map<uint8_t, std::string> byte_encoder_;
  int context_length_ = 77;
  std::vector<std::pair<std::string, std::string>> merges_;
  std::unordered_map<std::string, int> encoder_;
  std::unordered_map<std::pair<std::string, std::string>, int, PairHash>
      bpe_ranks_;
  mutable std::unordered_map<std::string, std::string> cache_;
  int sot_id_ = -1;
  int eot_id_ = -1;
};

inline std::filesystem::path
FindModel (const std::filesystem::path &models_dir, const std::string &base)
{
  namespace fs = std::filesystem;
  const std::vector<fs::path> candidates = {
    models_dir / (base + "_int8.onnx"),
    models_dir / (base + ".onnx"),
  };
  for (const auto &p : candidates)
    {
      if (fs::exists (p))
        return p;
    }
  throw std::runtime_error ("ImageBind model not found for: " + base);
}

inline void
L2NormalizeInPlace (std::vector<float> &v)
{
  double s2 = 0.0;
  for (float x : v)
    s2 += static_cast<double> (x) * static_cast<double> (x);
  const double n = std::sqrt (s2);
  if (n <= 0.0)
    return;
  for (float &x : v)
    x = static_cast<float> (static_cast<double> (x) / n);
}

inline void
StandardizeEmbeddingDim (std::vector<float> &v, std::size_t dim)
{
  if (v.size () > dim)
    {
      v.resize (dim);
    }
  else if (v.size () < dim)
    {
      v.resize (dim, 0.0f);
    }
}

inline void
FftInPlace (std::vector<std::complex<float>> &a)
{
  const std::size_t n = a.size ();
  // bit-reversal permutation
  for (std::size_t i = 1, j = 0; i < n; ++i)
    {
      std::size_t bit = n >> 1;
      for (; j & bit; bit >>= 1)
        j ^= bit;
      j ^= bit;
      if (i < j)
        std::swap (a[i], a[j]);
    }

  for (std::size_t len = 2; len <= n; len <<= 1)
    {
      const float ang = -2.0f * kPi
                        / static_cast<float> (len);
      const std::complex<float> wlen (std::cos (ang), std::sin (ang));
      for (std::size_t i = 0; i < n; i += len)
        {
          std::complex<float> w (1.0f, 0.0f);
          for (std::size_t j = 0; j < len / 2; ++j)
            {
              const std::complex<float> u = a[i + j];
              const std::complex<float> v = a[i + j + len / 2] * w;
              a[i + j] = u + v;
              a[i + j + len / 2] = u - v;
              w *= wlen;
            }
        }
    }
}

inline float
HtkMel (float hz)
{
  return 2595.0f * std::log10 (1.0f + hz / 700.0f);
}

inline float
HtkMelInv (float mel)
{
  return 700.0f * (std::pow (10.0f, mel / 2595.0f) - 1.0f);
}

inline std::vector<std::vector<float>>
BuildMelFilterbank (int sample_rate, int n_fft, int num_mel_bins)
{
  const int n_freq = n_fft / 2 + 1;
  const float f_min = 0.0f;
  const float f_max = static_cast<float> (sample_rate) / 2.0f;
  const float mel_min = HtkMel (f_min);
  const float mel_max = HtkMel (f_max);

  std::vector<float> mel_points;
  mel_points.resize (static_cast<std::size_t> (num_mel_bins + 2));
  for (int i = 0; i < num_mel_bins + 2; ++i)
    {
      mel_points[static_cast<std::size_t> (i)]
          = mel_min
            + (mel_max - mel_min) * static_cast<float> (i)
                  / static_cast<float> (num_mel_bins + 1);
    }

  std::vector<int> bins;
  bins.resize (static_cast<std::size_t> (num_mel_bins + 2));
  for (int i = 0; i < num_mel_bins + 2; ++i)
    {
      const float hz = HtkMelInv (mel_points[static_cast<std::size_t> (i)]);
      int b = static_cast<int> (
          std::floor ((static_cast<float> (n_fft) + 1.0f) * hz
                      / static_cast<float> (sample_rate)));
      if (b < 0)
        b = 0;
      if (b >= n_freq)
        b = n_freq - 1;
      bins[static_cast<std::size_t> (i)] = b;
    }

  std::vector<std::vector<float>> fbanks;
  fbanks.resize (static_cast<std::size_t> (num_mel_bins));
  for (int m = 0; m < num_mel_bins; ++m)
    {
      fbanks[static_cast<std::size_t> (m)].assign (
          static_cast<std::size_t> (n_freq), 0.0f);
      const int left = bins[static_cast<std::size_t> (m)];
      const int center = bins[static_cast<std::size_t> (m + 1)];
      const int right = bins[static_cast<std::size_t> (m + 2)];
      if (center == left || right == center)
        continue;
      for (int k = left; k < center; ++k)
        {
          fbanks[static_cast<std::size_t> (m)][static_cast<std::size_t> (k)]
              = (static_cast<float> (k - left)
                 / static_cast<float> (center - left));
        }
      for (int k = center; k < right; ++k)
        {
          fbanks[static_cast<std::size_t> (m)][static_cast<std::size_t> (k)]
              = (static_cast<float> (right - k)
                 / static_cast<float> (right - center));
        }
    }
  return fbanks;
}

inline std::vector<float>
WaveformToMelSpec (const float *pcm, std::size_t num_samples, int sample_rate,
                   int num_mel_bins, int target_length)
{
  if (!pcm)
    throw std::runtime_error ("EncodeAudio: pcm is null");
  if (num_samples == 0)
    throw std::runtime_error ("EncodeAudio: num_samples is 0");
  if (sample_rate != 16000)
    {
      // For now, require 16kHz to keep preprocessing deterministic.
      throw std::runtime_error ("EncodeAudio: expected 16kHz PCM");
    }

  // Subtract mean.
  double mean = 0.0;
  for (std::size_t i = 0; i < num_samples; ++i)
    mean += pcm[i];
  mean /= static_cast<double> (num_samples);

  // Kaldi fbank-like params (ImageBind defaults)
  const int frame_length = 400; // 25ms @ 16k
  const int frame_shift = 160;  // 10ms @ 16k
  const int n_fft = 512;
  const int n_freq = n_fft / 2 + 1;

  static const std::vector<std::vector<float>> mel_fbanks
      = BuildMelFilterbank (16000, n_fft, 128);

  // Hann window
  std::vector<float> window;
  window.resize (static_cast<std::size_t> (frame_length));
  for (int i = 0; i < frame_length; ++i)
    {
      window[static_cast<std::size_t> (i)]
          = 0.5f
            - 0.5f
                  * std::cos (2.0f * kPi
                              * static_cast<float> (i)
                              / static_cast<float> (frame_length - 1));
    }

  const std::size_t num_frames
      = (num_samples <= static_cast<std::size_t> (frame_length))
            ? 1
            : (1 + (num_samples - static_cast<std::size_t> (frame_length))
                     / static_cast<std::size_t> (frame_shift));

  // fbank: [mel_bins, frames], padded/truncated to target_length
  std::vector<float> fbank;
  fbank.assign (static_cast<std::size_t> (num_mel_bins * target_length), 0.0f);

  std::vector<std::complex<float>> buf;
  buf.resize (static_cast<std::size_t> (n_fft));
  std::vector<float> power;
  power.resize (static_cast<std::size_t> (n_freq));

  const std::size_t frames_to_compute
      = std::min (num_frames, static_cast<std::size_t> (target_length));

  for (std::size_t frame = 0; frame < frames_to_compute; ++frame)
    {
      const std::size_t start = frame * static_cast<std::size_t> (frame_shift);
      // fill buffer
      for (int i = 0; i < n_fft; ++i)
        buf[static_cast<std::size_t> (i)] = std::complex<float> (0.0f, 0.0f);
      for (int i = 0; i < frame_length; ++i)
        {
          const std::size_t idx = start + static_cast<std::size_t> (i);
          float x = 0.0f;
          if (idx < num_samples)
            x = pcm[idx] - static_cast<float> (mean);
          buf[static_cast<std::size_t> (i)]
              = std::complex<float> (x * window[static_cast<std::size_t> (i)],
                                     0.0f);
        }

      FftInPlace (buf);
      for (int k = 0; k < n_freq; ++k)
        {
          const auto &c = buf[static_cast<std::size_t> (k)];
          power[static_cast<std::size_t> (k)] = c.real () * c.real ()
                                                + c.imag () * c.imag ();
        }

      for (int m = 0; m < num_mel_bins; ++m)
        {
          double e = 0.0;
          const auto &w = mel_fbanks[static_cast<std::size_t> (m)];
          for (int k = 0; k < n_freq; ++k)
            e += static_cast<double> (power[static_cast<std::size_t> (k)])
                 * static_cast<double> (w[static_cast<std::size_t> (k)]);
          // log with epsilon
          const double eps = 1e-10;
          const float v = static_cast<float> (std::log (e + eps));
          fbank[static_cast<std::size_t> (m) * target_length + frame] = v;
        }
    }

  // Add channel dimension: return [1, mel_bins, target_length] flattened.
  std::vector<float> out;
  out.resize (static_cast<std::size_t> (1 * num_mel_bins * target_length));
  std::copy (fbank.begin (), fbank.end (), out.begin ());
  return out;
}

inline float
CubicWeight (float x)
{
  // Keys cubic kernel with a = -0.5 (Catmull-Rom)
  const float a = -0.5f;
  x = std::fabs (x);
  if (x < 1.0f)
    return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
  if (x < 2.0f)
    return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
  return 0.0f;
}

inline float
GetPixel (const std::uint8_t *data, int width, int height, int channels, int x,
          int y, int c)
{
  if (x < 0)
    x = 0;
  if (y < 0)
    y = 0;
  if (x >= width)
    x = width - 1;
  if (y >= height)
    y = height - 1;
  const int idx = (y * width + x) * channels + c;
  return static_cast<float> (data[idx]);
}

inline std::vector<float>
ResizeBicubicCHW (const std::uint8_t *data, int width, int height, int channels,
                  int out_w, int out_h)
{
  // Input assumed RGB (channels>=3). Output is float CHW in [0,1].
  std::vector<float> out;
  out.resize (static_cast<std::size_t> (3 * out_w * out_h), 0.0f);

  const float scale_x = static_cast<float> (width) / static_cast<float> (out_w);
  const float scale_y
      = static_cast<float> (height) / static_cast<float> (out_h);

  for (int oy = 0; oy < out_h; ++oy)
    {
      const float fy = (static_cast<float> (oy) + 0.5f) * scale_y - 0.5f;
      const int iy = static_cast<int> (std::floor (fy));
      for (int ox = 0; ox < out_w; ++ox)
        {
          const float fx = (static_cast<float> (ox) + 0.5f) * scale_x - 0.5f;
          const int ix = static_cast<int> (std::floor (fx));

          for (int c = 0; c < 3; ++c)
            {
              float acc = 0.0f;
              float wsum = 0.0f;
              for (int m = -1; m <= 2; ++m)
                {
                  const float wy = CubicWeight (fy - (iy + m));
                  for (int n = -1; n <= 2; ++n)
                    {
                      const float wx = CubicWeight (fx - (ix + n));
                      const float w = wy * wx;
                      const float p
                          = GetPixel (data, width, height, channels, ix + n,
                                      iy + m, c);
                      acc += w * p;
                      wsum += w;
                    }
                }
              if (wsum != 0.0f)
                acc /= wsum;
              const std::size_t ofs
                  = static_cast<std::size_t> (oy * out_w + ox);
              out[static_cast<std::size_t> (c) * out_w * out_h + ofs]
                  = acc / 255.0f;
            }
        }
    }
  return out;
}

inline std::vector<float>
VisionPreprocess (const std::uint8_t *data, int width, int height, int channels)
{
  if (!data || width <= 0 || height <= 0 || channels < 3)
    throw std::runtime_error ("EncodeImage: invalid input image");

  // Resize shortest side to 224, bicubic, then center crop 224x224.
  const int target = 224;
  int resized_w = 0;
  int resized_h = 0;
  if (width <= height)
    {
      resized_w = target;
      resized_h = static_cast<int> (
          std::round (static_cast<double> (height) * target
                      / static_cast<double> (width)));
    }
  else
    {
      resized_h = target;
      resized_w = static_cast<int> (
          std::round (static_cast<double> (width) * target
                      / static_cast<double> (height)));
    }

  auto resized = ResizeBicubicCHW (data, width, height, channels, resized_w,
                                   resized_h);

  // Center crop
  const int x0 = std::max (0, (resized_w - target) / 2);
  const int y0 = std::max (0, (resized_h - target) / 2);

  std::vector<float> cropped;
  cropped.resize (static_cast<std::size_t> (3 * target * target), 0.0f);
  for (int c = 0; c < 3; ++c)
    {
      for (int y = 0; y < target; ++y)
        {
          for (int x = 0; x < target; ++x)
            {
              const int sx = x0 + x;
              const int sy = y0 + y;
              const std::size_t s_ofs
                  = static_cast<std::size_t> (c) * resized_w * resized_h
                    + static_cast<std::size_t> (sy * resized_w + sx);
              const std::size_t d_ofs
                  = static_cast<std::size_t> (c) * target * target
                    + static_cast<std::size_t> (y * target + x);
              cropped[d_ofs] = resized[s_ofs];
            }
        }
    }

  // Normalize (ImageBind / OpenCLIP)
  const float mean[3] = { 0.48145466f, 0.4578275f, 0.40821073f };
  const float stdv[3] = { 0.26862954f, 0.26130258f, 0.27577711f };
  for (int c = 0; c < 3; ++c)
    {
      const std::size_t base = static_cast<std::size_t> (c) * target * target;
      for (int i = 0; i < target * target; ++i)
        {
          float v = cropped[base + static_cast<std::size_t> (i)];
          v = (v - mean[c]) / stdv[c];
          cropped[base + static_cast<std::size_t> (i)] = v;
        }
    }
  return cropped; // [3,224,224] CHW
}

#endif // CORTEXT_ENABLE_IMAGEBIND_ORT

} // namespace

struct ImageBindEncoder::Impl
{
  explicit Impl (std::string models_dir_in)
      : models_dir (std::move (models_dir_in))
  {
  }

  std::string models_dir;

#if defined(CORTEXT_ENABLE_IMAGEBIND_ORT)
  Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "cortext-imagebind" };
  std::unique_ptr<Ort::Session> text_session;
  std::unique_ptr<Ort::Session> audio_session;
  std::unique_ptr<Ort::Session> vision_session;
  std::unique_ptr<SimpleTokenizer> tokenizer;

  std::mutex init_mu;
  bool init_attempted = false;
  bool init_ok = false;
  std::string init_error;

  static void
  ConfigureSessionOptions (Ort::SessionOptions &opts)
  {
    auto threads = static_cast<int> (core::GetEmbedThreadCount ());
    opts.SetIntraOpNumThreads (threads);
    opts.SetInterOpNumThreads (threads);
    opts.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_BASIC);
  }

  enum class Provider
  {
    CoreML,
    Cuda,
    Cpu
  };

  static const char *
  ProviderName (Provider provider)
  {
    switch (provider)
      {
      case Provider::CoreML:
        return "CoreML";
      case Provider::Cuda:
        return "CUDA";
      case Provider::Cpu:
        return "CPU";
      }
    return "Unknown";
  }

  static bool
  TryAppendCoreMLProvider (Ort::SessionOptions &opts, std::string *error_out)
  {
#if defined(__APPLE__)
    OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_CoreML (
        opts, 0 /* default: allow ANE/CPU/GPU */);
    if (status != nullptr)
      {
        if (error_out != nullptr)
          {
            *error_out = Ort::GetApi ().GetErrorMessage (status);
          }
        Ort::GetApi ().ReleaseStatus (status);
        return false;
      }
    return true;
#else
    if (error_out != nullptr)
      {
        *error_out = "CoreML execution provider only available on Apple platforms";
      }
    return false;
#endif
  }

  static bool
  TryAppendCudaProvider (Ort::SessionOptions &opts, std::string *error_out)
  {
    OrtCUDAProviderOptions cuda_options{};
    OrtStatus *status
        = Ort::GetApi ().SessionOptionsAppendExecutionProvider_CUDA (
            opts, &cuda_options);
    if (status != nullptr)
      {
        if (error_out != nullptr)
          {
            *error_out = Ort::GetApi ().GetErrorMessage (status);
          }
        Ort::GetApi ().ReleaseStatus (status);
        return false;
      }
    return true;
  }

  bool
  TryCreateSessions (Provider provider,
                     const std::filesystem::path &text_model_path,
                     const std::filesystem::path &audio_model_path,
                     const std::filesystem::path &vision_model_path,
                     const std::filesystem::path &bpe_path,
                     std::string *error_out)
  {
    try
      {
        Ort::SessionOptions opts;
        ConfigureSessionOptions (opts);
        if (provider == Provider::CoreML)
          {
            if (!TryAppendCoreMLProvider (opts, error_out))
              {
                return false;
              }
          }
        else if (provider == Provider::Cuda)
          {
            if (!TryAppendCudaProvider (opts, error_out))
              {
                return false;
              }
          }

        auto text = std::make_unique<Ort::Session> (
            env, text_model_path.c_str (), opts);
        auto audio = std::make_unique<Ort::Session> (
            env, audio_model_path.c_str (), opts);
        auto vision = std::make_unique<Ort::Session> (
            env, vision_model_path.c_str (), opts);
        auto tok = std::make_unique<SimpleTokenizer> (bpe_path, 77);

        text_session = std::move (text);
        audio_session = std::move (audio);
        vision_session = std::move (vision);
        tokenizer = std::move (tok);
        return true;
      }
    catch (const std::exception &e)
      {
        if (error_out != nullptr)
          {
            *error_out = e.what ();
          }
        return false;
      }
  }

  void
  EnsureInitialized ()
  {
    std::lock_guard<std::mutex> lock (init_mu);
    if (init_ok)
      return;
    if (init_attempted)
      throw std::runtime_error (init_error);
    init_attempted = true;
    try
      {
        const std::filesystem::path md (models_dir);
        const std::filesystem::path text_model_path
            = FindModel (md, "text_encoder");
        const std::filesystem::path audio_model_path
            = FindModel (md, "audio_encoder");
        const std::filesystem::path vision_model_path
            = FindModel (md, "vision_encoder");
        const std::filesystem::path bpe_path = FindBpePath (md);

        std::string error;
#if defined(__APPLE__)
        const Provider providers[] = { Provider::CoreML, Provider::Cpu };
#else
        const Provider providers[] = { Provider::Cuda, Provider::Cpu };
#endif
        for (Provider provider : providers)
          {
            if (TryCreateSessions (provider, text_model_path, audio_model_path,
                                   vision_model_path, bpe_path, &error))
              {
                init_ok = true;
                return;
              }
            if (!error.empty ())
              {
                init_error = std::string ("ImageBindEncoder init failed on ")
                             + ProviderName (provider) + ": " + error;
              }
          }
        if (init_error.empty ())
          {
            init_error = "ImageBindEncoder init failed: unknown error";
          }
        throw std::runtime_error (init_error);
      }
    catch (const std::exception &e)
      {
        init_error = std::string ("ImageBindEncoder init failed: ") + e.what ();
        throw;
      }
  }
#endif
};

ImageBindEncoder::ImageBindEncoder (std::string models_dir)
    : impl_ (std::make_unique<Impl> (std::move (models_dir)))
{
}

ImageBindEncoder::~ImageBindEncoder () = default;

void
ImageBindEncoder::EncodeText (const std::string &text,
                              std::vector<float> &out_embedding)
{
#if !defined(CORTEXT_ENABLE_IMAGEBIND_ORT)
  (void)text;
  (void)out_embedding;
  throw std::runtime_error (
      "ImageBindEncoder requires CORTEXT_ENABLE_IMAGEBIND_ORT=ON");
#else
  if (!impl_)
    throw std::runtime_error ("ImageBindEncoder not initialized");
  impl_->EnsureInitialized ();
  if (!impl_->text_session || !impl_->tokenizer)
    throw std::runtime_error ("ImageBindEncoder not initialized");

  const auto tokens = impl_->tokenizer->Tokenize (text);
  std::vector<int64_t> shape{ 1, static_cast<int64_t> (tokens.size ()) };

  Ort::MemoryInfo mem_info
      = Ort::MemoryInfo::CreateCpu (OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::Value input = Ort::Value::CreateTensor<int64_t> (
      mem_info, const_cast<int64_t *> (tokens.data ()),
      static_cast<std::size_t> (tokens.size ()), shape.data (), shape.size ());

  const char *in_names[] = { "text_tokens" };
  const char *out_names[] = { "text_embedding" };
  auto outputs = impl_->text_session->Run (Ort::RunOptions{ nullptr }, in_names,
                                          &input, 1, out_names, 1);
  if (outputs.size () != 1 || !outputs[0].IsTensor ())
    throw std::runtime_error ("text_encoder produced no tensor output");

  auto type_info = outputs[0].GetTensorTypeAndShapeInfo ();
  const auto out_shape = type_info.GetShape ();
  std::size_t out_size = 1;
  for (auto d : out_shape)
    out_size *= static_cast<std::size_t> (d < 0 ? 1 : d);
  float *out = outputs[0].GetTensorMutableData<float> ();
  if (!out || out_size == 0)
    throw std::runtime_error ("text_encoder output is empty");

  out_embedding.assign (out, out + out_size);
  StandardizeEmbeddingDim (out_embedding, static_cast<std::size_t> (kDim));
  L2NormalizeInPlace (out_embedding);
#endif
}

void
ImageBindEncoder::EncodeAudio (const float *pcm, std::size_t num_samples,
                               std::vector<float> &out_embedding)
{
#if !defined(CORTEXT_ENABLE_IMAGEBIND_ORT)
  (void)pcm;
  (void)num_samples;
  (void)out_embedding;
  throw std::runtime_error (
      "ImageBindEncoder requires CORTEXT_ENABLE_IMAGEBIND_ORT=ON");
#else
  if (!impl_)
    throw std::runtime_error ("ImageBindEncoder not initialized");
  impl_->EnsureInitialized ();
  if (!impl_->audio_session)
    throw std::runtime_error ("ImageBindEncoder not initialized");

  constexpr int kSampleRate = 16000;
  constexpr int kNumMelBins = 128;
  constexpr int kTargetLen = 204;
  constexpr int kClipSamples = 2 * kSampleRate;
  constexpr int kClips = 3;
  constexpr float kMean = -4.268f;
  constexpr float kStd = 9.138f;

  // Choose up to 3 evenly spaced 2s clips (match ImageBind reference shape).
  std::vector<std::vector<float>> clips;
  clips.reserve (kClips);
  if (num_samples <= static_cast<std::size_t> (kClipSamples))
    {
      clips.push_back (WaveformToMelSpec (pcm, num_samples, kSampleRate,
                                          kNumMelBins, kTargetLen));
    }
  else
    {
      const std::size_t max_start
          = num_samples - static_cast<std::size_t> (kClipSamples);
      const std::size_t s0 = 0;
      const std::size_t s1 = max_start / 2;
      const std::size_t s2 = max_start;
      const std::size_t starts[3] = { s0, s1, s2 };
      for (int i = 0; i < kClips; ++i)
        {
          clips.push_back (WaveformToMelSpec (
              pcm + starts[i], static_cast<std::size_t> (kClipSamples),
              kSampleRate, kNumMelBins, kTargetLen));
        }
    }

  const int use_clips = static_cast<int> (clips.size ());
  // Flatten to [clips, 1, 128, 204]
  const std::size_t per_clip
      = static_cast<std::size_t> (1 * kNumMelBins * kTargetLen);
  std::vector<float> input_buf;
  input_buf.resize (static_cast<std::size_t> (use_clips) * per_clip);
  for (int c = 0; c < use_clips; ++c)
    {
      for (std::size_t i = 0; i < per_clip; ++i)
        {
          // Normalize (mean/std) as in ImageBind data.py
          input_buf[static_cast<std::size_t> (c) * per_clip + i]
              = (clips[static_cast<std::size_t> (c)][i] - kMean) / kStd;
        }
    }

  // Shape to match model input rank.
  // Some exported models omit shape metadata; if rank is unknown (0), try a
  // small set of canonical shapes used by ImageBind.
  auto RunWithShape = [&] (const float *data_ptr, std::size_t data_len,
                           const std::vector<int64_t> &shape) {
    Ort::MemoryInfo mem_info
        = Ort::MemoryInfo::CreateCpu (OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input = Ort::Value::CreateTensor<float> (
        mem_info, const_cast<float *> (data_ptr), data_len, shape.data (),
        shape.size ());

    const char *in_names[] = { "audio_melspec" };
    const char *out_names[] = { "audio_embedding" };
    return impl_->audio_session->Run (Ort::RunOptions{ nullptr }, in_names,
                                      &input, 1, out_names, 1);
  };

  std::vector<Ort::Value> outputs;
  std::string last_err;
  const std::vector<std::pair<std::vector<int64_t>, std::size_t>> tries = {
    { { 1, use_clips, 1, kNumMelBins, kTargetLen },
      static_cast<std::size_t> (use_clips) * per_clip },
    { { 1, 1, kNumMelBins, kTargetLen }, per_clip },
    { { 1, kNumMelBins, kTargetLen }, per_clip },
  };
  for (const auto &t : tries)
    {
      try
        {
          outputs = RunWithShape (input_buf.data (), t.second, t.first);
          break;
        }
      catch (const Ort::Exception &e)
        {
          last_err = e.what ();
          outputs.clear ();
        }
      catch (const std::exception &e)
        {
          last_err = e.what ();
          outputs.clear ();
        }
    }
  if (outputs.empty ())
    {
      throw std::runtime_error ("audio_encoder inference failed: " + last_err);
    }

  if (outputs.size () != 1 || !outputs[0].IsTensor ())
    throw std::runtime_error ("audio_encoder produced no tensor output");

  auto type_info = outputs[0].GetTensorTypeAndShapeInfo ();
  const auto out_shape = type_info.GetShape ();
  std::size_t out_size = 1;
  for (auto d : out_shape)
    out_size *= static_cast<std::size_t> (d < 0 ? 1 : d);
  float *out = outputs[0].GetTensorMutableData<float> ();
  if (!out || out_size == 0)
    throw std::runtime_error ("audio_encoder output is empty");

  out_embedding.assign (out, out + out_size);
  StandardizeEmbeddingDim (out_embedding, static_cast<std::size_t> (kDim));
  L2NormalizeInPlace (out_embedding);
#endif
}

void
ImageBindEncoder::EncodeImage (const std::uint8_t *data, int width, int height,
                               int channels, std::vector<float> &out_embedding)
{
#if !defined(CORTEXT_ENABLE_IMAGEBIND_ORT)
  (void)data;
  (void)width;
  (void)height;
  (void)channels;
  (void)out_embedding;
  throw std::runtime_error (
      "ImageBindEncoder requires CORTEXT_ENABLE_IMAGEBIND_ORT=ON");
#else
  if (!impl_)
    throw std::runtime_error ("ImageBindEncoder not initialized");
  impl_->EnsureInitialized ();
  if (!impl_->vision_session)
    throw std::runtime_error ("ImageBindEncoder not initialized");

  const auto chw = VisionPreprocess (data, width, height, channels);

  // Shape to match model input rank.
  auto in_info = impl_->vision_session->GetInputTypeInfo (0)
                     .GetTensorTypeAndShapeInfo ();
  const auto in_shape = in_info.GetShape ();
  const std::size_t rank = in_shape.size ();

  std::vector<float> input_buf = chw;
  auto RunWithShape = [&] (const std::vector<float> &buf,
                           const std::vector<int64_t> &shape) {
    Ort::MemoryInfo mem_info
        = Ort::MemoryInfo::CreateCpu (OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input = Ort::Value::CreateTensor<float> (
        mem_info, const_cast<float *> (buf.data ()), buf.size (), shape.data (),
        shape.size ());
    const char *in_names[] = { "vision_input" };
    const char *out_names[] = { "vision_embedding" };
    return impl_->vision_session->Run (Ort::RunOptions{ nullptr }, in_names,
                                       &input, 1, out_names, 1);
  };

  std::vector<Ort::Value> outputs;
  std::string last_err;

  // Candidate shapes (ImageBind uses images as [1,3,224,224] and video as
  // [1,3,2,224,224]).
  const std::vector<int64_t> shape4 = { 1, 3, 224, 224 };
  const int64_t t_from_model = (rank == 5 && in_shape.size () > 2 && in_shape[2] > 0)
                                  ? in_shape[2]
                                  : 2;
  const std::vector<int64_t> shape5 = { 1, 3, t_from_model, 224, 224 };
  std::vector<float> buf5;
  buf5.resize (static_cast<std::size_t> (3 * t_from_model * 224 * 224));
  const std::size_t frame_sz = static_cast<std::size_t> (3 * 224 * 224);
  for (int64_t i = 0; i < t_from_model; ++i)
    {
      std::copy (chw.begin (), chw.end (),
                 buf5.begin () + static_cast<std::size_t> (i) * frame_sz);
    }

  const std::vector<std::pair<const std::vector<float> *, const std::vector<int64_t> *>>
      tries = {
        { &input_buf, &shape4 },
        { &buf5, &shape5 },
      };

  for (const auto &t : tries)
    {
      try
        {
          outputs = RunWithShape (*t.first, *t.second);
          break;
        }
      catch (const Ort::Exception &e)
        {
          last_err = e.what ();
          outputs.clear ();
        }
      catch (const std::exception &e)
        {
          last_err = e.what ();
          outputs.clear ();
        }
    }
  if (outputs.empty ())
    {
      std::string msg = "vision_encoder inference failed: " + last_err
                        + " (input rank=" + std::to_string (rank) + " shape=[";
      for (std::size_t i = 0; i < in_shape.size (); ++i)
        {
          if (i)
            msg += ",";
          msg += std::to_string (in_shape[i]);
        }
      msg += "])";
      throw std::runtime_error (msg);
    }

  if (outputs.size () != 1 || !outputs[0].IsTensor ())
    throw std::runtime_error ("vision_encoder produced no tensor output");

  auto type_info = outputs[0].GetTensorTypeAndShapeInfo ();
  const auto out_shape = type_info.GetShape ();
  std::size_t out_size = 1;
  for (auto d : out_shape)
    out_size *= static_cast<std::size_t> (d < 0 ? 1 : d);
  float *out = outputs[0].GetTensorMutableData<float> ();
  if (!out || out_size == 0)
    throw std::runtime_error ("vision_encoder output is empty");

  out_embedding.assign (out, out + out_size);
  StandardizeEmbeddingDim (out_embedding, static_cast<std::size_t> (kDim));
  L2NormalizeInPlace (out_embedding);
#endif
}

} // namespace cortext
