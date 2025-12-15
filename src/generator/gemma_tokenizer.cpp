#include "cortext/generator/gemma_tokenizer.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace cortext
{

namespace
{

// SentencePiece uses ▁ (U+2581) to represent spaces
constexpr const char *kSpaceReplacement = "\xe2\x96\x81"; // UTF-8 for ▁

/// @brief Replace all occurrences of a substring.
std::string
ReplaceAll (const std::string &str, const std::string &from,
            const std::string &to)
{
  if (from.empty ())
    return str;

  std::string result;
  result.reserve (str.size ());
  std::size_t pos = 0;
  std::size_t prev = 0;

  while ((pos = str.find (from, prev)) != std::string::npos)
    {
      result.append (str, prev, pos - prev);
      result.append (to);
      prev = pos + from.size ();
    }
  result.append (str, prev, std::string::npos);
  return result;
}

/// @brief Hash function for pair of strings (for merge lookups).
struct PairHash
{
  std::size_t
  operator() (const std::pair<std::string, std::string> &p) const
  {
    auto h1 = std::hash<std::string>{}(p.first);
    auto h2 = std::hash<std::string>{}(p.second);
    return h1 ^ (h2 << 1);
  }
};

/// @brief BPE symbol during encoding.
struct BPESymbol
{
  std::string text;
  int prev = -1; // Index of previous symbol (-1 if none)
  int next = -1; // Index of next symbol (-1 if none)
};

/// @brief Priority queue item for BPE merges.
struct MergeCandidate
{
  int priority;  // Lower = higher priority (merge order in file)
  int left_idx;  // Index of left symbol
  int right_idx; // Index of right symbol

  bool
  operator> (const MergeCandidate &other) const
  {
    return priority > other.priority;
  }
};

} // namespace

struct GemmaTokenizer::Impl
{
  // Vocabulary: token string -> ID
  std::unordered_map<std::string, int64_t> vocab;

  // Reverse vocabulary: ID -> token string
  std::unordered_map<int64_t, std::string> id_to_token;

  // Merges: (left, right) -> priority (0 = highest priority)
  std::unordered_map<std::pair<std::string, std::string>, int, PairHash>
      merge_priority;

  // Special token IDs
  int64_t bos_id = 2;
  int64_t eos_id = 1;
  int64_t pad_id = 0;
  int64_t unk_id = 3;

  // Set of special tokens to skip during decode
  std::unordered_set<int64_t> special_token_ids;

  void
  LoadFromJson (const std::string &path)
  {
    std::ifstream file (path);
    if (!file.is_open ())
      {
        throw std::runtime_error ("Cannot open tokenizer file: " + path);
      }

    nlohmann::json j;
    try
      {
        file >> j;
      }
    catch (const nlohmann::json::exception &e)
      {
        throw std::runtime_error ("Failed to parse tokenizer JSON: "
                                  + std::string (e.what ()));
      }

    // Load vocabulary from model.vocab
    if (j.contains ("model") && j["model"].contains ("vocab"))
      {
        const auto &vocab_obj = j["model"]["vocab"];
        for (auto it = vocab_obj.begin (); it != vocab_obj.end (); ++it)
          {
            const std::string &token = it.key ();
            const int64_t id = it.value ().get<int64_t> ();
            vocab[token] = id;
            id_to_token[id] = token;
          }
      }

    // Load merges from model.merges
    if (j.contains ("model") && j["model"].contains ("merges"))
      {
        const auto &merges = j["model"]["merges"];
        int priority = 0;
        for (const auto &merge : merges)
          {
            std::string left, right;

            if (merge.is_array () && merge.size () == 2)
              {
                // Array format: ["left", "right"]
                left = merge[0].get<std::string> ();
                right = merge[1].get<std::string> ();
              }
            else if (merge.is_string ())
              {
                // String format: "left right"
                const std::string merge_str = merge.get<std::string> ();
                auto space_pos = merge_str.find (' ');
                if (space_pos != std::string::npos)
                  {
                    left = merge_str.substr (0, space_pos);
                    right = merge_str.substr (space_pos + 1);
                  }
              }

            if (!left.empty () && !right.empty ())
              {
                merge_priority[{ left, right }] = priority++;
              }
          }
      }

    // Load added tokens (for special tokens)
    if (j.contains ("added_tokens"))
      {
        for (const auto &token : j["added_tokens"])
          {
            const int64_t id = token["id"].get<int64_t> ();
            const std::string content = token["content"].get<std::string> ();
            const bool is_special = token.value ("special", false);

            // Add to vocab if not present
            if (vocab.find (content) == vocab.end ())
              {
                vocab[content] = id;
                id_to_token[id] = content;
              }

            // Track special tokens
            if (is_special)
              {
                special_token_ids.insert (id);

                // Identify specific special tokens
                if (content == "<bos>")
                  bos_id = id;
                else if (content == "<eos>")
                  eos_id = id;
                else if (content == "<pad>")
                  pad_id = id;
                else if (content == "<unk>")
                  unk_id = id;
              }
          }
      }
  }

  /// @brief Normalize text (replace spaces with ▁).
  std::string
  Normalize (const std::string &text) const
  {
    // Add leading space indicator (SentencePiece convention)
    std::string normalized = kSpaceReplacement + text;
    // Replace remaining spaces with ▁
    normalized = ReplaceAll (normalized, " ", kSpaceReplacement);
    return normalized;
  }

  /// @brief Denormalize text (replace ▁ with spaces).
  std::string
  Denormalize (const std::string &text) const
  {
    std::string result = ReplaceAll (text, kSpaceReplacement, " ");
    // Remove leading space if present
    if (!result.empty () && result[0] == ' ')
      {
        result.erase (0, 1);
      }
    return result;
  }

  /// @brief Split text into initial BPE symbols (UTF-8 characters).
  std::vector<BPESymbol>
  InitializeSymbols (const std::string &text) const
  {
    std::vector<BPESymbol> symbols;

    // Split into UTF-8 characters
    std::size_t i = 0;
    while (i < text.size ())
      {
        std::size_t char_len = 1;
        unsigned char c = static_cast<unsigned char> (text[i]);
        if ((c & 0x80) == 0)
          char_len = 1; // ASCII
        else if ((c & 0xE0) == 0xC0)
          char_len = 2; // 2-byte UTF-8
        else if ((c & 0xF0) == 0xE0)
          char_len = 3; // 3-byte UTF-8
        else if ((c & 0xF8) == 0xF0)
          char_len = 4; // 4-byte UTF-8

        if (i + char_len > text.size ())
          char_len = text.size () - i;

        BPESymbol sym;
        sym.text = text.substr (i, char_len);
        sym.prev = symbols.empty () ? -1 : static_cast<int> (symbols.size () - 1);
        sym.next = -1;

        if (!symbols.empty ())
          {
            symbols.back ().next = static_cast<int> (symbols.size ());
          }

        symbols.push_back (std::move (sym));
        i += char_len;
      }

    return symbols;
  }

  /// @brief Encode text using BPE algorithm.
  std::vector<int64_t>
  EncodeBPE (const std::string &text) const
  {
    if (text.empty ())
      return {};

    // Initialize symbols
    std::vector<BPESymbol> symbols = InitializeSymbols (text);
    if (symbols.empty ())
      return {};

    // Priority queue for merge candidates
    std::priority_queue<MergeCandidate, std::vector<MergeCandidate>,
                        std::greater<MergeCandidate> >
        pq;

    // Populate initial merge candidates
    auto add_merge = [&] (int left_idx) {
      if (left_idx < 0)
        return;
      const BPESymbol &left = symbols[left_idx];
      if (left.next < 0 || left.text.empty ())
        return;

      const BPESymbol &right = symbols[left.next];
      if (right.text.empty ())
        return;

      auto it = merge_priority.find ({ left.text, right.text });
      if (it != merge_priority.end ())
        {
          pq.push ({ it->second, left_idx, left.next });
        }
    };

    for (std::size_t i = 0; i < symbols.size (); ++i)
      {
        add_merge (static_cast<int> (i));
      }

    // Apply merges until no more possible
    while (!pq.empty ())
      {
        MergeCandidate mc = pq.top ();
        pq.pop ();

        // Check if this merge is still valid
        BPESymbol &left = symbols[mc.left_idx];
        if (left.text.empty () || left.next != mc.right_idx)
          continue;

        BPESymbol &right = symbols[mc.right_idx];
        if (right.text.empty ())
          continue;

        // Check merge still has correct priority
        auto it = merge_priority.find ({ left.text, right.text });
        if (it == merge_priority.end () || it->second != mc.priority)
          continue;

        // Perform merge
        left.text += right.text;
        left.next = right.next;
        if (right.next >= 0)
          {
            symbols[right.next].prev = mc.left_idx;
          }
        right.text.clear (); // Mark as deleted

        // Add new merge candidates
        add_merge (left.prev);
        add_merge (mc.left_idx);
      }

    // Collect final tokens
    std::vector<int64_t> ids;
    for (const auto &sym : symbols)
      {
        if (sym.text.empty ())
          continue;

        auto it = vocab.find (sym.text);
        if (it != vocab.end ())
          {
            ids.push_back (it->second);
          }
        else
          {
            // Unknown token - split into bytes or use UNK
            ids.push_back (unk_id);
          }
      }

    return ids;
  }
};

GemmaTokenizer::GemmaTokenizer (const std::string &tokenizer_json_path)
    : impl_ (std::make_unique<Impl> ())
{
  impl_->LoadFromJson (tokenizer_json_path);
}

GemmaTokenizer::~GemmaTokenizer () = default;

GemmaTokenizer::GemmaTokenizer (GemmaTokenizer &&) noexcept = default;
GemmaTokenizer &GemmaTokenizer::operator= (GemmaTokenizer &&) noexcept
    = default;

std::vector<int64_t>
GemmaTokenizer::Encode (const std::string &text, bool add_bos) const
{
  std::string normalized = impl_->Normalize (text);
  std::vector<int64_t> ids = impl_->EncodeBPE (normalized);

  if (add_bos)
    {
      ids.insert (ids.begin (), impl_->bos_id);
    }

  return ids;
}

std::string
GemmaTokenizer::Decode (const std::vector<int64_t> &ids) const
{
  std::string result;
  for (int64_t id : ids)
    {
      // Skip special tokens
      if (impl_->special_token_ids.count (id))
        continue;

      auto it = impl_->id_to_token.find (id);
      if (it != impl_->id_to_token.end ())
        {
          result += it->second;
        }
    }
  return impl_->Denormalize (result);
}

std::size_t
GemmaTokenizer::VocabSize () const
{
  return impl_->vocab.size ();
}

int64_t
GemmaTokenizer::BosTokenId () const
{
  return impl_->bos_id;
}

int64_t
GemmaTokenizer::EosTokenId () const
{
  return impl_->eos_id;
}

int64_t
GemmaTokenizer::PadTokenId () const
{
  return impl_->pad_id;
}

int64_t
GemmaTokenizer::UnkTokenId () const
{
  return impl_->unk_id;
}

int64_t
GemmaTokenizer::TokenToId (const std::string &token) const
{
  auto it = impl_->vocab.find (token);
  return (it != impl_->vocab.end ()) ? it->second : impl_->unk_id;
}

std::string
GemmaTokenizer::IdToToken (int64_t id) const
{
  auto it = impl_->id_to_token.find (id);
  return (it != impl_->id_to_token.end ()) ? it->second : "";
}

} // namespace cortext
