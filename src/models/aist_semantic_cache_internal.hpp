#pragma once

// Byte-bounded LRU of text -> semantic embedding for the AIST runtime.
//
// The cache only saves re-encoding identical text (the empty maintenance
// signal Consolidate encodes, repeated queries). Benchmark streams repeat
// <0.6% of their texts, so an unbounded map grew ~8.6 KB of live heap per
// ingested turn for almost no hits. The bound keeps repeat hits for recent
// text while holding resident memory flat.

#include <cstddef>
#include <list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cortext::internal
{

class AistSemanticCache
{
public:
  /// Resident budget for keys + embeddings (~200 short texts at 256-d).
  static constexpr std::size_t kDefaultCapacityBytes = 256U * 1024U;
  /// Bookkeeping charged per entry (list node, map node, two headers).
  static constexpr std::size_t kEntryOverheadBytes = 128U;

  explicit AistSemanticCache (std::size_t capacity_bytes
                              = kDefaultCapacityBytes)
      : capacity_bytes_ (capacity_bytes)
  {
  }

  /// @brief Embedding cached for @p text, refreshed as most recent; nullptr
  /// on a miss.
  const std::vector<float> *
  Find (const std::string &text)
  {
    const auto it = index_.find (text);
    if (it == index_.end ())
      {
        return nullptr;
      }
    entries_.splice (entries_.begin (), entries_, it->second);
    return &it->second->second;
  }

  /// @brief Insert (or replace) @p text and evict least recent entries until
  /// the byte budget holds. An entry larger than the whole budget is not kept.
  void
  Put (const std::string &text, std::vector<float> embedding)
  {
    const auto existing = index_.find (text);
    if (existing != index_.end ())
      {
        bytes_ -= EntryBytes (existing->second->first,
                              existing->second->second);
        entries_.erase (existing->second);
        index_.erase (existing);
      }
    const std::size_t bytes = EntryBytes (text, embedding);
    if (bytes > capacity_bytes_)
      {
        return;
      }
    entries_.emplace_front (text, std::move (embedding));
    index_.emplace (text, entries_.begin ());
    bytes_ += bytes;
    while (bytes_ > capacity_bytes_)
      {
        const auto &last = entries_.back ();
        bytes_ -= EntryBytes (last.first, last.second);
        index_.erase (last.first);
        entries_.pop_back ();
      }
  }

  std::size_t
  Size () const
  {
    return entries_.size ();
  }

  std::size_t
  Bytes () const
  {
    return bytes_;
  }

  std::size_t
  CapacityBytes () const
  {
    return capacity_bytes_;
  }

  static std::size_t
  EntryBytes (const std::string &text, const std::vector<float> &embedding)
  {
    // The key is held twice: in the LRU list and in the index.
    return 2U * text.size () + embedding.size () * sizeof (float)
           + kEntryOverheadBytes;
  }

private:
  using Entry = std::pair<std::string, std::vector<float>>;

  std::size_t capacity_bytes_;
  std::size_t bytes_ = 0;
  std::list<Entry> entries_;
  std::unordered_map<std::string, std::list<Entry>::iterator> index_;
};

} // namespace cortext::internal
