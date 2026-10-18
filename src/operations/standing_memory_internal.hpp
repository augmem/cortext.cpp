#pragma once

// Standing memories: a small, bounded set of stored memories that state a
// lasting directive or preference for future behaviour ("Always cite the
// version when I ask about libraries", "From now on reply in French", "I
// prefer short answers"). A directive is stated once and must shape answers
// long after it scrolls out of query similarity, so queries draw on this set
// on its own: the best few members join every query packet, ranked within
// the set, whatever their rank among all memories.
//
// Membership is a property of the stored memory (table standing_memories,
// schema migration 51), not of its modality. The detector here is the one
// that exists today, a cheap pattern check over a memory's text content;
// memories without text are never members until another modality has a
// detector of its own.

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cortext::operations::standing_memory_internal
{

/// Members kept per store; the oldest member leaves when a new one joins a
/// full set.
inline constexpr std::size_t kCapacity = 256;

/// Members joined to each query packet.
inline constexpr std::size_t kPacketMembers = 5;

/// Longest memory text (bytes) the detector considers: a standing directive
/// is a short statement, while long turns that merely contain "always" are
/// explanations or code.
inline constexpr std::size_t kMaxTextBytes = 480;

/// True when the text is short and one of its clauses opens with a
/// directive or preference marker ("always", "never", "from now on",
/// "going forward", "whenever I", "I prefer", ...), optionally after
/// "please". Clauses start at the text start or after : . ! ? ; or a line
/// break. Case-insensitive ASCII.
bool IsStandingDirective (std::string_view text);

/// The text's distinct non-stopword words, lowercased and cut to their first
/// kStemBytes bytes (a crude stem: "assemble" and "assembly" match),
/// space-separated. Stored with each member.
std::string TermsOf (std::string_view text);

inline constexpr std::size_t kStemBytes = 6;

/// BM25 (k1 = 1.2, b = 0.75, document frequencies from the set itself) of
/// each member's stored terms against the query text. Members without a
/// shared term are left out; best first, ties by newer memory id.
std::vector<std::pair<long long, double> >
RankByTerms (std::string_view query_text,
             const std::vector<std::pair<long long, std::string> > &members);

} // namespace cortext::operations::standing_memory_internal
