/// @file ingress_anchor_formation_bench.cpp
/// @brief ES-AIST chronological ingress-only anchor formation benchmark.
///
/// This benchmark intentionally has no non-model/oracle mode. It runs ES-AIST
/// on each incoming signal, updates shadow soft-anchor state before retrieval,
/// and audits whether later references reuse already-established anchors.

#include "cortext/models/aist_gguf_encoder.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using Json = nlohmann::json;

struct Options
{
  int episodes = 200;
  int max_conversations = 96;
  int max_turns_per_conversation = 96;
  int max_cases = 421;
  bool synthetic_chain = false;
  std::filesystem::path models_dir = "models";
  std::filesystem::path teacher_model;
  std::filesystem::path output_dir =
      "build/ingress_storage_attention_es_aist";
};

struct Relation
{
  std::string type;
  std::string source;
  std::string target;
};

struct IngressStep
{
  std::string episode_id;
  int step = 0;
  std::string family;
  std::string modality = "text";
  std::string text;
  std::vector<std::string> entities;
  std::vector<std::string> objects;
  std::vector<Relation> relations;
  bool no_anchor = false;
};

struct EncodedStep
{
  std::vector<float> semantic_key;
  std::vector<float> entity_key;
  std::vector<float> full_key;
  double latency_ms = 0.0;
};

struct Anchor
{
  std::string anchor_id;
  std::string episode_id;
  std::string kind;
  std::vector<float> key;
  int first_seen_step = 0;
  int last_seen_step = 0;
  int support_count = 0;
  std::set<std::string> eval_keys;
};

struct Match
{
  int index = -1;
  double score = -1.0;
  double margin = 0.0;
};

struct Decision
{
  std::string action = "none";
  std::string anchor_id;
  std::vector<std::string> eval_keys_before;
  double score = -1.0;
  double margin = 0.0;
  bool updated_existing = false;
};

struct Policy
{
  std::string name;
  double entity_update_threshold = 0.55;
  double object_update_threshold = 0.55;
  double margin_threshold = 0.02;
  bool use_margin = false;
};

struct StepRow
{
  std::string policy;
  IngressStep step;
  Decision entity_decision;
  Decision object_decision;
  bool success = false;
  bool false_existing_update = false;
  bool relation_success = false;
  int entity_anchor_count_before = 0;
  int object_anchor_count_before = 0;
};

struct PolicyRun
{
  Policy policy;
  std::vector<StepRow> rows;
  double mean_encode_ms = 0.0;
  double p95_encode_ms = 0.0;
};

struct Conversation
{
  std::string dataset;
  std::string conversation_id;
  std::vector<std::string> messages;
};

struct StorageEvalCase
{
  std::string case_id;
  std::string dataset;
  std::string conversation_id;
  std::string label_class;
  bool is_reference = false;
  bool is_baseline_missing = false;
  int query_turn = -1;
  int target_turn = -1;
  int target_distance = -1;
  int wrong_active_turn = -1;
  int stale_turn = -1;
};

struct StoredMemory
{
  std::string memory_id;
  int turn = -1;
  std::string text;
  EncodedStep views;
  std::string anchor_id;
};

struct StorageAnchor
{
  std::string anchor_id;
  std::vector<float> entity_state;
  std::vector<float> full_state;
  int first_seen_turn = -1;
  int last_seen_turn = -1;
  std::string first_text;
  std::string last_text;
  int support_count = 0;
};

struct StorageAttentionPolicy
{
  std::string name;
  double temperature = 0.08;
  double update_threshold = 0.72;
  double margin_threshold = 0.02;
  double wm_bias = 0.035;
  double stm_bias = 0.020;
  double ltm_bias = 0.000;
  bool store_unlinked_on_no_commit = false;
};

struct StorageAttentionDecision
{
  std::string action = "create_anchor";
  std::string selected_memory_id;
  std::string selected_anchor_id;
  std::string selected_tier = "none";
  double selected_score = -1.0;
  double selected_attention = 0.0;
  double margin = 0.0;
  double entropy = 1.0;
  double wm_mass = 0.0;
  double stm_mass = 0.0;
  double ltm_mass = 0.0;
  int wm_count = 0;
  int stm_count = 0;
  int ltm_count = 0;
};

struct StorageAttentionCaseRow
{
  std::string policy;
  StorageEvalCase meta;
  std::string current_signal_text;
  StorageAttentionDecision decision;
  std::string target_anchor_id;
  std::string wrong_active_anchor_id;
  std::string stale_anchor_id;
  bool selected_target = false;
  bool selected_wrong_active = false;
  bool selected_stale = false;
  bool no_anchor_abstained = false;
  bool target_anchor_existed = false;
  bool wrong_active_distinct = false;
  bool stale_inactive = false;
};

struct StorageAttentionLinkRow
{
  std::string policy;
  std::string dataset;
  std::string conversation_id;
  int turn = -1;
  std::string memory_id;
  std::string action;
  std::string anchor_id;
  std::string selected_memory_id;
  std::string selected_tier;
  double selected_score = 0.0;
  double margin = 0.0;
  double wm_mass = 0.0;
  double stm_mass = 0.0;
  double ltm_mass = 0.0;
};

struct StorageAttentionRun
{
  StorageAttentionPolicy policy;
  std::vector<StorageAttentionCaseRow> cases;
  std::vector<StorageAttentionLinkRow> links;
  double mean_encode_ms = 0.0;
  double p95_encode_ms = 0.0;
};

struct AdaptiveAnchor
{
  std::string anchor_id;
  std::vector<float> semantic_state;
  std::vector<float> entity_state;
  std::vector<float> full_state;
  int first_seen_turn = -1;
  int last_seen_turn = -1;
  std::string first_text;
  std::string last_text;
  int support_count = 0;
  int contradiction_count = 0;
  double confidence = 0.45;
  double stability = 0.0;
  double anchor_strength = 0.0;
  std::string anchor_label = "tentative";
  std::string status = "provisional";
};

struct AdaptiveAnchorPolicy
{
  std::string name;
  double attention_temperature = 0.08;
  double update_threshold = 0.64;
  double margin_threshold = 0.02;
  double max_entropy = 0.98;
  double event_mismatch_threshold = 0.18;
  double contradiction_penalty = 0.05;
  int active_support_count = 2;
  int active_ttl = 32;
  bool split_on_event_mismatch = false;
  double label_durable_strength_threshold = 0.78;
  double label_durable_state_strength_threshold = 0.72;
  double label_durable_margin_threshold = 0.04;
  double label_durable_max_entropy = 0.82;
  int label_durable_support_count = 3;
  bool label_demote_generic_to_tentative = false;
  bool label_block_event_mismatch_durable = false;
  double semantic_weight = 0.18;
  double entity_weight = 0.44;
  double full_weight = 0.18;
  double wm_bias = 0.020;
  double stm_bias = 0.012;
  double ltm_bias = 0.000;
  bool include_wm = true;
  bool include_stm = true;
  bool include_ltm = true;
  bool generic_action_suppressor = false;
  bool null_no_anchor_hypothesis = false;
  bool use_contradiction_penalty = true;
  int min_update_support = 0;
  double null_score_threshold = 0.62;
  double null_entropy_threshold = 0.94;
  bool soft_hypothesis_mode = false;
  double posterior_beta = 8.0;
  double generic_hard_threshold = 0.78;
  double new_anchor_threshold = 0.46;
  double posterior_keep_threshold = 0.035;
  double tentative_score_threshold = 0.38;
  double soft_update_posterior_threshold = 0.62;
  double focus = -1.0;
  double sensitivity = -1.0;
  double stability_knob = -1.0;
};

struct AdaptiveAnchorCandidate
{
  std::string anchor_id;
  std::string tier = "none";
  int last_seen_turn = -1;
  std::string last_text;
  double score = -1.0;
  double attention = 0.0;
  double entity_cosine = -1.0;
  double semantic_cosine = -1.0;
  double event_cosine = -1.0;
  double confidence = 0.0;
  int support_count = 0;
  int contradiction_count = 0;
  double anchor_strength = 0.0;
  std::string anchor_label = "none";
  bool retained_soft_link = false;
};

struct AdaptiveAnchorDecision
{
  std::string action = "create_anchor";
  std::string selected_anchor_id;
  std::string selected_tier = "none";
  double selected_score = -1.0;
  double selected_attention = 0.0;
  double margin = 0.0;
  double entropy = 1.0;
  double entity_cosine = -1.0;
  double semantic_cosine = -1.0;
  double event_cosine = -1.0;
  double split_pressure = 0.0;
  bool generic_signal = false;
  double generic_score = 0.0;
  double p_none = 0.0;
  double p_new = 0.0;
  double p_top_real = 0.0;
  double selected_anchor_strength = 0.0;
  std::string selected_anchor_label = "none";
  double confidence_before = 0.0;
  int support_before = 0;
  int contradiction_before = 0;
  double wm_mass = 0.0;
  double stm_mass = 0.0;
  double ltm_mass = 0.0;
  int wm_count = 0;
  int stm_count = 0;
  int ltm_count = 0;
  std::vector<AdaptiveAnchorCandidate> candidates;
};

struct AdaptiveAnchorCaseRow
{
  std::string policy;
  StorageEvalCase meta;
  std::string current_signal_text;
  AdaptiveAnchorDecision decision;
  std::string target_anchor_id;
  std::string wrong_active_anchor_id;
  std::string stale_anchor_id;
  bool selected_target = false;
  bool selected_wrong_active = false;
  bool selected_stale = false;
  bool no_anchor_abstained = false;
  bool target_anchor_existed = false;
  bool wrong_active_distinct = false;
  bool stale_inactive = false;
  int target_candidate_rank = 0;
  int wrong_active_candidate_rank = 0;
  int stale_candidate_rank = 0;
  bool target_in_top2 = false;
  bool target_in_top3 = false;
  bool wrong_active_in_top3 = false;
  bool hard_commit = false;
  bool hard_wrong_commit = false;
  bool useful_uncertain_reference = false;
  double selected_anchor_strength = 0.0;
  std::string selected_anchor_label = "none";
};

struct AdaptiveAnchorLinkRow
{
  std::string policy;
  std::string dataset;
  std::string conversation_id;
  int turn = -1;
  std::string memory_id;
  std::string action;
  std::string anchor_id;
  std::string selected_tier;
  double selected_score = 0.0;
  double margin = 0.0;
  double entropy = 1.0;
  double p_none = 0.0;
  double p_new = 0.0;
  double p_top_real = 0.0;
  double generic_score = 0.0;
  double split_pressure = 0.0;
  double anchor_strength = 0.0;
  std::string anchor_label = "none";
  double confidence_after = 0.0;
  int support_after = 0;
  int contradiction_after = 0;
  std::string status_after;
};

struct AdaptiveAnchorSoftLinkRow
{
  std::string policy;
  std::string dataset;
  std::string conversation_id;
  int turn = -1;
  std::string memory_id;
  std::string anchor_id;
  std::string tier;
  int rank = 0;
  double score = 0.0;
  double posterior = 0.0;
  double margin = 0.0;
  double entropy = 1.0;
  double p_none = 0.0;
  double p_new = 0.0;
  double p_top_real = 0.0;
  double generic_score = 0.0;
  double anchor_strength = 0.0;
  std::string anchor_label = "none";
  double entity_cosine = 0.0;
  double semantic_cosine = 0.0;
  double event_cosine = 0.0;
};

struct AdaptiveAnchorStateRow
{
  std::string policy;
  std::string dataset;
  std::string conversation_id;
  std::string anchor_id;
  int first_seen_turn = -1;
  int last_seen_turn = -1;
  int support_count = 0;
  int contradiction_count = 0;
  double confidence = 0.0;
  double stability = 0.0;
  double anchor_strength = 0.0;
  std::string anchor_label = "none";
  std::string status;
};

struct AdaptiveAnchorRun
{
  AdaptiveAnchorPolicy policy;
  std::vector<AdaptiveAnchorCaseRow> cases;
  std::vector<AdaptiveAnchorLinkRow> links;
  std::vector<AdaptiveAnchorSoftLinkRow> soft_links;
  std::vector<AdaptiveAnchorStateRow> states;
  double mean_encode_ms = 0.0;
  double p95_encode_ms = 0.0;
};

struct SoftAnchorConsumptionPolicy
{
  std::string name;
  std::string formation_policy;
  int max_links = 3;
  bool allow_tentative = true;
  bool allow_ambiguous = true;
  bool allow_durable = true;
  bool allow_wm = true;
  bool allow_stm = true;
  bool allow_ltm = true;
  double min_strength = 0.04;
  double min_score = 0.50;
  double min_posterior = 0.02;
  double min_margin = -1.0;
  double max_entropy = 1.0;
  double max_generic = 1.0;
  double focus = -1.0;
  double sensitivity = -1.0;
  double stability = -1.0;
};

struct SoftAnchorConsumptionSurfaceRow
{
  std::string consumer_policy;
  std::string formation_policy;
  StorageEvalCase meta;
  std::string current_signal_text;
  int surface_rank = 0;
  std::string anchor_id;
  std::string tier;
  double score = 0.0;
  double posterior = 0.0;
  double anchor_strength = 0.0;
  std::string anchor_label = "none";
  double margin = 0.0;
  double entropy = 1.0;
  double generic_score = 0.0;
  bool is_target = false;
  bool is_wrong_active = false;
  bool is_stale = false;
  int estimated_context_chars = 0;
};

struct SoftAnchorConsumptionCaseRow
{
  std::string consumer_policy;
  std::string formation_policy;
  StorageEvalCase meta;
  std::string current_signal_text;
  int surfaced_link_count = 0;
  int estimated_context_chars = 0;
  std::string top_anchor_id;
  std::string top_anchor_label = "none";
  std::string top_tier = "none";
  double top_strength = 0.0;
  double top_score = -1.0;
  double top_posterior = 0.0;
  double margin = 0.0;
  double entropy = 1.0;
  double generic_score = 0.0;
  bool surfaced_any = false;
  bool surfaced_target = false;
  bool surfaced_target_top1 = false;
  bool surfaced_wrong_active = false;
  bool surfaced_stale = false;
  bool surfaced_wrong_without_target = false;
  bool surfaced_stale_without_target = false;
  bool no_anchor_surfaced = false;
  bool useful_context = false;
  bool harmful_context = false;
  bool ambiguous_context = false;
};

struct SoftAnchorConsumptionRun
{
  SoftAnchorConsumptionPolicy policy;
  std::vector<SoftAnchorConsumptionCaseRow> cases;
  std::vector<SoftAnchorConsumptionSurfaceRow> surfaces;
};

std::string
CsvEscape (const std::string &value)
{
  if (value.find_first_of (",\"\n") == std::string::npos)
    {
      return value;
    }
  std::string out = "\"";
  for (char ch : value)
    {
      if (ch == '"')
        {
          out += "\"\"";
        }
      else
        {
          out += ch;
        }
    }
  out += '"';
  return out;
}

std::string
Join (const std::vector<std::string> &values, const std::string &sep = " ")
{
  std::ostringstream out;
  for (std::size_t i = 0; i < values.size (); ++i)
    {
      if (i != 0)
        {
          out << sep;
        }
      out << values[i];
    }
  return out.str ();
}

std::vector<std::string>
SetToVector (const std::set<std::string> &values)
{
  return std::vector<std::string> (values.begin (), values.end ());
}

std::string
TrimWhitespace (const std::string &input)
{
  std::size_t begin = 0;
  while (begin < input.size ()
         && std::isspace (static_cast<unsigned char> (input[begin])) != 0)
    {
      ++begin;
    }
  std::size_t end = input.size ();
  while (end > begin
         && std::isspace (static_cast<unsigned char> (input[end - 1])) != 0)
    {
      --end;
    }
  return input.substr (begin, end - begin);
}

double
Cosine (const std::vector<float> &lhs, const std::vector<float> &rhs)
{
  if (lhs.empty () || lhs.size () != rhs.size ())
    {
      return -1.0;
    }
  double dot = 0.0;
  double ln = 0.0;
  double rn = 0.0;
  for (std::size_t i = 0; i < lhs.size (); ++i)
    {
      dot += static_cast<double> (lhs[i]) * static_cast<double> (rhs[i]);
      ln += static_cast<double> (lhs[i]) * static_cast<double> (lhs[i]);
      rn += static_cast<double> (rhs[i]) * static_cast<double> (rhs[i]);
    }
  if (ln <= 0.0 || rn <= 0.0)
    {
      return -1.0;
    }
  return dot / std::sqrt (ln * rn);
}

double
Clamp01 (double value)
{
  return std::max (0.0, std::min (1.0, value));
}

double
MapCosine01 (double cosine)
{
  return Clamp01 ((cosine + 1.0) * 0.5);
}

std::string
LowerAlphaNumSpace (const std::string &text)
{
  std::string out;
  out.reserve (text.size ());
  bool last_space = true;
  for (unsigned char ch : text)
    {
      if (std::isalnum (ch))
        {
          out.push_back (static_cast<char> (std::tolower (ch)));
          last_space = false;
        }
      else if (!last_space)
        {
          out.push_back (' ');
          last_space = true;
        }
    }
  if (!out.empty () && out.back () == ' ')
    {
      out.pop_back ();
    }
  return out;
}

bool
IsGenericAnchorSignal (const std::string &text)
{
  const std::string normalized = LowerAlphaNumSpace (text);
  if (normalized.empty ())
    {
      return true;
    }
  static const std::unordered_set<std::string> kGeneric = {
    "ok",       "okay",      "k",          "yeah",       "yea",
    "yep",      "yes",       "no",         "nope",       "sure",
    "cool",     "nice",      "great",      "awesome",    "wow",
    "lol",      "haha",      "thanks",     "thank you",  "bye",
    "goodbye",  "see you",   "got it",     "sounds good", "that s ok",
    "thats ok", "that is ok", "alright",   "right",      "true",
  };
  if (kGeneric.find (normalized) != kGeneric.end ())
    {
      return true;
    }
  int token_count = 0;
  std::istringstream in (normalized);
  std::string token;
  while (in >> token)
    {
      ++token_count;
    }
  if (token_count <= 2)
    {
      return normalized.find ("ok") != std::string::npos
             || normalized.find ("bye") != std::string::npos
             || normalized.find ("thank") != std::string::npos;
    }
  return false;
}

double
AnchorSupportStrength (int support_count)
{
  return Clamp01 (std::log1p (static_cast<double> (support_count))
                  / std::log (9.0));
}

double
AnchorContradictionStrength (int contradiction_count, int support_count)
{
  return Clamp01 (
      static_cast<double> (contradiction_count)
      / static_cast<double> (std::max (1, support_count)));
}

double
CandidateAnchorStrength (double score, double attention, double margin,
                         double entropy, double confidence,
                         int support_count, int contradiction_count)
{
  const double score_term = Clamp01 ((score - 0.45) / 0.45);
  const double margin_term = Clamp01 (margin / 0.12);
  const double support = AnchorSupportStrength (support_count);
  const double contradiction =
      AnchorContradictionStrength (contradiction_count, support_count);
  return Clamp01 (0.54 * score_term + 0.16 * Clamp01 (attention)
                  + 0.10 * margin_term + 0.08 * Clamp01 (confidence)
                  + 0.08 * support + 0.08 * (1.0 - Clamp01 (entropy))
                  - 0.12 * contradiction);
}

double
SoftCandidateAnchorStrength (double score, double posterior, double margin,
                             double entropy, double confidence,
                             int support_count, int contradiction_count)
{
  const double score_term = Clamp01 ((score - 0.60) / 0.35);
  const double margin_term = Clamp01 (margin / 0.30);
  const double support = AnchorSupportStrength (support_count);
  const double contradiction =
      AnchorContradictionStrength (contradiction_count, support_count);
  return Clamp01 (0.22 * score_term + 0.30 * Clamp01 (posterior)
                  + 0.20 * margin_term + 0.08 * Clamp01 (confidence)
                  + 0.08 * support + 0.12 * (1.0 - Clamp01 (entropy))
                  - 0.18 * contradiction);
}

double
StateAnchorStrength (double confidence, double stability, int support_count,
                     int contradiction_count)
{
  const double support = AnchorSupportStrength (support_count);
  const double contradiction =
      AnchorContradictionStrength (contradiction_count, support_count);
  return Clamp01 (0.48 * Clamp01 (confidence) + 0.22 * Clamp01 (stability)
                  + 0.24 * support - 0.16 * contradiction);
}

std::string
AnchorLabel (double anchor_strength, double margin, double entropy,
             int support_count, int contradiction_count,
             const std::string &status, const AdaptiveAnchorPolicy &policy,
             bool generic_signal, bool event_mismatch)
{
  if (status == "closed" || status == "decayed")
    {
      return "decayed";
    }
  if (contradiction_count > support_count && anchor_strength < 0.25)
    {
      return "rejected";
    }
  if (anchor_strength < 0.10)
    {
      return "none";
    }
  const bool can_be_durable =
      anchor_strength >= policy.label_durable_strength_threshold
      && support_count >= policy.label_durable_support_count
      && contradiction_count == 0
      && margin >= policy.label_durable_margin_threshold
      && entropy <= policy.label_durable_max_entropy
      && !(policy.label_demote_generic_to_tentative && generic_signal)
      && !(policy.label_block_event_mismatch_durable && event_mismatch);
  if (can_be_durable)
    {
      return "durable";
    }
  if (entropy >= 0.92 || margin < 0.025 || contradiction_count > 0)
    {
      return "ambiguous";
    }
  return "tentative";
}

std::string
AnchorStateLabel (double anchor_strength, int support_count,
                  int contradiction_count, const std::string &status,
                  const AdaptiveAnchorPolicy &policy)
{
  if (status == "closed" || status == "decayed")
    {
      return "decayed";
    }
  if (contradiction_count > support_count && anchor_strength < 0.25)
    {
      return "rejected";
    }
  if (anchor_strength < 0.10)
    {
      return "none";
    }
  if (anchor_strength >= policy.label_durable_state_strength_threshold
      && support_count >= policy.label_durable_support_count
      && contradiction_count == 0)
    {
      return "durable";
    }
  if (contradiction_count > 0)
    {
      return "ambiguous";
    }
  return "tentative";
}

std::vector<float>
Blend (const std::vector<float> &old_key, const std::vector<float> &new_key,
       int support_count)
{
  if (old_key.empty () || old_key.size () != new_key.size ())
    {
      return new_key;
    }
  const float old_weight =
      static_cast<float> (std::max (1, support_count));
  std::vector<float> out (old_key.size (), 0.0F);
  double norm = 0.0;
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      out[i] = (old_weight * old_key[i] + new_key[i]) / (old_weight + 1.0F);
      norm += static_cast<double> (out[i]) * static_cast<double> (out[i]);
    }
  if (norm > 0.0)
    {
      const float inv = static_cast<float> (1.0 / std::sqrt (norm));
      for (float &value : out)
        {
          value *= inv;
        }
    }
  return out;
}

std::vector<float>
SoftBlend (const std::vector<float> &old_key, const std::vector<float> &new_key,
           double alpha)
{
  if (old_key.empty () || old_key.size () != new_key.size ())
    {
      return new_key;
    }
  const double clamped_alpha = Clamp01 (alpha);
  std::vector<float> out (old_key.size (), 0.0F);
  double norm = 0.0;
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      out[i] = static_cast<float> ((1.0 - clamped_alpha) * old_key[i]
                                   + clamped_alpha * new_key[i]);
      norm += static_cast<double> (out[i]) * static_cast<double> (out[i]);
    }
  if (norm > 0.0)
    {
      const float inv = static_cast<float> (1.0 / std::sqrt (norm));
      for (float &value : out)
        {
          value *= inv;
        }
    }
  return out;
}

double
Mean (std::vector<double> values)
{
  if (values.empty ())
    {
      return 0.0;
    }
  double sum = 0.0;
  for (double value : values)
    {
      sum += value;
    }
  return sum / static_cast<double> (values.size ());
}

double
Percentile (std::vector<double> values, double p)
{
  if (values.empty ())
    {
      return 0.0;
    }
  std::sort (values.begin (), values.end ());
  const std::size_t index = static_cast<std::size_t> (std::floor (
      p * static_cast<double> (values.size () - 1)));
  return values[index];
}

std::vector<IngressStep>
MakeEpisode (int index)
{
  std::string episode = "episode_" + std::to_string (index);
  std::string p = episode + ":";
  std::string jared = p + "person:jared";
  std::string alex = p + "person:alex";
  std::string maya = p + "person:maya";
  std::string image = p + "media:dog_image";
  std::string dog = p + "object:jared_dog";
  std::string work_doc = p + "object:work_doc";
  std::string pay_stubs = p + "object:pay_stubs";
  return {
    { episode, 0, "explicit_entity_create", "text",
      "Hey, I just got off the phone with Jared.", { jared }, {}, {}, false },
    { episode, 1, "person_reference", "text", "Great, what did he say?",
      { jared }, {}, {}, false },
    { episode, 2, "cross_modal_person_media_object", "image",
      "He sent me this blurry image of his dog.", { jared }, { image, dog },
      { { "sender", jared, image }, { "depicts", image, dog },
        { "owner", jared, dog } },
      false },
    { episode, 3, "object_media_reference", "text",
      "Can you send it to me?", {}, { image }, {}, false },
    { episode, 4, "media_continuation", "text",
      "Sure, it is kind of blurry but here you go.", {}, { image }, {},
      false },
    { episode, 5, "person_object_relation_reference", "text",
      "How long has he had it?", { jared }, { dog },
      { { "owner", jared, dog } }, false },
    { episode, 6, "no_anchor_topic_shift", "text",
      "The dishwasher broke this morning.", {}, {}, {}, true },
    { episode, 7, "wrong_active_intro", "text",
      "Alex is Gabe's coworker and is active on the work thread.", { alex },
      {}, {}, false },
    { episode, 8, "wrong_active_reference", "text",
      "He sent the work document.", { alex }, { work_doc },
      { { "sender", alex, work_doc } }, false },
    { episode, 9, "stale_same_source_intro", "text",
      "Maya is discussing housing paperwork now.", { maya }, {}, {}, false },
    { episode, 10, "stale_same_source_reference", "text",
      "She needs the pay stubs.", { maya }, { pay_stubs },
      { { "mentions", maya, pay_stubs } }, false },
  };
}

std::vector<IngressStep>
MakeSteps (int episodes)
{
  std::vector<IngressStep> steps;
  for (int i = 0; i < episodes; ++i)
    {
      auto episode = MakeEpisode (i);
      steps.insert (steps.end (), episode.begin (), episode.end ());
    }
  return steps;
}

std::string
ConversationKey (const std::string &dataset, const std::string &conversation_id)
{
  return dataset + "\t" + conversation_id;
}

std::string
ConversationKey (const Conversation &conv)
{
  return ConversationKey (conv.dataset, conv.conversation_id);
}

std::string
StorageMemoryId (const Conversation &conv, int turn)
{
  return conv.dataset + ":" + conv.conversation_id + ":turn_"
         + std::to_string (turn);
}

std::vector<Conversation>
LoadConversationsFromJsonl (const std::filesystem::path &path,
                            const std::string &dataset,
                            int max_conversations)
{
  std::ifstream in (path);
  if (!in)
    {
      return {};
    }
  std::vector<Conversation> conversations;
  std::string line;
  while (std::getline (in, line))
    {
      if (max_conversations > 0
          && static_cast<int> (conversations.size ()) >= max_conversations)
        {
          break;
        }
      if (TrimWhitespace (line).empty ())
        {
          continue;
        }
      Json row = Json::parse (line);
      if (!row.is_array () || row.size () < 2 || !row[1].contains ("content"))
        {
          continue;
        }
      Conversation conv;
      conv.dataset = dataset;
      conv.conversation_id = row[0].is_string ()
                                 ? row[0].get<std::string> ()
                                 : dataset + "_"
                                       + std::to_string (conversations.size ());
      for (const auto &turn : row[1]["content"])
        {
          if (!turn.contains ("message") || !turn["message"].is_string ())
            {
              continue;
            }
          const std::string message =
              TrimWhitespace (turn["message"].get<std::string> ());
          if (!message.empty ())
            {
              conv.messages.push_back (message);
            }
        }
      if (conv.messages.size () >= 20)
        {
          conversations.push_back (std::move (conv));
        }
    }
  return conversations;
}

std::vector<Conversation>
LoadStorageConversations (const Options &opts)
{
  std::vector<Conversation> conversations;
  const int per_dataset =
      std::max (1, (opts.max_conversations + 2) / 3);
  const std::vector<std::pair<std::filesystem::path, std::string>> inputs = {
    { "data/personachat/valid.jsonl", "personachat" },
    { "data/topical_chat/valid_freq.jsonl", "topical_chat" },
    { "data/taskmaster/valid.jsonl", "taskmaster" },
  };
  for (const auto &[path, dataset] : inputs)
    {
      auto loaded = LoadConversationsFromJsonl (path, dataset, per_dataset);
      conversations.insert (conversations.end (),
                            std::make_move_iterator (loaded.begin ()),
                            std::make_move_iterator (loaded.end ()));
    }
  if (static_cast<int> (conversations.size ()) > opts.max_conversations)
    {
      conversations.resize (static_cast<std::size_t> (opts.max_conversations));
    }
  if (conversations.empty ())
    {
      throw std::runtime_error ("Real ingress storage benchmark inputs were not found");
    }
  return conversations;
}

std::vector<StorageEvalCase>
BuildStorageEvalCases (const std::vector<Conversation> &conversations,
                       const Options &opts)
{
  std::vector<StorageEvalCase> cases;
  std::set<std::string> used_steps;
  int refs = 0;
  int controls = 0;
  int cursor = 0;
  const int max_cases = opts.max_cases <= 0 ? 421 : opts.max_cases;
  const int target_refs = std::max (1, static_cast<int> (
      std::round (static_cast<double> (max_cases) * 270.0 / 421.0)));
  const int target_controls = std::max (1, max_cases - target_refs);
  while ((refs < target_refs || controls < target_controls)
         && cursor < static_cast<int> (conversations.size ()) * 8)
    {
      const auto &conv = conversations[static_cast<std::size_t> (
          cursor % static_cast<int> (conversations.size ()))];
      ++cursor;
      const int size = std::min<int> (
          static_cast<int> (conv.messages.size ()),
          opts.max_turns_per_conversation > 0
              ? opts.max_turns_per_conversation
              : static_cast<int> (conv.messages.size ()));
      if (size < 20)
        {
          continue;
        }
      if (refs < target_refs)
        {
          int distance = 1;
          if (refs < target_refs * 37 / 100)
            {
              distance = 1;
            }
          else if (refs < target_refs * 79 / 100)
            {
              distance = 2 + (refs % 3);
            }
          else
            {
              distance = 5 + (refs % 8);
            }
          const int min_query = std::max (distance + 1, 14);
          const int query_span = std::max (1, size - min_query);
          int query = min_query + ((refs * 7 + cursor) % query_span);
          bool found_unused = false;
          for (int attempt = 0; attempt < query_span; ++attempt)
            {
              query = min_query
                      + ((refs * 7 + cursor + attempt) % query_span);
              if (used_steps.count (ConversationKey (conv) + "#"
                                    + std::to_string (query))
                  == 0)
                {
                  found_unused = true;
                  break;
                }
            }
          if (!found_unused)
            {
              continue;
            }
          used_steps.insert (ConversationKey (conv) + "#"
                             + std::to_string (query));
          StorageEvalCase row;
          row.case_id = "storage_" + conv.dataset + "_" + conv.conversation_id
                        + "_ref_" + std::to_string (refs);
          row.dataset = conv.dataset;
          row.conversation_id = conv.conversation_id;
          row.label_class = "reference";
          row.is_reference = true;
          row.is_baseline_missing = refs >= target_refs * 26 / 100;
          row.query_turn = query;
          row.target_turn = query - distance;
          row.target_distance = distance;
          row.wrong_active_turn = std::max (0, query - 1);
          row.stale_turn = std::max (0, query - 13);
          cases.push_back (std::move (row));
          ++refs;
        }
      if (controls < target_controls && (refs % 2 == 0 || refs >= target_refs))
        {
          const int query_span = std::max (1, size - 14);
          int query = 14 + ((controls * 11 + cursor) % query_span);
          bool found_unused = false;
          for (int attempt = 0; attempt < query_span; ++attempt)
            {
              query = 14 + ((controls * 11 + cursor + attempt)
                            % query_span);
              if (used_steps.count (ConversationKey (conv) + "#"
                                    + std::to_string (query))
                  == 0)
                {
                  found_unused = true;
                  break;
                }
            }
          if (!found_unused)
            {
              continue;
            }
          used_steps.insert (ConversationKey (conv) + "#"
                             + std::to_string (query));
          StorageEvalCase row;
          row.case_id = "storage_" + conv.dataset + "_" + conv.conversation_id
                        + "_noanchor_" + std::to_string (controls);
          row.dataset = conv.dataset;
          row.conversation_id = conv.conversation_id;
          row.label_class = "no_anchor_control";
          row.is_reference = false;
          row.query_turn = query;
          row.target_turn = -1;
          row.target_distance = -1;
          row.wrong_active_turn = std::max (0, query - 1);
          row.stale_turn = std::max (0, query - 13);
          cases.push_back (std::move (row));
          ++controls;
        }
    }
  return cases;
}

std::optional<std::filesystem::path>
ResolveEsAistModelPath (const Options &opts)
{
  if (!opts.teacher_model.empty ())
    {
      if (std::filesystem::exists (opts.teacher_model))
        {
          return opts.teacher_model;
        }
      throw std::runtime_error ("ES-AIST model override does not exist: "
                                + opts.teacher_model.string ());
    }
  if (const char *env = std::getenv ("CORTEXT_ES_AIST_MODEL_PATH"))
    {
      std::filesystem::path path (env);
      if (std::filesystem::exists (path))
        {
          return path;
        }
      throw std::runtime_error ("CORTEXT_ES_AIST_MODEL_PATH does not exist: "
                                + path.string ());
    }
  const std::vector<std::filesystem::path> preferred{
    opts.models_dir / "ES-AIST-81M-preview-GGUF" / "ES-AIST-81M_q8_0.gguf",
    opts.models_dir / "ES-AIST-81M-preview-GGUF",
    opts.models_dir,
  };
  for (const auto &path : preferred)
    {
      if (!std::filesystem::exists (path))
        {
          continue;
        }
      if (std::filesystem::is_regular_file (path)
          && path.extension () == ".gguf")
        {
          return path;
        }
      if (std::filesystem::is_directory (path))
        {
          std::vector<std::filesystem::path> matches;
          for (const auto &entry : std::filesystem::directory_iterator (path))
            {
              if ((entry.is_regular_file ()
                   || std::filesystem::is_symlink (entry.symlink_status ()))
                  && entry.path ().extension () == ".gguf"
                  && entry.path ().filename ().string ().find ("ES-AIST")
                         != std::string::npos)
                {
                  matches.push_back (entry.path ());
                }
            }
          std::sort (matches.begin (), matches.end (),
                     [] (const auto &lhs, const auto &rhs) {
                       const bool lq8 =
                           lhs.filename ().string ().find ("q8_0")
                           != std::string::npos;
                       const bool rq8 =
                           rhs.filename ().string ().find ("q8_0")
                           != std::string::npos;
                       if (lq8 != rq8)
                         {
                           return lq8;
                         }
                       return lhs < rhs;
                     });
          if (!matches.empty ())
            {
              return matches.front ();
            }
        }
    }
  return std::nullopt;
}

EncodedStep
EncodeStep (cortext::AaitGgufEncoder &encoder,
            std::unordered_map<std::string, EncodedStep> &cache,
            const std::string &text)
{
  auto found = cache.find (text);
  if (found != cache.end ())
    {
      return found->second;
    }
  auto started = std::chrono::steady_clock::now ();
  const auto output = encoder.EncodeTextWithAnchors (text);
  const auto views = cortext::BuildEssAistEmbeddingViews (
      output.semantic_vector);
  EncodedStep encoded;
  encoded.semantic_key = views.semantic_768_key;
  encoded.entity_key = views.entity_key;
  encoded.full_key = views.full_key;
  encoded.latency_ms =
      std::chrono::duration<double, std::milli> (
          std::chrono::steady_clock::now () - started)
          .count ();
  auto inserted = cache.emplace (text, encoded);
  return inserted.first->second;
}

std::vector<double>
Softmax (const std::vector<double> &logits)
{
  if (logits.empty ())
    {
      return {};
    }
  const double max_logit = *std::max_element (logits.begin (), logits.end ());
  std::vector<double> out (logits.size (), 0.0);
  double denom = 0.0;
  for (std::size_t i = 0; i < logits.size (); ++i)
    {
      out[i] = std::exp (logits[i] - max_logit);
      denom += out[i];
    }
  if (denom <= 0.0)
    {
      return std::vector<double> (logits.size (),
                                  1.0 / static_cast<double> (logits.size ()));
    }
  for (double &value : out)
    {
      value /= denom;
    }
  return out;
}

double
NormalizedEntropy (const std::vector<double> &weights)
{
  if (weights.size () <= 1)
    {
      return 0.0;
    }
  double entropy = 0.0;
  for (double weight : weights)
    {
      if (weight > 0.0)
        {
          entropy -= weight * std::log (weight);
        }
    }
  return entropy / std::log (static_cast<double> (weights.size ()));
}

std::string
MemoryTier (int current_turn, int memory_turn)
{
  const int age = current_turn - memory_turn;
  if (age <= 4)
    {
      return "wm";
    }
  if (age <= 32)
    {
      return "stm";
    }
  return "ltm";
}

double
TierBias (const StorageAttentionPolicy &policy, const std::string &tier)
{
  if (tier == "wm")
    {
      return policy.wm_bias;
    }
  if (tier == "stm")
    {
      return policy.stm_bias;
    }
  return policy.ltm_bias;
}

StorageAttentionDecision
AttendStoredMemories (const std::vector<StoredMemory> &memories,
                      const EncodedStep &current, int current_turn,
                      const StorageAttentionPolicy &policy)
{
  StorageAttentionDecision decision;
  if (memories.empty ())
    {
      return decision;
    }

  std::vector<double> logits;
  std::vector<double> base_scores;
  logits.reserve (memories.size ());
  base_scores.reserve (memories.size ());
  for (const auto &memory : memories)
    {
      const int age = std::max (1, current_turn - memory.turn);
      const std::string tier = MemoryTier (current_turn, memory.turn);
      const double semantic = Cosine (current.semantic_key,
                                      memory.views.semantic_key);
      const double entity = Cosine (current.entity_key, memory.views.entity_key);
      const double full = Cosine (current.full_key, memory.views.full_key);
      const double recency = 1.0 / (1.0 + static_cast<double> (age));
      const double base = 0.20 * semantic + 0.46 * entity + 0.26 * full
                          + 0.08 * recency;
      base_scores.push_back (base);
      logits.push_back ((base + TierBias (policy, tier))
                        / std::max (policy.temperature, 1.0e-6));
    }

  const auto weights = Softmax (logits);
  decision.entropy = NormalizedEntropy (weights);
  int best = -1;
  double best_logit = -std::numeric_limits<double>::infinity ();
  double second_base = -std::numeric_limits<double>::infinity ();
  for (std::size_t i = 0; i < memories.size (); ++i)
    {
      const std::string tier = MemoryTier (current_turn, memories[i].turn);
      if (tier == "wm")
        {
          decision.wm_mass += weights[i];
          decision.wm_count += 1;
        }
      else if (tier == "stm")
        {
          decision.stm_mass += weights[i];
          decision.stm_count += 1;
        }
      else
        {
          decision.ltm_mass += weights[i];
          decision.ltm_count += 1;
        }
      if (logits[i] > best_logit)
        {
          if (best >= 0)
            {
              second_base = std::max (second_base,
                                      base_scores[static_cast<std::size_t> (
                                          best)]);
            }
          best_logit = logits[i];
          best = static_cast<int> (i);
        }
      else
        {
          second_base = std::max (second_base, base_scores[i]);
        }
    }

  if (best < 0)
    {
      return decision;
    }
  const auto &selected = memories[static_cast<std::size_t> (best)];
  decision.selected_memory_id = selected.memory_id;
  decision.selected_anchor_id = selected.anchor_id;
  decision.selected_tier = MemoryTier (current_turn, selected.turn);
  decision.selected_attention = weights[static_cast<std::size_t> (best)];
  decision.selected_score =
      0.76 * base_scores[static_cast<std::size_t> (best)]
      + 0.14 * decision.selected_attention + 0.10 * (1.0 - decision.entropy);
  decision.margin = std::isfinite (second_base)
                        ? base_scores[static_cast<std::size_t> (best)]
                              - second_base
                        : base_scores[static_cast<std::size_t> (best)];

  if (!selected.anchor_id.empty ()
      && decision.selected_score >= policy.update_threshold
      && decision.margin >= policy.margin_threshold)
    {
      decision.action = "update_existing";
    }
  else if (policy.store_unlinked_on_no_commit)
    {
      decision.action = "store_unlinked";
      decision.selected_anchor_id.clear ();
    }
  return decision;
}

StorageAnchor *
FindStorageAnchor (std::vector<StorageAnchor> &anchors,
                   const std::string &anchor_id)
{
  for (auto &anchor : anchors)
    {
      if (anchor.anchor_id == anchor_id)
        {
          return &anchor;
        }
    }
  return nullptr;
}

std::string
MemoryAnchorAtTurn (const std::vector<StoredMemory> &memories, int turn)
{
  for (const auto &memory : memories)
    {
      if (memory.turn == turn)
        {
          return memory.anchor_id;
        }
    }
  return "";
}

std::string
AdaptiveAnchorAtTurn (const std::map<int, std::string> &turn_anchor_ids,
                      int turn)
{
  const auto found = turn_anchor_ids.find (turn);
  return found == turn_anchor_ids.end () ? "" : found->second;
}

AdaptiveAnchor *
FindAdaptiveAnchor (std::vector<AdaptiveAnchor> &anchors,
                    const std::string &anchor_id)
{
  for (auto &anchor : anchors)
    {
      if (anchor.anchor_id == anchor_id)
        {
          return &anchor;
        }
    }
  return nullptr;
}

int
AdaptiveCandidateRank (const AdaptiveAnchorDecision &decision,
                       const std::string &anchor_id)
{
  if (anchor_id.empty ())
    {
      return 0;
    }
  for (std::size_t i = 0; i < decision.candidates.size (); ++i)
    {
      if (decision.candidates[i].anchor_id == anchor_id)
        {
          return static_cast<int> (i + 1);
        }
    }
  return 0;
}

double
AdaptiveTierBias (const AdaptiveAnchorPolicy &policy, const std::string &tier)
{
  if (tier == "wm")
    {
      return policy.wm_bias;
    }
  if (tier == "stm")
    {
      return policy.stm_bias;
    }
  return policy.ltm_bias;
}

bool
AdaptiveTierIncluded (const AdaptiveAnchorPolicy &policy,
                      const std::string &tier)
{
  if (tier == "wm")
    {
      return policy.include_wm;
    }
  if (tier == "stm")
    {
      return policy.include_stm;
    }
  return policy.include_ltm;
}

AdaptiveAnchorDecision
AttendAdaptiveAnchors (const std::vector<AdaptiveAnchor> &anchors,
                       const EncodedStep &current, int current_turn,
                       const std::string &current_text,
                       const AdaptiveAnchorPolicy &policy)
{
  AdaptiveAnchorDecision decision;
  decision.generic_signal = IsGenericAnchorSignal (current_text);
  if (anchors.empty ())
    {
      if (policy.generic_action_suppressor && decision.generic_signal)
        {
          decision.generic_score = 1.0;
          decision.p_none = 1.0;
          decision.action = "abstain";
        }
      else if (policy.soft_hypothesis_mode)
        {
          decision.p_new = 1.0;
        }
      return decision;
    }

  std::vector<int> valid_indices;
  std::vector<double> logits;
  std::vector<double> scores;
  std::vector<double> semantic_cosines;
  std::vector<double> entity_cosines;
  std::vector<double> event_cosines;
  std::vector<double> contradictions;
  for (std::size_t i = 0; i < anchors.size (); ++i)
    {
      const auto &anchor = anchors[i];
      if (anchor.status == "closed" || anchor.entity_state.empty ())
        {
          continue;
        }
      const int age = std::max (1, current_turn - anchor.last_seen_turn);
      if (age > policy.active_ttl)
        {
          continue;
        }
      const std::string tier = MemoryTier (current_turn, anchor.last_seen_turn);
      if (!AdaptiveTierIncluded (policy, tier))
        {
          continue;
        }
      const double semantic = Cosine (current.semantic_key,
                                      anchor.semantic_state);
      const double entity = Cosine (current.entity_key, anchor.entity_state);
      const double event = Cosine (current.full_key, anchor.full_state);
      const double recency = 1.0 / (1.0 + static_cast<double> (age));
      const double support = Clamp01 (
          std::log1p (static_cast<double> (anchor.support_count))
          / std::log (9.0));
      const double contradiction = Clamp01 (
          static_cast<double> (anchor.contradiction_count)
          / static_cast<double> (std::max (1, anchor.support_count)));
      const double event_mismatch = std::max (0.0, entity - event);
      const double semantic_score = policy.soft_hypothesis_mode
                                        ? MapCosine01 (semantic)
                                        : semantic;
      const double entity_score = policy.soft_hypothesis_mode
                                      ? MapCosine01 (entity)
                                      : entity;
      const double event_score = policy.soft_hypothesis_mode
                                     ? MapCosine01 (event)
                                     : event;
      double score = 0.0;
      if (policy.soft_hypothesis_mode)
        {
          const double view_weight = std::max (
              1.0e-6, policy.entity_weight + policy.semantic_weight
                          + policy.full_weight);
          const double view_score =
              (policy.entity_weight * entity_score
               + policy.semantic_weight * semantic_score
               + policy.full_weight * event_score)
              / view_weight;
          score = 0.78 * view_score + 0.05 * recency
                  + 0.04 * anchor.confidence + 0.03 * support
                  + AdaptiveTierBias (policy, tier)
                  - (policy.use_contradiction_penalty
                         ? policy.contradiction_penalty * contradiction
                         : 0.0)
                  - 0.06 * event_mismatch;
        }
      else
        {
          score = policy.entity_weight * entity_score
                  + policy.semantic_weight * semantic_score
                  + policy.full_weight * event_score
                  + 0.08 * recency + 0.06 * anchor.confidence
                  + 0.04 * support + AdaptiveTierBias (policy, tier)
                  - (policy.use_contradiction_penalty
                         ? policy.contradiction_penalty * contradiction
                         : 0.0)
                  - 0.04 * event_mismatch;
        }
      if (policy.soft_hypothesis_mode)
        {
          score = Clamp01 (score);
        }
      valid_indices.push_back (static_cast<int> (i));
      scores.push_back (score);
      semantic_cosines.push_back (semantic);
      entity_cosines.push_back (entity);
      event_cosines.push_back (event);
      contradictions.push_back (contradiction);
      logits.push_back (
          policy.soft_hypothesis_mode
              ? policy.posterior_beta * score
              : score / std::max (policy.attention_temperature, 1.0e-6));
    }

  if (valid_indices.empty ())
    {
      if (policy.generic_action_suppressor && decision.generic_signal)
        {
          decision.generic_score = 1.0;
          decision.p_none = 1.0;
          decision.action = "abstain";
        }
      else if (policy.soft_hypothesis_mode)
        {
          decision.p_new = 1.0;
        }
      return decision;
    }

  std::vector<double> weights = Softmax (logits);
  decision.entropy = NormalizedEntropy (weights);
  std::vector<double> real_posteriors = weights;
  double p_none = 0.0;
  double p_new = 0.0;
  int best_hypothesis = -1;
  if (policy.soft_hypothesis_mode)
    {
      const auto sorted_scores = [&] {
        std::vector<double> values = scores;
        std::sort (values.begin (), values.end ());
        return values;
      } ();
      const double best_real =
          scores.empty () ? 0.0 : *std::max_element (scores.begin (),
                                                     scores.end ());
      const double median_real =
          sorted_scores.empty ()
              ? 0.0
              : sorted_scores[sorted_scores.size () / 2];
      const double specificity = Clamp01 (best_real - median_real);
      const double real_entropy = NormalizedEntropy (weights);
      const double low_information =
          Clamp01 (0.40 * 0.50 + 0.35 * real_entropy
                   + 0.25 * (1.0 - specificity));
      decision.generic_score =
          policy.generic_action_suppressor && decision.generic_signal ? 1.0
                                                                      : 0.0;
      const auto best_it = std::max_element (scores.begin (), scores.end ());
      const std::size_t best_pos =
          best_it == scores.end ()
              ? 0
              : static_cast<std::size_t> (std::distance (scores.begin (),
                                                         best_it));
      const double top_contra =
          best_pos < contradictions.size () ? contradictions[best_pos] : 0.0;
      const double info = Clamp01 ((1.0 + 0.50 + 1.0) / 3.0)
                          * (1.0 - decision.generic_score);
      const double score_none =
          policy.null_no_anchor_hypothesis
              ? Clamp01 (0.25 + 0.55 * decision.generic_score
                         + 0.20 * low_information
                         + 0.20 * real_entropy + 0.20 * (1.0 - best_real)
                         + 0.15 * top_contra)
              : -1.0;
      const double score_new =
          Clamp01 (0.15 + 0.45 * info + 0.20 * (1.0 - best_real)
                   - 0.35 * decision.generic_score);
      std::vector<double> hypothesis_logits;
      hypothesis_logits.reserve (scores.size () + 2);
      for (double score : scores)
        {
          hypothesis_logits.push_back (policy.posterior_beta * score);
        }
      hypothesis_logits.push_back (policy.posterior_beta * score_none);
      hypothesis_logits.push_back (policy.posterior_beta * score_new);
      const auto posterior = Softmax (hypothesis_logits);
      real_posteriors.assign (posterior.begin (),
                              posterior.begin ()
                                  + static_cast<std::ptrdiff_t> (
                                      scores.size ()));
      p_none = posterior[posterior.size () - 2];
      p_new = posterior[posterior.size () - 1];
      decision.p_none = p_none;
      decision.p_new = p_new;
      decision.entropy = NormalizedEntropy (posterior);
      best_hypothesis = static_cast<int> (std::distance (
          posterior.begin (), std::max_element (posterior.begin (),
                                                posterior.end ())));
      weights = real_posteriors;
    }
  int best = -1;
  double best_score = -std::numeric_limits<double>::infinity ();
  double second_score = -std::numeric_limits<double>::infinity ();
  for (std::size_t pos = 0; pos < valid_indices.size (); ++pos)
    {
      const auto &anchor =
          anchors[static_cast<std::size_t> (valid_indices[pos])];
      const std::string tier = MemoryTier (current_turn, anchor.last_seen_turn);
      AdaptiveAnchorCandidate candidate;
      candidate.anchor_id = anchor.anchor_id;
      candidate.tier = tier;
      candidate.last_seen_turn = anchor.last_seen_turn;
      candidate.last_text = anchor.last_text;
      candidate.score = scores[pos];
      candidate.attention = weights[pos];
      candidate.semantic_cosine = semantic_cosines[pos];
      candidate.entity_cosine = entity_cosines[pos];
      candidate.event_cosine = event_cosines[pos];
      candidate.confidence = anchor.confidence;
      candidate.support_count = anchor.support_count;
      candidate.contradiction_count = anchor.contradiction_count;
      decision.candidates.push_back (std::move (candidate));
      if (tier == "wm")
        {
          decision.wm_mass += weights[pos];
          decision.wm_count += 1;
        }
      else if (tier == "stm")
        {
          decision.stm_mass += weights[pos];
          decision.stm_count += 1;
        }
      else
        {
          decision.ltm_mass += weights[pos];
          decision.ltm_count += 1;
        }
      if (scores[pos] > best_score)
        {
          second_score = best_score;
          best_score = scores[pos];
          best = static_cast<int> (pos);
        }
      else if (scores[pos] > second_score)
        {
          second_score = scores[pos];
        }
    }
  std::sort (decision.candidates.begin (), decision.candidates.end (),
             [] (const AdaptiveAnchorCandidate &a,
                 const AdaptiveAnchorCandidate &b) {
               return a.score > b.score;
             });
  if (decision.candidates.size () > 5)
    {
      decision.candidates.resize (5);
    }

  if (best < 0)
    {
      return decision;
    }

  const auto &selected =
      anchors[static_cast<std::size_t> (valid_indices[best])];
  decision.selected_anchor_id = selected.anchor_id;
  decision.selected_tier = MemoryTier (current_turn, selected.last_seen_turn);
  decision.selected_score = best_score;
  decision.selected_attention = weights[static_cast<std::size_t> (best)];
  decision.margin = std::isfinite (second_score) ? best_score - second_score
                                                 : best_score;
  if (policy.soft_hypothesis_mode)
    {
      const double selected_p =
          static_cast<std::size_t> (best) < real_posteriors.size ()
              ? real_posteriors[static_cast<std::size_t> (best)]
              : 0.0;
      double next_p = std::max (p_none, p_new);
      for (std::size_t pos = 0; pos < real_posteriors.size (); ++pos)
        {
          if (pos == static_cast<std::size_t> (best))
            {
              continue;
            }
          next_p = std::max (next_p, real_posteriors[pos]);
        }
      decision.margin = selected_p - next_p;
    }
  decision.semantic_cosine = semantic_cosines[static_cast<std::size_t> (best)];
  decision.entity_cosine = entity_cosines[static_cast<std::size_t> (best)];
  decision.event_cosine = event_cosines[static_cast<std::size_t> (best)];
  decision.split_pressure = std::max (0.0, decision.entity_cosine
                                               - decision.event_cosine)
                            + std::max (0.0, decision.entropy
                                                 - policy.max_entropy);
  decision.confidence_before = selected.confidence;
  decision.support_before = selected.support_count;
  decision.contradiction_before = selected.contradiction_count;
  for (std::size_t rank = 0; rank < decision.candidates.size (); ++rank)
    {
      auto &candidate = decision.candidates[rank];
      const double candidate_margin = rank == 0 ? decision.margin : 0.0;
      const bool candidate_event_mismatch =
          candidate.entity_cosine - candidate.event_cosine
          >= policy.event_mismatch_threshold;
      candidate.anchor_strength =
          policy.soft_hypothesis_mode
              ? SoftCandidateAnchorStrength (
                    candidate.score, candidate.attention, candidate_margin,
                    decision.entropy, candidate.confidence,
                    candidate.support_count, candidate.contradiction_count)
              : CandidateAnchorStrength (
                    candidate.score, candidate.attention, candidate_margin,
                    decision.entropy, candidate.confidence,
                    candidate.support_count, candidate.contradiction_count);
      candidate.anchor_label =
          AnchorLabel (candidate.anchor_strength, candidate_margin,
                       decision.entropy, candidate.support_count,
                       candidate.contradiction_count, "active", policy,
                       decision.generic_signal, candidate_event_mismatch);
      candidate.retained_soft_link =
          !policy.soft_hypothesis_mode
          || (candidate.attention >= policy.posterior_keep_threshold
              && candidate.score >= policy.tentative_score_threshold
              && decision.generic_score < policy.generic_hard_threshold);
    }
  if (!decision.candidates.empty ())
    {
      decision.selected_anchor_strength =
          decision.candidates.front ().anchor_strength;
      decision.selected_anchor_label = decision.candidates.front ().anchor_label;
    }

  if (policy.generic_action_suppressor && !policy.soft_hypothesis_mode
      && decision.generic_signal)
    {
      decision.action = "abstain";
      return decision;
    }

  if (policy.soft_hypothesis_mode)
    {
      decision.p_top_real =
          best >= 0 && static_cast<std::size_t> (best) < real_posteriors.size ()
              ? real_posteriors[static_cast<std::size_t> (best)]
              : 0.0;
      const bool none_wins =
          best_hypothesis == static_cast<int> (scores.size ());
      const bool new_wins =
          best_hypothesis == static_cast<int> (scores.size () + 1);
      const bool generic_hard =
          decision.generic_score >= policy.generic_hard_threshold;
      if (none_wins || generic_hard)
        {
          decision.action = "abstain";
          return decision;
        }
      if (new_wins && p_new >= policy.new_anchor_threshold)
        {
          decision.action = "create_anchor";
          return decision;
        }
    }

  const bool event_mismatch = decision.entity_cosine
                                  - decision.event_cosine
                              >= policy.event_mismatch_threshold;
  const bool can_update =
      decision.selected_score >= policy.update_threshold
      && decision.margin >= policy.margin_threshold
      && decision.entropy <= policy.max_entropy
      && selected.support_count >= policy.min_update_support
      && (!policy.soft_hypothesis_mode
          || decision.p_top_real >= policy.soft_update_posterior_threshold)
      && !(policy.split_on_event_mismatch && event_mismatch);
  if (can_update)
    {
      decision.action = "update_existing";
    }
  else if (policy.soft_hypothesis_mode
           && decision.p_top_real >= policy.posterior_keep_threshold
           && decision.selected_score >= policy.tentative_score_threshold)
    {
      decision.action = "soft_update";
    }
  else if (!policy.soft_hypothesis_mode && policy.null_no_anchor_hypothesis
           && (decision.selected_score < policy.null_score_threshold
               || decision.entropy > policy.null_entropy_threshold))
    {
      decision.action = "abstain";
    }
  else if (!policy.soft_hypothesis_mode
           && (event_mismatch || decision.selected_score
                                    >= policy.update_threshold - 0.04))
    {
      decision.action = "split_anchor";
    }
  return decision;
}

std::string
StoreAdaptiveCurrentMemory (const Conversation &conv, int turn,
                            const EncodedStep &encoded,
                            AdaptiveAnchorDecision &decision,
                            std::vector<AdaptiveAnchor> &anchors,
                            std::map<int, std::string> &turn_anchor_ids,
                            const AdaptiveAnchorPolicy &policy)
{
  std::string anchor_id;
  if ((decision.action == "update_existing" || decision.action == "soft_update")
      && !decision.selected_anchor_id.empty ())
    {
      anchor_id = decision.selected_anchor_id;
      if (AdaptiveAnchor *anchor = FindAdaptiveAnchor (anchors, anchor_id))
        {
          const bool hard_update = decision.action == "update_existing";
          if (hard_update)
            {
              anchor->semantic_state = Blend (anchor->semantic_state,
                                              encoded.semantic_key,
                                              anchor->support_count);
              anchor->entity_state = Blend (anchor->entity_state,
                                            encoded.entity_key,
                                            anchor->support_count);
              anchor->full_state = Blend (anchor->full_state,
                                          encoded.full_key,
                                          anchor->support_count);
            }
          else
            {
              const double alpha =
                  0.04 + 0.10 * Clamp01 (decision.selected_anchor_strength);
              anchor->semantic_state = SoftBlend (anchor->semantic_state,
                                                  encoded.semantic_key, alpha);
              anchor->entity_state = SoftBlend (anchor->entity_state,
                                                encoded.entity_key, alpha);
              anchor->full_state =
                  SoftBlend (anchor->full_state, encoded.full_key, alpha);
            }
          anchor->last_seen_turn = turn;
          anchor->last_text = conv.messages[static_cast<std::size_t> (turn)];
          anchor->support_count += hard_update ? 1 : 0;
          const double support = Clamp01 (
              std::log1p (static_cast<double> (anchor->support_count))
              / std::log (9.0));
          anchor->confidence = Clamp01 (
              anchor->confidence + (hard_update ? 0.06 : 0.01)
              + (hard_update ? 0.04 : 0.01) * support
              - 0.03 * static_cast<double> (anchor->contradiction_count));
          anchor->stability = Clamp01 (0.62 * anchor->stability
                                       + 0.38 * decision.margin);
          if (anchor->support_count >= policy.active_support_count
              && anchor->confidence >= 0.50)
            {
              anchor->status = "active";
            }
          anchor->anchor_strength = StateAnchorStrength (
              anchor->confidence, anchor->stability, anchor->support_count,
              anchor->contradiction_count);
          anchor->anchor_label = AnchorStateLabel (
              anchor->anchor_strength, anchor->support_count,
              anchor->contradiction_count, anchor->status, policy);
        }
    }
  else if (decision.action == "abstain")
    {
      anchor_id.clear ();
    }
  else
    {
      if (decision.action == "split_anchor" && !decision.selected_anchor_id.empty ())
        {
          if (AdaptiveAnchor *anchor = FindAdaptiveAnchor (
                  anchors, decision.selected_anchor_id))
            {
              anchor->contradiction_count += 1;
              anchor->confidence = Clamp01 (anchor->confidence - 0.04);
              anchor->anchor_strength = StateAnchorStrength (
                  anchor->confidence, anchor->stability, anchor->support_count,
                  anchor->contradiction_count);
              anchor->anchor_label = AnchorStateLabel (
                  anchor->anchor_strength, anchor->support_count,
                  anchor->contradiction_count, anchor->status, policy);
            }
        }
      anchor_id = "adaptive_anchor_" + std::to_string (anchors.size ());
      AdaptiveAnchor anchor;
      anchor.anchor_id = anchor_id;
      anchor.semantic_state = encoded.semantic_key;
      anchor.entity_state = encoded.entity_key;
      anchor.full_state = encoded.full_key;
      anchor.first_seen_turn = turn;
      anchor.last_seen_turn = turn;
      anchor.first_text = conv.messages[static_cast<std::size_t> (turn)];
      anchor.last_text = anchor.first_text;
      anchor.support_count = 1;
      anchor.confidence = decision.action == "split_anchor" ? 0.52 : 0.45;
      anchor.stability = 0.0;
      anchor.status = "provisional";
      anchor.anchor_strength = StateAnchorStrength (
          anchor.confidence, anchor.stability, anchor.support_count,
          anchor.contradiction_count);
      anchor.anchor_label = decision.action == "split_anchor"
                                ? "ambiguous"
                                : AnchorStateLabel (
                                      anchor.anchor_strength,
                                      anchor.support_count,
                                      anchor.contradiction_count,
                                      anchor.status, policy);
      anchors.push_back (std::move (anchor));
      decision.selected_anchor_id = anchor_id;
      decision.selected_anchor_strength = anchors.back ().anchor_strength;
      decision.selected_anchor_label = anchors.back ().anchor_label;
    }
  (void)conv;
  if (!anchor_id.empty ())
    {
      turn_anchor_ids[turn] = anchor_id;
    }
  return anchor_id;
}

void
StoreCurrentMemory (const Conversation &conv, int turn,
                    const EncodedStep &encoded,
                    const StorageAttentionDecision &decision,
                    std::vector<StoredMemory> &memories,
                    std::vector<StorageAnchor> &anchors)
{
  std::string anchor_id;
  if (decision.action == "update_existing"
      && !decision.selected_anchor_id.empty ())
    {
      anchor_id = decision.selected_anchor_id;
      if (StorageAnchor *anchor = FindStorageAnchor (anchors, anchor_id))
        {
          anchor->entity_state = Blend (anchor->entity_state,
                                        encoded.entity_key,
                                        anchor->support_count);
          anchor->full_state = Blend (anchor->full_state, encoded.full_key,
                                      anchor->support_count);
          anchor->last_seen_turn = turn;
          anchor->support_count += 1;
        }
    }
  else if (decision.action == "create_anchor")
    {
      anchor_id = "anchor_" + std::to_string (anchors.size ());
      StorageAnchor anchor;
      anchor.anchor_id = anchor_id;
      anchor.entity_state = encoded.entity_key;
      anchor.full_state = encoded.full_key;
      anchor.first_seen_turn = turn;
      anchor.last_seen_turn = turn;
      anchor.support_count = 1;
      anchors.push_back (std::move (anchor));
    }

  StoredMemory memory;
  memory.memory_id = StorageMemoryId (conv, turn);
  memory.turn = turn;
  memory.text = conv.messages[static_cast<std::size_t> (turn)];
  memory.views = encoded;
  memory.anchor_id = anchor_id;
  memories.push_back (std::move (memory));
}

double
Auc (const std::vector<double> &scores, const std::vector<int> &labels)
{
  double wins = 0.0;
  double total = 0.0;
  for (std::size_t i = 0; i < scores.size (); ++i)
    {
      if (labels[i] != 1)
        {
          continue;
        }
      for (std::size_t j = 0; j < scores.size (); ++j)
        {
          if (labels[j] != 0)
            {
              continue;
            }
          if (scores[i] > scores[j])
            {
              wins += 1.0;
            }
          else if (scores[i] == scores[j])
            {
              wins += 0.5;
            }
          total += 1.0;
        }
    }
  return total <= 0.0 ? 0.5 : wins / total;
}

Match
FindBestMatch (const std::vector<Anchor> &anchors,
               const std::vector<float> &key, const std::string &episode_id)
{
  Match match;
  double second = -1.0;
  for (std::size_t i = 0; i < anchors.size (); ++i)
    {
      if (anchors[i].episode_id != episode_id)
        {
          continue;
        }
      double score = Cosine (anchors[i].key, key);
      if (score > match.score)
        {
          second = match.score;
          match.score = score;
          match.index = static_cast<int> (i);
        }
      else if (score > second)
        {
          second = score;
        }
    }
  match.margin = match.index >= 0 ? match.score - second : 0.0;
  return match;
}

Decision
ApplyDecision (std::vector<Anchor> &anchors, const std::vector<float> &key,
               const IngressStep &step, const std::string &kind,
               const Policy &policy, double threshold)
{
  Match match = FindBestMatch (anchors, key, step.episode_id);
  bool update = match.index >= 0 && match.score >= threshold;
  if (policy.use_margin)
    {
      update = update && match.margin >= policy.margin_threshold;
    }

  Decision decision;
  decision.score = match.score;
  decision.margin = match.margin;
  if (update)
    {
      Anchor &anchor = anchors[static_cast<std::size_t> (match.index)];
      decision.action = "update";
      decision.anchor_id = anchor.anchor_id;
      decision.eval_keys_before = SetToVector (anchor.eval_keys);
      decision.updated_existing = true;
      anchor.key = Blend (anchor.key, key, anchor.support_count);
      anchor.last_seen_step = step.step;
      anchor.support_count += 1;
      return decision;
    }

  Anchor anchor;
  anchor.anchor_id = kind + "_anchor_" + std::to_string (anchors.size () + 1);
  anchor.episode_id = step.episode_id;
  anchor.kind = kind;
  anchor.key = key;
  anchor.first_seen_step = step.step;
  anchor.last_seen_step = step.step;
  anchor.support_count = 1;
  decision.action = "create";
  decision.anchor_id = anchor.anchor_id;
  anchors.push_back (std::move (anchor));
  return decision;
}

bool
ContainsAnyExpected (const Decision &decision,
                     const std::vector<std::string> &expected)
{
  for (const auto &key : expected)
    {
      if (std::find (decision.eval_keys_before.begin (),
                     decision.eval_keys_before.end (), key)
          != decision.eval_keys_before.end ())
        {
          return true;
        }
    }
  return expected.empty ();
}

bool
UpdatedWrongExisting (const Decision &decision,
                      const std::vector<std::string> &expected)
{
  if (!decision.updated_existing || decision.eval_keys_before.empty ())
    {
      return false;
    }
  for (const auto &key : expected)
    {
      if (std::find (decision.eval_keys_before.begin (),
                     decision.eval_keys_before.end (), key)
          != decision.eval_keys_before.end ())
        {
          return false;
        }
    }
  return true;
}

void
AttachEvalKeys (std::vector<Anchor> &anchors, const Decision &decision,
                const std::vector<std::string> &keys)
{
  auto found = std::find_if (
      anchors.begin (), anchors.end (),
      [&] (const Anchor &anchor) { return anchor.anchor_id == decision.anchor_id; });
  if (found == anchors.end ())
    {
      return;
    }
  for (const auto &key : keys)
    {
      found->eval_keys.insert (key);
    }
}

bool
HasRelation (const std::set<std::string> &relations, const Relation &rel)
{
  std::string key = rel.type + "(" + rel.source + "," + rel.target + ")";
  return relations.find (key) != relations.end ();
}

void
MaybeAttachRelations (std::set<std::string> &relations,
                      const IngressStep &step, bool entity_ok, bool object_ok)
{
  if (!entity_ok || !object_ok)
    {
      return;
    }
  for (const auto &rel : step.relations)
    {
      relations.insert (rel.type + "(" + rel.source + "," + rel.target + ")");
    }
}

bool
EvaluateStep (const IngressStep &step, const Decision &entity,
              const Decision &object, const std::set<std::string> &relations,
              bool entity_ok, bool object_ok, bool false_existing_update,
              bool &relation_success)
{
  relation_success = true;
  for (const auto &rel : step.relations)
    {
      relation_success = relation_success && HasRelation (relations, rel);
    }
  if (step.no_anchor)
    {
      return !false_existing_update;
    }
  const bool needs_entity = !step.entities.empty ();
  const bool needs_object = !step.objects.empty ();
  const bool entity_part = !needs_entity || entity_ok
                           || (entity.action == "create"
                               && step.family.find ("intro") != std::string::npos)
                           || step.family == "explicit_entity_create";
  const bool object_part = !needs_object || object_ok
                           || (object.action == "create"
                               && step.modality != "text");
  return entity_part && object_part && relation_success
         && !false_existing_update;
}

PolicyRun
RunPolicy (const std::vector<IngressStep> &steps,
           const std::unordered_map<std::string, EncodedStep> &encoded_cache,
           const Policy &policy)
{
  PolicyRun run;
  run.policy = policy;
  std::vector<Anchor> entity_anchors;
  std::vector<Anchor> object_anchors;
  std::set<std::string> relations;
  std::vector<double> latencies;
  for (const auto &step : steps)
    {
      const EncodedStep &encoded = encoded_cache.at (step.text);
      latencies.push_back (encoded.latency_ms);
      StepRow row;
      row.policy = policy.name;
      row.step = step;
      row.entity_anchor_count_before = static_cast<int> (entity_anchors.size ());
      row.object_anchor_count_before = static_cast<int> (object_anchors.size ());
      row.entity_decision = ApplyDecision (
          entity_anchors, encoded.entity_key, step, "entity", policy,
          policy.entity_update_threshold);
      row.object_decision = ApplyDecision (
          object_anchors, encoded.full_key, step, "object_event", policy,
          policy.object_update_threshold);

      const bool entity_ok = ContainsAnyExpected (row.entity_decision,
                                                  step.entities);
      const bool object_ok = ContainsAnyExpected (row.object_decision,
                                                  step.objects);
      row.false_existing_update =
          UpdatedWrongExisting (row.entity_decision, step.entities)
          || UpdatedWrongExisting (row.object_decision, step.objects);

      if (!row.false_existing_update)
        {
          if (step.family == "explicit_entity_create"
              || step.family.find ("intro") != std::string::npos
              || entity_ok)
            {
              AttachEvalKeys (entity_anchors, row.entity_decision,
                              step.entities);
            }
          if (step.modality != "text" || object_ok)
            {
              AttachEvalKeys (object_anchors, row.object_decision,
                              step.objects);
            }
          MaybeAttachRelations (relations, step, entity_ok || !step.entities.empty (),
                                object_ok || step.modality != "text");
        }

      row.success = EvaluateStep (
          step, row.entity_decision, row.object_decision, relations, entity_ok,
          object_ok, row.false_existing_update, row.relation_success);
      run.rows.push_back (std::move (row));
    }
  run.mean_encode_ms = Mean (latencies);
  run.p95_encode_ms = Percentile (latencies, 0.95);
  return run;
}

Json
SummarizeRun (const PolicyRun &run)
{
  std::map<std::string, int> family_total;
  std::map<std::string, int> family_success;
  int success = 0;
  int false_existing_update = 0;
  int no_anchor = 0;
  int no_anchor_safe = 0;
  for (const auto &row : run.rows)
    {
      success += row.success ? 1 : 0;
      false_existing_update += row.false_existing_update ? 1 : 0;
      family_total[row.step.family] += 1;
      family_success[row.step.family] += row.success ? 1 : 0;
      if (row.step.no_anchor)
        {
          ++no_anchor;
          no_anchor_safe += row.success ? 1 : 0;
        }
    }
  Json family = Json::object ();
  for (const auto &[name, total] : family_total)
    {
      family[name] = {
        { "steps", total },
        { "success", family_success[name] },
        { "success_rate",
          total == 0 ? 0.0
                     : static_cast<double> (family_success[name])
                           / static_cast<double> (total) },
      };
    }
  const int total = static_cast<int> (run.rows.size ());
  return {
    { "entity_update_threshold", run.policy.entity_update_threshold },
    { "object_update_threshold", run.policy.object_update_threshold },
    { "margin_threshold", run.policy.margin_threshold },
    { "use_margin", run.policy.use_margin },
    { "steps", total },
    { "success", success },
    { "success_rate",
      total == 0 ? 0.0 : static_cast<double> (success) / total },
    { "false_existing_update", false_existing_update },
    { "false_existing_update_rate",
      total == 0 ? 0.0
                 : static_cast<double> (false_existing_update) / total },
    { "no_anchor_steps", no_anchor },
    { "no_anchor_safe", no_anchor_safe },
    { "no_anchor_safe_rate",
      no_anchor == 0 ? 0.0
                     : static_cast<double> (no_anchor_safe) / no_anchor },
    { "mean_encode_ms", run.mean_encode_ms },
    { "p95_encode_ms", run.p95_encode_ms },
    { "family", family },
  };
}

void
WriteRowsCsv (const std::filesystem::path &path,
              const std::vector<PolicyRun> &runs)
{
  std::ofstream out (path);
  out << "policy,episode_id,step,family,modality,current_signal_text,"
         "entity_action,entity_score,entity_margin,entity_eval_keys_before,"
         "object_action,object_score,object_margin,object_eval_keys_before,"
         "relation_success,success,false_existing_update,"
         "entity_anchor_count_before,object_anchor_count_before,"
         "retrieved_candidate_count,uses_retrieved_candidates,"
         "runtime_policy_uses_labels\n";
  for (const auto &run : runs)
    {
      for (const auto &row : run.rows)
        {
          out << CsvEscape (row.policy) << ','
              << CsvEscape (row.step.episode_id) << ',' << row.step.step
              << ',' << CsvEscape (row.step.family) << ','
              << CsvEscape (row.step.modality) << ','
              << CsvEscape (row.step.text) << ','
              << CsvEscape (row.entity_decision.action) << ','
              << row.entity_decision.score << ',' << row.entity_decision.margin
              << ','
              << CsvEscape (Join (row.entity_decision.eval_keys_before)) << ','
              << CsvEscape (row.object_decision.action) << ','
              << row.object_decision.score << ',' << row.object_decision.margin
              << ','
              << CsvEscape (Join (row.object_decision.eval_keys_before)) << ','
              << (row.relation_success ? 1 : 0) << ','
              << (row.success ? 1 : 0) << ','
              << (row.false_existing_update ? 1 : 0) << ','
              << row.entity_anchor_count_before << ','
              << row.object_anchor_count_before << ",0,0,0\n";
        }
    }
}

void
WriteFailuresCsv (const std::filesystem::path &path,
                  const std::vector<PolicyRun> &runs)
{
  std::ofstream out (path);
  out << "policy,episode_id,step,family,current_signal_text,entity_action,"
         "entity_score,entity_eval_keys_before,object_action,object_score,"
         "object_eval_keys_before,relation_success,false_existing_update\n";
  int count = 0;
  for (const auto &run : runs)
    {
      for (const auto &row : run.rows)
        {
          if (row.success)
            {
              continue;
            }
          out << CsvEscape (row.policy) << ','
              << CsvEscape (row.step.episode_id) << ',' << row.step.step
              << ',' << CsvEscape (row.step.family) << ','
              << CsvEscape (row.step.text) << ','
              << CsvEscape (row.entity_decision.action) << ','
              << row.entity_decision.score << ','
              << CsvEscape (Join (row.entity_decision.eval_keys_before)) << ','
              << CsvEscape (row.object_decision.action) << ','
              << row.object_decision.score << ','
              << CsvEscape (Join (row.object_decision.eval_keys_before)) << ','
              << (row.relation_success ? 1 : 0) << ','
              << (row.false_existing_update ? 1 : 0) << '\n';
          if (++count >= 240)
            {
              return;
            }
        }
    }
}

std::string
EvalCaseKey (const StorageEvalCase &row)
{
  return ConversationKey (row.dataset, row.conversation_id) + "#"
         + std::to_string (row.query_turn);
}

StorageAttentionRun
RunStorageAttentionPolicy (
    const std::vector<Conversation> &conversations,
    const std::vector<StorageEvalCase> &eval_cases,
    cortext::AaitGgufEncoder &encoder,
    std::unordered_map<std::string, EncodedStep> &encoded_cache,
    const StorageAttentionPolicy &policy,
    int max_turns_per_conversation)
{
  StorageAttentionRun run;
  run.policy = policy;
  std::map<std::string, std::vector<std::size_t>> cases_by_turn;
  for (std::size_t i = 0; i < eval_cases.size (); ++i)
    {
      cases_by_turn[EvalCaseKey (eval_cases[i])].push_back (i);
    }

  std::vector<double> latencies;
  for (const auto &conv : conversations)
    {
      const int max_turns = std::min<int> (
          static_cast<int> (conv.messages.size ()),
          max_turns_per_conversation > 0
              ? max_turns_per_conversation
              : static_cast<int> (conv.messages.size ()));
      std::vector<StoredMemory> memories;
      std::vector<StorageAnchor> anchors;
      for (int turn = 0; turn < max_turns; ++turn)
        {
          const EncodedStep current = EncodeStep (
              encoder, encoded_cache,
              conv.messages[static_cast<std::size_t> (turn)]);
          latencies.push_back (current.latency_ms);
          StorageAttentionDecision decision = AttendStoredMemories (
              memories, current, turn, policy);

          const std::string key = ConversationKey (conv) + "#"
                                  + std::to_string (turn);
          const auto found_cases = cases_by_turn.find (key);
          if (found_cases != cases_by_turn.end ())
            {
              for (std::size_t case_index : found_cases->second)
                {
                  const auto &meta = eval_cases[case_index];
                  StorageAttentionCaseRow row;
                  row.policy = policy.name;
                  row.meta = meta;
                  row.current_signal_text =
                      conv.messages[static_cast<std::size_t> (turn)];
                  row.decision = decision;
                  row.target_anchor_id = meta.target_turn >= 0
                                             ? MemoryAnchorAtTurn (
                                                   memories, meta.target_turn)
                                             : "";
                  row.wrong_active_anchor_id =
                      MemoryAnchorAtTurn (memories, meta.wrong_active_turn);
                  row.stale_anchor_id = MemoryAnchorAtTurn (memories,
                                                            meta.stale_turn);
                  row.target_anchor_existed = !row.target_anchor_id.empty ();
                  row.selected_target =
                      meta.is_reference && row.target_anchor_existed
                      && decision.action == "update_existing"
                      && decision.selected_anchor_id == row.target_anchor_id;
                  row.selected_wrong_active =
                      !row.wrong_active_anchor_id.empty ()
                      && decision.action == "update_existing"
                      && decision.selected_anchor_id
                             == row.wrong_active_anchor_id
                      && row.wrong_active_anchor_id != row.target_anchor_id;
                  row.selected_stale =
                      !row.stale_anchor_id.empty ()
                      && decision.action == "update_existing"
                      && decision.selected_anchor_id == row.stale_anchor_id
                      && row.stale_anchor_id != row.target_anchor_id;
                  row.no_anchor_abstained =
                      !meta.is_reference
                      && decision.action != "update_existing";
                  row.wrong_active_distinct =
                      meta.is_reference && row.target_anchor_existed
                      && !row.wrong_active_anchor_id.empty ()
                      && row.wrong_active_anchor_id != row.target_anchor_id;
                  row.stale_inactive =
                      meta.is_reference
                      && (row.stale_anchor_id.empty ()
                          || decision.selected_anchor_id
                                 != row.stale_anchor_id);
                  run.cases.push_back (std::move (row));
                }
            }

          StoreCurrentMemory (conv, turn, current, decision, memories, anchors);

          StorageAttentionLinkRow link;
          link.policy = policy.name;
          link.dataset = conv.dataset;
          link.conversation_id = conv.conversation_id;
          link.turn = turn;
          link.memory_id = StorageMemoryId (conv, turn);
          link.action = decision.action;
          link.anchor_id = memories.back ().anchor_id;
          link.selected_memory_id = decision.selected_memory_id;
          link.selected_tier = decision.selected_tier;
          link.selected_score = decision.selected_score;
          link.margin = decision.margin;
          link.wm_mass = decision.wm_mass;
          link.stm_mass = decision.stm_mass;
          link.ltm_mass = decision.ltm_mass;
          run.links.push_back (std::move (link));
        }
    }

  run.mean_encode_ms = Mean (latencies);
  run.p95_encode_ms = Percentile (latencies, 0.95);
  return run;
}

Json
SummarizeStorageAttentionRun (const StorageAttentionRun &run)
{
  int refs = 0;
  int controls = 0;
  int target_existed = 0;
  int selected_target = 0;
  int selected_wrong = 0;
  int selected_stale = 0;
  int no_anchor_abstain = 0;
  int wrong_distinct = 0;
  int stale_inactive = 0;
  std::vector<double> case_scores;
  std::vector<int> case_labels;
  std::vector<double> control_scores;
  std::vector<double> successful_ref_scores;
  for (const auto &row : run.cases)
    {
      const double score = row.decision.action == "update_existing"
                               ? row.decision.selected_score
                               : -1.0;
      if (row.meta.is_reference)
        {
          ++refs;
          target_existed += row.target_anchor_existed ? 1 : 0;
          selected_target += row.selected_target ? 1 : 0;
          selected_wrong += row.selected_wrong_active ? 1 : 0;
          selected_stale += row.selected_stale ? 1 : 0;
          wrong_distinct += row.wrong_active_distinct ? 1 : 0;
          stale_inactive += row.stale_inactive ? 1 : 0;
          successful_ref_scores.push_back (row.selected_target ? score : -1.0);
          case_scores.push_back (score);
          case_labels.push_back (1);
        }
      else
        {
          ++controls;
          no_anchor_abstain += row.no_anchor_abstained ? 1 : 0;
          control_scores.push_back (score);
          case_scores.push_back (score);
          case_labels.push_back (0);
        }
    }

  auto recovery = [&] (int allowed_controls) {
    std::vector<double> thresholds = control_scores;
    thresholds.push_back (std::numeric_limits<double>::infinity ());
    thresholds.push_back (-std::numeric_limits<double>::infinity ());
    int best = 0;
    for (double threshold : thresholds)
      {
        int control_hits = 0;
        for (double score : control_scores)
          {
            if (score >= threshold)
              {
                ++control_hits;
              }
          }
        if (control_hits > allowed_controls)
          {
            continue;
          }
        int recovered = 0;
        for (double score : successful_ref_scores)
          {
            if (score >= threshold)
              {
                ++recovered;
              }
          }
        best = std::max (best, recovered);
      }
    return best;
  };

  return {
    { "reference_count", refs },
    { "control_count", controls },
    { "stored_memory_count", run.links.size () },
    { "target_anchor_existed_count", target_existed },
    { "target_anchor_existed_rate",
      refs == 0 ? 0.0 : static_cast<double> (target_existed) / refs },
    { "selected_target_count", selected_target },
    { "selected_target_rate",
      refs == 0 ? 0.0 : static_cast<double> (selected_target) / refs },
    { "selected_wrong_active_count", selected_wrong },
    { "selected_wrong_active_rate",
      refs == 0 ? 0.0 : static_cast<double> (selected_wrong) / refs },
    { "selected_stale_count", selected_stale },
    { "selected_stale_rate",
      refs == 0 ? 0.0 : static_cast<double> (selected_stale) / refs },
    { "no_anchor_abstain_count", no_anchor_abstain },
    { "no_anchor_abstain_rate",
      controls == 0 ? 0.0
                    : static_cast<double> (no_anchor_abstain) / controls },
    { "wrong_active_distinct_count", wrong_distinct },
    { "wrong_active_distinct_rate",
      refs == 0 ? 0.0 : static_cast<double> (wrong_distinct) / refs },
    { "stale_inactive_count", stale_inactive },
    { "stale_inactive_rate",
      refs == 0 ? 0.0 : static_cast<double> (stale_inactive) / refs },
    { "reference_vs_no_anchor_auc", Auc (case_scores, case_labels) },
    { "zero_fpr_recovery", recovery (0) },
    { "five_pct_fpr_recovery",
      recovery (static_cast<int> (std::floor (
          0.05 * static_cast<double> (std::max (0, controls)))) ) },
    { "mean_encode_ms", run.mean_encode_ms },
    { "p95_encode_ms", run.p95_encode_ms },
  };
}

void
WriteStorageCasesCsv (const std::filesystem::path &path,
                      const std::vector<StorageAttentionRun> &runs)
{
  std::ofstream out (path);
  out << "policy,case_id,dataset,conversation_id,label_class,is_reference,"
         "current_ingress_step,current_signal_text,target_turn,"
         "target_distance,action,selected_memory_id,selected_anchor_id,"
         "selected_tier,selected_score,selected_attention,margin,entropy,"
         "wm_mass,stm_mass,ltm_mass,wm_count,stm_count,ltm_count,"
         "target_anchor_id,wrong_active_anchor_id,stale_anchor_id,"
         "target_anchor_existed,selected_target,selected_wrong_active,"
         "selected_stale,no_anchor_abstained,wrong_active_distinct,"
         "stale_inactive,retrieved_candidate_count,uses_retrieved_candidates,"
         "runtime_policy_uses_labels\n";
  for (const auto &run : runs)
    {
      for (const auto &row : run.cases)
        {
          const auto &d = row.decision;
          out << CsvEscape (run.policy.name) << ','
              << CsvEscape (row.meta.case_id) << ','
              << CsvEscape (row.meta.dataset) << ','
              << CsvEscape (row.meta.conversation_id) << ','
              << CsvEscape (row.meta.label_class) << ','
              << (row.meta.is_reference ? 1 : 0) << ','
              << row.meta.query_turn << ','
              << CsvEscape (row.current_signal_text) << ','
              << row.meta.target_turn << ',' << row.meta.target_distance << ','
              << CsvEscape (d.action) << ','
              << CsvEscape (d.selected_memory_id) << ','
              << CsvEscape (d.selected_anchor_id) << ','
              << CsvEscape (d.selected_tier) << ',' << d.selected_score << ','
              << d.selected_attention << ',' << d.margin << ',' << d.entropy
              << ',' << d.wm_mass << ',' << d.stm_mass << ',' << d.ltm_mass
              << ',' << d.wm_count << ',' << d.stm_count << ',' << d.ltm_count
              << ',' << CsvEscape (row.target_anchor_id) << ','
              << CsvEscape (row.wrong_active_anchor_id) << ','
              << CsvEscape (row.stale_anchor_id) << ','
              << (row.target_anchor_existed ? 1 : 0) << ','
              << (row.selected_target ? 1 : 0) << ','
              << (row.selected_wrong_active ? 1 : 0) << ','
              << (row.selected_stale ? 1 : 0) << ','
              << (row.no_anchor_abstained ? 1 : 0) << ','
              << (row.wrong_active_distinct ? 1 : 0) << ','
              << (row.stale_inactive ? 1 : 0) << ",0,0,0\n";
        }
    }
}

void
WriteAdaptiveCandidatesCsv (const std::filesystem::path &path,
                            const std::vector<AdaptiveAnchorRun> &runs)
{
  std::ofstream out (path);
  out << "policy,case_id,dataset,conversation_id,label_class,is_reference,"
         "current_ingress_step,candidate_rank,anchor_id,tier,score,attention,"
         "anchor_strength,anchor_label,entity_cosine,semantic_cosine,"
         "event_cosine,last_seen_turn,"
         "last_seen_text,is_target,"
         "is_wrong_active,is_stale\n";
  for (const auto &run : runs)
    {
      for (const auto &row : run.cases)
        {
          for (std::size_t i = 0; i < row.decision.candidates.size (); ++i)
            {
              const auto &candidate = row.decision.candidates[i];
              out << CsvEscape (run.policy.name) << ','
                  << CsvEscape (row.meta.case_id) << ','
                  << CsvEscape (row.meta.dataset) << ','
                  << CsvEscape (row.meta.conversation_id) << ','
                  << CsvEscape (row.meta.label_class) << ','
                  << (row.meta.is_reference ? 1 : 0) << ','
                  << row.meta.query_turn << ',' << (i + 1) << ','
                  << CsvEscape (candidate.anchor_id) << ','
                  << CsvEscape (candidate.tier) << ',' << candidate.score
                  << ',' << candidate.attention << ','
                  << candidate.anchor_strength << ','
                  << CsvEscape (candidate.anchor_label) << ','
                  << candidate.entity_cosine << ','
                  << candidate.semantic_cosine << ','
                  << candidate.event_cosine << ','
                  << candidate.last_seen_turn << ','
                  << CsvEscape (candidate.last_text) << ','
                  << (candidate.anchor_id == row.target_anchor_id ? 1 : 0)
                  << ','
                  << (candidate.anchor_id == row.wrong_active_anchor_id ? 1
                                                                        : 0)
                  << ','
                  << (candidate.anchor_id == row.stale_anchor_id ? 1 : 0)
                  << '\n';
            }
        }
    }
}

void
WriteStorageLinksCsv (const std::filesystem::path &path,
                      const std::vector<StorageAttentionRun> &runs)
{
  std::ofstream out (path);
  out << "policy,dataset,conversation_id,turn,memory_id,action,anchor_id,"
         "selected_memory_id,selected_tier,selected_score,margin,wm_mass,"
         "stm_mass,ltm_mass,created_before_any_future_query\n";
  for (const auto &run : runs)
    {
      for (const auto &link : run.links)
        {
          out << CsvEscape (run.policy.name) << ','
              << CsvEscape (link.dataset) << ','
              << CsvEscape (link.conversation_id) << ',' << link.turn << ','
              << CsvEscape (link.memory_id) << ',' << CsvEscape (link.action)
              << ',' << CsvEscape (link.anchor_id) << ','
              << CsvEscape (link.selected_memory_id) << ','
              << CsvEscape (link.selected_tier) << ',' << link.selected_score
              << ',' << link.margin << ',' << link.wm_mass << ','
              << link.stm_mass << ',' << link.ltm_mass << ",1\n";
        }
    }
}

void
WriteStorageFailuresCsv (const std::filesystem::path &path,
                         const std::vector<StorageAttentionRun> &runs)
{
  std::ofstream out (path);
  out << "policy,case_id,label_class,failure_type,current_ingress_step,"
         "current_signal_text,target_turn,action,selected_memory_id,"
         "selected_anchor_id,selected_tier,"
         "selected_score,margin,wm_mass,stm_mass,ltm_mass,target_anchor_id,"
         "wrong_active_anchor_id,stale_anchor_id\n";
  int count = 0;
  for (const auto &run : runs)
    {
      for (const auto &row : run.cases)
        {
          std::string failure;
          if (row.meta.is_reference && !row.selected_target)
            {
              failure = row.selected_wrong_active ? "selected_wrong_active"
                                                  : "missed_target_anchor";
            }
          else if (!row.meta.is_reference && !row.no_anchor_abstained)
            {
              failure = "no_anchor_false_bind";
            }
          if (failure.empty ())
            {
              continue;
            }
          const auto &d = row.decision;
          out << CsvEscape (run.policy.name) << ','
              << CsvEscape (row.meta.case_id) << ','
              << CsvEscape (row.meta.label_class) << ','
              << CsvEscape (failure) << ',' << row.meta.query_turn << ','
              << CsvEscape (row.current_signal_text) << ','
              << row.meta.target_turn << ',' << CsvEscape (d.action) << ','
              << CsvEscape (d.selected_memory_id) << ','
              << CsvEscape (d.selected_anchor_id) << ','
              << CsvEscape (d.selected_tier) << ',' << d.selected_score << ','
              << d.margin << ',' << d.wm_mass << ',' << d.stm_mass << ','
              << d.ltm_mass << ',' << CsvEscape (row.target_anchor_id) << ','
              << CsvEscape (row.wrong_active_anchor_id) << ','
              << CsvEscape (row.stale_anchor_id) << '\n';
          if (++count >= 240)
            {
              return;
            }
        }
    }
}

AdaptiveAnchorRun
RunAdaptiveAnchorPolicy (
    const std::vector<Conversation> &conversations,
    const std::vector<StorageEvalCase> &eval_cases,
    cortext::AaitGgufEncoder &encoder,
    std::unordered_map<std::string, EncodedStep> &encoded_cache,
    const AdaptiveAnchorPolicy &policy,
    int max_turns_per_conversation)
{
  AdaptiveAnchorRun run;
  run.policy = policy;
  std::map<std::string, std::vector<std::size_t>> cases_by_turn;
  for (std::size_t i = 0; i < eval_cases.size (); ++i)
    {
      cases_by_turn[EvalCaseKey (eval_cases[i])].push_back (i);
    }

  std::vector<double> latencies;
  for (const auto &conv : conversations)
    {
      const int max_turns = std::min<int> (
          static_cast<int> (conv.messages.size ()),
          max_turns_per_conversation > 0
              ? max_turns_per_conversation
              : static_cast<int> (conv.messages.size ()));
      std::vector<AdaptiveAnchor> anchors;
      std::map<int, std::string> turn_anchor_ids;
      for (int turn = 0; turn < max_turns; ++turn)
        {
          for (auto &anchor : anchors)
            {
              if (turn - anchor.last_seen_turn > policy.active_ttl)
                {
                  anchor.status = "decayed";
                  anchor.confidence = Clamp01 (anchor.confidence - 0.02);
                  anchor.anchor_strength = StateAnchorStrength (
                      anchor.confidence, anchor.stability,
                      anchor.support_count, anchor.contradiction_count);
                  anchor.anchor_label = AnchorStateLabel (
                      anchor.anchor_strength, anchor.support_count,
                      anchor.contradiction_count, anchor.status, policy);
                }
            }

          const EncodedStep current = EncodeStep (
              encoder, encoded_cache,
              conv.messages[static_cast<std::size_t> (turn)]);
          latencies.push_back (current.latency_ms);
          AdaptiveAnchorDecision decision = AttendAdaptiveAnchors (
              anchors, current, turn,
              conv.messages[static_cast<std::size_t> (turn)], policy);

          const std::string key = ConversationKey (conv) + "#"
                                  + std::to_string (turn);
          const auto found_cases = cases_by_turn.find (key);
          if (found_cases != cases_by_turn.end ())
            {
              for (std::size_t case_index : found_cases->second)
                {
                  const auto &meta = eval_cases[case_index];
                  AdaptiveAnchorCaseRow row;
                  row.policy = policy.name;
                  row.meta = meta;
                  row.current_signal_text =
                      conv.messages[static_cast<std::size_t> (turn)];
                  row.decision = decision;
                  row.target_anchor_id = meta.target_turn >= 0
                                             ? AdaptiveAnchorAtTurn (
                                                   turn_anchor_ids,
                                                   meta.target_turn)
                                             : "";
                  row.wrong_active_anchor_id =
                      AdaptiveAnchorAtTurn (turn_anchor_ids,
                                            meta.wrong_active_turn);
                  row.stale_anchor_id = AdaptiveAnchorAtTurn (turn_anchor_ids,
                                                              meta.stale_turn);
                  row.target_anchor_existed = !row.target_anchor_id.empty ();
                  row.target_candidate_rank =
                      AdaptiveCandidateRank (decision, row.target_anchor_id);
                  row.wrong_active_candidate_rank = AdaptiveCandidateRank (
                      decision, row.wrong_active_anchor_id);
                  row.stale_candidate_rank =
                      AdaptiveCandidateRank (decision, row.stale_anchor_id);
                  row.target_in_top2 = row.target_candidate_rank > 0
                                       && row.target_candidate_rank <= 2;
                  row.target_in_top3 = row.target_candidate_rank > 0
                                       && row.target_candidate_rank <= 3;
                  row.wrong_active_in_top3 =
                      row.wrong_active_candidate_rank > 0
                      && row.wrong_active_candidate_rank <= 3;
                  row.selected_target =
                      meta.is_reference && row.target_anchor_existed
                      && decision.action == "update_existing"
                      && decision.selected_anchor_id == row.target_anchor_id;
                  row.selected_wrong_active =
                      !row.wrong_active_anchor_id.empty ()
                      && decision.action == "update_existing"
                      && decision.selected_anchor_id
                             == row.wrong_active_anchor_id
                      && row.wrong_active_anchor_id != row.target_anchor_id;
                  row.selected_stale =
                      !row.stale_anchor_id.empty ()
                      && decision.action == "update_existing"
                      && decision.selected_anchor_id == row.stale_anchor_id
                      && row.stale_anchor_id != row.target_anchor_id;
                  row.no_anchor_abstained =
                      !meta.is_reference
                      && decision.action != "update_existing";
                  row.wrong_active_distinct =
                      meta.is_reference && row.target_anchor_existed
                      && !row.wrong_active_anchor_id.empty ()
                      && row.wrong_active_anchor_id != row.target_anchor_id;
                  row.stale_inactive =
                      meta.is_reference
                      && (row.stale_anchor_id.empty ()
                          || decision.selected_anchor_id
                                 != row.stale_anchor_id);
                  row.hard_commit = decision.action == "update_existing";
                  row.selected_anchor_strength =
                      decision.selected_anchor_strength;
                  row.selected_anchor_label = decision.selected_anchor_label;
                  row.hard_wrong_commit =
                      (meta.is_reference
                       && (row.selected_wrong_active || row.selected_stale))
                      || (!meta.is_reference && row.hard_commit);
                  row.useful_uncertain_reference =
                      meta.is_reference && row.target_in_top3
                      && !row.selected_target;
                  run.cases.push_back (std::move (row));
                }
            }

          const std::string anchor_id = StoreAdaptiveCurrentMemory (
              conv, turn, current, decision, anchors, turn_anchor_ids,
              policy);
          for (std::size_t i = 0; i < decision.candidates.size (); ++i)
            {
              const auto &candidate = decision.candidates[i];
              if (!candidate.retained_soft_link)
                {
                  continue;
                }
              AdaptiveAnchorSoftLinkRow soft_link;
              soft_link.policy = policy.name;
              soft_link.dataset = conv.dataset;
              soft_link.conversation_id = conv.conversation_id;
              soft_link.turn = turn;
              soft_link.memory_id = StorageMemoryId (conv, turn);
              soft_link.anchor_id = candidate.anchor_id;
              soft_link.tier = candidate.tier;
              soft_link.rank = static_cast<int> (i + 1);
              soft_link.score = candidate.score;
              soft_link.posterior = candidate.attention;
              soft_link.margin = i == 0 ? decision.margin : 0.0;
              soft_link.entropy = decision.entropy;
              soft_link.p_none = decision.p_none;
              soft_link.p_new = decision.p_new;
              soft_link.p_top_real = decision.p_top_real;
              soft_link.generic_score = decision.generic_score;
              soft_link.anchor_strength = candidate.anchor_strength;
              soft_link.anchor_label = candidate.anchor_label;
              soft_link.entity_cosine = candidate.entity_cosine;
              soft_link.semantic_cosine = candidate.semantic_cosine;
              soft_link.event_cosine = candidate.event_cosine;
              run.soft_links.push_back (std::move (soft_link));
            }
          AdaptiveAnchorLinkRow link;
          link.policy = policy.name;
          link.dataset = conv.dataset;
          link.conversation_id = conv.conversation_id;
          link.turn = turn;
          link.memory_id = StorageMemoryId (conv, turn);
          link.action = decision.action;
          link.anchor_id = anchor_id;
          link.selected_tier = decision.selected_tier;
          link.selected_score = decision.selected_score;
          link.margin = decision.margin;
          link.entropy = decision.entropy;
          link.p_none = decision.p_none;
          link.p_new = decision.p_new;
          link.p_top_real = decision.p_top_real;
          link.generic_score = decision.generic_score;
          link.split_pressure = decision.split_pressure;
          link.anchor_strength = decision.selected_anchor_strength;
          link.anchor_label = decision.selected_anchor_label;
          if (AdaptiveAnchor *anchor = FindAdaptiveAnchor (anchors, anchor_id))
            {
              link.confidence_after = anchor->confidence;
              link.support_after = anchor->support_count;
              link.contradiction_after = anchor->contradiction_count;
              link.status_after = anchor->status;
              link.anchor_strength = anchor->anchor_strength;
              link.anchor_label = anchor->anchor_label;
            }
          run.links.push_back (std::move (link));
        }

      for (const auto &anchor : anchors)
        {
          AdaptiveAnchorStateRow row;
          row.policy = policy.name;
          row.dataset = conv.dataset;
          row.conversation_id = conv.conversation_id;
          row.anchor_id = anchor.anchor_id;
          row.first_seen_turn = anchor.first_seen_turn;
          row.last_seen_turn = anchor.last_seen_turn;
          row.support_count = anchor.support_count;
          row.contradiction_count = anchor.contradiction_count;
          row.confidence = anchor.confidence;
          row.stability = anchor.stability;
          row.anchor_strength = anchor.anchor_strength;
          row.anchor_label = anchor.anchor_label;
          row.status = anchor.status;
          run.states.push_back (std::move (row));
        }
    }

  run.mean_encode_ms = Mean (latencies);
  run.p95_encode_ms = Percentile (latencies, 0.95);
  return run;
}

Json
SummarizeAdaptiveAnchorRun (const AdaptiveAnchorRun &run)
{
  int refs = 0;
  int controls = 0;
  int target_existed = 0;
  int selected_target = 0;
  int selected_wrong = 0;
  int selected_stale = 0;
  int no_anchor_abstain = 0;
  int wrong_distinct = 0;
  int stale_inactive = 0;
  int target_top1 = 0;
  int target_top2 = 0;
  int target_top3 = 0;
  int useful_uncertain = 0;
  int hard_commit_count = 0;
  int hard_wrong_commit_count = 0;
  int no_anchor_hard_bind = 0;
  int selected_tentative = 0;
  int selected_ambiguous = 0;
  int selected_durable = 0;
  int no_anchor_durable_bind = 0;
  int retained_soft_links = 0;
  int retained_durable_soft_links = 0;
  int update_count = 0;
  int soft_update_count = 0;
  int create_count = 0;
  int split_count = 0;
  int abstain_count = 0;
  std::vector<double> case_scores;
  std::vector<int> case_labels;
  std::vector<double> control_scores;
  std::vector<double> successful_ref_scores;
  for (const auto &link : run.links)
    {
      update_count += link.action == "update_existing" ? 1 : 0;
      soft_update_count += link.action == "soft_update" ? 1 : 0;
      create_count += link.action == "create_anchor" ? 1 : 0;
      split_count += link.action == "split_anchor" ? 1 : 0;
      abstain_count += link.action == "abstain" ? 1 : 0;
    }
  for (const auto &row : run.cases)
    {
      const double score = row.decision.action == "update_existing"
                               ? row.decision.selected_score
                               : -1.0;
      if (row.meta.is_reference)
        {
          ++refs;
          target_existed += row.target_anchor_existed ? 1 : 0;
          selected_target += row.selected_target ? 1 : 0;
          selected_wrong += row.selected_wrong_active ? 1 : 0;
          selected_stale += row.selected_stale ? 1 : 0;
          wrong_distinct += row.wrong_active_distinct ? 1 : 0;
          stale_inactive += row.stale_inactive ? 1 : 0;
          target_top1 += row.target_candidate_rank == 1 ? 1 : 0;
          target_top2 += row.target_in_top2 ? 1 : 0;
          target_top3 += row.target_in_top3 ? 1 : 0;
          useful_uncertain += row.useful_uncertain_reference ? 1 : 0;
          successful_ref_scores.push_back (row.selected_target ? score : -1.0);
          case_scores.push_back (score);
          case_labels.push_back (1);
        }
      else
        {
          ++controls;
          no_anchor_abstain += row.no_anchor_abstained ? 1 : 0;
          no_anchor_hard_bind += row.hard_commit ? 1 : 0;
          no_anchor_durable_bind +=
              row.hard_commit && row.selected_anchor_label == "durable" ? 1
                                                                          : 0;
          control_scores.push_back (score);
          case_scores.push_back (score);
          case_labels.push_back (0);
        }
      hard_commit_count += row.hard_commit ? 1 : 0;
      hard_wrong_commit_count += row.hard_wrong_commit ? 1 : 0;
      selected_tentative += row.selected_anchor_label == "tentative" ? 1 : 0;
      selected_ambiguous += row.selected_anchor_label == "ambiguous" ? 1 : 0;
      selected_durable += row.selected_anchor_label == "durable" ? 1 : 0;
    }
  for (const auto &link : run.soft_links)
    {
      ++retained_soft_links;
      retained_durable_soft_links += link.anchor_label == "durable" ? 1 : 0;
    }

  auto recovery = [&] (int allowed_controls) {
    std::vector<double> thresholds = control_scores;
    thresholds.push_back (std::numeric_limits<double>::infinity ());
    thresholds.push_back (-std::numeric_limits<double>::infinity ());
    int best = 0;
    for (double threshold : thresholds)
      {
        int control_hits = 0;
        for (double score : control_scores)
          {
            if (score >= threshold)
              {
                ++control_hits;
              }
          }
        if (control_hits > allowed_controls)
          {
            continue;
          }
        int recovered = 0;
        for (double score : successful_ref_scores)
          {
            if (score >= threshold)
              {
                ++recovered;
              }
          }
        best = std::max (best, recovered);
      }
    return best;
  };

  return {
    { "reference_count", refs },
    { "control_count", controls },
    { "stored_memory_count", run.links.size () },
    { "anchor_state_count", run.states.size () },
    { "update_count", update_count },
    { "soft_update_count", soft_update_count },
    { "create_count", create_count },
    { "split_count", split_count },
    { "abstain_count", abstain_count },
    { "target_anchor_existed_count", target_existed },
    { "target_anchor_existed_rate",
      refs == 0 ? 0.0 : static_cast<double> (target_existed) / refs },
    { "selected_target_count", selected_target },
    { "selected_target_rate",
      refs == 0 ? 0.0 : static_cast<double> (selected_target) / refs },
    { "tentative_target_top1_count", target_top1 },
    { "tentative_target_top1_rate",
      refs == 0 ? 0.0 : static_cast<double> (target_top1) / refs },
    { "tentative_target_top2_count", target_top2 },
    { "tentative_target_top2_rate",
      refs == 0 ? 0.0 : static_cast<double> (target_top2) / refs },
    { "tentative_target_top3_count", target_top3 },
    { "tentative_target_top3_rate",
      refs == 0 ? 0.0 : static_cast<double> (target_top3) / refs },
    { "useful_uncertain_reference_count", useful_uncertain },
    { "useful_uncertain_reference_rate",
      refs == 0 ? 0.0 : static_cast<double> (useful_uncertain) / refs },
    { "selected_wrong_active_count", selected_wrong },
    { "selected_wrong_active_rate",
      refs == 0 ? 0.0 : static_cast<double> (selected_wrong) / refs },
    { "selected_stale_count", selected_stale },
    { "selected_stale_rate",
      refs == 0 ? 0.0 : static_cast<double> (selected_stale) / refs },
    { "no_anchor_abstain_count", no_anchor_abstain },
    { "no_anchor_abstain_rate",
      controls == 0 ? 0.0
                    : static_cast<double> (no_anchor_abstain) / controls },
    { "no_anchor_hard_bind_count", no_anchor_hard_bind },
    { "no_anchor_hard_bind_rate",
      controls == 0 ? 0.0
                    : static_cast<double> (no_anchor_hard_bind) / controls },
    { "hard_commit_count", hard_commit_count },
    { "hard_commit_rate",
      run.cases.empty ()
          ? 0.0
          : static_cast<double> (hard_commit_count) / run.cases.size () },
    { "hard_wrong_commit_count", hard_wrong_commit_count },
    { "hard_wrong_commit_rate",
      run.cases.empty ()
          ? 0.0
          : static_cast<double> (hard_wrong_commit_count) / run.cases.size () },
    { "selected_tentative_count", selected_tentative },
    { "selected_ambiguous_count", selected_ambiguous },
    { "selected_durable_count", selected_durable },
    { "retained_soft_link_count", retained_soft_links },
    { "retained_durable_soft_link_count", retained_durable_soft_links },
    { "no_anchor_durable_bind_count", no_anchor_durable_bind },
    { "wrong_active_distinct_count", wrong_distinct },
    { "wrong_active_distinct_rate",
      refs == 0 ? 0.0 : static_cast<double> (wrong_distinct) / refs },
    { "stale_inactive_count", stale_inactive },
    { "stale_inactive_rate",
      refs == 0 ? 0.0 : static_cast<double> (stale_inactive) / refs },
    { "reference_vs_no_anchor_auc", Auc (case_scores, case_labels) },
    { "zero_fpr_recovery", recovery (0) },
    { "five_pct_fpr_recovery",
      recovery (static_cast<int> (std::floor (
          0.05 * static_cast<double> (std::max (0, controls)))) ) },
    { "mean_encode_ms", run.mean_encode_ms },
    { "p95_encode_ms", run.p95_encode_ms },
	  };
}

bool
SoftConsumptionLabelAllowed (const SoftAnchorConsumptionPolicy &policy,
                             const std::string &label)
{
  if (label == "durable")
    {
      return policy.allow_durable;
    }
  if (label == "ambiguous")
    {
      return policy.allow_ambiguous;
    }
  if (label == "tentative")
    {
      return policy.allow_tentative;
    }
  return false;
}

bool
SoftConsumptionTierAllowed (const SoftAnchorConsumptionPolicy &policy,
                            const std::string &tier)
{
  if (tier == "wm")
    {
      return policy.allow_wm;
    }
  if (tier == "stm")
    {
      return policy.allow_stm;
    }
  if (tier == "ltm")
    {
      return policy.allow_ltm;
    }
  return false;
}

int
EstimatedSoftAnchorContextChars (const AdaptiveAnchorCandidate &candidate)
{
  const int text_chars = static_cast<int> (
      std::min<std::size_t> (candidate.last_text.size (), 160));
  return 48 + text_chars;
}

std::vector<const AdaptiveAnchorCandidate *>
SelectSoftAnchorConsumptionCandidates (
    const AdaptiveAnchorCaseRow &case_row,
    const SoftAnchorConsumptionPolicy &policy)
{
  std::vector<const AdaptiveAnchorCandidate *> selected;
  if (policy.max_links <= 0)
    {
      return selected;
    }
  if (case_row.decision.generic_score > policy.max_generic)
    {
      return selected;
    }
  if (case_row.decision.entropy > policy.max_entropy)
    {
      return selected;
    }
  if (policy.min_margin >= 0.0 && case_row.decision.margin < policy.min_margin)
    {
      return selected;
    }
  for (const auto &candidate : case_row.decision.candidates)
    {
      if (!candidate.retained_soft_link)
        {
          continue;
        }
      if (!SoftConsumptionTierAllowed (policy, candidate.tier))
        {
          continue;
        }
      if (!SoftConsumptionLabelAllowed (policy, candidate.anchor_label))
        {
          continue;
        }
      if (candidate.anchor_strength < policy.min_strength)
        {
          continue;
        }
      if (candidate.score < policy.min_score)
        {
          continue;
        }
      if (candidate.attention < policy.min_posterior)
        {
          continue;
        }
      selected.push_back (&candidate);
      if (static_cast<int> (selected.size ()) >= policy.max_links)
        {
          break;
        }
    }
  return selected;
}

SoftAnchorConsumptionRun
RunSoftAnchorConsumptionPolicy (const AdaptiveAnchorRun &formation_run,
                                const SoftAnchorConsumptionPolicy &policy)
{
  SoftAnchorConsumptionRun run;
  run.policy = policy;
  for (const auto &case_row : formation_run.cases)
    {
      const auto selected = SelectSoftAnchorConsumptionCandidates (case_row,
                                                                   policy);
      SoftAnchorConsumptionCaseRow row;
      row.consumer_policy = policy.name;
      row.formation_policy = formation_run.policy.name;
      row.meta = case_row.meta;
      row.current_signal_text = case_row.current_signal_text;
      row.margin = case_row.decision.margin;
      row.entropy = case_row.decision.entropy;
      row.generic_score = case_row.decision.generic_score;
      row.surfaced_link_count = static_cast<int> (selected.size ());
      row.surfaced_any = !selected.empty ();
      row.ambiguous_context = selected.size () > 1;

      for (std::size_t i = 0; i < selected.size (); ++i)
        {
          const auto &candidate = *selected[i];
          const int chars = EstimatedSoftAnchorContextChars (candidate);
          row.estimated_context_chars += chars;
          const bool is_target =
              !case_row.target_anchor_id.empty ()
              && candidate.anchor_id == case_row.target_anchor_id;
          const bool is_wrong =
              !case_row.wrong_active_anchor_id.empty ()
              && candidate.anchor_id == case_row.wrong_active_anchor_id
              && candidate.anchor_id != case_row.target_anchor_id;
          const bool is_stale =
              !case_row.stale_anchor_id.empty ()
              && candidate.anchor_id == case_row.stale_anchor_id
              && candidate.anchor_id != case_row.target_anchor_id;
          row.surfaced_target = row.surfaced_target || is_target;
          row.surfaced_wrong_active = row.surfaced_wrong_active || is_wrong;
          row.surfaced_stale = row.surfaced_stale || is_stale;
          if (i == 0)
            {
              row.top_anchor_id = candidate.anchor_id;
              row.top_anchor_label = candidate.anchor_label;
              row.top_tier = candidate.tier;
              row.top_strength = candidate.anchor_strength;
              row.top_score = candidate.score;
              row.top_posterior = candidate.attention;
              row.surfaced_target_top1 = is_target;
            }

          SoftAnchorConsumptionSurfaceRow surface;
          surface.consumer_policy = policy.name;
          surface.formation_policy = formation_run.policy.name;
          surface.meta = case_row.meta;
          surface.current_signal_text = case_row.current_signal_text;
          surface.surface_rank = static_cast<int> (i + 1);
          surface.anchor_id = candidate.anchor_id;
          surface.tier = candidate.tier;
          surface.score = candidate.score;
          surface.posterior = candidate.attention;
          surface.anchor_strength = candidate.anchor_strength;
          surface.anchor_label = candidate.anchor_label;
          surface.margin = case_row.decision.margin;
          surface.entropy = case_row.decision.entropy;
          surface.generic_score = case_row.decision.generic_score;
          surface.is_target = is_target;
          surface.is_wrong_active = is_wrong;
          surface.is_stale = is_stale;
          surface.estimated_context_chars = chars;
          run.surfaces.push_back (std::move (surface));
        }

      row.no_anchor_surfaced = !row.meta.is_reference && row.surfaced_any;
      row.surfaced_wrong_without_target =
          row.meta.is_reference && row.surfaced_wrong_active
          && !row.surfaced_target;
      row.surfaced_stale_without_target =
          row.meta.is_reference && row.surfaced_stale && !row.surfaced_target;
      row.useful_context = row.meta.is_reference && row.surfaced_target;
      row.harmful_context =
          row.no_anchor_surfaced || row.surfaced_wrong_without_target
          || row.surfaced_stale_without_target;
      run.cases.push_back (std::move (row));
    }
  return run;
}

Json
SummarizeSoftAnchorConsumptionRun (const SoftAnchorConsumptionRun &run)
{
  int refs = 0;
  int controls = 0;
  int surfaced_any = 0;
  int target_surfaced = 0;
  int target_top1 = 0;
  int wrong_surfaced = 0;
  int wrong_without_target = 0;
  int stale_surfaced = 0;
  int stale_without_target = 0;
  int no_anchor_surfaced = 0;
  int useful = 0;
  int harmful = 0;
  int ambiguous = 0;
  int durable_surfaced = 0;
  int tentative_surfaced = 0;
  int ambiguous_surfaced = 0;
  int wm_surfaces = 0;
  int stm_surfaces = 0;
  int ltm_surfaces = 0;
  std::vector<double> surfaced_counts;
  std::vector<double> context_chars;
  std::vector<double> control_scores;
  std::vector<double> successful_ref_scores;
  for (const auto &row : run.cases)
    {
      if (row.meta.is_reference)
        {
          ++refs;
          target_surfaced += row.surfaced_target ? 1 : 0;
          target_top1 += row.surfaced_target_top1 ? 1 : 0;
          wrong_surfaced += row.surfaced_wrong_active ? 1 : 0;
          wrong_without_target += row.surfaced_wrong_without_target ? 1 : 0;
          stale_surfaced += row.surfaced_stale ? 1 : 0;
          stale_without_target += row.surfaced_stale_without_target ? 1 : 0;
          successful_ref_scores.push_back (row.surfaced_target ? row.top_score
                                                               : -1.0);
        }
      else
        {
          ++controls;
          no_anchor_surfaced += row.no_anchor_surfaced ? 1 : 0;
          control_scores.push_back (row.surfaced_any ? row.top_score : -1.0);
        }
      surfaced_any += row.surfaced_any ? 1 : 0;
      useful += row.useful_context ? 1 : 0;
      harmful += row.harmful_context ? 1 : 0;
      ambiguous += row.ambiguous_context ? 1 : 0;
      surfaced_counts.push_back (static_cast<double> (row.surfaced_link_count));
      context_chars.push_back (static_cast<double> (row.estimated_context_chars));
    }
  for (const auto &surface : run.surfaces)
    {
      durable_surfaced += surface.anchor_label == "durable" ? 1 : 0;
      tentative_surfaced += surface.anchor_label == "tentative" ? 1 : 0;
      ambiguous_surfaced += surface.anchor_label == "ambiguous" ? 1 : 0;
      wm_surfaces += surface.tier == "wm" ? 1 : 0;
      stm_surfaces += surface.tier == "stm" ? 1 : 0;
      ltm_surfaces += surface.tier == "ltm" ? 1 : 0;
    }

  auto recovery = [&] (int allowed_controls) {
    std::vector<double> thresholds = control_scores;
    thresholds.push_back (std::numeric_limits<double>::infinity ());
    thresholds.push_back (-std::numeric_limits<double>::infinity ());
    int best = 0;
    for (double threshold : thresholds)
      {
        int control_hits = 0;
        for (double score : control_scores)
          {
            if (score >= threshold)
              {
                ++control_hits;
              }
          }
        if (control_hits > allowed_controls)
          {
            continue;
          }
        int recovered = 0;
        for (double score : successful_ref_scores)
          {
            if (score >= threshold)
              {
                ++recovered;
              }
          }
        best = std::max (best, recovered);
      }
    return best;
  };

  return {
    { "formation_policy", run.policy.formation_policy },
    { "max_links", run.policy.max_links },
    { "allow_tentative", run.policy.allow_tentative },
    { "allow_ambiguous", run.policy.allow_ambiguous },
    { "allow_durable", run.policy.allow_durable },
    { "allow_wm", run.policy.allow_wm },
    { "allow_stm", run.policy.allow_stm },
    { "allow_ltm", run.policy.allow_ltm },
    { "min_strength", run.policy.min_strength },
    { "min_score", run.policy.min_score },
    { "min_posterior", run.policy.min_posterior },
    { "min_margin", run.policy.min_margin },
    { "max_entropy", run.policy.max_entropy },
    { "max_generic", run.policy.max_generic },
    { "focus", run.policy.focus },
    { "sensitivity", run.policy.sensitivity },
    { "stability", run.policy.stability },
    { "reference_count", refs },
    { "control_count", controls },
    { "surface_case_count", surfaced_any },
    { "surface_case_rate",
      run.cases.empty () ? 0.0
                         : static_cast<double> (surfaced_any) / run.cases.size () },
    { "target_surfaced_count", target_surfaced },
    { "target_surfaced_rate",
      refs == 0 ? 0.0 : static_cast<double> (target_surfaced) / refs },
    { "target_top1_surfaced_count", target_top1 },
    { "target_top1_surfaced_rate",
      refs == 0 ? 0.0 : static_cast<double> (target_top1) / refs },
    { "wrong_active_surfaced_count", wrong_surfaced },
    { "wrong_active_surfaced_rate",
      refs == 0 ? 0.0 : static_cast<double> (wrong_surfaced) / refs },
    { "wrong_active_without_target_count", wrong_without_target },
    { "wrong_active_without_target_rate",
      refs == 0 ? 0.0 : static_cast<double> (wrong_without_target) / refs },
    { "stale_surfaced_count", stale_surfaced },
    { "stale_surfaced_rate",
      refs == 0 ? 0.0 : static_cast<double> (stale_surfaced) / refs },
    { "stale_without_target_count", stale_without_target },
    { "stale_without_target_rate",
      refs == 0 ? 0.0 : static_cast<double> (stale_without_target) / refs },
    { "no_anchor_surfaced_count", no_anchor_surfaced },
    { "no_anchor_surfaced_rate",
      controls == 0 ? 0.0
                    : static_cast<double> (no_anchor_surfaced) / controls },
    { "useful_context_count", useful },
    { "useful_context_rate",
      refs == 0 ? 0.0 : static_cast<double> (useful) / refs },
    { "harmful_context_count", harmful },
    { "harmful_context_rate",
      run.cases.empty () ? 0.0
                         : static_cast<double> (harmful) / run.cases.size () },
    { "useful_to_harmful_ratio",
      static_cast<double> (useful) / static_cast<double> (std::max (1, harmful)) },
    { "ambiguous_context_count", ambiguous },
    { "ambiguous_context_rate",
      run.cases.empty () ? 0.0
                         : static_cast<double> (ambiguous) / run.cases.size () },
    { "surfaced_link_count", run.surfaces.size () },
    { "surfaced_tentative_link_count", tentative_surfaced },
    { "surfaced_ambiguous_link_count", ambiguous_surfaced },
    { "surfaced_durable_link_count", durable_surfaced },
    { "wm_surface_count", wm_surfaces },
    { "stm_surface_count", stm_surfaces },
    { "ltm_surface_count", ltm_surfaces },
    { "mean_surfaced_links", Mean (surfaced_counts) },
    { "p95_surfaced_links", Percentile (surfaced_counts, 0.95) },
    { "mean_context_chars", Mean (context_chars) },
    { "p95_context_chars", Percentile (context_chars, 0.95) },
    { "zero_fpr_recovery", recovery (0) },
    { "five_pct_fpr_recovery",
      recovery (static_cast<int> (std::floor (
          0.05 * static_cast<double> (std::max (0, controls)))) ) },
  };
}

void
WriteAdaptiveCasesCsv (const std::filesystem::path &path,
                       const std::vector<AdaptiveAnchorRun> &runs)
{
  std::ofstream out (path);
  out << "policy,case_id,dataset,conversation_id,label_class,is_reference,"
         "current_ingress_step,current_signal_text,target_turn,"
         "target_distance,action,selected_anchor_id,selected_tier,"
         "selected_score,selected_attention,anchor_strength,anchor_label,"
         "margin,entropy,p_none,p_new,p_top_real,generic_score,entity_cosine,"
         "semantic_cosine,event_cosine,split_pressure,confidence_before,"
         "support_before,contradiction_before,wm_mass,stm_mass,ltm_mass,"
         "target_anchor_id,wrong_active_anchor_id,stale_anchor_id,"
         "target_anchor_existed,selected_target,selected_wrong_active,"
         "selected_stale,no_anchor_abstained,wrong_active_distinct,"
         "stale_inactive,target_candidate_rank,wrong_active_candidate_rank,"
         "stale_candidate_rank,target_in_top2,target_in_top3,"
         "wrong_active_in_top3,hard_commit,hard_wrong_commit,"
         "useful_uncertain_reference,retrieved_candidate_count,"
         "uses_retrieved_candidates,runtime_policy_uses_labels\n";
  for (const auto &run : runs)
    {
      for (const auto &row : run.cases)
        {
          const auto &d = row.decision;
          out << CsvEscape (run.policy.name) << ','
              << CsvEscape (row.meta.case_id) << ','
              << CsvEscape (row.meta.dataset) << ','
              << CsvEscape (row.meta.conversation_id) << ','
              << CsvEscape (row.meta.label_class) << ','
              << (row.meta.is_reference ? 1 : 0) << ','
              << row.meta.query_turn << ','
              << CsvEscape (row.current_signal_text) << ','
              << row.meta.target_turn << ',' << row.meta.target_distance << ','
              << CsvEscape (d.action) << ','
              << CsvEscape (d.selected_anchor_id) << ','
              << CsvEscape (d.selected_tier) << ',' << d.selected_score << ','
              << d.selected_attention << ','
              << row.selected_anchor_strength << ','
              << CsvEscape (row.selected_anchor_label) << ','
              << d.margin << ',' << d.entropy << ',' << d.p_none << ','
              << d.p_new << ',' << d.p_top_real << ',' << d.generic_score
              << ',' << d.entity_cosine << ',' << d.semantic_cosine << ','
              << d.event_cosine << ',' << d.split_pressure << ','
              << d.confidence_before << ',' << d.support_before << ','
              << d.contradiction_before << ',' << d.wm_mass << ','
              << d.stm_mass << ',' << d.ltm_mass << ','
              << CsvEscape (row.target_anchor_id) << ','
              << CsvEscape (row.wrong_active_anchor_id) << ','
              << CsvEscape (row.stale_anchor_id) << ','
              << (row.target_anchor_existed ? 1 : 0) << ','
              << (row.selected_target ? 1 : 0) << ','
              << (row.selected_wrong_active ? 1 : 0) << ','
              << (row.selected_stale ? 1 : 0) << ','
              << (row.no_anchor_abstained ? 1 : 0) << ','
              << (row.wrong_active_distinct ? 1 : 0) << ','
              << (row.stale_inactive ? 1 : 0) << ','
              << row.target_candidate_rank << ','
              << row.wrong_active_candidate_rank << ','
              << row.stale_candidate_rank << ','
              << (row.target_in_top2 ? 1 : 0) << ','
              << (row.target_in_top3 ? 1 : 0) << ','
              << (row.wrong_active_in_top3 ? 1 : 0) << ','
              << (row.hard_commit ? 1 : 0) << ','
              << (row.hard_wrong_commit ? 1 : 0) << ','
              << (row.useful_uncertain_reference ? 1 : 0)
              << ",0,0,0\n";
        }
    }
}

void
WriteAdaptiveManualAuditCsv (const std::filesystem::path &path,
                             const std::vector<AdaptiveAnchorRun> &runs)
{
  std::ofstream out (path);
  out << "policy,case_id,dataset,conversation_id,label_class,is_reference,"
         "current_ingress_step,current_signal_text,target_turn,"
         "target_candidate_rank,wrong_active_candidate_rank,"
         "suggested_acceptability,manual_verdict,manual_notes,action,"
         "anchor_strength,anchor_label,hard_commit,hard_wrong_commit,"
         "top1_anchor_id,top1_score,top1_attention,top1_strength,"
         "top1_label,top1_last_seen_turn,top1_last_seen_text,"
         "top2_anchor_id,top2_score,top2_attention,top2_strength,"
         "top2_label,top2_last_seen_turn,"
         "top2_last_seen_text,top3_anchor_id,top3_score,top3_attention,"
         "top3_strength,top3_label,top3_last_seen_turn,"
         "top3_last_seen_text\n";

  for (const auto &run : runs)
    {
      for (const auto &row : run.cases)
        {
          std::string suggested = "needs_review";
          if (row.meta.is_reference && row.selected_target)
            {
              suggested = "acceptable_hard_bind";
            }
          else if (row.meta.is_reference && row.useful_uncertain_reference)
            {
              suggested = "acceptable_uncertain_top3";
            }
          else if (row.hard_wrong_commit)
            {
              suggested = "unacceptable_wrong_hard_commit";
            }
          else if (!row.meta.is_reference && row.no_anchor_abstained)
            {
              suggested = "acceptable_control_no_hard_bind";
            }
          else if (row.meta.is_reference)
            {
              suggested = "reference_not_recovered";
            }

          const auto write_candidate = [&] (std::size_t index) {
            if (index >= row.decision.candidates.size ())
              {
                out << ",,,,,,";
                return;
              }
            const auto &candidate = row.decision.candidates[index];
            out << CsvEscape (candidate.anchor_id) << ',' << candidate.score
                << ',' << candidate.attention << ','
                << candidate.anchor_strength << ','
                << CsvEscape (candidate.anchor_label) << ','
                << candidate.last_seen_turn << ','
                << CsvEscape (candidate.last_text);
          };

          out << CsvEscape (run.policy.name) << ','
              << CsvEscape (row.meta.case_id) << ','
              << CsvEscape (row.meta.dataset) << ','
              << CsvEscape (row.meta.conversation_id) << ','
              << CsvEscape (row.meta.label_class) << ','
              << (row.meta.is_reference ? 1 : 0) << ','
              << row.meta.query_turn << ','
              << CsvEscape (row.current_signal_text) << ','
              << row.meta.target_turn << ',' << row.target_candidate_rank
              << ',' << row.wrong_active_candidate_rank << ','
              << CsvEscape (suggested) << ",,,"
              << CsvEscape (row.decision.action) << ','
              << row.selected_anchor_strength << ','
              << CsvEscape (row.selected_anchor_label) << ','
              << (row.hard_commit ? 1 : 0) << ','
              << (row.hard_wrong_commit ? 1 : 0) << ',';
          write_candidate (0);
          out << ',';
          write_candidate (1);
          out << ',';
          write_candidate (2);
          out << '\n';
        }
    }
}

void
WriteAdaptiveLinksCsv (const std::filesystem::path &path,
                       const std::vector<AdaptiveAnchorRun> &runs)
{
  std::ofstream out (path);
  out << "policy,dataset,conversation_id,turn,memory_id,action,anchor_id,"
         "selected_tier,selected_score,margin,entropy,p_none,p_new,"
         "p_top_real,generic_score,split_pressure,"
         "anchor_strength,anchor_label,confidence_after,support_after,"
         "contradiction_after,status_after,"
         "created_before_any_future_query\n";
  for (const auto &run : runs)
    {
      for (const auto &link : run.links)
        {
          out << CsvEscape (run.policy.name) << ','
              << CsvEscape (link.dataset) << ','
              << CsvEscape (link.conversation_id) << ',' << link.turn << ','
              << CsvEscape (link.memory_id) << ',' << CsvEscape (link.action)
              << ',' << CsvEscape (link.anchor_id) << ','
              << CsvEscape (link.selected_tier) << ',' << link.selected_score
              << ',' << link.margin << ',' << link.entropy << ','
              << link.p_none << ',' << link.p_new << ','
              << link.p_top_real << ',' << link.generic_score << ','
              << link.split_pressure << ',' << link.anchor_strength << ','
              << CsvEscape (link.anchor_label) << ','
              << link.confidence_after << ','
              << link.support_after << ',' << link.contradiction_after << ','
              << CsvEscape (link.status_after) << ",1\n";
        }
    }
}

void
WriteAdaptiveSoftLinksCsv (const std::filesystem::path &path,
                           const std::vector<AdaptiveAnchorRun> &runs)
{
  std::ofstream out (path);
  out << "policy,dataset,conversation_id,turn,memory_id,anchor_id,tier,rank,"
         "score,posterior,margin,entropy,p_none,p_new,p_top_real,"
         "generic_score,anchor_strength,anchor_label,entity_cosine,"
         "semantic_cosine,event_cosine,created_before_any_future_query\n";
  for (const auto &run : runs)
    {
      for (const auto &link : run.soft_links)
        {
          out << CsvEscape (run.policy.name) << ','
              << CsvEscape (link.dataset) << ','
              << CsvEscape (link.conversation_id) << ',' << link.turn << ','
              << CsvEscape (link.memory_id) << ','
              << CsvEscape (link.anchor_id) << ','
              << CsvEscape (link.tier) << ',' << link.rank << ','
              << link.score << ',' << link.posterior << ',' << link.margin
              << ',' << link.entropy << ',' << link.p_none << ','
              << link.p_new << ',' << link.p_top_real << ','
              << link.generic_score << ',' << link.anchor_strength << ','
              << CsvEscape (link.anchor_label) << ','
              << link.entity_cosine << ',' << link.semantic_cosine << ','
              << link.event_cosine << ",1\n";
        }
    }
}

void
WriteAdaptiveStatesCsv (const std::filesystem::path &path,
                        const std::vector<AdaptiveAnchorRun> &runs)
{
  std::ofstream out (path);
  out << "policy,dataset,conversation_id,anchor_id,first_seen_turn,"
         "last_seen_turn,support_count,contradiction_count,confidence,"
         "stability,anchor_strength,anchor_label,status\n";
  for (const auto &run : runs)
    {
      for (const auto &state : run.states)
        {
          out << CsvEscape (run.policy.name) << ','
              << CsvEscape (state.dataset) << ','
              << CsvEscape (state.conversation_id) << ','
              << CsvEscape (state.anchor_id) << ',' << state.first_seen_turn
              << ',' << state.last_seen_turn << ',' << state.support_count
              << ',' << state.contradiction_count << ',' << state.confidence
              << ',' << state.stability << ',' << state.anchor_strength << ','
              << CsvEscape (state.anchor_label) << ','
              << CsvEscape (state.status)
              << '\n';
        }
    }
}

void
WriteAdaptiveFailuresCsv (const std::filesystem::path &path,
                          const std::vector<AdaptiveAnchorRun> &runs)
{
  std::ofstream out (path);
  out << "policy,case_id,label_class,failure_type,current_ingress_step,"
         "current_signal_text,target_turn,action,selected_anchor_id,"
         "selected_tier,selected_score,anchor_strength,anchor_label,"
         "margin,entropy,split_pressure,"
         "target_anchor_id,wrong_active_anchor_id,stale_anchor_id\n";
  int count = 0;
  for (const auto &run : runs)
    {
      for (const auto &row : run.cases)
        {
          std::string failure;
          if (row.meta.is_reference && !row.selected_target)
            {
              failure = row.selected_wrong_active ? "selected_wrong_active"
                                                  : "missed_target_anchor";
            }
          else if (!row.meta.is_reference && !row.no_anchor_abstained)
            {
              failure = "no_anchor_false_bind";
            }
          if (failure.empty ())
            {
              continue;
            }
          const auto &d = row.decision;
          out << CsvEscape (run.policy.name) << ','
              << CsvEscape (row.meta.case_id) << ','
              << CsvEscape (row.meta.label_class) << ','
              << CsvEscape (failure) << ',' << row.meta.query_turn << ','
              << CsvEscape (row.current_signal_text) << ','
              << row.meta.target_turn << ',' << CsvEscape (d.action) << ','
              << CsvEscape (d.selected_anchor_id) << ','
              << CsvEscape (d.selected_tier) << ',' << d.selected_score << ','
              << row.selected_anchor_strength << ','
              << CsvEscape (row.selected_anchor_label) << ','
              << d.margin << ',' << d.entropy << ',' << d.split_pressure
              << ',' << CsvEscape (row.target_anchor_id) << ','
              << CsvEscape (row.wrong_active_anchor_id) << ','
              << CsvEscape (row.stale_anchor_id) << '\n';
          if (++count >= 240)
            {
              return;
            }
	    }
	}
}

void
WriteSoftAnchorConsumptionCasesCsv (
    const std::filesystem::path &path,
    const std::vector<SoftAnchorConsumptionRun> &runs)
{
  std::ofstream out (path);
  out << "consumer_policy,formation_policy,case_id,dataset,conversation_id,"
         "label_class,is_reference,current_ingress_step,current_signal_text,"
         "target_turn,target_distance,surfaced_link_count,"
         "estimated_context_chars,top_anchor_id,top_anchor_label,top_tier,"
         "top_strength,top_score,top_posterior,margin,entropy,generic_score,"
         "surfaced_any,surfaced_target,surfaced_target_top1,"
         "surfaced_wrong_active,surfaced_stale,"
         "surfaced_wrong_without_target,surfaced_stale_without_target,"
         "no_anchor_surfaced,useful_context,harmful_context,"
         "ambiguous_context,runtime_policy_uses_labels\n";
  for (const auto &run : runs)
    {
      for (const auto &row : run.cases)
        {
          out << CsvEscape (row.consumer_policy) << ','
              << CsvEscape (row.formation_policy) << ','
              << CsvEscape (row.meta.case_id) << ','
              << CsvEscape (row.meta.dataset) << ','
              << CsvEscape (row.meta.conversation_id) << ','
              << CsvEscape (row.meta.label_class) << ','
              << (row.meta.is_reference ? 1 : 0) << ',' << row.meta.query_turn
              << ',' << CsvEscape (row.current_signal_text) << ','
              << row.meta.target_turn << ',' << row.meta.target_distance << ','
              << row.surfaced_link_count << ',' << row.estimated_context_chars
              << ',' << CsvEscape (row.top_anchor_id) << ','
              << CsvEscape (row.top_anchor_label) << ','
              << CsvEscape (row.top_tier) << ',' << row.top_strength << ','
              << row.top_score << ',' << row.top_posterior << ','
              << row.margin << ',' << row.entropy << ',' << row.generic_score
              << ',' << (row.surfaced_any ? 1 : 0) << ','
              << (row.surfaced_target ? 1 : 0) << ','
              << (row.surfaced_target_top1 ? 1 : 0) << ','
              << (row.surfaced_wrong_active ? 1 : 0) << ','
              << (row.surfaced_stale ? 1 : 0) << ','
              << (row.surfaced_wrong_without_target ? 1 : 0) << ','
              << (row.surfaced_stale_without_target ? 1 : 0) << ','
              << (row.no_anchor_surfaced ? 1 : 0) << ','
              << (row.useful_context ? 1 : 0) << ','
              << (row.harmful_context ? 1 : 0) << ','
              << (row.ambiguous_context ? 1 : 0) << ",0\n";
        }
    }
}

void
WriteSoftAnchorConsumptionSurfacesCsv (
    const std::filesystem::path &path,
    const std::vector<SoftAnchorConsumptionRun> &runs)
{
  std::ofstream out (path);
  out << "consumer_policy,formation_policy,case_id,dataset,conversation_id,"
         "label_class,is_reference,current_ingress_step,surface_rank,anchor_id,"
         "tier,score,posterior,anchor_strength,anchor_label,margin,entropy,"
         "generic_score,is_target,is_wrong_active,is_stale,"
         "estimated_context_chars,current_signal_text\n";
  for (const auto &run : runs)
    {
      for (const auto &surface : run.surfaces)
        {
          out << CsvEscape (surface.consumer_policy) << ','
              << CsvEscape (surface.formation_policy) << ','
              << CsvEscape (surface.meta.case_id) << ','
              << CsvEscape (surface.meta.dataset) << ','
              << CsvEscape (surface.meta.conversation_id) << ','
              << CsvEscape (surface.meta.label_class) << ','
              << (surface.meta.is_reference ? 1 : 0) << ','
              << surface.meta.query_turn << ',' << surface.surface_rank << ','
              << CsvEscape (surface.anchor_id) << ','
              << CsvEscape (surface.tier) << ',' << surface.score << ','
              << surface.posterior << ',' << surface.anchor_strength << ','
              << CsvEscape (surface.anchor_label) << ',' << surface.margin << ','
              << surface.entropy << ',' << surface.generic_score << ','
              << (surface.is_target ? 1 : 0) << ','
              << (surface.is_wrong_active ? 1 : 0) << ','
              << (surface.is_stale ? 1 : 0) << ','
              << surface.estimated_context_chars << ','
              << CsvEscape (surface.current_signal_text) << '\n';
        }
    }
}

void
WriteSoftAnchorConsumptionFailuresCsv (
    const std::filesystem::path &path,
    const std::vector<SoftAnchorConsumptionRun> &runs)
{
  std::ofstream out (path);
  out << "consumer_policy,formation_policy,case_id,label_class,failure_type,"
         "current_ingress_step,current_signal_text,target_turn,"
         "surfaced_link_count,top_anchor_id,top_anchor_label,top_tier,"
         "top_strength,top_score,margin,entropy,generic_score,"
         "surfaced_target,surfaced_wrong_active,surfaced_stale,"
         "no_anchor_surfaced,estimated_context_chars\n";
  int count = 0;
  for (const auto &run : runs)
    {
      for (const auto &row : run.cases)
        {
          std::string failure;
          if (row.meta.is_reference && !row.surfaced_target)
            {
              failure = "target_not_surfaced";
            }
          else if (row.surfaced_wrong_without_target)
            {
              failure = "wrong_active_surfaced_without_target";
            }
          else if (row.surfaced_stale_without_target)
            {
              failure = "stale_surfaced_without_target";
            }
          else if (row.no_anchor_surfaced)
            {
              failure = "no_anchor_context_surfaced";
            }
          if (failure.empty ())
            {
              continue;
            }
          out << CsvEscape (row.consumer_policy) << ','
              << CsvEscape (row.formation_policy) << ','
              << CsvEscape (row.meta.case_id) << ','
              << CsvEscape (row.meta.label_class) << ','
              << CsvEscape (failure) << ',' << row.meta.query_turn << ','
              << CsvEscape (row.current_signal_text) << ','
              << row.meta.target_turn << ',' << row.surfaced_link_count << ','
              << CsvEscape (row.top_anchor_id) << ','
              << CsvEscape (row.top_anchor_label) << ','
              << CsvEscape (row.top_tier) << ',' << row.top_strength << ','
              << row.top_score << ',' << row.margin << ',' << row.entropy << ','
              << row.generic_score << ',' << (row.surfaced_target ? 1 : 0)
              << ',' << (row.surfaced_wrong_active ? 1 : 0) << ','
              << (row.surfaced_stale ? 1 : 0) << ','
              << (row.no_anchor_surfaced ? 1 : 0) << ','
              << row.estimated_context_chars << '\n';
          if (++count >= 360)
            {
              return;
            }
        }
    }
}

void
WriteSoftAnchorConsumptionSummaryCsv (
    const std::filesystem::path &path,
    const std::vector<SoftAnchorConsumptionRun> &runs)
{
  std::ofstream out (path);
  out << "consumer_policy,formation_policy,focus,sensitivity,stability,"
         "max_links,min_strength,min_score,min_posterior,min_margin,"
         "max_entropy,max_generic,reference_count,control_count,"
         "target_surfaced_count,target_surfaced_rate,"
         "target_top1_surfaced_count,target_top1_surfaced_rate,"
         "no_anchor_surfaced_count,no_anchor_surfaced_rate,"
         "wrong_active_without_target_count,wrong_active_without_target_rate,"
         "stale_without_target_count,stale_without_target_rate,"
         "useful_context_count,useful_context_rate,harmful_context_count,"
         "harmful_context_rate,useful_to_harmful_ratio,"
         "ambiguous_context_count,ambiguous_context_rate,"
         "surfaced_link_count,surfaced_tentative_link_count,"
         "surfaced_ambiguous_link_count,surfaced_durable_link_count,"
         "wm_surface_count,stm_surface_count,ltm_surface_count,"
         "mean_surfaced_links,p95_surfaced_links,mean_context_chars,"
         "p95_context_chars,zero_fpr_recovery,five_pct_fpr_recovery\n";
  for (const auto &run : runs)
    {
      const Json s = SummarizeSoftAnchorConsumptionRun (run);
      out << CsvEscape (run.policy.name) << ','
          << CsvEscape (run.policy.formation_policy) << ','
          << run.policy.focus << ',' << run.policy.sensitivity << ','
          << run.policy.stability << ',' << run.policy.max_links << ','
          << run.policy.min_strength << ',' << run.policy.min_score << ','
          << run.policy.min_posterior << ',' << run.policy.min_margin << ','
          << run.policy.max_entropy << ',' << run.policy.max_generic << ','
          << s["reference_count"].get<int> () << ','
          << s["control_count"].get<int> () << ','
          << s["target_surfaced_count"].get<int> () << ','
          << s["target_surfaced_rate"].get<double> () << ','
          << s["target_top1_surfaced_count"].get<int> () << ','
          << s["target_top1_surfaced_rate"].get<double> () << ','
          << s["no_anchor_surfaced_count"].get<int> () << ','
          << s["no_anchor_surfaced_rate"].get<double> () << ','
          << s["wrong_active_without_target_count"].get<int> () << ','
          << s["wrong_active_without_target_rate"].get<double> () << ','
          << s["stale_without_target_count"].get<int> () << ','
          << s["stale_without_target_rate"].get<double> () << ','
          << s["useful_context_count"].get<int> () << ','
          << s["useful_context_rate"].get<double> () << ','
          << s["harmful_context_count"].get<int> () << ','
          << s["harmful_context_rate"].get<double> () << ','
          << s["useful_to_harmful_ratio"].get<double> () << ','
          << s["ambiguous_context_count"].get<int> () << ','
          << s["ambiguous_context_rate"].get<double> () << ','
          << s["surfaced_link_count"].get<int> () << ','
          << s["surfaced_tentative_link_count"].get<int> () << ','
          << s["surfaced_ambiguous_link_count"].get<int> () << ','
          << s["surfaced_durable_link_count"].get<int> () << ','
          << s["wm_surface_count"].get<int> () << ','
          << s["stm_surface_count"].get<int> () << ','
          << s["ltm_surface_count"].get<int> () << ','
          << s["mean_surfaced_links"].get<double> () << ','
          << s["p95_surfaced_links"].get<double> () << ','
          << s["mean_context_chars"].get<double> () << ','
          << s["p95_context_chars"].get<double> () << ','
          << s["zero_fpr_recovery"].get<int> () << ','
          << s["five_pct_fpr_recovery"].get<int> () << '\n';
    }
}

Json
SoftAnchorConsumptionMonotonicity (
    const std::vector<SoftAnchorConsumptionRun> &runs)
{
  struct Point
  {
    double focus = 0.0;
    double sensitivity = 0.0;
    double stability = 0.0;
    Json summary;
  };
  std::vector<Point> points;
  for (const auto &run : runs)
    {
      if (run.policy.focus < 0.0 || run.policy.sensitivity < 0.0
          || run.policy.stability < 0.0)
        {
          continue;
        }
      points.push_back ({ run.policy.focus, run.policy.sensitivity,
                          run.policy.stability,
                          SummarizeSoftAnchorConsumptionRun (run) });
    }

  const auto close = [] (double lhs, double rhs) {
    return std::abs (lhs - rhs) < 0.0001;
  };
  const auto metric = [] (const Point &point, const std::string &key) {
    return point.summary[key].get<double> ();
  };
  const auto check = [&] (const std::string &axis, const std::string &key,
                          bool increasing) {
    int total = 0;
    int pass = 0;
    int changed = 0;
    std::vector<double> values = { 0.25, 0.50, 0.75 };
    for (double a : values)
      {
        for (double b : values)
          {
            std::vector<Point> series;
            for (const auto &point : points)
              {
                if ((axis == "focus"
                     && close (point.sensitivity, a)
                     && close (point.stability, b))
                    || (axis == "sensitivity"
                        && close (point.focus, a)
                        && close (point.stability, b))
                    || (axis == "stability"
                        && close (point.focus, a)
                        && close (point.sensitivity, b)))
                  {
                    series.push_back (point);
                  }
              }
            std::sort (series.begin (), series.end (),
                       [&] (const Point &lhs, const Point &rhs) {
                         if (axis == "focus")
                           {
                             return lhs.focus < rhs.focus;
                           }
                         if (axis == "sensitivity")
                           {
                             return lhs.sensitivity < rhs.sensitivity;
                           }
                         return lhs.stability < rhs.stability;
                       });
            for (std::size_t i = 1; i < series.size (); ++i)
              {
                ++total;
                const double previous = metric (series[i - 1], key);
                const double current = metric (series[i], key);
                if (std::abs (current - previous) > 1.0e-9)
                  {
                    ++changed;
                  }
                if ((increasing && current + 1.0e-9 >= previous)
                    || (!increasing && current <= previous + 1.0e-9))
                  {
                    ++pass;
                  }
              }
          }
      }
    return Json {
      { "axis", axis },
      { "metric", key },
      { "expected_direction", increasing ? "increase" : "decrease" },
      { "adjacent_checks", total },
      { "passing_checks", pass },
      { "changed_checks", changed },
      { "pass_rate", total == 0 ? 0.0 : static_cast<double> (pass) / total },
      { "non_flat_rate",
        total == 0 ? 0.0 : static_cast<double> (changed) / total },
    };
  };

  return {
    { "sweep_point_count", points.size () },
    { "checks",
      Json::array ({
          check ("focus", "mean_surfaced_links", false),
          check ("focus", "harmful_context_rate", false),
          check ("focus", "no_anchor_surfaced_rate", false),
          check ("sensitivity", "target_surfaced_rate", true),
          check ("sensitivity", "mean_surfaced_links", true),
          check ("stability", "target_surfaced_rate", true),
          check ("stability", "mean_surfaced_links", true),
          check ("stability", "ltm_surface_count", true),
      }) },
  };
}

Options
ParseOptions (int argc, char **argv)
{
  Options opts;
  for (int i = 1; i < argc; ++i)
    {
      std::string arg = argv[i];
      if (arg == "--episodes" && i + 1 < argc)
        {
          opts.episodes = std::stoi (argv[++i]);
        }
      else if (arg == "--max-conversations" && i + 1 < argc)
        {
          opts.max_conversations = std::stoi (argv[++i]);
        }
      else if (arg == "--max-turns" && i + 1 < argc)
        {
          opts.max_turns_per_conversation = std::stoi (argv[++i]);
        }
      else if (arg == "--max-cases" && i + 1 < argc)
        {
          opts.max_cases = std::stoi (argv[++i]);
        }
      else if (arg == "--models" && i + 1 < argc)
        {
          opts.models_dir = argv[++i];
        }
      else if (arg == "--teacher-model" && i + 1 < argc)
        {
          opts.teacher_model = argv[++i];
        }
      else if (arg == "--output-dir" && i + 1 < argc)
        {
          opts.output_dir = argv[++i];
        }
      else if (arg == "--help")
        {
          std::cout
              << "Usage: cortext_ingress_anchor_formation_bench "
                 "[--models DIR] [--teacher-model ES_AIST.gguf] "
                 "[--max-conversations N] [--max-turns N] "
                 "[--max-cases N] [--output-dir DIR]\n\n"
              << "Default mode is chronological storage-time ES-AIST "
                 "attention over prior WM/STM/LTM memories. It does not "
                 "score retrieved candidates and does not change production "
                 "retrieval.\n";
          std::exit (0);
        }
    }
  return opts;
}

} // namespace

int
main (int argc, char **argv)
{
  Options opts = ParseOptions (argc, argv);
  std::filesystem::create_directories (opts.output_dir);
  const auto model_path = ResolveEsAistModelPath (opts);
  if (!model_path)
    {
      throw std::runtime_error (
          "ES-AIST ingress benchmark requires ES-AIST GGUF. Use --teacher-model "
          "or place ES-AIST-81M_q8_0.gguf under models/ES-AIST-81M-preview-GGUF/.");
    }

  cortext::AaitGgufConfig config;
  config.model_path = model_path->string ();
  config.context_length = 128;
  cortext::AaitGgufEncoder encoder (config);
  std::unordered_map<std::string, EncodedStep> encoded_cache;
  if (opts.synthetic_chain)
    {
      const std::vector<IngressStep> steps = MakeSteps (opts.episodes);
      for (const auto &step : steps)
        {
          (void)EncodeStep (encoder, encoded_cache, step.text);
        }

      const std::vector<Policy> policies = {
        { "es_aist_loose", 0.30, 0.30, 0.00, false },
        { "es_aist_mid", 0.50, 0.50, 0.00, false },
        { "es_aist_strict", 0.70, 0.70, 0.00, false },
        { "es_aist_mid_margin", 0.50, 0.50, 0.02, true },
        { "es_aist_strict_margin", 0.70, 0.70, 0.02, true },
      };

      std::vector<PolicyRun> runs;
      for (const auto &policy : policies)
        {
          runs.push_back (RunPolicy (steps, encoded_cache, policy));
        }

      WriteRowsCsv (
          opts.output_dir / "ingress_anchor_formation_es_aist_steps.csv",
          runs);
      WriteRowsCsv (
          opts.output_dir / "ingress_anchor_formation_es_aist_audit.csv",
          runs);
      WriteFailuresCsv (
          opts.output_dir
              / "ingress_anchor_formation_es_aist_failure_examples.csv",
          runs);

      Json results = {
        { "mode", "ingress_anchor_formation_es_aist_synthetic_chain" },
        { "runtime_effect", "none" },
        { "production_retrieval_changed", false },
        { "model_path", model_path->string () },
        { "model_backend", encoder.KernelOpsBackend () },
        { "uses_kernel_ops", encoder.UsesKernelOps () },
        { "uses_full_text_graph_ops", encoder.UsesFullTextGraphOps () },
        { "retrieved_candidate_count", 0 },
        { "uses_retrieved_candidates", false },
        { "runtime_policy_uses_labels", false },
        { "episodes", opts.episodes },
        { "steps", steps.size () },
        { "unique_text_encodes", encoded_cache.size () },
        { "variants", Json::object () },
      };
      for (const auto &run : runs)
        {
          results["variants"][run.policy.name] = SummarizeRun (run);
        }
      std::ofstream out (
          opts.output_dir / "ingress_anchor_formation_es_aist_results.json");
      out << std::setw (2) << results << '\n';
      std::cout << std::setw (2) << results << '\n';
      return 0;
    }

  const auto conversations = LoadStorageConversations (opts);
  const auto eval_cases = BuildStorageEvalCases (conversations, opts);
  const std::vector<StorageAttentionPolicy> policies = {
    { "storage_attention_loose", 0.08, 0.60, 0.00, 0.035, 0.020, 0.000,
      false },
    { "storage_attention_mid", 0.08, 0.66, 0.02, 0.030, 0.030, 0.000,
      false },
    { "storage_attention_stm_weighted", 0.08, 0.68, 0.03, 0.010, 0.055,
      0.005, false },
    { "storage_attention_strict", 0.08, 0.72, 0.04, 0.020, 0.035, 0.000,
      false },
    { "storage_attention_high_precision", 0.06, 0.78, 0.06, 0.010, 0.040,
      0.000, false },
  };
  std::vector<AdaptiveAnchorPolicy> adaptive_policies = {
    { "adaptive_evidence_loose", 0.08, 0.58, 0.00, 1.00, 0.22, 0.04, 2,
      32, false },
    { "adaptive_evidence_mid", 0.08, 0.63, 0.015, 0.98, 0.20, 0.05, 2,
      32, false },
    { "adaptive_hysteresis", 0.08, 0.67, 0.025, 0.94, 0.18, 0.06, 2, 32,
      true },
    { "adaptive_split_guard", 0.08, 0.65, 0.025, 0.92, 0.14, 0.08, 2, 24,
      true },
    { "adaptive_high_precision", 0.06, 0.72, 0.045, 0.88, 0.12, 0.10, 3,
      24, true },
    { "adaptive_hp_label_generic_guard", 0.06, 0.72, 0.045, 0.88, 0.12,
      0.10, 3, 24, true, 0.78, 0.72, 0.04, 0.82, 3, true, false },
    { "adaptive_hp_label_event_guard", 0.06, 0.72, 0.045, 0.88, 0.12, 0.10,
      3, 24, true, 0.78, 0.72, 0.04, 0.82, 3, false, true },
    { "adaptive_hp_label_support5", 0.06, 0.72, 0.045, 0.88, 0.12, 0.10,
      3, 24, true, 0.78, 0.72, 0.04, 0.82, 5, false, false },
    { "adaptive_hp_label_margin_guard", 0.06, 0.72, 0.045, 0.88, 0.12,
      0.10, 3, 24, true, 0.90, 0.85, 0.08, 0.70, 3, false, true },
    { "adaptive_hp_label_conservative", 0.06, 0.72, 0.045, 0.88, 0.12,
      0.10, 3, 24, true, 0.90, 0.85, 0.08, 0.70, 5, true, true },
    { "adaptive_knob_low_focus_high_sensitivity", 0.08, 0.60, 0.010, 0.98,
      0.20, 0.05, 2, 32, false, 0.86, 0.80, 0.05, 0.76, 4, true, false },
    { "adaptive_knob_high_focus", 0.05, 0.78, 0.060, 0.82, 0.12, 0.12, 3,
      24, true, 0.90, 0.85, 0.08, 0.70, 5, true, true },
    { "adaptive_knob_high_stability", 0.06, 0.68, 0.040, 0.88, 0.14, 0.08,
      3, 48, true, 0.88, 0.82, 0.07, 0.74, 5, true, true },
    { "adaptive_ultra_precision", 0.05, 0.84, 0.080, 0.78, 0.10, 0.12, 3,
      16, true },
  };
  auto add_adaptive_ablation = [&] (const std::string &name,
                                    const std::string &base_name,
                                    auto mutate) {
    auto found = std::find_if (
        adaptive_policies.begin (), adaptive_policies.end (),
        [&] (const AdaptiveAnchorPolicy &policy) {
          return policy.name == base_name;
        });
    if (found == adaptive_policies.end ())
      {
        throw std::runtime_error ("Missing adaptive ablation base policy: "
                                  + base_name);
      }
    AdaptiveAnchorPolicy policy = *found;
    policy.name = name;
    mutate (policy);
    adaptive_policies.push_back (policy);
  };
  add_adaptive_ablation (
      "adaptive_soft_null_hypothesis", "adaptive_high_precision",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.null_no_anchor_hypothesis = true;
        policy.null_score_threshold = 0.70;
        policy.null_entropy_threshold = 0.88;
      });
  add_adaptive_ablation (
      "adaptive_soft_generic_suppressor", "adaptive_high_precision",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.generic_action_suppressor = true;
        policy.label_demote_generic_to_tentative = true;
      });
  add_adaptive_ablation (
      "adaptive_soft_null_generic", "adaptive_high_precision",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.null_no_anchor_hypothesis = true;
        policy.generic_action_suppressor = true;
        policy.label_demote_generic_to_tentative = true;
        policy.null_score_threshold = 0.70;
        policy.null_entropy_threshold = 0.88;
      });
  add_adaptive_ablation (
      "adaptive_soft_repeated_support_gate", "adaptive_high_precision",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.min_update_support = 2;
        policy.label_durable_support_count = 5;
        policy.label_demote_generic_to_tentative = true;
      });
  add_adaptive_ablation (
      "adaptive_soft_no_contradiction_penalty", "adaptive_high_precision",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.use_contradiction_penalty = false;
        policy.contradiction_penalty = 0.0;
      });
  add_adaptive_ablation (
      "adaptive_soft_no_entropy_margin_gate", "adaptive_high_precision",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.margin_threshold = 0.0;
        policy.max_entropy = 1.0;
        policy.label_durable_margin_threshold = 0.0;
        policy.label_durable_max_entropy = 1.0;
      });
  add_adaptive_ablation (
      "adaptive_view_semantic_only", "adaptive_high_precision",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.semantic_weight = 0.80;
        policy.entity_weight = 0.0;
        policy.full_weight = 0.0;
      });
  add_adaptive_ablation (
      "adaptive_view_entity_only", "adaptive_high_precision",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.semantic_weight = 0.0;
        policy.entity_weight = 0.80;
        policy.full_weight = 0.0;
      });
  add_adaptive_ablation (
      "adaptive_view_full_only", "adaptive_high_precision",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.semantic_weight = 0.0;
        policy.entity_weight = 0.0;
        policy.full_weight = 0.80;
      });
  add_adaptive_ablation (
      "adaptive_view_semantic_entity", "adaptive_high_precision",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.semantic_weight = 0.40;
        policy.entity_weight = 0.40;
        policy.full_weight = 0.0;
      });
  add_adaptive_ablation (
      "adaptive_tier_wm_only", "adaptive_high_precision",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.include_wm = true;
        policy.include_stm = false;
        policy.include_ltm = false;
      });
  add_adaptive_ablation (
      "adaptive_tier_wm_stm", "adaptive_high_precision",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.include_wm = true;
        policy.include_stm = true;
        policy.include_ltm = false;
      });
  add_adaptive_ablation (
      "adaptive_tier_stm_ltm_no_wm", "adaptive_high_precision",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.include_wm = false;
        policy.include_stm = true;
        policy.include_ltm = true;
      });
  add_adaptive_ablation (
      "adaptive_soft_anchor_v1", "adaptive_high_precision",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.soft_hypothesis_mode = true;
        policy.null_no_anchor_hypothesis = true;
        policy.generic_action_suppressor = true;
        policy.posterior_beta = 3.0;
        policy.semantic_weight = 0.34;
        policy.entity_weight = 0.34;
        policy.full_weight = 0.22;
        policy.update_threshold = 0.76;
        policy.margin_threshold = 0.16;
        policy.max_entropy = 0.72;
        policy.soft_update_posterior_threshold = 0.70;
        policy.new_anchor_threshold = 0.42;
        policy.generic_hard_threshold = 0.82;
        policy.posterior_keep_threshold = 0.05;
        policy.tentative_score_threshold = 0.54;
        policy.label_demote_generic_to_tentative = true;
        policy.label_block_event_mismatch_durable = true;
        policy.label_durable_strength_threshold = 0.90;
        policy.label_durable_state_strength_threshold = 0.88;
        policy.label_durable_margin_threshold = 0.24;
        policy.label_durable_max_entropy = 0.42;
        policy.label_durable_support_count = 4;
      });
  add_adaptive_ablation (
      "adaptive_soft_anchor_v1_no_generic", "adaptive_soft_anchor_v1",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.generic_action_suppressor = false;
      });
  add_adaptive_ablation (
      "adaptive_soft_anchor_v1_no_null", "adaptive_soft_anchor_v1",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.null_no_anchor_hypothesis = false;
      });
  add_adaptive_ablation (
      "adaptive_soft_anchor_v1_update_strict", "adaptive_soft_anchor_v1",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.posterior_keep_threshold = 0.09;
        policy.tentative_score_threshold = 0.70;
      });
  add_adaptive_ablation (
      "adaptive_soft_anchor_v1_update_very_strict", "adaptive_soft_anchor_v1",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.posterior_keep_threshold = 0.12;
        policy.tentative_score_threshold = 0.74;
      });
  add_adaptive_ablation (
      "adaptive_soft_anchor_v1_high_focus", "adaptive_soft_anchor_v1",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.posterior_beta = 5.0;
        policy.update_threshold = 0.82;
        policy.margin_threshold = 0.22;
        policy.max_entropy = 0.55;
        policy.soft_update_posterior_threshold = 0.82;
        policy.new_anchor_threshold = 0.54;
        policy.generic_hard_threshold = 0.70;
        policy.semantic_weight = 0.24;
        policy.entity_weight = 0.46;
        policy.full_weight = 0.24;
      });
  add_adaptive_ablation (
      "adaptive_soft_anchor_v1_high_sensitivity", "adaptive_soft_anchor_v1",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.posterior_beta = 2.0;
        policy.update_threshold = 0.68;
        policy.margin_threshold = 0.08;
        policy.max_entropy = 0.86;
        policy.soft_update_posterior_threshold = 0.52;
        policy.new_anchor_threshold = 0.36;
        policy.generic_hard_threshold = 0.84;
        policy.semantic_weight = 0.42;
        policy.entity_weight = 0.28;
        policy.full_weight = 0.22;
      });
  add_adaptive_ablation (
      "adaptive_soft_anchor_v1_high_stability", "adaptive_soft_anchor_v1",
      [] (AdaptiveAnchorPolicy &policy) {
        policy.active_ttl = 48;
        policy.label_durable_support_count = 5;
        policy.label_durable_max_entropy = 0.50;
        policy.new_anchor_threshold = 0.42;
        policy.soft_update_posterior_threshold = 0.70;
      });
  const auto add_soft_anchor_knob_sweep = [&] {
    auto base = std::find_if (
        adaptive_policies.begin (), adaptive_policies.end (),
        [] (const AdaptiveAnchorPolicy &policy) {
          return policy.name == "adaptive_soft_anchor_v1";
        });
    if (base == adaptive_policies.end ())
      {
        throw std::runtime_error (
            "Missing adaptive_soft_anchor_v1 base policy for knob sweep");
      }
    const auto lerp = [] (double low, double high, double value) {
      return low + (high - low) * value;
    };
    const auto tag = [] (double value) {
      return std::to_string (
          static_cast<int> (std::round (100.0 * value)));
    };
    const std::vector<double> values = { 0.25, 0.50, 0.75 };
    for (double focus : values)
      {
        for (double sensitivity : values)
          {
            for (double stability : values)
              {
                AdaptiveAnchorPolicy policy = *base;
                policy.name = "adaptive_soft_anchor_v1_f" + tag (focus)
                              + "_s" + tag (sensitivity) + "_t"
                              + tag (stability);
                policy.focus = focus;
                policy.sensitivity = sensitivity;
                policy.stability_knob = stability;

                policy.posterior_beta = lerp (2.0, 5.0, focus);
                policy.semantic_weight =
                    0.28 + 0.16 * sensitivity + 0.08 * (1.0 - focus);
                policy.entity_weight = 0.22 + 0.34 * focus;
                policy.full_weight = 0.24;
                policy.update_threshold = 1.01;
                policy.margin_threshold = Clamp01 (
                    0.08 + 0.18 * focus - 0.05 * sensitivity);
                policy.max_entropy = Clamp01 (
                    0.88 - 0.30 * focus + 0.10 * sensitivity);
                policy.soft_update_posterior_threshold = Clamp01 (
                    0.50 + 0.24 * focus - 0.08 * sensitivity);
                policy.new_anchor_threshold = Clamp01 (
                    0.34 + 0.16 * focus - 0.06 * sensitivity);
                policy.generic_hard_threshold = Clamp01 (
                    0.86 - 0.16 * focus + 0.04 * sensitivity);
                policy.posterior_keep_threshold = Clamp01 (
                    0.030 + 0.090 * focus - 0.035 * sensitivity);
                policy.tentative_score_threshold = Clamp01 (
                    0.52 + 0.32 * focus - 0.10 * sensitivity);

                policy.active_ttl = static_cast<int> (
                    std::round (lerp (4.0, 40.0, stability)));
                policy.label_durable_support_count = static_cast<int> (
                    std::round (lerp (3.0, 6.0, stability)));
                policy.label_durable_strength_threshold = Clamp01 (
                    0.82 + 0.12 * focus + 0.04 * stability
                    - 0.03 * sensitivity);
                policy.label_durable_state_strength_threshold = Clamp01 (
                    0.80 + 0.12 * focus + 0.04 * stability
                    - 0.03 * sensitivity);
                policy.label_durable_margin_threshold = Clamp01 (
                    0.12 + 0.24 * focus + 0.04 * stability
                    - 0.05 * sensitivity);
                policy.label_durable_max_entropy = Clamp01 (
                    0.62 - 0.32 * focus - 0.08 * stability
                    + 0.08 * sensitivity);
                adaptive_policies.push_back (policy);
              }
          }
      }
	  };
	  add_soft_anchor_knob_sweep ();

	  std::vector<SoftAnchorConsumptionPolicy> consumption_policies;
	  auto add_consumption = [&] (SoftAnchorConsumptionPolicy policy) {
	    consumption_policies.push_back (std::move (policy));
	  };
	  SoftAnchorConsumptionPolicy consume_none;
	  consume_none.name = "consume_none_baseline";
	  consume_none.formation_policy = "adaptive_soft_anchor_v1";
	  consume_none.max_links = 0;
	  add_consumption (consume_none);

	  SoftAnchorConsumptionPolicy consume_default;
	  consume_default.name = "consume_soft_anchor_v1_default";
	  consume_default.formation_policy = "adaptive_soft_anchor_v1";
	  consume_default.max_links = 3;
	  consume_default.min_strength = 0.05;
	  consume_default.min_score = 0.54;
	  consume_default.min_posterior = 0.04;
	  consume_default.max_entropy = 0.86;
	  consume_default.max_generic = 0.82;
	  add_consumption (consume_default);

	  SoftAnchorConsumptionPolicy consume_top1 = consume_default;
	  consume_top1.name = "consume_top1";
	  consume_top1.max_links = 1;
	  add_consumption (consume_top1);

	  SoftAnchorConsumptionPolicy consume_top5 = consume_default;
	  consume_top5.name = "consume_top5_high_recall";
	  consume_top5.max_links = 5;
	  consume_top5.min_strength = 0.025;
	  consume_top5.min_score = 0.46;
	  consume_top5.min_posterior = 0.025;
	  consume_top5.max_entropy = 0.95;
	  add_consumption (consume_top5);

	  SoftAnchorConsumptionPolicy consume_durable = consume_default;
	  consume_durable.name = "consume_durable_only";
	  consume_durable.allow_tentative = false;
	  consume_durable.allow_ambiguous = false;
	  consume_durable.allow_durable = true;
	  consume_durable.min_strength = 0.80;
	  consume_durable.min_score = 0.70;
	  add_consumption (consume_durable);

	  SoftAnchorConsumptionPolicy consume_no_ambiguous = consume_default;
	  consume_no_ambiguous.name = "consume_no_ambiguous";
	  consume_no_ambiguous.allow_ambiguous = false;
	  add_consumption (consume_no_ambiguous);

	  SoftAnchorConsumptionPolicy consume_ambiguous_only = consume_default;
	  consume_ambiguous_only.name = "consume_ambiguous_only";
	  consume_ambiguous_only.allow_tentative = false;
	  consume_ambiguous_only.allow_durable = false;
	  consume_ambiguous_only.allow_ambiguous = true;
	  add_consumption (consume_ambiguous_only);

	  SoftAnchorConsumptionPolicy consume_strict_margin = consume_default;
	  consume_strict_margin.name = "consume_strict_margin";
	  consume_strict_margin.min_margin = 0.10;
	  consume_strict_margin.max_entropy = 0.70;
	  consume_strict_margin.min_strength = 0.08;
	  consume_strict_margin.min_score = 0.60;
	  add_consumption (consume_strict_margin);

	  SoftAnchorConsumptionPolicy consume_wm_only = consume_default;
	  consume_wm_only.name = "consume_wm_only";
	  consume_wm_only.allow_stm = false;
	  consume_wm_only.allow_ltm = false;
	  add_consumption (consume_wm_only);

	  SoftAnchorConsumptionPolicy consume_wm_stm = consume_default;
	  consume_wm_stm.name = "consume_wm_stm";
	  consume_wm_stm.allow_ltm = false;
	  add_consumption (consume_wm_stm);

	  SoftAnchorConsumptionPolicy consume_all_tiers = consume_default;
	  consume_all_tiers.name = "consume_wm_stm_ltm";
	  add_consumption (consume_all_tiers);

	  SoftAnchorConsumptionPolicy consume_unsafe_all = consume_default;
	  consume_unsafe_all.name = "consume_unsafe_all_retained";
	  consume_unsafe_all.max_links = 8;
	  consume_unsafe_all.min_strength = 0.0;
	  consume_unsafe_all.min_score = 0.0;
	  consume_unsafe_all.min_posterior = 0.0;
	  consume_unsafe_all.max_entropy = 1.0;
	  consume_unsafe_all.max_generic = 1.0;
	  add_consumption (consume_unsafe_all);

	  const auto add_consumption_knob_sweep = [&] {
	    const auto lerp = [] (double low, double high, double value) {
	      return low + (high - low) * value;
	    };
	    const auto tag = [] (double value) {
	      return std::to_string (
	          static_cast<int> (std::round (100.0 * value)));
	    };
	    const std::vector<double> values = { 0.25, 0.50, 0.75 };
	    for (double focus : values)
	      {
	        for (double sensitivity : values)
	          {
	            for (double stability : values)
	              {
	                SoftAnchorConsumptionPolicy policy = consume_default;
	                policy.name = "consume_soft_anchor_v1_f" + tag (focus)
	                              + "_s" + tag (sensitivity) + "_t"
	                              + tag (stability);
	                policy.formation_policy = "adaptive_soft_anchor_v1_f"
	                                          + tag (focus) + "_s"
	                                          + tag (sensitivity) + "_t"
	                                          + tag (stability);
	                policy.focus = focus;
	                policy.sensitivity = sensitivity;
	                policy.stability = stability;
	                policy.max_links = std::clamp (
	                    static_cast<int> (std::round (
	                        lerp (6.0, 2.0, focus)
	                        + lerp (0.0, 2.0, sensitivity)
	                        + lerp (0.0, 1.0, stability))),
	                    1, 8);
	                policy.min_strength = Clamp01 (
	                    0.025 + 0.080 * focus - 0.020 * sensitivity);
	                policy.min_score = Clamp01 (
	                    0.46 + 0.18 * focus - 0.06 * sensitivity);
	                policy.min_posterior = Clamp01 (
	                    0.020 + 0.050 * focus - 0.015 * sensitivity);
	                policy.max_entropy = Clamp01 (
	                    0.92 - 0.30 * focus + 0.08 * sensitivity);
	                policy.max_generic = Clamp01 (
	                    0.86 - 0.12 * focus + 0.04 * sensitivity);
	                policy.min_margin = focus >= 0.75 ? 0.04 : -1.0;
	                policy.allow_ltm = true;
	                add_consumption (policy);
	              }
	          }
	      }
	  };
	  add_consumption_knob_sweep ();

	  std::vector<StorageAttentionRun> storage_runs;
  for (const auto &policy : policies)
    {
      storage_runs.push_back (RunStorageAttentionPolicy (
          conversations, eval_cases, encoder, encoded_cache, policy,
          opts.max_turns_per_conversation));
    }
	  std::vector<AdaptiveAnchorRun> adaptive_runs;
	  for (const auto &policy : adaptive_policies)
	    {
	      adaptive_runs.push_back (RunAdaptiveAnchorPolicy (
	          conversations, eval_cases, encoder, encoded_cache, policy,
	          opts.max_turns_per_conversation));
	    }
	  std::map<std::string, const AdaptiveAnchorRun *> adaptive_run_by_name;
	  for (const auto &run : adaptive_runs)
	    {
	      adaptive_run_by_name[run.policy.name] = &run;
	    }
	  std::vector<SoftAnchorConsumptionRun> consumption_runs;
	  std::vector<SoftAnchorConsumptionRun> consumption_knob_runs;
	  for (const auto &policy : consumption_policies)
	    {
	      const auto found = adaptive_run_by_name.find (policy.formation_policy);
	      if (found == adaptive_run_by_name.end ())
	        {
	          continue;
	        }
	      SoftAnchorConsumptionRun run =
	          RunSoftAnchorConsumptionPolicy (*found->second, policy);
	      if (policy.focus >= 0.0 && policy.sensitivity >= 0.0
	          && policy.stability >= 0.0)
	        {
	          consumption_knob_runs.push_back (run);
	        }
	      consumption_runs.push_back (std::move (run));
	    }

	  WriteStorageCasesCsv (opts.output_dir / "ingress_storage_attention_cases.csv",
	                        storage_runs);
  WriteStorageLinksCsv (opts.output_dir / "ingress_storage_attention_links.csv",
                        storage_runs);
  WriteStorageFailuresCsv (
      opts.output_dir / "ingress_storage_attention_failure_examples.csv",
      storage_runs);
  WriteAdaptiveCasesCsv (opts.output_dir / "ingress_adaptive_anchor_cases.csv",
                         adaptive_runs);
  WriteAdaptiveLinksCsv (opts.output_dir / "ingress_adaptive_anchor_links.csv",
                         adaptive_runs);
  WriteAdaptiveSoftLinksCsv (
      opts.output_dir / "ingress_adaptive_anchor_soft_links.csv",
      adaptive_runs);
  WriteAdaptiveCandidatesCsv (
      opts.output_dir / "ingress_adaptive_anchor_candidates.csv",
      adaptive_runs);
  WriteAdaptiveManualAuditCsv (
      opts.output_dir / "ingress_adaptive_anchor_manual_audit.csv",
      adaptive_runs);
  WriteAdaptiveStatesCsv (
      opts.output_dir / "ingress_adaptive_anchor_states.csv", adaptive_runs);
	  WriteAdaptiveFailuresCsv (
	      opts.output_dir / "ingress_adaptive_anchor_failure_examples.csv",
	      adaptive_runs);
	  WriteSoftAnchorConsumptionCasesCsv (
	      opts.output_dir / "soft_anchor_consumption_cases.csv",
	      consumption_runs);
	  WriteSoftAnchorConsumptionSurfacesCsv (
	      opts.output_dir / "soft_anchor_consumption_surfaces.csv",
	      consumption_runs);
	  WriteSoftAnchorConsumptionFailuresCsv (
	      opts.output_dir / "soft_anchor_consumption_failure_examples.csv",
	      consumption_runs);
	  WriteSoftAnchorConsumptionSummaryCsv (
	      opts.output_dir / "soft_anchor_consumption_ablation_summary.csv",
	      consumption_runs);
	  WriteSoftAnchorConsumptionSummaryCsv (
	      opts.output_dir / "soft_anchor_consumption_knob_sweep_summary.csv",
	      consumption_knob_runs);

	  Json results = {
    { "mode", "ingress_storage_attention_es_aist" },
    { "runtime_effect", "none" },
    { "production_retrieval_changed", false },
    { "model_path", model_path->string () },
    { "model_backend", encoder.KernelOpsBackend () },
    { "uses_kernel_ops", encoder.UsesKernelOps () },
    { "uses_full_text_graph_ops", encoder.UsesFullTextGraphOps () },
    { "retrieved_candidate_count", 0 },
    { "uses_retrieved_candidates", false },
    { "runtime_policy_uses_labels", false },
    { "storage_time_signal", true },
    { "candidate_pool_runtime_count", 0 },
    { "conversations", conversations.size () },
    { "eval_cases", eval_cases.size () },
    { "unique_text_encodes", encoded_cache.size () },
    { "max_turns_per_conversation", opts.max_turns_per_conversation },
    { "memory_tiers",
      { { "wm", "prior stored turns with age <= 4" },
        { "stm", "prior stored turns with age 5..32" },
        { "ltm", "prior stored turns with age > 32" } } },
    { "contract",
      "Each conversation is replayed chronologically. The current signal is "
      "encoded once at ingress with ES-AIST, attention is computed over "
      "already-stored prior WM/STM/LTM memory records, and the shadow link is "
      "created before the current memory is written. Retrieved candidates are "
      "not built or scored at runtime." },
    { "variants", Json::object () },
	    { "adaptive_variants", Json::object () },
	    { "adaptive_policy_configs", Json::object () },
	    { "soft_anchor_consumption_variants", Json::object () },
	    { "soft_anchor_consumption_policy_configs", Json::object () },
	  };
  for (const auto &run : storage_runs)
    {
      results["variants"][run.policy.name] = SummarizeStorageAttentionRun (
          run);
    }
  for (const auto &run : adaptive_runs)
    {
      results["adaptive_variants"][run.policy.name] =
          SummarizeAdaptiveAnchorRun (run);
      results["adaptive_policy_configs"][run.policy.name] = {
        { "focus", run.policy.focus },
        { "sensitivity", run.policy.sensitivity },
        { "stability", run.policy.stability_knob },
        { "attention_temperature", run.policy.attention_temperature },
        { "update_threshold", run.policy.update_threshold },
        { "margin_threshold", run.policy.margin_threshold },
        { "max_entropy", run.policy.max_entropy },
        { "event_mismatch_threshold", run.policy.event_mismatch_threshold },
        { "contradiction_penalty", run.policy.contradiction_penalty },
        { "active_support_count", run.policy.active_support_count },
        { "active_ttl", run.policy.active_ttl },
        { "split_on_event_mismatch", run.policy.split_on_event_mismatch },
        { "label_durable_strength_threshold",
          run.policy.label_durable_strength_threshold },
        { "label_durable_state_strength_threshold",
          run.policy.label_durable_state_strength_threshold },
        { "label_durable_margin_threshold",
          run.policy.label_durable_margin_threshold },
        { "label_durable_max_entropy", run.policy.label_durable_max_entropy },
        { "label_durable_support_count",
          run.policy.label_durable_support_count },
        { "label_demote_generic_to_tentative",
          run.policy.label_demote_generic_to_tentative },
        { "label_block_event_mismatch_durable",
          run.policy.label_block_event_mismatch_durable },
        { "semantic_weight", run.policy.semantic_weight },
        { "entity_weight", run.policy.entity_weight },
        { "full_weight", run.policy.full_weight },
        { "wm_bias", run.policy.wm_bias },
        { "stm_bias", run.policy.stm_bias },
        { "ltm_bias", run.policy.ltm_bias },
        { "include_wm", run.policy.include_wm },
        { "include_stm", run.policy.include_stm },
        { "include_ltm", run.policy.include_ltm },
        { "generic_action_suppressor",
          run.policy.generic_action_suppressor },
        { "null_no_anchor_hypothesis",
          run.policy.null_no_anchor_hypothesis },
        { "use_contradiction_penalty",
          run.policy.use_contradiction_penalty },
        { "min_update_support", run.policy.min_update_support },
        { "null_score_threshold", run.policy.null_score_threshold },
        { "null_entropy_threshold", run.policy.null_entropy_threshold },
        { "soft_hypothesis_mode", run.policy.soft_hypothesis_mode },
        { "posterior_beta", run.policy.posterior_beta },
        { "generic_hard_threshold", run.policy.generic_hard_threshold },
        { "new_anchor_threshold", run.policy.new_anchor_threshold },
        { "posterior_keep_threshold", run.policy.posterior_keep_threshold },
        { "tentative_score_threshold", run.policy.tentative_score_threshold },
        { "soft_update_posterior_threshold",
          run.policy.soft_update_posterior_threshold },
	      };
	    }
	  for (const auto &run : consumption_runs)
	    {
	      results["soft_anchor_consumption_variants"][run.policy.name] =
	          SummarizeSoftAnchorConsumptionRun (run);
	      results["soft_anchor_consumption_policy_configs"][run.policy.name] = {
	        { "formation_policy", run.policy.formation_policy },
	        { "max_links", run.policy.max_links },
	        { "allow_tentative", run.policy.allow_tentative },
	        { "allow_ambiguous", run.policy.allow_ambiguous },
	        { "allow_durable", run.policy.allow_durable },
	        { "allow_wm", run.policy.allow_wm },
	        { "allow_stm", run.policy.allow_stm },
	        { "allow_ltm", run.policy.allow_ltm },
	        { "min_strength", run.policy.min_strength },
	        { "min_score", run.policy.min_score },
	        { "min_posterior", run.policy.min_posterior },
	        { "min_margin", run.policy.min_margin },
	        { "max_entropy", run.policy.max_entropy },
	        { "max_generic", run.policy.max_generic },
	        { "focus", run.policy.focus },
	        { "sensitivity", run.policy.sensitivity },
	        { "stability", run.policy.stability },
	      };
	    }
	  results["soft_anchor_consumption_monotonicity"] =
	      SoftAnchorConsumptionMonotonicity (consumption_knob_runs);
	  {
	    std::ofstream out (opts.output_dir
	                       / "ingress_storage_attention_results.json");
	    out << std::setw (2) << results << '\n';
	  }
	  {
	    Json consumption_results = {
	      { "mode", "soft_anchor_consumption_ablation" },
	      { "runtime_effect", "none" },
	      { "production_retrieval_changed", false },
	      { "uses_retrieved_candidates", false },
	      { "runtime_policy_uses_labels", false },
	      { "formation_stage", "chronological ingress soft anchor" },
	      { "consumption_stage",
	        "surface bounded top-k soft links as uncertain context" },
	      { "variants", results["soft_anchor_consumption_variants"] },
	      { "policy_configs",
	        results["soft_anchor_consumption_policy_configs"] },
	      { "monotonicity",
	        results["soft_anchor_consumption_monotonicity"] },
	    };
	    std::ofstream out (opts.output_dir
	                       / "soft_anchor_consumption_results.json");
	    out << std::setw (2) << consumption_results << '\n';
	  }
	  {
	    std::ofstream out (opts.output_dir
	                       / "soft_anchor_consumption_monotonicity.json");
	    out << std::setw (2)
	        << results["soft_anchor_consumption_monotonicity"] << '\n';
	  }
  {
    Json metadata = {
      { "model_path", model_path->string () },
      { "backend", encoder.KernelOpsBackend () },
      { "uses_kernel_ops", encoder.UsesKernelOps () },
      { "uses_full_text_graph_ops", encoder.UsesFullTextGraphOps () },
      { "semantic_key", Json::array ({ 0, 768 }) },
      { "entity_key", Json::array ({ 768, 1536 }) },
      { "full_key", Json::array ({ 0, 1536 }) },
      { "runtime_role", "storage-time signal encoder" },
    };
    std::ofstream out (opts.output_dir
                       / "ingress_storage_attention_model.json");
    out << std::setw (2) << metadata << '\n';
  }

  std::cout << std::setw (2) << results << '\n';
  return 0;
}
