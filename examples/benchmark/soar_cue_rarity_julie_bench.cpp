#include "../../src/operations/cognitive_mechanisms.hpp"
#include "include/benchmark_text_encoder.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
namespace cognitive = cortext::operations::cognitive;

namespace
{

struct Message
{
  int index = 0;
  bool from_contact = false;
  std::string text;
};

struct TokenStats
{
  std::unordered_map<std::string, int> df;
  int documents = 0;
};

struct Features
{
  double base = 0.0;
  double overlap = 0.0;
  double rare = 0.0;
  double specificity = 0.0;
};

struct Trial
{
  std::string query;
  int target = -1;
  int negative = -1;
  int fold = 0;
};

struct Metrics
{
  int count = 0;
  int pair_correct = 0;
  double pair_accuracy = 0.0;
  double mrr = 0.0;
  double hit10 = 0.0;
  double mean_rank = 0.0;
  double mean_margin = 0.0;
};

bool
StartsWithDate (const std::string &line)
{
  return line.size () >= 19 && std::isdigit (line[0])
         && std::isdigit (line[1]) && std::isdigit (line[2])
         && std::isdigit (line[3]) && line[4] == '-'
         && std::isdigit (line[5]) && std::isdigit (line[6])
         && line[7] == '-' && std::isdigit (line[8])
         && std::isdigit (line[9]) && line[10] == ' '
         && std::isdigit (line[11]) && std::isdigit (line[12])
         && line[13] == ':' && std::isdigit (line[14])
         && std::isdigit (line[15]) && line[16] == ':'
         && std::isdigit (line[17]) && std::isdigit (line[18]);
}

double
Clamp (double value, double lo = 0.0, double hi = 1.0)
{
  return std::max (lo, std::min (hi, value));
}

std::vector<Message>
ParseMessages (const fs::path &path)
{
  std::ifstream in (path);
  if (!in)
    {
      throw std::runtime_error ("could not open transcript: "
                                + path.string ());
    }

  std::vector<Message> messages;
  std::string line;
  std::string header;
  std::string text;
  bool collecting = false;
  auto flush = [&] {
    while (!text.empty ()
           && (text.back () == '\n' || text.back () == '\r'
               || text.back () == ' ' || text.back () == '\t'))
      {
        text.pop_back ();
      }
    if (!text.empty ())
      {
        messages.push_back ({
          static_cast<int> (messages.size ()),
          header.find (" from ") != std::string::npos,
          text,
        });
      }
    text.clear ();
    collecting = false;
  };

  while (std::getline (in, line))
    {
      if (line.rfind ("---", 0) == 0)
        {
          flush ();
          continue;
        }
      if (StartsWithDate (line))
        {
          flush ();
          header = line;
          collecting = true;
          continue;
        }
      if (!collecting)
        {
          continue;
        }
      if (line.empty () && text.empty ())
        {
          continue;
        }
      text += line;
      text.push_back (' ');
    }
  flush ();
  return messages;
}

std::vector<std::string>
Tokens (const std::string &text)
{
  static const std::set<std::string> stop = {
    "the", "and", "for", "you", "your", "are", "was", "were", "that",
    "this", "with", "have", "has", "had", "but", "not", "can", "will",
    "just", "now", "then", "from", "our", "out", "get", "got", "its",
    "it",  "ill", "im",  "is",  "to",  "of",  "in",  "on",  "at",
    "a",   "i",   "we",  "he",  "she", "they", "my", "me", "be",
    "do",  "so",  "ok",  "or",  "as",  "if",  "up", "all", "what",
    "when", "where", "how", "why", "there", "here", "than", "more"
  };

  std::vector<std::string> tokens;
  std::string current;
  for (unsigned char ch : text)
    {
      if (std::isalnum (ch))
        {
          current.push_back (static_cast<char> (std::tolower (ch)));
        }
      else if (!current.empty ())
        {
          if (current.size () > 2 && stop.count (current) == 0)
            {
              tokens.push_back (current);
            }
          current.clear ();
        }
    }
  if (!current.empty () && current.size () > 2 && stop.count (current) == 0)
    {
      tokens.push_back (current);
    }
  return tokens;
}

TokenStats
BuildStats (const std::vector<Message> &messages)
{
  TokenStats stats;
  stats.documents = static_cast<int> (messages.size ());
  for (const auto &msg : messages)
    {
      std::set<std::string> unique;
      for (const auto &token : Tokens (msg.text))
        {
          unique.insert (token);
        }
      for (const auto &token : unique)
        {
          ++stats.df[token];
        }
    }
  return stats;
}

int
Df (const TokenStats &stats, const std::string &token)
{
  const auto it = stats.df.find (token);
  return it == stats.df.end () ? 1 : it->second;
}

double
Cosine (const std::vector<float> &a, const std::vector<float> &b)
{
  double dot = 0.0;
  double an = 0.0;
  double bn = 0.0;
  const std::size_t n = std::min (a.size (), b.size ());
  for (std::size_t i = 0; i < n; ++i)
    {
      dot += static_cast<double> (a[i]) * b[i];
      an += static_cast<double> (a[i]) * a[i];
      bn += static_cast<double> (b[i]) * b[i];
    }
  if (an <= 1e-12 || bn <= 1e-12)
    {
      return 0.0;
    }
  return dot / (std::sqrt (an) * std::sqrt (bn));
}

double
SemanticScore (cortext::benchmark::BenchmarkTextEncoder &encoder,
               const std::string &query, const std::string &candidate)
{
  std::vector<float> q;
  std::vector<float> c;
  encoder.EncodeText (query, q);
  encoder.EncodeText (candidate, c);
  return Clamp (Cosine (q, c));
}

double
Overlap (const std::vector<std::string> &a, const std::vector<std::string> &b)
{
  if (a.empty ())
    {
      return 0.0;
    }
  std::set<std::string> bset (b.begin (), b.end ());
  int hits = 0;
  for (const auto &token : a)
    {
      if (bset.count (token) > 0)
        {
          ++hits;
        }
    }
  return Clamp (static_cast<double> (hits) / static_cast<double> (a.size ()));
}

double
RareAverage (const TokenStats &stats, const std::vector<std::string> &tokens)
{
  if (tokens.empty ())
    {
      return 0.0;
    }
  double total = 0.0;
  for (const auto &token : tokens)
    {
      total += cognitive::CueRarityWeight (stats.documents, Df (stats, token));
    }
  return Clamp (total / static_cast<double> (tokens.size ()));
}

bool
ContainsDigit (const std::string &text)
{
  return std::any_of (text.begin (), text.end (), [] (unsigned char ch) {
    return std::isdigit (ch) != 0;
  });
}

double
Specificity (const TokenStats &stats, const Message &msg)
{
  const auto tokens = Tokens (msg.text);
  const double length = Clamp (static_cast<double> (tokens.size ()) / 12.0);
  const double rare = RareAverage (stats, tokens);
  const double digit = ContainsDigit (msg.text) ? 1.0 : 0.0;
  return Clamp (0.48 * rare + 0.32 * length + 0.20 * digit);
}

std::string
JoinTokens (const std::vector<std::string> &tokens)
{
  std::ostringstream out;
  for (std::size_t i = 0; i < tokens.size (); ++i)
    {
      if (i > 0)
        {
          out << ' ';
        }
      out << tokens[i];
    }
  return out.str ();
}

std::string
RareQuery (const TokenStats &stats, const Message &msg)
{
  auto tokens = Tokens (msg.text);
  if (tokens.empty ())
    {
      return msg.text;
    }
  std::sort (tokens.begin (), tokens.end (), [&] (const auto &a,
                                                  const auto &b) {
    if (Df (stats, a) == Df (stats, b))
      {
        return a < b;
      }
    return Df (stats, a) < Df (stats, b);
  });
  std::vector<std::string> query{ tokens.front () };
  std::sort (tokens.begin (), tokens.end (), [&] (const auto &a,
                                                  const auto &b) {
    if (Df (stats, a) == Df (stats, b))
      {
        return a < b;
      }
    return Df (stats, a) > Df (stats, b);
  });
  if (!tokens.empty () && tokens.front () != query.front ())
    {
      query.push_back (tokens.front ());
    }
  return JoinTokens (query);
}

Features
BuildFeatures (cortext::benchmark::BenchmarkTextEncoder &encoder,
               const TokenStats &stats, const std::vector<Message> &messages,
               const std::string &query, const Message &msg)
{
  (void)messages;
  const auto query_tokens = Tokens (query);
  const auto msg_tokens = Tokens (msg.text);
  return {
    SemanticScore (encoder, query, msg.text),
    Overlap (query_tokens, msg_tokens),
    RareAverage (stats, msg_tokens),
    Specificity (stats, msg),
  };
}

double
ProductScore (const Features &features,
              const std::vector<std::string> &query_tokens,
              const Message &msg, const TokenStats &stats)
{
  const auto msg_tokens = Tokens (msg.text);
  std::vector<cognitive::CueEvidence> cues;
  for (const auto &token : query_tokens)
    {
      if (std::find (msg_tokens.begin (), msg_tokens.end (), token)
          != msg_tokens.end ())
        {
          cues.push_back ({ 0.95, static_cast<double> (Df (stats, token)),
                            false });
        }
    }
  return cognitive::CueRarityProductScore (
      features.base, cues, stats.documents, features.specificity, 0.35, 0.25,
      static_cast<double> (std::max<std::size_t> (1, query_tokens.size ())));
}

std::vector<Trial>
GenerateTrials (cortext::benchmark::BenchmarkTextEncoder &encoder,
                const std::vector<Message> &messages, const TokenStats &stats)
{
  std::vector<Trial> trials;
  constexpr int kMaxTrials = 24;
  for (const auto &target : messages)
    {
      const auto target_tokens = Tokens (target.text);
      if (target_tokens.size () < 3)
        {
          continue;
        }
      const std::string query = RareQuery (stats, target);
      if (query.empty ())
        {
          continue;
        }
      const auto query_tokens = Tokens (query);
      const Features tf = BuildFeatures (encoder, stats, messages, query,
                                         target);
      if (tf.rare < 0.48 || tf.overlap <= 0.0)
        {
          continue;
        }

      int best_negative = -1;
      double best_negative_raw = -1.0;
      for (const auto &candidate : messages)
        {
          if (candidate.index == target.index)
            {
              continue;
            }
          const auto candidate_tokens = Tokens (candidate.text);
          if (Overlap (query_tokens, candidate_tokens) <= 0.0)
            {
              continue;
            }
          const Features cf = BuildFeatures (encoder, stats, messages, query,
                                             candidate);
          const bool negative_ok = cf.base >= tf.base - 0.06
                                   && cf.overlap <= std::max (0.34, tf.overlap)
                                   && cf.rare <= tf.rare + 0.05;
          if (negative_ok && cf.base > best_negative_raw)
            {
              best_negative = candidate.index;
              best_negative_raw = cf.base;
            }
        }
      if (best_negative >= 0)
        {
          trials.push_back ({
            query,
            target.index,
            best_negative,
            static_cast<int> (trials.size ()) % 3,
          });
          if (static_cast<int> (trials.size ()) >= kMaxTrials)
            {
              break;
            }
        }
    }
  return trials;
}

Metrics
Evaluate (const std::vector<Trial> &trials, int fold,
          cortext::benchmark::BenchmarkTextEncoder &encoder,
          const std::vector<Message> &messages, const TokenStats &stats,
          bool product)
{
  Metrics metrics;
  for (const auto &trial : trials)
    {
      if (trial.fold != fold)
        {
          continue;
        }
      const auto query_tokens = Tokens (trial.query);
      const auto &target_msg = messages[static_cast<std::size_t> (trial.target)];
      const auto &negative_msg
          = messages[static_cast<std::size_t> (trial.negative)];
      const Features target = BuildFeatures (encoder, stats, messages,
                                             trial.query, target_msg);
      const Features negative = BuildFeatures (encoder, stats, messages,
                                               trial.query, negative_msg);
      const double target_score = product
                                      ? ProductScore (target, query_tokens,
                                                      target_msg, stats)
                                      : target.base;
      const double negative_score = product
                                        ? ProductScore (negative, query_tokens,
                                                        negative_msg, stats)
                                        : negative.base;
      metrics.pair_correct += target_score > negative_score ? 1 : 0;
      metrics.mean_margin += target_score - negative_score;

      int rank = 1;
      for (const auto &candidate : messages)
        {
          if (candidate.index == trial.target)
            {
              continue;
            }
          const Features cf = BuildFeatures (encoder, stats, messages,
                                             trial.query, candidate);
          const double score = product
                                   ? ProductScore (cf, query_tokens, candidate,
                                                   stats)
                                   : cf.base;
          if (score > target_score
              || (score == target_score && candidate.index < trial.target))
            {
              ++rank;
            }
        }
      metrics.mrr += 1.0 / static_cast<double> (rank);
      metrics.hit10 += rank <= 10 ? 1.0 : 0.0;
      metrics.mean_rank += rank;
      ++metrics.count;
    }

  if (metrics.count > 0)
    {
      const double count = static_cast<double> (metrics.count);
      metrics.pair_accuracy = metrics.pair_correct / count;
      metrics.mrr /= count;
      metrics.hit10 /= count;
      metrics.mean_rank /= count;
      metrics.mean_margin /= count;
    }
  return metrics;
}

Metrics
AverageFolds (const std::vector<Trial> &trials,
              cortext::benchmark::BenchmarkTextEncoder &encoder,
              const std::vector<Message> &messages, const TokenStats &stats,
              bool product)
{
  Metrics total;
  for (int fold = 0; fold < 3; ++fold)
    {
      const auto fold_metrics = Evaluate (trials, fold, encoder, messages,
                                          stats, product);
      std::cout << "fold=" << fold << " mode="
                << (product ? "soar_product" : "baseline")
                << " test=" << fold_metrics.count << std::fixed
                << std::setprecision (6)
                << " pair=" << fold_metrics.pair_accuracy
                << " mrr=" << fold_metrics.mrr
                << " hit10=" << fold_metrics.hit10
                << " mean_rank=" << fold_metrics.mean_rank
                << " margin=" << fold_metrics.mean_margin << "\n";
      total.count += fold_metrics.count;
      total.pair_accuracy += fold_metrics.pair_accuracy;
      total.mrr += fold_metrics.mrr;
      total.hit10 += fold_metrics.hit10;
      total.mean_rank += fold_metrics.mean_rank;
      total.mean_margin += fold_metrics.mean_margin;
    }
  total.pair_accuracy /= 3.0;
  total.mrr /= 3.0;
  total.hit10 /= 3.0;
  total.mean_rank /= 3.0;
  total.mean_margin /= 3.0;
  return total;
}

std::string
ArgValue (int argc, char **argv, const std::string &prefix,
          const std::string &fallback)
{
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      if (arg.rfind (prefix, 0) == 0)
        {
          return arg.substr (prefix.size ());
        }
    }
  return fallback;
}

} // namespace

int
main (int argc, char **argv)
{
  try
    {
      const std::string models_dir = ArgValue (argc, argv, "--models=", "models");
      const fs::path transcript = ArgValue (
          argc, argv, "--transcript=",
          "build/julie_mixed_media_week_2025_03_20_input/"
          "Messages - Julie Willen.txt");

      auto messages = ParseMessages (transcript);
      if (messages.empty ())
        {
          throw std::runtime_error ("no Julie messages parsed");
        }
      const auto stats = BuildStats (messages);
      cortext::benchmark::BenchmarkTextEncoder encoder (models_dir);
      std::cout << "encoder_backend=" << encoder.backend_name ()
                << " model=" << encoder.resolved_model_path ().string ()
                << " messages=" << messages.size () << "\n";

      const auto trials = GenerateTrials (encoder, messages, stats);
      const auto baseline = AverageFolds (trials, encoder, messages, stats,
                                          false);
      const auto product = AverageFolds (trials, encoder, messages, stats,
                                         true);

      std::cout << "summary trials=" << trials.size () << std::fixed
                << std::setprecision (6)
                << " baseline_pair=" << baseline.pair_accuracy
                << " product_pair=" << product.pair_accuracy
                << " baseline_mrr=" << baseline.mrr
                << " product_mrr=" << product.mrr
                << " baseline_hit10=" << baseline.hit10
                << " product_hit10=" << product.hit10
                << " baseline_mean_rank=" << baseline.mean_rank
                << " product_mean_rank=" << product.mean_rank
                << " baseline_margin=" << baseline.mean_margin
                << " product_margin=" << product.mean_margin << "\n";

      const bool passed = product.pair_accuracy >= baseline.pair_accuracy + 0.20
                          && product.mrr >= baseline.mrr + 0.20
                          && product.hit10 >= baseline.hit10 + 0.20;
      return passed ? 0 : 1;
    }
  catch (const std::exception &e)
    {
      std::cerr << "soar_cue_rarity_julie_bench failed: " << e.what ()
                << "\n";
      return 2;
    }
}
