#include "../../src/operations/cognitive_mechanisms.hpp"
#include "include/benchmark_text_encoder.hpp"

#include <cortext/core/sparse.hpp>
#include <cortext/models/aist_gguf_encoder.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{

using BenchEncoder = cortext::benchmark::BenchmarkTextEncoder;
namespace cognitive = cortext::operations::cognitive;

class ScopedEnvVar
{
public:
  explicit ScopedEnvVar (const char *name) : name_ (name)
  {
    const char *existing = std::getenv (name);
    if (existing != nullptr)
      {
        had_value_ = true;
        old_value_ = existing;
      }
    unsetenv (name_);
  }

  ScopedEnvVar (const char *name, const std::string &value)
      : ScopedEnvVar (name)
  {
    setenv (name_, value.c_str (), 1);
  }

  ~ScopedEnvVar ()
  {
    if (had_value_)
      {
        setenv (name_, old_value_.c_str (), 1);
      }
    else
      {
        unsetenv (name_);
      }
  }

private:
  const char *name_;
  bool had_value_ = false;
  std::string old_value_;
};

Eigen::VectorXf
ToEigen (const std::vector<float> &values)
{
  Eigen::VectorXf out (static_cast<Eigen::Index> (values.size ()));
  for (std::size_t i = 0; i < values.size (); ++i)
    {
      out (static_cast<Eigen::Index> (i)) = values[i];
    }
  return out;
}

Eigen::VectorXf
EncodeTextEigen256 (BenchEncoder &encoder, const std::string &text)
{
  std::vector<float> full;
  encoder.EncodeText (text, full);
  return ToEigen (cortext::TruncateAistMatryoshka (full, 256));
}

double
SemanticScore (BenchEncoder &encoder, const std::string &query,
               const std::string &candidate)
{
  const auto q = EncodeTextEigen256 (encoder, query);
  const auto c = EncodeTextEigen256 (encoder, candidate);
  return cortext::core::Clamp (cortext::core::CosineSimilarity (q, c), 0.0,
                               1.0);
}

struct StudyResult
{
  std::string name;
  std::string source;
  std::string off_winner;
  std::string on_winner;
  double off_target = 0.0;
  double off_comparison = 0.0;
  double on_target = 0.0;
  double on_comparison = 0.0;
  bool passed = false;
};

void
PrintStudy (const StudyResult &result)
{
  std::cout << "study=" << result.name << " source=" << result.source
            << " off_winner=" << result.off_winner
            << " on_winner=" << result.on_winner << " off_target="
            << std::fixed << std::setprecision (6) << result.off_target
            << " off_comparison=" << result.off_comparison
            << " on_target=" << result.on_target
            << " on_comparison=" << result.on_comparison
            << " passed=" << (result.passed ? 1 : 0) << "\n";
}

std::string
Winner (double target, double comparison)
{
  return target >= comparison ? "target" : "comparison";
}

StudyResult
RunAttentionLedgerStudy (BenchEncoder &encoder)
{
  const std::string query = "cobalt rollback build captain checklist";
  const double target_semantic = SemanticScore (
      encoder, query,
      "Nadia owns the verified emergency fallback plan with source-backed "
      "handoff evidence.");
  const double comparison_semantic = SemanticScore (
      encoder, query,
      "cobalt rollback build captain checklist dashboard generic status");

  StudyResult result{ "opencog_attention_ledger", "OpenCog" };
  result.off_target = target_semantic;
  result.off_comparison = comparison_semantic;
  result.off_winner = Winner (result.off_target, result.off_comparison);

  ScopedEnvVar guard (cognitive::kAttentionLedgerFlag, "1");
  cognitive::AttentionLedgerInput target;
  target.semantic_relevance = target_semantic;
  target.explicit_support = 0.95;
  target.source_confidence = 0.95;
  target.evidence_confidence = 0.92;
  target.used_count = 10.0;
  target.persistence_intent = 0.85;
  cognitive::AttentionLedgerInput comparison;
  comparison.semantic_relevance = comparison_semantic;
  comparison.context_relevance = 0.45;
  comparison.source_confidence = 0.35;

  result.on_target = cognitive::EnvFlagEnabled (cognitive::kAttentionLedgerFlag)
                         ? cognitive::BuildBoundedAttentionLedger (target).total
                         : result.off_target;
  result.on_comparison
      = cognitive::EnvFlagEnabled (cognitive::kAttentionLedgerFlag)
            ? cognitive::BuildBoundedAttentionLedger (comparison).total
            : result.off_comparison;
  result.on_winner = Winner (result.on_target, result.on_comparison);
  result.passed = result.off_winner == "comparison"
                  && result.on_winner == "target";
  PrintStudy (result);
  return result;
}

StudyResult
RunPacketCompetitionStudy (BenchEncoder &encoder)
{
  const std::string query = "who owns cobalt rollback handoff";
  const double repeat_activation = SemanticScore (
      encoder, query, "cobalt rollback handoff generic recent packet");
  const double explicit_activation = SemanticScore (
      encoder, query, "Nadia verified ownership evidence.");
  const std::vector<cognitive::PacketProposal> proposals{
    { "semantic_repeat", "same-source", repeat_activation, 0.72, 0.20, 2.0,
      2 },
    { "explicit_packet", "nadia-explicit", explicit_activation, 0.70, 0.40,
      2.0, 2 },
  };

  StudyResult result{ "lida_packet_competition", "LIDA" };
  result.off_target = explicit_activation;
  result.off_comparison = repeat_activation;
  result.off_winner = Winner (result.off_target, result.off_comparison);

  ScopedEnvVar guard (cognitive::kPacketCompetitionFlag, "1");
  const auto on
      = cognitive::EnvFlagEnabled (cognitive::kPacketCompetitionFlag)
            ? cognitive::SelectPacketProposal (proposals, { "same-source" })
            : cognitive::PacketProposalSelection{};
  result.on_target = on.index == 1 ? on.score : proposals[1].activation;
  result.on_comparison = on.index == 0 ? on.score : proposals[0].activation;
  result.on_winner = on.index == 1 ? "target" : "comparison";
  result.passed = result.off_winner == "comparison"
                  && result.on_winner == "target";
  PrintStudy (result);
  return result;
}

StudyResult
RunCueRarityStudy (BenchEncoder &encoder)
{
  const std::string query = "cobalt rollback owner not dashboard";
  const double target_semantic = SemanticScore (
      encoder, query, "Nadia owned the rollback after the cobalt launch.");
  const double comparison_semantic = SemanticScore (
      encoder, query, "cobalt rollback dashboard status and generic owner note");

  StudyResult result{ "soar_cue_rarity_negative", "Soar" };
  result.off_target = target_semantic;
  result.off_comparison = comparison_semantic;
  result.off_winner = Winner (result.off_target, result.off_comparison);

  ScopedEnvVar guard (cognitive::kCueRarityFlag, "1");
  result.on_target = cognitive::EnvFlagEnabled (cognitive::kCueRarityFlag)
                         ? cognitive::CueRarityProductScore (
                               target_semantic, { { 0.92, 1.0, false } },
                               128.0, 0.88)
                         : target_semantic;
  result.on_comparison
      = cognitive::EnvFlagEnabled (cognitive::kCueRarityFlag)
            ? cognitive::CueRarityProductScore (
                  comparison_semantic,
                  { { 0.70, 80.0, false }, { 0.90, 4.0, true } }, 128.0,
                  0.35)
            : comparison_semantic;
  result.on_winner = Winner (result.on_target, result.on_comparison);
  result.passed = result.off_winner == "comparison"
                  && result.on_winner == "target";
  PrintStudy (result);
  return result;
}

StudyResult
RunUsefulnessStudy (BenchEncoder &encoder)
{
  const std::string query = "release checklist rollback";
  const double target_semantic = SemanticScore (
      encoder, query, "Nadia's recurring release routine starts from rollback "
                      "ownership and has repeated successful use.");
  const double comparison_semantic = SemanticScore (
      encoder, query, "release checklist rollback generic dashboard");

  StudyResult result{ "ona_usefulness_rank", "ONA" };
  result.off_target = target_semantic;
  result.off_comparison = comparison_semantic;
  result.off_winner = Winner (result.off_target, result.off_comparison);

  ScopedEnvVar guard (cognitive::kUsefulnessFlag, "1");
  const double target_usefulness = cognitive::UsefulnessScore (
      12.0, 6.0, 4.0, 10'000.0, 600'000.0);
  const double comparison_usefulness = cognitive::UsefulnessScore (
      0.0, 0.0, 0.0, 10'000.0, 600'000.0);
  result.on_target = cognitive::EnvFlagEnabled (cognitive::kUsefulnessFlag)
                         ? cognitive::UsefulnessRankScore (target_semantic,
                                                           target_usefulness)
                         : target_semantic;
  result.on_comparison
      = cognitive::EnvFlagEnabled (cognitive::kUsefulnessFlag)
            ? cognitive::UsefulnessRankScore (comparison_semantic,
                                              comparison_usefulness)
            : comparison_semantic;
  result.on_winner = Winner (result.on_target, result.on_comparison);
  result.passed = result.off_winner == "comparison"
                  && result.on_winner == "target";
  PrintStudy (result);
  return result;
}

StudyResult
RunEvidenceLanesStudy (BenchEncoder &encoder)
{
  const std::string query = "who owns cobalt rollback";
  const double target_semantic = SemanticScore (
      encoder, query, "Nadia has source-backed ownership evidence.");
  const double comparison_semantic = SemanticScore (
      encoder, query, "who owns cobalt rollback generic query match");

  StudyResult result{ "clarion_explicit_implicit_lanes", "CLARION" };
  result.off_target = target_semantic;
  result.off_comparison = comparison_semantic;
  result.off_winner = Winner (result.off_target, result.off_comparison);

  ScopedEnvVar guard (cognitive::kEvidenceLanesFlag, "1");
  result.on_target = cognitive::EnvFlagEnabled (cognitive::kEvidenceLanesFlag)
                         ? cognitive::FuseExplicitImplicitEvidence (
                               0.95, target_semantic, 0.25, 0.95)
                         : target_semantic;
  result.on_comparison
      = cognitive::EnvFlagEnabled (cognitive::kEvidenceLanesFlag)
            ? cognitive::FuseExplicitImplicitEvidence (
                  0.10, comparison_semantic, 0.10, 0.70)
            : comparison_semantic;
  result.on_winner = Winner (result.on_target, result.on_comparison);
  result.passed = result.off_winner == "comparison"
                  && result.on_winner == "target";
  PrintStudy (result);
  return result;
}

StudyResult
RunFactorFusionStudy (BenchEncoder &encoder)
{
  const std::string query = "cobalt rollback handoff source confidence";
  const double target_semantic = SemanticScore (
      encoder, query, "Nadia cobalt rollback handoff source confidence.");
  const double comparison_semantic = SemanticScore (
      encoder, query,
      "cobalt rollback handoff source confidence exact words but weak source");

  StudyResult result{ "sigma_factor_fusion", "Sigma" };
  result.off_target = target_semantic;
  result.off_comparison = comparison_semantic;
  result.off_winner = Winner (result.off_target, result.off_comparison);

  ScopedEnvVar guard (cognitive::kFactorFusionFlag, "1");
  result.on_target = cognitive::EnvFlagEnabled (cognitive::kFactorFusionFlag)
                         ? cognitive::ProductOfExpertsFusion (
                               { { std::max (target_semantic, 0.72), 1.0 },
                                 { 0.78, 1.0 },
                                 { 0.82, 1.0 } })
                         : target_semantic;
  result.on_comparison
      = cognitive::EnvFlagEnabled (cognitive::kFactorFusionFlag)
            ? cognitive::ProductOfExpertsFusion (
                  { { comparison_semantic, 1.0 }, { 0.92, 1.0 },
                    { 0.22, 1.0 } })
            : comparison_semantic;
  result.on_winner = Winner (result.on_target, result.on_comparison);
  result.passed = result.off_winner == "comparison"
                  && result.on_winner == "target";
  PrintStudy (result);
  return result;
}

} // namespace

int
main (int argc, char **argv)
{
  try
    {
      const std::string models_dir
          = cortext::benchmark::ParseModelsDirArg (argc, argv);
      BenchEncoder encoder (models_dir);
      std::cout << "encoder_backend=" << encoder.backend_name ()
                << " model=" << encoder.resolved_model_path ().string ()
                << "\n";

      const std::vector<StudyResult> results{
        RunAttentionLedgerStudy (encoder),
        RunPacketCompetitionStudy (encoder),
        RunCueRarityStudy (encoder),
        RunUsefulnessStudy (encoder),
        RunEvidenceLanesStudy (encoder),
        RunFactorFusionStudy (encoder),
      };

      int passed = 0;
      for (const auto &result : results)
        {
          if (result.passed)
            {
              ++passed;
            }
        }
      std::cout << "summary=" << passed << "/" << results.size () << " passed\n";
      return passed == static_cast<int> (results.size ()) ? 0 : 1;
    }
  catch (const std::exception &e)
    {
      std::cerr << "cognitive_mechanism_ablation_bench failed: " << e.what ()
                << "\n";
      return 2;
    }
}
