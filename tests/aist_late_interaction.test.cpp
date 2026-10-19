#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "../src/models/aist_embedded_model.hpp"
#include "../src/models/aist_late_interaction.hpp"
#include "test_helpers.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <unistd.h>

using cortext::EncoderPositions;
using cortext::LateInteractionSet;
using cortext::internal::LateInteractionModel;

namespace
{

const std::filesystem::path kSource = CORTEXT_TEST_SOURCE_DIR;

// The reranker ships as Git shards: the test resolves it the way the engine
// does (env override, beside the AIST model, else the linked shards).
std::unique_ptr<LateInteractionModel>
ShippedModel ()
{
  const auto path = cortext::internal::ResolveLateInteractionModelPath (
      kSource / "models/AIST-87M-GGUF/AIST-87M_q8_0.gguf");
  REQUIRE (path.has_value ());
  return LateInteractionModel::Load (*path);
}

struct FloatBlocks
{
  std::vector<float> data;
  std::size_t at = 0;

  std::vector<float>
  Take (std::size_t n)
  {
    REQUIRE (at + n <= data.size ());
    std::vector<float> out (data.begin () + static_cast<std::ptrdiff_t> (at),
                            data.begin () + static_cast<std::ptrdiff_t> (at + n));
    at += n;
    return out;
  }
};

EncoderPositions
Positions (const std::string &tower, std::vector<float> values, int count,
           int dim)
{
  EncoderPositions p;
  p.tower = tower;
  p.count = count;
  p.dim = dim;
  p.values = std::move (values);
  return p;
}

} // namespace

TEST_CASE ("Late-interaction codes serialize round trip", "[late_interaction]")
{
  LateInteractionSet set;
  set.count = 3;
  set.dim = 4;
  set.codes = { 1, -2, 3, -4, 127, -127, 0, 5, 6, 7, 8, 9 };
  const auto blob = LateInteractionModel::Serialize (set);
  REQUIRE (blob.size () == 4 + 12);
  LateInteractionSet back;
  REQUIRE (LateInteractionModel::Deserialize (blob.data (), blob.size (), back));
  CHECK (back.count == 3);
  CHECK (back.dim == 4);
  CHECK (back.codes == set.codes);
  CHECK_FALSE (LateInteractionModel::Deserialize (blob.data (), blob.size () - 1, back));
  CHECK (LateInteractionModel::Serialize (LateInteractionSet{}).empty ());
}

TEST_CASE ("Late-interaction model matches the Python reference",
           "[late_interaction]")
{
  // tools/reranker/parity_fixture.py: same positions through codes.kmeans +
  // li.py scoring.
  const auto model = ShippedModel ();
  std::ifstream js (kSource / "tests/fixtures/late_interaction/parity.json");
  REQUIRE (js);
  const nlohmann::json fx = nlohmann::json::parse (js);
  std::ifstream bin (kSource / "tests/fixtures/late_interaction/parity.bin",
                     std::ios::binary | std::ios::ate);
  REQUIRE (bin);
  FloatBlocks blocks;
  blocks.data.resize (static_cast<std::size_t> (bin.tellg ()) / sizeof (float));
  bin.seekg (0);
  bin.read (reinterpret_cast<char *> (blocks.data.data ()),
            static_cast<std::streamsize> (blocks.data.size () * sizeof (float)));
  REQUIRE (model->Dim () == fx["dim"].get<int> ());
  REQUIRE (model->CodesPerMember () == fx["k"].get<int> ());

  const int qn = fx["query"]["positions"][0].get<int> ();
  const auto qpos = blocks.Take (static_cast<std::size_t> (qn) * 384);
  const auto qglob = blocks.Take (256);
  const LateInteractionSet query
      = model->QuerySet (Positions ("text", qpos, qn, 384));
  REQUIRE (query.count == qn);
  Eigen::VectorXf query_global = Eigen::Map<const Eigen::VectorXf> (qglob.data (), 256);

  std::vector<std::vector<LateInteractionModel::Member>> memories;
  std::size_t code_total = 0;
  std::size_t code_close = 0;
  for (const auto &mem : fx["memories"])
    {
      std::vector<LateInteractionModel::Member> members;
      for (const auto &member : mem)
        {
          // members default to the text tower; image members carry patches
          const std::string tower = member.value ("tower", "text");
          const int n = member["positions"][0].get<int> ();
          const int dim = member["positions"][1].get<int> ();
          const auto pos = blocks.Take (static_cast<std::size_t> (n * dim));
          const auto glob = blocks.Take (256);
          LateInteractionModel::Member m;
          m.codes = model->MemorySet (Positions (tower, pos, n, dim));
          const auto expected = member["codes"].get<std::vector<int>> ();
          REQUIRE (m.codes.codes.size () == expected.size ());
          for (std::size_t i = 0; i < expected.size (); ++i)
            {
              ++code_total;
              code_close += std::abs (m.codes.codes[i] - expected[i]) <= 1 ? 1 : 0;
            }
          m.global = Eigen::Map<const Eigen::VectorXf> (glob.data (), 256);
          members.push_back (std::move (m));
        }
      memories.push_back (std::move (members));
    }
  // int8 rounding at a .5 boundary may differ by one step between runtimes
  CHECK (code_close == code_total);
  const auto scores = model->Score (query, query_global, memories);
  const auto expected = fx["scores"].get<std::vector<float>> ();
  REQUIRE (scores.size () == expected.size ());
  for (std::size_t i = 0; i < scores.size (); ++i)
    {
      CHECK (scores[i] == Catch::Approx (expected[i]).margin (2e-3));
    }
}

TEST_CASE ("Late-interaction memories without members score last",
           "[late_interaction]")
{
  const auto model = ShippedModel ();
  // A memory with no members scores below one that has any.
  LateInteractionSet query;
  Eigen::VectorXf g = Eigen::VectorXf::Zero (256);
  g (0) = 1.0F;
  std::vector<std::vector<LateInteractionModel::Member>> memories (2);
  LateInteractionModel::Member member;
  member.global = g;
  memories[1].push_back (member);
  const auto scores = model->Score (query, g, memories);
  REQUIRE (scores.size () == 2);
  CHECK (scores[1] > scores[0]);
}

TEST_CASE ("Late-interaction model resolves from the linked shards when none "
           "sits beside the AIST model",
           "[late_interaction]")
{
  if (!cortext::HasEmbeddedAistModel ())
    {
      SKIP ("build links no AIST shards (CORTEXT_EMBED_AIST_MODEL=OFF)");
    }
  const auto root = std::filesystem::temp_directory_path ()
                    / ("cortext-li-shards-" + std::to_string (::getpid ()));
  std::filesystem::remove_all (root);
  cortext::testing::ScopedEnvVar cache ("CORTEXT_MODEL_CACHE_DIR",
                                        (root / "cache").string ());
  cortext::testing::ScopedEnvVar no_override ("CORTEXT_LI_MODEL_PATH");
  // A caller-supplied AIST model with nothing beside it (a binding's own
  // reassembled cache) still gets the reranker the library ships.
  const auto lone = root / "elsewhere" / "AIST-87M_q8_0.gguf";
  const auto path = cortext::internal::ResolveLateInteractionModelPath (lone);
  REQUIRE (path.has_value ());
  CHECK (path->parent_path ().parent_path () == root / "cache");

  // Byte-identical to the tracked parts listed in the manifest, in order.
  const auto manifest = nlohmann::json::parse (
      std::ifstream (kSource / "models/manifest.json"));
  std::string expected;
  for (const auto &asset : manifest.at ("assets"))
    {
      if (asset.at ("filename") != "AIST-87M-GGUF/AIST-87M-LI.gguf")
        {
          continue;
        }
      for (const auto &chunk : asset.at ("chunks"))
        {
          std::ifstream part (kSource / "models"
                                  / chunk.at ("filename").get<std::string> (),
                              std::ios::binary);
          expected.append (std::istreambuf_iterator<char> (part), {});
        }
      CHECK (expected.size () == asset.at ("size").get<std::size_t> ());
    }
  std::ifstream got_file (*path, std::ios::binary);
  const std::string got (std::istreambuf_iterator<char> (got_file), {});
  REQUIRE (!expected.empty ());
  CHECK (got == expected);
  CHECK (LateInteractionModel::Load (*path) != nullptr);
  std::filesystem::remove_all (root);
}
