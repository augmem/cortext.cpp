#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/working_memory.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::WorkingMemory;

TEST_CASE ("Alg24 inserts new WM slot when margin exceeds cost",
           "[operations][working_memory][insert]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Constant (4, 1.0f);
  s.timestamp = 100;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;

  // Ensure empty WM and permissive threshold
  pctx.T_dynamic = 0.2;
  pctx.weight_relevance = 0.5;

  WorkingMemory op;
  OperationContext ctx (s, pctx, cfg, buf);
  ctx.SetCompositeScore (0.9); // benefit high
  op.Execute (ctx);

  REQUIRE (pctx.wm_last_accepted == true);
  REQUIRE (pctx.wm_last_chunked == false);
  REQUIRE (pctx.wm_slots.size () == 1);
  REQUIRE (pctx.wm_slots.back ().embedding.size () == s.embedding.size ());
}

TEST_CASE ("Alg24 chunks into best-matching slot above threshold",
           "[operations][working_memory][chunk]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.embedding[0] = 1.0f; // unit along x
  s.timestamp = 200;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 1.0;       // high focus → high chunk threshold ~0.9
  cfg.sensitivity = 0.3; // modest maintenance cost
  cfg.stability = 0.6;
  std::vector<BufferedWriteInstruction> buf;

  // Seed an existing slot with the same direction
  ProcessorContext::WMSlot slot;
  slot.embedding = s.embedding;
  slot.strength = 1.0;
  slot.last_ts = static_cast<double> (s.timestamp);
  slot.pos_index = 0;
  pctx.wm_slots.push_back (slot);
  const std::size_t before_size = pctx.wm_slots.size ();
  const double before_strength = pctx.wm_slots.front ().strength;

  // Threshold permissive enough with high benefit
  pctx.T_dynamic = 0.2;
  WorkingMemory op;
  OperationContext ctx (s, pctx, cfg, buf);
  ctx.SetCompositeScore (0.95);
  op.Execute (ctx);

  REQUIRE (pctx.wm_last_accepted == true);
  REQUIRE (pctx.wm_last_chunked == true);
  REQUIRE (pctx.wm_slots.size () == before_size); // no new slot
  REQUIRE (pctx.wm_slots.front ().strength
           >= Catch::Approx (before_strength).epsilon (1e-6));
}

TEST_CASE ("Alg24 maintenance decays and removes empty slots",
           "[operations][working_memory][maintenance]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Constant (3, 0.5f);
  s.timestamp = 10;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5; // cost_per_slot = 0.10
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;

  // One slot at t=0 with strength 1.0 → after 10s at 0.1/s → 0
  ProcessorContext::WMSlot slot;
  slot.embedding = s.embedding;
  slot.strength = 1.0;
  slot.last_ts = 0.0;
  slot.pos_index = 0;
  pctx.wm_slots.push_back (slot);
  pctx.T_dynamic = 0.95; // ensure reject to avoid adding strength

  WorkingMemory op;
  OperationContext ctx (s, pctx, cfg, buf);
  ctx.SetCompositeScore (0.2); // margin negative
  op.Execute (ctx);

  REQUIRE (pctx.wm_slots.size () == 0); // removed after decay to zero
  REQUIRE (pctx.wm_last_accepted == false);
  REQUIRE (pctx.wm_last_chunked == false);
}

TEST_CASE ("Alg24 maintenance reduces strength without removal when dt small",
           "[operations][working_memory][maintenance][decay]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Constant (3, 0.5f);
  s.timestamp = 5; // 5s
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5; // cost_per_slot = 0.10
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;

  ProcessorContext::WMSlot slot;
  slot.embedding = s.embedding;
  slot.strength = 1.0;
  slot.last_ts = 0.0;
  slot.pos_index = 0;
  pctx.wm_slots.push_back (slot);
  pctx.T_dynamic = 0.95; // force rejection

  WorkingMemory op;
  OperationContext ctx (s, pctx, cfg, buf);
  ctx.SetCompositeScore (0.2);
  op.Execute (ctx);

  REQUIRE (pctx.wm_slots.size () == 1);
  const double expected = 1.0 - 0.10 * 5.0;
  REQUIRE (pctx.wm_slots.front ().strength == Catch::Approx (expected));
  REQUIRE (pctx.wm_slots.front ().last_ts
           == Catch::Approx (static_cast<double> (s.timestamp)));
}
