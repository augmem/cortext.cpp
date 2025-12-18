#include "cortext/operations/signal_metrics_persistence.hpp"

#include "cortext/operations/metrics.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"

namespace cortext::operations
{

namespace
{

inline double
GetMetricValue (const std::unordered_map<Metric, double> &metrics, Metric m)
{
  auto it = metrics.find (m);
  return (it != metrics.end ()) ? it->second : 0.0;
}

} // namespace

void
PersistSignalMetrics::Execute (OperationContext &context, Transaction &tx) const
{
  const auto &metrics = context.GetAllMetrics ();
  const auto &signal = context.GetSignal ();

  // Extract all metric values
  double relevance = GetMetricValue (metrics, Metric::relevance);
  double mismatch = GetMetricValue (metrics, Metric::mismatch);
  double surprise = GetMetricValue (metrics, Metric::surprise);
  double rarity = GetMetricValue (metrics, Metric::rarity);
  double drift = GetMetricValue (metrics, Metric::drift);
  double contradiction = GetMetricValue (metrics, Metric::contradiction);
  double utility = GetMetricValue (metrics, Metric::utility);
  double periphery = GetMetricValue (metrics, Metric::periphery);
  double coverage = GetMetricValue (metrics, Metric::coverage);
  double salience = GetMetricValue (metrics, Metric::salience);
  double valence = GetMetricValue (metrics, Metric::valence);
  double arousal = GetMetricValue (metrics, Metric::arousal);

  // Get composite score and threshold
  auto composite_opt = context.GetCompositeScore ();
  double composite_score = composite_opt.has_value () ? *composite_opt : 0.0;
  double threshold_t = context.GetThresholdTDynamic ();

  // Write decision: 1 if composite_score > (T_dynamic - hysteresis)
  int write_decision = context.GetWriteDecision () ? 1 : 0;

  // Get structural metrics from context
  double coherence = context.GetCoherence ();
  double focus_spread = GetMetricValue (metrics, Metric::focus_spread);
  double f_effective = context.GetEffectiveFocus ();

  // v2: Update the most recent signal for this source_id with metrics
  // Metrics are stored inline on the signals table (merged from signal_metrics)
  tx.Execute (
      "UPDATE signals "
      "SET relevance = ?, mismatch = ?, surprise = ?, rarity = ?, "
      "    drift = ?, contradiction = ?, utility = ?, periphery = ?, "
      "    coverage = ?, salience = ?, valence = ?, arousal = ?, "
      "    score = ?, threshold_t = ?, write_decision = ?, "
      "    coherence = ?, focus_spread = ?, f_effective = ? "
      "WHERE signal_id = ("
      "  SELECT signal_id FROM signals "
      "  WHERE source_id = ? AND timestamp = ? "
      "  ORDER BY signal_id DESC LIMIT 1"
      ")",
      { relevance, mismatch, surprise, rarity, drift, contradiction, utility,
        periphery, coverage, salience, valence, arousal, composite_score,
        threshold_t, write_decision, coherence, focus_spread, f_effective,
        signal.source_id, static_cast<long long> (signal.timestamp) });
}

} // namespace cortext::operations
