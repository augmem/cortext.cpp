#pragma once

#include <memory>
#include <vector>

namespace cortext
{

class OperationContext;
class Transaction;

/// @brief Defines the interface for an operation in the signal processing
/// model.
///
/// An operation represents a single, atomic step in the process.
class IOperation
{
public:
  virtual ~IOperation () = default;

  /// @brief Executes the operation.
  /// @param context The context object for the current signal.
  /// @param tx The active database transaction for this signal.
  virtual void Execute (OperationContext &context, Transaction &tx) const = 0;
};

/// @brief Declares the context values an operation consumes.
///
/// Each parameter is a tag type identifying a value on the operation
/// context. Every required value must be satisfied by an earlier operation
/// in the assembled pipeline (or be a declared pipeline input); a
/// construction-time validator checks this for each pipeline variant.
template <typename... Tags>
struct Requires
{
};

/// @brief Declares the context values an operation produces.
///
/// Each parameter is a tag type identifying a value on the operation
/// context. Producing a value satisfies the requirements of later
/// operations in the pipeline.
template <typename... Tags>
struct Satisfies
{
};

/// @brief Contract-declaring base for operations.
///
/// Every pipeline operation declares its per-signal dataflow contract,
/// e.g.:
///
///   class ComputeCompositeScore
///       : public Operation<Requires<tags::MetricValues>,
///                          Satisfies<tags::CompositeScore>>
///
/// Tags live in cortext/processor/contract_tags.hpp. Validation happens at
/// compile time: OperationSet aggregates member contracts in sequence
/// (operation_set.hpp), and the pipeline build site static_asserts that
/// both pipeline variants are self-contained - an operation consuming a
/// value no earlier operation produces fails the build, with the missing
/// tags listed in the set's Input alias.
template <typename TRequires, typename TSatisfies>
class Operation : public IOperation
{
public:
  using Input = TRequires;
  using Output = TSatisfies;
};

} // namespace cortext
