#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <memory>
#include <typeinfo>
#include <utility>
#include <vector>

namespace cortext
{

/// @brief An operation that executes a series of other operations
/// sequentially.
class OperationSet : public IOperation
{
public:
  /// @brief Variadic template constructor for clean construction.
  ///
  /// Takes ownership of the provided operations.
  template <typename... TOps>
  explicit OperationSet (std::unique_ptr<TOps>... ops)
  {
    // Use a fold expression to move all operations into the vector.
    (operations_.push_back (std::move (ops)), ...);
  }

  /// @brief Executes each contained operation in sequence.
  /// @param context The context to pass to each operation.
  /// @param tx The active database transaction for this signal.
  void
  Execute (OperationContext &context, Transaction &tx) const override
  {
    for (const auto &op : operations_)
      {
        const IOperation *op_ptr = op.get ();
        const char *op_type = op_ptr ? typeid (*op_ptr).name () : "unknown";
        telemetry::ScopedSpan span (
            "cortext.operation",
            { telemetry::Attribute::String ("cortext.operation_type",
                                            op_type ? op_type : "unknown") });
        op->Execute (context, tx);
      }
  }

private:
  std::vector<std::unique_ptr<IOperation> > operations_;
};

} // namespace cortext
