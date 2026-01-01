#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <chrono>
#include <memory>
#include <typeinfo>
#include <utility>
#include <vector>
#include <string>

#if defined(__GNUG__) || defined(__clang__)
#include <cxxabi.h>
#include <cstdlib>
#endif

namespace cortext
{

namespace
{
inline std::string
DemangleOpType (const char *name)
{
  if (!name)
    {
      return "unknown";
    }
#if defined(__GNUG__) || defined(__clang__)
  int status = 0;
  char *demangled = abi::__cxa_demangle (name, nullptr, nullptr, &status);
  if (status == 0 && demangled)
    {
      std::string out (demangled);
      std::free (demangled);
      return out;
    }
  if (demangled)
    {
      std::free (demangled);
    }
#endif
  return std::string (name);
}
} // namespace

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
        const std::string op_type_name = DemangleOpType (op_type);
        context.SetCurrentOperationType (op_type_name);
        telemetry::ScopedSpan span (
            "cortext.operation",
            { telemetry::Attribute::String ("cortext.operation_type",
                                            op_type_name) });
        const auto op_start = std::chrono::steady_clock::now ();
        op->Execute (context, tx);
        const auto op_end = std::chrono::steady_clock::now ();
        const double ms = std::chrono::duration_cast<
            std::chrono::duration<double, std::milli> > (op_end - op_start)
                              .count ();
        context.AddOperationTiming (op_type_name, ms);
      }
  }

private:
  std::vector<std::unique_ptr<IOperation> > operations_;
};

} // namespace cortext
