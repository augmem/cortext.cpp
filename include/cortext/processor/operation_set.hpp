#pragma once

#include "cortext/internal/cancellation.hpp"
#include "cortext/processor/operation.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <chrono>
#include <memory>
#include <tuple>
#include <type_traits>
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
/// sequentially, composed at runtime via Add().
///
/// Prefer the statically composed OperationSet<Ops...> template; this
/// dynamic variant remains for tests and benchmarks that assemble small
/// ad-hoc sequences.
class DynamicOperationSet : public IOperation
{
public:
  /// @brief Variadic template constructor for clean construction.
  ///
  /// Takes ownership of the provided operations.
  template <typename... TOps>
  explicit DynamicOperationSet (std::unique_ptr<TOps>... ops)
  {
    // Use a fold expression to move all operations into the vector.
    (operations_.push_back (std::move (ops)), ...);
  }

  void
  Add (std::unique_ptr<IOperation> op)
  {
    operations_.push_back (std::move (op));
  }

  /// @brief Executes each contained operation in sequence.
  /// @param context The context to pass to each operation.
  /// @param tx The active database transaction for this signal.
  void
  Execute (OperationContext &context, Transaction &tx) const override
  {
    for (const auto &op : operations_)
      {
        internal::ThrowIfStopRequested ();
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
        internal::ThrowIfStopRequested ();
      }
  }

private:
  std::vector<std::unique_ptr<IOperation> > operations_;
};

template <typename... Ops> class OperationSet;

namespace operation_set_detail
{

inline std::string
Demangle (const char *name)
{
  if (!name)
    {
      return "unknown";
    }
#if defined(__GNUG__) || defined(__clang__)
  int status = 0;
  char *demangled = abi::__cxa_demangle (name, nullptr, nullptr, &status);
  if (demangled)
    {
      std::string out = (status == 0) ? std::string (demangled)
                                      : std::string (name);
      std::free (demangled);
      return out;
    }
#endif
  return std::string (name);
}

/// Demangled once per type; DynamicOperationSet pays this per execution.
template <typename T>
const std::string &
TypeName ()
{
  static const std::string name = Demangle (typeid (T).name ());
  return name;
}

template <typename T> struct IsOperationSet : std::false_type
{
};

template <typename... Ops>
struct IsOperationSet<OperationSet<Ops...> > : std::true_type
{
};

/// True when Op declares its contract (Input/Output aliases, normally via
/// the Operation<Requires<...>, Satisfies<...>> base).
template <typename Op, typename = void> struct HasContract : std::false_type
{
};

template <typename Op>
struct HasContract<Op, std::void_t<typename Op::Input, typename Op::Output> >
    : std::true_type
{
};

/// Contract accessor with a permissive default for unported operations:
/// requires nothing, satisfies nothing. Sound only while validation is
/// gated on kContractsComplete (see OperationSet).
template <typename Op, bool = HasContract<Op>::value> struct ContractOf
{
  using Input = Requires<>;
  using Output = Satisfies<>;
};

template <typename Op> struct ContractOf<Op, true>
{
  using Input = typename Op::Input;
  using Output = typename Op::Output;
};

template <typename Tag, typename List> struct Contains;

template <typename Tag, template <typename...> class List, typename... Ts>
struct Contains<Tag, List<Ts...> >
    : std::bool_constant<(std::is_same_v<Tag, Ts> || ...)>
{
};

template <typename A, typename B> struct Concat;

template <template <typename...> class LA, typename... As,
          template <typename...> class LB, typename... Bs>
struct Concat<LA<As...>, LB<Bs...> >
{
  using type = LA<As..., Bs...>;
};

template <typename Tag, typename List> struct Prepend;

template <typename Tag, template <typename...> class List, typename... Ts>
struct Prepend<Tag, List<Ts...> >
{
  using type = List<Tag, Ts...>;
};

/// Tags of R that are not contained in S, as a Requires<...>.
template <typename R, typename S> struct Unsatisfied;

template <template <typename...> class L, typename S>
struct Unsatisfied<L<>, S>
{
  using type = Requires<>;
};

template <template <typename...> class L, typename T0, typename... Ts,
          typename S>
struct Unsatisfied<L<T0, Ts...>, S>
{
  using Rest = typename Unsatisfied<L<Ts...>, S>::type;
  using type = std::conditional_t<Contains<T0, S>::value, Rest,
                                  typename Prepend<T0, Rest>::type>;
};

/// Walks the operation sequence accumulating what has been satisfied so
/// far and which requirements must come from outside the set.
template <typename Satisfied, typename External, typename... Rest>
struct ChainContract
{
  using SatisfiedOut = Satisfied;
  using ExternalInput = External;
};

template <typename Satisfied, typename External, typename First,
          typename... Rest>
struct ChainContract<Satisfied, External, First, Rest...>
{
private:
  using FirstInput = typename ContractOf<First>::Input;
  using FirstOutput = typename ContractOf<First>::Output;
  using NextSatisfied = typename Concat<Satisfied, FirstOutput>::type;
  using NextExternal = typename Concat<
      External, typename Unsatisfied<FirstInput, Satisfied>::type>::type;
  using Next = ChainContract<NextSatisfied, NextExternal, Rest...>;

public:
  using SatisfiedOut = typename Next::SatisfiedOut;
  using ExternalInput = typename Next::ExternalInput;
};

} // namespace operation_set_detail

/// @brief A statically composed sequence of operations.
///
/// OperationSet<A, B, C> executes A, B, C in order, with the same
/// per-operation cancellation, telemetry span, and timing behavior as
/// DynamicOperationSet, but the composition is a type: members are stored
/// by value (no per-operation heap allocation), operation names are
/// demangled once per type instead of per execution, and contract checking
/// can run at compile time.
///
/// An OperationSet is itself an IOperation, so sets nest:
/// OperationSet<CoreStage, RetrievalStage> where each stage is an
/// OperationSet. Nested sets execute their members directly (no extra
/// span/timing entry for the stage wrapper). The type-list shape is also
/// where future combinators land (e.g. a Fork<D, E> member executing D and
/// E concurrently once the transaction story permits, with the
/// Requires/Satisfies contracts proving the absence of conflicts).
///
/// Contract aggregation: Input lists every context value required by a
/// member but not satisfied by an earlier member; Output lists everything
/// the members satisfy. Both are meaningful only once members declare
/// contracts; kContractsComplete reports whether they all do. Call sites
/// that need a fully self-contained set assert IsSelfContained<S> once
/// migration is complete.
template <typename... Ops> class OperationSet final : public IOperation
{
  static_assert (sizeof...(Ops) > 0,
                 "OperationSet needs at least one operation");
  static_assert ((std::is_base_of_v<IOperation, Ops> && ...),
                 "every OperationSet member must implement IOperation");

  using Chain = operation_set_detail::ChainContract<Satisfies<>, Requires<>,
                                                    Ops...>;

public:
  using Input = typename Chain::ExternalInput;
  using Output = typename Chain::SatisfiedOut;

  /// True once every member (recursively, via nested sets' own aggregation)
  /// declares its contract; until then Input/Output understate the truth
  /// and must not be used for validation.
  static constexpr bool kContractsComplete
      = (operation_set_detail::HasContract<Ops>::value && ...);

  OperationSet () = default;

  explicit OperationSet (Ops... ops) : ops_ (std::move (ops)...) {}

  void
  Execute (OperationContext &context, Transaction &tx) const override
  {
    std::apply ([&] (const Ops &...ops)
                { (this->ExecuteOne (ops, context, tx), ...); },
                ops_);
  }

private:
  template <typename Op>
  void
  ExecuteOne (const Op &op, OperationContext &context, Transaction &tx) const
  {
    if constexpr (operation_set_detail::IsOperationSet<Op>::value)
      {
        op.Execute (context, tx);
      }
    else
      {
        internal::ThrowIfStopRequested ();
        const std::string &op_type_name
            = operation_set_detail::TypeName<Op> ();
        context.SetCurrentOperationType (op_type_name);
        telemetry::ScopedSpan span (
            "cortext.operation",
            { telemetry::Attribute::String ("cortext.operation_type",
                                            op_type_name) });
        const auto op_start = std::chrono::steady_clock::now ();
        op.Execute (context, tx);
        const auto op_end = std::chrono::steady_clock::now ();
        const double ms = std::chrono::duration_cast<
            std::chrono::duration<double, std::milli> > (op_end - op_start)
                              .count ();
        context.AddOperationTiming (op_type_name, ms);
        internal::ThrowIfStopRequested ();
      }
  }

  std::tuple<Ops...> ops_;
};

/// True when S's members all declare contracts and every requirement is
/// satisfied inside S itself. The root build site asserts this once all
/// operations are ported to Operation<Requires<...>, Satisfies<...>>.
template <typename S>
inline constexpr bool IsSelfContained
    = S::kContractsComplete && std::is_same_v<typename S::Input, Requires<> >;

} // namespace cortext
