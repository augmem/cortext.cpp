#pragma once

#include "cortext/internal/cancellation.hpp"
#include "cortext/processor/operation.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/processor/operation_fork.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <chrono>
#include <memory>
#include <stdexcept>
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
    // While any Fork is outstanding, main-thread members share the store
    // with running branches: route their transaction through the same
    // per-method serialization the branches use (unless an enclosing set
    // already did).
    std::optional<SerializedTransaction> guarded;
    Transaction *effective = &tx;
    if (context.HasPendingForks ()
        && dynamic_cast<SerializedTransaction *> (&tx) == nullptr)
      {
        guarded.emplace (tx, context.ForkTransactionMutex ());
        effective = &*guarded;
      }

    if constexpr (operation_set_detail::IsOperationSet<Op>::value)
      {
        op.Execute (context, *effective);
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
        op.Execute (context, *effective);
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

/// @brief Executes its branches concurrently on the persistent worker
/// pool and returns immediately; the matching Join is the barrier.
///
/// Contract shape encodes fork-join correctness: Fork requires the union
/// of its branches' requirements (they must be satisfied before the fork
/// point) but satisfies NOTHING - branch outputs become visible only at
/// the matching Join, so any member between Fork and Join that consumes a
/// branch output fails the compile-time chain validation.
///
/// Branch safety is proven at compile time: branches must have pairwise
/// disjoint outputs (no write/write conflict) and no branch may consume a
/// sibling's output (such an edge demands sequencing, not forking). Tags
/// map to distinct context storage, so tag-disjoint branches do not race
/// on the OperationContext.
///
/// Transaction access from branches - and from main-thread members while
/// any fork is outstanding - serializes per method call through the
/// context's fork mutex (see SerializedTransaction): compute runs fully
/// parallel, store access interleaves safely on the one connection.
template <typename... Branches> class Fork final : public IOperation
{
  static_assert (sizeof...(Branches) >= 2,
                 "Fork needs at least two branches; use the operation "
                 "directly otherwise");
  static_assert ((std::is_base_of_v<IOperation, Branches> && ...),
                 "every Fork branch must implement IOperation");
  static_assert ((operation_set_detail::HasContract<Branches>::value && ...),
                 "every Fork branch must declare its contract");
  static_assert (ForkBranchesConflictFree<Branches...>,
                 "Fork branches conflict: overlapping outputs or a branch "
                 "consuming a sibling's output");

public:
  using Input = typename fork_detail::UnionInputs<Branches...>::type;
  using Output = Satisfies<>;

  Fork () = default;

  explicit Fork (Branches... branches) : branches_ (std::move (branches)...)
  {
  }

  void
  Execute (OperationContext &context, Transaction &tx) const override
  {
    internal::ThrowIfStopRequested ();
    auto fork = std::make_shared<PendingFork> ();
    fork->remaining = static_cast<int> (sizeof...(Branches));
    context.PushPendingFork (fork);

    // If a surrounding fork already routed us through the serialized
    // wrapper, reuse the incoming transaction directly: it shares the
    // same context-wide mutex and wrapping again would self-deadlock.
    Transaction *branch_tx = &tx;
    std::shared_ptr<SerializedTransaction> wrapper;
    if (dynamic_cast<SerializedTransaction *> (&tx) == nullptr)
      {
        wrapper = std::make_shared<SerializedTransaction> (
            tx, context.ForkTransactionMutex ());
        branch_tx = wrapper.get ();
      }

    auto &pool = fork_detail::OperationWorkerPool::Instance ();
    std::apply (
        [&] (const Branches &...branches)
        {
          (pool.Submit (
               [fork, wrapper, branch_tx, &context, &branches] ()
               {
                 std::exception_ptr failure;
                 try
                   {
                     branches.Execute (context, *branch_tx);
                   }
                 catch (...)
                   {
                     failure = std::current_exception ();
                   }
                 fork->BranchDone (failure);
               }),
           ...);
        },
        branches_);
  }

private:
  std::tuple<Branches...> branches_;
};

/// @brief Barrier for the oldest outstanding Fork.
///
/// Join<D, E> names the branch types of the Fork it pairs with (forks and
/// joins pair FIFO at runtime) and satisfies the union of their outputs -
/// this is where branch results become visible to the compile-time chain,
/// so consumers of forked work must sit after the Join. The first branch
/// failure rethrows here.
template <typename... Branches> class Join final : public IOperation
{
  static_assert (sizeof...(Branches) >= 1,
                 "Join names the branch types of the Fork it pairs with");
  static_assert ((operation_set_detail::HasContract<Branches>::value && ...),
                 "every joined branch must declare its contract");

public:
  using Input = Requires<>;
  using Output = typename fork_detail::UnionOutputs<Branches...>::type;

  void
  Execute (OperationContext &context, Transaction &tx) const override
  {
    (void)tx;
    auto fork = context.PopPendingFork ();
    if (!fork)
      {
        throw std::logic_error (
            "Join executed with no outstanding Fork in this context");
      }
    fork->Wait ();
    if (!fork->failures.empty ())
      {
        std::rethrow_exception (fork->failures.front ());
      }
  }
};

} // namespace cortext
