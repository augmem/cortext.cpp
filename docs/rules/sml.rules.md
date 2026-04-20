# sml.rules.md

## 1. scope
these rules define how to use boost.SML (boost::ext SML) to build a high-performance, real-time friendly architecture that behaves like a pure actor model while remaining synchronous run-to-completion (RTC) and using no message queue.

these rules apply to:
- boost.SML state machines (`boost::sml::sm<...>`) and their composition (composite state machines, orthogonal regions).
- synchronous dispatch only (no background workers, no mailboxes, no async buffering).

the rules assume the project-pinned boost.SML semantics as implemented in the
local header and utility dispatch table, including typed completion propagation
via `sml::completion<TEvent>`.

primary sources consulted (non-exhaustive)
- docs: https://boost-ext.github.io/sml/ (introduction), https://boost-ext.github.io/sml/tutorial.html (tutorial), https://boost-ext.github.io/sml/user_guide.html (user guide), https://boost-ext.github.io/sml/benchmarks.html (benchmarks), https://boost-ext.github.io/sml/overview.html (overview)
- source code: https://github.com/boost-ext/sml/blob/v1.1.13/include/boost/sml.hpp , https://github.com/boost-ext/sml/blob/v1.1.13/include/boost/sml/utility/dispatch_table.hpp
- author talk (overview of design/perf goals): https://www.youtube.com/watch?v=lg3t_iact5_fw

## 2. definitions
- actor: an isolated unit owning (1) exactly one SML `sm` instance and (2) its private context, processed only via event dispatch.
- event: a value (usually a trivial type) passed to `process_event`. publicly
  exposed events are immutable. internal-only events that are not publicly
  exposed MAY carry mutable fields when needed for synchronous RTC handoff.
- RTC chain: the complete, synchronous computation triggered by one top-level dispatch call, including SML internal anonymous transitions.
- quiescence: a stable configuration where no further internal (anonymous) transitions are enabled.
- orchestrator: the external driver that calls `process_event` on actors and provides time and ordering.
- no message queue: no SML `process_queue`, no SML `defer_queue`, no user mailbox, and no “post for later” mechanism.

## 3. core invariants
1. RTC invariant: A top-level `actor.dispatch(e)` MUST return only after the machine has reached quiescence for that event (and any internal anonymous transitions).
2. no-queue invariant: actors MUST NOT enable or use SML queuing policies (`sml::process_queue`, `sml::defer_queue`) and MUST NOT implement a mailbox.
3. determinism invariant: given identical initial state, identical event sequence, and identical event payloads, the sequence of executed actions and state changes MUST be identical.
4. single-writer invariant: during any RTC chain, exactly one thread MUST be executing inside any given actor’s `process_event`.
5. allocation invariant: no dynamic allocation (heap) MUST occur during dispatch (guards/actions/entry/exit/anonymous progress).
6. bounded-work invariant: each top-level dispatch MUST have a provable upper bound on executed transitions and on total work.

## 4. event model
1. event types SHOULD be small, trivially copyable, and contain only immutable payload.
   exception: internal-only events that are not publicly exposed MAY use mutable
   pointers/references for synchronous same-RTC handoff. such mutable payload
   MUST NOT be exposed via public API types, MUST NOT be retained beyond the
   top-level dispatch call, and MUST preserve deterministic bounded behavior.
2. events MUST NOT contain owning pointers or dynamic containers (e.g., `std::string`, `std::vector`) unless a custom allocator and strict “no allocate during dispatch” enforcement is in place.
3. event dispatch SHOULD be compile-time typed (`sm.process_event(TEvent{...})`). runtime-polymorphic “base event” dispatch SHOULD be avoided.
4. if runtime event IDs are required, the system MAY use `sml::utility::make_dispatch_table` (static jump table indexed by ID) and MUST validate the ID range before indexing. see `make_dispatch_table` implementation using a static function-pointer table indexed by `(id - event_range_begin)`. (source: `dispatch_table.hpp` in SML repo.)
5. each event MUST have a single “owning” actor responsible for interpreting it. cross-actor effects MUST be explicit (see section 8).
6. when internal RTC phases need the originating event payload, transitions
   SHOULD use typed completion propagation (`sml::completion<TEvent>`) so
   guards/actions can consume the original event directly.
7. event payload pointers/references MUST remain valid for the full top-level
   `process_event` call (including internal completion/anonymous progress and
   synchronous nested actor dispatch triggered within that RTC chain).
8. required event fields MUST be references, not pointers. pointers in event
   payloads MUST be reserved for optional/nullable fields or C ABI boundary
   types where references cannot be expressed.

## 5. state and context
1. state types SHOULD be “labels”, not storage. actor data MUST live in an explicit context object owned by the actor and passed into SML via dependency injection (constructor args to `sml::sm`).
2. actor context MUST have stable addresses for the lifetime of the state machine (no moves that change addresses) if references are injected.
3. construction-time dependencies order MUST NOT be relied on. the tutorial explicitly notes parameter order is “not specified”. prefer a single context aggregate or named dependency wrappers. (docs: tutorial section “create a state machine”.)
4. context mutation MUST be confined to actions (not guards) unless a guard mutation is proven to be side-effect free in terms of externally observable behavior (recommended: treat guards as pure).
5. state machine member functions MUST NOT read or write context directly. context
   access MUST be explicit via action/guard parameters or internal transitions.
6. state queries for external observers MUST be done using `visit_current_states` or `is(...)` and MUST NOT require locks in the steady state.
7. context MUST NOT mirror per-dispatch event payload solely to hand data from
   one internal phase to another.
8. context SHOULD store only machine-owned runtime state that truly spans
   phases/events; ephemeral request payload SHOULD stay in events and be
   propagated with typed transitions (for example `event<TEvent>`,
   `completion<TEvent>`).
9. context is for persistent actor-owned state only; per-dispatch orchestration
   scratch MUST NOT be stored in context.
10. per-dispatch phase handoff MUST use typed internal events
    (`events::*_done`, `events::*_error`, `sml::completion<TEvent>`) and MUST
    NOT mirror request/event data into context.
11. error progression MUST be modeled by explicit error states/events; do not
    use context error/status fields as control state.

## 6. actions and guards
### selection and evaluation order
1. guards MUST be pure predicates of `(event, context)` and MUST return `bool`.
2. for a given (state, event), transitions are attempted in transition-table order, and the first transition whose guard returns true is taken. this follows SML’s `transitions<T, ts...>::execute` recursion: it executes `T` first and only tries `ts...` if `T` does not execute. (source: `boost/sml.hpp`, `transitions<T, ts...>`.)
3. for an external transition with entry/exit enabled, the order MUST be: guard, on-exit, state update, action, on-entry. this follows `transition<...>::execute` which calls `on_exit`, updates current state, executes action, then calls `on_entry`. (source: `boost/sml.hpp`, `transition<state<s1>, state<s2>, event<E>, G, A>::execute`.)

### real-time and determinism constraints
4. guards and actions MUST be bounded time and MUST NOT block (no I/O waits, no mutex waits, no sleeps).
5. guards and actions MUST NOT allocate. if an action MUST allocate for rare paths (e.g., error reporting), it MUST do so outside dispatch and only pass references into dispatch.
6. guards MUST NOT read wall-clock time. time MUST be provided explicitly via events (section 10).
7. actions MUST NOT contain orchestration branching or validation logic. any runtime control-flow
   decision (success vs error, retries, mode selection) MUST be expressed as guarded transitions or
   explicit choice states.
8. runtime branching statements MUST NOT appear inside actions, state machine member methods, or
   functions called from actions/member methods. this ban includes `if`, `else if`, `switch`, and
   conditional operators (`?:`). all runtime control flow MUST be modeled as explicit guarded
   transitions or explicit choices/states in the transition graph. only compile-time conditionals
   (e.g., `if constexpr`, `#if`) are allowed inside actions, member methods, or functions called
   from actions/member methods.
9. actions SHOULD be short. long-running work MUST be split:
   - action initiates work and transitions to a “waiting” state.
   - A later external event represents completion (still no queues).
10. actions SHOULD be `noexcept` in production builds. if exceptions are enabled, the system MUST define a hard policy for exception events and document action-throws semantics (overview page notes different semantics for guard-throws vs action-throws).
11. avoid try/catch in guards/actions. exceptions MUST be avoided unless
    absolutely necessary and explicitly justified.
12. actors are independent: do not share a model between actors unless
    explicitly authorized by the user. only common actions/guards that are not
    variant-specific may be shared.
13. unexpected-event handling MUST NOT consume internal SML events. do NOT use
    `event<sml::_>` as an unexpected-event wildcard. use `sml::unexpected_event`
    for unexpected-event handling; it is only raised for unhandled external events.
14. when modeling unexpected events, always prefer:
    - `sml::unexpected_event<specific_external_event>` for explicit unexpected handling, or
    - `sml::unexpected_event<sml::_>` as the catchall (NO guard). guards that exclude
      `boost::sml::back::internal_event` will suppress the unexpected event itself because
      `unexpected_event<_>` is an internal_event.
15. guards MAY branch only on `(event, persistent_context)` and MUST NOT depend
    on dispatch-local context fields.
16. runtime control-flow emulation is forbidden in actions, state-machine member
    methods, and any function they call. forbidden patterns include:
    - single-pass conditional loops (for example `for (bool c = cond; c; c = false)`).
    - branch-case loops (for example `for (size_t k = branch; k == 0u/1u; k = 2u)`).
    - any loop whose purpose is choosing a control path rather than iterating data.
17. runtime-indexed selection used as branch substitution is forbidden in actions,
    state-machine member methods, and action callees. examples include choosing
    handlers/callbacks/return behavior from arrays using runtime branch indices.
    runtime control decisions MUST be modeled in transitions/guards.
18. loops in actions/details/member-method callees are allowed only for data-plane
    iteration with monotonic progress and bounded work. loops MUST NOT encode
    success/error/mode/retry/routing control decisions.
19. compile-time conditionals remain allowed in actions/member methods/action
    callees (`if constexpr`, `#if`, `#ifdef`). this allowance does not permit
    runtime control-flow emulation.

## 7. reentrancy and nested dispatch
1. an actor MUST NOT call its own `process_event` (directly or indirectly) from inside a guard/action. this prevents unbounded recursion and makes WCET analysis tractable. (motivation: `process_event` is synchronous and can be re-entered; SML users report deep call stacks if they do this.)
2. internal multi-step “microflows” within a single RTC chain MUST be modeled
   using typed completion transitions (`sml::completion<TEvent>`), anonymous
   transitions (eventless transitions), and/or entry actions, not by
   self-dispatch.
3. `sml::completion<TEvent>` and anonymous transitions MUST NOT be used to
   model per-element or per-item data loops. those loops MUST execute inside
   one allocation-free action/detail kernel per phase so compiler
   vectorization and predictable throughput are preserved.
4. anonymous transition graphs MUST be acyclic or MUST have a statically provable bound on firings per top-level event. SML’s `process_event` loops internal anonymous processing to quiescence (`while (process_internal_events(anonymous{}, ...)) {}`), so cycles can create unbounded work. (source: `boost/sml.hpp`, `sm_impl::process_event` loop.)
5. cross-actor nested dispatch (A calls B synchronously) MAY be used, but MUST obey:
   - no re-entrancy into the same actor instance within a single RTC chain.
   - acyclic call graph per top-level dispatch (enforced by orchestrator stack tracking or design discipline).
6. if multi-thread calls are possible, thread-safety MUST be enforced outside SML using a single external scheduler; using `sml::thread_safe<std::mutex>` inside actors SHOULD be avoided for real-time because lock acquisition is not bounded. (docs: overview “thread safety” shows optional locking policy.)

## 8. cross-actor interaction (no queues)
### allowed interaction patterns
1. synchronous send (recommended): actor A calls `actor_b.dispatch(ev)` directly as a normal function call. this provides actor-like isolation while preserving RTC and no-queue semantics.
2. synchronous request-reply: actor A calls `actor_b.dispatch(req)` and reads results via:
   - return values from `actor_b` non-SML methods, or
   - context fields exposed as read-only snapshots after `dispatch` returns.
3. publish-subscribe (no queue): the orchestrator MAY fan out an input event to multiple actors by deterministic ordering. each actor MUST treat the event as read-only input.
4. synchronous callbacks (allowed): events MAY carry `emel::callback`-style functors for
   immediate notification within the same RTC chain (e.g., request-reply without storing
   results in context). callbacks MUST be invoked before `dispatch` returns, MUST be bounded
   and non-allocating, MUST NOT call back into `process_event`, and MUST NOT be stored in
   actor context or retained for later.

### forbidden interaction patterns
5. actors MUST NOT store “callback handles” that later re-enter `process_event` asynchronously
   (this becomes a mailbox by another name).
6. actors MUST NOT enqueue follow-up work for later (no `sml::process_queue`,
   no `sml::defer_queue`, no user mailbox).

### deterministic ordering rules
6. cross-actor calls MUST have a single total order defined by the orchestrator. the order MUST be deterministic and testable (e.g., fixed actor ID ordering).
7. if an event needs to affect multiple actors, the orchestrator MUST define whether the operation is:
   - sequential (A then B then C) with well-defined intermediate visibility, or
   - transactional (collect decisions, then commit), implemented explicitly as a two-phase protocol with no queues.

## 9. performance rules
1. actors MUST compile with optimizations appropriate for latency (`-o2` or `-o3`) and with exceptions disabled where possible. SML is designed to not use exceptions internally and can be built with `-fno-exceptions`. (docs: overview “exception safety”.)
2. the dispatch mechanism SHOULD be constant-time over states. SML provides dispatch policies including a jump table (`dispatch_table[current_state]`) using a constexpr static function-pointer array. (source: `boost/sml.hpp`, `back::policies::jump_table`.)
3. for maximum predictability, actors SHOULD:
   - avoid RTTI and virtual dispatch in guards/actions,
   - avoid type erasure and `std::function` in hot paths,
   - keep transition tables in a single translation unit to maximize inlining and compile-time resolution.
4. each actor MUST have a hard cap on:
   - maximum number of transitions executed per top-level event,
   - maximum number of anonymous transitions executed per top-level event,
   - maximum number of cross-actor synchronous calls per top-level event.
5. memory footprint SHOULD be fixed and minimal. SML benchmarks report very small “memory usage” for the state machine itself when no queues are used. (docs: benchmarks page.)

## 10. time and scheduling
1. actors MUST be driven by an external scheduler (orchestrator). actors MUST NOT create threads or timers.
2. time MUST enter the system only as explicit event payload (e.g., `tick{now_ns, dt_ns}` or `deadline_reached{...}`).
3. timeout logic MUST be modeled as:
   - store a deadline in context on entry to a waiting state,
   - on each `tick`, guard checks `now >= deadline` and transitions.
4. each dispatch MUST have a time budget. the orchestrator SHOULD measure per-event execution time and enforce a policy:
   - hard fail (assert/terminate) in debug,
   - degrade gracefully (enter fault state) in production.
5. for determinism in simulation/replay, “now” MUST be supplied by the caller, not read from the OS clock.

## 11. composition rules
### hierarchy (composite state machines)
1. composite state machines MAY be used to represent hierarchical actors or subcomponents, but each submachine MUST still obey the same no-queue and RTC rules.
2. if using SML submachines, designers MUST account for SML’s event propagation behavior:
   - submachine gets first chance to handle events,
   - if unhandled, parent transitions may run (fallback). (source: `boost/sml.hpp`, `transitions_sub<sm<tsm>, ...>::execute_impl`.)
3. entry into a composite state MUST initialize its submachine(s) deterministically. SML updates composite state initialization during state updates. (source: `boost/sml.hpp`, `update_current_state` overload for `state<back::sm<T>>`.)
4. do NOT introduce ad-hoc shared base classes for machine wrappers (e.g. `sm_base`). inheritance from project base wrappers in `emel/sm.hpp` (for example `emel::sm<model>`) is acceptable. each machine MUST own its context and define its own `process_event` wrapper. share behavior via `actions.hpp`/`detail.hpp` helpers and `sm_any` dispatch.

### orthogonal regions
4. orthogonal regions (multiple initial states) MUST be designed so that the same event does not cause side effects in more than one region, unless those side effects commute and ordering does not matter.
5. if a cross-region coordination is required, it MUST be expressed via explicit context variables and deterministic guards, not by relying on region evaluation order.

## 12. observability and tracing
1. production tracing MUST be constant-time and bounded-memory. it MUST NOT allocate during dispatch.
2. the preferred integration point is `sml::logger<your_logger>` which can observe:
   - `log_process_event`, `log_state_change`, `log_action`, `log_guard`. (docs: user guide “policies” and logger requirements.)
3. logger callbacks MUST be deterministic and MUST NOT call back into `process_event`.
4. state inspection for debugging SHOULD use `visit_current_states` and `is(...)` (both are synchronous and can be used without instrumentation).

## 13. testing and validation
1. every actor MUST have tests that validate:
   - determinism (same initial state + same event sequence => same trace),
   - bounded internal work (max anonymous transitions per event),
   - no allocations during dispatch (instrument global new/delete in tests),
   - time budget compliance (measure cycles or ns per dispatch).
2. unit tests SHOULD use `sml::testing::sm` and `set_current_states(...)` to force configurations and test edge cases. (docs: user guide “testing::sm”.)
3. property-based tests SHOULD generate event sequences and assert invariants (never re-enter, never allocate, never exceed transition cap).
4. integration tests MUST include a single-threaded orchestrator that reproduces the exact dispatch order used in production.

## 14. reference examples
### example A: actor wrapper with synchronous, no-queue dispatch
```cpp
#include <boost/sml.hpp>
namespace sml = boost::sml;

struct tick { uint64_t now_ns; };
struct start { uint64_t now_ns; };

struct ctx {
  uint64_t deadline_ns = 0;
  // references to other actors go here, injected by reference.
};

struct actor_sm {
  auto operator()() const {
    using namespace sml;
    const auto set_deadline = [](ctx& c, const start& s) { c.deadline_ns = s.now_ns + 10'000'000; };
    const auto timed_out = [](const ctx& c, const tick& t) { return t.now_ns >= c.deadline_ns; };

    return make_transition_table(
      *"idle"_s + event<start> / set_deadline = "waiting"_s,
       "waiting"_s + event<tick> [ timed_out ] = "idle"_s
    );
  }
};

struct actor {
  ctx ctx;
  sml::sm<actor_sm> sm{ctx};

  template<class ev>
  bool dispatch(const ev& ev) { return sm.process_event(ev); }
};
```

### example B: bounded internal progress using anonymous transitions (no self-dispatch)
```cpp
#include <boost/sml.hpp>
namespace sml = boost::sml;

struct step {};
struct ctx { int i = 0; };

struct sm {
  auto operator()() const {
    using namespace sml;
    const auto inc = [](ctx& c) { ++c.i; };
    const auto more = [](const ctx& c) { return c.i < 3; };

    return make_transition_table(
      *"s0"_s + event<step> / inc = "s1"_s,
       "s1"_s [ more ] / inc = "s1"_s  // guarded anonymous transition
    );
  }
};
```

## 15. naming and style
1. exported/public C++ types MUST use PascalCase (not C API types).
2. non-exported/internal types MUST use lower_snake_case.
3. SML state names and events MUST use lower_snake_case unless explicitly exported.
4. transition rows in `make_transition_table(...)` MUST be written in
   destination-first form (`sml::state<dst> <= src + event [guard] / action`)
   for all new and modified state machines.
5. in destination-first rows, the destination state and `<=` MUST appear on
   the same line.
6. transition tables with more than a few rows MUST be organized in explicit
   visual sections (for example `//------------------------------------------------------------------------------//`
   and phase labels).
7. transition rows after the first row in a table MUST use leading-comma style
   to keep diffs stable and scanning consistent.
8. when table readability would be degraded by auto-formatting, transition
   tables MAY be wrapped with narrowly scoped `// clang-format off/on`.
9. NEVER use macros in models.
rule check: the anonymous self-loop is bounded by `i < 3`; cycles without bounds are forbidden.
rule check: the anonymous self-loop is bounded by `i < 3`; cycles without bounds are forbidden.

### example C: runtime event dispatch without dynamic allocation
```cpp
#include <boost/sml.hpp>
#include <boost/sml/utility/dispatch_table.hpp>
namespace sml = boost::sml;

struct runtime_ev { int id; /* payload */ };
struct ev1 { static constexpr auto id = 1; ev1(const runtime_ev&) {} };
struct ev2 { static constexpr auto id = 2; ev2(const runtime_ev&) {} };

struct sm {
  auto operator()() const {
    using namespace sml;
    return make_transition_table(
      *"idle"_s + event<ev1> = "idle"_s,
       "idle"_s + event<ev2> = "idle"_s
    );
  }
};

int main() {
  sml::sm<sm> sm;

  // build once, use repeatedly.
  auto dispatch_runtime = sml::utility::make_dispatch_table<runtime_ev, 1, 2>(sm);

  runtime_ev e{.id = 1};
  if (e.id < 1 || e.id > 2) return 1;  // MUST validate before indexing.
  dispatch_runtime(e, e.id);
}
```

## 16. `sml::utility::sm_pool` rules
`sm_pool` is the preferred utility when one SML router machine must drive a large indexed pool of logical actors/items with shared storage.

1. `sm_pool` storage type MUST be either:
   - default-constructible, or
   - constructible from `std::size_t` (pool size).
2. storage SHOULD provide `reset()` if `sm_pool::reset()` is used.
3. indexed dispatch MUST use `sml::utility::indexed_event<TEvent>` and route by `ev.id`.
4. router actions/guards SHOULD access storage by `ev.id` only and MUST bounds-check indirectly via validated IDs at the call site.
5. for contiguous ID/event buffers, callers SHOULD use pointer/count overloads to minimize iterator overhead:
   - `process_indexed_batch<TEvent>(ids_ptr, count, event)`
   - `process_event_batch(events_ptr, count)`
6. iterator/range overloads MUST preserve deterministic input order; batch APIs are still synchronous RTC dispatch.
7. batch APIs MUST NOT allocate during dispatch. all memory for IDs/events/storage MUST be prepared before the call.
8. `sm_pool` usage MUST stay generic (event-driven state logic), not tailored to a single domain workflow.

### example D: indexed pooled dispatch
```cpp
#include <boost/sml.hpp>
#include <boost/sml/utility/sm_pool.hpp>
namespace sml = boost::sml;

struct ev_tick {};
using pooled_tick = sml::utility::indexed_event<ev_tick>;

struct storage {
  explicit storage(std::size_t n) : flags(n) {}
  void reset() { for (auto& f : flags) f = 0; }
  std::vector<unsigned char> flags;
};

struct router {
  auto operator()() const {
    using namespace sml;
    const auto hot = "hot"_s;
    const auto toggle = [](storage& s, const pooled_tick& ev) { s.flags[ev.id] ^= 1; };
    return make_transition_table(*hot + event<pooled_tick> / toggle);
  }
};

using pool_t = sml::utility::sm_pool<storage, router>;
```
this uses a static dispatch table indexed by event ID; the ID range must be validated by the caller.
