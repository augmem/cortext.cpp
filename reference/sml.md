All code examples include `boost/sml.hpp` as well as declare a convienent `sml` namespace alias.

```cpp
#include <boost/sml.hpp>
namespace sml = boost::sml;
```

\###0. Read Boost.MSM - eUML documentation

* [Boost.MSM - UML Short Guide](http://www.boost.org/doc/libs/1_60_0/libs/msm/doc/HTML/ch02.html)
* [Boost.MSM - eUML Documentation](http://www.boost.org/doc/libs/1_60_0/libs/msm/doc/HTML/ch03s04.html)

\###1. Create events and states

State machine is composed of finite number of states and transitions which are triggered via events.

An Event is just a unique type, which will be processed by the state machine.

```cpp
struct my_event { ... };
```

You can also create event instance in order to simplify transition table notation.

```cpp
auto event = sml::event<my_event>;
```

If you happen to have a Clang/GCC compiler, you can create an Event on the fly.

```cpp
using namespace sml;
auto event  = "event"_e;
```

However, such event will not store any data.

***

A State can have entry/exit behaviour executed whenever machine enters/leaves State and
represents current location of the state machine flow.

To create a state below snippet might be used.

```cpp
auto idle = sml::state<class idle>;
```

If you happen to have a Clang/GCC compiler, you can create a State on the fly.

```cpp
using namespace sml;
auto state  = "idle"_s;
```

However, please notice that above solution is a non-standard extension for Clang/GCC.

`SML` states cannot have data as data is injected directly into guards/actions instead.

A state machine might be a State itself.

```cpp
sml::state<state_machine> composite;
```

`SML` supports `terminate` state, which stops events to be processed. It defined by `X`.

```cpp
"state"_s = X;
```

States are printable too.

```cpp
assert(string("idle") == "idle"_s.c_str());
```

![CPP(BTN)](Run_Events_Example|https://raw.githubusercontent.com/boost-ext/sml/master/example/events.cpp)
![CPP(BTN)](Run_States_Example|https://raw.githubusercontent.com/boost-ext/sml/master/example/states.cpp)
![CPP(BTN)](Run_Composite_Example|https://raw.githubusercontent.com/boost-ext/sml/master/example/composite.cpp)

 

***

\###2. Create guards and actions

Guards and actions are callable objects which will be executed by the state machine in order to verify whether a transition, followed by an action should take place.

Guard MUST return boolean value.

```cpp
auto guard1 = [] {
  return true;
};

auto guard2 = [](int, double) { // guard with dependencies
  return true;
};

auto guard3 = [](int, auto event, double) { // guard with an event and dependencies
  return true;
};

struct guard4 {
    bool operator()() const noexcept {
        return true;
    }
};
```

Action MUST not return.

```cpp
auto action1 = [] { };
auto action2 = [](int, double) { }; // action with dependencies
auto action3 = [](int, auto event, double) { }; // action with an event and dependencies
struct action4 {
    void operator()() noexcept { }
};
```

![CPP(BTN)](Run_Actions_Guards_Example|https://raw.githubusercontent.com/boost-ext/sml/master/example/actions_guards.cpp)

 

***

\###3. Create a transition table

When we have states and events handy we can finally create a transition table which represents
our transitions.

`SML` is using eUML-like DSL in order to be as close as possible to UML design.

* Transition Table DSL

  * Postfix Notation

  | Expression                                                                                                 | Description                                                               |
  | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
  | state + event<e> \[ guard ]                                                                                | internal transition on event e when guard                                 |
  | src\_state / \[] {} = dst\_state                                                                           | anonymous transition with action                                          |
  | src\_state / \[] {} = src\_state                                                                           | self transition (calls on\_exit/on\_entry)                                |
  | src\_state + event<e> = dst\_state                                                                         | external transition on event e without guard or action                    |
  | src\_state + event<e> \[ guard ] / action = dst\_state                                                     | transition from src\_state to dst\_state on event e with guard and action |
  | src\_state + event<e> \[ guard && (!\[]{return true;} && guard2) ] / (action, action2, \[]{}) = dst\_state | transition from src\_state to dst\_state on event e with guard and action |

  * Prefix Notation

  | Expression                                                                                                  | Description                                                               |
  | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
  | state + event<e> \[ guard ]                                                                                 | internal transition on event e when guard                                 |
  | dst\_state <= src\_state / \[] {}                                                                           | anonymous transition with action                                          |
  | src\_state <= src\_state / \[] {}                                                                           | self transition (calls on\_exit/on\_entry)                                |
  | dst\_state <= src\_state + event<e>                                                                         | external transition on event e without guard or action                    |
  | dst\_state <= src\_state + event<e> \[ guard ] / action                                                     | transition from src\_state to dst\_state on event e with guard and action |
  | dst\_state <= src\_state + event<e> \[ guard && (!\[]{return true;} && guard2) ] / (action, action2, \[]{}) | transition from src\_state to dst\_state on event e with guard and action |

* Transition flow

```
src_state + event [ guard ] / action = dst_state
                                     ^
                                     |
                                     |
                                    1. src_state + on_exit
                                    2. dst_state + on_entry
```

To create a transition table [`make_transition_table`](user_guide.md#make_transition_table-state-machine) is provided.

```cpp
using namespace sml; // Postfix Notation

make_transition_table(
 *"src_state"_s + event<my_event> [ guard ] / action = "dst_state"_s
, "dst_state"_s + "other_event"_e = X
);
```

or

```cpp
using namespace sml; // Prefix Notation

make_transition_table(
  "dst_state"_s <= *"src_state"_s + event<my_event> [ guard ] / action
, X             <= "dst_state"_s  + "other_event"_e
);
```

![CPP(BTN)](Run_Transition_Table_Example|https://raw.githubusercontent.com/boost-ext/sml/master/example/transitions.cpp)
![CPP(BTN)](Run_eUML_Emulation_Example|https://raw.githubusercontent.com/boost-ext/sml/master/example/euml_emulation.cpp)

 

***

\###4. Set initial states

Initial state tells the state machine where to start. It can be set by prefixing a State with `*`.

```cpp
using namespace sml;
make_transition_table(
 *"src_state"_s + event<my_event> [ guard ] / action = "dst_state"_s,
  "dst_state"_s + event<game_over> = X
);
```

Initial/Current state might be remembered by the State Machine so that whenever it will reentered
the last active state will reactivated. In order to enable history you just have
to replace `*` with postfixed `(H)` when declaring the initial state.

```cpp
using namespace sml;
make_transition_table(
  "src_state"_s(H) + event<my_event> [ guard ] / action = "dst_state"_s,
  "dst_state"_s    + event<game_over>                   = X
);
```

You can have more than one initial state. All initial states will be executed in pseudo-parallel way
. Such states are called `Orthogonal regions`.

```cpp
using namespace sml;
make_transition_table(
 *"region_1"_s   + event<my_event1> [ guard ] / action = "dst_state1"_s,
  "dst_state1"_s + event<game_over> = X,

 *"region_2"_s   + event<my_event2> [ guard ] / action = "dst_state2"_s,
  "dst_state2"_s + event<game_over> = X
);
```

![CPP(BTN)](Run_Orthogonal_Regions_Example|https://raw.githubusercontent.com/boost-ext/sml/master/example/orthogonal_regions.cpp)
![CPP(BTN)](Run_History_Example|https://raw.githubusercontent.com/boost-ext/sml/master/example/history.cpp)

 

***

\###5. Create a state machine

State machine is an abstraction for transition table holding current states and processing events.
To create a state machine, we have to add a transition table.

```cpp
class example {
public:
  auto operator()() {
    using namespace sml;
    return make_transition_table(
     *"src_state"_s + event<my_event> [ guard ] / action = "dst_state"_s,
      "dst_state"_s + event<game_over> = X
    );
  }
};
```

Having transition table configured we can create a state machine.

```cpp
sml::sm<example> sm;
```

State machine constructor provides required dependencies for actions and guards.

```cpp
                            /---- event (injected from process_event)
                            |
auto guard = [](double d, auto event) { return true; }
                   |
                   \--------\
                            |
auto action = [](int i){}   |
                  |         |
                  |         |
                  \-\   /---/
                    |   |
sml::sm<example> s{42, 87.0};

sml::sm<example> s{87.0, 42}; // order in which parameters have to passed is not specificied
```

Passing and maintaining a lot of dependencies might be tedious and requires huge amount of boilerplate code.
In order to avoid it, Dependency Injection Library might be used to automate this process.
For example, we can use [ext Boost.DI](https://github.com/boost-ext/di).

```cpp
auto injector = di::make_injector(
    di::bind<>.to(42)
  , di::bind<interface>.to<implementation>()
);

auto sm = injector.create<sm<example>>();
sm.process_event(e1{});
```

![CPP(BTN)](Run_Hello_World_Example|https://raw.githubusercontent.com/boost-ext/sml/master/example/hello_world.cpp)
![CPP(BTN)](Run_Dependency_Injection_Example|https://raw.githubusercontent.com/boost-ext/sml/master/example/dependency_injection.cpp)

 

***

\###6. Process events

State machine is a simple creature. Its main purpose is to process events.
In order to do it, `process_event` method might be used.

```cpp
sml::sm<example> sm;

sm.process_event(my_event{}); // handled
sm.process_event(int{}); // not handled -> unexpected_event<int>
```

Process event might be also triggered on transition table.

```
using namespace sml;
return make_transition_table(
 *"s1"_s + event<my_event> / process(other_event{}) = "s2"_s,
  "s2"_s + event<other_event> = X
);
```

`SML` also provides a way to dispatch dynamically created events into the state machine.

```cpp
struct game_over {
  static constexpr auto id = SDL_QUIT;
  // explicit game_over(const SDL_Event&) noexcept; // optional, when defined runtime event will be passed
};
enum SDL_EventType { SDL_FIRSTEVENT = 0, SDL_QUIT, SDL_KEYUP, SDL_MOUSEBUTTONUP, SDL_LASTEVENT };
//create dispatcher from state machine and range of events
auto dispatch_event = sml::utility::make_dispatch_table<SDL_Event, SDL_FIRSTEVENT, SDL_LASTEVENT>(sm);
SDL_Event event{SDL_QUIT};
dispatch_event(event, event.type); // will call sm.process(game_over{});
```

![CPP(BTN)](Run_Hello_World_Example|https://raw.githubusercontent.com/boost-ext/sml/master/example/hello_world.cpp)
![CPP(BTN)](Run_Dispatch_Table_Example|https://raw.githubusercontent.com/boost-ext/sml/master/example/dispatch_table.cpp)
![CPP(BTN)](Run_SDL2_Integration_Example|https://raw.githubusercontent.com/boost-ext/sml/master/example/sdl2.cpp)

 

***

\###8. Handle errors

In case when a State Machine can't handle given event an `unexpected_event` is fired.

```cpp
make_transition_table(
 *"src_state"_s + event<my_event> [ guard ] / action = "dst_state"_s
, "src_state"_s + unexpected_event<some_event> = X
);
```

Any unexpected event might be handled too by using `unexpected_event<_>`.

```cpp
make_transition_table(
 *"src_state"_s + event<my_event> [ guard ] / action = "dst_state"_s
, "src_state"_s + unexpected_event<some_event> / [] { std::cout << "unexpected 'some_event' << '\n'; "}
, "src_state"_s + unexpected_event<_> = X // any event
);
```

In such case...

```cpp
sm.process_event(some_event{}); // "unexpected 'some_event'
sm.process_event(int{}); // terminate
assert(sm.is(X));
```

Usually, it's handy to create additional `Orthogonal region` to cover this scenario,
This way State causing unexpected event does not matter.

```cpp
make_transition_table(
 *"idle"_s + event<my_event> [ guard ] / action = "s1"_s
, "s1"_s + event<other_event> [ guard ] / action = "s2"_s
, "s2"_s + event<yet_another_event> [ guard ] / action = X
// terminate (=X) the Machine or reset to another state
,*"error_handler"_s + unexpected_event<some_event> = X
);
```

We can always check whether a State Machine is in terminate state by.

```cpp
assert(sm.is(sml::X)); // doesn't matter how many regions there are
```

When exceptions are enabled (project is NOT compiled with `-fno-exceptions`) they
can be caught using `exception<name>` syntax. Exception handlers will be processed
in the order they were defined, and `exception<>` might be used to catch anything (equivalent to `catch (...)`).
Please, notice that when there is no exception handler defined in the Transition Table, exception will not be handled by the State Machine.

```cpp
make_transition_table(
 *"idle"_s + event<event> / [] { throw std::runtime_error{"error"}; }
,*"error_handler"_s + exception<std::runtime_error> = X
, "error_handler"_s + exception<std::logic_error> = X
, "error_handler"_s + exception<> / [] { cleanup...; } = X // any exception
);
```

![CPP(BTN)](Run_Error_Handling_Example|https://raw.githubusercontent.com/boost-ext/sml/master/example/error_handling.cpp)

 

***

\###9. Test it

Sometimes it's useful to verify whether a state machine is in a specific state, for example, if
we are in a terminate state or not. We can do it with `SML` using `is` or `visit_current_states`
functionality.

```cpp
sml::sm<example> sm;
sm.process_event(my_event{});
assert(sm.is(X)); // is(X, s1, ...) when you have orthogonal regions

//or

sm.visit_current_states([](auto state) { std::cout << state.c_str() << std::endl; });
```

On top of that, `SML` provides testing facilities to check state machine as a whole.
`set_current_states` method is available from `testing::sm` in order to set state machine
in a requested state.

```cpp
sml::sm<example, sml::testing> sm{fake_data...};
sm.set_current_states("s3"_s); // set_current_states("s3"_s, "s1"_s, ...) for orthogonal regions
sm.process_event(event{});
assert(sm.is(X));
```

![CPP(BTN)](Run_Testing_Example|https://raw.githubusercontent.com/boost-ext/sml/master/example/testing.cpp)

 

***

\###10. Debug it

`SML` provides logging capabilities in order to inspect state machine flow.
To enable logging you can use (Logger Policy)(user\_guide.md#policies)

````cpp
struct my_logger {
  template <class SM, class TEvent>
  void log_process_event(const TEvent&) {
    printf("[%s][process_event] %s\n", sml::aux::get_type_name<SM>(), sml::aux::get_type_name<TEvent>());
  }

  template <class SM, class TGuard, class TEvent>
  void log_guard(const TGuard&, const TEvent&, bool result) {
    printf("[%s][guard] %s %s %s\n", sml::aux::get_type_name<SM>(), sml::aux::get_type_name<TGuard>(),
           sml::aux::get_type_name<TEvent>(), (result ? "[OK]" : "[Reject]"));
  }

  template <class SM, class TAction, class TEvent>
  void log_action(const TAction&, const TEvent&) {
    printf("[%s][action] %s %s\n", sml::aux::get_type_name<SM>(), sml::aux::get_type_name<TAction>(),
           sml::aux::get_type_name<TEvent>());
  }

  template <class SM, class TSrcState, class TDstState>
  void log_state_change(const TSrcState& src, const TDstState& dst) {
    printf("[%s][transition] %s -> %s\n", sml::aux::get_type_name<SM>(), src.c_str(), dst.c_str());
  }
};

my_logger logger;
sml::sm<logging, sml::logger<my_logger>> sm{logger};
sm.process_event(my_event{}); // will call logger appropriately
``` 

***

<iframe style="width: 100%; height: 600px;" src="https://boost-ext.github.io/sml/embo-2018" />
\###transitional \[concept]

***Header***

````

\#include \<boost/sml.hpp>

```

***Description***

Requirements for transition.

***Synopsis***

```

template <class T>
concept bool transitional() {
return requires(T transition) {
typename T::src\_state;
typename T::dst\_state;
typename T::event;
typename T::deps;
T::initial;
T::history;
{ transition.execute() } -> bool;
}
};

```

***Semantics***

```

transitional<T>

```

***Example***

```

using namespace sml;

{
auto transition = ("idle"\_s = X); // Postfix Notation
static\_assert(transitional\<decltype(transition)>::value);
}

{
auto transition = (X <= "idle"\_s); // Prefix Notation
static\_assert(transitional\<decltype(transition)>::value);
}

````

```cpp
#include <boost/sml.hpp>

using namespace sml;

{
auto transition = ("idle"_s = X); // Postfix Notation
static_assert(transitional<decltype(transition)>::value);
}

{
auto transition = (X <= "idle"_s); // Prefix Notation
static_assert(transitional<decltype(transition)>::value);
}
````

 

***

\###configurable \[concept]

***Header***

```
#include <boost/sml.hpp>
```

***Description***

Requirements for the state machine.

***Synopsis***

```
template <class SM>
concept bool configurable() {
  return requires(SM sm) {
    { sm.operator()() };
  }
};
```

***Semantics***

```
configurable<SM>
```

***Example***

```
class example {
  auto operator()() const noexcept {
    return make_transition_table();
  }
};

static_assert(configurable<example>::value);
```

```cpp
#include <boost/sml.hpp>

class example {
  auto operator()() const noexcept {
    return make_transition_table();
  }
};

static_assert(configurable<example>::value);
```

 

***

\###callable \[concept]

***Header***

```
#include <boost/sml.hpp>
```

***Description***

Requirements for action and guards.

***Synopsis***

```
template <class TResult, class T>
concept bool callable() {
  return requires(T object) {
    { object(...) } -> TResult;
  }
}
```

***Semantics***

```
callable<SM>
```

***Example***

```
auto guard = [] { return true; };
auto action = [] { };

static_assert(callable<bool, decltype(guard)>::value);
static_assert(callable<void, decltype(action)>::value);
```

```cpp
#include <boost/sml.hpp>

auto guard = [] { return true; };
auto action = [] { };

static_assert(callable<bool, decltype(guard)>::value);
static_assert(callable<void, decltype(action)>::value);
```

 

***

\###dispatchable \[concept]

***Header***

```
#include <boost/sml.hpp>
```

***Description***

Requirements for the dispatch table.

***Synopsis***

```
template <class TDynamicEvent, TEvent>
concept bool dispatchable() {
  return requires(T) {
    typename TEvent::id;
    { TEvent(declval<TDynamicEvent>()) };
  }
};
```

***Semantics***

```
dispatchable<SM>
```

***Example***

```
struct runtime_event { };

struct event1 {
  static constexpr auto id = 1;
};

struct event2 {
  static constexpr auto id = 2;
  explicit event2(const runtime_event&) {}
};

static_assert(dispatchable<runtime_event, event1>::value);
static_assert(dispatchable<runtime_event, event2>::value);
```

```cpp
#include <boost/sml.hpp>

struct runtime_event { };

struct event1 {
  static constexpr auto id = 1;
};

struct event2 {
  static constexpr auto id = 2;
  explicit event2(const runtime_event&) {}
};

static_assert(dispatchable<runtime_event, event1>::value);
static_assert(dispatchable<runtime_event, event2>::value);
```

```cpp
//
// Copyright (c) 2016-2020 Kris Jusiak (kris at jusiak dot net)
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#include <boost/sml.hpp>
#include <cassert>
#include <iostream>

#include "boost/sml/utility/dispatch_table.hpp"

namespace sml = boost::sml;

// clang-format off
#if __has_include(<SDL2/SDL_events.h>)
#include <SDL2/SDL_events.h>
// clang-format on

namespace {

#else

namespace {

enum { SDLK_SPACE = ' ' };
enum SDL_EventType { SDL_FIRSTEVENT = 0, SDL_QUIT, SDL_KEYUP, SDL_MOUSEBUTTONUP, SDL_LASTEVENT };
struct SDL_KeyboardEvent {
  SDL_EventType type;
  struct {
    int sym;
  } keysym;
};
struct SDL_MouseButtonEvent {
  SDL_EventType type;
  int button;
};
struct SDL_QuitEvent {
  SDL_EventType type;
};
union SDL_Event {
  SDL_EventType type;
  SDL_KeyboardEvent key;
  SDL_MouseButtonEvent button;
  SDL_QuitEvent quit;
};
#endif

template <SDL_EventType Id>
struct sdl_event_impl {
  static constexpr auto id = Id;
  explicit sdl_event_impl(const SDL_Event& data) noexcept : data(data) {}
  SDL_Event data;
};

template <SDL_EventType Id>
decltype(sml::event<sdl_event_impl<Id>>) sdl_event{};

struct IsKey {
  auto operator()(int key) {
    return [=](auto event) { return event.data.key.keysym.sym == key; };
  }
} is_key;

struct sdl2 {
  auto operator()() const noexcept {
    using namespace sml;
    // clang-format off
    return make_transition_table(
      //------------------------------------------------------------------------------//
        "wait_for_user_input"_s <= *"idle"_s
          / [] { std::cout << "initialization" << std::endl; }

      , "key_pressed"_s <= "wait_for_user_input"_s + sdl_event<SDL_KEYUP> [ is_key(SDLK_SPACE) ]
          / [] { std::cout << "space pressed" << std::endl; }

      , X <= "key_pressed"_s + sdl_event<SDL_MOUSEBUTTONUP>
          / [] { std::cout << "mouse button pressed" << std::endl; }
      //------------------------------------------------------------------------------//
      , X <= *"waiting_for_quit"_s + sdl_event<SDL_QUIT>
          / [] { std::cout << "quit" << std::endl; }
      //------------------------------------------------------------------------------//
    );
    // clang-format on
  }
};
}

int main() {
  sml::sm<sdl2> sm;
  auto dispatch_event = sml::utility::make_dispatch_table<SDL_Event, SDL_FIRSTEVENT, SDL_LASTEVENT>(sm);

  SDL_Event event;

  // while (SDL_PollEvent(&event)) {
  //   dispatch_event(event, event.type)
  // };

  {
    SDL_KeyboardEvent keyboard_event;
    keyboard_event.type = SDL_KEYUP;
    keyboard_event.keysym.sym = SDLK_SPACE;
    event.key = keyboard_event;
    dispatch_event(event, event.type);
  }

  {
    SDL_MouseButtonEvent mousebutton_event;
    mousebutton_event.type = SDL_MOUSEBUTTONUP;
    mousebutton_event.button = 1;
    event.button = mousebutton_event;
    dispatch_event(event, event.type);
  }

  {
    SDL_QuitEvent quit_event;
    quit_event.type = SDL_QUIT;
    event.quit = quit_event;
    dispatch_event(event, event.type);
  }

  assert(sm.is(sml::X, sml::X));
}
```

 

***

\###state \[core]

***Header***

```
#include <boost/sml.hpp>
```

***Description***

Represents a state machine state.

***Synopsis***

```
template<class TState> // no requirements, TState may be a state machine
class state {
public:
  initial operator*() const noexcept; // no requirements

  template <class T> // no requirements
  auto operator<=(const T &) const noexcept;

  template <class T> // no requirements
  auto operator=(const T &) const noexcept;

  template <class T> // no requirements
  auto operator+(const T &) const noexcept;

  template <class T> requires callable<bool, T>
  auto operator[](const T) const noexcept;

  template <class T> requires callable<void, T>
  auto operator/(const T &t) const noexcept;

  const char* c_str() noexcept;
};

template <class T, T... Chrs>
state<unspecified> operator""_s() noexcept;

// predefined states
state<unspecified> X;
```

***Requirements***

* [callable](#callable-concept)

***Semantics***

```
state<T>{}
```

***Example***

```
auto idle = state<class idle>;
auto idle = "idle"_s;

auto initial_state = *idle;
auto history_state = idle(H);
auto terminate_state = X;
```

```cpp
//
// Copyright (c) 2016-2020 Kris Jusiak (kris at jusiak dot net)
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#include <boost/sml.hpp>
#include <cassert>
#include <iostream>

namespace sml = boost::sml;

namespace {
struct e1 {};
struct e2 {};
struct e3 {};

struct states {
  auto operator()() const noexcept {
    using namespace sml;
    const auto idle = state<class idle>;
    // clang-format off
    return make_transition_table(
       *idle + event<e1> = "s1"_s
      , "s1"_s + sml::on_entry<_> / [] { std::cout << "s1 on entry" << std::endl; }
      , "s1"_s + sml::on_exit<_> / [] { std::cout << "s1 on exit" << std::endl; }
      , "s1"_s + event<e2> = state<class s2>
      , state<class s2> + event<e3> = X
    );
    // clang-format on
  }
};
}  // namespace

int main() {
  sml::sm<states> sm;
  sm.process_event(e1{});
  sm.process_event(e2{});
  sm.process_event(e3{});
  assert(sm.is(sml::X));
}
```

```cpp
//
// Copyright (c) 2016-2020 Kris Jusiak (kris at jusiak dot net)
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#include <boost/sml.hpp>
#include <cassert>
#include <iostream>

namespace sml = boost::sml;

namespace {
struct e1 {};
struct e2 {};
struct e3 {};
struct e4 {};
struct e5 {};

struct sub {
  auto operator()() const noexcept {
    using namespace sml;
    // clang-format off
      return make_transition_table(
       *"idle"_s + event<e3> / [] { std::cout << "in sub sm" << std::endl; } = "s1"_s
      , "s1"_s + event<e4> / [] { std::cout << "finish sub sm" << std::endl; } = X
      );
    // clang-format on
  }
};

struct composite {
  auto operator()() const noexcept {
    using namespace sml;
    // clang-format off
    return make_transition_table(
     *"idle"_s + event<e1> = "s1"_s
    , "s1"_s + event<e2> / [] { std::cout << "enter sub sm" << std::endl; } = state<sub>
    , state<sub> + event<e5> / [] { std::cout << "exit sub sm" << std::endl; } = X
    );
    // clang-format on
  }
};
}  // namespace

int main() {
  sml::sm<composite> sm;

  using namespace sml;
  assert(sm.is("idle"_s));
  assert(sm.is<decltype(state<sub>)>("idle"_s));

  sm.process_event(e1{});
  assert(sm.is("s1"_s));
  assert(sm.is<decltype(state<sub>)>("idle"_s));

  sm.process_event(e2{});  // enter sub sm
  assert(sm.is(state<sub>));
  assert(sm.is<decltype(state<sub>)>("idle"_s));

  sm.process_event(e3{});  // in sub sm
  assert(sm.is(state<sub>));
  assert(sm.is<decltype(state<sub>)>("s1"_s));

  sm.process_event(e4{});  // finish sub sm
  assert(sm.is(state<sub>));
  assert(sm.is<decltype(state<sub>)>(X));

  sm.process_event(e5{});  // exit sub sm
  assert(sm.is(X));
  assert(sm.is<decltype(state<sub>)>(X));
}
```

```cpp
//
// Copyright (c) 2016-2020 Kris Jusiak (kris at jusiak dot net)
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#include <boost/sml.hpp>
#include <cassert>

namespace sml = boost::sml;

namespace {
struct e1 {};
struct e2 {};
struct e3 {};

struct orthogonal_regions {
  auto operator()() const noexcept {
    using namespace sml;
    // clang-format off
    return make_transition_table(
     *"idle"_s + event<e1> = "s1"_s
    , "s1"_s + event<e2> = X

    ,*"idle2"_s + event<e2> = "s2"_s
    , "s2"_s + event<e3> = X
    );
    // clang-format on
  }
};
}  // namespace

int main() {
  sml::sm<orthogonal_regions> sm;
  using namespace sml;
  assert(sm.is("idle"_s, "idle2"_s));
  sm.process_event(e1{});
  assert(sm.is("s1"_s, "idle2"_s));
  sm.process_event(e2{});
  assert(sm.is(X, "s2"_s));
  sm.process_event(e3{});
  assert(sm.is(X, X));
}
```

 

***

\###event \[core]

***Header***

```
#include <boost/sml.hpp>
```

***Description***

Represents a state machine event.

***Synopsis***

```
template<TEvent> // no requirements
class event {
public:
  template <class T> requires callable<bool, T>
  auto operator[](const T &) const noexcept;

  template <class T> requires callable<void, T>
  auto operator/(const T &t) const noexcept;
};

template<class TEvent>
event<TEvent> event{};

// predefined events
auto on_entry = event<unspecified>;
auto on_exit = event<unspecified>;

template<class TEvent> unexpected_event{};
template<class T> exception{};
```

***Requirements***

* [callable](#callable-concept)

***Semantics***

```
event<T>
```

***Example***

```
auto my_int_event = event<int>;
```

```cpp
// Note: action_guards.cpp file not found, this is a placeholder for the events example
```

```cpp
//
// Copyright (c) 2016-2020 Kris Jusiak (kris at jusiak dot net)
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#include <boost/sml.hpp>
#include <cassert>
#include <iostream>
#include <stdexcept>

namespace sml = boost::sml;

namespace {
struct some_event {};

struct error_handling {
  auto operator()() const {
    using namespace sml;
    // clang-format off
    return make_transition_table(
        *("idle"_s) + "event1"_e / [] { throw std::runtime_error{"error"}; }
      ,   "idle"_s  + "event2"_e / [] { throw 0; }

      , *("exceptions handling"_s) + exception<std::runtime_error> / [] { std::cout << "exception caught" << std::endl; }
      ,   "exceptions handling"_s  + exception<_> / [] { std::cout << "generic exception caught" << std::endl; } = X

      , *("unexpected events handling"_s) + unexpected_event<some_event> / [] { std::cout << "unexpected event 'some_event'" << std::endl; }
      ,   "unexpected events handling"_s  + unexpected_event<_> / [] { std::cout << "generic unexpected event" << std::endl; } = X
    );
    // clang-format on
  }
};
}  // namespace

int main() {
  using namespace sml;
  sm<error_handling> sm;

  sm.process_event("event1"_e());  // throws runtime_error
  assert(sm.is("idle"_s, "exceptions handling"_s, "unexpected events handling"_s));

  sm.process_event("event2"_e());  // throws 0
  assert(sm.is("idle"_s, X, "unexpected events handling"_s));

  sm.process_event(some_event{});  // unexpected event
  assert(sm.is("idle"_s, X, "unexpected events handling"_s));

  sm.process_event(int{});  // unexpected any event
  assert(sm.is("idle"_s, X, X));
}
```

 

***

\###make\_transition\_table \[state machine]

***Header***

```
#include <boost/sml.hpp>
```

***Description***

Creates a transition table.

***Synopsis***

```
template <class... Ts> requires transitional<Ts>...
auto make_transition_table(Ts...) noexcept;
```

***Requirements***

* [transitional](#transitional-concept)

***Semantics***

```
make_transition_table(transitions...);
```

***Example***

```
auto transition_table_postfix_notation = make_transition_table(
  *"idle_s" + event<int> / [] {} = X
);

auto transition_table_prefix_notation = make_transition_table(
  X <= *"idle_s" + event<int> / [] {}
);

class example {
public:
  auto operator()() const noexcept {
    return make_transition_table();
  }
};
```

```cpp
//
// Copyright (c) 2016-2020 Kris Jusiak (kris at jusiak dot net)
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#include <boost/sml.hpp>
#include <cassert>
#include <iostream>

namespace sml = boost::sml;

namespace {
struct e1 {};
struct e2 {};
struct e3 {};

struct transitions {
  auto operator()() const noexcept {
    using namespace sml;
    // clang-format off
    return make_transition_table(
       *"idle"_s                  / [] { std::cout << "anonymous transition" << std::endl; } = "s1"_s
      , "s1"_s + event<e1>        / [] { std::cout << "internal transition" << std::endl; }
      , "s1"_s + event<e2>        / [] { std::cout << "self transition" << std::endl; } = "s1"_s
      , "s1"_s + sml::on_entry<_> / [] { std::cout << "s1 entry" << std::endl; }
      , "s1"_s + sml::on_exit<_>  / [] { std::cout << "s1 exit" << std::endl; }
      , "s1"_s + event<e3>        / [] { std::cout << "external transition" << std::endl; } = X
    );
    // clang-format on
  }
};
}  // namespace

int main() {
  sml::sm<transitions> sm;
  sm.process_event(e1{});
  sm.process_event(e2{});
  sm.process_event(e3{});
  assert(sm.is(sml::X));
}
```

 

***

\###sm \[state machine]

***Header***

```
#include <boost/sml.hpp>
```

***Description***

Creates a State Machine.

***Synopsis***

```
template<class T> requires configurable<T>
class sm {
public:
  using states = unspecified; // unique list of states
  using events = unspecified; // unique list of events which can be handled by the State Machine
  using transitions = unspecified; // list of transitions

  sm(sm &&) = default;
  sm(const sm &) = delete;
  sm &operator=(const sm &) = delete;

  template <class... TDeps> requires is_base_of<TDeps, dependencies>...
  sm(TDeps&&...) noexcept;

  template<class TEvent> // no requirements
  bool process_event(const TEvent&)

  template <class TVisitor> requires callable<void, TVisitor>
  void visit_current_states(const TVisitor &) const noexcept(noexcept(visitor(state{})));

  template <class TState>
  bool is(const state<TState> &) const noexcept;

  template <class... TStates> requires sizeof...(TStates) == number_of_initial_states
  bool is(const state<TStates> &...) const noexcept;
};
```

| Expression                       | Requirement                                        | Description                                          | Returns                                                       |
| -------------------------------- | -------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------- |
| `TDeps...`                       | is\_base\_of dependencies                          | constructor                                          |                                                               |
| `process_event<TEvent>`          | -                                                  | process event `TEvent`                               | returns true when handled, false otherwise                    |
| `visit_current_states<TVisitor>` | [callable](#callable-concept)                      | visit current states                                 | -                                                             |
| `is<TState>`                     | -                                                  | verify whether any of current states equals `TState` | true when any current state matches `TState`, false otherwise |
| `is<TStates...>`                 | size of TStates... equals number of initial states | verify whether all current states match `TStates...` | true when all states match `TState...`, false otherwise       |

***Semantics***

```
sml::sm<T>{...};
sm.process_event(TEvent{});
sm.visit_current_states([](auto state){});
sm.is(X);
sm.is(s1, s2);
```

***Example***

```
struct my_event {};

class example {
public:
  auto operator()() const noexcept {
    using namespace sml;
    return make_transition_table(
      *"idle"_s + event<my_event> / [](int i) { std::cout << i << std::endl; } = X
    );
  }
};

sml::sm<example> sm{42};
assert(sm.is("idle"_s));
sm.process_event(int{}); // no handled, will call unexpected_event<int>
sm.process_event(my_event{}); // handled
assert(sm.is(X));

sm.visit_current_states([](auto state) { std::cout << state.c_str() << std::endl; });
```

```cpp
//
// Copyright (c) 2016-2020 Kris Jusiak (kris at jusiak dot net)
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#include <boost/sml.hpp>
#include <cassert>

namespace sml = boost::sml;

namespace {
// events
struct release {};
struct ack {};
struct fin {};
struct timeout {};

// guards
const auto is_ack_valid = [](const ack&) { return true; };
const auto is_fin_valid = [](const fin&) { return true; };

// actions
const auto send_fin = [] {};
const auto send_ack = [] {};

#if !defined(_MSC_VER)
struct hello_world {
  auto operator()() const {
    using namespace sml;
    // clang-format off
    return make_transition_table(
      *"established"_s + event<release> / send_fin = "fin wait 1"_s,
       "fin wait 1"_s + event<ack> [ is_ack_valid ] = "fin wait 2"_s,
       "fin wait 2"_s + event<fin> [ is_fin_valid ] / send_ack = "timed wait"_s,
       "timed wait"_s + event<timeout> / send_ack = X
    );
    // clang-format on
  }
};
}

int main() {
  using namespace sml;

  sm<hello_world> sm;
  static_assert(1 == sizeof(sm), "sizeof(sm) != 1b");
  assert(sm.is("established"_s));

  sm.process_event(release{});
  assert(sm.is("fin wait 1"_s));

  sm.process_event(ack{});
  assert(sm.is("fin wait 2"_s));

  sm.process_event(fin{});
  assert(sm.is("timed wait"_s));

  sm.process_event(timeout{});
  assert(sm.is(X));  // released
}
#else
class established;
class fin_wait_1;
class fin_wait_2;
class timed_wait;

struct hello_world {
  auto operator()() const {
    using namespace sml;
    // clang-format off
    return make_transition_table(
      *state<established> + event<release> / send_fin = state<fin_wait_1>,
       state<fin_wait_1> + event<ack> [ is_ack_valid ] = state<fin_wait_2>,
       state<fin_wait_2> + event<fin> [ is_fin_valid ] / send_ack = state<timed_wait>,
       state<timed_wait> + event<timeout> / send_ack = X
    );
    // clang-format on
  }
};
}

int main() {
  using namespace sml;

  sm<hello_world> sm;
  assert(sm.is(state<established>));

  sm.process_event(release{});
  assert(sm.is(state<fin_wait_1>));

  sm.process_event(ack{});
  assert(sm.is(state<fin_wait_2>));

  sm.process_event(fin{});
  assert(sm.is(state<timed_wait>));

  sm.process_event(timeout{});
  assert(sm.is(X));  // released
}
#endif
```

```cpp
//
// Copyright (c) 2016-2020 Kris Jusiak (kris at jusiak dot net)
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#if __has_include(<boost/di.hpp>)
// clang-format on
#include <boost/sml.hpp>
#include <boost/di.hpp>
#include <cassert>
#include <typeinfo>
#include <iostream>

namespace sml = boost::sml;
namespace di = boost::di;

namespace {
struct e1 {};
struct e2 {};
struct e3 {};

auto guard = [](int i, double d) {
  assert(42 == i);
  assert(87.0 == d);
  std::cout << "guard" << std::endl;
  return true;
};

auto action = [](int i, auto e) {
  assert(42 == i);
  std::cout << "action: " << typeid(e).name() << std::endl;
};

struct example {
  auto operator()() const noexcept {
    using namespace sml;
    // clang-format off
    return make_transition_table(
       *"idle"_s + event<e1> = "s1"_s
      , "s1"_s + event<e2> [ guard ] / action = "s2"_s
      , "s2"_s + event<e3> / [] { std::cout << "in place action" << std::endl; } = X
    );
    // clang-format on
  }
};

class controller {
 public:
  explicit controller(sml::sm<example>& sm) : sm(sm) {}

  void start() {
    sm.process_event(e1{});
    sm.process_event(e2{});
    sm.process_event(e3{});
    assert(sm.is(sml::X));
  }

 private:
  sml::sm<example>& sm;
};
}  // namespace

int main() {
  // clang-format off
  const auto injector = di::make_injector(
    di::bind<>.to(42)
  , di::bind<>.to(87.0)
  );
  // clang-format off
  injector.create<controller>().start();
}
#else
int main() {}
#endif
```

```cpp
//
// Copyright (c) 2016-2020 Kris Jusiak (kris at jusiak dot net)
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#include <boost/sml.hpp>
#include <cassert>

namespace sml = boost::sml;

namespace {
struct e1 {};
struct e2 {};
struct e3 {};

auto event1 = sml::event<e1>;
auto event2 = sml::event<e2>;
auto event3 = sml::event<e3>;

auto idle = sml::state<class idle>;
auto s1 = sml::state<class s1>;
auto s2 = sml::state<class s2>;

class euml_emulation;

struct Guard {
  template <class TEvent>
  bool operator()(euml_emulation&, const TEvent&) const;
} guard;

struct Action {
  template <class TEvent>
  void operator()(euml_emulation&, const TEvent&);
} action;

class euml_emulation {
 public:
  auto operator()() const {
    using namespace sml;
    // clang-format off
    return make_transition_table(
      s1 <= *idle + event1,
      s2 <= s1    + event2 [ guard ],
      X  <= s2    + event3 [ guard ] / action
    );
    // clang-format on
  }

  template <class TEvent>
  bool call_guard(const TEvent&) {
    return true;
  }

  void call_action(const e3&) {}
};

template <class TEvent>
bool Guard::operator()(euml_emulation& sm, const TEvent& event) const {
  return sm.call_guard(event);
}

template <class TEvent>
void Action::operator()(euml_emulation& sm, const TEvent& event) {
  sm.call_action(event);
}
}  // namespace

int main() {
  euml_emulation euml;
  sml::sm<euml_emulation> sm{euml};
  assert(sm.is(idle));
  sm.process_event(e1{});
  assert(sm.is(s1));
  sm.process_event(e2{});
  assert(sm.is(s2));
  sm.process_event(e3{});
  assert(sm.is(sml::X));
}
```

 

***

\###policies \[state machine]

***Header***

```
#include <boost/sml.hpp>
```

***Description***

Additional State Machine configurations.

***Synopsis***

```
thread_safe<Lockable>
logger<Loggable>
```

| Expression | Requirement                                               | Description   | Example                              |
| ---------- | --------------------------------------------------------- | ------------- | ------------------------------------ |
| `Lockable` | `lock/unlock`                                             | Lockable type | `std::mutex`, `std::recursive_mutex` |
| `Loggable` | `log_process_event/log_state_change/log_action/log_guard` | Loggable type | -                                    |

***Example***

```
sml::sm<example, sml::thread_safe<std::recursive_mutex>> sm; // thread safe policy
sml::sm<example, sml::logger<my_logger>> sm; // logger policy
sml::sm<example, sml::thread_safe<std::recursive_mutex>, sml::logger<my_logger>> sm; // thread safe and logger policy
sml::sm<example, sml::logger<my_logger>, sml::thread_safe<std::recursive_mutex>> sm; // thread safe and logger policy
```

```cpp
//
// Copyright (c) 2016-2020 Kris Jusiak (kris at jusiak dot net)
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#include <boost/sml.hpp>
#include <cassert>
#include <cstdio>
#include <iostream>

namespace sml = boost::sml;

namespace {
struct my_logger {
  template <class SM, class TEvent>
  void log_process_event(const TEvent&) {
    printf("[%s][process_event] %s\n", sml::aux::get_type_name<SM>(), sml::aux::get_type_name<TEvent>());
  }

  template <class SM, class TGuard, class TEvent>
  void log_guard(const TGuard&, const TEvent&, bool result) {
    printf("[%s][guard] %s %s %s\n", sml::aux::get_type_name<SM>(), sml::aux::get_type_name<TGuard>(),
           sml::aux::get_type_name<TEvent>(), (result ? "[OK]" : "[Reject]"));
  }

  template <class SM, class TAction, class TEvent>
  void log_action(const TAction&, const TEvent&) {
    printf("[%s][action] %s %s\n", sml::aux::get_type_name<SM>(), sml::aux::get_type_name<TAction>(),
           sml::aux::get_type_name<TEvent>());
  }

  template <class SM, class TSrcState, class TDstState>
  void log_state_change(const TSrcState& src, const TDstState& dst) {
    printf("[%s][transition] %s -> %s\n", sml::aux::get_type_name<SM>(), src.c_str(), dst.c_str());
  }
};

struct e1 {};
struct e2 {};

struct guard {
  bool operator()() const { return true; }
} guard;

struct action {
  void operator()() {}
} action;

struct logging {
  auto operator()() const noexcept {
    using namespace sml;
    // clang-format off
    return make_transition_table(
       *"idle"_s + event<e1> [ guard && guard ] / action = "s1"_s
    );
    // clang-format on
  }
};
}  // namespace

int main() {
  my_logger logger;
  sml::sm<logging, sml::logger<my_logger>> sm{logger};
  sm.process_event(e1{});
  sm.process_event(e2{});
}
```

 

***

\###testing::sm \[testing]

***Header***

```
#include <boost/sml/testing/state_machine.hpp>
```

***Description***

Creates a state machine with testing capabilities.

***Synopsis***

```
namespace testing {
  template <class T>
  class sm : public sml::sm<T> {
   public:
    using sml::sm<T>::sm;

    template <class... TStates>
    void set_current_states(const detail::state<TStates> &...) noexcept;
  };
}
```

| Expression                       | Requirement | Description        | Returns |
| -------------------------------- | ----------- | ------------------ | ------- |
| `set_current_states<TStates...>` | -           | set current states |         |

***Semantics***

```
sml::testing::sm<T>{...};
sm.set_current_states("s1"_s);
```

***Example***

```
sml::testing::sm<T>{inject_fake_data...};
sm.set_current_states("s1"_s);
sm.process_event(TEvent{});
sm.is(X);
```

```cpp
//
// Copyright (c) 2016-2020 Kris Jusiak (kris at jusiak dot net)
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#include <boost/sml.hpp>
#include <cassert>

namespace sml = boost::sml;

namespace {
struct e1 {};
struct e2 {};
struct e3 {};

struct data {
  int value = 0;
};

struct testing {
  auto operator()() const noexcept {
    using namespace sml;

    const auto guard = [](data& d) { return !d.value; };
    const auto action = [](data& d) { d.value = 42; };

    // clang-format off
    return make_transition_table(
       *"idle"_s + event<e1> = "s1"_s
      , "s1"_s + event<e2> = "s2"_s
      , "s2"_s + event<e3> [guard] / action = X // transition under test
    );
    // clang-format on
  }
};
}  // namespace

int main() {
  using namespace sml;
  data fake_data{0};
  sml::sm<::testing, sml::testing> sm{fake_data};
  sm.set_current_states("s2"_s);
  sm.process_event(e3{});
  assert(sm.is(X));
  assert(fake_data.value == 42);
}
```

 

***

\###make\_dispatch\_table \[utility]

***Header***

```
#include <boost/sml/utility/dispatch_table.hpp>
```

***Description***

Creates a dispatch table to handle runtime events.

***Synopsis***

```
namespace utility {
  template<class TEvent, int EventRangeBegin, int EventRangeBegin, class SM> requires dispatchable<TEvent, typename SM::events>
  callable<bool, (TEvent, int)> make_dispatch_table(sm<SM>&) noexcept;
}
```

***Requirements***

* [dispatchable](#dispatchable-concept)

***Semantics***

```
sml::utility::make_dispatch_table<T, 0, 10>(sm);
```

***Example***

```
struct runtime_event {
  int id = 0;
};
struct event1 {
  static constexpr auto id = 1;
  event1(const runtime_event &) {}
};

auto dispatch_event = sml::utility::make_dispatch_table<runtime_event, 1 /*min*/, 5 /*max*/>(sm);
dispatch_event(event, event.id);
```
