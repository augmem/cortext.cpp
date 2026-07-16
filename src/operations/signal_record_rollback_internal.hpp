#pragma once

#include "cortext/processor/processor_context.hpp"
#include "cortext/processor/operation.hpp"

#include <memory>

namespace cortext::operations::signal_record_rollback_internal
{

class JournalAwareOperation
{
public:
  virtual ~JournalAwareOperation () = default;
};

class JournalAwareOperationAdapter final : public IOperation,
                                           public JournalAwareOperation
{
public:
  explicit JournalAwareOperationAdapter (std::unique_ptr<IOperation> operation)
      : operation_ (std::move (operation))
  {
  }

  void
  Execute (OperationContext &context, Transaction &tx) const override
  {
    operation_->Execute (context, tx);
  }

private:
  std::unique_ptr<IOperation> operation_;
};

inline std::unique_ptr<IOperation>
MarkJournalAware (std::unique_ptr<IOperation> operation)
{
  return std::make_unique<JournalAwareOperationAdapter> (
      std::move (operation));
}

// Internal operations must call EnsureBackedUp before destructively changing
// the active source's accumulator records or working-memory record/blob-id
// vectors. Append-only active-source changes are journaled by retained
// ownership and do not require a backup. A path that mutates foreign
// accumulator topology must instead call EnsureAllBackedUp explicitly.

class Scope
{
public:
  using BackupFn = void (*) (void *owner);
  using PreserveSlotFn = void (*) (void *owner, std::size_t slot_index);

  Scope (ProcessorContext &context, void *owner, BackupFn backup,
         BackupFn backup_all, PreserveSlotFn preserve_slot)
      : context_ (&context), owner_ (owner), backup_ (backup),
        backup_all_ (backup_all), preserve_slot_ (preserve_slot),
        previous_ (active_)
  {
    active_ = this;
  }

  ~Scope ()
  {
    active_ = previous_;
  }

  Scope (const Scope &) = delete;
  Scope &operator= (const Scope &) = delete;

  static void
  EnsureBackedUp (ProcessorContext &context)
  {
    for (auto *scope = active_; scope; scope = scope->previous_)
      {
        if (scope->context_ == &context)
          {
            scope->backup_ (scope->owner_);
            return;
          }
      }
  }

  static void
  EnsureAllBackedUp (ProcessorContext &context)
  {
    for (auto *scope = active_; scope; scope = scope->previous_)
      {
        if (scope->context_ == &context)
          {
            scope->backup_all_ (scope->owner_);
            return;
          }
      }
  }

  static void
  PreserveWorkingMemorySlotBeforeErase (ProcessorContext &context,
                                        std::size_t slot_index)
  {
    for (auto *scope = active_; scope; scope = scope->previous_)
      {
        if (scope->context_ == &context)
          {
            scope->preserve_slot_ (scope->owner_, slot_index);
            return;
          }
      }
  }

private:
  ProcessorContext *context_ = nullptr;
  void *owner_ = nullptr;
  BackupFn backup_ = nullptr;
  BackupFn backup_all_ = nullptr;
  PreserveSlotFn preserve_slot_ = nullptr;
  Scope *previous_ = nullptr;
  static inline thread_local Scope *active_ = nullptr;
};

inline void
EnsureBackedUp (ProcessorContext &context)
{
  Scope::EnsureBackedUp (context);
}

inline void
EnsureAllBackedUp (ProcessorContext &context)
{
  Scope::EnsureAllBackedUp (context);
}

inline void
PreserveWorkingMemorySlotBeforeErase (ProcessorContext &context,
                                      std::size_t slot_index)
{
  Scope::PreserveWorkingMemorySlotBeforeErase (context, slot_index);
}

} // namespace cortext::operations::signal_record_rollback_internal
