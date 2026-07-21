#pragma once

#include "cortext/processor/processor_context.hpp"
#include "cortext/processor/operation.hpp"

#include <memory>
#include <string>

namespace cortext::operations::signal_record_rollback_internal
{

class JournalAwareOperation
{
public:
  virtual ~JournalAwareOperation () = default;
};

class EngineOwnedOperation
{
public:
  virtual ~EngineOwnedOperation () = default;
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

class EngineOwnedJournalAwareOperationAdapter final
    : public IOperation,
      public JournalAwareOperation,
      public EngineOwnedOperation
{
public:
  explicit EngineOwnedJournalAwareOperationAdapter (
      std::unique_ptr<IOperation> operation)
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
MarkEngineOwnedJournalAware (std::unique_ptr<IOperation> operation)
{
  return std::make_unique<EngineOwnedJournalAwareOperationAdapter> (
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
  using PreserveSparseIndexFn = void (*) (
      void *owner, const std::string &key, long long memory_id);
  using PreserveProceduralFn = void (*) (
      void *owner, const std::string &key, long long memory_id);

  Scope (ProcessorContext &context, void *owner, BackupFn backup,
         BackupFn backup_all, PreserveSlotFn preserve_slot,
         PreserveSparseIndexFn preserve_sparse_index,
         PreserveProceduralFn preserve_procedural)
      : context_ (&context), owner_ (owner), backup_ (backup),
        backup_all_ (backup_all), preserve_slot_ (preserve_slot),
        preserve_sparse_index_ (preserve_sparse_index),
        preserve_procedural_ (preserve_procedural),
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

  static void
  PreserveSparseIndexBeforeInsert (ProcessorContext &context,
                                   const std::string &key,
                                   long long memory_id)
  {
    for (auto *scope = active_; scope; scope = scope->previous_)
      {
        if (scope->context_ == &context)
          {
            scope->preserve_sparse_index_ (
                scope->owner_, key, memory_id);
            return;
          }
      }
  }

  static void
  PreserveProceduralValueBeforeUpdate (ProcessorContext &context,
                                       const std::string &key,
                                       long long memory_id)
  {
    for (auto *scope = active_; scope; scope = scope->previous_)
      {
        if (scope->context_ == &context)
          {
            scope->preserve_procedural_ (scope->owner_, key, memory_id);
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
  PreserveSparseIndexFn preserve_sparse_index_ = nullptr;
  PreserveProceduralFn preserve_procedural_ = nullptr;
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

inline void
PreserveSparseIndexBeforeInsert (ProcessorContext &context,
                                 const std::string &key,
                                 long long memory_id)
{
  Scope::PreserveSparseIndexBeforeInsert (context, key, memory_id);
}

inline void
PreserveProceduralValueBeforeUpdate (ProcessorContext &context,
                                     const std::string &key,
                                     long long memory_id)
{
  Scope::PreserveProceduralValueBeforeUpdate (context, key, memory_id);
}

} // namespace cortext::operations::signal_record_rollback_internal
