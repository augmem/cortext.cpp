#include "cortext/operations/graph_schema.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/schema.hpp"

namespace cortext::operations
{

void
EnsureGraphSchema::Execute (OperationContext &context, Transaction &tx) const
{
  // No-op: schema is managed via core schema.cpp migration 0.
  (void)context;
}

void
EnsureGraphSchema::CollectSchema (cortext::store::SchemaRegistry &registry) const
{
  // No-op: all graph/consolidation/extraction tables are now in core schema.cpp.
  // This method is kept for interface compatibility but registers nothing.
  (void)registry;
}

} // namespace cortext::operations
