#ifndef OBJSTORE_BLAKE3_H
#define OBJSTORE_BLAKE3_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>
#include <stdint.h>

#include "objstore/backend.h"
#include <blake3.h>

  /**
   * Thin wrapper around the upstream BLAKE3 hasher so objstore can hash payloads
   * without leaking third-party headers into dependents.
   */
  typedef struct objstore_blake3
  {
    blake3_hasher hasher;
  } objstore_blake3;

  /** Initializes a hashing context. */
  void objstore_blake3_init (objstore_blake3 *ctx);

  /** Feeds bytes into an active hashing context. */
  void objstore_blake3_update (objstore_blake3 *ctx, const void *data,
                               size_t len);

  /** Finalizes the hash and writes OBJSTORE_ID_SIZE bytes into out_id. */
  void objstore_blake3_final (objstore_blake3 *ctx, objstore_id *out_id);

  /** Convenience helper that hashes a contiguous blob in one call. */
  void objstore_blake3_hash_blob (const uint8_t *data, size_t size,
                                  objstore_id *out_id);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OBJSTORE_BLAKE3_H */
