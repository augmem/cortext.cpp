#include "objstore/blake3.h"

#include <stddef.h>

void
objstore_blake3_init (objstore_blake3 *ctx)
{
  if (ctx == NULL)
    {
      return;
    }
  blake3_hasher_init (&ctx->hasher);
}

void
objstore_blake3_update (objstore_blake3 *ctx, const void *data, size_t len)
{
  if (ctx == NULL || len == 0 || data == NULL)
    {
      return;
    }
  blake3_hasher_update (&ctx->hasher, data, len);
}

void
objstore_blake3_final (objstore_blake3 *ctx, objstore_id *out_id)
{
  if (ctx == NULL || out_id == NULL)
    {
      return;
    }
  blake3_hasher_finalize (&ctx->hasher, out_id->bytes, OBJSTORE_ID_SIZE);
  objstore_blake3_init (ctx);
}

void
objstore_blake3_hash_blob (const uint8_t *data, size_t size,
                           objstore_id *out_id)
{
  if (out_id == NULL)
    {
      return;
    }
  objstore_blake3 ctx;
  objstore_blake3_init (&ctx);
  if (data != NULL && size > 0)
    {
      objstore_blake3_update (&ctx, data, size);
    }
  objstore_blake3_final (&ctx, out_id);
}
