#ifndef OBJSTORE_TEST_HARNESS_H
#define OBJSTORE_TEST_HARNESS_H

#include <stddef.h>
#include <stdint.h>

#include <sqlite3.h>

#include "objstore/backend.h"
#include "objstore/objstore.h"

#ifdef __cplusplus
extern "C"
{
#endif

  int objstore_open_shared_memory_db (sqlite3 **out_db, const char *label);
  sqlite3 *objstore_open_ephemeral_db (void);
  void objstore_close_ephemeral_db (sqlite3 *db);
  int objstore_remove_tree (const char *path);
  char *objstore_create_temp_root (const char *tag);
  void objstore_ensure_dir (const char *path);
  void objstore_prepare_file_roots (const char *base_path,
                                    char **out_objects_root,
                                    char **out_staging_root);
  char *objstore_build_sharded_path (const char *objects_root,
                                     const char *hex, uint8_t shard_width);
  void objstore_exec_or_fail (sqlite3 *db, const char *sql);
  void objstore_exec_expect_error (sqlite3 *db, const char *sql);
  void objstore_id_to_hex (const objstore_id *id, char *hex_out);
  void objstore_compute_zero_payload_id (size_t size, objstore_id *out_id);
  sqlite3 *objstore_open_file_backend_db (char **out_base, char **out_objects,
                                          char **out_staging,
                                          const char *label);
  int objstore_exists_helper (sqlite3 *db, const uint8_t *id_bytes);

#ifdef __cplusplus
}
#endif

#endif /* OBJSTORE_TEST_HARNESS_H */

