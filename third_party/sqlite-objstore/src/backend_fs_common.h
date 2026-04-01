#ifndef OBJSTORE_BACKEND_FS_COMMON_H
#define OBJSTORE_BACKEND_FS_COMMON_H

#include <stdbool.h>
#include <stddef.h>

#include <sqlite3.h>

#include "objstore/backend.h"

#define OBJSTORE_FILE_SUFFIX ".dat"
#define OBJSTORE_PUT_DIR "put"
#define OBJSTORE_DELETE_DIR "delete"
#define OBJSTORE_MANIFEST_FILE "manifest.log"

char *objstore_fs_path_join (const char *lhs, const char *rhs);
int objstore_fs_mkdirs (const char *path);
int objstore_fs_remove_tree (const char *path);
int objstore_fs_rename (const char *from, const char *to);
bool objstore_fs_path_exists (const char *path);
void objstore_fs_id_to_hex (const objstore_id *id, char *hex_out);
int objstore_fs_hex_to_id (const char *hex, objstore_id *out_id);
void objstore_fs_random_hex (char *buffer, size_t length);
size_t objstore_fs_effective_shard_width (const objstore_config *config,
                                          size_t default_width);
char *objstore_fs_object_directory (const char *objects_root,
                                    size_t shard_width, const char *hex_id);
char *objstore_fs_object_path (const char *objects_root, size_t shard_width,
                               const char *hex_id);

#endif /* OBJSTORE_BACKEND_FS_COMMON_H */
