#include "objstore/backend.h"

#include "backend_portable.h"

static int
opfs_open_env (sqlite3 *db, const objstore_config *config,
               objstore_backend_env **out_env)
{
  return objstore_portable_open_env (OBJSTORE_PORTABLE_OPFS, db, config,
                                     out_env);
}

const objstore_backend objstore_backend_opfs = {
  .kind = OBJSTORE_BACKEND_OPFS,
  .name = "opfs",
  .open_env = opfs_open_env,
  .close_env = objstore_portable_close_env,
  .begin_txn = objstore_portable_begin_txn,
  .commit_txn = objstore_portable_commit_txn,
  .rollback_txn = objstore_portable_rollback_txn,
  .savepoint_begin = objstore_portable_savepoint_begin,
  .savepoint_release = objstore_portable_savepoint_release,
  .savepoint_rollback = objstore_portable_savepoint_rollback,
  .staged_write_begin = objstore_portable_staged_write_begin,
  .staged_write_push = objstore_portable_staged_write_push,
  .staged_write_finalize = objstore_portable_staged_write_finalize,
  .staged_write_set_size_hint = objstore_portable_staged_write_set_hint,
  .commit_staged = objstore_portable_commit_staged,
  .rollback_staged = objstore_portable_rollback_staged,
  .put = objstore_portable_put,
  .get = objstore_portable_get,
  .get_range = objstore_portable_get_range,
  .get_size = objstore_portable_get_size,
  .delete_fn = objstore_portable_delete,
  .exists = objstore_portable_exists,
  .scan_open = objstore_portable_scan_open,
  .scan_next = objstore_portable_scan_next,
  .scan_close = objstore_portable_scan_close,
};
