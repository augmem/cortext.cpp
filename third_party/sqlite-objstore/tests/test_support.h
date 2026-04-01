#ifndef OBJSTORE_TEST_SUPPORT_H
#define OBJSTORE_TEST_SUPPORT_H

#include <sqlite3.h>

#include "objstore/backend.h"

#ifdef __cplusplus
extern "C"
{
#endif

  typedef enum objstore_fixture_kind
  {
    OBJSTORE_FIXTURE_SMOKE = 0,
    OBJSTORE_FIXTURE_SQLITE_BACKEND,
  } objstore_fixture_kind;

  void objstore_tests_set_fixture (objstore_fixture_kind kind);
  const objstore_backend *objstore_tests_sqlite_backend (void);
  objstore_backend_env *objstore_tests_backend_env (void);
  sqlite3 *objstore_tests_sqlite_db (void);

  void core_api_register_tests (void);
  void vtab_crud_register_tests (void);
  void scalar_function_register_tests (void);
  void sqlite_helper_register_tests (void);
  void file_backend_vtab_register_tests (void);
  void txn_log_register_tests (void);
  void backend_sqlite_register_tests (void);
  void backend_file_register_tests (void);
  void backend_vfs_register_tests (void);
  void object_manager_register_tests (void);

#ifdef __cplusplus
}
#endif

#endif /* OBJSTORE_TEST_SUPPORT_H */
