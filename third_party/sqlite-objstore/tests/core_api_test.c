
#include "unity.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "objstore/backend.h"
#include "objstore/blake3.h"
#include "objstore/objstore.h"
#include "test_harness.h"
#include "test_support.h"


static void
test_objstore_version_exposes_semantic_version_string (void)
{
  const char *version = objstore_version ();
  TEST_ASSERT_NOT_NULL (version);
  TEST_ASSERT_EQUAL_STRING (OBJSTORE_VERSION_STRING, version);
}


static void
test_objstore_register_validates_inputs (void)
{
  TEST_ASSERT_EQUAL_INT (SQLITE_MISUSE, objstore_register (NULL, NULL));

  sqlite3 *db = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, objstore_open_shared_memory_db (&db, "register"));
  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_AUTO,
    .storage_root = NULL,
    .chunk_size_bytes = 0,
    .reserved_flags = 0,
  };

  TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_register (db, &cfg));

  cfg.reserved_flags = 1;
  TEST_ASSERT_EQUAL_INT (SQLITE_MISUSE, objstore_register (db, &cfg));
  sqlite3_close (db);
}

void
core_api_register_tests (void)
{
  RUN_TEST (test_objstore_version_exposes_semantic_version_string);
  RUN_TEST (test_objstore_register_validates_inputs);
}
