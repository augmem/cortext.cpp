
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

static void
test_objstore_register_is_silent_on_success (void)
{
  sqlite3 *db = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, objstore_open_shared_memory_db (&db, "register_silent"));

  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_AUTO,
    .storage_root = NULL,
    .chunk_size_bytes = 0,
    .reserved_flags = 0,
  };

  int pipefd[2] = { -1, -1 };
  TEST_ASSERT_EQUAL_INT (0, pipe (pipefd));
  int saved_stderr = dup (STDERR_FILENO);
  TEST_ASSERT_TRUE (saved_stderr >= 0);
  fflush (stderr);
  TEST_ASSERT_EQUAL_INT (STDERR_FILENO, dup2 (pipefd[1], STDERR_FILENO));
  close (pipefd[1]);
  pipefd[1] = -1;

  const int rc = objstore_register (db, &cfg);

  fflush (stderr);
  TEST_ASSERT_EQUAL_INT (STDERR_FILENO, dup2 (saved_stderr, STDERR_FILENO));
  close (saved_stderr);
  saved_stderr = -1;

  char buffer[64];
  const int nread = (int)read (pipefd[0], buffer, sizeof (buffer));
  close (pipefd[0]);
  pipefd[0] = -1;

  TEST_ASSERT_EQUAL_INT (SQLITE_OK, rc);
  TEST_ASSERT_TRUE (nread >= 0);
  TEST_ASSERT_EQUAL_INT (0, (int)nread);
  sqlite3_close (db);
}

void
core_api_register_tests (void)
{
  RUN_TEST (test_objstore_version_exposes_semantic_version_string);
  RUN_TEST (test_objstore_register_validates_inputs);
  RUN_TEST (test_objstore_register_is_silent_on_success);
}
