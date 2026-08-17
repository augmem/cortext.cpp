#include "backend_fs_common.h"

#if defined(_WIN32)
#include "win_dirent.h"
#else
#include <dirent.h>
#endif
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#if defined(_WIN32)
#include <direct.h>
#endif

static int
objstore_fs_mkdir_one (const char *path)
{
#if defined(_WIN32)
  return _mkdir (path);
#else
  return mkdir (path, 0777);
#endif
}

char *
objstore_fs_path_join (const char *lhs, const char *rhs)
{
  if (lhs == NULL || lhs[0] == '\0')
    {
      return sqlite3_mprintf ("%s", rhs ? rhs : "");
    }
  if (rhs == NULL || rhs[0] == '\0')
    {
      return sqlite3_mprintf ("%s", lhs);
    }
  const size_t len = strlen (lhs);
  const char separator = '/';
  if (lhs[len - 1] == separator)
    {
      return sqlite3_mprintf ("%s%s", lhs, rhs);
    }
  return sqlite3_mprintf ("%s%c%s", lhs, separator, rhs);
}

int
objstore_fs_mkdirs (const char *path)
{
  if (path == NULL || path[0] == '\0')
    {
      return SQLITE_MISUSE;
    }
  char *mutable_path = sqlite3_mprintf ("%s", path);
  if (mutable_path == NULL)
    {
      return SQLITE_NOMEM;
    }

  int rc = SQLITE_OK;
  for (char *p = mutable_path + 1; *p != '\0'; ++p)
    {
      if (*p == '/')
        {
          *p = '\0';
          if (objstore_fs_mkdir_one (mutable_path) != 0 && errno != EEXIST)
            {
              rc = SQLITE_IOERR;
              goto done;
            }
          *p = '/';
        }
    }
  if (objstore_fs_mkdir_one (mutable_path) != 0 && errno != EEXIST)
    {
      rc = SQLITE_IOERR;
    }

done:
  sqlite3_free (mutable_path);
  return rc;
}

int
objstore_fs_remove_tree (const char *path)
{
  DIR *dir = opendir (path);
  if (dir == NULL)
    {
      if (errno == ENOENT)
        {
          return SQLITE_OK;
        }
      return SQLITE_IOERR;
    }

  struct dirent *entry = NULL;
  while ((entry = readdir (dir)) != NULL)
    {
      if (strcmp (entry->d_name, ".") == 0
          || strcmp (entry->d_name, "..") == 0)
        {
          continue;
        }
      char *child = objstore_fs_path_join (path, entry->d_name);
      if (child == NULL)
        {
          closedir (dir);
          return SQLITE_NOMEM;
        }
      struct stat st;
      if (stat (child, &st) == 0)
        {
          if (S_ISDIR (st.st_mode))
            {
              int rc = objstore_fs_remove_tree (child);
              sqlite3_free (child);
              if (rc != SQLITE_OK)
                {
                  closedir (dir);
                  return rc;
                }
              continue;
            }
          else
            {
              if (unlink (child) != 0)
                {
                  const int err = errno;
                  sqlite3_free (child);
                  closedir (dir);
                  return (err == ENOENT) ? SQLITE_OK : SQLITE_IOERR;
                }
            }
        }
      sqlite3_free (child);
    }
  closedir (dir);
  return (rmdir (path) == 0 || errno == ENOENT) ? SQLITE_OK : SQLITE_IOERR;
}

int
objstore_fs_rename (const char *from, const char *to)
{
  if (from == NULL || to == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (rename (from, to) != 0)
    {
      return (errno == ENOENT) ? SQLITE_NOTFOUND : SQLITE_IOERR;
    }
  return SQLITE_OK;
}

bool
objstore_fs_path_exists (const char *path)
{
  if (path == NULL)
    {
      return false;
    }
  struct stat st;
  return stat (path, &st) == 0;
}

static int
objstore_fs_hex_value (char c)
{
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'a' && c <= 'f')
    return 10 + (c - 'a');
  if (c >= 'A' && c <= 'F')
    return 10 + (c - 'A');
  return -1;
}

void
objstore_fs_id_to_hex (const objstore_id *id, char *hex_out)
{
  static const char kHexDigits[] = "0123456789abcdef";
  if (id == NULL || hex_out == NULL)
    {
      return;
    }
  for (size_t i = 0; i < OBJSTORE_ID_SIZE; ++i)
    {
      hex_out[i * 2] = kHexDigits[(id->bytes[i] >> 4) & 0xF];
      hex_out[i * 2 + 1] = kHexDigits[id->bytes[i] & 0xF];
    }
  hex_out[OBJSTORE_ID_SIZE * 2] = '\0';
}

int
objstore_fs_hex_to_id (const char *hex, objstore_id *out_id)
{
  if (hex == NULL || out_id == NULL || strlen (hex) < OBJSTORE_ID_SIZE * 2)
    {
      return SQLITE_MISUSE;
    }
  for (size_t i = 0; i < OBJSTORE_ID_SIZE; ++i)
    {
      int hi = objstore_fs_hex_value (hex[i * 2]);
      int lo = objstore_fs_hex_value (hex[i * 2 + 1]);
      if (hi < 0 || lo < 0)
        {
          return SQLITE_MISUSE;
        }
      out_id->bytes[i] = (uint8_t)((hi << 4) | lo);
    }
  return SQLITE_OK;
}

void
objstore_fs_random_hex (char *buffer, size_t length)
{
  if (buffer == NULL || length < 2)
    {
      return;
    }
  uint8_t bytes[16];
  sqlite3_randomness (sizeof (bytes), bytes);
  const size_t hex_len = (length - 1) / 2;
  static const char digits[] = "0123456789abcdef";
  for (size_t i = 0; i < hex_len && i < sizeof (bytes); ++i)
    {
      buffer[i * 2] = digits[(bytes[i] >> 4) & 0xF];
      buffer[i * 2 + 1] = digits[bytes[i] & 0xF];
    }
  buffer[hex_len * 2] = '\0';
}

size_t
objstore_fs_effective_shard_width (const objstore_config *config,
                                   size_t default_width)
{
  uint8_t requested
      = (config != NULL && config->shard_width > 0) ? config->shard_width : 0;
  if (requested == 0)
    {
      requested = (uint8_t)default_width;
    }
  const size_t max_hex = OBJSTORE_ID_SIZE * 2u;
  if (requested > max_hex)
    {
      requested = (uint8_t)max_hex;
    }
  if ((requested % 2u) != 0u)
    {
      --requested;
    }
  return requested;
}

char *
objstore_fs_object_directory (const char *objects_root, size_t shard_width,
                              const char *hex_id)
{
  if (objects_root == NULL)
    {
      return NULL;
    }
  char *path = sqlite3_mprintf ("%s", objects_root);
  if (path == NULL)
    {
      return NULL;
    }
  if (shard_width == 0 || hex_id == NULL)
    {
      return path;
    }
  for (size_t i = 0; i < shard_width; i += 2)
    {
      if (hex_id[i] == '\0' || hex_id[i + 1] == '\0')
        {
          break;
        }
      char segment[3] = { hex_id[i], hex_id[i + 1], '\0' };
      char *next = objstore_fs_path_join (path, segment);
      sqlite3_free (path);
      path = next;
      if (path == NULL)
        {
          return NULL;
        }
    }
  return path;
}

char *
objstore_fs_object_path (const char *objects_root, size_t shard_width,
                         const char *hex_id)
{
  char *dir = objstore_fs_object_directory (objects_root, shard_width, hex_id);
  if (dir == NULL)
    {
      return NULL;
    }
  size_t hex_len = hex_id != NULL ? strlen (hex_id) : 0;
  char *filename
      = sqlite3_mprintf ("%.*s%s", (int)hex_len, hex_id, OBJSTORE_FILE_SUFFIX);
  if (filename == NULL)
    {
      sqlite3_free (dir);
      return NULL;
    }
  char *path = objstore_fs_path_join (dir, filename);
  sqlite3_free (dir);
  sqlite3_free (filename);
  return path;
}
