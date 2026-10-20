#include "win_dirent.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>

DIR *
opendir (const char *path)
{
  DIR *dir;
  WIN32_FIND_DATAA find_data;
  HANDLE handle;

  if (path == NULL)
    {
      errno = ENOENT;
      return NULL;
    }

  dir = (DIR *) malloc (sizeof (DIR));
  if (dir == NULL)
    {
      errno = ENOMEM;
      return NULL;
    }

  handle = FindFirstFileA (path, &find_data);
  if (handle == INVALID_HANDLE_VALUE)
    {
      free (dir);
      errno = ENOENT;
      return NULL;
    }

  dir->handle = (void *) handle;
  memcpy (dir->entry.d_name, find_data.cFileName,
          sizeof (find_data.cFileName));
  dir->entry_valid = 1;
  return dir;
}

struct dirent *
readdir (DIR *dir)
{
  WIN32_FIND_DATAA find_data;

  if (dir == NULL)
    {
      errno = EBADF;
      return NULL;
    }

  if (dir->entry_valid)
    {
      dir->entry_valid = 0;
      return &dir->entry;
    }

  if (!FindNextFileA ((HANDLE) dir->handle, &find_data))
    {
      return NULL;
    }

  memcpy (dir->entry.d_name, find_data.cFileName,
          sizeof (find_data.cFileName));
  dir->entry_valid = 1;
  return &dir->entry;
}

int
closedir (DIR *dir)
{
  if (dir == NULL)
    {
      errno = EBADF;
      return -1;
    }

  FindClose ((HANDLE) dir->handle);
  free (dir);
  return 0;
}
