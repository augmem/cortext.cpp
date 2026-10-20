#include "win_posix.h"

#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>

int
flock (int fd, int how)
{
  HANDLE handle;
  BOOL exclusive = (how & LOCK_EX) != 0;

  handle = (HANDLE) _get_osfhandle (fd);
  if (handle == INVALID_HANDLE_VALUE)
    {
      errno = EBADF;
      return -1;
    }

  if (how & LOCK_UN)
    {
      if (UnlockFileEx (handle, 0, 1, 0, (LPOVERLAPPED) 0) != 0)
        {
          return 0;
        }
      errno = (GetLastError () == ERROR_NOT_LOCKED) ? EINVAL : EIO;
      return -1;
    }

  /*
   * LockFileEx is always non-blocking; the POSIX callers of this shim use
   * both modes (LOCK_NB for the primary probe, blocking waits while the
   * primary finishes), so emulate the blocking form with a short retry
   * loop on the dedicated lock file.
   */
  for (;;)
    {
      if (LockFileEx (handle, exclusive ? LOCKFILE_EXCLUSIVE_LOCK : 0,
                      0, 1, 0, (LPOVERLAPPED) 0) != 0)
        {
          return 0;
        }
      if (how & LOCK_NB)
        {
          errno = (GetLastError () == ERROR_LOCK_VIOLATION) ? EACCES : EIO;
          return -1;
        }
      Sleep (10);
    }
}

int
ftruncate (int fd, off_t length)
{
  errno_t err;

  err = _chsize_s (fd, length);
  if (err != 0)
    {
      errno = (int) err;
      return -1;
    }
  return 0;
}
