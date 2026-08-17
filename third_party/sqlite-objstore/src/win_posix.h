#ifndef OBJSTORE_WIN_POSIX_H
#define OBJSTORE_WIN_POSIX_H

/*
 * Minimal POSIX unistd/fcntl/sys-file subset for Windows (MSVC and clang-cl).
 *
 * The objstore file backends use open/read/write/close, fsync, ftruncate,
 * flock, the O_* open flags, and a few unistd functions. UCRT provides the
 * underscore-prefixed primitives (_open, _read, _write, _close, _commit,
 * _chsize_s, _get_osfhandle) and the prefixed _O_* flags, but not the bare
 * POSIX names or the un-prefixed O_* aliases in plain C mode. On other
 * platforms the real <unistd.h> and <sys/file.h> are used instead.
 *
 * Every definition is guarded so this header also compiles under MinGW,
 * where the bare names already exist.
 */

#define WIN32_LEAN_AND_MEAN 1
#include <direct.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef O_RDONLY
#define O_RDONLY _O_RDONLY
#define O_WRONLY _O_WRONLY
#define O_RDWR _O_RDWR
#define O_CREAT _O_CREAT
#define O_TRUNC _O_TRUNC
#define O_EXCL _O_EXCL
#define O_APPEND _O_APPEND
#endif
#ifndef O_CLOEXEC
#define O_CLOEXEC _O_NOINHERIT
#endif
#ifndef open
#define open _open
#define read _read
#define write _write
#define close _close
#endif
#ifndef fsync
#define fsync(fd) _commit(fd)
#endif
#ifndef LOCK_SH
#define LOCK_SH 1
#define LOCK_EX 2
#define LOCK_NB 4
#define LOCK_UN 8
#endif

#ifdef __cplusplus
extern "C" {
#endif

int flock (int fd, int how);
int ftruncate (int fd, off_t length);

#ifdef __cplusplus
}
#endif

#endif /* OBJSTORE_WIN_POSIX_H */
