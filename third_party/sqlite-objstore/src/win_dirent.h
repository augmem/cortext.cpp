#ifndef OBJSTORE_WIN_DIRENT_H
#define OBJSTORE_WIN_DIRENT_H

/*
 * Minimal POSIX dirent subset for Windows (MSVC and clang-cl).
 *
 * sqlite-objstore only needs opendir(), readdir(), closedir(), and
 * d_name; the system <dirent.h> does not exist on Windows. On other
 * platforms the real <dirent.h> is used instead.
 */

#ifdef __cplusplus
extern "C" {
#endif

struct dirent
{
  char d_name[260];
};

typedef struct
{
  void *handle;
  int entry_valid;
  struct dirent entry;
} DIR;

DIR *opendir (const char *path);
struct dirent *readdir (DIR *dir);
int closedir (DIR *dir);

#ifdef __cplusplus
}
#endif

#endif /* OBJSTORE_WIN_DIRENT_H */
