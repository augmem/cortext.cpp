#include <objstore/objstore.h>

#include <stdio.h>
#include <string.h>

int
main (void)
{
  const char *version = objstore_version ();
  if (version == NULL || strcmp (version, OBJSTORE_VERSION_STRING) != 0)
    {
      fprintf (stderr, "unexpected objstore version\n");
      return 1;
    }
  return 0;
}
