#include <catch2/catch_session.hpp>

#include <cstdlib>
#include <string>
#include <vector>

#if defined(_WIN32)
extern "C" char **_environ;
#else
extern char **environ;
#endif

namespace
{

void
UnsetEnv (const std::string &name)
{
#if defined(_WIN32)
  _putenv_s (name.c_str (), "");
#else
  unsetenv (name.c_str ());
#endif
}

void
ScrubCortextEnvironment ()
{
#if defined(_WIN32)
  char **env = _environ;
#else
  char **env = environ;
#endif
  std::vector<std::string> names;
  for (char **entry = env; entry != nullptr && *entry != nullptr; ++entry)
    {
      const std::string value (*entry);
      const auto equals = value.find ('=');
      const std::string name = value.substr (0, equals);
      if (name.rfind ("CORTEXT_", 0) == 0)
        {
          names.push_back (name);
        }
    }
  for (const auto &name : names)
    {
      UnsetEnv (name);
    }
}

} // namespace

int
main (int argc, char *argv[])
{
  ScrubCortextEnvironment ();
  return Catch::Session ().run (argc, argv);
}
