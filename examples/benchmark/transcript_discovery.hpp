#pragma once

/// Shared transcript discovery for the chat-replay benchmarks.
///
/// A chat-replay corpus is a directory containing exactly one top-level
/// `.txt` transcript plus optional media files. Each message in the
/// transcript starts with a header line `YYYY-MM-DD HH:MM:SS <marker
/// containing " to " or " from ">`, followed by the body on subsequent
/// lines; messages are separated by a long dash rule. `" from "` in the
/// header marks a message from the contact; anything else is from the
/// user.

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace chat_replay
{

/// Returns the transcript path for `input_dir`. `override_path` (from a
/// `--transcript` flag) wins when non-empty. Otherwise the directory must
/// contain exactly one top-level `.txt` file; when several are present, a
/// single legacy `Messages - *.txt` export is accepted as the transcript.
inline std::filesystem::path
DiscoverTranscript (const std::filesystem::path &input_dir,
                    const std::filesystem::path &override_path = {})
{
  namespace fs = std::filesystem;
  if (!override_path.empty ())
    {
      if (!fs::is_regular_file (override_path))
        throw std::runtime_error ("transcript not found: "
                                  + override_path.string ());
      return override_path;
    }
  if (input_dir.empty () || !fs::is_directory (input_dir))
    throw std::runtime_error (
        "--input-dir is required and must be a directory containing the "
        "chat export (one top-level .txt transcript plus optional media)");
  std::vector<fs::path> texts;
  std::vector<fs::path> legacy;
  for (const auto &entry : fs::directory_iterator (input_dir))
    {
      if (!entry.is_regular_file ()
          || entry.path ().extension () != ".txt")
        continue;
      texts.push_back (entry.path ());
      if (entry.path ().filename ().string ().rfind ("Messages - ", 0) == 0)
        legacy.push_back (entry.path ());
    }
  if (texts.size () == 1)
    return texts.front ();
  if (legacy.size () == 1)
    return legacy.front ();
  if (texts.empty ())
    throw std::runtime_error (
        "no .txt transcript found in " + input_dir.string ()
        + "; expected a chat export with timestamped "
          "\"YYYY-MM-DD HH:MM:SS ... to|from ...\" message headers");
  std::string names;
  for (const auto &p : texts)
    names += (names.empty () ? "" : ", ") + p.filename ().string ();
  throw std::runtime_error ("multiple .txt transcripts in "
                            + input_dir.string () + " (" + names
                            + "); pass --transcript to choose one");
}

} // namespace chat_replay
