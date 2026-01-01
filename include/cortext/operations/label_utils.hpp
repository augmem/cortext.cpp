#pragma once

#include <cctype>
#include <string>

namespace cortext::operations
{

inline std::string
TrimLabel (const std::string &label)
{
  auto start = label.begin ();
  auto end = label.end ();
  while (start != end && std::isspace (static_cast<unsigned char> (*start)))
    {
      ++start;
    }
  while (end != start)
    {
      auto prev = end;
      --prev;
      if (!std::isspace (static_cast<unsigned char> (*prev)))
        {
          break;
        }
      end = prev;
    }
  return std::string (start, end);
}

inline std::string
NormalizeLabelKey (const std::string &label)
{
  std::string trimmed = TrimLabel (label);
  std::string out;
  out.reserve (trimmed.size ());
  for (unsigned char c : trimmed)
    {
      out.push_back (static_cast<char> (std::tolower (c)));
    }
  return out;
}

} // namespace cortext::operations
