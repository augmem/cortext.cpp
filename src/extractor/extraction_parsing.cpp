#include "cortext/extractor/extraction_parsing.hpp"

#include <nlohmann/json.hpp>

#include <cctype>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cortext
{

namespace
{

std::string
Trim (const std::string &text)
{
  size_t begin = 0;
  while (begin < text.size ()
         && std::isspace (static_cast<unsigned char> (text[begin])))
    {
      ++begin;
    }
  size_t end = text.size ();
  while (end > begin
         && std::isspace (static_cast<unsigned char> (text[end - 1])))
    {
      --end;
    }
  return text.substr (begin, end - begin);
}

bool
LooksLikeConcreteLooseLabel (const std::string &label)
{
  return label.find (' ') != std::string::npos
         || (!label.empty ()
             && std::isupper (static_cast<unsigned char> (label.front ())));
}

std::vector<std::string>
ParseLooseBraceLabels (const std::string &content)
{
  const size_t begin = content.find ('{');
  const size_t end = content.find ('}', begin == std::string::npos ? 0 : begin);
  if (begin == std::string::npos || end == std::string::npos || end <= begin)
    {
      return {};
    }

  std::vector<std::string> labels;
  std::stringstream stream (content.substr (begin + 1, end - begin - 1));
  std::string item;
  while (std::getline (stream, item, ','))
    {
      std::string label = Trim (item);
      const size_t colon = label.find (':');
      if (colon != std::string::npos)
        {
          label = Trim (label.substr (colon + 1));
        }
      if (!label.empty () && LooksLikeConcreteLooseLabel (label))
        {
          labels.push_back (label);
        }
    }
  return labels;
}

std::optional<nlohmann::json>
TryParseJsonObject (const std::string &content)
{
  for (std::size_t start = content.find ('{'); start != std::string::npos;
       start = content.find ('{', start + 1))
    {
      int depth = 0;
      bool in_string = false;
      bool escaped = false;
      for (std::size_t pos = start; pos < content.size (); ++pos)
        {
          const char c = content[pos];
          if (escaped)
            {
              escaped = false;
              continue;
            }
          if (c == '\\' && in_string)
            {
              escaped = true;
              continue;
            }
          if (c == '"')
            {
              in_string = !in_string;
              continue;
            }
          if (in_string)
            {
              continue;
            }
          if (c == '{')
            {
              ++depth;
            }
          else if (c == '}')
            {
              --depth;
              if (depth == 0)
                {
                  try
                    {
                      return nlohmann::json::parse (
                          content.substr (start, pos - start + 1));
                    }
                  catch (const nlohmann::json::exception &)
                    {
                      break;
                    }
                }
            }
        }
    }
  return std::nullopt;
}

} // namespace

operations::ExtractionResult
ParseExtractionResponse (const std::string &content)
{
  operations::ExtractionResult result;
  auto json_opt = TryParseJsonObject (content);
  if (!json_opt)
    {
      for (const auto &label : ParseLooseBraceLabels (content))
        {
          result.labels.push_back ({ label, 0.5 });
        }
      return result;
    }

  const auto &json_output = *json_opt;
  if (json_output.contains ("labels"))
    {
      for (const auto &label : json_output["labels"])
        {
          if (label.is_string ())
            {
              operations::ExtractedLabel e;
              e.label = label.get<std::string> ();
              e.salience = 0.5;
              result.labels.push_back (std::move (e));
            }
          else if (label.is_object ())
            {
              // Tolerate object-shaped labels if the model fails to follow the prompt.
              operations::ExtractedLabel e;
              e.label = label.value ("label", label.value ("name", ""));
              e.salience = 0.5;
              if (!e.label.empty ())
                {
                  result.labels.push_back (std::move (e));
                }
            }
        }
    }

  if (result.labels.empty () && json_output.contains ("subject")
      && json_output.contains ("object") && json_output["subject"].is_string ()
      && json_output["object"].is_string ())
    {
      const std::string subject = json_output.value ("subject", "");
      const std::string object = json_output.value ("object", "");
      if (!subject.empty ())
        {
          result.labels.push_back ({ subject, 0.5 });
        }
      if (!object.empty () && object != subject)
        {
          result.labels.push_back ({ object, 0.5 });
        }
    }

  if (json_output.contains ("relations"))
    {
      for (const auto &relation : json_output["relations"])
        {
          if (!relation.is_object ())
            {
              continue;
            }
          operations::ExtractedRelation r;
          r.subject = relation.value ("subject", "");
          r.predicate = relation.value ("predicate", "");
          r.object = relation.value ("object", "");
          r.confidence = relation.value ("confidence", 0.5);
          result.relations.push_back (std::move (r));
        }
    }
  else if (json_output.contains ("subject") && json_output.contains ("object")
           && json_output["subject"].is_string ()
           && json_output["object"].is_string ())
    {
      operations::ExtractedRelation r;
      r.subject = json_output.value ("subject", "");
      r.predicate = json_output.value ("predicate", "");
      r.object = json_output.value ("object", "");
      r.confidence = json_output.value ("confidence", 0.5);
      if (!r.subject.empty () && !r.predicate.empty () && !r.object.empty ())
        {
          result.relations.push_back (std::move (r));
        }
    }

  if (json_output.contains ("facts"))
    {
      for (const auto &fact : json_output["facts"])
        {
          if (!fact.is_object ())
            {
              continue;
            }
          operations::ExtractedFact f;
          f.subject = fact.value ("subject", "");
          f.predicate = fact.value ("predicate", "");
          f.object = fact.value ("object", "");
          f.confidence = fact.value ("confidence", 0.5);
          if (fact.contains ("valid_start_ts")
              && fact["valid_start_ts"].is_number ())
            {
              f.valid_start_ts = fact["valid_start_ts"].get<std::uint64_t> ();
            }
          if (fact.contains ("valid_end_ts")
              && fact["valid_end_ts"].is_number ())
            {
              f.valid_end_ts = fact["valid_end_ts"].get<std::uint64_t> ();
            }
          if (!f.subject.empty () && !f.predicate.empty ()
              && !f.object.empty ())
            {
              result.facts.push_back (std::move (f));
            }
        }
    }

  return result;
}

} // namespace cortext
