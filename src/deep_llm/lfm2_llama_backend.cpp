#include "lfm2_llama_backend.hpp"

#include "cortext/internal/cancellation.hpp"
#include "cortext/core/thread_config.hpp"
#include "llama_cpp_support.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(CORTEXT_ENABLE_LLAMA_CPP)
using llama_sampler = cortext::internal::llama_sampler;
using llama_vocab = cortext::internal::llama_vocab;
using llama_token = cortext::internal::llama_token;
#endif

namespace cortext
{

namespace
{

std::string
GetEnvOrDefault (const char *name, const std::string &fallback = {})
{
  const char *value = std::getenv (name);
  if (value == nullptr || *value == '\0')
    {
      return fallback;
    }
  return value;
}

int
GetEnvInt (const char *name, int fallback)
{
  const std::string value = GetEnvOrDefault (name);
  if (value.empty ())
    {
      return fallback;
    }
  try
    {
      return std::stoi (value);
    }
  catch (...)
    {
      return fallback;
    }
}

bool
GetEnvBool (const char *name, bool fallback)
{
  const std::string value = GetEnvOrDefault (name);
  if (value.empty ())
    {
      return fallback;
    }

  std::string normalized = value;
  std::transform (normalized.begin (), normalized.end (), normalized.begin (),
                  [] (unsigned char c) { return static_cast<char> (std::tolower (c)); });
  if (normalized == "1" || normalized == "true" || normalized == "yes"
      || normalized == "on")
    {
      return true;
    }
  if (normalized == "0" || normalized == "false" || normalized == "no"
      || normalized == "off")
    {
      return false;
    }
  return fallback;
}

std::string
ResolveExistingPath (const std::string &path, const char *component)
{
  if (path.empty ())
    {
      throw std::runtime_error (std::string (component) + ": model path is empty");
    }
  const std::filesystem::path p (path);
  if (!std::filesystem::exists (p))
    {
      throw std::runtime_error (std::string (component) + ": file not found: "
                                + p.string ());
    }
  return p.string ();
}

std::string
TrimToWordLimit (const std::string &text, int max_words)
{
  if (max_words <= 0)
    {
      return text;
    }
  std::ostringstream out;
  int count = 0;
  bool in_word = false;
  for (char c : text)
    {
      const bool is_space = (c == ' ' || c == '\n' || c == '\t' || c == '\r');
      if (!is_space && !in_word)
        {
          in_word = true;
          count++;
          if (count > max_words)
            {
              break;
            }
        }
      if (count <= max_words)
        {
          out << c;
        }
      if (is_space)
        {
          in_word = false;
        }
    }
  return out.str ();
}

std::string
TrimAsciiWhitespace (std::string value)
{
  auto is_space = [] (unsigned char c) { return std::isspace (c) != 0; };
  value.erase (
      value.begin (),
      std::find_if (value.begin (), value.end (),
                    [&] (unsigned char c) { return !is_space (c); }));
  value.erase (
      std::find_if (value.rbegin (), value.rend (),
                    [&] (unsigned char c) { return !is_space (c); })
          .base (),
      value.end ());
  return value;
}

std::string
SanitizeSummaryText (std::string summary)
{
  if (summary.empty ())
    {
      return summary;
    }

  summary = std::regex_replace (
      summary,
      std::regex (R"(^\s*(Final summary|Output summary|Summary)\s*:\s*)",
                  std::regex_constants::icase),
      "");
  summary = std::regex_replace (
      summary,
      std::regex (R"(\b(Final summary|Output summary|Summary)\s*:\s*)",
                  std::regex_constants::icase),
      "");
  summary = std::regex_replace (
      summary, std::regex (R"(Excerpt\s+\d+\s+(mentions|indicates)\s+)",
                           std::regex_constants::icase),
      "");
  summary = std::regex_replace (
      summary,
      std::regex (R"(,\s*which is a user\b)", std::regex_constants::icase), "");
  summary = std::regex_replace (
      summary,
      std::regex (R"(\bthe user is focusing on\b)",
                  std::regex_constants::icase),
      "focuses on");
  summary = std::regex_replace (
      summary, std::regex (R"(\bthe user is\b)", std::regex_constants::icase),
      "");
  summary = std::regex_replace (
      summary, std::regex (R"(\bthe user\b)", std::regex_constants::icase),
      "");
  summary = std::regex_replace (
      summary,
      std::regex (R"(\bthe assistant\b)", std::regex_constants::icase), "");
  summary = std::regex_replace (summary, std::regex (R"([ \t]+)"), " ");
  summary = std::regex_replace (summary, std::regex (R"(\s+\.)"), ".");
  summary = std::regex_replace (summary, std::regex (R"(\n{3,})"), "\n\n");
  return TrimAsciiWhitespace (summary);
}

bool
UseStructuredSummaryOutput ()
{
  return GetEnvBool ("CORTEXT_LFM2_SUMMARY_STRUCTURED", true);
}

std::string
BuildSummaryFormatInstruction ()
{
  if (UseStructuredSummaryOutput ())
    {
      return "Return only a single JSON object of the form "
             "{\"summary\":\"...\"}.\n";
    }
  return "Return only the summary text.\n";
}

int
ComputeSummaryMaxTokens (const std::vector<std::string> &texts, int max_words)
{
  if (max_words > 0)
    {
      return std::clamp (max_words * 8, 64, 1024);
    }

  std::size_t total_chars = 0;
  for (const auto &text : texts)
    {
      total_chars += text.size ();
    }

  const int approx_input_tokens
      = static_cast<int> (std::max<std::size_t> (1, total_chars / 4));
  return std::clamp (approx_input_tokens, 256, 1024);
}

std::string
BuildSummarySystemPrompt ()
{
  const std::string style
      = GetEnvOrDefault ("CORTEXT_LFM2_SUMMARY_PROMPT_STYLE",
                         "transcript_fewshot");
  const std::string format = BuildSummaryFormatInstruction ();
  const std::string example_one_summary
      = UseStructuredSummaryOutput ()
            ? "{\"summary\":\"Gabriel lives in Chicago and is debugging a SQLite migration for Cortext before a demo.\"}"
            : "Gabriel lives in Chicago and is debugging a SQLite migration for Cortext before a demo.";
  const std::string example_two_summary
      = UseStructuredSummaryOutput ()
            ? "{\"summary\":\"Sarah is helping with a housing application in Logan Square, and missing pay stubs are delaying the submission.\"}"
            : "Sarah is helping with a housing application in Logan Square, and missing pay stubs are delaying the submission.";
  const std::string example_precision_one_summary
      = UseStructuredSummaryOutput ()
            ? "{\"summary\":\"Alice is leading the Acme pilot in Seattle, and the customer needs SSO and audit exports before June.\"}"
            : "Alice is leading the Acme pilot in Seattle, and the customer needs SSO and audit exports before June.";
  const std::string example_precision_two_summary
      = UseStructuredSummaryOutput ()
            ? "{\"summary\":\"Finance is revising the renewal forecast this week.\"}"
            : "Finance is revising the renewal forecast this week.";
  const std::string example_transcript_two_summary
      = UseStructuredSummaryOutput ()
            ? "{\"summary\":\"Alice is leading the Acme pilot in Seattle, the customer needs SSO and audit exports before June, and Finance is revising the renewal forecast this week.\"}"
            : "Alice is leading the Acme pilot in Seattle, the customer needs SSO and audit exports before June, and Finance is revising the renewal forecast this week.";

  if (style == "durable")
    {
      return std::string (
                 "You are writing a durable memory note from conversation excerpts.\n"
                 "Write a concise factual summary in 1-3 sentences.\n")
             + format
             + "Preserve durable facts about names, places, projects, goals, preferences, plans, blockers, and outcomes.\n"
               "State the facts directly instead of describing who said them.\n"
               "Prefer concrete named subjects and specific noun phrases.\n"
               "If the excerpts contain multiple topics, keep them separate instead of implying they caused each other.\n"
               "Do not use speaker-role subjects or second-person phrasing. Avoid 'the user', 'the assistant', 'you', and 'your'.\n"
               "Do not write meta narration like 'in a conversation', 'they discussed', 'the user said', or 'the assistant asked'.\n"
               "Start directly with the summary sentence. Do not write 'Summary:', 'Final summary:', 'Output summary:', or any example headers.\n"
               "Do not speculate beyond the text.";
    }

  if (style == "precision")
    {
      return std::string (
                 "Write a compact durable summary from the conversation excerpts.\n")
             + format
             + "Use 1-2 sentences.\n"
               "Keep only concrete durable facts that are directly supported by the text.\n"
               "Prefer names, places, projects, deadlines, blockers, and outcomes.\n"
               "Omit banter, greetings, rhetorical questions, and anything uncertain.\n"
               "Start directly with the summary sentence. Do not write 'Summary:', 'Final summary:', 'Output summary:', or any example headers.\n"
               "Do not use speaker-role language, second-person phrasing, or meta commentary.\n"
               "Avoid 'the user', 'the assistant', 'you', 'your', 'in a conversation', 'discussed', 'said', and 'asked'.";
    }

  if (style == "fewshot_durable")
    {
      return std::string (
                 "You are writing a durable memory note from conversation excerpts.\n"
                 "Write a concise factual summary in 1-3 sentences.\n")
             + format
             + "Preserve durable facts about names, places, projects, goals, preferences, plans, blockers, and outcomes.\n"
               "State facts directly instead of describing who said them.\n"
               "Do not use speaker-role subjects or second-person phrasing. Avoid 'the user', 'the assistant', 'you', and 'your'.\n"
               "Do not write meta narration like 'in a conversation', 'they discussed', 'the user said', or 'the assistant asked'.\n"
               "When answering the real task, start directly with the summary sentence. Do not write 'Summary:', 'Final summary:', 'Output summary:', or copy example headers.\n"
               "Do not speculate beyond the text.\n"
               "\n"
               "Example 1\n"
               "Input excerpts:\n"
               "Excerpt 1:\n"
               "User: My name is Gabriel and I live in Chicago.\n"
               "Excerpt 2:\n"
               "Assistant: Noted.\n"
               "Excerpt 3:\n"
               "User: I am debugging a SQLite migration for Cortext before the demo.\n"
               "Output summary:\n"
             + example_one_summary
             + "\n\n"
               "Example 2\n"
               "Input excerpts:\n"
               "Excerpt 1:\n"
               "User: My neighbor Sarah is helping with a housing application in Logan Square.\n"
               "Excerpt 2:\n"
               "Assistant: Missing pay stubs are delaying the submission.\n"
               "Output summary:\n"
             + example_two_summary;
    }

  if (style == "fewshot_precision")
    {
      return std::string (
                 "Write a compact durable summary from the conversation excerpts.\n")
             + format
             + "Use 1-2 sentences.\n"
               "Keep only concrete durable facts directly supported by the text.\n"
               "Prefer names, places, projects, deadlines, blockers, and outcomes.\n"
               "Omit banter, greetings, rhetorical questions, and uncertain details.\n"
               "Do not use speaker-role language, second-person phrasing, or meta commentary.\n"
               "When answering the real task, start directly with the summary sentence. Do not write 'Summary:', 'Final summary:', 'Output summary:', or copy example headers.\n"
               "\n"
               "Example 1\n"
               "Input excerpts:\n"
               "Excerpt 1:\n"
               "User: Alice is leading the Acme pilot in Seattle.\n"
               "Excerpt 2:\n"
               "Assistant: The customer needs SSO and audit exports before June.\n"
               "Output summary:\n"
             + example_precision_one_summary
             + "\n\n"
               "Example 2\n"
               "Input excerpts:\n"
               "Excerpt 1:\n"
               "User: Finance is revising the renewal forecast this week.\n"
               "Excerpt 2:\n"
               "Assistant: Noted.\n"
               "Output summary:\n"
             + example_precision_two_summary;
    }

  if (style == "transcript_fewshot")
    {
      return std::string ("Write a compact durable summary from a chat transcript.\n")
             + format
             + "Use 1-2 sentences.\n"
               "Capture the durable facts across the whole transcript, not just the last turn.\n"
               "Prefer names, places, projects, deadlines, blockers, and outcomes.\n"
               "Use direct factual sentences.\n"
               "Do not use speaker-role language or prompt headers. Avoid 'User:', 'Assistant:', 'Summary:', 'Final summary:', 'the user', 'the assistant', and meta narration.\n"
               "\n"
               "Example 1 transcript:\n"
               "User: My name is Gabriel and I live in Chicago.\n"
               "Assistant: Noted.\n"
               "User: I am debugging a SQLite migration for Cortext before the demo.\n"
               "Example 1 summary:\n"
             + example_one_summary
             + "\n\n"
               "Example 2 transcript:\n"
               "User: Alice is leading the Acme pilot in Seattle.\n"
               "Assistant: The customer needs SSO and audit exports before June.\n"
               "User: Finance is revising the renewal forecast this week.\n"
               "Example 2 summary:\n"
             + example_transcript_two_summary;
    }

  return std::string (
             "You are writing a durable memory note from conversation excerpts.\n"
             "Write a concise factual summary in 1-3 sentences.\n")
         + format
         + "Treat lines labeled 'User:' as a human user and lines labeled 'Assistant:' as the assistant.\n"
           "Summarize the underlying facts and topics, not the mechanics of the conversation.\n"
           "Prioritize durable facts about people, events, names, preferences, plans, and outcomes over banter, greetings, or rhetorical questions.\n"
           "Include all major durable facts that fit, especially named people, projects, technologies, and goals.\n"
           "If the excerpts contain multiple topics, list them as separate facts instead of implying they caused each other.\n"
           "If both user and assistant excerpts restate the same fact, prefer the underlying fact itself instead of narrating who said it.\n"
           "Do not repeat the same fact from different perspectives.\n"
           "State facts directly when possible. Prefer direct factual sentences over wording like 'The user...' or 'The assistant...'.\n"
           "Do not use speaker-role subjects or second-person phrasing in the summary. Avoid 'the user', 'the assistant', 'you', and 'your' when a concrete named subject or neutral phrasing is available.\n"
           "Do not write phrases like 'the user said', 'the assistant asked', 'in a conversation', 'they discussed', or 'this occurred after' unless that wording is necessary for clarity.\n"
           "Do not infer causality, chronology, identity, or shared beliefs beyond the text.\n"
           "Avoid speculation, role confusion, and meta commentary.";
}

std::string
BuildSummaryUserPrompt (const std::vector<std::string> &texts)
{
  const std::string style
      = GetEnvOrDefault ("CORTEXT_LFM2_SUMMARY_PROMPT_STYLE",
                         "transcript_fewshot");
  const bool structured = UseStructuredSummaryOutput ();
  std::ostringstream combined;
  if (style == "transcript_fewshot")
    {
      combined << "Chat transcript:\n";
      for (size_t i = 0; i < texts.size (); ++i)
        {
          combined << texts[i] << "\n";
        }
      combined << "\n" << (structured ? "JSON response:\n" : "Summary:\n");
      return combined.str ();
    }

  combined << "Conversation excerpts:\n\n";
  for (size_t i = 0; i < texts.size (); ++i)
    {
      combined << "Excerpt " << (i + 1) << ":\n" << texts[i] << "\n\n";
    }
  if (style != "baseline")
    {
      combined << (structured ? "JSON response:\n" : "Final summary:\n");
    }
  return combined.str ();
}

std::string
BuildExtractionSystemPrompt ()
{
  const std::string style
      = GetEnvOrDefault ("CORTEXT_LFM2_EXTRACT_PROMPT_STYLE",
                         "fewshot_durable");

  if (style == "durable")
    {
      return "Extract labels, relations, and durable facts from the provided "
             "text.\n"
             "Return only a single JSON object with keys \"labels\", "
             "\"relations\", and optional \"facts\".\n"
             "Labels are durable graph nodes, not every salient word.\n"
             "Prefer precise durable entities and concepts over exhaustive "
             "recall.\n"
             "Choose concrete people, places, organizations, projects, tools, "
             "named goals, and stable topics directly supported by the text.\n"
             "Prefer multi-word noun phrases when they are more specific than "
             "single tokens.\n"
             "Do not output generic verbs, helper words, schema keys, JSON "
             "field names, timestamps, or meta terms like valid_start_ts, "
             "valid_end_ts, explanation, support, salient, get, or ready "
             "unless they are literally the main subject of the text.\n"
             "Avoid labels that are only function words, modifiers, or broad "
             "process words.\n"
             "\"labels\" must be an array of non-empty strings copied from the "
             "text.\n"
             "Return between 0 and 8 labels; use [] when no durable label is "
             "present.\n"
             "Each relation must include non-empty \"subject\", \"predicate\", "
             "and \"object\" strings taken from the text.\n"
             "Each fact must include non-empty \"subject\", \"predicate\", and "
             "\"object\" strings taken from the text.\n"
             "If confidence is present, it must be a JSON number.\n"
             "If valid_start_ts or valid_end_ts are present for facts, they "
             "must be JSON integers in milliseconds.\n"
             "Do not emit any explanation outside the JSON object.";
    }

  if (style == "precision")
    {
      return "Extract labels, relations, and durable facts from the provided "
             "text.\n"
             "Return only a single JSON object with keys \"labels\", "
             "\"relations\", and optional \"facts\".\n"
             "Optimize for label precision, not recall.\n"
             "Labels should be the few memory-worthy nodes a long-term memory "
             "graph should keep: people, places, projects, organizations, "
             "tools, conditions, and durable topics directly supported by the "
             "text.\n"
             "Prefer named entities and concrete noun phrases.\n"
             "Only use a state or activity label if it is clearly enduring or "
             "central to the text.\n"
             "Do not output schema terms, meta words, field names, bare "
             "generic verbs, or filler tokens.\n"
             "\"labels\" must be an array of non-empty strings copied from the "
             "text.\n"
             "Return between 0 and 6 labels; use [] when no durable label is "
             "present.\n"
             "Each relation must include non-empty \"subject\", \"predicate\", "
             "and \"object\" strings taken from the text.\n"
             "Each fact must include non-empty \"subject\", \"predicate\", and "
             "\"object\" strings taken from the text.\n"
             "If confidence is present, it must be a JSON number.\n"
             "If valid_start_ts or valid_end_ts are present for facts, they "
             "must be JSON integers in milliseconds.\n"
             "Do not emit any explanation outside the JSON object.";
    }

  if (style == "fewshot_durable")
    {
      return "Extract labels, relations, and durable facts from the provided "
             "text.\n"
             "Return only a single JSON object with keys \"labels\", "
             "\"relations\", and optional \"facts\".\n"
             "Labels are durable graph nodes, not every salient word.\n"
             "Choose concrete people, places, organizations, projects, tools, "
             "named goals, stable conditions, and enduring topics directly "
             "supported by the text.\n"
             "Prefer multi-word noun phrases when they are more specific than "
             "single tokens.\n"
             "Do not output schema keys, JSON field names, timestamps, helper "
             "words, generic verbs, or meta terms like valid_start_ts, "
             "valid_end_ts, explanation, support, salient, get, or ready "
             "unless they are literally the main subject of the text.\n"
             "\"labels\" must be an array of non-empty strings copied from the "
             "text.\n"
             "Return between 0 and 8 labels; use [] when no durable label is "
             "present.\n"
             "Each relation must include non-empty \"subject\", \"predicate\", "
             "and \"object\" strings taken from the text.\n"
             "Each fact must include non-empty \"subject\", \"predicate\", and "
             "\"object\" strings taken from the text.\n"
             "If confidence is present, it must be a JSON number.\n"
             "If valid_start_ts or valid_end_ts are present for facts, they "
             "must be JSON integers in milliseconds.\n"
             "Examples:\n"
             "Text:\n"
             "User: My name is Gabriel. I live in Chicago.\n"
             "Assistant: Noted. You are debugging a SQLite migration for "
             "Cortext.\n"
             "{\"labels\":[\"Gabriel\",\"Chicago\",\"SQLite migration\","
             "\"Cortext\"],\"relations\":[],\"facts\":[{\"subject\":"
             "\"Gabriel\",\"predicate\":\"lives in\",\"object\":\"Chicago\"}]"
             "}\n"
             "Text:\n"
             "User: I need help getting ready for an interview tomorrow. I "
             "want more confidence and a clear explanation.\n"
             "Assistant: We can prep the interview.\n"
             "{\"labels\":[\"interview\",\"confidence\"],\"relations\":[],"
             "\"facts\":[]}\n"
             "Do not emit any explanation outside the JSON object.";
    }

  if (style == "fewshot_precision")
    {
      return "Extract labels, relations, and durable facts from the provided "
             "text.\n"
             "Return only a single JSON object with keys \"labels\", "
             "\"relations\", and optional \"facts\".\n"
             "Optimize for label precision.\n"
             "Labels should be the few memory-worthy nodes a long-term memory "
             "graph should keep: people, places, projects, organizations, "
             "tools, conditions, and durable topics directly supported by the "
             "text.\n"
             "Prefer named entities and concrete noun phrases.\n"
             "Only use a state or activity label if it is clearly enduring or "
             "central to the text.\n"
             "Do not output schema terms, field names, timestamps, generic "
             "verbs, helper words, or filler tokens.\n"
             "\"labels\" must be an array of non-empty strings copied from the "
             "text.\n"
             "Return between 0 and 6 labels; use [] when no durable label is "
             "present.\n"
             "Each relation must include non-empty \"subject\", \"predicate\", "
             "and \"object\" strings taken from the text.\n"
             "Each fact must include non-empty \"subject\", \"predicate\", and "
             "\"object\" strings taken from the text.\n"
             "If confidence is present, it must be a JSON number.\n"
             "If valid_start_ts or valid_end_ts are present for facts, they "
             "must be JSON integers in milliseconds.\n"
             "Examples:\n"
             "Text:\n"
             "User: My neighbor Sarah is helping me prepare documents for a "
             "housing application in Logan Square.\n"
             "Assistant: We can organize the paperwork.\n"
             "{\"labels\":[\"Sarah\",\"housing application\",\"Logan Square\"],"
             "\"relations\":[],\"facts\":[]}\n"
             "Text:\n"
             "User: The explanation and support were useful, but my goal is a "
             "better interview plan.\n"
             "Assistant: We'll focus on the interview plan.\n"
             "{\"labels\":[\"interview plan\"],\"relations\":[],\"facts\":[]}"
             "\n"
             "Do not emit any explanation outside the JSON object.";
    }

  return "Extract labels, relations, and durable facts from the provided "
         "text.\n"
         "Return only a single JSON object with keys \"labels\", "
         "\"relations\", and optional \"facts\".\n"
         "Prefer recall over compression: include all salient labels and "
         "relations that are directly supported by the text.\n"
         "\"labels\" must be an array of non-empty strings copied from the "
         "text.\n"
         "Each relation must include non-empty \"subject\", \"predicate\", "
         "and \"object\" strings taken from the text.\n"
         "Each fact must include non-empty \"subject\", \"predicate\", and "
         "\"object\" strings taken from the text.\n"
         "If confidence is present, it must be a JSON number.\n"
         "If valid_start_ts or valid_end_ts are present for facts, they must "
         "be JSON integers in milliseconds.\n"
         "Return an empty labels array when the text has no durable label.\n"
         "Do not emit any explanation outside the JSON object.";
}

std::string
BuildLabelOnlyExtractionSystemPrompt ()
{
  const std::string style
      = GetEnvOrDefault ("CORTEXT_LFM2_EXTRACT_PROMPT_STYLE",
                         "fewshot_durable");

  if (style == "durable")
    {
      return "Extract durable graph labels from the provided text.\n"
             "Return only a single JSON object with the key \"labels\".\n"
             "Labels are durable graph nodes, not every salient word.\n"
             "Prefer people, places, organizations, projects, tools, stable "
             "conditions, and enduring topics directly supported by the "
             "text.\n"
             "Prefer multi-word noun phrases when they are more specific.\n"
             "Do not output helper words, generic verbs, schema terms, JSON "
             "field names, timestamps, or meta labels.\n"
             "\"labels\" must be an array of non-empty strings copied from the "
             "text.\n"
             "Return between 0 and 8 labels; use [] when no durable label is "
             "present.\n"
             "Do not emit any explanation outside the JSON object.";
    }

  if (style == "precision")
    {
      return "Extract only the highest-value labels from the provided text.\n"
             "Return only a single JSON object with the key \"labels\".\n"
             "Optimize for label precision.\n"
             "Prefer named entities and concrete noun phrases that should be "
             "remembered later.\n"
             "Do not output schema terms, field names, generic verbs, helper "
             "words, or filler labels.\n"
             "\"labels\" must be an array of non-empty strings copied from the "
             "text.\n"
             "Return between 0 and 6 labels; use [] when no durable label is "
             "present.\n"
             "Do not emit any explanation outside the JSON object.";
    }

  if (style == "fewshot_durable")
    {
      return "Extract durable graph labels from the provided text.\n"
             "Return only a single JSON object with the key \"labels\".\n"
             "Labels are durable graph nodes, not every salient word.\n"
             "Prefer people, places, organizations, projects, tools, stable "
             "conditions, and enduring topics directly supported by the "
             "text.\n"
             "Prefer multi-word noun phrases when they are more specific.\n"
             "Do not output helper words, generic verbs, schema terms, JSON "
             "field names, timestamps, or meta labels.\n"
             "\"labels\" must be an array of non-empty strings copied from the "
             "text.\n"
             "Return between 0 and 8 labels; use [] when no durable label is "
             "present.\n"
             "Examples:\n"
             "Text:\n"
             "User: My name is Gabriel. I live in Chicago.\n"
             "Assistant: You are debugging a SQLite migration for Cortext.\n"
             "{\"labels\":[\"Gabriel\",\"Chicago\",\"SQLite migration\","
             "\"Cortext\"]}\n"
             "Text:\n"
             "User: I need help getting ready for an interview tomorrow. I "
             "want more confidence.\n"
             "Assistant: We can prep the interview.\n"
             "{\"labels\":[\"interview\",\"confidence\"]}\n"
             "Do not emit any explanation outside the JSON object.";
    }

  if (style == "fewshot_precision")
    {
      return "Extract only the highest-value labels from the provided text.\n"
             "Return only a single JSON object with the key \"labels\".\n"
             "Optimize for label precision.\n"
             "Prefer named entities and concrete noun phrases that should be "
             "remembered later.\n"
             "Do not output schema terms, field names, generic verbs, helper "
             "words, or filler labels.\n"
             "\"labels\" must be an array of non-empty strings copied from the "
             "text.\n"
             "Return between 0 and 6 labels; use [] when no durable label is "
             "present.\n"
             "Examples:\n"
             "Text:\n"
             "User: My neighbor Sarah is helping me prepare documents for a "
             "housing application in Logan Square.\n"
             "{\"labels\":[\"Sarah\",\"housing application\",\"Logan Square\"]}"
             "\n"
             "Text:\n"
             "User: The explanation and support were useful, but my goal is a "
             "better interview plan.\n"
             "{\"labels\":[\"interview plan\"]}\n"
             "Do not emit any explanation outside the JSON object.";
    }

  return "Extract labels from the provided text.\n"
         "Return only a single JSON object with the key \"labels\".\n"
         "Prefer recall over compression: include all salient labels that are "
         "directly supported by the text.\n"
         "\"labels\" must be an array of non-empty strings copied from the "
         "text.\n"
         "Return an empty labels array when the text has no durable label.\n"
         "Do not emit any explanation outside the JSON object.";
}

std::string
BuildExtractionUserPrompt (const std::string &text)
{
  return "Text:\n" + text;
}

std::string
FallbackChatTemplatePrompt (
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant)
{
  std::ostringstream prompt;
  prompt << "<|startoftext|>";
  for (const auto &message : messages)
    {
      prompt << "<|im_start|>" << message.first << "\n"
             << message.second << "<|im_end|>\n";
    }
  if (add_assistant)
    {
      prompt << "<|im_start|>assistant\n";
    }
  return prompt.str ();
}

std::optional<nlohmann::json>
TryParseJsonObject (const std::string &content)
{
  const auto start = content.find ('{');
  const auto end = content.rfind ('}');
  if (start == std::string::npos || end == std::string::npos || end <= start)
    {
      return std::nullopt;
    }
  try
    {
      return nlohmann::json::parse (content.substr (start, end - start + 1));
    }
  catch (const nlohmann::json::exception &)
    {
      return std::nullopt;
    }
}

operations::ExtractionResult
ParseExtractionResponse (const std::string &content)
{
  operations::ExtractionResult result;
  const auto json_opt = TryParseJsonObject (content);
  if (!json_opt)
    {
      return result;
    }

  const auto &json_output = *json_opt;
  if (json_output.contains ("labels"))
    {
      for (const auto &label : json_output["labels"])
        {
          if (label.is_string ())
            {
              const std::string value
                  = TrimAsciiWhitespace (label.get<std::string> ());
              if (!value.empty ())
                {
                  result.labels.push_back (
                      operations::ExtractedLabel{ value, 0.5 });
                }
            }
          else if (label.is_object ())
            {
              const std::string value = TrimAsciiWhitespace (
                  label.value ("label", label.value ("name", "")));
              if (!value.empty ())
                {
                  result.labels.push_back (
                      operations::ExtractedLabel{ value, 0.5 });
                }
            }
        }
    }

  if (json_output.contains ("relations"))
    {
      for (const auto &relation : json_output["relations"])
        {
          operations::ExtractedRelation r;
          r.subject = TrimAsciiWhitespace (relation.value ("subject", ""));
          r.predicate
              = TrimAsciiWhitespace (relation.value ("predicate", ""));
          r.object = TrimAsciiWhitespace (relation.value ("object", ""));
          r.confidence = relation.value ("confidence", 0.5);
          if (!r.subject.empty () && !r.predicate.empty () && !r.object.empty ())
            {
              result.relations.push_back (std::move (r));
            }
        }
    }

  if (json_output.contains ("facts"))
    {
      for (const auto &fact : json_output["facts"])
        {
          operations::ExtractedFact f;
          f.subject = TrimAsciiWhitespace (fact.value ("subject", ""));
          f.predicate = TrimAsciiWhitespace (fact.value ("predicate", ""));
          f.object = TrimAsciiWhitespace (fact.value ("object", ""));
          f.confidence = fact.value ("confidence", 0.5);
          if (fact.contains ("valid_start_ts") && fact["valid_start_ts"].is_number ())
            {
              f.valid_start_ts = fact["valid_start_ts"].get<std::uint64_t> ();
            }
          if (fact.contains ("valid_end_ts") && fact["valid_end_ts"].is_number ())
            {
              f.valid_end_ts = fact["valid_end_ts"].get<std::uint64_t> ();
            }
          if (!f.subject.empty () && !f.predicate.empty () && !f.object.empty ())
            {
              result.facts.push_back (std::move (f));
            }
        }
    }

  return result;
}

std::string
ParseSummaryResponse (const std::string &content)
{
  const auto json_opt = TryParseJsonObject (content);
  if (!json_opt)
    {
      return {};
    }

  const auto &json_output = *json_opt;
  if (!json_output.contains ("summary") || !json_output["summary"].is_string ())
    {
      return {};
    }

  return TrimAsciiWhitespace (json_output["summary"].get<std::string> ());
}

bool
HasNonEmptyLabel (const operations::ExtractionResult &result)
{
  return std::any_of (result.labels.begin (), result.labels.end (),
                      [] (const auto &label) {
                        return !label.label.empty ();
                      });
}

bool
SchemaAllowsEmptyLabels (const nlohmann::json &schema)
{
  if (!schema.contains ("properties") || !schema["properties"].is_object ())
    {
      return false;
    }
  const auto &properties = schema["properties"];
  if (!properties.contains ("labels") || !properties["labels"].is_object ())
    {
      return false;
    }
  return properties["labels"].value ("minItems", 0) == 0;
}

bool
SchemaRequiresRelations (const nlohmann::json &schema)
{
  if (!schema.contains ("required") || !schema["required"].is_array ())
    {
      return false;
    }

  for (const auto &required : schema["required"])
    {
      if (required.is_string () && required.get<std::string> () == "relations")
        {
          return true;
        }
    }
  return false;
}

nlohmann::json
BuildLabelsOnlySchema (const nlohmann::json &schema)
{
  nlohmann::json labels_only;
  labels_only["type"] = "object";
  labels_only["properties"] = nlohmann::json::object ();
  labels_only["properties"]["labels"] = schema.at ("properties").at ("labels");
  labels_only["required"] = nlohmann::json::array ({ "labels" });
  return labels_only;
}

void
RequireBoolean (bool condition, const std::string &message)
{
  if (!condition)
    {
      throw std::runtime_error (message);
    }
}

void
ValidateSchemaSubset (const nlohmann::json &schema)
{
  RequireBoolean (schema.is_object (),
                  "LFM2 extractor schema must be a JSON object");
  RequireBoolean (schema.value ("type", "") == "object",
                  "LFM2 extractor schema root must have type=object");
  RequireBoolean (schema.contains ("properties")
                      && schema["properties"].is_object (),
                  "LFM2 extractor schema must define object properties");

  const auto &properties = schema["properties"];
  RequireBoolean (properties.contains ("labels"),
                  "LFM2 extractor schema must define labels");
  const auto &labels = properties["labels"];
  RequireBoolean (labels.is_object () && labels.value ("type", "") == "array",
                  "LFM2 extractor schema labels must be an array");
  RequireBoolean (labels.contains ("items") && labels["items"].is_object ()
                      && labels["items"].value ("type", "") == "string",
                  "LFM2 extractor schema labels items must be strings");

  const int min_items = labels.value ("minItems", 0);
  RequireBoolean (min_items == 0 || min_items == 1,
                  "LFM2 extractor schema only supports labels minItems 0 or 1");

  if (properties.contains ("relations"))
    {
      const auto &relations = properties["relations"];
      RequireBoolean (relations.is_object ()
                          && relations.value ("type", "") == "array",
                      "LFM2 extractor schema relations must be an array");
      RequireBoolean (relations.contains ("items")
                          && relations["items"].is_object (),
                      "LFM2 extractor schema relations items must be objects");
      const auto &items = relations["items"];
      RequireBoolean (items.value ("type", "") == "object",
                      "LFM2 extractor schema relation items must have type=object");
      RequireBoolean (items.contains ("properties")
                          && items["properties"].is_object (),
                      "LFM2 extractor schema relation items must define properties");
      const auto &relation_props = items["properties"];
      for (const char *key : { "subject", "predicate", "object" })
        {
          RequireBoolean (
              relation_props.contains (key)
                  && relation_props[key].is_object ()
                  && relation_props[key].value ("type", "") == "string",
              std::string ("LFM2 extractor schema relation property ") + key
                  + " must be a string");
        }
      if (relation_props.contains ("confidence"))
        {
          RequireBoolean (relation_props["confidence"].is_object (),
                          "LFM2 extractor schema confidence must be an object");
          const std::string confidence_type
              = relation_props["confidence"].value ("type", "");
          RequireBoolean (confidence_type == "number"
                              || confidence_type == "integer",
                          "LFM2 extractor schema confidence must be numeric");
        }
    }

  if (properties.contains ("facts"))
    {
      const auto &facts = properties["facts"];
      RequireBoolean (facts.is_object () && facts.value ("type", "") == "array",
                      "LFM2 extractor schema facts must be an array");
      RequireBoolean (facts.contains ("items") && facts["items"].is_object (),
                      "LFM2 extractor schema facts items must be objects");
      const auto &items = facts["items"];
      RequireBoolean (items.value ("type", "") == "object",
                      "LFM2 extractor schema fact items must have type=object");
      RequireBoolean (items.contains ("properties")
                          && items["properties"].is_object (),
                      "LFM2 extractor schema fact items must define properties");
      const auto &fact_props = items["properties"];
      for (const char *key : { "subject", "predicate", "object" })
        {
          RequireBoolean (
              fact_props.contains (key)
                  && fact_props[key].is_object ()
                  && fact_props[key].value ("type", "") == "string",
              std::string ("LFM2 extractor schema fact property ") + key
                  + " must be a string");
        }
      if (fact_props.contains ("confidence"))
        {
          RequireBoolean (fact_props["confidence"].is_object (),
                          "LFM2 extractor schema fact confidence must be an object");
          const std::string confidence_type
              = fact_props["confidence"].value ("type", "");
          RequireBoolean (confidence_type == "number"
                              || confidence_type == "integer",
                          "LFM2 extractor schema fact confidence must be numeric");
        }
      for (const char *key : { "valid_start_ts", "valid_end_ts" })
        {
          if (fact_props.contains (key))
            {
              RequireBoolean (fact_props[key].is_object (),
                              std::string ("LFM2 extractor schema ") + key
                                  + " must be an object");
              const std::string ts_type = fact_props[key].value ("type", "");
              RequireBoolean (ts_type == "integer" || ts_type == "number",
                              std::string ("LFM2 extractor schema ") + key
                                  + " must be numeric");
            }
        }
    }

  if (schema.contains ("required"))
    {
      RequireBoolean (schema["required"].is_array (),
                      "LFM2 extractor schema required must be an array");
      for (const auto &required : schema["required"])
        {
          RequireBoolean (required.is_string (),
                          "LFM2 extractor schema required entries must be strings");
          const std::string key = required.get<std::string> ();
          RequireBoolean (key == "labels" || key == "relations"
                              || key == "facts",
                          "LFM2 extractor schema only supports labels/relations/facts in required");
        }
      bool labels_required = false;
      for (const auto &required : schema["required"])
        {
          if (required.get<std::string> () == "labels")
            {
              labels_required = true;
              break;
            }
        }
      RequireBoolean (labels_required,
                      "LFM2 extractor schema must require labels");
    }
}

std::string
BuildStringGrammar ()
{
  return R"(char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
nonblank-char ::= [^"\\\x7F\x00-\x1F \t\r\n] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
string ::= "\"" char* "\"" space
nonempty-string ::= "\"" nonblank-char char* "\"" space
)";
}

std::string
BuildNumberGrammar ()
{
  return R"(decimal-part ::= [0-9]{1,16}
integral-part ::= [0] | [1-9] [0-9]{0,15}
number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
)";
}

} // namespace

namespace internal
{

std::string
BuildLfm2SummaryGrammar ()
{
  std::ostringstream grammar;
  grammar << "root ::= \"{\" space summary-kv \"}\" space\n";
  grammar << "summary-kv ::= \"\\\"summary\\\"\" space \":\" space nonempty-string\n";
  grammar << BuildStringGrammar ();
  grammar << "space ::= | \" \" | \"\\n\"{1,2} [ \\t]{0,20}\n";
  return grammar.str ();
}

std::string
BuildLfm2ExtractionGrammar (const nlohmann::json &schema)
{
  ValidateSchemaSubset (schema);

  const auto &properties = schema["properties"];
  const auto &labels = properties["labels"];
  const bool labels_nonempty = labels.value ("minItems", 0) > 0;
  const bool has_relations = properties.contains ("relations");
  const bool has_facts = properties.contains ("facts");
  bool relations_required = false;
  bool facts_required = false;
  bool confidence_supported = false;
  bool fact_confidence_supported = false;
  bool valid_start_supported = false;
  bool valid_end_supported = false;
  if (schema.contains ("required"))
    {
      for (const auto &required : schema["required"])
        {
          if (required.is_string () && required.get<std::string> () == "relations")
            {
              relations_required = true;
            }
          if (required.is_string () && required.get<std::string> () == "facts")
            {
              facts_required = true;
            }
        }
    }
  if (has_relations)
    {
      const auto &relation_props = properties["relations"]["items"]["properties"];
      confidence_supported = relation_props.contains ("confidence");
    }
  if (has_facts)
    {
      const auto &fact_props = properties["facts"]["items"]["properties"];
      fact_confidence_supported = fact_props.contains ("confidence");
      valid_start_supported = fact_props.contains ("valid_start_ts");
      valid_end_supported = fact_props.contains ("valid_end_ts");
    }

  std::ostringstream grammar;
  if (has_relations || has_facts)
    {
      grammar << "root ::= \"{\" space labels-kv";
      if (has_relations && has_facts)
        {
          if (relations_required)
            {
              grammar << " \",\" space relations-kv";
              if (facts_required)
                {
                  grammar << " \",\" space facts-kv";
                }
              else
                {
                  grammar << " ( \",\" space facts-kv )?";
                }
            }
          else if (facts_required)
            {
              grammar << " \",\" space facts-kv";
            }
          else
            {
              grammar << " ( \",\" space ( relations-kv ( \",\" space facts-kv )? | facts-kv ) )?";
            }
        }
      else if (has_relations)
        {
          if (relations_required)
            {
              grammar << " \",\" space relations-kv";
            }
          else
            {
              grammar << " ( \",\" space ( relations-kv ) )?";
            }
        }
      else
        {
          if (facts_required)
            {
              grammar << " \",\" space facts-kv";
            }
          else
            {
              grammar << " ( \",\" space ( facts-kv ) )?";
            }
        }
      grammar << " \"}\" space\n";
    }
  else
    {
      grammar << "root ::= \"{\" space labels-kv \"}\" space\n";
    }
  grammar << "labels ::= \"[\" space ";
  if (labels_nonempty)
    {
      grammar << "nonempty-string (\",\" space nonempty-string)*";
    }
  else
    {
      grammar << "(nonempty-string (\",\" space nonempty-string)*)?";
    }
  grammar << " \"]\" space\n";
  grammar << "labels-kv ::= \"\\\"labels\\\"\" space \":\" space labels\n";
  if (has_relations)
    {
      grammar << "relations ::= \"[\" space (relations-item (\",\" space "
                 "relations-item)*)? \"]\" space\n";
      grammar << "relations-item ::= \"{\" space relations-item-subject-kv "
                 "\",\" space relations-item-predicate-kv \",\" space "
                 "relations-item-object-kv";
      if (confidence_supported)
        {
          grammar << " ( \",\" space ( relations-item-confidence-kv ) )?";
        }
      grammar << " \"}\" space\n";
      grammar << "relations-item-subject-kv ::= \"\\\"subject\\\"\" space "
                 "\":\" space nonempty-string\n";
      grammar << "relations-item-predicate-kv ::= \"\\\"predicate\\\"\" space "
                 "\":\" space nonempty-string\n";
      grammar << "relations-item-object-kv ::= \"\\\"object\\\"\" space "
                 "\":\" space nonempty-string\n";
      if (confidence_supported)
        {
          grammar
              << "relations-item-confidence-kv ::= \"\\\"confidence\\\"\" "
                 "space \":\" space number\n";
        }
      grammar << "relations-kv ::= \"\\\"relations\\\"\" space \":\" space "
                 "relations\n";
    }
  if (has_facts)
    {
      grammar << "facts ::= \"[\" space (facts-item (\",\" space facts-item)*)? \"]\" space\n";
      grammar << "facts-item ::= \"{\" space facts-item-subject-kv "
                 "\",\" space facts-item-predicate-kv \",\" space "
                 "facts-item-object-kv";
      if (fact_confidence_supported)
        {
          grammar << " ( \",\" space ( facts-item-confidence-kv ) )?";
        }
      if (valid_start_supported)
        {
          grammar << " ( \",\" space ( facts-item-valid-start-kv ) )?";
        }
      if (valid_end_supported)
        {
          grammar << " ( \",\" space ( facts-item-valid-end-kv ) )?";
        }
      grammar << " \"}\" space\n";
      grammar << "facts-item-subject-kv ::= \"\\\"subject\\\"\" space "
                 "\":\" space nonempty-string\n";
      grammar << "facts-item-predicate-kv ::= \"\\\"predicate\\\"\" space "
                 "\":\" space nonempty-string\n";
      grammar << "facts-item-object-kv ::= \"\\\"object\\\"\" space "
                 "\":\" space nonempty-string\n";
      if (fact_confidence_supported)
        {
          grammar << "facts-item-confidence-kv ::= \"\\\"confidence\\\"\" "
                     "space \":\" space number\n";
        }
      if (valid_start_supported)
        {
          grammar << "facts-item-valid-start-kv ::= \"\\\"valid_start_ts\\\"\" "
                     "space \":\" space number\n";
        }
      if (valid_end_supported)
        {
          grammar << "facts-item-valid-end-kv ::= \"\\\"valid_end_ts\\\"\" "
                     "space \":\" space number\n";
        }
      grammar << "facts-kv ::= \"\\\"facts\\\"\" space \":\" space facts\n";
    }
  grammar << BuildStringGrammar ();
  grammar << BuildNumberGrammar ();
  grammar << "space ::= | \" \" | \"\\n\"{1,2} [ \\t]{0,20}\n";
  return grammar.str ();
}

} // namespace internal

namespace
{

struct GenerationConfig
{
  int max_tokens = 256;
  float temperature = 0.0f;
  float min_p = 0.0f;
  float repetition_penalty = 1.0f;
  bool greedy = true;
  std::string grammar;
};

llama_sampler *
BuildGenerationSampler (const llama_vocab *vocab, const GenerationConfig &config)
{
#if !defined(CORTEXT_ENABLE_LLAMA_CPP)
  (void)vocab;
  (void)config;
  return nullptr;
#else
  llama_sampler_chain_params params = llama_sampler_chain_default_params ();
  llama_sampler *chain = llama_sampler_chain_init (params);
  if (chain == nullptr)
    {
      return nullptr;
    }

  if (!config.grammar.empty ())
    {
      llama_sampler *grammar
          = llama_sampler_init_grammar (vocab, config.grammar.c_str (), "root");
      if (grammar == nullptr)
        {
          llama_sampler_free (chain);
          throw std::runtime_error ("failed to initialize llama.cpp grammar sampler");
        }
      llama_sampler_chain_add (chain, grammar);
    }

  if (config.repetition_penalty > 1.0f)
    {
      llama_sampler_chain_add (
          chain, llama_sampler_init_penalties (64, config.repetition_penalty,
                                               0.0f, 0.0f));
    }
  if (config.min_p > 0.0f)
    {
      llama_sampler_chain_add (chain,
                               llama_sampler_init_min_p (config.min_p, 1));
    }

  if (config.greedy || config.temperature <= 0.0f)
    {
      llama_sampler_chain_add (chain, llama_sampler_init_greedy ());
    }
  else
    {
      llama_sampler_chain_add (chain,
                               llama_sampler_init_temp (config.temperature));
      llama_sampler_chain_add (chain, llama_sampler_init_dist (1234));
    }

  return chain;
#endif
}

class LlamaCppTextModel
{
public:
  explicit LlamaCppTextModel (std::string component_name,
                              const std::string &model_path)
      : component_name_ (std::move (component_name))
  {
    try
      {
        Initialize (model_path);
        available_ = true;
      }
    catch (const std::exception &e)
      {
        available_ = false;
        init_error_ = e.what ();
      }
  }

  ~LlamaCppTextModel ()
  {
#if defined(CORTEXT_ENABLE_LLAMA_CPP)
    if (ctx_ != nullptr)
      {
        llama_free (ctx_);
      }
    if (model_ != nullptr)
      {
        llama_model_free (model_);
      }
#endif
  }

  LlamaCppTextModel (const LlamaCppTextModel &) = delete;
  LlamaCppTextModel &operator= (const LlamaCppTextModel &) = delete;
  LlamaCppTextModel (LlamaCppTextModel &&) = delete;
  LlamaCppTextModel &operator= (LlamaCppTextModel &&) = delete;

  bool
  IsAvailable () const
  {
    return available_;
  }

  const std::string &
  InitializationError () const
  {
    return init_error_;
  }

  std::string
  Generate (const std::vector<std::pair<std::string, std::string>> &messages,
            const GenerationConfig &config)
  {
    if (!available_)
      {
        throw std::runtime_error (component_name_ + ": " + init_error_);
      }

    std::lock_guard<std::mutex> lock (mu_);
#if !defined(CORTEXT_ENABLE_LLAMA_CPP)
    (void)messages;
    (void)config;
    throw std::runtime_error (component_name_
                              + ": llama.cpp backend unavailable: link libllama");
#else
    auto *memory = llama_get_memory (ctx_);
    if (memory == nullptr)
      {
        throw std::runtime_error (component_name_
                                  + ": llama.cpp context has no memory");
      }
    internal::ThrowIfStopRequested ();
    llama_memory_clear (memory, false);

    const std::string prompt = ApplyChatTemplate (messages);
    std::vector<llama_token> prompt_tokens = TokenizePrompt (prompt);
    if (prompt_tokens.empty ())
      {
        throw std::runtime_error (component_name_
                                  + ": prompt tokenization returned no tokens");
      }
    if (static_cast<int> (prompt_tokens.size ()) >= max_context_tokens_)
      {
        throw std::runtime_error (component_name_
                                  + ": prompt exceeds llama.cpp context window");
      }

    llama_batch batch = llama_batch_init (
        static_cast<int32_t> (prompt_tokens.size ()), 0, 1);
    if (batch.token == nullptr || batch.pos == nullptr
        || batch.n_seq_id == nullptr || batch.seq_id == nullptr
        || batch.logits == nullptr)
      {
        throw std::runtime_error (
            component_name_ + ": failed to allocate llama.cpp prompt batch");
      }

    batch.n_tokens = static_cast<int32_t> (prompt_tokens.size ());
    for (int32_t i = 0; i < batch.n_tokens; ++i)
      {
        batch.token[i] = prompt_tokens[static_cast<size_t> (i)];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == batch.n_tokens - 1) ? 1 : 0;
      }

    const int prompt_rc = llama_decode (ctx_, batch);
    llama_batch_free (batch);
    if (prompt_rc != 0)
      {
        throw std::runtime_error (component_name_
                                  + ": llama.cpp prompt decode failed: "
                                  + std::to_string (prompt_rc));
      }
    internal::ThrowIfStopRequested ();

    std::unique_ptr<llama_sampler, void (*) (llama_sampler *)> sampler (
        BuildGenerationSampler (vocab_, config), llama_sampler_free);
    if (!sampler)
      {
        throw std::runtime_error (component_name_
                                  + ": failed to create llama.cpp sampler");
      }

    std::vector<llama_token> output_tokens;
    output_tokens.reserve (static_cast<size_t> (config.max_tokens));

    for (int produced = 0; produced < config.max_tokens; ++produced)
      {
        internal::ThrowIfStopRequested ();
        const llama_token token = llama_sampler_sample (sampler.get (), ctx_, -1);
        if (token == LLAMA_TOKEN_NULL || llama_vocab_is_eog (vocab_, token))
          {
            break;
          }

        output_tokens.push_back (token);

        llama_batch next = llama_batch_get_one (&output_tokens.back (), 1);
        const int decode_rc = llama_decode (ctx_, next);
        if (decode_rc != 0)
          {
            throw std::runtime_error (component_name_
                                      + ": llama.cpp token decode failed: "
                                      + std::to_string (decode_rc));
          }
        internal::ThrowIfStopRequested ();
      }

    return Detokenize (output_tokens);
#endif
  }

private:
  void
  Initialize (const std::string &model_path)
  {
    const std::string resolved
        = ResolveExistingPath (model_path, component_name_.c_str ());
#if !defined(CORTEXT_ENABLE_LLAMA_CPP)
    (void)resolved;
    throw std::runtime_error ("llama.cpp backend unavailable: link libllama");
#else
	    static std::once_flag backend_once;
	    std::call_once (backend_once, [] () {
	      internal::InstallLlamaCppLogFilter ();
	      internal::LoadGgmlBackendsOnce ();
	      llama_backend_init ();
	    });

    llama_model_params model_params = llama_model_default_params ();
    model_params.n_gpu_layers = 0;
    model_params.use_mmap = true;
    model_params.use_mlock = false;

    model_ = llama_model_load_from_file (resolved.c_str (), model_params);
    if (model_ == nullptr)
      {
        throw std::runtime_error ("failed to load model " + resolved);
      }

    vocab_ = llama_model_get_vocab (model_);
    if (vocab_ == nullptr)
      {
        throw std::runtime_error ("loaded model has no vocabulary");
      }

    llama_context_params context_params = llama_context_default_params ();
    max_context_tokens_
        = std::max (1024, GetEnvInt ("CORTEXT_LLAMA_CPP_N_CTX", 8192));
    context_params.n_ctx = static_cast<uint32_t> (max_context_tokens_);
    context_params.n_batch = static_cast<uint32_t> (max_context_tokens_);
    context_params.n_ubatch = static_cast<uint32_t> (
        std::max (512, GetEnvInt ("CORTEXT_LLAMA_CPP_UBATCH", 2048)));
    context_params.n_seq_max = 1;
    context_params.n_threads = static_cast<uint32_t> (
        std::max (1, static_cast<int> (core::GetInferThreadCount ())));
    context_params.n_threads_batch = context_params.n_threads;
    context_params.embeddings = false;
    context_params.offload_kqv = false;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    context_params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;

    ctx_ = llama_init_from_model (model_, context_params);
    if (ctx_ == nullptr)
      {
        throw std::runtime_error ("failed to create llama.cpp context");
      }
    llama_set_warmup (ctx_, false);
#endif
  }

  std::string
  ApplyChatTemplate (
      const std::vector<std::pair<std::string, std::string>> &messages) const
  {
#if !defined(CORTEXT_ENABLE_LLAMA_CPP)
    (void)messages;
    return {};
#else
    std::vector<llama_chat_message> chat_messages;
    chat_messages.reserve (messages.size ());
    for (const auto &message : messages)
      {
        chat_messages.push_back (
            llama_chat_message{ message.first.c_str (), message.second.c_str () });
      }

    const char *tmpl = llama_model_chat_template (model_, nullptr);
    if (tmpl == nullptr)
      {
        return FallbackChatTemplatePrompt (messages, true);
      }

    size_t estimated_size = 256;
    for (const auto &message : messages)
      {
        estimated_size += message.first.size () + message.second.size () + 32;
      }

    std::vector<char> buffer (estimated_size);
    int32_t actual = llama_chat_apply_template (
        tmpl, chat_messages.data (), chat_messages.size (), true, buffer.data (),
        static_cast<int32_t> (buffer.size ()));
    if (actual < 0)
      {
        throw std::runtime_error (component_name_
                                  + ": llama.cpp chat template application failed");
      }
    if (actual >= static_cast<int32_t> (buffer.size ()))
      {
        buffer.resize (static_cast<size_t> (actual) + 1);
        actual = llama_chat_apply_template (
            tmpl, chat_messages.data (), chat_messages.size (), true,
            buffer.data (), static_cast<int32_t> (buffer.size ()));
        if (actual < 0)
          {
            throw std::runtime_error (
                component_name_ + ": llama.cpp chat template retry failed");
          }
      }
    return std::string (buffer.data (), static_cast<size_t> (actual));
#endif
  }

  std::vector<llama_token>
  TokenizePrompt (const std::string &prompt) const
  {
#if !defined(CORTEXT_ENABLE_LLAMA_CPP)
    (void)prompt;
    return {};
#else
    const int32_t needed
        = llama_tokenize (vocab_, prompt.c_str (),
                          static_cast<int32_t> (prompt.size ()), nullptr, 0,
                          false, true);
    if (needed == INT32_MIN)
      {
        throw std::runtime_error (component_name_
                                  + ": llama.cpp tokenization overflow");
      }
    const int32_t token_count = needed < 0 ? -needed : needed;
    std::vector<llama_token> tokens (static_cast<size_t> (token_count));
    const int32_t actual = llama_tokenize (
        vocab_, prompt.c_str (), static_cast<int32_t> (prompt.size ()),
        tokens.data (), token_count, false, true);
    if (actual < 0)
      {
        throw std::runtime_error (component_name_
                                  + ": llama.cpp tokenization failed");
      }
    tokens.resize (static_cast<size_t> (actual));
    return tokens;
#endif
  }

  std::string
  Detokenize (const std::vector<llama_token> &tokens) const
  {
#if !defined(CORTEXT_ENABLE_LLAMA_CPP)
    (void)tokens;
    return {};
#else
    if (tokens.empty ())
      {
        return {};
      }
    std::vector<char> buffer (tokens.size () * 16 + 32);
    int32_t actual = llama_detokenize (vocab_, tokens.data (),
                                       static_cast<int32_t> (tokens.size ()),
                                       buffer.data (),
                                       static_cast<int32_t> (buffer.size ()),
                                       true, false);
    if (actual < 0)
      {
        buffer.resize (static_cast<size_t> (-actual) + 1);
        actual = llama_detokenize (
            vocab_, tokens.data (), static_cast<int32_t> (tokens.size ()),
            buffer.data (), static_cast<int32_t> (buffer.size ()), true, false);
      }
    if (actual < 0)
      {
        throw std::runtime_error (component_name_
                                  + ": llama.cpp detokenization failed");
      }
    return std::string (buffer.data (), static_cast<size_t> (actual));
#endif
  }

  std::string component_name_;
  mutable std::mutex mu_;
  bool available_ = false;
  std::string init_error_;
  int max_context_tokens_ = 8192;
#if defined(CORTEXT_ENABLE_LLAMA_CPP)
  llama_model *model_ = nullptr;
  llama_context *ctx_ = nullptr;
  const llama_vocab *vocab_ = nullptr;
#endif
};

} // namespace

struct Lfm2LlamaSummarizer::Impl
{
  explicit Impl (const std::string &model_path)
      : model ("Lfm2LlamaSummarizer", model_path)
  {
  }

  LlamaCppTextModel model;
};

Lfm2LlamaSummarizer::Lfm2LlamaSummarizer (const std::string &model_path)
    : impl_ (std::make_unique<Impl> (model_path))
{
}

Lfm2LlamaSummarizer::~Lfm2LlamaSummarizer () = default;
Lfm2LlamaSummarizer::Lfm2LlamaSummarizer (Lfm2LlamaSummarizer &&) noexcept
    = default;
Lfm2LlamaSummarizer &
Lfm2LlamaSummarizer::operator= (Lfm2LlamaSummarizer &&) noexcept = default;

std::string
Lfm2LlamaSummarizer::SummarizeTexts (const std::vector<std::string> &texts)
{
  return SummarizeTextsLimited (texts, 0);
}

std::string
Lfm2LlamaSummarizer::SummarizeTextsLimited (
    const std::vector<std::string> &texts, int max_words)
{
  if (texts.empty ())
    {
      return {};
    }

  GenerationConfig config;
  config.max_tokens = ComputeSummaryMaxTokens (texts, max_words);
  config.temperature = 0.3f;
  config.min_p = 0.15f;
  config.repetition_penalty = 1.05f;
  config.greedy = false;
  if (UseStructuredSummaryOutput ())
    {
      config.grammar = internal::BuildLfm2SummaryGrammar ();
      config.greedy = true;
      config.temperature = 0.0f;
      config.min_p = 0.0f;
    }

  std::vector<std::pair<std::string, std::string>> messages;
  messages.emplace_back ("system", BuildSummarySystemPrompt ());
  messages.emplace_back ("user", BuildSummaryUserPrompt (texts));
  const std::string response = impl_->model.Generate (messages, config);

  std::string summary;
  if (UseStructuredSummaryOutput ())
    {
      summary = ParseSummaryResponse (response);
      if (summary.empty ())
        {
          GenerationConfig fallback_config = config;
          fallback_config.grammar.clear ();
          summary = SanitizeSummaryText (
              impl_->model.Generate (messages, fallback_config));
        }
    }
  else
    {
      summary = SanitizeSummaryText (response);
    }

  return TrimToWordLimit (SanitizeSummaryText (summary), max_words);
}

std::string
Lfm2LlamaSummarizer::SummarizeAudio (const float * /*pcm*/,
                                     size_t /*num_samples*/)
{
  throw std::runtime_error (
      "Lfm2LlamaSummarizer: audio summarization is unsupported for transcript-only GGUF models");
}

std::string
Lfm2LlamaSummarizer::SummarizeAudioSegments (
    const std::vector<AudioSegment> & /*segments*/)
{
  throw std::runtime_error (
      "Lfm2LlamaSummarizer: audio summarization is unsupported for transcript-only GGUF models");
}

bool
Lfm2LlamaSummarizer::IsAvailable () const
{
  return impl_ && impl_->model.IsAvailable ();
}

struct Lfm2LlamaExtractor::Impl
{
  explicit Impl (const std::string &model_path)
      : model ("Lfm2LlamaExtractor", model_path)
  {
  }

  LlamaCppTextModel model;
};

Lfm2LlamaExtractor::Lfm2LlamaExtractor (const std::string &model_path)
    : impl_ (std::make_unique<Impl> (model_path))
{
}

Lfm2LlamaExtractor::~Lfm2LlamaExtractor () = default;
Lfm2LlamaExtractor::Lfm2LlamaExtractor (Lfm2LlamaExtractor &&) noexcept
    = default;
Lfm2LlamaExtractor &
Lfm2LlamaExtractor::operator= (Lfm2LlamaExtractor &&) noexcept = default;

operations::ExtractionResult
Lfm2LlamaExtractor::ExtractFromText (const std::string &text,
                                     const nlohmann::json &schema)
{
  if (text.empty ())
    {
      return {};
    }

  std::vector<std::pair<std::string, std::string>> messages;
  messages.emplace_back ("system", BuildExtractionSystemPrompt ());
  messages.emplace_back ("user", BuildExtractionUserPrompt (text));

  GenerationConfig config;
  config.max_tokens = 1024;
  config.greedy = true;
  config.grammar = internal::BuildLfm2ExtractionGrammar (schema);

  const std::string response = impl_->model.Generate (messages, config);
  const auto parsed = ParseExtractionResponse (response);
  if (!HasNonEmptyLabel (parsed))
    {
      if (SchemaAllowsEmptyLabels (schema))
        {
          return parsed;
        }
      if (!SchemaRequiresRelations (schema))
        {
          GenerationConfig fallback_config;
          fallback_config.max_tokens = 256;
          fallback_config.greedy = true;
          fallback_config.grammar = internal::BuildLfm2ExtractionGrammar (
              BuildLabelsOnlySchema (schema));

          std::vector<std::pair<std::string, std::string>> fallback_messages;
          fallback_messages.emplace_back ("system",
                                          BuildLabelOnlyExtractionSystemPrompt ());
          fallback_messages.emplace_back ("user",
                                          BuildExtractionUserPrompt (text));

          const auto fallback = ParseExtractionResponse (
              impl_->model.Generate (fallback_messages, fallback_config));
          if (HasNonEmptyLabel (fallback))
            {
              return fallback;
            }
        }

      throw std::runtime_error (
          "Lfm2LlamaExtractor: failed to produce valid constrained labels");
    }
  return parsed;
}

operations::ExtractionResult
Lfm2LlamaExtractor::ExtractFromAudio (const float * /*pcm*/,
                                      size_t /*num_samples*/,
                                      const nlohmann::json & /*schema*/)
{
  throw std::runtime_error (
      "Lfm2LlamaExtractor: audio extraction is unsupported for text-only GGUF models");
}

bool
Lfm2LlamaExtractor::IsAvailable () const
{
  return impl_ && impl_->model.IsAvailable ();
}

} // namespace cortext
