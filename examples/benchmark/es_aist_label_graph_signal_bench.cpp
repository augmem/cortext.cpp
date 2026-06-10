#include <cortext/models/aist_gguf_encoder.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using Json = nlohmann::json;

struct Options
{
  std::filesystem::path source_records
      = "build/label_centroid_sources/label_centroid_source_records.jsonl";
  std::filesystem::path assets_dir = "build/real_multimodal_episode_assets";
  std::filesystem::path output_dir
      = "build/es_aist_label_graph_signal_bench";
  std::filesystem::path models_dir = "models";
  std::filesystem::path model_path;
  int max_labels = 0;
  int top_k = 50;
  int label_prompt_count = 4;
  std::string label_text_mode = "label";
  std::string label_match_mode = "label";
};

struct LabelRecord
{
  int index = 0;
  std::string label_id;
  std::string source;
  std::string kind;
  std::string label;
  std::string definition;
  std::string encoded_text;
  std::string match_text;
  std::vector<std::string> aliases;
  std::vector<std::string> prompts;
  std::vector<float> embedding;
  std::map<std::string, std::vector<float>> views;
};

struct SignalRecord
{
  std::string scenario;
  std::string signal_id;
  std::string modality;
  std::string offline_group;
  std::vector<float> embedding;
  std::map<std::string, std::vector<float>> views;
};

struct Candidate
{
  int label_index = 0;
  double score = 0.0;
};

struct EdgeRow
{
  std::string scenario;
  std::string view;
  std::string variant;
  std::string source_signal_id;
  std::string target_signal_id;
  std::string source_group;
  std::string target_group;
  double weight = 0.0;
  bool same_group = false;
  bool cross_group = false;
  std::string detail;
};

std::string
CsvEscape (const std::string &s)
{
  if (s.find_first_of (",\"\n\r") == std::string::npos)
    {
      return s;
    }
  std::string out = "\"";
  for (char c : s)
    {
      if (c == '"')
        {
          out += "\"\"";
        }
      else
        {
          out += c;
        }
    }
  out += '"';
  return out;
}

double
Millis (std::chrono::steady_clock::duration duration)
{
  return std::chrono::duration<double, std::milli> (duration).count ();
}

void
Normalize (std::vector<float> &values)
{
  double norm_sq = 0.0;
  for (float value : values)
    {
      norm_sq += static_cast<double> (value) * static_cast<double> (value);
    }
  if (norm_sq <= 0.0)
    {
      return;
    }
  const auto inv = static_cast<float> (1.0 / std::sqrt (norm_sq));
  for (float &value : values)
    {
      value *= inv;
    }
}

double
Dot (const std::vector<float> &a, const std::vector<float> &b)
{
  const std::size_t n = std::min (a.size (), b.size ());
  double out = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      out += static_cast<double> (a[i]) * static_cast<double> (b[i]);
    }
  return out;
}

std::vector<float>
Slice (const std::vector<float> &values, std::size_t begin, std::size_t end)
{
  if (begin >= values.size ())
    {
      return {};
    }
  end = std::min (end, values.size ());
  std::vector<float> out (values.begin () + static_cast<std::ptrdiff_t> (begin),
                          values.begin () + static_cast<std::ptrdiff_t> (end));
  Normalize (out);
  return out;
}

std::map<std::string, std::vector<float>>
ViewsFor (const std::vector<float> &embedding)
{
  std::map<std::string, std::vector<float>> views;
  views["prefix256"] = Slice (embedding, 0, 256);
  views["semantic768"] = Slice (embedding, 0, 768);
  views["entity768"] = Slice (embedding, 768, 1536);
  std::vector<float> full = embedding;
  Normalize (full);
  views["full1536"] = full;
  return views;
}

std::vector<unsigned char>
ReadBytes (const std::filesystem::path &path)
{
  std::ifstream in (path, std::ios::binary | std::ios::ate);
  if (!in)
    {
      throw std::runtime_error ("Failed to open " + path.string ());
    }
  const auto size = in.tellg ();
  in.seekg (0, std::ios::beg);
  std::vector<unsigned char> out (static_cast<std::size_t> (size));
  in.read (reinterpret_cast<char *> (out.data ()),
           static_cast<std::streamsize> (out.size ()));
  return out;
}

std::vector<float>
ReadFloat32 (const std::filesystem::path &path)
{
  const auto bytes = ReadBytes (path);
  if (bytes.size () % sizeof (float) != 0)
    {
      throw std::runtime_error ("Invalid f32 file " + path.string ());
    }
  std::vector<float> out (bytes.size () / sizeof (float));
  std::memcpy (out.data (), bytes.data (), bytes.size ());
  return out;
}

bool
Exists (const std::filesystem::path &path)
{
  std::error_code ec;
  return std::filesystem::exists (path, ec);
}

Options
ParseArgs (int argc, char **argv)
{
  Options opts;
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      auto take = [&] () -> std::string {
        if (i + 1 >= argc)
          {
            throw std::runtime_error ("Missing value for " + arg);
          }
        return argv[++i];
      };
      if (arg == "--source-records")
        {
          opts.source_records = take ();
        }
      else if (arg == "--assets-dir")
        {
          opts.assets_dir = take ();
        }
      else if (arg == "--output-dir")
        {
          opts.output_dir = take ();
        }
      else if (arg == "--models")
        {
          opts.models_dir = take ();
        }
      else if (arg == "--model")
        {
          opts.model_path = take ();
        }
      else if (arg == "--max-labels")
        {
          opts.max_labels = std::stoi (take ());
        }
      else if (arg == "--top-k")
        {
          opts.top_k = std::stoi (take ());
        }
      else if (arg == "--label-prompt-count")
        {
          opts.label_prompt_count = std::stoi (take ());
        }
      else if (arg == "--label-text-mode")
        {
          opts.label_text_mode = take ();
        }
      else if (arg == "--label-match-mode")
        {
          opts.label_match_mode = take ();
        }
      else
        {
          throw std::runtime_error ("Unknown argument: " + arg);
        }
    }
  return opts;
}

std::vector<std::string>
ReadStringArray (const Json &json, const char *key)
{
  std::vector<std::string> out;
  const auto found = json.find (key);
  if (found == json.end () || !found->is_array ())
    {
      return out;
    }
  for (const auto &value : *found)
    {
      if (value.is_string ())
        {
          out.push_back (value.get<std::string> ());
        }
    }
  return out;
}

std::string
JoinUnique (const std::vector<std::string> &items, int limit)
{
  std::set<std::string> seen;
  std::ostringstream out;
  int count = 0;
  for (const auto &item : items)
    {
      if (item.empty () || seen.count (item) > 0)
        {
          continue;
        }
      seen.insert (item);
      if (count > 0)
        {
          out << ". ";
        }
      out << item;
      ++count;
      if (limit > 0 && count >= limit)
        {
          break;
        }
    }
  return out.str ();
}

std::string
BuildLabelText (const LabelRecord &label, const Options &opts)
{
  std::vector<std::string> parts;
  parts.push_back (label.label);

  if (opts.label_text_mode == "label")
    {
      return label.label;
    }

  if (opts.label_text_mode == "prompts")
    {
      return JoinUnique (label.prompts, opts.label_prompt_count);
    }

  if (opts.label_text_mode != "expanded")
    {
      throw std::runtime_error ("Unknown label text mode: "
                                + opts.label_text_mode);
    }

  for (const auto &alias : label.aliases)
    {
      parts.push_back (alias);
    }
  if (!label.definition.empty ())
    {
      parts.push_back (label.definition);
    }
  int prompt_count = 0;
  for (const auto &prompt : label.prompts)
    {
      if (prompt.empty () || prompt == label.label)
        {
          continue;
        }
      parts.push_back (prompt);
      ++prompt_count;
      if (opts.label_prompt_count > 0
          && prompt_count >= opts.label_prompt_count)
        {
          break;
        }
    }
  return JoinUnique (parts, 0);
}

std::string
BuildMatchText (const LabelRecord &label, const Options &opts)
{
  if (opts.label_match_mode == "label")
    {
      return label.label;
    }
  if (opts.label_match_mode != "expanded")
    {
      throw std::runtime_error ("Unknown label match mode: "
                                + opts.label_match_mode);
    }
  std::vector<std::string> parts = { label.label };
  for (const auto &alias : label.aliases)
    {
      parts.push_back (alias);
    }
  if (!label.definition.empty ())
    {
      parts.push_back (label.definition);
    }
  return JoinUnique (parts, 0);
}

std::vector<LabelRecord>
LoadLabels (const Options &opts)
{
  std::ifstream in (opts.source_records);
  if (!in)
    {
      throw std::runtime_error ("Missing source records: "
                                + opts.source_records.string ());
    }
  std::vector<LabelRecord> labels;
  std::string line;
  while (std::getline (in, line))
    {
      if (line.empty ())
        {
          continue;
        }
      Json json = Json::parse (line, nullptr, false);
      if (json.is_discarded ())
        {
          continue;
        }
      LabelRecord label;
      label.index = static_cast<int> (labels.size ());
      label.label_id = json.value ("label_id", "");
      label.source = json.value ("source", "");
      label.kind = json.value ("kind", "");
      label.label = json.value ("label", "");
      label.definition = json.value ("definition", "");
      label.aliases = ReadStringArray (json, "aliases");
      label.prompts = ReadStringArray (json, "prompts");
      if (label.label.empty () || label.label_id.empty ())
        {
          continue;
        }
      label.encoded_text = BuildLabelText (label, opts);
      label.match_text = BuildMatchText (label, opts);
      if (label.encoded_text.empty ())
        {
          label.encoded_text = label.label;
        }
      if (label.match_text.empty ())
        {
          label.match_text = label.label;
        }
      labels.push_back (std::move (label));
      if (opts.max_labels > 0
          && static_cast<int> (labels.size ()) >= opts.max_labels)
        {
          break;
        }
    }
  return labels;
}

std::optional<std::filesystem::path>
ResolveModel (const Options &opts)
{
  if (!opts.model_path.empty ())
    {
      if (!std::filesystem::exists (opts.model_path))
        {
          throw std::runtime_error ("Model does not exist: "
                                    + opts.model_path.string ());
        }
      return opts.model_path;
    }
  return cortext::ResolveEssAistGgufModelPath (opts.models_dir);
}

std::vector<SignalRecord>
BuildSignals (cortext::AaitGgufEncoder &encoder,
              const std::filesystem::path &assets_dir)
{
  const auto raw = assets_dir / "raw";
  const auto extended_raw = std::filesystem::path (
      "build/extended_label_graph_assets/raw");
  const auto dog_image = ReadBytes (raw / "dog_384x384.rgb");
  const auto car_image = ReadBytes (raw / "car_crash_384x384.rgb");
  const auto dog_audio = ReadFloat32 (raw / "bailey_16k_mono.f32");
  const auto crash_audio = ReadFloat32 (raw / "crash_16k_mono.f32");

  std::vector<unsigned char> cat_image;
  std::vector<unsigned char> train_image;
  std::vector<unsigned char> bell_image;
  std::vector<float> cat_audio;
  std::vector<float> train_audio;
  std::vector<float> bell_audio;
  const bool has_extended = Exists (extended_raw / "cat_384x384.rgb")
                            && Exists (extended_raw / "train_384x384.rgb")
                            && Exists (extended_raw / "bell_384x384.rgb")
                            && Exists (extended_raw / "cat_16k_mono.f32")
                            && Exists (extended_raw / "train_16k_mono.f32")
                            && Exists (extended_raw / "bell_16k_mono.f32");
  if (has_extended)
    {
      cat_image = ReadBytes (extended_raw / "cat_384x384.rgb");
      train_image = ReadBytes (extended_raw / "train_384x384.rgb");
      bell_image = ReadBytes (extended_raw / "bell_384x384.rgb");
      cat_audio = ReadFloat32 (extended_raw / "cat_16k_mono.f32");
      train_audio = ReadFloat32 (extended_raw / "train_16k_mono.f32");
      bell_audio = ReadFloat32 (extended_raw / "bell_16k_mono.f32");
    }

  struct Event
  {
    std::string scenario;
    std::string id;
    std::string modality;
    std::string group;
    std::string text;
    const std::vector<unsigned char> *image = nullptr;
    const std::vector<float> *audio = nullptr;
  };

  std::vector<Event> events = {
    { "wikimedia_dog_multimodal", "dog_image", "image", "dog_entity", "",
      &dog_image, nullptr },
    { "wikimedia_dog_multimodal", "dog_text", "text", "dog_entity",
      "Golden Retriever", nullptr, nullptr },
    { "wikimedia_dog_multimodal", "dog_audio", "audio", "dog_entity", "",
      nullptr, &dog_audio },
    { "wikimedia_car_crash_multimodal", "car_image", "image", "car_event",
      "", &car_image, nullptr },
    { "wikimedia_car_crash_multimodal", "car_text", "text", "car_event",
      "Car crash 1", nullptr, nullptr },
    { "wikimedia_car_crash_multimodal", "crash_audio", "audio",
      "car_event", "", nullptr, &crash_audio },
    { "wikimedia_dog_then_car", "dog_image", "image", "dog_entity", "",
      &dog_image, nullptr },
    { "wikimedia_dog_then_car", "dog_text", "text", "dog_entity",
      "Golden Retriever", nullptr, nullptr },
    { "wikimedia_dog_then_car", "dog_audio", "audio", "dog_entity", "",
      nullptr, &dog_audio },
    { "wikimedia_dog_then_car", "car_image", "image", "car_event", "",
      &car_image, nullptr },
    { "wikimedia_dog_then_car", "car_text", "text", "car_event",
      "Car crash 1", nullptr, nullptr },
    { "wikimedia_dog_then_car", "crash_audio", "audio", "car_event", "",
      nullptr, &crash_audio },
    { "wikimedia_audio_image_interleave", "dog_audio", "audio",
      "dog_entity", "", nullptr, &dog_audio },
    { "wikimedia_audio_image_interleave", "dog_image", "image",
      "dog_entity", "", &dog_image, nullptr },
    { "wikimedia_audio_image_interleave", "crash_audio", "audio",
      "car_event", "", nullptr, &crash_audio },
    { "wikimedia_audio_image_interleave", "car_image", "image",
      "car_event", "", &car_image, nullptr },
  };

  if (has_extended)
    {
      events.insert (
          events.end (),
          {
              { "wikimedia_cat_multimodal", "cat_image", "image",
                "cat_entity", "", &cat_image, nullptr },
              { "wikimedia_cat_multimodal", "cat_text", "text",
                "cat_entity", "Cat", nullptr, nullptr },
              { "wikimedia_cat_multimodal", "cat_audio", "audio",
                "cat_entity", "", nullptr, &cat_audio },
              { "wikimedia_train_multimodal", "train_image", "image",
                "train_entity", "", &train_image, nullptr },
              { "wikimedia_train_multimodal", "train_text", "text",
                "train_entity", "Train", nullptr, nullptr },
              { "wikimedia_train_multimodal", "train_audio", "audio",
                "train_entity", "", nullptr, &train_audio },
              { "wikimedia_bell_multimodal", "bell_image", "image",
                "bell_entity", "", &bell_image, nullptr },
              { "wikimedia_bell_multimodal", "bell_text", "text",
                "bell_entity", "Bell", nullptr, nullptr },
              { "wikimedia_bell_multimodal", "bell_audio", "audio",
                "bell_entity", "", nullptr, &bell_audio },
              { "wikimedia_cat_train_bell_sequence", "cat_image", "image",
                "cat_entity", "", &cat_image, nullptr },
              { "wikimedia_cat_train_bell_sequence", "cat_text", "text",
                "cat_entity", "Cat", nullptr, nullptr },
              { "wikimedia_cat_train_bell_sequence", "cat_audio", "audio",
                "cat_entity", "", nullptr, &cat_audio },
              { "wikimedia_cat_train_bell_sequence", "train_image", "image",
                "train_entity", "", &train_image, nullptr },
              { "wikimedia_cat_train_bell_sequence", "train_text", "text",
                "train_entity", "Train", nullptr, nullptr },
              { "wikimedia_cat_train_bell_sequence", "train_audio", "audio",
                "train_entity", "", nullptr, &train_audio },
              { "wikimedia_cat_train_bell_sequence", "bell_image", "image",
                "bell_entity", "", &bell_image, nullptr },
              { "wikimedia_cat_train_bell_sequence", "bell_text", "text",
                "bell_entity", "Bell", nullptr, nullptr },
              { "wikimedia_cat_train_bell_sequence", "bell_audio", "audio",
                "bell_entity", "", nullptr, &bell_audio },
              { "wikimedia_extended_audio_image_interleave", "cat_audio",
                "audio", "cat_entity", "", nullptr, &cat_audio },
              { "wikimedia_extended_audio_image_interleave", "cat_image",
                "image", "cat_entity", "", &cat_image, nullptr },
              { "wikimedia_extended_audio_image_interleave", "train_audio",
                "audio", "train_entity", "", nullptr, &train_audio },
              { "wikimedia_extended_audio_image_interleave", "train_image",
                "image", "train_entity", "", &train_image, nullptr },
              { "wikimedia_extended_audio_image_interleave", "bell_audio",
                "audio", "bell_entity", "", nullptr, &bell_audio },
              { "wikimedia_extended_audio_image_interleave", "bell_image",
                "image", "bell_entity", "", &bell_image, nullptr },
          });
    }

  std::vector<SignalRecord> signals;
  for (const auto &event : events)
    {
      SignalRecord signal;
      signal.scenario = event.scenario;
      signal.signal_id = event.id;
      signal.modality = event.modality;
      signal.offline_group = event.group;
      if (event.modality == "text")
        {
          encoder.EncodeText (event.text, signal.embedding);
        }
      else if (event.modality == "image")
        {
          encoder.EncodeImage (event.image->data (), 384, 384, 3,
                               signal.embedding);
        }
      else
        {
          encoder.EncodeAudio (event.audio->data (), event.audio->size (),
                               signal.embedding);
        }
      Normalize (signal.embedding);
      signal.views = ViewsFor (signal.embedding);
      signals.push_back (std::move (signal));
    }
  return signals;
}

std::set<std::string>
Tokens (const std::string &text)
{
  static const std::set<std::string> stop = {
    "a", "an", "the", "of", "to", "in", "on", "with", "for", "and",
    "or", "by", "someone", "something", "about", "memory", "audio",
    "video", "text", "evidence", "scene", "containing", "involving", "1"
  };
  std::set<std::string> out;
  std::string token;
  for (unsigned char ch : text)
    {
      if (std::isalnum (ch))
        {
          token.push_back (static_cast<char> (std::tolower (ch)));
        }
      else
        {
          if (token.size () > 1 && stop.count (token) == 0)
            {
              out.insert (token);
            }
          token.clear ();
        }
    }
  if (token.size () > 1 && stop.count (token) == 0)
    {
      out.insert (token);
    }
  return out;
}

std::set<std::string>
ExpectedTokens (const std::string &group)
{
  if (group == "dog_entity")
    {
      return { "dog", "retriever", "canine", "puppy", "animal", "bailey" };
    }
  if (group == "car_event")
    {
      return { "car", "vehicle", "automobile", "crash", "collision",
               "accident", "wreck", "fender" };
    }
  if (group == "cat_entity")
    {
      return { "cat", "feline", "kitty", "animal" };
    }
  if (group == "train_entity")
    {
      return { "train", "rail", "locomotive", "vehicle" };
    }
  if (group == "bell_entity")
    {
      return { "bell", "chime", "liberty" };
    }
  return {};
}

std::vector<Candidate>
RankLabels (const std::vector<float> &query,
            const std::vector<LabelRecord> &labels,
            const std::string &view, int top_k)
{
  std::vector<Candidate> scores;
  scores.reserve (labels.size ());
  for (const auto &label : labels)
    {
      auto found = label.views.find (view);
      if (found == label.views.end () || found->second.empty ())
        {
          continue;
        }
      scores.push_back ({ label.index, Dot (query, found->second) });
    }
  const int keep = std::min<int> (top_k, scores.size ());
  std::partial_sort (scores.begin (), scores.begin () + keep, scores.end (),
                     [] (const Candidate &a, const Candidate &b) {
                       if (a.score != b.score)
                         {
                           return a.score > b.score;
                         }
                       return a.label_index < b.label_index;
                     });
  scores.resize (keep);
  return scores;
}

bool
ExpectedHit (const std::vector<Candidate> &candidates,
             const std::vector<LabelRecord> &labels,
             const std::string &group, int k)
{
  const auto expected = ExpectedTokens (group);
  for (int i = 0; i < std::min<int> (k, candidates.size ()); ++i)
    {
      const auto tokens
          = Tokens (labels[candidates[i].label_index].match_text);
      for (const auto &token : tokens)
        {
          if (expected.count (token) > 0)
            {
              return true;
            }
        }
    }
  return false;
}

std::pair<bool, std::string>
TopLabelTokenOverlap (const std::vector<Candidate> &a,
                      const std::vector<Candidate> &b,
                      const std::vector<LabelRecord> &labels, int k)
{
  for (int i = 0; i < std::min<int> (k, a.size ()); ++i)
    {
      const auto left = Tokens (labels[a[i].label_index].match_text);
      for (int j = 0; j < std::min<int> (k, b.size ()); ++j)
        {
          const auto right = Tokens (labels[b[j].label_index].match_text);
          std::vector<std::string> common;
          std::set_intersection (left.begin (), left.end (), right.begin (),
                                 right.end (), std::back_inserter (common));
          if (!common.empty ())
            {
              return { true,
                       labels[a[i].label_index].label + "<->"
                           + labels[b[j].label_index].label };
            }
        }
    }
  return { false, "" };
}

void
WriteSignalLabels (
    const std::filesystem::path &path, const std::vector<SignalRecord> &signals,
    const std::vector<LabelRecord> &labels,
    const std::map<std::string,
                   std::map<std::string, std::vector<Candidate>>> &rankings,
    int top_k)
{
  std::ofstream out (path);
  out << "scenario,signal_id,modality,offline_group,view,rank,label_id,kind,"
         "label,score,hit_top10,hit_top25,hit_top50\n";
  for (const auto &signal : signals)
    {
      for (const auto &[view, candidates] : rankings.at (signal.signal_id))
        {
          const bool hit10 = ExpectedHit (candidates, labels,
                                          signal.offline_group, 10);
          const bool hit25 = ExpectedHit (candidates, labels,
                                          signal.offline_group, 25);
          const bool hit50 = ExpectedHit (candidates, labels,
                                          signal.offline_group, 50);
          for (int i = 0; i < std::min<int> (top_k, candidates.size ()); ++i)
            {
              const auto &candidate = candidates[i];
              const auto &label = labels[candidate.label_index];
              out << CsvEscape (signal.scenario) << ","
                  << CsvEscape (signal.signal_id) << ","
                  << CsvEscape (signal.modality) << ","
                  << CsvEscape (signal.offline_group) << ","
                  << CsvEscape (view) << "," << (i + 1) << ","
                  << CsvEscape (label.label_id) << ","
                  << CsvEscape (label.kind) << ","
                  << CsvEscape (label.label) << "," << candidate.score << ","
                  << (hit10 ? 1 : 0) << "," << (hit25 ? 1 : 0) << ","
                  << (hit50 ? 1 : 0) << "\n";
            }
        }
    }
}

void
WriteEdges (const std::filesystem::path &path, const std::vector<EdgeRow> &edges)
{
  std::ofstream out (path);
  out << "scenario,view,variant,source_signal_id,target_signal_id,"
         "source_group,target_group,weight,same_group,cross_group,detail\n";
  for (const auto &edge : edges)
    {
      out << CsvEscape (edge.scenario) << "," << CsvEscape (edge.view) << ","
          << CsvEscape (edge.variant) << ","
          << CsvEscape (edge.source_signal_id) << ","
          << CsvEscape (edge.target_signal_id) << ","
          << CsvEscape (edge.source_group) << ","
          << CsvEscape (edge.target_group) << "," << edge.weight << ","
          << (edge.same_group ? 1 : 0) << "," << (edge.cross_group ? 1 : 0)
          << "," << CsvEscape (edge.detail) << "\n";
    }
}

void
WriteFailures (
    const std::filesystem::path &path, const std::vector<SignalRecord> &signals,
    const std::vector<LabelRecord> &labels,
    const std::map<std::string,
                   std::map<std::string, std::vector<Candidate>>> &rankings)
{
  std::ofstream out (path);
  out << "scenario,signal_id,modality,offline_group,view,failure_type,"
         "top_labels\n";
  for (const auto &signal : signals)
    {
      for (const auto &[view, candidates] : rankings.at (signal.signal_id))
        {
          if (ExpectedHit (candidates, labels, signal.offline_group, 25))
            {
              continue;
            }
          std::ostringstream top;
          for (int i = 0; i < std::min<int> (8, candidates.size ()); ++i)
            {
              if (i != 0)
                {
                  top << " | ";
                }
              const auto &candidate = candidates[i];
              const auto &label = labels[candidate.label_index];
              top << label.kind << ":" << label.label << ":"
                  << candidate.score;
            }
          out << CsvEscape (signal.scenario) << ","
              << CsvEscape (signal.signal_id) << ","
              << CsvEscape (signal.modality) << ","
              << CsvEscape (signal.offline_group) << ","
              << CsvEscape (view) << ",expected_group_not_in_top25,"
              << CsvEscape (top.str ()) << "\n";
        }
    }
}

} // namespace

int
main (int argc, char **argv)
{
  try
    {
      const Options opts = ParseArgs (argc, argv);
      std::filesystem::create_directories (opts.output_dir);
      const auto model_path = ResolveModel (opts);
      if (!model_path)
        {
          throw std::runtime_error ("ES-AIST GGUF model not found");
        }

      cortext::AaitGgufConfig config;
      config.model_path = model_path->string ();
      cortext::AaitGgufEncoder encoder (config);
      if (!encoder.IsLoaded () || !encoder.IsRuntimeAvailable ())
        {
          throw std::runtime_error ("ES-AIST runtime unavailable");
        }

      auto labels = LoadLabels (opts);
      auto encode_started = std::chrono::steady_clock::now ();
      for (auto &label : labels)
        {
          encoder.EncodeText (label.encoded_text, label.embedding);
          Normalize (label.embedding);
          label.views = ViewsFor (label.embedding);
        }
      const double label_encode_ms = Millis (std::chrono::steady_clock::now ()
                                             - encode_started);

      auto signal_started = std::chrono::steady_clock::now ();
      auto signals = BuildSignals (encoder, opts.assets_dir);
      const double signal_encode_ms = Millis (std::chrono::steady_clock::now ()
                                              - signal_started);

      const std::vector<std::string> views = {
        "prefix256",
        "semantic768",
        "entity768",
        "full1536",
      };
      std::map<std::string, std::map<std::string, std::vector<Candidate>>>
          rankings;
      for (const auto &signal : signals)
        {
          for (const auto &view : views)
            {
              rankings[signal.signal_id][view] = RankLabels (
                  signal.views.at (view), labels, view, opts.top_k);
            }
        }

      std::vector<EdgeRow> edges;
      std::map<std::string, std::vector<const SignalRecord *>> by_scenario;
      for (const auto &signal : signals)
        {
          by_scenario[signal.scenario].push_back (&signal);
        }
      const std::vector<int> edge_top_ks = { 10, 25, 50 };
      for (const auto &[scenario, scenario_signals] : by_scenario)
        {
          for (const auto &view : views)
            {
              for (int top_k : edge_top_ks)
                {
                  for (std::size_t i = 0; i < scenario_signals.size (); ++i)
                    {
                      for (std::size_t j = i + 1; j < scenario_signals.size ();
                           ++j)
                        {
                          const auto *a = scenario_signals[i];
                          const auto *b = scenario_signals[j];
                          auto [hit, detail] = TopLabelTokenOverlap (
                              rankings[a->signal_id][view],
                              rankings[b->signal_id][view], labels, top_k);
                          if (!hit)
                            {
                              continue;
                            }
                          EdgeRow edge;
                          edge.scenario = scenario;
                          edge.view = view;
                          edge.variant = "token_overlap_top"
                                         + std::to_string (top_k);
                          edge.source_signal_id = a->signal_id;
                          edge.target_signal_id = b->signal_id;
                          edge.source_group = a->offline_group;
                          edge.target_group = b->offline_group;
                          edge.weight = 1.0;
                          edge.same_group = a->offline_group == b->offline_group;
                          edge.cross_group = a->offline_group != b->offline_group;
                          edge.detail = detail;
                          edges.push_back (std::move (edge));
                        }
                    }
                }
            }
        }

      Json label_summary = Json::array ();
      for (const auto &view : views)
        {
          std::map<std::string, std::map<std::string, int>> by_modality;
          for (const auto &signal : signals)
            {
              const auto &candidates = rankings[signal.signal_id][view];
              auto &row = by_modality[signal.modality];
              row["count"] += 1;
              row["hit10"] += ExpectedHit (candidates, labels,
                                            signal.offline_group, 10)
                                  ? 1
                                  : 0;
              row["hit25"] += ExpectedHit (candidates, labels,
                                            signal.offline_group, 25)
                                  ? 1
                                  : 0;
              row["hit50"] += ExpectedHit (candidates, labels,
                                            signal.offline_group, 50)
                                  ? 1
                                  : 0;
            }
          for (const auto &[modality, counts] : by_modality)
            {
              const double denom = static_cast<double> (counts.at ("count"));
              label_summary.push_back ({
                { "view", view },
                { "modality", modality },
                { "count", counts.at ("count") },
                { "hit10", counts.at ("hit10") },
                { "hit25", counts.at ("hit25") },
                { "hit50", counts.at ("hit50") },
                { "hit10_rate", counts.at ("hit10") / denom },
                { "hit25_rate", counts.at ("hit25") / denom },
                { "hit50_rate", counts.at ("hit50") / denom },
              });
            }
        }

      Json graph_summary = Json::array ();
      for (const auto &view : views)
        {
          for (int top_k : edge_top_ks)
            {
              const std::string variant = "token_overlap_top"
                                          + std::to_string (top_k);
              int possible_same = 0;
              int possible_cross = 0;
              for (const auto &[scenario, scenario_signals] : by_scenario)
                {
                  (void)scenario;
                  for (std::size_t i = 0; i < scenario_signals.size (); ++i)
                    {
                      for (std::size_t j = i + 1;
                           j < scenario_signals.size (); ++j)
                        {
                          if (scenario_signals[i]->offline_group
                              == scenario_signals[j]->offline_group)
                            {
                              ++possible_same;
                            }
                          else
                            {
                              ++possible_cross;
                            }
                        }
                    }
                }
              int same = 0;
              int cross = 0;
              int edge_count = 0;
              for (const auto &edge : edges)
                {
                  if (edge.view != view || edge.variant != variant)
                    {
                      continue;
                    }
                  ++edge_count;
                  same += edge.same_group ? 1 : 0;
                  cross += edge.cross_group ? 1 : 0;
                }
              const int scored = same + cross;
              graph_summary.push_back ({
                { "view", view },
                { "variant", variant },
                { "edge_count", edge_count },
                { "same_group_edges", same },
                { "cross_group_edges", cross },
                { "possible_same_group_pairs", possible_same },
                { "possible_cross_group_pairs", possible_cross },
                { "scored_edge_precision",
                  scored > 0 ? static_cast<double> (same) / scored : 1.0 },
                { "same_group_pair_recall",
                  possible_same > 0 ? static_cast<double> (same)
                                          / possible_same
                                    : 0.0 },
                { "cross_group_pair_rate",
                  possible_cross > 0 ? static_cast<double> (cross)
                                           / possible_cross
                                     : 0.0 },
              });
            }
        }

      WriteSignalLabels (opts.output_dir / "es_aist_label_graph_signal_labels.csv",
                         signals, labels, rankings, opts.top_k);
      WriteEdges (opts.output_dir / "es_aist_label_graph_edges.csv", edges);
      WriteFailures (opts.output_dir / "es_aist_label_graph_failures.csv",
                     signals, labels, rankings);

      Json summary = {
        { "benchmark_only", true },
        { "production_behavior_changed", false },
        { "config",
          {
              { "label_text_mode", opts.label_text_mode },
              { "label_match_mode", opts.label_match_mode },
              { "label_prompt_count", opts.label_prompt_count },
              { "top_k", opts.top_k },
              { "max_labels", opts.max_labels },
          } },
        { "model_path", model_path->string () },
        { "kernel_ops", encoder.UsesKernelOps () },
        { "kernel_backend", encoder.KernelOpsBackend () },
        { "label_count", labels.size () },
        { "signal_count", signals.size () },
        { "label_encode_ms", label_encode_ms },
        { "signal_encode_ms", signal_encode_ms },
        { "views", views },
        { "label_summary", label_summary },
        { "graph_summary", graph_summary },
        { "outputs",
          Json::array ({ "es_aist_label_graph_results.json",
                         "es_aist_label_graph_signal_labels.csv",
                         "es_aist_label_graph_edges.csv",
                         "es_aist_label_graph_failures.csv" }) },
      };
      {
        std::ofstream out (opts.output_dir
                           / "es_aist_label_graph_results.json");
        out << summary.dump (2) << "\n";
      }
      std::cout << summary.dump (2) << "\n";
      return 0;
    }
  catch (const std::exception &e)
    {
      std::cerr << "error: " << e.what () << "\n";
      return 1;
    }
}
