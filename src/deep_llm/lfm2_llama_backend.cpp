#include "lfm2_llama_backend.hpp"

#include "cortext/core/thread_config.hpp"
#include "llama_cpp_support.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
BuildSummarySystemPrompt ()
{
  return "You are writing a durable memory note from conversation excerpts.\n"
         "Write a concise factual summary in 1-3 sentences.\n"
         "Return only the summary text.\n"
         "Treat lines labeled 'User:' as a human user and lines labeled "
         "'Assistant:' as the assistant.\n"
         "Summarize the underlying facts and topics, not the mechanics of the "
         "conversation.\n"
         "Prioritize durable facts about people, events, names, preferences, "
         "plans, and outcomes over banter, greetings, or rhetorical "
         "questions.\n"
         "Include all major durable facts that fit, especially named people, "
         "projects, technologies, and goals.\n"
         "If the excerpts contain multiple topics, list them as separate facts "
         "instead of implying they caused each other.\n"
         "If both user and assistant excerpts restate the same fact, prefer "
         "the underlying fact itself instead of narrating who said it.\n"
         "Do not repeat the same fact from different perspectives.\n"
         "State facts directly when possible. Prefer direct factual sentences "
         "over wording like 'The user...' or 'The assistant...'.\n"
         "Do not use speaker-role subjects or second-person phrasing in the "
         "summary. Avoid 'the user', 'the assistant', 'you', and 'your' when "
         "a concrete named subject or neutral phrasing is available.\n"
         "Do not write phrases like 'the user said', 'the assistant asked', "
         "'in a conversation', 'they discussed', or 'this occurred after' "
         "unless that wording is necessary for clarity.\n"
         "Do not infer causality, chronology, identity, or shared beliefs "
         "beyond the text.\n"
         "Avoid speculation, role confusion, and meta commentary.";
}

std::string
BuildSummaryUserPrompt (const std::vector<std::string> &texts)
{
  std::ostringstream combined;
  combined << "Conversation excerpts:\n\n";
  for (size_t i = 0; i < texts.size (); ++i)
    {
      combined << "Excerpt " << (i + 1) << ":\n" << texts[i] << "\n\n";
    }
  return combined.str ();
}

std::string
BuildExtractionSystemPrompt ()
{
  return "Extract labels and relations from the provided text.\n"
         "Return only a single JSON object with keys \"labels\" and "
         "\"relations\".\n"
         "Prefer recall over compression: include all salient labels and "
         "relations that are directly supported by the text.\n"
         "\"labels\" must be an array of non-empty strings copied from the "
         "text.\n"
         "Each relation must include non-empty \"subject\", \"predicate\", "
         "and \"object\" strings taken from the text.\n"
         "If confidence is present, it must be a JSON number.\n"
         "Always return at least one label; if nothing is obvious, choose the "
         "single most salient term from the text.\n"
         "Do not emit any explanation outside the JSON object.";
}

std::string
BuildLabelOnlyExtractionSystemPrompt ()
{
  return "Extract labels from the provided text.\n"
         "Return only a single JSON object with the key \"labels\".\n"
         "Prefer recall over compression: include all salient labels that are "
         "directly supported by the text.\n"
         "\"labels\" must be an array of non-empty strings copied from the "
         "text.\n"
         "Always return at least one label; if nothing is obvious, choose the "
         "single most salient term from the text.\n"
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

  return result;
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

  if (schema.contains ("required"))
    {
      RequireBoolean (schema["required"].is_array (),
                      "LFM2 extractor schema required must be an array");
      for (const auto &required : schema["required"])
        {
          RequireBoolean (required.is_string (),
                          "LFM2 extractor schema required entries must be strings");
          const std::string key = required.get<std::string> ();
          RequireBoolean (key == "labels" || key == "relations",
                          "LFM2 extractor schema only supports labels/relations in required");
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
BuildLfm2ExtractionGrammar (const nlohmann::json &schema)
{
  ValidateSchemaSubset (schema);

  const auto &properties = schema["properties"];
  const auto &labels = properties["labels"];
  const bool labels_nonempty = labels.value ("minItems", 0) > 0;
  const bool has_relations = properties.contains ("relations");
  bool relations_required = false;
  bool confidence_supported = false;
  if (schema.contains ("required"))
    {
      for (const auto &required : schema["required"])
        {
          if (required.is_string () && required.get<std::string> () == "relations")
            {
              relations_required = true;
            }
        }
    }
  if (has_relations)
    {
      const auto &relation_props = properties["relations"]["items"]["properties"];
      confidence_supported = relation_props.contains ("confidence");
    }

  std::ostringstream grammar;
  if (has_relations)
    {
      grammar << "root ::= \"{\" space labels-kv";
      if (relations_required)
        {
          grammar << " \",\" space relations-kv";
        }
      else
        {
          grammar << " ( \",\" space ( relations-kv ) )?";
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
      ggml_backend_load_all ();
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
  config.max_tokens = std::clamp ((max_words > 0 ? max_words * 8 : 256), 64, 512);
  config.temperature = 0.3f;
  config.min_p = 0.15f;
  config.repetition_penalty = 1.05f;
  config.greedy = false;

  std::vector<std::pair<std::string, std::string>> messages;
  messages.emplace_back ("system", BuildSummarySystemPrompt ());
  messages.emplace_back ("user", BuildSummaryUserPrompt (texts));
  return TrimToWordLimit (impl_->model.Generate (messages, config), max_words);
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
