#include <node_api.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "cortext/capi.h"

namespace
{

#define NAPI_RETURN_IF_FAILED(env, call)                                         \
  do                                                                             \
    {                                                                            \
      const napi_status status__ = (call);                                       \
      if (status__ != napi_ok)                                                   \
        {                                                                        \
          const napi_extended_error_info *info__ = nullptr;                      \
          napi_get_last_error_info ((env), &info__);                             \
          const char *message__                                                  \
              = (info__ && info__->error_message) ? info__->error_message        \
                                                  : "Node-API call failed";      \
          napi_throw_error ((env), nullptr, message__);                          \
          return nullptr;                                                        \
        }                                                                        \
    }                                                                            \
  while (0)

struct BindingHandle
{
  cortext_handle handle = nullptr;
};

napi_value
ThrowTypeError (napi_env env, const char *message)
{
  napi_throw_type_error (env, nullptr, message);
  return nullptr;
}

napi_value
ThrowCortextError (napi_env env, const char *fallback)
{
  const char *detail = cortext_last_error ();
  napi_throw_error (env, nullptr,
                    (detail && *detail != '\0') ? detail : fallback);
  return nullptr;
}

BindingHandle *
UnwrapHandle (napi_env env, napi_value jsthis)
{
  BindingHandle *wrapped = nullptr;
  if (napi_unwrap (env, jsthis, reinterpret_cast<void **> (&wrapped)) != napi_ok
      || wrapped == nullptr || wrapped->handle == nullptr)
    {
      return nullptr;
    }
  return wrapped;
}

bool
GetString (napi_env env, napi_value value, std::string &out)
{
  size_t size = 0;
  if (napi_get_value_string_utf8 (env, value, nullptr, 0, &size) != napi_ok)
    {
      return false;
    }

  out.resize (size + 1);
  if (napi_get_value_string_utf8 (env, value, out.data (), out.size (), &size)
      != napi_ok)
    {
      return false;
    }
  out.resize (size);
  return true;
}

bool
GetOptionalStringProperty (napi_env env, napi_value obj, const char *name,
                           std::string &out)
{
  bool has = false;
  if (napi_has_named_property (env, obj, name, &has) != napi_ok || !has)
    {
      return true;
    }

  napi_value value;
  if (napi_get_named_property (env, obj, name, &value) != napi_ok)
    {
      return false;
    }

  napi_valuetype type = napi_undefined;
  if (napi_typeof (env, value, &type) != napi_ok)
    {
      return false;
    }
  if (type == napi_null || type == napi_undefined)
    {
      out.clear ();
      return true;
    }
  return GetString (env, value, out);
}

bool
GetOptionalDoubleProperty (napi_env env, napi_value obj, const char *name,
                           double &out)
{
  bool has = false;
  if (napi_has_named_property (env, obj, name, &has) != napi_ok || !has)
    {
      return true;
    }

  napi_value value;
  if (napi_get_named_property (env, obj, name, &value) != napi_ok)
    {
      return false;
    }

  return napi_get_value_double (env, value, &out) == napi_ok;
}

bool
GetOptionalBoolProperty (napi_env env, napi_value obj, const char *name,
                         int &out)
{
  bool has = false;
  if (napi_has_named_property (env, obj, name, &has) != napi_ok || !has)
    {
      return true;
    }

  napi_value value;
  bool flag = false;
  if (napi_get_named_property (env, obj, name, &value) != napi_ok
      || napi_get_value_bool (env, value, &flag) != napi_ok)
    {
      return false;
    }

  out = flag ? 1 : 0;
  return true;
}

bool
ParseConfigObject (napi_env env, napi_value value, cortext_config &cfg,
                   std::string &label_bank_path,
                   std::string &summarizer_provider_uri,
                   std::string &extractor_provider_uri)
{
  if (!GetOptionalDoubleProperty (env, value, "focus", cfg.focus)
      || !GetOptionalDoubleProperty (env, value, "sensitivity",
                                    cfg.sensitivity)
      || !GetOptionalDoubleProperty (env, value, "stability", cfg.stability)
      || !GetOptionalBoolProperty (env, value, "affectInterrupt",
                                   cfg.affect_interrupt)
      || !GetOptionalBoolProperty (env, value, "affectRetrieval",
                                   cfg.affect_retrieval)
      || !GetOptionalBoolProperty (env, value, "reinforcementEnabled",
                                   cfg.reinforcement_enabled)
      || !GetOptionalBoolProperty (env, value, "proceduralEnabled",
                                   cfg.procedural_enabled)
      || !GetOptionalBoolProperty (env, value, "sequentialEdgesEnabled",
                                   cfg.sequential_edges_enabled)
      || !GetOptionalStringProperty (env, value, "labelBankPath",
                                     label_bank_path)
      || !GetOptionalStringProperty (env, value, "summarizerProviderUri",
                                     summarizer_provider_uri)
      || !GetOptionalStringProperty (env, value, "extractorProviderUri",
                                     extractor_provider_uri))
    {
      return false;
    }

  if (!label_bank_path.empty ())
    {
      cfg.label_bank_path = label_bank_path.c_str ();
    }
  if (!summarizer_provider_uri.empty ())
    {
      cfg.summarizer_provider_uri = summarizer_provider_uri.c_str ();
    }
  if (!extractor_provider_uri.empty ())
    {
      cfg.extractor_provider_uri = extractor_provider_uri.c_str ();
    }
  return true;
}

bool
GetFloat32Array (napi_env env, napi_value value, const float **data,
                 size_t &length)
{
  bool is_typed_array = false;
  if (napi_is_typedarray (env, value, &is_typed_array) != napi_ok
      || !is_typed_array)
    {
      return false;
    }

  napi_typedarray_type type;
  size_t element_count = 0;
  void *raw = nullptr;
  napi_value array_buffer;
  size_t byte_offset = 0;
  if (napi_get_typedarray_info (env, value, &type, &element_count, &raw,
                                &array_buffer, &byte_offset)
          != napi_ok
      || type != napi_float32_array)
    {
      return false;
    }

  *data = static_cast<const float *> (raw);
  length = element_count;
  return true;
}

bool
GetUint8Data (napi_env env, napi_value value, const uint8_t **data,
              size_t &length)
{
  bool is_buffer = false;
  if (napi_is_buffer (env, value, &is_buffer) == napi_ok && is_buffer)
    {
      void *raw = nullptr;
      if (napi_get_buffer_info (env, value, &raw, &length) != napi_ok)
        {
          return false;
        }
      *data = static_cast<const uint8_t *> (raw);
      return true;
    }

  bool is_typed_array = false;
  if (napi_is_typedarray (env, value, &is_typed_array) != napi_ok
      || !is_typed_array)
    {
      return false;
    }

  napi_typedarray_type type;
  void *raw = nullptr;
  napi_value array_buffer;
  size_t byte_offset = 0;
  if (napi_get_typedarray_info (env, value, &type, &length, &raw,
                                &array_buffer, &byte_offset)
          != napi_ok)
    {
      return false;
    }

  if (type != napi_uint8_array && type != napi_uint8_clamped_array)
    {
      return false;
    }

  *data = static_cast<const uint8_t *> (raw);
  return true;
}

napi_value
JSONStringResult (napi_env env, char *raw, const char *fallback)
{
  if (raw == nullptr)
    {
      return ThrowCortextError (env, fallback);
    }

  napi_value result;
  const napi_status status
      = napi_create_string_utf8 (env, raw, NAPI_AUTO_LENGTH, &result);
  cortext_string_free (raw);
  if (status != napi_ok)
    {
      const napi_extended_error_info *info = nullptr;
      napi_get_last_error_info (env, &info);
      napi_throw_error (env, nullptr,
                        (info && info->error_message) ? info->error_message
                                                      : "Failed to create JS string");
      return nullptr;
    }
  return result;
}

void
FinalizeHandle (napi_env env, void *data, void *hint)
{
  auto *wrapped = static_cast<BindingHandle *> (data);
  if (wrapped != nullptr)
    {
      if (wrapped->handle != nullptr)
        {
          cortext_free (wrapped->handle);
        }
      delete wrapped;
    }
}

napi_value
CortextCtor (napi_env env, napi_callback_info info)
{
  size_t argc = 3;
  napi_value args[3];
  napi_value jsthis;
  NAPI_RETURN_IF_FAILED (
      env, napi_get_cb_info (env, info, &argc, args, &jsthis, nullptr));

  cortext_config cfg{};
  cortext_config_init (&cfg);

  std::string label_bank_path;
  std::string summarizer_provider_uri;
  std::string extractor_provider_uri;
  std::string db_path = ":memory:";
  std::string models_dir;
  bool first_arg_is_db_path = false;

  if (argc > 0)
    {
      napi_valuetype type = napi_undefined;
      NAPI_RETURN_IF_FAILED (env, napi_typeof (env, args[0], &type));
      if (type == napi_string)
        {
          first_arg_is_db_path = true;
          if (!GetString (env, args[0], db_path))
            {
              return ThrowTypeError (env, "dbPath must be a string");
            }
        }
      else if (type != napi_undefined && type != napi_null)
        {
          if (type != napi_object
              || !ParseConfigObject (env, args[0], cfg, label_bank_path,
                                     summarizer_provider_uri,
                                     extractor_provider_uri))
            {
              return ThrowTypeError (env, "config must be an object");
            }
        }
    }

  if (argc > 1)
    {
      napi_valuetype type = napi_undefined;
      NAPI_RETURN_IF_FAILED (env, napi_typeof (env, args[1], &type));
      if (type != napi_undefined && type != napi_null)
        {
          if (!GetString (env, args[1], first_arg_is_db_path ? models_dir : db_path))
            {
              return ThrowTypeError (
                  env, first_arg_is_db_path ? "modelsDir must be a string"
                                            : "dbPath must be a string");
            }
        }
    }

  if (!first_arg_is_db_path && argc > 2)
    {
      napi_valuetype type = napi_undefined;
      NAPI_RETURN_IF_FAILED (env, napi_typeof (env, args[2], &type));
      if (type != napi_undefined && type != napi_null)
        {
          if (!GetString (env, args[2], models_dir))
            {
              return ThrowTypeError (env, "modelsDir must be a string");
            }
        }
    }

  auto *wrapped = new BindingHandle ();
  wrapped->handle = cortext_create_with_config (
      &cfg, db_path.c_str (), models_dir.empty () ? nullptr : models_dir.c_str ());
  if (wrapped->handle == nullptr)
    {
      delete wrapped;
      return ThrowCortextError (env, "cortext_create_with_config failed");
    }

  NAPI_RETURN_IF_FAILED (
      env, napi_wrap (env, jsthis, wrapped, FinalizeHandle, nullptr, nullptr));
  return jsthis;
}

napi_value
ProcessTextJSON (napi_env env, napi_callback_info info)
{
  size_t argc = 2;
  napi_value args[2];
  napi_value jsthis;
  NAPI_RETURN_IF_FAILED (
      env, napi_get_cb_info (env, info, &argc, args, &jsthis, nullptr));
  BindingHandle *wrapped = UnwrapHandle (env, jsthis);
  if (wrapped == nullptr)
    {
      return ThrowCortextError (env, "invalid Cortext handle");
    }

  std::string text;
  std::string source_id;
  if (argc < 2 || !GetString (env, args[0], text)
      || !GetString (env, args[1], source_id))
    {
      return ThrowTypeError (env, "processTextJson(text, sourceId) expects strings");
    }

  return JSONStringResult (
      env, cortext_process_text_json (wrapped->handle, text.c_str (),
                                      source_id.c_str ()),
      "cortext_process_text_json failed");
}

napi_value
ProcessAudioJSON (napi_env env, napi_callback_info info)
{
  size_t argc = 2;
  napi_value args[2];
  napi_value jsthis;
  NAPI_RETURN_IF_FAILED (
      env, napi_get_cb_info (env, info, &argc, args, &jsthis, nullptr));
  BindingHandle *wrapped = UnwrapHandle (env, jsthis);
  if (wrapped == nullptr)
    {
      return ThrowCortextError (env, "invalid Cortext handle");
    }

  const float *pcm = nullptr;
  size_t sample_count = 0;
  std::string source_id;
  if (argc < 2 || !GetFloat32Array (env, args[0], &pcm, sample_count)
      || !GetString (env, args[1], source_id))
    {
      return ThrowTypeError (
          env, "processAudioJson(pcm, sourceId) expects Float32Array and string");
    }

  return JSONStringResult (
      env, cortext_process_audio_json (wrapped->handle, pcm, sample_count,
                                       source_id.c_str ()),
      "cortext_process_audio_json failed");
}

napi_value
ProcessImageJSON (napi_env env, napi_callback_info info)
{
  size_t argc = 5;
  napi_value args[5];
  napi_value jsthis;
  NAPI_RETURN_IF_FAILED (
      env, napi_get_cb_info (env, info, &argc, args, &jsthis, nullptr));
  BindingHandle *wrapped = UnwrapHandle (env, jsthis);
  if (wrapped == nullptr)
    {
      return ThrowCortextError (env, "invalid Cortext handle");
    }

  const uint8_t *data = nullptr;
  size_t byte_count = 0;
  int32_t width = 0;
  int32_t height = 0;
  int32_t channels = 0;
  std::string source_id;
  if (argc < 5 || !GetUint8Data (env, args[0], &data, byte_count)
      || napi_get_value_int32 (env, args[1], &width) != napi_ok
      || napi_get_value_int32 (env, args[2], &height) != napi_ok
      || napi_get_value_int32 (env, args[3], &channels) != napi_ok
      || !GetString (env, args[4], source_id))
    {
      return ThrowTypeError (
          env,
          "processImageJson(data, width, height, channels, sourceId) expects bytes, numbers, and string");
    }

  if (width <= 0 || height <= 0 || channels <= 0)
    {
      return ThrowTypeError (env, "width, height, and channels must be positive");
    }

  const size_t expected
      = static_cast<size_t> (width) * static_cast<size_t> (height)
        * static_cast<size_t> (channels);
  if (byte_count < expected)
    {
      return ThrowTypeError (
          env, "image buffer is smaller than width * height * channels");
    }

  return JSONStringResult (
      env, cortext_process_image_json (wrapped->handle, data, width, height,
                                       channels, source_id.c_str ()),
      "cortext_process_image_json failed");
}

napi_value
ConsolidateJSON (napi_env env, napi_callback_info info)
{
  napi_value jsthis;
  NAPI_RETURN_IF_FAILED (
      env, napi_get_cb_info (env, info, nullptr, nullptr, &jsthis, nullptr));
  BindingHandle *wrapped = UnwrapHandle (env, jsthis);
  if (wrapped == nullptr)
    {
      return ThrowCortextError (env, "invalid Cortext handle");
    }

  return JSONStringResult (env, cortext_consolidate_json (wrapped->handle),
                           "cortext_consolidate_json failed");
}

napi_value
ConsolidateModeJSON (napi_env env, napi_callback_info info)
{
  size_t argc = 1;
  napi_value args[1];
  napi_value jsthis;
  NAPI_RETURN_IF_FAILED (
      env, napi_get_cb_info (env, info, &argc, args, &jsthis, nullptr));
  BindingHandle *wrapped = UnwrapHandle (env, jsthis);
  if (wrapped == nullptr)
    {
      return ThrowCortextError (env, "invalid Cortext handle");
    }

  int32_t mode = 0;
  if (argc < 1 || napi_get_value_int32 (env, args[0], &mode) != napi_ok)
    {
      return ThrowTypeError (env, "consolidateModeJson(mode) expects a number");
    }

  return JSONStringResult (
      env, cortext_consolidate_mode_json (wrapped->handle, mode),
      "cortext_consolidate_mode_json failed");
}

napi_value
Flush (napi_env env, napi_callback_info info)
{
  napi_value jsthis;
  NAPI_RETURN_IF_FAILED (
      env, napi_get_cb_info (env, info, nullptr, nullptr, &jsthis, nullptr));
  BindingHandle *wrapped = UnwrapHandle (env, jsthis);
  if (wrapped == nullptr)
    {
      return ThrowCortextError (env, "invalid Cortext handle");
    }

  if (cortext_flush (wrapped->handle) != 0)
    {
      return ThrowCortextError (env, "cortext_flush failed");
    }

  napi_value undefined_value;
  NAPI_RETURN_IF_FAILED (env, napi_get_undefined (env, &undefined_value));
  return undefined_value;
}

napi_value
Version (napi_env env, napi_callback_info info)
{
  napi_value result;
  NAPI_RETURN_IF_FAILED (
      env, napi_create_string_utf8 (env, cortext_version (), NAPI_AUTO_LENGTH,
                                    &result));
  return result;
}

napi_value
LastError (napi_env env, napi_callback_info info)
{
  napi_value result;
  NAPI_RETURN_IF_FAILED (
      env, napi_create_string_utf8 (env, cortext_last_error (),
                                    NAPI_AUTO_LENGTH, &result));
  return result;
}

napi_value
Init (napi_env env, napi_value exports)
{
  const napi_property_descriptor ctor_props[] = {
    { "processTextJson", nullptr, ProcessTextJSON, nullptr, nullptr, nullptr,
      napi_default, nullptr },
    { "processAudioJson", nullptr, ProcessAudioJSON, nullptr, nullptr, nullptr,
      napi_default, nullptr },
    { "processImageJson", nullptr, ProcessImageJSON, nullptr, nullptr, nullptr,
      napi_default, nullptr },
    { "consolidateJson", nullptr, ConsolidateJSON, nullptr, nullptr, nullptr,
      napi_default, nullptr },
    { "consolidateModeJson", nullptr, ConsolidateModeJSON, nullptr, nullptr,
      nullptr, napi_default, nullptr },
    { "flush", nullptr, Flush, nullptr, nullptr, nullptr, napi_default,
      nullptr },
  };

  napi_value ctor;
  NAPI_RETURN_IF_FAILED (
      env, napi_define_class (env, "NativeCortext", NAPI_AUTO_LENGTH,
                              CortextCtor, nullptr,
                              sizeof (ctor_props) / sizeof (ctor_props[0]),
                              ctor_props, &ctor));
  NAPI_RETURN_IF_FAILED (
      env, napi_set_named_property (env, exports, "NativeCortext", ctor));

  const napi_property_descriptor exports_props[] = {
    { "version", nullptr, Version, nullptr, nullptr, nullptr, napi_default,
      nullptr },
    { "lastError", nullptr, LastError, nullptr, nullptr, nullptr, napi_default,
      nullptr },
  };
  NAPI_RETURN_IF_FAILED (
      env, napi_define_properties (env, exports,
                                   sizeof (exports_props)
                                       / sizeof (exports_props[0]),
                                   exports_props));

  napi_value value;
  NAPI_RETURN_IF_FAILED (
      env, napi_create_int32 (env, CORTEXT_CONSOLIDATE_SHALLOW, &value));
  NAPI_RETURN_IF_FAILED (
      env, napi_set_named_property (env, exports, "CONSOLIDATE_SHALLOW", value));
  NAPI_RETURN_IF_FAILED (
      env, napi_create_int32 (env, CORTEXT_CONSOLIDATE_DEEP, &value));
  NAPI_RETURN_IF_FAILED (
      env, napi_set_named_property (env, exports, "CONSOLIDATE_DEEP", value));
  NAPI_RETURN_IF_FAILED (
      env, napi_create_int32 (env, CORTEXT_CONSOLIDATE_BOTH, &value));
  NAPI_RETURN_IF_FAILED (
      env, napi_set_named_property (env, exports, "CONSOLIDATE_BOTH", value));

  return exports;
}

} // namespace

NAPI_MODULE (cortext, Init)
