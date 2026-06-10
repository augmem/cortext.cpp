#include "cortext/generator/json_decoder.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <tuple>

namespace cortext
{

namespace
{

/// @brief Infer JSON Schema primitive type name for a value.
std::string
InferJsonType (const nlohmann::json &value)
{
  if (value.is_null ())
    return "null";
  if (value.is_boolean ())
    return "boolean";
  if (value.is_number_integer ())
    return "integer";
  if (value.is_number_float ())
    return "number";
  if (value.is_string ())
    return "string";
  if (value.is_array ())
    return "array";
  if (value.is_object ())
    return "object";
  return "unknown";
}

/// @brief Check if a string starts with a prefix.
bool
StartsWith (const std::string &str, const std::string &prefix)
{
  return str.size () >= prefix.size ()
         && str.compare (0, prefix.size (), prefix) == 0;
}

} // namespace

struct StreamingJSONParser::Impl
{
  nlohmann::json schema;
  std::string buffer;
  JSONState state = JSONState::kExpectingObjectStart;

  // Structure tracking
  struct ContainerFrame
  {
    enum class Kind
    {
      Object,
      Array
    };

    Kind kind;
    nlohmann::json schema;
    bool last_was_comma = false;
    std::size_t item_count = 0;
    std::size_t min_items = 0;
    bool is_root = false;
  };

  struct ObjectContext
  {
    std::set<std::string> seen_fields;
    std::set<std::string> required_fields;
    std::string current_key;
  };

  std::vector<ContainerFrame> container_stack;
  std::vector<ObjectContext> object_context_stack;
  std::string current_key;
  std::vector<std::string> current_path;

  // Schema satisfaction tracking
  std::set<std::string> seen_fields;
  std::set<std::string> required_fields;

  // oneOf handling
  std::vector<nlohmann::json> oneof_branches;
  std::optional<OneOfDiscriminator> oneof_discriminator;
  std::set<std::string> oneof_union_properties;
  std::optional<int> active_branch_index;
  std::vector<int> candidate_branch_order;
  int heuristic_idx = 0;
  bool heuristic_fallback_used = false;
  bool oneof_union_mode = false;
  std::optional<int> preferred_branch_index;
  std::vector<std::string> preferred_fields_prelock;
  std::vector<std::set<std::string>> oneof_unique_props_by_branch;

  // Value parsing buffers
  std::string partial_key;
  std::string partial_value;
  bool escape_next = false;
  std::optional<nlohmann::json> current_value_schema;

  // Value type context
  std::optional<std::string> current_value_type;
  std::optional<std::vector<nlohmann::json>> current_enum_values;
  std::optional<std::string> current_enum_type;

  // Parsed result
  nlohmann::json parsed_object = nlohmann::json::object ();

  void
  Initialize (const nlohmann::json &s)
  {
    schema = s;
    if (schema.contains ("required") && schema["required"].is_array ())
      {
        for (const auto &field : schema["required"])
          {
            if (field.is_string ())
              {
                required_fields.insert (field.get<std::string> ());
              }
          }
      }
    AnalyzeOneOf ();
  }

  void
  ClearValueContext ()
  {
    current_value_type = std::nullopt;
    current_enum_values = std::nullopt;
    current_enum_type = std::nullopt;
    current_value_schema = std::nullopt;
    partial_value.clear ();
    escape_next = false;
  }

  void
  EnterObject (const nlohmann::json &obj_schema, bool is_root)
  {
    if (!is_root)
      {
        object_context_stack.push_back (
            { seen_fields, required_fields, current_key });
      }

    required_fields.clear ();
    if (obj_schema.contains ("required") && obj_schema["required"].is_array ())
      {
        for (const auto &field : obj_schema["required"])
          {
            if (field.is_string ())
              {
                required_fields.insert (field.get<std::string> ());
              }
          }
      }
    seen_fields.clear ();
    current_key.clear ();

    container_stack.push_back (
        { ContainerFrame::Kind::Object, obj_schema, false, 0, 0, is_root });
    state = JSONState::kExpectingKeyOrClose;
  }

  void
  EnterArray (const nlohmann::json &array_schema)
  {
    nlohmann::json item_schema = nlohmann::json::object ();
    if (array_schema.contains ("items"))
      item_schema = array_schema["items"];

    std::size_t min_items = 0;
    if (array_schema.contains ("minItems") && array_schema["minItems"].is_number_integer ())
      {
        int value = array_schema["minItems"].get<int> ();
        if (value > 0)
          min_items = static_cast<std::size_t> (value);
      }

    container_stack.push_back (
        { ContainerFrame::Kind::Array, item_schema, false, 0, min_items, false });
    state = JSONState::kInArray;
    BeginValueContext (item_schema);
  }

  void
  AnalyzeOneOf ()
  {
    if (!schema.contains ("oneOf") || !schema["oneOf"].is_array ())
      return;

    const auto &branches = schema["oneOf"];
    std::vector<nlohmann::json> obj_branches;
    std::vector<nlohmann::json> props_per_branch;

    for (const auto &b : branches)
      {
        if (b.is_object ()
            && (b.value ("type", "") == "object" || b.contains ("properties")))
          {
            obj_branches.push_back (b);
            props_per_branch.push_back (b.value ("properties", nlohmann::json::object ()));
          }
      }

    if (obj_branches.empty ())
      return;

    oneof_branches = obj_branches;

    // Find discriminator candidate: property present in all branches with
    // const or single enum
    std::set<std::string> common_props;
    bool first = true;
    for (const auto &props : props_per_branch)
      {
        std::set<std::string> keys;
        for (auto it = props.begin (); it != props.end (); ++it)
          keys.insert (it.key ());

        if (first)
          {
            common_props = keys;
            first = false;
          }
        else
          {
            std::set<std::string> intersect;
            std::set_intersection (
                common_props.begin (), common_props.end (), keys.begin (),
                keys.end (), std::inserter (intersect, intersect.begin ()));
            common_props = intersect;
          }
      }

    std::unordered_map<std::string, std::vector<nlohmann::json>> candidates;
    for (const auto &prop : common_props)
      {
        std::vector<nlohmann::json> values;
        bool valid = true;
        for (const auto &props : props_per_branch)
          {
            const auto &ps = props.value (prop, nlohmann::json::object ());
            if (ps.contains ("const"))
              {
                values.push_back (ps["const"]);
              }
            else if (ps.contains ("enum") && ps["enum"].is_array ()
                     && ps["enum"].size () == 1)
              {
                values.push_back (ps["enum"][0]);
              }
            else
              {
                valid = false;
                break;
              }
          }
        if (valid)
          candidates[prop] = values;
      }

    if (!candidates.empty ())
      {
        const auto &[prop, values] = *candidates.begin ();
        OneOfDiscriminator disc;
        disc.prop = prop;
        disc.values_by_branch = values;
        for (const auto &v : values)
          disc.value_set.insert (v);
        oneof_discriminator = disc;
      }

    // Union of properties across branches
    for (const auto &props : props_per_branch)
      {
        for (auto it = props.begin (); it != props.end (); ++it)
          oneof_union_properties.insert (it.key ());
      }

    // Build candidate branch order (fewest required, then fewest properties)
    std::vector<std::tuple<int, int, int>> ordering;
    for (int i = 0; i < static_cast<int> (oneof_branches.size ()); ++i)
      {
        const auto &b = oneof_branches[i];
        int req_len = 0;
        if (b.contains ("required") && b["required"].is_array ())
          req_len = static_cast<int> (b["required"].size ());
        int prop_len = 0;
        if (b.contains ("properties") && b["properties"].is_object ())
          prop_len = static_cast<int> (b["properties"].size ());
        ordering.emplace_back (req_len, prop_len, i);
      }
    std::sort (ordering.begin (), ordering.end ());
    for (const auto &[req, prop, idx] : ordering)
      candidate_branch_order.push_back (idx);

    // Heuristic: choose preferred branch and detect tie/union mode
    if (!oneof_discriminator && !oneof_branches.empty ())
      {
        std::unordered_map<std::string, int> prop_counts;
        std::vector<std::set<std::string>> branch_props_list;
        for (const auto &b : oneof_branches)
          {
            std::set<std::string> props;
            if (b.contains ("properties") && b["properties"].is_object ())
              {
                for (auto it = b["properties"].begin ();
                     it != b["properties"].end (); ++it)
                  {
                    props.insert (it.key ());
                    prop_counts[it.key ()]++;
                  }
              }
            branch_props_list.push_back (props);
          }

        // Find unique properties per branch
        for (const auto &props : branch_props_list)
          {
            std::set<std::string> uniq;
            for (const auto &p : props)
              {
                if (prop_counts[p] == 1)
                  uniq.insert (p);
              }
            oneof_unique_props_by_branch.push_back (uniq);
          }

        // Find best branch by (required_count, property_count)
        std::vector<std::pair<int, int>> metrics;
        int best_idx = -1;
        std::pair<int, int> best_key = { INT_MAX, INT_MAX };
        for (int i = 0; i < static_cast<int> (oneof_branches.size ()); ++i)
          {
            const auto &b = oneof_branches[i];
            int req = 0;
            if (b.contains ("required") && b["required"].is_array ())
              req = static_cast<int> (b["required"].size ());
            int prop = static_cast<int> (branch_props_list[i].size ());
            auto key = std::make_pair (req, prop);
            metrics.push_back (key);
            if (key < best_key)
              {
                best_key = key;
                best_idx = i;
              }
          }
        preferred_branch_index = best_idx;

        // Check for tie (union mode)
        int tie_count = static_cast<int> (
            std::count (metrics.begin (), metrics.end (), best_key));
        oneof_union_mode = (tie_count >= 2);

        // Precompute preferred fields pre-lock
        if (best_idx >= 0)
          {
            const auto &uniq = oneof_unique_props_by_branch[best_idx];
            std::vector<std::string> ordered;
            for (const auto &p : uniq)
              ordered.push_back (p);
            if (oneof_branches[best_idx].contains ("required"))
              {
                for (const auto &r :
                     oneof_branches[best_idx]["required"])
                  {
                    std::string rs = r.get<std::string> ();
                    if (std::find (ordered.begin (), ordered.end (), rs)
                        == ordered.end ())
                      ordered.push_back (rs);
                  }
              }
            preferred_fields_prelock = ordered;
          }
      }
  }

  const nlohmann::json &
  GetCurrentSchema () const
  {
    if (!container_stack.empty ()
        && container_stack.back ().kind == ContainerFrame::Kind::Object)
      {
        const auto &frame = container_stack.back ();
        if (frame.is_root && active_branch_index
            && *active_branch_index >= 0
            && *active_branch_index < static_cast<int> (oneof_branches.size ()))
          {
            return oneof_branches[*active_branch_index];
          }
        return frame.schema;
      }

    if (active_branch_index && *active_branch_index >= 0
        && *active_branch_index < static_cast<int> (oneof_branches.size ()))
      {
        return oneof_branches[*active_branch_index];
      }
    return schema;
  }

  std::set<std::string>
  GetEffectiveRequiredFields () const
  {
    const bool in_root_object
        = !container_stack.empty ()
          && container_stack.back ().kind == ContainerFrame::Kind::Object
          && container_stack.back ().is_root;

    if (in_root_object && active_branch_index)
      return required_fields;

    if (in_root_object && !oneof_branches.empty ())
      {
        if (oneof_discriminator)
          {
            if (seen_fields.find (oneof_discriminator->prop)
                == seen_fields.end ())
              return { oneof_discriminator->prop };
            return {};
          }
        if (oneof_union_mode)
          return {};
        if (!candidate_branch_order.empty ())
          {
            int idx = candidate_branch_order[std::min (
                heuristic_idx,
                static_cast<int> (candidate_branch_order.size ()) - 1)];
            if (idx >= 0
                && idx < static_cast<int> (oneof_branches.size ()))
              {
                std::set<std::string> req;
                const auto &b = oneof_branches[idx];
                if (b.contains ("required") && b["required"].is_array ())
                  {
                    for (const auto &r : b["required"])
                      req.insert (r.get<std::string> ());
                  }
                return req;
              }
          }
      }
    return required_fields;
  }

  std::pair<std::optional<std::vector<nlohmann::json>>, std::optional<std::string>>
  ExtractEnumInfo (const nlohmann::json &prop_schema) const
  {
    if (prop_schema.contains ("const"))
      {
        return { std::vector<nlohmann::json>{ prop_schema["const"] },
                 InferJsonType (prop_schema["const"]) };
      }
    if (prop_schema.contains ("enum") && prop_schema["enum"].is_array ())
      {
        std::vector<nlohmann::json> vals;
        for (const auto &v : prop_schema["enum"])
          vals.push_back (v);

        std::set<std::string> types;
        for (const auto &v : vals)
          types.insert (InferJsonType (v));

        // Unify numeric enums to 'number'
        if (types.size () == 1
            || (types.size () == 2
                && types.count ("integer") && types.count ("number")))
          {
            std::string t = (types.count ("number") || types.count ("integer"))
                                ? "number"
                                : *types.begin ();
            return { vals, t };
          }
      }
    return { std::nullopt, std::nullopt };
  }

  void
  BeginValueContext (const nlohmann::json &prop_schema)
  {
    current_value_type = std::nullopt;
    current_enum_values = std::nullopt;
    current_enum_type = std::nullopt;
    current_value_schema = prop_schema;

    if (prop_schema.contains ("type") && prop_schema["type"].is_string ())
      current_value_type = prop_schema["type"].get<std::string> ();

    auto [enum_vals, enum_type] = ExtractEnumInfo (prop_schema);
    if (!current_value_type && enum_type)
      current_value_type = *enum_type;

    if (enum_type)
      {
        current_enum_values = enum_vals;
        current_enum_type = enum_type;
      }

    // If key is discriminator, derive enum from branch values (root only)
    const bool in_root_object
        = !container_stack.empty ()
          && container_stack.back ().kind == ContainerFrame::Kind::Object
          && container_stack.back ().is_root;
    if (!current_enum_values && in_root_object && oneof_discriminator
        && current_key == oneof_discriminator->prop)
      {
        const auto &vals = oneof_discriminator->values_by_branch;
        if (!vals.empty ())
          {
            std::set<std::string> types;
            for (const auto &v : vals)
              types.insert (InferJsonType (v));
            if (types.size () == 1)
              {
                std::string t = *types.begin ();
                current_enum_values = vals;
                current_enum_type = t;
                if (!current_value_type)
                  current_value_type = t;
              }
          }
      }
  }

  void
  StoreValue (const nlohmann::json &value)
  {
    if (!container_stack.empty ()
        && container_stack.back ().kind == ContainerFrame::Kind::Array)
      {
        container_stack.back ().item_count += 1;
        ClearValueContext ();
        return;
      }

    if (current_key.empty ())
      {
        ClearValueContext ();
        return;
      }

    std::string key_just_set = current_key;
    parsed_object[current_key] = value;
    seen_fields.insert (current_key);
    current_key.clear ();
    ClearValueContext ();

    // Maybe lock oneOf branch via discriminator
    const bool in_root_object
        = !container_stack.empty ()
          && container_stack.back ().kind == ContainerFrame::Kind::Object
          && container_stack.back ().is_root;
    if (in_root_object && oneof_discriminator && !active_branch_index
        && key_just_set == oneof_discriminator->prop)
      {
        for (int i = 0;
             i < static_cast<int> (oneof_discriminator->values_by_branch.size ());
             ++i)
          {
            if (oneof_discriminator->values_by_branch[i] == value)
              {
                active_branch_index = i;
                if (!container_stack.empty ()
                    && container_stack.back ().is_root)
                  {
                    container_stack.back ().schema = oneof_branches[i];
                  }
                // Update required fields to branch
                required_fields.clear ();
                if (oneof_branches[i].contains ("required"))
                  {
                    for (const auto &r : oneof_branches[i]["required"])
                      required_fields.insert (r.get<std::string> ());
                  }
                break;
              }
          }
      }

    // Heuristic lock: if only one branch contains this property compatible
    if (in_root_object && !active_branch_index && !oneof_branches.empty ())
      {
        std::string vtype = InferJsonType (value);
        std::vector<int> candidates;
        for (int i = 0; i < static_cast<int> (oneof_branches.size ()); ++i)
          {
            const auto &b = oneof_branches[i];
            if (!b.contains ("properties"))
              continue;
            const auto &props = b["properties"];
            if (!props.contains (key_just_set))
              continue;

            const auto &ps = props[key_just_set];
            if (ps.contains ("type")
                && ps["type"].get<std::string> () != vtype)
              {
                // Allow number/integer interop
                std::string ptype = ps["type"].get<std::string> ();
                if (!((ptype == "number" || ptype == "integer")
                      && (vtype == "number" || vtype == "integer")))
                  continue;
              }
            if (ps.contains ("const") && ps["const"] != value)
              continue;
            if (ps.contains ("enum"))
              {
                bool found = false;
                for (const auto &e : ps["enum"])
                  if (e == value)
                    found = true;
                if (!found)
                  continue;
              }
            candidates.push_back (i);
          }
        if (candidates.size () == 1)
          {
            active_branch_index = candidates[0];
            if (!container_stack.empty ()
                && container_stack.back ().is_root)
              {
                container_stack.back ().schema
                    = oneof_branches[*active_branch_index];
              }
            required_fields.clear ();
            if (oneof_branches[*active_branch_index].contains ("required"))
              {
                for (const auto &r :
                     oneof_branches[*active_branch_index]["required"])
                  required_fields.insert (r.get<std::string> ());
              }
          }
      }
  }

  bool
  CloseObject ()
  {
    if (container_stack.empty ()
        || container_stack.back ().kind != ContainerFrame::Kind::Object)
      return false;

    container_stack.pop_back ();

    const bool has_parent_context = !object_context_stack.empty ();
    if (has_parent_context)
      {
        ObjectContext parent = std::move (object_context_stack.back ());
        object_context_stack.pop_back ();
        seen_fields = std::move (parent.seen_fields);
        required_fields = std::move (parent.required_fields);
        current_key = std::move (parent.current_key);
      }

    if (!container_stack.empty ()
        && container_stack.back ().kind == ContainerFrame::Kind::Array)
      {
        container_stack.back ().item_count += 1;
        ClearValueContext ();
      }
    else
      {
        if (has_parent_context)
          {
            StoreValue (nlohmann::json::object ());
          }
        else
          {
            ClearValueContext ();
          }
      }

    state = JSONState::kExpectingCommaOrClose;
    return true;
  }

  bool
  CloseArray ()
  {
    if (container_stack.empty ()
        || container_stack.back ().kind != ContainerFrame::Kind::Array)
      return false;

    container_stack.pop_back ();

    if (!container_stack.empty ()
        && container_stack.back ().kind == ContainerFrame::Kind::Object)
      {
        StoreValue (nlohmann::json::array ());
      }
    else if (!container_stack.empty ()
             && container_stack.back ().kind == ContainerFrame::Kind::Array)
      {
        container_stack.back ().item_count += 1;
        ClearValueContext ();
      }
    else
      {
        ClearValueContext ();
      }

    state = JSONState::kExpectingCommaOrClose;
    return true;
  }

  bool
  IsSchemaSatisfied () const
  {
    for (const auto &field : required_fields)
      {
        if (seen_fields.find (field) == seen_fields.end ())
          return false;
      }
    if (!container_stack.empty ())
      return false;
    return state == JSONState::kExpectingCommaOrClose;
  }

  bool
  ProcessChar (char c)
  {
    if (std::isspace (static_cast<unsigned char> (c))
        && state != JSONState::kInStringValue
        && state != JSONState::kExpectingKeyString)
      {
        return true;
      }

    switch (state)
      {
      case JSONState::kExpectingObjectStart:
        if (c == '{')
          {
            EnterObject (schema, true);
            return true;
          }
        return false;

      case JSONState::kExpectingKeyOrClose:
        if (c == '"')
          {
            state = JSONState::kExpectingKeyString;
            partial_key.clear ();
            heuristic_idx = 0;
            heuristic_fallback_used = false;
            if (!container_stack.empty ())
              container_stack.back ().last_was_comma = false;
            return true;
          }
        if (c == '}')
          {
            if (!container_stack.empty ()
                && container_stack.back ().last_was_comma)
              return false;
            return CloseObject ();
          }
        return false;

      case JSONState::kExpectingKeyString:
        if (c == '"' && !escape_next)
          {
            current_key = partial_key;
            state = JSONState::kExpectingColon;
            if (!container_stack.empty ())
              container_stack.back ().last_was_comma = false;
            return true;
          }
        if (c == '\\' && !escape_next)
          {
            escape_next = true;
            return true;
          }
        partial_key += c;
        escape_next = false;
        return true;

      case JSONState::kExpectingColon:
        if (c == ':')
          {
            current_value_type = std::nullopt;
            current_enum_values = std::nullopt;
            current_enum_type = std::nullopt;
            current_value_schema = std::nullopt;
            if (!current_key.empty ())
              {
                const auto &cur_schema = GetCurrentSchema ();
                nlohmann::json prop_schema;
                if (cur_schema.contains ("properties")
                    && cur_schema["properties"].is_object ()
                    && cur_schema["properties"].contains (current_key))
                  {
                    prop_schema = cur_schema["properties"][current_key];
                  }
                BeginValueContext (prop_schema);
              }
            state = JSONState::kExpectingValue;
            return true;
          }
        return false;

      case JSONState::kExpectingValue:
        if (c == '"')
          {
            state = JSONState::kInStringValue;
            partial_value.clear ();
            return true;
          }
        if (c == '-' || std::isdigit (static_cast<unsigned char> (c)))
          {
            state = JSONState::kInNumberValue;
            partial_value = std::string (1, c);
            return true;
          }
        if (c == 't' || c == 'f')
          {
            state = JSONState::kInBooleanValue;
            partial_value = std::string (1, c);
            return true;
          }
        if (c == 'n')
          {
            state = JSONState::kInNullValue;
            partial_value = std::string (1, c);
            return true;
          }
        if (c == '{')
          {
            const nlohmann::json obj_schema
                = current_value_schema.value_or (nlohmann::json::object ());
            EnterObject (obj_schema, false);
            return true;
          }
        if (c == '[')
          {
            const nlohmann::json array_schema
                = current_value_schema.value_or (nlohmann::json::object ());
            EnterArray (array_schema);
            return true;
          }
        return false;

      case JSONState::kInStringValue:
        if (c == '"' && !escape_next)
          {
            // Enum validation
            if (current_enum_values
                && (current_enum_type == "string"
                    || current_value_type == "string"))
              {
                bool found = false;
                for (const auto &v : *current_enum_values)
                  {
                    if (v.is_string ()
                        && v.get<std::string> () == partial_value)
                      {
                        found = true;
                        break;
                      }
                  }
                if (!found)
                  return false;
              }
            StoreValue (partial_value);
            state = JSONState::kExpectingCommaOrClose;
            return true;
          }
        if (c == '\\' && !escape_next)
          {
            escape_next = true;
            return true;
          }
        partial_value += c;
        escape_next = false;
        return true;

      case JSONState::kInNumberValue:
        {
          const auto &vt = current_value_type;
          if (std::isdigit (static_cast<unsigned char> (c)))
            {
              partial_value += c;
              return true;
            }
          if (c == '-')
            {
              if (partial_value.empty ()
                  || partial_value.back () == 'e'
                  || partial_value.back () == 'E')
                {
                  partial_value += c;
                  return true;
                }
              return false;
            }
          if (c == '+')
            {
              if (!partial_value.empty ()
                  && (partial_value.back () == 'e'
                      || partial_value.back () == 'E'))
                {
                  partial_value += c;
                  return true;
                }
              return false;
            }
          if (c == '.')
            {
              if (vt && *vt == "integer")
                return false;
              std::string lower = partial_value;
              std::transform (lower.begin (), lower.end (), lower.begin (),
                              ::tolower);
              if (lower.find ('e') != std::string::npos
                  || partial_value.find ('.') != std::string::npos)
                return false;
              bool has_digit = false;
              for (char ch : partial_value)
                if (std::isdigit (static_cast<unsigned char> (ch)))
                  has_digit = true;
              if (!has_digit)
                return false;
              partial_value += c;
              return true;
            }
          if (c == 'e' || c == 'E')
            {
              if (vt && *vt == "integer")
                return false;
              std::string lower = partial_value;
              std::transform (lower.begin (), lower.end (), lower.begin (),
                              ::tolower);
              if (lower.find ('e') != std::string::npos)
                return false;
              bool has_digit = false;
              for (char ch : partial_value)
                if (std::isdigit (static_cast<unsigned char> (ch)))
                  has_digit = true;
              if (!has_digit)
                return false;
              partial_value += c;
              return true;
            }
          if (c == ',' || c == '}' || c == ']')
            {
              nlohmann::json value_to_store;
              bool has_digit = false;
              for (char ch : partial_value)
                if (std::isdigit (static_cast<unsigned char> (ch)))
                  has_digit = true;
              if (!has_digit)
                return false;

              if (vt && *vt == "integer")
                {
                  value_to_store = std::stoll (partial_value);
                }
              else
                {
                  value_to_store = std::stod (partial_value);
                }

              // Enum validation
              if (current_enum_values
                  && (current_enum_type == "number"
                      || current_enum_type == "integer"
                      || current_value_type == "number"
                      || current_value_type == "integer"))
                {
                  bool found = false;
                  for (const auto &ev : *current_enum_values)
                    {
                      if (ev == value_to_store)
                        found = true;
                    }
                  if (!found)
                    return false;
                }

              StoreValue (value_to_store);
              state = JSONState::kExpectingCommaOrClose;
              return ProcessChar (c);
            }
          return false;
        }

      case JSONState::kInBooleanValue:
        partial_value += c;
        if (partial_value == "true" || partial_value == "false")
          {
            bool val = (partial_value == "true");
            if (current_enum_values
                && (current_enum_type == "boolean"
                    || current_value_type == "boolean"))
              {
                bool found = false;
                for (const auto &ev : *current_enum_values)
                  if (ev.is_boolean () && ev.get<bool> () == val)
                    found = true;
                if (!found)
                  return false;
              }
            StoreValue (val);
            state = JSONState::kExpectingCommaOrClose;
            return true;
          }
        if (partial_value == "t" || partial_value == "tr"
            || partial_value == "tru" || partial_value == "f"
            || partial_value == "fa" || partial_value == "fal"
            || partial_value == "fals")
          return true;
        return false;

      case JSONState::kInNullValue:
        partial_value += c;
        if (partial_value == "null")
          {
            if (current_enum_values
                && (current_enum_type == "null"
                    || current_value_type == "null"))
              {
                bool found = false;
                for (const auto &ev : *current_enum_values)
                  if (ev.is_null ())
                    found = true;
                if (!found)
                  return false;
              }
            StoreValue (nullptr);
            state = JSONState::kExpectingCommaOrClose;
            return true;
          }
        if (partial_value == "n" || partial_value == "nu"
            || partial_value == "nul")
          return true;
        return false;

      case JSONState::kInArray:
        if (std::isspace (static_cast<unsigned char> (c)))
          return true;
        if (c == ']')
          {
            if (!container_stack.empty ()
                && container_stack.back ().kind == ContainerFrame::Kind::Array)
              {
                if (container_stack.back ().last_was_comma)
                  return false;
                if (container_stack.back ().item_count
                    < container_stack.back ().min_items)
                  return false;
                return CloseArray ();
              }
            return false;
          }
        if (!container_stack.empty ())
          container_stack.back ().last_was_comma = false;
        state = JSONState::kExpectingValue;
        return ProcessChar (c);

      case JSONState::kExpectingCommaOrClose:
        if (container_stack.empty ())
          return false;
        if (container_stack.back ().kind == ContainerFrame::Kind::Object)
          {
            if (c == ',')
              {
                state = JSONState::kExpectingKeyOrClose;
                container_stack.back ().last_was_comma = true;
                return true;
              }
            if (c == '}')
              {
                container_stack.back ().last_was_comma = false;
                return CloseObject ();
              }
            return false;
          }
        if (container_stack.back ().kind == ContainerFrame::Kind::Array)
          {
            if (c == ',')
              {
                container_stack.back ().last_was_comma = true;
                state = JSONState::kInArray;
                BeginValueContext (container_stack.back ().schema);
                return true;
              }
            if (c == ']')
              {
                if (container_stack.back ().last_was_comma)
                  return false;
                if (container_stack.back ().item_count
                    < container_stack.back ().min_items)
                  return false;
                container_stack.back ().last_was_comma = false;
                return CloseArray ();
              }
          }
        return false;

      case JSONState::kInNestedObject:
        return false;

      default:
        return true;
      }
  }

  std::vector<std::string>
  GetAllowedTokens () const
  {
    std::vector<std::string> allowed;

    switch (state)
      {
      case JSONState::kExpectingObjectStart:
        allowed.push_back ("{");
        break;

      case JSONState::kExpectingKeyOrClose:
        if (!container_stack.empty ()
            && container_stack.back ().last_was_comma)
          {
            allowed.push_back ("\"");
          }
        else
          {
            const bool in_root_object
                = !container_stack.empty ()
                  && container_stack.back ().kind == ContainerFrame::Kind::Object
                  && container_stack.back ().is_root;
            auto eff_required = GetEffectiveRequiredFields ();
            std::set<std::string> missing;
            for (const auto &f : eff_required)
              if (seen_fields.find (f) == seen_fields.end ())
                missing.insert (f);

            if (!missing.empty ())
              {
                allowed.push_back ("\"");
              }
            else
              {
                allowed.push_back ("\"");
                allowed.push_back ("}");
              }
            // Pre-lock oneOf: disallow closing until branch locked
            if (in_root_object && !oneof_branches.empty () && !active_branch_index)
              {
                allowed.erase (
                    std::remove (allowed.begin (), allowed.end (), "}"),
                    allowed.end ());
              }
          }
        break;

      case JSONState::kExpectingKeyString:
        {
          const auto &cur_schema = GetCurrentSchema ();
          std::vector<std::string> property_names;
          if (cur_schema.contains ("properties")
              && cur_schema["properties"].is_object ())
            {
              for (auto it = cur_schema["properties"].begin ();
                   it != cur_schema["properties"].end (); ++it)
                property_names.push_back (it.key ());
            }

          // Pre-lock restrictions
          const bool in_root_object
              = !container_stack.empty ()
                && container_stack.back ().kind == ContainerFrame::Kind::Object
                && container_stack.back ().is_root;
          if (in_root_object && !active_branch_index)
            {
              if (oneof_discriminator)
                {
                  property_names = { oneof_discriminator->prop };
                }
              else if (oneof_union_mode && !oneof_union_properties.empty ())
                {
                  property_names.assign (oneof_union_properties.begin (),
                                         oneof_union_properties.end ());
                }
              else if (!candidate_branch_order.empty ())
                {
                  int cand_idx = candidate_branch_order[std::min (
                      heuristic_idx,
                      static_cast<int> (candidate_branch_order.size ())
                          - 1)];
                  if (cand_idx >= 0
                      && cand_idx
                             < static_cast<int> (oneof_branches.size ()))
                    {
                      const auto &b = oneof_branches[cand_idx];
                      property_names.clear ();
                      if (b.contains ("properties"))
                        for (auto it = b["properties"].begin ();
                             it != b["properties"].end (); ++it)
                          property_names.push_back (it.key ());
                    }
                }
              else if (!oneof_union_properties.empty ())
                {
                  property_names.assign (oneof_union_properties.begin (),
                                         oneof_union_properties.end ());
                }
            }

          // Filter to unseen
          std::vector<std::string> unseen;
          for (const auto &p : property_names)
            if (seen_fields.find (p) == seen_fields.end ())
              unseen.push_back (p);

          // Filter to prefix matches
          std::vector<std::string> valid;
          for (const auto &p : unseen)
            if (StartsWith (p, partial_key))
              valid.push_back (p);

          // If missing required, filter to required only
          auto eff_required = GetEffectiveRequiredFields ();
          std::set<std::string> missing;
          for (const auto &f : eff_required)
            if (seen_fields.find (f) == seen_fields.end ())
              missing.insert (f);
          if (!missing.empty ())
            {
              std::vector<std::string> required_valid;
              for (const auto &v : valid)
                if (missing.count (v))
                  required_valid.push_back (v);
              valid = required_valid;
            }

          // If exact match, allow closing quote
          bool exact_match = false;
          for (const auto &p : unseen)
            if (p == partial_key)
              exact_match = true;

          if (exact_match)
            {
              allowed.push_back ("\"");
            }
          else if (!valid.empty ())
            {
              allowed = valid;
            }
        }
        break;

      case JSONState::kExpectingColon:
        allowed.push_back (":");
        break;

      case JSONState::kExpectingValue:
        {
          const auto &cur_schema = GetCurrentSchema ();
          nlohmann::json prop_schema;
          if (!current_key.empty () && cur_schema.contains ("properties")
              && cur_schema["properties"].contains (current_key))
            prop_schema = cur_schema["properties"][current_key];

          std::string value_type;
          if (current_value_type)
            value_type = *current_value_type;
          else if (prop_schema.contains ("type"))
            value_type = prop_schema["type"].get<std::string> ();

          bool has_enum = current_enum_values && current_enum_type;

          if (has_enum)
            {
              const std::string &et = *current_enum_type;
              if (et == "string")
                allowed.push_back ("\"");
              else if (et == "boolean")
                {
                  for (const auto &v : *current_enum_values)
                    {
                      if (v.is_boolean () && v.get<bool> () == true)
                        allowed.push_back ("t");
                      if (v.is_boolean () && v.get<bool> () == false)
                        allowed.push_back ("f");
                    }
                }
              else if (et == "null")
                {
                  for (const auto &v : *current_enum_values)
                    if (v.is_null ())
                      allowed.push_back ("n");
                }
              else if (et == "number" || et == "integer")
                {
                  for (char d = '0'; d <= '9'; ++d)
                    allowed.push_back (std::string (1, d));
                  allowed.push_back ("-");
                }
              else
                {
                  allowed = { "\"", "0", "1", "2", "3", "4", "5",
                              "6",  "7", "8", "9", "-", "t", "f",
                              "n",  "{", "[" };
                }
            }
          else
            {
              if (value_type == "string")
                allowed.push_back ("\"");
              else if (value_type == "number" || value_type == "integer")
                {
                  for (char d = '0'; d <= '9'; ++d)
                    allowed.push_back (std::string (1, d));
                  allowed.push_back ("-");
                }
              else if (value_type == "boolean")
                {
                  allowed.push_back ("t");
                  allowed.push_back ("f");
                }
              else if (value_type == "null")
                allowed.push_back ("n");
              else if (value_type == "array")
                allowed.push_back ("[");
              else if (value_type == "object")
                allowed.push_back ("{");
              else
                allowed = { "\"", "0", "1", "2", "3", "4", "5",
                            "6",  "7", "8", "9", "-", "t", "f",
                            "n",  "{", "[" };
            }
        }
        break;

      case JSONState::kInStringValue:
        if (current_enum_values
            && (current_enum_type == "string"
                || current_value_type == "string"))
          {
            std::vector<std::string> candidates;
            for (const auto &s : *current_enum_values)
              {
                if (s.is_string ()
                    && StartsWith (s.get<std::string> (), partial_value))
                  candidates.push_back (s.get<std::string> ());
              }
            bool exact = false;
            for (const auto &c : candidates)
              if (c == partial_value)
                exact = true;
            if (exact)
              {
                allowed.push_back ("\"");
              }
            else if (!candidates.empty ())
              {
                allowed = candidates;
              }
          }
        else
          {
            allowed.push_back ("ANY");
          }
        break;

      case JSONState::kInNumberValue:
        {
          for (char d = '0'; d <= '9'; ++d)
            allowed.push_back (std::string (1, d));

          std::set<std::string> will_have = seen_fields;
          if (!current_key.empty ())
            will_have.insert (current_key);
          auto eff_required = GetEffectiveRequiredFields ();
          std::set<std::string> missing;
          for (const auto &f : eff_required)
            if (will_have.find (f) == will_have.end ())
              missing.insert (f);

          const bool in_array
              = !container_stack.empty ()
                && container_stack.back ().kind == ContainerFrame::Kind::Array;
          const auto &vt = current_value_type;
          if (vt && *vt == "integer")
            {
              if (partial_value.empty ())
                allowed.push_back ("-");
              allowed.push_back (",");
              if (in_array)
                allowed.push_back ("]");
              else if (current_key.empty () || missing.empty ())
                allowed.push_back ("}");
            }
          else
            {
              if (partial_value.empty ())
                allowed.push_back ("-");
              std::string lower = partial_value;
              std::transform (lower.begin (), lower.end (), lower.begin (),
                              ::tolower);
              bool has_e = lower.find ('e') != std::string::npos;
              bool has_dot = partial_value.find ('.') != std::string::npos;

              if (!has_dot && !has_e)
                allowed.push_back (".");
              if (!has_e)
                {
                  allowed.push_back ("e");
                  allowed.push_back ("E");
                }
              if (!partial_value.empty ()
                  && (partial_value.back () == 'e'
                      || partial_value.back () == 'E'))
                {
                  allowed.push_back ("-");
                  allowed.push_back ("+");
                }
              allowed.push_back (",");
              if (in_array)
                allowed.push_back ("]");
              else if (current_key.empty () || missing.empty ())
                allowed.push_back ("}");
            }
        }
        break;

      case JSONState::kInBooleanValue:
        if (partial_value == "t")
          allowed.push_back ("r");
        else if (partial_value == "tr")
          allowed.push_back ("u");
        else if (partial_value == "tru")
          allowed.push_back ("e");
        else if (partial_value == "f")
          allowed.push_back ("a");
        else if (partial_value == "fa")
          allowed.push_back ("l");
        else if (partial_value == "fal")
          allowed.push_back ("s");
        else if (partial_value == "fals")
          allowed.push_back ("e");
        break;

      case JSONState::kInNullValue:
        if (partial_value == "n")
          allowed.push_back ("u");
        else if (partial_value == "nu")
          allowed.push_back ("l");
        else if (partial_value == "nul")
          allowed.push_back ("l");
        break;

      case JSONState::kInArray:
        {
          if (container_stack.empty ()
              || container_stack.back ().kind != ContainerFrame::Kind::Array
              || (!container_stack.back ().last_was_comma
                  && container_stack.back ().item_count
                         >= container_stack.back ().min_items))
            {
              allowed.push_back ("]");
            }

          std::string value_type;
          if (current_value_type)
            value_type = *current_value_type;

          bool has_enum = current_enum_values && current_enum_type;

          if (has_enum)
            {
              const std::string &et = *current_enum_type;
              if (et == "string")
                allowed.push_back ("\"");
              else if (et == "boolean")
                {
                  for (const auto &v : *current_enum_values)
                    {
                      if (v.is_boolean () && v.get<bool> () == true)
                        allowed.push_back ("t");
                      if (v.is_boolean () && v.get<bool> () == false)
                        allowed.push_back ("f");
                    }
                }
              else if (et == "null")
                {
                  for (const auto &v : *current_enum_values)
                    if (v.is_null ())
                      allowed.push_back ("n");
                }
              else if (et == "number" || et == "integer")
                {
                  for (char d = '0'; d <= '9'; ++d)
                    allowed.push_back (std::string (1, d));
                  allowed.push_back ("-");
                }
              else
                {
                  allowed = { "\"", "0", "1", "2", "3", "4", "5",
                              "6",  "7", "8", "9", "-", "t", "f",
                              "n",  "{", "[" };
                }
            }
          else
            {
              if (value_type == "string")
                allowed.push_back ("\"");
              else if (value_type == "number" || value_type == "integer")
                {
                  for (char d = '0'; d <= '9'; ++d)
                    allowed.push_back (std::string (1, d));
                  allowed.push_back ("-");
                }
              else if (value_type == "boolean")
                {
                  allowed.push_back ("t");
                  allowed.push_back ("f");
                }
              else if (value_type == "null")
                allowed.push_back ("n");
              else if (value_type == "array")
                allowed.push_back ("[");
              else if (value_type == "object")
                allowed.push_back ("{");
              else
                allowed = { "\"", "0", "1", "2", "3", "4", "5",
                            "6",  "7", "8", "9", "-", "t", "f",
                            "n",  "{", "[" };
            }
        }
        break;

      case JSONState::kExpectingCommaOrClose:
        {
          if (!container_stack.empty ()
              && container_stack.back ().kind == ContainerFrame::Kind::Array)
            {
              allowed.push_back (",");
              if (!container_stack.back ().last_was_comma
                  && container_stack.back ().item_count
                         >= container_stack.back ().min_items)
                allowed.push_back ("]");
              break;
            }

          auto eff_required = GetEffectiveRequiredFields ();
          std::set<std::string> missing;
          for (const auto &f : eff_required)
            if (seen_fields.find (f) == seen_fields.end ())
              missing.insert (f);

          if (!missing.empty ())
            {
              allowed.push_back (",");
            }
          else
            {
              allowed.push_back (",");
              allowed.push_back ("}");
            }
          // Pre-lock oneOf: disallow closing
          if (!oneof_branches.empty () && !active_branch_index
              && !container_stack.empty ()
              && container_stack.back ().kind == ContainerFrame::Kind::Object
              && container_stack.back ().is_root)
            {
              allowed.erase (
                  std::remove (allowed.begin (), allowed.end (), "}"),
                  allowed.end ());
            }
        }
        break;

      default:
        break;
      }

    return allowed;
  }
};

// Public interface implementations

StreamingJSONParser::StreamingJSONParser (const nlohmann::json &schema)
    : impl_ (std::make_unique<Impl> ())
{
  impl_->Initialize (schema);
}

StreamingJSONParser::~StreamingJSONParser () = default;
StreamingJSONParser::StreamingJSONParser (StreamingJSONParser &&) noexcept
    = default;
StreamingJSONParser &
StreamingJSONParser::operator= (StreamingJSONParser &&) noexcept = default;

ParserStatus
StreamingJSONParser::AddToken (const std::string &token_text)
{
  impl_->buffer += token_text;

  for (char c : token_text)
    {
      if (!impl_->ProcessChar (c))
        {
          impl_->state = JSONState::kInvalid;
          return ParserStatus::kInvalid;
        }
    }

  if (impl_->IsSchemaSatisfied ())
    return ParserStatus::kComplete;

  return ParserStatus::kValidPartial;
}

std::vector<std::string>
StreamingJSONParser::GetAllowedTokens () const
{
  return impl_->GetAllowedTokens ();
}

const nlohmann::json &
StreamingJSONParser::GetParsedObject () const
{
  return impl_->parsed_object;
}

JSONState
StreamingJSONParser::State () const
{
  return impl_->state;
}

const std::string &
StreamingJSONParser::Buffer () const
{
  return impl_->buffer;
}

void
StreamingJSONParser::Reset ()
{
  impl_->buffer.clear ();
  impl_->state = JSONState::kExpectingObjectStart;
  impl_->container_stack.clear ();
  impl_->object_context_stack.clear ();
  impl_->current_key.clear ();
  impl_->seen_fields.clear ();
  impl_->partial_key.clear ();
  impl_->partial_value.clear ();
  impl_->escape_next = false;
  impl_->current_value_schema = std::nullopt;
  impl_->current_value_type = std::nullopt;
  impl_->current_enum_values = std::nullopt;
  impl_->current_enum_type = std::nullopt;
  impl_->parsed_object = nlohmann::json::object ();
  impl_->active_branch_index = std::nullopt;
  impl_->heuristic_idx = 0;
  impl_->heuristic_fallback_used = false;
}

const std::string &
StreamingJSONParser::CurrentKey () const
{
  return impl_->current_key;
}

const std::string &
StreamingJSONParser::PartialKey () const
{
  return impl_->partial_key;
}

const std::string &
StreamingJSONParser::PartialValue () const
{
  return impl_->partial_value;
}

const std::set<std::string> &
StreamingJSONParser::SeenFields () const
{
  return impl_->seen_fields;
}

const std::set<std::string> &
StreamingJSONParser::RequiredFields () const
{
  return impl_->required_fields;
}

const std::optional<std::string> &
StreamingJSONParser::CurrentValueType () const
{
  return impl_->current_value_type;
}

const std::optional<std::vector<nlohmann::json>> &
StreamingJSONParser::CurrentEnumValues () const
{
  return impl_->current_enum_values;
}

const std::optional<std::string> &
StreamingJSONParser::CurrentEnumType () const
{
  return impl_->current_enum_type;
}

const std::vector<nlohmann::json> &
StreamingJSONParser::OneOfBranches () const
{
  return impl_->oneof_branches;
}

std::optional<int>
StreamingJSONParser::ActiveBranchIndex () const
{
  return impl_->active_branch_index;
}

const std::optional<OneOfDiscriminator> &
StreamingJSONParser::OneOfDiscriminatorInfo () const
{
  return impl_->oneof_discriminator;
}

const std::set<std::string> &
StreamingJSONParser::OneOfUnionProperties () const
{
  return impl_->oneof_union_properties;
}

bool
StreamingJSONParser::IsOneOfUnionMode () const
{
  return impl_->oneof_union_mode;
}

const std::vector<std::string> &
StreamingJSONParser::PreferredFieldsPrelock () const
{
  return impl_->preferred_fields_prelock;
}

const nlohmann::json &
StreamingJSONParser::CurrentSchema () const
{
  return impl_->GetCurrentSchema ();
}

std::set<std::string>
StreamingJSONParser::EffectiveRequiredFields () const
{
  return impl_->GetEffectiveRequiredFields ();
}

} // namespace cortext
