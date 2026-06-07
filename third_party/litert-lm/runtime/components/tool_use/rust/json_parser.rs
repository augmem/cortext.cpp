// Copyright 2025 The Google AI Edge Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use serde_json::{Map, Value};

fn parse_tool_call(map: &Map<String, Value>) -> Result<Value, String> {
    let name = map.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if name.is_empty() {
        return Err("Tool call missing name.".to_string());
    }
    let args = map.get("arguments").or(map.get("args"));

    let mut tool_call_map = Map::new();
    tool_call_map.insert("name".to_string(), serde_json::json!(name));

    match args {
        Some(Value::Object(args_map)) => {
            tool_call_map.insert("arguments".to_string(), Value::Object(args_map.clone()));
        }
        Some(_) => return Err("Tool call arguments are not an object.".to_string()),
        None => {
            tool_call_map.insert("arguments".to_string(), serde_json::json!({}));
        }
    }
    Ok(Value::Object(tool_call_map))
}

fn process_json_root(root: Value) -> Result<Vec<Value>, String> {
    let mut tool_calls = Vec::new();
    let items = match root {
        Value::Array(arr) => arr,
        _ => vec![root],
    };

    for item in items {
        if let Value::Object(map) = item {
            tool_calls.push(parse_tool_call(&map)?);
        }
    }
    Ok(tool_calls)
}

pub fn parse_json_expression(text: &str) -> Result<Vec<Value>, String> {
    let json_result: Result<Value, _> = serde_json::from_str(text);

    match json_result {
        Ok(root) => process_json_root(root),
        Err(e) => Err(e.to_string()),
    }
}
