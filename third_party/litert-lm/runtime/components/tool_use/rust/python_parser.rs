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

use antlr4rust::common_token_stream::CommonTokenStream;
use antlr4rust::error_strategy::BailErrorStrategy;
use antlr4rust::tree::{ParseTree, ParseTreeListener};
use antlr4rust::InputStream;
use antlr_python_tool_call_parser::{
    antlrpythonlexer, antlrpythonparser, antlrpythonparserlistener,
};
use antlrpythonlexer::AntlrPythonLexer;
use antlrpythonparser::{
    AntlrPythonParser, AntlrPythonParserContextType, AntlrPythonParserTreeWalker,
    ArgValContextAttrs, ArgValExprContextAttrs, DictContext, DictContextAttrs,
    EmptyFunctionCallContextAttrs, FullFunctionCallContextAttrs, FunctionCallContext,
    FunctionCallContextAttrs, ListContext, ListContextAttrs, ObjectContext, ObjectContextAttrs,
    ValueContext, ValueContextAttrs,
};
use antlrpythonparserlistener::AntlrPythonParserListener;
use serde_json::{json, Map, Value};
use std::collections::HashSet;

fn strip_quotes(s: &str) -> &str {
    if s.len() >= 2 {
        let first = s.chars().next().unwrap();
        let last = s.chars().last().unwrap();
        if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
            return &s[1..s.len() - 1];
        }
    }
    s
}

fn parse_value(value_ctx: &ValueContext) -> Result<Value, String> {
    if let Some(int_token) = value_ctx.INT() {
        let text = int_token.get_text();
        if let Ok(val) = text.parse::<i64>() {
            Ok(json!(val))
        } else {
            Err(format!("Invalid int: {}", text))
        }
    } else if let Some(float_token) = value_ctx.FLOAT() {
        let text = float_token.get_text();
        if let Ok(val) = text.parse::<f64>() {
            Ok(json!(val))
        } else {
            Err(format!("Invalid float: {}", text))
        }
    } else if let Some(str_token) = value_ctx.STRING() {
        let text_raw = str_token.get_text();
        let text = strip_quotes(&text_raw);
        Ok(json!(text))
    } else if let Some(bool_token) = value_ctx.BOOL() {
        Ok(json!(bool_token.get_text() == "True"))
    } else if let Some(_) = value_ctx.NONE() {
        Ok(Value::Null)
    } else if let Some(list_ctx) = value_ctx.list() {
        parse_list(&list_ctx)
    } else if let Some(dict_ctx) = value_ctx.dict() {
        parse_dict(&dict_ctx)
    } else if let Some(obj_ctx) = value_ctx.object() {
        parse_object(&obj_ctx)
    } else {
        Err(format!("Unhandled value type: {}", value_ctx.get_text()))
    }
}

fn parse_list(list_ctx: &ListContext) -> Result<Value, String> {
    let mut list = vec![];
    for value in list_ctx.value_all() {
        let parsed_value = parse_value(&value)?;
        list.push(parsed_value);
    }
    Ok(Value::Array(list))
}

fn parse_dict(dict_ctx: &DictContext) -> Result<Value, String> {
    let mut map = Map::new();
    let keys = dict_ctx.key_all();
    let values = dict_ctx.value_all();

    if keys.len() != values.len() {
        return Err("Dictionary keys and values count mismatch".to_string());
    }

    let mut seen_keys = HashSet::new();

    for (i, key_token) in keys.iter().enumerate() {
        let key_raw = key_token.get_text();
        let key = strip_quotes(&key_raw).to_string();

        if seen_keys.contains(&key) {
            // Log duplicate key but don't treat it as an error.
            eprintln!("Ignoring duplicate key: {}", key);
            continue;
        }
        seen_keys.insert(key.clone());

        let value_ctx = &values[i];
        let parsed_value = parse_value(value_ctx)?;
        map.insert(key, parsed_value);
    }
    Ok(Value::Object(map))
}

fn parse_object(object_ctx: &ObjectContext) -> Result<Value, String> {
    let mut map = Map::new();

    let name_token = object_ctx.NAME().ok_or("Missing object name")?;
    let type_name = name_token.get_text();

    map.insert("__type__".to_string(), json!(type_name));

    if let Some(arg_val_expr) = object_ctx.argValExpr() {
        let mut seen_keys = HashSet::new();
        for arg_val in arg_val_expr.argVal_all() {
            let arg_name_token = arg_val.NAME().ok_or("Missing argument name")?;
            let arg_name = arg_name_token.get_text();

            if seen_keys.contains(&arg_name) {
                return Err(format!("Duplicate key: {}", arg_name));
            }
            seen_keys.insert(arg_name.clone());

            let val_ctx = arg_val.value().ok_or("Missing argument value")?;
            let parsed_val = parse_value(&val_ctx)?;
            map.insert(arg_name, parsed_val);
        }
    }

    Ok(Value::Object(map))
}

struct PythonListener {
    tool_calls: Result<Vec<Value>, String>,
}

impl PythonListener {
    fn new() -> Self {
        PythonListener { tool_calls: Ok(Vec::new()) }
    }

    fn tool_calls(self) -> Result<Vec<Value>, String> {
        self.tool_calls
    }
}

impl<'input> ParseTreeListener<'input, AntlrPythonParserContextType> for PythonListener {}

impl<'input> AntlrPythonParserListener<'input> for PythonListener {
    #[allow(non_snake_case)]
    fn enter_functionCall(&mut self, ctx: &FunctionCallContext<'input>) {
        if self.tool_calls.is_err() {
            return;
        }

        if let Some(full_ctx) = ctx.fullFunctionCall() {
            let name = match full_ctx.NAME() {
                Some(n) => n.get_text(),
                None => {
                    self.tool_calls = Err("Missing function name".to_string());
                    return;
                }
            };

            let mut tool_call_map = Map::new();
            tool_call_map.insert("name".to_string(), json!(name));

            let mut args_map = Map::new();
            if let Some(arg_val_expr) = full_ctx.argValExpr() {
                let mut seen_keys = HashSet::new();
                for arg_val in arg_val_expr.argVal_all() {
                    let arg_name_token = match arg_val.NAME() {
                        Some(n) => n,
                        None => {
                            self.tool_calls = Err("Missing argument name".to_string());
                            return;
                        }
                    };
                    let arg_name = arg_name_token.get_text();

                    if seen_keys.contains(&arg_name) {
                        self.tool_calls = Err(format!("Duplicate key: {}", arg_name));
                        return;
                    }
                    seen_keys.insert(arg_name.clone());

                    let val_ctx = match arg_val.value() {
                        Some(v) => v,
                        None => {
                            self.tool_calls = Err("Missing argument value".to_string());
                            return;
                        }
                    };

                    match parse_value(&val_ctx) {
                        Ok(v) => {
                            args_map.insert(arg_name, v);
                        }
                        Err(e) => {
                            self.tool_calls = Err(e);
                            return;
                        }
                    }
                }
            }
            tool_call_map.insert("arguments".to_string(), Value::Object(args_map));
            self.tool_calls.as_mut().unwrap().push(Value::Object(tool_call_map));
        } else if let Some(empty_ctx) = ctx.emptyFunctionCall() {
            let name = match empty_ctx.NAME() {
                Some(n) => n.get_text(),
                None => {
                    self.tool_calls = Err("Missing function name".to_string());
                    return;
                }
            };
            let mut tool_call_map = Map::new();
            tool_call_map.insert("name".to_string(), json!(name));
            tool_call_map.insert("arguments".to_string(), json!({}));
            self.tool_calls.as_mut().unwrap().push(Value::Object(tool_call_map));
        }
    }
}

pub fn parse_python_expression(text: &str) -> Result<Vec<Value>, String> {
    if text.len() == 0 {
        return Ok(Vec::new());
    }
    let lexer = AntlrPythonLexer::new(InputStream::new(text));
    let mut parser = AntlrPythonParser::with_strategy(
        CommonTokenStream::new(lexer),
        Box::new(BailErrorStrategy::new()),
    );
    let start = match parser.main() {
        Ok(start) => start,
        Err(e) => return Err(e.to_string()),
    };
    match AntlrPythonParserTreeWalker::walk(Box::new(PythonListener::new()), start.as_ref()) {
        Ok(listener) => listener.tool_calls(),
        Err(e) => Err(e.to_string()),
    }
}
