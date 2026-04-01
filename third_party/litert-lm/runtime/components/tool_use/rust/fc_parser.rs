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
use antlr_fc_tool_call_parser::{antlrfclexer, antlrfcparser, antlrfcparserlistener};
use antlrfclexer::AntlrFcLexer;
use antlrfcparser::{
    AntlrFcParser, AntlrFcParserContextType, AntlrFcParserTreeWalker, ArrayContext,
    ArrayContextAttrs, FunctionCallContext, FunctionCallContextAttrs, ObjectContext,
    ObjectContextAttrs, PairContextAttrs, ValueContext, ValueContextAttrs,
};
use antlrfcparserlistener::AntlrFcParserListener;
use protobuf::{prelude::*, proto};
use std::collections::HashSet;
use tool_call_rust_proto::{Field, ListValue, NullValue, Struct, ToolCall, ToolCalls, Value};

#[cxx::bridge(namespace = "litert::lm")]
pub mod ffi {
    struct ToolCallResult {
        serialized_tool_calls: Vec<u8>,
        is_ok: bool,
        error: String,
    }

    extern "Rust" {
        fn parse_fc_expression(text: &str) -> ToolCallResult;
    }
}

impl ffi::ToolCallResult {
    pub fn with_tool_calls(tool_calls: Vec<u8>) -> Self {
        Self { serialized_tool_calls: tool_calls, is_ok: true, error: String::new() }
    }

    pub fn with_error(error: String) -> Self {
        Self { serialized_tool_calls: Vec::new(), is_ok: false, error: error }
    }
}

impl Default for ffi::ToolCallResult {
    fn default() -> Self {
        Self { serialized_tool_calls: Vec::new(), is_ok: true, error: String::new() }
    }
}

fn strip_escape_tokens(text: &str) -> &str {
    const ESCAPE: &str = "<escape>";
    let mut s = text;
    if s.starts_with(ESCAPE) {
        s = &s[ESCAPE.len()..];
    }
    if s.ends_with(ESCAPE) {
        s = &s[..(s.len() - ESCAPE.len())];
    }
    s
}

fn parse_value(value_ctx: &ValueContext) -> Result<Value, String> {
    if let Some(escaped_string_ctx) = value_ctx.ESCAPED_STRING() {
        Ok(proto!(Value {
            string_value: strip_escape_tokens(&escaped_string_ctx.get_text()).to_string()
        }))
    } else if let Some(number_ctx) = value_ctx.NUMBER() {
        let text = number_ctx.get_text();
        if let Ok(double_val) = text.parse::<f64>() {
            Ok(proto!(Value { number_value: double_val }))
        } else {
            Err(format!("Failed to parse number: {}", text))
        }
    } else if let Some(object_ctx) = value_ctx.object() {
        let s = parse_object(&object_ctx)?;
        Ok(proto!(Value { struct_value: s }))
    } else if let Some(array_ctx) = value_ctx.array() {
        let l = parse_array(&array_ctx)?;
        Ok(proto!(Value { list_value: l }))
    } else if let Some(boolean_ctx) = value_ctx.BOOLEAN() {
        Ok(proto!(Value { bool_value: boolean_ctx.get_text() == "true" }))
    } else if let Some(_null_literal_ctx) = value_ctx.NULL_LITERAL() {
        Ok(proto!(Value { null_value: NullValue::default() }))
    } else {
        Err(format!("Unhandled value type: {}", value_ctx.get_text()))
    }
}

fn parse_array(array_ctx: &ArrayContext) -> Result<ListValue, String> {
    let mut list_value = ListValue::new();
    for value in array_ctx.value_all() {
        let parsed_value = parse_value(&value)?;
        list_value.values_mut().push(parsed_value);
    }
    Ok(list_value)
}

fn parse_object(object_ctx: &ObjectContext) -> Result<Struct, String> {
    let mut object = Struct::new();
    let mut seen_keys = HashSet::new();

    for pair_ctx in object_ctx.pair_all() {
        let id_token =
            pair_ctx.ID().ok_or_else(|| "Invalid pair in object: ID missing".to_string())?;
        let value_ctx =
            pair_ctx.value().ok_or_else(|| "Invalid pair in object: Value missing".to_string())?;

        let key = id_token.get_text();
        if key.is_empty() {
            return Err("Object key is empty".to_string());
        }

        if seen_keys.contains(&key) {
            // Log duplicate key but don't treat it as an error.
            eprintln!("Ignoring duplicate key: {}", key);
            continue;
        }
        seen_keys.insert(key.clone());

        let parsed_value = parse_value(&value_ctx)
            .map_err(|e| format!("Error parsing value for key '{}': {}", key, e))?;

        let mut field = Field::new();
        field.set_name(key);
        field.set_value(parsed_value);
        object.fields_mut().push(field);
    }
    Ok(object)
}

struct FcListener {
    tool_calls: Result<ToolCalls, String>,
}

impl FcListener {
    fn new() -> Self {
        FcListener { tool_calls: Ok(ToolCalls::default()) }
    }

    fn tool_calls(self) -> Result<ToolCalls, String> {
        self.tool_calls
    }
}

impl<'input> ParseTreeListener<'input, AntlrFcParserContextType> for FcListener {}

impl<'input> AntlrFcParserListener<'input> for FcListener {
    fn enter_functionCall(&mut self, ctx: &FunctionCallContext<'input>) {
        println!("enter_functionCall: {:?}", ctx);
        if let Ok(tool_calls) = &mut self.tool_calls {
            let mut tool_call = ToolCall::new();
            let name =
                if let Some(id_token) = ctx.ID() { id_token.get_text() } else { "".to_string() };
            tool_call.set_name(name);

            if let Some(object_ctx) = ctx.object() {
                match parse_object(&object_ctx) {
                    Ok(args) => tool_call.set_arguments(args),
                    Err(e) => {
                        self.tool_calls = Err(e);
                        return;
                    }
                }
            }

            println!("Parsed tool_call: {:?}", tool_call);
            tool_calls.tool_calls_mut().push(tool_call);
        }
    }
}

pub fn parse_fc_expression(text: &str) -> ffi::ToolCallResult {
    if text.len() == 0 {
        return ffi::ToolCallResult::default();
    }
    let lexer = AntlrFcLexer::new(InputStream::new(text));
    let mut parser = AntlrFcParser::with_strategy(
        CommonTokenStream::new(lexer),
        Box::new(BailErrorStrategy::new()),
    );
    let start = match parser.start() {
        Ok(start) => start,
        Err(e) => return ffi::ToolCallResult::with_error(e.to_string()),
    };
    match AntlrFcParserTreeWalker::walk(Box::new(FcListener::new()), start.as_ref()) {
        Ok(listener) => match listener.tool_calls() {
            Ok(tool_calls) => match tool_calls.serialize() {
                Ok(serialized_tool_calls) => {
                    ffi::ToolCallResult::with_tool_calls(serialized_tool_calls)
                }
                Err(e) => ffi::ToolCallResult::with_error(e.to_string()),
            },
            Err(e) => ffi::ToolCallResult::with_error(e.to_string()),
        },
        Err(e) => ffi::ToolCallResult::with_error(e.to_string()),
    }
}
