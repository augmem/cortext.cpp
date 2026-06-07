parser grammar AntlrFcParser;

options {
    tokenVocab = AntlrFcLexer;
}

start : functionCall EOF;

functionCall: CALL COLON ID object?;

object : OPEN_BRACE ( pair (COMMA pair)* )? CLOSE_BRACE;

pair : ID COLON value;

value
    : ESCAPED_STRING
    | NUMBER
    | BOOLEAN
    | NULL_LITERAL
    | object
    | array
    ;

array: OPEN_BRACKET ( value (COMMA value)* )? CLOSE_BRACKET;
