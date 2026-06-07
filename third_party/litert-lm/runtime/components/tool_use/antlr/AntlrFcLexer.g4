lexer grammar AntlrFcLexer;

// Structural Characters
OPEN_BRACE : '{';
CLOSE_BRACE : '}';
OPEN_BRACKET : '[';
CLOSE_BRACKET : ']';
COMMA : ',';
COLON : ':';
ESCAPE : '<escape>' | '<ctrl46>' | '<|"|>';

// Literals
BOOLEAN : 'true' | 'false';
NULL_LITERAL : 'null';

// Number: Integer and floating-point, including exponents
NUMBER : '-'? INT ( FRAC | EXP )? | '-'? FRAC | '-'? EXP ;

fragment INT : '0' | [1-9] [0-9]*;
fragment FRAC : '.' [0-9]+;
fragment EXP : [eE] [+-]? [0-9]+;

ESCAPED_STRING : ESCAPE .*? ESCAPE ;

CALL : 'call';

// An identifier must start with a letter or an underscore.
// The remaining characters may be letters, numbers, underscores, dots, or dashes.
ID : [a-zA-Z_] [a-zA-Z0-9_.-]*;

// Whitespace: Skipped
WS : [ \t\n\r]+ -> skip;
