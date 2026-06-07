parser grammar AntlrPythonParser;


options {
    tokenVocab = AntlrPythonLexer;
}

main: expr EOF;
expr: functionCall+ | functionCallList;

key: STRING;
value: INT | FLOAT | BOOL | STRING | NONE | list | dict | object;

list: LIST_OPEN (value (SEP value)* SEP?)? LIST_CLOSE;

dict
  : OPEN_BRACE (key COLON value (SEP key COLON value)* SEP?)? CLOSE_BRACE
  ;

argVal: NAME EQ value;
argValExpr: argVal (SEP argVal)* SEP?;
object
  : NAME OPEN_PAR CLOSE_PAR
  | NAME OPEN_PAR argValExpr CLOSE_PAR
  ;

emptyFunctionCall: NAME OPEN_PAR CLOSE_PAR;
fullFunctionCall: NAME OPEN_PAR argValExpr CLOSE_PAR;

functionCall
  : fullFunctionCall
  | emptyFunctionCall
  ;

functionCallList
  : LIST_OPEN functionCall (SEP functionCall)* SEP? LIST_CLOSE
  | LIST_OPEN LIST_CLOSE;