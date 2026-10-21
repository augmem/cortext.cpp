"""Python ports of the English keyword rules the head replaces (the baselines).

`engine_rule` is a byte-exact port of IsStandingDirective in src/operations/standing_memory_internal.cpp
(origin/main 55793178): text <= 480 bytes, some clause opens (after optional "please" and comma) with one of
the markers below at a word boundary; clauses start at the text start or after : . ! ? ; or a newline;
ASCII case folding over UTF-8 bytes; a "word byte" is ASCII alnum, apostrophe, or any byte >= 0x80.

`brief_rule` is the variant described in the task brief (utterance-initial markers incl. "do not",
"don't", "make sure", "remember to", "keep in mind", "whenever", "every time", plus >= 3 content tokens).
It is not on main; it is reported as a second baseline only.
"""
from __future__ import annotations

MAX_TEXT_BYTES = 480
MARKERS = [b"always", b"never", b"from now on", b"going forward", b"in the future", b"in future", b"whenever i",
           b"every time i", b"do not ever", b"don't ever", b"i prefer", b"i'd prefer", b"i would prefer",
           b"my preference is"]
_SPACE = set(b" \t\n\v\f\r")
_BREAK = set(b":.!?;\n")


def _alnum(c: int) -> bool:
    return 48 <= c <= 57 or 65 <= c <= 90 or 97 <= c <= 122


def _word(c: int) -> bool:
    return _alnum(c) or c == 0x27 or c >= 0x80


def _lower(c: int) -> int:
    return c + 32 if 65 <= c <= 90 else c


def _match(t: bytes, pos: int, w: bytes) -> bool:
    if pos + len(w) > len(t):
        return False
    for i, ch in enumerate(w):
        if _lower(t[pos + i]) != ch:
            return False
    end = pos + len(w)
    return end == len(t) or not _word(t[end])


def _skip(t: bytes, pos: int) -> int:
    while pos < len(t) and t[pos] in _SPACE:
        pos += 1
    return pos


def _opens(t: bytes, pos: int) -> bool:
    pos = _skip(t, pos)
    if _match(t, pos, b"please"):
        pos = _skip(t, pos + 6)
        if pos < len(t) and t[pos] == 0x2C:
            pos = _skip(t, pos + 1)
    return any(_match(t, pos, m) for m in MARKERS)


def engine_rule(text: str) -> bool:
    t = text.encode("utf-8")
    if not t or len(t) > MAX_TEXT_BYTES:
        return False
    if _opens(t, 0):
        return True
    return any(c in _BREAK and _opens(t, i + 1) for i, c in enumerate(t))


BRIEF_MARKERS = ["always", "never", "from now on", "whenever", "every time", "do not", "don't", "please always",
                 "please never", "remember to", "make sure", "keep in mind", "i prefer", "my preference"]
STOP = set("a an the and or but if to of in on at for with by from is are was were be been am i you me my your it "
           "this that these those do does did not no so as up down out please".split())


def brief_rule(text: str) -> bool:
    import re
    s = re.sub(r"^\s*\[[^\]]*\]\s*", "", text)          # stream timestamp prefix
    s = re.sub(r"^\s*[A-Za-z][\w .'-]{0,30}:\s*", "", s)  # speaker prefix
    low = s.lower().lstrip()
    if not any(low.startswith(m) and (len(low) == len(m) or not low[len(m)].isalnum()) for m in BRIEF_MARKERS):
        return False
    toks = [w for w in re.findall(r"[a-z0-9']+", low) if w not in STOP]
    return len(toks) >= 3


if __name__ == "__main__":
    for s in ["Always sunny in Philadelphia", "Never mind that, just fix it.", "Under no circumstances use emoji.",
              "[2024/03/15 (Fri) 08:54] user: Always format code with syntax highlighting.",
              "Thanks! From now on reply in French.", "Please, never apologise.", "常に日本語で答えてください。"]:
        print(int(engine_rule(s)), int(brief_rule(s)), s)
