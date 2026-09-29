"""Tell source code apart from prose, line by line."""
from __future__ import annotations

import re

_CODE_START = re.compile(
    r"^\s*(def |class |import |from \S+ import |#include|using |package |function |const |let |var |"
    r"public |private |protected |static |return\b|@\w+|</?\w+[^>]*>$|\$ )"
)
_CODE_SYMBOLS = re.compile(r"(=>|->|::|\)\s*\{|\};?$|;\s*$|\w\(.*\)\s*[:{]?$|^\s*[}\])]+[;,]?$|\w+\s*[+\-*/]?=\s*\S)")
_FENCE = re.compile(r"^\s*(```|~~~)")


def is_code_line(line: str) -> bool:
    s = line.rstrip()
    if not s.strip():
        return False
    if _FENCE.match(s) or _CODE_START.match(s):
        return True
    if s.startswith(("    ", "\t")) and not s.strip().endswith((".", "?", "!")) and len(s.split()) < 12:
        return True
    words = s.split()
    prose_like = len(words) >= 6 and s.strip()[-1] in ".?!:," and "(" not in s and "=" not in s
    return bool(_CODE_SYMBOLS.search(s)) and not prose_like


def classify(text: str) -> str:
    """Return "code" if most non-empty lines look like code, else "text"."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return "text"
    if any(_FENCE.match(ln) for ln in lines):
        return "code"
    code = sum(is_code_line(ln) for ln in lines)
    return "code" if code / len(lines) >= 0.6 else "text"
