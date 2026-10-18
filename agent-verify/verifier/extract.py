"""Pull Python code out of a model reply. Bad extraction causes false failures, so it is
tested carefully and every result says HOW the code was found."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Optional

_CLOSED_FENCE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
_OPEN_FENCE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\r?\n(.*)\Z", re.DOTALL | re.IGNORECASE)
_CODE_START = re.compile(r"^(def |class |import |from |@)", re.MULTILINE)


@dataclass
class Extraction:
    code: str
    ok: bool
    method: str   # fenced_entry | fenced_longest | unclosed_fence | raw | heuristic | none


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_code(text: str, entry_point: Optional[str] = None) -> Extraction:
    text = text or ""
    blocks = [b.strip("\n") for b in _CLOSED_FENCE.findall(text) if b.strip()]

    if blocks:
        if entry_point:
            pat = re.compile(rf"^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\b", re.MULTILINE)
            with_entry = [b for b in blocks if pat.search(b)]
            if with_entry:
                return Extraction(with_entry[-1], True, "fenced_entry")  # last = final answer
        return Extraction(max(blocks, key=len), True, "fenced_longest")

    m = _OPEN_FENCE.search(text)  # reply was cut off before the closing fence
    if m and m.group(1).strip():
        return Extraction(m.group(1).strip("\n"), True, "unclosed_fence")

    stripped = text.strip()
    if stripped and _parses(stripped):
        return Extraction(stripped, True, "raw")

    m = _CODE_START.search(text)
    if m:
        candidate = text[m.start():].strip()
        if _parses(candidate):
            return Extraction(candidate, True, "heuristic")

    return Extraction(stripped, False, "none")
