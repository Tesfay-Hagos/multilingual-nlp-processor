from __future__ import annotations

import re
from typing import List


# Sentence boundaries for modern Tigrinya
_SENT_BOUNDARY = {"።", "?", "!", "¡"}

# Punctuation that should be separated into its own token in word tokenization.
# Decisions:
# - '/' is intentionally excluded (abbreviation marker, intra-token)
# - "'" is intentionally excluded (contraction/inner-token marker, intra-token)
# - '፡' is handled specially (word separator except between digits)
# - '.' is treated as punctuation token, but NOT a sentence boundary
_PUNCT_TOKENS = set(list("።፣፤፥፦፨?!.!,;:\"“”«»()[]{}"))

# Recognize digits (ASCII). If you later want Ethiopic numerals treated similarly,
# we can extend this set/range.
_DIGIT_RE = re.compile(r"\d")

# Split on whitespace (after converting ፡ separators appropriately)
_WHITESPACE_SPLIT_RE = re.compile(r"\s+")


def sentences(text: str) -> List[str]:
    """
    Split text into sentences for modern Tigrinya.

    Sentence boundaries:
    - ።
    - ?
    - !

    Notes:
    - '.' is NOT treated as a sentence boundary.
    - We preserve punctuation in the returned sentences.
    - Trims leading/trailing whitespace per sentence.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    out: List[str] = []
    buf: List[str] = []

    for ch in text:
        buf.append(ch)
        if ch in _SENT_BOUNDARY:
            s = "".join(buf).strip()
            if s:
                out.append(s)
            buf = []

    tail = "".join(buf).strip()
    if tail:
        out.append(tail)

    return out


def _is_digit(ch: str) -> bool:
    return bool(_DIGIT_RE.match(ch))


def _normalize_word_separators(text: str) -> str:
    """
    Handle Ethiopic word separator ፡.

    Policy:
    - Treat ፡ as a word boundary EXCEPT when between digits (e.g., 08፡30).
    - When it is a boundary, convert it to a space.
    - When between digits, keep it as-is.
    """
    if "፡" not in text:
        return text

    chars = list(text)
    out: List[str] = []

    for i, ch in enumerate(chars):
        if ch != "፡":
            out.append(ch)
            continue

        prev_ch = chars[i - 1] if i - 1 >= 0 else ""
        next_ch = chars[i + 1] if i + 1 < len(chars) else ""

        if _is_digit(prev_ch) and _is_digit(next_ch):
            out.append("፡")  # keep inside numeric token
        else:
            out.append(" ")  # treat as boundary

    return "".join(out)


def words(text: str) -> List[str]:
    """
    Split text into tokens (words and punctuation) for Tigrinya.

    Rules:
    - Split on whitespace.
    - Treat ፡ as a word boundary except between digits (08፡30 stays one token).
    - Separate punctuation into its own tokens (e.g., ።, ?, !, «, », quotes).
    - Preserve '/' and apostrophe "'" inside tokens (do not split them).
    - Do not perform stopword removal.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = _normalize_word_separators(text)

    rough = [t for t in _WHITESPACE_SPLIT_RE.split(text.strip()) if t]
    tokens: List[str] = []

    for piece in rough:
        current: List[str] = []

        def flush_current() -> None:
            if current:
                tokens.append("".join(current))
                current.clear()

        for ch in piece:
            if ch in _PUNCT_TOKENS:
                flush_current()
                tokens.append(ch)
            else:
                # includes '/', apostrophe "'", and all Ethiopic letters
                current.append(ch)

        flush_current()

    return [t for t in tokens if t]
