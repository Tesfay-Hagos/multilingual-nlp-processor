from __future__ import annotations

import re
import unicodedata


# ----------------------------
# Phase 2: Unicode / canonicalization
# ----------------------------

_INVISIBLE_CHARS = [
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\ufeff",  # BOM
]

# Punctuation that should trigger spacing rules
# NOTE: '/' is intentionally excluded (abbreviation marker, intra-token)
ETHIOPIC_SPACING_PUNCT = "።፣፤፥፦፡፨"
ASCII_SPACING_PUNCT = r"\.,;:\?!¡»\""

SPACING_PUNCT = ETHIOPIC_SPACING_PUNCT + ASCII_SPACING_PUNCT

_SPACE_BEFORE_PUNCT_RE = re.compile(rf"\s+([{SPACING_PUNCT}])")
_SPACE_AFTER_PUNCT_RE = re.compile(rf"([{SPACING_PUNCT}])([^\s])")

_MULTISPACE_RE = re.compile(r"[ \t]{2,}")


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """
    Normalize Unicode representation to a canonical form (default: NFC).
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return unicodedata.normalize(form, text)


def strip_control_chars(text: str, keep_newlines: bool = True) -> str:
    """
    Remove invisible/control characters while preserving readable content.
    """
    for ch in _INVISIBLE_CHARS:
        text = text.replace(ch, "")

    cleaned = []
    for ch in text:
        if keep_newlines and ch == "\n":
            cleaned.append(ch)
            continue

        if unicodedata.category(ch).startswith("C"):
            continue

        cleaned.append(ch)

    return "".join(cleaned)


def normalize_punctuation_spacing(text: str) -> str:
    """
    Normalize spacing around punctuation WITHOUT changing the punctuation symbols.
    - Remove spaces before punctuation.
    - Ensure a single space after punctuation when followed immediately by non-space.
    """
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_PUNCT_RE.sub(r"\1 \2", text)
    return text


def normalize_whitespace(text: str, preserve_newlines: bool = True) -> str:
    """
    Normalize whitespace:
    - Tabs → spaces
    - Collapse repeated spaces
    - Trim lines (if preserve_newlines)
    """
    text = text.replace("\t", " ")

    if preserve_newlines:
        lines = []
        for line in text.splitlines():
            line = _MULTISPACE_RE.sub(" ", line).strip()
            lines.append(line)
        return "\n".join(lines)

    text = text.replace("\n", " ")
    return _MULTISPACE_RE.sub(" ", text).strip()


# ----------------------------
# Phase 4: Tigrinya linguistic normalization
# ----------------------------

# Your canonicalization rules (directional):
# - ሠ-family -> ሰ-family
# - ጸ-family -> ፀ-family
# - ኀ-family -> ሀ-family
#
# IMPORTANT: No vowel normalization across different orders.
# We only map SAME ORDER within the family.

_SA_FAMILY = "ሠሡሢሣሤሥሦ"
_SE_FAMILY = "ሰሱሲሳሴስሶ"

_TSA_FAMILY = "ጸጹጺጻጼጽጾ"
_TSA_CANON = "ፀፁፂፃፄፅፆ"

_KHA_FAMILY = "ኀኁኂኃኄኅኆ"
_HA_CANON = "ሀሁሂሃሄህሆ"

_LINGUISTIC_MAP: dict[str, str] = {
    **{src: dst for src, dst in zip(_SA_FAMILY, _SE_FAMILY)},
    **{src: dst for src, dst in zip(_TSA_FAMILY, _TSA_CANON)},
    **{src: dst for src, dst in zip(_KHA_FAMILY, _HA_CANON)},
}


def normalize_tigrinya_letters(text: str) -> str:
    """
    Phase 4 linguistic normalization for Tigrinya.

    Applies ONLY the agreed directional mappings:
    - ሠ-family -> ሰ-family
    - ጸ-family -> ፀ-family
    - ኀ-family -> ሀ-family

    Does NOT alter:
    - ሐ (or its family)
    - vowels (orders are preserved via 1-to-1 mapping within families)
    - morphology
    """
    # Fast path: translate via dict (safe and explicit)
    return "".join(_LINGUISTIC_MAP.get(ch, ch) for ch in text)


# ----------------------------
# Public normalize() API
# ----------------------------

def normalize(text: str, mode: str = "conservative") -> str:
    """
    Normalization pipeline.

    Modes:
    - "conservative": Phase 2 only (Unicode + control chars + ፧->? + spacing + whitespace)
    - "linguistic": Phase 2 + Phase 4 (Tigrinya letter-family normalization)

    Notes:
    - ፧ is canonicalized to '?' (modern usage).
    - '/' is not treated as punctuation for spacing.
    """
    if mode not in {"conservative", "linguistic"}:
        raise ValueError("mode must be 'conservative' or 'linguistic'")

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Phase 2
    text = normalize_unicode(text)
    text = strip_control_chars(text)

    # Canonicalize traditional Ethiopic question mark
    text = text.replace("፧", "?")

    text = normalize_punctuation_spacing(text)
    text = normalize_whitespace(text)

    # Phase 4 (optional)
    if mode == "linguistic":
        text = normalize_tigrinya_letters(text)

    return text
