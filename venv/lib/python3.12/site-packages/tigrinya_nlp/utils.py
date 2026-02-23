"""Utility helpers."""

from typing import Iterable, List


def to_text(value) -> str:
    """Convert common input types to a string."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def deduplicate_tokens(tokens: Iterable[str]) -> List[str]:
    """Preserve order while removing duplicate tokens."""
    seen = set()
    unique = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique
