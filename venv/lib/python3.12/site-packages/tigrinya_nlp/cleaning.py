from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


# ============================================================
# Cleaning configuration (policy layer)
# ============================================================

@dataclass(frozen=True)
class CleaningConfig:
    """
    Cleaning policy configuration.

    NOTE:
    These configurations are *presets* for convenience, not ground truth.
    They encode reasonable defaults for common domains and tasks
    and can be overridden by users.
    """

    remove_urls: bool = True
    remove_emojis: bool = True
    remove_mentions: bool = True
    remove_hashtags: bool = True
    collapse_repeated_punct: bool = True

    @classmethod
    def research_default(cls) -> "CleaningConfig":
        """
        General-purpose research cleaning policy.

        Suitable for:
        - information retrieval
        - text classification
        - general NLP pipelines
        """
        return cls()

    @classmethod
    def social_media(cls) -> "CleaningConfig":
        """
        Cleaning policy for social media text.

        Preserves broad social signals such as:
        - emojis
        - mentions
        - hashtags
        - repeated punctuation
        """
        return cls(
            remove_urls=True,
            remove_emojis=False,
            remove_mentions=False,
            remove_hashtags=False,
            collapse_repeated_punct=False,
        )

    @classmethod
    def emotion_aware(cls) -> "CleaningConfig":
        """
        Cleaning policy for emotion- and sentiment-aware NLP tasks.

        Preserves affective cues while avoiding identity/network signals:
        - emojis preserved
        - repeated punctuation preserved
        - hashtags preserved (often emotion-bearing)
        - mentions removed (default)
        """
        return cls(
            remove_urls=True,
            remove_emojis=False,
            remove_mentions=True,
            remove_hashtags=False,
            collapse_repeated_punct=False,
        )

    @classmethod
    def formal_text(cls) -> "CleaningConfig":
        """
        Cleaning policy for formal text (news, academic, legal).

        Aggressively removes non-linguistic noise.
        """
        return cls(
            remove_urls=True,
            remove_emojis=True,
            remove_mentions=True,
            remove_hashtags=True,
            collapse_repeated_punct=True,
        )


# ============================================================
# Internal regex patterns
# ============================================================

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")

# NOTE: '/' deliberately excluded (abbreviation marker)
_REPEATED_PUNCT_RE = re.compile(r"([!?።፣፤፥፦]){2,}")

_MULTISPACE_RE = re.compile(r"\s{2,}")


# ============================================================
# Internal helpers
# ============================================================

def _is_emoji(char: str) -> bool:
    """
    Conservative emoji / pictograph detection.

    Uses Unicode general categories:
    - So: Symbol, Other
    - Sk: Symbol, Modifier
    """
    return unicodedata.category(char) in {"So", "Sk"}


def _remove_emojis(text: str) -> str:
    return "".join(ch for ch in text if not _is_emoji(ch))


def _remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def _remove_mentions(text: str) -> str:
    return _MENTION_RE.sub(" ", text)


def _remove_hashtags(text: str) -> str:
    return _HASHTAG_RE.sub(" ", text)


def _collapse_repeated_punctuation(text: str) -> str:
    return _REPEATED_PUNCT_RE.sub(r"\1", text)


def _final_whitespace_cleanup(text: str) -> str:
    return _MULTISPACE_RE.sub(" ", text).strip()


# ============================================================
# Public API
# ============================================================

def clean(text: str, config: CleaningConfig | None = None) -> str:
    """
    Clean text according to a configurable cleaning policy.

    Phase 3 responsibilities:
    - optional removal of URLs, emojis, mentions, hashtags
    - optional collapsing of repeated punctuation
    - preservation of Ethiopic letters and abbreviations
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if config is None:
        config = CleaningConfig.research_default()

    if config.remove_urls:
        text = _remove_urls(text)

    if config.remove_emojis:
        text = _remove_emojis(text)

    if config.remove_mentions:
        text = _remove_mentions(text)

    if config.remove_hashtags:
        text = _remove_hashtags(text)

    if config.collapse_repeated_punct:
        text = _collapse_repeated_punctuation(text)

    text = _final_whitespace_cleanup(text)
    return text
