from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Set


# ============================================================
# Category word lists (v1)
# ============================================================

# Safe default stopwords (minimal): conjunctions + prepositions + light copula
_FUNCTION_WORDS: Set[str] = {
    # Conjunctions
    "እና",
    "ወይ",
    "ከም",

    # Prepositions
    "ኣብ",
    "ናብ",
    "ካብ",
    "ምስ",
    "ናይ",

    # Copula / light auxiliaries (conservative)
    "እዩ",
    "እያ",
    "እዮም",

    # Short forms (tokenization keeps apostrophe inside token)
    "'ያ",
    "'ዮም",
}

# Demonstratives / articles (optional)
_DEMONSTRATIVES: Set[str] = {
    "እዚ",
    "እቲ",
    "እዛ",
    "እታ",
    "እቶም",
    "እዞም",

    # Short forms
    "'ዚ",
    "'ቲ",
    "'ዛ",
    "'ታ",
    "'ቶም",
    "'ዞም",
}

# Pronouns (optional)
_PRONOUNS: Set[str] = {
    "ኣነ",
    "ንስኻ",
    "ንስኺ",
    "ንሱ",
    "ንሳ",
    "ንሕና",
    "ንስኻትኩም",
    "ንስኦም",
    "ንሳቶም",
}

# Question / filler words (optional)
_QUESTION_WORDS: Set[str] = {
    "ድዩ",
    "ድያ",
    "ዶ",
    "እንታይ",
    "ከመይ",
    "ንምንታይ",
    "ኣበይ",
    "መን",
}

# Focus / limiter words (optional)
_FOCUS_WORDS: Set[str] = {
    "ጥራይ",
}

# Negation words (PROTECTED by default)
_NEGATION_PROTECTED: Set[str] = {
    "ኣይ",
    "ኣይኮነን",
    "የለን",
}


# ============================================================
# Policy object
# ============================================================

@dataclass(frozen=True)
class StopwordConfig:
    """
    Stopword removal policy configuration.

    Category-based design:
    - select which categories are removed
    - support extra stopwords and protected words
    - protect negation by default (unless explicitly overridden)

    Notes:
    - Token-level matching only (no prefix stripping in v1).
    """
    remove_function_words: bool = True
    remove_demonstratives: bool = False
    remove_pronouns: bool = False
    remove_question_words: bool = False
    remove_focus_words: bool = False

    keep_punctuation: bool = True

    # User-defined additions
    extra_stopwords: Optional[Set[str]] = None

    # Safety mechanism
    protected_words: Optional[Set[str]] = None
    allow_remove_protected: bool = False  # must be explicit

    @classmethod
    def minimal(cls, *, keep_punctuation: bool = True, extra_stopwords: Optional[Set[str]] = None) -> "StopwordConfig":
        """
        Conservative default:
        - remove function words only
        - protect negation
        """
        return cls(
            remove_function_words=True,
            remove_demonstratives=False,
            remove_pronouns=False,
            remove_question_words=False,
            remove_focus_words=False,
            keep_punctuation=keep_punctuation,
            extra_stopwords=extra_stopwords,
            protected_words=set(_NEGATION_PROTECTED),
            allow_remove_protected=False,
        )

    @classmethod
    def topic_modeling(cls, *, keep_punctuation: bool = True, extra_stopwords: Optional[Set[str]] = None) -> "StopwordConfig":
        """
        More aggressive than minimal, useful for topic modeling:
        - remove function words
        - remove demonstratives
        - remove focus words (e.g., ጥራይ)
        - protect negation
        """
        return cls(
            remove_function_words=True,
            remove_demonstratives=True,
            remove_pronouns=False,
            remove_question_words=False,
            remove_focus_words=True,
            keep_punctuation=keep_punctuation,
            extra_stopwords=extra_stopwords,
            protected_words=set(_NEGATION_PROTECTED),
            allow_remove_protected=False,
        )

    @classmethod
    def qa_sensitive(cls, *, keep_punctuation: bool = True, extra_stopwords: Optional[Set[str]] = None) -> "StopwordConfig":
        """
        QA / dialogue friendly:
        - remove only function words (conservative)
        - keep demonstratives, pronouns, question words
        - protect negation
        """
        return cls(
            remove_function_words=True,
            remove_demonstratives=False,
            remove_pronouns=False,
            remove_question_words=False,
            remove_focus_words=False,
            keep_punctuation=keep_punctuation,
            extra_stopwords=extra_stopwords,
            protected_words=set(_NEGATION_PROTECTED),
            allow_remove_protected=False,
        )

    @classmethod
    def custom(
        cls,
        *,
        remove_function_words: bool = True,
        remove_demonstratives: bool = False,
        remove_pronouns: bool = False,
        remove_question_words: bool = False,
        remove_focus_words: bool = False,
        keep_punctuation: bool = True,
        extra_stopwords: Optional[Set[str]] = None,
        protected_words: Optional[Set[str]] = None,
        allow_remove_protected: bool = False,
    ) -> "StopwordConfig":
        """
        Explicit custom configuration.
        If protected_words is not provided, negation is protected by default.
        """
        if protected_words is None:
            protected_words = set(_NEGATION_PROTECTED)

        return cls(
            remove_function_words=remove_function_words,
            remove_demonstratives=remove_demonstratives,
            remove_pronouns=remove_pronouns,
            remove_question_words=remove_question_words,
            remove_focus_words=remove_focus_words,
            keep_punctuation=keep_punctuation,
            extra_stopwords=extra_stopwords,
            protected_words=set(protected_words),
            allow_remove_protected=allow_remove_protected,
        )


# ============================================================
# Public API
# ============================================================

def stopwords_for_config(config: StopwordConfig) -> Set[str]:
    """
    Materialize the stopword set implied by a StopwordConfig, excluding protected words.
    """
    sw: Set[str] = set()

    if config.remove_function_words:
        sw |= _FUNCTION_WORDS
    if config.remove_demonstratives:
        sw |= _DEMONSTRATIVES
    if config.remove_pronouns:
        sw |= _PRONOUNS
    if config.remove_question_words:
        sw |= _QUESTION_WORDS
    if config.remove_focus_words:
        sw |= _FOCUS_WORDS

    if config.extra_stopwords:
        sw |= set(config.extra_stopwords)

    protected = set(config.protected_words or set())
    if not config.allow_remove_protected:
        sw -= protected

    return sw


def remove_stopwords(tokens: Iterable[str], *, config: Optional[StopwordConfig] = None) -> List[str]:
    """
    Remove stopwords from a token sequence according to StopwordConfig.

    Default behavior:
    - uses StopwordConfig.minimal()
    - keeps punctuation tokens
    - protects negation words

    This function expects tokenized input (typically from words()).
    """
    if tokens is None:
        raise TypeError("tokens must be an iterable of strings")

    if config is None:
        config = StopwordConfig.minimal()

    sw = stopwords_for_config(config)
    protected = set(config.protected_words or set())

    out: List[str] = []
    for tok in tokens:
        if not isinstance(tok, str):
            raise TypeError("all tokens must be strings")

        # Punctuation policy
        if not config.keep_punctuation and _is_punctuation(tok):
            continue

        # Protected words override removal unless explicitly allowed
        if (not config.allow_remove_protected) and (tok in protected):
            out.append(tok)
            continue

        # Normal stopword removal
        if tok in sw:
            continue

        out.append(tok)

    return out


def _is_punctuation(token: str) -> bool:
    """
    Heuristic: token is punctuation if it contains no alphanumeric characters.
    Works with Ethiopic punctuation tokens (።, ፣, «, » etc.).
    """
    for ch in token:
        if ch.isalnum():
            return False
    return True
