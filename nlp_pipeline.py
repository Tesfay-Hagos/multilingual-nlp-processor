"""
NLP Pipeline for Research Assignment Option 2 - Graphic Tool

Implements all professor requirements:
- 2.1 Eliminate stopwords
- 2.2 Lemmatize terms
- 2.3 Compute frequencies
- 2.4 Measure distances from strategic points (start and end)
- 2.5 Compute compound relevance indices (50% frequency + 50% earliness)

Supports Tigrinya (Ethiopic) and English text.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Tigrinya support
try:
    from tigrinya_nlp import (
        clean,
        normalize,
        words as tigrinya_words,
        remove_stopwords,
        StopwordConfig,
    )
    TIGRINYA_AVAILABLE = True
except ImportError:
    TIGRINYA_AVAILABLE = False

# English support (NLTK)
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Language auto-detection
try:
    from langdetect import detect, LangDetectException
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0  # Deterministic results
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Module-level flag to avoid redundant NLTK data checks on every call
_NLTK_DATA_READY = False


@dataclass
class TermStats:
    """Statistics for a single term in the document."""
    term: str
    frequency: int
    first_position: int
    last_position: int
    distance_from_start: float
    distance_from_end: float
    freq_score: float
    earliness_score: float
    compound_relevance: float


@dataclass
class PipelineResult:
    """Result of the full NLP pipeline."""
    raw_text: str
    cleaned_text: str
    tokens: List[str]
    tokens_no_stopwords: List[str]
    lemmas: List[str]
    frequency_table: Dict[str, int]
    term_stats: List[TermStats] = field(default_factory=list)
    language: str = "tigrinya"


def _ensure_nltk_data() -> None:
    """Download required NLTK data. Only runs once per process via module-level flag."""
    global _NLTK_DATA_READY
    if _NLTK_DATA_READY or not NLTK_AVAILABLE:
        return

    RESOURCE_PATHS = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    }
    for name, path in RESOURCE_PATHS.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)
    _NLTK_DATA_READY = True


def _detect_language(text: str) -> str:
    """
    Auto-detect whether input text is Tigrinya (Ethiopic script) or English.

    Strategy (in order):
    1. Check for Ethiopic Unicode characters (U+1200–U+137F) — zero false positives,
       no library needed, handles all Tigrinya/Amharic text.
    2. Fall back to langdetect for Latin-script ambiguous cases.
    3. Default to 'english' if detection fails or text is too short.
    """
    # Step 1: Ethiopic script check — definitive signal, fastest path
    ethiopic_count = sum(1 for c in text if '\u1200' <= c <= '\u137F')
    if ethiopic_count > 0:
        return "tigrinya"

    # Step 2: langdetect for Latin-script text
    if LANGDETECT_AVAILABLE:
        try:
            detected = detect(text)
            if detected == "ti":
                return "tigrinya"
            elif detected == "en":
                return "english"
            # Unknown language — fall through to default
        except LangDetectException:
            pass

    # Step 3: Safe default
    return "english"


def _tokenize_tigrinya(text: str) -> List[str]:
    """Tokenize Tigrinya text."""
    if not TIGRINYA_AVAILABLE:
        raise ImportError("tigrinya-nlp is required for Tigrinya. Install: pip install tigrinya-nlp")
    cleaned = clean(text)
    normalized = normalize(cleaned)
    return tigrinya_words(normalized)


def _tokenize_english(text: str) -> List[str]:
    """Tokenize English text."""
    if not NLTK_AVAILABLE:
        raise ImportError("NLTK is required for English. Install: pip install nltk")
    _ensure_nltk_data()
    return [w.lower() for w in word_tokenize(text) if w.isalnum()]


def _remove_stopwords_tigrinya(tokens: List[str]) -> List[str]:
    """Remove Tigrinya stopwords."""
    if not TIGRINYA_AVAILABLE:
        return tokens
    return remove_stopwords(tokens, config=StopwordConfig.minimal())


def _remove_stopwords_english(tokens: List[str]) -> List[str]:
    """Remove English stopwords."""
    if not NLTK_AVAILABLE:
        return tokens
    _ensure_nltk_data()
    sw = set(nltk_stopwords.words("english"))
    return [t for t in tokens if t not in sw]


# Tigrinya suffix rules: (suffix, min_stem_length)
# Ordered longest-first to prevent partial matches
# Note: Tigrinya is a syllabary, so a 2-character stem (e.g. ሰብ 'man', ገዛ 'house') is common and complete.
_TIGRINYA_SUFFIXES = [
    ("ኩም", 2), ("ኩን", 2), ("ናት", 2), ("ያት", 2), ("ታት", 2),
    ("ዎም", 2), ("ዮ", 2), ("ዎ", 2), ("ካ", 2), ("ኪ", 2),
    ("ኩ", 2), ("ስ", 3), ("ድ", 3),
]

def _stem_tigrinya_word(word: str) -> str:
    """Strip common morphological suffixes."""
    for suffix, min_stem in _TIGRINYA_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_stem:
            return word[:-len(suffix)]
    return word

def _lemmatize_tigrinya(tokens: List[str]) -> List[str]:
    """
    Rule-based Tigrinya stemmer/lemmatizer.
    Strips common morphological suffixes documented in Ge'ez-derived Semitic NLP.
    Falls back to identity for unseen patterns.
    """
    return [_stem_tigrinya_word(t) for t in tokens]


def _get_wordnet_pos(treebank_tag: str) -> str:
    """Map POS tag to WordNet POS constant. Requires NLTK_AVAILABLE."""
    if not NLTK_AVAILABLE:
        return "n"  # safe fallback string, avoids NameError on wordnet.NOUN
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default

def _lemmatize_english(tokens: List[str]) -> List[str]:
    """Lemmatize English tokens using WordNet and POS tagging."""
    if not NLTK_AVAILABLE:
        return tokens
    _ensure_nltk_data()
    lemmatizer = WordNetLemmatizer()
    
    # Tag tokens to get parts of speech
    tagged_tokens = pos_tag(tokens)
    
    # Lemmatize based on POS
    return [lemmatizer.lemmatize(word, _get_wordnet_pos(tag)) for word, tag in tagged_tokens]


def _compute_distances_and_relevance(
    tokens: List[str],
    freq_table: Dict[str, int],
) -> List[TermStats]:
    """
    Compute distance from start/end and compound relevance for each term.
    
    - Distance from start: first_occurrence / total_tokens (0 = at start, 1 = at end)
    - Distance from end: (total_tokens - last_occurrence) / total_tokens
    - Earliness: 1 - (avg_position / total_tokens) — terms appearing earlier score higher
    - Compound relevance: 0.5 * freq_score + 0.5 * earliness_score
    """
    if not tokens:
        return []

    total = len(tokens)
    first_pos: Dict[str, int] = {}
    last_pos: Dict[str, int] = {}
    positions: Dict[str, List[int]] = {}

    for i, t in enumerate(tokens):
        if t not in first_pos:
            first_pos[t] = i
            positions[t] = []
        last_pos[t] = i
        positions[t].append(i)

    max_freq = max(freq_table.values()) if freq_table else 1
    results: List[TermStats] = []

    for term, freq in freq_table.items():
        first = first_pos.get(term, 0)
        last = last_pos.get(term, 0)
        pos_list = positions.get(term, [0])

        dist_start = first / total if total > 0 else 0.0
        # dist_end uses LAST occurrence — gives genuinely independent metric from dist_start.
        # A term spanning the full doc has low dist_start AND low dist_end.
        dist_end = (total - last - 1) / total if total > 0 else 0.0
        dist_end = max(0.0, dist_end)

        avg_pos = sum(pos_list) / len(pos_list)
        earliness = 1.0 - (avg_pos / total) if total > 0 else 0.0
        earliness = max(0.0, min(1.0, earliness))

        freq_score = freq / max_freq if max_freq > 0 else 0.0

        compound_relevance = 0.5 * freq_score + 0.5 * earliness

        results.append(
            TermStats(
                term=term,
                frequency=freq,
                first_position=first,
                last_position=last,
                distance_from_start=round(dist_start, 4),
                distance_from_end=round(dist_end, 4),
                freq_score=round(freq_score, 4),
                earliness_score=round(earliness, 4),
                compound_relevance=round(compound_relevance, 4),
            )
        )

    return sorted(results, key=lambda x: x.compound_relevance, reverse=True)


def run_pipeline(
    text: str,
    language: str = "auto",
    remove_stopwords_flag: bool = True,
    lemmatize_flag: bool = True,
) -> PipelineResult:
    """
    Run the full NLP pipeline on input text.

    Args:
        text: Input text (Tigrinya or English)
        language: "tigrinya", "english", or "auto" (auto-detects from script/langdetect)
        remove_stopwords_flag: Whether to remove stopwords (Req 2.1)
        lemmatize_flag: Whether to lemmatize (Req 2.2)

    Returns:
        PipelineResult with all computed statistics (Req 2.3, 2.4, 2.5)
    """
    if not text or not text.strip():
        return PipelineResult(
            raw_text=text,
            cleaned_text="",
            tokens=[],
            tokens_no_stopwords=[],
            lemmas=[],
            frequency_table={},
            language=language,
        )

    # Auto-detect language if not explicitly specified
    if language.lower() == "auto":
        language = _detect_language(text)
    lang = language.lower()
    if lang == "tigrinya":
        if not TIGRINYA_AVAILABLE:
            raise ImportError("tigrinya-nlp required. pip install tigrinya-nlp")
        cleaned = clean(text)
        normalized = normalize(cleaned)
        tokens = tigrinya_words(normalized)
        tokens_no_sw = _remove_stopwords_tigrinya(tokens) if remove_stopwords_flag else tokens
        lemmas = _lemmatize_tigrinya(tokens_no_sw) if lemmatize_flag else tokens_no_sw
    else:
        if not NLTK_AVAILABLE:
            raise ImportError("NLTK required. pip install nltk")
        cleaned = text
        tokens = _tokenize_english(text)
        tokens_no_sw = _remove_stopwords_english(tokens) if remove_stopwords_flag else tokens
        lemmas = _lemmatize_english(tokens_no_sw) if lemmatize_flag else tokens_no_sw

    freq_table = dict(Counter(lemmas))
    term_stats = _compute_distances_and_relevance(lemmas, freq_table)

    return PipelineResult(
        raw_text=text,
        cleaned_text=cleaned if lang == "tigrinya" else text,
        tokens=tokens,
        tokens_no_stopwords=tokens_no_sw,
        lemmas=lemmas,
        frequency_table=freq_table,
        term_stats=term_stats,
        language=language,
    )
