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
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


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
    """Download required NLTK data."""
    if not NLTK_AVAILABLE:
        return
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


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


def _lemmatize_tigrinya(tokens: List[str]) -> List[str]:
    """
    Lemmatize Tigrinya tokens.
    Tigrinya has no standard lemmatizer; we use a simple identity/placeholder.
    In production, a rule-based stemmer or morphological analyzer could be added.
    """
    # Identity mapping for now - Tigrinya lemmatization is research-level
    return list(tokens)


from nltk.corpus import wordnet
from nltk import pos_tag

def _get_wordnet_pos(treebank_tag: str) -> str:
    """Map POS tag to first character used by WordNetLemmatizer"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # Default

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
    language: str = "tigrinya",
    remove_stopwords_flag: bool = True,
    lemmatize_flag: bool = True,
) -> PipelineResult:
    """
    Run the full NLP pipeline on input text.

    Args:
        text: Input text (Tigrinya or English)
        language: "tigrinya" or "english"
        remove_stopwords_flag: Whether to remove stopwords
        lemmatize_flag: Whether to lemmatize

    Returns:
        PipelineResult with all computed statistics
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
