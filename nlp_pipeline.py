"""
NLP Pipeline for Research Assignment Option 2 - Graphic Tool
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    from tigrinya_nlp import (clean, normalize, words as tigrinya_words, remove_stopwords, StopwordConfig)
    TIGRINYA_AVAILABLE = True
except ImportError:
    TIGRINYA_AVAILABLE = False

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

try:
    from langdetect import detect, LangDetectException
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

_NLTK_DATA_READY = False

@dataclass
class TermStats:
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
    raw_text: str
    cleaned_text: str
    tokens: List[str]
    tokens_no_stopwords: List[str]
    lemmas: List[str]
    frequency_table: Dict[str, int]
    term_stats: List[TermStats] = field(default_factory=list)
    language: str = "tigrinya"

def _ensure_nltk_data() -> None:
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
    ethiopic_count = sum(1 for c in text if '\u1200' <= c <= '\u137F')
    if ethiopic_count > 0:
        return "tigrinya"
    if LANGDETECT_AVAILABLE:
        try:
            detected = detect(text)
            if detected == "ti":
                return "tigrinya"
            elif detected == "en":
                return "english"
        except LangDetectException:
            pass
    return "english"

def _tokenize_english(text: str) -> List[str]:
    if not NLTK_AVAILABLE:
        raise ImportError("NLTK is required for English.")
    _ensure_nltk_data()
    return [w.lower() for w in word_tokenize(text) if w.isalnum()]

def _remove_stopwords_english(tokens: List[str]) -> List[str]:
    if not NLTK_AVAILABLE:
        return tokens
    _ensure_nltk_data()
    sw = set(nltk_stopwords.words("english"))
    return [t for t in tokens if t not in sw]

# ---------------------------------------------------------------------------
# Tigrinya Rule-Based Stemmer
# ---------------------------------------------------------------------------
# PRIMARY SOURCE: Native Tigrinya speaker verification (all examples below
# marked [NS] were confirmed directly by a native speaker during development).
# Supporting academic sources:
#   [3] Ibrahim & Mikami (2012). COLING 2012. https://aclanthology.org/C12-3043.pdf
#   [4] Swarthmore LING073 — Tigrinya Grammar
#
# HOW TIGRINYA VERB+OBJECT WORKS (native speaker explanation):
#   The verb STEM changes with the SUBJECT, and the SUFFIX changes with the OBJECT.
#   Example paradigm for "knew him":
#     ፈሊጠዮ   I/you knew him        (stem ፈሊጠ + suffix ዮ)
#     ፈሊጡዎ   he knew him           (stem ፈሊጡ + suffix ዎ)
#     ፈሊጣቶ   she knew him          (stem ፈሊጣ + suffix ቶ)  ← ቶ = 3sg.m obj after she-stem
#     ፈሊጥናዮ  we knew him           (stem ፈሊጥና + suffix ዮ)
#     ፈሊጠምዎ  they knew him         (stem ፈሊጠም + suffix ዎ)
#   This means -ዮ, -ዎ, -ቶ are all "him" but conditioned by the subject.
#   For stemming we strip whichever suffix appears — the stem may vary but
#   that is acceptable for a rule-based approach.
#
# NOUN PLURAL SUFFIXES [NS + Src 3,4]:
#   -ታት  primary plural  ከባቢ→ከባቢታት, ጻዕሪ→ጻዕርታት
#   -ኣት  consonant-stem plural (written form varies with stem vowel)
#
# FEMININE NOUN MARKER [NS]:
#   -ት  (U+1275, NOT ቲ U+1272)   ተማሃሪ→ተማሃሪት
#
# OBJECT PRONOUN SUFFIXES — all confirmed by native speaker [NS]:
#   -ኒ        1sg obj    "me"          ፈሊጡኒ
#   -ካ / -ኻ   2sg.m obj  "you(m)"      ፈሊጡካ / ፈሊጡኻ   (dual spelling, both in use)
#   -ኪ / -ኺ   2sg.f obj  "you(f)"      confirmed [NS]   (dual spelling, both in use)
#   -ና        1pl  obj   "us"          ፈሊጡና
#   -ኹም       2pl.m obj  "you(pl.m)"   ኣሎኹም             [NS + Src 4]
#   -ኽን       2pl.f obj  "you(pl.f)"   ኣሎኽን             [NS + Src 4]
#   -ዮ        3sg.m obj  "him" after I/you/we stems    ፈሊጠዮ  [NS]
#   -ዎ        3sg.m obj  "him" after he/they stems     ፈሊጡዎ  [NS]
#   -ቶ        3sg.m obj  "him" after she stem          ፈሊጣቶ  [NS]  ← ቶ = U+1276
#   -ዋ        3sg.f obj  "her"         ፈሊጡዋ             [NS]
#   -ዎም       3pl.m obj  "them(m)"     ፈሊጡዎም            [NS]
#
# POSSESSIVE SUFFIX [NS]:
#   -ይ  1sg "my"    ከልበይ
#
# NOTE ON min_stem: Each Ethiopic character = one full syllable.
#   min_stem=2 (syllables) is the correct guard — equivalent to ~4 Latin chars.
#
# NOT INCLUDED:
#   -ያት, -ናት → Amharic, NOT Tigrinya
#   -ኩ, -ኩም, -ኩን → wrong forms; correct Tigrinya is -ኹም / -ኻ / -ኪ
#   -ትካ/-ትኻ → this is she-stem (ፈሊጣ) + 2sg.m suffix (ካ/ኻ), NOT a single suffix
# ---------------------------------------------------------------------------
_TIGRINYA_SUFFIXES: List[tuple] = [
    # ── 3-syllable suffixes (longest first — prevents partial matches) ────
    ("ታት", 2),   # noun plural            ከባቢ  → ከባቢታት   [NS, Src 3,4]
    ("ዎም", 2),   # 3pl.m obj "them(m)"    ፈሊጡ → ፈሊጡዎም   [NS]
    ("ኹም", 2),   # 2pl.m obj "you(pl.m)"  ኣሎ  → ኣሎኹም    [NS, Src 4]
    ("ኽን", 2),   # 2pl.f obj "you(pl.f)"  ኣሎ  → ኣሎኽን    [NS, Src 4]
    # ── 2-syllable suffixes ───────────────────────────────────────────────
    ("ኣት", 2),   # noun plural (consonant-final stem)      [Src 3,4]
    ("ኻ",  2),   # 2sg.m obj "you(m)" labiovelar  ፈሊጡኻ   [NS]
    ("ካ",  2),   # 2sg.m obj "you(m)" regular k   ፈሊጡካ   [NS]
    ("ኺ",  2),   # 2sg.f obj "you(f)" labiovelar  ኣሎኺ    [NS]
    ("ኪ",  2),   # 2sg.f obj "you(f)" regular k   confirmed [NS]
    ("ና",  2),   # 1pl obj   "us"                 ፈሊጡና   [NS]
    ("ዮ",  2),   # 3sg.m obj "him" after I/you    ፈሊጠዮ   [NS]
    ("ዎ",  2),   # 3sg.m obj "him" after he/they  ፈሊጡዎ   [NS]
    ("ቶ",  2),   # 3sg.m obj "him" after she      ፈሊጣቶ   [NS] ← ቶ U+1276
    ("ዋ",  2),   # 3sg.f obj "her"                ፈሊጡዋ   [NS]
    # ── 1-syllable suffixes ───────────────────────────────────────────────
    ("ኒ",  2),   # 1sg obj   "me"                 ፈሊጡኒ   [NS]
    ("ይ",  2),   # 1sg poss  "my"                 ከልበይ   [NS]
    ("ት",  2),   # feminine noun marker (U+1275)   ተማሃሪት  [NS]
                 # NOTE: ት (U+1275) ≠ ቲ (U+1272)
]


def _stem_tigrinya_word(word: str) -> str:
    """
    Strip the first matching suffix from a Tigrinya word.
    Returns the unstemmed word if no rule applies.
    """
    for suffix, min_stem in _TIGRINYA_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_stem:
            return word[: -len(suffix)]
    return word


def _lemmatize_tigrinya(tokens: List[str]) -> List[str]:
    """
    Rule-based Tigrinya stemmer / lemmatizer.

    Strips common morphological suffixes documented in Ge'ez-derived Semitic NLP
    literature (possessive, plural, verb-aspect, and focus markers).
    Falls back to the identity form for tokens where no rule matches.

    Note: Full Tigrinya lemmatization would require a morphological database
    that does not yet exist as open-source software. This rule-based approach
    is academically defensible and transparent.
    """
    return [_stem_tigrinya_word(t) for t in tokens]

def _get_wordnet_pos(treebank_tag: str) -> str:
    if not NLTK_AVAILABLE:
        return "n"
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def _lemmatize_english(tokens: List[str]) -> List[str]:
    if not NLTK_AVAILABLE:
        return tokens
    _ensure_nltk_data()
    lemmatizer = WordNetLemmatizer()
    tagged_tokens = pos_tag(tokens)
    return [lemmatizer.lemmatize(word, _get_wordnet_pos(tag)) for word, tag in tagged_tokens]

def _compute_distances_and_relevance(tokens: List[str], freq_table: Dict[str, int]) -> List[TermStats]:
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
        results.append(TermStats(
            term=term, frequency=freq,
            first_position=first, last_position=last,
            distance_from_start=round(dist_start, 4),
            distance_from_end=round(dist_end, 4),
            freq_score=round(freq_score, 4),
            earliness_score=round(earliness, 4),
            compound_relevance=round(compound_relevance, 4),
        ))
    return sorted(results, key=lambda x: x.compound_relevance, reverse=True)

def run_pipeline(
    text: str,
    language: str = "auto",
    remove_stopwords_flag: bool = True,
    lemmatize_flag: bool = True,
) -> PipelineResult:
    if not text or not text.strip():
        return PipelineResult(raw_text=text, cleaned_text="", tokens=[],
                              tokens_no_stopwords=[], lemmas=[], frequency_table={}, language=language)
    if language.lower() == "auto":
        language = _detect_language(text)
    lang = language.lower()
    if lang == "tigrinya":
        if not TIGRINYA_AVAILABLE:
            raise ImportError("tigrinya-nlp required. pip install tigrinya-nlp")
        cleaned = clean(text)
        normalized = normalize(cleaned)
        tokens = tigrinya_words(normalized)
        tokens_no_sw = remove_stopwords(tokens, config=StopwordConfig.minimal()) if remove_stopwords_flag else tokens
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
        raw_text=text, cleaned_text=cleaned if lang == "tigrinya" else text,
        tokens=tokens, tokens_no_stopwords=tokens_no_sw, lemmas=lemmas,
        frequency_table=freq_table, term_stats=term_stats, language=language,
    )
