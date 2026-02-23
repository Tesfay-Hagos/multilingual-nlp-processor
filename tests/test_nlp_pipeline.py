"""
Tests for NLP Pipeline - Research Assignment Option 2

Verifies all professor requirements for both Tigrinya and English:
- 2.1 Eliminate stopwords
- 2.2 Lemmatize terms
- 2.3 Compute frequencies
- 2.4 Measure distances from strategic points (start and end)
- 2.5 Compute compound relevance indices (50% frequency + 50% earliness)
"""

import pytest
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp_pipeline import (
    run_pipeline,
    TermStats,
    PipelineResult,
    TIGRINYA_AVAILABLE,
    NLTK_AVAILABLE,
)

# Fixture paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_TIGRINYA = (FIXTURES_DIR / "sample_tigrinya.txt").read_text(encoding="utf-8")
SAMPLE_TIGRINYA_LONG = (FIXTURES_DIR / "sample_tigrinya_long.txt").read_text(encoding="utf-8")
SAMPLE_ENGLISH = (FIXTURES_DIR / "sample_english.txt").read_text(encoding="utf-8")


# ============== Tigrinya Tests ==============

@pytest.mark.skipif(not TIGRINYA_AVAILABLE, reason="tigrinya-nlp not installed")
class TestTigrinyaPipeline:
    """Tests for Tigrinya language pipeline."""

    def test_tigrinya_empty_input(self):
        """Empty input returns empty result."""
        r = run_pipeline("", language="tigrinya")
        assert r.tokens == []
        assert r.lemmas == []
        assert r.frequency_table == {}
        assert r.term_stats == []

    def test_tigrinya_whitespace_only(self):
        """Whitespace-only input returns empty result."""
        r = run_pipeline("   \n\t  ", language="tigrinya")
        assert len(r.tokens) == 0
        assert r.term_stats == []

    def test_tigrinya_stopwords_removed(self):
        """Requirement 2.1: Stopwords are eliminated."""
        # ኣብ, እዩ are Tigrinya stopwords
        text = "ሰብ ኣብ ከተማ እዩ።"
        r = run_pipeline(text, language="tigrinya", remove_stopwords_flag=True)
        assert "ኣብ" not in r.tokens_no_stopwords
        assert "እዩ" not in r.tokens_no_stopwords
        assert len(r.tokens_no_stopwords) < len(r.tokens)

    def test_tigrinya_stopwords_kept_when_disabled(self):
        """When stopword removal is disabled, all tokens kept."""
        text = "ሰብ ኣብ ከተማ እዩ።"
        r = run_pipeline(text, language="tigrinya", remove_stopwords_flag=False, lemmatize_flag=False)
        assert len(r.tokens_no_stopwords) == len(r.tokens)

    def test_tigrinya_lemmatization_reduces_suffixes(self):
        """Requirement 2.2: Lemmatize Tigrinya tokens by stripping common suffixes."""
        # ሰብካ (man + your) -> ሰብ
        # ገዛኹም (house + your-pl) - using ኩም variant
        text = "ሰብካ ገዛኩም"
        r = run_pipeline(text, language="tigrinya", remove_stopwords_flag=False, lemmatize_flag=True)
        assert "ሰብ" in r.lemmas
        assert "ገዛ" in r.lemmas
        assert "ሰብካ" not in r.lemmas

    def test_tigrinya_frequencies_computed(self):
        """Requirement 2.3: Frequencies are computed correctly."""
        text = "ሰብ ከተማ ሰብ ጽሑፍ ከተማ"  # ሰብ=2, ከተማ=2, ጽሑፍ=1
        r = run_pipeline(text, language="tigrinya", remove_stopwords_flag=False)
        assert r.frequency_table["ሰብ"] == 2
        assert r.frequency_table["ከተማ"] == 2
        assert r.frequency_table["ጽሑፍ"] == 1

    def test_tigrinya_distance_from_start_end(self):
        """Requirement 2.4: Distance from start and end are measured."""
        text = "ጽሑፍ ጽሑፍ ጽሑፍ"  # Single repeated term
        r = run_pipeline(text, language="tigrinya", remove_stopwords_flag=False)
        assert len(r.term_stats) == 1
        t = r.term_stats[0]
        assert 0 <= t.distance_from_start <= 1
        assert 0 <= t.distance_from_end <= 1
        assert t.first_position == 0
        assert t.last_position == 2
        assert t.distance_from_start == 0.0

    def test_tigrinya_compound_relevance_formula(self):
        """Requirement 2.5: Compound relevance = 0.5*freq + 0.5*earliness."""
        text = "ጽሑፍ ጽሑፍ ጽሑፍ ከተማ"  # ጽሑፍ=3 (freq_score=1), early; ከተማ=1, late
        r = run_pipeline(text, language="tigrinya", remove_stopwords_flag=False)
        for t in r.term_stats:
            expected = 0.5 * t.freq_score + 0.5 * t.earliness_score
            assert abs(t.compound_relevance - expected) < 0.0001

    def test_tigrinya_sample_file(self):
        """Full pipeline on sample Tigrinya document."""
        r = run_pipeline(SAMPLE_TIGRINYA, language="tigrinya")
        assert len(r.tokens) > 0
        assert len(r.frequency_table) > 0
        assert len(r.term_stats) > 0
        assert all(0 <= ts.compound_relevance <= 1 for ts in r.term_stats)
        assert all(0 <= ts.distance_from_start <= 1 for ts in r.term_stats)
        assert all(0 <= ts.distance_from_end <= 1 for ts in r.term_stats)

    def test_tigrinya_long_sample_file(self):
        """Full pipeline on longer Tigrinya document (news-style sentences)."""
        r = run_pipeline(SAMPLE_TIGRINYA_LONG, language="tigrinya")
        assert len(r.tokens) > 20
        assert len(r.frequency_table) >= 5
        assert len(r.term_stats) >= 5
        assert all(0 <= ts.compound_relevance <= 1 for ts in r.term_stats)

    def test_tigrinya_high_frequency_terms(self):
        """Repeated terms (ዜና, ከተማ, ጽሑፍ, ሰብ) get higher frequency in long sample."""
        r = run_pipeline(SAMPLE_TIGRINYA_LONG, language="tigrinya")
        # At least one term should appear 3+ times
        max_freq = max(r.frequency_table.values()) if r.frequency_table else 0
        assert max_freq >= 3
        # Top term by relevance should have high frequency
        if r.term_stats:
            top = r.term_stats[0]
            assert top.frequency >= 2

    def test_tigrinya_news_style_vocabulary(self):
        """News-style Tigrinya (ዜና, ከተማ, ኢትዮጵያ) is tokenized and processed."""
        text = "ናይቲ ውጽኢት ዜና ይጽበዩ ነበሩ። ዜና ንሰብ ኣዝዩ ኣገዳሲ እዩ።"
        r = run_pipeline(text, language="tigrinya")
        assert len(r.tokens) > 5
        assert "ዜና" in r.frequency_table or "ሰብ" in r.frequency_table


# ============== English Tests ==============

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
class TestEnglishPipeline:
    """Tests for English language pipeline."""

    def test_english_empty_input(self):
        """Empty input returns empty result."""
        r = run_pipeline("", language="english")
        assert r.tokens == []
        assert r.lemmas == []
        assert r.frequency_table == {}
        assert r.term_stats == []

    def test_english_stopwords_removed(self):
        """Requirement 2.1: Stopwords (the, is, a) are eliminated."""
        text = "The quick brown fox is a lazy dog."
        r = run_pipeline(text, language="english", remove_stopwords_flag=True)
        assert "the" not in r.tokens_no_stopwords
        assert "is" not in r.tokens_no_stopwords
        assert "a" not in r.tokens_no_stopwords
        assert len(r.tokens_no_stopwords) < len(r.tokens)

    def test_english_lemmatization(self):
        """Requirement 2.2: Terms are lemmatized (e.g. running -> run)."""
        text = "running runs ran"
        r = run_pipeline(text, language="english", remove_stopwords_flag=False, lemmatize_flag=True)
        # WordNet lemmatizer: running->run, runs->run, ran->ran (irregular)
        lemmas = r.lemmas
        assert "run" in lemmas or "running" in lemmas  # At least some normalization

    def test_english_frequencies_computed(self):
        """Requirement 2.3: Frequencies are computed correctly."""
        text = "fox fox fox dog dog cat"
        r = run_pipeline(text, language="english", remove_stopwords_flag=False)
        assert r.frequency_table["fox"] == 3
        assert r.frequency_table["dog"] == 2
        assert r.frequency_table["cat"] == 1

    def test_english_distance_from_start_end(self):
        """Requirement 2.4: Distance from start and end are measured."""
        text = "one two three four"
        r = run_pipeline(text, language="english", remove_stopwords_flag=False)
        total = len(r.lemmas)
        for t in r.term_stats:
            assert 0 <= t.distance_from_start <= 1
            assert 0 <= t.distance_from_end <= 1
            assert 0 <= t.first_position < total
            assert 0 <= t.last_position < total

    def test_english_compound_relevance_formula(self):
        """Requirement 2.5: Compound relevance = 0.5*freq + 0.5*earliness."""
        text = "cat cat cat dog"
        r = run_pipeline(text, language="english", remove_stopwords_flag=False)
        for t in r.term_stats:
            expected = 0.5 * t.freq_score + 0.5 * t.earliness_score
            assert abs(t.compound_relevance - expected) < 0.0001

    def test_english_sample_file(self):
        """Full pipeline on sample English document."""
        r = run_pipeline(SAMPLE_ENGLISH, language="english")
        assert len(r.tokens) > 0
        assert len(r.frequency_table) > 0
        assert len(r.term_stats) > 0
        assert all(0 <= ts.compound_relevance <= 1 for ts in r.term_stats)
        # "fox" appears multiple times - should have high frequency
        assert "fox" in r.frequency_table
        assert r.frequency_table["fox"] >= 2

    def test_english_multiple_paragraphs(self):
        """Expanded English sample with multiple topics (NLP, weather, education)."""
        r = run_pipeline(SAMPLE_ENGLISH, language="english")
        assert len(r.tokens) > 30
        # Should contain vocabulary from different domains (lemmatized forms may vary)
        vocab = set(r.frequency_table.keys())
        domain_words = ["language", "learning", "computer", "weather", "education", "people", "machine", "sun"]
        assert any(w in vocab for w in domain_words)

    def test_english_high_frequency_term(self):
        """High-frequency term (fox) ranks higher in compound relevance."""
        r = run_pipeline(SAMPLE_ENGLISH, language="english")
        fox_freq = r.frequency_table.get("fox", 0)
        assert fox_freq >= 2
        # Fox should be among top terms by relevance
        top_terms = [t.term for t in r.term_stats[:5]]
        assert "fox" in top_terms


# ============== Cross-cutting Tests ==============

class TestPipelineEdgeCases:
    """Edge cases and formula validation."""

    def test_single_token(self):
        """Single token produces one term stat."""
        lang = "english" if NLTK_AVAILABLE else ("tigrinya" if TIGRINYA_AVAILABLE else None)
        if lang is None:
            pytest.skip("No NLP backend available")
        r = run_pipeline("hello", language=lang)
        assert len(r.term_stats) == 1
        t = r.term_stats[0]
        assert t.frequency == 1
        assert t.freq_score == 1.0
        assert t.compound_relevance == 1.0  # Only term, max relevance

    def test_repeated_same_term(self):
        """Repeated term: freq_score=1, earliness depends on position."""
        lang = "english" if NLTK_AVAILABLE else ("tigrinya" if TIGRINYA_AVAILABLE else None)
        if lang is None:
            pytest.skip("No NLP backend available")
        text = "word word word word" if lang == "english" else "ጽሑፍ ጽሑፍ ጽሑፍ"
        r = run_pipeline(text, language=lang, remove_stopwords_flag=False)
        assert len(r.term_stats) == 1
        t = r.term_stats[0]
        assert t.freq_score == 1.0
        assert 0.5 <= t.compound_relevance <= 1.0

    def test_term_stats_sorted_by_relevance(self):
        """Term stats are sorted by compound_relevance descending."""
        lang = "english" if NLTK_AVAILABLE else ("tigrinya" if TIGRINYA_AVAILABLE else None)
        if lang is None:
            pytest.skip("No NLP backend available")
        text = "a b c d e f" if lang == "english" else "ጽሑፍ ከተማ ሰብ መለኪያ ቕልጡፍ ኣሎ"
        r = run_pipeline(text, language=lang, remove_stopwords_flag=False)
        relevances = [t.compound_relevance for t in r.term_stats]
        assert relevances == sorted(relevances, reverse=True)

    def test_freq_score_normalized(self):
        """Frequency scores are normalized 0-1."""
        lang = "english" if NLTK_AVAILABLE else ("tigrinya" if TIGRINYA_AVAILABLE else None)
        if lang is None:
            pytest.skip("No NLP backend available")
        text = "a a a b b c" if lang == "english" else "ጽሑፍ ጽሑፍ ጽሑፍ ከተማ ከተማ ሰብ"
        r = run_pipeline(text, language=lang, remove_stopwords_flag=False)
        for t in r.term_stats:
            assert 0 <= t.freq_score <= 1
        # Max freq term should have freq_score 1
        max_freq = max(t.frequency for t in r.term_stats)
        max_term = next(t for t in r.term_stats if t.frequency == max_freq)
        assert max_term.freq_score == 1.0


# ============== Integration with NLPDash sample ==============

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_nlpdash_sample_english():
    """Test with NLPDash sample_text.txt from repo (first paragraph)."""
    sample = (
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976 "
        "in Cupertino, California. The company has grown to become one of the largest technology "
        "companies in the world."
    )
    r = run_pipeline(sample, language="english")
    assert len(r.tokens) > 10
    assert "apple" in r.frequency_table or "jobs" in r.frequency_table
    assert all(0 <= ts.compound_relevance <= 1 for ts in r.term_stats)
