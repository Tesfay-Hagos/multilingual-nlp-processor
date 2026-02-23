"""
Tigrinya NLP preprocessing toolkit.
"""

from .cleaning import CleaningConfig, clean
from .normalization import normalize
from .tokenization import sentences, words
from .stopwords import StopwordConfig, remove_stopwords, stopwords_for_config

__all__ = [
    "clean",
    "CleaningConfig",
    "normalize",
    "sentences",
    "words",
    "remove_stopwords",
    "stopwords_for_config",
    "StopwordConfig",
]
