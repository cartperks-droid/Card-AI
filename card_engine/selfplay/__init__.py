"""Experimental candidate generation; model scores are not battle labels."""

from .search import (
    CardVariant, LegalInventory, SearchCandidate, SearchConfig, SearchResult,
    generate_candidates,
)

__all__ = ["CardVariant", "LegalInventory", "SearchCandidate", "SearchConfig",
           "SearchResult", "generate_candidates"]
