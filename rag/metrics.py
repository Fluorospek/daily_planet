"""Retrieval metrics: hit rate and MRR, both gives a score which determines did we fetch the right chunk, and
how high did it rank respectively, both measuerd aainst a golden set of questions instead of a guess
"""

from __future__ import annotations

def hit_rate(retrieved_sources, expected_sources, k=None):
    """1 if any expected source appears in the retrieved list of expected sources, else 0

    'retrieved_sources' is a ranked list of source filesnames (best first)
    'k' optionally limits how far down the ranked list counts as 'found'
    """
    if not expected_sources:
        return None
    window = retrieved_sources[:k] if k else retrieved_sources
    return 1.0 if any(s in window for s in expected_sources) else 0.0

def reciprocal_rank(retrieved_sources, expected_sources):
    """1/rank of the first correct source (1-indexed) in the retrieved list, 0 if never found
    """

    if not expected_sources:
        return None
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected_sources:
            return 1.0/rank
    return 0.0

def mean_reciprocal_rank(all_reciprocal_ranks):
    """Average reciprocal rank across all retrieval question in the set

    None entries(non-retrieval questions, like out-of-corpus cases) are excluded,
    not treated as zero, they didn't fail retrieval, they were never a retrieval question to
    begin with
    """
    scored = [r for r in all_reciprocal_ranks if r is not None]
    return sum(scored)/len(scored) if scored else 0.0
        