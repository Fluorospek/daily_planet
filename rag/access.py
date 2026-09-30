"""Access Control: role-based visibility filtering for retrieval.

Based on the section filter applied in qdrant demo. This filtering happens at the retrieval stage,
before any of the chunks reaches the LLMs, as a chunk that's not visible is not retrieved at all.
"""

from __future__ import annotations
from qdrant_client.models import Filter, FieldCondition, MatchAny

ROLE_ACCESS = {
    "public": ["public"],
    "reporter": ["public", "reporter"],
    "editor": ["public", "reporter", "editor"],
}

def visibility_filter(role):
    """Build a Qdrant filter allowing only the tiers this role can see.

    Fail closed: an unrecognized role gets the same access as "public", the least, not the most
    """
    allowed = ROLE_ACCESS.get(role, ["public"])
    return Filter(must=[
        FieldCondition(
            key="metadata.visibility",
            match=MatchAny(
                any=allowed
            )
        )
    ])

def normalize_visibility(chunks):
    """Fail closed at ingestion too: a chunk with no visibility tag is treated as the most
    restrictive tier (editor-only), never assumed public
    """
    for chunk in chunks:
        chunk["metadata"].setdefault("visibility", "editor")
    return chunks