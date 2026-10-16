"""The golden evaluation set: real newsroom questions with known-correct
sources, so retrieval and answer quality can be measured, not guessed at.
"""

GOLDEN_SET = [
    {
        "id": "cost-overruns",
        "question": "How is the city protecting taxpayers from cost overruns on the arena?",
        "expected_sources": ["arena-budget-approved.md"],
        "answerable": True,
    },
    {
        "id": "lawsuit-reason",
        "question": "Why did a community group sue over the arena budget?",
        "expected_sources": ["arena-lawsuit-filed.md"],
        "answerable": True,
    },
    {
        "id": "case-number",
        "question": "24-CV-1099",
        "expected_sources": ["arena-lawsuit-filed.md"],
        "answerable": True,
    },
    {
        "id": "mayor-statement",
        "question": "What does the mayor say about the arena budget?",
        "expected_sources": ["mayor-statement.md"],
        "answerable": True,
    },
    {
        "id": "transit-funding",
        "question": "Are officials arguing about paying for buses and trains ahead of the fall vote?",
        "expected_sources": ["transit-budget-debate.md"],
        "answerable": True,
    },
    {
        "id": "jobs-impact",
        "question": "What do local businesses think about the arena's promised jobs?",
        "expected_sources": ["arena-jobs-impact.md"],
        "answerable": True,
    },
    {
        "id": "multi-hop-conflict",
        "question": "Which officials who voted for the arena budget have also previously worked with the stadium's developer?",
        "expected_sources": [],
        "answerable": False,
        "note": (
            "A genuinely GraphRAG-shaped question (Week 5, Video 3 / quiz Q5). "
            "Flat chunk retrieval structurally cannot answer this - the connection "
            "lives across two separate documents, not inside any single chunk. "
            "Deliberately included as a known-hard case: a good eval harness "
            "measures where the system is honestly weak, not just where it's strong."
        ),
    },
    {
        "id": "out-of-corpus",
        "question": "How is the city handling the recent water main break on Elm Street?",
        "expected_sources": [],
        "answerable": False,
        "note": "Genuinely fabricated, nothing in the archive covers this. Tests the honest refusal.",
    },
]
