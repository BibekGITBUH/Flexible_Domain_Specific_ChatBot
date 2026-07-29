"""
evaluation/test_cases.py
--------------------------
Curated evaluation set for the college_regulations domain.

Each case has:
  - query: the user's question
  - expected_keywords: key facts/numbers that MUST appear in a correct
    answer (used for automatic keyword-coverage scoring)
  - in_domain: whether this question should be answered from the KB
    (True) or correctly refused as out-of-scope (False)

This is not a training set (no fine-tuning happens in this project) --
it's purely for evaluating prompt quality across zero-shot / few-shot
/ CoT.
"""

TEST_CASES = [
    {
        "query": "What is the minimum attendance required to sit for exams?",
        "expected_keywords": ["75%"],
        "in_domain": True,
    },
    {
        "query": "My attendance is 68%, can I still take the exam?",
        "expected_keywords": ["condonation", "medical certificate"],
        "in_domain": True,
    },
    {
        "query": "How much do CIA and end-semester exams count toward my grade?",
        "expected_keywords": ["40%", "60%"],
        "in_domain": True,
    },
    {
        "query": "What grade do I need to pass a course?",
        "expected_keywords": ["P", "4"],
        "in_domain": True,
    },
    {
        "query": "I got caught copying on my first assignment, what happens to me?",
        "expected_keywords": ["zero", "warning"],
        "in_domain": True,
    },
    {
        "query": "How long can I take a leave of absence for?",
        "expected_keywords": ["two semesters"],
        "in_domain": True,
    },
    {
        "query": "What's a good recipe for chocolate chip cookies?",
        "expected_keywords": [],
        "in_domain": False,
    },
    {
        "query": "Who won the last FIFA World Cup?",
        "expected_keywords": [],
        "in_domain": False,
    },
]
