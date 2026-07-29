"""
domain/college_regulations.py
------------------------------
Demo domain: undergraduate college academic regulations.

The facts below are a SAMPLE curated policy set written for this
project (loosely modeled on typical university regulations, not any
one real institution's actual policy document). In a real deployment
you would replace `KNOWLEDGE_BASE` with the actual regulation text
from your institution's handbook.

Dataset description (for submission writeup):
  - Type: curated, hand-written knowledge snippets (no scraping)
  - Size: ~20 policy facts, grouped by topic
  - Format: plain text, injected directly into the LLM system prompt
  - No PII, no copyrighted material
"""

from domain.base import DomainConfig

KNOWLEDGE_BASE = """
ATTENDANCE
- Minimum attendance required to sit for the semester exam: 73%.
- Students with 65-74% attendance may apply for condonation with a
  medical certificate or valid supporting document, subject to
  department approval.
- Below 65% attendance: the student is debarred from the exam and
  must repeat the course.

GRADING
- Grading is on a 10-point scale: O (10), A+ (9), A (8), B+ (7),
  B (6), C (5), P (4), F (0).
- A minimum grade of P (4) is required to pass a course.
- CGPA is the credit-weighted average of all completed semesters.

EXAMINATIONS
- Continuous Internal Assessment (CIA) contributes 40% of the final
  grade; the End-Semester Exam contributes 60%.
- A student who misses the end-semester exam for a valid,
  documented reason may apply for a supplementary exam within 15
  days of the result being published.
- Re-evaluation requests must be submitted within 10 days of result
  publication and require a re-evaluation fee.

ACADEMIC PROBATION
- A student whose semester GPA falls below 5.0 is placed on academic
  probation for the following semester.
- Two consecutive semesters on probation may result in the student
  being asked to repeat the year, per department policy.

LEAVE OF ABSENCE
- Students may apply for a leave of absence of up to two semesters
  for medical, personal, or family reasons, with Dean approval.
- A leave of absence does not count toward the maximum program
  duration, but attendance and academic standing rules still apply
  upon return.

PLAGIARISM & ACADEMIC INTEGRITY
- First offense: grade of zero on the assignment/exam and a formal
  warning on record.
- Second offense: automatic F grade in the course and referral to
  the academic integrity committee.
- Severe or repeated cases can lead to suspension or expulsion,
  decided by the disciplinary committee.

FEES & REGISTRATION
- Late registration is allowed up to 2 weeks into the semester with
  a late fee.
- Fee payment deadlines are published each semester; non-payment by
  the deadline can result in de-registration from courses.
""".strip()


PERSONA = (
    "You are RegBot, an academic-regulations assistant for a university. "
    "You answer ONLY using the official regulation knowledge provided to "
    "you below. You are precise, cite the specific rule (e.g. percentages, "
    "grades, deadlines) when relevant, and never invent a rule that isn't "
    "in the knowledge base. If a student's situation isn't covered by the "
    "knowledge base, say so honestly and suggest they contact the "
    "registrar's office instead of guessing."
)


# Few-shot exemplars: demonstrate the exact style/format we want the
# model to imitate (grounded, cites the specific rule, concise).
FEW_SHOT_EXAMPLES = [
    (
        "What happens if my attendance is 70%?",
        "With 70% attendance, you fall in the 65-74% band, so you are not "
        "automatically debarred. You may apply for condonation by submitting "
        "a medical certificate or other valid supporting document for "
        "department approval."
    ),
    (
        "I missed my final exam because I was hospitalized, what do I do?",
        "Since this is a valid, documented reason, you can apply for a "
        "supplementary exam. You must submit the application, along with "
        "your medical documentation, within 15 days of the result being "
        "published."
    ),
    (
        "What's the capital of France?",
        "That's outside my scope -- I can only help with this university's "
        "academic regulations (attendance, grading, exams, probation, "
        "leave, integrity, and fees)."
    ),
]

CONFIG = DomainConfig(
    name="college academic regulations",
    persona=PERSONA,
    knowledge_base=KNOWLEDGE_BASE,
    few_shot_examples=FEW_SHOT_EXAMPLES,
)
