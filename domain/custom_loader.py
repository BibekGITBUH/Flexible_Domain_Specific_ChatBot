"""
domain/custom_loader.py
--------------------------
Lets a user build a DomainConfig at RUNTIME -- via pasted text, a .txt
upload, or a .pdf upload -- instead of writing a domain/*.py file ahead
of time. This is what "swap name/persona/knowledge_base/few_shot" means
in practice: the exact same core/ prompt-engineering and generation
code (zero-shot/few-shot/CoT, the domain guard, generation) will start
answering from WHATEVER content is passed in here, no code changes.

Used by both streamlit_app.py (file upload widget) and cli.py (a
/custom interactive command).
"""

from io import BytesIO

from domain.base import DomainConfig

# Since this project is explicitly no-retrieval, the ENTIRE knowledge
# base is injected into every single prompt (see
# core/prompt_engineering.py::_system_message). There is no chunking
# or search step. That means knowledge base size is capped by the
# model's context window, not by any code limit here -- a huge PDF
# would blow past context, slow generation, and dilute what the model
# attends to. This constant is a practical safety cap, not the
# retrieval that this project intentionally does not implement.
MAX_KB_CHARS = 12_000


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts the text layer of a PDF. Only works for PDFs that HAVE a
    text layer (e.g. exported from Word, LaTeX, or any "real" digital
    document). Scanned/photographed pages have no text layer and will
    extract as empty or near-empty -- this project does no OCR, since
    that's a vision task outside an LLM-only, no-retrieval scope.
    """
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    return text


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore").strip()


def parse_few_shot_block(raw_text: str) -> list:
    """
    Parses a simple pasted-text format for few-shot examples:

        Q: What is the return window?
        A: 30 days from the purchase date with a receipt.

        Q: Can I return a used item?
        A: Only if defective; otherwise it must be unopened.

    Blank line separates examples. Lines not starting with Q:/A: are
    ignored. Returns list of (question, answer) tuples.
    """
    examples = []
    current_q, current_a = None, None
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("Q:"):
            if current_q and current_a:
                examples.append((current_q, current_a))
            current_q = stripped[2:].strip()
            current_a = None
        elif stripped.upper().startswith("A:") and current_q is not None:
            current_a = stripped[2:].strip()
    if current_q and current_a:
        examples.append((current_q, current_a))
    return examples


def build_custom_domain(
    name: str,
    persona: str,
    knowledge_base: str,
    few_shot_examples=None,
):
    """
    Returns (DomainConfig, warning_message_or_None).
    `few_shot_examples`: list of (question, answer) tuples, or None.
    Truncates knowledge_base if it exceeds MAX_KB_CHARS and returns a
    warning string so the caller (UI/CLI) can surface it.
    """
    warning = None
    knowledge_base = (knowledge_base or "").strip()

    if len(knowledge_base) > MAX_KB_CHARS:
        knowledge_base = knowledge_base[:MAX_KB_CHARS]
        warning = (
            f"Knowledge base was truncated to {MAX_KB_CHARS} characters. "
            f"This project injects the full knowledge base into every "
            f"prompt (no retrieval) -- documents larger than this need a "
            f"RAG-based design instead."
        )

    clean_name = (name or "").strip() or "custom domain"
    clean_persona = (persona or "").strip() or (
        f"You are a helpful assistant that only answers questions about "
        f"{clean_name}, using the reference knowledge you're given."
    )

    config = DomainConfig(
        name=clean_name,
        persona=clean_persona,
        knowledge_base=knowledge_base,
        few_shot_examples=few_shot_examples or [],
    )
    return config, warning
