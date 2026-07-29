"""
domain/base.py
---------------
Defines the shape every domain module must provide. This is what makes
the chatbot "domain-agnostic core + swappable domain data": to add a
new domain (HR policy, government forms, a research paper, etc.) you
only write a new file like domain/college_regulations.py that fills in
this dataclass -- nothing in core/ changes.

Since this project is explicitly LLM-only / no-retrieval, the domain's
"knowledge" is not stored in a vector DB. It is a curated block of text
(`knowledge_base`) that gets injected directly into the system prompt.
This only works because the knowledge is small and curated (a few KB
of text) -- for a large corpus you'd need retrieval, which is
intentionally out of scope here.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DomainConfig:
    name: str
    persona: str                       # who the bot is / how it should behave
    knowledge_base: str                # curated facts injected into every prompt
    few_shot_examples: List[Tuple[str, str]] = field(default_factory=list)
    out_of_domain_reply: str = (
        "I'm only able to answer questions about {name}. "
        "That question looks outside my scope."
    )
