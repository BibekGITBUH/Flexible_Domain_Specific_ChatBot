"""
core/mock_chatbot.py
----------------------
A drop-in replacement for DomainChatbot that does NOT load any real
model. It exists purely so we can unit-test the prompt-building /
domain-guard / evaluation pipeline in environments without a GPU or
internet access to Hugging Face (like this sandbox).

It has the exact same `.ask(query, mode)` interface as DomainChatbot,
so evaluate.py and cli.py work unchanged against either one -- swap
`USE_MOCK` in cli.py / evaluate.py to False once you run this on your
RTX 3060 with the real model.
"""

from domain.base import DomainConfig
from core.prompt_engineering import build_prompt


class MockDomainChatbot:
    def __init__(self, domain: DomainConfig, *_, **__):
        self.domain = domain

    def ask(self, user_query: str, mode: str = "few_shot", history: list = None) -> dict:
        messages = build_prompt(mode, self.domain, user_query, history=history)

        # extremely naive canned response so we can sanity check the
        # domain guard + prompt structure without a real LLM
        lowered = user_query.lower()
        kb_lower = self.domain.knowledge_base.lower()
        on_topic = any(
            word in kb_lower
            for word in lowered.split()
            if len(word) > 4
        )

        if not on_topic:
            response = self.domain.out_of_domain_reply.format(
                name=self.domain.name
            )
        else:
            response = (
                "[MOCK RESPONSE] This would be answered from the "
                f"knowledge base using '{mode}' prompting."
            )

        return {
            "query": user_query,
            "mode": mode,
            "messages_sent": messages,
            "response": response,
        }
