"""
core/prompt_engineering.py
----------------------------
Builds the actual message list sent to the LLM, for each of the three
required prompting techniques. All three share the same domain
grounding (persona + knowledge base) as the system message -- the
technique only changes what comes AFTER the system message.

Returned format is a list of {"role": ..., "content": ...} dicts,
matching the chat-template format used by HF `apply_chat_template`
and the OpenAI-style API.
"""

from domain.base import DomainConfig


def _system_message(domain: DomainConfig) -> dict:
    content = (
        f"{domain.persona}\n\n"
        f"--- REFERENCE KNOWLEDGE ({domain.name}) ---\n"
        f"{domain.knowledge_base}\n"
        f"--- END REFERENCE KNOWLEDGE ---\n\n"
        f"Rules:\n"
        f"1. Only answer using the reference knowledge above.\n"
        f"2. If the question is unrelated to {domain.name}, respond with "
        f"something like: \"{domain.out_of_domain_reply.format(name=domain.name)}\"\n"
        f"3. Never fabricate a fact, number, or detail that isn't in the "
        f"reference knowledge.\n"
    )
    return {"role": "system", "content": content}


def _append_history(messages: list, history: list) -> list:
    """
    history: list of {"role": "user"|"assistant", "content": str} from
    the REAL conversation so far (not few-shot examples). Appended
    after the system/few-shot setup and before the current question,
    so the model can resolve references like "okay sure" or "what
    about the second one" correctly.
    """
    if history:
        messages.extend(history)
    return messages


def build_zero_shot(domain: DomainConfig, user_query: str, history: list = None) -> list:
    """
    Zero-shot: only the system (domain) prompt + conversation history
    (if any) + the user's question. No examples of desired output format.
    """
    messages = [_system_message(domain)]
    _append_history(messages, history)
    messages.append({"role": "user", "content": user_query})
    return messages


def build_few_shot(domain: DomainConfig, user_query: str, history: list = None) -> list:
    """
    Few-shot: same system prompt, a handful of (question, ideal answer)
    pairs as prior turns so the model imitates the style/grounding
    pattern, THEN the real conversation history, THEN the current
    question. Real history comes after the examples so the model
    doesn't confuse a stale few-shot example with what the user just
    said.
    """
    messages = [_system_message(domain)]
    for example_q, example_a in domain.few_shot_examples:
        messages.append({"role": "user", "content": example_q})
        messages.append({"role": "assistant", "content": example_a})
    _append_history(messages, history)
    messages.append({"role": "user", "content": user_query})
    return messages


def build_chain_of_thought(domain: DomainConfig, user_query: str, history: list = None) -> list:
    """
    Chain-of-thought: instructs the model to reason step by step over
    the knowledge base BEFORE giving the final answer. We ask it to
    clearly separate reasoning from the final answer so the UI/eval
    code can parse out just the final answer if needed.
    """
    messages = [_system_message(domain)]
    _append_history(messages, history)
    cot_instruction = (
        "Think through this step by step before answering:\n"
        "1. Identify which part(s) of the reference knowledge are relevant.\n"
        "2. Reason about how that specific fact or detail applies to the "
        "question asked.\n"
        "3. Then give the final answer clearly, prefixed with "
        "'Final Answer:'.\n\n"
        f"Question: {user_query}"
    )
    messages.append({"role": "user", "content": cot_instruction})
    return messages


PROMPT_BUILDERS = {
    "zero_shot": build_zero_shot,
    "few_shot": build_few_shot,
    "cot": build_chain_of_thought,
}


def build_prompt(mode: str, domain: DomainConfig, user_query: str, history: list = None) -> list:
    if mode not in PROMPT_BUILDERS:
        raise ValueError(
            f"Unknown prompting mode '{mode}'. Choose from {list(PROMPT_BUILDERS)}"
        )
    return PROMPT_BUILDERS[mode](domain, user_query, history)
