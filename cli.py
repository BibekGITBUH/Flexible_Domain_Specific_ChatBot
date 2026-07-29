"""
cli.py
-------
Terminal chat loop -- this IS the "system core" for now, deliberately
with no server/Streamlit layer yet (that comes later per the project
plan).

Commands inside the chat:
  /mode zero_shot | few_shot | cot     switch prompting technique
  /domain <key>                        switch domain (see domain/registry.py)
  /exit                                quit

Usage:
    python cli.py            # mock backend, no GPU/model needed
    python cli.py --real     # loads the real LLM (needs GPU + internet)
"""

import argparse

from domain.registry import get_domain, DOMAIN_REGISTRY
from config import MODEL_CONFIG, GEN_CONFIG, DEFAULT_DOMAIN


def main(use_real_model: bool):
    domain_key = DEFAULT_DOMAIN
    domain = get_domain(domain_key)

    if use_real_model:
        from core.chatbot import DomainChatbot
        bot = DomainChatbot(domain, MODEL_CONFIG, GEN_CONFIG)
        print(f"[loading {MODEL_CONFIG.model_name} ... this can take a minute]")
    else:
        from core.mock_chatbot import MockDomainChatbot
        bot = MockDomainChatbot(domain)
        print("[running in MOCK mode -- no real LLM loaded. Use --real on your GPU machine.]")

    mode = "few_shot"
    history = []          # real conversation memory: list of {"role", "content"}
    MAX_HISTORY_TURNS = 6  # cap so the prompt doesn't grow unbounded on a small model
    print(f"Domain: {domain.name} | Mode: {mode}")
    print("Type /exit to quit, /mode <zero_shot|few_shot|cot> to switch technique,")
    print("/custom to build a new domain from a PDF/TXT file or pasted text.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("Bye.")
            break

        if user_input == "/reset":
            history = []
            print("[conversation memory cleared]")
            continue

        if user_input == "/custom":
            from domain.custom_loader import (
                build_custom_domain,
                extract_text_from_pdf,
                extract_text_from_txt,
            )

            print("\nLet's build a custom domain.")
            new_name = input("  Domain name: ").strip()
            new_persona = input("  Persona (how the bot should behave): ").strip()
            kb_source = input(
                "  Knowledge base: type a .txt/.pdf file path, or paste text directly: "
            ).strip()

            if kb_source.lower().endswith(".pdf"):
                with open(kb_source, "rb") as f:
                    kb_text = extract_text_from_pdf(f.read())
            elif kb_source.lower().endswith(".txt"):
                with open(kb_source, "rb") as f:
                    kb_text = extract_text_from_txt(f.read())
            else:
                kb_text = kb_source  # treated as raw pasted text

            few_shot = []
            while input("  Add a few-shot example? (y/n): ").strip().lower() == "y":
                q = input("    Example question: ").strip()
                a = input("    Ideal answer: ").strip()
                few_shot.append((q, a))

            new_domain, warning = build_custom_domain(new_name, new_persona, kb_text, few_shot)
            if warning:
                print(f"  [warning] {warning}")

            new_key = f"custom_{new_domain.name.lower().replace(' ', '_')}"
            DOMAIN_REGISTRY[new_key] = new_domain
            domain = new_domain
            domain_key = new_key
            bot.domain = domain
            history = []
            print(f"[custom domain '{new_domain.name}' activated as '{new_key}']\n")
            continue

        if user_input == "/mode":
            print(f"[current mode: {mode}. Usage: /mode zero_shot|few_shot|cot]")
            continue

        if user_input.startswith("/mode "):
            requested = user_input.split(" ", 1)[1].strip()
            if requested in {"zero_shot", "few_shot", "cot"}:
                mode = requested
                print(f"[mode switched to {mode}]")
            else:
                print("[unknown mode; choose zero_shot, few_shot, or cot]")
            continue

        if user_input.startswith("/domain "):
            requested = user_input.split(" ", 1)[1].strip()
            try:
                domain = get_domain(requested)
                domain_key = requested
                bot.domain = domain
                print(f"[domain switched to {domain.name}]")
            except KeyError as e:
                print(f"[{e}] Available: {list(DOMAIN_REGISTRY)}")
            continue

        result = bot.ask(user_input, mode=mode, history=history)
        print(f"RegBot ({mode}): {result['response']}\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": result["response"]})
        history = history[-(MAX_HISTORY_TURNS * 2):]  # keep last N user+assistant pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use the real LLM instead of the mock backend")
    args = parser.parse_args()
    main(use_real_model=args.real)
