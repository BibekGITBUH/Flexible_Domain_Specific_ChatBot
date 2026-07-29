"""
evaluation/evaluate.py
-------------------------
Runs the curated TEST_CASES against the chatbot for each prompting
mode (zero_shot / few_shot / cot), scores:
  1. Keyword coverage: fraction of expected_keywords found in the
     response (for in-domain questions)
  2. Refusal accuracy: for out-of-domain questions, did the bot
     correctly decline instead of hallucinating an answer?

Writes results to logs/eval_results.json and prints a summary table.

Usage:
    python -m evaluation.evaluate            # uses the mock backend
    python -m evaluation.evaluate --real      # uses the real LLM
"""

import argparse
import json
import time
from pathlib import Path

from domain.registry import get_domain
from config import MODEL_CONFIG, GEN_CONFIG, DEFAULT_DOMAIN
from evaluation.test_cases import TEST_CASES

MODES = ["zero_shot", "few_shot", "cot"]
LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "eval_results.json"


def score_case(case: dict, response: str) -> dict:
    response_lower = response.lower()

    if case["in_domain"]:
        hits = [kw for kw in case["expected_keywords"] if kw.lower() in response_lower]
        coverage = len(hits) / len(case["expected_keywords"]) if case["expected_keywords"] else 1.0
        return {"type": "keyword_coverage", "score": coverage, "hits": hits}
    else:
        # crude refusal heuristic: response should NOT confidently answer;
        # look for refusal-ish language
        refusal_markers = ["outside", "scope", "only", "cannot", "can't", "not able"]
        refused = any(m in response_lower for m in refusal_markers)
        return {"type": "refusal_accuracy", "score": 1.0 if refused else 0.0}


def run_evaluation(use_real_model: bool):
    domain = get_domain(DEFAULT_DOMAIN)

    if use_real_model:
        from core.chatbot import DomainChatbot
        bot = DomainChatbot(domain, MODEL_CONFIG, GEN_CONFIG)
    else:
        from core.mock_chatbot import MockDomainChatbot
        bot = MockDomainChatbot(domain)

    results = []
    for case in TEST_CASES:
        for mode in MODES:
            t0 = time.time()
            output = bot.ask(case["query"], mode=mode)
            latency = time.time() - t0

            scoring = score_case(case, output["response"])
            results.append({
                "query": case["query"],
                "mode": mode,
                "in_domain": case["in_domain"],
                "response": output["response"],
                "scoring": scoring,
                "latency_sec": round(latency, 3),
            })

    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(results, f, indent=2)

    _print_summary(results)
    return results


def _print_summary(results):
    print(f"\n{'Mode':<10} {'Query':<55} {'Score':<8} {'Type'}")
    print("-" * 90)
    for r in results:
        q = (r["query"][:52] + "...") if len(r["query"]) > 52 else r["query"]
        print(f"{r['mode']:<10} {q:<55} {r['scoring']['score']:<8.2f} {r['scoring']['type']}")

    for mode in MODES:
        mode_scores = [r["scoring"]["score"] for r in results if r["mode"] == mode]
        avg = sum(mode_scores) / len(mode_scores)
        print(f"\nAverage score for {mode}: {avg:.2f}")

    print(f"\nFull results written to {LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use the real LLM instead of the mock backend")
    args = parser.parse_args()
    run_evaluation(use_real_model=args.real)
