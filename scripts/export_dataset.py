"""
scripts/export_dataset.py
----------------------------
The knowledge base lives in domain/college_regulations.py because
that's what the running code imports. But your submission checklist
separately asks for a "Dataset (either uploaded zip file, link)" --
reviewers shouldn't have to read Python to see your data. This script
exports the SAME data (single source of truth stays in domain/) to
plain JSON files under dataset/, safe to zip and submit on their own.

Usage:
    python -m scripts.export_dataset
"""

import json
from pathlib import Path

from domain.registry import DOMAIN_REGISTRY
from evaluation.test_cases import TEST_CASES

OUT_DIR = Path(__file__).resolve().parent.parent / "dataset"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    for key, domain in DOMAIN_REGISTRY.items():
        payload = {
            "domain_name": domain.name,
            "persona": domain.persona,
            "knowledge_base": domain.knowledge_base,
            "few_shot_examples": [
                {"question": q, "answer": a} for q, a in domain.few_shot_examples
            ],
        }
        out_path = OUT_DIR / f"{key}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {out_path}")

    eval_path = OUT_DIR / "evaluation_test_cases.json"
    with open(eval_path, "w") as f:
        json.dump(TEST_CASES, f, indent=2)
    print(f"Wrote {eval_path}")


if __name__ == "__main__":
    main()
