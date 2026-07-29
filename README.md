# Domain-Specific Chatbot (Track 3) — College Academic Regulations

An LLM-based chatbot that answers questions about college academic
regulations using **prompt engineering only** — no retrieval (RAG),
no fine-tuning.

## 1. Model Architecture

**Type:** LLM-based, no retrieval. There is no vector database, no
embedding search, and no fine-tuning step.

**Base model:** `Qwen/Qwen2.5-3B-Instruct`, loaded in 4-bit
(NF4 via `bitsandbytes`) — confirmed target GPU is an RTX 3060 with
6GB VRAM (Windows/WDDM, so some VRAM is reserved by the OS), which is
too tight for a 7B model in 4-bit once KV-cache overhead is added. 3B
in 4-bit uses ~2GB, leaving comfortable headroom. Swappable in
`config.py` — 7B models (Mistral-7B-Instruct, Llama-3.1-8B-Instruct)
are documented there for use on a card with more free VRAM.

**How domain-specificity is achieved without retrieval:** the domain's
factual knowledge (`domain/college_regulations.py:KNOWLEDGE_BASE`) is
small and curated (~20 policy facts, a few KB of text), so it is
injected wholesale into the **system prompt** on every call, instead
of being retrieved dynamically. This is the key architectural decision
required by the assignment ("LLM-based, without retrieval") — it only
scales to a knowledge base small enough to fit in context; a larger
corpus would need RAG.

```
User query
   │
   ▼
prompt_engineering.build_prompt(mode, domain, query)
   │   mode ∈ {zero_shot, few_shot, cot}
   │   always prepends: persona + full knowledge base + rules
   ▼
DomainChatbot.ask() → tokenizer.apply_chat_template → model.generate()
   │
   ▼
Response (grounded in the injected knowledge, or a refusal if
off-topic)
```

### Prompt engineering techniques implemented (`core/prompt_engineering.py`)
| Technique | What it does |
|---|---|
| **Zero-shot** | System prompt (persona + knowledge base + rules) + the raw user question. No examples. |
| **Few-shot** | Same system prompt, plus 3 curated (question → ideal answer) example turns before the real question, so the model imitates the grounded, concise answer style. |
| **Chain-of-thought** | Instructs the model to (1) identify the relevant KB section, (2) reason about how it applies, (3) then output `Final Answer: ...`, before answering. |

All three share the same domain-guard rule in the system prompt:
*"If the question is unrelated to the domain, say so instead of
guessing."* This is how out-of-domain queries (e.g. "capital of
France") are handled, without any separate classifier.

## 2. Dataset Description

There is no training dataset — this project does not fine-tune a
model. The "dataset" is:

- **Knowledge base** (`domain/college_regulations.py`): ~20 hand-written
  policy facts across 7 topics (attendance, grading, exams, probation,
  leave, academic integrity, fees/registration). Curated for this
  project, not scraped, no PII/copyrighted text.
- **Evaluation set** (`evaluation/test_cases.py`): 8 curated queries —
  6 in-domain (each with required keywords the answer must contain)
  and 2 out-of-domain (should be politely refused).

Swapping domains = writing a new file in `domain/` following the same
`DomainConfig` schema (see `domain/base.py`) and registering it in
`domain/registry.py`. No other code changes.

## 3. Training Details

**N/A by design.** The assignment specifies prompt-engineering
techniques (zero-shot / few-shot / CoT) on top of an existing LLM, not
fine-tuning. No weights are updated; all domain grounding happens at
inference time through the system prompt. This is documented here
explicitly to match the "Training details" field expected in
submission, with the reasoning for why it's empty.

## 4. Evaluation Metrics & Results

Run with the mock backend (no GPU needed, tests the *pipeline* logic):
```
python -m evaluation.evaluate
```
Run with the real model (needs your RTX 3060 + `pip install -r requirements.txt`):
```
python -m evaluation.evaluate --real
```

**Metrics:**
- **Keyword coverage** (in-domain questions): fraction of
  `expected_keywords` (specific rule numbers/terms) present in the
  response. Proxy for factual grounding without needing manual
  scoring for every run.
- **Refusal accuracy** (out-of-domain questions): did the bot decline
  instead of hallucinating an answer?

Results are written to `logs/eval_results.json` (full transcript: query,
mode, response, score, latency) and a summary table is printed per mode,
so you can compare zero-shot vs few-shot vs CoT quality side by side.

## 5. Running the chatbot

```bash
pip install -r requirements.txt
```

### Terminal (CLI)
```bash
python cli.py            # mock mode — no model download, tests logic instantly
python cli.py --real     # real mode — loads Qwen2.5-3B-Instruct in 4-bit on your GPU
```
Inside the chat:
- `/mode zero_shot|few_shot|cot` — switch prompting technique live
- `/domain <key>` — switch domain (see `domain/registry.py`)
- `/reset` — clear conversation memory, start fresh
- `/exit` — quit

### Streamlit (the deployment deliverable)
```bash
streamlit run streamlit_app.py
```
Opens a browser UI with the same underlying chatbot (`core/chatbot.py` /
`core/mock_chatbot.py` — no logic is duplicated). Sidebar toggle switches
between mock and real model, lets you switch domain/prompting technique
live, and shows the injected knowledge base for transparency. Chat
history persists across turns via `st.session_state` for the browser
session; it does not persist across restarts (same as the CLI).

## 6. Evaluation

```bash
python -m evaluation.evaluate --real     # scores against the real model, not the mock
```
Writes results to `logs/eval_results.json` and prints a per-mode
summary table comparing zero-shot / few-shot / CoT on keyword-coverage
(in-domain accuracy) and refusal-accuracy (out-of-domain handling).

## 7. Exporting the dataset as a standalone submission file

The knowledge base lives in `domain/college_regulations.py` (that's
what the code imports), but for a clean submittable "dataset" artifact:
```bash
python -m scripts.export_dataset
```
Writes `dataset/college_regulations.json` (knowledge base + few-shot
examples) and `dataset/evaluation_test_cases.json` — plain JSON, no
Python needed to read them.

## 8. Project Structure

```
domain_chatbot/
├── config.py                    # model + generation settings (edit to swap models)
├── requirements.txt
├── cli.py                       # terminal chat loop
├── streamlit_app.py             # Streamlit deployment (deliverable #5)
├── domain/
│   ├── base.py                  # DomainConfig schema
│   ├── college_regulations.py   # curated knowledge + few-shot examples
│   └── registry.py              # domain key → DomainConfig lookup
├── core/
│   ├── prompt_engineering.py    # zero-shot / few-shot / CoT prompt builders
│   ├── chatbot.py                # real LLM backend (transformers + bitsandbytes)
│   └── mock_chatbot.py           # no-GPU stand-in, same .ask() interface
├── evaluation/
│   ├── test_cases.py             # curated eval queries + expected keywords
│   └── evaluate.py               # scoring harness, writes logs/eval_results.json
├── scripts/
│   └── export_dataset.py         # exports domain/ data to plain JSON for submission
├── dataset/                      # generated by export_dataset.py
│   ├── college_regulations.json
│   └── evaluation_test_cases.json
└── logs/
    └── eval_results.json         # generated by evaluate.py
```
