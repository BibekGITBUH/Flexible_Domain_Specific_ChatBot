"""
core/chatbot.py
-----------------
The chatbot "engine": loads the LLM once, builds a prompt using the
requested prompt-engineering technique, and generates a response.

This is intentionally the ONLY file that imports `transformers` /
`torch`, so the rest of the codebase (prompt building, evaluation,
domain config) can be tested without a GPU or the model downloaded.
"""

from domain.base import DomainConfig
from core.prompt_engineering import build_prompt
from config import ModelConfig, GenerationConfig


class DomainChatbot:
    def __init__(
        self,
        domain: DomainConfig,
        model_config: ModelConfig,
        gen_config: GenerationConfig,
    ):
        self.domain = domain
        self.model_config = model_config
        self.gen_config = gen_config
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Model loading (lazy: only happens on first `ask()` call, so you can
    # import/instantiate this class in tests/eval without a GPU present)
    # ------------------------------------------------------------------
    def _load_model(self):
        if self._model is not None:
            return

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch.cuda.is_available() is False -- no GPU detected by "
                "PyTorch. This is almost always because torch was installed "
                "as a CPU-only build. Fix with:\n"
                "    pip uninstall torch\n"
                "    pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
                "(use cu118 instead of cu121 if `nvidia-smi` shows an older "
                "CUDA driver version). Then re-run with --real."
            )

        quant_config = None
        if self.model_config.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        # Force the whole model onto GPU 0 rather than letting "auto"
        # decide -- on borderline-VRAM cards (e.g. 6GB RTX 3060 laptop
        # GPUs) device_map="auto" can be overly conservative and try to
        # offload a few layers to CPU/disk, which 4-bit bitsandbytes
        # refuses to do without an explicit fp32 CPU-offload flag. Since
        # a 7B model in 4-bit is only ~5GB of weights, forcing it fully
        # onto the GPU is the right call for a 3060 (6GB or 12GB).
        device_map = {"": 0}

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            quantization_config=quant_config,
            device_map=device_map,
            dtype=torch.float16,
        )

    # ------------------------------------------------------------------
    def ask(self, user_query: str, mode: str = "few_shot", history: list = None) -> dict:
        """
        mode: "zero_shot" | "few_shot" | "cot"
        history: optional list of {"role": "user"|"assistant", "content": str}
            from the real conversation so far (NOT few-shot examples).
            Pass this if you want multi-turn context; omit for a single
            stateless question (e.g. in the evaluation harness).
        Returns dict with the raw response text plus the prompt used
        (useful for the eval harness / demo logs).
        """
        self._load_model()

        messages = build_prompt(mode, self.domain, user_query, history=history)

        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._model.device)

        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.gen_config.max_new_tokens,
            temperature=self.gen_config.temperature,
            top_p=self.gen_config.top_p,
            do_sample=self.gen_config.do_sample,
            repetition_penalty=self.gen_config.repetition_penalty,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        response_text = self._tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        return {
            "query": user_query,
            "mode": mode,
            "messages_sent": messages,
            "response": response_text,
        }
