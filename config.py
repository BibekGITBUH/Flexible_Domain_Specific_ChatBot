"""
config.py
---------
Single place to control which model, quantization, and generation
settings the chatbot uses. Change MODEL_NAME to swap models without
touching any other file.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Default: Qwen2.5-3B-Instruct. Confirmed target GPU is an RTX 3060
    # with 6GB VRAM (WDDM/Windows, so some VRAM is reserved by the OS
    # display driver) -- a 7B model in 4-bit (~4.5GB weights alone) is
    # too tight there once you add KV-cache + generation overhead.
    # 3B in 4-bit is ~2GB, leaving comfortable headroom.
    #
    # If you later run this on a card with more free VRAM (8GB+), swap
    # to:  "Qwen/Qwen2.5-7B-Instruct"
    #      "mistralai/Mistral-7B-Instruct-v0.3"
    #      "meta-llama/Meta-Llama-3.1-8B-Instruct"  (gated, needs HF token)
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    # 4-bit quantization keeps a 7B model comfortably under ~6GB VRAM,
    # which fits an RTX 3060. Set to False if you have more VRAM and
    # want full/half precision quality.
    load_in_4bit: bool = True

    device_map: str = "auto"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 400
    temperature: float = 0.3      # lower = more factual/deterministic
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1


MODEL_CONFIG = ModelConfig()
GEN_CONFIG = GenerationConfig()

# Which domain module to load by default. See domain/registry.py
DEFAULT_DOMAIN = "college_regulations"
