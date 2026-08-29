"""
harness/models.py
Modular LLM model runner interface supporting Hugging Face, Unsloth, PEFT LoRA,
and API backends for forensic evaluation.
"""

import os
import torch
from abc import ABC, abstractmethod

class BaseModelRunner(ABC):
    """Abstract base class for all LLM inference runners in the harness."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generates the forensic activity reconstruction text for a given prompt."""
        pass


class HuggingFaceRunner(BaseModelRunner):
    """Runner for standard Hugging Face Transformers models with MPS / CUDA / CPU support."""

    def __init__(self, model_name_or_path: str, adapter_path: str = None, max_seq_length: int = 8192, torch_dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name_or_path
        self.adapter_path = adapter_path
        
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"[HuggingFaceRunner] Loading {model_name_or_path} on device: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        if adapter_path:
            from peft import PeftModel
            print(f"[HuggingFaceRunner] Loading LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt_text = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        else:
            prompt_text = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )

        gen_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


class UnslothRunner(BaseModelRunner):
    """Runner using Unsloth optimized inference kernels."""

    def __init__(self, model_name_or_path: str, adapter_path: str = None, max_seq_length: int = 8192, chat_template: str = "gemma"):
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        
        target_path = adapter_path if adapter_path else model_name_or_path
        print(f"[UnslothRunner] Loading {target_path} with Unsloth...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=target_path,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        FastLanguageModel.for_inference(self.model)
        self.tokenizer = get_chat_template(self.tokenizer, chat_template=chat_template)

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                use_cache=True,
            )

        gen_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def get_model_runner(model_name_or_path: str, adapter_path: str = None, backend: str = "auto", chat_template: str = "gemma") -> BaseModelRunner:
    """Factory function to instantiate the appropriate model runner."""
    if backend == "unsloth" or (backend == "auto" and torch.cuda.is_available()):
        try:
            return UnslothRunner(model_name_or_path, adapter_path, chat_template=chat_template)
        except Exception as e:
            print(f"[get_model_runner] Unsloth initialization failed ({e}), falling back to HuggingFaceRunner.")
    
    return HuggingFaceRunner(model_name_or_path, adapter_path)
