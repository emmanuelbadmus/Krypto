"""
Krypto: Mobile Forensic Activity Reconstruction Fine-Tuning Script (Unsloth / GPU Server)
Fine-tunes Gemma-4-E2B-it with Response-Only Loss Masking.
"""

import os
import sys
import gc
import json
import re
import argparse
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    args = parser.parse_args()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_name = "models/gemma-4-E2B-it" if os.path.exists("models/gemma-4-E2B-it/model.safetensors") else "google/gemma-4-E2B-it"
    print(f"Loading {model_name}...")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    try:
        from transformers import Gemma4ForConditionalGeneration
        model = Gemma4ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)

    model = model.to(device)

    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        num_layers = len(model.model.language_model.layers)
        target_mods = [f"model.language_model.layers.{i}.self_attn.{p}" for i in range(num_layers) for p in ["q_proj", "k_proj", "v_proj", "o_proj"]] + \
                      [f"model.language_model.layers.{i}.mlp.{p}" for i in range(num_layers) for p in ["gate_proj", "up_proj", "down_proj"]]
    else:
        target_mods = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=target_mods,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config, low_cpu_mem_usage=False)
    model.print_trainable_parameters()

    def prepare_dataset(filepath, max_len=2048):
        samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                msgs = item.get("messages", [])
                if len(msgs) >= 2:
                    gemma_msgs = []
                    system_text = ""
                    for m in msgs:
                        if m["role"] == "system":
                            system_text = m["content"].strip() + "\n\n"
                        elif m["role"] == "user":
                            gemma_msgs.append({"role": "user", "content": (system_text + m["content"]).strip()})
                            system_text = ""
                        elif m["role"] in ("assistant", "model"):
                            gemma_msgs.append({"role": "assistant", "content": m["content"]})
                    full_text = tokenizer.apply_chat_template(gemma_msgs, tokenize=False)
                    prompt_text = tokenizer.apply_chat_template([gemma_msgs[0]], tokenize=False, add_generation_prompt=True)
                    samples.append({"full_text": full_text, "prompt_text": prompt_text})
        raw_ds = Dataset.from_list(samples)
        def tok_fn(batch):
            input_ids_list, attention_mask_list, labels_list = [], [], []
            for full_t, prompt_t in zip(batch["full_text"], batch["prompt_text"]):
                encoded = tokenizer(full_t, truncation=True, max_length=max_len)
                prompt_enc = tokenizer(prompt_t, truncation=True, max_length=max_len, add_special_tokens=False)
                input_ids = encoded["input_ids"]
                labels = list(input_ids)
                prompt_len = min(len(prompt_enc["input_ids"]), len(labels))
                labels[:prompt_len] = [-100] * prompt_len
                input_ids_list.append(input_ids)
                attention_mask_list.append(encoded["attention_mask"])
                labels_list.append(labels)
            return {"input_ids": input_ids_list, "attention_mask": attention_mask_list, "labels": labels_list}
        return raw_ds.map(tok_fn, batched=True, remove_columns=["full_text", "prompt_text"])

    train_ds = prepare_dataset("data/splits/train.jsonl")
    val_ds = prepare_dataset("data/splits/val.jsonl")

    training_args = TrainingArguments(
        output_dir="outputs_forensic_gemma4_2b",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        warmup_steps=10,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_steps=1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
    )

    print("Starting training...")
    trainer.train()
    model.save_pretrained("outputs_forensic_gemma4_2b/final_adapter")
    tokenizer.save_pretrained("outputs_forensic_gemma4_2b/final_adapter")
    print("Adapter saved successfully!")


if __name__ == "__main__":
    main()
