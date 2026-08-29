#!/usr/bin/env python3
"""
train_unsloth.py
Production-grade LoRA fine-tuning script for H100 / A100 GPUs.
Fine-tunes lightweight models (e.g. google/gemma-2-2b-it, Qwen/Qwen2.5-14B) on mobile digital forensic
timeline reconstruction, 100% citation discipline, absence/SQLCipher auditing, and indirect commute inference.
"""

import os
import sys
import json
import argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for Forensic Auditing with LoRA")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it",
                        help="Base model from Hugging Face (e.g. google/gemma-2-2b-it, Qwen/Qwen2.5-14B-Instruct, meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--train_file", type=str, default="data/splits/train.jsonl",
                        help="Path to training jsonl (90 date-held-out windows)")
    parser.add_argument("--val_file", type=str, default="data/splits/val.jsonl",
                        help="Path to validation jsonl (23 held-out windows)")
    parser.add_argument("--output_dir", type=str, default="outputs_forensic_gemma_2b",
                        help="Directory to save checkpoints and final LoRA adapter")
    parser.add_argument("--max_seq_length", type=int, default=8192,
                        help="Maximum sequence length (8k context for long SQLite dumps)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch size = 8)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for LoRA adapter")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank (r)")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    return parser.parse_args()


def load_and_format_dataset(filepath, tokenizer):
    """Loads a JSONL dataset and formats messages using the tokenizer chat template."""
    print(f"[Dataset] Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            messages = data.get("messages", [])
            
            if len(messages) >= 2:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                samples.append({"text": text})

    print(f"[Dataset] Formatted {len(samples)} valid training samples from {filepath}.")
    return Dataset.from_list(samples)


def main():
    args = parse_args()

    print("=" * 75)
    print("  KRYPTO: MOBILE FORENSIC LLM FINE-TUNING PIPELINE (LORA)")
    print(f"  Base Model:       {args.model_name}")
    print(f"  Train Split:      {args.train_file}")
    print(f"  Val Split:        {args.val_file}")
    print(f"  Output Dir:       {args.output_dir}")
    print(f"  Max Context Len:  {args.max_seq_length} tokens")
    print(f"  LoRA Rank (r):    {args.lora_rank}, Alpha: {args.lora_alpha}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Device:           {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 75)

    # 1. Load Tokenizer
    print(f"\n[1/5] Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model
    print(f"[2/5] Loading base model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # 3. Add LoRA Adapters
    print("[3/5] Injecting LoRA adapters into projection layers...")
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Prepare Datasets
    print("\n[4/5] Loading and formatting dataset splits...")
    train_dataset = load_and_format_dataset(args.train_file, tokenizer)
    val_dataset = load_and_format_dataset(args.val_file, tokenizer) if args.val_file else None

    # 5. Configure Training Arguments with SFTConfig
    print("\n[5/5] Configuring SFT Trainer hyper-parameters...")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_steps=1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    print("\nLaunching GPU Training...")
    trainer_stats = trainer.train()
    print(f"\n[Training Complete] Runtime: {trainer_stats.metrics['train_runtime']:.2f}s | Global Steps: {trainer_stats.global_step}")

    # Save LoRA Adapter & Tokenizer
    final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
    print(f"\n[Saving] Saving fine-tuned LoRA adapter to {final_adapter_dir}...")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    print("\n" + "=" * 75)
    print(f"  SUCCESS: Fine-tuned LoRA adapter saved to '{final_adapter_dir}'")
    print("=" * 75)

if __name__ == "__main__":
    main()
