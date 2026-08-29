#!/usr/bin/env python3
"""
train_unsloth.py
Fine-tunes a lightweight base LLM (default: Gemma-2-2B-it) using Unsloth on H100 GPU
for forensic activity auditing, citation discipline, and absence reasoning.
"""

import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for Forensic Auditing with Unsloth")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-2-2b-it",
                        help="Base model from Hugging Face / Unsloth (e.g. unsloth/gemma-2-2b-it, unsloth/gemma-2-9b-it, unsloth/Qwen2.5-14B-Instruct)")
    parser.add_argument("--train_file", type=str, default="data/train_split.jsonl")
    parser.add_argument("--val_file", type=str, default="data/val_split.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs_forensic_gemma_2b")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--load_in_4bit", action="store_true", help="Set to train with 4-bit QLoRA")
    parser.add_argument("--chat_template", type=str, default="gemma",
                        help="Chat template to use ('gemma', 'chatml', 'llama-3')")
    return parser.parse_args()

def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    args = parse_args()
    print("=" * 60)
    print(f"Starting Fine-Tuning Pipeline for Forensic Auditing")
    print(f"Model: {args.model_name}")
    print(f"Max Sequence Length: {args.max_seq_length}")
    print(f"LoRA Rank: {args.lora_rank}, Alpha: {args.lora_alpha}")
    print(f"4-bit Quantization: {args.load_in_4bit}")
    print("=" * 60)

    # 1. Load Base Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=torch.bfloat16,
    )

    # 2. Configure PEFT / LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 3. Setup Chat Template
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    def format_prompts(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    # 4. Load Datasets
    print(f"Loading training data from {args.train_file}...")
    train_raw = load_jsonl(args.train_file)
    train_dataset = Dataset.from_list(train_raw).map(format_prompts, batched=True)

    val_dataset = None
    if os.path.exists(args.val_file):
        print(f"Loading validation data from {args.val_file}...")
        val_raw = load_jsonl(args.val_file)
        val_dataset = Dataset.from_list(val_raw).map(format_prompts, batched=True)

    # 5. Trainer Configuration
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=5,
        evaluation_strategy="epoch" if val_dataset is not None else "no",
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # 6. Train
    print("Beginning Training...")
    trainer.train()

    # 7. Save Model Adapter
    adapter_path = os.path.join(args.output_dir, "final_adapter")
    print(f"Saving fine-tuned LoRA adapter to {adapter_path}...")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print("Training successfully finished!")

if __name__ == "__main__":
    main()
