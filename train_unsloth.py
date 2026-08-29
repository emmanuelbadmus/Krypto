#!/usr/bin/env python3
"""
train_unsloth.py
Production-grade LoRA fine-tuning script optimized with Unsloth for H100 / A100 GPUs.
Fine-tunes lightweight models (e.g., Gemma-2-2B-it, Gemma-4-E2B-it) on mobile digital forensic
timeline reconstruction, 100% citation discipline, absence/SQLCipher auditing, and indirect commute inference.
"""

import os
import sys
import json
import argparse
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for Forensic Auditing with Unsloth")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-2-2b-it",
                        help="Base model from Hugging Face / Unsloth (e.g. unsloth/gemma-2-2b-it, google/gemma-4-E2B-it, unsloth/Qwen2.5-14B-Instruct)")
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
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout (0.0 optimized for Unsloth)")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Enable 4-bit QLoRA quantization (saves VRAM on smaller GPUs)")
    parser.add_argument("--chat_template", type=str, default="gemma",
                        help="Chat template ('gemma', 'chatml', 'llama-3')")
    parser.add_argument("--push_to_hub", type=str, default=None,
                        help="Optional Hugging Face repo ID to push final adapter")
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
            
            # Ensure valid conversation structure
            if len(messages) >= 2:
                # Format into single prompt string using chat template
                try:
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    samples.append({"text": text})
                except Exception as e:
                    # Fallback standard formatting
                    sys_text = messages[0]["content"] if messages[0]["role"] == "system" else ""
                    usr_text = messages[1]["content"] if messages[1]["role"] == "user" else ""
                    ast_text = messages[2]["content"] if len(messages) > 2 and messages[2]["role"] == "assistant" else ""
                    formatted = f"<start_of_turn>user\n{sys_text}\n\n{usr_text}<end_of_turn>\n<start_of_turn>model\n{ast_text}<end_of_turn>"
                    samples.append({"text": formatted})

    print(f"[Dataset] Formatted {len(samples)} valid training samples from {filepath}.")
    return Dataset.from_list(samples)


def main():
    args = parse_args()

    print("=" * 75)
    print("  KRYPTO: MOBILE FORENSIC LLM FINE-TUNING PIPELINE (UNSLOTH)")
    print(f"  Base Model:       {args.model_name}")
    print(f"  Train Split:      {args.train_file}")
    print(f"  Val Split:        {args.val_file}")
    print(f"  Output Dir:       {args.output_dir}")
    print(f"  Max Context Len:  {args.max_seq_length} tokens")
    print(f"  LoRA Rank (r):    {args.lora_rank}, Alpha: {args.lora_alpha}")
    print(f"  4-bit QLoRA:      {args.load_in_4bit}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Device:           {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 75)

    # 1. Initialize Unsloth FastLanguageModel
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        print(f"\n[1/5] Initializing base model {args.model_name} with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        # 2. Add LoRA Adapters targetting all attention & MLP projections
        print("[2/5] Injecting LoRA adapters into attention and MLP projection layers...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    except ImportError:
        print("\n[Warning] Unsloth is not installed in the local environment. Falling back to Standard HuggingFace PEFT/TRL.")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    # 3. Prepare Datasets
    print("\n[3/5] Loading and formatting dataset splits...")
    train_dataset = load_and_format_dataset(args.train_file, tokenizer)
    val_dataset = load_and_format_dataset(args.val_file, tokenizer) if args.val_file else None

    # 4. Configure Training Arguments
    print("\n[4/5] Configuring SFT Trainer hyper-parameters...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_steps=1,
        optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        save_strategy="epoch",
        evaluation_strategy="epoch" if val_dataset else "no",
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
        packing=False, # Set to False for forensic boundary integrity
        args=training_args,
    )

    # 5. Execute Fine-Tuning
    print("\n[5/5] Launching GPU Training...")
    trainer_stats = trainer.train()
    print(f"\n[Training Complete] Runtime: {trainer_stats.metrics['train_runtime']:.2f}s | Global Steps: {trainer_stats.global_step}")

    # 6. Save LoRA Adapters & Tokenizer
    final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
    print(f"\n[Saving] Saving fine-tuned LoRA adapter to {final_adapter_dir}...")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    if args.push_to_hub:
        print(f"[Hub] Pushing fine-tuned model adapter to Hugging Face Hub: {args.push_to_hub}...")
        model.push_to_hub_merged(args.push_to_hub, tokenizer, save_method="lora")

    print("\n" + "=" * 75)
    print(f"  SUCCESS: Fine-tuned LoRA adapter saved to '{final_adapter_dir}'")
    print(f"  Next step: Run evaluation using ./finetune_and_eval.sh to verify metrics!")
    print("=" * 75)

if __name__ == "__main__":
    main()
