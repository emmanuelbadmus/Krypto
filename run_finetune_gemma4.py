"""
Krypto: Mobile Forensic Activity Reconstruction Fine-Tuning Script
Fine-tunes Gemma-4-E2B-it on GPU / Google Colab with Response-Only Loss Masking.
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
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-4-E2B-it for Forensic Reconstruction")
    parser.add_argument("--model_name", type=str, default="google/gemma-4-E2B-it", help="Model name or local path")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face access token")
    parser.add_argument("--train_file", type=str, default="data/splits/train.jsonl", help="Train dataset path")
    parser.add_argument("--val_file", type=str, default="data/splits/val.jsonl", help="Val dataset path")
    parser.add_argument("--output_dir", type=str, default="outputs_forensic_gemma4_2b", help="Output directory")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device train batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (for testing)")
    return parser.parse_args()


def setup_auth(token_arg):
    token = token_arg
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
        except Exception:
            pass
    if not token:
        token = os.environ.get('HF_TOKEN')
    
    if token:
        from huggingface_hub import login
        try:
            login(token=token)
            print("✅ Successfully authenticated with Hugging Face!")
        except Exception as e:
            print(f"⚠️ Hugging Face login warning: {e}")
    else:
        print("ℹ️ No HF_TOKEN provided; attempting public access.")
    return token


def load_model_and_tokenizer(model_path, hf_token=None):
    if os.path.exists("models/gemma-4-E2B-it/model.safetensors"):
        model_path = "models/gemma-4-E2B-it"
    
    print(f"📦 Loading Tokenizer and Model from: {model_path}")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            token=hf_token,
            trust_remote_code=True,
        )
    except Exception:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            token=hf_token,
            trust_remote_code=True,
        )

    model = model.to(device)
    print(f"✅ Model successfully loaded on {device} (dtype: {dtype})")
    return model, tokenizer, device


def apply_lora(model):
    print("💉 Injecting LoRA Target Adapters into Language Model...")
    
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        num_layers = len(model.model.language_model.layers)
        target_mods = [f"model.language_model.layers.{i}.self_attn.{p}" for i in range(num_layers) for p in ["q_proj", "k_proj", "v_proj", "o_proj"]] + \\\n                      [f"model.language_model.layers.{i}.mlp.{p}" for i in range(num_layers) for p in ["gate_proj", "up_proj", "down_proj"]]
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
    return model


def prepare_and_tokenize(filepath, tokenizer, max_len=2048):
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            msgs = item.get("messages", [])
            if len(msgs) >= 2:
                gemma_msgs = []
                system_text = ""
                for m in msgs:
                    if m["role"] == "system":
                        system_text = m["content"].strip() + "\n\n"
                    elif m["role"] == "user":
                        gemma_msgs.append({
                            "role": "user",
                            "content": (system_text + m["content"]).strip()
                        })
                        system_text = ""
                    elif m["role"] in ("assistant", "model"):
                        gemma_msgs.append({
                            "role": "assistant",
                            "content": m["content"]
                        })
                
                full_text = tokenizer.apply_chat_template(gemma_msgs, tokenize=False, add_generation_prompt=False)
                user_only_msgs = [gemma_msgs[0]]
                prompt_text = tokenizer.apply_chat_template(user_only_msgs, tokenize=False, add_generation_prompt=True)
                
                samples.append({"full_text": full_text, "prompt_text": prompt_text})
    
    raw_ds = Dataset.from_list(samples)
    
    def tok_fn(batch):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for full_t, prompt_t in zip(batch["full_text"], batch["prompt_text"]):
            encoded = tokenizer(full_t, truncation=True, max_length=max_len)
            prompt_enc = tokenizer(prompt_t, truncation=True, max_length=max_len, add_special_tokens=False)
            
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            labels = list(input_ids)
            
            # Mask input prompt with -100 so loss is computed ONLY on assistant response!
            prompt_len = min(len(prompt_enc["input_ids"]), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
            
        return {"input_ids": input_ids_list, "attention_mask": attention_mask_list, "labels": labels_list}
        
    tokenized_ds = raw_ds.map(tok_fn, batched=True, remove_columns=["full_text", "prompt_text"])
    return tokenized_ds


def evaluate_model(model, tokenizer, val_file, device):
    print("\n" + "=" * 65)
    print("  AUTOMATED IN-NOTEBOOK BENCHMARK EVALUATION")
    print("=" * 65)
    
    model.eval()
    model.to(device)

    total_cited = 0
    valid_cited = 0
    hallucinated_cited = 0
    absence_detected_count = 0
    results = []

    with open(val_file, "r", encoding="utf-8") as f:
        val_lines = [json.loads(l) for l in f if l.strip()]

    for idx, item in enumerate(val_lines):
        msgs = item.get("messages", [])
        sys_prompt = msgs[0]["content"] if len(msgs) > 0 and msgs[0]["role"] == "system" else ""
        user_prompt = msgs[1]["content"] if len(msgs) > 1 and msgs[1]["role"] == "user" else ""

        valid_prompt_eids = set(re.findall(r"EVT-([a-f0-9]+)", user_prompt, flags=re.IGNORECASE))

        inp_msgs = [{"role": "user", "content": (sys_prompt + "\n\n" + user_prompt).strip()}]
        inputs = tokenizer.apply_chat_template(inp_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        input_ids = inputs.input_ids if hasattr(inputs, "input_ids") else inputs
        prompt_len = input_ids.shape[1]

        with torch.inference_mode():
            outputs = model.generate(input_ids=input_ids, max_new_tokens=300, temperature=0.1, use_cache=True)
        pred_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        pred_eids = re.findall(r"EVT-([a-f0-9]+)", pred_text, flags=re.IGNORECASE)
        v_c = [e for e in pred_eids if e in valid_prompt_eids]
        h_c = [e for e in pred_eids if e not in valid_prompt_eids]

        total_cited += len(pred_eids)
        valid_cited += len(v_c)
        hallucinated_cited += len(h_c)

        has_absence = any(kw in pred_text.lower() for kw in ["without direct artifact support", "sqlcipher", "encrypted", "unrecoverable", "no matching artifact"])
        if has_absence:
            absence_detected_count += 1

        prec = (len(v_c) / len(pred_eids)) if pred_eids else 1.0
        results.append([f"Window #{idx+1}", len(pred_eids), len(v_c), f"{prec*100:.1f}%", "✅" if has_absence else "-"])

    precision = (valid_cited / total_cited * 100) if total_cited > 0 else 100.0
    print("\n" + "=" * 65)
    print("  FINAL FORENSIC EVALUATION SCORECARD")
    print("=" * 65)
    print(f"  Total Windows Evaluated:     {len(val_lines)}")
    print(f"  Overall Citation Precision:  {precision:.2f}%\")\n",
    print(f"  Hallucinated Phantom IDs:    {hallucinated_cited}")
    print(f"  Absence Auditing Accuracy:   {(absence_detected_count / len(val_lines) * 100):.1f}%\")\n",
    print("=" * 65)
    print(tabulate(results[:10], headers=["Window", "Total Cited", "Valid", "Precision", "Absence"], tablefmt="grid"))


def main():
    args = parse_args()
    
    print("🧹 Cleaning GPU Cache...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    hf_token = setup_auth(args.hf_token)
    model, tokenizer, device = load_model_and_tokenizer(args.model_name, hf_token=hf_token)
    model = apply_lora(model)

    print(f"📊 Preparing datasets with Response-Only Loss Masking from {args.train_file} and {args.val_file}...")
    train_dataset = prepare_and_tokenize(args.train_file, tokenizer, max_len=args.max_seq_len)
    val_dataset = prepare_and_tokenize(args.val_file, tokenizer, max_len=args.max_seq_len)
    print(f"✅ Ready: {len(train_dataset)} training samples | {len(val_dataset)} validation samples.")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        warmup_steps=10,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
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

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print(f"\n🚀 Starting GPU Fine-Tuning ({args.epochs} Epochs with Cosine Decay)...")
    trainer_stats = trainer.train()
    print(f"✅ Training Complete! Runtime: {trainer_stats.metrics['train_runtime']:.2f}s")

    final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
    print(f"💾 Saving LoRA adapter to {final_adapter_dir}...")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print("✅ Adapter saved successfully!")

    evaluate_model(model, tokenizer, args.val_file, device)


if __name__ == "__main__":
    main()

