#!/bin/bash
set -e

echo "=== 1. Splitting Dataset (Date-Held-Out Split) ==="
python3 prepare_splits.py

echo "=== 2. Fine-Tuning Gemma-2-2B-it with Unsloth on H100 ==="
python3 train_unsloth.py \
  --model_name "unsloth/gemma-2-2b-it" \
  --train_file "data/splits/train.jsonl" \
  --val_file "data/splits/val.jsonl" \
  --output_dir "outputs_forensic_gemma_2b" \
  --max_seq_length 8192 \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 4 \
  --learning_rate 2e-4 \
  --chat_template "gemma"

echo "=== 3. Evaluating Base Model Baseline ==="
python3 evaluate_forensic.py \
  --model_name "unsloth/gemma-2-2b-it" \
  --val_file "data/splits/val.jsonl" \
  --output_eval_file "eval_reports/eval_base_gemma_2b.json" \
  --chat_template "gemma"

echo "=== 4. Evaluating Fine-Tuned LoRA Adapter ==="
python3 evaluate_forensic.py \
  --model_name "unsloth/gemma-2-2b-it" \
  --adapter_path "outputs_forensic_gemma_2b/final_adapter" \
  --val_file "data/splits/val.jsonl" \
  --output_eval_file "eval_reports/eval_finetuned_gemma_2b.json" \
  --chat_template "gemma"

echo "=== Pipeline Completed Successfully! ==="
