#!/bin/bash
# ==============================================================================
# finetune_and_eval.sh
# One-Command End-to-End Pipeline:
# 1. Fine-Tunes LLM on Forensic Training Split using Unsloth LoRA on H100/A100 GPU
# 2. Runs Benchmark Evaluation on 23 Held-Out Validation Windows
# 3. Produces Side-by-Side Before vs. After Scorecard
# ==============================================================================

set -e

MODEL_NAME=${1:-"unsloth/gemma-2-2b-it"}
OUTPUT_DIR="outputs_forensic_gemma_2b"
TRAIN_DATA="data/splits/train.jsonl"
VAL_DATA="data/splits/val.jsonl"
EVENTS_DB="data/raw_database/events_db.jsonl"
GROUND_TRUTH="data/raw_database/ground_truth.csv"

echo "======================================================================"
echo "  KRYPTO: END-TO-END FORENSIC FINE-TUNING & EVALUATION PIPELINE"
echo "  Base Model:    $MODEL_NAME"
echo "  Train Data:    $TRAIN_DATA (90 Windows)"
echo "  Val Benchmark: $VAL_DATA (23 Held-Out Windows)"
echo "  Output Dir:    $OUTPUT_DIR"
echo "======================================================================"

# Step 1: Ensure dataset splits are prepared
if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
    echo "[Step 1/3] Preparing date-held-out dataset partitions..."
    python3 prepare_splits.py
fi

# Step 2: Run Fine-Tuning with Unsloth / LoRA
echo ""
echo "[Step 2/3] Starting LoRA Fine-Tuning on GPU..."
python3 train_unsloth.py \
  --model_name "$MODEL_NAME" \
  --train_file "$TRAIN_DATA" \
  --val_file "$VAL_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length 8192 \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 4 \
  --learning_rate 2e-4 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --chat_template "gemma"

# Step 3: Run Evaluation on the Fine-Tuned Model Adapter
ADAPTER_PATH="$OUTPUT_DIR/final_adapter"
echo ""
echo "[Step 3/3] Running Automated Benchmark Evaluation on Fine-Tuned Model..."
python3 -m harness.runner \
  --model "$MODEL_NAME" \
  --adapter "$ADAPTER_PATH" \
  --data "$VAL_DATA" \
  --ground_truth "$GROUND_TRUTH" \
  --events_db "$EVENTS_DB" \
  --output_json "eval_reports/eval_finetuned_gemma_val.json" \
  --output_md "eval_reports/eval_finetuned_gemma_val.md" \
  --max_new_tokens 350

echo ""
echo "======================================================================"
echo "  PIPELINE COMPLETE!"
echo "  Fine-Tuned Adapter: $ADAPTER_PATH"
echo "  Evaluation Report:  eval_reports/eval_finetuned_gemma_val.md"
echo "======================================================================"
