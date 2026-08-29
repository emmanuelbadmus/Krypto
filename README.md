# Forensic LLM Activity Auditing & Reconstruction

A specialized framework and evaluation harness for fine-tuning Large Language Models (LLMs) to perform **digital forensic activity auditing** and **timeline reconstruction** directly from logical device extractions (SQLite databases and system logs).

Based on principles from *"AI Agents in Depth: Design Principles and Engineering Practice"* ($\text{Agent} = \text{LLM} + \text{Context} + \text{Tools}$).

---

## 🎯 Core Objectives

Generic LLMs fail at digital forensic auditing because they summarize events rather than maintaining evidential audit trails, hallucinate unsupported claims, and misinterpret server synchronization timestamps.

This repository enforces **four core forensic behaviors**:
1. **Strict Citation Discipline**: Every assertion must end with `[EVT-xxxxxxxxxxxx]`, mechanically resolving to `database :: table :: row_id :: timestamp`.
2. **Absence as Evidence**: Distinguishes between inactivity and unrecoverable activity (e.g. encrypted stores, external state).
3. **Residue / Lifecycle Analysis**: Recognizes orphaned artifacts (e.g. cookies) to detect installed, used, and uninstalled apps.
4. **Server vs. Device Timestamp Discrimination**: Detects and flags server-synced content predating device provisioning.

---

## 📂 Repository Structure

```
.
├── harness/                      # Pluggable Forensic Evaluation Harness
│   ├── __init__.py
│   ├── models.py                 # Runners for Hugging Face, Unsloth, LoRA, and APIs
│   ├── evaluators.py             # Deterministic citation and absence verifiers
│   └── runner.py                 # CLI driver & Markdown/JSON audit report generator
├── xx/
│   ├── unseen_training_data/     # Original handoff data (ground_truth.csv, events.jsonl, unlabelled.jsonl)
│   ├── train_augmented.jsonl     # Augmented 113-sample dataset with session/episode windowing
│   ├── train_split.jsonl         # Date-held-out training split (90 windows)
│   ├── val_split.jsonl           # Date-held-out validation split (23 windows)
│   ├── Dataset_Handoff.pdf       # Project specifications and baseline extraction metrics
│   └── evaluation_report.txt     # Baseline extraction benchmark
├── prepare_splits.py             # Script to generate date-held-out dataset partitions
├── train_unsloth.py              # Turnkey Unsloth LoRA fine-tuning script (H100 / A100 optimized)
├── evaluate_forensic.py          # Standalone metric evaluator
├── requirements_h100.txt         # GPU environment dependencies
└── run_h100.sh                   # End-to-end execution pipeline script
```

---

## 🚀 Quick Start

### 1. Fine-Tuning on GPU (H100 / A100)
```bash
pip install -r requirements_h100.txt

# Run the complete fine-tuning and evaluation pipeline
./run_h100.sh
```

### 2. Evaluating Models with the Harness
Run evaluation on unseen test windows with any model or LoRA adapter:
```bash
# Evaluate base Gemma-4-E2B-it
python3 -m harness.runner \
  --model "models/gemma-4-E2B-it" \
  --data "xx/unseen_training_data/unlabelled.jsonl" \
  --ground_truth "xx/unseen_training_data/ground_truth.csv" \
  --events_db "xx/unseen_training_data/events.jsonl" \
  --output_md "eval_base_gemma4.md"

# Evaluate fine-tuned checkpoint
python3 -m harness.runner \
  --model "models/gemma-4-E2B-it" \
  --adapter "outputs_forensic_gemma_2b/final_adapter" \
  --data "xx/unseen_training_data/unlabelled.jsonl" \
  --output_md "eval_finetuned_gemma4.md"
```
