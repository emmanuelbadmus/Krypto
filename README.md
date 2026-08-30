# 🔬 Krypto: Mobile Forensic Activity Reconstruction Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emmanuelbadmus/Krypto/blob/main/FineTune_Gemma_Forensics.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/Model-Gemma--4--E2B--it-purple.svg)](https://huggingface.co/google/gemma-4-E2B-it)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An open-weight, privacy-preserving LLM benchmark and fine-tuning pipeline for **ground-truth mobile forensic timeline reconstruction** from multi-app SQLite databases, system logs, and communication artifacts.

---

## 🎯 Benchmark Overview

In digital forensics and criminal investigations (ISO/IEC 27037 & Daubert standards), forensic timeline reconstruction requires:
1. **Zero Hallucinations**: Every activity claim must cite valid, verifiable `[EVT-...]` artifact IDs extracted directly from on-disk databases.
2. **Strict Chronological Sequencing**: Events across multiple applications (SMS, Twitter, Line, Battery power events, WhatsApp, Signal) must be temporally aligned.
3. **Absence & SQLCipher Auditing**: If an encrypted application (e.g. Signal, SQLCipher) or unrecoverable artifact is encountered, it must be explicitly audited rather than omitted.
4. **Air-Gapped Local Deployment**: Forensic data cannot be transmitted to commercial cloud APIs (GPT-4 / Claude) due to privacy and chain-of-custody requirements.

---

## 📊 Benchmark Results (Gemma-4-E2B-it)

Evaluated on **Date-Held-Out test splits** (`data/splits/val.jsonl`):

| Metric | Zero-Shot Baseline | Fine-Tuned Gemma-4-E2B-it (LoRA) |
| :--- | :---: | :---: |
| **Citation Precision** | 96.58% | **98.61% - 100.0%** |
| **Grounded Evidence Retrieval** | ~70 events | **140 Valid Forensic Events** |
| **Perfect 100% Precision Windows** | 17.4% (4/23) | **82.6% (19/23)** |
| **Training Loss Descent** | N/A | **3.91 ➔ 0.056** |
| **Air-Gapped / Local Inference** | ✅ | ✅ |

---

## 🏗️ Repository Architecture

```
Krypto/
├── FineTune_Gemma_Forensics.ipynb  # Interactive 10-step Google Colab notebook
├── run_finetune_gemma4.py          # Standalone GPU training & evaluation script
├── evaluate_forensic.py            # CLI benchmark evaluation suite
├── prepare_splits.py               # Date-held-out dataset split generator
├── requirements_gpu.txt            # Python dependencies for GPU/Colab
│
├── data/
│   ├── splits/                     # Date-held-out train/val JSONL splits
│   │   ├── train.jsonl             # 90 training daily windows
│   │   └── val.jsonl               # 23 validation daily windows
│   ├── raw_database/               # Raw database events and ground truth
│   └── reports_and_docs/           # Benchmark baseline documentation
│
├── eval_reports/                   # Generated evaluation scorecards & markdown reports
└── harness/                        # Evaluation harnesses and metrics calculator
```

---

## 🚀 Quickstart

### 1. Interactive Google Colab (Recommended)
Open and run directly on a GPU instance:
👉 **[`FineTune_Gemma_Forensics.ipynb`](https://colab.research.google.com/github/emmanuelbadmus/Krypto/blob/main/FineTune_Gemma_Forensics.ipynb)**

### 2. Standalone GPU Execution
```bash
# Clone the repository
git clone https://github.com/emmanuelbadmus/Krypto.git
cd Krypto

# Install dependencies
pip install -r requirements_gpu.txt

# Run 6-epoch LoRA fine-tuning with Response-Only Loss Masking
python run_finetune_gemma4.py \
    --model_name "google/gemma-4-E2B-it" \
    --epochs 6 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 1.5e-4 \
    --output_dir outputs_forensic_gemma4_2b
```

---

## 🔬 Methodology & Key Innovations

* **Response-Only Loss Masking (`labels = -100`)**: Masks the 80+ input artifacts in the user prompt during loss calculation, focusing 100% of gradient capacity on generating accurate chronological reasoning and `[EVT-...]` citations.
* **LoRA Target Adaptation**: Targets all 35 language model projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with rank $r=32$ and $\alpha=64$ (48.3M trainable parameters, 0.94% of total model size).
* **Strict Whitelist Verification**: Automated validation harness to cross-check all emitted event IDs against the raw database rows.
