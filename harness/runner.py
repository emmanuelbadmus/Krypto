#!/usr/bin/env python3
"""
harness/runner.py
CLI and main driver for the Forensic Activity Reconstruction Evaluation Harness.
Supports pluggable models (Gemma-4-E2B, Gemma-2-2B, Qwen2.5, Llama, LoRA adapters)
and outputs structured JSON and Markdown audit reports.
"""

import os
import json
import argparse
from typing import List, Dict, Any
from harness.models import get_model_runner
from harness.evaluators import ForensicEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Forensic LLM Evaluation Harness")
    parser.add_argument("--model", type=str, default="google/gemma-4-E2B-it",
                        help="Model name, path, or Hugging Face repository ID")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Optional path to fine-tuned PEFT / LoRA adapter")
    parser.add_argument("--data", type=str, default="xx/unseen_training_data/unlabelled.jsonl",
                        help="Evaluation dataset file (e.g., unlabelled.jsonl, train.jsonl, or val_split.jsonl)")
    parser.add_argument("--ground_truth", type=str, default="xx/unseen_training_data/ground_truth.csv",
                        help="Path to ground_truth.csv")
    parser.add_argument("--events_db", type=str, default="xx/unseen_training_data/events.jsonl",
                        help="Path to events.jsonl database (optional)")
    parser.add_argument("--output_json", type=str, default="eval_report.json",
                        help="Path for output JSON metrics")
    parser.add_argument("--output_md", type=str, default="eval_report.md",
                        help="Path for human-readable markdown report")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of windows to evaluate (optional)")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "hf", "unsloth"])
    parser.add_argument("--chat_template", type=str, default="gemma")
    return parser.parse_args()

def extract_date_from_prompt(user_prompt: str) -> str:
    for line in user_prompt.split("\n"):
        if line.startswith("Date:") or line.startswith("Window:"):
            parts = line.split(":", 1)[1].strip().split()
            if parts:
                return parts[0]
    return None

def main():
    args = parse_args()
    print("=" * 70)
    print("  FORENSIC ACTIVITY RECONSTRUCTION EVALUATION HARNESS")
    print(f"  Model:        {args.model}")
    print(f"  Adapter:      {args.adapter if args.adapter else 'None (Base Model)'}")
    print(f"  Dataset:      {args.data}")
    print(f"  Ground Truth: {args.ground_truth}")
    print("=" * 70)

    # 1. Initialize Evaluator
    evaluator = ForensicEvaluator(
        ground_truth_csv=args.ground_truth if os.path.exists(args.ground_truth) else None,
        events_jsonl=args.events_db if (args.events_db and os.path.exists(args.events_db)) else None,
    )

    # 2. Load Evaluation Data
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Evaluation dataset not found at {args.data}")

    with open(args.data, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"[Harness] Loaded {len(samples)} windows for evaluation.")

    # 3. Load Model Runner
    runner = get_model_runner(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        backend=args.backend,
        chat_template=args.chat_template,
    )

    # 4. Run Evaluation Loop
    results = []
    for idx, sample in enumerate(samples):
        messages = sample.get("messages", [])
        if len(messages) < 2:
            continue

        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        target_answer = messages[2]["content"] if len(messages) > 2 else None

        window_date = extract_date_from_prompt(user_prompt)

        print(f"\n--- Evaluating Window [{idx+1}/{len(samples)}] (Date: {window_date}) ---")
        prediction = runner.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        metrics = evaluator.evaluate_window(
            prediction=prediction,
            user_prompt=user_prompt,
            target_answer=target_answer,
            window_date=window_date,
        )

        print(f"  Citations: {metrics['valid_citations']}/{metrics['total_citations']} valid (Precision: {metrics['citation_precision']*100:.1f}%)")
        print(f"  Absence Handling: {'Detected' if metrics['pred_has_absence_reasoning'] else 'Not Detected'}")

        results.append({
            "window_index": idx + 1,
            "date": window_date,
            "prediction": prediction,
            "target": target_answer,
            "metrics": metrics,
        })

    # 5. Compute Aggregate Summary
    summary = evaluator.aggregate_results(results)
    summary["model_name"] = args.model
    summary["adapter_path"] = args.adapter
    summary["eval_dataset"] = args.data

    # 6. Save JSON Report
    full_output = {
        "summary": summary,
        "detailed_windows": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n[Harness] JSON metrics saved to: {args.output_json}")

    # 7. Generate Markdown Report
    generate_markdown_report(summary, results, args.output_md)
    print(f"[Harness] Markdown audit report saved to: {args.output_md}")

    # 8. Print Terminal Summary
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print(f"  Total Windows Evaluated:       {summary.get('total_windows_evaluated', 0)}")
    print(f"  Overall Citation Precision:    {summary.get('overall_citation_precision', 0.0) * 100:.2f}%")
    print(f"  Hallucination Rate:            {summary.get('hallucination_rate', 0.0) * 100:.2f}%")
    print(f"  Absence Handling Accuracy:     {summary.get('absence_handling_accuracy', 0.0) * 100:.2f}%")
    print(f"  Ground-Truth Matches:          {summary.get('total_ground_truth_matches', 0)}")
    print("=" * 70)


def generate_markdown_report(summary: Dict[str, Any], results: List[Dict[str, Any]], md_path: str):
    """Formats evaluation results into a clean markdown document."""
    lines = [
        "# Forensic Activity Reconstruction Evaluation Report",
        "",
        f"- **Model**: `{summary.get('model_name')}`",
        f"- **LoRA Adapter**: `{summary.get('adapter_path', 'None (Base Model)')}`",
        f"- **Dataset Evaluated**: `{summary.get('eval_dataset')}`",
        f"- **Total Windows**: {summary.get('total_windows_evaluated', 0)}",
        "",
        "## Executive Summary",
        "",
        "| Metric | Score | Description |",
        "| :--- | :--- | :--- |",
        f"| **Citation Precision** | **{summary.get('overall_citation_precision', 0.0) * 100:.2f}%** | Percentage of emitted `[EVT-...]` IDs that exist in input provenance |",
        f"| **Hallucination Rate** | **{summary.get('hallucination_rate', 0.0) * 100:.2f}%** | Claims made with phantom/fabricated `[EVT-...]` IDs |",
        f"| **Absence Handling** | **{summary.get('absence_handling_accuracy', 0.0) * 100:.2f}%** | Correctly identifying missing/encrypted/unrecoverable artifacts |",
        f"| **Ground-Truth Matches** | **{summary.get('total_ground_truth_matches', 0)}** | Matched documented events from `ground_truth.csv` |",
        "",
        "## Window-by-Window Audit Breakdown",
        "",
        "| Window # | Date | Total Citations | Valid Citations | Precision | Absence Detected |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        m = r["metrics"]
        lines.append(
            f"| {r['window_index']} | {r['date']} | {m['total_citations']} | {m['valid_citations']} | {m['citation_precision']*100:.1f}% | {'✅' if m['pred_has_absence_reasoning'] else '❌'} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
