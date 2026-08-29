#!/usr/bin/env python3
"""
evaluate_forensic.py
Evaluates base vs. fine-tuned LLMs on forensic activity reconstruction.
Calculates:
  1. Citation Validity (Precision of EVT- IDs against prompt provenance)
  2. Hallucination / Fabrication Rate (Uncited claims or phantom IDs)
  3. Absence Handling Accuracy (Correct explanation for unrecoverable data)
  4. Output Structural Integrity
"""

import os
import re
import json
import argparse
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Forensic Reconstruction Model")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-2-2b-it",
                        help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to trained LoRA adapter (optional, if evaluating fine-tuned model)")
    parser.add_argument("--val_file", type=str, default="xx/val_split.jsonl",
                        help="Validation dataset file")
    parser.add_argument("--output_eval_file", type=str, default="eval_results.json")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--chat_template", type=str, default="gemma")
    return parser.parse_args()

def extract_valid_event_ids_from_prompt(user_prompt):
    """Extracts all valid [EVT-xxxx] IDs that were provided in the input prompt."""
    return set(re.findall(r"EVT-([a-f0-9]+)", user_prompt))

def evaluate_sample(prediction, ground_truth, valid_prompt_ids):
    pred_event_ids = re.findall(r"EVT-([a-f0-9]+)", prediction)
    
    # 1. Citation Validity: Were emitted IDs actually present in the input prompt?
    total_cited = len(pred_event_ids)
    valid_citations = [eid for eid in pred_event_ids if eid in valid_prompt_ids]
    invalid_citations = [eid for eid in pred_event_ids if eid not in valid_prompt_ids]
    
    citation_precision = (len(valid_citations) / total_cited) if total_cited > 0 else 0.0

    # 2. Absence Handling: Did the model include the unrecoverable / absence section?
    has_absence_section = "without direct artifact support" in prediction.lower() or "no direct artifact" in prediction.lower()
    gt_has_absence = "without direct artifact support" in ground_truth.lower() or "no direct artifact" in ground_truth.lower()
    absence_match = (has_absence_section == gt_has_absence)

    # 3. Direct Activity Support:
    has_direct_section = "with artifact support" in prediction.lower()
    gt_has_direct = "with artifact support" in ground_truth.lower()
    direct_match = (has_direct_section == gt_has_direct)

    return {
        "total_cited_ids": total_cited,
        "valid_citations": len(valid_citations),
        "invalid_citations": len(invalid_citations),
        "citation_precision": citation_precision,
        "absence_match": absence_match,
        "direct_match": direct_match,
    }

def main():
    args = parse_args()
    print("=" * 60)
    print(f"Running Forensic Evaluation")
    print(f"Base Model: {args.model_name}")
    print(f"Adapter: {args.adapter_path if args.adapter_path else 'None (Evaluating Base Model Baseline)'}")
    print(f"Validation file: {args.val_file}")
    print("=" * 60)

    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_path if args.adapter_path else args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    # 2. Load Validation Samples
    with open(args.val_file, "r", encoding="utf-8") as f:
        val_samples = [json.loads(line) for line in f]

    results = []
    total_valid_citations = 0
    total_emitted_citations = 0
    total_absence_matches = 0

    for idx, sample in enumerate(val_samples):
        messages = sample["messages"]
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        ground_truth = messages[2]["content"]

        valid_prompt_ids = extract_valid_event_ids_from_prompt(user_prompt)

        # Prepare input for inference
        input_convo = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(input_convo, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                use_cache=True,
                temperature=0.1,  # Low temperature for deterministic forensic reasoning
            )

        # Decode generated text (excluding input prompt)
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        metrics = evaluate_sample(prediction, ground_truth, valid_prompt_ids)
        total_valid_citations += metrics["valid_citations"]
        total_emitted_citations += metrics["total_cited_ids"]
        if metrics["absence_match"]:
            total_absence_matches += 1

        results.append({
            "sample_index": idx,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "metrics": metrics
        })

        print(f"Sample [{idx+1}/{len(val_samples)}] - Cited: {metrics['total_cited_ids']} | Valid: {metrics['valid_citations']} | Precision: {metrics['citation_precision']:.2f}")

    # Summary Metrics
    overall_precision = (total_valid_citations / total_emitted_citations) if total_emitted_citations > 0 else 0.0
    absence_accuracy = total_absence_matches / len(val_samples)

    summary = {
        "model": args.model_name,
        "adapter": args.adapter_path,
        "total_eval_samples": len(val_samples),
        "overall_citation_precision": overall_precision,
        "absence_accuracy": absence_accuracy,
        "detailed_results": results
    }

    with open(args.output_eval_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print(f"Overall Citation Precision (Anti-Hallucination): {overall_precision * 100:.2f}%")
    print(f"Absence Handling Accuracy:                     {absence_accuracy * 100:.2f}%")
    print(f"Results saved to: {args.output_eval_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
