"""
harness/evaluators.py
Automated verifiers and forensic metric evaluators.
Evaluates Citation Validity, Anti-Hallucination Rate, Absence Reasoning,
and Ground-Truth Recall against documented forensic records.
"""

import re
import csv
import json
from collections import defaultdict
from typing import Dict, List, Set, Any

class ForensicEvaluator:
    """Comprehensive forensic evaluation engine."""

    def __init__(self, ground_truth_csv: str = None, events_jsonl: str = None):
        self.ground_truth = []
        self.events_db = {}
        
        if ground_truth_csv:
            self.load_ground_truth(ground_truth_csv)
        if events_jsonl:
            self.load_events_db(events_jsonl)

    def load_ground_truth(self, csv_path: str):
        print(f"[ForensicEvaluator] Loading ground truth from {csv_path}...")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.ground_truth = list(reader)
        print(f"[ForensicEvaluator] Loaded {len(self.ground_truth)} ground truth events.")

    def load_events_db(self, jsonl_path: str):
        print(f"[ForensicEvaluator] Indexing events database from {jsonl_path}...")
        count = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                eid = event.get("event_id")
                if eid:
                    self.events_db[eid] = event
                    count += 1
        print(f"[ForensicEvaluator] Indexed {count} provenance events.")

    def extract_prompt_event_ids(self, user_prompt: str) -> Set[str]:
        """Extracts all [EVT-xxxx] IDs that exist in the user's prompt."""
        return set(re.findall(r"EVT-([a-f0-9]+)", user_prompt))

    def evaluate_window(self, prediction: str, user_prompt: str, target_answer: str = None, window_date: str = None) -> Dict[str, Any]:
        """Evaluates a single window prediction."""
        valid_prompt_ids = self.extract_prompt_event_ids(user_prompt)
        pred_eids = re.findall(r"EVT-([a-f0-9]+)", prediction)

        # 1. Citation Discipline & Anti-Hallucination
        total_cited = len(pred_eids)
        valid_citations = [eid for eid in pred_eids if eid in valid_prompt_ids]
        hallucinated_citations = [eid for eid in pred_eids if eid not in valid_prompt_ids]
        
        citation_precision = (len(valid_citations) / total_cited) if total_cited > 0 else 1.0 if not pred_eids else 0.0

        # Check citations against master DB if loaded
        in_master_db = [eid for eid in pred_eids if eid in self.events_db]

        # 2. Absence & Negative Constraint Reasoning
        absence_keywords = [
            "without direct artifact support",
            "no direct artifact",
            "outside sqlite",
            "encrypted",
            "no application data present",
            "documented activity",
        ]
        pred_has_absence = any(k in prediction.lower() for k in absence_keywords)
        
        target_has_absence = False
        if target_answer:
            target_has_absence = any(k in target_answer.lower() for k in absence_keywords)

        absence_handled_correctly = (pred_has_absence == target_has_absence) if target_answer else pred_has_absence

        # 3. Ground Truth Recall (if date provided and GT available)
        matched_gt_ids = []
        if window_date and self.ground_truth:
            gt_for_date = [gt for gt in self.ground_truth if gt.get("date") == window_date]
            for gt in gt_for_date:
                app = gt.get("app", "").lower()
                content = gt.get("content", "").lower()
                # Check if app and content snippet are mentioned in prediction
                if app in prediction.lower():
                    words = [w for w in content.split() if len(w) > 3]
                    if any(w in prediction.lower() for w in words):
                        matched_gt_ids.append(gt.get("gt_id"))

        return {
            "total_citations": total_cited,
            "valid_citations": len(valid_citations),
            "hallucinated_citations": len(hallucinated_citations),
            "citation_precision": citation_precision,
            "pred_has_absence_reasoning": pred_has_absence,
            "absence_handled_correctly": absence_handled_correctly,
            "matched_ground_truth_count": len(matched_gt_ids),
            "matched_gt_ids": matched_gt_ids,
        }

    def aggregate_results(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Computes aggregate summary metrics across all evaluated windows."""
        total_samples = len(window_results)
        if total_samples == 0:
            return {}

        total_citations = sum(r["metrics"]["total_citations"] for r in window_results)
        valid_citations = sum(r["metrics"]["valid_citations"] for r in window_results)
        hallucinated = sum(r["metrics"]["hallucinated_citations"] for r in window_results)
        
        avg_precision = (valid_citations / total_citations) if total_citations > 0 else 0.0
        absence_accuracy = sum(1 for r in window_results if r["metrics"]["absence_handled_correctly"]) / total_samples
        total_gt_matches = sum(r["metrics"]["matched_ground_truth_count"] for r in window_results)

        return {
            "total_windows_evaluated": total_samples,
            "total_citations_emitted": total_citations,
            "valid_citations": valid_citations,
            "hallucinated_citations": hallucinated,
            "overall_citation_precision": avg_precision,
            "hallucination_rate": (hallucinated / total_citations) if total_citations > 0 else 0.0,
            "absence_handling_accuracy": absence_accuracy,
            "total_ground_truth_matches": total_gt_matches,
        }

