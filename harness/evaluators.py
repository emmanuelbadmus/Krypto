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
                raw_eid = event.get("event_id", "")
                clean_eid = raw_eid.replace("EVT-", "").strip().lower()
                
                # Normalize schema fields
                norm_event = {
                    "event_id": f"EVT-{clean_eid}" if clean_eid else raw_eid,
                    "clean_id": clean_eid,
                    "app": event.get("source") or event.get("app"),
                    "event_type": event.get("event_type"),
                    "timestamp": event.get("ts_local") or event.get("timestamp") or event.get("ts_utc"),
                    "ts_utc": event.get("ts_utc"),
                    "ts_local": event.get("ts_local"),
                    "db_path": event.get("artifact") or event.get("db_path"),
                    "table": event.get("table"),
                    "row_id": event.get("row_id"),
                    "time_column": event.get("time_column"),
                    "raw_timestamp": event.get("raw_timestamp"),
                    "epoch_type": event.get("epoch_format") or event.get("epoch_type"),
                    "party": event.get("party"),
                    "content": event.get("content"),
                    "raw_data": event.get("raw_data", {}),
                }

                # Store under both keys for instantaneous resolution
                if raw_eid:
                    self.events_db[raw_eid] = norm_event
                if clean_eid:
                    self.events_db[clean_eid] = norm_event
                count += 1
        print(f"[ForensicEvaluator] Indexed {count} provenance events.")

    def extract_prompt_event_ids(self, user_prompt: str) -> Set[str]:
        """Extracts all [EVT-xxxx] IDs that exist in the user's prompt."""
        return set(re.findall(r"EVT-([a-f0-9]+)", user_prompt, flags=re.IGNORECASE))

    def evaluate_window(self, prediction: str, user_prompt: str, target_answer: str = None, window_date: str = None) -> Dict[str, Any]:
        """Evaluates a single window prediction."""
        valid_prompt_ids = set(x.lower() for x in self.extract_prompt_event_ids(user_prompt))
        pred_eids = [x.lower() for x in re.findall(r"EVT-([a-f0-9]+)", prediction, flags=re.IGNORECASE)]

        # 1. Citation Discipline & Anti-Hallucination
        total_cited = len(pred_eids)
        valid_citations = [eid for eid in pred_eids if eid in valid_prompt_ids]
        hallucinated_citations = [eid for eid in pred_eids if eid not in valid_prompt_ids]
        
        citation_precision = (len(valid_citations) / total_cited) if total_cited > 0 else 1.0 if not pred_eids else 0.0

        # 2. Absence Reasoning Detection
        absence_keywords = [
            "without direct artifact support",
            "sqlcipher",
            "encrypted",
            "protobuf",
            "uninstalled",
            "no direct sqlite",
            "outside sqlite",
            "cookie only",
            "residue",
            "unrecoverable",
        ]
        pred_lower = prediction.lower()
        has_absence_reasoning = any(kw in pred_lower for kw in absence_keywords)
        
        target_has_absence = False
        if target_answer:
            target_has_absence = any(kw in target_answer.lower() for kw in absence_keywords)

        # 3. Ground Truth Matching (if date provided)
        gt_matches = 0
        total_gt = 0
        if window_date and self.ground_truth:
            day_gt = [row for row in self.ground_truth if row.get("date") == window_date]
            total_gt = len(day_gt)
            for row in day_gt:
                app_name = row.get("app", "").lower()
                if app_name and app_name in pred_lower:
                    gt_matches += 1

        return {
            "window_date": window_date,
            "total_citations": total_cited,
            "valid_citations": len(valid_citations),
            "hallucinated_citations": len(hallucinated_citations),
            "citation_precision": citation_precision,
            "pred_has_absence_reasoning": has_absence_reasoning,
            "target_has_absence": target_has_absence,
            "gt_matches": gt_matches,
            "total_gt_events": total_gt,
            "hallucinated_ids": hallucinated_citations,
        }

    def aggregate_results(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates window results into comprehensive benchmark summary."""
        total_windows = len(window_results)
        if total_windows == 0:
            return {}

        total_citations = sum(r["metrics"]["total_citations"] for r in window_results)
        valid_citations = sum(r["metrics"]["valid_citations"] for r in window_results)
        hallucinated_citations = sum(r["metrics"]["hallucinated_citations"] for r in window_results)
        
        overall_precision = (valid_citations / total_citations) if total_citations > 0 else 1.0
        hallucination_rate = (hallucinated_citations / total_citations) if total_citations > 0 else 0.0

        absence_correct = sum(
            1 for r in window_results
            if r["metrics"]["pred_has_absence_reasoning"] == r["metrics"]["target_has_absence"]
        )
        absence_accuracy = absence_correct / total_windows

        total_gt_matches = sum(r["metrics"]["gt_matches"] for r in window_results)

        return {
            "total_windows_evaluated": total_windows,
            "total_citations_emitted": total_citations,
            "valid_citations": valid_citations,
            "hallucinated_citations": hallucinated_citations,
            "overall_citation_precision": overall_precision,
            "hallucination_rate": hallucination_rate,
            "absence_handling_accuracy": absence_accuracy,
            "total_gt_matches": total_gt_matches,
        }
