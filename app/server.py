"""
app/server.py
FastAPI backend for the Forensic Activity Reconstruction Workbench.
Serves interactive UI, provides endpoints for data browsing, live model inference,
deterministic metric evaluation, and audit report generation.
"""

import os
import re
import json
import csv
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from harness.evaluators import ForensicEvaluator
from harness.models import get_model_runner

app = FastAPI(title="Forensic Activity Reconstruction Workbench", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clean Primary Data Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive", "legacy_raw_extractions")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

EVENTS_DB_PATH = os.path.join(DATA_DIR, "events.jsonl")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.csv")

# Available Datasets
DATASETS = {
    "val_split": {
        "name": "Date-Held-Out Validation Split (23 windows)",
        "path": os.path.join(DATA_DIR, "val_split.jsonl"),
    },
    "train_augmented": {
        "name": "Primary Augmented Session Windows (113 windows)",
        "path": os.path.join(DATA_DIR, "train_augmented.jsonl"),
    },
    "train_split": {
        "name": "Training Split (90 windows)",
        "path": os.path.join(DATA_DIR, "train_split.jsonl"),
    },
    "unlabelled_archive": {
        "name": "Archived Unlabelled Test Windows (52 windows)",
        "path": os.path.join(ARCHIVE_DIR, "unlabelled.jsonl"),
    }
}

# Cached evaluator instance
evaluator = ForensicEvaluator(
    ground_truth_csv=GROUND_TRUTH_PATH if os.path.exists(GROUND_TRUTH_PATH) else None,
    events_jsonl=EVENTS_DB_PATH if os.path.exists(EVENTS_DB_PATH) else None,
)

# In-memory cached active runner to prevent reloading model on every request
active_runner = None
active_model_key = None

class InferenceRequest(BaseModel):
    model_name: str = "models/gemma-4-E2B-it"
    adapter_path: Optional[str] = None
    system_prompt: str
    user_prompt: str
    temperature: float = 0.1
    max_new_tokens: int = 1024
    chat_template: str = "gemma"

class EvaluationRequest(BaseModel):
    prediction: str
    user_prompt: str
    target_answer: Optional[str] = None
    window_date: Optional[str] = None


@app.get("/api/status")
def get_system_status():
    """Returns dataset stats, available models, and system status."""
    available_models = [
        {"id": "models/gemma-4-E2B-it", "name": "Gemma 4 E2B-IT (Local 131k context)", "type": "local"},
        {"id": "unsloth/gemma-2-2b-it", "name": "Gemma 2 2B-IT (Unsloth / Hugging Face)", "type": "hub"},
        {"id": "Qwen/Qwen2.5-14B-Instruct", "name": "Qwen 2.5 14B-Instruct", "type": "hub"},
        {"id": "meta-llama/Llama-3.1-8B-Instruct", "name": "Llama 3.1 8B-Instruct", "type": "hub"},
    ]
    
    adapters = []
    output_dirs = ["outputs_forensic_gemma_2b", "outputs_forensic_qwen_14b"]
    for od in output_dirs:
        full_od = os.path.join(BASE_DIR, od, "final_adapter")
        if os.path.exists(full_od):
            adapters.append({"id": full_od, "name": od})

    datasets_list = [{"id": k, "name": v["name"], "path": v["path"]} for k, v in DATASETS.items()]

    return {
        "device": "Google Pixel 3 (Android 9 Pie, PQ2A.190205.001)",
        "corpus": "Joshua Hickman Digital Corpora (2019-02-13 to 2019-04-06)",
        "indexed_events_count": len(evaluator.events_db),
        "ground_truth_count": len(evaluator.ground_truth_events),
        "available_models": available_models,
        "available_adapters": adapters,
        "available_datasets": datasets_list,
    }


@app.get("/api/windows")
def get_windows(dataset: str = Query("val_split", description="Dataset identifier: val_split, train_augmented, train_split, unlabelled_archive")):
    """Returns parsed timeline windows for UI exploration."""
    if dataset not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found. Available: {list(DATASETS.keys())}")

    filepath = DATASETS[dataset]["path"]
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Dataset file '{filepath}' does not exist on disk.")

    windows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            messages = data.get("messages", [])

            sys_prompt = ""
            user_prompt = ""
            target_answer = None

            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    sys_prompt = content
                elif role == "user":
                    user_prompt = content
                elif role == "assistant":
                    target_answer = content

            date_str = None
            for pline in user_prompt.split("\n"):
                if pline.startswith("Date:") or pline.startswith("Window:"):
                    parts = pline.split(":", 1)[1].strip().split()
                    if parts:
                        date_str = parts[0]
                    break

            evt_matches = re.findall(r"\[EVT-([a-f0-9]+)\]", user_prompt)

            windows.append({
                "index": idx,
                "date": date_str,
                "title": date_str if date_str else f"Window #{idx + 1}",
                "event_count": len(evt_matches),
                "has_target": target_answer is not None,
                "system_prompt": sys_prompt,
                "user_prompt": user_prompt,
                "target_answer": target_answer,
            })

    return {"dataset": dataset, "total_windows": len(windows), "windows": windows}


@app.get("/api/provenance/{event_id}")
def get_provenance(event_id: str):
    """Deep inspects an event ID to return SQLite provenance row details."""
    clean_id = event_id.replace("EVT-", "").lower()
    event_data = evaluator.events_db.get(clean_id)
    if not event_data:
        raise HTTPException(status_code=404, detail=f"Event ID '[EVT-{clean_id}]' not found in indexed database.")
    
    return {
        "event_id": f"EVT-{clean_id}",
        "timestamp": event_data.get("timestamp"),
        "epoch_type": event_data.get("epoch_type"),
        "raw_timestamp": event_data.get("raw_timestamp"),
        "app": event_data.get("app"),
        "artifact_type": event_data.get("artifact_type"),
        "db_path": event_data.get("db_path"),
        "table": event_data.get("table"),
        "row_id": event_data.get("row_id"),
        "raw_data": event_data.get("raw_data", {}),
    }


@app.post("/api/generate")
def run_model_inference(req: InferenceRequest):
    """Executes live LLM model generation for a forensic prompt."""
    global active_runner, active_model_key

    model_key = f"{req.model_name}::{req.adapter_path}"
    if active_runner is None or active_model_key != model_key:
        try:
            print(f"[API] Initializing runner for {req.model_name}...")
            model_target = req.model_name
            if os.path.exists(os.path.join(BASE_DIR, req.model_name)):
                model_target = os.path.join(BASE_DIR, req.model_name)

            active_runner = get_model_runner(
                model_name_or_path=model_target,
                adapter_path=req.adapter_path,
                chat_template=req.chat_template,
            )
            active_model_key = model_key
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize model '{req.model_name}': {str(e)}")

    try:
        prediction = active_runner.generate(
            system_prompt=req.system_prompt,
            user_prompt=req.user_prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )
        return {"model": req.model_name, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/api/evaluate")
def evaluate_prediction(req: EvaluationRequest):
    """Computes deterministic forensic metrics on a generated output."""
    metrics = evaluator.evaluate_window(
        prediction_text=req.prediction,
        user_prompt=req.user_prompt,
        target_answer=req.target_answer,
        window_date=req.window_date,
    )
    return {"metrics": metrics}


# Mount Static UI Files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
