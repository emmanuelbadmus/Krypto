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

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "xx")
UNSEEN_DATA_DIR = os.path.join(DATA_DIR, "unseen_training_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Cached evaluator instance
evaluator = ForensicEvaluator(
    ground_truth_csv=os.path.join(UNSEEN_DATA_DIR, "ground_truth.csv") if os.path.exists(os.path.join(UNSEEN_DATA_DIR, "ground_truth.csv")) else None,
    events_jsonl=os.path.join(UNSEEN_DATA_DIR, "events.jsonl") if os.path.exists(os.path.join(UNSEEN_DATA_DIR, "events.jsonl")) else None,
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
    
    # Check if local adapters exist
    adapters = []
    output_dirs = ["outputs_forensic_gemma_2b", "outputs_forensic_qwen_14b"]
    for od in output_dirs:
        full_od = os.path.join(BASE_DIR, od, "final_adapter")
        if os.path.exists(full_od):
            adapters.append({"id": full_od, "name": od})

    datasets = [
        {"id": "unseen", "name": "Unseen Test Windows (52 windows)", "path": "xx/unseen_training_data/unlabelled.jsonl"},
        {"id": "val_split", "name": "Date-Held-Out Validation Split (23 windows)", "path": "xx/val_split.jsonl"},
        {"id": "train_augmented", "name": "Augmented Session Windows (113 windows)", "path": "xx/train_augmented.jsonl"},
        {"id": "train_split", "name": "Training Split (90 windows)", "path": "xx/train_split.jsonl"},
    ]

    return {
        "device": "Google Pixel 3 (Android 9 Pie, PQ2A.190205.001)",
        "corpus": "Joshua Hickman Digital Corpora (2019-02-13 to 2019-04-06)",
        "indexed_events_count": len(evaluator.events_db),
        "ground_truth_count": len(evaluator.ground_truth),
        "available_models": available_models,
        "available_adapters": adapters,
        "available_datasets": datasets,
    }


@app.get("/api/windows")
def list_windows(dataset: str = Query("unseen")):
    """Returns list of windows in the selected dataset with metadata."""
    dataset_map = {
        "unseen": os.path.join(UNSEEN_DATA_DIR, "unlabelled.jsonl"),
        "val_split": os.path.join(DATA_DIR, "val_split.jsonl"),
        "train_augmented": os.path.join(DATA_DIR, "train_augmented.jsonl"),
        "train_split": os.path.join(DATA_DIR, "train_split.jsonl"),
    }
    
    file_path = dataset_map.get(dataset, dataset_map["unseen"])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Dataset file {file_path} not found.")

    windows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            msgs = item.get("messages", [])
            user_content = msgs[1]["content"] if len(msgs) > 1 else ""
            system_content = msgs[0]["content"] if len(msgs) > 0 else ""
            assistant_content = msgs[2]["content"] if len(msgs) > 2 else None

            # Extract window/date name
            window_title = f"Window #{idx+1}"
            date_str = ""
            for l in user_content.split("\n"):
                if l.startswith("Window:") or l.startswith("Date:"):
                    window_title = l.replace("Window:", "").replace("Date:", "").strip()
                    parts = window_title.split()
                    date_str = parts[0] if parts else ""
                    break

            # Count artifacts in prompt
            event_ids = re.findall(r"\[EVT-([a-f0-9]+)\]", user_content)

            windows.append({
                "index": idx,
                "title": window_title,
                "date": date_str,
                "event_count": len(event_ids),
                "has_target": assistant_content is not None,
                "system_prompt": system_content,
                "user_prompt": user_content,
                "target_answer": assistant_content,
            })

    return {"dataset": dataset, "total_windows": len(windows), "windows": windows}


@app.get("/api/provenance/{event_id}")
def get_event_provenance(event_id: str):
    """Resolves an [EVT-xxxx] event ID to its exact SQLite database, table, row, and timestamp."""
    event = evaluator.events_db.get(event_id)
    if not event:
        # Check partial match
        for k, v in evaluator.events_db.items():
            if k.startswith(event_id) or event_id in k:
                event = v
                break

    if not event:
        raise HTTPException(status_code=404, detail=f"Event ID [EVT-{event_id}] not found in master database.")

    return {
        "event_id": event.get("event_id"),
        "timestamp": event.get("timestamp"),
        "epoch_type": event.get("epoch_type"),
        "raw_timestamp": event.get("raw_timestamp"),
        "app": event.get("app"),
        "artifact_type": event.get("artifact_type"),
        "db_path": event.get("db_path"),
        "table": event.get("table"),
        "row_id": event.get("row_id"),
        "raw_data": event.get("raw_data", {}),
    }


@app.post("/api/generate")
def run_model_inference(req: InferenceRequest):
    """Runs inference using the selected model on the given prompt."""
    global active_runner, active_model_key

    model_key = f"{req.model_name}::{req.adapter_path}"
    try:
        if active_runner is None or active_model_key != model_key:
            # Resolve relative local path
            model_path = req.model_name
            if model_path.startswith("models/"):
                model_path = os.path.join(BASE_DIR, model_path)
            
            adapter_path = req.adapter_path
            if adapter_path and adapter_path.startswith("outputs"):
                adapter_path = os.path.join(BASE_DIR, adapter_path)

            print(f"[API] Initializing runner for {model_path}...")
            active_runner = get_model_runner(
                model_name_or_path=model_path,
                adapter_path=adapter_path,
                chat_template=req.chat_template,
            )
            active_model_key = model_key

        prediction = active_runner.generate(
            system_prompt=req.system_prompt,
            user_prompt=req.user_prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )

        return {"prediction": prediction, "model_key": model_key}

    except Exception as e:
        print(f"[API Error in generate] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate")
def evaluate_reconstruction(req: EvaluationRequest):
    """Deterministically evaluates a reconstructed output against prompt and ground truth."""
    metrics = evaluator.evaluate_window(
        prediction=req.prediction,
        user_prompt=req.user_prompt,
        target_answer=req.target_answer,
        window_date=req.window_date,
    )
    return {"metrics": metrics}


# Mount Static Files
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=True)

