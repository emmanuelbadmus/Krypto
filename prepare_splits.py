#!/usr/bin/env python3
"""
prepare_splits.py
Splits train_augmented.jsonl into train_split.jsonl and val_split.jsonl
using date-held-out partitioning to prevent temporal data leakage.
"""

import json
import os
from collections import defaultdict

INPUT_FILE = "xx/train_augmented.jsonl"
TRAIN_FILE = "xx/train_split.jsonl"
VAL_FILE = "xx/val_split.jsonl"

# Held-out dates selected for balanced representation of Direct, Absence, and AndroidAuto events
VAL_DATES = {"2019-02-22", "2019-03-15", "2019-03-28", "2019-04-03"}

def get_date(item):
    user_content = item["messages"][1]["content"]
    for line in user_content.split("\n"):
        if line.startswith("Window:"):
            return line.replace("Window:", "").strip().split()[0]
    return "unknown"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    train_data = []
    val_data = []

    for item in data:
        date = get_date(item)
        if date in VAL_DATES:
            val_data.append(item)
        else:
            train_data.append(item)

    print(f"Total samples: {len(data)}")
    print(f"Train samples ({len(set(get_date(x) for x in train_data))} dates): {len(train_data)}")
    print(f"Val samples   ({len(set(get_date(x) for x in val_data))} dates): {len(val_data)}")

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved: {TRAIN_FILE} and {VAL_FILE}")

if __name__ == "__main__":
    main()
