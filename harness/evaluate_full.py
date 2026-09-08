"""
evaluate_full.py  —  Chapter Four evaluation harness

Produces every figure Chapter Three commits to reporting, for BOTH conditions,
in ONE execution, under ONE procedure.

Run this as a Colab cell AFTER training, while the PEFT-wrapped `model` and
`tokenizer` are still in memory. Nothing else needs to be loaded.

The three hazards named in Chapter Three §3.9.2 are eliminated structurally:

  1. Model identity      - the SAME weights serve both conditions. The baseline
                           is produced with model.disable_adapter(), so the only
                           difference between conditions is whether the LoRA
                           adapter is active. Not a different model, not a
                           different checkpoint, not a different script.
  2. Prompt construction - one build_prompt() function serves both conditions,
                           reproducing the training format exactly (system text
                           concatenated into the user turn, because the Gemma
                           chat template defines no system role).
  3. Scoring convention  - one scorer, applied identically. Micro-averaging is
                           used for the headline figures, so the undefined
                           per-window precision case never enters the totals.

Determinism: do_sample=False (greedy decoding) with a fixed seed. Output is
reproducible on identical input, which temperature-based sampling is not.
"""

import json
import re
import os
import torch

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
VAL_FILE       = "data/splits/val.jsonl"
OUT_JSON       = "eval_reports/chapter4_results.json"
OUT_MD         = "eval_reports/chapter4_tables.md"
MAX_NEW_TOKENS = 1024        # identical for both conditions
SEED           = 42

EVT_RE = re.compile(r"EVT-([a-f0-9]+)", re.IGNORECASE)

# Section markers written by the dataset builder. A window "warrants an absence
# report" when its ground truth contains this heading.
ABSENCE_MARKER = "without direct artifact support"


# --------------------------------------------------------------------------
# PROMPT — identical for both conditions, matching the training format
# --------------------------------------------------------------------------
def build_prompt(sample):
    msgs = sample["messages"]
    sys_txt = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
    usr_txt = msgs[1]["content"]
    merged = (sys_txt + "\n\n" + usr_txt).strip()
    return [{"role": "user", "content": merged}]


@torch.inference_mode()
def generate(model, tokenizer, sample, device):
    torch.manual_seed(SEED)
    ids = tokenizer.apply_chat_template(
        build_prompt(sample), tokenize=True,
        add_generation_prompt=True, return_tensors="pt").to(device)
    input_ids = ids.input_ids if hasattr(ids, "input_ids") else ids
    n = input_ids.shape[1]
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,          # greedy: deterministic
        use_cache=True,
    )
    return tokenizer.decode(out[0][n:], skip_special_tokens=True)


# --------------------------------------------------------------------------
# SCORING — one scorer, both conditions
# --------------------------------------------------------------------------
def score_window(prediction, sample):
    msgs = sample["messages"]
    user_prompt  = msgs[1]["content"]
    ground_truth = msgs[2]["content"]

    whitelist = set(m.lower() for m in EVT_RE.findall(user_prompt))
    gt_ids    = set(m.lower() for m in EVT_RE.findall(ground_truth))
    pred_ids  = [m.lower() for m in EVT_RE.findall(prediction)]
    pred_set  = set(pred_ids)

    valid   = [e for e in pred_ids if e in whitelist]
    invalid = [e for e in pred_ids if e not in whitelist]

    # Per-window precision is undefined when nothing is cited. Recorded as None
    # rather than assigned an arbitrary value; totals use micro-averaging.
    precision = (len(valid) / len(pred_ids)) if pred_ids else None
    recall    = (len(gt_ids & pred_set) / len(gt_ids)) if gt_ids else None

    # Absence handling: agreement with ground truth on whether an absence report
    # is warranted. A window correctly reporting no unrecoverable activity is a
    # correct outcome and is scored as such. This is an accuracy, not a
    # detection rate.
    gt_absence   = ABSENCE_MARKER in ground_truth.lower()
    pred_absence = ABSENCE_MARKER in prediction.lower()

    return {
        "emitted":        len(pred_ids),
        "valid":          len(valid),
        "invalid":        len(invalid),
        "gt_available":   len(gt_ids),
        "gt_recovered":   len(gt_ids & pred_set),
        "precision":      precision,
        "recall":         recall,
        "gt_absence":     gt_absence,
        "pred_absence":   pred_absence,
        "absence_correct": gt_absence == pred_absence,
    }


def aggregate(rows):
    """Micro-averaged totals: the undefined per-window case never enters."""
    emit = sum(r["emitted"] for r in rows)
    val  = sum(r["valid"] for r in rows)
    avail = sum(r["gt_available"] for r in rows)
    rec   = sum(r["gt_recovered"] for r in rows)

    p = (val / emit) if emit else 0.0
    r_ = (rec / avail) if avail else 0.0
    f1 = (2 * p * r_ / (p + r_)) if (p + r_) else 0.0

    return {
        "windows":              len(rows),
        "total_emitted":        emit,
        "total_valid":          val,
        "total_unsupported":    emit - val,
        "gt_ids_available":     avail,
        "gt_ids_recovered":     rec,
        "citation_precision":   p,
        "citation_recall":      r_,
        "f1_score":             f1,
        "absence_accuracy":     sum(r["absence_correct"] for r in rows) / len(rows),
        "windows_perfect_precision":
            sum(1 for r in rows if r["precision"] == 1.0),
        "windows_no_citation":
            sum(1 for r in rows if r["emitted"] == 0),
        "windows_absence_warranted":
            sum(1 for r in rows if r["gt_absence"]),
    }


# --------------------------------------------------------------------------
# RUN
# --------------------------------------------------------------------------
def run_condition(model, tokenizer, samples, device, label):
    print(f"\n{'='*70}\n  GENERATING — {label}\n{'='*70}")
    rows = []
    for i, s in enumerate(samples, 1):
        pred = generate(model, tokenizer, s, device)
        m = score_window(pred, s)
        m["window"] = i
        m["prediction"] = pred
        rows.append(m)
        pr = "  n/a" if m["precision"] is None else f"{m['precision']*100:5.1f}%"
        rc = "  n/a" if m["recall"] is None else f"{m['recall']*100:5.1f}%"
        print(f"  window {i:>2}/{len(samples)}  cited {m['emitted']:>3}  "
              f"valid {m['valid']:>3}  P {pr}  R {rc}  "
              f"absence {'ok' if m['absence_correct'] else 'MISS'}")
    return rows


def main(model, tokenizer):
    device = next(model.parameters()).device
    model.eval()

    with open(VAL_FILE, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(samples)} evaluation windows from {VAL_FILE}")

    # ---- BASELINE : same weights, adapter switched off --------------------
    with model.disable_adapter():
        base_rows = run_condition(model, tokenizer, samples, device,
                                  "BASELINE  (adapter disabled)")

    # ---- FINE-TUNED : adapter active --------------------------------------
    ft_rows = run_condition(model, tokenizer, samples, device,
                            "FINE-TUNED  (adapter active)")

    base, ft = aggregate(base_rows), aggregate(ft_rows)

    # ---- summary ----------------------------------------------------------
    print(f"\n{'='*70}\n  CHAPTER FOUR SUMMARY\n{'='*70}")
    hdr = f"  {'Metric':<34}{'Baseline':>15}{'Fine-tuned':>15}"
    print(hdr + "\n  " + "-" * (len(hdr) - 2))

    def row(name, key, pct=True):
        b, f = base[key], ft[key]
        if pct:
            print(f"  {name:<34}{b*100:>14.2f}%{f*100:>14.2f}%")
        else:
            print(f"  {name:<34}{b:>15}{f:>15}")

    row("Citation precision", "citation_precision")
    row("Citation recall", "citation_recall")
    row("F1 score", "f1_score")
    row("Absence handling accuracy", "absence_accuracy")
    row("Identifiers emitted", "total_emitted", pct=False)
    row("Identifiers valid", "total_valid", pct=False)
    row("Identifiers unsupported", "total_unsupported", pct=False)
    row("Ground truth IDs recovered", "gt_ids_recovered", pct=False)
    row("Windows at 100% precision", "windows_perfect_precision", pct=False)
    row("Windows citing nothing", "windows_no_citation", pct=False)
    print(f"\n  Ground truth identifiers available : {base['gt_ids_available']}")
    print(f"  Windows warranting an absence report: {base['windows_absence_warranted']}"
          f" of {base['windows']}")

    # ---- persist ----------------------------------------------------------
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    payload = {
        "config": {
            "val_file": VAL_FILE, "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False, "seed": SEED,
            "baseline_method": "model.disable_adapter() — identical weights",
            "precision_convention":
                "micro-averaged; per-window precision undefined when no "
                "identifier is emitted and recorded as null",
            "absence_definition":
                "agreement with ground truth on whether an absence report is "
                "warranted, across all windows",
        },
        "summary": {"baseline": base, "fine_tuned": ft},
        "per_window": {"baseline": base_rows, "fine_tuned": ft_rows},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # ---- per-window table for Chapter Four --------------------------------
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("| Window | Base cited | Base valid | Base P | Base R | "
                "FT cited | FT valid | FT P | FT R |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for b, t in zip(base_rows, ft_rows):
            fmt = lambda v: "n/a" if v is None else f"{v*100:.1f}%"
            f.write(f"| {b['window']} | {b['emitted']} | {b['valid']} | "
                    f"{fmt(b['precision'])} | {fmt(b['recall'])} | "
                    f"{t['emitted']} | {t['valid']} | "
                    f"{fmt(t['precision'])} | {fmt(t['recall'])} |\n")

    print(f"\n  Written: {OUT_JSON}")
    print(f"  Written: {OUT_MD}")
    print("=" * 70)
    return payload


# In Colab, after training, simply call:
#     results = main(model, tokenizer)
if __name__ == "__main__":
    raise SystemExit(
        "Import this in the notebook and call main(model, tokenizer) while the "
        "trained PEFT model is in memory."
    )
