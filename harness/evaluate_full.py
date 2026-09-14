"""
evaluate_full.py  —  Chapter Four evaluation harness (corrected scoring)

Produces every figure Chapter Three commits to reporting, for BOTH conditions,
in ONE execution, under ONE procedure.

Run as a Colab cell AFTER training, while the PEFT-wrapped `model` and
`tokenizer` are still in memory:  results = main(model, tokenizer)

--------------------------------------------------------------------------
THE THREE COMPARABILITY HAZARDS (Chapter Three 3.9.2) — unchanged
--------------------------------------------------------------------------
  1. Model identity      the SAME weights serve both conditions; the baseline
                         is model.disable_adapter()
  2. Prompt construction one build_prompt() serves both conditions
  3. Scoring convention  one scorer, applied identically, micro-averaged

--------------------------------------------------------------------------
WHAT CHANGED IN THE SCORING, AND WHY
--------------------------------------------------------------------------
The previous scorer reported 99.65% precision and 100% recall for the
zero-shot baseline. Those figures were artefacts of how the metrics were
defined, not evidence of good reconstruction.

(a) "Citation precision" counted an identifier correct if it appeared
    ANYWHERE in the prompt. A model that transcribes every identifier placed
    in front of it therefore scores 100%, and recovers every ground-truth
    identifier as a side effect, scoring 100% recall too. Both ceilings are
    reachable without reconstructing anything.

    Corrected: precision is now measured against the ground truth — of the
    identifiers a model chose to cite, how many support a documented event.
    Citing everything now costs precision, as it should.

    Whitelist membership is still computed, but reported separately as
    GROUNDING VALIDITY: the share of emitted identifiers that exist at all.
    That is the verification component's job and answers RQ3; it is not a
    measure of reconstruction quality and is no longer presented as one.

(b) "Absence handling accuracy" was a substring match on a section heading,
    so it measured whether a model reproduced a phrase. It scored the
    fine-tuned model at 69.6% — below the 78.3% a model achieves by emitting
    the heading unconditionally.

    Corrected: absence reporting is scored per ENTRY, on (time, application)
    pairs parsed from the absence section, with precision, recall and F1. A
    model must name the right undocumented activities, not produce the header.

(c) Trivial baselines are computed from the same data and printed alongside
    every headline figure. A metric a constant predictor can match is
    identified as such in the output rather than left for a reader to notice.

Determinism: do_sample=False (greedy) with a fixed seed, per section 3.9.4.
"""

import json
import os
import re
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

# Section headings written by the dataset builder.
SUPPORTED_HEAD = "reconstructed activity with artifact support"
ABSENCE_HEAD   = "without direct artifact support"

# "- 07:40 - GooglePodcasts: Downloaded Up First  [no direct artifact; ...]"
ENTRY_RE = re.compile(r"^\s*[-*]\s*(\d{1,2}:\d{2})\s*[-–]\s*([^:]+):", re.M)


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
# PARSING
# --------------------------------------------------------------------------
def split_sections(text):
    """Return (supported_text, absence_text). Either may be empty.

    Free-form output that uses neither heading is treated as entirely
    supported-section prose, which is the correct reading for a zero-shot
    baseline that has never seen the target format.
    """
    low = text.lower()
    a = low.find(ABSENCE_HEAD)
    if a == -1:
        return text, ""
    line_start = text.rfind("\n", 0, a)
    line_start = 0 if line_start == -1 else line_start
    return text[:line_start], text[line_start:]


def entries(section_text):
    """Set of (HH:MM, application) pairs named in a section."""
    out = set()
    for t, app in ENTRY_RE.findall(section_text):
        hh, mm = t.split(":")
        out.add((f"{int(hh):02d}:{mm}", app.strip().lower()))
    return out


def prf(hit, pred_n, true_n):
    p = hit / pred_n if pred_n else None
    r = hit / true_n if true_n else None
    if p and r:
        f = 2 * p * r / (p + r)
    else:
        f = 0.0 if (p is not None and r is not None) else None
    return p, r, f


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

    # -- grounding validity: does the identifier exist at all? (RQ3) ---------
    valid   = [e for e in pred_ids if e in whitelist]
    invalid = [e for e in pred_ids if e not in whitelist]

    # -- citation precision/recall: measured against the GROUND TRUTH -------
    # Citing every identifier on offer no longer scores well: the identifiers
    # that support no documented event now count against precision.
    correct = pred_set & gt_ids
    c_p, c_r, c_f = prf(len(correct), len(pred_set), len(gt_ids))

    # -- absence reporting: per entry, not per heading -----------------------
    _, gt_abs_text   = split_sections(ground_truth)
    _, pred_abs_text = split_sections(prediction)
    gt_abs   = entries(gt_abs_text)
    pred_abs = entries(pred_abs_text)
    a_hit = len(gt_abs & pred_abs)
    a_p, a_r, a_f = prf(a_hit, len(pred_abs), len(gt_abs))

    return {
        # volume
        "emitted":          len(pred_ids),
        "emitted_unique":   len(pred_set),
        "shown":            len(whitelist),
        # grounding (verification component, RQ3)
        "valid":            len(valid),
        "invalid":          len(invalid),
        # citation quality against ground truth
        "gt_available":     len(gt_ids),
        "gt_recovered":     len(correct),
        "precision":        c_p,
        "recall":           c_r,
        "f1":               c_f,
        # absence reporting
        "gt_absence_n":     len(gt_abs),
        "pred_absence_n":   len(pred_abs),
        "absence_hit":      a_hit,
        "absence_precision": a_p,
        "absence_recall":   a_r,
        "absence_f1":       a_f,
    }


def aggregate(rows):
    """Micro-averaged totals: undefined per-window cases never enter."""
    S = lambda k: sum(r[k] for r in rows)

    emit_u, shown = S("emitted_unique"), S("shown")
    avail, rec    = S("gt_available"), S("gt_recovered")
    p, r, f = prf(rec, emit_u, avail)

    a_hit, a_pred, a_true = S("absence_hit"), S("pred_absence_n"), S("gt_absence_n")
    ap, ar, af = prf(a_hit, a_pred, a_true)

    return {
        "windows":               len(rows),
        "total_emitted":         S("emitted"),
        "total_emitted_unique":  emit_u,
        "total_shown":           shown,
        "total_valid":           S("valid"),
        "total_invalid":         S("invalid"),
        "grounding_validity":    (S("valid") / S("emitted")) if S("emitted") else None,
        "gt_ids_available":      avail,
        "gt_ids_recovered":      rec,
        "citation_precision":    p,
        "citation_recall":       r,
        "f1_score":              f,
        "absence_entries_gt":    a_true,
        "absence_entries_pred":  a_pred,
        "absence_entries_hit":   a_hit,
        "absence_precision":     ap,
        "absence_recall":        ar,
        "absence_f1":            af,
        "windows_no_citation":   sum(1 for r in rows if r["emitted"] == 0),
        "citation_rate":         (emit_u / shown) if shown else None,
    }


def trivial_baselines(samples):
    """What a model achieves WITHOUT reconstructing anything.

    Any reported figure at or below these is uninformative, and the output
    says so rather than leaving it to be noticed later.
    """
    shown = correct = avail = 0
    abs_true = abs_all = 0
    for s in samples:
        prompt, gt = s["messages"][1]["content"], s["messages"][2]["content"]
        wl = set(m.lower() for m in EVT_RE.findall(prompt))
        ids = set(m.lower() for m in EVT_RE.findall(gt))
        shown += len(wl); avail += len(ids); correct += len(wl & ids)
        _, gt_abs_text = split_sections(gt)
        gt_abs = entries(gt_abs_text)
        abs_true += len(gt_abs)
        abs_all += len(entries(gt))          # "declare everything absent"
    cp, cr, cf = prf(correct, shown, avail)
    ap, ar, af = prf(abs_true, abs_all, abs_true)
    return {
        "cite_everything": {"precision": cp, "recall": cr, "f1": cf},
        "declare_all_absent": {"precision": ap, "recall": ar, "f1": af},
    }


# --------------------------------------------------------------------------
# RUN
# --------------------------------------------------------------------------
def run_condition(model, tokenizer, samples, device, label):
    print(f"\n{'='*78}\n  GENERATING — {label}\n{'='*78}")
    rows = []
    for i, s in enumerate(samples, 1):
        pred = generate(model, tokenizer, s, device)
        m = score_window(pred, s)
        m["window"] = i
        m["prediction"] = pred
        rows.append(m)
        pc = lambda v: "  n/a" if v is None else f"{v*100:5.1f}%"
        print(f"  window {i:>2}/{len(samples)}  cited {m['emitted_unique']:>3}"
              f"/{m['shown']:>3}  P {pc(m['precision'])}  R {pc(m['recall'])}"
              f"  abs {m['absence_hit']:>2}/{m['gt_absence_n']:<2}"
              f"  ungrounded {m['invalid']}")
    return rows


def main(model, tokenizer):
    device = next(model.parameters()).device
    model.eval()

    with open(VAL_FILE, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(samples)} evaluation windows from {VAL_FILE}")

    with model.disable_adapter():
        base_rows = run_condition(model, tokenizer, samples, device,
                                  "BASELINE  (adapter disabled)")
    ft_rows = run_condition(model, tokenizer, samples, device,
                            "FINE-TUNED  (adapter active)")

    base, ft = aggregate(base_rows), aggregate(ft_rows)
    triv = trivial_baselines(samples)

    # ---- summary ----------------------------------------------------------
    print(f"\n{'='*78}\n  CHAPTER FOUR SUMMARY\n{'='*78}")
    hdr = f"  {'Metric':<36}{'Baseline':>13}{'Fine-tuned':>13}{'Trivial':>13}"
    print(hdr + "\n  " + "-" * (len(hdr) - 2))

    def row(name, key, triv_val=None, pct=True):
        b, f = base[key], ft[key]
        fm = (lambda v: "n/a" if v is None
              else (f"{v*100:.2f}%" if pct else str(v)))
        print(f"  {name:<36}{fm(b):>13}{fm(f):>13}{fm(triv_val):>13}")

    print("  CITATION QUALITY (against ground truth)")
    row("  precision", "citation_precision", triv["cite_everything"]["precision"])
    row("  recall", "citation_recall", triv["cite_everything"]["recall"])
    row("  F1", "f1_score", triv["cite_everything"]["f1"])
    print("  ABSENCE REPORTING (per entry)")
    row("  precision", "absence_precision", triv["declare_all_absent"]["precision"])
    row("  recall", "absence_recall", triv["declare_all_absent"]["recall"])
    row("  F1", "absence_f1", triv["declare_all_absent"]["f1"])
    print("  GROUNDING (verification component, RQ3)")
    row("  validity of emitted identifiers", "grounding_validity")
    row("  ungrounded identifiers", "total_invalid", pct=False)
    print("  VOLUME")
    row("  identifiers cited (unique)", "total_emitted_unique", pct=False)
    row("  identifiers shown", "total_shown", pct=False)
    row("  citation rate (cited/shown)", "citation_rate")
    row("  windows citing nothing", "windows_no_citation", pct=False)

    print(f"\n  Ground truth identifiers available : {base['gt_ids_available']}")
    print(f"  Absence entries in ground truth    : {base['absence_entries_gt']}")
    print("\n  The 'Trivial' column is what a model scores without reconstructing:")
    print("  citing every identifier shown, and declaring every entry absent.")
    print("  A result at or below that column carries no evidence of capability.")

    # ---- persist ----------------------------------------------------------
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    payload = {
        "config": {
            "val_file": VAL_FILE, "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False, "seed": SEED,
            "baseline_method": "model.disable_adapter() — identical weights",
            "citation_convention":
                "precision and recall measured against ground-truth "
                "identifiers, micro-averaged; whitelist membership reported "
                "separately as grounding validity",
            "absence_convention":
                "per-entry (time, application) matching within the absence "
                "section; precision, recall and F1, not a heading match",
        },
        "trivial_baselines": triv,
        "summary": {"baseline": base, "fine_tuned": ft},
        "per_window": {"baseline": base_rows, "fine_tuned": ft_rows},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # ---- per-window table -------------------------------------------------
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("| Window | Shown | Base cited | Base P | Base R | Base abs "
                "| FT cited | FT P | FT R | FT abs |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for b, t in zip(base_rows, ft_rows):
            g = lambda v: "n/a" if v is None else f"{v*100:.1f}%"
            f.write(f"| {b['window']} | {b['shown']} | {b['emitted_unique']} | "
                    f"{g(b['precision'])} | {g(b['recall'])} | "
                    f"{b['absence_hit']}/{b['gt_absence_n']} | "
                    f"{t['emitted_unique']} | {g(t['precision'])} | "
                    f"{g(t['recall'])} | {t['absence_hit']}/{t['gt_absence_n']} |\n")

    print(f"\n  Written: {OUT_JSON}\n  Written: {OUT_MD}\n" + "=" * 78)
    return payload


if __name__ == "__main__":
    raise SystemExit(
        "Import this in the notebook and call main(model, tokenizer) while the "
        "trained PEFT model is in memory."
    )
