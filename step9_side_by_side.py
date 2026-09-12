# ============================================================================
#  STEP 9 (replacement) — side-by-side reconstruction on a REAL window
#
#  The original Step 9 fed the model a hand-written prompt containing two
#  Twitter events. That is not the task: a real window carries tens of events
#  across several applications, plus a provenance block. Demonstrating on two
#  events shows almost nothing, and the output cannot be compared with the
#  evaluation figures.
#
#  This runs both conditions on the same real validation window, using the
#  same prompt construction and the same deterministic decoding as
#  evaluate_full.py, so the output is directly comparable with Chapter Four.
# ============================================================================

import json, re, torch

WINDOW = 0          # which validation window to demonstrate (0-22)
MAX_NEW_TOKENS = 1024
SEED = 42

with open("data/splits/val.jsonl", encoding="utf-8") as f:
    samples = [json.loads(l) for l in f if l.strip()]

sample = samples[WINDOW]
msgs   = sample["messages"]
sys_p, usr_p, target = msgs[0]["content"], msgs[1]["content"], msgs[2]["content"]

label = next((l.replace("Window:", "").strip()
              for l in usr_p.split("\n") if l.startswith("Window:")), f"#{WINDOW+1}")
n_events = len(re.findall(r"\[EVT-[0-9a-f]+\]", usr_p.split("PROVENANCE:")[0]))
whitelist = set(m.lower() for m in re.findall(r"EVT-([a-f0-9]+)", usr_p))

print("=" * 78)
print(f"  WINDOW {WINDOW+1} of {len(samples)}   {label}")
print(f"  events presented: {n_events}   identifiers available: {len(whitelist)}")
print("=" * 78)


@torch.inference_mode()
def run():
    torch.manual_seed(SEED)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": (sys_p + "\n\n" + usr_p).strip()}],
        tokenize=True, add_generation_prompt=True,
        return_tensors="pt").to(next(model.parameters()).device)
    ids = ids.input_ids if hasattr(ids, "input_ids") else ids
    out = model.generate(input_ids=ids, max_new_tokens=MAX_NEW_TOKENS,
                         do_sample=False, use_cache=True)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def score(text):
    ids = [m.lower() for m in re.findall(r"EVT-([a-f0-9]+)", text)]
    ok = [e for e in ids if e in whitelist]
    return len(ids), len(ok), (len(ok) / len(ids) * 100) if ids else None


model.eval()

with model.disable_adapter():
    base = run()
fine = run()

for name, text in (("BASELINE  (adapter disabled)", base),
                   ("FINE-TUNED  (adapter active)", fine)):
    n, v, p = score(text)
    print("\n" + "=" * 78)
    print(f"  {name}")
    print(f"  cited {n}   valid {v}   precision " +
          ("n/a (nothing cited)" if p is None else f"{p:.1f}%"))
    print("=" * 78)
    print(text.strip()[:2600])

print("\n" + "=" * 78)
print("  GROUND TRUTH")
print("=" * 78)
print(target.strip()[:2600])

with open("eval_reports/step9_side_by_side.txt", "w", encoding="utf-8") as f:
    f.write(f"WINDOW {WINDOW+1} of {len(samples)}  {label}\n")
    f.write(f"events presented: {n_events}  identifiers available: {len(whitelist)}\n\n")
    for name, text in (("BASELINE", base), ("FINE-TUNED", fine),
                       ("GROUND TRUTH", target)):
        n, v, p = score(text)
        f.write("=" * 78 + f"\n{name}\n")
        if name != "GROUND TRUTH":
            f.write(f"cited {n}  valid {v}  precision "
                    + ("n/a\n" if p is None else f"{p:.1f}%\n"))
        f.write("=" * 78 + "\n" + text.strip() + "\n\n")

print("\nWritten: eval_reports/step9_side_by_side.txt")
