"""
rebuild_dataset.py  —  corrected window construction

WHY THIS EXISTS
---------------
The original builder capped the PROVENANCE block at 150 entries but left the
EVENT list uncapped:

    ev = "\\n".join(fmt_event(e) for e in evs)        # uncapped
    pv = "\\n".join(fmt_prov(e) for e in evs[:150])   # capped at 150

Two consequences followed.

1. Prompts ran to a median of ~7,200 tokens against a 2,048 limit. Because
   truncation cuts from the end and the target sits last, 87 of 90 training
   windows had every label set to -100 and contributed no gradient at all.

2. Events beyond the 150th appeared in the event list with no provenance entry,
   so the model was shown identifiers it could not resolve.

This script rebuilds the windows so that a prompt always fits, and so that every
listed event carries a provenance line.

SELECTION RULE (must be declared in Chapter Three)
--------------------------------------------------
Events are admitted to a window under a token budget, in two tiers:

  Tier 1  communication, browsing and concealment artefacts - admitted first
  Tier 2  system instrumentation (notifications, app usage, launcher, power,
          media store) - admitted only while budget remains, sampled evenly
          across the day so coverage is not front-loaded

The rationale is forensic rather than technical: of 30,609 extracted events,
roughly 97% are system instrumentation. Admitted without limit they crowd out
the communication evidence the tool exists to reconstruct.

The target is rebuilt from the SAME admitted event set, so a target never cites
an identifier absent from its prompt.
"""

import csv
import json
import os
import sys
from collections import defaultdict

EVENTS_FILE = "data/raw_database/events_db.jsonl"
GT_FILE     = "data/raw_database/ground_truth.csv"
OUT_DIR     = "data/splits"

# held-out dates, unchanged from prepare_splits.py
VAL_DATES = {"2019-02-22", "2019-03-15", "2019-03-28", "2019-04-03"}

MAX_SEQ_LENGTH = 4096          # must match max_len in Step 5 tokenisation
TEMPLATE_OVERHEAD = 96          # chat template + generation prompt margin

# Tier 1: forensically salient. Tier 2: everything else (instrumentation).
TIER1 = {
    "SMS", "MMS", "Phone", "Chrome", "WhatsApp", "Telegram", "Instagram",
    "Line", "Viber", "Kik", "TikTok", "TextNow", "Twitter", "Snapchat",
    "FacebookMessenger", "Skype", "Messages", "Duo", "Contacts",
    "GalleryVault", "AppLock", "WickrMe", "Maps",
}

# Ground-truth application name -> extraction source name(s). Anything unlisted
# matches itself. build_target() already lowercases both sides, so case was never
# the issue -- differing NAMES are (Google/Chrome, Phone/Contacts, Camera/MediaStore).
SOURCE_ALIASES = {
    "google":   {"chrome"},
    "phone":    {"phone", "contacts"},
    "camera":   {"mediastore"},
    "messages": {"messages", "sms", "mms"},
    "sms":      {"sms", "messages", "mms"},
}

GT_TOLERANCE_MIN = 5            # Chapter Three section 3.5.3


def sources_for(app):
    a = (app or "").lower()
    return SOURCE_ALIASES.get(a, {a})


SYSTEM_PROMPT = (
    "You are a digital forensic analyst. You are given timestamped artifacts "
    "extracted from an Android device. Reconstruct what the user did, in "
    "chronological order.\n\n"
    "Rules:\n"
    "1. Every claim must cite the event_id(s) that support it, in square brackets.\n"
    "2. Never state an activity that no event_id supports.\n"
    "3. If an app is present but its data is absent or encrypted, say so "
    "explicitly rather than omitting it.\n"
    "4. Residual artifacts from a removed app indicate prior installation and use.\n"
    "5. An artifact timestamp may reflect when content was created on a server, "
    "not when the user acted on this device. Flag such cases.\n"
    "6. Group related events into episodes rather than listing rows."
)


# ---------------------------------------------------------------- formatting
def fmt_event(e):
    p = [f"[{e['event_id']}]", e["ts_local"][11:16], e["source"],
         e.get("event_type") or ""]
    if e.get("direction"):
        p.append(f"({e['direction']})")
    if e.get("party"):
        p.append(f"<{str(e['party'])[:24]}>")
    c = (e.get("content") or "").replace("\n", " ")
    if c:
        p.append(f'"{c[:80]}"')
    return " ".join(x for x in p if x)


def fmt_prov(e):
    return (f"{e['event_id']} -> {e['artifact']} :: {e['table']} :: "
            f"row {e['row_id']} :: {e['time_column']}={e['raw_timestamp']} "
            f"({e['epoch_format']})")


def to_min(hhmm):
    try:
        return int(hhmm[:2]) * 60 + int(hhmm[3:5])
    except (ValueError, IndexError, TypeError):
        return None


def build_target(gts, evs):
    """Unchanged from the original builder, but evaluated against the ADMITTED
    events only, so a citation can never reference an identifier the model was
    not shown."""
    by_app = defaultdict(list)
    for e in evs:
        by_app[e["source"].lower()].append(e)

    ok, gap = [], []
    for g in sorted(gts, key=lambda x: x["time"]):
        note = g.get("notes") or ""
        desc = f"{g['time']} - {g['app']}: {g['content']}"

        if "ENCRYPTED" in note:
            gap.append(f"- {desc}  [the application store is encrypted and not "
                       f"recoverable from a logical extraction]"); continue
        if "NO_SQLITE" in note:
            gap.append(f"- {desc}  [the application stores its state outside "
                       f"SQLite; no database record exists]"); continue
        if "PACKAGE_ABSENT" in note:
            gap.append(f"- {desc}  [no application data present in this extraction]"); continue
        if "RESIDUE_ONLY" in note:
            gap.append(f"- {desc}  [only residual artifacts remain, indicating "
                       f"prior installation and removal]"); continue
        if "INDIRECT_ONLY" in note:
            gap.append(f"- {desc}  [no direct artifact; inferable from Bluetooth "
                       f"pairing, power state and app-usage correlation]"); continue

        gm = to_min(g["time"])
        cite = ""
        cands = [e for src in sources_for(g["app"]) for e in by_app.get(src, [])]
        if cands and gm is not None:
            sc = []
            for e in cands:
                em = to_min(e["ts_local"][11:16])
                if em is not None:
                    sc.append((abs(em - gm), e))
            if sc:
                sc.sort(key=lambda x: x[0])
                if sc[0][0] <= GT_TOLERANCE_MIN:
                    cite = f" [{sc[0][1]['event_id']}]"
        (ok if cite else gap).append(
            f"- {desc}{cite}" if cite else
            f"- {desc}  [documented activity; no matching artifact located]")

    out = []
    if ok:
        out.append("Reconstructed activity with artifact support:")
        out.extend(ok)
    if gap:
        if ok:
            out.append("")
        out.append("Documented activity without direct artifact support:")
        out.extend(gap)
    return "\n".join(out)


def make_prompt(label, evs):
    ev = "\n".join(fmt_event(e) for e in evs) or "(no events extracted)"
    pv = "\n".join(fmt_prov(e) for e in evs)          # SAME set - no mismatch
    return (f"Device: Google Pixel 3, Android 9. Timezone: America/New_York.\n"
            f"Window: {label}\n\n"
            f"EXTRACTED ARTIFACTS ({len(evs)} events):\n{ev}\n\n"
            f"PROVENANCE:\n{pv}\n\n"
            f"Reconstruct the user's activity for this window.")


# ------------------------------------------------------------ event admission
def spread(items, k):
    """Take k items spread evenly across the sequence rather than the first k,
    so instrumentation coverage is not confined to the start of the day."""
    if k >= len(items):
        return list(items)
    if k <= 0:
        return []
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def gt_relevant(evs, gts):
    """Tier 0: events a ground-truth entry in this window could actually cite.

    Rows already carrying a recoverability caveat are skipped -- they are
    absence lines by definition and need no artefact.
    """
    wanted = []
    for g in gts:
        if (g.get("notes") or "").strip():
            continue
        gm = to_min(g.get("time", ""))
        if gm is not None:
            wanted.append((sources_for(g.get("app")), gm))
    if not wanted:
        return []

    keep = []
    for e in evs:
        em = to_min(e["ts_local"][11:16])
        if em is None:
            continue
        src = (e.get("source") or "").lower()
        if any(src in srcs and abs(em - gm) <= GT_TOLERANCE_MIN
               for srcs, gm in wanted):
            keep.append(e)
    return keep


def largest_fitting(base, pool, fits):
    """Biggest evenly-spread subset of pool that still fits alongside base."""
    lo, hi, best = 0, len(pool), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(base + spread(pool, mid)):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return spread(pool, best), best


def admit(evs, gts, label, tokenizer):
    """Budgeted admission: Tier 0 (citable) -> Tier 1 (salient) -> Tier 2 (rest).

    CORRECTED. The previous version trimmed Tier 1 chronologically from the end:

        sel = list(t1)
        while sel and total_tokens(sel) > MAX_SEQ_LENGTH:
            sel = sel[:-1]

    Tier 1 is time-sorted, so a budget overflow amputated the later part of the
    day. On 2019-03-30 that admitted 09:03-14:38 and dropped all six Kik events
    (15:00-15:11) and all six WhatsApp events (15:16-15:19); the window labelled
    "kik episode" stopped four minutes before kik appeared. Since build_target()
    evaluates the ground truth against the ADMITTED set, every dropped event
    became a "no matching artifact located" line.

    Returns (selected, n_tier0, n_tier1, n_tier2).
    """
    def fits(sel):
        sel = sorted(sel, key=lambda e: e["ts_local"])
        prompt = (SYSTEM_PROMPT + "\n\n" + make_prompt(label, sel)).strip()
        target = build_target(gts, sel)
        n = (len(tokenizer(prompt)["input_ids"])
             + len(tokenizer(target)["input_ids"]) + TEMPLATE_OVERHEAD)
        return n <= MAX_SEQ_LENGTH

    t0 = sorted(gt_relevant(evs, gts), key=lambda e: e["ts_local"])
    t0_ids = {e["event_id"] for e in t0}
    rest = [e for e in evs if e["event_id"] not in t0_ids]
    t1 = sorted([e for e in rest if e.get("source") in TIER1],
                key=lambda e: e["ts_local"])
    t2 = sorted([e for e in rest if e.get("source") not in TIER1],
                key=lambda e: e["ts_local"])

    # Tier 0 is protected. If it alone overflows, thin it evenly rather than
    # truncating -- and say so: a window that cannot show its own citable
    # evidence is a finding, not a routine trim.
    sel = list(t0)
    if sel and not fits(sel):
        sel, _ = largest_fitting([], t0, fits)
        print(f"  WARNING: {label}: tier-0 overflow, "
              f"{len(sel)} of {len(t0)} citable events admitted")
    n0 = len(sel)

    add1, n1 = largest_fitting(sel, t1, fits)
    sel = sel + add1
    add2, n2 = largest_fitting(sel, t2, fits)
    sel = sel + add2

    return sorted(sel, key=lambda e: e["ts_local"]), n0, n1, n2


# --------------------------------------------------------------------- driver
def main(tokenizer):
    for p in (EVENTS_FILE, GT_FILE):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    events = [json.loads(l) for l in open(EVENTS_FILE, encoding="utf-8") if l.strip()]
    with open(GT_FILE, encoding="utf-8") as f:
        gt = list(csv.DictReader(f))
    print(f"loaded {len(events):,} events and {len(gt):,} ground truth rows")

    ev_by_day, gt_by_day = defaultdict(list), defaultdict(list)
    for e in events:
        if e.get("ts_local"):
            ev_by_day[e["ts_local"][:10]].append(e)
    for g in gt:
        gt_by_day[g["date"]].append(g)

    days = sorted(gt_by_day)
    print(f"{len(days)} labelled days  |  held out: {sorted(VAL_DATES)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    tr = open(f"{OUT_DIR}/train.jsonl", "w", encoding="utf-8")
    va = open(f"{OUT_DIR}/val.jsonl", "w", encoding="utf-8")
    n_tr = n_va = 0
    stats = []
    scale_counts = defaultdict(int)

    def hour_of(x, is_gt):
        try:
            return int(x["time"][:2]) if is_gt else int(x["ts_local"][11:13])
        except (ValueError, IndexError, KeyError, TypeError):
            return -1

    def emit(label, evs, gts, day, scale):
        nonlocal n_tr, n_va
        if not gts:
            return
        sel, k0, k1, k2 = admit(evs, gts, label, tokenizer)
        prompt = make_prompt(label, sel)
        target = build_target(gts, sel)
        rec = {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]}
        tok = len(tokenizer((SYSTEM_PROMPT + "\n\n" + prompt).strip())["input_ids"])
        cited = target.count("[EVT-")
        absence_only = ("Reconstructed activity with artifact support:" not in target)
        stats.append((label, len(evs), len(sel), k0, k1, k2, tok, cited, absence_only))
        scale_counts[scale] += 1
        (va if day in VAL_DATES else tr).write(json.dumps(rec, ensure_ascii=False) + "\n")
        if day in VAL_DATES:
            n_va += 1
        else:
            n_tr += 1

    SESSIONS = [("morning", 5, 12), ("afternoon", 12, 17), ("evening", 17, 24)]

    for d in days:
        evs = sorted(ev_by_day.get(d, []), key=lambda e: e["ts_local"])
        gts = gt_by_day[d]

        emit(d, evs, gts, d, "day")

        for lab, lo, hi in (("morning", 0, 12), ("afternoon", 12, 24)):
            se = [e for e in evs if lo <= hour_of(e, False) < hi]
            sg = [g for g in gts if lo <= hour_of(g, True) < hi]
            if sg:
                emit(f"{d} {lab}", se, sg, d, "half-day")

        for lab, lo, hi in SESSIONS:
            se = [e for e in evs if lo <= hour_of(e, False) < hi]
            sg = [g for g in gts if lo <= hour_of(g, True) < hi]
            if len(sg) >= 2:
                emit(f"{d} {lab}", se, sg, d, "session")

        apps = defaultdict(list)
        for g in gts:
            apps[g["app"]].append(g)
        for app, ag in apps.items():
            if len(ag) < 3:
                continue
            lo = min(hour_of(g, True) for g in ag)
            hi = max(hour_of(g, True) for g in ag)
            se = [e for e in evs if lo - 1 <= hour_of(e, False) <= hi + 1]
            emit(f"{d} {app} episode", se, ag, d, "episode")

    tr.close(); va.close()

    over = [s for s in stats if s[6] > MAX_SEQ_LENGTH]
    print(f"\nwrote {n_tr} training and {n_va} validation windows")
    print(f"prompts over {MAX_SEQ_LENGTH} tokens: {len(over)} of {len(stats)}")
    print(f"max prompt tokens: {max(s[6] for s in stats)}")
    print("\nwindows by scale:")
    for k in ("day", "half-day", "session", "episode"):
        print(f"  {k:<12}{scale_counts[k]:>5}")
    # ---- citation-density check: read this BEFORE spending a training run ----
    n_abs = sum(1 for s in stats if s[8])
    n_cit = sum(s[7] for s in stats)
    print("\nCITATION DENSITY (compare against the pre-fix run)")
    print(f"  absence-only targets : {n_abs} of {len(stats)}"
          f"   ({100*n_abs/len(stats):.0f}%)      [was 83 of 122, 68%]")
    print(f"  identifiers cited    : {n_cit}"
          f"                   [was 129]")
    print(f"  mean per target      : {n_cit/len(stats):.1f}")
    if n_abs / len(stats) > 0.55:
        print("  >> STILL ABSENCE-DOMINATED. Do not train. Investigate first.")

    print(f"\n{'window':<28}{'events':>8}{'admit':>7}{'t0':>5}{'t1':>5}{'t2':>5}{'tok':>7}{'cite':>6}")
    for d, tot, kept, k0, k1, k2, tok, cited, _ in stats[:10]:
        print(f"{str(d):<28}{tot:>8}{kept:>7}{k0:>5}{k1:>5}{k2:>5}{tok:>7}{cited:>6}")
    if len(stats) > 10:
        print(f"... {len(stats)-10} more")
    print("\nNote: windows overlap by construction, so they are not independent")
    print("samples. Partitioning is by DATE, so no held-out date appears in training.")


if __name__ == "__main__":
    raise SystemExit("Call main(tokenizer) from the notebook.")
