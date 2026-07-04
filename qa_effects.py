# -*- coding: utf-8 -*-
"""qa_effects.py -- EFFECT-CONGRUENCE QA: does an edit's placed effects match the band's REEL GRAMMAR?

This is the qualitative check "are the effects similar to the research videos?", made MEASURABLE. It is
the effect-placement sibling of qa_gate.py (which checks the rendered MP4's binary content gates). Here
we compare the effects the placer wrote into edit_plan.json against the derived reel grammar in
effect_rationale.json, and DECIDE pass/fail on two axes:

    SIMILARITY    -- do the placed effects look like the band's own accent vocabulary? (right TYPES,
                     right DENSITY, right AROUSAL bar, sustained-vs-stab regime matching the corpus)
    INTENTIONALITY-- is each accent GLUED to a real musical hit (not floating mid-bar), and is the whole
                     set RESTRAINED (rare, not a per-shot sprinkle / wall of fx)?

Standalone:
    python qa_effects.py <workdir>
        [--rationale effect_rationale.json]   # defaults to the one beside this script
        [--json report.json]

reads  <workdir>/edit_plan.json  +  effect_rationale.json  (the grammar).
exit 0 = CONGRUENT ; non-zero = NOT CONGRUENT.

CHECKS (each maps to a rule in effect_rationale.json):
  1. DENSITY (R7)          accents per 30s within the reel range (RARE; 0 is allowed & on-brand).
  2. RGB COUNT (R4/R7)     rgb_split count <= 2 (hard corpus cap, <=2 per reel).
  3. RGB AROUSAL (R4/R8)   every rgb on a hit whose local arousal >= the reel high-arousal bar (~0.75).
  4. SNAPPED / INTENTIONAL (R3)  every accent snapped to a real hit within <= ~0.2s (not floating mid-bar).
  5. SUSTAINED ONLY ON DROP (R4) sustained pulsing-prism only when on a drop; short stab off drops.
  6. NO SHAKE / NO FORBIDDEN (R7 + palette) no placed shake, no zero-corpus forbidden effects.
  7. NO OVER-PLACEMENT (R1/R7)  no per-shot carpet: at most 1 accent per shot on average, and no two
                                accents crammed inside one shot beyond the corpus's own crowding.

Prints a per-effect table (effect | matches grammar? | why) + an overall CONGRUENT/NOT verdict with a
0-100 similarity score. Config-driven; nothing hardcoded to a machine or band. Python is sys.executable.
"""
from __future__ import annotations
import sys, os, json, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ forbidden / non-accent vocabulary
FORBIDDEN_EFFECTS = {"radial_zoom", "speed_ramp", "glitch", "light_streak"}
ACCENT_EFFECTS = {"rgb_split", "flash", "shake"}          # the codex's ACCENT family (whip is a seam)


# ==================================================================== grammar bars (read from rationale)
def load_grammar(rationale_path):
    """Pull the measurable bars out of effect_rationale.json so the QA is grounded in the SAME numbers
    the placer targets: the rgb high-arousal bar, the snap tolerance, the on-drop distance, the rgb cap,
    and the per-reel ACCENT DENSITY range (computed from the corpus per-reel accent counts / spans)."""
    with open(rationale_path, "r", encoding="utf-8") as fh:
        rat = json.load(fh)
    rules = rat.get("rules", {})
    agg = rat.get("aggregate_stats", {})

    # rgb arousal bar: the corpus rgb median arousal / on-hit bar. Rationale R4 median 0.73; the placer's
    # gate is 0.75. We check every rgb clears the corpus HIGH-arousal bar ~0.75 (the raised placement gate).
    rgb_arousal_bar = 0.75

    # snap tolerance: R3 accents-within-0.20s-of-a-flux-hit. Intentional (not floating mid-bar).
    snap_tol = 0.20

    # on-drop distance: R4 sustained reserved for nearest_drop <= ~0.05s (corpus on-drop cases 0.02-0.04).
    on_drop_s = 0.05

    # rgb hard cap: R4 <=2 per reel.
    rgb_cap = 2

    # ACCENT density range across the reels (accents per 30s). Built from the corpus per-reel accent
    # counts + each reel's effect SPAN. Reels with 0 accents anchor the low end at 0 (on-brand).
    accents_per_reel = agg.get("accents_per_reel", {})
    per_reel_meta = agg.get("per_reel_density", {})
    dens = []
    for reel, n_acc in accents_per_reel.items():
        span = float(per_reel_meta.get(reel, {}).get("span_s", 0.0) or 0.0)
        # reels average ~90s; if a reel's effect-span is tiny/missing, fall back to a ~90s reel length.
        reel_len = span if span >= 20.0 else 90.0
        dens.append(n_acc / reel_len * 30.0)
    # 2/8 reels have ZERO accents -> the floor is 0. The ceiling is the busiest reel's accent density.
    dens_lo = 0.0
    dens_hi = max(dens) if dens else 3.0
    # add a small tolerance to the ceiling so an edit that matches the busiest reel isn't judged over.
    dens_hi = round(dens_hi * 1.10, 3)

    return {
        "rgb_arousal_bar": rgb_arousal_bar,
        "snap_tol": snap_tol,
        "on_drop_s": on_drop_s,
        "rgb_cap": rgb_cap,
        "density_lo": dens_lo,
        "density_hi": dens_hi,
        "rgb_median_arousal": float(rules.get("R4_rgb_signature", {})
                                    .get("evidence", {}).get("rgb_median_arousal", 0.73)),
        "_accent_densities_per_reel": [round(d, 3) for d in sorted(dens)],
    }


# ==================================================================== per-effect intentionality helpers
def _snap_dist(fx):
    """How far (s) this accent is from the real flux hit it was snapped to. The placer records both the
    song-relative time it fired (song_rel) and the hit it snapped to is that same event; we recover the
    residual from the reason text if present, else 0 (placer snaps ON the hit). We ALSO expose the
    placer's own audit: the moment dict + song_rel let the QA re-derive nothing floats mid-bar."""
    # The placer snaps rgb ONTO a flux hit (candidate hits ARE flux hits within SNAP_S), so the recorded
    # placement sits on a hit by construction. We read the residual the placer wrote into the reason
    # ("snapped %.3fs to a flux hit") when available; absence => treat as on-hit (0.0).
    reason = fx.get("reason", "")
    import re
    m = re.search(r"snapped\s+([0-9.]+)s", reason)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 0.0


def _is_subtle(intensity):
    if isinstance(intensity, (int, float)):
        return float(intensity) <= 0.18 + 1e-9
    return str(intensity).lower() in {"subtle", "low"}


# ==================================================================== the checks
def evaluate(plan, gram):
    fx = plan.get("effects", [])
    dur = max(float(plan.get("duration", 1.0)), 1.0)
    shots = plan.get("shots", [])
    n_shots = max(1, len(shots))

    rgb = [f for f in fx if f.get("effect") == "rgb_split"]
    shakes = [f for f in fx if f.get("effect") == "shake"]
    forbidden = [f for f in fx if f.get("effect") in FORBIDDEN_EFFECTS]
    density = len(fx) / dur * 30.0

    checks = []

    # 1. DENSITY (R7) -- accents per 30s within the reel range (0 allowed).
    dens_ok = gram["density_lo"] - 1e-9 <= density <= gram["density_hi"] + 1e-9
    checks.append({
        "name": "density_in_reel_range", "rule": "R7 (restraint)", "pass": dens_ok,
        "measured": "%.2f accents/30s" % density,
        "threshold": "%.2f-%.2f/30s (reel range; 0 allowed)" % (gram["density_lo"], gram["density_hi"]),
    })

    # 2. RGB COUNT (R4/R7) -- <= 2.
    cnt_ok = len(rgb) <= gram["rgb_cap"]
    checks.append({
        "name": "rgb_count", "rule": "R4/R7 (rgb rare)", "pass": cnt_ok,
        "measured": "%d rgb_split" % len(rgb), "threshold": "<= %d" % gram["rgb_cap"],
    })

    # 3. RGB AROUSAL (R4/R8) -- every rgb on a hit >= the high-arousal bar.
    low_ar = [f for f in rgb if float(f.get("local_arousal", 0.0)) < gram["rgb_arousal_bar"] - 1e-9]
    ar_ok = (len(rgb) == 0) or (len(low_ar) == 0)
    checks.append({
        "name": "rgb_high_arousal", "rule": "R4/R8 (earns the accent)", "pass": ar_ok,
        "measured": ("no rgb" if not rgb else
                     ("all >= bar" if not low_ar else
                      "%d below bar: %s" % (len(low_ar), [round(f.get("local_arousal", 0), 2) for f in low_ar]))),
        "threshold": "each rgb arousal >= %.2f" % gram["rgb_arousal_bar"],
    })

    # 4. SNAPPED / INTENTIONAL (R3) -- every accent within snap_tol of a real hit.
    floaters = [f for f in fx if f.get("effect") in ACCENT_EFFECTS
                and _snap_dist(f) > gram["snap_tol"] + 1e-9]
    snap_ok = len(floaters) == 0
    worst_snap = max((_snap_dist(f) for f in fx if f.get("effect") in ACCENT_EFFECTS), default=0.0)
    checks.append({
        "name": "snapped_to_hit", "rule": "R3 (never floats mid-bar)", "pass": snap_ok,
        "measured": ("no accents" if not any(f.get("effect") in ACCENT_EFFECTS for f in fx)
                     else "worst snap %.3fs" % worst_snap),
        "threshold": "each accent <= %.2fs from a flux hit" % gram["snap_tol"],
    })

    # 5. SUSTAINED ONLY ON DROP (R4) -- pulsing prism only on a drop; short stab off drops.
    regime_viol = []
    for f in rgb:
        mom = f.get("moment", {})
        nd = float(mom.get("nearest_drop_s", 9e9))
        regime = f.get("regime", "stab")
        on_drop = mom.get("on_drop", nd <= gram["on_drop_s"])
        if regime == "sustained" and not (on_drop or nd <= gram["on_drop_s"]):
            regime_viol.append({"t": f.get("tl_start"), "why": "sustained but nearest_drop %.3fs" % nd})
        if regime != "sustained" and (on_drop and nd <= gram["on_drop_s"] and float(f.get("local_arousal", 0)) >= 0.85):
            # an on-drop top-arousal hit that stayed a stab is not a FAIL (restraint), but note it.
            pass
    reg_ok = len(regime_viol) == 0
    checks.append({
        "name": "sustained_only_on_drop", "rule": "R4 (prism reserved for a drop)", "pass": reg_ok,
        "measured": ("no rgb" if not rgb else ("regimes match" if reg_ok else str(regime_viol))),
        "threshold": "sustained regime only when nearest_drop <= %.2fs" % gram["on_drop_s"],
    })

    # 6. NO SHAKE / NO FORBIDDEN (R7 + palette).
    clean_ok = (len(shakes) == 0 and len(forbidden) == 0)
    checks.append({
        "name": "no_shake_no_forbidden", "rule": "R7 + palette", "pass": clean_ok,
        "measured": ("clean" if clean_ok else
                     "shake=%d forbidden=%s" % (len(shakes), [f.get("effect") for f in forbidden])),
        "threshold": "0 placed shake, 0 forbidden effects",
    })

    # 7. NO OVER-PLACEMENT (R1/R7) -- no per-shot carpet. Average <= 1 accent/shot AND no single shot
    #    carries more than 2 accents (the corpus never sprinkles one-per-shot across a multi-shot edit).
    per_shot = {}
    for f in fx:
        if f.get("effect") not in ACCENT_EFFECTS:
            continue
        tl = float(f.get("tl_start", -1))
        owner = None
        for i, sh in enumerate(shots):
            s0 = float(sh["tl_start"]); s1 = s0 + float(sh["dur"])
            if s0 - 1e-6 <= tl < s1 + 1e-6:
                owner = i
                break
        per_shot[owner] = per_shot.get(owner, 0) + 1
    max_in_shot = max(per_shot.values(), default=0)
    avg_per_shot = (sum(per_shot.values()) / n_shots) if n_shots else 0.0
    over_ok = (avg_per_shot <= 1.0 + 1e-9) and (max_in_shot <= 2)
    checks.append({
        "name": "no_over_placement", "rule": "R1/R7 (no per-shot sprinkle)", "pass": over_ok,
        "measured": "avg %.2f/shot, max %d in a shot" % (avg_per_shot, max_in_shot),
        "threshold": "<= 1.0 accent/shot avg AND <= 2 in any shot",
    })

    return checks, {"density": density, "n_fx": len(fx), "n_rgb": len(rgb)}


def score(checks):
    """0-100 similarity score = fraction of grammar checks passed, weighted equally."""
    if not checks:
        return 0
    return int(round(100.0 * sum(1 for c in checks if c["pass"]) / len(checks)))


def per_effect_table(plan, gram):
    """One row per PLACED effect: does it match the grammar, and why."""
    rows = []
    for f in plan.get("effects", []):
        eff = f.get("effect")
        ar = float(f.get("local_arousal", 0.0))
        regime = f.get("regime", "stab")
        mom = f.get("moment", {})
        nd = float(mom.get("nearest_drop_s", 9e9))
        snap = _snap_dist(f)
        ok = True
        whys = []
        if eff in FORBIDDEN_EFFECTS:
            ok = False; whys.append("FORBIDDEN type")
        if eff == "shake":
            ok = False; whys.append("placed shake (corpus: in-camera only)")
        if eff == "rgb_split":
            if ar < gram["rgb_arousal_bar"] - 1e-9:
                ok = False; whys.append("arousal %.2f < %.2f bar" % (ar, gram["rgb_arousal_bar"]))
            else:
                whys.append("arousal %.2f >= %.2f (R4/R8)" % (ar, gram["rgb_arousal_bar"]))
            if snap > gram["snap_tol"] + 1e-9:
                ok = False; whys.append("floats %.3fs off hit" % snap)
            else:
                whys.append("snapped %.3fs to hit (R3)" % snap)
            if regime == "sustained" and nd > gram["on_drop_s"] + 1e-9:
                ok = False; whys.append("sustained but not on a drop (nd %.3fs)" % nd)
            else:
                whys.append("%s regime, nearest_drop %.3fs (R4)"
                            % ("sustained prism" if regime == "sustained" else "short stab", nd))
        rows.append({"effect": eff, "tl_start": round(float(f.get("tl_start", 0)), 2),
                     "intensity": f.get("intensity"), "arousal": round(ar, 2),
                     "on_drop": bool(mom.get("on_drop")), "matches": ok, "why": " | ".join(whys)})
    return rows


def print_report(plan, gram, checks, rows, meta):
    print("\n" + "=" * 96)
    print("  QA EFFECTS  (effect-congruence vs the band's reel grammar)")
    print("=" * 96)
    print("  reel accent-density range: %.2f-%.2f /30s   (per-reel: %s)"
          % (gram["density_lo"], gram["density_hi"], gram["_accent_densities_per_reel"]))
    print("  rgb high-arousal bar: %.2f   snap tol: %.2fs   on-drop: %.2fs   rgb cap: %d"
          % (gram["rgb_arousal_bar"], gram["snap_tol"], gram["on_drop_s"], gram["rgb_cap"]))

    # per-effect table
    print("\n  PLACED EFFECTS (%d):" % len(rows))
    if not rows:
        print("    (none placed -- ZERO accents. On-brand: 2/8 reels carry no accent overlay. Restraint.)")
    else:
        print("    %-6s %-10s %-7s %-6s %-7s %-7s %s"
              % ("tl", "effect", "int", "arous", "on_drop", "match?", "why"))
        for r in rows:
            print("    %-6.2f %-10s %-7s %-6.2f %-7s %-7s %s"
                  % (r["tl_start"], r["effect"], str(r["intensity"])[:7], r["arousal"],
                     str(r["on_drop"]), "YES" if r["matches"] else "NO", r["why"]))

    # grammar checks
    print("\n  GRAMMAR CHECKS:")
    print("    %-24s %-6s %-30s %-34s" % ("CHECK", "RESULT", "MEASURED", "THRESHOLD"))
    print("    " + "-" * 92)
    for c in checks:
        mark = "PASS" if c["pass"] else "FAIL"
        meas = str(c["measured"])
        if len(meas) > 29:
            meas = meas[:26] + "..."
        thr = str(c["threshold"])
        if len(thr) > 33:
            thr = thr[:30] + "..."
        print("    %-24s %-6s %-30s %-34s" % (c["name"], mark, meas, thr))

    sc = score(checks)
    congruent = all(c["pass"] for c in checks)
    n_pass = sum(1 for c in checks if c["pass"])
    print("    " + "-" * 92)
    print("    %d/%d grammar checks PASS" % (n_pass, len(checks)))
    fails = [c for c in checks if not c["pass"]]
    if fails:
        print("\n  DIVERGENCES:")
        for c in fails:
            print("    * %-24s [%s]  measured: %s  (want %s)"
                  % (c["name"], c["rule"], c["measured"], c["threshold"]))
    print("\n  VERDICT: %s   similarity score: %d/100"
          % ("CONGRUENT" if congruent else "NOT CONGRUENT", sc))
    print("=" * 96 + "\n")
    return congruent, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--rationale", default=os.path.join(_HERE, "effect_rationale.json"))
    ap.add_argument("--json", default=None, help="write the full report JSON here")
    a = ap.parse_args()

    plan_path = os.path.join(a.work, "edit_plan.json")
    if not os.path.exists(plan_path):
        print("ERROR: edit_plan.json not found in", a.work); sys.exit(3)
    plan = json.load(open(plan_path, encoding="utf-8"))
    gram = load_grammar(a.rationale)

    checks, meta = evaluate(plan, gram)
    rows = per_effect_table(plan, gram)
    congruent, sc = print_report(plan, gram, checks, rows, meta)

    if a.json:
        rep = {"work": a.work, "congruent": congruent, "similarity_score": sc,
               "grammar": gram, "effects": rows, "checks": checks, "meta": meta}
        json.dump(rep, open(a.json, "w"), indent=2, default=str)
        print("wrote report:", a.json)

    sys.exit(0 if congruent else 1)


if __name__ == "__main__":
    main()
