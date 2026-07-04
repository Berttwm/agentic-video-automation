# -*- coding: utf-8 -*-
"""Effect inference v4 -- REEL-GRAMMAR placement (rebuilt 2026-07-04 from effect_rationale.json/.md).

WHAT CHANGED (v3 -> v4)
    v3 placed accents PER SHOT (strongest hit per shot), gated arousal loosely, and GUARANTEED >=1
    accent via a forced fallback. That reads as a "sprinkle" -- one effect per shot -- and violates the
    band's measured restraint. v4 rebuilds placement to the DERIVED GRAMMAR in effect_rationale.json:

      * rgb_split is a RARE signature accent (corpus: 6 instances, <=2 per reel, only 4/8 reels use it),
        ALWAYS on a HIGH-arousal HIT, snapped to within ~0.15s of that hit (corpus median 0.148s).
      * Placement runs over the WHOLE ARRANGEMENT's event spine, not per shot: collect every usable hit,
        rank by LOCAL-PEAK arousal, and place at most the top ~1-2 whose local arousal clears a RAISED
        gate (>=0.75, the corpus high-arousal bar). If NO hit clears the bar -> PLACE NOTHING. Zero
        accents is a valid, on-brand outcome (2/8 real reels carry no accent overlay at all).
      * SUSTAINED pulsing-prism regime ONLY when the chosen hit sits ON a DROP (nearest_drop <= ~0.05s,
        the corpus on-drop distance); otherwise a SHORT chromatic stab (~0.13-0.35s, instant off).
      * A per-EDIT accent budget (~1 rgb per 30-40s, HARD max 2), not per-shot. Prefer chorus/hook
        sections as a soft prior.
      * flash is OFF by default -- it is a special hook device (white strobe on a repeated vocal hook,
        OR a warm wash at a section entry), not a generic per-hit accent, so we don't auto-place it.
      * NO placed shake (corpus shake reads as in-camera bumps). NO forced fallback. NO downbeat
        constraint (corpus doesn't support it -- accents scatter across the bar).

    WHIP is UNCHANGED and CORRECT: whips ride the section-seam CUT, so each JOIN in the arrangement is
    tagged as a SeamlessTransition(style='whip') the renderer builds at that join -- never a mid-shot
    overlay. Accents ride a hit MID-SHOT; whips ride the seam.

HOW IT PLACES (grounded in the musical-event SPINE, events.py)
    * Events come from events.spine(song_wav): hits (spectral-flux accents), drops (energy jumps),
      builds, resolutions, and an arousal (energy) envelope.
    * AROUSAL IS SAMPLED LOCALLY at the hit (a +/-win peak), not at the shot midpoint -- the hit's local
      energy spike is what earns the accent.
    * effects_lab still supplies each accent's intensity ladder + envelope/duration/regime
      (make_placement / choose_duration_envelope); v4 only decides WHICH hits get an accent and WHY.

OUTPUT (into edit_plan.json)
    plan["effects"]        : list of EL placement dicts (effect/tl_start/intensity/envelope/
                             envelope_duration/regime/event/reason/local_arousal/song_rel/moment) --
                             what to BAKE. Each keeps an AUDITABLE `reason` naming the reel rule it
                             satisfies.
    plan["transitions"]    : per-join {tl_start, style:'whip', overlap, reason} -- SeamlessTransition spec.
    plan["events_summary"] : bpm + event counts + placement provenance for QA.

Usage: python infer_effects.py <workdir>
"""
import sys, os, json
import numpy as np
import events as EV
import effects_lab as EL

WORK = sys.argv[1]

# ---------------------------------------------------------------- REEL GRAMMAR CONSTANTS (from rationale)
# These numbers come straight from effect_rationale.json (the derived grammar over the 8 reels). They are
# the placement law v4 enforces; qa_effects.py checks the placed effects back against these same bars.
RGB_AROUSAL_GATE = 0.75      # R4/R8: raise the accent gate to the corpus high-arousal bar (rgb median 0.73,
                             #        on-hit median ~0.83). Below this -> the hit does NOT earn an accent.
SNAP_S = 0.15                # R3: an accent must snap to a real flux hit within ~0.15s (median 0.148s).
ON_DROP_S = 0.05             # R4: SUSTAINED pulsing-prism only when the hit is essentially ON a drop
                             #     (corpus on-drop cases: nearest_drop 0.02-0.04s). Else a short stab.
RGB_HARD_MAX = 2             # R4/R7: rgb is <=2 per reel, HARD cap.
ACCENT_BUDGET_PER_S = 1.0 / 35.0   # R7: ~1 rgb per 30-40s (use 35s), a per-EDIT budget (not per-shot).
ACCENT_MIN_GAP_S = 6.0       # R7: accents are sparse; keep a wide floor between placed accents.

plan = json.load(open(os.path.join(WORK, "edit_plan.json"), encoding="utf-8"))
songs = json.load(open(os.path.join(WORK, "songs.json")))
songs = songs if isinstance(songs, list) else songs.get("songs", songs)
idx = plan["song_index"]
song_start = songs[idx - 1].get("start", 0.0) if idx - 1 < len(songs) else 0.0
song_wav = os.path.join(WORK, "song_%02d.wav" % idx)

# ---------------------------------------------------------------- event spine (song time)
sp = EV.spine(song_wav)
drops = np.asarray(sp["drops"], float)
hits = np.asarray(sp["hits"], float)
builds = list(sp["builds"])
arr = np.asarray(sp["arousal"], float)
a_hz = float(sp["arousal_hz"])


def arousal_at(t):
    """instantaneous arousal (0..1) at song-time t."""
    return float(arr[int(np.clip(t * a_hz, 0, len(arr) - 1))])


def arousal_peak(t, win=0.4):
    """LOCAL PEAK arousal in [t-win, t+win] -- the hit's energy spike (a shot average would wash it out)."""
    i0 = int(np.clip((t - win) * a_hz, 0, len(arr) - 1))
    i1 = int(np.clip((t + win) * a_hz, 0, len(arr) - 1))
    return float(arr[i0:i1 + 1].max()) if i1 >= i0 else arousal_at(t)


def nearest_drop_s(t):
    return float(min((abs(t - dt) for dt in drops), default=9e9))


def nearest_hit_s(t):
    return float(min((abs(t - ht) for ht in hits), default=9e9))


shots = plan["shots"]


# ---------------------------------------------------------------- shot mapping (song time <-> timeline)
# A hit at song-time t is "usable" if it falls inside a shot window, off the very edges/cuts. We keep the
# owning shot so we can map to timeline time and read its section label (the chorus/hook soft prior).
def _owning_shot(t):
    for sh in shots:
        r0 = sh["master_start_abs"] - song_start
        if r0 + 0.30 <= t <= r0 + sh["dur"] - 0.40:
            return sh
    return None


def _to_tl(sh, t_song):
    return sh["tl_start"] + (t_song - (sh["master_start_abs"] - song_start))


def _is_chorus(sh):
    lbl = (sh.get("part_label") or "").lower()
    return ("chorus" in lbl) or ("hook" in lbl) or ("solo" in lbl)


# ============================================================================================ PLACEMENT
# REEL-GRAMMAR accent placement, per-ARRANGEMENT (not per-shot):
#   1. Collect every usable HIT across the whole edit, snap each to the nearest real flux hit (they ARE
#      flux hits, so this just guarantees the <=0.15s intentionality), read its LOCAL-PEAK arousal.
#   2. Gate hard: keep only hits whose local arousal >= RGB_AROUSAL_GATE (0.75). If none clear it,
#      place NOTHING -- zero accents is on-brand.
#   3. Rank survivors; prefer chorus/hook sections (soft prior), then higher arousal. Place at most
#      the per-EDIT budget (~1 per 35s) and at most RGB_HARD_MAX (2), keeping a wide min gap.
#   4. rgb regime: SUSTAINED pulsing-prism iff the chosen hit is ON a drop (nearest_drop <= 0.05s),
#      else a short stab. make_placement/choose_duration_envelope pick the actual envelope + intensity.
def _moment_at(t):
    """The events.py 'moment' dict effects_lab consumes. on_drop is TRUE only when the hit sits ~ON a
    drop (nearest_drop <= ON_DROP_S) -- the corpus condition for the SUSTAINED pulsing prism. is_climax
    is left False; the sustained regime is driven purely by on_drop here (grammar R4), so exactly the
    on-drop accent rides sustained and every other accent stays a short stab."""
    ar = arousal_peak(t)
    nd = nearest_drop_s(t)
    on_drop = nd <= ON_DROP_S
    # sustain_s: how long arousal holds >= 0.85*peak after t (kept for audit; the regime is on_drop-led).
    i0 = int(np.clip(t * a_hz, 0, len(arr) - 1))
    thr = 0.85 * ar
    j = i0
    while j < len(arr) and arr[j] >= thr:
        j += 1
    sustain_s = (j - i0) / a_hz
    return {"arousal": ar, "on_drop": bool(on_drop), "nearest_drop_s": round(nd, 3),
            "is_climax": bool(on_drop and ar >= 0.85), "sustain_s": round(sustain_s, 3)}


# 1) candidate HITS across the whole arrangement (mapped to a shot, off the edges).
candidates = []
for t in sorted(set(round(float(h), 3) for h in hits)):
    sh = _owning_shot(t)
    if sh is None:
        continue
    if nearest_hit_s(t) > SNAP_S:        # must be a real flux hit within the snap window (intentionality)
        continue
    ar = arousal_peak(t)
    candidates.append({"t": t, "shot": sh, "arousal": ar})

# 2) HARD arousal gate -- only hits that clear the corpus high-arousal bar earn an accent.
gated = [c for c in candidates if c["arousal"] >= RGB_AROUSAL_GATE]

# 3) rank: chorus/hook sections first (soft prior), then higher local arousal, then earlier in time.
gated.sort(key=lambda c: (0 if _is_chorus(c["shot"]) else 1, -c["arousal"], c["t"]))

# per-EDIT budget: ~1 accent per 35s, hard-capped at RGB_HARD_MAX (2).
budget = min(RGB_HARD_MAX, max(0, int(round(float(plan.get("duration", 0.0)) * ACCENT_BUDGET_PER_S))))
# a one-song showcase of any real length is allowed at least ONE accent IF a hit clears the gate (but
# never a forced one -- if `gated` is empty this stays 0). round() gives 0 only for very short edits.
if budget == 0 and gated and float(plan.get("duration", 0.0)) >= 15.0:
    budget = 1

placements = []
placed_ts = []
for c in gated:
    if len(placements) >= budget:
        break
    t = c["t"]
    if any(abs(t - u) < ACCENT_MIN_GAP_S for u in placed_ts):
        continue                                     # keep accents sparse (R7)
    sh = c["shot"]
    mom = _moment_at(t)
    p = EL.make_placement("rgb_split", _to_tl(sh, t), mom["arousal"],
                          label=sh.get("part_label", "section"), moment=mom)
    if not p:                                        # below the effects_lab gate (shouldn't happen post-0.75)
        continue
    # AUDITABLE reason: name the reel rule this placement satisfies.
    regime = "SUSTAINED pulsing prism (R4: hit ON a drop, nearest_drop %.3fs)" % mom["nearest_drop_s"] \
        if p.get("regime") == "sustained" else \
        "short chromatic stab (R4: hit OFF any drop, nearest_drop %.3fs)" % mom["nearest_drop_s"]
    p["song_rel"] = round(t, 3)
    p["local_arousal"] = round(mom["arousal"], 3)
    p["moment"] = {k: mom[k] for k in ("on_drop", "nearest_drop_s", "is_climax", "sustain_s")}
    p["reason"] = ("rgb signature accent on HIGH-arousal HIT (a=%.2f>=%.2f) in %s, snapped %.3fs to a "
                   "flux hit | %s | reel rules R3+R4+R7+R8"
                   % (mom["arousal"], RGB_AROUSAL_GATE, sh.get("part_label", "section"),
                      nearest_hit_s(t), regime))
    placements.append(p)
    placed_ts.append(t)

# enforce the same corpus caps + spacing effects_lab knows about (belt-and-braces; budget already limits).
placements = EL.enforce_density(placements, global_min_spacing=ACCENT_MIN_GAP_S)
# force SUBTLE intensity on short stabs (taste lock: soft accents); leave the sustained ride as chosen so
# its intensity bump (a touch stronger to read as a deliberate prism) survives -- still within the codex.
for p in placements:
    if p.get("regime") != "sustained" and p.get("intensity") not in ("subtle", "low"):
        p["intensity"] = "subtle"
        p["reason"] = p.get("reason", "") + " | SUBTLE (taste lock: soft accents only)"
placements.sort(key=lambda p: p["tl_start"])

# NOTE: NO forced fallback. NO placed shake. flash is NOT auto-placed (special hook device only). The
# forbidden zero-corpus types (speed_ramp/radial_zoom/glitch/light_streak) are never emitted.

# ---------------------------------------------------------------- WHIP -> seamless transitions at joins
# UNCHANGED (correct): whip rides the section-seam CUT. Tag every join so the renderer builds
# EL.SeamlessTransition(style='whip') (video xfade + audio acrossfade) there -- never a mid-shot overlay.
transitions = []
for sh in shots:
    if sh.get("is_join") and sh["tl_start"] > 0.1:
        transitions.append({
            "tl_start": round(sh["tl_start"], 3),
            "style": "whip",
            "overlap": 0.7,
            "reason": "seamless WHIP transition at %s join (R2: whip = the section seam)" % sh["part_label"],
        })
plan["transitions"] = transitions

plan["effects"] = placements
plan["events_summary"] = {
    "bpm": sp["bpm"], "n_drops": int(len(drops)), "n_hits": int(len(hits)),
    "n_resolutions": int(len(sp["resolutions"])), "n_builds": int(len(builds)),
    "arousal_hz": round(a_hz, 3),
    "placer": "v4 reel-grammar (per-arrangement rgb on high-arousal hits, sustained only on-drop, "
              "hard max 2, no fallback/shake); whip=seam at joins",
    "rgb_arousal_gate": RGB_AROUSAL_GATE, "accent_budget": budget,
    "n_candidate_hits": len(candidates), "n_gated_hits": len(gated),
}
json.dump(plan, open(os.path.join(WORK, "edit_plan.json"), "w"), indent=2)

# ---------------------------------------------------------------- report
from collections import Counter
dens = len(placements) / max(plan["duration"], 1) * 30
print("inferred %d effects (%.2f/30s) via v4 reel-grammar, grounded in events (bpm %s)"
      % (len(placements), dens, sp["bpm"]))
print("  candidate hits: %d | cleared arousal gate (>=%.2f): %d | accent budget: %d"
      % (len(candidates), RGB_AROUSAL_GATE, len(gated), budget))
print("  by effect:", dict(Counter(p["effect"] for p in placements)))
print("  seamless transitions (whip):", len(transitions))
for p in placements:
    m = p.get("moment", {})
    print("   %6.2fs %-10s int=%-6s regime=%-9s a=%.2f on_drop=%s  (%s)"
          % (p["tl_start"], p["effect"], p["intensity"], p.get("regime", "?"),
             p.get("local_arousal", 0), m.get("on_drop"), p["reason"]))
for t in transitions:
    print("   %6.2fs TRANS %-6s overlap=%.2f  (%s)" % (t["tl_start"], t["style"], t["overlap"], t["reason"]))
if not placements:
    print("  (ZERO accents placed -- no hit cleared the arousal bar. This is a VALID, on-brand outcome:")
    print("   2/8 real reels carry no accent overlay. Restraint over a forced accent.)")
