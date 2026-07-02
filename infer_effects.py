# -*- coding: utf-8 -*-
"""Effect inference v3 -- RESEARCH-DRIVEN placement via the code-rendered effect library.

WHAT CHANGED (v2 -> v3)
    v2 emitted hardcoded CapCut effect NAMES ("Leak 2"/"Shake 2") chosen by ad-hoc thresholds.
    v3 drives placement entirely from effects_lab (EL): its TRIGGER_RULES grammar (measured from the
    reel corpus / style_model.json), EL.make_placement() (auditable placement + REASON + gate), and
    EL.enforce_density() (per-effect + global min-spacing = restraint). Effects map to the names the
    library actually RENDERS: rgb_split, shake, radial_zoom, light_leak, speed_ramp (+ whip, which the
    library recommends using as the SEAMLESS TRANSITION seam at a join, not as a standalone overlay).

HOW IT PLACES (grounded in the musical-event SPINE, events.py)
    * Events come from events.spine(song_wav): hits (spectral-flux accents), drops (energy jumps),
      builds (ramp into a drop), resolutions, and an arousal (energy) envelope.
    * AROUSAL IS SAMPLED LOCALLY at the event (a +/-win peak), NOT at the shot midpoint -- an accent's
      local energy spike is what earns an effect; a shot's average would wash it out and never clear the
      research gates. This is the key fix that makes the gates fire on the right moments.
    * Per shot window [rel0, rel1] (song time), candidate events are mapped to TIMELINE time and handed
      to EL.make_placement(name, tl, local_arousal, label). The rule's arousal_gate decides if it fires;
      the rule's ladder decides intensity; the rule's envelope/duration ride along. Below-gate => dropped.
    * WHIP is NOT emitted as a standalone effect (the library flags standalone whip as too strong on
      bright footage). Instead each JOIN in the arrangement is tagged transition_style='whip' so the
      renderer builds EL.SeamlessTransition(style='whip') there (video xfade + audio acrossfade).

OUTPUT (into edit_plan.json)
    plan["effects"]        : list of EL placement dicts (effect/tl_start/intensity/envelope/
                             envelope_duration/event/reason/local_arousal/song_rel) -- what to BAKE.
    plan["transitions"]    : per-join {tl_start, style:'whip', overlap, reason} -- SeamlessTransition spec.
    plan["events_summary"] : bpm + event counts for QA.
    Every placement stays a plain dict a QA agent can thin/soften/drop before rendering (nothing baked).

Usage: python infer_effects.py <workdir>
"""
import sys, os, json
import numpy as np
import events as EV
import effects_lab as EL

WORK = sys.argv[1]
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
builds = list(sp["builds"])                                   # [(t0, t_drop), ...]
arr = np.asarray(sp["arousal"], float)
a_hz = float(sp["arousal_hz"])


def arousal_at(t):
    """instantaneous arousal (0..1) at song-time t."""
    return float(arr[int(np.clip(t * a_hz, 0, len(arr) - 1))])


def arousal_peak(t, win=0.4):
    """LOCAL PEAK arousal in [t-win, t+win] -- the accent's energy spike (see module docstring)."""
    i0 = int(np.clip((t - win) * a_hz, 0, len(arr) - 1))
    i1 = int(np.clip((t + win) * a_hz, 0, len(arr) - 1))
    return float(arr[i0:i1 + 1].max()) if i1 >= i0 else arousal_at(t)


def strongest(cands, k, keyfn):
    """top-k candidates by keyfn (desc)."""
    return [t for _, t in sorted(((keyfn(t), t) for t in cands), reverse=True)[:k]]


# ---------------------------------------------------------------- place effects per shot
shots = plan["shots"]
placements = []

for si, sh in enumerate(shots):
    rel0 = sh["master_start_abs"] - song_start                # shot window in SONG time
    rel1 = rel0 + sh["dur"]
    lo, hi = rel0 + 0.30, rel1 - 0.40                         # keep effects off the very edges/cuts

    def to_tl(t_song):
        return sh["tl_start"] + (t_song - rel0)

    # -- DROP in this shot -> radial_zoom (punch-in impact). take the single strongest drop. --
    d_in = [t for t in drops if lo <= t <= hi]
    for t in strongest(d_in, 1, arousal_peak):
        p = EL.make_placement("radial_zoom", to_tl(t), arousal_peak(t), label=sh["part_label"])
        if p:
            p["song_rel"] = round(t, 3); p["local_arousal"] = round(arousal_peak(t), 3)
            placements.append(p)

    # -- SECTION ENTRY (shot start, when a drop lands near the head) -> light_leak warm flash. --
    # research: light_leak fires on a section entry right after a drop; use the shot's opening beat.
    entry_t = rel0 + 0.15
    near_drop = any(abs(dt - rel0) <= 1.2 for dt in drops)     # a drop kicks off this section
    if si > 0 and near_drop:                                   # not the very first shot (no entry flash there)
        p = EL.make_placement("light_leak", to_tl(entry_t), arousal_peak(rel0 + 0.4), label=sh["part_label"])
        if p:
            # keep leaks tasteful on bright daylight footage: never stronger than medium (report note)
            if p["intensity"] == "strong":
                p["intensity"] = "medium"
            p["song_rel"] = round(entry_t, 3); p["local_arousal"] = round(arousal_peak(rel0 + 0.4), 3)
            placements.append(p)

    # -- STRONG HIT -> shake (biggest accents, gate 0.6) OR rgb_split (chromatic stab, gate 0.55). --
    # prefer shake on the very strongest hit; use rgb_split on the next strong hit for variety.
    h_in = [t for t in hits if lo <= t <= hi and all(abs(t - dt) > 1.0 for dt in d_in)]
    h_sorted = strongest(h_in, 4, arousal_peak)
    used_hit_times = []
    if h_sorted:
        # strongest hit -> shake
        t = h_sorted[0]
        p = EL.make_placement("shake", to_tl(t), arousal_peak(t), label=sh["part_label"])
        if p:
            p["song_rel"] = round(t, 3); p["local_arousal"] = round(arousal_peak(t), 3)
            placements.append(p); used_hit_times.append(t)
    # next distinct strong hit -> rgb_split (chromatic aberration stab)
    for t in h_sorted[1:]:
        if all(abs(t - u) > 2.0 for u in used_hit_times):
            p = EL.make_placement("rgb_split", to_tl(t), arousal_peak(t), label=sh["part_label"])
            if p:
                p["song_rel"] = round(t, 3); p["local_arousal"] = round(arousal_peak(t), 3)
                placements.append(p); used_hit_times.append(t)
            break

# -- SPEED_RAMP: at most ONE, into the strongest build->drop that has enough clean lead in a single --
# -- shot (needs envelope_duration s of continuous source). Reads only in motion; keep it rare. --
SR_DUR = EL.TRIGGER_RULES["speed_ramp"]["envelope_duration"]
sr_cands = []
for (t0, td) in builds:
    for si, sh in enumerate(shots):
        rel0 = sh["master_start_abs"] - song_start
        rel1 = rel0 + sh["dur"]
        # the drop and a full SR_DUR lead must fit inside this one shot, away from its edges
        if rel0 + 0.4 <= (td - SR_DUR) and td <= rel1 - 0.4:
            a = arousal_peak(td)
            sr_cands.append((a, si, sh, td - SR_DUR, td))
            break
if sr_cands:
    a, si, sh, t_song0, td = max(sr_cands, key=lambda x: x[0])
    rel0 = sh["master_start_abs"] - song_start
    tl0 = sh["tl_start"] + (t_song0 - rel0)
    p = EL.make_placement("speed_ramp", tl0, a, label=sh["part_label"])
    if p:
        p["song_rel"] = round(t_song0, 3); p["local_arousal"] = round(a, 3)
        p["ramp_to_song_rel"] = round(td, 3)
        placements.append(p)

# ---------------------------------------------------------------- restraint: enforce spacing (research)
# global floor 2.0s (a touch above the library's 1.2 tasteful floor) thins redundant stacking; the
# per-effect min_spacing_s in TRIGGER_RULES still governs same-effect crowding.
placements = EL.enforce_density(placements, global_min_spacing=2.0)
# extra restraint: no two effect TIME WINDOWS (tl_start .. tl_start+envelope_duration) may overlap or
# butt right up against each other -- a >=0.5s clear gap between windows keeps accents legible and
# prevents a "wall of fx" where one effect starts before the last has resolved. Keep the earlier one.
placements.sort(key=lambda p: p["tl_start"])
thinned, last_end = [], -1e9
for p in placements:
    dur = float(p.get("envelope_duration") or 0.7)
    if p["tl_start"] - last_end < 0.5:                        # starts before prior window fully clears
        continue
    thinned.append(p)
    last_end = p["tl_start"] + dur
placements = thinned
placements.sort(key=lambda p: p["tl_start"])

# ---------------------------------------------------------------- WHIP -> seamless transitions at joins
# The library recommends whip as the TRANSITION SEAM, not a standalone overlay. Tag every join so the
# renderer builds EL.SeamlessTransition(style='whip') (video xfade + audio acrossfade) there.
transitions = []
for sh in shots:
    if sh.get("is_join") and sh["tl_start"] > 0.1:
        transitions.append({
            "tl_start": round(sh["tl_start"], 3),
            "style": "whip",
            "overlap": 0.7,
            "reason": "seamless WHIP transition at %s join (video xfade + audio acrossfade)" % sh["part_label"],
        })
plan["transitions"] = transitions

plan["effects"] = placements
plan["events_summary"] = {
    "bpm": sp["bpm"], "n_drops": int(len(drops)), "n_hits": int(len(hits)),
    "n_resolutions": int(len(sp["resolutions"])), "n_builds": int(len(builds)),
    "arousal_hz": round(a_hz, 3),
    "placer": "effects_lab TRIGGER_RULES + make_placement + enforce_density (local-peak arousal)",
}
json.dump(plan, open(os.path.join(WORK, "edit_plan.json"), "w"), indent=2)

# ---------------------------------------------------------------- report
from collections import Counter
dens = len(placements) / max(plan["duration"], 1) * 30
print("inferred %d effects (%.1f/30s) via effects_lab, grounded in events (bpm %s)"
      % (len(placements), dens, sp["bpm"]))
print("  by effect:", dict(Counter(p["effect"] for p in placements)))
print("  seamless transitions (whip):", len(transitions))
for p in placements:
    print("   %6.2fs %-11s int=%-6s env=%-16s a=%.2f  (%s)"
          % (p["tl_start"], p["effect"], p["intensity"], p["envelope"],
             p.get("local_arousal", 0), p["reason"]))
for t in transitions:
    print("   %6.2fs TRANS %-6s overlap=%.2f  (%s)" % (t["tl_start"], t["style"], t["overlap"], t["reason"]))
