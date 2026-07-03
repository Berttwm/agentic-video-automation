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
import grammar as G                              # SHARED grammar loader (style_model + effect_codex)

WORK = sys.argv[1]

# ---------------------------------------------------------------- READ THE MEASURED GRAMMAR (codex-led)
# effect_budget() = per-effect placement budget from effect_codex.corpus_frequency (which types, on
# which event kind, at what MEASURED max-per-edit). placeable_overlay_effects() = the codex overlay
# palette in tasteful/attested order (rgb_split + light_leak lead). This is what makes placement
# codex-DRIVEN: the loop below places the accents the CODEX supports, up to the codex's own caps,
# rather than a hardcoded one-off. effects_lab (fitted to the same codex) still supplies each accent's
# intensity ladder + envelope/duration -- one grammar, read the same way in both scripts.
EFFECT_BUDGET = G.effect_budget()
OVERLAY_PALETTE = G.placeable_overlay_effects()          # [(name, budget_rec), ...] tasteful order
# density ceiling anchored to the codex's LOW measured rate. Added-overlay effects in the corpus are
# sparse -- across 8 reels only rgb_split (6) and light_leak (1) are the tasteful added overlays, i.e.
# ~0.9 added accents per ~90s reel, ~0.3/30s. Placing the two lead tasteful types up to their measured
# caps (rgb_split max 2 + light_leak max 1) gives an absolute ceiling of ~3 accents on a full edit,
# which for a ~60s reel is ~1.5/30s -- clearly LOW and under the QA density gate (<3/30s). The per-type
# max_per_edit caps + enforce_density spacing do the fine limiting; this is the hard top.
_LEAD_CAP = sum(r["max_per_edit"] for n, r in OVERLAY_PALETTE
                if n in ("rgb_split", "light_leak"))                 # = 3 (2 rgb + 1 leak)
CODEX_MAX_OVERLAYS = max(1, _LEAD_CAP)
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

# THE CLIMAX (spine-faithful): the single most impactful DROP of the ARRANGEMENT -- the strongest drop
# (by local-peak arousal) that actually lands inside a usable shot window (not the song edge / a cut).
# Exactly this one instant is eligible for the SUSTAINED regime; everything else stays a short stab.
# Rationale: arousal is a RELATIVE normalized envelope, so the arrangement's single highest in-window
# drop IS its climax/solo peak by definition -- reserving ONE sustained effect for it matches the
# approved model (a tasteful sustained ride on THE peak) without carpeting the edit.
def _usable(t):
    for sh in shots:
        r0 = sh["master_start_abs"] - song_start
        if r0 + 0.30 <= t <= r0 + sh["dur"] - 0.40:
            return True
    return False

_drop_cands = [t for t in drops if _usable(t)]
_climax_t = max(_drop_cands, key=arousal_peak) if _drop_cands else None
# only treat it as a genuine climax if it clears a real impact floor (guards flat, low-dynamic songs:
# if nothing in the edit is punchy, NO sustained effect fires -- exactly the desired restraint).
CLIMAX_AROUSAL_FLOOR = 0.75
if _climax_t is not None and arousal_peak(_climax_t) < CLIMAX_AROUSAL_FLOOR:
    _climax_t = None


def moment_at(t):
    """Build the events.py 'moment' dict effects_lab.moment_is_sustained() consumes, so a genuinely
    impactful passage (a drop that rings out / the climax) earns the SUSTAINED envelope while an
    isolated hit stays a short stab. Fields: arousal (local peak), on_drop, nearest_drop_s,
    is_climax, sustain_s (how long energy stays >= 0.85*peak after t)."""
    ar = arousal_peak(t)
    nearest = float(min((abs(t - dt) for dt in drops), default=9e9))
    # sustain_s: how long arousal holds >= 0.85 * its local peak starting at t (a passage that rings out)
    i0 = int(np.clip(t * a_hz, 0, len(arr) - 1))
    thr = 0.85 * ar
    j = i0
    while j < len(arr) and arr[j] >= thr:
        j += 1
    sustain_s = (j - i0) / a_hz
    is_climax = (_climax_t is not None) and (abs(t - _climax_t) <= 0.30)
    return {"arousal": ar, "on_drop": nearest <= 0.25, "nearest_drop_s": nearest,
            "is_climax": bool(is_climax), "sustain_s": round(sustain_s, 3)}
placements = []

for si, sh in enumerate(shots):
    rel0 = sh["master_start_abs"] - song_start                # shot window in SONG time
    rel1 = rel0 + sh["dur"]
    lo, hi = rel0 + 0.30, rel1 - 0.40                         # keep effects off the very edges/cuts

    def to_tl(t_song):
        return sh["tl_start"] + (t_song - rel0)

    def add(name, t_song):
        """Place `name` at song-time t_song, driving duration/envelope from the MOMENT there.
        make_placement() picks the SUSTAINED regime (long attack/hold/release) only when the moment
        is a genuinely impactful drop/climax (moment_is_sustained), else a short instant stab."""
        mom = moment_at(t_song)
        p = EL.make_placement(name, to_tl(t_song), mom["arousal"], label=sh["part_label"], moment=mom)
        if p:
            p["song_rel"] = round(t_song, 3)
            p["local_arousal"] = round(mom["arousal"], 3)
            p["moment"] = {k: mom[k] for k in ("on_drop", "nearest_drop_s", "is_climax", "sustain_s")}
            placements.append(p)
        return p

    # ============================================================ CODEX-DRIVEN PLACEMENT (per shot)
    # Instead of a hardcoded one-off, we place the accents the effect_codex actually SUPPORTS, driven
    # by grammar.effect_budget(): for each placeable overlay type, fire it on the event kind the codex
    # measured it on (rgb_split/shake/flash on a HIT, light_leak on a section ENTRY), at the codex's
    # measured density (max_per_edit, enforced globally below). Every placement goes through
    # EL.make_placement() so the SAME taste ladder / envelope / arousal-gate governs it.
    d_in = [t for t in drops if lo <= t <= hi]
    h_in = [t for t in hits if lo <= t <= hi and all(abs(t - dt) > 1.0 for dt in d_in)]
    h_sorted = strongest(h_in, 4, arousal_peak)               # strongest hits in this shot
    used_here = []                                            # times already consumed in this shot

    def _spaced(t, gap=2.0):
        return all(abs(t - u) > gap for u in used_here)

    for name, rec in OVERLAY_PALETTE:
        ev = rec["event"]
        if ev == "hit":
            # rgb_split (and, if budgeted, flash/shake) ride the strongest HITS -- codex: effects sit
            # ON hits (median 0.13s). The strongest in-window drop is ALSO a hit context and earns the
            # SUSTAINED regime via moment_at(); place rgb on it first, then on the next strong hits.
            cands = []
            if name == "rgb_split":
                cands = strongest(d_in, 1, arousal_peak) + [t for t in h_sorted]
            else:
                cands = [t for t in h_sorted]
            placed_this_type = 0
            for t in cands:
                if placed_this_type >= rec["max_per_edit"]:
                    break                                     # codex per-type cap (per shot slice)
                if not _spaced(t):
                    continue
                if add(name, t):                              # gate/ladder/envelope decided in effects_lab
                    used_here.append(t)
                    placed_this_type += 1
        elif ev == "section_entry":
            # light_leak = the measured warm wash on a SECTION ENTRY right after a drop (codex: 1 inst).
            entry_t = rel0 + 0.15
            near_drop = any(abs(dt - rel0) <= 1.2 for dt in drops)
            if si > 0 and near_drop:                          # never on the very first shot
                p = add(name, entry_t)
                if p and p["intensity"] == "strong":          # keep the wash tasteful on bright footage
                    p["intensity"] = "medium"
        # ev == "build_to_drop" (blur_build): only valid immediately before a transition seam; the
        # renderer owns those joins (whip). We do not free-stand it here (codex placement principle).

# NOTE: speed_ramp / radial_zoom / glitch / light_streak have ZERO corpus instances and are FORBIDDEN
# (TRIGGER_RULES + grammar.effect_budget exclude them). They are intentionally never emitted here.

# ---------------------------------------------------------------- protect the SUSTAINED climax effect
# enforce_density keeps the EARLIER placement on a spacing tie, which would let an ordinary nearby stab
# evict the one SUSTAINED climax ride (the whole point of the approved model). So first drop any stab
# that sits inside a sustained placement's spacing window, guaranteeing the climax effect survives.
sustained = [p for p in placements if p.get("regime") == "sustained"]
if sustained:
    protected = []
    for p in placements:
        if p.get("regime") == "sustained":
            protected.append(p); continue
        clash = any(s["effect"] == p["effect"] and abs(p["tl_start"] - s["tl_start"]) < s["min_spacing_s"]
                    for s in sustained)
        if not clash:
            protected.append(p)
    placements = protected

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

# ---------------------------------------------------------------- TASTE FILTER: codex accents, SUBTLE
# The taste filter is now a SUBTLETY + RESTRAINT pass over the CODEX-DRIVEN placements, NOT a "keep only
# the sustained ride" veto (that veto is what previously deleted every codex accent and forced the lone
# fallback). It keeps the tasteful accents the codex supports -- the short rgb chromatic stabs on hits,
# the ONE warm light_leak on a section entry, and (when a real climax exists) the ONE sustained rgb
# prism ride -- while enforcing the user's LOCKS:
#   * SUBTLE intensity everywhere (small offset / soft bloom -- never a heavy chromatic smear/wash).
#   * NO carpet: keep at most CODEX_MAX_OVERLAYS accents (the codex's summed measured max-per-edit),
#     and only ONE sustained ride (the single climax); the rest stay short stabs.
#   * DROP shake (codex: the 3 instances read as in-camera bumps -> "QA should default to dropping").
#   * forbidden effects are already excluded upstream (grammar.effect_budget).
# This makes the codex-driven path NORMALLY fire (subtle stabs + entry leak), with the fallback below
# staying only as a floor for arrangements where no gated event survives.
def _taste_filter(pls):
    # 1) DROP shake outright (codex: probably in-camera, reads as a fault in stills).
    survivors = [p for p in pls if p["effect"] != "shake"]
    # 2) among rgb_split, at most ONE sustained climax ride survives as sustained; if several sustained
    #    somehow appear, keep the highest-arousal one and DEMOTE the others back to short stabs (they are
    #    still valid codex accents, just not the single climax -- so they are NOT deleted).
    sustained_rgb = [p for p in survivors if p["effect"] == "rgb_split" and p.get("regime") == "sustained"]
    rgb_keep = max(sustained_rgb, key=lambda p: p.get("local_arousal", 0)) if sustained_rgb else None
    for p in sustained_rgb:
        if p is not rgb_keep:
            p["regime"] = "stab"
            de = EL.choose_duration_envelope("rgb_split", p.get("local_arousal", 0.6), moment=None)
            if de:
                p["envelope"] = de["shape"]; p["envelope_duration"] = de["duration_s"]
                p["attack_s"], p["hold_s"], p["release_s"] = de["attack_s"], de["hold_s"], de["release_s"]
                p["hold_pulses"] = de["hold_pulses"]
    kept = list(survivors)
    # 3) force SUBTLE intensity on every survivor (tasteful, not glitchy). The light_leak even at the
    #    'subtle' ladder value (0.18) is a heavy orange wash on bright daylight footage, so the kept leak
    #    gets a REDUCED NUMERIC intensity (~half of subtle) -> a gentle warm bloom (numeric passes
    #    straight through effects_lab.level_value, no ladder lookup).
    LEAK_SOFT = 0.09      # brightness add @peak (vs subtle=0.18); soft bloom, not a wash
    for p in kept:
        if p["effect"] == "light_leak":
            p["intensity"] = LEAK_SOFT
            p["reason"] = p.get("reason", "") + " | SOFT bloom (taste lock: leak below subtle)"
        elif p.get("intensity") != "subtle":
            p["intensity"] = "subtle"
            p["reason"] = p.get("reason", "") + " | SUBTLE (taste lock: soft accents only)"
    # 4) the kept sustained rgb rides a touch longer so the attack/hold/fall-off breathes (~1.4s)
    if rgb_keep is not None and rgb_keep.get("regime") == "sustained":
        rgb_keep["envelope_duration"] = 1.4
        a, r = rgb_keep.get("attack_s") or 0.15, rgb_keep.get("release_s") or 0.15
        rgb_keep["hold_s"] = round(max(0.0, 1.4 - a - r), 3)
    # 5) restraint cap: keep at most CODEX_MAX_OVERLAYS accents (the codex's summed measured max-per-edit,
    #    a LOW number), prioritising the sustained climax ride, then the section-entry leak, then the
    #    strongest short stabs -- so we place the best-supported subtle accents and never a carpet.
    kept.sort(key=lambda p: p["tl_start"])
    if len(kept) > CODEX_MAX_OVERLAYS:
        def _prio(p):
            if p is rgb_keep:                 return (0, 0)                      # the one sustained ride
            if p["effect"] == "light_leak":   return (1, 0)                      # the entry wash
            return (2, -float(p.get("local_arousal", 0)))                        # strongest stabs first
        kept = sorted(sorted(kept, key=_prio)[:CODEX_MAX_OVERLAYS], key=lambda p: p["tl_start"])
    return kept
placements = _taste_filter(placements)

# ---------------------------------------------------------------- GUARANTEE >=1 subtle accent
# On some arrangements NO drop/hit inside the chosen shots clears the research gates, so the taste
# filter leaves ZERO effects (the defect the user hit on the last hand-built edit). A one-song showcase
# should still carry ONE intentional, SUBTLE accent. So if nothing survived, place exactly one -- a
# subtle rgb_split on the strongest HIT inside a CHORUS shot, or (failing that) a soft light_leak at the
# SOLO entry. We build it through effects_lab so it stays a legitimate, auditable, taste-filtered
# placement (subtle intensity, sustained-on-impactful envelope) -- never heavy, never frequent.
def _fallback_accent():
    # candidate 1: strongest hit that sits inside a chorus shot
    chorus_shots = [sh for sh in shots if "chorus" in sh.get("part_label", "").lower()]
    best = None  # (arousal, tl_start, song_rel, label)
    for sh in chorus_shots:
        r0 = sh["master_start_abs"] - song_start
        lo, hi = r0 + 0.30, r0 + sh["dur"] - 0.40
        cands = [t for t in hits if lo <= t <= hi]
        for t in cands:
            ar = arousal_peak(t)
            tl = sh["tl_start"] + (t - r0)
            if best is None or ar > best[0]:
                best = (ar, tl, t, sh["part_label"])
    if best is not None:
        ar, tl, t_song, lbl = best
        mom = moment_at(t_song)
        # place at (forced) subtle: use a high arousal so the ladder returns a level, then override.
        p = EL.make_placement("rgb_split", tl, max(ar, EL.TRIGGER_RULES["rgb_split"]["arousal_gate"]),
                              label=lbl, moment=mom)
        if p:
            p["intensity"] = "subtle"
            p["song_rel"] = round(t_song, 3)
            p["local_arousal"] = round(ar, 3)
            p["moment"] = {k: mom[k] for k in ("on_drop", "nearest_drop_s", "is_climax", "sustain_s")}
            p["reason"] = ("FALLBACK subtle rgb accent on strongest hit inside %s (no gated event survived)"
                           % lbl)
            return p
    # candidate 2: soft light_leak at the solo entry (or the last shot's head)
    solo_shot = next((sh for sh in shots if "solo" in sh.get("part_label", "").lower()), None)
    target = solo_shot or (shots[-1] if shots else None)
    if target is not None and target["tl_start"] > 0.1:
        t_song = (target["master_start_abs"] - song_start) + 0.15
        p = EL.make_placement("light_leak", target["tl_start"] + 0.15,
                              max(0.5, EL.TRIGGER_RULES["light_leak"]["arousal_gate"]),
                              label=target["part_label"], moment=moment_at(t_song))
        if p:
            p["intensity"] = 0.09     # soft bloom (below 'subtle'), same as the taste-filter leak
            p["song_rel"] = round(t_song, 3)
            p["local_arousal"] = round(arousal_peak(t_song), 3)
            p["reason"] = ("FALLBACK soft light_leak at %s entry (no gated event survived)"
                           % target["part_label"])
            return p
    return None


if not placements:
    fb = _fallback_accent()
    if fb:
        placements = [fb]
        print("  (fallback: placed 1 subtle accent -- no gated event survived the taste filter)")

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
