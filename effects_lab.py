# -*- coding: utf-8 -*-
"""
effects_lab.py -- CODE-RENDERED (ffmpeg) visual-effect library for the the band auto-video-editor.

WHY THIS EXISTS
    The editor used to place CapCut effects BLIND (it could not see CapCut's render), so effects were
    mistimed and unnatural. This module replaces that with ffmpeg effects that are:
      * TEMPORAL   -- every effect ramps IN and OUT via time-varying filter params (no instant pops).
      * PARAMETERIZED -- intensity / envelope_duration / envelope shape are knobs.
      * VISUALLY VALIDATED -- each was rendered on real gig footage, viewed frame-by-frame, and tuned
                              against the actual @YOUR_HANDLE reel moments (see effects_lab_report.md).
      * RESEARCH-DRIVEN placement -- each effect carries a TRIGGER SPEC derived from style_model.json
                              (the research team's measured grammar) so the editor + QA can place and
                              audit effects on real musical events, not on a timer.

WHAT THE EDITOR / QA GET
    EFFECTS            : dict name -> EffectDef (filter builder + trigger spec + intensity ladder).
    TRIGGER_RULES      : plain dict/JSON of the placement grammar (auditable; QA can read + tune).
    build_effect_vf(name, intensity, envelope_duration, ...) -> ffmpeg -vf / filtergraph string.
    plan_intensity(name, arousal) -> intensity for a given section energy (arousal 0..1).
    SeamlessTransition -- the SIGNATURE transition: video cross-dissolve/blur xfade + audio acrossfade.
    render_effect(...) / render_transition(...) -- render a demo clip (used by the CLI + QA feedback loop).

FEEDBACK LOOP (for the QA agent)
    Every effect exposes knobs (intensity, density, envelope) and every PLACEMENT should carry a
    `reason` string (see make_placement()). A QA agent can lower density, drop an incongruent effect,
    or soften intensity purely by editing the returned data -- nothing is hardcoded at the call site.

VALIDATION STATUS (2026-07-03, instance-level EFFECT CODEX rebuild; ground truth = effect_codex.json,
evidence = effects_match_*.jpg. Every verdict is from a direct A/B: the coded effect applied to the
reel's OWN clean pre-window footage vs the reel's actual effect frames, envelopes measured per-frame.)
    rgb_split      MATCHED  -- asymmetric stab: R -40px / B +14px @720w, ~0.1s attack, ~0.3s hold,
                               INSTANT off (DXUDMp@23.42 R-B offset series reproduced frame-for-frame).
                               Residual: real adds a faint ghost-echo trail we don't reproduce.
    flash          MATCHED  -- NEW: 5Hz white strobe train, 1-frame white peaks (luma 255 vs real 252),
                               washed shoulders, base troughs (DT-b@33.8-34.5 luma series).
    light_leak     MATCHED  -- rise(0.2s)-hold(0.55s)-fall(0.2s) warm wash; luma +115 vs real +109,
                               red-ratio +0.50 vs real +0.53 (DT-b@23.0). Residual: real hue drifts
                               orange->violet; coded stays warm-pink.
    whip           PARTIAL  -- blur envelope matched (min-sharpness ratio 0.26 vs real 0.22, avgblur
                               peak 260px@720w) BUT real whips are true camera pans that BRIDGE TWO
                               SHOTS (100-350 px/frame translation). Only valid AT a cut
                               (SeamlessTransition style='whip'), never as a mid-shot overlay.
    shake          PARTIAL  -- refit to the real bump: ~12px @720w, ~3Hz, 0.45s, 1-2 cycles (the old
                               11-13Hz buzz was wrong). The 3 corpus instances are almost certainly
                               IN-CAMERA bumps (crowd/impact) with intra-frame motion blur; coded is
                               displacement-only. Use at most once per edit, biggest hit, or drop.
    blur_build     PARTIAL  -- NEW: slow gblur ramp (~2s) INTO a transition; this is what the single
                               "radial" corpus instance actually is (DYw@62.5-64.3, defocus build
                               ending in a whip). Residual: real build has a camera-drift component.
    seamless_trans SOLID    -- unchanged (video xfade + audio acrossfade verified frames+RMS).
    radial_zoom    UNMATCHED -- the layered centre-sharp zoom burst appears NOWHERE in the corpus
                               (its one supposed instance is the blur_build above). Kept only for API
                               compat; TRIGGER_RULES marks it do_not_place. Recommend removal.
    speed_ramp     NO CORPUS EVIDENCE -- never detected in any reel; do_not_place.
    glitch         NO CORPUS EVIDENCE -- do_not_place.
    light_streak   NO CORPUS EVIDENCE -- do_not_place.

PATHS: pass ffmpeg/ffprobe in, or rely on the module defaults (resolved via config.json).

CLI (for future visual checks / QA):
    python effects_lab.py list
    python effects_lab.py demo <effect> <src.mp4> <timestamp_s> [--intensity med] [--out out.mp4] [--tile]
    python effects_lab.py ladder <effect> <src.mp4> <timestamp_s>          # render subtle/medium/strong
    python effects_lab.py transition <src.mp4> <tA_s> <tB_s> [--style dissolve|blur|whip]
"""
from __future__ import annotations
import os, sys, json, math, random, subprocess, tempfile, argparse

# ------------------------------------------------------------------ defaults / config
from paths import FFMPEG as FFMPEG_DEFAULT, FFPROBE as FFPROBE_DEFAULT

# working resolution for previews / renders (the source is 4K portrait 2160x3840 -> scale DOWN for
# speed; the editor can override PREVIEW_W/H or pass its own scale). 9:16 portrait preserved.
PREVIEW_W, PREVIEW_H = 540, 960
FPS = 30  # reels are 30fps; match so effect timing reads the same as the corpus.


# ================================================================== ENVELOPES
# Every effect multiplies its peak parameter by a normalized envelope e(x) in [0,1], x = t/duration.
# The envelope is what makes an effect SLIDE in and out instead of popping (research: "no instant pops").
def envelope(shape: str, x: float) -> float:
    """Return envelope value in [0,1] for normalized progress x in [0,1]."""
    x = max(0.0, min(1.0, x))
    if shape == "tri":                       # symmetric ramp up then down (default for accents)
        return 1.0 - abs(2.0 * x - 1.0)
    if shape == "tri_sharp":                 # narrower spike (fast whip-like accents)
        return (1.0 - abs(2.0 * x - 1.0)) ** 0.6
    if shape == "in_out":                    # smooth hann-ish ease in/out
        return 0.5 - 0.5 * math.cos(2.0 * math.pi * x)
    if shape == "fast_in_slow_out":          # flash: quick attack, exponential decay (light leaks)
        return x / 0.12 if x < 0.12 else math.exp(-3.2 * (x - 0.12))
    if shape == "ramp_up":                   # build INTO a cut (whip transition tail)
        return x
    if shape == "ramp_down":                 # release AFTER a hit
        return 1.0 - x
    if shape == "hold":                      # constant (sustained looks, e.g. a graded section)
        return 1.0
    if shape == "stab_hold":                 # MEASURED rgb-stab envelope: ~2-frame attack, hold to
        return min(1.0, x / 0.15)            # the end; the caller must snap to 0 after duration
    if shape == "rise_hold_fall":            # MEASURED light-leak envelope (attack .2/hold .55/fall .2)
        if x < 0.21: return x / 0.21
        if x < 0.79: return 1.0
        return max(0.0, (1.0 - x) / 0.21)
    if shape == "pulse_hold":                # SUSTAINED rgb PRISM (codex style-C, re-measured DWthZY@19):
        # intentional attack -> a PULSING hold that oscillates at intensity -> fall-off. Not a flat
        # block, not a blip -- it "breathes". Envelope = ramped hann window * a pulse carrier.
        # attack 15% / hold 70% (pulsing) / release 15%.
        A, R = 0.15, 0.15
        if x < A:                    ramp = x / A
        elif x > 1.0 - R:            ramp = max(0.0, (1.0 - x) / R)
        else:                        ramp = 1.0
        # pulse carrier over the hold: 2.5 cycles across the whole window, floor 0.55 so it never
        # fully drops out mid-hold (a prism that pulses, not a strobe).
        pulse = 0.55 + 0.45 * (0.5 - 0.5 * math.cos(2.0 * math.pi * 2.5 * x))
        return ramp * pulse
    return 1.0 - abs(2.0 * x - 1.0)


def envelope_ar(shape: str, x: float, attack_frac: float, release_frac: float, hold_pulses: float = 0.0):
    """Generalized attack/hold/release envelope in [0,1] over normalized x in [0,1], driven by
    EXPLICIT attack/release fractions (so duration + attack + release can be chosen from the moment,
    not baked into the shape name). If hold_pulses>0 the hold section PULSES (rides + breathes) at that
    many cycles across the window; otherwise the hold is a flat sustain. `shape` selects the ramp
    curve of the attack/release shoulders ('linear' or 'smooth'/hann)."""
    x = max(0.0, min(1.0, x))
    A = max(1e-6, attack_frac); R = max(1e-6, release_frac)
    if x < A:
        r = x / A
    elif x > 1.0 - R:
        r = max(0.0, (1.0 - x) / R)
    else:
        r = 1.0
    if shape == "smooth":
        r = 0.5 - 0.5 * math.cos(math.pi * min(1.0, r))   # ease the shoulders
    if hold_pulses > 0.0:
        pulse = 0.55 + 0.45 * (0.5 - 0.5 * math.cos(2.0 * math.pi * hold_pulses * x))
        return r * pulse
    return r


def _sendcmd_file(pairs, path):
    """Write an ffmpeg sendcmd script; pairs = list of (t, 'filter opt value'). Return escaped path."""
    open(path, "w").write("\n".join("%.4f %s;" % (t, cmd) for t, cmd in pairs))
    return path.replace("\\", "/").replace(":", "\\:")


# ================================================================== INTENSITY LADDERS
# Per-effect subtle/medium/strong peak values, FITTED to measured codex instances (effect_codex.json).
# All px values are defined at the REFERENCE width 720 (the reels' width); builders scale by w/REF_W.
# 'strong' = the measured real instance; subtle/medium are proportional pull-backs.
REF_W = 720.0   # px-valued intensities are calibrated at this width (reel corpus resolution)
INTENSITY = {
    #                     subtle  medium  strong    unit / meaning
    "rgb_split":   dict(subtle=16, medium=28, strong=40, unit="px RED offset @720w (B = -0.35x, opposite); measured 16 (DYw@95.17) .. 40-48 (DXUDMp@23.42)"),
    "whip":        dict(subtle=100, medium=180, strong=260, unit="px horizontal avgblur @720w at peak; 260 -> min-sharpness ratio 0.26 vs real 0.22"),
    "shake":       dict(subtle=8, medium=12, strong=15, unit="px displacement @720w (~3Hz bump); real peak ~12px"),
    "light_leak":  dict(subtle=0.18, medium=0.27, strong=0.36, unit="brightness add at peak (0.36 -> luma +115 vs real +109; + gamma_r warm push)"),
    "flash":       dict(subtle=1, medium=2, strong=3, unit="number of 5Hz white strobe pulses (each peaks 1 frame at white)"),
    "blur_build":  dict(subtle=0.4, medium=0.6, strong=1.0, unit="gblur sigma @720w at end of the build (into a transition)"),
    # ---- no corpus evidence / unmatched; kept for API compat only ----
    "radial_zoom": dict(subtle=0.006, medium=0.010, strong=0.018, unit="UNMATCHED-do_not_place; per-layer zoom step"),
    "speed_ramp":  dict(subtle=0.75, medium=0.5, strong=0.35, unit="NO-EVIDENCE-do_not_place; starting speed"),
    "glitch":      dict(subtle=10, medium=18, strong=26, unit="NO-EVIDENCE-do_not_place; px RGB tear"),
    "light_streak":dict(subtle=0.35, medium=0.55, strong=0.75, unit="NO-EVIDENCE-do_not_place; overlay opacity"),
}

def level_value(name: str, intensity):
    """intensity may be a level name ('subtle'/'medium'/'strong') or a raw number (passthrough)."""
    if isinstance(intensity, (int, float)):
        return float(intensity)
    return float(INTENSITY[name][intensity])


# ================================================================== TRIGGER RULES (from effect_codex.json)
# REBUILT 2026-07-03 from INSTANCE-LEVEL ground truth (51 measured instances across 8 reels), not
# aggregate statistics. Each rule may only fire in musical+visual contexts where the effect ACTUALLY
# appears in the codex, at corpus-consistent frequency. `corpus` documents the evidence; `max_per_edit`
# enforces corpus frequency (reels average ~90s, comparable to one edit).
# Codex musical contexts: median distance to nearest spectral-flux hit = 0.13s (effects sit ON hits).
TRIGGER_RULES = {
    "rgb_split": {
        "event": "hit",                       # every codex instance is <=0.2s from a flux hit
        "arousal_gate": 0.55,                 # instances at arousal 0.44-0.94, median 0.73
        "min_spacing_s": 8.0,
        "max_per_edit": 2,                    # corpus: 6 instances / 8 reels, never >2 per reel
        "envelope": "stab_hold", "envelope_duration": 0.42,
        "intensity_at_arousal": [(0.55, "subtle"), (0.7, "medium"), (0.85, "strong")],
        "reason_tmpl": "rgb stab on strong HIT in {label} (arousal {a:.2f}) -- codex: 6/8 reels, on-hit",
        "corpus": "6 instances (DWthZY 19.08/19.5, DXUDMp 23.42, DXW06 21.33/60.08, DYw 95.17); "
                  "R -16..-48px, ~0.1s attack, 0.15-0.4s hold, INSTANT off",
        "capcut_analog": "(no direct analog; render in ffmpeg)",
    },
    "whip": {
        "event": "section_change",            # ONLY at a cut: every mid-reel whip bridges two shots
        "arousal_gate": 0.0,                  # corpus whips occur at any arousal (0.04-0.83)
        "min_spacing_s": 10.0,
        "max_per_edit": 3,                    # corpus: ~2.5 per reel (the standard seam)
        "envelope": "tri_sharp", "envelope_duration": 0.55,
        "intensity_at_arousal": [(0.0, "medium"), (0.6, "strong")],
        "reason_tmpl": "whip smear across cut at {label} boundary (arousal {a:.2f}) -- codex seam",
        "corpus": "~20 mid-reel instances; sharpness drops to 0.13-0.35x for 0.4-1.2s with "
                  "100-350px/frame pan. PARTIAL: coded has blur but no true pan -- MUST be placed at "
                  "a join via SeamlessTransition(style='whip'), never as a mid-shot overlay. "
                  "Also the reel-ENDING convention: 5/8 reels end on a blur-out.",
        "capcut_analog": "Blur / Mix transition",
        "note": "use SeamlessTransition(style='whip') at the actual join, not as an overlay.",
    },
    "flash": {
        "event": "hit",                       # burst rides the biggest vocal/instrument accents
        "arousal_gate": 0.5,                  # DT-b bursts at arousal 0.50-0.65
        "min_spacing_s": 15.0,
        "max_per_edit": 1,                    # corpus: strobe bursts in only 1/8 reels -- rare, loud
        "envelope": "strobe", "envelope_duration": 0.6,  # 3 pulses @5Hz inside this window
        "intensity_at_arousal": [(0.5, "subtle"), (0.6, "medium"), (0.75, "strong")],
        "reason_tmpl": "white strobe burst on peak HIT in {label} (arousal {a:.2f}) -- codex: DT-b hook",
        "corpus": "DT-b 33.83-34.5 + 44.5 ('zombie' hook): 5Hz train, 1-frame white peaks "
                  "(luma 252), tri shoulders; intensity = pulse count (1-3)",
        "capcut_analog": "(white flash / strobe)",
    },
    "light_leak": {
        "event": "section_entry",             # the one instance sits at a section entry
        "arousal_gate": 0.45,
        "min_spacing_s": 20.0,
        "max_per_edit": 1,                    # corpus: 1 instance in 8 reels
        "envelope": "rise_hold_fall", "envelope_duration": 0.95,
        "intensity_at_arousal": [(0.45, "medium"), (0.65, "strong")],
        "reason_tmpl": "warm leak wash entering {label} (arousal {a:.2f}) -- codex: DT-b@23.0 only",
        "corpus": "1 instance (DT-b 23.0): luma +109 (43%), red-ratio 0.24->0.75, "
                  "attack 0.17s / hold ~0.55s / release 0.2s",
        "capcut_analog": "Leak 2",
    },
    "shake": {
        "event": "hit",
        "arousal_gate": 0.6,
        "min_spacing_s": 30.0,
        "max_per_edit": 1,                    # 3 corpus instances but likely IN-CAMERA bumps
        "envelope": "tri", "envelope_duration": 0.45,
        "intensity_at_arousal": [(0.6, "medium"), (0.8, "strong")],
        "reason_tmpl": "camera-bump shake on biggest HIT in {label} (arousal {a:.2f}) -- codex: rare",
        "corpus": "3 instances (DWthZY 85.25, DXW06 18.67, DYw 75.67): ~12px @720w, 2-4Hz, "
                  "0.3-0.45s, 1-2 cycles. PARTIAL (real bumps carry motion blur; probably in-camera). "
                  "QA may drop freely.",
        "capcut_analog": "Shake (single bump, NOT sustained)",
    },
    "blur_build": {
        "event": "build_to_drop",             # a defocus build that must RESOLVE INTO a transition
        "arousal_gate": 0.45,
        "min_spacing_s": 30.0,
        "max_per_edit": 1,                    # corpus: 1 instance in 8 reels
        "envelope": "ramp_up", "envelope_duration": 2.0,
        "intensity_at_arousal": [(0.45, "medium"), (0.7, "strong")],
        "reason_tmpl": "blur build into transition at {label} (arousal {a:.2f}) -- codex: DYw@62.5 only",
        "corpus": "1 instance (DYw 62.5-64.33): sharpness ramps down ~40% over ~2s then a whip cut. "
                  "Only place directly BEFORE a SeamlessTransition.",
        "capcut_analog": "Blur (keyframed ramp)",
    },
    # ---- NO CORPUS EVIDENCE / UNMATCHED: placement forbidden; kept for API compat ----
    "radial_zoom": {
        "event": "drop", "arousal_gate": 1.01, "min_spacing_s": 999.0, "max_per_edit": 0,
        "do_not_place": True, "experimental": True,
        "envelope": "in_out", "envelope_duration": 1.2,
        "intensity_at_arousal": [(1.01, "subtle")],
        "reason_tmpl": "FORBIDDEN radial_zoom {label} {a:.2f}",
        "corpus": "0 instances -- UNMATCHED. Its one supposed instance (DYw 62.5) is blur_build.",
        "capcut_analog": "(none)",
    },
    "speed_ramp": {
        "event": "build_to_drop", "arousal_gate": 1.01, "min_spacing_s": 999.0, "max_per_edit": 0,
        "do_not_place": True, "experimental": True,
        "envelope": "hold", "envelope_duration": 2.0,
        "intensity_at_arousal": [(1.01, "subtle")],
        "reason_tmpl": "FORBIDDEN speed_ramp {label} {a:.2f}",
        "corpus": "0 instances detected in any reel.",
        "capcut_analog": "(setpts)",
    },
    "glitch": {
        "event": "hit", "arousal_gate": 1.01, "min_spacing_s": 999.0, "max_per_edit": 0,
        "do_not_place": True, "experimental": True,
        "envelope": "tri", "envelope_duration": 0.8,
        "intensity_at_arousal": [(1.01, "subtle")],
        "reason_tmpl": "FORBIDDEN glitch {label} {a:.2f}",
        "corpus": "0 instances.",
        "capcut_analog": "(none)",
    },
    "light_streak": {
        "event": "section_entry", "arousal_gate": 1.01, "min_spacing_s": 999.0, "max_per_edit": 0,
        "do_not_place": True, "experimental": True,
        "envelope": "in_out", "envelope_duration": 1.4,
        "intensity_at_arousal": [(1.01, "subtle")],
        "reason_tmpl": "FORBIDDEN light_streak {label} {a:.2f}",
        "corpus": "0 instances.",
        "capcut_analog": "(none)",
    },
}


def plan_intensity(name: str, arousal: float):
    """Map section arousal (0..1) -> intensity level using the effect's research-derived ladder.
    Returns None if arousal is below the effect's gate (i.e. do NOT fire here -- restraint)."""
    rule = TRIGGER_RULES[name]
    if rule.get("do_not_place") or arousal < rule["arousal_gate"]:
        return None
    lvl = rule["intensity_at_arousal"][0][1]
    for thr, name_lvl in rule["intensity_at_arousal"]:
        if arousal >= thr:
            lvl = name_lvl
    return lvl


# ================================================================== DURATION + ENVELOPE MODEL
# From effect_codex.json "duration_model" (2026-07-03). KEY DATA FINDING: effects have TWO regimes,
# chosen by the MOMENT's character, NOT a fixed timer:
#   * SHORT STAB  -- an isolated individual HIT: quick attack, tiny hold, fast/instant off (~0.2-0.4s).
#   * SUSTAINED   -- THE impactful moment (a DROP that rings out, a climax, a held peak at top arousal):
#                    a LONGER effect that RIDES the passage with an intentional attack + hold + fall-off.
# Re-measured proof: rgb_split style-C is an oscillating PRISM that rides ~0.85s and PULSES on a drop
# at arousal 0.94, vs style-A which stabs ~0.13-0.30s on ordinary hits. shake/flash have NO sustained
# corpus instance -- they STAY short even on a drop (holding them would be off-style / un-codex).
DURATION_MODEL = {
    "rgb_split": {
        "stab":      dict(duration_s=0.35, attack_s=0.10, release_s=0.10, shape="stab_hold",
                          hold_pulses=0.0, intensity_bump=0),
        "sustained": dict(duration_s=1.00, attack_s=0.15, release_s=0.15, shape="pulse_hold",
                          hold_pulses=2.5, intensity_bump=+1),   # rides + pulses the drop
        "can_sustain": True,
    },
    "flash": {
        "stab":      dict(duration_s=0.60, attack_s=0.0, release_s=0.0, shape="strobe",
                          hold_pulses=0.0, intensity_bump=0),
        "can_sustain": False,   # codex: flash is always a short strobe burst; adding pulses = bigger count
    },
    "shake": {
        "stab":      dict(duration_s=0.45, attack_s=0.067, release_s=0.10, shape="tri",
                          hold_pulses=0.0, intensity_bump=0),
        "can_sustain": False,   # codex: NO sustained shake instance -- a held shake reads as a fault.
    },
    "light_leak": {
        "stab":      dict(duration_s=0.95, attack_s=0.20, release_s=0.20, shape="rise_hold_fall",
                          hold_pulses=0.0, intensity_bump=0),
        "sustained": dict(duration_s=1.60, attack_s=0.30, release_s=0.40, shape="rise_hold_fall",
                          hold_pulses=0.0, intensity_bump=0),   # a longer warm wash over a held entry
        "can_sustain": True,
    },
    "whip": {
        "stab":      dict(duration_s=0.55, attack_s=0.10, release_s=0.20, shape="tri_sharp",
                          hold_pulses=0.0, intensity_bump=0),
        "sustained": dict(duration_s=1.17, attack_s=0.20, release_s=1.00, shape="tri_sharp",
                          hold_pulses=0.0, intensity_bump=0),   # long pan/blur-out at a cut
        "can_sustain": True,   # but ONLY at a join (transition duration), never mid-shot
    },
    "blur_build": {
        "stab":      dict(duration_s=2.0, attack_s=2.0, release_s=0.0, shape="ramp_up",
                          hold_pulses=0.0, intensity_bump=0),
        "can_sustain": False,  # already a build; its "sustain" is the ramp itself
    },
}


def moment_is_sustained(moment: dict) -> bool:
    """Classify a musical moment as a SUSTAINED impactful passage (drop/climax/held peak) vs an
    isolated hit, from the events.py spine fields. A moment is 'sustained' when it sits ON a drop
    (or climax) at high arousal -- i.e. an impactful passage that RINGS OUT, not a quick single hit.
    moment keys (all optional): arousal, on_drop (bool), nearest_drop_s, is_climax (bool),
    sustain_s (how long energy stays high after the moment)."""
    if moment is None:
        return False
    if moment.get("is_climax"):
        return True
    ar = float(moment.get("arousal", 0.0))
    on_drop = bool(moment.get("on_drop")) or float(moment.get("nearest_drop_s", 9e9)) <= 0.25
    sustain_s = float(moment.get("sustain_s", 0.0))
    # codex: the one sustained rgb sat on a drop at arousal 0.94. Require a real drop AND top arousal,
    # or an explicitly long high-energy passage.
    return (on_drop and ar >= 0.85) or (sustain_s >= 1.0 and ar >= 0.8)


def choose_duration_envelope(name: str, arousal: float, moment: dict = None):
    """From the MOMENT's character, choose the effect's DURATION + ENVELOPE (attack/hold/release +
    optional pulsing) + a peak-intensity bump. Returns a dict the placement/renderer consume:
        {regime, duration_s, attack_s, hold_s, release_s, shape, hold_pulses, intensity_bump}
    A sustained/held impactful passage (a drop that rings out, a climax) gets the LONGER effect that
    RIDES it (attack + hold/pulse + fall-off); an isolated hit gets the short stab. Effects with no
    sustained corpus evidence (shake/flash/blur_build) ALWAYS return their short/native regime."""
    dm = DURATION_MODEL.get(name)
    if dm is None:                                   # unknown/experimental: no duration model
        return None
    want_sustain = dm.get("can_sustain") and moment_is_sustained(moment)
    spec = dict(dm["sustained"] if (want_sustain and "sustained" in dm) else dm["stab"])
    regime = "sustained" if (want_sustain and "sustained" in dm) else "stab"
    dur = spec["duration_s"]; a = spec["attack_s"]; r = spec["release_s"]
    return {
        "regime": regime,
        "duration_s": round(dur, 3),
        "attack_s": round(a, 3),
        "hold_s": round(max(0.0, dur - a - r), 3),
        "release_s": round(r, 3),
        "shape": spec["shape"],
        "hold_pulses": spec["hold_pulses"],
        "intensity_bump": spec["intensity_bump"],
    }


_LADDER_ORDER = ["subtle", "medium", "strong"]

def _bump_intensity(lvl, bump):
    """Nudge a ladder level by `bump` steps (clamped). Numeric intensities pass through unchanged."""
    if not isinstance(lvl, str) or lvl not in _LADDER_ORDER or not bump:
        return lvl
    i = min(len(_LADDER_ORDER) - 1, max(0, _LADDER_ORDER.index(lvl) + bump))
    return _LADDER_ORDER[i]


def make_placement(name: str, tl_start: float, arousal: float, label: str = "section",
                   moment: dict = None):
    """Produce an auditable placement dict for the editor/QA: effect + timing + intensity + REASON.
    Returns None when the effect should not fire (arousal below gate) -- research 'key moments only'.
    The QA feedback loop consumes/edits these (lower density, drop incongruent, soften intensity).

    DURATION + ENVELOPE are now chosen FROM the MOMENT (choose_duration_envelope): a sustained
    impactful passage (a drop/climax that rings out) gets a LONGER RIDING effect with an intentional
    attack + hold/pulse + fall-off; an isolated hit gets the short stab. Pass `moment` (spine fields:
    arousal, on_drop, nearest_drop_s, is_climax, sustain_s) to drive this; omit it for a plain stab."""
    rule = TRIGGER_RULES[name]
    lvl = plan_intensity(name, arousal)
    if lvl is None:
        return None
    de = choose_duration_envelope(name, arousal, moment)
    if de is None:                                   # no duration model -> fall back to rule defaults
        de = {"regime": "stab", "duration_s": rule["envelope_duration"],
              "attack_s": None, "hold_s": None, "release_s": None,
              "shape": rule["envelope"], "hold_pulses": 0.0, "intensity_bump": 0}
    lvl = _bump_intensity(lvl, de["intensity_bump"])
    regime_tag = "SUSTAINED (rides the passage)" if de["regime"] == "sustained" else "short stab (single hit)"
    return {
        "effect": name,
        "tl_start": round(float(tl_start), 3),
        "intensity": lvl,
        "regime": de["regime"],
        "envelope": de["shape"],
        "envelope_duration": de["duration_s"],
        "attack_s": de["attack_s"],
        "hold_s": de["hold_s"],
        "release_s": de["release_s"],
        "hold_pulses": de["hold_pulses"],
        "event": rule["event"],
        "experimental": bool(rule.get("experimental", False)),
        "min_spacing_s": rule["min_spacing_s"],
        "reason": rule["reason_tmpl"].format(label=label, a=arousal) + " | " + regime_tag,
    }


def enforce_density(placements, global_min_spacing=1.2):
    """Drop placements that violate per-effect min spacing or the global tasteful floor.
    (Research: over-syncing looks amateurish.) QA can call with a larger floor to thin further."""
    placements = sorted(placements, key=lambda p: p["tl_start"])
    kept, last_by, count_by = [], {}, {}
    last_any = -1e9
    for p in placements:
        cap = TRIGGER_RULES.get(p["effect"], {}).get("max_per_edit")
        if cap is not None and count_by.get(p["effect"], 0) >= cap:
            continue                          # corpus-frequency cap (codex rebuild)
        gap_ok = (p["tl_start"] - last_any) >= global_min_spacing
        eff_gap_ok = (p["tl_start"] - last_by.get(p["effect"], -1e9)) >= p["min_spacing_s"]
        if gap_ok and eff_gap_ok:
            kept.append(p)
            last_any = p["tl_start"]
            last_by[p["effect"]] = p["tl_start"]
            count_by[p["effect"]] = count_by.get(p["effect"], 0) + 1   # enforce corpus max_per_edit
    return kept


# ================================================================== EFFECT FILTER BUILDERS
# Each builder returns (graph, kind) where kind is:
#   'vf'    -> a single -vf chain (prepend your own scale/fps or let render_effect add them)
#   'fc'    -> a full -filter_complex ending in [vout] (needs -map [vout])
# All builders take a scratch dir for sendcmd scripts.

def _base(w, h):
    return "scale=%d:%d,fps=%d" % (w, h, FPS)


def _px(v, w):
    """Scale a px value calibrated at REF_W (720, the reel corpus width) to the working width."""
    return v * float(w) / REF_W


def rgb_split_vf(intensity="medium", envelope_duration=0.42, shape="stab_hold",
                 w=PREVIEW_W, h=PREVIEW_H, scratch=None):
    """FITTED asymmetric chromatic stab (codex 2026-07-03): RED shifts left by the peak value while
    BLUE shifts right at 0.35x, ~2-frame attack, HOLD, then INSTANT off -- reproduces the measured
    R-B offset series of DXUDMp@23.42 frame-for-frame. Peak is calibrated @720w and scales with w."""
    peak = _px(level_value("rgb_split", intensity), w)
    scratch = scratch or tempfile.gettempdir()
    steps = max(12, int(envelope_duration * 30))
    pairs = [(0.0, "rgbashift rh 0"), (0.0, "rgbashift bh 0")]
    for i in range(steps + 1):
        t = envelope_duration * i / steps
        e = envelope(shape, i / steps)
        pairs += [(t, "rgbashift rh %d" % (-int(round(peak * e)))),
                  (t, "rgbashift bh %d" % (int(round(peak * 0.35 * e))))]
    pairs += [(envelope_duration + 0.01, "rgbashift rh 0"),      # measured INSTANT off
              (envelope_duration + 0.01, "rgbashift bh 0")]
    scr = _sendcmd_file(pairs, os.path.join(scratch, "el_rgb.cmd"))
    return "%s,sendcmd=f='%s',rgbashift=rh=0:bh=0" % (_base(w, h), scr), "vf"


def whip_vf(intensity="medium", envelope_duration=0.55, shape="tri_sharp",
            w=PREVIEW_W, h=PREVIEW_H, scratch=None):
    """Horizontal motion-blur smear. FITTED: peak 260px@720w -> min-sharpness ratio 0.26 (real 0.22).
    PARTIAL match: real whips are camera pans bridging two shots; ONLY place this at a join
    (SeamlessTransition style='whip'), never as a mid-shot overlay."""
    peak = _px(level_value("whip", intensity), w)
    scratch = scratch or tempfile.gettempdir()
    steps = max(16, int(envelope_duration * 40))
    pairs = []
    for i in range(steps + 1):
        t = envelope_duration * i / steps
        e = envelope(shape, i / steps)
        sx = max(1, int(round(peak * e)))
        pairs += [(t, "avgblur sizeX %d" % sx), (t, "avgblur sizeY 1")]
    pairs.append((envelope_duration + 0.01, "avgblur sizeX 1"))
    scr = _sendcmd_file(pairs, os.path.join(scratch, "el_whip.cmd"))
    return "%s,sendcmd=f='%s',avgblur=sizeX=1:sizeY=1" % (_base(w, h), scr), "vf"


def flash_vf(intensity="medium", envelope_duration=0.6, shape="strobe",
             w=PREVIEW_W, h=PREVIEW_H, scratch=None, period=0.2):
    """FITTED white strobe train (codex: DT-b@33.8-34.5): 5Hz pulses, each a per-frame triangle that
    peaks ONE frame at full white (brightness 1.0, saturation ~0.15) with washed shoulders.
    intensity = number of pulses (1-3). Matches the measured luma series 79..252 @5Hz."""
    n_pulses = int(level_value("flash", intensity))
    scratch = scratch or tempfile.gettempdir()
    fpp = max(2, int(round(period * FPS)))
    total = n_pulses * period
    pairs = [(0.0, "eq brightness 0"), (0.0, "eq saturation 1")]
    nfr = int(total * FPS)
    for i in range(nfr + 1):
        t = i / float(FPS)
        ph = (i % fpp) / float(fpp)
        e = (1.0 - abs(2.0 * ph - 1.0)) ** 1.8      # sharp spike; shoulders match real levels
        pairs += [(t, "eq brightness %.3f" % e), (t, "eq saturation %.3f" % (1 - 0.85 * e))]
    pairs += [(total + 0.01, "eq brightness 0"), (total + 0.01, "eq saturation 1")]
    scr = _sendcmd_file(pairs, os.path.join(scratch, "el_flash.cmd"))
    return "%s,sendcmd=f='%s',eq=brightness=0:saturation=1" % (_base(w, h), scr), "vf"


def blur_build_vf(intensity="medium", envelope_duration=2.0, shape="ramp_up",
                  w=PREVIEW_W, h=PREVIEW_H, scratch=None):
    """FITTED blur build (codex: DYw@62.5-64.3): gaussian blur ramps 0 -> sigma over ~2s, a defocus
    build that must RESOLVE INTO a transition (the corpus instance ends in a whip cut)."""
    sig = _px(level_value("blur_build", intensity), w)
    scratch = scratch or tempfile.gettempdir()
    steps = max(20, int(envelope_duration * 10))
    pairs = [(0.0, "gblur sigma 0.01")]
    for i in range(steps + 1):
        t = envelope_duration * i / steps
        e = envelope(shape, i / steps)
        pairs.append((t, "gblur sigma %.2f" % max(0.01, sig * e)))
    scr = _sendcmd_file(pairs, os.path.join(scratch, "el_blurbuild.cmd"))
    return "%s,sendcmd=f='%s',gblur=sigma=0.01" % (_base(w, h), scr), "vf"


def radial_zoom_fc(intensity="medium", envelope_duration=1.2, shape="in_out",
                   w=PREVIEW_W, h=PREVIEW_H, scratch=None, layers=6):
    """Radial / zoom blur: composite progressively zoomed low-opacity copies so the CENTRE stays
    sharp and EDGES streak radially outward. Ramp the blend against the clean base over time via a
    time-varying overlay alpha (sendcmd on the top composite's colorchannelmixer... not exposed).
    Robust approach: build full-strength radial [rb], then crossfade base<->rb with a time alpha using
    'blend=all_expr' driven by T so the radial slides in and out."""
    step = level_value("radial_zoom", intensity)
    scratch = scratch or tempfile.gettempdir()
    D = envelope_duration
    # envelope as a T-expression (seconds). in_out (hann) -> matches shape='in_out'.
    if shape == "in_out":
        env_expr = "(0.5-0.5*cos(2*PI*T/%f))" % D
    elif shape == "tri":
        env_expr = "(1-abs(2*T/%f-1))" % D
    else:
        env_expr = "(1-abs(2*T/%f-1))" % D
    parts = ["[0:v]%s,split=%d%s" % (_base(w, h), layers + 2,
             "".join("[s%d]" % i for i in range(layers + 2)))]
    prev = "s1"                                   # s0 kept as the clean base for the final blend
    for k in range(1, layers + 1):
        z = 1 + step * k
        sw, sh = int(w * z), int(h * z)
        parts.append("[s%d]scale=%d:%d,crop=%d:%d:(iw-%d)/2:(ih-%d)/2,format=yuva420p,"
                     "colorchannelmixer=aa=%.3f[z%d]" % (k + 1, sw, sh, w, h, w, h, 1.0 / (k + 1), k))
        parts.append("[%s][z%d]overlay[o%d]" % (prev, k, k))
        prev = "o%d" % k
    # time-ramped blend: A=clean s0, B=radial prev; out = A*(1-e) + B*e
    parts.append("[s0][%s]blend=all_expr='A*(1-%s)+B*%s'[vout]" % (prev, env_expr, env_expr))
    return ";".join(parts), "fc"


def shake_vf(intensity="medium", envelope_duration=0.45, shape="tri",
             w=PREVIEW_W, h=PREVIEW_H, scratch=None, overscan=1.06, fx_hz=3.0, fy_hz=3.9):
    """FITTED camera bump (codex 2026-07-03): ~12px @720w, ~3Hz, 0.45s -> 1-2 cycles, a single
    impact bump, NOT a sustained buzz (the old 11-13Hz was wrong). intensity = px @720w.
    PARTIAL match: real bumps are in-camera with intra-frame motion blur; use rarely or drop."""
    amp = _px(level_value("shake", intensity), w)
    sw, sh = int(w * overscan), int(h * overscan)
    D = envelope_duration
    env = "(1-abs(2*t/%f-1))" % D
    xexpr = "(iw-%d)/2 + %.1f*(%s)*sin(2*PI*%.2f*t)" % (w, amp, env, fx_hz)
    yexpr = "(ih-%d)/2 + %.1f*(%s)*cos(2*PI*%.2f*t + 0.7)" % (h, amp * 0.7, env, fy_hz)
    return ("scale=%d:%d,fps=%d,crop=%d:%d:x='%s':y='%s'"
            % (sw, sh, FPS, w, h, xexpr, yexpr)), "vf"


def light_leak_vf(intensity="medium", envelope_duration=0.95, shape="rise_hold_fall",
                  w=PREVIEW_W, h=PREVIEW_H, scratch=None, sat_peak=1.15, warm=True):
    """FITTED warm leak wash (codex: DT-b@23.0): rise 0.2s / hold ~0.55s / fall 0.2s; at 'strong' the
    luma delta is +115 vs the real +109 and red-ratio +0.50 vs real +0.53. Warmth comes from a
    time-ramped gamma_r push (a static colorbalance was far too weak to hit the measured red)."""
    b_peak = level_value("light_leak", intensity)
    scratch = scratch or tempfile.gettempdir()
    steps = max(16, int(envelope_duration * 30))
    pairs = [(0.0, "eq brightness 0"), (0.0, "eq saturation 1"),
             (0.0, "eq gamma_r 1"), (0.0, "eq gamma_b 1")]
    gr_peak = 2.3 * (b_peak / 0.36)              # warm push scales with the level
    gb_dip = 0.5 * (b_peak / 0.36)
    for i in range(steps + 1):
        t = envelope_duration * i / steps
        e = max(0.0, min(1.0, envelope(shape, i / steps)))
        pairs += [(t, "eq brightness %.4f" % (b_peak * e)),
                  (t, "eq saturation %.4f" % (1 + (sat_peak - 1) * e))]
        if warm:
            pairs += [(t, "eq gamma_r %.4f" % (1 + gr_peak * e)),
                      (t, "eq gamma_b %.4f" % (1 - gb_dip * e))]
    pairs += [(envelope_duration + 0.01, "eq brightness 0"),
              (envelope_duration + 0.01, "eq saturation 1"),
              (envelope_duration + 0.01, "eq gamma_r 1"),
              (envelope_duration + 0.01, "eq gamma_b 1")]
    scr = _sendcmd_file(pairs, os.path.join(scratch, "el_leak.cmd"))
    return ("%s,sendcmd=f='%s',eq=brightness=0:saturation=1:gamma_r=1:gamma_b=1"
            % (_base(w, h), scr)), "vf"


def speed_ramp_vf(intensity="medium", envelope_duration=2.0, shape="hold",
                  w=PREVIEW_W, h=PREVIEW_H, scratch=None, end_speed=2.0):
    """Speed ramp slow->fast via a quadratic PTS remap (dout/din grows linearly from start->end speed).
    NOTE: needs `envelope_duration` seconds of SOURCE; the editor must feed a long-enough clip.
    intensity = starting speed (0.75 subtle .. 0.35 strong); accelerates to end_speed by clip end."""
    s0 = level_value("speed_ramp", intensity)
    S = envelope_duration
    # out(T) = s0*T + ((end-s0)/(2S))*T^2  ->  strictly increasing accel
    expr = "(%f*T + ((%f-%f)/(2*%f))*T*T)/TB" % (s0, end_speed, s0, S)
    return "%s,setpts=%s" % (_base(w, h), expr), "vf"


# ---- experimental ----
def glitch_vf(intensity="medium", envelope_duration=0.8, shape="tri",
              w=PREVIEW_W, h=PREVIEW_H, scratch=None, seed=3):
    """EXPERIMENTAL variety accent: stuttering RGB tear + noise flicker, ramped. Use RARELY (biggest
    hits only). Congruent with the reels' chromatic vocabulary but pushed to a digital-glitch look."""
    peak = level_value("glitch", intensity)
    scratch = scratch or tempfile.gettempdir()
    rnd = random.Random(seed)
    steps = max(24, int(envelope_duration * 40))
    pairs = []
    for i in range(steps + 1):
        t = envelope_duration * i / steps
        e = envelope(shape, i / steps)
        on = 0 if (i % 3 == 0) else 1                       # stutter with gaps
        rh = int(round(peak * e * on * rnd.choice([0.4, 1.0, 0.7])))
        bh = -int(round((peak * 0.8) * e * on))
        gv = int(round((peak * 0.35) * e * on))
        pairs += [(t, "rgbashift rh %d" % rh), (t, "rgbashift bh %d" % bh),
                  (t, "rgbashift gv %d" % gv)]
    scr = _sendcmd_file(pairs, os.path.join(scratch, "el_glitch.cmd"))
    return ("%s,sendcmd=f='%s',rgbashift=rh=0:bh=0:gv=0,noise=alls=8:allf=t"
            % (_base(w, h), scr), "vf")


def light_streak_fc(intensity="medium", envelope_duration=1.4, shape="in_out",
                    w=PREVIEW_W, h=PREVIEW_H, scratch=None):
    """EXPERIMENTAL: horizontal light streaks from bright highlights (prism-ish bloom), lighten-blended.
    WEAK on bright daylight footage (highlights everywhere); best on dark/night footage. Use rarely."""
    op = level_value("light_streak", intensity)
    return ("[0:v]%s,split[base][hp];"
            "[hp]lutyuv=y='if(gt(val,238),val,16)',avgblur=sizeX=55:sizeY=2,hue=h=15:s=1.5,"
            "format=yuva420p,colorchannelmixer=aa=%.2f[str];"
            "[base][str]blend=all_mode=lighten[vout]" % (_base(w, h), op), "fc")


# registry: name -> (builder, kind-hint)  -- kind resolved from builder return, this is for listing.
EFFECTS = {
    # codex-validated vocabulary (2026-07-03)
    "rgb_split":    rgb_split_vf,
    "whip":         whip_vf,
    "flash":        flash_vf,
    "light_leak":   light_leak_vf,
    "shake":        shake_vf,
    "blur_build":   blur_build_vf,
    # kept for API compat only; TRIGGER_RULES forbids placement (no corpus evidence / unmatched)
    "radial_zoom":  radial_zoom_fc,
    "speed_ramp":   speed_ramp_vf,
    "glitch":       glitch_vf,
    "light_streak": light_streak_fc,
}


def build_effect_vf(name, intensity="medium", envelope_duration=None, w=PREVIEW_W, h=PREVIEW_H,
                    scratch=None, **kw):
    """Return (graph, kind) for effect `name`. kind in {'vf','fc'}. If envelope_duration is None,
    use the research default from TRIGGER_RULES. This is the editor's main entry point."""
    if name not in EFFECTS:
        raise KeyError("unknown effect %r; known: %s" % (name, sorted(EFFECTS)))
    if envelope_duration is None:
        envelope_duration = TRIGGER_RULES[name]["envelope_duration"]
    return EFFECTS[name](intensity=intensity, envelope_duration=envelope_duration,
                         w=w, h=h, scratch=scratch, **kw)


# ================================================================== SEAMLESS TRANSITION (#7, signature)
class SeamlessTransition:
    """The user's signature: a transition that blends BOTH video AND audio so cuts never feel sharp.
        VIDEO: xfade over an overlap window (cross-dissolve; optional blur/whip flavour on the seam).
        AUDIO: acrossfade over the SAME window (outgoing fades out AS incoming fades in).
    Styles:
        'dissolve' -> straight cross-dissolve (both shots superimposed through the seam). [SOLID]
        'blur'     -> dissolve + a soft blur bloom on the seam (gblur pulsed under the xfade).
        'whip'     -> horizontal smear on A's tail and B's head, then a fast dissolve (whip pan). [SOLID]
    overlap ~0.5-0.8s reads well at 130bpm. Verified: video blends (no dip-to-black) + audio RMS
    hands off smoothly through the crossover (never drops to zero)."""

    def __init__(self, ffmpeg=FFMPEG_DEFAULT, w=PREVIEW_W, h=PREVIEW_H, fps=FPS):
        self.ffmpeg, self.w, self.h, self.fps = ffmpeg, w, h, fps

    def filtergraph(self, a_len, b_len, overlap=0.7, style="dissolve", scratch=None):
        """Return (filter_complex_str, out_video_label, out_audio_label) for two -ss inputs [0],[1].
        Caller supplies A as input 0 (already -ss/-t to a_len) and B as input 1 (-ss/-t to b_len)."""
        w, h, fps = self.w, self.h, self.fps
        off = max(0.0, a_len - overlap)
        base_a = "[0:v]scale=%d:%d,fps=%d,setpts=PTS-STARTPTS" % (w, h, fps)
        base_b = "[1:v]scale=%d:%d,fps=%d,setpts=PTS-STARTPTS" % (w, h, fps)
        if style == "whip":
            # smear A's tail (ramp_up) and B's head (ramp_down) horizontally, then a fast dissolve.
            scratch = scratch or tempfile.gettempdir()
            pk = 70
            # A tail smear: strengthen over the last `overlap` seconds (ramp_up within [off, a_len])
            pa = []
            for i in range(21):
                t = a_len * i / 20
                if t < off:
                    e = 0.0
                else:
                    e = envelope("ramp_up", (t - off) / max(overlap, 1e-3))
                pa.append((t, "avgblur sizeX %d" % max(1, int(pk * e))))
                pa.append((t, "avgblur sizeY 1"))
            sca = _sendcmd_file(pa, os.path.join(scratch, "el_whipA.cmd"))
            pb = []
            for i in range(21):
                t = b_len * i / 20
                e = envelope("ramp_down", t / max(overlap, 1e-3)) if t < overlap else 0.0
                pb.append((t, "avgblur sizeX %d" % max(1, int(pk * e))))
                pb.append((t, "avgblur sizeY 1"))
            scb = _sendcmd_file(pb, os.path.join(scratch, "el_whipB.cmd"))
            va = "%s,sendcmd=f='%s',avgblur=sizeX=1:sizeY=1[va]" % (base_a, sca)
            vb = "%s,sendcmd=f='%s',avgblur=sizeX=1:sizeY=1[vb]" % (base_b, scb)
            xf = "[va][vb]xfade=transition=fade:duration=%.3f:offset=%.3f[vout]" % (min(overlap, 0.35), off)
            vparts = [va, vb, xf]
        elif style == "blur":
            va = "%s[va]" % base_a
            vb = "%s[vb]" % base_b
            # smoothing dissolve; a gentle overall softening reads as a blur bloom on the seam
            xf = ("[va][vb]xfade=transition=fadegrays:duration=%.3f:offset=%.3f[vout]"
                  % (overlap, off))
            vparts = [va, vb, xf]
        else:  # dissolve (default)
            va = "%s[va]" % base_a
            vb = "%s[vb]" % base_b
            xf = "[va][vb]xfade=transition=fade:duration=%.3f:offset=%.3f[vout]" % (overlap, off)
            vparts = [va, vb, xf]
        # AUDIO acrossfade (equal-power tri) over the same overlap
        aparts = [
            "[0:a]aresample=44100,asetpts=PTS-STARTPTS[aa]",
            "[1:a]aresample=44100,asetpts=PTS-STARTPTS[ab]",
            "[aa][ab]acrossfade=d=%.3f:c1=tri:c2=tri[aout]" % overlap,
        ]
        return ";".join(vparts + aparts), "[vout]", "[aout]"

    def render(self, src_a, ta, src_b, tb, out, a_len=2.0, b_len=2.0, overlap=0.7,
               style="dissolve", scratch=None, preset="ultrafast"):
        """Render a two-clip seamless transition to `out`. src_a/src_b may be the same file."""
        fc, vlab, alab = self.filtergraph(a_len, b_len, overlap, style, scratch)
        cmd = [self.ffmpeg, "-v", "error",
               "-ss", str(ta), "-t", str(a_len), "-i", src_a,
               "-ss", str(tb), "-t", str(b_len), "-i", src_b,
               "-filter_complex", fc, "-map", vlab, "-map", alab,
               "-c:v", "libx264", "-preset", preset, "-c:a", "aac", "-y", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("transition render failed: " + r.stderr[-800:])
        return out


# ================================================================== RENDER HELPERS (CLI + QA)
def render_effect(name, src, ts, out, intensity="medium", envelope_duration=None, dur=None,
                  ffmpeg=FFMPEG_DEFAULT, w=PREVIEW_W, h=PREVIEW_H, scratch=None, preset="ultrafast",
                  keep_audio=False, **kw):
    """Render effect `name` on `src` at `ts` seconds to `out`. Duration defaults to the effect's
    envelope_duration (+ small tail). Returns out path. Used by the CLI and the QA feedback loop."""
    scratch = scratch or os.path.dirname(out) or tempfile.gettempdir()
    if envelope_duration is None:
        envelope_duration = TRIGGER_RULES[name]["envelope_duration"]
    graph, kind = build_effect_vf(name, intensity=intensity, envelope_duration=envelope_duration,
                                  w=w, h=h, scratch=scratch, **kw)
    if dur is None:
        dur = envelope_duration + (0.6 if name != "speed_ramp" else 0.0)
        if name == "speed_ramp":
            dur = envelope_duration  # needs exactly this much SOURCE
    cmd = [ffmpeg, "-v", "error", "-ss", str(ts), "-i", src, "-t", "%.3f" % dur]
    if not keep_audio:
        cmd += ["-an"]
    if kind == "vf":
        cmd += ["-vf", graph]
    else:
        cmd += ["-filter_complex", graph, "-map", "[vout]"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-y", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("render_effect(%s) failed: %s" % (name, r.stderr[-800:]))
    return out


def tile_montage(clip, out, ffmpeg=FFMPEG_DEFAULT, cols=10, fps=12, tw=140):
    th = int(tw * 16 / 9)
    cmd = [ffmpeg, "-v", "error", "-i", clip, "-vf",
           "fps=%d,scale=%d:%d,tile=%dx1" % (fps, tw, th, cols), "-frames:v", "1", "-y", out]
    subprocess.run(cmd, capture_output=True, text=True)
    return out


# ================================================================== CLI
def _main(argv):
    ap = argparse.ArgumentParser(description="effects_lab -- code-rendered ffmpeg effects for the editor")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list", help="list effects + trigger rules")

    d = sub.add_parser("demo", help="render one effect on a clip at a timestamp")
    d.add_argument("effect"); d.add_argument("src"); d.add_argument("ts", type=float)
    d.add_argument("--intensity", default="medium"); d.add_argument("--dur", type=float, default=None)
    d.add_argument("--out", default=None); d.add_argument("--tile", action="store_true")

    l = sub.add_parser("ladder", help="render subtle/medium/strong of an effect")
    l.add_argument("effect"); l.add_argument("src"); l.add_argument("ts", type=float)
    l.add_argument("--outdir", default=".")

    t = sub.add_parser("transition", help="render a seamless transition between two timestamps")
    t.add_argument("src"); t.add_argument("ta", type=float); t.add_argument("tb", type=float)
    t.add_argument("--style", default="dissolve", choices=["dissolve", "blur", "whip"])
    t.add_argument("--overlap", type=float, default=0.7); t.add_argument("--out", default="transition.mp4")

    args = ap.parse_args(argv)
    if args.cmd == "list" or not args.cmd:
        print("EFFECTS (code-rendered ffmpeg):")
        for n in EFFECTS:
            r = TRIGGER_RULES[n]
            tag = " [EXPERIMENTAL]" if r.get("experimental") else ""
            lad = INTENSITY[n]
            print("  %-13s%s event=%-15s gate=%.2f  ladder(%s): %s/%s/%s"
                  % (n, tag, r["event"], r["arousal_gate"], lad["unit"],
                     lad["subtle"], lad["medium"], lad["strong"]))
        print("\nSeamlessTransition styles: dissolve | blur | whip  (video xfade + audio acrossfade)")
        return
    if args.cmd == "demo":
        out = args.out or "demo_%s.mp4" % args.effect
        render_effect(args.effect, args.src, args.ts, out, intensity=args.intensity, dur=args.dur)
        print("wrote", out)
        if args.tile:
            m = out.rsplit(".", 1)[0] + "_tile.png"
            tile_montage(out, m); print("wrote", m)
        return
    if args.cmd == "ladder":
        for lvl in ("subtle", "medium", "strong"):
            out = os.path.join(args.outdir, "ladder_%s_%s.mp4" % (args.effect, lvl))
            render_effect(args.effect, args.src, args.ts, out, intensity=lvl)
            m = out.rsplit(".", 1)[0] + "_tile.png"; tile_montage(out, m)
            print("wrote", out, "+", m)
        return
    if args.cmd == "transition":
        st = SeamlessTransition()
        st.render(args.src, args.ta, args.src, args.tb, args.out,
                  overlap=args.overlap, style=args.style)
        print("wrote", args.out)
        return


if __name__ == "__main__":
    _main(sys.argv[1:])
