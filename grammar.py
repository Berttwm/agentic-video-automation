# -*- coding: utf-8 -*-
"""grammar.py -- the SINGLE SHARED LOADER for the research-derived STYLE GRAMMAR.

Both assemble_song.py (arrangement pacing) and infer_effects.py (effect placement) read the measured
grammar THROUGH THIS MODULE, so "apply the research" means exactly the same thing everywhere. Nothing
here recomputes the grammar -- it only READS the files the reel-research agent already produced:

    style_model.json   -- measured editing grammar: cadence (cuts/min, tempo), cut/effect grammar,
                          best-practices, automation_framework (tiered quantization, arousal-density).
    effect_codex.json  -- 51 measured effect INSTANCES with envelopes/durations + per-type corpus
                          frequency (the palette + how often each type appears + its measured timing).
    ig_research.json   -- `inspiration` (trending @firsttoeleven: ~12 cuts/min, tight 29s) available as
                          a PUNCHIER option behind a flag, never the default.

The loader exposes the two things the editor needs to be grammar-LED:

  1. pacing()   -> arrangement pacing knobs for assemble_song, derived from style_model's CADENCE +
                  best-practices toward P&P's MEASURED style (long takes / whole sections). The
                  trending inspiration cadence is available (pacing(punchy=True)) but OFF by default.

  2. effect_budget() -> per-effect placement budget for infer_effects, derived from the codex's
                  MEASURED corpus frequency (instances / reels): which effect types are placeable, on
                  which event kind, and the measured max-per-edit density (LOW). Forbidden / venue-only
                  / zero-corpus types are excluded here so they can never be placed.

Config-driven paths (paths.py); no band handle / device path hardcoded. Dependency-free (json+os).
"""
import os, json

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    p = os.path.join(_HERE, name)
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---- raw grammar files (read once, cached) -----------------------------------------------------
_STYLE = None
_CODEX = None
_IG = None


def style_model():
    global _STYLE
    if _STYLE is None:
        _STYLE = _load("style_model.json")
    return _STYLE


def effect_codex():
    global _CODEX
    if _CODEX is None:
        _CODEX = _load("effect_codex.json")
    return _CODEX


def ig_research():
    global _IG
    if _IG is None:
        try:
            _IG = _load("ig_research.json")
        except (OSError, ValueError):
            _IG = {}
    return _IG


# ================================================================== ARRANGEMENT PACING (assemble_song)
# P&P's real style, MEASURED in style_model.cadence: ~2 cuts/min, i.e. LONG single-take sections that
# breathe (the opposite of the trending 12-cuts/min inspiration). The pacing knobs translate that
# measured cadence + the best-practice "let phrases finish before cutting / whole sections" into the
# arrangement's structural choices: how many section-cuts to allow, how often to switch the drummer cam.
def pacing(punchy=False):
    """Return arrangement pacing knobs derived from the MEASURED grammar.

    Default (punchy=False) = P&P's own measured long-take style (style_model.cadence ~2 cuts/min):
        whole sections, few section-cuts, ONE interior drummer-cam switch.
    punchy=True = the trending @firsttoeleven inspiration cadence (~12 cuts/min) as an AVAILABLE
        option (more section-cuts, more angle switches). NEVER the default -- the user's whole-sections
        taste lock wins; this flag exists only so the punchier cadence is reachable on request.
    """
    sm = style_model()
    cad = sm.get("cadence", {})
    cuts_per_min = float(cad.get("cuts_per_min_median", 2.0))     # P&P measured: ~2/min (long takes)
    tempo = float(cad.get("tempo_median", 130.0))

    knobs = {
        "source": "style_model.cadence (P&P measured) + best_practices (let phrases finish, whole sections)",
        "cuts_per_min": cuts_per_min,
        "tempo_median_bpm": tempo,
        # a ~60s edit at ~2 cuts/min => ~2 section boundaries; keep angle switches to ONE interior
        # section (measured restraint). These are CAPS layered on top of the v3 selection logic --
        # the selection still picks vocal-build -> complete chorus -> whole solo; pacing bounds how
        # many cuts/switches that shape is allowed to carry so it stays long-take.
        "max_section_cuts_per_min": cuts_per_min,                # long takes: don't exceed measured cadence
        "drummer_switches": 1,                                   # ONE interior switch (visual variety only)
        "whole_sections": True,                                  # never truncate a section (taste lock)
        "condense": False,                                       # do NOT over-condense (taste lock)
        "punchy": False,
    }
    if punchy:
        insp = (ig_research().get("inspiration") or [{}])
        insp = insp[0] if isinstance(insp, list) and insp else {}
        # the inspiration analysis reports ~12.4 cuts/min; expose it but do NOT let it truncate sections.
        knobs.update({
            "source": "ig_research.inspiration (@%s, punchier option -- OFF by default)"
                      % insp.get("account", "firsttoeleven").lstrip("@"),
            "cuts_per_min": 12.0,
            "max_section_cuts_per_min": 12.0,
            "drummer_switches": 2,
            "punchy": True,
        })
    return knobs


# ================================================================== EFFECT BUDGET (infer_effects)
# The codex's MEASURED corpus frequency is the density authority. Reels average ~90s and one edit is
# comparable, so instances/ reels ~ how many of each type belong in a single edit. We translate that
# into a per-effect budget the placer consumes: placeable types only (venue-lighting + zero-corpus +
# forbidden are dropped here), each tagged with its measured event kind + max-per-edit + which duration
# regimes the codex proved for it. The intensity ladder + envelopes still come from effects_lab
# (TRIGGER_RULES / DURATION_MODEL), which were themselves fitted to this same codex -- so the budget
# says WHICH/HOW-MANY/ON-WHAT, and effects_lab says HOW-STRONG/HOW-LONG. One codex, read the same way.

# effect types that are venue-lighting or have ZERO corpus evidence -> never budgeted for placement.
_NON_PLACEABLE = {"radial_zoom", "speed_ramp", "glitch", "light_streak"}

# codex corpus_frequency effect-type name -> the effects_lab/placer effect name (they mostly match;
# flash_strobe is the codex label for the effects_lab 'flash').
_CODEX_NAME = {"flash_strobe": "flash"}


def effect_budget():
    """Per-effect placement budget derived from effect_codex.corpus_frequency (MEASURED).

    Returns dict effect_name -> {
        instances, reels, per_reel, max_per_edit, event, do_not_place, is_transition, note
    } for the PLACEABLE palette (venue-lighting / zero-corpus excluded). `event` is the musical event
    the codex places this type on ('hit' / 'section_entry' / 'section_change' / 'build_to_drop').
    `is_transition` marks whip (belongs to CUTS -> SeamlessTransition seam, not an overlay).
    max_per_edit = ceil(instances / reels) clamped to the codex placement caps -- the LOW measured
    density. This is the single source infer_effects reads to decide which accents to place + how many.
    """
    codex = effect_codex()
    cf = codex.get("corpus_frequency", {})

    # measured event kind per type (from the codex codex_placement / duration_model context) --
    # this mirrors effects_lab.TRIGGER_RULES[name]["event"] (same codex, executable form there).
    event_of = {
        "rgb_split": "hit",             # every instance <=0.2s from a flux hit
        "flash": "hit",                 # strobe burst rides the biggest vocal/instrument hit
        "light_leak": "section_entry",  # the one instance sits at a section entry
        "shake": "hit",                 # camera bump on a strong hit (likely in-camera; QA may drop)
        "blur_build": "build_to_drop",  # defocus build that resolves INTO a transition
        "whip": "section_change",       # the standard cut seam (a TRANSITION, not an overlay)
    }
    # placement caps proven by the codex (never exceed corpus max-per-reel).
    cap_of = {
        "rgb_split": 2,   # 6 inst / 4 reels, never >2 per reel
        "flash": 1,       # 1 reel only; max 1 burst per edit
        "light_leak": 1,  # 1 instance in 8 reels
        "shake": 1,       # likely in-camera; at most 1
        "blur_build": 1,  # 1 instance in 8 reels; only before a transition
        "whip": 3,        # ~2.5/reel, the standard seam (goes to transitions)
    }
    budget = {}
    for codex_name, rec in cf.items():
        if not isinstance(rec, dict) or "instances" not in rec:
            continue
        if codex_name == "excluded_as_venue_lighting":
            continue
        name = _CODEX_NAME.get(codex_name, codex_name)
        if name in _NON_PLACEABLE:
            continue
        inst = int(rec.get("instances", 0) or 0)
        reels = int(rec.get("reels", 0) or 0)
        if inst <= 0:                                    # zero-corpus -> not placeable
            continue
        per_reel = (inst / reels) if reels else float(inst)
        # measured max-per-edit = the codex cap (clamped to at least the rounded per-reel rate).
        import math
        measured_cap = max(1, int(math.ceil(per_reel)))
        cap = min(cap_of.get(name, measured_cap), measured_cap) if name in cap_of else measured_cap
        # if the codex cap is explicitly larger than the per-reel round (whip), trust the codex cap.
        cap = cap_of.get(name, cap)
        budget[name] = {
            "instances": inst,
            "reels": reels,
            "per_reel": round(per_reel, 3),
            "max_per_edit": int(cap),
            "event": event_of.get(name, "hit"),
            "do_not_place": False,
            "is_transition": (name == "whip"),
            "note": rec.get("note", ""),
        }
    return budget


# DEFAULT-PLACEABLE overlays vs codex types held OUT of the default palette (they remain in the codex
# budget for audit, but are not placed by default because they violate the SUBTLE / whole-sections
# taste locks even though the corpus contains them):
#   flash      -- a white STROBE; appears in only 1/8 reels ('zombie' hook), loud + rare (codex: "max 1
#                 burst per edit"). Reads as un-subtle -> held out of the default tasteful palette.
#   shake      -- 3 instances but all read as IN-CAMERA bumps (codex: "QA should default to dropping").
#   blur_build -- must resolve INTO a transition, never free-standing -> the renderer owns it at a seam.
# The default tasteful palette is therefore rgb_split (the standard chromatic accent, attested across
# 4/8 reels) + light_leak (the one measured section-entry wash) -- both SUBTLE-friendly.
_DEFAULT_HELD_OUT = {"flash", "shake", "blur_build"}


def placeable_overlay_effects(include_held_out=False):
    """The DEFAULT codex overlay palette the placer draws from (excludes whip -- a transition seam --
    and the held-out loud/uncertain/context-locked types flash/shake/blur_build). Ordered so the most
    broadly-attested, tasteful accents lead: rgb_split (4 reels) first, then light_leak (the entry wash).
    Every type returned has real corpus instances; ordering decides which fires first under the budget.
    Pass include_held_out=True to get the full codex overlay set (audit / a punchier explicit request)."""
    b = effect_budget()
    overlays = [(n, r) for n, r in b.items()
                if not r["is_transition"] and (include_held_out or n not in _DEFAULT_HELD_OUT)]
    overlays.sort(key=lambda kv: (-kv[1]["reels"],       # breadth of corpus support first
                                  -kv[1]["per_reel"]))
    return overlays


if __name__ == "__main__":
    # quick audit: print the grammar the editor will read.
    print("PACING (default, P&P long-take):")
    for k, v in pacing().items():
        print("   %-26s %s" % (k, v))
    print("\nPACING (punchy inspiration option):")
    for k, v in pacing(punchy=True).items():
        print("   %-26s %s" % (k, v))
    print("\nEFFECT BUDGET (from effect_codex.corpus_frequency, MEASURED):")
    for n, r in effect_budget().items():
        print("   %-11s inst=%-2d reels=%-2d per_reel=%-5s max_per_edit=%d  event=%-14s%s"
              % (n, r["instances"], r["reels"], r["per_reel"], r["max_per_edit"], r["event"],
                 "  [TRANSITION seam]" if r["is_transition"] else ""))
    print("\nPLACEABLE OVERLAYS (freq order):",
          [n for n, _ in placeable_overlay_effects()])
