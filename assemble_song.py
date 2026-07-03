# -*- coding: utf-8 -*-
"""Assemble ONE song into edit_plan.json from the FINE SECTION MAP (structure2.py), v3.

WHAT CHANGED (v2 -> v3)
    v2 read the OLD fixed-k structure.json (structure_analyze.py) and condensed the whole song by a
    forward-skip walk. That produced blobby, rushed arrangements on a harmonically uniform cover.
    v3 reads the FINE section map produced by structure2.analyze_song (demucs vocal-aware novelty
    boundaries + repetition clustering + chord-progression verse/chorus split), cached as
    <workdir>/vocal_structure_song<N>.json, and builds the shape the user converged on by hand:

        a VOCAL BUILD (verse -> COMPLETE chorus, WHOLE sections at natural length)
        -> the GUITAR SOLO played WHOLE
        -> END on the solo (or a complete chorus), never the outro banter.

    Selection rules (all read off the fine map + chords):
      * COMPLETE chorus only: a section labeled 'chorus' whose prog_cluster == the map's
        chorus_prog_cluster (the borrowed-minor "lift" pattern) AND whose length is full (>= the median
        complete-chorus length, i.e. not cut short by a bar). The defective/short chorus (chords flags it
        as a chorus cut short) is NEVER selected.
      * Prefer the complete chorus that flows CONTIGUOUSLY into the solo (chorus.abs_end == solo.abs_start
        within a small tolerance) so verse->chorus->solo runs unbroken.
      * SKIP interludes / instrumental breakdowns and any low-quality section.
      * Let it BREATHE: target ~50-80s; do NOT truncate sections -- whole sections only.
      * END on a MUSICAL section: verify the last ~2s is music (sub-160Hz bass fraction > MUSIC_BASS_FRAC)
        via the MASTER audio, so we never land on crowd banter.
      * ONE drummer-cam switch on an interior whole section for visual variety (audio stays continuous).

Output schema is UNCHANGED (consumed by infer_effects + render_timeline_fx_v2): top-level
angles/duration/effects/transitions + shots[] carrying tl_start/dur/angle/master_start_abs/
angle_start_abs/is_join/transition_in.

Config-driven paths (paths.py); no band handle / device path hardcoded. Python is sys.executable.
Usage: python assemble_song.py <workdir> <draft_name> [--song N] [--target 65] [--no-switch]
                               [--mode single|gig]
"""
import sys, os, json, argparse, subprocess, io
import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("work"); ap.add_argument("name")
ap.add_argument("--song", type=int, default=0)
ap.add_argument("--target", type=float, default=65.0,
                help="target edit duration (s); the vocal build fills toward this, whole-sections only")
ap.add_argument("--min", type=float, default=45.0)
ap.add_argument("--max", type=float, default=85.0)
ap.add_argument("--no-switch", action="store_true")
ap.add_argument("--punchy", action="store_true",
                help="use the trending inspiration cadence (@firsttoeleven ~12 cuts/min) instead of "
                     "P&P's measured long-take cadence. OFF by default -- whole-sections taste lock wins.")
ap.add_argument("--mode", choices=["single", "gig"], default="single")
ap.add_argument("--exclude-abs", default="",
                help="comma list of section abs_start values to EXCLUDE from selection (QA-retry knob: "
                     "drop an offending defective chorus so the assembler picks a complete one instead)")
a = ap.parse_args()
# QA-retry knob: sections whose abs_start matches one of these (within tolerance) are removed from the
# fine map before any selection, so a section the QA gate flagged can never re-enter the arrangement.
_EXCLUDE_ABS = [float(x) for x in a.exclude_abs.split(",") if x.strip()]
SR = 22050
from paths import FFPROBE, FFMPEG
import grammar as G

# ---------------------------------------------------------------- READ THE MEASURED GRAMMAR (pacing)
# style_model's CADENCE + best-practices drive the arrangement's PACING knobs (how long takes run, how
# many section-cuts / drummer-cam switches are allowed). Default = P&P's own measured long-take cadence
# (~2 cuts/min, whole sections); --punchy swaps in the trending inspiration cadence (~12 cuts/min) but
# that is OFF by default because the user's whole-sections/let-it-breathe taste lock wins. We LAYER
# these knobs on top of the v3 selection logic (vocal build -> complete chorus -> whole solo, avoid the
# defective chorus, end on music) -- the selection is unchanged; pacing only bounds cuts/switches.
PACING = G.pacing(punchy=a.punchy)
# max angle switches allowed (P&P long-take default = 1 interior drummer-cam switch). --no-switch still
# forces 0. --punchy raises it toward the inspiration cadence.
MAX_SWITCHES = 0 if a.no_switch else int(PACING.get("drummer_switches", 1))
print("=> PACING: %s | cuts/min~%.1f, drummer_switches<=%d, whole_sections=%s, condense=%s"
      % ("PUNCHY (inspiration)" if PACING.get("punchy") else "P&P long-take (measured)",
         PACING.get("cuts_per_min", 2.0), MAX_SWITCHES,
         PACING.get("whole_sections", True), PACING.get("condense", False)))

A = json.load(open(os.path.join(a.work, "analysis.json")))
meta = A["_meta"]; master = meta["master"]
ranking = meta["ranking"]; second = ranking[1] if len(ranking) > 1 else master
offs = meta.get("sync_offset_refined") or meta.get("sync_offset") or {}
songs = json.load(open(os.path.join(a.work, "songs.json")))
songs = songs if isinstance(songs, list) else songs.get("songs", songs)

CONTIG_TOL = 1.2          # sections are "contiguous" if their abs boundaries meet within this (s)
MUSIC_BASS_FRAC = 0.18    # >this fraction of energy below 160Hz in the last window => music, not speech
END_WIN = 2.0             # window (s) checked at the edit end for music-vs-speech


def media_dur(p):
    try:
        return float(subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                    "-of", "csv=p=0", p], stdout=subprocess.PIPE).stdout.decode().strip())
    except Exception:
        return 0.0


# ---------------------------------------------------------------- fine section map
def load_map(idx):
    """Load the fine section map for song idx. Prefer the full dict (has chorus_prog_cluster);
    fall back to the bare sections list written by structure2.main()."""
    p = os.path.join(a.work, "vocal_structure_song%d.json" % idx)
    if not os.path.exists(p):
        print("ERROR: fine section map not found: %s\n"
              "       run_gig stage 4 (structure2) must produce it first." % p)
        sys.exit(2)
    data = json.load(open(p, encoding="utf-8"))
    if isinstance(data, dict) and "sections" in data:
        return data["sections"], data.get("chorus_prog_cluster"), data.get("chorus_cluster")
    # bare list -> infer the chorus progression cluster from the sections themselves
    secs = data
    cp = _infer_chorus_prog(secs)
    return secs, cp, None


def _infer_chorus_prog(secs):
    """When only the bare list is available: the chorus prog_cluster is the prog_cluster carried by the
    sections labeled 'chorus' (majority)."""
    from collections import Counter
    c = Counter(s.get("prog_cluster") for s in secs
                if s.get("label") == "chorus" and s.get("prog_cluster") is not None)
    return c.most_common(1)[0][0] if c else None


# ---------------------------------------------------------------- choose the song
if a.song:
    idx = a.song
else:
    # auto: the song with a solo AND at least one complete chorus, longest such wins
    best = None
    for s in songs:
        si = s.get("index")
        p = os.path.join(a.work, "vocal_structure_song%d.json" % si)
        if not os.path.exists(p):
            continue
        secs, cp, _ = load_map(si)
        has_solo = any(x["label"] == "solo" for x in secs)
        has_ch = any(x["label"] == "chorus" for x in secs)
        if has_solo and has_ch:
            dur = sum(x["dur"] for x in secs)
            if best is None or dur > best[1]:
                best = (si, dur)
    if best is None:
        print("ERROR: no song has both a solo and a chorus in its fine map; pass --song N")
        sys.exit(2)
    idx = best[0]

secs, chorus_prog, chorus_cl = load_map(idx)
if _EXCLUDE_ABS:
    _before = len(secs)
    secs = [s for s in secs
            if not any(abs(s.get("abs_start", -1e9) - ex) <= 0.75 for ex in _EXCLUDE_ABS)]
    print("   [QA-retry] excluded %d section(s) at abs %s (%d -> %d sections)"
          % (_before - len(secs), _EXCLUDE_ABS, _before, len(secs)))
song_abs = songs[idx - 1].get("start", 0.0) if idx - 1 < len(songs) else 0.0
print("=> song %d  (%d sections, chorus_prog_cluster=%s)" % (idx, len(secs), chorus_prog))

# ---------------------------------------------------------------- classify sections
SKIP_LABELS = {"interlude", "outro", "intro"}   # instrumental breakdowns + banter outro are skipped

# Chords carrying the "chorus lift" (borrowed minor). A HEALTHY chorus resolves the lift arc (it reaches
# a plain major like Cmaj/Dmaj at the phrase end); a DEFECTIVE chorus cut short by a bar gets stuck on
# the lift (all-Fmin/Cmin, never resolving) and carries much less sung content. chords.py uses the same
# vocabulary.
_LIFT = {"Fmin", "Cmin", "G#maj", "D#min", "A#min"}
_RESOLVE = {"Cmaj", "Dmaj", "Gmaj", "Amaj", "Emaj"}

choruses = [s for s in secs if s["label"] == "chorus"]
# reference vocal content of a full chorus (median over prog-matched choruses)
_ch_matched = [c for c in choruses
               if chorus_prog is None or c.get("prog_cluster") == chorus_prog]
_med_voc = float(np.median([c.get("vocal_rms", 0.0) for c in _ch_matched])) if _ch_matched else 0.0
_med_len = float(np.median([c["dur"] for c in _ch_matched])) if _ch_matched else 0.0


def _defect_reason(s):
    """Return why a chorus is defective/cut-short, or None if it is COMPLETE.

    A chorus is defective (cut short by a bar -- the ~807-818 case) when ANY of:
      * its progression cluster does NOT match the chorus lift pattern, or
      * it carries far less sung content than a full chorus (vocal_rms < 0.6 * median chorus), which is
        the acoustic signature of a chorus guillotined a bar early, or
      * its chord progression never RESOLVES the lift (no plain-major bar) -- it is stuck on the
        borrowed-minor lift, i.e. the phrase was cut before it landed."""
    if chorus_prog is not None and s.get("prog_cluster") is not None and s["prog_cluster"] != chorus_prog:
        return "prog_cluster %s != chorus %s" % (s.get("prog_cluster"), chorus_prog)
    if _med_voc > 0 and s.get("vocal_rms", 0.0) < 0.6 * _med_voc:
        return "low sung content (vocal_rms %.4f < 0.6*%.4f)" % (s.get("vocal_rms", 0.0), _med_voc)
    prog = s.get("progression") or []
    if prog and not (set(prog) & _RESOLVE):
        return "progression never resolves the lift (%s)" % "/".join(prog)
    return None


def is_complete_chorus(s, _complete_len=None):
    return s["label"] == "chorus" and _defect_reason(s) is None


complete_len = _med_len
solo = next((s for s in secs if s["label"] == "solo"), None)

print("   choruses:")
for c in choruses:
    dr = _defect_reason(c)
    tag = "COMPLETE" if dr is None else "short/defective -> AVOID (%s)" % dr
    print("     abs %.1f-%.1f dur=%.1fs prog_cl=%s voc=%.4f  %s"
          % (c["abs_start"], c["abs_end"], c["dur"], c.get("prog_cluster"), c.get("vocal_rms", 0.0), tag))
if solo:
    print("   solo: abs %.1f-%.1f dur=%.1fs" % (solo["abs_start"], solo["abs_end"], solo["dur"]))

# ---------------------------------------------------------------- pick the COMPLETE chorus -> solo
# Prefer the complete chorus that flows CONTIGUOUSLY into the solo; else the longest complete chorus.
complete = [c for c in choruses if is_complete_chorus(c, complete_len)]
if not complete:
    print("ERROR: no COMPLETE chorus found (all choruses look short/defective); cannot build showcase")
    sys.exit(2)

chorus = None
if solo is not None:
    for c in complete:
        if abs(c["abs_end"] - solo["abs_start"]) <= CONTIG_TOL:
            chorus = c; break
if chorus is None:
    chorus = max(complete, key=lambda c: c["dur"])   # fall back to the fullest complete chorus
contig_to_solo = solo is not None and abs(chorus["abs_end"] - solo["abs_start"]) <= CONTIG_TOL
print("   => chosen chorus: abs %.1f-%.1f (%s solo)"
      % (chorus["abs_start"], chorus["abs_end"], "contiguous into" if contig_to_solo else "NOT contiguous to"))


def by_index(s):
    return secs.index(s)


# ---------------------------------------------------------------- assemble the VOCAL BUILD window
# Build backwards from the chosen chorus: (1) gather the CONTIGUOUS clean sung lead-in that runs straight
# into the chosen chorus (verse[, earlier complete chorus...]), stopping at any interlude/defective chorus
# so the run stays unbroken. Then (2) the chosen chorus, then (3) the whole solo. Whole sections only.
ci = by_index(chorus)


def _clean_leadin(anchor_i, budget):
    """Whole sung sections immediately preceding index anchor_i, back to back, stopping at the first
    interlude / defective chorus (so the run is contiguous). Returns the list in time order."""
    lead = []
    j = anchor_i - 1
    while j >= 0 and budget > 0:
        s = secs[j]
        if s["label"] in SKIP_LABELS:
            break
        if s["label"] == "chorus" and not is_complete_chorus(s):
            break
        lead.insert(0, s)
        budget -= s["dur"]
        j -= 1
    while lead and lead[0]["label"] != "verse":    # a proper build STARTS on a verse
        lead.pop(0)
    return lead


solo_dur = solo["dur"] if (solo is not None and contig_to_solo) else 0.0
lead = _clean_leadin(ci, a.target - chorus["dur"] - solo_dur)
window = lead + [chorus]

# (2) LET IT BREATHE: if the contiguous build is still under --min, prepend an EARLIER complete
# verse->chorus build (the best one before the gap), accepting ONE clean join. This is the fuller vocal
# build the user approved (verse->chorus  ...  verse->chorus->solo). We only add a whole verse->chorus
# pair, never a fragment, and never the defective chorus.
def _cur_dur():
    return sum(s["dur"] for s in window) + solo_dur


if _cur_dur() < a.min and window:
    first_i = by_index(window[0])
    # find the latest COMPLETE chorus strictly before the current window start...
    earlier_ch = None
    for k in range(first_i - 1, -1, -1):
        if secs[k]["label"] == "chorus" and is_complete_chorus(secs[k]):
            earlier_ch = k; break
    if earlier_ch is not None:
        pre_lead = _clean_leadin(earlier_ch, a.max - _cur_dur() - secs[earlier_ch]["dur"])
        pre = pre_lead + [secs[earlier_ch]]
        # only take it if it actually adds a verse->chorus and doesn't overshoot --max
        if any(s["label"] == "verse" for s in pre) and _cur_dur() + sum(s["dur"] for s in pre) <= a.max:
            window = pre + window

# (3) append the whole solo (played WHOLE) -- the payoff we END on
if solo is not None:
    window = window + [solo]

# guarantee the build actually contains a verse (the vocals)
if not any(s["label"] == "verse" for s in window):
    v = None
    for k in range(ci - 1, -1, -1):
        if secs[k]["label"] == "verse":
            v = secs[k]; break
    if v is not None:
        window = [v] + window

print("   window sections: %s" % " -> ".join("%s[%.0fs]" % (s["label"], s["dur"]) for s in window))
print("   window duration: %.1fs" % sum(s["dur"] for s in window))

# ---------------------------------------------------------------- MUSIC-vs-SPEECH end guard
def bass_fraction(abs_end, win=END_WIN):
    """Fraction of spectral energy below 160Hz in the last `win` seconds before abs_end, read from the
    MASTER source. Music (drums/bass/guitar chord) carries strong sub-160Hz energy; crowd banter/speech
    does not. >MUSIC_BASS_FRAC => music."""
    try:
        raw = subprocess.run([FFMPEG, "-v", "error", "-ss", "%.3f" % max(abs_end - win, 0.0),
                              "-t", "%.3f" % win, "-i", A[master]["path"], "-vn", "-ar", str(SR),
                              "-ac", "1", "-f", "wav", "pipe:1"], stdout=subprocess.PIPE).stdout
        y, sr = sf.read(io.BytesIO(raw)); y = y.astype(np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) < sr // 4:
            return 0.0
        spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
        e = float((spec ** 2).sum()) + 1e-12
        eb = float((spec[freqs < 160.0] ** 2).sum())
        return eb / e
    except Exception as ex:
        print("   (bass-fraction check skipped: %s)" % ex)
        return 1.0


end_abs = window[-1]["abs_end"]
bf = bass_fraction(end_abs)
print("   ending bass-fraction @ abs %.1f = %.3f (%s)"
      % (end_abs, bf, "MUSIC" if bf > MUSIC_BASS_FRAC else "SPEECH -- trimming"))
# if the end is speech, drop the last section and land on the previous MUSICAL one
while len(window) > 1 and bf <= MUSIC_BASS_FRAC:
    dropped = window.pop()
    print("     dropped trailing '%s' (looked like speech); new end abs %.1f"
          % (dropped["label"], window[-1]["abs_end"]))
    end_abs = window[-1]["abs_end"]
    bf = bass_fraction(end_abs)

# ---------------------------------------------------------------- resolve the ending (ride the decay)
# Structural boundaries land on a beat, not on the note RELEASE, so the final note can be guillotined
# while still ringing. Extend the last section until the ringing note decays to the noise floor.
def resolve_ending(abs_end):
    try:
        raw = subprocess.run([FFMPEG, "-v", "error", "-ss", "%.3f" % max(abs_end - 1.0, 0.0),
                              "-t", "5.0", "-i", A[master]["path"], "-vn", "-ar", str(SR),
                              "-ac", "1", "-f", "wav", "pipe:1"], stdout=subprocess.PIPE).stdout
        yy, ss = sf.read(io.BytesIO(raw)); yy = yy.astype(np.float32)
        if yy.ndim > 1:
            yy = yy.mean(axis=1)
        hop = int(0.05 * ss)
        if len(yy) < hop * 4:
            return 0.0
        rms = np.array([np.sqrt(np.mean(yy[i:i + hop] ** 2)) for i in range(0, len(yy) - hop, hop)])
        tt = (abs_end - 1.0) + np.arange(len(rms)) * 0.05
        floor = float(np.percentile(rms, 10))
        cut_i = int(np.argmin(np.abs(tt - abs_end)))
        rms_cut = float(rms[cut_i])
        if rms_cut < 1.8 * floor:
            return 0.0
        thr = max(0.40 * rms_cut, 1.8 * floor)
        need, low = 3, 0
        for i in range(cut_i + 1, len(rms)):
            low = low + 1 if rms[i] < thr else 0
            if low >= need:
                return float(np.clip(tt[i - need + 1] - abs_end, 0.0, 3.0))
        return min(3.0, float(tt[-1] - abs_end))
    except Exception as e:
        print("   (ending-resolve skipped: %s)" % e); return 0.0


ext = resolve_ending(window[-1]["abs_end"])
if ext > 0.05:
    window[-1] = dict(window[-1])
    window[-1]["abs_end"] = round(window[-1]["abs_end"] + ext, 3)
    window[-1]["dur"] = round(window[-1]["dur"] + ext, 3)
    print("   ENDING: extended final '%s' by +%.2fs so the last note rings out"
          % (window[-1]["label"], ext))

# ---------------------------------------------------------------- ONE drummer-cam switch (interior)
drum_off = offs.get(second, 0.0)
drum_dur = media_dur(A[second]["path"]) if second != master else 0.0


def covers(s):
    """drummer cam covers this section (angle time = master time + offset stays within the drummer clip)."""
    return (second != master and drum_dur > 0
            and (s["abs_start"] + drum_off + s["dur"]) <= drum_dur - 1.0)


# PACING-DRIVEN: the number of interior drummer-cam switches is bounded by MAX_SWITCHES (from the
# measured cadence -- P&P long-take default = 1). We pick the MAX_SWITCHES highest-energy interior whole
# sections the drummer cam covers (never first/last), so more switches only happen under --punchy.
switch_set = set()
if MAX_SWITCHES > 0 and second != master and drum_dur > 0 and len(window) >= 3:
    interior = [k for k in range(1, len(window) - 1) if covers(window[k])]
    interior.sort(key=lambda k: window[k].get("mix_rms", 0.0), reverse=True)
    switch_set = set(interior[:MAX_SWITCHES])
switch_i = max(switch_set, key=lambda k: window[k].get("mix_rms", 0.0)) if switch_set else -1

# ---------------------------------------------------------------- build shots
shots = []; tl = 0.0; prev_end_abs = None
for i, s in enumerate(window):
    join = (prev_end_abs is not None and abs(s["abs_start"] - prev_end_abs) > CONTIG_TOL)
    angle = "drummer" if i in switch_set else "master"
    ang_off = drum_off if angle == "drummer" else 0.0
    # a Blur transition marks a real join, the switch INTO the drummer cam, and the switch back to master
    trans = "Blur" if (join or i in switch_set or (i - 1) in switch_set) else None
    shots.append({
        "tl_start": round(tl, 3), "dur": round(s["dur"], 3),
        "part_label": s["label"], "cluster": s.get("cluster", -1),
        "energy_ratio": round(s.get("mix_rms", 0.0) / (np.mean([x.get("mix_rms", 0.0) for x in secs]) + 1e-9), 3),
        "angle": angle, "master_start_abs": round(s["abs_start"], 3),
        "angle_start_abs": round(s["abs_start"] + ang_off, 3),
        "transition_in": trans, "is_join": join, "crossfade_in": join,
        "reason": ("crossfade join into %s (skipped a section)" % s["label"] if join
                   else "drummer cam on %s" % s["label"] if angle == "drummer"
                   else "continue %s" % s["label"]),
    })
    tl += s["dur"]; prev_end_abs = s["abs_start"] + s["dur"]

plan = {
    "name": a.name, "song_index": idx, "song_title": "Song %d" % idx,
    "master": master,
    "angles": {"master": {"path": A[master]["path"], "offset": 0.0},
               "drummer": {"path": A[second]["path"], "offset": drum_off}},
    "duration": round(tl, 3),
    "n_shots": len(shots),
    "n_joins": sum(1 for s in shots if s["is_join"]),
    "n_angle_switches": sum(1 for s in shots if s["angle"] != "master"),
    "n_transitions": sum(1 for s in shots if s["transition_in"]),
    "crossfade_s": 0.6, "fade_in_s": 0.8, "fade_out_s": 1.2,
    "ending_bass_fraction": round(bf, 3),
    "chorus_prog_cluster": chorus_prog,
    "pacing": {                          # the MEASURED grammar knobs this arrangement was paced by
        "mode": "punchy" if PACING.get("punchy") else "pnp_long_take",
        "source": PACING.get("source"),
        "cuts_per_min": PACING.get("cuts_per_min"),
        "max_drummer_switches": MAX_SWITCHES,
        "whole_sections": PACING.get("whole_sections", True),
        "condense": PACING.get("condense", False),
    },
    "title_card": {"text": "THE BAND", "sub": "(song title - edit in CapCut)", "style": "white_top"},
    "shots": shots,
    "effects": [],           # filled by infer_effects.py
    "transitions": [],       # filled by infer_effects.py
}

out = os.path.join(a.work, "edit_plan.json")
json.dump(plan, open(out, "w"), indent=2)
print("\nWROTE %s  dur=%.1fs shots=%d joins=%d switches=%d transitions=%d  end=%s(bf=%.3f)"
      % (out, tl, len(shots), plan["n_joins"], plan["n_angle_switches"], plan["n_transitions"],
         shots[-1]["part_label"], bf))
for s in shots:
    print("   %5.1f-%5.1fs %-8s %-7s%s  (%s)"
          % (s["tl_start"], s["tl_start"] + s["dur"], s["part_label"], s["angle"],
             "  <" + s["transition_in"] + ">" if s["transition_in"] else "", s["reason"]))
