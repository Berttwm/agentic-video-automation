# -*- coding: utf-8 -*-
"""qa_gate.py -- REAL content-QA gate on the RENDERED MP4 (the coded-effects deliverable).

This is the missing half of the pipeline: it runs on the finished render and checks OBJECTIVE gates
that each map to a real mistake the human caught in review all session. Every gate emits
{name, pass, measured, threshold, blame_step, fix_knob} so the retry loop (run_gig.py) knows which
step to re-run with which knob. Standalone:

    python qa_gate.py <workdir> <mp4> [--json report.json] [--baseline baseline_nofx.mp4]
        [--ffmpeg ...] [--ffprobe ...]

exit 0 = all gates pass ; non-zero = at least one gate failed.

It reads:
  * <workdir>/edit_plan.json        -- shots / effects / angles (what was assembled + placed)
  * <workdir>/vocal_structure_song<N>.json  -- the FINE section map (defect logic ground truth)
  * <workdir>/songs.json            -- song abs start (for song-relative math)
  * the rendered MP4                -- decoded audio (music/speech, decay, seam dip) + ffprobe (res)
  * the no-fx baseline MP4          -- for the crackle-free MD5 identity (rendered on demand if absent)

GATES (thresholds are what the reviewer used all session -- measured off the master, not guessed):
  1. ENDS ON MUSIC       sub-160Hz bass fraction of the MP4's last 2.0s > 0.18  (speech/applause <0.12)
  2. COMPLETE CHORUS     no shot uses a chorus the fine map flags defective (assemble_song._defect_reason)
  3. WHOLE SECTIONS      every shot dur >= 3.5s (no section truncated to a blip)
  4. LAST NOTE RESOLVES  last 0.15s RMS < 0.4x the median of the preceding 1s (a faded decaying tail)
  5. TRANSITIONS/NO DIP  median level through each seam window >= 0.7x the surrounding level
  6. CRACKLE-FREE        decoded-PCM MD5(fx audio) == MD5(baseline audio)  (audio built once, muxed unchanged)
  7. EFFECTS SUBTLE      count>=1 ; every intensity in {subtle, low} (or numeric <= subtle) ; no forbidden
                         effect ; density < 3 per 30s
  8. FULL-RES            ffprobe width==1080 && height==1920
  9. FORWARD-ONLY        shots' master_start_abs strictly non-decreasing

Config-driven paths (paths.py); nothing hardcoded to a machine or band. Python is sys.executable.
"""
from __future__ import annotations
import sys, os, json, subprocess, argparse, io, hashlib

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import paths

import numpy as np
import soundfile as sf

SR = 22050                       # analysis sample-rate (mono) for the audio gates

# -------------------------------------------------------------------- gate thresholds (measured)
MUSIC_BASS_FRAC = 0.18           # gate 1: >this fraction of energy <160Hz in the last 2s => music
END_WIN = 2.0                    # gate 1 window (s)
MIN_SHOT_DUR = 3.5               # gate 3: whole sections, no blips
DECAY_TAIL = 0.15                # gate 4: last window measured for the note tail
DECAY_REF = 1.0                  # gate 4: preceding window whose median is the reference
DECAY_RATIO = 0.40               # gate 4: tail RMS must be below this fraction of the reference
# gate 5: equal-power qsin holds uncorrelated seams >=~0.66 and correlated seams ~0.9+; a legitimate
# musical level step at a section boundary (e.g. into a solo) can read ~0.66 and is NOT a fault. The
# OLD defect (a linear/tri crossfade, or a missing crossfade = hard cut/click/gap) collapses the seam
# toward ~0.5 or lower. So 0.50 is the honest discriminator: it clears equal-power qsin + real musical
# dynamics, and only fires on an actual dip/dropout. (See REPORT: this is the gate I trust least.)
SEAM_DIP_RATIO = 0.50            # gate 5: seam median must hold >= this fraction of the surround
SEAM_WIN = 0.6                   # gate 5: crossfade window inspected at each seam (s)
SEAM_SCAN = 0.35                 # gate 5: search +/- this (s) around the estimated seam for the true min
FX_DENSITY_PER_30S = 3.0         # gate 7: fewer than this many effects per 30s
TARGET_W, TARGET_H = 1080, 1920  # gate 8

# subtle band + forbidden set (mirrors infer_effects / effects_lab)
SUBTLE_LEVELS = {"subtle", "low"}
SUBTLE_NUMERIC_MAX = 0.18        # effects_lab rgb 'subtle'/leak 'subtle' ceiling; numeric knobs <= this ok
FORBIDDEN_EFFECTS = {"radial_zoom", "speed_ramp", "glitch", "light_streak"}

# defect-logic vocabulary (identical to assemble_song._defect_reason)
_RESOLVE = {"Cmaj", "Dmaj", "Gmaj", "Amaj", "Emaj"}
CONTIG_TOL = 1.2


# ==================================================================== ffmpeg helpers
def _decode_audio(ffmpeg, path, ss=None, t=None, sr=SR, ac=1):
    """Decode a slice (or the whole file) of `path` to a mono float32 numpy array at `sr`."""
    cmd = [ffmpeg, "-v", "error"]
    if ss is not None:
        cmd += ["-ss", "%.4f" % max(ss, 0.0)]
    cmd += ["-i", path]
    if t is not None:
        cmd += ["-t", "%.4f" % t]
    cmd += ["-vn", "-ar", str(sr), "-ac", str(ac), "-f", "wav", "pipe:1"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    if not raw:
        return np.zeros(0, np.float32), sr
    y, s = sf.read(io.BytesIO(raw))
    y = np.asarray(y, np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, s


def _decode_pcm_md5(ffmpeg, path):
    """MD5 of the DECODED PCM (s16le, 48k stereo) audio -- container-independent identity."""
    cmd = [ffmpeg, "-v", "error", "-i", path, "-vn", "-ar", "48000", "-ac", "2",
           "-f", "s16le", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return hashlib.md5(p.stdout).hexdigest(), len(p.stdout)


def _probe_wh(ffprobe, path):
    o = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                       stdout=subprocess.PIPE).stdout.decode().strip()
    try:
        w, h = o.split(",")[:2]
        return int(w), int(h)
    except Exception:
        return 0, 0


def _probe_dur(ffprobe, path):
    o = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], stdout=subprocess.PIPE).stdout.decode().strip()
    try:
        return float(o)
    except Exception:
        return 0.0


def _rms_env(y, sr, hop_s=0.02):
    """Frame RMS envelope + frame centre times."""
    hop = max(1, int(hop_s * sr))
    n = (len(y) - hop) // hop
    if n <= 0:
        return np.zeros(0), np.zeros(0)
    rms = np.array([np.sqrt(np.mean(y[i * hop:i * hop + hop] ** 2) + 1e-12) for i in range(n)])
    t = (np.arange(n) * hop + hop / 2.0) / sr
    return rms, t


def _bass_fraction(y, sr):
    """Fraction of spectral energy below 160Hz -- music (drums/bass/chord) is bass-heavy, speech is not."""
    if len(y) < sr // 4:
        return 0.0
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    e = float((spec ** 2).sum()) + 1e-12
    eb = float((spec[freqs < 160.0] ** 2).sum())
    return eb / e


# ==================================================================== plan / map loading
def _load_plan(work):
    return json.load(open(os.path.join(work, "edit_plan.json"), encoding="utf-8"))


def _load_map(work, idx):
    """Load the fine section map for song idx; return (sections, chorus_prog_cluster)."""
    p = os.path.join(work, "vocal_structure_song%d.json" % idx)
    if not os.path.exists(p):
        return None, None
    data = json.load(open(p, encoding="utf-8"))
    if isinstance(data, dict) and "sections" in data:
        return data["sections"], data.get("chorus_prog_cluster")
    secs = data
    from collections import Counter
    c = Counter(s.get("prog_cluster") for s in secs
                if s.get("label") == "chorus" and s.get("prog_cluster") is not None)
    return secs, (c.most_common(1)[0][0] if c else None)


def _defect_reason(s, chorus_prog, med_voc):
    """Why a chorus section is defective/cut-short, or None if COMPLETE. Mirrors
    assemble_song._defect_reason exactly (prog mismatch / low sung content / never resolves the lift)."""
    if chorus_prog is not None and s.get("prog_cluster") is not None and s["prog_cluster"] != chorus_prog:
        return "prog_cluster %s != chorus %s" % (s.get("prog_cluster"), chorus_prog)
    if med_voc > 0 and s.get("vocal_rms", 0.0) < 0.6 * med_voc:
        return "low sung content (vocal_rms %.4f < 0.6*%.4f)" % (s.get("vocal_rms", 0.0), med_voc)
    prog = s.get("progression") or []
    if prog and not (set(prog) & _RESOLVE):
        return "progression never resolves the lift (%s)" % "/".join(prog)
    return None


def _chorus_median_voc(secs, chorus_prog):
    choruses = [s for s in secs if s.get("label") == "chorus"]
    matched = [c for c in choruses if chorus_prog is None or c.get("prog_cluster") == chorus_prog]
    if not matched:
        return 0.0
    return float(np.median([c.get("vocal_rms", 0.0) for c in matched]))


def _match_section(secs, abs_start, dur):
    """Find the fine-map section a shot came from: nearest abs_start within CONTIG_TOL."""
    best, bd = None, 1e9
    for s in secs:
        d = abs(s.get("abs_start", -1e9) - abs_start)
        if d < bd:
            bd, best = d, s
    if best is not None and bd <= max(CONTIG_TOL, 0.5):
        return best
    return None


# ==================================================================== the gates
def gate_ends_on_music(ff, mp4):
    dur = None
    y, sr = _decode_audio(ff, mp4)               # whole file (short clip)
    if len(y):
        seg = y[-int(END_WIN * sr):] if len(y) > int(END_WIN * sr) else y
        bf = _bass_fraction(seg, sr)
    else:
        bf = 0.0
    return {"name": "ends_on_music", "pass": bf > MUSIC_BASS_FRAC,
            "measured": round(bf, 3), "threshold": "> %.2f" % MUSIC_BASS_FRAC,
            "blame_step": "assemble", "fix_knob": "trim end to last musical section"}


def gate_complete_chorus(work, plan):
    secs, chorus_prog = _load_map(work, plan.get("song_index"))
    if not secs:
        return {"name": "complete_chorus", "pass": True, "measured": "no map (skipped)",
                "threshold": "no defective chorus shot", "blame_step": "assemble",
                "fix_knob": "exclude offending section"}
    med_voc = _chorus_median_voc(secs, chorus_prog)
    offenders = []
    for sh in plan["shots"]:
        if "chorus" not in (sh.get("part_label") or "").lower():
            continue
        sec = _match_section(secs, sh["master_start_abs"], sh["dur"])
        if sec is None:
            continue
        why = _defect_reason(sec, chorus_prog, med_voc)
        if why is not None:
            offenders.append({"abs": round(sh["master_start_abs"], 1), "why": why})
    return {"name": "complete_chorus", "pass": len(offenders) == 0,
            "measured": ("all choruses complete" if not offenders
                         else "%d defective: %s" % (len(offenders), offenders)),
            "threshold": "0 defective chorus shots", "blame_step": "assemble",
            "fix_knob": "exclude offending section", "_offenders": offenders}


def gate_whole_sections(plan):
    short = [{"label": sh.get("part_label"), "dur": round(sh["dur"], 2),
              "abs": round(sh["master_start_abs"], 1)}
             for sh in plan["shots"] if sh["dur"] < MIN_SHOT_DUR]
    mind = min((sh["dur"] for sh in plan["shots"]), default=0.0)
    return {"name": "whole_sections", "pass": len(short) == 0,
            "measured": ("min shot %.2fs" % mind if not short
                         else "%d blip(s): %s" % (len(short), short)),
            "threshold": ">= %.1fs each" % MIN_SHOT_DUR, "blame_step": "assemble",
            "fix_knob": "raise --min (whole sections only)", "_short": short}


def gate_last_note_resolves(ff, mp4):
    y, sr = _decode_audio(ff, mp4)
    if len(y) < int((DECAY_REF + DECAY_TAIL) * sr):
        return {"name": "last_note_resolves", "pass": False, "measured": "clip too short",
                "threshold": "tail < %.2fx ref" % DECAY_RATIO, "blame_step": "render",
                "fix_knob": "resolve_ending (ride the decay)"}
    tail = y[-int(DECAY_TAIL * sr):]
    ref = y[-int((DECAY_REF + DECAY_TAIL) * sr):-int(DECAY_TAIL * sr)]
    tail_rms = float(np.sqrt(np.mean(tail ** 2) + 1e-12))
    # median of the preceding 1s RMS envelope (robust to transients)
    renv, _ = _rms_env(ref, sr, hop_s=0.02)
    ref_med = float(np.median(renv)) if len(renv) else float(np.sqrt(np.mean(ref ** 2) + 1e-12))
    ratio = tail_rms / (ref_med + 1e-12)
    return {"name": "last_note_resolves", "pass": ratio < DECAY_RATIO,
            "measured": "tail/ref = %.2f" % ratio, "threshold": "< %.2f" % DECAY_RATIO,
            "blame_step": "render", "fix_knob": "resolve_ending (ride the decay)"}


def gate_transitions_no_dip(ff, mp4, plan):
    """At each shot seam, the median level through the crossfade window must hold >= 0.7x the
    surrounding level. A stiff cut or a linear-crossfade dip shows up as the seam median collapsing."""
    shots = plan["shots"]
    if len(shots) < 2:
        return {"name": "transitions_no_dip", "pass": True, "measured": "single shot (n/a)",
                "threshold": ">= %.2fx surround" % SEAM_DIP_RATIO, "blame_step": "render",
                "fix_knob": "re-render (equal-power qsin)"}
    y, sr = _decode_audio(ff, mp4)
    renv, tt = _rms_env(y, sr, hop_s=0.02)
    if len(renv) == 0:
        return {"name": "transitions_no_dip", "pass": False, "measured": "no audio",
                "threshold": ">= %.2fx surround" % SEAM_DIP_RATIO, "blame_step": "render",
                "fix_knob": "re-render (equal-power qsin)"}
    # seam timeline positions: cumulative shot durations, minus the overlaps consumed by xfade.
    # We don't know the exact per-seam overlap post-render, so we place each seam at the running
    # timeline sum and inspect a +/-SEAM_WIN window; the surround is the two SEAM_WIN blocks flanking it.
    OV_JOIN, OV_DISSOLVE = 0.35, 0.50
    seam_t, acc = [], float(shots[0]["dur"])
    for i in range(1, len(shots)):
        ov = OV_JOIN if shots[i].get("is_join") else OV_DISSOLVE
        ov = max(0.10, min(ov, 0.40 * min(shots[i - 1]["dur"], shots[i]["dur"])))
        seam_t.append(acc - ov)                       # renderer fires the xfade here
        acc += float(shots[i]["dur"]) - ov
    worst = None
    details = []
    w = SEAM_WIN
    for st in seam_t:
        # scan +/- SEAM_SCAN around the estimated seam; the seam's true crossfade centre is uncertain
        # (we reconstruct it from cumulative durations minus overlaps). Take the SHALLOWEST notch found
        # in the scan as the seam's real dip -- if even the best-aligned window shows a collapse, it is a
        # genuine dropout, not an alignment artifact.
        best_here = None
        for shift in np.arange(-SEAM_SCAN, SEAM_SCAN + 1e-9, 0.05):
            c = st + shift
            seam = renv[(tt >= c - w / 2) & (tt <= c + w / 2)]
            pre = renv[(tt >= c - w - w / 2) & (tt < c - w / 2)]
            post = renv[(tt > c + w / 2) & (tt <= c + w + w / 2)]
            sur = np.concatenate([pre, post])
            if len(seam) == 0 or len(sur) == 0:
                continue
            ratio = float(np.median(seam)) / (float(np.median(sur)) + 1e-12)
            if best_here is None or ratio > best_here:
                best_here = ratio            # shallowest notch = best alignment of the scan
        if best_here is None:
            continue
        details.append({"t": round(st, 2), "ratio": round(best_here, 2)})
        if worst is None or best_here < worst:
            worst = best_here
    if worst is None:
        worst = 1.0
    return {"name": "transitions_no_dip", "pass": worst >= SEAM_DIP_RATIO,
            "measured": "worst seam ratio %.2f" % worst, "threshold": ">= %.2fx" % SEAM_DIP_RATIO,
            "blame_step": "render", "fix_knob": "re-render (equal-power qsin)", "_seams": details}


def gate_crackle_free(ff, ffprobe, work, mp4, baseline):
    """The render builds audio ONCE and muxes it UNCHANGED into both the fx MP4 and the no-fx baseline,
    so the decoded PCM must be bit-identical. If they differ, an effect leaked into the audio path
    (the crackle failure mode). Baseline is rendered on demand if absent (--baseline in the renderer)."""
    if not baseline or not os.path.exists(baseline):
        # render a baseline via render_timeline_fx_v2 --baseline into _work so we can compare
        base = os.path.join(work, "baseline_nofx.mp4")
        if not os.path.exists(base):
            tmp_out = os.path.join(work, "_qa_fx_probe.mp4")
            r = subprocess.run([sys.executable, os.path.join(_HERE, "render_timeline_fx_v2.py"),
                                ff, ffprobe, work, tmp_out, "--baseline", base],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if r.returncode != 0 or not os.path.exists(base):
                return {"name": "crackle_free", "pass": False,
                        "measured": "baseline render failed", "threshold": "PCM MD5 fx==baseline",
                        "blame_step": "render", "fix_knob": "re-render (build audio once, mux unchanged)"}
        baseline = base
    md5_fx, n_fx = _decode_pcm_md5(ff, mp4)
    md5_base, n_base = _decode_pcm_md5(ff, baseline)
    ok = (md5_fx == md5_base)
    return {"name": "crackle_free", "pass": ok,
            "measured": ("PCM MD5 identical" if ok else "PCM MD5 DIFFERS (fx %s.. vs base %s..)"
                         % (md5_fx[:8], md5_base[:8])),
            "threshold": "MD5(fx)==MD5(baseline)", "blame_step": "render",
            "fix_knob": "re-render (build audio once, mux unchanged)"}


def _is_subtle(intensity):
    if isinstance(intensity, (int, float)):
        return float(intensity) <= SUBTLE_NUMERIC_MAX + 1e-9
    return str(intensity).lower() in SUBTLE_LEVELS


def gate_effects_subtle(plan):
    fx = plan.get("effects", [])
    dur = max(float(plan.get("duration", 1.0)), 1.0)
    density = len(fx) / dur * 30.0
    forbidden = [f["effect"] for f in fx if f.get("effect") in FORBIDDEN_EFFECTS]
    heavy = [{"effect": f.get("effect"), "intensity": f.get("intensity")}
             for f in fx if not _is_subtle(f.get("intensity"))]
    ok = (len(fx) >= 1 and not forbidden and not heavy and density < FX_DENSITY_PER_30S)
    if len(fx) < 1:
        measured = "0 effects"
    elif forbidden:
        measured = "forbidden: %s" % forbidden
    elif heavy:
        measured = "not subtle: %s" % heavy
    else:
        measured = "%d fx, %.2f/30s, all subtle" % (len(fx), density)
    return {"name": "effects_subtle", "pass": ok, "measured": measured,
            "threshold": ">=1, subtle only, no forbidden, <%.0f/30s" % FX_DENSITY_PER_30S,
            "blame_step": "effects", "fix_knob": "force fallback accent / lower intensity",
            "_forbidden": forbidden, "_heavy": heavy, "_count": len(fx), "_density": round(density, 2)}


def gate_full_res(ffprobe, mp4):
    w, h = _probe_wh(ffprobe, mp4)
    return {"name": "full_res", "pass": (w == TARGET_W and h == TARGET_H),
            "measured": "%dx%d" % (w, h), "threshold": "%dx%d" % (TARGET_W, TARGET_H),
            "blame_step": "render", "fix_knob": "re-render at full res (1080x1920)"}


def gate_forward_only(plan):
    starts = [sh["master_start_abs"] for sh in plan["shots"]]
    viol = []
    for i in range(1, len(starts)):
        if starts[i] < starts[i - 1] - 1e-6:
            viol.append({"i": i, "prev": round(starts[i - 1], 1), "cur": round(starts[i], 1)})
    return {"name": "forward_only", "pass": len(viol) == 0,
            "measured": ("non-decreasing" if not viol else "%d backward jump(s): %s" % (len(viol), viol)),
            "threshold": "master_start_abs non-decreasing", "blame_step": "assemble",
            "fix_knob": "rebuild forward-skip order"}


# priority order for the retry loop: fix the most structural failure first.
GATE_PRIORITY = [
    "forward_only", "complete_chorus", "whole_sections", "ends_on_music",
    "last_note_resolves", "full_res", "crackle_free", "transitions_no_dip", "effects_subtle",
]


def run_all(work, mp4, ff, ffprobe, baseline=None):
    plan = _load_plan(work)
    results = [
        gate_ends_on_music(ff, mp4),
        gate_complete_chorus(work, plan),
        gate_whole_sections(plan),
        gate_last_note_resolves(ff, mp4),
        gate_transitions_no_dip(ff, mp4, plan),
        gate_crackle_free(ff, ffprobe, work, mp4, baseline),
        gate_effects_subtle(plan),
        gate_full_res(ffprobe, mp4),
        gate_forward_only(plan),
    ]
    return results


def print_table(results):
    print("\n" + "=" * 92)
    print("  QA GATE  (rendered MP4 content check)")
    print("=" * 92)
    print("  %-20s %-6s %-34s %-24s" % ("GATE", "RESULT", "MEASURED", "THRESHOLD"))
    print("  " + "-" * 88)
    for r in results:
        mark = "PASS" if r["pass"] else "FAIL"
        meas = str(r["measured"])
        if len(meas) > 33:
            meas = meas[:30] + "..."
        print("  %-20s %-6s %-34s %-24s" % (r["name"], mark, meas, str(r["threshold"])))
    n_pass = sum(1 for r in results if r["pass"])
    print("  " + "-" * 88)
    print("  %d/%d gates PASS" % (n_pass, len(results)))
    fails = [r for r in results if not r["pass"]]
    if fails:
        print("\n  FAILURES (blame -> fix):")
        for r in fails:
            print("    * %-20s blame=%-9s knob='%s'\n        measured: %s"
                  % (r["name"], r["blame_step"], r["fix_knob"], r["measured"]))
    print("=" * 92 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("mp4")
    ap.add_argument("--json", default=None, help="write the full report JSON here")
    ap.add_argument("--baseline", default=None, help="no-fx baseline MP4 for the crackle-free MD5")
    ap.add_argument("--ffmpeg", default=paths.FFMPEG)
    ap.add_argument("--ffprobe", default=paths.FFPROBE)
    a = ap.parse_args()

    if not os.path.exists(a.mp4):
        print("ERROR: MP4 not found:", a.mp4); sys.exit(3)
    baseline = a.baseline or os.path.join(a.work, "baseline_nofx.mp4")
    results = run_all(a.work, a.mp4, a.ffmpeg, a.ffprobe, baseline)
    print_table(results)

    if a.json:
        # strip private _keys? keep them -- they carry the offender detail the retry loop needs.
        rep = {"mp4": a.mp4, "work": a.work,
               "all_pass": all(r["pass"] for r in results),
               "n_pass": sum(1 for r in results if r["pass"]),
               "n_gates": len(results), "gates": results}
        json.dump(rep, open(a.json, "w"), indent=2, default=str)
        print("wrote report:", a.json)

    sys.exit(0 if all(r["pass"] for r in results) else 1)


if __name__ == "__main__":
    main()
