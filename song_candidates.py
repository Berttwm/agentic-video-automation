# -*- coding: utf-8 -*-
"""song_candidates.py -- GIG-LEVEL "which songs are good reel candidates" scorer.

Standalone:  python song_candidates.py <workdir> [--json out.json] [--no-structure]

For EVERY detected song in a gig workdir it:
  1. ENSURES the fine structure map exists (vocal_structure_song<N>.json). If missing it calls
     structure2.analyze_song, which reuses cached demucs stems under _stems/htdemucs/song<N>/ and
     NEVER re-separates when both stem wavs already exist. (--no-structure skips this and scores only
     songs whose map already exists -- useful for a pure re-score.)
  2. SCORES reel-potential from the fine map + the song wav. The score rewards what makes a good
     highlight reel: a real SOLO, at least one COMPLETE (payoff) chorus, a clear verse->chorus vocal
     build, clean audio, compelling energy dynamics, drummer-cam coverage for cut variety, and a sane
     duration. Weights are exposed as constants below.

Output: a ranked table (best first) + optional JSON list of per-song dicts:
  {song_index, score, dur, has_solo, n_complete_choruses, has_build, clip_pct, snr,
   drummer_covered, arousal_peak, note}

GENERIC: nothing hardcodes a song title or a setlist -- everything is reported by song_index. The
defect / complete-chorus logic mirrors assemble_song._defect_reason and qa_gate exactly, so the score
agrees with what the real pipeline will actually be able to build.

Config-driven paths (paths.py); Python is sys.executable; byte-compilable. Deps: numpy, soundfile
(+ librosa/torch/demucs only if a map must actually be computed via structure2).
"""
from __future__ import annotations
import sys, os, json, argparse
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import paths  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

# ======================================================================= SCORING WEIGHTS (exposed)
# Each ingredient contributes points; the total is the reel-candidate score. A song that lacks a
# payoff (no complete chorus AND no solo) can never make a satisfying highlight, so it is capped.
W_SOLO            = 22.0   # a real instrumental SOLO -- the strongest single highlight ingredient
W_CHORUS_FIRST    = 18.0   # the FIRST complete (payoff) chorus -- you need >=1 to land the hook
W_CHORUS_EXTRA    = 6.0    # each additional complete chorus (diminishing; capped at CHORUS_EXTRA_CAP)
CHORUS_EXTRA_CAP  = 2      # count at most this many extra choruses toward the score
W_BUILD           = 14.0   # a clear verse -> chorus vocal build (tension then release)
W_AUDIO_QUALITY   = 12.0   # clean audio: low clipping, decent SNR/crest (scaled 0..1 -> points)
W_DYNAMICS        = 12.0   # energy dynamics: a compelling arousal range / peak (scaled 0..1)
W_DRUMMER         = 8.0    # drummer-cam covers the song span -> we can cut for variety
W_DURATION        = 8.0    # duration sanity (0..1 triangular around the sweet spot)

NO_PAYOFF_CAP     = 30.0   # hard ceiling if a song has neither a solo nor any complete chorus

# duration sweet spot (s): plenty of material but not a marathon
DUR_MIN, DUR_LO, DUR_HI, DUR_MAX = 90.0, 150.0, 340.0, 480.0

# audio-quality scaling
CLIP_PCT_BAD      = 1.0     # >= this clip% -> 0 quality credit from clipping
SNR_GOOD_DB       = 14.0    # >= this SNR -> full SNR credit
SNR_BAD_DB        = 4.0     # <= this SNR -> no SNR credit
CREST_GOOD_DB     = 12.0    # healthy crest factor (dynamic, un-squashed)
CREST_BAD_DB      = 5.0

# dynamics scaling: arousal = short-window loudness envelope; we reward a wide range + a real peak.
AR_RANGE_GOOD_DB  = 12.0    # peak-to-median loudness spread that reads as "dynamic"
DRUMMER_TAIL_PAD  = 1.0     # drummer clip must extend >= song_end + this (mirrors assemble.covers)

# defect vocabulary (identical to assemble_song / qa_gate)
_RESOLVE = {"Cmaj", "Dmaj", "Gmaj", "Amaj", "Emaj"}
SR = 22050


# ======================================================================= audio helpers
def _load_mono(path, sr=SR):
    y, file_sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, np.float32)
    if file_sr != sr:
        # cheap linear resample (we only need loudness statistics, not fidelity)
        n = int(round(len(y) * sr / float(file_sr)))
        if n > 1:
            y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y).astype(np.float32)
    return y, sr


def _clip_pct(y):
    if len(y) == 0:
        return 0.0
    return 100.0 * float(np.mean(np.abs(y) >= 0.999))


def _snr_crest_db(y):
    """Rough SNR (loud-frame vs quiet-frame RMS) + crest factor (peak vs RMS), both in dB."""
    if len(y) < SR:
        return 0.0, 0.0
    hop = int(0.05 * SR)
    n = len(y) // hop
    if n < 4:
        return 0.0, 0.0
    rms = np.array([np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    hi = float(np.percentile(rms, 90))
    lo = float(np.percentile(rms, 10))
    snr = 20.0 * np.log10((hi + 1e-9) / (lo + 1e-9))
    peak = float(np.max(np.abs(y))) + 1e-9
    rms_all = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    crest = 20.0 * np.log10(peak / (rms_all + 1e-9))
    return float(snr), float(crest)


def _arousal_range_db(y):
    """Arousal proxy = short-window loudness envelope. Return the peak-to-median spread in dB
    (a compelling song swings; a flat one does not) and the normalized peak position 0..1."""
    if len(y) < SR:
        return 0.0, 0.0
    win = int(0.5 * SR)
    hop = int(0.25 * SR)
    n = max(0, (len(y) - win) // hop)
    if n < 4:
        return 0.0, 0.0
    env = np.array([np.sqrt(np.mean(y[i * hop:i * hop + win] ** 2) + 1e-12) for i in range(n)])
    env_db = 20.0 * np.log10(env + 1e-9)
    peak = float(np.percentile(env_db, 98))
    med = float(np.median(env_db))
    rng = peak - med
    peak_pos = float(np.argmax(env)) / max(1, n - 1)
    return float(rng), float(peak_pos)


# ======================================================================= map / defect logic
def _load_map(work, idx):
    p = os.path.join(work, "vocal_structure_song%d.json" % idx)
    if not os.path.exists(p):
        return None, None
    data = json.load(open(p, encoding="utf-8"))
    if isinstance(data, dict) and "sections" in data:
        secs = data["sections"]
        return secs, data.get("chorus_prog_cluster")
    secs = data
    c = Counter(s.get("prog_cluster") for s in secs
                if s.get("label") == "chorus" and s.get("prog_cluster") is not None)
    return secs, (c.most_common(1)[0][0] if c else None)


def _chorus_median_voc(secs, chorus_prog):
    choruses = [s for s in secs if s.get("label") == "chorus"]
    matched = [c for c in choruses if chorus_prog is None or c.get("prog_cluster") == chorus_prog]
    if not matched:
        return 0.0
    return float(np.median([c.get("vocal_rms", 0.0) for c in matched]))


def _defect_reason(s, chorus_prog, med_voc):
    """Why a chorus is defective/cut-short, or None if COMPLETE. Mirrors assemble_song/qa_gate."""
    if chorus_prog is not None and s.get("prog_cluster") is not None and s["prog_cluster"] != chorus_prog:
        return "prog_cluster mismatch"
    if med_voc > 0 and s.get("vocal_rms", 0.0) < 0.6 * med_voc:
        return "low sung content"
    prog = s.get("progression") or []
    if prog and not (set(prog) & _RESOLVE):
        return "never resolves the lift"
    return None


def _has_build(secs):
    """A clear verse -> chorus vocal build: some verse immediately (or near-) precedes a chorus and
    the chorus carries at least as much sung energy (tension in the verse, release in the chorus)."""
    for i in range(1, len(secs)):
        if secs[i].get("label") == "chorus":
            # look back through a pre-chorus to the nearest verse
            j = i - 1
            if j >= 0 and secs[j].get("label") == "pre-chorus":
                j -= 1
            if j >= 0 and secs[j].get("label") == "verse":
                return True
    return False


# ======================================================================= per-song scoring
def _quality_unit(clip_pct, snr, crest):
    clip_credit = max(0.0, 1.0 - clip_pct / CLIP_PCT_BAD)
    snr_credit = np.clip((snr - SNR_BAD_DB) / (SNR_GOOD_DB - SNR_BAD_DB), 0.0, 1.0)
    crest_credit = np.clip((crest - CREST_BAD_DB) / (CREST_GOOD_DB - CREST_BAD_DB), 0.0, 1.0)
    # clipping is the hard disqualifier; SNR + crest share the rest
    return float(0.5 * clip_credit + 0.3 * snr_credit + 0.2 * crest_credit)


def _duration_unit(dur):
    if dur <= DUR_MIN or dur >= DUR_MAX:
        return 0.0
    if DUR_LO <= dur <= DUR_HI:
        return 1.0
    if dur < DUR_LO:
        return (dur - DUR_MIN) / (DUR_LO - DUR_MIN)
    return (DUR_MAX - dur) / (DUR_MAX - DUR_HI)


def score_song(idx, wav, secs, chorus_prog, drum_off, drum_dur):
    y, sr = _load_mono(wav)
    dur = len(y) / float(sr)
    clip_pct = _clip_pct(y)
    snr, crest = _snr_crest_db(y)
    ar_range, ar_peak_pos = _arousal_range_db(y)

    med_voc = _chorus_median_voc(secs, chorus_prog)
    choruses = [s for s in secs if s.get("label") == "chorus"]
    complete = [c for c in choruses if _defect_reason(c, chorus_prog, med_voc) is None]
    n_complete = len(complete)
    has_solo = any(s.get("label") == "solo" for s in secs)
    has_build = _has_build(secs)

    # drummer-cam coverage: does the drummer clip span the whole song (mirrors assemble.covers)?
    # The fine-map abs_start/abs_end are on the MASTER timeline; drum_off is the master->drummer sync.
    song_abs_end = max((s.get("abs_end", 0.0) for s in secs), default=0.0)
    drummer_covered = bool(drum_dur > 0 and (song_abs_end + drum_off + DRUMMER_TAIL_PAD) <= drum_dur)

    # ---- ingredient points ----
    pts = 0.0
    pts += W_SOLO if has_solo else 0.0
    if n_complete >= 1:
        pts += W_CHORUS_FIRST
        pts += W_CHORUS_EXTRA * min(CHORUS_EXTRA_CAP, n_complete - 1)
    pts += W_BUILD if has_build else 0.0
    q_unit = _quality_unit(clip_pct, snr, crest)
    pts += W_AUDIO_QUALITY * q_unit
    dyn_unit = float(np.clip(ar_range / AR_RANGE_GOOD_DB, 0.0, 1.0))
    pts += W_DYNAMICS * dyn_unit
    pts += W_DRUMMER if drummer_covered else 0.0
    dur_unit = _duration_unit(dur)
    pts += W_DURATION * dur_unit

    # ---- no-payoff cap: neither a solo nor a complete chorus => can't land a highlight ----
    capped = False
    if not has_solo and n_complete == 0:
        if pts > NO_PAYOFF_CAP:
            pts = NO_PAYOFF_CAP
            capped = True

    # ---- human-readable note ----
    bits = []
    bits.append("solo" if has_solo else "no-solo")
    bits.append("%dxCC" % n_complete if n_complete else "0CC")
    bits.append("build" if has_build else "no-build")
    if capped:
        bits.append("CAPPED(no payoff)")
    if clip_pct >= 0.5:
        bits.append("clipping %.1f%%" % clip_pct)
    if not drummer_covered:
        bits.append("drum-uncovered")
    note = ", ".join(bits)

    return {
        "song_index": idx,
        "score": round(pts, 1),
        "dur": round(dur, 1),
        "has_solo": has_solo,
        "n_complete_choruses": n_complete,
        "has_build": has_build,
        "clip_pct": round(clip_pct, 3),
        "snr": round(snr, 1),
        "drummer_covered": drummer_covered,
        "arousal_peak": round(ar_range, 1),
        "note": note,
        # extra context (not in the required schema but handy)
        "_q_unit": round(q_unit, 2),
        "_dyn_unit": round(dyn_unit, 2),
        "_dur_unit": round(dur_unit, 2),
        "_n_choruses_total": len(choruses),
        "_arousal_peak_pos": round(ar_peak_pos, 2),
    }


# ======================================================================= structure ensure
def _ensure_map(work, idx, songs_meta, do_structure):
    """Ensure vocal_structure_song<idx>.json exists. Reuse cached; else run structure2 (which reuses
    cached demucs stems). Returns True if a map is available."""
    smap = os.path.join(work, "vocal_structure_song%d.json" % idx)
    if os.path.exists(smap):
        return True
    if not do_structure:
        return False
    wav = os.path.join(work, "song_%02d.wav" % idx)
    if not os.path.exists(wav):
        print("  [song %d] wav missing (%s) -- skip" % (idx, os.path.basename(wav)))
        return False
    abs_start = 0.0
    for m in songs_meta:
        if m.get("index") == idx:
            abs_start = m.get("start", 0.0)
            break
    stem_voc = os.path.join(work, "_stems", "htdemucs", "song%d" % idx, "vocals.wav")
    cached_stems = os.path.exists(stem_voc)
    print("  [song %d] building fine map (demucs stems %s) ..."
          % (idx, "CACHED reuse" if cached_stems else "SEPARATING first run"), flush=True)
    import structure2  # noqa: E402  (heavy import deferred to when a map is actually needed)
    res = structure2.analyze_song(wav, work, "song%d" % idx, abs_start=abs_start)
    json.dump(res["sections"], open(smap, "w"), indent=2)
    return True


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser(description="Rank songs by reel-candidate potential.")
    ap.add_argument("workdir")
    ap.add_argument("--json", default=None, help="write the ranked list JSON here")
    ap.add_argument("--no-structure", action="store_true",
                    help="do NOT build missing maps; score only songs whose map already exists")
    a = ap.parse_args()
    work = a.workdir

    songs_json = os.path.join(work, "songs.json")
    if not os.path.exists(songs_json):
        print("ERROR: no songs.json in", work); sys.exit(2)
    sm = json.load(open(songs_json))
    songs_meta = sm.get("songs", sm) if isinstance(sm, dict) else sm

    # drummer-cam geometry from analysis.json (second-ranked angle + its master->drummer sync offset)
    drum_off, drum_dur = 0.0, 0.0
    ana_p = os.path.join(work, "analysis.json")
    if os.path.exists(ana_p):
        A = json.load(open(ana_p))
        meta = A.get("_meta", {})
        ranking = meta.get("ranking", [])
        master = meta.get("master")
        second = ranking[1] if len(ranking) > 1 else master
        offs = meta.get("sync_offset_refined") or meta.get("sync_offset") or {}
        if second and second != master and second in A:
            drum_off = float(offs.get(second, 0.0))
            drum_dur = float(A[second].get("dur", 0.0))

    print("=" * 96)
    print("  SONG REEL-CANDIDATE SCORING  (workdir: %s)" % work)
    print("  drummer clip: dur=%.1fs  sync_offset=+%.2fs  (covers a song if song_end+off+%.0f <= dur)"
          % (drum_dur, drum_off, DRUMMER_TAIL_PAD))
    print("=" * 96)

    rows = []
    for m in songs_meta:
        idx = m.get("index")
        if idx is None:
            continue
        if not _ensure_map(work, idx, songs_meta, not a.no_structure):
            print("  [song %d] no fine map -- skipped" % idx)
            continue
        secs, chorus_prog = _load_map(work, idx)
        if not secs:
            print("  [song %d] empty map -- skipped" % idx)
            continue
        wav = os.path.join(work, "song_%02d.wav" % idx)
        if not os.path.exists(wav):
            print("  [song %d] wav missing -- skipped" % idx)
            continue
        rows.append(score_song(idx, wav, secs, chorus_prog, drum_off, drum_dur))

    rows.sort(key=lambda r: r["score"], reverse=True)

    # ---- ranked table ----
    print("\n" + "=" * 112)
    print("  REEL-CANDIDATE RANKING  (best first)")
    print("=" * 112)
    hdr = ("%-4s %-6s %-7s %-5s %-4s %-6s %-7s %-6s %-5s %-8s  %s"
           % ("rank", "song", "score", "dur", "solo", "#CC", "build", "clip%", "snr", "arousal", "note"))
    print(hdr)
    print("-" * 112)
    for rank, r in enumerate(rows, 1):
        print("%-4d %-6d %-7.1f %-5.0f %-4s %-6d %-7s %-6.2f %-5.1f %-8.1f  %s"
              % (rank, r["song_index"], r["score"], r["dur"],
                 "Y" if r["has_solo"] else "-", r["n_complete_choruses"],
                 "Y" if r["has_build"] else "-", r["clip_pct"], r["snr"],
                 r["arousal_peak"], r["note"]))
    print("-" * 112)
    print("  weights: solo=%.0f chorus_first=%.0f chorus_extra=%.0f(x<=%d) build=%.0f "
          "audio=%.0f dynamics=%.0f drummer=%.0f duration=%.0f  |  no-payoff cap=%.0f"
          % (W_SOLO, W_CHORUS_FIRST, W_CHORUS_EXTRA, CHORUS_EXTRA_CAP, W_BUILD,
             W_AUDIO_QUALITY, W_DYNAMICS, W_DRUMMER, W_DURATION, NO_PAYOFF_CAP))
    print("=" * 112)

    if a.json:
        json.dump(rows, open(a.json, "w"), indent=2, default=str)
        print("wrote:", a.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
