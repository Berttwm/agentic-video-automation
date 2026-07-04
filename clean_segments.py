# -*- coding: utf-8 -*-
"""EXPERIMENTAL selective pitch/timing "cleanup" for the auto-editor.

>>> OFF BY DEFAULT. This is a research/experimental module. It is NOT wired into
>>> run_gig / the render pipeline. It only does something when you call it
>>> explicitly (CLI or import) with enable flags. Nothing here is applied to a
>>> reel unless you ask for it.

WHAT IT DOES (and does NOT do)
------------------------------
The gig audio is a full LIVE BAND MIX. You cannot pitch-correct one instrument
inside a mix. We use the CACHED Demucs stems (same cache vocal_structure.py
builds) to isolate the VOCAL, correct ONLY the isolated vocal stem for a few
clearly-wrong sustained notes, then REMIX corrected-vocal + the UNTOUCHED
no_vocals stem back to a full mix. The other stems are never altered.

  * VOCAL pitch cleanup  -> FEASIBLE (vocals stem is clean & monophonic).
  * VOCAL / hit timing   -> DETECT only by default (nudging attack timing on a
                            live mono take reliably enough to help is dicey;
                            correction is available but conservative & opt-in).
  * GUITAR pitch cleanup -> effectively INFEASIBLE and intentionally NOT done.
        The guitar lives in `no_vocals` mixed with bass+drums; a monophonic
        tracker locks onto the bass / glitches on drums, htdemucs_6s' guitar
        stem bleeds badly (and pitch-shifting a bleedy stem surfaces the
        artifacts), and live rhythm guitar is polyphonic (chords) which has no
        automatic corrector. We DETECT candidate guitar-ish infractions for
        REPORTING only, clearly flagged low-confidence, and never modify them.

DESIGN PRINCIPLES (why it should not over-correct)
--------------------------------------------------
  * CONSERVATIVE detection. A note is flagged ONLY when it SUSTAINS clearly off
    the nearest in-scale note. Vibrato / slides / short passing tones / grace
    notes are excluded by (a) a minimum sustained-duration gate, (b) requiring
    the note's *stable core* (not its onset/tail) to be off, and (c) rejecting
    notes whose pitch is still moving (slide/vibrato) rather than settled.
  * PARTIAL correction. We move the note only a FRACTION (strength, default 0.6)
    of the way toward the scale note -- a nudge, not a snap. No hard quantize.
  * FORMANT-PRESERVING. WORLD (pyworld): scale f0, keep spectral envelope (sp)
    and aperiodicity (ap) exactly -> formants unchanged, no "chipmunk". PSOLA
    (parselmouth) is available as an alternative engine for tiny shifts.
  * LOCAL. Only flagged windows are touched; each is EQUAL-POWER cross-faded
    back against the dry stem so the rest of the vocal is bit-for-bit original.

Thresholds were calibrated on the actual gig (song2 vocal stem): abs cents-off
the nearest semitone ran median ~21c, p90 ~41c, worst frame ~50c over a chorus,
so the default 45c note-level gate flags only genuine outliers, not the take.

DEPENDENCIES: numpy, scipy, soundfile, librosa (always). pyworld (correction
engine, has a cp312 win wheel -- no compiler). Optional: psola/parselmouth
(alt engine), beats.py (timing grid). Correction deps are imported LAZILY so
this module always IMPORTS and DETECTS even where they are absent. Config-driven
(paths.py); no device path / band handle / username is hardcoded.

CLI
---
  # DETECT + report only (default; nothing written to audio):
  python clean_segments.py detect <song_wav> <workdir> <song_key> [out.json]

  # CORRECT vocal + remix (opt-in), writing a new mix + before/after clips:
  python clean_segments.py correct <song_wav> <workdir> <song_key> \
        --out <mix_out.wav> [--engine world|psola] [--strength 0.6] \
        [--clips <dir>] [--sections a,b] [--apply-timing]

Public API
----------
  detect_infractions(song_wav, workdir, song_key, **kw) -> dict (report)
  correct_song(song_wav, workdir, song_key, out_wav, **kw)  -> dict (report)
"""
import sys, os, json, argparse
import numpy as np
import soundfile as sf
import librosa

# Reuse the editor's stem cache + config exactly as vocal_structure.py does.
from vocal_structure import separate_stems, SR  # SR == 22050

# --------------------------------------------------------------------------
# tuning defaults (all overridable). Calibrated on the actual gig audio.
# --------------------------------------------------------------------------
DEFAULTS = dict(
    # --- pitch detection ---
    fmin=80.0, fmax=1000.0,          # vocal f0 search range (Hz)
    reg_floor_hz=150.0,              # notes whose core sits below this are treated as bass bleed /
                                     # octave-halving errors, not the lead vocal -> never flagged (~D3)
    frame_length=2048, hop=512,      # pyin frames (default resolution -> ~65x faster)
    note_cents_gate=50.0,            # a note must be >= this many cents off the nearest CHROMATIC
                                     # semitone (median of its stable core) to count as an infraction
    min_note_dur=0.16,               # seconds; sustained-note gate (skip grace/passing notes)
    stable_frac=(0.25, 0.85),        # use only the middle of the note as its "stable core"
    max_core_drift=35.0,             # cents; if the core is still MOVING > this, it's a slide/vibrato -> skip
    min_voiced_frac=0.7,             # a note-region must be this voiced to be trusted
    vibrato_reject=True,             # reject notes whose core oscillates (vibrato) rather than sits off
    # --- correction ---
    engine="world",                  # 'world' (pyworld) or 'psola' (parselmouth)
    strength=0.6,                    # move this fraction toward the scale note (0..1). partial, not a snap.
    max_correct_cents=120.0,         # never move a note by more than this (safety; huge = probably mis-detection)
    xfade_ms=25.0,                   # equal-power crossfade of corrected window <-> dry stem
    frame_period=5.0,                # WORLD analysis/synth frame period (ms)
    # --- timing (detect always; correct only if apply_timing) ---
    onset_off_beat=0.09,             # seconds; onset this far from the nearest beat = "off-rhythm" (flag)
    timing_max_nudge=0.05,           # seconds; max attack time-nudge if timing correction is enabled
    apply_timing=False,
)

# 12-TET pitch classes, and Krumhansl-Schmuckler key profiles (major/minor).
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# scale-degree pitch classes relative to the tonic
_MAJ_SCALE = [0, 2, 4, 5, 7, 9, 11]
_MIN_SCALE = [0, 2, 3, 5, 7, 8, 10]   # natural minor


# --------------------------------------------------------------------------
# io helpers
# --------------------------------------------------------------------------
def _load_mono(path, sr=SR):
    y, file_sr = sf.read(path, always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y


def _pcorr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# --------------------------------------------------------------------------
# 1. key / scale estimate (Krumhansl-Schmuckler on the MIX chroma)
# --------------------------------------------------------------------------
def estimate_key(mix_y, sr=SR):
    """Return (tonic_pc, mode, scale_pcs, corr). Uses the full-mix chroma so it
    reflects the harmonic backing (guitar/bass), which is a more reliable scale
    reference than the vocal alone."""
    chroma = librosa.feature.chroma_cqt(y=mix_y, sr=sr).mean(axis=1)
    best = None
    for i in range(12):
        cm = _pcorr(chroma, np.roll(_KS_MAJ, i))
        cn = _pcorr(chroma, np.roll(_KS_MIN, i))
        for mode, c in (("maj", cm), ("min", cn)):
            if best is None or c > best[3]:
                scale = _MAJ_SCALE if mode == "maj" else _MIN_SCALE
                pcs = sorted(((i + s) % 12 for s in scale))
                best = (i, mode, pcs, c)
    return best


def _nearest_scale_midi(midi, scale_pcs):
    """Nearest in-scale MIDI note to a (float) midi value, and signed cents to it."""
    base = int(np.floor(midi))
    cands = []
    for octave in (base - 12, base, base + 12):
        oc = (octave // 12) * 12
        for pc in scale_pcs:
            cands.append(oc + pc)
    cands = np.array(sorted(set(cands)), dtype=float)
    j = int(np.argmin(np.abs(cands - midi)))
    target = cands[j]
    cents = (midi - target) * 100.0
    return target, cents


# --------------------------------------------------------------------------
# 2. f0 track -> note segmentation -> conservative infraction flags
# --------------------------------------------------------------------------
def track_f0(y, sr=SR, cfg=None):
    cfg = cfg or DEFAULTS
    f0, vflag, vprob = librosa.pyin(
        y, fmin=cfg["fmin"], fmax=cfg["fmax"], sr=sr,
        frame_length=cfg["frame_length"], hop_length=cfg["hop"])
    times = librosa.times_like(f0, sr=sr, hop_length=cfg["hop"])
    return f0, times, vflag, vprob


def _note_regions(f0, times, cfg):
    """Group consecutive voiced frames whose pitch is roughly stable into note
    regions. A new note starts on a voicing gap or a > ~0.8-semitone jump."""
    midi = librosa.hz_to_midi(f0)
    regions = []
    cur = None
    for i in range(len(f0)):
        voiced = not np.isnan(f0[i])
        if not voiced:
            if cur:
                regions.append(cur); cur = None
            continue
        if cur is None:
            cur = [i, i]
        else:
            prev_m = midi[cur[1]]
            if abs(midi[i] - prev_m) > 0.8:   # pitch jumped -> new note
                regions.append(cur); cur = [i, i]
            else:
                cur[1] = i
    if cur:
        regions.append(cur)
    return regions


def flag_pitch_infractions(f0, times, scale_pcs, cfg):
    """Conservative infraction flags.

    The TRIGGER is CHROMATIC off-ness -- how far the note's stable core sits from
    the nearest 12-TET semitone (the note the singer was aiming for). This is the
    correct definition of a pitch ERROR and, crucially, does NOT punish blue notes
    / chromatic passing tones sung in tune (they're ~0c off chromatic even though
    they're outside the diatonic scale). Flagging against the diatonic scale
    instead would flag every in-tune accidental -- exactly the over-correction we
    must avoid.

    A note is flagged ONLY when ALL hold:
      * sustained (>= min_note_dur) and well-voiced,
      * its stable CORE is >= note_cents_gate off the nearest CHROMATIC semitone,
      * the core is settled (not drifting/sliding) and not vibrato,
      * it sits in a plausible vocal register (>= reg_floor_hz) -> rejects
        sub-bass bleed / octave-halving errors.

    We record whether the aimed-at semitone is IN the estimated key. If it is
    out-of-key we mark `in_key=False` (advisory) but STILL only trigger on
    chromatic off-ness, so a cleanly-sung blue note is never "corrected".
    Returns a list of candidate-correction dicts.
    """
    midi = librosa.hz_to_midi(f0)
    hop_dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.023
    reg_floor = librosa.hz_to_midi(cfg["reg_floor_hz"])
    scale_set = set(scale_pcs)
    out = []
    for a, b in _note_regions(f0, times, cfg):
        dur = times[b] - times[a] + hop_dt
        if dur < cfg["min_note_dur"]:
            continue                                  # grace/passing note -> skip
        seg = midi[a:b + 1]
        voiced = ~np.isnan(seg)
        if voiced.mean() < cfg["min_voiced_frac"]:
            continue
        n = len(seg)
        lo = int(cfg["stable_frac"][0] * n); hi = max(lo + 1, int(cfg["stable_frac"][1] * n))
        core = seg[lo:hi]
        core = core[~np.isnan(core)]
        if len(core) < 3:
            continue
        core_med = float(np.median(core))
        if core_med < reg_floor:
            continue                                  # sub-bass -> bleed/octave error, not the vocal
        core_drift = float(np.percentile(core, 90) - np.percentile(core, 10)) * 100.0  # cents spread
        if core_drift > cfg["max_core_drift"]:
            continue                                  # still moving -> slide/expressive -> skip
        if cfg["vibrato_reject"] and _looks_like_vibrato(core):
            continue
        # --- CHROMATIC off-ness = the trigger ---
        nearest_semi = float(np.round(core_med))
        chroma_cents = (core_med - nearest_semi) * 100.0
        if abs(chroma_cents) < cfg["note_cents_gate"]:
            continue                                  # on a real note (in tune) -> leave it
        # correction target: the nearest IN-SCALE note if that is also the nearest
        # semitone (normal case); otherwise still pull to the nearest chromatic note
        # (never invent an out-of-scale->in-scale leap the singer didn't intend).
        target_midi = nearest_semi
        in_key = int(round(nearest_semi)) % 12 in scale_set
        out.append({
            "t_start": round(float(times[a]), 3),
            "t_end": round(float(times[a] + dur), 3),
            "dur": round(float(dur), 3),
            "f0_median_hz": round(float(librosa.midi_to_hz(core_med)), 2),
            "note_sung": _midi_name(core_med),
            "note_target": _midi_name(target_midi),
            "cents_off": round(float(chroma_cents), 1),   # signed, vs nearest chromatic note
            "in_key": bool(in_key),
            "core_drift_cents": round(core_drift, 1),
            "voiced_frac": round(float(voiced.mean()), 2),
            "region": [a, b],
        })
    return out


def _looks_like_vibrato(core, min_cycles=2, min_depth_cents=20.0):
    """Detect an oscillation around the mean (vibrato) vs a static offset. If the
    de-meaned core crosses zero a few times with meaningful depth, call it vibrato."""
    x = core - np.mean(core)
    depth = (np.percentile(core, 90) - np.percentile(core, 10)) * 100.0
    if depth < min_depth_cents:
        return False
    signs = np.sign(x); signs[signs == 0] = 1
    crossings = int(np.sum(signs[1:] != signs[:-1]))
    return crossings >= (2 * min_cycles)


def _midi_name(m):
    mr = int(round(m))
    return "%s%d" % (_NOTE_NAMES[mr % 12], mr // 12 - 1)


# --------------------------------------------------------------------------
# 3. rhythmic / timing infractions (onset vs beat grid). DETECT by default.
# --------------------------------------------------------------------------
def flag_timing_infractions(stem_y, sr, cfg, beat_times=None):
    """Flag vocal onsets that land clearly off the nearest beat. ADVISORY ONLY:
    off-rhythm is frequently intentional (syncopation, laid-back phrasing), so
    this is never auto-corrected by default -- it is reported for the human to
    judge. `beat_times` may be supplied (e.g. from beats.py's ML grid on the full
    mix); otherwise we beat-track the stem with librosa."""
    if beat_times is not None and len(beat_times) >= 4:
        beats = np.asarray(beat_times, float)
    else:
        _tempo, bframes = librosa.beat.beat_track(y=stem_y, sr=sr, hop_length=cfg["hop"], trim=False)
        beats = librosa.frames_to_time(bframes, sr=sr, hop_length=cfg["hop"])
    if beats is None or len(beats) < 4:
        return {"beats": 0, "infractions": []}
    onset_env = librosa.onset.onset_strength(y=stem_y, sr=sr, hop_length=cfg["hop"])
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                         hop_length=cfg["hop"], units="time", backtrack=True)
    beats = np.asarray(beats, float)
    infr = []
    for ot in onsets:
        j = int(np.argmin(np.abs(beats - ot)))
        d = float(ot - beats[j])
        if abs(d) >= cfg["onset_off_beat"]:
            infr.append({"t_onset": round(float(ot), 3),
                         "nearest_beat": round(float(beats[j]), 3),
                         "offset_s": round(d, 3),
                         "suggest_nudge_s": round(float(np.clip(-d, -cfg["timing_max_nudge"],
                                                                cfg["timing_max_nudge"])), 3)})
    return {"beats": int(len(beats)), "infractions": infr}


# --------------------------------------------------------------------------
# 4. CORRECTION engines (vocal stem only). Formant-preserving, partial.
# --------------------------------------------------------------------------
def _cents_to_ratio(cents):
    return 2.0 ** (cents / 1200.0)


def correct_stem_world(voc_y, sr, flags, cfg):
    """WORLD (pyworld) partial pitch nudge on flagged windows only. Returns a NEW
    stem: dry everywhere, corrected+crossfaded inside each flagged window.
    Formants preserved (sp/ap untouched); unvoiced frames untouched."""
    import pyworld as pw
    x = voc_y.astype(np.float64)
    fp = cfg["frame_period"]
    _f0, t = pw.harvest(x, sr, frame_period=fp)        # robust for live/low-SNR
    f0 = pw.stonemask(x, _f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    f0n = f0.copy()
    applied = []
    frame_t = t  # seconds per WORLD frame
    for fl in flags:
        move = -fl["cents_off"] * cfg["strength"]                 # partial move toward target
        move = float(np.clip(move, -cfg["max_correct_cents"], cfg["max_correct_cents"]))
        ratio = _cents_to_ratio(move)
        m = (frame_t >= fl["t_start"]) & (frame_t <= fl["t_end"]) & (f0n > 0)
        if not np.any(m):
            continue
        f0n[m] = f0[m] * ratio
        applied.append({**fl, "applied_cents": round(move, 1), "engine": "world"})
    y = pw.synthesize(f0n, sp, ap, sr, frame_period=fp).astype(np.float32)
    y = _match_len_gain(voc_y, y)
    out = _local_blend(voc_y, y, applied, sr, cfg)
    return out, applied


def correct_stem_psola(voc_y, sr, flags, cfg):
    """PSOLA (psola/parselmouth) alternative engine for small shifts. Each flagged
    window is corrected independently by building a per-frame target-f0 curve that
    equals the tracked f0 * a partial ratio, then TD-PSOLA re-pitches the slice.
    The result is equal-power crossfaded back into the dry stem (rest untouched)."""
    out = voc_y.astype(np.float32).copy()
    fmin, fmax = cfg["fmin"], cfg["fmax"]
    applied = []
    for fl in flags:
        s = int(fl["t_start"] * sr); e = int(fl["t_end"] * sr)
        s = max(0, s); e = min(len(voc_y), e)
        if e - s < int(0.05 * sr):
            continue
        move = -fl["cents_off"] * cfg["strength"]
        move = float(np.clip(move, -cfg["max_correct_cents"], cfg["max_correct_cents"]))
        ratio = _cents_to_ratio(move)
        seg = voc_y[s:e].astype(np.float64)
        try:
            shifted = _psola_shift(seg, sr, ratio, fmin, fmax)
        except Exception as ex:
            # psola can fail on breathy/short live slices -> skip that window, leave dry
            applied.append({**fl, "applied_cents": 0.0, "engine": "psola", "skipped": str(ex)[:60]})
            continue
        shifted = _match_len_gain(voc_y[s:e], np.asarray(shifted, dtype=np.float32))
        out[s:e] = _xfade_replace(out[s:e], shifted, sr, cfg["xfade_ms"])
        applied.append({**fl, "applied_cents": round(move, 1), "engine": "psola"})
    return out, applied


def _psola_shift(seg, sr, ratio, fmin, fmax):
    """Small pitch shift of a short segment: TD-PSOLA re-pitch toward a target f0
    curve = tracked_f0 * ratio (partial move already baked into `ratio`)."""
    import psola
    tracked = _track_for_psola(seg, sr, fmin, fmax)
    target = np.where(tracked > 0, tracked * ratio, tracked)
    return psola.vocode(seg, sample_rate=int(sr), target_pitch=target, fmin=fmin, fmax=fmax)


def _track_for_psola(seg, sr, fmin, fmax):
    import parselmouth
    snd = parselmouth.Sound(seg.astype(np.float64), sampling_frequency=float(sr))
    pitch = snd.to_pitch_ac(pitch_floor=float(fmin), pitch_ceiling=float(fmax))
    return np.asarray(pitch.selected_array['frequency'], dtype=np.float64)


# --------------------------------------------------------------------------
# blend / gain helpers
# --------------------------------------------------------------------------
def _match_len_gain(dry, wet):
    """Trim/pad wet to dry length and match RMS so the corrected region does not
    jump in level (WORLD synth can change amplitude)."""
    if len(wet) < len(dry):
        wet = np.pad(wet, (0, len(dry) - len(wet)))
    elif len(wet) > len(dry):
        wet = wet[:len(dry)]
    rd = float(np.sqrt(np.mean(dry.astype(np.float64) ** 2)) + 1e-9)
    rw = float(np.sqrt(np.mean(wet.astype(np.float64) ** 2)) + 1e-9)
    g = min(4.0, rd / rw)
    return (wet * g).astype(np.float32)


def _equal_power(n):
    tp = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    return np.cos(tp), np.sin(tp)   # fade-out, fade-in (constant power)


def _xfade_replace(dry_win, wet_win, sr, xfade_ms):
    """Replace dry_win with wet_win but equal-power crossfade the first/last
    xfade_ms so there is no click at the window edges."""
    n = len(dry_win)
    xf = min(int(xfade_ms * 1e-3 * sr), n // 2)
    out = wet_win.astype(np.float32).copy()
    if xf > 0:
        fo, fi = _equal_power(xf)   # fo: cos (1->0), fi: sin (0->1); equal power
        # leading edge: dry fades OUT, wet fades IN
        out[:xf] = dry_win[:xf] * fo + wet_win[:xf] * fi
        # trailing edge: wet fades OUT, dry fades IN
        out[-xf:] = wet_win[-xf:] * fo + dry_win[-xf:] * fi
    return out


def _local_blend(dry, wet, applied, sr, cfg):
    """For a WORLD full-resynth `wet`, keep `dry` everywhere except inside each
    applied window, where we equal-power crossfade wet<->dry at the edges."""
    out = dry.astype(np.float32).copy()
    for fl in applied:
        s = int(fl["t_start"] * sr); e = int(fl["t_end"] * sr)
        s = max(0, s); e = min(len(dry), e)
        if e - s < 8:
            continue
        out[s:e] = _xfade_replace(dry[s:e], wet[s:e], sr, cfg["xfade_ms"])
    return out


# --------------------------------------------------------------------------
# 5. remix corrected vocal + untouched no_vocals
# --------------------------------------------------------------------------
def remix(corrected_voc, workdir, song_key, sr=SR, model_name="htdemucs"):
    """Sum the corrected vocal stem with the UNTOUCHED no_vocals stem. Because the
    editor's stems are demucs (vocals + no_vocals), corrected_voc + no_vocals
    reconstructs the full mix with only the vocal changed."""
    stem_dir = os.path.join(workdir, "_stems", model_name, song_key)
    acc = _load_mono(os.path.join(stem_dir, "no_vocals.wav"), sr)
    n = min(len(corrected_voc), len(acc))
    mix = corrected_voc[:n].astype(np.float32) + acc[:n].astype(np.float32)
    peak = float(np.max(np.abs(mix))) or 1.0
    if peak > 0.999:
        mix = mix / peak * 0.999
    return mix


# --------------------------------------------------------------------------
# 6. top-level: detect / correct
# --------------------------------------------------------------------------
def _merge_cfg(**kw):
    cfg = dict(DEFAULTS)
    for k, v in kw.items():
        if v is not None and k in cfg:
            cfg[k] = v
    return cfg


def detect_infractions(song_wav, workdir, song_key, model_name="htdemucs",
                       sections=None, **kw):
    """DETECT only. Returns a report dict; writes nothing to audio."""
    cfg = _merge_cfg(**kw)
    voc_path, acc_path = separate_stems(song_wav, workdir, song_key, model_name)
    voc_y = _load_mono(voc_path)
    mix_y = _load_mono(song_wav)
    tonic, mode, scale_pcs, kcorr = estimate_key(mix_y)
    f0, times, vflag, vprob = track_f0(voc_y, cfg=cfg)
    pitch_flags = flag_pitch_infractions(f0, times, scale_pcs, cfg)
    # timing grid: prefer the editor's shared beat grid on the FULL MIX (reliable);
    # fall back to a librosa track on the vocal stem inside flag_timing_infractions.
    beat_times = None
    try:
        from beats import grid
        beat_times = grid(song_wav).get("beats")
    except Exception:
        beat_times = None
    timing = flag_timing_infractions(voc_y, SR, cfg, beat_times=beat_times)
    if sections:
        pitch_flags = [f for f in pitch_flags if _in_any(f["t_start"], sections)]
    voiced = f0[~np.isnan(f0)]
    cents_stat = {}
    if len(voiced):
        m = librosa.hz_to_midi(voiced)
        c = np.abs((m - np.round(m)) * 100.0)
        cents_stat = {"median": round(float(np.median(c)), 1),
                      "p90": round(float(np.percentile(c, 90)), 1),
                      "max": round(float(np.max(c)), 1)}
    return {
        "song_key": song_key,
        "wav": os.path.basename(song_wav),
        "duration_s": round(len(mix_y) / SR, 1),
        "key": {"tonic": _NOTE_NAMES[tonic], "mode": mode,
                "scale": [_NOTE_NAMES[p] for p in scale_pcs], "corr": round(kcorr, 3)},
        "vocal_cents_offness": cents_stat,
        "thresholds": {"note_cents_gate": cfg["note_cents_gate"],
                       "min_note_dur": cfg["min_note_dur"],
                       "onset_off_beat": cfg["onset_off_beat"]},
        "pitch_infractions": pitch_flags,
        "n_pitch_infractions": len(pitch_flags),
        "timing": {"n_off_beat_onsets": len(timing["infractions"]),
                   "detail": timing["infractions"]},
        "guitar_note": ("Guitar pitch cleanup is intentionally NOT performed: the "
                        "guitar is mixed with bass+drums in no_vocals; monophonic "
                        "tracking is unreliable there and rhythm guitar is "
                        "polyphonic. Detect/correct here covers the VOCAL only."),
    }


def correct_song(song_wav, workdir, song_key, out_wav, model_name="htdemucs",
                 sections=None, clips_dir=None, **kw):
    """DETECT then CORRECT the vocal stem (opt-in) and remix. Writes the new mix
    to out_wav. If clips_dir is given, also writes short BEFORE/AFTER clips around
    each applied correction so you can A/B exactly where cleanup fired."""
    cfg = _merge_cfg(**kw)
    rep = detect_infractions(song_wav, workdir, song_key, model_name, sections=sections, **kw)
    voc_path, acc_path = separate_stems(song_wav, workdir, song_key, model_name)
    voc_y = _load_mono(voc_path)
    flags = rep["pitch_infractions"]

    if cfg["engine"] == "psola":
        corrected_voc, applied = correct_stem_psola(voc_y, SR, flags, cfg)
    else:
        corrected_voc, applied = correct_stem_world(voc_y, SR, flags, cfg)

    # optional (opt-in) timing nudge of the corrected vocal stem
    timing_applied = []
    if cfg["apply_timing"] and rep["timing"]["detail"]:
        corrected_voc, timing_applied = _apply_timing_nudges(
            corrected_voc, SR, rep["timing"]["detail"], cfg)

    mix = remix(corrected_voc, workdir, song_key, SR, model_name)
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    sf.write(out_wav, mix, SR, subtype="PCM_16")

    # dry remix (original vocal + acc) for a fair full-mix A/B baseline
    dry_mix = remix(voc_y, workdir, song_key, SR, model_name)

    clips = []
    if clips_dir and applied:
        os.makedirs(clips_dir, exist_ok=True)
        for i, fl in enumerate(applied):
            pad = 0.6
            s = max(0.0, fl["t_start"] - pad); e = fl["t_end"] + pad
            a = int(s * SR); b = int(e * SR)
            before = dry_mix[a:min(b, len(dry_mix))]
            after = mix[a:min(b, len(mix))]
            bn = os.path.join(clips_dir, "%s_fix%02d_%.1fs_%s_to_%s_BEFORE.wav" %
                              (song_key, i, fl["t_start"], fl["note_sung"], fl["note_target"]))
            an = bn.replace("BEFORE.wav", "AFTER.wav")
            sf.write(bn, before, SR, subtype="PCM_16")
            sf.write(an, after, SR, subtype="PCM_16")
            clips.append({"fix": i, **{k: fl[k] for k in ("t_start", "t_end", "note_sung",
                          "note_target", "cents_off", "applied_cents")},
                          "before": os.path.basename(bn), "after": os.path.basename(an)})

    rep["correction"] = {
        "engine": cfg["engine"], "strength": cfg["strength"],
        "n_applied": len(applied), "applied": applied,
        "timing_applied": timing_applied,
        "out_wav": os.path.abspath(out_wav),
        "clips_dir": os.path.abspath(clips_dir) if clips_dir else None,
        "clips": clips,
    }
    return rep


def _apply_timing_nudges(voc_y, sr, timing_detail, cfg):
    """VERY conservative attack time-nudge: for each flagged off-beat onset, shift
    a short window around the attack toward the beat by resampling that window.
    This is experimental and easily artifact-prone; off by default."""
    out = voc_y.astype(np.float32).copy()
    applied = []
    for d in timing_detail:
        nudge = d.get("suggest_nudge_s", 0.0)
        if abs(nudge) < 0.005:
            continue
        # window: from onset-0.05 to onset+0.20
        c = d["t_onset"]
        s = int(max(0, (c - 0.05)) * sr); e = int((c + 0.20) * sr)
        e = min(e, len(out))
        if e - s < int(0.05 * sr):
            continue
        seg = out[s:e]
        # time-shift by resampling seg to a slightly different length then pad/trim
        factor = 1.0 + (nudge / (len(seg) / sr))
        factor = float(np.clip(factor, 0.9, 1.1))
        shifted = librosa.effects.time_stretch(seg.astype(np.float32), rate=1.0 / factor)
        shifted = _match_len_gain(seg, shifted)
        out[s:e] = _xfade_replace(seg, shifted, sr, cfg["xfade_ms"])
        applied.append({**d, "applied_nudge_s": round(nudge, 3)})
    return out, applied


def _in_any(t, sections):
    for a, b in sections:
        if a <= t <= b:
            return True
    return False


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_sections(s):
    if not s:
        return None
    out = []
    for part in s.split(";"):
        a, b = part.split(",")
        out.append((float(a), float(b)))
    return out


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description="EXPERIMENTAL selective vocal pitch/timing cleanup (off by default).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("detect", help="detect+report infractions; writes nothing to audio")
    pd.add_argument("song_wav"); pd.add_argument("workdir"); pd.add_argument("song_key")
    pd.add_argument("out_json", nargs="?", default=None)
    pd.add_argument("--sections", default=None, help='windows "a,b;c,d" (seconds) to restrict reporting')
    pd.add_argument("--note-cents-gate", type=float, default=None)
    pd.add_argument("--min-note-dur", type=float, default=None)

    pc = sub.add_parser("correct", help="detect + correct vocal + remix (opt-in)")
    pc.add_argument("song_wav"); pc.add_argument("workdir"); pc.add_argument("song_key")
    pc.add_argument("--out", required=True, help="output remixed wav")
    pc.add_argument("--engine", choices=["world", "psola"], default=None)
    pc.add_argument("--strength", type=float, default=None)
    pc.add_argument("--note-cents-gate", type=float, default=None)
    pc.add_argument("--min-note-dur", type=float, default=None)
    pc.add_argument("--clips", default=None, help="dir to write BEFORE/AFTER A/B clips")
    pc.add_argument("--sections", default=None, help='restrict corrections to windows "a,b;c,d"')
    pc.add_argument("--apply-timing", action="store_true")
    pc.add_argument("--report", default=None, help="write full JSON report here")

    args = ap.parse_args(argv)

    if args.cmd == "detect":
        rep = detect_infractions(
            args.song_wav, args.workdir, args.song_key,
            sections=_parse_sections(args.sections),
            note_cents_gate=args.note_cents_gate, min_note_dur=args.min_note_dur)
        out = args.out_json or os.path.join(args.workdir, "cleanup_detect_%s.json" % args.song_key)
        json.dump(rep, open(out, "w"), indent=2)
        _print_detect(rep)
        print("WROTE", out)
        return 0

    if args.cmd == "correct":
        rep = correct_song(
            args.song_wav, args.workdir, args.song_key, args.out,
            sections=_parse_sections(args.sections), clips_dir=args.clips,
            engine=args.engine, strength=args.strength,
            note_cents_gate=args.note_cents_gate, min_note_dur=args.min_note_dur,
            apply_timing=args.apply_timing)
        _print_detect(rep)
        c = rep["correction"]
        print("--- CORRECTION ---")
        print("engine=%s strength=%s  applied=%d  out=%s"
              % (c["engine"], c["strength"], c["n_applied"], c["out_wav"]))
        for cl in c["clips"]:
            print("  fix%02d %.1fs %s->%s  off=%.0fc applied=%.0fc  %s | %s"
                  % (cl["fix"], cl["t_start"], cl["note_sung"], cl["note_target"],
                     cl["cents_off"], cl["applied_cents"], cl["before"], cl["after"]))
        if args.report:
            json.dump(rep, open(args.report, "w"), indent=2)
            print("WROTE", args.report)
        return 0
    return 2


def _print_detect(rep):
    k = rep["key"]
    print("song=%s dur=%.0fs  KEY=%s %s (corr=%.2f)  scale=%s"
          % (rep["song_key"], rep["duration_s"], k["tonic"], k["mode"], k["corr"], ",".join(k["scale"])))
    co = rep.get("vocal_cents_offness", {})
    if co:
        print("vocal off-ness cents: median=%.0f p90=%.0f max=%.0f  (gate=%.0f)"
              % (co.get("median", 0), co.get("p90", 0), co.get("max", 0),
                 rep["thresholds"]["note_cents_gate"]))
    print("PITCH infractions: %d  (trigger = >=%.0fc off nearest CHROMATIC note)"
          % (rep["n_pitch_infractions"], rep["thresholds"]["note_cents_gate"]))
    for f in rep["pitch_infractions"]:
        print("  %6.2fs-%6.2fs dur=%.2f sung=%-4s target=%-4s off=%+.0fc drift=%.0fc voiced=%.0f%% in_key=%s"
              % (f["t_start"], f["t_end"], f["dur"], f["note_sung"], f["note_target"],
                 f["cents_off"], f["core_drift_cents"], f["voiced_frac"] * 100, f.get("in_key")))
    print("TIMING off-beat onsets: %d (advisory)" % rep["timing"]["n_off_beat_onsets"])


if __name__ == "__main__":
    sys.exit(main())
