# -*- coding: utf-8 -*-
"""Beat/downbeat GRID via beat_this (ML, 2024 SOTA). Shared by build_style_model + the editor.
Gives an accurate downbeat grid to quantize cuts/effects to (replaces the kick-phase heuristic).
Usage: from beats import grid ;  g = grid(path)  ->  {'beats':[...], 'downbeats':[...], 'bpm':..}"""
import os, subprocess, tempfile, functools
import numpy as np
from paths import FFMPEG

@functools.lru_cache(maxsize=1)
def _f2b():
    from beat_this.inference import File2Beats
    return File2Beats(checkpoint_path="final0", device="cpu", dbn=False)

def _as_wav(path):
    if path.lower().endswith('.wav'):
        return path, False
    w = os.path.join(tempfile.gettempdir(), os.path.basename(path) + '.beats.wav')
    subprocess.run([FFMPEG, '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', '22050', '-y', w], check=True)
    return w, True

def _librosa_grid(wav):
    """Fallback beat/downbeat grid using librosa when the beat_this ML model is unavailable.
    Beat-tracks the audio and derives 4/4 downbeats by phase-aligning to the strongest beat energy.
    Not as accurate as beat_this, but keeps the whole pipeline runnable on a machine without the model."""
    import soundfile as sf, librosa
    y, sr = sf.read(wav, always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050); sr = 22050
    _tempo, bframes = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, trim=False)
    beats = librosa.frames_to_time(bframes, sr=sr, hop_length=512)
    # downbeats: assume 4/4; pick the phase (0..3) whose beats carry the most onset energy.
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    ot = librosa.times_like(onset, sr=sr, hop_length=512)
    def _e(t):
        i = int(np.clip(np.argmin(np.abs(ot - t)), 0, len(onset) - 1)); return float(onset[i])
    best_ph, best = 0, -1.0
    for ph in range(4):
        e = float(np.mean([_e(beats[i]) for i in range(ph, len(beats), 4)])) if len(beats) > ph else 0.0
        if e > best:
            best, best_ph = e, ph
    downbeats = beats[best_ph::4]
    return np.asarray(beats, float), np.asarray(downbeats, float)


def grid(path, normalize_tempo=True):
    wav, tmp = _as_wav(path)
    try:
        try:
            beats, downbeats = _f2b()(wav)
        except ImportError:
            # beat_this (ML) not installed on this machine -> degrade to the librosa grid so the
            # events spine / effect inference still run. structure2 already uses librosa beats too.
            beats, downbeats = _librosa_grid(wav)
    finally:
        if tmp:
            try: os.remove(wav)
            except Exception: pass
    beats = np.asarray(beats, float); downbeats = np.asarray(downbeats, float)
    bpm = 60.0 / float(np.median(np.diff(beats))) if len(beats) > 2 else 0.0
    # beat_this sometimes locks to a subdivision (2x/4x). If bpm is very high, thin the beat grid
    # so "beat" ~ musical quarter-note; downbeats are kept as-is (they're the reliable bar anchors).
    if normalize_tempo and bpm > 180 and len(beats) > 4:
        beats = beats[::2]; bpm /= 2
    return {"beats": beats.tolist(), "downbeats": downbeats.tolist(), "bpm": round(bpm, 1)}

def snap(t, gridvals, max_dist=0.35):
    """snap a timestamp to the nearest grid value within max_dist; else return t unchanged."""
    if not len(gridvals):
        return t
    a = np.asarray(gridvals, float)
    i = int(np.argmin(np.abs(a - t)))
    return float(a[i]) if abs(a[i] - t) <= max_dist else t

if __name__ == "__main__":
    import sys, json
    print(json.dumps({k: (v[:8] if isinstance(v, list) else v) for k, v in grid(sys.argv[1]).items()}, indent=1))
