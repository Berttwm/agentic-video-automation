# -*- coding: utf-8 -*-
"""Musical-EVENT SPINE for Editor v2 -- the shared timeline every edit decision references.
Fuses: beat_this grid (beats.py) + spectral-flux ACCENTS(hits) + energy BUILDS/DROPS +
phrase RESOLUTIONS + ANY-time AROUSAL(energy). The editor snaps cuts to downbeats, ends clips at
resolutions, triggers effects on hits/drops, and scales effect density by arousal.
Usage:  from events import spine ;  ev = spine(wav_path)
"""
import os, subprocess, tempfile
import numpy as np
import soundfile as sf
import librosa
from beats import grid, snap

SR = 22050
HOP = 512
def _mad(x): return float(np.median(np.abs(x - np.median(x)))) + 1e-9

def _load(path):
    if path.lower().endswith('.wav'):
        y = sf.read(path)[0].astype(np.float32)
    else:
        from beats import FFMPEG
        w = os.path.join(tempfile.gettempdir(), os.path.basename(path) + '.ev.wav')
        subprocess.run([FFMPEG, '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(SR), '-y', w], check=True)
        y = sf.read(w)[0].astype(np.float32)
        try: os.remove(w)
        except Exception: pass
    return y.mean(1) if y.ndim > 1 else y

def spine(path):
    y = _load(path)
    dur = len(y) / SR
    g = grid(path)                                   # beats / downbeats / bpm (ML)
    beats = np.asarray(g['beats'], float); downbeats = np.asarray(g['downbeats'], float)

    # SPECTRAL-FLUX accents (hits) -> peak-pick, snap to nearest beat
    flux = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)   # spectral-flux based
    ft = librosa.times_like(flux, sr=SR, hop_length=HOP)
    pk = librosa.util.peak_pick(flux, pre_max=3, post_max=3, pre_avg=5, post_avg=5,
                                delta=_mad(flux) * 1.5, wait=int(SR / HOP * 0.25))
    hits = sorted({round(snap(float(ft[i]), beats), 3) for i in pk if flux[i] > np.median(flux) + 2 * _mad(flux)})

    # ENERGY envelope -> arousal, builds, drops, resolutions
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rt = librosa.times_like(rms, sr=SR, hop_length=HOP)
    rms_s = np.convolve(rms, np.ones(15) / 15, mode='same')
    arousal = rms_s / (rms_s.max() + 1e-9)
    d = np.diff(rms_s, prepend=rms_s[0])
    drop_idx = np.where(d > d.mean() + 2 * d.std())[0]
    drops = sorted({round(snap(float(rt[i]), downbeats if len(downbeats) else beats), 4) for i in drop_idx})
    # builds = ~1.5s rising ramp immediately preceding each drop
    builds = [(round(max(t - 1.5, 0), 2), round(t, 2)) for t in drops]
    # resolutions = local energy minima, snapped to a downbeat (clean cut / phrase-end points)
    from scipy.signal import argrelextrema
    mins = argrelextrema(rms_s, np.less, order=int(SR / HOP * 1.2))[0]
    resolutions = sorted({round(snap(float(rt[i]), downbeats if len(downbeats) else beats), 3) for i in mins})

    def arousal_at(t):
        i = int(np.clip(t / dur * (len(arousal) - 1), 0, len(arousal) - 1))
        return float(arousal[i])

    return {"dur": round(dur, 2), "bpm": g['bpm'], "beats": beats.tolist(), "downbeats": downbeats.tolist(),
            "hits": hits, "drops": drops, "builds": builds, "resolutions": resolutions,
            "arousal": [round(float(a), 3) for a in arousal[::4]], "arousal_hz": SR / HOP / 4,
            "_arousal_at": arousal_at}

if __name__ == "__main__":
    import sys, json
    ev = spine(sys.argv[1])
    print("dur=%.1fs bpm=%s | beats=%d downbeats=%d hits=%d drops=%d resolutions=%d"
          % (ev['dur'], ev['bpm'], len(ev['beats']), len(ev['downbeats']),
             len(ev['hits']), len(ev['drops']), len(ev['resolutions'])))
    print("drops:", ev['drops'][:8])
    print("resolutions:", ev['resolutions'][:8])
    print("hits(first8):", ev['hits'][:8])
