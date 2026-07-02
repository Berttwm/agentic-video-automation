# -*- coding: utf-8 -*-
"""Music STRUCTURE detection via Laplacian segmentation (McFee/librosa method).
Beat-synchronous CQT+MFCC -> combined recurrence(repetition)+sequence graph -> normalized
Laplacian -> spectral clustering into part-types. Finds REPEATED parts (same cluster label)
so the assembler can dedup repeats (keep the cleaner take) and arrange intentionally.
Works on the FULL song audio (not the arbitrary energy sections).
Usage: python structure_analyze.py <workdir>  -> writes structure.json"""
import sys, os, json
import numpy as np
import soundfile as sf
import librosa
import scipy
from scipy.cluster.vq import kmeans2

WORK = sys.argv[1]
SR = 22050
sections_data = json.load(open(os.path.join(WORK, "sections.json")))

def segment_song(y):
    # ---- beat-synchronous features ----
    C = np.abs(librosa.cqt(y=y, sr=SR, hop_length=512, bins_per_octave=12, n_bins=84))
    C = librosa.amplitude_to_db(C, ref=np.max)
    tempo, beats = librosa.beat.beat_track(y=y, sr=SR, hop_length=512, trim=False)
    if len(beats) < 8:
        return None
    Csync = librosa.util.sync(C, beats, aggregate=np.median)
    M = librosa.feature.mfcc(y=y, sr=SR, hop_length=512, n_mfcc=13)
    Msync = librosa.util.sync(M, beats)
    nb = Csync.shape[1]

    # ---- recurrence (repetition) graph on chroma-like CQT ----
    R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity', sym=True)
    from scipy.ndimage import median_filter
    Rf = librosa.segment.timelag_filter(median_filter)(R, size=(1, 7))

    # ---- local sequence (path) graph on timbre ----
    path_dist = np.sum(np.diff(Msync, axis=1) ** 2, axis=0)
    sigma = np.median(path_dist) + 1e-9
    path_sim = np.exp(-path_dist / sigma)
    R_path = np.diag(path_sim, 1) + np.diag(path_sim, -1)

    # ---- combine + normalized Laplacian ----
    deg_path = np.sum(R_path, axis=1); deg_rec = np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / (np.sum((deg_path + deg_rec) ** 2) + 1e-9)
    mu = float(np.clip(mu, 0.1, 0.9))
    A = mu * Rf + (1 - mu) * R_path
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    evals, evecs = scipy.linalg.eigh(L)

    # ---- choose k parts by song length; spectral-cluster beats ----
    dur = len(y) / SR
    k = int(min(7, max(3, round(dur / 22.0))))
    k = min(k, nb - 1)
    X = evecs[:, :k]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    _, labels = kmeans2(Xn, k, minit='++', seed=0, iter=50)

    # ---- beats -> time; merge contiguous same-label into segments ----
    bt = librosa.frames_to_time(beats, sr=SR, hop_length=512)
    bounds = np.concatenate([[0.0], bt, [dur]])
    seg_times = []  # (start,end,label)
    cur = labels[0]; start = bounds[0]
    for i in range(1, len(labels)):
        if labels[i] != cur:
            seg_times.append((start, bounds[i], int(cur)))
            start = bounds[i]; cur = labels[i]
    seg_times.append((start, bounds[len(labels)], int(cur)))
    # drop tiny segments (<4s) by merging into previous
    merged = []
    for s, e, lab in seg_times:
        if merged and (e - s) < 4.0:
            merged[-1] = (merged[-1][0], e, merged[-1][2])
        else:
            merged.append((s, e, lab))
    return merged, float(np.atleast_1d(tempo)[0])

def energy_of(y, s, e):
    seg = y[int(s * SR):int(e * SR)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0

out = []
for song in sections_data:
    idx = song["song_index"] or (sections_data.index(song) + 1)
    wav = os.path.join(WORK, "song_%02d.wav" % idx)
    if not os.path.exists(wav):
        continue
    y = sf.read(wav)[0].astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    res = segment_song(y)
    if not res:
        print("song %d: too short/uncertain" % idx); continue
    segs, tempo = res
    dur = len(y) / SR
    e_all = np.mean([energy_of(y, s, e) for s, e, _ in segs]) + 1e-9
    from collections import Counter
    labcnt = Counter(l for _, _, l in segs)
    # heuristic human labels: recurring high-energy=chorus; first=intro; last=outro;
    # non-recurring high-energy interior=solo/bridge; else verse
    recur = {l for l, c in labcnt.items() if c > 1}
    eseg = [(s, e, l, energy_of(y, s, e) / e_all) for s, e, l in segs]
    hi = sorted(eseg, key=lambda t: t[3], reverse=True)
    chorus_lab = next((l for _, _, l, _ in hi if l in recur), None)
    named = []
    for i, (s, e, l, er) in enumerate(eseg):
        if i == 0:
            nm = "intro"
        elif i == len(eseg) - 1:
            nm = "outro"
        elif l == chorus_lab:
            nm = "chorus"
        elif l in recur:
            nm = "verse"
        elif er > 1.15:
            nm = "solo"
        else:
            nm = "bridge"
        named.append({"start": round(s, 2), "end": round(e, 2), "dur": round(e - s, 2),
                      "cluster": l, "is_repeat": l in recur, "label": nm, "energy_ratio": round(er, 2)})
    out.append({"song_index": idx, "duration": round(dur, 1), "tempo": round(tempo, 1),
                "n_parts": len(named), "n_distinct": len(labcnt),
                "n_repeat_groups": sum(1 for c in labcnt.values() if c > 1), "parts": named})
    seq = " ".join("%s%s" % (p["label"][:3], "*" if p["is_repeat"] else "") for p in named)
    print("song %d (%.0fs, %.0fbpm): %d parts, %d distinct -> %s"
          % (idx, dur, tempo, len(named), len(labcnt), seq), flush=True)

json.dump(out, open(os.path.join(WORK, "structure.json"), "w"), indent=2)
print("WROTE structure.json  (* = repeated part)")
