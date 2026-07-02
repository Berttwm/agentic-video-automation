# -*- coding: utf-8 -*-
"""Segment each detected song into verse/chorus/bridge sections via energy, spectral contrast, and repetition.
Usage: python section_segment.py <workdir>
Reads songs.json, writes sections.json."""
import sys, os, json
import numpy as np
import soundfile as sf
import librosa

WORK = sys.argv[1]
SR = 22050

songs_data = json.load(open(os.path.join(WORK, "songs.json")))
songs = songs_data["songs"]

all_sections = []

for song in songs:
    idx = song["index"]
    wav = os.path.join(WORK, "song_%02d.wav" % idx)
    if not os.path.exists(wav):
        print("WARN: %s missing, skip" % wav, flush=True)
        continue

    x, _ = sf.read(wav)
    x = x.astype(np.float32)
    dur = len(x) / SR
    print("song %d (%.0fs): segmenting..." % (idx, dur), flush=True)

    hop = 512
    # energy contour
    rms = librosa.feature.rms(y=x, frame_length=2048, hop_length=hop)[0]
    tf = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=hop)

    # spectral contrast (distinguishes verse=sparse from chorus=full)
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=hop))
    contrast = librosa.feature.spectral_contrast(S=S, sr=SR, hop_length=hop)
    contrast_mean = np.mean(contrast, axis=0)

    # smoothed energy novelty (section boundaries = big energy changes)
    k = max(1, int(2.0 * SR / hop))
    sm_rms = np.convolve(rms, np.ones(k)/k, mode='same')
    novelty = np.abs(np.gradient(sm_rms))

    # combined feature for boundary detection
    sm_contrast = np.convolve(contrast_mean, np.ones(k)/k, mode='same')
    contrast_nov = np.abs(np.gradient(sm_contrast))
    combined = novelty / (novelty.max() + 1e-9) + 0.5 * contrast_nov / (contrast_nov.max() + 1e-9)

    # beat tracking for snap
    tempo, beats = librosa.beat.beat_track(y=x, sr=SR, hop_length=hop, units='time')
    beats = np.asarray(beats, float)
    beats = beats[(beats > 2.0) & (beats < dur - 2.0)]
    def snap(t):
        if len(beats) == 0: return t
        return float(beats[np.argmin(np.abs(beats - t))])

    # peak-pick boundaries with min spacing ~8s
    MIN_SECTION = 8.0
    order = np.argsort(combined)[::-1]
    boundaries = [0.0]
    for ci in order:
        t = float(tf[ci])
        if t < 4.0 or t > dur - 4.0:
            continue
        if all(abs(t - b) >= MIN_SECTION for b in boundaries):
            boundaries.append(round(snap(t), 2))
        if len(boundaries) >= min(12, int(dur / 15)):
            break
    boundaries.append(round(dur, 2))
    boundaries = sorted(set(boundaries))

    # label sections by energy level relative to song mean
    mean_energy = float(np.mean(sm_rms))
    sections = []
    for j in range(len(boundaries) - 1):
        s, e = boundaries[j], boundaries[j+1]
        si, ei = np.searchsorted(tf, s), np.searchsorted(tf, e)
        seg_energy = float(np.mean(sm_rms[si:ei])) if ei > si else mean_energy
        seg_contrast = float(np.mean(contrast_mean[si:ei])) if ei > si else 0

        # heuristic labeling
        energy_ratio = seg_energy / (mean_energy + 1e-9)
        if energy_ratio > 1.3:
            label = "chorus"
        elif energy_ratio > 0.9:
            label = "verse"
        elif energy_ratio > 0.5:
            label = "bridge"
        else:
            label = "intro" if j == 0 else ("outro" if j == len(boundaries)-2 else "break")

        sections.append({
            "section_index": j+1,
            "label": label,
            "start": s,
            "end": e,
            "duration": round(e - s, 2),
            "energy": round(seg_energy, 6),
            "energy_ratio": round(energy_ratio, 2),
            "contrast": round(seg_contrast, 4),
            "start_absolute": round(song["start"] + s, 2),
            "end_absolute": round(song["start"] + e, 2)
        })

    print("  %d sections: %s" % (len(sections), [s["label"] for s in sections]), flush=True)
    all_sections.append({
        "song_index": idx,
        "title": song.get("title"),
        "song_start": song["start"],
        "song_end": song["end"],
        "sections": sections
    })

out = os.path.join(WORK, "sections.json")
json.dump(all_sections, open(out, "w"), indent=2)
print("\nWROTE %s" % out, flush=True)
