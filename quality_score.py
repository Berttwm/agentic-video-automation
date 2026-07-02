# -*- coding: utf-8 -*-
"""Score audio quality per section: SNR, dynamics, clipping, spectral balance.
Usage: python quality_score.py <workdir>
Reads sections.json + per-song WAVs, writes quality.json."""
import sys, os, json
import numpy as np
import soundfile as sf
import librosa

WORK = sys.argv[1]
SR = 22050

sections_data = json.load(open(os.path.join(WORK, "sections.json")))

def db(x): return 20 * np.log10(max(x, 1e-12))

def score_section(x):
    if len(x) < SR:
        return {"score": 0, "snr_db": 0, "crest_db": 0, "clip_pct": 0, "dynamics": 0}
    rms_val = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    clip = float(np.mean(np.abs(x) > 0.985)) * 100

    frame, hop = 2048, 1024
    n = max(1, 1 + (len(x) - frame) // hop)
    rms_frames = np.array([np.sqrt(np.mean(x[i*hop:i*hop+frame]**2)) for i in range(n)])
    rms_frames = rms_frames[rms_frames > 0]
    if len(rms_frames) < 2:
        return {"score": 0, "snr_db": 0, "crest_db": 0, "clip_pct": clip, "dynamics": 0}

    noise = np.percentile(rms_frames, 5)
    sig = np.percentile(rms_frames, 75)
    snr = db(sig) - db(noise)
    crest = db(peak) - db(rms_val)

    # onset regularity = rhythmic tightness (lower std of inter-onset intervals = tighter)
    onsets = librosa.onset.onset_detect(y=x, sr=SR, hop_length=512, units='time')
    if len(onsets) > 3:
        ioi = np.diff(onsets)
        rhythm_score = 1.0 / (1.0 + float(np.std(ioi) / (np.mean(ioi) + 1e-9)))
    else:
        rhythm_score = 0.5

    composite = (snr * 1.0 - clip * 3.0 + crest * 0.3 + rhythm_score * 10.0)

    return {
        "score": round(composite, 2),
        "snr_db": round(float(snr), 2),
        "crest_db": round(float(crest), 2),
        "clip_pct": round(clip, 4),
        "rhythm_tightness": round(rhythm_score, 3),
        "rms_db": round(db(rms_val), 2)
    }

results = []
for song_entry in sections_data:
    idx = song_entry["song_index"]
    wav_path = os.path.join(WORK, "song_%02d.wav" % idx)
    if not os.path.exists(wav_path):
        print("WARN: %s missing" % wav_path, flush=True)
        continue
    x_full, _ = sf.read(wav_path)
    x_full = x_full.astype(np.float32)

    # GATE thresholds: drop only genuinely BAD sections; keep everything else in temporal order.
    # (Per the user: a per-song edit keeps the WHOLE song minus true failures, not "top N".)
    FAIL_CLIP_PCT = 1.0      # audible clipping
    FAIL_RHYTHM = 0.30       # very loose/off timing
    FAIL_SNR_DB = 5.0        # buried / noisy

    scored_sections = []
    for sec in song_entry["sections"]:
        si = int(sec["start"] * SR)
        ei = int(sec["end"] * SR)
        x_sec = x_full[si:ei]
        q = score_section(x_sec)
        reasons = []
        if q.get("clip_pct", 0) > FAIL_CLIP_PCT:
            reasons.append("clipping %.2f%%" % q["clip_pct"])
        if q.get("rhythm_tightness", 1) < FAIL_RHYTHM:
            reasons.append("loose timing %.2f" % q.get("rhythm_tightness", 0))
        if q.get("snr_db", 99) < FAIL_SNR_DB:
            reasons.append("low SNR %.1fdB" % q.get("snr_db", 0))
        passed = len(reasons) == 0
        scored_sections.append({**sec, **q, "passed": passed, "gate_reasons": reasons})

    # keep TEMPORAL order (do NOT sort by score); compute pass/fail stats for the log
    n_pass = sum(1 for s in scored_sections if s["passed"])
    by_score = sorted(scored_sections, key=lambda s: s["score"], reverse=True)
    print("song %d (%s): %d sections, %d PASS / %d DROP, best=%s(%.1f) worst=%s(%.1f)" % (
        idx, song_entry.get("title") or "?",
        len(scored_sections), n_pass, len(scored_sections) - n_pass,
        by_score[0]["label"], by_score[0]["score"],
        by_score[-1]["label"], by_score[-1]["score"]
    ), flush=True)

    results.append({
        "song_index": idx,
        "title": song_entry.get("title"),
        "sections": scored_sections
    })

out = os.path.join(WORK, "quality.json")
json.dump(results, open(out, "w"), indent=2)
print("\nWROTE %s" % out, flush=True)
