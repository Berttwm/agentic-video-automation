# -*- coding: utf-8 -*-
"""Detect songs in a full gig recording via Whisper transcription + silence/energy segmentation.
Outputs songs.json: list of {title, start, end, confidence, lyrics} per detected song.
Usage: python song_detect.py <ffmpeg> <workdir> <master_video>"""
import sys, os, json, subprocess
import numpy as np
import soundfile as sf
import librosa

FFMPEG, WORK, SRC = sys.argv[1], sys.argv[2], sys.argv[3]
SR = 22050
LIB_PATH = os.path.join(os.path.dirname(__file__), 'song_library.json')

os.makedirs(WORK, exist_ok=True)

# ---- extract mono audio if not cached ----
base = os.path.splitext(os.path.basename(SRC))[0]
wav = os.path.join(WORK, base + ".ana.wav")
if not os.path.exists(wav):
    print("extracting audio...", flush=True)
    subprocess.run([FFMPEG, '-y', '-i', SRC, '-vn', '-ac', '1', '-ar', str(SR), wav],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
x, _ = sf.read(wav)
x = x.astype(np.float32)
total_dur = len(x) / SR
print("audio: %.0fs (%.1f min)" % (total_dur, total_dur/60), flush=True)

# ---- find song boundaries via energy segmentation ----
# songs have music; gaps between songs have low energy (talking, tuning, silence)
hop = 1024
rms = librosa.feature.rms(y=x, frame_length=4096, hop_length=hop)[0]
tf = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=hop)

# smooth RMS over ~3s windows
k = max(1, int(3.0 * SR / hop))
sm = np.convolve(rms, np.ones(k)/k, mode='same')

# threshold: adaptive — use percentile-based detection
# music typically occupies the louder portions of the recording
p25, p75 = np.percentile(sm, 25), np.percentile(sm, 75)
thresh = p25 + (p75 - p25) * 0.3
active = sm > thresh
print("energy thresh: p25=%.6f p75=%.6f thresh=%.6f" % (p25, p75, thresh), flush=True)

# find contiguous active regions (min 30s = a song, not just applause)
MIN_SONG = 30.0
MIN_GAP = 5.0
regions = []
in_region = False
start_i = 0
for i in range(len(active)):
    if active[i] and not in_region:
        start_i = i; in_region = True
    elif not active[i] and in_region:
        dur = tf[i] - tf[start_i]
        if dur >= MIN_SONG:
            regions.append((tf[start_i], tf[i]))
        in_region = False
if in_region:
    dur = tf[-1] - tf[start_i]
    if dur >= MIN_SONG:
        regions.append((tf[start_i], tf[-1]))

# merge regions separated by < MIN_GAP
merged = [regions[0]] if regions else []
for s, e in regions[1:]:
    if s - merged[-1][1] < MIN_GAP:
        merged[-1] = (merged[-1][0], e)
    else:
        merged.append((s, e))

print("detected %d song regions:" % len(merged), flush=True)
for i, (s, e) in enumerate(merged):
    print("  song %d: %.1fs - %.1fs (%.0fs)" % (i+1, s, e, e-s), flush=True)

# ---- Whisper transcription per song region ----
print("\nrunning Whisper transcription...", flush=True)
from faster_whisper import WhisperModel
model = WhisperModel("base", device="cpu", compute_type="int8")

songs = []
# first write all song WAVs
for i, (s, e) in enumerate(merged):
    region_wav = os.path.join(WORK, "song_%02d.wav" % (i+1))
    si, ei = int(s * SR), int(e * SR)
    sf.write(region_wav, x[si:ei], SR)
    print("  wrote %s (%.0fs)" % (os.path.basename(region_wav), e-s), flush=True)

# then transcribe each
for i, (s, e) in enumerate(merged):
    region_wav = os.path.join(WORK, "song_%02d.wav" % (i+1))
    print("  transcribing song %d..." % (i+1), flush=True)
    segments_gen, info = model.transcribe(region_wav, language="en", beam_size=3,
                                          vad_filter=True, vad_parameters=dict(min_silence_duration_ms=1000))
    lyrics = []
    for seg in segments_gen:
        lyrics.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()})

    full_text = " ".join(l["text"] for l in lyrics).strip()
    print("  song %d lyrics preview: %s" % (i+1, full_text[:120]), flush=True)

    songs.append({
        "index": i+1,
        "start": round(s, 2),
        "end": round(e, 2),
        "duration": round(e - s, 2),
        "lyrics": lyrics,
        "full_text": full_text,
        "title": None,
        "confidence": None
    })

# ---- match against song library ----
if os.path.exists(LIB_PATH):
    lib = json.load(open(LIB_PATH))
else:
    lib = {}

def fuzzy_match(text, lib):
    text_lower = text.lower()
    best_title, best_score = None, 0
    for title, info in lib.items():
        keywords = info.get("keywords", [title.lower()])
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > best_score:
            best_score, best_title = score, title
    if best_score >= 1:
        return best_title, min(best_score / 3.0, 1.0)
    return None, 0.0

for song in songs:
    title, conf = fuzzy_match(song["full_text"], lib)
    if title:
        song["title"] = title
        song["confidence"] = round(conf, 2)
        print("  song %d matched: '%s' (conf=%.2f)" % (song["index"], title, conf), flush=True)
    else:
        print("  song %d: no library match (new song?)" % song["index"], flush=True)

# ---- save ----
out = os.path.join(WORK, "songs.json")
json.dump({"source": SRC, "total_duration": round(total_dur, 2), "songs": songs}, open(out, "w"), indent=2)
print("\nWROTE %s (%d songs)" % (out, len(songs)), flush=True)

# ---- update library with new songs (unmatched get placeholder entries) ----
updated = False
for song in songs:
    if not song["title"] and song["full_text"]:
        key = "unknown_song_%d" % song["index"]
        if key not in lib:
            lib[key] = {"keywords": [], "sample_lyrics": song["full_text"][:200],
                        "note": "auto-detected, needs manual title assignment"}
            updated = True
if updated:
    json.dump(lib, open(LIB_PATH, "w"), indent=2)
    print("updated song_library.json with %d new placeholder entries" % sum(1 for s in songs if not s["title"]))
