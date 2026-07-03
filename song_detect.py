# -*- coding: utf-8 -*-
"""Segment a full live-gig recording into individual songs.

Timbre-aware boundary detection (replaces the old raw energy-threshold split, which
broke on soft ballad intros and gapless / applause-hidden boundaries):

  1. MUSIC vs NON-MUSIC per frame.  The dominant discriminator on a live PA mix is
     BASS ENERGY (sub-band <160 Hz): songs carry kick/bass, while applause / talk /
     tuning are loud but bass-poor.  Loudness is folded in so a soft-but-loud passage
     (ballad intro, quiet verse) still reads as music, while loud-but-bass-poor
     applause reads as non-music.
  2. Smooth that likelihood, threshold with hysteresis, MERGE short internal dips
     (a quiet bar mid-song must not create a boundary) and enforce a MIN song length
     so nothing short survives.  -> coarse music regions.
  3. NOVELTY pass for gapless / near-gapless boundaries: any region long enough to
     hold >1 song is split at self-similarity novelty peaks over coarse timbre
     (MFCC) + key (chroma) features.  A wide checkerboard kernel is used so a
     sustained song-to-song timbre change wins over momentary internal breaks.
  4. Snap each boundary to the surrounding low-bass valley (song end = last bass-full
     frame before the dip; next start = where music sustains again).

The ground-truth timings are NOT encoded anywhere here; they were only used offline
to validate the thresholds below.

Config-driven: tool paths come from paths.py / config.json (gitignored).  No
per-machine paths and no band identifier are baked into this file.

Usage: python song_detect.py <ffmpeg> <workdir> <master_video>
Outputs songs.json {source,total_duration,songs:[{index,start,end,duration,...}]}
and one song_%02d.wav per detected song.
"""
import sys, os, json, subprocess
import numpy as np
import soundfile as sf
import librosa

# ---- CLI (ffmpeg passed explicitly by the pipeline; fall back to paths.py) ----
try:
    from paths import FFMPEG as _CFG_FFMPEG
except Exception:  # pragma: no cover - fresh clone without paths on sys.path
    _CFG_FFMPEG = "ffmpeg"

FFMPEG = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else _CFG_FFMPEG
WORK = sys.argv[2]
SRC = sys.argv[3]
SR = 22050
LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'song_library.json')

os.makedirs(WORK, exist_ok=True)

# ======================================================================
# Tunable detection parameters (calibrated on a full 40-min 9-song gig).
# ======================================================================
FPS = 10.0            # analysis frame rate (0.1 s frames)
SMOOTH_BASS = 6.0     # s, smoothing of the bass ratio
SMOOTH_ML = 4.0       # s, smoothing of the combined music-likelihood
RMS_BOOST = 0.25      # loudness contribution to music-likelihood
MUSIC_THR = 0.40      # music-likelihood threshold (region detection)
EDGE_THR = 0.28       # looser threshold to grow region edges (soft intros/outros)
MIN_MUSIC = 8.0       # s, drop music blips shorter than this
MERGE_GAP = 22.0      # s, fill non-music dips shorter than this (internal bars)
MIN_SONG = 120.0      # s, a real song is at least this long
TYP_SONG = 230.0      # s, typical song length (used to estimate #songs / region)
MIN_SEG = 150.0       # s, minimum separation between novelty boundaries
NOV_L = 32            # novelty checkerboard half-width (s) -- wide = song-scale
GAP_THR = 0.42        # bass level considered "music present" for the next-song onset
END_THR = 0.50        # bass level marking the end of the outgoing song
HOLD = 20             # s the incoming song must sustain music before its start counts


def smooth(v, sec):
    k = int(sec * FPS)
    k += 1 - (k % 2)                       # force odd
    if k < 1:
        return v
    return np.convolve(v, np.ones(k) / k, mode='same')


def runs_of(mask):
    """Return [start,end,value] runs of a boolean array."""
    out = []
    prev = None
    st = 0
    for i in range(len(mask)):
        if mask[i] != prev:
            if prev is not None:
                out.append([st, i, prev])
            st = i
            prev = mask[i]
    out.append([st, len(mask), prev])
    return out


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
print("audio: %.0fs (%.1f min)" % (total_dur, total_dur / 60), flush=True)

# ======================================================================
# 1. Per-frame features
# ======================================================================
hop = int(round(SR / FPS))                 # 0.1 s hop -> 10 fps
n_fft = 4096
S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)) ** 2   # power spectrogram
freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
total_energy = S.sum(0) + 1e-10

sub_ratio = S[freqs < 160].sum(0) / total_energy               # bass energy fraction
rms_db = librosa.power_to_db(S.sum(0) + 1e-10)

# ======================================================================
# 2. Music-likelihood = bass, boosted where the signal is loud (rescues soft-but-
#    loud passages) but only when SOME bass is present (applause stays low).
# ======================================================================
bass = smooth(sub_ratio, SMOOTH_BASS)
rlo, rhi = np.percentile(rms_db, 10), np.percentile(rms_db, 95)
rms_n = smooth(np.clip((rms_db - rlo) / (rhi - rlo + 1e-9), 0, 1), SMOOTH_BASS)
ml = smooth(bass + RMS_BOOST * rms_n * (bass > 0.12), SMOOTH_ML)   # 10 fps
ml1 = ml[::int(round(FPS))]                                        # 1 fps view

# ======================================================================
# 3. Coarse music regions
# ======================================================================
music = ml > MUSIC_THR
for r in runs_of(music):                                          # kill short blips
    if r[2] and (r[1] - r[0]) < MIN_MUSIC * FPS:
        music[r[0]:r[1]] = False
for r in runs_of(music):                                          # fill internal dips
    if (not r[2]) and (r[1] - r[0]) < MERGE_GAP * FPS:
        music[r[0]:r[1]] = True
regions = [[r[0] / FPS, r[1] / FPS] for r in runs_of(music)
           if r[2] and (r[1] - r[0]) >= MIN_SONG * FPS]

# grow region edges outward while still music-ish (recover soft intros/outros)
for R in regions:
    s = int(R[0])
    while s > 0 and ml1[s - 1] > EDGE_THR:
        s -= 1
    e = int(R[1])
    while e < len(ml1) - 1 and ml1[e] > EDGE_THR:
        e += 1
    R[0], R[1] = s, e
# the very first song often opens with a soft intro riding just under EDGE_THR;
# extend its start back to where sound first rises above the silence floor.
if regions:
    s = int(regions[0][0])
    while s > 0 and rms_n[int((s - 1) * FPS)] > 0.25:
        s -= 1
    regions[0][0] = s

print("music regions: " + " ".join("%.0f-%.0f" % (s, e) for s, e in regions), flush=True)

# ======================================================================
# 4. Novelty features (coarse timbre + key at 1 s) for splitting merged regions
# ======================================================================
hop1 = SR                                                         # 1 s frames
mfcc = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=13, hop_length=hop1, n_fft=n_fft)
chroma = librosa.feature.chroma_cqt(y=x, sr=SR, hop_length=hop1)


def _zn(M):
    return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)


Fmat = np.vstack([_zn(mfcc[1:]), _zn(chroma)]).T                  # drop mfcc0 (energy)
Fn = Fmat / (np.linalg.norm(Fmat, axis=1, keepdims=True) + 1e-9)
SSM = Fn @ Fn.T                                                   # cosine self-similarity
T = Fmat.shape[0]


def novelty(L):
    g = np.arange(-L, L + 1)
    gauss = np.exp(-(g[:, None] ** 2 + g[None, :] ** 2) / (2 * (L / 2) ** 2))
    kernel = gauss * np.outer(np.sign(g + 0.5), np.sign(g + 0.5))  # checkerboard
    nov = np.zeros(T)
    for i in range(T):
        a = max(0, i - L)
        b = min(T, i + L + 1)
        ka = a - (i - L)
        kb = ka + (b - a)
        nov[i] = np.sum(SSM[a:b, a:b] * kernel[ka:kb, ka:kb])
    return np.maximum(nov, 0)


NOV = novelty(NOV_L)


def _sustained_music(t):
    seg = ml1[t:t + HOLD]
    return len(seg) > 0 and np.all(seg >= GAP_THR)


def find_gap(anchor, s, e):
    """Given a novelty anchor inside [s,e], return (song_end, next_start).

    song_end  = last bass-full frame before the bass valley nearest the anchor.
    next_start= first frame after which music sustains (skips mid-gap flourishes /
                deeper later valleys, so a long applause gap is fully skipped).
    """
    lo = max(s + 15, anchor - 18)
    hi = min(e - 15, anchor + 18)
    if hi <= lo:
        lo = max(s + 5, anchor - 18)
        hi = min(e - 5, anchor + 18)
    vc = lo + int(np.argmin(ml1[lo:hi]))
    a = vc
    while a > s and ml1[a] < END_THR:
        a -= 1
    song_end = a + 1
    b = song_end
    while b < e - HOLD and not _sustained_music(b):
        b += 1
    return song_end, b


def split_region(s, e):
    length = e - s
    n = max(1, int(round(length / TYP_SONG)))
    if n <= 1:
        return [(s, e)]
    score = NOV[int(s):int(e)].copy()
    picks = []
    for idx in np.argsort(score)[::-1]:
        t = s + idx
        if all(abs(t - p) > MIN_SEG for p in picks) and (t - s) > MIN_SEG and (e - t) > MIN_SEG:
            picks.append(t)
        if len(picks) >= n - 1:
            break
    picks = sorted(picks)
    segs = []
    prev = s
    for c in picks:
        gs, ge = find_gap(c, prev, e)
        segs.append((prev, gs))
        prev = ge
    segs.append((prev, e))
    return segs


boundaries = []
for s, e in regions:
    boundaries.extend(split_region(s, e))
boundaries.sort()

print("detected %d songs:" % len(boundaries), flush=True)
for i, (s, e) in enumerate(boundaries):
    print("  song %d: %.1fs - %.1fs (%.0fs)" % (i + 1, s, e, e - s), flush=True)

# ======================================================================
# 5. Write per-song WAVs
# ======================================================================
songs = []
for i, (s, e) in enumerate(boundaries):
    region_wav = os.path.join(WORK, "song_%02d.wav" % (i + 1))
    si, ei = int(s * SR), int(e * SR)
    sf.write(region_wav, x[si:ei], SR)
    songs.append({
        "index": i + 1,
        "start": round(float(s), 2),
        "end": round(float(e), 2),
        "duration": round(float(e - s), 2),
        "lyrics": [],
        "full_text": "",
        "title": None,
        "confidence": None,
    })
    print("  wrote %s (%.0fs)" % (os.path.basename(region_wav), e - s), flush=True)

# ======================================================================
# 6. OPTIONAL Whisper transcription + library title match.
#    Purely for naming -- it does NOT gate boundary detection and any failure here
#    is swallowed so the songs.json boundaries are always produced.
# ======================================================================
try:
    print("\nrunning Whisper transcription (titles only)...", flush=True)
    from faster_whisper import WhisperModel
    model = WhisperModel("base", device="cpu", compute_type="int8")
    for song in songs:
        region_wav = os.path.join(WORK, "song_%02d.wav" % song["index"])
        segments_gen, _ = model.transcribe(
            region_wav, language="en", beam_size=3, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000))
        lyrics = [{"start": round(seg.start, 2), "end": round(seg.end, 2),
                   "text": seg.text.strip()} for seg in segments_gen]
        song["lyrics"] = lyrics
        song["full_text"] = " ".join(l["text"] for l in lyrics).strip()
        print("  song %d lyrics: %s" % (song["index"], song["full_text"][:100]), flush=True)

    lib = json.load(open(LIB_PATH)) if os.path.exists(LIB_PATH) else {}

    def fuzzy_match(text, lib):
        tl = text.lower()
        best_title, best_score = None, 0
        for title, info in lib.items():
            keywords = info.get("keywords", [title.lower()])
            score = sum(1 for kw in keywords if kw.lower() in tl)
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
except Exception as exc:  # transcription is best-effort only
    print("  Whisper step skipped (%s: %s)" % (type(exc).__name__, exc), flush=True)

# ======================================================================
# 7. Save songs.json
# ======================================================================
out = os.path.join(WORK, "songs.json")
json.dump({"source": SRC, "total_duration": round(total_dur, 2), "songs": songs},
          open(out, "w"), indent=2)
print("\nWROTE %s (%d songs)" % (out, len(songs)), flush=True)
