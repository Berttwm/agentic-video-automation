# -*- coding: utf-8 -*-
"""Vocal-aware music STRUCTURE detection for the auto-editor.

Pipeline (config-driven, band/username-free, byte-compilable, drop-in):
  1. demucs source-separation (htdemucs, CPU) of a song wav into vocals / accompaniment
     stems. Stems are CACHED under <workdir>/_stems/<songkey>/ so re-runs are cheap.
     We call demucs via its Python apply API and save with soundfile, so we do NOT
     depend on torchaudio's torchcodec save backend (which needs a matching ffmpeg).
  2. VOCAL-ACTIVITY envelope: RMS of the VOCALS stem over ~0.5 s frames -> a
     vocal-presence curve. HIGH => singing (verse/pre-chorus/chorus); LOW =>
     instrumental (intro/solo/outro). This is the signal that tells a CHORUS
     (vocal) apart from a SOLO (lead instrument, little/no vocal).
  3. STRUCTURE boundaries + repetition: Laplacian self-similarity segmentation
     (McFee/librosa) on the accompaniment stem -> contiguous segments + which
     segments are REPEATS (same spectral-cluster id).
  4. LABEL each WHOLE segment by combining vocal-activity + repetition + energy +
     position, fitting the human template:
       intro -> verse -> pre-chorus(may merge) -> chorus -> SOLO ->
       post-solo verse -> post-solo chorus -> outro(may split in 2)

Public entry points:
  separate_stems(wav_path, workdir, song_key, model_name="htdemucs") -> (vocals_wav, accomp_wav)
  analyze_song(wav_path, workdir, song_key, abs_start=0.0, template=None) -> dict
  main() : CLI  `python vocal_structure.py <wav> <workdir> <song_key> [abs_start]`

Dependencies: numpy, scipy, soundfile, librosa, torch, demucs. Uses sys.executable's
interpreter; nothing here hardcodes a device path.
"""
import sys, os, json, hashlib
import numpy as np
import soundfile as sf
import librosa
import scipy
import scipy.linalg
import scipy.sparse.csgraph
from scipy.ndimage import median_filter
from scipy.cluster.vq import kmeans2
from collections import Counter

SR = 22050
VOCAL_FRAME = 0.5          # seconds per RMS frame for the vocal envelope
MIN_SEG = 4.0              # merge segments shorter than this (seconds)


# --------------------------------------------------------------------------
# 1. demucs separation (cached), saved via soundfile to dodge torchcodec
# --------------------------------------------------------------------------
def _load_mono(path, sr=SR):
    y, file_sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y


def separate_stems(wav_path, workdir, song_key, model_name="htdemucs"):
    """Return (vocals_wav_path, accompaniment_wav_path), computing + caching stems.

    Stems live in <workdir>/_stems/<model_name>/<song_key>/{vocals,no_vocals}.wav .
    If both already exist, demucs is NOT re-run.
    """
    stem_dir = os.path.join(workdir, "_stems", model_name, song_key)
    os.makedirs(stem_dir, exist_ok=True)
    voc_path = os.path.join(stem_dir, "vocals.wav")
    acc_path = os.path.join(stem_dir, "no_vocals.wav")
    if os.path.exists(voc_path) and os.path.exists(acc_path):
        return voc_path, acc_path

    # Lazy imports so the module still imports on machines without demucs/torch.
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    model = get_model(model_name)
    model.cpu().eval()
    model_sr = model.samplerate            # htdemucs => 44100
    model_ch = model.audio_channels        # => 2

    # Load at the model's native SR/channels for best separation quality.
    wav, file_sr = sf.read(wav_path, always_2d=True)          # (n, ch)
    wav_t = torch.tensor(wav.T, dtype=torch.float32)          # (ch, n)
    wav_t = convert_audio(wav_t, file_sr, model_sr, model_ch)

    ref = wav_t.mean(0)
    wav_in = (wav_t - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(model, wav_in[None], device="cpu",
                              split=True, overlap=0.25, progress=True)[0]
    sources = sources * ref.std() + ref.mean()                # (n_src, ch, n)

    names = model.sources                                     # ['drums','bass','other','vocals']
    vi = names.index("vocals")
    vocals = sources[vi]
    accomp = sum(sources[i] for i in range(len(names)) if i != vi)

    def _save(t, path):
        # t: (ch, n) at model_sr -> mono at SR, saved as 16-bit-ish float wav
        y = t.mean(0).cpu().numpy().astype(np.float32)
        if model_sr != SR:
            y = librosa.resample(y, orig_sr=model_sr, target_sr=SR)
        peak = float(np.max(np.abs(y))) or 1.0
        if peak > 1.0:
            y = y / peak
        sf.write(path, y, SR, subtype="PCM_16")

    _save(vocals, voc_path)
    _save(accomp, acc_path)
    return voc_path, acc_path


# --------------------------------------------------------------------------
# 2. vocal-activity envelope
# --------------------------------------------------------------------------
def vocal_envelope(vocals_wav, frame=VOCAL_FRAME, sr=SR):
    """Return (times, rms) : per-frame RMS of the vocals stem (vocal-presence curve)."""
    y = _load_mono(vocals_wav, sr)
    hop = int(frame * sr)
    n = max(1, len(y) // hop)
    rms = np.array([np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2) + 1e-12)
                    for i in range(n)], dtype=np.float32)
    times = (np.arange(n) + 0.5) * frame
    return times, rms


def _mean_in(times, values, s, e):
    m = (times >= s) & (times < e)
    if not np.any(m):
        idx = int(np.clip(np.searchsorted(times, (s + e) / 2), 0, len(values) - 1))
        return float(values[idx])
    return float(np.mean(values[m]))


# --------------------------------------------------------------------------
# 3. Laplacian self-similarity segmentation + repetition (on accompaniment)
# --------------------------------------------------------------------------
def segment_song(y, sr=SR):
    """McFee/librosa Laplacian segmentation. Returns (segments, tempo) where
    segments is a list of (start, end, cluster_id)."""
    C = np.abs(librosa.cqt(y=y, sr=sr, hop_length=512, bins_per_octave=12, n_bins=84))
    C = librosa.amplitude_to_db(C, ref=np.max)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, trim=False)
    if len(beats) < 8:
        return None
    Csync = librosa.util.sync(C, beats, aggregate=np.median)
    M = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    Msync = librosa.util.sync(M, beats)
    nb = Csync.shape[1]

    R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity', sym=True)
    Rf = librosa.segment.timelag_filter(median_filter)(R, size=(1, 7))

    path_dist = np.sum(np.diff(Msync, axis=1) ** 2, axis=0)
    sigma = np.median(path_dist) + 1e-9
    path_sim = np.exp(-path_dist / sigma)
    R_path = np.diag(path_sim, 1) + np.diag(path_sim, -1)

    deg_path = np.sum(R_path, axis=1); deg_rec = np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / (np.sum((deg_path + deg_rec) ** 2) + 1e-9)
    mu = float(np.clip(mu, 0.1, 0.9))
    A = mu * Rf + (1 - mu) * R_path
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    evals, evecs = scipy.linalg.eigh(L)

    dur = len(y) / sr
    k = int(min(7, max(3, round(dur / 22.0))))
    k = min(k, nb - 1)
    X = evecs[:, :k]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    _, labels = kmeans2(Xn, k, minit='++', seed=0, iter=50)

    bt = librosa.frames_to_time(beats, sr=sr, hop_length=512)
    bounds = np.concatenate([[0.0], bt, [dur]])
    seg_times = []
    cur = labels[0]; start = bounds[0]
    for i in range(1, len(labels)):
        if labels[i] != cur:
            seg_times.append((start, bounds[i], int(cur)))
            start = bounds[i]; cur = labels[i]
    seg_times.append((start, bounds[len(labels)], int(cur)))
    merged = []
    for s, e, lab in seg_times:
        if merged and (e - s) < MIN_SEG:
            merged[-1] = (merged[-1][0], e, merged[-1][2])
        else:
            merged.append((s, e, lab))
    return merged, float(np.atleast_1d(tempo)[0])


def energy_of(y, s, e, sr=SR):
    seg = y[int(s * sr):int(e * sr)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0


# --------------------------------------------------------------------------
# 4. vocal-aware labeling fit to the template
# --------------------------------------------------------------------------
DEFAULT_TEMPLATE = ["intro", "verse", "pre-chorus", "chorus", "solo",
                    "verse", "chorus", "outro"]


def label_segments(segs, mix_y, voc_times, voc_rms, sr=SR):
    """Vocal-aware whole-segment labeling, fit to the human template
    (intro -> verse -> pre-chorus -> chorus -> SOLO -> post-solo verse ->
    post-solo chorus -> outro). segs: list of (s,e,cluster).

    Strategy: the VOCAL envelope is the anchor. The single most reliable landmark
    is the long instrumental gap in the latter half = the SOLO. Around it we lay
    down the template in order, using the repetition-cluster to tell CHORUS (the
    most-repeated sung cluster) from VERSE, and a rising sung run just before a
    chorus as PRE-CHORUS. Everything is assigned per-segment then merged so each
    labeled section is WHOLE.

    Returns (records, chorus_cluster, vocal_threshold).
    """
    dur = segs[-1][1]
    feats = []
    for s, e, cl in segs:
        vr = _mean_in(voc_times, voc_rms, s, e)
        mr = energy_of(mix_y, s, e, sr)
        feats.append([s, e, cl, vr, mr])
    voc_vals = np.array([f[3] for f in feats])
    mix_vals = np.array([f[4] for f in feats])
    n = len(feats)
    mid = dur / 2.0

    # ---- adaptive vocal threshold: split "singing" from "instrumental" ----
    # The vocal stem has a near-zero floor (intro/solo) and a clearly higher
    # sung level. Put the threshold well above the floor.
    v_active = voc_vals[voc_vals > 0.005]
    v_hi = np.percentile(voc_vals, 75)
    floor = np.percentile(voc_vals, 15)
    v_thresh = max(0.015, floor + 0.30 * (v_hi - floor))
    is_vocal = voc_vals >= v_thresh

    labcnt = Counter(cl for _, _, cl, _, _ in feats)
    recur = {cl for cl, c in labcnt.items() if c > 1}

    # ---- CHORUS cluster: the most-REPEATED sung cluster (repeat count * vocal) ----
    cluster_voc = {}
    for cl in labcnt:
        idxs = [i for i, f in enumerate(feats) if f[2] == cl]
        cluster_voc[cl] = float(np.mean([voc_vals[i] for i in idxs]))
    chorus_lab = None
    best = -1.0
    for cl in recur:
        if cluster_voc[cl] < v_thresh * 0.9:      # chorus must be sung
            continue
        score = labcnt[cl] * cluster_voc[cl]
        if score > best:
            best = score; chorus_lab = cl
    if chorus_lab is None and recur:
        chorus_lab = max(recur, key=lambda c: labcnt[c] * cluster_voc[c])

    # ---- SOLO: the longest low-vocal instrumental segment in the LATTER half ----
    # (the human constraint: solo is late, instrumental-led).
    solo_idx = None
    solo_best = -1.0
    for i, (s, e, cl, vr, mr) in enumerate(feats):
        if i == 0 or i == n - 1:
            continue
        if e <= mid:                # must be in the latter half
            continue
        if vr >= v_thresh * 0.5:    # solos are instrumental (very low vocal)
            continue
        length = e - s
        # prefer long + low-vocal + energetic
        score = length + 30.0 * (mr - vr)
        if score > solo_best:
            solo_best = score; solo_idx = i

    labels = [None] * n

    # ---- INTRO: leading run of low-vocal segments from the start ----
    intro_end = 0
    for i in range(n):
        if not is_vocal[i]:
            intro_end = i
        else:
            break
    for i in range(intro_end + 1):
        labels[i] = "intro"

    # ---- core assignment, in template order around the solo ----
    for i in range(n):
        if labels[i] is not None:
            continue
        s, e, cl, vr, mr = feats[i]
        if i == solo_idx:
            labels[i] = "solo"
        elif i == n - 1:
            labels[i] = "outro"
        elif is_vocal[i]:
            labels[i] = "chorus" if cl == chorus_lab else "verse"
        else:
            # instrumental interior segment that is NOT the solo:
            # if it sits between two sung sections it's a short break -> fold into
            # the surrounding section by giving it the previous label; if it's in
            # the trailing region it's outro.
            if solo_idx is not None and i > solo_idx and s > mid:
                labels[i] = "outro"
            elif i > 0 and labels[i - 1] is not None:
                labels[i] = labels[i - 1]
            else:
                labels[i] = "verse"

    # any remaining Nones (shouldn't happen) default forward-fill
    for i in range(n):
        if labels[i] is None:
            labels[i] = labels[i - 1] if i > 0 else "verse"

    # ---- PRE-CHORUS: a rising sung VERSE right before the FIRST (pre-solo) chorus
    solo_pos = solo_idx if solo_idx is not None else n
    for i in range(1, n):
        if i >= solo_pos:
            break
        if labels[i] == "chorus" and labels[i - 1] == "verse":
            if voc_vals[i - 1] >= v_thresh and voc_vals[i] >= voc_vals[i - 1] * 0.85:
                labels[i - 1] = "pre-chorus"
            break

    # ---- build records, merge adjacent identical labels into WHOLE sections ----
    recs = []
    for i, (s, e, cl, vr, mr) in enumerate(feats):
        recs.append({"start": round(s, 2), "end": round(e, 2), "label": labels[i],
                     "vocal_rms": round(vr, 5), "mix_rms": round(mr, 5),
                     "cluster": cl, "is_repeat": cl in recur})
    merged = _merge_same_label(recs)

    # ---- OUTRO split: if the trailing outro spans a clear cluster sub-change,
    # the template allows splitting it into two (outro / outro_2). Find the outro
    # record and, if it internally covers >1 distinct cluster with a boundary,
    # split at the dominant internal boundary.
    merged = _maybe_split_outro(merged, feats, recur)
    return merged, chorus_lab, float(v_thresh)


def _maybe_split_outro(merged, feats, recur):
    out = []
    for r in merged:
        if r["label"] != "outro":
            out.append(r); continue
        inner = [f for f in feats if f[0] >= r["start"] - 1e-6 and f[1] <= r["end"] + 1e-6]
        # distinct clusters inside the outro, in order
        clusters = [f[2] for f in inner]
        if len(inner) >= 2 and len(set(clusters)) >= 2:
            # split at the first cluster change that leaves >=3s on both sides
            split_at = None
            for k in range(1, len(inner)):
                if inner[k][2] != inner[k - 1][2]:
                    left = inner[k][0] - r["start"]; right = r["end"] - inner[k][0]
                    if left >= 3.0 and right >= 3.0:
                        split_at = inner[k][0]; k_idx = k; break
            if split_at is not None:
                a = [f for f in inner if f[1] <= split_at + 1e-6]
                b = [f for f in inner if f[0] >= split_at - 1e-6]
                out.append(_rec_from(a, "outro", recur))
                out.append(_rec_from(b, "outro_2", recur))
                continue
        out.append(r)
    return out


def _rec_from(inner, label, recur):
    s = inner[0][0]; e = inner[-1][1]
    tot = sum(f[1] - f[0] for f in inner) or 1.0
    vr = sum(f[3] * (f[1] - f[0]) for f in inner) / tot
    mr = sum(f[4] * (f[1] - f[0]) for f in inner) / tot
    # dominant cluster = longest
    dom = max(inner, key=lambda f: f[1] - f[0])[2]
    return {"start": round(s, 2), "end": round(e, 2), "label": label,
            "vocal_rms": round(vr, 5), "mix_rms": round(mr, 5),
            "cluster": dom, "is_repeat": dom in recur}


def _merge_same_label(recs):
    """Merge adjacent records that share the same label into whole sections.
    Recompute vocal/mix rms as duration-weighted means; keep the dominant cluster."""
    out = []
    for r in recs:
        if out and out[-1]["label"] == r["label"]:
            p = out[-1]
            dp = p["end"] - p["start"]; dr = r["end"] - r["start"]
            tot = dp + dr
            p["vocal_rms"] = round((p["vocal_rms"] * dp + r["vocal_rms"] * dr) / tot, 5)
            p["mix_rms"] = round((p["mix_rms"] * dp + r["mix_rms"] * dr) / tot, 5)
            # keep the longer piece's cluster
            if dr > dp:
                p["cluster"] = r["cluster"]
            p["is_repeat"] = p["is_repeat"] or r["is_repeat"]
            p["end"] = r["end"]
        else:
            out.append(dict(r))
    return out


# --------------------------------------------------------------------------
# top-level
# --------------------------------------------------------------------------
def analyze_song(wav_path, workdir, song_key, abs_start=0.0,
                 model_name="htdemucs", template=None):
    """End-to-end. Returns a dict with the labeled section map + metadata."""
    voc_path, acc_path = separate_stems(wav_path, workdir, song_key, model_name)
    voc_times, voc_rms = vocal_envelope(voc_path)
    acc_y = _load_mono(acc_path)
    mix_y = _load_mono(wav_path)
    res = segment_song(acc_y)
    if not res:
        raise RuntimeError("song too short / no beats for segmentation")
    segs, tempo = res
    recs, chorus_lab, v_thresh = label_segments(segs, mix_y, voc_times, voc_rms)

    sections = []
    for r in recs:
        r["abs_start"] = round(abs_start + r["start"], 2)
        r["abs_end"] = round(abs_start + r["end"], 2)
        r["dur"] = round(r["end"] - r["start"], 2)
        sections.append(r)

    return {
        "song_key": song_key,
        "wav": os.path.basename(wav_path),
        "duration": round(len(mix_y) / SR, 1),
        "tempo": round(tempo, 1),
        "abs_start": abs_start,
        "model": model_name,
        "stem_dir": os.path.relpath(os.path.dirname(voc_path), workdir),
        "vocal_threshold": round(v_thresh, 5),
        "chorus_cluster": chorus_lab,
        "template": template or DEFAULT_TEMPLATE,
        "sections": sections,
    }


def main():
    if len(sys.argv) < 4:
        print("usage: python vocal_structure.py <wav> <workdir> <song_key> [abs_start] [out.json]")
        return 2
    wav = sys.argv[1]; workdir = sys.argv[2]; song_key = sys.argv[3]
    abs_start = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    out = sys.argv[5] if len(sys.argv) > 5 else os.path.join(workdir, "vocal_structure_%s.json" % song_key)
    result = analyze_song(wav, workdir, song_key, abs_start)
    json.dump(result["sections"], open(out, "w"), indent=2)
    # human-readable summary to stdout
    print("song_key=%s  dur=%.1fs tempo=%.1f  chorus_cluster=%s  v_thresh=%.5f"
          % (song_key, result["duration"], result["tempo"],
             result["chorus_cluster"], result["vocal_threshold"]))
    print("%-11s %7s %7s %8s %8s %7s %6s %6s" %
          ("label", "start", "end", "abs_st", "vocRMS", "mixRMS", "clus", "rep"))
    for s in result["sections"]:
        print("%-11s %7.2f %7.2f %8.2f %8.5f %7.5f %6d %6s" %
              (s["label"], s["start"], s["end"], s["abs_start"],
               s["vocal_rms"], s["mix_rms"], s["cluster"], s["is_repeat"]))
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
