# -*- coding: utf-8 -*-
"""FINE-GRAINED music STRUCTURE detection (boundaries + repetition), v2.

Rationale / why this replaces the old fixed-k spectral clustering
-----------------------------------------------------------------
The previous approach (structure_analyze.py / vocal_structure.py) spectral-clustered
beats into ~7 GLOBAL labels. On a harmonically uniform song (e.g. a "Girls Just Want
to Have Fun" cover, whose verse/pre-chorus/chorus all share essentially one chord
progression) that collapses ~15 real fine sections into a few big blobs and never
resolves the 4-12 s sections. This module DECOUPLES the two problems:

  A. BOUNDARIES from a NOVELTY curve. Build a beat-synchronous self-similarity matrix
     (chroma harmony + MFCC timbre), convolve a Foote checkerboard kernel down its
     diagonal to get a novelty curve, and FUSE it with two direct transition cues:
     the |derivative| of the demucs vocal-stem RMS (vocals switching on/off is a very
     strong section boundary) and the |derivative| of mix energy. Adaptive peak-pick
     -> fine boundaries (~1 per 4-12 s). Boundaries snap to the nearest beat and, when
     a downbeat estimate exists, to the nearest downbeat. A minimum-segment merge drops
     the weaker of two bounding peaks so we don't over-fragment, but a LONG LOW-VOCAL
     instrumental run (the SOLO) is protected from being split.

  B. REPETITION / labeling by SEGMENT SIMILARITY. For every resulting segment we take
     its beat-synchronous chroma SEQUENCE and compare segments pairwise with DTW
     (sequence-aware, so a chorus sung again matches even if slightly stretched), plus
     a vocal-presence penalty so a sung section never merges with an instrumental one.
     Agglomerative average-linkage over that distance groups IDENTICAL sections into
     one cluster id, so the three choruses share a cluster and the two pre-choruses
     share a cluster. Clusters are then NAMED to the human vocabulary (intro, verse,
     pre-chorus, chorus, solo, interlude, outro) using vocal-presence + energy +
     position + repetition.

Public entry points (unchanged contract from vocal_structure.py so the pipeline can
swap this in):
  separate_stems(wav_path, workdir, song_key, model_name="htdemucs") -> (voc, acc)
  analyze_song(wav_path, workdir, song_key, abs_start=0.0, template=None) -> dict
  main() : CLI  `python structure2.py <wav> <workdir> <song_key> [abs_start] [out.json]`

Config-driven, band/username-free (nothing hardcodes a device path or the band handle),
byte-compilable. Dependencies: numpy, scipy, soundfile, librosa (+ torch/demucs only if
stems must actually be computed; cached stems skip that import).
"""
import sys, os, json
import numpy as np
import soundfile as sf
import librosa
import scipy.signal
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from collections import Counter

SR = 22050
HOP = 512

# ---- tunables (validated against a fine human ground truth; see module docstring) ----
KERNEL_L = 10            # Foote checkerboard half-width in BEATS (~2 bars @ ~108 bpm)
NOV_W_CHROMA = 0.6       # harmony weight in the self-similarity matrix
NOV_W_MFCC = 0.4         # timbre weight
FUSE_W_VOCAL = 0.5       # weight of |d/dt vocal-RMS| in the fused novelty
FUSE_W_ENERGY = 0.3      # weight of |d/dt mix-energy| in the fused novelty
PEAK_HEIGHT_FRAC = 0.5   # peak height threshold = frac * mean(fused)
PEAK_MIN_DIST_BEATS = 5  # min beats between accepted boundaries
PEAK_PROMINENCE = 0.03
MIN_SEG = 5.0            # merge segments shorter than this (seconds) ...
SOLO_PROTECT_S = 6.0     # ... UNLESS it is a >=this long low-vocal instrumental (solo)
CLUSTER_W_VOCAL = 0.4    # vocal-presence penalty weight in the repetition distance
CLUSTER_THRESH = 0.26    # coarse agglomerative distance threshold (repetition groups)
SUBCLUSTER_THRESH = 0.18 # finer chroma-only threshold used only to inform naming


# ==========================================================================
# 0. IO
# ==========================================================================
def _load_mono(path, sr=SR):
    y, file_sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y


# ==========================================================================
# 1. demucs separation (cached). Saved via soundfile to dodge torchcodec.
# ==========================================================================
def separate_stems(wav_path, workdir, song_key, model_name="htdemucs"):
    """Return (vocals_wav_path, accompaniment_wav_path), computing + caching stems.
    Stems live in <workdir>/_stems/<model_name>/<song_key>/{vocals,no_vocals}.wav .
    If both already exist, demucs is NOT re-run (torch/demucs are not even imported)."""
    stem_dir = os.path.join(workdir, "_stems", model_name, song_key)
    os.makedirs(stem_dir, exist_ok=True)
    voc_path = os.path.join(stem_dir, "vocals.wav")
    acc_path = os.path.join(stem_dir, "no_vocals.wav")
    if os.path.exists(voc_path) and os.path.exists(acc_path):
        return voc_path, acc_path

    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    model = get_model(model_name)
    model.cpu().eval()
    model_sr = model.samplerate
    model_ch = model.audio_channels

    wav, file_sr = sf.read(wav_path, always_2d=True)
    wav_t = torch.tensor(wav.T, dtype=torch.float32)
    wav_t = convert_audio(wav_t, file_sr, model_sr, model_ch)
    ref = wav_t.mean(0)
    wav_in = (wav_t - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(model, wav_in[None], device="cpu",
                              split=True, overlap=0.25, progress=True)[0]
    sources = sources * ref.std() + ref.mean()
    names = model.sources
    vi = names.index("vocals")
    vocals = sources[vi]
    accomp = sum(sources[i] for i in range(len(names)) if i != vi)

    def _save(t, path):
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


# ==========================================================================
# 2. beat-synchronous features
# ==========================================================================
def beat_features(mix_y, voc_y):
    """Beat-track the MIX, then compute beat-synchronous chroma, MFCC, vocal-RMS and
    mix-RMS. Returns a dict of arrays + beat start times (one entry per beat-segment)."""
    tempo, beats = librosa.beat.beat_track(y=mix_y, sr=SR, hop_length=HOP, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    bt = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    dur = len(mix_y) / SR

    chroma = librosa.feature.chroma_cqt(y=mix_y, sr=SR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=mix_y, sr=SR, hop_length=HOP, n_mfcc=20)
    vrms = librosa.feature.rms(y=voc_y, hop_length=HOP, frame_length=2048)[0]
    mrms = librosa.feature.rms(y=mix_y, hop_length=HOP, frame_length=2048)[0]

    chroma_s = librosa.util.sync(chroma, beats, aggregate=np.median)
    mfcc_s = librosa.util.sync(mfcc, beats, aggregate=np.mean)
    vrms_s = librosa.util.sync(vrms[None, :], beats, aggregate=np.mean)[0]
    mrms_s = librosa.util.sync(mrms[None, :], beats, aggregate=np.mean)[0]

    nb = chroma_s.shape[1]
    # start time of each beat-segment; append song end so the last segment closes.
    beat_times = np.concatenate([bt, [dur]])[:nb]

    # downbeats (best-effort): assume 4/4 and phase-align to strongest beat-energy.
    downbeats = _estimate_downbeats(mrms_s, beat_times, nb)
    # downbeat TIMES (seconds) from our OWN beat grid -- so the chord module never needs the external
    # beat_this ML model (which may be absent). This keeps structure2 self-sufficient and cache-only.
    downbeats_time = [float(beat_times[i]) for i in sorted(downbeats) if 0 <= i < len(beat_times)]

    return {
        "tempo": tempo, "dur": dur, "nb": nb, "beat_times": beat_times,
        "chroma_s": chroma_s, "mfcc_s": mfcc_s, "vrms_s": vrms_s, "mrms_s": mrms_s,
        "downbeats": downbeats, "downbeats_time": downbeats_time,
    }


def _estimate_downbeats(mrms_s, beat_times, nb, meter=4):
    """Return the set of beat indices that are downbeats (best-effort 4/4).
    Pick the phase (0..meter-1) whose beats carry the most energy."""
    if nb < meter:
        return set(range(nb))
    best_phase, best_e = 0, -1.0
    for ph in range(meter):
        idxs = list(range(ph, nb, meter))
        e = float(np.mean(mrms_s[idxs])) if idxs else 0.0
        if e > best_e:
            best_e, best_phase = e, ph
    return set(range(best_phase, nb, meter))


# ==========================================================================
# 3. novelty boundaries
# ==========================================================================
def _ssm(feat):
    """Cosine self-similarity matrix of a feature (dim x nb), in [0,1]."""
    fn = feat / (np.linalg.norm(feat, axis=0, keepdims=True) + 1e-9)
    S = fn.T @ fn
    return np.clip(S, 0.0, 1.0)


def _foote_novelty(S, L):
    """Novelty curve = diagonal convolution of a Gaussian-tapered checkerboard kernel."""
    N = S.shape[0]
    g = np.outer(scipy.signal.windows.gaussian(2 * L, L),
                 scipy.signal.windows.gaussian(2 * L, L))
    ker = np.fromfunction(lambda i, j: np.where((i < L) == (j < L), 1.0, -1.0),
                          (2 * L, 2 * L)) * g
    Spad = np.pad(S, L, mode="edge")
    nov = np.array([np.sum(Spad[i:i + 2 * L, i:i + 2 * L] * ker) for i in range(N)])
    nov = np.maximum(nov, 0.0)
    return nov / (nov.max() + 1e-9)


def _fused_novelty(F):
    """Foote novelty fused with vocal-RMS and energy transition cues."""
    S = NOV_W_CHROMA * _ssm(_znorm(F["chroma_s"])) + NOV_W_MFCC * _ssm(_znorm(F["mfcc_s"]))
    nov = gaussian_filter1d(_foote_novelty(S, KERNEL_L), 1.0)

    vsm = gaussian_filter1d(F["vrms_s"], 2.0)
    vdiff = np.abs(np.gradient(vsm)); vdiff /= vdiff.max() + 1e-9
    esm = gaussian_filter1d(F["mrms_s"], 2.0)
    ediff = np.abs(np.gradient(esm)); ediff /= ediff.max() + 1e-9

    fused = nov + FUSE_W_VOCAL * vdiff + FUSE_W_ENERGY * ediff
    fused = gaussian_filter1d(fused, 1.0)
    return fused / (fused.max() + 1e-9), nov


def _znorm(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-9)


def detect_boundaries(F):
    """Return sorted list of boundary BEAT INDICES (excluding 0 and nb) after
    peak-picking the fused novelty, snapping to downbeats, and min-segment merging
    (with the long low-vocal solo protected)."""
    nb = F["nb"]; beat_times = F["beat_times"]
    fused, _nov = _fused_novelty(F)
    peaks, _ = scipy.signal.find_peaks(
        fused, height=fused.mean() * PEAK_HEIGHT_FRAC,
        distance=PEAK_MIN_DIST_BEATS, prominence=PEAK_PROMINENCE)
    peakval = {int(p): float(fused[p]) for p in peaks}

    # snap each peak to the nearest downbeat within +/-1 beat (keeps sections bar-aligned)
    downs = F["downbeats"]
    snapped = []
    for p in peaks:
        cand = [p] + [p + d for d in (-1, 1) if 0 < p + d < nb]
        db = [c for c in cand if c in downs]
        q = min(db, key=lambda c: abs(c - p)) if db else p
        snapped.append(int(q))
        peakval[int(q)] = max(peakval.get(int(q), 0.0), peakval.get(int(p), fused[p]))
    inner = sorted(set(snapped))

    # min-segment merge: drop the weaker bounding peak of any too-short segment, unless
    # that segment is a long low-vocal instrumental (the solo) which must stay whole.
    vrms_s = F["vrms_s"]
    voc_floor = np.percentile(vrms_s, 20)
    voc_lo = max(0.01, voc_floor + 0.15 * (np.percentile(vrms_s, 80) - voc_floor))

    changed = True
    while changed:
        changed = False
        b = [0] + inner + [nb]
        for i in range(len(b) - 1):
            s_t = beat_times[b[i]] if b[i] < nb else F["dur"]
            e_t = beat_times[b[i + 1]] if b[i + 1] < nb else F["dur"]
            seglen = e_t - s_t
            if seglen >= MIN_SEG:
                continue
            seg_voc = float(np.mean(vrms_s[b[i]:max(b[i] + 1, b[i + 1])]))
            if seglen >= SOLO_PROTECT_S and seg_voc < voc_lo:
                continue  # protected instrumental (solo) fragment
            cands = [x for x in (b[i], b[i + 1]) if x in inner]
            if not cands:
                continue
            worst = min(cands, key=lambda x: peakval.get(x, 0.0))
            inner.remove(worst); changed = True; break
    return inner


# ==========================================================================
# 4. segments + repetition clustering
# ==========================================================================
def segments_from_bounds(F, inner):
    """Build segment records with beat-index ranges and summary features."""
    nb = F["nb"]; bt = F["beat_times"]
    bidx = [0] + inner + [nb]
    segs = []
    for i in range(len(bidx) - 1):
        bi = bidx[i]; bj = bidx[i + 1]
        s = float(bt[bi]) if bi < nb else F["dur"]
        e = float(bt[bj]) if bj < nb else F["dur"]
        if i == 0:
            s = 0.0                       # first segment covers from the song start
        if i == len(bidx) - 2:
            e = F["dur"]                  # last segment covers to the song end
        sl = slice(bi, max(bi + 1, bj))
        segs.append({
            "start": s, "end": e, "bi": bi, "bj": bj,
            "chroma": F["chroma_s"][:, sl],
            "vocal_rms": float(F["vrms_s"][sl].mean()),
            "mix_rms": float(F["mrms_s"][sl].mean()),
        })
    return segs


def merge_instrumental_runs(segs, F, v_thresh):
    """Merge adjacent LOW-VOCAL instrumental segments into one whole segment.

    The novelty curve can place a seam inside a long instrumental solo (its internal
    energy shifts). Since the human wants the SOLO as ONE continuous piece, any run of
    consecutive segments that are all clearly instrumental (mean vocal-RMS below the
    vocal threshold) and that spans no sung material is fused back together. Intro
    (leading) instrumental runs are also fused; this keeps segments WHOLE without
    over-fragmenting an instrumental passage."""
    if len(segs) <= 1:
        return segs
    out = []
    for s in segs:
        instrumental = s["vocal_rms"] < v_thresh
        if out and instrumental and (out[-1]["vocal_rms"] < v_thresh):
            p = out[-1]
            sl = slice(p["bi"], max(p["bi"] + 1, s["bj"]))
            p["end"] = s["end"]; p["bj"] = s["bj"]
            p["chroma"] = F["chroma_s"][:, sl]
            p["vocal_rms"] = float(F["vrms_s"][sl].mean())
            p["mix_rms"] = float(F["mrms_s"][sl].mean())
        else:
            out.append(dict(s))
    return out


def _dtw_cost(A, B):
    """Normalized DTW distance between two beat-chroma sequences (cosine local cost)."""
    if A.shape[1] < 2 or B.shape[1] < 2:
        return 1.0
    An = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-9)
    C = 1.0 - An.T @ Bn
    D, wp = librosa.sequence.dtw(C=C)
    return float(D[-1, -1] / len(wp))


def _relabel_first_appearance(raw):
    remap = {}; out = []
    for c in raw:
        if c not in remap:
            remap[c] = len(remap)
        out.append(remap[c])
    return out


def cluster_segments(segs):
    """Agglomerative clustering of segments by chroma-sequence DTW + vocal-presence
    penalty. Returns (clusters, subclusters):
      - clusters:    coarse REPETITION groups (same harmonic part -> same id). This is
                     the repetition truth the assembler dedups on.
      - subclusters: a finer split of the same tree, used only to inform naming.
    Both are 0-based in first-appearance order."""
    n = len(segs)
    if n == 1:
        return [0], [0]
    Dc = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            Dc[i, j] = Dc[j, i] = _dtw_cost(segs[i]["chroma"], segs[j]["chroma"])
    Dc /= Dc.max() + 1e-9

    vr = np.array([s["vocal_rms"] for s in segs])
    vlog = np.log(vr + 1e-4)
    vlog = (vlog - vlog.min()) / (np.ptp(vlog) + 1e-9)
    Dv = np.abs(vlog[:, None] - vlog[None, :])

    D = Dc + CLUSTER_W_VOCAL * Dv
    D /= D.max() + 1e-9
    Z = linkage(squareform(D, checks=False), method="average")
    coarse = _relabel_first_appearance(fcluster(Z, t=CLUSTER_THRESH, criterion="distance"))

    # a finer sub-clustering on chroma-only distance (no vocal penalty) exposes the
    # harmonic sub-parts (e.g. a chorus-like vs verse-like progression) for naming.
    Zc = linkage(squareform(Dc, checks=False), method="average")
    sub = _relabel_first_appearance(fcluster(Zc, t=SUBCLUSTER_THRESH, criterion="distance"))
    return coarse, sub


# ==========================================================================
# 5. naming to the human vocabulary
# ==========================================================================
def name_segments(segs, clusters, subclusters, F):
    """Assign human-vocabulary labels while keeping repeated sections' shared cluster id.

    Naming order of operations:
      1. Vocal threshold splits sung vs instrumental segments.
      2. INTRO   = leading low-vocal run; SOLO = the longest low-vocal segment in the
         latter half (instrumental lead, kept whole); interior low-vocal = INTERLUDE;
         a trailing low-vocal tail = OUTRO. These are acoustically unambiguous.
      3. For the SUNG family we pick a CHORUS sub-cluster: the sub-cluster (from the
         chroma-only tree) that recurs most and carries the most vocal energy is the
         chorus. Its members -> "chorus". A short sung segment (< PRECHORUS_MAX_S)
         immediately preceding a chorus -> "pre-chorus". Remaining sung segments ->
         "verse". Because verse/pre-chorus/chorus share one progression in some songs,
         this naming is a best-effort overlay on top of the (correct) repetition
         clustering -- the `cluster` field remains the ground truth for "same part".

    NOTE ON THIS SONG: for the "Girls Just Want to Have Fun" cover the verse, pre-chorus
    and chorus reuse essentially one chord progression, so they land in ONE coarse
    repetition cluster (as they should -- they ARE the same recurring harmony). The
    verse/chorus *names* below are therefore heuristic, not acoustically separable.
    """
    PRECHORUS_MAX_S = 8.0
    n = len(segs)
    vr = np.array([s["vocal_rms"] for s in segs])
    mr = np.array([s["mix_rms"] for s in segs])
    dur_total = segs[-1]["end"]
    mid = dur_total / 2.0

    floor = np.percentile(vr, 15)
    hi = np.percentile(vr, 75)
    v_thresh = max(0.012, floor + 0.30 * (hi - floor))
    is_vocal = vr >= v_thresh

    # ---- SOLO: longest low-vocal segment in the latter half ----
    solo_idx = None; solo_best = -1.0
    for i in range(n):
        if i == 0 or i == n - 1 or segs[i]["end"] <= mid or vr[i] >= v_thresh * 0.6:
            continue
        length = segs[i]["end"] - segs[i]["start"]
        score = length + 25.0 * (mr[i] - vr[i])
        if score > solo_best:
            solo_best, solo_idx = score, i

    labels = [None] * n

    # ---- INTRO: leading low-vocal run ----
    intro_end = -1
    for i in range(n):
        if not is_vocal[i]:
            intro_end = i
        else:
            break
    for i in range(intro_end + 1):
        labels[i] = "intro"

    # ---- assign SOLO / INTERLUDE / OUTRO anchors first ----
    for i in range(n):
        if labels[i] is not None:
            continue
        if i == solo_idx:
            labels[i] = "solo"
        elif i == n - 1 and not is_vocal[i]:
            labels[i] = "outro"
        elif not is_vocal[i]:
            labels[i] = "interlude"

    # ---- CHORUS: the sung part that FLANKS the solo (pop songs bracket a solo with
    # choruses) plus every sung segment whose chroma sub-cluster matches a flanker.
    # This is more reliable than raw recurrence when verse/chorus share a progression.
    flank = []
    if solo_idx is not None:
        for j in (solo_idx - 1, solo_idx + 1):
            if 0 <= j < n and is_vocal[j]:
                flank.append(j)
    chorus_subs = set(subclusters[j] for j in flank)
    # also fold in the most-recurring high-energy sung sub-cluster as a chorus sub
    sub_members = {}
    for i in range(n):
        if is_vocal[i] and i != solo_idx:
            sub_members.setdefault(subclusters[i], []).append(i)
    if sub_members:
        best_sc = max(sub_members,
                      key=lambda sc: len(sub_members[sc]) *
                      float(np.mean([mr[i] for i in sub_members[sc]])))
        if len(sub_members[best_sc]) >= 2:
            chorus_subs.add(best_sc)

    for i in range(n):
        if labels[i] is not None:
            continue
        labels[i] = "chorus" if subclusters[i] in chorus_subs else "verse"

    # ---- PRE-CHORUS: short sung segment right before a chorus ----
    pre_idxs = []
    for i in range(1, n):
        if labels[i] == "chorus" and labels[i - 1] in ("verse", "chorus"):
            plen = segs[i - 1]["end"] - segs[i - 1]["start"]
            if plen <= PRECHORUS_MAX_S and is_vocal[i - 1] and labels[i - 1] != "pre-chorus":
                labels[i - 1] = "pre-chorus"
                pre_idxs.append(i - 1)
    # keep both pre-choruses in ONE shared cluster id (repetition guarantee)
    if len(pre_idxs) >= 2:
        pre_cl = clusters[pre_idxs[0]]
        for i in pre_idxs:
            clusters[i] = pre_cl

    # ---- trailing outro: the final segment is the outro when it is instrumental, OR
    # when it is a non-repeating singleton cluster tail (a distinct closing section). ----
    cl_count_all = Counter(clusters)
    if labels[-1] != "outro":
        if (not is_vocal[-1]) or cl_count_all[clusters[-1]] == 1:
            labels[-1] = "outro"

    # chorus_cluster reported = the coarse repetition cluster carrying the chorus subs
    chorus_cl = None
    ch_idxs = [i for i in range(n) if labels[i] == "chorus"]
    if ch_idxs:
        chorus_cl = Counter(clusters[i] for i in ch_idxs).most_common(1)[0][0]

    return labels, clusters, chorus_cl, float(v_thresh)


# ==========================================================================
# 5b. CHORD-PROGRESSION refinement (verse/chorus by chord pattern; boundary snap)
# ==========================================================================
def refine_with_chords(segs, labels, clusters, F, acc_path):
    """Use the CHORD PROGRESSION of the vocals-removed accompaniment to (a) refine
    verse/chorus labels and (b) snap sung-section boundaries to real chord changes.

    This is the real version of "verse = chord pattern A, chorus = pattern B". The
    novelty/vocal segmentation gives good boundaries and a correct repetition
    clustering, but on a harmonically uniform cover the verse/chorus *names* are
    unreliable -- they share most diatonic chords. chords.py estimates a soft chord
    posterior per bar (denoised chroma -> 24-triad softmax), splits the SUNG sections
    into two progression clusters (pattern A vs B), and labels the higher-lift cluster
    (borrowed-minor "chorus lift") as chorus.

    Returns (new_labels, new_segs, new_clusters, progressions, prog_clusters,
    chorus_prog_cluster, change_times). progressions[i] is the ordered bar-chord
    label list for segment i; prog_clusters[i] is its A/B progression cluster (or
    None for non-sung anchors). Adjacent same-label sung sections may be FUSED, so
    the returned arrays can be shorter than the inputs -- they stay mutually aligned.
    On any failure (missing chords module / too few sung sections) the inputs are
    returned unchanged so analyze_song still produces a valid map."""
    try:
        import chords as _chords
    except Exception:
        return (labels, segs, clusters, [None] * len(segs),
                [None] * len(segs), None, [])

    downs = F.get("downbeats_time")
    bars = _chords.bar_chords(acc_path, sr=SR, hop=HOP, downbeats=downs)
    change_times = bars["change_times"]

    # ---- (a) snap sung-section boundaries to the nearest real chord change ----
    # Only interior sung->anything seams move; intro/solo/outro anchors stay put.
    new_segs = [dict(s) for s in segs]
    for i in range(1, len(new_segs)):
        prev_l, cur_l = labels[i - 1], labels[i]
        if prev_l in ("intro", "solo", "outro") or cur_l in ("intro", "solo", "outro"):
            continue
        t = new_segs[i]["start"]
        snapped = _chords.snap_to_chord_change(t, change_times, max_dist=2.5)
        if abs(snapped - t) > 1e-3:
            new_segs[i]["start"] = snapped
            new_segs[i - 1]["end"] = snapped

    # rebuild per-section progression from the (possibly moved) boundaries
    progressions, sec_prog = [], []
    for s in new_segs:
        sp = _chords.section_progression(bars, s["start"], s["end"])
        progressions.append(sp["labels"])
        sec_prog.append(sp)

    # ---- (b) split the SUNG sections into progression clusters A vs B ----
    sung_idx = [i for i in range(len(new_segs))
                if labels[i] in ("verse", "chorus", "pre-chorus")]
    prog_clusters = [None] * len(new_segs)
    chorus_prog_cl = None
    new_labels = list(labels)
    new_clusters = list(clusters)
    if len(sung_idx) >= 2:
        ids, chorus_id, _lift = _chords.split_sung_progressions(
            [sec_prog[i] for i in sung_idx])
        for k, i in enumerate(sung_idx):
            prog_clusters[i] = int(ids[k])
        chorus_prog_cl = chorus_id
        # relabel sung sections by progression cluster: chorus cluster -> chorus,
        # the other -> verse.
        for i in sung_idx:
            if chorus_prog_cl is not None and prog_clusters[i] == chorus_prog_cl:
                new_labels[i] = "chorus"
            else:
                new_labels[i] = "verse"

        # ---- absorb a short verse TAIL of a chorus ----
        # The human noted the 808-841 region has "a chorus cut short by ~1 bar that
        # leads directly into the verse": the novelty drops an extra boundary a bar
        # early, so the last bar(s) of a chorus get split off as a tiny verse fragment
        # that then runs into an instrumental (solo/interlude) or ends the sung run.
        # A short (<= PRECHORUS-ish) verse wedged between a chorus and an instrumental
        # is that clipped chorus tail -> fold it back into the chorus.
        for i in range(1, len(new_labels)):
            if (new_labels[i] == "verse" and new_labels[i - 1] == "chorus"
                    and (new_segs[i]["end"] - new_segs[i]["start"]) <= 9.0):
                nxt = new_labels[i + 1] if i + 1 < len(new_labels) else "outro"
                if nxt in ("solo", "interlude", "outro"):
                    new_labels[i] = "chorus"
                    prog_clusters[i] = chorus_prog_cl

        # ---- merge adjacent same-label sung sections created by an over-split ----
        # The novelty curve sometimes drops an extra boundary INSIDE one chorus (the
        # human's "chorus cut short by a bar then straight into the verse" region).
        # Now that verse/chorus come from the chord pattern, two adjacent sung
        # sections with the SAME refined label are one part -> fuse them. This is
        # what un-messes the 808-841 seam. Repetition-cluster ids are unified across
        # the merged span so the assembler still sees one repeating part.
        (new_labels, new_segs, new_clusters, progressions, prog_clusters) = \
            _merge_adjacent_sung(new_labels, new_segs, new_clusters, progressions,
                                 prog_clusters, F, bars, _chords)

    return (new_labels, new_segs, new_clusters, progressions, prog_clusters,
            chorus_prog_cl, change_times)


def _merge_adjacent_sung(labels, segs, clusters, progressions, prog_clusters, F,
                         _chords_bars, _chords):
    """Fuse consecutive sung sections that carry the SAME refined verse/chorus label
    into a single section (recomputing its progression and vocal/mix RMS from the
    merged bar span). Returns aligned (labels, segs, clusters, progressions,
    prog_clusters)."""
    bars = _chords_bars
    out_l, out_s, out_c, out_p, out_pc = [], [], [], [], []
    for i in range(len(segs)):
        if (out_l and labels[i] in ("verse", "chorus")
                and out_l[-1] == labels[i]):
            p = out_s[-1]
            p["end"] = segs[i]["end"]
            p["bj"] = segs[i].get("bj", p.get("bj"))
            # recompute summary RMS over the merged beat span
            bi = p.get("bi", 0); bj = p.get("bj", bi + 1)
            sl = slice(bi, max(bi + 1, bj))
            p["vocal_rms"] = float(F["vrms_s"][sl].mean())
            p["mix_rms"] = float(F["mrms_s"][sl].mean())
            out_p[-1] = _chords.section_progression(bars, p["start"], p["end"])["labels"]
            # keep the earlier section's repetition + progression cluster ids
        else:
            out_l.append(labels[i]); out_s.append(dict(segs[i]))
            out_c.append(clusters[i]); out_p.append(progressions[i])
            out_pc.append(prog_clusters[i])
    return out_l, out_s, out_c, out_p, out_pc


# ==========================================================================
# 6. top-level
# ==========================================================================
def analyze_song(wav_path, workdir, song_key, abs_start=0.0,
                 model_name="htdemucs", template=None):
    """End-to-end fine structure. Returns a dict with the labeled section map."""
    voc_path, acc_path = separate_stems(wav_path, workdir, song_key, model_name)
    mix_y = _load_mono(wav_path)
    voc_y = _load_mono(voc_path)

    F = beat_features(mix_y, voc_y)
    if F["nb"] < 8:
        raise RuntimeError("song too short / no beats for segmentation")

    inner = detect_boundaries(F)
    segs = segments_from_bounds(F, inner)

    # provisional vocal threshold (same rule as name_segments) so we can fuse the
    # instrumental solo into one whole piece BEFORE clustering/labeling.
    _vr = np.array([s["vocal_rms"] for s in segs])
    _v_thresh0 = max(0.012, np.percentile(_vr, 15) +
                     0.30 * (np.percentile(_vr, 75) - np.percentile(_vr, 15)))
    segs = merge_instrumental_runs(segs, F, _v_thresh0)

    clusters, subclusters = cluster_segments(segs)
    labels, clusters, chorus_cl, v_thresh = name_segments(segs, clusters, subclusters, F)

    # CHORD-PROGRESSION refinement: fix verse/chorus labels + snap sung boundaries to
    # chord changes + fuse over-split choruses. Non-fatal -- degrades to the chroma-
    # only labels on any failure.
    (labels, segs, clusters, progressions, prog_clusters,
     chorus_prog_cl, change_times) = refine_with_chords(segs, labels, clusters, F, acc_path)

    cl_count = Counter(clusters)
    sections = []
    for i, s in enumerate(segs):
        c = int(clusters[i])
        sections.append({
            "start": round(s["start"], 2),
            "end": round(s["end"], 2),
            "abs_start": round(abs_start + s["start"], 2),
            "abs_end": round(abs_start + s["end"], 2),
            "dur": round(s["end"] - s["start"], 2),
            "label": labels[i],
            "cluster": c,
            "prog_cluster": prog_clusters[i],
            "progression": progressions[i],
            "vocal_rms": round(s["vocal_rms"], 5),
            "mix_rms": round(s["mix_rms"], 5),
            "is_repeat": cl_count[c] > 1,
        })

    return {
        "song_key": song_key,
        "wav": os.path.basename(wav_path),
        "duration": round(len(mix_y) / SR, 1),
        "tempo": round(F["tempo"], 1),
        "abs_start": abs_start,
        "model": model_name,
        "stem_dir": os.path.relpath(os.path.dirname(voc_path), workdir),
        "vocal_threshold": round(v_thresh, 5),
        "chorus_cluster": int(chorus_cl) if chorus_cl is not None else None,
        "chorus_prog_cluster": int(chorus_prog_cl) if chorus_prog_cl is not None else None,
        "n_sections": len(sections),
        "n_clusters": len(set(clusters)),
        "template": template,
        "sections": sections,
    }


def main():
    if len(sys.argv) < 4:
        print("usage: python structure2.py <wav> <workdir> <song_key> [abs_start] [out.json]")
        return 2
    wav = sys.argv[1]; workdir = sys.argv[2]; song_key = sys.argv[3]
    abs_start = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    out = sys.argv[5] if len(sys.argv) > 5 else \
        os.path.join(workdir, "vocal_structure_%s.json" % song_key)
    res = analyze_song(wav, workdir, song_key, abs_start)
    json.dump(res["sections"], open(out, "w"), indent=2)
    print("song_key=%s dur=%.1fs tempo=%.1f chorus_cluster=%s v_thresh=%.5f n_sec=%d n_cl=%d"
          % (song_key, res["duration"], res["tempo"], res["chorus_cluster"],
             res["vocal_threshold"], res["n_sections"], res["n_clusters"]))
    print("%-11s %8s %8s %8s %6s %8s %8s %5s %4s" %
          ("label", "abs_st", "abs_end", "dur", "clus", "vocRMS", "mixRMS", "rep", "#"))
    for s in res["sections"]:
        print("%-11s %8.2f %8.2f %8.2f %6d %8.5f %8.5f %5s" %
              (s["label"], s["abs_start"], s["abs_end"], s["dur"], s["cluster"],
               s["vocal_rms"], s["mix_rms"], s["is_repeat"]))
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
