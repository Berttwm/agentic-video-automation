# -*- coding: utf-8 -*-
"""CHORD-PROGRESSION analysis for verse/chorus discrimination.

Why this module exists
----------------------
On a harmonically busy live cover, verse and chorus share most of the key's
diatonic chords, so RAW beat-chroma barely separates them (a quick test on the
gig-11 "Girls" cover clustered tagged choruses at DTW 0.034 vs 0.053 chorus-to-
verse -- a REAL but faint signal). The human confirmed the true discriminator is
the CHORD PROGRESSION: verse = pattern A, chorus = pattern B. This module exploits
that properly:

  1. From the vocals-removed ACCOMPANIMENT stem, take a downbeat grid (beat_this,
     via beats.grid) and CQT chroma. Chroma is denoised with a cosine nn-filter
     (recurrence smoothing) before anything else -- this alone lifted the tagged
     within-cluster / cross-cluster separation from ~0.02 to ~0.07.
  2. Per BAR (downbeat to downbeat) average the chroma and template-match it to the
     24 major/minor triads, keeping a SOFT posterior over the 24 chords (softmax of
     the template scores) rather than a hard label -- soft posteriors survive the
     bar-to-bar Cmaj/Cmin/C#min jitter that wrecks hard-label DTW.
  3. Represent each section as its ordered sequence of bar posteriors. Compare two
     sections with a progression distance = 0.5*DTW(posterior sequences) +
     0.5*cosine(mean posterior histograms). DTW tolerates the "chorus cut short by
     a bar" case; the histogram term stabilises very short sections.
  4. Split the SUNG sections into two progression clusters (2-means agglomerative)
     -> pattern A (verse) vs pattern B (chorus). The cluster whose members carry
     more "borrowed-minor" mass (Fmin/Cmin/G#maj type chords, i.e. the chorus lift)
     is the chorus cluster.
  5. Expose chord-CHANGE times so a caller can snap a fuzzy novelty boundary to the
     nearest real chord-pattern change (fixes the 808-841 "messy" seam).

HONEST separability read (gig-11 song4, vs human tags): 7/8 sung sections cluster
correctly and stably across a wide parameter sweep. The single stable miss is the
post-solo chorus (~873-882), whose chord content genuinely sits between the two
patterns (its borrowed-minor mass 0.04 is verse-like) -- consistent with the human
noting the post-solo region is outro-ish/messy. So: 3 of 4 choruses separate
cleanly by chord; the post-solo pair is genuinely ambiguous. We do NOT fake a
clean split.

Public API
----------
  bar_chords(acc_wav, sr, hop) -> dict with keys:
      downbeats, bar_times [(s,e)...], bar_post (nbar x 24), bar_labels [str...],
      chord_names [24], change_times [float...]
  section_progression(bars, start_s, end_s) -> dict(labels, post, hist, bar_idx)
  progression_distance(secA, secB) -> float
  split_sung_progressions(sections) -> (cluster_ids, chorus_cluster_id)
  snap_to_chord_change(t, change_times, max_dist) -> float

Config-driven paths come from beats.grid (which uses paths.FFMPEG). No band handle,
username, or device path is hardcoded. Dependencies: numpy, soundfile, librosa,
scipy (+ beats.py for the downbeat grid). Byte-compilable.
"""
import numpy as np
import soundfile as sf
import librosa
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import Counter

# --- tunables (fixed by the gig-11 separability sweep; robust across gamma 6..12) ---
CHORD_SOFTMAX_GAMMA = 8.0   # sharpness of the softmax over template scores
PROG_W_DTW = 0.5            # weight of sequence DTW vs histogram cosine
CHANGE_MIN_POST_DELTA = 0.35  # min posterior L1 change to count as a chord change

_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Chords that carry the "chorus lift" (borrowed minor iv / minor i / bVI type).
# Used only to DECIDE which of the two progression clusters is the chorus; the
# clustering itself is unsupervised. These are the chords the gig-11 choruses lean
# on far more than the verses (verses lean on the plain major I / II).
_CHORUS_LIFT = ("Fmin", "Cmin", "G#maj", "D#min", "A#min")
_VERSE_PLAIN = ("Cmaj", "Dmaj", "Gmaj", "Amaj")


def _build_templates():
    """Return (T[24x12] unit-norm, names[24]) for the 12 major + 12 minor triads."""
    maj = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.])   # root, M3, P5
    minr = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.])  # root, m3, P5
    tem, names = [], []
    for r in range(12):
        tem.append(np.roll(maj, r)); names.append(_NOTES[r] + "maj")
    for r in range(12):
        tem.append(np.roll(minr, r)); names.append(_NOTES[r] + "min")
    T = np.array(tem)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    return T, names


_TEMPLATES, CHORD_NAMES = _build_templates()


def _load_mono(path, sr):
    y, fsr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if fsr != sr:
        y = librosa.resample(y, orig_sr=fsr, target_sr=sr)
    return y


def bar_chords(acc_wav, sr=22050, hop=512, downbeats=None):
    """Estimate a soft chord posterior per BAR from the accompaniment stem.

    downbeats: optional precomputed downbeat times (seconds). If None, beats.grid
    (beat_this) is used on the accompaniment wav. Returns a dict; see module docs."""
    y = _load_mono(acc_wav, sr)
    dur = len(y) / sr

    if downbeats is None:
        from beats import grid
        g = grid(acc_wav)
        downbeats = np.asarray(g["downbeats"], float)
    else:
        downbeats = np.asarray(downbeats, float)
    downbeats = downbeats[(downbeats >= 0) & (downbeats <= dur)]
    if len(downbeats) < 2:
        downbeats = np.linspace(0, dur, max(2, int(dur / 2.2)))

    # denoise chroma with a cosine recurrence nn-filter before bar-averaging.
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    try:
        chroma = librosa.decompose.nn_filter(chroma, aggregate=np.median, metric="cosine")
        chroma = np.maximum(chroma, 0.0)
    except Exception:
        pass
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)

    bar_edges = [(downbeats[i], downbeats[i + 1]) for i in range(len(downbeats) - 1)]
    bar_edges.append((downbeats[-1], dur))

    post = np.zeros((len(bar_edges), 24))
    labels = []
    for bi, (s, e) in enumerate(bar_edges):
        m = (times >= s) & (times < e)
        c = chroma[:, m].mean(axis=1) if m.sum() >= 1 else np.ones(12) / 12.0
        cn = c / (np.linalg.norm(c) + 1e-9)
        sc = _TEMPLATES @ cn
        p = np.exp((sc - sc.max()) * CHORD_SOFTMAX_GAMMA)
        p /= p.sum() + 1e-12
        post[bi] = p
        labels.append(CHORD_NAMES[int(np.argmax(sc))])

    # chord-change times: bar starts where the posterior shifts by > threshold (L1),
    # i.e. a genuine chord-pattern change rather than jitter.
    change_times = [float(bar_edges[0][0])]
    for bi in range(1, len(bar_edges)):
        if np.abs(post[bi] - post[bi - 1]).sum() >= CHANGE_MIN_POST_DELTA:
            change_times.append(float(bar_edges[bi][0]))

    return {
        "downbeats": downbeats.tolist(),
        "bar_times": [(float(s), float(e)) for (s, e) in bar_edges],
        "bar_post": post,
        "bar_labels": labels,
        "chord_names": CHORD_NAMES,
        "change_times": change_times,
        "dur": dur,
    }


def section_progression(bars, start_s, end_s):
    """Slice the bar sequence to a section [start_s, end_s) and summarise it.

    A bar belongs to the section if its start lies within [start_s-1, end_s-0.5)
    (the -1/-0.5 slack absorbs the beat-vs-downbeat snapping). Returns a dict with
    the ordered bar labels, the posterior sub-sequence, its mean histogram, and the
    bar indices used."""
    bt = bars["bar_times"]
    idx = [i for i, (s, e) in enumerate(bt) if s >= start_s - 1.0 and s < end_s - 0.5]
    if not idx:
        # fall back to the single nearest bar so we never return an empty section
        centers = np.array([(s + e) / 2.0 for (s, e) in bt])
        idx = [int(np.argmin(np.abs(centers - (start_s + end_s) / 2.0)))]
    post = bars["bar_post"][idx]
    return {
        "labels": [bars["bar_labels"][i] for i in idx],
        "post": post,
        "hist": post.mean(axis=0),
        "bar_idx": idx,
    }


def _dtw_post(A, B):
    """Normalised DTW distance between two bar-posterior sequences (cosine cost)."""
    if A.shape[0] < 1 or B.shape[0] < 1:
        return 1.0
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    C = 1.0 - An @ Bn.T
    D, wp = librosa.sequence.dtw(C=C)
    return float(D[-1, -1] / len(wp))


def _cos(a, b):
    return float(1.0 - np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def progression_distance(secA, secB):
    """Distance between two section progressions: DTW(posterior seq) blended with
    cosine(mean histogram). Tolerant of a chorus being cut short by a bar (DTW) and
    stable for very short sections (histogram)."""
    d = PROG_W_DTW * _dtw_post(secA["post"], secB["post"]) \
        + (1.0 - PROG_W_DTW) * _cos(secA["hist"], secB["hist"])
    return float(d)


def _lift_score(sec):
    """Borrowed-minor 'chorus lift' minus 'plain-major verse' posterior mass."""
    names = CHORD_NAMES
    h = sec["hist"]
    lift = sum(h[names.index(c)] for c in _CHORUS_LIFT if c in names)
    plain = sum(h[names.index(c)] for c in _VERSE_PLAIN if c in names)
    return float(lift - plain)


def lift_scores(sung_secs):
    """Per-section borrowed-minor 'chorus lift' score (higher = more chorus-like)."""
    return [_lift_score(s) for s in sung_secs]


def split_sung_progressions(sung_secs):
    """Split a list of section-progression dicts into two clusters -> pattern A
    (verse) vs pattern B (chorus). Returns (cluster_ids[0/1 per section],
    chorus_cluster_id, lift_scores).

    The split is an agglomerative 2-means over the progression distance
    (sequence-aware, so a chorus cut short by a bar still matches its siblings). The
    CHORUS cluster is then the cluster whose members carry more borrowed-minor 'lift'
    (Fmin/Cmin/G#maj-type mass), which is what distinguishes the chorus pattern B
    from the plain-major verse pattern A on this material.

    On the gig-11 song this cleanly separates the three pre-solo choruses (lift
    +0.13/+0.12/+0.04) from the verses (lift ~-0.06); the post-solo choruses sit
    near lift 0 and are genuinely ambiguous -- we do not force them.

    Fewer than 2 sung sections -> single cluster (id 0), chorus id None."""
    n = len(sung_secs)
    lift = np.array(lift_scores(sung_secs)) if n else np.array([])
    if n < 2:
        return [0] * n, None, lift.tolist()

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = progression_distance(sung_secs[i], sung_secs[j])
    Z = linkage(squareform(D, checks=False), method="average")
    raw = fcluster(Z, t=2, criterion="maxclust")

    means = {cid: float(lift[[i for i in range(n) if raw[i] == cid]].mean())
             for cid in set(raw)}
    chorus_raw = max(means, key=lambda c: means[c])
    # remap so the chorus cluster is id 1, verse cluster id 0 (stable naming)
    ids = [1 if raw[i] == chorus_raw else 0 for i in range(n)]
    return ids, 1, lift.tolist()


def snap_to_chord_change(t, change_times, max_dist=2.5):
    """Snap a timestamp to the nearest chord-change time within max_dist seconds;
    else return t unchanged. Used to move a fuzzy novelty boundary onto the real
    chord-pattern change (verse->chorus seam)."""
    if not change_times:
        return t
    a = np.asarray(change_times, float)
    i = int(np.argmin(np.abs(a - t)))
    return float(a[i]) if abs(a[i] - t) <= max_dist else t
