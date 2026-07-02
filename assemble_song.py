# -*- coding: utf-8 -*-
"""Assemble ONE song into edit_plan.json using detected STRUCTURE (structure.json).
v2: gate bad parts, DEDUP repeated parts keeping the CLEANER take, keep musical order,
mark joins for beat-matched audio crossfade + transition, fade in/out, plan a title card,
and ONE drummer-cam switch on an interior peak part (if the drummer cam covers it).
Usage: python assemble_song.py <workdir> <draft_name> [--song N] [--min 45] [--max 120]"""
import sys, os, json, argparse, subprocess
import numpy as np
import soundfile as sf
import librosa

ap = argparse.ArgumentParser()
ap.add_argument("work"); ap.add_argument("name")
ap.add_argument("--song", type=int, default=0)
ap.add_argument("--min", type=float, default=45.0)
ap.add_argument("--max", type=float, default=120.0)
ap.add_argument("--target", type=float, default=85.0, help="target duration; trim excess repeats down to this")
ap.add_argument("--no-switch", action="store_true")
a = ap.parse_args()
SR = 22050
from paths import FFPROBE

A = json.load(open(os.path.join(a.work, "analysis.json")))
meta = A["_meta"]; master = meta["master"]
ranking = meta["ranking"]; second = ranking[1] if len(ranking) > 1 else master
offs = meta.get("sync_offset_refined", meta["sync_offset"])
structure = json.load(open(os.path.join(a.work, "structure.json")))
songs = json.load(open(os.path.join(a.work, "songs.json")))
songs = songs if isinstance(songs, list) else songs.get("songs", songs)

def media_dur(p):
    try:
        return float(subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                    "-of", "csv=p=0", p], stdout=subprocess.PIPE).stdout.decode().strip())
    except Exception:
        return 0.0

def cleanliness(x):
    """higher = cleaner take (good SNR/dynamics, tight timing, no clipping)."""
    if len(x) < SR // 2:
        return -1e9, {"clip": 0}
    peak = float(np.max(np.abs(x))); rms = float(np.sqrt(np.mean(x ** 2)))
    clip = float(np.mean(np.abs(x) > 0.985)) * 100
    fr, hop = 2048, 1024
    n = max(1, 1 + (len(x) - fr) // hop)
    rf = np.array([np.sqrt(np.mean(x[i*hop:i*hop+fr]**2)) for i in range(n)]); rf = rf[rf > 0]
    if len(rf) < 2:
        return -1e9, {"clip": clip}
    snr = 20*np.log10(np.percentile(rf, 75)/max(np.percentile(rf, 5), 1e-9))
    crest = 20*np.log10(max(peak, 1e-9)/max(rms, 1e-9))
    onsets = librosa.onset.onset_detect(y=x, sr=SR, units='time')
    if len(onsets) > 3:
        ioi = np.diff(onsets); rhythm = 1.0/(1.0+float(np.std(ioi)/(np.mean(ioi)+1e-9)))
    else:
        rhythm = 0.5
    score = snr*1.0 - clip*3.0 + crest*0.3 + rhythm*10.0
    return round(float(score), 2), {"snr": round(float(snr), 1), "crest": round(float(crest), 1),
                                    "clip": round(clip, 3), "rhythm": round(rhythm, 3)}

def gate_ok(m):
    return m.get("clip", 0) <= 1.0 and m.get("rhythm", 1) >= 0.30 and m.get("snr", 99) >= 5.0

# ---- score every part of every song; build candidate arrangements ----
def build_song(st):
    idx = st["song_index"]
    wav = os.path.join(a.work, "song_%02d.wav" % idx)
    y = sf.read(wav)[0].astype(np.float32)
    if y.ndim > 1: y = y.mean(axis=1)
    song_abs = songs[idx-1].get("start", 0.0) if idx-1 < len(songs) else 0.0
    parts = []
    for p in st["parts"]:
        seg = y[int(p["start"]*SR):int(p["end"]*SR)]
        sc, m = cleanliness(seg)
        parts.append({**p, "score": sc, "metrics": m, "passed": gate_ok(m),
                      "abs_start": round(song_abs + p["start"], 3)})
    return idx, parts

# ---- choose song ----
scored = {}
print("song  keptParts  keptDur  meanScore  structure")
cands = []
for st in structure:
    idx, parts = build_song(st)
    scored[idx] = parts
    kept = [p for p in parts if p["passed"]]
    dur = sum(p["dur"] for p in kept); mean = np.mean([p["score"] for p in kept]) if kept else -1e9
    seq = " ".join("%s%s" % (p["label"][:3], "*" if p["is_repeat"] else "") for p in parts)
    print("  %2d      %d/%d      %5.1fs   %6.1f    %s" % (idx, len(kept), len(parts), dur, mean, seq))
    if kept: cands.append((idx, dur, mean))

if a.song:
    idx = a.song
else:
    inrange = [(i, d, m) for (i, d, m) in cands if a.min <= d <= a.max]
    idx = max(inrange or cands, key=lambda x: x[2])[0]
parts = scored[idx]
print("\n=> chosen song %d" % idx)

# ---- FORWARD-ONLY, SKIP-REPEATS arrangement (the user: always move forward; only skip repeats) ----
# Walk the song in time; keep each section the first time it appears (the hook/most-repeated cluster
# may return ONCE for a reprise); skip further repeats. Strictly forward -> can never jump backwards.
from collections import Counter
kept = [p for p in parts if p["passed"]]
kept.sort(key=lambda p: p["start"])
for p in kept:
    p["_id"] = id(p)
cnt = Counter(p["cluster"] for p in kept)
hook = max(cnt, key=lambda c: (cnt[c], sum(q["energy_ratio"] for q in kept if q["cluster"] == c))) if cnt else None
used, arr = Counter(), []
for p in kept:
    c = p["cluster"]
    limit = 2 if (c == hook and cnt[c] >= 2) else 1     # hook/chorus may recur once; all else kept once
    if used[c] < limit:
        arr.append(p); used[c] += 1                     # else: SKIP this repeat (keep moving forward)
if kept and kept[-1]["_id"] not in {x["_id"] for x in arr}:
    arr.append(kept[-1])                                # always land on the real ending
arr.sort(key=lambda p: p["start"])                       # STRICTLY forward
def tot(): return sum(p["dur"] for p in arr)
while tot() > a.target and len(arr) > 3:                  # trim: drop weakest interior part
    victim = min(arr[1:-1], key=lambda p: p["score"])
    arr.remove(victim)
for i, p in enumerate(arr):                               # display-only role names (logic is cluster-based)
    p["role_name"] = ("intro" if i == 0 else "outro" if i == len(arr) - 1
                      else "chorus" if p["cluster"] == hook
                      else "solo" if (p["energy_ratio"] > 1.1 and cnt[p["cluster"]] == 1) else "verse")
print("  forward-skip: %d parts -> %d kept (%.0fs). order: %s"
      % (len(parts), len(arr), tot(), " -> ".join(p["role_name"] for p in arr)))

# ---- drummer switch on interior peak part, if covered ----
drum_off = offs.get(second, 0.0)
drum_dur = media_dur(A[second]["path"]) if second != master else 0.0
def covers(p):
    return second != master and drum_dur > 0 and (p["abs_start"] + drum_off + p["dur"]) <= drum_dur - 1.0
interior = list(range(1, len(arr) - 1))
peak_i = max(interior, key=lambda i: arr[i].get("energy_ratio", 0)) if interior else -1
if peak_i >= 0 and (a.no_switch or not covers(arr[peak_i])):
    peak_i = -1

# ---- build shots; joins where kept parts are non-contiguous (a part was dropped between) ----
shots = []; tl = 0.0; prev_end_abs = None
for i, p in enumerate(arr):
    join = (prev_end_abs is not None and abs(p["abs_start"] - prev_end_abs) > 0.3)
    angle = "drummer" if i == peak_i else "master"
    ang_off = offs.get(second if angle == "drummer" else master, 0.0)
    trans = "Blur" if (join or i == peak_i or (peak_i >= 0 and i == peak_i + 1)) else None
    lbl = p.get("role_name", p["label"])
    shots.append({
        "tl_start": round(tl, 3), "dur": round(p["dur"], 3),
        "part_label": lbl, "cluster": p["cluster"], "energy_ratio": p.get("energy_ratio", 1.0),
        "angle": angle, "master_start_abs": p["abs_start"],
        "angle_start_abs": round(p["abs_start"] + ang_off, 3),
        "transition_in": trans, "is_join": join,
        "crossfade_in": join,   # beat-matched audio crossfade at joins
        "reason": ("crossfade join between %s sections" % lbl if join
                   else "drummer cam on %s" % lbl if angle == "drummer"
                   else "continue %s" % lbl),
    })
    tl += p["dur"]; prev_end_abs = p["abs_start"] + p["dur"]

plan = {
    "name": a.name, "song_index": idx, "song_title": "Song %d" % idx,
    "master": master,
    "angles": {"master": {"path": A[master]["path"], "offset": 0.0},
               "drummer": {"path": A[second]["path"], "offset": drum_off}},
    "duration": round(tl, 3),
    "n_shots": len(shots),
    "n_joins": sum(1 for s in shots if s["is_join"]),
    "n_angle_switches": sum(1 for s in shots if s["angle"] != "master"),
    "n_transitions": sum(1 for s in shots if s["transition_in"]),
    "crossfade_s": 0.6,
    "fade_in_s": 0.8, "fade_out_s": 1.2,
    "title_card": {"text": "THE BAND", "sub": "(song title - edit in CapCut)", "style": "white_top"},
    "shots": shots,
    "effects": [],  # filled below
}
# effects: Blur/Shake 2 on the 2 highest-energy kept parts (longer, on the hit)
by_e = sorted(range(len(shots)), key=lambda i: shots[i]["energy_ratio"], reverse=True)[:2]
FX = ["Shake 2", "Blur"]
plan["effects"] = [{"tl_start": round(shots[i]["tl_start"] + 0.1, 2), "name": FX[k % 2], "dur": 1.5,
                    "reason": "hit on high-energy %s" % shots[i]["part_label"]}
                   for k, i in enumerate(sorted(by_e))]

out = os.path.join(a.work, "edit_plan.json")
json.dump(plan, open(out, "w"), indent=2)
print("WROTE %s  dur=%.1fs shots=%d joins=%d switches=%d transitions=%d"
      % (out, tl, len(shots), plan["n_joins"], plan["n_angle_switches"], plan["n_transitions"]))
for s in shots:
    print("   %5.1f-%5.1fs %-7s %-7s%s  (%s)" % (s["tl_start"], s["tl_start"]+s["dur"],
          s["part_label"], s["angle"], "  <"+s["transition_in"]+">" if s["transition_in"] else "", s["reason"]))
