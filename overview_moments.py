"""
overview_moments.py  -  CROSS-GIG moment miner (D10 / OV-10, OV-24).

A recap's FOOTAGE must come from ACROSS the whole gig, not one song (the user: "there
are way more interesting/engaging moments if you review the entire gig"). This scores
candidate moments across ALL songs from the whole-gig motion timelines + per-section
energy map, and emits a ranked moment_pool.json. The bed timeline (song 9) decides WHEN
cuts land; this pool decides WHICH gig-wide footage fills each slot.

Scoring per candidate window (per camera):
  motion    - frame-difference energy (YAVG of tblend) at that instant, normalized
  energy    - the section's audio energy_ratio (choruses/peaks score high)
  chorus    - bonus if the section is a chorus (hands-up moments)
  entrance  - bonus just after a section start (a hit / a moment lands)
Then diversified: capped per song, both cameras represented, min-spacing so we don't
pick near-duplicate frames. No crowd needed (gig 11) -- the highest-motion / peak-energy
stage moments are the "equally engaging alternative" (D8).

Usage: python overview_moments.py <workdir>  ->  <workdir>/moment_pool.json
"""
import json, os, re, sys

DRUM_OFF = 15.883          # drummer-cam file-time = master-abs instant + DRUM_OFF
WIN = 1.6                  # candidate window length (s)
STEP = 1.2                # candidate spacing within a section (s)


def load_motion(path, to_real=0.0):
    """Return [(real_instant, motion)] from an ffmpeg signalstats(tblend) dump.
    to_real is subtracted from the file timestamp to map into master-abs (real) time."""
    if not os.path.exists(path):
        return []
    txt = open(path, encoding="utf-8", errors="ignore").read()
    ts = [float(x) for x in re.findall(r'pts_time:([\d.]+)', txt)]
    ys = [float(x) for x in re.findall(r'YAVG=([\d.]+)', txt)]
    n = min(len(ts), len(ys))
    return [(ts[i] - to_real, ys[i]) for i in range(n)]


def sampler(series):
    """Nearest-sample lookup for a sparse (time, value) series."""
    if not series:
        return lambda t: 0.0
    xs = [p[0] for p in series]
    def at(t):
        lo, hi = 0, len(xs) - 1
        if t <= xs[0]: return series[0][1]
        if t >= xs[-1]: return series[-1][1]
        while lo < hi:
            mid = (lo + hi) // 2
            if xs[mid] < t: lo = mid + 1
            else: hi = mid
        # nearest of lo-1, lo
        a = series[max(0, lo - 1)]; b = series[lo]
        return a[1] if abs(a[0] - t) <= abs(b[0] - t) else b[1]
    return at


def window_motion(at, t, half=WIN / 2, n=5):
    return sum(at(t + (i / (n - 1) - 0.5) * 2 * half) for i in range(n)) / n


def main(wd):
    songs = json.load(open(f"{wd}/songs.json", encoding="utf-8"))["songs"]
    sec_rows = [json.loads(l) for l in open(f"{wd}/sections.json", encoding="utf-8") if l.strip()] \
        if not _is_json_array(f"{wd}/sections.json") else json.load(open(f"{wd}/sections.json", encoding="utf-8"))

    front = load_motion(f"{wd}/motion_front.txt", 0.0)                 # master-abs already
    drummer = load_motion(f"{wd}/motion_drummer.txt", DRUM_OFF)        # file-time -> master-abs
    at_front, at_drum = sampler(front), sampler(drummer)
    drum_real_max = max((p[0] for p in drummer), default=0.0)

    # section lookup keyed by absolute (master) time
    sects = []
    for row in sec_rows:
        si = row.get("song_index")
        for s in row.get("sections", []):
            sects.append({"song": si, "label": s.get("label", "verse"),
                          "a": s.get("start_absolute"), "b": s.get("end_absolute"),
                          "er": s.get("energy_ratio", 1.0)})

    def section_at(t):
        for s in sects:
            if s["a"] is not None and s["a"] <= t < s["b"]:
                return s
        return None

    cams = [("front", at_front, 1e18), ("drummer", at_drum, drum_real_max)]
    cand = []
    for s in sects:
        a, b = s["a"], s["b"]
        if a is None or b - a < 1.0:
            continue
        t = a + 0.4
        while t < b - 0.4:
            entrance = max(0.0, 1.0 - (t - a) / 2.0)          # 1.0 at section start -> 0 at +2s
            chorus = 0.45 if s["label"] == "chorus" else (0.15 if s["label"] == "outro" else 0.0)
            for cam, at, tmax in cams:
                if t > tmax - 0.5:
                    continue
                m = window_motion(at, t)
                cand.append({"src_master": round(t, 2), "angle": cam, "song": s["song"],
                             "label": s["label"], "motion_raw": round(m, 2),
                             "energy_ratio": round(s["er"], 3),
                             "_entr": entrance, "_chor": chorus})
            t += STEP

    if not cand:
        raise SystemExit("no candidates - check motion/section inputs")

    # normalize motion PER-CAMERA (drummer always moves more; a great front moment must
    # compete against front, not lose to drummer's baseline), then score.
    for cam, _, _ in cams:
        cm = sorted(c["motion_raw"] for c in cand if c["angle"] == cam)
        if not cm:
            continue
        lo_m, hi_m = cm[int(0.05 * len(cm))], cm[int(0.95 * len(cm))]
        span = max(1e-6, hi_m - lo_m)
        for c in cand:
            if c["angle"] == cam:
                c["motion"] = round(min(1.0, max(0.0, (c["motion_raw"] - lo_m) / span)), 3)
    for c in cand:
        c["score"] = round(1.00 * c["motion"] + 1.4 * (c["energy_ratio"] - 1.0)
                           + c["_chor"] + 0.30 * c["_entr"], 3)
        # vibe is MUSICAL energy (for matching to a bed slot); motion is separate (hero pick)
        c["vibe"] = "peak" if c["energy_ratio"] >= 1.15 else \
                    ("low" if c["energy_ratio"] < 0.88 else "mid")

    cand.sort(key=lambda c: -c["score"])

    # stratified diversify: a balanced PALETTE, not just the loudest choruses.
    # caps keep both angles and a real spread of energy so calm slots (intro/groove) and
    # peak slots (drop/chorus) can each be filled from gig-wide footage.
    TARGET = 90
    cap_angle = {"front": int(TARGET * 0.55), "drummer": int(TARGET * 0.55)}
    cap_vibe = {"peak": int(TARGET * 0.55), "mid": int(TARGET * 0.60), "low": int(TARGET * 0.30)}
    picked, per_song, n_angle, n_vibe = [], {}, {"front": 0, "drummer": 0}, {"peak": 0, "mid": 0, "low": 0}

    def spaced(c):
        for p in picked:
            if p["song"] == c["song"] and p["angle"] == c["angle"] and abs(p["src_master"] - c["src_master"]) < 3.0:
                return False
        return per_song.get(c["song"], 0) < 14

    def take(c):
        picked.append(c); per_song[c["song"]] = per_song.get(c["song"], 0) + 1
        n_angle[c["angle"]] += 1; n_vibe[c["vibe"]] += 1

    # pass 1: respect angle+vibe caps for balance
    for c in cand:
        if len(picked) >= TARGET: break
        if spaced(c) and n_angle[c["angle"]] < cap_angle[c["angle"]] and n_vibe[c["vibe"]] < cap_vibe[c["vibe"]]:
            take(c)
    # pass 2: fill any remaining slots ignoring caps (still spaced)
    for c in cand:
        if len(picked) >= TARGET: break
        if c not in picked and spaced(c):
            take(c)

    picked.sort(key=lambda c: -c["score"])
    for i, c in enumerate(picked):
        c["rank"] = i + 1
        for k in ("_entr", "_chor", "motion_raw"):
            c.pop(k, None)

    pool = {"drum_off": DRUM_OFF, "n": len(picked),
            "songs_represented": sorted(per_song.keys()),
            "by_vibe": {v: sum(1 for c in picked if c["vibe"] == v) for v in ("peak", "mid", "low")},
            "moments": picked}
    json.dump(pool, open(f"{wd}/moment_pool.json", "w", encoding="utf-8"), indent=2)
    print(f"moment_pool.json: {len(picked)} moments across songs {sorted(per_song.keys())}")
    print(f"  by vibe: {pool['by_vibe']}")
    print(f"  per song: {dict(sorted(per_song.items()))}")
    print("  top 12:")
    for c in picked[:12]:
        print(f"    #{c['rank']:2d} song{c['song']} {c['angle']:7} @{c['src_master']:7.1f}s "
              f"{c['label']:6} motion={c['motion']:.2f} er={c['energy_ratio']:.2f} score={c['score']:.2f} [{c['vibe']}]")
    return pool


def _is_json_array(path):
    with open(path, encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch: return False
            if ch.isspace(): continue
            return ch == "["


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python overview_moments.py <workdir>")
    main(sys.argv[1])
