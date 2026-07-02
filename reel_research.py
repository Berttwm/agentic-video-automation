#!/usr/bin/env python3
"""
reel_research.py - Frame-accurate analysis engine for the band reel research.

Given a reel .mp4 (downloaded locally via the IG in-page workaround), extracts frames
at a granular FPS and computes editing-style + effect signals:
  - shot segmentation (cut/transition detection) -> cadence (cuts/min, shot lengths)
  - camera-motion class per shot (locked / handheld / dynamic) + whip-pan spikes
  - color & tint timeline; effect-window flags (color-cast, saturation spike, blur/zoom-blur)
  - title-card / caption band activity (top & bottom text regions)
Outputs a compact JSON report next to the video (<name>.research.json) and prints a summary.

Usage:
  python reel_research.py <file_or_dir> [--fps 12] [--jobs 4]
Parallelizes across files with a process pool (trade CPU for wall-clock).
"""
import subprocess, sys, os, json, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from paths import FFMPEG
W, H = 108, 192  # analysis resolution (keeps aspect ~9:16)

def load_frames(path, fps):
    cmd = [FFMPEG, "-v", "error", "-i", path, "-vf", f"fps={fps},scale={W}:{H}",
           "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(np.float32)

def ffmpeg_cuts(path, thr=0.35):
    """Content-based scene-change detection via ffmpeg (robust vs handheld/strobe)."""
    cmd = [FFMPEG, "-i", path, "-filter:v", f"select='gt(scene,{thr})',showinfo",
           "-f", "null", "-"]
    err = subprocess.run(cmd, capture_output=True).stderr.decode("utf-8", "ignore")
    times = []
    for line in err.splitlines():
        if "pts_time:" in line:
            try:
                times.append(float(line.split("pts_time:")[1].split()[0]))
            except Exception:
                pass
    return sorted(times)

def contiguous(idx):
    """group sorted indices into (start,end) runs"""
    runs = []
    for i in idx:
        if runs and i - runs[-1][1] <= 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return runs

def analyze(path, fps):
    F = load_frames(path, fps)
    N = len(F)
    if N < 3:
        return {"file": os.path.basename(path), "error": "too few frames"}
    t = lambda i: round(i / fps, 2)
    R, G, B = F[..., 0], F[..., 1], F[..., 2]
    Rm, Gm, Bm = R.mean((1, 2)), G.mean((1, 2)), B.mean((1, 2))
    bright = F.mean((1, 2, 3))
    mx, mn = F.max(3), F.min(3)
    sat = ((mx - mn) / (mx + 1e-3)).mean((1, 2))
    # tint / color-cast
    redRatio = Rm / (Gm + Bm + 1)
    blueRatio = Bm / (Rm + Gm + 1)
    grnRatio = Gm / (Rm + Bm + 1)
    magenta = (Rm + Bm) / (2 * Gm + 1)          # high when pink/magenta
    # edge vs center (vignette / edge-glow effects like Red Alert)
    yy, xx = np.mgrid[0:H, 0:W]
    edge = (xx < W * 0.13) | (xx > W * 0.87) | (yy < H * 0.09) | (yy > H * 0.91)
    edgeRed = R[:, edge].mean(1) - R[:, ~edge].mean(1)
    # sharpness (gradient energy) - dips on motion/zoom blur
    gray = F.mean(3)
    grad = (np.abs(np.diff(gray, axis=1))[:, :, :-1] + np.abs(np.diff(gray, axis=2))[:, :-1, :]).mean((1, 2))
    # raw frame-to-frame diff (used only for per-shot motion class)
    diff = np.concatenate([[0], np.abs(np.diff(F.reshape(N, -1), axis=0)).mean(1)])
    # brightness delta -> count lighting strobes (informational, NOT counted as cuts)
    bmean = gray.mean((1, 2))
    bdiff = np.concatenate([[0], np.abs(np.diff(bmean))])
    strobe_ct = len(contiguous([i for i in range(1, N)
                                if bdiff[i] > np.median(bdiff) + 6 * (float(np.median(np.abs(bdiff - np.median(bdiff)))) + 1e-6)]))

    # ---- shot segmentation via ffmpeg content-based scene detection ----
    cut_times = ffmpeg_cuts(path)
    cuts = sorted({min(N - 1, max(1, int(round(ts * fps)))) for ts in cut_times})
    bounds = [0] + cuts + [N]
    shots = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        seg = slice(a, b)
        intra = diff[a + 1:b]
        motion = float(intra.mean()) if len(intra) else 0.0
        jit = float(intra.std()) if len(intra) else 0.0
        cls = "locked" if motion < 6 else ("handheld" if motion < 16 else "dynamic")
        shots.append({"start": t(a), "end": t(b - 1), "dur": round((b - a) / fps, 2),
                      "motion": round(motion, 1), "class": cls})
    durs = [s["dur"] for s in shots] or [N / fps]
    secs = N / fps

    # ---- effect windows ----
    def flag(mask, base, k, extra=None, min_frames=2):
        m = mask
        if extra is not None:
            m = m & extra
        runs = contiguous([i for i in range(N) if m[i]])
        return [{"win": [t(a), t(b)], "peak": round(float(base[a:b + 1].max()), 2)}
                for a, b in runs if b - a + 1 >= min_frames]
    rr0 = float(np.median(redRatio)); mg0 = float(np.median(magenta))
    sat0 = float(np.median(sat)); gr0 = float(np.median(grad))
    effects = {
        "red_cast":      flag(redRatio > rr0 * 1.22, redRatio, None),
        "edge_glow":     flag(edgeRed > 8, edgeRed, None),        # Red-Alert-like edge vignette
        "magenta_cast":  flag(magenta > mg0 * 1.30, magenta, None),
        "sat_spike":     flag(sat > sat0 * 1.30, sat, None),
        "blur_dip":      flag(grad < gr0 * 0.55, grad, None),     # motion/zoom blur (often transitions)
        "whip_or_cut":   [{"t": t(c), "diff": round(float(diff[c]), 0)} for c in cuts],
    }

    # ---- text bands (title top / caption bottom) ----
    topb = bright[:] * 0  # placeholder to keep shape (unused)
    top_region = F[:, :int(H * 0.16), :, :].mean((1, 2, 3))
    bot_region = F[:, int(H * 0.80):, :, :].mean((1, 2, 3))
    # a text band shows localized temporal change vs the frame; report activity
    text = {
        "top_band_activity": round(float(np.std(top_region)), 1),
        "bottom_band_activity": round(float(np.std(bot_region)), 1),
        "note": "high bottom activity ~ karaoke captions; steady bright top ~ persistent title card",
    }

    dominant = ("blue" if np.median(blueRatio) > 0.42 else
                "green" if np.median(grnRatio) > 0.42 else
                "magenta/red" if mg0 > 0.9 else "mixed")

    return {
        "file": os.path.basename(path),
        "fps": fps, "duration_s": round(secs, 1), "frames": N,
        "cadence": {"n_shots": len(shots), "cuts_per_min": round(len(shots) / secs * 60, 1),
                    "mean_shot_s": round(float(np.mean(durs)), 1),
                    "median_shot_s": round(float(np.median(durs)), 1),
                    "min_shot_s": round(float(np.min(durs)), 1),
                    "max_shot_s": round(float(np.max(durs)), 1)},
        "lighting_strobes": strobe_ct,
        "single_take": bool(len(shots) <= 1 or max(durs) > 0.7 * secs),
        "motion_mix": {c: sum(1 for s in shots if s["class"] == c) for c in ("locked", "handheld", "dynamic")},
        "dominant_lighting": dominant,
        "color_baseline": {"redRatio": round(rr0, 2), "magenta": round(mg0, 2), "sat": round(sat0, 2)},
        "effects": effects,
        "text_bands": text,
        "shots": shots,
    }

def run_one(path, fps):
    try:
        return analyze(path, fps)
    except Exception as e:
        return {"file": os.path.basename(path), "error": str(e)[:200]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if os.path.isdir(a.target):
        files = [os.path.join(a.target, f) for f in os.listdir(a.target) if f.lower().endswith(".mp4")]
    else:
        files = [a.target]
    files.sort()
    results = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_one, f, a.fps): f for f in files}
        for fut in as_completed(futs):
            r = fut.result()
            results[r["file"]] = r
            outp = os.path.splitext(futs[fut])[0] + ".research.json"
            with open(outp, "w") as o:
                json.dump(r, o, indent=1)
    # summary
    for name in sorted(results):
        r = results[name]
        if "error" in r:
            print(f"{name}: ERROR {r['error']}"); continue
        c = r["cadence"]; e = r["effects"]
        fx = [k for k in ("red_cast", "edge_glow", "magenta_cast", "sat_spike", "blur_dip") if e[k]]
        tag = "SINGLE-TAKE" if r["single_take"] else "multi-cut"
        print(f"\n{name}  {r['duration_s']}s  {r['dominant_lighting']}  [{tag}]")
        print(f"  true-cuts={c['n_shots']-1} ({c['cuts_per_min']}/min) med-shot={c['median_shot_s']}s "
              f"range {c['min_shot_s']}-{c['max_shot_s']}s  strobes={r['lighting_strobes']}  motion={r['motion_mix']}")
        print(f"  effect-flags: {', '.join(fx) if fx else 'none'}  transitions={len(e['whip_or_cut'])}")
        print(f"  text: top={r['text_bands']['top_band_activity']} bottom={r['text_bands']['bottom_band_activity']}")
    print(f"\nAnalyzed {len(results)} file(s) @ {a.fps}fps. Per-reel JSON written alongside each mp4.")

if __name__ == "__main__":
    main()
