# -*- coding: utf-8 -*-
"""Strong, effect-SPECIFIC detectors for added post effects in a reel (not generic metrics).
Detects: RGB split (chromatic aberration), radial/zoom blur, global motion blur/whip,
camera shake, and pulsing color flash (Red-Alert-like). Signature-based, per-frame.
Usage: python detect_effects.py <reel.mp4> [--fps 12]"""
import sys, subprocess, argparse
import numpy as np
from paths import FFMPEG as FF
ap = argparse.ArgumentParser(); ap.add_argument("reel"); ap.add_argument("--fps", type=int, default=12)
a = ap.parse_args()
W, H = 192, 341

raw = subprocess.run([FF, "-v", "error", "-i", a.reel, "-vf", f"fps={a.fps},scale={W}:{H}",
                      "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True).stdout
F = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(np.float32)
N = len(F); t = lambda i: round(i / a.fps, 2)
R, G, B = F[..., 0], F[..., 1], F[..., 2]
gray = F.mean(3)
print(f"frames={N} @ {a.fps}fps ({N/a.fps:.1f}s)")

def grad_energy(g):  # sharpness
    return (np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1, :]).mean()

# masks for radial analysis
yy, xx = np.mgrid[0:H, 0:W]
cy, cx = H/2, W/2
rr = np.sqrt(((yy-cy)/cy)**2 + ((xx-cx)/cx)**2)
center = rr < 0.45; ring = rr > 0.75

# ---- 1) RGB split: best horizontal R-vs-B shift per frame ----
def rgb_shift(fr):
    r, b = fr[..., 0], fr[..., 2]
    best, bv = 0, 1e9
    for s in range(-4, 5):
        d = np.abs(r[:, max(0, s):W+min(0, s)] - b[:, max(0, -s):W+min(0, -s)]).mean()
        if d < bv: bv, best = d, s
    return abs(best)
# ---- 2) shake: phase-correlation displacement between consecutive frames ----
def phase_shift(g1, g2):
    A = np.fft.rfft2(g1); Bf = np.fft.rfft2(g2)
    Rp = A * np.conj(Bf); Rp /= (np.abs(Rp) + 1e-6)
    r = np.fft.irfft2(Rp, s=g1.shape)
    p = np.unravel_index(np.argmax(r), r.shape)
    dy, dx = p
    if dy > H/2: dy -= H
    if dx > W/2: dx -= W
    return dx, dy

sharp = np.array([grad_energy(gray[i]) for i in range(N)])
edge_sharp = np.array([grad_energy(gray[i]*ring) for i in range(N)])
cen_sharp = np.array([grad_energy(gray[i]*center) for i in range(N)])
rgb = np.array([rgb_shift(F[i]) for i in range(N)])
disp = np.array([phase_shift(gray[i-1], gray[i]) if i else (0, 0) for i in range(N)])
dmag = np.hypot(disp[:, 0], disp[:, 1])
redr = R.mean((1, 2)) / (G.mean((1, 2)) + B.mean((1, 2)) + 1)
mx, mn = F.max(3), F.min(3); sat = ((mx-mn)/(mx+1e-3)).mean((1, 2))

def runs(mask, minlen=2):
    out = []; s = None
    for i in range(N):
        if mask[i] and s is None: s = i
        elif not mask[i] and s is not None:
            if i-s >= minlen: out.append((s, i-1))
            s = None
    if s is not None: out.append((s, N-1))
    return out

# radial blur = edges much less sharp than centre, beyond baseline
radial_ratio = edge_sharp / (cen_sharp + 1e-6)
rb_base = np.median(radial_ratio)
radial = runs(radial_ratio < rb_base*0.6)
# global motion blur / whip = sharpness dips hard vs local median
from numpy.lib.stride_tricks import sliding_window_view
sm = np.median(sharp)
whip = runs(sharp < sm*0.5)
# shake = sustained oscillating displacement (sign flips) with moderate magnitude
signflip = np.array([1 if i > 1 and (disp[i, 0]*disp[i-1, 0] < 0 or disp[i, 1]*disp[i-1, 1] < 0) else 0 for i in range(N)])
shake_score = np.convolve(signflip*(dmag > 1.0), np.ones(5)/5, mode='same')
shake = runs(shake_score > 0.5, minlen=3)
# rgb split = consistent nonzero shift
rgbsplit = runs(rgb >= 2, minlen=2)
# pulsing color flash (red-alert): redRatio OR sat spikes above robust baseline, brief
rmad = np.median(np.abs(redr-np.median(redr)))+1e-6
smad = np.median(np.abs(sat-np.median(sat)))+1e-6
flash = runs(((redr > np.median(redr)+4*rmad) | (sat > np.median(sat)+4*smad)))

def report(name, wins, series=None):
    if not wins:
        print(f"  {name}: none"); return
    segs = []
    for s, e in wins[:8]:
        v = ("pk%.1f" % series[s:e+1].max()) if series is not None else ""
        segs.append(f"{t(s)}-{t(e)}s{('('+v+')') if v else ''}")
    print(f"  {name}: {len(wins)} -> " + ", ".join(segs))

print("STRONG effect detection:")
report("RGB split", rgbsplit, rgb)
report("radial/zoom blur", radial, (1-radial_ratio))
report("motion blur / whip", whip)
report("camera shake", shake, shake_score)
report("color flash (red-alert)", flash, redr)
print("  baselines: sharp med=%.1f radialRatio med=%.2f rgb med=%.1f dispMag med=%.2f"
      % (sm, rb_base, np.median(rgb), np.median(dmag)))
