# -*- coding: utf-8 -*-
"""render_preview_fx_v2.py -- CRACKLE-FREE, HIGH-QUALITY single-effect PREVIEW renderer.

WHY THIS REPLACES render_preview_fx.py's approach
    The old renderer split each clip at every effect boundary, cut a fresh sub-segment (with its own
    audio) from the master for each piece, then CONCATENATED + acrossfaded them. Every audio splice
    re-encoded and re-joined the PCM -> clicks / crackle at each boundary.

THE FIX (this file)
    A preview is ONE effect on ONE moment in ONE short clip. So:
      * VIDEO  -- decode the clip ONCE. Apply the effect as a TIME-GATED filter over that single
                  continuous decode, using ffmpeg `enable='between(t,S,E)'` (or, for filters that have
                  no timeline support -- crop/blend -- an envelope expression that is exactly 0 outside
                  the moment window). The effect ramps in/out on the moment; the rest is untouched.
      * AUDIO  -- map the ORIGINAL audio straight through as a SINGLE unbroken stream. It is never cut,
                  never split, never re-concatenated, never crossfaded. No splice = no crackle.

    Effects are taken from effects_lab (same intensity ladders + envelope shapes), but their internal
    time expressions are OFFSET to the moment window so the effect fires mid-clip and the clip's own
    head/tail stay clean.

USAGE
    python render_preview_fx_v2.py <ffmpeg> <ffprobe> <src.mp4> <out.mp4> \
        --clip-start ABS_SECONDS --clip-dur 2.5 \
        --effect shake --moment-at 1.25 --intensity medium [--envd 1.0] \
        [--w 1080] [--h 1920] [--crf 18] [--preset slow] [--fps 60]
    Effect "none" renders a clean baseline (no video filter at all) -- same decode/encode settings.
"""
from __future__ import annotations
import sys, os, math, argparse, subprocess, tempfile, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import effects_lab as EL


def _sendcmd_file_offset(pairs, path, t_offset):
    """Write a sendcmd script but SHIFT every command time by t_offset seconds (so the effect's ramp
    lands on the moment mid-clip, not at t=0). Returns the ffmpeg-escaped path."""
    lines = []
    for t, cmd in pairs:
        lines.append("%.4f %s;" % (t + t_offset, cmd))
    open(path, "w").write("\n".join(lines))
    return path.replace("\\", "/").replace(":", "\\:")


def _base(w, h, fps):
    return "scale=%d:%d,fps=%d" % (w, h, fps)


# ------------------------------------------------------------------ per-effect gated graph builders
# Each returns (graph_str, kind) where kind is 'vf' (single -vf chain) or 'fc' (filter_complex ->[vout]).
# m0 = moment start (s, clip-relative), m1 = moment end, envd = envelope duration (= m1 - m0).

def gated_rgb_split(intensity, envd, m0, w, h, fps, scratch, shape="stab_hold"):
    # FITTED (codex 2026-07-03). TWO regimes, chosen by the caller's shape:
    #   shape='stab_hold'  -> asymmetric STAB R=-peak / B=+0.35*peak, ~2-frame attack, hold, INSTANT
    #                         off (ordinary hit).  ~0.2-0.4s.
    #   shape='pulse_hold' -> SUSTAINED oscillating PRISM (re-measured DWthZY@19): ramp-in attack,
    #                         a PULSING hold that rides at intensity, then a fall-off. ~0.8-1.2s on a
    #                         DROP/climax. Reproduces the measured multi-burst prism, not a flat block.
    # peak calibrated @720w (reel corpus), scaled to the working width.
    peak = EL.level_value("rgb_split", intensity) * w / EL.REF_W
    steps = max(12, int(envd * fps))
    pairs = [(0.0, "rgbashift rh 0"), (0.0, "rgbashift bh 0")]
    for i in range(steps + 1):
        t = envd * i / steps
        e = EL.envelope(shape, i / steps)
        pairs += [(t, "rgbashift rh %d" % (-int(round(peak * e)))),
                  (t, "rgbashift bh %d" % (int(round(peak * 0.35 * e))))]
    pairs += [(envd + 0.01, "rgbashift rh 0"), (envd + 0.01, "rgbashift bh 0")]
    scr = _sendcmd_file_offset(pairs, os.path.join(scratch, "v2_rgb.cmd"), m0)
    # enable-gate the rgbashift so it is a pure passthrough outside [m0, m0+envd]
    g = ("%s,sendcmd=f='%s',rgbashift=rh=0:bh=0:enable='between(t,%.4f,%.4f)'"
         % (_base(w, h, fps), scr, m0, m0 + envd))
    return g, "vf"


def gated_whip(intensity, envd, m0, w, h, fps, scratch, shape="tri_sharp"):
    # FITTED: peak 260px@720w -> min-sharpness ratio 0.26 (real 0.22). PARTIAL: only place at a cut.
    peak = EL.level_value("whip", intensity) * w / EL.REF_W
    steps = max(16, int(envd * fps))
    pairs = [(0.0, "avgblur sizeX 1"), (0.0, "avgblur sizeY 1")]
    for i in range(steps + 1):
        t = envd * i / steps
        e = EL.envelope(shape, i / steps)
        sx = max(1, int(round(peak * e)))
        pairs += [(t, "avgblur sizeX %d" % sx), (t, "avgblur sizeY 1")]
    pairs.append((envd + 0.01, "avgblur sizeX 1"))
    scr = _sendcmd_file_offset(pairs, os.path.join(scratch, "v2_whip.cmd"), m0)
    g = ("%s,sendcmd=f='%s',avgblur=sizeX=1:sizeY=1:enable='between(t,%.4f,%.4f)'"
         % (_base(w, h, fps), scr, m0, m0 + envd))
    return g, "vf"


def gated_light_leak(intensity, envd, m0, w, h, fps, scratch, shape="rise_hold_fall",
                     sat_peak=1.15, warm=True):
    # FITTED (codex: DT-b@23.0): rise .2 / hold .55 / fall .2; warm push via time-ramped gamma_r
    # (static colorbalance was far too weak to hit the measured red-ratio +0.53).
    b_peak = EL.level_value("light_leak", intensity)
    gr_peak = 2.3 * (b_peak / 0.36); gb_dip = 0.5 * (b_peak / 0.36)
    steps = max(16, int(envd * fps))
    pairs = [(0.0, "eq brightness 0"), (0.0, "eq saturation 1"),
             (0.0, "eq gamma_r 1"), (0.0, "eq gamma_b 1")]
    for i in range(steps + 1):
        t = envd * i / steps
        e = max(0.0, min(1.0, EL.envelope(shape, i / steps)))
        pairs += [(t, "eq brightness %.4f" % (b_peak * e)),
                  (t, "eq saturation %.4f" % (1 + (sat_peak - 1) * e))]
        if warm:
            pairs += [(t, "eq gamma_r %.4f" % (1 + gr_peak * e)),
                      (t, "eq gamma_b %.4f" % (1 - gb_dip * e))]
    pairs += [(envd + 0.01, "eq brightness 0"), (envd + 0.01, "eq saturation 1"),
              (envd + 0.01, "eq gamma_r 1"), (envd + 0.01, "eq gamma_b 1")]
    scr = _sendcmd_file_offset(pairs, os.path.join(scratch, "v2_leak.cmd"), m0)
    en = "enable='between(t,%.4f,%.4f)'" % (m0, m0 + envd)
    g = ("%s,sendcmd=f='%s',eq=brightness=0:saturation=1:gamma_r=1:gamma_b=1:%s"
         % (_base(w, h, fps), scr, en))
    return g, "vf"


def gated_flash(intensity, envd, m0, w, h, fps, scratch, period=0.2):
    # FITTED white strobe train (codex: DT-b@33.8-34.5): 5Hz pulses, 1-frame white peaks, washed
    # shoulders. intensity = pulse count (1-3); envd is derived from it (n * period).
    n_pulses = int(EL.level_value("flash", intensity))
    fpp = max(2, int(round(period * fps)))
    total = n_pulses * period
    pairs = [(0.0, "eq brightness 0"), (0.0, "eq saturation 1")]
    nfr = int(total * fps)
    for i in range(nfr + 1):
        t = i / float(fps)
        ph = (i % fpp) / float(fpp)
        e = (1.0 - abs(2.0 * ph - 1.0)) ** 1.8
        pairs += [(t, "eq brightness %.3f" % e), (t, "eq saturation %.3f" % (1 - 0.85 * e))]
    pairs += [(total + 0.01, "eq brightness 0"), (total + 0.01, "eq saturation 1")]
    scr = _sendcmd_file_offset(pairs, os.path.join(scratch, "v2_flash.cmd"), m0)
    en = "enable='between(t,%.4f,%.4f)'" % (m0, m0 + total + 0.02)
    g = "%s,sendcmd=f='%s',eq=brightness=0:saturation=1:%s" % (_base(w, h, fps), scr, en)
    return g, "vf"


def gated_blur_build(intensity, envd, m0, w, h, fps, scratch):
    # FITTED blur build (codex: DYw@62.5-64.3): gblur ramps 0 -> sigma over envd (~2s) INTO a cut.
    sig = EL.level_value("blur_build", intensity) * w / EL.REF_W
    steps = max(20, int(envd * 10))
    pairs = [(0.0, "gblur sigma 0.01")]
    for i in range(steps + 1):
        t = envd * i / steps
        pairs.append((t, "gblur sigma %.2f" % max(0.01, sig * i / steps)))
    scr = _sendcmd_file_offset(pairs, os.path.join(scratch, "v2_blurbuild.cmd"), m0)
    en = "enable='between(t,%.4f,%.4f)'" % (m0, m0 + envd)
    g = "%s,sendcmd=f='%s',gblur=sigma=0.01:%s" % (_base(w, h, fps), scr, en)
    return g, "vf"


def gated_glitch(intensity, envd, m0, w, h, fps, scratch, shape="tri", seed=3):
    peak = EL.level_value("glitch", intensity)
    rnd = random.Random(seed)
    steps = max(24, int(envd * fps))
    pairs = []
    for i in range(steps + 1):
        t = envd * i / steps
        e = EL.envelope(shape, i / steps)
        on = 0 if (i % 3 == 0) else 1
        rh = int(round(peak * e * on * rnd.choice([0.4, 1.0, 0.7])))
        bh = -int(round((peak * 0.8) * e * on))
        gv = int(round((peak * 0.35) * e * on))
        pairs += [(t, "rgbashift rh %d" % rh), (t, "rgbashift bh %d" % bh),
                  (t, "rgbashift gv %d" % gv)]
    scr = _sendcmd_file_offset(pairs, os.path.join(scratch, "v2_glitch.cmd"), m0)
    en = "enable='between(t,%.4f,%.4f)'" % (m0, m0 + envd)
    g = ("%s,sendcmd=f='%s',rgbashift=rh=0:bh=0:gv=0:%s,noise=alls=8:allf=t:%s"
         % (_base(w, h, fps), scr, en, en))
    return g, "vf"


def gated_shake(intensity, envd, m0, w, h, fps, scratch, overscan=1.06, fx_hz=3.0, fy_hz=3.9):
    """FITTED camera bump (codex): ~12px @720w, ~3Hz, 0.45s, 1-2 cycles (NOT a sustained buzz).
    crop has NO timeline support, so we gate via the AMPLITUDE expression: the envelope term is
    multiplied by between(t,m0,m1) and the oscillation phase uses (t-m0). Outside the window the
    displacement is exactly 0, so the (centered) overscan crop is a static, invisible slight zoom."""
    amp = EL.level_value("shake", intensity) * w / EL.REF_W   # px @720w -> working width
    sw, sh = int(w * overscan), int(h * overscan)
    D = envd
    m1 = m0 + D
    tt = "(t-%.4f)" % m0                                   # moment-relative time
    gate = "between(t,%.4f,%.4f)" % (m0, m1)
    env = "(1-abs(2*%s/%f-1))*%s" % (tt, D, gate)          # triangle envelope, zero outside window
    xexpr = "(iw-%d)/2 + %.1f*(%s)*sin(2*PI*%.2f*%s)" % (w, amp, env, fx_hz, tt)
    yexpr = "(ih-%d)/2 + %.1f*(%s)*cos(2*PI*%.2f*%s + 0.7)" % (h, amp * 0.7, env, fy_hz, tt)
    g = ("scale=%d:%d,fps=%d,crop=%d:%d:x='%s':y='%s'"
         % (sw, sh, fps, w, h, xexpr, yexpr))
    return g, "vf"


def gated_radial_zoom(intensity, envd, m0, w, h, fps, scratch, layers=6, shape="in_out"):
    """blend=all_expr with a T-envelope. We offset T by m0 and multiply the envelope by between(T,..)
    so outside the moment window the blend output is exactly the clean base A (no radial)."""
    step = EL.level_value("radial_zoom", intensity)
    D = envd
    m1 = m0 + D
    TT = "(T-%.4f)" % m0
    gate = "between(T,%.4f,%.4f)" % (m0, m1)
    if shape == "in_out":
        env_expr = "((0.5-0.5*cos(2*PI*%s/%f))*%s)" % (TT, D, gate)
    else:  # tri
        env_expr = "((1-abs(2*%s/%f-1))*%s)" % (TT, D, gate)
    parts = ["[0:v]%s,split=%d%s" % (_base(w, h, fps), layers + 2,
             "".join("[s%d]" % i for i in range(layers + 2)))]
    prev = "s1"
    for k in range(1, layers + 1):
        z = 1 + step * k
        sw, sh = int(w * z), int(h * z)
        parts.append("[s%d]scale=%d:%d,crop=%d:%d:(iw-%d)/2:(ih-%d)/2,format=yuva420p,"
                     "colorchannelmixer=aa=%.3f[z%d]" % (k + 1, sw, sh, w, h, w, h, 1.0 / (k + 1), k))
        parts.append("[%s][z%d]overlay[o%d]" % (prev, k, k))
        prev = "o%d" % k
    parts.append("[s0][%s]blend=all_expr='A*(1-%s)+B*%s'[vout]" % (prev, env_expr, env_expr))
    return ";".join(parts), "fc"


GATED = {
    "rgb_split":   gated_rgb_split,
    "whip":        gated_whip,
    "flash":       gated_flash,
    "light_leak":  gated_light_leak,
    "blur_build":  gated_blur_build,
    "shake":       gated_shake,
    # kept for API compat; TRIGGER_RULES forbids placement of these:
    "glitch":      gated_glitch,
    "radial_zoom": gated_radial_zoom,
}


_SHAPE_AWARE = {"rgb_split", "whip", "light_leak"}   # gated builders that accept a `shape` kwarg

def build_gated(effect, intensity, envd, m0, w, h, fps, scratch, shape=None):
    if effect not in GATED:
        raise KeyError("effect %r not supported by v2 gated renderer; known: %s"
                       % (effect, sorted(GATED)))
    if shape and effect in _SHAPE_AWARE:
        return GATED[effect](intensity, envd, m0, w, h, fps, scratch, shape=shape)
    return GATED[effect](intensity, envd, m0, w, h, fps, scratch)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ffmpeg"); ap.add_argument("ffprobe")
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--clip-start", type=float, required=True, help="abs seconds into src")
    ap.add_argument("--clip-dur", type=float, default=2.5)
    ap.add_argument("--effect", required=True, help="effect name or 'none' for baseline")
    ap.add_argument("--moment-at", type=float, default=None,
                    help="clip-relative seconds where the effect PEAK/window centre sits (default: centre)")
    ap.add_argument("--intensity", default="medium")
    ap.add_argument("--envd", type=float, default=None, help="envelope duration (default: effect rule)")
    ap.add_argument("--shape", default=None,
                    help="envelope shape override (e.g. stab_hold | pulse_hold). Drives short-stab vs "
                         "sustained regime for shape-aware effects (rgb_split, whip, light_leak).")
    ap.add_argument("--w", type=int, default=1080); ap.add_argument("--h", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="slow")
    A = ap.parse_args()

    FF, FP = A.ffmpeg, A.ffprobe
    scratch = tempfile.mkdtemp(prefix="rfxv2_")
    W, H, FPS = A.w, A.h, A.fps

    # video codec: HIGH quality preview
    CV = ["-c:v", "libx264", "-preset", A.preset, "-crf", str(A.crf), "-pix_fmt", "yuv420p",
          "-r", str(FPS)]
    # audio: single unbroken stream, straight AAC re-encode of the ORIGINAL audio (no filter graph).
    CA = ["-c:a", "aac", "-b:a", "256k"]

    os.makedirs(os.path.dirname(A.out), exist_ok=True)

    # base input: ONE continuous decode of [clip_start, clip_start+clip_dur]
    inp = ["-ss", "%.4f" % A.clip_start, "-i", A.src, "-t", "%.4f" % A.clip_dur]

    # ALWAYS build a filter_complex that consumes [0:v] and produces [vout]; map [vout] + 0:a.
    # (Using filter_complex for both plain and fx cases avoids all -vf/-map interaction gotchas.
    #  The AUDIO 0:a is mapped straight through untouched -> one unbroken stream, no splice, no crackle.)
    if A.effect == "none":
        # clean baseline: same scale/encode as the fx clips, so it is a fair A/B comparison
        graph = "[0:v]%s,setsar=1[vout]" % _base(W, H, FPS)
        what = "baseline"
    else:
        envd = A.envd if A.envd is not None else EL.TRIGGER_RULES[A.effect]["envelope_duration"]
        # window centre defaults to clip centre; m0 = window start
        centre = A.moment_at if A.moment_at is not None else (A.clip_dur / 2.0)
        m0 = max(0.0, centre - envd / 2.0)
        if m0 + envd > A.clip_dur:                          # keep window inside the clip
            m0 = max(0.0, A.clip_dur - envd)
        vf_graph, kind = build_gated(A.effect, A.intensity, envd, m0, W, H, FPS, scratch, shape=A.shape)
        if kind == "vf":
            graph = "[0:v]%s,setsar=1[vout]" % vf_graph      # wrap the vf chain into a labelled fc
        else:
            graph = vf_graph                                 # already a full fc ending in [vout]
        what = "fx(%s)" % A.effect
    cmd = [FF, "-v", "error", *inp, "-filter_complex", graph,
           "-map", "[vout]", "-map", "0:a:0?", *CV, *CA,
           "-movflags", "+faststart", "-y", A.out]
    _run(cmd, what)

    _report(FF, FP, A.out, W, H, FPS, A.effect)


def _run(cmd, what):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("%s failed:\n%s" % (what, r.stderr[-2000:]))
    return r


def _report(FF, FP, out, W, H, FPS, effect):
    import re
    dur = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out],
                         stdout=subprocess.PIPE).stdout.decode().strip()
    vol = subprocess.run([FF, "-i", out, "-af", "volumedetect", "-f", "null", "-"],
                         stderr=subprocess.PIPE).stderr.decode()
    mean = re.search(r"mean_volume:\s*([-\d.]+)", vol)
    mx = re.search(r"max_volume:\s*([-\d.]+)", vol)
    print("OK %s | effect=%s | %ss | %dx%d@%dfps | mean=%s max=%s"
          % (os.path.basename(out), effect, dur, W, H, FPS,
             mean.group(1) if mean else "?", mx.group(1) if mx else "?"))


if __name__ == "__main__":
    main()
