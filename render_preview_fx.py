# -*- coding: utf-8 -*-
"""render_preview_fx.py -- export the edit to a preview MP4 with the REAL effects_lab effects BAKED in.

This REPLACES the old export_render.py crude approximations (gblur pulse for every effect, brightness bump
for leaks) with the actual, visually-validated ffmpeg renders from effects_lab (EL):

  * PER EFFECT: EL.build_effect_vf(name, intensity, envelope_duration) is applied over the effect's exact
    TIME WINDOW with its ramp envelope (no instant pops), on the correct footage.
  * PER JOIN:   EL.SeamlessTransition().render(style='whip') blends BOTH video (xfade) AND audio
    (acrossfade) across the seam so cuts are seamless -- never a hard chop or a dip to black.
  * AUDIO:      the MASTER audio rides natively on every segment (cut from the same source as the video),
    so it is inherently in sync; joins hand off via acrossfade. Output is loudness-normalised toward the
    reel target (~-14 LUFS) as a final pass.
  * OUTPUT:     a small portrait proxy (default 480x854, 30fps) suitable for a quick visual QA.

ARCHITECTURE (why it stays in sync)
    The forward-skip arrangement collapses to a few CONTIGUOUS master spans joined by transitions. Within
    a span, video+audio is one continuous cut from the master (perfect sync). Each span is rendered by
    splitting it at its effect windows: plain sub-segments are copied straight; effect sub-segments run
    through the effects_lab graph (video) while KEEPING their native audio. Spans are then joined pairwise
    with SeamlessTransition(whip). Everything is re-encoded consistently so concat/xfade is clean.

Usage:
    python render_preview_fx.py <ffmpeg> <ffprobe> <workdir> <out_mp4> [--w 480] [--h 854]
                                [--lufs -14] [--no-normalize]
"""
import sys, os, json, subprocess, tempfile, re, argparse, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import effects_lab as EL

ap = argparse.ArgumentParser()
ap.add_argument("ffmpeg"); ap.add_argument("ffprobe"); ap.add_argument("work"); ap.add_argument("out")
ap.add_argument("--w", type=int, default=480); ap.add_argument("--h", type=int, default=854)
ap.add_argument("--lufs", type=float, default=-14.0)
ap.add_argument("--no-normalize", action="store_true")
ap.add_argument("--preset", default="veryfast")
A = ap.parse_args()
FF, FP, WORK, OUT, W, H = A.ffmpeg, A.ffprobe, A.work, A.out, A.w, A.h
FPS = 30
EL.PREVIEW_W, EL.PREVIEW_H, EL.FPS = W, H, FPS      # make library graphs render at the proxy size/fps

plan = json.load(open(os.path.join(WORK, "edit_plan.json"), encoding="utf-8"))
shots = plan["shots"]
master = plan["angles"]["master"]["path"]
drummer = plan["angles"]["drummer"]["path"]
song_start = None
tmp = tempfile.mkdtemp(prefix="rfx_")
scratch = tmp

AR = 48000                                          # common audio rate for clean concat/xfade
CV = ["-c:v", "libx264", "-preset", A.preset, "-pix_fmt", "yuv420p", "-r", str(FPS)]
CA = ["-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", str(AR)]
# letterbox/scale into the proxy frame (portrait 9:16 preserved)
FIT = ("scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2,setsar=1"
       % (W, H, W, H))


def run(cmd, what):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("%s failed:\n%s" % (what, r.stderr[-1200:]))
    return r


def probe_dur(path):
    o = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       stdout=subprocess.PIPE).stdout.decode().strip()
    try:
        return float(o)
    except Exception:
        return 0.0


# ------------------------------------------------------------------ contiguous master spans
# (a "span" = a run of shots that is continuous in the master source; joins split spans)
spans = []
for sh in shots:
    if not spans or sh.get("is_join"):
        spans.append({"tl0": sh["tl_start"], "abs0": sh["master_start_abs"], "shots": [sh]})
    else:
        spans[-1]["shots"].append(sh)
for s in spans:
    last = s["shots"][-1]
    s["tl1"] = round(last["tl_start"] + last["dur"], 4)
    s["dur"] = round(s["tl1"] - s["tl0"], 4)
    s["abs1"] = round(last["master_start_abs"] + last["dur"], 4)

effects = sorted(plan.get("effects", []), key=lambda f: f["tl_start"])


# ------------------------------------------------------------------ render one segment (plain or FX)
def render_plain(src_abs, dur, out):
    """Cut [src_abs, src_abs+dur] from the master, keep audio, fit to proxy frame."""
    run([FF, "-v", "error", "-ss", "%.4f" % src_abs, "-i", master, "-t", "%.4f" % dur,
         "-vf", FIT, *CV, *CA, "-y", out], "plain segment")


def render_fx(src_abs, dur, f, out):
    """Cut the window, apply the effects_lab graph over the whole window (its ramp is inside), keep audio."""
    name = f["effect"]
    intensity = f["intensity"]
    envd = f.get("envelope_duration") or EL.TRIGGER_RULES[name]["envelope_duration"]
    graph, kind = EL.build_effect_vf(name, intensity=intensity, envelope_duration=envd,
                                     w=W, h=H, scratch=scratch)
    # the library graph already starts with scale=W:H,fps -- but our source is 9:16 like the target,
    # so a plain scale (not letterbox) is what the graph does; that's fine (no black bars needed here).
    base = [FF, "-v", "error", "-ss", "%.4f" % src_abs, "-i", master, "-t", "%.4f" % dur]
    if kind == "vf":
        cmd = base + ["-vf", graph, *CV, *CA, "-y", out]
    else:  # 'fc' -> full filter_complex ending in [vout]; map video from it, audio straight from input
        cmd = base + ["-filter_complex", graph, "-map", "[vout]", "-map", "0:a?", *CV, *CA, "-y", out]
    run(cmd, "fx segment (%s)" % name)


# ------------------------------------------------------------------ build each span (split at fx windows)
def build_span(s, sidx):
    """Render span `s` to a single mp4 with its effects baked; return (path, duration)."""
    tl0, dur_span, abs0 = s["tl0"], s["dur"], s["abs0"]
    # effects whose START falls inside this span (windows are guaranteed inside a span by infer_effects)
    fxs = [f for f in effects if tl0 - 1e-6 <= f["tl_start"] < s["tl1"] - 1e-6]
    # cut points along the span, in span-relative seconds
    cuts = []  # list of (rel_start, rel_end, effect_or_None)
    cursor = 0.0
    for f in fxs:
        rs = f["tl_start"] - tl0
        envd = f.get("envelope_duration") or EL.TRIGGER_RULES[f["effect"]]["envelope_duration"]
        re_ = min(rs + envd, dur_span)
        if rs > cursor + 1e-3:
            cuts.append((cursor, rs, None))
        cuts.append((rs, re_, f))
        cursor = re_
    if cursor < dur_span - 1e-3:
        cuts.append((cursor, dur_span, None))
    if not cuts:
        cuts = [(0.0, dur_span, None)]

    seg_files = []
    for j, (rs, re_, f) in enumerate(cuts):
        seg = os.path.join(tmp, "s%d_%02d.mp4" % (sidx, j))
        seg_abs = abs0 + rs
        seg_dur = re_ - rs
        if seg_dur <= 0.02:
            continue
        if f is None:
            render_plain(seg_abs, seg_dur, seg)
        else:
            render_fx(seg_abs, seg_dur, f, seg)
        seg_files.append(seg)
    # concat the span's sub-segments (tight, same encoder params)
    span_out = os.path.join(tmp, "span%d.mp4" % sidx)
    if len(seg_files) == 1:
        shutil.copy2(seg_files[0], span_out)
    else:
        cat = os.path.join(tmp, "span%d.txt" % sidx)
        open(cat, "w").write("".join("file '%s'\n" % p.replace("\\", "/") for p in seg_files))
        run([FF, "-v", "error", "-f", "concat", "-safe", "0", "-i", cat, *CV, *CA, "-y", span_out],
            "span %d concat" % sidx)
    return span_out, probe_dur(span_out)


print("=== rendering %d spans with baked effects ===" % len(spans))
span_paths = []
for i, s in enumerate(spans):
    p, d = build_span(s, i)
    nfx = sum(1 for f in effects if s["tl0"] - 1e-6 <= f["tl_start"] < s["tl1"] - 1e-6)
    span_paths.append((p, d))
    print("  span %d  tl %.2f-%.2f  dur %.2fs  fx=%d  -> %s"
          % (i, s["tl0"], s["tl1"], d, nfx, os.path.basename(p)))


# ------------------------------------------------------------------ join spans with SeamlessTransition
# pairwise: acc = join(acc, span_i, whip). Each join xfades video + acrossfades audio over `overlap`.
st = EL.SeamlessTransition(ffmpeg=FF, w=W, h=H, fps=FPS)
trans = {round(t["tl_start"], 2): t for t in plan.get("transitions", [])}
DEFAULT_OVERLAP = 0.7

acc_path, acc_dur = span_paths[0]
for i in range(1, len(spans)):
    b_path, b_dur = span_paths[i]
    tj = round(spans[i]["tl0"], 2)
    tinfo = trans.get(tj, {})
    style = tinfo.get("style", "whip")
    overlap = float(tinfo.get("overlap", DEFAULT_OVERLAP))
    overlap = min(overlap, acc_dur - 0.2, b_dur - 0.2)      # never exceed either clip
    joined = os.path.join(tmp, "join_%d.mp4" % i)
    # feed the FULL accumulated clip as A and the FULL next span as B (ta=tb=0, a_len/b_len = full dur)
    st.render(acc_path, 0.0, b_path, 0.0, joined, a_len=acc_dur, b_len=b_dur,
              overlap=overlap, style=style, scratch=scratch, preset=A.preset)
    acc_path, acc_dur = joined, probe_dur(joined)
    print("  join %d  <%s overlap=%.2fs>  -> total %.2fs" % (i, style, overlap, acc_dur))

# ------------------------------------------------------------------ loudness normalise toward ~-14 LUFS
os.makedirs(os.path.dirname(OUT), exist_ok=True)
if A.no_normalize:
    shutil.copy2(acc_path, OUT)
    print("copied (no normalize) -> %s" % OUT)
else:
    # single-pass loudnorm toward target; keeps it audible (~-14 LUFS) and consistent with the reels.
    run([FF, "-v", "error", "-i", acc_path,
         "-af", "loudnorm=I=%.1f:TP=-1.5:LRA=11" % A.lufs,
         *CV, *CA, "-movflags", "+faststart", "-y", OUT], "loudnorm export")
    print("loudnorm (I=%.1f) export -> %s" % (A.lufs, OUT))

# ------------------------------------------------------------------ self-QA the audio + basic stats
out_dur = probe_dur(OUT)
vol = subprocess.run([FF, "-i", OUT, "-af", "volumedetect", "-f", "null", "-"],
                     stderr=subprocess.PIPE).stderr.decode()
mean = re.search(r"mean_volume:\s*([-\d.]+)", vol)
mx = re.search(r"max_volume:\s*([-\d.]+)", vol)
astream = subprocess.run([FP, "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=codec_name,channels", "-of", "default=nw=1", OUT],
                         stdout=subprocess.PIPE).stdout.decode().replace("\n", " ").strip()
print("\n=== PREVIEW EXPORT QA ===")
print("output: %s" % OUT)
print("duration: %.2fs  (%dx%d @%dfps)" % (out_dur, W, H, FPS))
print("audio: %s | mean_volume: %s dB | max_volume: %s dB"
      % (astream or "NONE", mean.group(1) if mean else "?", mx.group(1) if mx else "?"))
print("effects baked: %d | seamless transitions: %d" % (len(effects), len(spans) - 1))

# leave tmp for the caller's montage step; caller cleans scratch at the end
print("\nTMP:", tmp)
