# -*- coding: utf-8 -*-
"""render_timeline_fx_v2.py -- CRACKLE-FREE full-TIMELINE preview renderer for the assembled edit.

This is the timeline sibling of render_preview_fx_v2.py (which renders ONE effect on ONE clip). It
bakes ALL placed effects across the WHOLE assembled edit_plan AND renders the plan's TRANSITIONS and
FADES so the preview matches the intended edit (no more stiff hard cuts).

    TRANSITIONS (defect fix) -- shots are NO LONGER hard-concatenated (`concat -c copy`). Every shot
              boundary is a musical section seam, so adjacent shots are CROSS-DISSOLVED with `xfade`
              (video) + `acrossfade` (audio) over a matching overlap: a fast WHIP-style dissolve
              (~0.35s) at the big JOINS flagged is_join, a gentle dissolve (~0.5s) everywhere else.
              Because both the video xfade and the audio acrossfade use the SAME per-boundary overlap,
              A/V stay locked in sync (each boundary shortens both streams by the same amount).
              A fade-IN from black (fade_in_s) opens the edit and a fade-OUT to black (~0.8s) closes it,
              riding the newly-extended final-note decay (only the last ~0.8s fades, never the ring).

    AUDIO  -- built ONCE from the MASTER at each shot's master-time, then the per-shot segments are
              chained with `acrossfade` over the same overlaps as the video (equal-power tri, no
              clicks -- acrossfade superimposes, it never hard-splices). The result is loudnorm'd ONCE
              and muxed UNCHANGED into BOTH the fx preview and the no-effects baseline, so the two
              outputs are AUDIO-BIT-IDENTICAL by construction (they differ ONLY by the video effects;
              the SAME transitions/fades are applied to both).

    VIDEO  -- each SHOT is decoded ONCE from its own angle source (master or drummer-cam) at
              angle_start_abs. Its effects are applied as TIME-GATED filters over that single decode
              (effects_lab intensity ladders + envelope shapes; rgb_split rides pulse_hold on the
              sustained climax). The per-shot fx mp4s are then blended with the xfade chain above.
              NOTHING in the video path touches audio.

Effects are the SAME gated builders as render_preview_fx_v2 (imported), so a moment renders identically
whether previewed alone or in the timeline. Duration/envelope/regime come from the plan (the approved
moment-driven model): SUSTAINED (pulse_hold) only on the genuinely impactful climax; forbidden effects
are never in the plan.

Usage:
    python render_timeline_fx_v2.py <ffmpeg> <ffprobe> <workdir> <out_mp4>
        [--baseline <baseline_mp4>]   # also render a NO-EFFECTS baseline muxing the SAME audio
        [--w 1080] [--h 1920] [--fps 60] [--crf 18] [--preset slow] [--lufs -14]
"""
from __future__ import annotations
import sys, os, json, subprocess, tempfile, argparse, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_preview_fx_v2 as RV   # reuse the exact gated effect builders
import effects_lab as EL


def run(ff_cmd, what):
    r = subprocess.run(ff_cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("%s failed:\n%s" % (what, r.stderr[-1800:]))
    return r


def probe_dur(fp, path):
    o = subprocess.run([fp, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       stdout=subprocess.PIPE).stdout.decode().strip()
    try:
        return float(o)
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ffmpeg"); ap.add_argument("ffprobe"); ap.add_argument("work"); ap.add_argument("out")
    ap.add_argument("--baseline", default=None, help="also render a no-effects baseline (same audio)")
    ap.add_argument("--w", type=int, default=1080); ap.add_argument("--h", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--lufs", type=float, default=-14.0)
    A = ap.parse_args()
    FF, FP, WORK, OUT = A.ffmpeg, A.ffprobe, A.work, A.out
    W, H, FPS = A.w, A.h, A.fps

    plan = json.load(open(os.path.join(WORK, "edit_plan.json"), encoding="utf-8"))
    shots = plan["shots"]
    angles = plan["angles"]
    master_path = angles["master"]["path"]
    master_off = float(angles["master"].get("offset", 0.0))

    effects = sorted(plan.get("effects", []), key=lambda f: f["tl_start"])
    scratch = tempfile.mkdtemp(prefix="rtlfx_")

    CV = ["-c:v", "libx264", "-preset", A.preset, "-crf", str(A.crf), "-pix_fmt", "yuv420p",
          "-r", str(FPS), "-video_track_timescale", "15360"]
    # portrait fit for any source (drummer cam may differ in aspect); matches the target frame.
    FIT = ("scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2:color=black,"
           "setsar=1,fps=%d" % (W, H, W, H, FPS))

    # ------------------------------------------------------------------ VIDEO: render each shot once
    # A shot is a single continuous decode of its angle source at angle_start_abs for `dur`. Effects
    # whose window falls inside the shot are baked as TIME-GATED (shot-relative) filters over that one
    # decode. No audio in the video path (-an) -> audio can never be spliced by an effect.
    def render_shot_video(sh, si, with_fx, out_mp4):
        angle = sh.get("angle", "master")
        src = angles[angle]["path"]
        src_start = sh.get("angle_start_abs", sh["master_start_abs"])
        dur = float(sh["dur"])
        tl0 = float(sh["tl_start"])
        # effects whose START lands within this shot's timeline window
        fxs = [f for f in effects if tl0 - 1e-6 <= f["tl_start"] < tl0 + dur - 1e-6] if with_fx else []

        inp = ["-ss", "%.4f" % max(src_start, 0.0), "-i", src, "-t", "%.4f" % dur]
        if not fxs:
            graph = "[0:v]%s,setsar=1[vout]" % FIT
        else:
            # chain: FIT first, then each gated effect graph in sequence. Every gated builder is a
            # self-contained vf/fc; we keep to 'vf' builders (all placed effects are vf) and chain them,
            # each time-gated shot-relative so it fires only on its moment and passes through elsewhere.
            chain = FIT
            for f in fxs:
                name = f["effect"]
                envd = float(f.get("envelope_duration") or EL.TRIGGER_RULES[name]["envelope_duration"])
                m_centre = (f["tl_start"] - tl0) + envd / 2.0        # shot-relative window centre
                m0 = max(0.0, m_centre - envd / 2.0)
                if m0 + envd > dur:
                    m0 = max(0.0, dur - envd)
                shape = f.get("envelope")                            # pulse_hold / stab_hold / etc.
                g, kind = RV.build_gated(name, f["intensity"], envd, m0, W, H, FPS, scratch, shape=shape)
                if kind != "vf":
                    raise RuntimeError("timeline renderer expects vf builders; %s is %s" % (name, kind))
                # RV builders start with their own scale/fps (RV._base). Strip that leading base so we
                # can chain after our FIT (which already scaled+padded+fps'd the frame).
                base_prefix = RV._base(W, H, FPS)                    # "scale=..,fps=.."
                if g.startswith(base_prefix + ","):
                    g = g[len(base_prefix) + 1:]
                chain = chain + "," + g
            graph = "[0:v]%s,setsar=1[vout]" % chain
        cmd = [FF, "-v", "error", *inp, "-filter_complex", graph, "-map", "[vout]", "-an", *CV,
               "-y", out_mp4]
        run(cmd, "shot %d video (%s)" % (si, "fx" if with_fx else "plain"))

    # ------------------------------------------------------------------ TRANSITION / FADE geometry
    # Each shot boundary is a musical section seam -> cross-dissolve, never a hard cut. A fast
    # WHIP-style dissolve at the big joins (is_join), a gentle dissolve elsewhere. The overlap for a
    # boundary is capped so it can never exceed a fraction of either neighbouring shot.
    OV_JOIN = 0.35        # whip-style fast dissolve at is_join seams
    OV_DISSOLVE = 0.50    # gentle dissolve at ordinary section seams
    FADE_IN = float(plan.get("fade_in_s", 0.8) or 0.8)
    FADE_OUT = 0.8        # ride only the last ~0.8s so the extended final note still rings out

    def overlaps():
        """overlap seconds for each of the N-1 boundaries, capped to <=40% of the shorter neighbour."""
        ov = []
        for i in range(1, len(shots)):
            want = OV_JOIN if shots[i].get("is_join") else OV_DISSOLVE
            cap = 0.40 * min(float(shots[i - 1]["dur"]), float(shots[i]["dur"]))
            ov.append(max(0.10, min(want, cap)))
        return ov
    OVS = overlaps()

    def build_video(with_fx, tag):
        # 1) render each shot once (fx baked, full dur -- overlaps are consumed by xfade, not by trims)
        seg_files = []
        for si, sh in enumerate(shots):
            seg = os.path.join(scratch, "%s_shot_%02d.mp4" % (tag, si))
            render_shot_video(sh, si, with_fx, seg)
            seg_files.append(seg)
        vid = os.path.join(scratch, "%s_video.mp4" % tag)
        # 2) single filter_complex: xfade-chain all shots, then fade-in-from-black + fade-out-to-black.
        if len(seg_files) == 1:
            dur0 = float(shots[0]["dur"])
            fc = ("[0:v]setpts=PTS-STARTPTS,fade=t=in:st=0:d=%.3f,"
                  "fade=t=out:st=%.3f:d=%.3f[vout]"
                  % (FADE_IN, max(0.0, dur0 - FADE_OUT), FADE_OUT))
            inputs = ["-i", seg_files[0]]
        else:
            inputs = []
            for p in seg_files:
                inputs += ["-i", p]
            parts = []
            for i in range(len(seg_files)):
                parts.append("[%d:v]setpts=PTS-STARTPTS[v%d]" % (i, i))
            # accumulate xfade offsets: boundary k fires at (sum dur 0..k) - (sum overlaps 0..k)
            cur = "[v0]"
            acc = float(shots[0]["dur"])
            total = float(shots[0]["dur"])
            for i in range(1, len(seg_files)):
                ov = OVS[i - 1]
                off = acc - ov
                trans = "fade"                              # cross-DISSOLVE on every seam (video blend)
                out = "[x%d]" % i
                parts.append("%s[v%d]xfade=transition=%s:duration=%.3f:offset=%.3f%s"
                             % (cur, i, trans, ov, off, out))
                cur = out
                acc = acc + float(shots[i]["dur"]) - ov     # chain length after this xfade
                total = acc
            # fades on the final blended stream
            parts.append("%sfade=t=in:st=0:d=%.3f,fade=t=out:st=%.3f:d=%.3f[vout]"
                         % (cur, FADE_IN, max(0.0, total - FADE_OUT), FADE_OUT))
            fc = ";".join(parts)
        run([FF, "-v", "error", *inputs, "-filter_complex", fc, "-map", "[vout]", "-an",
             *CV, "-y", vid], "%s video xfade-chain" % tag)
        return vid

    # ------------------------------------------------------------------ AUDIO: build ONCE, shared
    # Per-shot audio comes from the MASTER at the master-time of that section (a non-master angle maps
    # back to master time via its offset). Segments are concatenated into ONE continuous stream and
    # loudnorm'd ONCE. This exact wav is muxed into BOTH outputs -> audio is bit-identical.
    def build_audio():
        # per-shot MASTER-time segments (full dur -- overlaps consumed by acrossfade, same as video)
        seg_files = []
        for si, sh in enumerate(shots):
            dur = float(sh["dur"])
            angle = sh.get("angle", "master")
            if angle == "master":
                master_time = sh["master_start_abs"]
            else:
                # angle_start_abs is in the angle's source clock; master time = angle_time - angle_offset
                master_time = sh.get("angle_start_abs", sh["master_start_abs"]) - float(angles[angle].get("offset", 0.0))
            seg = os.path.join(scratch, "aud_%02d.wav" % si)
            run([FF, "-v", "error", "-ss", "%.4f" % max(master_time, 0.0), "-i", master_path,
                 "-t", "%.4f" % dur, "-vn", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
                 "-y", seg], "audio seg %d" % si)
            seg_files.append(seg)
        # chain segments with acrossfade over the SAME per-boundary overlaps as the video xfade, so the
        # audio bed shortens by exactly the same amount at each seam and A/V stay locked. acrossfade
        # superimposes (equal-power tri) rather than hard-splicing -> crackle-free at every join.
        raw = os.path.join(scratch, "aud_bed.wav")
        if len(seg_files) == 1:
            shutil.copyfile(seg_files[0], raw)
        else:
            inputs = []
            for p in seg_files:
                inputs += ["-i", p]
            parts = []
            for i in range(len(seg_files)):
                parts.append("[%d:a]aresample=48000,asetpts=PTS-STARTPTS[a%d]" % (i, i))
            cur = "[a0]"
            for i in range(1, len(seg_files)):
                ov = OVS[i - 1]
                out = "[ax%d]" % i
                # qsin = constant-POWER curve (gain_out^2 + gain_in^2 = 1): incoming/outgoing audio
                # sum to a flat loudness through the seam -> no mid-crossfade volume dip. (tri is
                # linear and DIPS when blending two different signals -- reviewers heard that dip.)
                parts.append("%s[a%d]acrossfade=d=%.3f:c1=qsin:c2=qsin%s" % (cur, i, ov, out))
                cur = out
            fc = ";".join(parts)
            run([FF, "-v", "error", *inputs, "-filter_complex", fc, "-map", cur,
                 "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", "-y", raw], "audio acrossfade-chain")
        # fade the audio bed in/out to match the video's fade-from/to-black (same windows)
        adur = probe_dur(FP, raw)
        faded = os.path.join(scratch, "aud_bed_faded.wav")
        run([FF, "-v", "error", "-i", raw, "-af",
             "afade=t=in:st=0:d=%.3f,afade=t=out:st=%.3f:d=%.3f"
             % (FADE_IN, max(0.0, adur - FADE_OUT), FADE_OUT),
             "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", "-y", faded], "audio fades")
        bed = os.path.join(scratch, "aud_bed_norm.wav")
        run([FF, "-v", "error", "-i", faded, "-af", "loudnorm=I=%.1f:TP=-1.0:LRA=11" % A.lufs,
             "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", "-y", bed], "audio loudnorm")
        return bed

    print("=== building shared audio bed (continuous, loudnorm) ===", flush=True)
    bed = build_audio()

    def mux(video, out_mp4, label):
        os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
        run([FF, "-v", "error", "-i", video, "-i", bed, "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest",
             "-movflags", "+faststart", "-y", out_mp4], "%s mux" % label)
        print("  %-9s -> %s (%.2fs)" % (label, out_mp4, probe_dur(FP, out_mp4)), flush=True)

    print("=== rendering FX video (%d effects baked, gated, video-only) ===" % len(effects), flush=True)
    fx_video = build_video(with_fx=True, tag="fx")
    mux(fx_video, OUT, "preview")

    if A.baseline:
        print("=== rendering NO-EFFECTS baseline video (same shots, same audio) ===", flush=True)
        base_video = build_video(with_fx=False, tag="base")
        mux(base_video, A.baseline, "baseline")

    print("DONE. scratch:", scratch)


if __name__ == "__main__":
    main()
