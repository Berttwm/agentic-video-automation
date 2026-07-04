"""
build_overview.py  -  LONG-FORM RECAP renderer (gig 11).

Executes the spec-driven EDL from overview_plan.py. The signature move is the LONG-FORM
GRADE HELD ACROSS THE CUTS: every clip in a section carries the SAME grade filter, so the
colour is continuous through the fast cuts (energy from cuts, unity from colour). Frame-exact
base render (no drift), then the song-9 bed. Long-form fx per shot:
  bloom_in     cold-open bloom from white
  build_flash  1-2 frame brightness pop on the sharpest build kicks
  whip_smear   motion-blur + rotational smear resolving ON the release hit
  drop_wash    held red/magenta wash + chromatic glitch + decaying shake
  film_burn_out held VHS/chromatic + escalating burn-to-white on the ring-out
  slowmo       pre-drop slow-mo + zoom into the silence
Grades held across cuts: warm / blue (bass groove) / flip (warm->cool final build) /
vibrant (drop+chorus) / warm_cool (pause) / outro (VHS). Reads the gitignored workdir.
Usage: python build_overview.py <workdir>
"""
import sys, os, json, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared"))
import paths, av_sync, camera_match

FF, FP = paths.FFMPEG, paths.FFPROBE
FPS = int(os.environ.get("OV_FPS", "60"))     # source is 60fps -> OV_FPS=60 keeps motion + tighter cuts
FS = FPS / 30.0                                # frame-scale for per-frame zoompan params
OUTW, OUTH = 1080, 1920
UPW, UPH = 1620, 2880
BED = "song_09"; DRUM_OFF = 15.883
CROPS = {"wide": (1.0, .5, .5), "band": (1.28, .5, .37), "tight": (1.7, .5, .40)}

GRADES = {
    "warm": "curves=all='0/0.04 1/0.97',eq=contrast=1.06:saturation=1.10,"
            "colorbalance=rs=0.05:gs=0.03:rh=0.10:gh=0.05,vignette=a=PI/7",
    "vibrant": "curves=all='0/0.03 1/0.99',eq=contrast=1.12:saturation=1.40,"
               "colorbalance=rs=0.12:rh=0.14:bh=-0.04,vignette=a=PI/8",
    "blue": "colorchannelmixer=rr=0.50:rb=0.15:gg=0.70:gb=0.10:br=0.10:bb=1.15,"
            "eq=saturation=1.15:contrast=1.06,vignette=a=PI/7",   # strong held-blue field (overrides venue light)
    "teal_soft": "eq=saturation=1.05:contrast=1.04,colorbalance=bs=0.07:bm=0.04:rm=-0.03,vignette=a=PI/7",
    "warm_cool": "eq=contrast=1.06:saturation=0.88,colorbalance=bs=0.14:rm=-0.04,vignette=a=PI/6",
    "outro": "eq=contrast=1.06:saturation=1.12,colorbalance=rs=0.05:bh=0.04,vignette=a=PI/6",
}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("cmd failed:\n" + " ".join(str(c) for c in cmd[:8]) + " ...\n" + r.stderr[-1400:])
    return r


def grade_of(sh):
    g = sh.get("grade", "warm")
    if g == "flip":                                   # warm<-cool FLIP across the final build
        f = float(sh.get("flip_frac", 0.5))
        rs = round(0.02 + 0.22 * f, 3); bs = round(0.16 * (1 - f), 3)
        rm = round(0.10 * f, 3); sat = round(1.12 + 0.32 * f, 3)
        return f"eq=contrast=1.06:saturation={sat},colorbalance=rs={rs}:rm={rm}:bs={bs},vignette=a=PI/7"
    return GRADES.get(g, GRADES["warm"])


def frame_pipeline(sh):
    """crop/framing -> OUTWxOUTH."""
    crop = sh["crop"]; parts = []
    if crop == "zoomin":                              # slow push-in (pre-drop pause)
        parts.append(f"scale={UPW}:{UPH},zoompan=z='min(zoom+{0.0016/FS:.5f},1.45)':d=1:"
                     f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={OUTW}x{OUTH}")
    elif crop == "punch":                             # zoom-punch entrance (drop)
        parts.append(f"scale={UPW}:{UPH},zoompan=z='if(lte(on,{6*FS:.0f}),1.32-0.32*(on/{6*FS:.0f}),1.0)':d=1:"
                     f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={OUTW}x{OUTH}")
    else:
        z, cx, cy = CROPS.get(crop, CROPS["band"])
        if z != 1.0:
            parts.append(f"crop=iw/{z}:ih/{z}:(iw-iw/{z})*{cx}:(ih-ih/{z})*{cy},scale={OUTW}:{OUTH}")
        else:
            parts.append(f"scale={OUTW}:{OUTH}:force_original_aspect_ratio=increase,crop={OUTW}:{OUTH}")
    return parts


def clip_vf(sh, cm):
    parts = frame_pipeline(sh)
    if sh["angle"] == "drummer" and cm:
        parts.append(cm)                              # camera colour-match
    if sh.get("fx") == "hero_hold":                   # gentle slow push-in on held shots
        parts.append(f"scale={UPW}:{UPH},zoompan=z='min(zoom+{0.0009/FS:.5f},1.16)':d=1:"
                     f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={OUTW}x{OUTH}")
    parts.append(grade_of(sh))                        # HELD grade (same across a section's clips)
    if sh.get("speed", 1.0) != 1.0:
        parts.append(f"setpts={sh['speed']}*PTS")     # slow-mo (video only)
    parts.append(f"fps={FPS}"); parts.append("setsar=1")

    fx = sh.get("fx"); d = sh["dur"]
    if fx == "slowmo":                                # pre-drop: defocus blur on top of the slow-mo
        parts.append("gblur=sigma=6")
    elif fx == "predrop_ramp":                        # INCREASING defocus + chromatic build INTO the drop
        steps = "; ".join(f"{max(0.0, d*f):.3f} gblur sigma {round(1 + 26 * f, 1)}" for f in (0.0, 0.3, 0.6, 0.85, 1.0))
        parts.append(f"sendcmd=c='{steps}',gblur=sigma=1")
        parts.append("rgbashift=rh=-4:bh=4")
    elif fx == "bloom_in":                            # cold-open bloom from white
        parts.append("fade=t=in:st=0:d=0.55:color=white")
    elif fx == "build_flash":                         # sharp 1-2 frame pop on the kick
        parts.append("eq=brightness='0.34*max(0,1-t/0.05)':eval=frame")
    elif fx == "whip_smear":                          # motion-blur + rotate resolving to sharp on the cut
        parts.append("tmix=frames=4")
        parts.append(f"scale={int(OUTW*1.18)}:{int(OUTH*1.18)},"
                     f"rotate='0.16*max(0,1-t/{max(0.12,d):.3f})':c=black:ow={int(OUTW*1.18)}:oh={int(OUTH*1.18)},"
                     f"crop={OUTW}:{OUTH}:(iw-{OUTW})/2:(ih-{OUTH})/2")
        parts.append("colorbalance=rs=0.10:bs=0.10,eq=saturation=1.3")   # magenta swell
    elif fx == "drop_wash":                           # hot red/magenta wash + chromatic glitch + decaying shake
        parts.append("colorbalance=rm=0.22:rs=0.18:rh=0.15:bh=0.06,eq=saturation=1.5:contrast=1.08")
        parts.append("rgbashift=rh=-8:bh=8")
        parts.append("crop=w='iw*0.92':h='ih*0.92':x='(iw*0.08/2)+9*max(0,1-t/0.3)*sin(2*PI*11*t)'"
                     ":y='(ih*0.08/2)+7*max(0,1-t/0.3)*cos(2*PI*13*t)'," + f"scale={OUTW}:{OUTH}")
        parts.append("eq=brightness='0.30*max(0,1-t/0.08)':eval=frame")
    elif fx == "film_burn_out":                       # held VHS + escalating burn-to-white on the ring-out
        parts.append("rgbashift=rh=-5:bh=5")
        parts.append(f"fade=t=out:st={max(0.0, d-1.4):.3f}:d=1.4:color=white")
    # GUARANTEE exact frame count: clone-pad the tail so a filter warm-up (tmix) or a short
    # source read can never drop below the clip's frame bound -> no accumulated drift.
    parts.append("tpad=stop_mode=clone:stop_duration=1.0")
    return ",".join(parts)


def render_base(shots, angles, cm, tmp, total):
    # ABSOLUTE frame boundaries -> each clip an EXACT frame count -> no accumulated drift
    bounds = [round(s["tl_start"] * FPS) for s in shots] + [round(total * FPS)]
    parts = []
    for i, sh in enumerate(shots):
        nf = max(1, bounds[i + 1] - bounds[i])
        src = angles["drummer"]["path"] if sh["angle"] == "drummer" else angles["master"]["path"]
        ss = sh["src_master"] + (DRUM_OFF if sh["angle"] == "drummer" else 0.0)
        # slow-mo needs more source frames read (speed x duration)
        readdur = (nf / FPS) * sh.get("speed", 1.0) + 0.3
        p = os.path.join(tmp, f"s{i:03d}.mp4")
        run([FF, "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{readdur:.3f}", "-i", src,
             "-vf", clip_vf(sh, cm), "-frames:v", str(nf), "-an", "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
             "-fps_mode", "cfr", "-y", p])
        parts.append(p)
    lst = os.path.join(tmp, "list.txt"); open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    silent = os.path.join(tmp, "silent.mp4")
    run([FF, "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-y", silent])
    return silent


def render_final(silent, bed, total, out):
    inp = [FF, "-v", "error", "-i", silent, "-ss", "0", "-t", f"{total:.3f}", "-i", bed]
    fc = f"[1:a]afade=t=out:st={max(0, total-1.4):.2f}:d=1.4[aud]"
    inp += ["-filter_complex", fc, "-map", "0:v", "-map", "[aud]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-y", out]
    run(inp)


def main():
    wd = sys.argv[1]
    angles = json.load(open(f"{wd}/edit_plan.json"))["angles"]
    cm = camera_match.correction_filter(wd)
    edl = json.load(open(f"{wd}/overview_edl.json"))
    shots = edl["shots"]; total = edl["duration"]
    tmp = tempfile.mkdtemp(prefix="ovrecap_")
    silent = render_base(shots, angles, cm, tmp, total)
    out = f"{os.path.dirname(angles['master']['path'])}/_auto_output/OVERVIEW_RECAP{'_60fps' if FPS != 30 else ''}.mp4"
    render_final(silent, f"{wd}/{BED}.wav", total, out)
    v, a, dr, ok = av_sync.verify_av(FP, out, FPS)
    print(f"RECAP v14: {len(shots)} shots, {total:.1f}s, footage songs {edl.get('songs_used')}")
    print(f"A/V lock: video {v:.3f}s / audio {a:.3f}s / drift {dr}f [{'OK' if ok else 'DRIFT'}]")
    print("OUT ->", out)


if __name__ == "__main__":
    main()
