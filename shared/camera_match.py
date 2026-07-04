"""
shared/camera_match.py  -  colour-match a non-master angle (the drummer cam) to the
master (front) cam so the grade is CONSISTENT across cameras. SHARED device: both the
per-song reel and the gig overview cut between these two cameras, so both need it.
This is a consistency/correctness step, not a stylistic grade.
"""
import json, os, subprocess


def correction_filter(workdir):
    """The ffmpeg filter mapping the drummer cam -> the master cam's colour for this
    gig, or '' if not computed. Written by compute() into <workdir>/camera_match.json."""
    p = os.path.join(workdir, "camera_match.json")
    if os.path.exists(p):
        return json.load(open(p)).get("filter", "")
    return ""


def compute(ffmpeg, master_path, drummer_path, drummer_offset, out_json,
            sample_times=None, drum_maxsrc=2320):
    """Sample both cameras across the gig, derive a per-channel gain that aligns the
    drummer cam's mean colour to the master, and write camera_match.json."""
    import numpy as np
    from PIL import Image
    st = sample_times or [300, 600, 900, 1200, 1500, 1800, 2100, 2300]

    def mean_rgb(path, times):
        acc = []
        for i, t in enumerate(times):
            p = out_json + f".cm{i}.png"
            subprocess.run([ffmpeg, "-v", "error", "-ss", f"{t}", "-i", path, "-frames:v", "1",
                            "-vf", "scale=200:356", "-y", p], check=True)
            acc.append(np.asarray(Image.open(p).convert("RGB"), float).reshape(-1, 3).mean(0))
            os.remove(p)
        return np.mean(acc, 0)

    fm = mean_rgb(master_path, st)
    dm = mean_rgb(drummer_path, [t + drummer_offset for t in st if t + drummer_offset < drum_maxsrc])
    g = np.clip(fm / np.maximum(dm, 1e-3), 0.75, 1.35)
    filt = f"colorchannelmixer=rr={g[0]:.3f}:gg={g[1]:.3f}:bb={g[2]:.3f}"
    json.dump({"drummer_gain": [round(float(x), 4) for x in g], "filter": filt},
              open(out_json, "w"), indent=2)
    return filt
