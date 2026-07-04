"""
shared/av_sync.py  -  A/V-sync correctness utilities SHARED by both renderers
(per-song reel + gig overview). Keeping video frames locked to the audio is a
CORRECTNESS concern, not a stylistic one, so it lives here and both pipelines use it.

The bug this prevents: rendering clips whose length is expressed in SECONDS and then
concatenating makes each clip's length round to whole frames; the errors accumulate,
so the video drifts off the audio (overview: cuts slip off the beat; per-song: the
video ends ~1 frame before the audio -> a frozen tail / creeping lip-sync).
"""
import subprocess


def frame_align(t, fps):
    """Snap a time (s) to the nearest frame boundary."""
    return round(round(float(t) * fps) / fps, 6)


def exact_frames(dur, fps):
    """Frame COUNT for a duration -> use with `-frames:v N` (never `-t seconds`)."""
    return max(1, int(round(float(dur) * fps)))


def probe_dur(ffprobe, path, stream=None):
    sel = ["-select_streams", stream] if stream else []
    o = subprocess.run([ffprobe, "-v", "error", *sel, "-show_entries",
                        ("stream=duration" if stream else "format=duration"), "-of", "csv=p=0", path],
                       capture_output=True, text=True).stdout.strip().splitlines()
    return float(o[0]) if o and o[0] not in ("", "N/A") else 0.0


def verify_av(ffprobe, path, fps, tol_frames=1.0):
    """Return (video_dur, audio_dur, drift_frames, ok). ok=True if within tol."""
    v = probe_dur(ffprobe, path, "v:0"); a = probe_dur(ffprobe, path, "a:0")
    drift = abs(v - a) * fps
    return v, a, round(drift, 2), drift <= tol_frames


def mux_locked(ff, ffprobe, video, audio, out, fps):
    """Mux video+audio with the VIDEO length locked to the audio's frame-aligned
    length: pad the video's last frame if short, then cap both to the exact frame
    count. Guarantees video_frames == round(audio_dur*fps) and A/V never drift."""
    a = probe_dur(ffprobe, audio); n = exact_frames(a, fps); dur = n / fps
    subprocess.run([ff, "-v", "error", "-i", video, "-i", audio,
                    "-vf", "tpad=stop_mode=clone:stop_duration=1,fps=%d" % fps,
                    "-map", "0:v:0", "-map", "1:a:0", "-t", "%.5f" % dur,
                    "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k",
                    "-video_track_timescale", "%d" % (fps * 512), "-y", out], check=True)
    return out
