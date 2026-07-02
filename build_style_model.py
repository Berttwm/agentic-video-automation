# -*- coding: utf-8 -*-
"""Research v2 -> STYLE MODEL. The bridge from research to editing.
For each reel: detect cut/effect moments (reuse .research.json) and the MUSICAL EVENTS at those
timestamps (beats/downbeats, energy drops, hits, phrase resolutions). CORRELATE them to learn the
GRAMMAR (which musical event triggers each edit action), then merge best-practices + the CapCut
effect catalog. Output style_model.json that the editor and QA both consume.
Usage: python build_style_model.py <ffmpeg> <reels_dir> <out_json>
"""
import sys, os, json, glob, subprocess, tempfile
import numpy as np
import soundfile as sf
import librosa

FF, REELS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

def load_audio(mp4):
    w = os.path.join(tempfile.gettempdir(), os.path.basename(mp4) + '.sm.wav')
    subprocess.run([FF, '-v', 'error', '-i', mp4, '-vn', '-ac', '1', '-ar', '22050', '-y', w],
                   check=True)
    y = sf.read(w)[0].astype(np.float32)
    try: os.remove(w)
    except Exception: pass
    return y

SR = 22050
def musical_events(y):
    hop = 512
    tempo, beats = librosa.beat.beat_track(y=y, sr=SR, hop_length=hop, units='time')
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.asarray(beats, float)
    # DOWNBEAT phase = the beat-phase (0-3 in 4/4) with the most low-frequency (kick) energy
    mel = librosa.feature.melspectrogram(y=y, sr=SR, hop_length=hop, n_mels=64, fmax=250)
    low = mel[:8].sum(0)
    lt = librosa.frames_to_time(np.arange(low.shape[0]), sr=SR, hop_length=hop)
    if len(beats) >= 4:
        be = np.interp(beats, lt, low)
        phase = max(range(4), key=lambda p: be[p::4].mean() if len(be[p::4]) else 0)
        downbeats = beats[phase::4]
    else:
        downbeats = beats[::4]
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rt = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=hop)
    rms_s = np.convolve(rms, np.ones(9)/9, mode='same')
    # drops = sharp sustained energy jump; builds precede them; resolutions = energy settle (local minima)
    d = np.diff(rms_s, prepend=rms_s[0])
    thr = d.mean() + 2*d.std()
    drops = rt[np.where(d > thr)[0]]
    # phrase resolutions: local minima of smoothed energy (candidate clean cut points)
    from scipy.signal import argrelextrema
    mins = argrelextrema(rms_s, np.less, order=int(SR/hop*1.5))[0]
    resolutions = rt[mins]
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    ot = librosa.times_like(onset, sr=SR, hop_length=hop)
    hits = ot[onset > (np.median(onset) + 3*np.median(np.abs(onset-np.median(onset))))]
    return dict(tempo=tempo, beats=beats, downbeats=downbeats, drops=drops,
                resolutions=resolutions, hits=hits)

def nearest(arr, t):
    return float(np.min(np.abs(arr - t))) if len(arr) else 9e9

cut_downbeat, cut_drop, cut_res, cut_total = 0, 0, 0, 0
eff_drop, eff_hit, eff_total = 0, 0, 0
cadences, tempos = [], []
reels = sorted(glob.glob(os.path.join(REELS, '*.mp4')))
for mp4 in reels:
    rj = os.path.splitext(mp4)[0] + '.research.json'
    if not os.path.exists(rj):
        continue
    R = json.load(open(rj))
    cuts = [c['t'] for c in R.get('effects', {}).get('whip_or_cut', [])]
    eff_wins = []
    for k in ('red_cast', 'magenta_cast', 'sat_spike', 'blur_dip', 'edge_glow'):
        eff_wins += [w['win'][0] for w in R.get('effects', {}).get(k, [])]
    try:
        y = load_audio(mp4)
    except Exception as e:
        print("skip", os.path.basename(mp4), e); continue
    ev = musical_events(y)
    tempos.append(ev['tempo'])
    cadences.append(len(cuts) / (len(y)/SR) * 60)
    for t in cuts:
        cut_total += 1
        if nearest(ev['downbeats'], t) < 0.25: cut_downbeat += 1
        if nearest(ev['drops'], t) < 0.8: cut_drop += 1
        if nearest(ev['resolutions'], t) < 0.8: cut_res += 1
    for t in eff_wins:
        eff_total += 1
        if nearest(ev['drops'], t) < 0.8: eff_drop += 1
        if nearest(ev['hits'], t) < 0.35: eff_hit += 1
    print("  %-30s cuts=%d effs=%d tempo=%.0f" % (os.path.basename(mp4)[10:], len(cuts), len(eff_wins), ev['tempo']), flush=True)

rate = lambda a, b: round(a / b, 2) if b else None
catalog = {}
cp = os.path.join(os.path.dirname(OUT), 'capcut_effects_catalog.json')
if os.path.exists(cp):
    cc = json.load(open(cp))
    catalog = {"effects_available": list(cc.get('used_in_drafts', {}).get('effect', {}).keys()),
               "transitions_available": list(cc.get('used_in_drafts', {}).get('transition', {}).keys())}

model = {
  "generated": "2026-07-02",
  "measured_from": {"reels": len(cadences), "cut_events": cut_total, "effect_events": eff_total},
  "cadence": {"cuts_per_min_median": round(float(np.median(cadences)), 1) if cadences else None,
              "tempo_median": round(float(np.median(tempos)), 0) if tempos else None},
  "cut_grammar": {"on_downbeat_rate": rate(cut_downbeat, cut_total),
                  "at_energy_drop_rate": rate(cut_drop, cut_total),
                  "at_phrase_resolution_rate": rate(cut_res, cut_total),
                  "rule": "snap every cut to the nearest DOWNBEAT; prefer cuts at a phrase RESOLUTION or an energy DROP"},
  "effect_grammar": {"at_drop_rate": rate(eff_drop, eff_total), "on_hit_rate": rate(eff_hit, eff_total),
     "rules": {
        "whip_transition": {"trigger": "section change / energy drop, on a downbeat", "envelope": "motion-blur builds into the cut then resolves", "capcut": "Blur/Mix transition"},
        "shake":  {"trigger": "a strong HIT on a downbeat (chorus/solo accent)", "envelope": "ramp 0->peak->0 over ~1s (keyframed)", "intensity": 0.4, "capcut": "Shake 2"},
        "blur/zoom": {"trigger": "a DROP or the moment of impact after a build", "envelope": "ramp up over the build, release after", "intensity": 0.5, "capcut": "Blur (radial approx via zoom)"},
        "light_leak_flash": {"trigger": "section ENTRY right after a drop", "envelope": "quick in, slow out (~1s)", "capcut": "Leak 2"}
     }},
  "boundary_rules": {
     "clip_end": "extend every clip to the next phrase RESOLUTION (energy settle on a downbeat) so it never ends mid-phrase or on an unresolved note",
     "cut_snap": "downbeat",
     "song_end": "must end on a resolution (final chord/decay), never a hard chop"},
  "tempo_aware_selection": {
     "note": "choose transition/effect family by song tempo & mood (external best-practice)",
     "fast_gt_120bpm": {"transitions": ["whip/Blur", "Mix"], "effects": ["Shake 2", "flash/Leak 2"]},
     "slow_le_120bpm": {"transitions": ["cross-dissolve", "Blur"], "effects": ["subtle Blur", "light leak"]}},
  "techniques": {
     "speed_ramp": "on dynamic beats / into a drop, ramp speed (slow->fast or fast->slow) in time with the track - high impact for band footage, was unused",
     "key_moments_only": "effects/flashy transitions ONLY on key moments (drop/solo/final chorus) - over-syncing looks amateurish"},
  "best_practices": [
     "Effects earn their place: only on a real musical event (drop/hit/build/section change), never on a timer",
     "Every effect ramps in and out (keyframed) matching the build/release of the moment - no instant pops",
     "Cut on the DOWNBEAT for major cuts; don't cut every beat; let phrases finish before cutting",
     "Pick transitions by mood: whip/shake/flash for fast rock, cross-dissolve/blur for slow",
     "Speed ramps on dynamic beats add energy without more footage",
     "Density stays near the reel corpus median; restraint in verses, punch in choruses/solo",
     "Effects support the music, never distract; grade > discrete effects for the overall look"],
  "best_practice_sources": [
     "editorskeys.com - how to edit music videos like a pro",
     "beat2cut.com - beat-sync video editing guide",
     "capcut.com - concert editing (upscale, lighting, synced transitions/text)"],
  "capcut_catalog": catalog,
}
json.dump(model, open(OUT, 'w'), indent=1)
print("\nWROTE", OUT)
print("cadence median: %s cuts/min | cut-on-downbeat: %s | cut-at-drop: %s | cut-at-resolution: %s"
      % (model['cadence']['cuts_per_min_median'], model['cut_grammar']['on_downbeat_rate'],
         model['cut_grammar']['at_energy_drop_rate'], model['cut_grammar']['at_phrase_resolution_rate']))
print("effect-at-drop: %s | effect-on-hit: %s" % (model['effect_grammar']['at_drop_rate'], model['effect_grammar']['on_hit_rate']))
