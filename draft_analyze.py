# -*- coding: utf-8 -*-
"""Quantify the user's real CapCut editing style from draft_content.json files.
Usage: python draft_analyze.py <draft_content.json> [more...]"""
import sys, os, json
import numpy as np
from collections import Counter

def load(p): return json.load(open(p, encoding='utf-8'))

for path in sys.argv[1:]:
    name = os.path.basename(os.path.dirname(path))
    j = load(path)
    mats = j.get('materials', {})
    idmap = {}
    for cat, arr in mats.items():
        if isinstance(arr, list):
            for m in arr:
                if isinstance(m, dict) and 'id' in m:
                    idmap[m['id']] = (cat, m)

    tracks = j.get('tracks', [])
    vtracks = [t for t in tracks if t.get('type') == 'video']
    etracks = [t for t in tracks if t.get('type') == 'effect']
    ttracks = [t for t in tracks if t.get('type') == 'text']

    segs = []
    for ti, t in enumerate(vtracks):
        for s in t.get('segments', []):
            tr = s.get('target_timerange', {}) or {}
            start = tr.get('start', 0) / 1e6
            dur = tr.get('duration', 0) / 1e6
            trans = None
            speed = s.get('speed', None)
            effects, anims = [], []
            for r in (s.get('extra_material_refs', []) or []):
                if r in idmap:
                    cat, m = idmap[r]
                    if cat == 'transitions':
                        trans = m.get('name')
                    elif cat == 'speeds':
                        sp = m.get('speed', None)
                        if sp is not None: speed = sp
                    elif cat == 'video_effects':
                        effects.append(m.get('name'))
                    elif cat == 'material_animations':
                        for a in (m.get('animations') or []):
                            anims.append(a.get('name'))
            segs.append(dict(track=ti, start=start, dur=dur, trans=trans,
                             speed=speed, effects=effects, anims=anims))
    segs.sort(key=lambda x: (x['start']))

    main = [s for s in segs if s['track'] == 0] if vtracks else []
    durs = [s['dur'] for s in main if s['dur'] > 0.05]
    total = j.get('duration', 0) / 1e6
    n = len(main)

    # transitions
    trans_named = [s['trans'] for s in segs if s['trans']]
    # speeds
    speeds = [round(s['speed'], 3) for s in segs if s['speed'] and abs(s['speed'] - 1.0) > 0.01]
    # effects: attached + on effect tracks
    eff_attached = [e for s in segs for e in s['effects']]
    eff_track = []
    for t in etracks:
        for s in t.get('segments', []):
            r = s.get('material_id')
            if r in idmap and idmap[r][0] == 'video_effects':
                eff_track.append(idmap[r][1].get('name'))
            tr = s.get('target_timerange', {}) or {}
    anims = [a for s in segs for a in s['anims']]

    print("=" * 64)
    print("DRAFT: %s   total=%.1fs  fps=%s" % (name, total, j.get('fps')))
    print("  video tracks=%d (multicam layers)  text tracks=%d  effect tracks=%d" % (len(vtracks), len(ttracks), len(etracks)))
    print("  MAIN-track clips=%d" % n)
    if durs:
        a = np.array(durs)
        print("  shot length s: mean=%.2f median=%.2f p25=%.2f p75=%.2f min=%.2f max=%.2f" % (
            a.mean(), np.median(a), np.percentile(a, 25), np.percentile(a, 75), a.min(), a.max()))
        print("  => avg cuts/min on main track: %.1f" % (60.0 / a.mean()))
    print("  TRANSITIONS: %d total across all tracks; coverage on cuts ~ %.0f%%" % (
        len(trans_named), 100.0 * len(trans_named) / max(len(segs) - len(vtracks), 1)))
    print("     names:", dict(Counter(trans_named)))
    print("  SPEED ramps (!=1.0): %d  values:" % len(speeds), dict(Counter(speeds)))
    print("  VIDEO EFFECTS: attached=%d track=%d  names:" % (len(eff_attached), len(eff_track)),
          dict(Counter(eff_attached + eff_track)))
    print("  TEXT/clip ANIMATIONS:", dict(Counter([x for x in anims if x])))
