# -*- coding: utf-8 -*-
"""QA Layer A - hard structural gates on a generated CapCut draft.
Verifies the draft will open and matches the edit intent before the user reviews it.
Usage: python qa_check.py <draft_root> <draft_name> <workdir>
Exit 0 = all gates pass; exit 1 = a gate failed (prints specifics)."""
import sys, os, json

DRAFT_ROOT, NAME, WORK = sys.argv[1], sys.argv[2], sys.argv[3]
folder = os.path.join(DRAFT_ROOT, NAME)
doc = json.load(open(os.path.join(folder, 'draft_content.json'), encoding='utf-8'))
plan = json.load(open(os.path.join(WORK, 'edit_plan.json'), encoding='utf-8'))

fails, warns = [], []

# gather all material ids
mat_ids = set()
mat_by_id = {}
for cat, lst in doc['materials'].items():
    if isinstance(lst, list):
        for m in lst:
            if isinstance(m, dict) and 'id' in m:
                mat_ids.add(m['id']); mat_by_id[m['id']] = (cat, m)

# 1) every segment material_id + extra_material_refs resolve
n_seg = 0
for tr in doc['tracks']:
    for seg in tr.get('segments', []):
        n_seg += 1
        mid = seg.get('material_id')
        if mid and mid not in mat_ids:
            fails.append("segment %s -> missing material_id %s" % (seg.get('id','?')[:8], mid[:8]))
        for r in seg.get('extra_material_refs', []):
            if r not in mat_ids:
                fails.append("segment %s -> dangling ref %s" % (seg.get('id','?')[:8], r[:8]))

# 2) video source files exist + source_timerange within media duration
for m in doc['materials'].get('videos', []):
    p = m.get('path')
    if p and not os.path.exists(p):
        fails.append("source file missing: %s" % p)
for tr in doc['tracks']:
    for seg in tr.get('segments', []):
        cat_m = mat_by_id.get(seg.get('material_id'))
        st = seg.get('source_timerange')
        if cat_m and cat_m[0] == 'videos' and st:
            mdur = cat_m[1].get('duration', 0)
            if st['start'] + st['duration'] > mdur + 100000:  # 0.1s tol
                fails.append("source range exceeds media: seg %s (%.1fs>%.1fs)" %
                             (seg.get('id','?')[:8], (st['start']+st['duration'])/1e6, mdur/1e6))

# 3) single song
if not plan.get('song_index'):
    fails.append("edit_plan has no single song_index")

# 4) effects present + placed
n_fx_mat = len(doc['materials'].get('video_effects', []))
n_fx_seg = sum(len(t.get('segments', [])) for t in doc['tracks'] if t.get('type') == 'effect')
if n_fx_mat == 0 or n_fx_seg == 0:
    warns.append("no effects placed (effects=%d seg=%d)" % (n_fx_mat, n_fx_seg))
# RENDER gate: effect-track effects MUST be apply_target_type=2 (global) or they render onto the
# hidden base track and never appear (the recurring "effects don't show" bug).
ve_by_id = {m['id']: m for m in doc['materials'].get('video_effects', [])}
bad_apply = 0
for t in doc['tracks']:
    if t.get('type') != 'effect':
        continue
    for s in t.get('segments', []):
        m = ve_by_id.get(s.get('material_id'))
        if m and int(m.get('apply_target_type', 0)) != 2:
            bad_apply += 1
if bad_apply:
    fails.append("%d effect-track effects have apply_target_type!=2 -> will NOT render over the video" % bad_apply)
n_efftracks = sum(1 for t in doc['tracks'] if t.get('type') == 'effect')
if n_efftracks > 1:
    warns.append("%d effect tracks (expected 1) -> possible stale/duplicate track" % n_efftracks)

# 5) transitions present (>=1 expected when there is a stitch or angle switch)
n_trans = len(doc['materials'].get('transitions', []))
expect_trans = plan.get('n_transitions', plan.get('n_stitches', 0) + plan.get('n_angle_switches', 0))
if expect_trans > 0 and n_trans < expect_trans:
    fails.append("expected %d transitions, found %d" % (expect_trans, n_trans))

# 6) no trace of 0218
if '0218' in json.dumps(doc.get('materials', {})) or '0218' in json.dumps(plan):
    fails.append("0218 reference found in draft/plan")

# 7) audio present: master audio rides on the video clips (volume>0), like the template draft
audio_tracks = [t for t in doc['tracks'] if t.get('type') == 'audio']
vtracks = [t for t in doc['tracks'] if t.get('type') == 'video']
audio_on = [s for t in audio_tracks for s in t.get('segments', []) if s.get('volume', 0) > 0]
audio_on += [s for t in vtracks for s in t.get('segments', []) if s.get('volume', 0) > 0]
if not audio_on:
    fails.append("no audio-on segment anywhere -> silent (video clips must carry audio or an audio track must exist)")
# audio-fade SILENCING gate: a material audio_fade whose fade covers most of a short clip mutes it
vids_by_id = {m['id']: m for m in doc['materials'].get('videos', [])}
for t in vtracks:
    for s in t.get('segments', []):
        if s.get('volume', 0) <= 0:
            continue
        m = vids_by_id.get(s.get('material_id')) or {}
        af = m.get('audio_fade')
        dur = s.get('target_timerange', {}).get('duration', 0)
        if isinstance(af, dict) and dur:
            fi = af.get('fade_in_duration', 0); fo = af.get('fade_out_duration', 0)
            if fi > dur * 0.5 or fo > dur * 0.5:
                fails.append("clip audio_fade (in %.1fs/out %.1fs) >= half of %.1fs clip -> SILENCES audio. "
                             "SUGGESTION: set material audio_fade=None" % (fi/1e6, fo/1e6, dur/1e6))

# 9) EFFECT NATURALNESS gate (catch jarring effects; emit suggestions)
for t in doc['tracks']:
    if t.get('type') != 'effect':
        continue
    for s in t.get('segments', []):
        m = ve_by_id.get(s.get('material_id'), {})
        dur = s.get('target_timerange', {}).get('duration', 0) / 1e6
        has_kf = bool(s.get('common_keyframes'))
        nm = m.get('name', '?')
        if not has_kf:
            warns.append("effect '%s' has NO intensity keyframes -> hard pop. SUGGESTION: ramp its param 0->peak->0" % nm)
        if dur < 0.8:
            warns.append("effect '%s' is %.2fs -> too short to slide in/out. SUGGESTION: >=1.0s" % (nm, dur))
        # shake intensity check
        if 'Shake' in nm:
            ap = m.get('adjust_params') or []
            vals = [p.get('value') for p in ap if isinstance(p, dict) and isinstance(p.get('value'), (int, float))]
            if vals and max(vals) > 0.5:
                warns.append("shake '%s' peak intensity %.2f is jarring. SUGGESTION: lower to ~0.4 and keyframe the ramp" % (nm, max(vals)))

# 8) duration sanity
if abs(doc.get('duration', 0)/1e6 - plan['duration']) > 1.0:
    warns.append("draft duration != plan duration")

print("=== QA Layer A (hard gates) ===")
print("segments checked: %d | materials: %d | transitions: %d | effect-segs: %d"
      % (n_seg, len(mat_ids), n_trans, n_fx_seg))
print("song_index: %s | duration: %.1fs" % (plan.get('song_index'), doc.get('duration',0)/1e6))
for w in warns: print("  WARN:", w)
if fails:
    print("\nRESULT: FAIL (%d)" % len(fails))
    for f in fails: print("  FAIL:", f)
    sys.exit(1)
print("\nRESULT: PASS - structural gates OK; safe to open in CapCut.")
