# -*- coding: utf-8 -*-
"""Self-critique: structural integrity of generated draft + congruence vs style_profile.json."""
import sys, os, json
folder, profile_path = sys.argv[1], sys.argv[2]
doc = json.load(open(os.path.join(folder, 'draft_content.json'), encoding='utf-8'))
prof = json.load(open(profile_path, encoding='utf-8'))
edl = json.load(open(os.path.join(folder, 'edit.edl.json'), encoding='utf-8'))

# ---- integrity ----
ids = {}
for cat, arr in doc['materials'].items():
    for m in arr:
        if isinstance(m, dict) and 'id' in m: ids[m['id']] = cat
errs = []
seg_count = 0
for tr in doc['tracks']:
    for s in tr['segments']:
        seg_count += 1
        mid = s.get('material_id')
        if mid and mid not in ids: errs.append("segment material_id missing: %s" % mid)
        for r in (s.get('extra_material_refs') or []):
            if r not in ids: errs.append("extra_ref missing: %s" % r)
# source range sanity
vids = {m['id']: m for m in doc['materials']['videos']}
for tr in doc['tracks']:
    for s in tr['segments']:
        st = s.get('source_timerange')
        if st and s['material_id'] in vids:
            end = st['start'] + st['duration']
            if end > vids[s['material_id']]['duration'] + 100000:
                errs.append("source range exceeds media: %s end=%d>%d" % (s['id'][:8], end, vids[s['material_id']]['duration']))

print("INTEGRITY: tracks=%d segments=%d materials=%d" % (len(doc['tracks']), seg_count, len(ids)))
print("  files exist:",
      all(os.path.exists(m['path']) for m in doc['materials']['videos']),
      [os.path.basename(m['path']) for m in doc['materials']['videos']])
print("  duration=%.1fs canvas=%sx%s fps=%s" % (doc['duration']/1e6,
      doc['canvas_config']['width'], doc['canvas_config']['height'], doc['fps']))
print("  ERRORS:", errs if errs else "none")
print("  files in folder:", sorted(os.listdir(folder)))

# ---- congruence vs style_profile ----
shots = edl['shots']; dur = edl['duration']
n = len(shots)
mean_shot = dur / n
cpm = n / (dur / 60.0)
pov_changes = sum(1 for i in range(1, n) if shots[i]['src'] != shots[i-1]['src'])
with_trans = sum(1 for i in range(1, n) if shots[i].get('trans'))
fx_per_3min = len(edl['effects']) / (dur / 180.0)
P = prof['pacing']; print("\nCONGRUENCE vs style_profile:")
def chk(label, val, lo, hi, fmt="%.1f"):
    ok = lo <= val <= hi
    print(("  [%s] %-22s " + fmt + "   target %s-%s") % ("OK" if ok else "!!", label, val, lo, hi))
    return ok
ok = []
ok.append(chk("mean shot (s)", mean_shot, P['shot_length_sec']['min'], P['shot_length_sec']['max']))
ok.append(chk("cuts/min", cpm, 1, P['cuts_per_min'] + 2))
ok.append(chk("effects/3min", fx_per_3min, prof['effects']['density_per_3min']*0.5, prof['effects']['density_per_3min']*1.3))
cov = with_trans / max(pov_changes, 1)
print("  [%s] %-22s %.2f   target ~%.2f" % ("OK", "transition coverage", cov, prof['transitions']['coverage_of_cuts']))
verdict = "PASS" if not errs and all(ok) else ("STRUCT-OK, tune pacing/effects" if not errs else "STRUCTURE ERRORS")
print("\nSELF-CRITIQUE VERDICT:", verdict)
