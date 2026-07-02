# -*- coding: utf-8 -*-
"""Generate an editable CapCut draft from edit_plan.json (template-cloned from the template draft ONLY).
Architecture (research-faithful): base track = continuous MASTER audio bed (per contiguous run);
overlay track = the visible cut sequence (master + optional drummer-cam), muted, Blur transitions at
stitch/angle-switch points; effect track = subtle real effects (Shake 2 / Blur) on peak sections.
NO draft 0218. NO synthesized grade (color is applied later / in CapCut).
Usage: python build_draft.py <ffprobe> <workdir> <draft_root> <draft_name>"""
import sys, os, json, uuid, time, shutil, subprocess

FFPROBE, WORK, DRAFT_ROOT, NAME = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
FFMPEG = FFPROBE.replace('ffprobe.exe', 'ffmpeg.exe')
US = 1_000_000
def us(sec): return int(round(sec * US))
def fwd(p): return p.replace('\\', '/')
def newid(): return str(uuid.uuid4()).upper()
def clone(o): return json.loads(json.dumps(o))

plan = json.load(open(os.path.join(WORK, 'edit_plan.json'), encoding='utf-8'))
shots = plan['shots']
DUR = plan['duration']
master_path = plan['angles']['master']['path']
drummer_path = plan['angles']['drummer']['path']

import json as _json
_cfgp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
TPL = ((_json.load(open(_cfgp)).get('template_draft') if os.path.exists(_cfgp) else None) or 'YOUR_TEMPLATE_DRAFT')
tpov = json.load(open(os.path.join(DRAFT_ROOT, TPL, 'draft_content.json'), encoding='utf-8'))
assert '0218' not in json.dumps(plan), "edit_plan must not reference 0218"

# ---- harvest CapCut-cached effect/transition OBJECTS from ALL drafts (effect objects only, not
#      any draft's footage/structure) so the inferred palette (Shake/Leak 2/etc.) is placeable ----
trans_lib, fx_lib = {}, {}
for proj in os.listdir(DRAFT_ROOT):
    dcp = os.path.join(DRAFT_ROOT, proj, 'draft_content.json')
    if not os.path.isfile(dcp):
        continue
    try:
        dd = json.load(open(dcp, encoding='utf-8'))
    except Exception:
        continue
    for m in dd['materials'].get('transitions', []) or []:
        if m.get('name'): trans_lib.setdefault(m['name'], m)
    for m in dd['materials'].get('video_effects', []) or []:
        if m.get('name'): fx_lib.setdefault(m['name'], m)
vid_tpl = tpov['materials']['videos'][0]
vseg_tpl = [t for t in tpov['tracks'] if t['type'] == 'video'][0]['segments'][0]
efx_tracks = [t for t in tpov['tracks'] if t['type'] == 'effect']
efx_track_tpl = efx_tracks[0] if efx_tracks else None
eseg_tpl = efx_track_tpl['segments'][0] if (efx_track_tpl and efx_track_tpl['segments']) else None
aud_tpl = (tpov['materials'].get('audios') or [None])[0]
_atrk = [t for t in tpov['tracks'] if t['type'] == 'audio']
aseg_tpl = _atrk[0]['segments'][0] if (_atrk and _atrk[0]['segments']) else None
print("the template draft transitions:", sorted(trans_lib))
print("the template draft effects:", sorted(fx_lib))

def probe(path):
    o = subprocess.run([FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                        'stream=width,height,duration', '-show_entries', 'format=duration',
                        '-of', 'json', path], stdout=subprocess.PIPE).stdout.decode()
    j = json.loads(o); st = j['streams'][0]
    dur = float(st.get('duration') or j['format']['duration'])
    a = subprocess.run([FFPROBE, '-v', 'error', '-select_streams', 'a', '-show_entries',
                        'stream=index', '-of', 'csv=p=0', path], stdout=subprocess.PIPE).stdout.decode().strip()
    return int(st['width']), int(st['height']), us(dur), bool(a)
mw, mh, mdur, mha = probe(master_path)
dw, dh, ddur, dha = probe(drummer_path)

# ---- resolve transition/effect names to what actually exists (fallbacks) ----
def pick_trans(name):
    if name in trans_lib: return name
    for alt in ('Blur', 'Floodlight', 'Twinkle Zoom'):
        if alt in trans_lib: return alt
    return next(iter(trans_lib)) if trans_lib else None
def pick_fx(name):
    if name in fx_lib: return name
    for alt in ('Shake 2', 'Blur'):
        if alt in fx_lib: return alt
    return next(iter(fx_lib)) if fx_lib else None

# =================== MATERIALS ===================
materials = {k: [] for k in tpov['materials'].keys()}
def add(cat, obj):
    materials.setdefault(cat, []).append(obj); return obj['id']

def mk_video(src, w, h, dur_us, ha, name):
    m = clone(vid_tpl); m['id'] = newid(); m['local_material_id'] = str(uuid.uuid4())
    m['path'] = fwd(src); m['media_path'] = ''; m['material_name'] = name
    m['width'], m['height'], m['duration'], m['has_audio'] = w, h, dur_us, ha
    m['material_id'] = ''; m['origin_material_id'] = ''
    m['audio_fade'] = None   # CRITICAL: drop pov1's inherited 6.7s audio fade-in that silenced short clips
    m['crop'] = {"lower_left_x":0.0,"lower_left_y":1.0,"lower_right_x":1.0,"lower_right_y":1.0,
                 "upper_left_x":0.0,"upper_left_y":0.0,"upper_right_x":1.0,"upper_right_y":0.0}
    m['crop_ratio'] = 'free'; m['crop_scale'] = 1.0
    return add('videos', m)
master_mid = mk_video(master_path, mw, mh, mdur, mha, os.path.basename(master_path))
drummer_mid = mk_video(drummer_path, dw, dh, ddur, dha, os.path.basename(drummer_path))

def mk_speed(v=1.0): return add('speeds', {"curve_speed":None,"id":newid(),"mode":0,"speed":v,"type":"speed"})
def mk_canvas(): return add('canvases', {"album_image":"","blur":0.0,"color":"#ffffffff","id":newid(),
    "image":"","image_id":"","image_name":"","source_platform":0,"team_id":"","type":"canvas_color"})
def mk_anim(): return add('material_animations', {"animations":[],"id":newid(),
    "multi_language_current":"none","type":"sticker_animation"})
def mk_scm(): return add('sound_channel_mappings', {"audio_channel_mapping":0,"id":newid(),
    "is_config_open":False,"type":"none"})
def mk_vsep(): return add('vocal_separations', {"choice":0,"id":newid(),"production_path":"",
    "removed_sounds":[],"time_range":None,"type":"vocal_separation"})
def mk_trans(name):
    t = clone(trans_lib[name]); t['id'] = newid(); return add('transitions', t)
def mk_fx(name, intensity=0.6):
    e = clone(fx_lib[name]); e['id'] = newid()
    e['apply_target_type'] = 2   # 2 = GLOBAL: render over the whole composite (the visible cut track),
                                 # not target the hidden base track (0). This is why effects didn't show.
    # gentle the intensity so effects punctuate rather than disrupt the shot
    ap = e.get('adjust_params')
    if isinstance(ap, list):
        for p in ap:
            if isinstance(p, dict) and isinstance(p.get('value'), (int, float)):
                p['value'] = round(float(p['value']) * intensity, 4)
    if isinstance(e.get('value'), (int, float)):
        e['value'] = round(float(e['value']) * intensity, 4)
    return add('video_effects', e)

DEF_CLIP = {"alpha":1.0,"flip":{"horizontal":False,"vertical":False},"rotation":0.0,
            "scale":{"x":1.0,"y":1.0},"transform":{"x":0.0,"y":0.0}}
def mk_vseg(mid, tl_s, dur_s, src_s, volume, rndr, trk):
    s = clone(vseg_tpl); s['id'] = newid(); s['material_id'] = mid
    s['target_timerange'] = {"start": us(tl_s), "duration": us(dur_s)}
    s['source_timerange'] = {"start": us(max(src_s, 0)), "duration": us(dur_s)}
    s['speed'] = 1.0; s['volume'] = volume; s['last_nonzero_volume'] = 1.0
    s['clip'] = clone(DEF_CLIP); s['common_keyframes'] = []; s['keyframe_refs'] = []
    s['uniform_scale'] = {"on": True, "value": 1.0}; s['visible'] = True; s['group_id'] = ''
    s['render_index'] = rndr; s['track_render_index'] = trk
    s['extra_material_refs'] = [mk_speed(), mk_canvas(), mk_anim(), mk_scm(), mk_vsep()]
    return s

def mk_audio_mat(path, dur_us, name):
    m = clone(aud_tpl); m['id'] = newid(); m['local_material_id'] = str(uuid.uuid4())
    m['type'] = 'video_original_sound'; m['path'] = fwd(path); m['name'] = name
    m['duration'] = dur_us; m['music_id'] = ''; m['resource_id'] = ''; m['music_source'] = ''
    return add('audios', m)

def mk_aseg(mid, tl_s, dur_s, src_s, vol):
    s = clone(aseg_tpl); s['id'] = newid(); s['material_id'] = mid
    s['target_timerange'] = {"start": us(tl_s), "duration": us(dur_s)}
    s['source_timerange'] = {"start": us(max(src_s, 0)), "duration": us(dur_s)}
    s['volume'] = vol; s['last_nonzero_volume'] = 1.0; s['speed'] = 1.0
    s['common_keyframes'] = []; s['keyframe_refs'] = []; s['group_id'] = ''
    s['extra_material_refs'] = [mk_speed(), mk_scm(), mk_vsep()]
    return s

def mk_eseg(fx_name, tl_s, dur_s, rndr, intensity=0.6):
    base = clone(eseg_tpl) if eseg_tpl else {}
    base['id'] = newid(); base['material_id'] = mk_fx(fx_name, intensity)
    fx_mat = materials['video_effects'][-1]
    base['target_timerange'] = {"start": us(tl_s), "duration": us(dur_s)}
    base['source_timerange'] = None; base['clip'] = None
    base['extra_material_refs'] = []; base['render_index'] = rndr; base['track_render_index'] = 2
    # INTENSITY KEYFRAMES: ramp the effect's primary param 0 -> peak -> 0 so it slides in and out
    # (schema cloned from the template draft: common_keyframes w/ property_type = the adjust-param name).
    ap = fx_mat.get('adjust_params') or []
    prim = next((p for p in ap if isinstance(p, dict) and p.get('name')), None)
    if prim:
        peak = float(prim.get('value', intensity) or intensity)
        D = us(dur_s)
        pts = [(0, 0.0), (int(D*0.35), peak), (int(D*0.65), peak), (D, 0.0)]
        def kf(t, v):
            return {"curveType": "Line", "graphID": "", "id": newid(),
                    "left_control": {"x": 0.0, "y": 0.0}, "right_control": {"x": 0.0, "y": 0.0},
                    "time_offset": t, "values": [v]}
        base['common_keyframes'] = [{"id": newid(), "material_id": "", "property_type": prim['name'],
                                     "keyframe_list": [kf(t, v) for t, v in pts]}]
    else:
        base['common_keyframes'] = []
    base['keyframe_refs'] = []
    return base

# =================== TRACKS ===================
# runs = contiguous shots split at stitch points (each run is continuous in the master source)
runs = []
for sh in shots:
    if not runs or sh.get('is_join') or sh.get('is_stitch'):
        runs.append([sh])
    else:
        runs[-1].append(sh)

# SINGLE VIDEO track: the visible cut sequence; clips carry their OWN (master) audio at volume 1.0,
# exactly like the template draft -> audio plays reliably and it's ONE clean track. Transitions at joins.
ov_segs = []
for i, sh in enumerate(shots):
    if sh['angle'] == 'drummer':
        mid, src_s, vol = drummer_mid, sh['angle_start_abs'], 0.0   # (drummer switch off by default now)
    else:
        mid, src_s, vol = master_mid, sh['master_start_abs'], 1.0
    ov_segs.append(mk_vseg(mid, sh['tl_start'], sh['dur'], src_s, vol, rndr=i, trk=0))
for i, sh in enumerate(shots):
    if sh.get('transition_in') and i >= 1:
        tn = pick_trans(sh['transition_in'])
        if tn: ov_segs[i-1]['extra_material_refs'].append(mk_trans(tn))

# EFFECT track: effects, global (apply_target_type=2), with intensity keyframes; shakes kept gentle.
# The PREVIEW MP4 bakes the REAL effects_lab (ffmpeg) renders; this CapCut effect track is the EDITABLE
# handoff, so each effects_lab effect maps to its closest available CapCut analog (from TRIGGER_RULES'
# capcut_analog, with a Blur fallback). effects_lab names that have NO CapCut object (rgb_split /
# radial_zoom / speed_ramp) fall back to the nearest CapCut look so the draft still shows *something*
# tasteful at the moment; the true effect lives in the exported preview. New plan schema: each effect is
# {effect, intensity(level str), envelope_duration, ...}; older {name, dur} is still accepted.
try:
    import effects_lab as _EL
    _TRIG = _EL.TRIGGER_RULES
except Exception:
    _TRIG = {}
# effects_lab level -> a gentle CapCut intensity scalar (draft-side softening; preview uses real ramps)
_LEVEL_SCALAR = {'subtle': 0.35, 'medium': 0.55, 'strong': 0.8}
# effects_lab name -> preferred CapCut analog names (first that exists in fx_lib wins; else Blur)
_CAPCUT_ANALOG = {
    'shake':       ['Shake 2', 'Shake'],
    'light_leak':  ['Leak 2'],
    'radial_zoom': ['Blur', 'Twinkle Zoom'],
    'rgb_split':   ['Blur'],            # no CapCut chroma-split object locally -> Blur placeholder
    'speed_ramp':  ['Blur'],            # a speed curve, not an fx object -> Blur placeholder in draft
    'whip':        ['Blur', 'Mix'],     # normally a transition, not a standalone fx
}

def _resolve_capcut(effname):
    for cand in _CAPCUT_ANALOG.get(effname, []):
        if cand in fx_lib:
            return cand
    return pick_fx(effname)              # generic fallback (Shake 2 / Blur / first available)

efx_segs = []
for i, f in enumerate(plan.get('effects', [])):
    libname = f.get('effect') or f.get('name')          # new schema 'effect' | legacy 'name'
    fn = _resolve_capcut(libname)
    if not fn:
        continue
    # clip length: envelope_duration (+ small tail so it can slide out); legacy plans carry 'dur'
    envd = f.get('envelope_duration')
    if envd is None:
        envd = (_TRIG.get(libname, {}) or {}).get('envelope_duration')
    dur = float(f.get('dur') or ((envd or 0.7) + 0.5))
    dur = max(dur, 1.0)                                   # QA gate: >=1.0s so it can ramp in CapCut
    # intensity: map effects_lab level -> scalar; keep shakes extra gentle (QA naturalness gate)
    lvl = f.get('intensity')
    inten = _LEVEL_SCALAR.get(lvl, 0.55) if isinstance(lvl, str) else 0.55
    if 'Shake' in fn:
        inten = min(inten, 0.4)
    efx_segs.append(mk_eseg(fn, f['tl_start'], dur, 11000 + i, inten))

def vtrack(segs, name):
    return {"attribute":0,"flag":0,"id":newid(),"is_default_name":True,"name":name,"segments":segs,"type":"video"}
tracks = [vtrack(ov_segs, "video")]
if efx_segs and efx_track_tpl:
    et = clone(efx_track_tpl); et['id'] = newid(); et['segments'] = efx_segs; tracks.append(et)

# =================== ASSEMBLE draft_content.json ===================
doc = clone(tpov)
doc['materials'] = materials
doc['tracks'] = tracks
doc['duration'] = us(DUR)
doc['id'] = newid(); doc['name'] = NAME
doc['canvas_config'] = {"background": None, "height": 1920, "ratio": "9:16", "width": 1080}
doc['fps'] = 30.0
doc['create_time'] = int(time.time()); doc['update_time'] = int(time.time())
folder = os.path.join(DRAFT_ROOT, NAME)
doc['path'] = fwd(folder)
for k in ('cover', 'retouch_cover', 'static_cover_image_path'):
    if k in doc and not isinstance(doc.get(k), (dict, list)): doc[k] = ''

# =================== draft_meta_info.json (from the template draft) ===================
mi = json.load(open(os.path.join(DRAFT_ROOT, TPL, 'draft_meta_info.json'), encoding='utf-8'))
now_s, now_us = int(time.time()), int(time.time() * US)
def matentry(src, w, h, dur_us):
    return {"create_time":now_s,"duration":dur_us,"extra_info":os.path.basename(src),
            "file_Path":fwd(src),"height":h,"id":str(uuid.uuid4()),"import_time":now_s,
            "import_time_ms":now_us,"item_source":1,"md5":"","metetype":"video",
            "roughcut_time_range":{"duration":dur_us,"start":0},
            "sub_time_range":{"duration":-1,"start":-1},"type":0,"width":w}
mi['draft_fold_path'] = fwd(folder)
mi['draft_id'] = newid(); mi['draft_name'] = NAME
mi['draft_materials'] = [{"type":0,"value":[matentry(master_path, mw, mh, mdur),
                                            matentry(drummer_path, dw, dh, ddur)]}]
mi['tm_draft_create'] = now_us; mi['tm_draft_modified'] = now_us; mi['tm_duration'] = us(DUR)
mi['draft_cover'] = 'draft_cover.jpg'

# =================== WRITE FOLDER ===================
os.makedirs(folder, exist_ok=True)
for aux in ('draft_virtual_store.json', 'draft_settings'):   # NOT draft_cover.jpg (that's the template's)
    sp = os.path.join(DRAFT_ROOT, TPL, aux)
    if os.path.exists(sp):
        try: shutil.copy2(sp, os.path.join(folder, aux))
        except Exception as ex: print("aux copy skip", aux, ex)
# COVER from THIS gig's footage: grab a frame from the peak shot of the master source
cover_t = max(shots, key=lambda s: s.get('energy_ratio', 0))['master_start_abs'] + 1.0
cover_path = os.path.join(folder, 'draft_cover.jpg')
try:
    subprocess.run([FFMPEG, '-v', 'error', '-ss', '%.2f' % cover_t, '-i', master_path,
                    '-frames:v', '1', '-vf', 'scale=1080:-1', '-y', cover_path], check=True)
    print("cover extracted from gig @ %.1fs -> draft_cover.jpg" % cover_t)
except Exception as ex:
    print("cover extract failed:", ex)
json.dump(doc, open(os.path.join(folder, 'draft_content.json'), 'w', encoding='utf-8'), ensure_ascii=False)
json.dump(mi, open(os.path.join(folder, 'draft_meta_info.json'), 'w', encoding='utf-8'), ensure_ascii=False)

print("\nWROTE DRAFT:", folder)
print("  duration=%.1fs | video shots=%d | effects=%d (single video track, audio on clips)"
      % (DUR, len(ov_segs), len(efx_segs)))
trans_ids = {m['id'] for m in materials['transitions']}
print("  transitions placed:", sum(1 for s in ov_segs
      for r in s['extra_material_refs'] if r in trans_ids))
