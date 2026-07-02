# -*- coding: utf-8 -*-
"""Generate a preview MP4 from edit_plan.json — clean cuts over one audio bed, no effects.
Usage: python gen_preview.py <ffmpeg> <ffprobe> <workdir> <outdir>"""
import sys, os, json, subprocess
import numpy as np
import soundfile as sf

FF, FP, WORK, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)

plan = json.load(open(os.path.join(WORK, 'edit_plan.json')))
analysis = json.load(open(os.path.join(WORK, 'analysis.json')))
meta = analysis['_meta']
master = meta['master']
master_src = analysis[master]['path']

shots = plan['shots']
if not shots:
    print("ERROR: no shots in edit_plan.json"); sys.exit(1)

total_dur = shots[-1]['end_timeline']
print("rendering %.1fs preview (%d shots)" % (total_dur, len(shots)), flush=True)

log = os.path.join(WORK, 'gen_preview.log')
open(log, 'w').close()
def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    open(log, 'a', encoding='utf-8').write(' '.join(map(str, cmd))[:300] + "\n" + p.stdout.decode('utf-8','ignore')[-600:] + "\n---\n")
    return p.returncode

def probe_dims(path):
    o = subprocess.run([FP, '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=width,height', '-of', 'csv=p=0', path],
                       stdout=subprocess.PIPE).stdout.decode().strip()
    w, h = o.split(',')[:2]
    return int(w), int(h)

# render each shot
vf_portrait = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps=30,format=yuv420p"
vf_landscape = "scale=1080:-2,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps=30,format=yuv420p"

lst = os.path.join(WORK, 'preview_concat.txt')
lf = open(lst, 'w', encoding='utf-8')

for i, sh in enumerate(shots):
    src = sh['source_path']
    w, h = probe_dims(src)
    vf = vf_portrait if h > w else vf_landscape
    dur = sh['end_timeline'] - sh['start_timeline']
    src_start = sh['source_start']
    f = os.path.join(WORK, "prev_%03d.mp4" % i)
    rc = run([FF, '-y', '-ss', "%.3f" % src_start, '-i', src, '-t', "%.3f" % dur,
              '-vf', vf, '-an', '-r', '30',
              '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22',
              '-pix_fmt', 'yuv420p', '-video_track_timescale', '15360', f])
    angle = sh.get('angle_type', '?')
    reason = sh.get('reason', '')
    print("  shot %d [%s] rc=%d (%.1fs) %s" % (i, angle, rc, dur, reason), flush=True)
    lf.write("file '%s'\n" % f.replace('\\', '/'))
lf.close()

concat = os.path.join(WORK, 'preview_concat.mp4')
print("concat rc=%d" % run([FF, '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', concat]), flush=True)

# audio bed: per-shot segments from master at each shot's ABSOLUTE time (not one continuous stream)
# this ensures audio matches what's happening on screen even when shots jump between songs
bed_lst = os.path.join(WORK, 'bed_concat.txt')
bed_lf = open(bed_lst, 'w', encoding='utf-8')
offset_map = {}
if os.path.exists(os.path.join(WORK, 'analysis.json')):
    import json as _j
    _a = _j.load(open(os.path.join(WORK, 'analysis.json')))
    _m = _a['_meta']
    offset_map = _m.get('sync_offset_refined', _m['sync_offset'])

for i, sh in enumerate(shots):
    bed_seg = os.path.join(WORK, 'bed_%03d.wav' % i)
    dur = sh['end_timeline'] - sh['start_timeline']
    # always take audio from master at the absolute time corresponding to this section
    # if this shot uses a non-master angle, compute the master-time from the source_start
    src_name = sh.get('source_name', '')
    offset = offset_map.get(src_name, 0)
    if src_name == analysis['_meta']['master']:
        master_time = sh['source_start']
    else:
        master_time = sh['source_start'] - offset
    run([FF, '-y', '-ss', "%.3f" % max(master_time, 0), '-i', master_src,
         '-t', "%.3f" % dur, '-vn', '-ar', '48000', '-ac', '2', bed_seg])
    bed_lf.write("file '%s'\n" % bed_seg.replace('\\', '/'))
bed_lf.close()

bed_cat = os.path.join(WORK, 'bed_concat.wav')
run([FF, '-y', '-f', 'concat', '-safe', '0', '-i', bed_lst, '-c', 'copy', bed_cat])

# loudnorm the concatenated bed
bed_norm = os.path.join(WORK, 'bed_norm.wav')
run([FF, '-y', '-i', bed_cat, '-af', 'loudnorm=I=-14:TP=-1.0:LRA=11', '-ar', '48000', '-ac', '2', bed_norm])

out_path = os.path.join(OUT, plan.get('name', 'preview') + '_preview.mp4')
rc = run([FF, '-y', '-i', concat, '-i', bed_norm, '-map', '0:v', '-map', '1:a',
          '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', out_path])
print("FINAL rc=%d -> %s" % (rc, out_path), flush=True)
