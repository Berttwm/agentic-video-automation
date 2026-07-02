# -*- coding: utf-8 -*-
"""Render a low-res PROXY preview from edit_plan.json (sync/structure check + QA Layer-B input).
Overlay cut sequence (per-shot angle video) + continuous master audio bed.
Usage: python render_preview.py <ffmpeg> <workdir> <out_mp4>"""
import sys, os, json, subprocess, tempfile

FF, WORK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
plan = json.load(open(os.path.join(WORK, 'edit_plan.json'), encoding='utf-8'))
shots = plan['shots']
master = plan['angles']['master']['path']
drummer = plan['angles']['drummer']['path']
W, H = 480, 854
tmp = tempfile.mkdtemp(prefix='prev_')
VF = ("scale=%d:%d:force_original_aspect_ratio=decrease,"
      "pad=%d:%d:(ow-iw)/2:(oh-ih)/2,setsar=1" % (W, H, W, H))

segfiles = []
for i, sh in enumerate(shots):
    src = drummer if sh['angle'] == 'drummer' else master
    ss = sh['angle_start_abs'] if sh['angle'] == 'drummer' else sh['master_start_abs']
    seg = os.path.join(tmp, 'seg_%02d.mp4' % i)
    cmd = [FF, '-v', 'error', '-ss', '%.3f' % ss, '-i', src, '-t', '%.3f' % sh['dur'],
           '-vf', VF, '-an', '-r', '30', '-c:v', 'libx264', '-preset', 'ultrafast', '-y', seg]
    subprocess.run(cmd, check=True)
    segfiles.append(seg)
    print('  seg %d/%d %s %.1fs' % (i + 1, len(shots), sh['angle'], sh['dur']), flush=True)

concat_txt = os.path.join(tmp, 'concat.txt')
with open(concat_txt, 'w') as f:
    for s in segfiles:
        f.write("file '%s'\n" % s.replace('\\', '/'))
video = os.path.join(tmp, 'video.mp4')
subprocess.run([FF, '-v', 'error', '-f', 'concat', '-safe', '0', '-i', concat_txt,
                '-c', 'copy', '-y', video], check=True)

# master audio bed from the first shot's master position, full duration
audio = os.path.join(tmp, 'bed.m4a')
subprocess.run([FF, '-v', 'error', '-ss', '%.3f' % shots[0]['master_start_abs'], '-i', master,
                '-t', '%.3f' % plan['duration'], '-vn', '-c:a', 'aac', '-b:a', '160k', '-y', audio], check=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
subprocess.run([FF, '-v', 'error', '-i', video, '-i', audio, '-c:v', 'copy', '-c:a', 'copy',
                '-shortest', '-y', OUT], check=True)
print('WROTE preview:', OUT)
