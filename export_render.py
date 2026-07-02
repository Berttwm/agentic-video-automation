# -*- coding: utf-8 -*-
"""Export the edit (video + master audio per shot, in sync) to an MP4 and self-QA the audio.
Renders what the CapCut draft should produce, so we can verify audio is present without CapCut.
Usage: python export_render.py <ffmpeg> <ffprobe> <workdir> <out_mp4>"""
import sys, os, json, subprocess, tempfile, re

FF, FP, WORK, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
plan = json.load(open(os.path.join(WORK, 'edit_plan.json'), encoding='utf-8'))
shots = plan['shots']
master = plan['angles']['master']['path']
drummer = plan['angles']['drummer']['path']
W, H = 480, 854
tmp = tempfile.mkdtemp(prefix='exp_')
VF = ("scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2,setsar=1"
      % (W, H, W, H))
segs = []
for i, sh in enumerate(shots):
    src = drummer if sh['angle'] == 'drummer' else master
    ss = sh['angle_start_abs'] if sh['angle'] == 'drummer' else sh['master_start_abs']
    seg = os.path.join(tmp, 'e%02d.mp4' % i)
    # keep AUDIO this time (no -an); re-encode a/v so concat is clean
    subprocess.run([FF, '-v', 'error', '-ss', '%.3f' % ss, '-i', src, '-t', '%.3f' % sh['dur'],
                    '-vf', VF, '-r', '30', '-c:v', 'libx264', '-preset', 'ultrafast',
                    '-c:a', 'aac', '-b:a', '160k', '-ac', '2', '-ar', '48000', '-y', seg], check=True)
    segs.append(seg)
    print('  seg %d/%d %s %.1fs (audio kept)' % (i+1, len(shots), sh['angle'], sh['dur']), flush=True)
cat = os.path.join(tmp, 'c.txt')
open(cat, 'w').write(''.join("file '%s'\n" % s.replace('\\', '/') for s in segs))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
# bake APPROXIMATE effects into the proxy so it represents the final (blur/shake -> blur pulse; leak -> flash)
vfl = []
for e in plan.get('effects', []):
    s = e['tl_start']; en = round(s + e['dur'], 2); nm = e['name']
    if 'Leak' in nm or e.get('role') == 'flash':
        vfl.append("eq=brightness=0.20:saturation=1.35:enable='between(t\\,%.2f\\,%.2f)'" % (s, en))
    else:
        vfl.append("gblur=sigma=11:enable='between(t\\,%.2f\\,%.2f)'" % (s, en))
vf = ",".join(vfl) if vfl else "null"
subprocess.run([FF, '-v', 'error', '-f', 'concat', '-safe', '0', '-i', cat, '-vf', vf,
                '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'copy', '-y', OUT], check=True)
print("baked %d approximate effects into the proxy" % len(vfl))

# ---- self-QA the exported audio ----
probe = subprocess.run([FP, '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                        'stream=codec_name,channels,duration', '-of', 'default=nw=1', OUT],
                       stdout=subprocess.PIPE).stdout.decode()
vol = subprocess.run([FF, '-i', OUT, '-af', 'volumedetect', '-f', 'null', '-'],
                     stderr=subprocess.PIPE).stderr.decode()
mean = re.search(r'mean_volume:\s*([-\d.]+)', vol)
mx = re.search(r'max_volume:\s*([-\d.]+)', vol)
print("\n=== EXPORT AUDIO QA ===")
print("audio stream:", probe.replace('\n', ' ').strip() or "NONE")
print("mean_volume:", mean.group(1) if mean else "?", "dB | max_volume:", mx.group(1) if mx else "?", "dB")
mv = float(mean.group(1)) if mean else -99
verdict = "AUDIO PRESENT & AUDIBLE" if mv > -40 else ("VERY QUIET/near-silent" if mv > -70 else "SILENT")
print("verdict:", verdict)
print("export:", OUT)
