# -*- coding: utf-8 -*-
"""Orchestrator: run the full auto-editor pipeline on a gig folder.
Usage: python run_gig.py <gig_folder> <draft_name> [--dur 90] [--prompt "..."]
Outputs: preview MP4 + CapCut draft."""
import sys, os, json, subprocess, argparse, glob

PROJ = os.path.dirname(os.path.abspath(__file__))
from paths import FFMPEG as FF, FFPROBE as FP, CAPCUT_DRAFT_ROOT as DRAFT_ROOT
PY = sys.executable

parser = argparse.ArgumentParser()
parser.add_argument('gig', help='Path to gig folder')
parser.add_argument('name', help='Draft project name')
parser.add_argument('--dur', type=float, default=90, help='Target duration in seconds')
parser.add_argument('--prompt', default='auto', help='Edit prompt or "auto"')
args = parser.parse_args()

WORK = os.path.join(PROJ, '_work', args.name)
OUT = os.path.join(args.gig, '_auto_output')
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

def run_step(label, cmd):
    print("\n" + "="*60)
    print("[%s]" % label)
    print("="*60, flush=True)
    r = subprocess.run([PY] + cmd, cwd=PROJ)
    if r.returncode != 0:
        print("FAILED: %s (rc=%d)" % (label, r.returncode))
        sys.exit(1)

# ---- find full-show recordings (>400MB video files) ----
srcs = sorted([f for f in glob.glob(os.path.join(args.gig, '*'))
               if os.path.isfile(f)
               and os.path.splitext(f)[1].lower() in ('.mp4', '.mov', '.m4v')
               and os.path.getsize(f) > 400_000_000],
              key=os.path.getsize, reverse=True)
if not srcs:
    print("ERROR: no full-show recordings (>400MB) found in", args.gig)
    sys.exit(1)
print("Sources:")
for s in srcs:
    print("  %s (%.1f GB)" % (os.path.basename(s), os.path.getsize(s)/1e9))

# ---- Step 1: Multicam sync + audio quality ----
if not os.path.exists(os.path.join(WORK, 'analysis.json')):
    run_step("1/6 ANALYZE", ['analyze.py', FF, FP, WORK] + srcs)
else:
    print("\n[1/6 ANALYZE] cached")

analysis = json.load(open(os.path.join(WORK, 'analysis.json')))
master = analysis['_meta']['master']
master_src = analysis[master]['path']

# ---- Step 2: Song detection ----
if not os.path.exists(os.path.join(WORK, 'songs.json')):
    run_step("2/6 SONG DETECT", ['song_detect.py', FF, WORK, master_src])
else:
    print("\n[2/6 SONG DETECT] cached")

# ---- Step 3: Section segmentation ----
if not os.path.exists(os.path.join(WORK, 'sections.json')):
    run_step("3/6 SECTION SEGMENT", ['section_segment.py', WORK])
else:
    print("\n[3/6 SECTION SEGMENT] cached")

# ---- Step 4: Quality scoring ----
if not os.path.exists(os.path.join(WORK, 'quality.json')):
    run_step("4/6 QUALITY SCORE", ['quality_score.py', WORK])
else:
    print("\n[4/6 QUALITY SCORE] cached")

# ---- Step 5: Section assembly (produces edit_plan.json) ----
# This is the "judgment" step — in full pipeline a sub-agent handles this.
# For now: auto-assemble from quality-ranked sections.
if not os.path.exists(os.path.join(WORK, 'edit_plan.json')):
    print("\n" + "="*60)
    print("[5/6 SECTION ASSEMBLY]")
    print("="*60, flush=True)

    quality = json.load(open(os.path.join(WORK, 'quality.json')))
    sections = json.load(open(os.path.join(WORK, 'sections.json')))
    meta = analysis['_meta']
    offset_map = meta.get('sync_offset_refined', meta['sync_offset'])

    # collect all sections with absolute times, ranked by quality
    all_secs = []
    for song in quality:
        for sec in song['sections']:
            all_secs.append({**sec, 'song_title': song.get('title'), 'song_index': song['song_index']})

    # sort by score, pick top sections up to target duration
    all_secs.sort(key=lambda s: s['score'], reverse=True)
    target = args.dur
    picked = []
    total = 0
    for sec in all_secs:
        if total + sec['duration'] > target * 1.2:
            continue
        picked.append(sec)
        total += sec['duration']
        if total >= target:
            break

    # sort picked by absolute time for coherent playback
    picked.sort(key=lambda s: s['start_absolute'])

    # assign angles: alternate master/second, with intentional reasoning
    ranking = meta['ranking']
    sources = {name: analysis[name] for name in ranking}

    # detect orientation per source
    def get_dims(path):
        o = subprocess.run([FP, '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=width,height', '-of', 'csv=p=0', path],
                           stdout=subprocess.PIPE).stdout.decode().strip()
        w, h = o.split(',')[:2]
        return int(w), int(h)

    src_info = {}
    for name in ranking:
        p = sources[name]['path']
        w, h = get_dims(p)
        orient = 'portrait' if h > w else 'landscape'
        src_info[name] = {'path': p, 'w': w, 'h': h, 'orientation': orient}
        print("  %s: %dx%d (%s)" % (name, w, h, orient))

    # build shots with POV reasoning
    shots = []
    tl_cursor = 0.0
    for i, sec in enumerate(picked):
        # POV logic: chorus/high-energy -> master (usually wider/better audio),
        # verse/bridge -> alternate to second for variety
        if sec['label'] in ('chorus',) or sec['energy_ratio'] > 1.2:
            angle_name = ranking[0]
            reason = "high energy %s -> best audio angle" % sec['label']
        elif sec['label'] in ('bridge', 'break', 'outro'):
            angle_name = ranking[1] if len(ranking) > 1 else ranking[0]
            reason = "%s -> alternate angle for variety" % sec['label']
        elif i % 2 == 0:
            angle_name = ranking[0]
            reason = "verse -> master angle"
        else:
            angle_name = ranking[1] if len(ranking) > 1 else ranking[0]
            reason = "verse -> alternate angle for variety"

        src_path = src_info[angle_name]['path']
        offset = offset_map.get(angle_name, 0)
        source_start = sec['start_absolute'] + offset

        shots.append({
            'start_timeline': round(tl_cursor, 3),
            'end_timeline': round(tl_cursor + sec['duration'], 3),
            'source_path': src_path,
            'source_start': round(max(source_start, 0), 3),
            'source_name': angle_name,
            'angle_type': src_info[angle_name]['orientation'],
            'section_label': sec['label'],
            'song_index': sec['song_index'],
            'song_title': sec.get('song_title'),
            'energy_ratio': sec['energy_ratio'],
            'quality_score': sec['score'],
            'reason': reason
        })
        tl_cursor += sec['duration']

    # audio bed: continuous from master starting at the first shot's absolute time
    first_abs = picked[0]['start_absolute']

    edit_plan = {
        'name': args.name,
        'prompt': args.prompt,
        'target_duration': target,
        'actual_duration': round(tl_cursor, 2),
        'audio_bed_start': round(first_abs, 3),
        'audio_bed_source': master_src,
        'shots': shots,
        'source_info': src_info
    }
    json.dump(edit_plan, open(os.path.join(WORK, 'edit_plan.json'), 'w'), indent=2)
    print("WROTE edit_plan.json: %d shots, %.1fs" % (len(shots), tl_cursor))
else:
    print("\n[5/6 SECTION ASSEMBLY] cached")

# ---- Step 6: Generate preview MP4 ----
run_step("6/6 PREVIEW RENDER", ['gen_preview.py', FF, FP, WORK, OUT])

print("\n" + "="*60)
print("DONE!")
print("Preview: %s" % os.path.join(OUT, args.name + '_preview.mp4'))
print("Edit plan: %s" % os.path.join(WORK, 'edit_plan.json'))
print("="*60)
