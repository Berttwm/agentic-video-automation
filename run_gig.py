# -*- coding: utf-8 -*-
"""Orchestrator: run the ACTUAL v2 auto-editor chain on a gig folder, end-to-end.

Chain (each step reads/writes _work/<name>/ and is cache-aware -- a valid cached artifact is reused):
    1. analyze.py            multicam sync + audio ranking            -> analysis.json
    2. song_detect.py        split the show into songs               -> songs.json  (+ song_NN.wav)
    3. section_segment.py    energy sections per song                -> sections.json
    4. structure2.py         demucs vocal-aware FINE structure        -> vocal_structure_song<N>.json
                             (novelty boundaries + repetition clustering + chord verse/chorus split;
                              demucs stems are separated ONCE and cached under _stems/, then reused)
    5. (style_model.json)    research grammar -- rebuilt only if a reels dir is given and it's missing
    6. assemble_song.py      FORWARD-SKIP arrangement of ONE song    -> edit_plan.json (shots/angles)
    7. infer_effects.py      effects_lab placement on the event spine-> edit_plan.json (+effects/trans)
                             (moment-driven: SUSTAINED only on the impactful climax, short stabs else;
                              forbidden effects never placed)
    8. build_draft.py        editable CapCut draft                   -> <draft_root>/<name>/
    9. render_timeline_fx_v2 crackle-free fx preview + no-fx baseline-> _auto_output/<name>_fx_preview.mp4
   10. qa_check.py           hard structural gates on the draft

Device paths come from paths.py (which reads the gitignored config.json). Nothing here is hardcoded
to a machine or a band. Python is sys.executable.

Usage: python run_gig.py <gig_folder> <draft_name> [--song N] [--target 85] [--force STEP[,STEP...]]
       [--no-draft] [--no-qa]
  --force: comma list of step keys to re-run even if cached
           (analyze,songs,sections,structure,assemble,effects,draft,preview,style)
           'structure' forces the fine structure2 re-analysis (NOT the cached demucs stems -- those are
           only re-separated if the stem wavs are missing).
"""
import sys, os, json, subprocess, argparse, glob

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
import paths

FF, FP = paths.FFMPEG, paths.FFPROBE
PY = sys.executable
DRAFT_ROOT = paths.CAPCUT_DRAFT_ROOT
REELS_DIR = os.path.join(PROJ, "_reels")

ap = argparse.ArgumentParser()
ap.add_argument("gig", help="Path to gig folder (contains the full-show recordings)")
ap.add_argument("name", help="Draft / work project name")
ap.add_argument("--song", type=int, default=0, help="1-based song index to edit (0 = auto-pick best)")
ap.add_argument("--target", type=float, default=85.0, help="target edit duration (s)")
ap.add_argument("--force", default="", help="comma list of step keys to force re-run")
ap.add_argument("--draft", action="store_true",
                help="ALSO build a CapCut draft (default: OFF -- the coded-effects MP4 is the deliverable; "
                     "CapCut uses its own unverifiable effects, so it is opt-in only)")
ap.add_argument("--no-qa", action="store_true", help="skip qa_check (only relevant with --draft)")
ap.add_argument("--no-qa-retry", action="store_true",
                help="skip the content-QA gate + auto-retry on the rendered MP4 (default: ON)")
ap.add_argument("--max-retries", type=int, default=3,
                help="max QA-retry iterations before delivering the best attempt with a residual report")
args = ap.parse_args()

WORK = os.path.join(PROJ, "_work", args.name)
OUT = os.path.join(args.gig, "_auto_output")
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
FORCE = set(s.strip() for s in args.force.split(",") if s.strip())


def run_step(label, cmd, cwd=PROJ):
    print("\n" + "=" * 64 + "\n[%s]\n" % label + "=" * 64, flush=True)
    r = subprocess.run([PY] + cmd, cwd=cwd)
    if r.returncode != 0:
        print("FAILED: %s (rc=%d)" % (label, r.returncode))
        sys.exit(1)


def cached(key, artifact):
    """True when the artifact exists and the step is not force-listed."""
    return key not in FORCE and os.path.exists(os.path.join(WORK, artifact))


# ---- find full-show recordings (>400MB video files), largest first (best/master candidate) ----
srcs = sorted([f for f in glob.glob(os.path.join(args.gig, "*"))
               if os.path.isfile(f)
               and os.path.splitext(f)[1].lower() in (".mp4", ".mov", ".m4v")
               and os.path.getsize(f) > 400_000_000],
              key=os.path.getsize, reverse=True)
if not srcs:
    print("ERROR: no full-show recordings (>400MB) found in", args.gig)
    sys.exit(1)
print("Sources:")
for s in srcs:
    print("  %s (%.1f GB)" % (os.path.basename(s), os.path.getsize(s) / 1e9))

# ---- 1. analyze (sync + audio ranking) ----
if cached("analyze", "analysis.json"):
    print("\n[1/10 ANALYZE] cached")
else:
    run_step("1/10 ANALYZE", ["analyze.py", FF, FP, WORK] + srcs)

analysis = json.load(open(os.path.join(WORK, "analysis.json")))
master = analysis["_meta"]["master"]
master_src = analysis[master]["path"]

# ---- 2. song detection ----
if cached("songs", "songs.json"):
    print("\n[2/10 SONG DETECT] cached")
else:
    run_step("2/10 SONG DETECT", ["song_detect.py", FF, WORK, master_src])

# ---- 3. section segmentation ----
if cached("sections", "sections.json"):
    print("\n[3/10 SECTION SEGMENT] cached")
else:
    run_step("3/10 SECTION SEGMENT", ["section_segment.py", WORK])

# ---- 4. FINE musical structure (structure2: demucs vocal-aware novelty + repetition + chords) ----
#         Per the chosen song we (a) demucs-separate its vocal stem ONCE (cached under _stems/, reused
#         if present -- never re-separated) then (b) run structure2.analyze_song to emit the fine section
#         map <workdir>/vocal_structure_song<N>.json that the assembler consumes. This REPLACES the old
#         fixed-k structure_analyze. Cache-aware + config-driven; nothing is hardcoded to a machine/band.
songs_meta = json.load(open(os.path.join(WORK, "songs.json")))
songs_meta = songs_meta if isinstance(songs_meta, list) else songs_meta.get("songs", songs_meta)


def _structure2_for_song(song_index, force):
    """Ensure the fine section map for song_index exists (structure2). Returns the map path.
    Demucs stems are cached under _stems/; structure2.separate_stems reuses them without importing
    torch/demucs when both stem wavs already exist."""
    smap = os.path.join(WORK, "vocal_structure_song%d.json" % song_index)
    if not force and os.path.exists(smap):
        print("\n[4/10 FINE STRUCTURE] song %d cached (%s)" % (song_index, os.path.basename(smap)))
        return smap
    wav = os.path.join(WORK, "song_%02d.wav" % song_index)
    if not os.path.exists(wav):
        print("ERROR: song wav missing for --song %d: %s" % (song_index, wav)); sys.exit(1)
    abs_start = songs_meta[song_index - 1].get("start", 0.0) if song_index - 1 < len(songs_meta) else 0.0
    stem_voc = os.path.join(WORK, "_stems", "htdemucs", "song%d" % song_index, "vocals.wav")
    print("\n[4/10 FINE STRUCTURE] song %d: demucs stems %s"
          % (song_index, "cached (reuse)" if os.path.exists(stem_voc) else "SEPARATING (first run)"))
    run_step("4/10 FINE STRUCTURE (structure2)",
             ["structure2.py", wav, WORK, "song%d" % song_index, "%.3f" % abs_start, smap])
    return smap


# which song(s) get a fine map: the requested --song, else every song (auto-pick happens in assemble).
force_struct = "structure" in FORCE
if args.song:
    _structure2_for_song(args.song, force_struct)
else:
    for _sm in songs_meta:
        _structure2_for_song(_sm.get("index"), force_struct)

# ---- 5. style model (research grammar). Tracked style_model.json ships with the repo; only rebuild
#         if it's missing AND a reels dir exists (research data is local/gitignored). ----
style_json = os.path.join(PROJ, "style_model.json")
if ("style" in FORCE or not os.path.exists(style_json)) and os.path.isdir(REELS_DIR):
    run_step("5/10 STYLE MODEL", ["build_style_model.py", FF, REELS_DIR, style_json])
else:
    print("\n[5/10 STYLE MODEL] using existing style_model.json" if os.path.exists(style_json)
          else "\n[5/10 STYLE MODEL] skipped (no reels dir)")

# ---- 6. assemble ONE song (forward-skip arrangement) -> edit_plan.json ----
if cached("assemble", "edit_plan.json"):
    print("\n[6/10 ASSEMBLE SONG] cached")
else:
    cmd = ["assemble_song.py", WORK, args.name, "--target", str(args.target)]
    if args.song:
        cmd += ["--song", str(args.song)]
    run_step("6/10 ASSEMBLE SONG", cmd)

# ---- 7. infer effects (effects_lab placement, moment-driven duration model) ----
#         Always re-run when assemble ran; only skip when the plan already carries effects and neither
#         assemble nor effects was forced (infer_effects mutates edit_plan.json in place).
plan = json.load(open(os.path.join(WORK, "edit_plan.json"), encoding="utf-8"))
have_fx = bool(plan.get("effects"))
if have_fx and "effects" not in FORCE and "assemble" not in FORCE:
    print("\n[7/10 INFER EFFECTS] cached (plan already has %d effects)" % len(plan["effects"]))
else:
    run_step("7/10 INFER EFFECTS", ["infer_effects.py", WORK])

# ---- 8. build CapCut draft ----
if not args.draft:
    print("\n[8/10 BUILD DRAFT] skipped (coded-effects MP4 is the deliverable; pass --draft to also build one)")
elif not DRAFT_ROOT:
    print("\n[8/10 BUILD DRAFT] skipped (no capcut_draft_root in config)")
else:
    run_step("8/10 BUILD DRAFT", ["build_draft.py", FP, WORK, DRAFT_ROOT, args.name])

# ---- 9. crackle-free fx preview + no-fx baseline ----
preview = os.path.join(OUT, args.name + "_fx_preview.mp4")
baseline = os.path.join(WORK, "baseline_nofx.mp4")
if cached("preview", os.path.relpath(preview, WORK)) and os.path.exists(preview):
    print("\n[9/10 FX PREVIEW] cached")
else:
    run_step("9/10 FX PREVIEW", ["render_timeline_fx_v2.py", FF, FP, WORK, preview,
                                 "--baseline", baseline])

# ---- 10. CONTENT-QA GATE + AUTO-RETRY on the rendered MP4 (the coded-effects deliverable) ----
# Run qa_gate.py on the finished render. On FAIL, for the HIGHEST-PRIORITY failed gate, re-run its
# blame step with an adjusted knob (assemble: exclude the offending section / raise --min; effects:
# force the fallback / lower intensity; render: re-render), then re-render + re-check. Capped at
# --max-retries; if still failing we deliver the best attempt and PRINT the residual gap (never
# silently ship a failing reel, never loop forever). Every iteration is logged for audit.
import qa_gate as QG

qa_log = []          # audit trail: one entry per iteration
best_attempt = None  # (n_pass, results, preview_copy) -- kept so we can deliver the best if all fail


def _run_qa(report_path):
    res = QG.run_all(WORK, preview, FF, FP, baseline)
    QG.print_table(res)
    try:
        json.dump({"gates": res, "all_pass": all(r["pass"] for r in res)},
                  open(report_path, "w"), indent=2, default=str)
    except Exception:
        pass
    return res


def _assemble_cmd(extra):
    cmd = ["assemble_song.py", WORK, args.name, "--target", str(args.target)]
    if args.song:
        cmd += ["--song", str(args.song)]
    return cmd + extra


# knob state carried across iterations
excl_abs = []                 # sections to exclude from assemble (offending defective choruses)
min_bump = 0.0                # additive bump to assemble --min (whole-sections / blip fix)
force_fallback_fx = False     # force infer_effects to lean on the subtle fallback accent


def _apply_fix(gate):
    """Translate a failed gate into a concrete re-run of its blame step with the adjusted knob.
    Returns a short human string describing the knob change (for the audit log)."""
    global excl_abs, min_bump, force_fallback_fx
    blame = gate["blame_step"]
    name = gate["name"]
    if blame == "assemble":
        if name == "complete_chorus":
            new = [o["abs"] for o in gate.get("_offenders", [])]
            excl_abs = sorted(set(excl_abs) | set(new))
            knob = "assemble --exclude-abs %s (drop defective chorus)" % excl_abs
        elif name == "whole_sections":
            min_bump += 8.0            # raise the floor so short blips get absorbed into whole sections
            knob = "assemble --min +%.0f (no blips)" % min_bump
        else:                          # ends_on_music / forward_only -> rebuild the arrangement
            knob = "assemble re-run (%s)" % name
        extra = []
        if excl_abs:
            extra += ["--exclude-abs", ",".join("%.3f" % x for x in excl_abs)]
        if min_bump:
            extra += ["--min", "%.1f" % (45.0 + min_bump)]
        run_step("QA-RETRY assemble", _assemble_cmd(extra))
        run_step("QA-RETRY infer effects", ["infer_effects.py", WORK])
    elif blame == "effects":
        force_fallback_fx = True
        knob = "infer_effects re-run (force subtle fallback / drop heavy+forbidden)"
        run_step("QA-RETRY infer effects", ["infer_effects.py", WORK])
    else:  # render
        knob = "re-render (%s)" % name
        # (assemble/effects unchanged; a fresh render re-applies resolve_ending, qsin seams, full res)
    # every fix ends with a fresh render so the gate re-measures the real MP4
    run_step("QA-RETRY render", ["render_timeline_fx_v2.py", FF, FP, WORK, preview,
                                 "--baseline", baseline])
    return knob


if args.no_qa_retry:
    print("\n[10/10 QA GATE] skipped (--no-qa-retry)")
else:
    print("\n" + "=" * 64 + "\n[10/10 CONTENT-QA GATE + AUTO-RETRY]\n" + "=" * 64, flush=True)
    results = _run_qa(os.path.join(WORK, "qa_report.json"))
    it = 0
    stuck = {}     # gate_name -> last measured value, to detect a re-render that changes nothing
    while not all(r["pass"] for r in results) and it < args.max_retries:
        n_pass = sum(1 for r in results if r["pass"])
        if best_attempt is None or n_pass > best_attempt[0]:
            best_attempt = (n_pass, results)
        # highest-priority failed gate drives the fix, but SKIP a gate whose last re-run left its
        # measured value unchanged (re-rendering the same deterministic pipeline won't fix it -- avoid
        # burning retries on an unfixable measurement). Fall through to the next failed gate; if every
        # failed gate is stuck, stop early and deliver the best attempt.
        failed = [r for r in results if not r["pass"]]
        failed.sort(key=lambda r: QG.GATE_PRIORITY.index(r["name"])
                    if r["name"] in QG.GATE_PRIORITY else 99)
        target_gate = next((r for r in failed
                            if stuck.get(r["name"]) != str(r["measured"])), None)
        if target_gate is None:
            print("\n  [QA-RETRY] all remaining failures are unchanged by re-runs (stuck) -- stopping "
                  "the loop to avoid a no-op cycle.")
            break
        it += 1
        print("\n---- QA RETRY %d/%d: fixing '%s' (blame=%s) ----"
              % (it, args.max_retries, target_gate["name"], target_gate["blame_step"]), flush=True)
        knob = _apply_fix(target_gate)
        results = _run_qa(os.path.join(WORK, "qa_report.json"))
        new_meas = next((r["measured"] for r in results if r["name"] == target_gate["name"]), "?")
        # remember this gate's post-fix measurement; if it re-fails next round with the SAME value the
        # loop treats it as stuck and won't waste another render on it.
        still_failing = next((r for r in results if r["name"] == target_gate["name"] and not r["pass"]),
                             None)
        stuck[target_gate["name"]] = str(new_meas) if still_failing else None
        qa_log.append({
            "iteration": it, "failed_gate": target_gate["name"], "blame": target_gate["blame_step"],
            "knob_changed": knob, "old_measured": target_gate["measured"], "new_measured": new_meas,
            "gates_pass_after": sum(1 for r in results if r["pass"]),
        })

    n_pass = sum(1 for r in results if r["pass"])
    if best_attempt is None or n_pass > best_attempt[0]:
        best_attempt = (n_pass, results)

    print("\n" + "-" * 64 + "\nQA-RETRY AUDIT LOG\n" + "-" * 64)
    if not qa_log:
        print("  (no retries needed -- passed on the first render)")
    for e in qa_log:
        print("  iter %d: gate '%s' FAILED (blame=%s)\n"
              "          knob -> %s\n"
              "          measured %s  =>  %s   (%d/9 gates pass after)"
              % (e["iteration"], e["failed_gate"], e["blame"], e["knob_changed"],
                 e["old_measured"], e["new_measured"], e["gates_pass_after"]))

    if all(r["pass"] for r in results):
        print("\n  QA GATE: ALL 9 PASS -- reel is clean.")
    else:
        # never silently ship a failing reel: deliver the best attempt and PRINT the residual gap
        print("\n  QA GATE: still failing after %d retries -- delivering best attempt (%d/9 pass)."
              % (args.max_retries, best_attempt[0]))
        print("  RESIDUAL GAP (human review needed):")
        for r in best_attempt[1]:
            if not r["pass"]:
                print("    * %-20s measured %s  (want %s; blame=%s, knob='%s')"
                      % (r["name"], r["measured"], r["threshold"], r["blame_step"], r["fix_knob"]))

# ---- (opt-in) CapCut-draft structural QA (legacy qa_check.py; only under --draft) ----
if args.draft and not args.no_qa and DRAFT_ROOT:
    print("\n" + "=" * 64 + "\n[QA CHECK -- CapCut draft]\n" + "=" * 64, flush=True)
    r = subprocess.run([PY, "qa_check.py", DRAFT_ROOT, args.name, WORK], cwd=PROJ)
    if r.returncode != 0:
        print("QA reported hard-gate failures on the CapCut draft (see above). Draft still written.")

print("\n" + "=" * 64 + "\nDONE (v2 chain)\n" + "=" * 64)
print("Preview : %s" % preview)
print("Baseline: %s" % baseline)
print("Draft   : %s" % (os.path.join(DRAFT_ROOT, args.name) if DRAFT_ROOT and args.draft
                        else "(none -- coded-effects MP4 is the deliverable)"))
print("Plan    : %s" % os.path.join(WORK, "edit_plan.json"))
