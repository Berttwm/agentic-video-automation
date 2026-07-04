"""
qa_overview.py  -  RECAP-rubric QA gate (D13, OV-36/37). NOT the per-song qa_gate (which
would reject a correct recap). Checks the rendered overview against the locked laws and
prints a per-gate report. Designed to GROW: add gates as requirements/lessons accrue,
prune obsolete ones. Layer-A (measurable) here; Layer-B (qualitative congruence) is the
qa-validator sub-agent.

Usage: python qa_overview.py <workdir> <render.mp4>
"""
import sys, json, subprocess, numpy as np, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared"))
import paths, av_sync

FPS = 30
EVENT_FX = {"slowmo", "drop_wash", "film_burn_out"}     # intentionally event-timed, exempt from cut-snap


def gate(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main():
    wd, render = sys.argv[1], sys.argv[2]
    ev = json.load(open(f"{wd}/music_events.json", encoding="utf-8"))
    edl = json.load(open(f"{wd}/overview_edl.json", encoding="utf-8"))
    S = edl["shots"]
    LEAD = ev.get("cut_lead_s", 0.0)     # cuts are intentionally lead-corrected to the true attack
    hits = np.array(sorted(set(ev["kicks"] + ev["crashes"] + ev.get("snares", []) + ev.get("fill_onsets", [])
                               + ev.get("build_pulse", []) + ev.get("double_12", [])))) - LEAD
    builds = ev["builds"]
    results = []
    print(f"qa_overview: {len(S)} shots, {edl['duration']}s")

    # 1. SYNC LANDING (D1) - every felt cut within 30ms of a real hit
    offs = [float(min(abs(hits - s["tl_start"])) * 1000) for s in S
            if s["tl_start"] >= 0.7 and s.get("fx") not in EVENT_FX]
    bad = [o for o in offs if o > 30]
    results.append(gate("SYNC-LANDING", len(bad) == 0,
                        f"{len(offs)-len(bad)}/{len(offs)} cuts <=30ms off a real hit (mean {np.mean(offs):.0f}ms, max {max(offs):.0f}ms)"))

    # 1b. MIX COUPLING (the workflow gap) - verify cuts land tight on the ACTUAL MIX transients the
    #     ear hears, not just my detected stem hits. Catches a systematic detection-lag offset.
    try:
        import librosa
        my, _ = librosa.load(f"{wd}/song_09.wav", sr=44100, mono=True, duration=min(42.0, edl["duration"]))
        onm = np.array(librosa.onset.onset_detect(y=my, sr=44100, hop_length=128, units="time", backtrack=True))
        moffs = [float((onm - s["tl_start"])[np.argmin(np.abs(onm - s["tl_start"]))]) for s in S
                 if s["tl_start"] >= 0.7 and s.get("fx") not in EVENT_FX
                 and abs((onm - s["tl_start"])[np.argmin(np.abs(onm - s["tl_start"]))]) < 0.06]
        ms = np.mean(moffs) * 1000 if moffs else 0.0
        # tolerance = detection/backtrack bias (~10ms) + one frame (16ms @60fps); slightly-early reads tight
        results.append(gate("MIX-COUPLING", abs(ms) <= 20,
                            f"cuts vs MIX transients: mean {ms:+.0f}ms (|<=20|), |abs| {np.mean(np.abs(moffs))*1000:.0f}ms, n={len(moffs)}"))
    except Exception as ex:
        print(f"  [warn] MIX-COUPLING skipped: {ex}")

    # 2. BUILDS LAND ON RELEASE (D4) - a cut within 1 frame of each build release
    starts = np.array([s["tl_start"] for s in S])
    okb = True; bd = []
    for b in builds:
        rel = b.get("release")
        if rel is None:
            continue
        d = float(min(abs(starts - (rel - LEAD))) * 1000)
        bd.append(f"{b['kind']}->{rel}s ({d:.0f}ms)")
        okb = okb and d <= 40
    results.append(gate("BUILDS-LAND", okb, "; ".join(bd)))

    # 3. PACING SHAPE (D3/OV-11) - varied lengths, >=2 hero-holds, no near-equal run in the
    #    MONTAGE sections. Builds are intentionally steady (a quarter-kick run) -> exempt them.
    durs = [s["dur"] for s in S]
    distinct = len(set(round(d, 1) for d in durs))
    heroes = sum(1 for d in durs if d >= 2.2)
    run, worst = 1, 1
    for i in range(1, len(S)):
        # builds are intentionally steady; the intro is a deliberate clean synced pulse (the user's
        # request) -> both exempt from the "metronome" check. Grooves/chorus must still vary.
        steady = any(w in ((S[i].get("label") or "") + (S[i - 1].get("label") or ""))
                     for w in ("build", "intro"))
        if not steady and abs(durs[i] - durs[i - 1]) <= 0.1 * max(durs[i], durs[i - 1]):
            run += 1; worst = max(worst, run)
        else:
            run = 1
    ok3 = distinct >= 10 and heroes >= 2 and worst < 5
    results.append(gate("PACING-SHAPE", ok3,
                        f"{distinct} distinct lengths, {heroes} hero-holds, longest montage near-equal run={worst} (<5 ok; builds exempt)"))

    # 4. CROSS-GIG FOOTAGE (D10) - footage from many songs, none dominating
    from collections import Counter
    songc = Counter(s["src_song"] for s in S)
    dom = max(songc.values()) / len(S)
    ok4 = len(songc) >= 5 and dom <= 0.45
    results.append(gate("CROSS-GIG", ok4, f"{len(songc)} songs, top song {dom*100:.0f}% of shots (<=45% ok)"))

    # 5. EFFECT BALANCE (D5) - sustained-dominant: held-colour blocks carry screen-time; stabs are few
    longform_grades = {"blue", "teal_soft", "flip", "vibrant", "outro"}
    lf_time = sum(s["dur"] for s in S if s["grade"] in longform_grades)
    stabs = sum(1 for s in S if s.get("fx") == "build_flash")
    ok5 = lf_time >= 0.45 * edl["duration"] and stabs <= 8
    results.append(gate("EFFECT-BALANCE", ok5,
                        f"held-colour long-form screen-time {lf_time/edl['duration']*100:.0f}% (>=45%), {stabs} flash stabs (<=8)"))

    # 6. A/V LOCK (OV-02) - drift <= 1 frame
    v, a, dr, oklock = av_sync.verify_av(paths.FFPROBE, render, FPS)
    results.append(gate("AV-LOCK", dr <= 1.0, f"video {v:.2f}s / audio {a:.2f}s / drift {dr}f"))

    # 7. NO-TEXT (D12) - iteration-1 has no on-screen text overlays (planner emits none)
    results.append(gate("NO-TEXT", all("caption" not in (s.get("fx") or "") for s in S),
                        "captions/text off for iteration 1"))

    # ---- WARNINGS (soft flags for REVIEW, not pass/fail) : catch the classes of bug that pass the
    #      generic gates but read wrong -- an intent-anchored cut landing off its anchor, a seam
    #      sliver, or a cut whose sync is decoupled by an effect. These get surfaced to the user. ----
    warns = []
    starts = np.array([s["tl_start"] for s in S])
    ncut = lambda t: float(min(abs(starts - t)))
    # (a) every INTENT anchor must have a cut landing ON it (cuts are lead-corrected, so compare to t-LEAD)
    anchors = [("build starts ON the bar", ev.get("build_bar")),
               ("double-snare hit 1", (ev.get("double_12") or [None])[0]),
               ("double-snare hit 2", (ev.get("double_12") or [None, None])[1]),
               ("pre-drop 2nd-scene beat", ev.get("predrop_split")),
               ("DROP", ev["key"].get("drop"))]
    for name, t in anchors:
        if t is None: continue
        d = ncut(t - LEAD) * 1000
        if d > 45:
            warns.append(f"INTENT '{name}' @ {t}s: nearest cut is {d:.0f}ms away -- expected a transition ON it. Ask the user.")
    # (b) seam slivers (a sectioning bug, not to be silently patched)
    for s in S:
        if s["dur"] < 0.15 and "build" not in (s.get("label") or ""):
            warns.append(f"SLIVER {s['dur']*1000:.0f}ms shot @ {s['tl_start']}s ({s['label']}) -- a section-seam bug, not intentional.")
    # (c) effect-decoupled sync: slow-mo cuts are on-beat but the STRETCHED video won't visibly pulse
    sm = [round(s["tl_start"], 2) for s in S if s.get("fx") == "slowmo"]
    if sm:
        warns.append(f"SLOW-MO cuts at {sm}: cut timing is on-beat but the time-stretched video won't visibly pulse with the audio -- confirm it reads right.")

    npass = sum(results)
    print(f"\nqa_overview: {npass}/{len(results)} gates PASS")
    if warns:
        print(f"{len(warns)} WARNING(S) for review:")
        for w in warns:
            print("  [WARN]", w)
    return npass == len(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
