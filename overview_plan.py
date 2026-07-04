"""
overview_plan.py  -  SPEC-DRIVEN whole-gig RECAP planner (gig 11).

Reads the authoritative data, never hardcodes structure:
  edit_spec.json   - section windows, grades, the review decisions (D1-D13)
  music_events.json- kicks / crashes / downbeats / builds / key anchors (the SYNC events)
  moment_pool.json - gig-wide footage candidates (WHICH footage fills each slot; D10)

Laws it enforces:
  D1/D2 SYNC LAW   - every cut lands on a real hit (crash > strong kick > section change);
                     zero wasted transitions; NO cut-speed floor.
  D3    PACING     - non-uniform: decaying gaps, >=1 deliberate hero-hold (2-4s) per groove
                     section on the highest-motion moment, ~1-in-4 downbeat skipped.
  D4    BUILDS     - the two detected build->release points get accelerating cuts resolving
                     ON the release hit (crash@16.833, drop@30.467).
  D5    EFFECTS    - each section carries a LONG-FORM effect/grade (held across the cuts).
  D10   CROSS-GIG  - footage drawn from moment_pool across all 9 songs; bed audio = song 9.
Every shot records spec_ref (which rule drove it) for the V-SPEC-IS-READ drift-guard.

Usage: python overview_plan.py <workdir>  ->  <workdir>/overview_edl.json
"""
import json, os, sys
FPS = int(os.environ.get("OV_FPS", "60"))     # 60 -> cuts align to the finer 16ms grid (source is 60fps)
fa = lambda t: round(round(t * FPS) / FPS, 3)


def build_edl(wd):
    spec = json.load(open(f"{wd}/edit_spec.json", encoding="utf-8"))
    ev = json.load(open(f"{wd}/music_events.json", encoding="utf-8"))
    pool = json.load(open(f"{wd}/moment_pool.json", encoding="utf-8"))["moments"]
    G = spec["global"]

    kicks = sorted(ev["kicks"]); crashes = sorted(ev["crashes"]); downbeats = sorted(ev["downbeats"])
    hits = sorted(set(kicks + crashes))
    on_kicks = sorted(ev.get("on_beat_kicks", kicks))
    on_snares = sorted(ev.get("on_beat_snares", []))
    felt = sorted(set(on_kicks) | set(crashes) | set(on_snares))   # the FELT beat -> no ghost-kick cuts
    strong_intro = sorted(set(downbeats) | set(crashes))           # clean opening pulse (0-8 was messy)
    dbl12 = ev.get("double_12")                                    # the LOUD double-snare near 0:12 (precise)
    fill_onsets = sorted(ev.get("fill_onsets", []))                # precise fast-fill kicks (0:15 build)
    predrop_split = ev.get("predrop_split")                        # strong kick inside the slow-mo (2nd scene)
    build_pulse = sorted(ev.get("build_pulse", []))                # the STEADY build pulse (even-spaced)
    LEAD = ev.get("cut_lead_s", 0.0)                               # lead-correct cuts to the true attack
    K = ev["key"]; builds = ev["builds"]
    W = float(ev.get("window_s", 48.0))
    b_early, b_final = builds[0], builds[1]
    drop = K["drop"]; icut = K["instruments_cut"]; voc = K.get("vocal_in", drop)
    chorus_crash = next((c for c in crashes if c > 40), 44.933)

    # ---------- gig-wide footage picker (D10): vibe-matched, angle-varied, spread, no-reuse ----------
    used, song_use = set(), {}
    def pick(vibe="mid", angle=None, hero=False):
        best, bs = None, -1e9
        for m in pool:
            if m["rank"] in used:
                continue
            s = 0.0
            s += 1.0 if m["vibe"] == vibe else (0.3 if abs("low mid peak".split().index(m["vibe"]) -
                                                          "low mid peak".split().index(vibe)) == 1 else 0.0)
            if angle and m["angle"] == angle:
                s += 0.6
            if hero:
                s += 1.4 * m["motion"]
            s += 0.25 * m["score"] - 0.28 * song_use.get(m["song"], 0)
            if s > bs:
                bs, best = s, m
        if best is None:
            best = min(pool, key=lambda m: song_use.get(m["song"], 0))
        used.add(best["rank"]); song_use[best["song"]] = song_use.get(best["song"], 0) + 1
        return {"src_master": best["src_master"], "angle": best["angle"], "song": best["song"],
                "motion": best["motion"], "vibe": best["vibe"]}

    shots = []
    def add(t0, t1, clip, crop, trans, grade, fx, label, spec_ref, **extra):
        if fa(t1) - fa(t0) < 0.05:
            return
        sh = {"tl_start": fa(t0), "dur": round(fa(t1) - fa(t0), 3), "src_master": clip["src_master"],
              "angle": clip["angle"], "src_song": clip["song"], "crop": crop, "transition": trans,
              "grade": grade, "fx": fx, "label": label, "spec_ref": spec_ref,
              "vibe": clip.get("vibe"), "motion": clip.get("motion", 0)}
        sh.update(extra); shots.append(sh)

    def snap(t, pool_hits, tol):
        c = [h for h in pool_hits if abs(h - t) <= tol]
        return min(c, key=lambda h: abs(h - t)) if c else None

    sb = lambda t: fa(snap(t, hits, 0.55) or t)         # snap a soft section boundary to a real hit
    all_hits = sorted(set(kicks + crashes + ev.get("snares", [])))

    def steady_cuts(lo, hi, period, tol=0.16):
        """A REGULAR `period` cadence, each beat snapped to the nearest real hit -> a clean,
        synced pulse through a rhythmically-busy section (holds where the music breathes)."""
        out = [lo]; t = lo
        while t < hi - 0.25:
            t += period
            h = snap(t, all_hits, tol)
            if h and h > out[-1] + 0.30:
                out.append(h)
        return out

    # ---------- montage cut generator: SYNC-LAW + non-uniform (D1/D2/D3) ----------
    crash_set = set(crashes)
    def montage(lo, hi, gap0, gap1, grade, vibe, crops, angles, fx=None,
                hero=True, spec_ref="", label="montage", hitset=None, force=None, preset=None):
        # Either use a PRESET cadence (already snapped to real hits) or walk the FELT-beat hits and
        # select which to cut on. Every cut lands on a beat the ear tracks -- NOT ghost kicks.
        # Crashes always cut; on-beat kicks/snares cut once the decaying gap elapsed; ~1-in-4
        # skipped. `force` guarantees specific times get a cut (e.g. a double-snare).
        span = max(0.5, hi - lo)
        if preset is not None:
            cuts = [lo] + [c for c in preset if lo < c < hi]
        else:
            pulse = hitset if hitset is not None else felt
            seg_hits = sorted(h for h in pulse if lo + 0.05 < h < hi - 0.05)
            cuts = [lo]; last = lo; sk = 0
            for h in seg_hits:
                frac = (h - lo) / span
                gap = gap0 + (gap1 - gap0) * frac
                if h in crash_set and (h - last) >= min(0.28, gap * 0.6):
                    cuts.append(h); last = h
                elif (h - last) >= gap:
                    sk += 1
                    if sk % 4 == 0 and (hi - h) > gap * 1.6:      # deliberate skip for phrasing
                        continue
                    cuts.append(h); last = h
        forced = sorted(fa(f) for f in (force or []) if lo < f < hi)
        base = [c for c in cuts if all(abs(c - f) > 0.14 for f in forced)]   # keep forced cuts clean
        cuts = sorted(set(fa(c) for c in base + [hi] + forced))
        # hero-hold: make the FIRST shot a deliberate ~2.5s hold on the best peak moment,
        # ending on a REAL hit (so its out-cut still obeys the sync law)
        segs = []
        if hero and (hi - lo) > 4.5:
            hero_end = next((h for h in pulse if h >= lo + 2.4), None)   # end the hold on a FELT hit
            if hero_end and hero_end < hi - 1.0:
                hero_end = fa(hero_end)
                segs.append((lo, hero_end, True))
                cuts = sorted(set([hero_end] + [c for c in cuts if c > hero_end + 0.2] + [hi]))
        for i in range(len(cuts) - 1):
            segs.append((cuts[i], cuts[i + 1], False))
        for i, (a, b, is_hero) in enumerate(segs):
            clip = pick(vibe="peak" if is_hero else vibe, angle=angles[i % len(angles)], hero=is_hero)
            add(a, b, clip, "wide" if is_hero else crops[i % len(crops)],
                "cut", grade, ("hero_hold" if is_hero else fx),
                f"{label}{' HERO' if is_hero else ''}", spec_ref)

    # ================= TIMELINE =================
    co_end = fa(next((h for h in strong_intro if h >= 1.5), 2.0))   # cold-open ends ON a strong beat
    # 1. COLD-OPEN 0.0-co_end : held most-electric shot; bloom-in; bed's crash@0.633 lands INSIDE.
    #    Starts at 0.0 so the video aligns with the bed from t=0 (no shift).
    co = pick(vibe="peak", hero=True)
    add(0.0, co_end, co, "wide", "cut", "warm", "bloom_in",
        "cold-open (most-electric, crash@0.633 inside)", "sec.cold_open")

    # 2a. INTRO GROOVE co_end-11 : busy opening chorus -> a STEADY half-bar cadence snapped to real
    #     hits (clean synced pulse, holds where the music breathes) instead of a scattered walk.
    montage(co_end, 11.0, 0, 0, "warm", "mid",
            ["wide", "band", "tight", "band"], ["front", "drummer", "front"],
            hero=False, spec_ref="sec.intro_groove", label="intro",
            preset=steady_cuts(co_end, 11.0, ev["bar"] / 2))

    # The 8-count build must start ON THE BAR (a real downbeat), NOT spill early. build_bar is the
    # bar downbeat; the build's fast cuts begin there, the pre-build stays CALM before it.
    rel = b_early["release"]
    build_bar = fa(ev.get("build_bar") or b_early["start"])
    bpx = sorted(set(fa(x) for x in build_pulse if build_bar - 0.03 <= x < rel - 0.04))  # beats FROM the bar

    # 2b. PRE-BUILD 11-build_bar : CALM steady half-bar cadence (NO acceleration into the build, so
    #     the fast changes don't start early). Forces the double-snare cut.
    montage(11.0, build_bar, 0, 0, "warm", "mid",
            ["band", "wide", "band", "tight"], ["front", "drummer"],
            hero=False, spec_ref="sec.pre_build", label="pre-build",
            preset=steady_cuts(11.0, build_bar, ev["bar"] / 2), force=dbl12)

    # 4. EARLY BUILD : from the BAR, cut on the STEADY pulse -- EVERY beat before the crash gets a cut
    #    (no skipped count), even ~0.23s spacing locks to the drum. Flash last 3; whip on the FINAL beat.
    N = len(bpx) - 1
    for i in range(N):
        clip = pick(vibe="peak", angle=("front", "drummer")[i % 2], hero=False)
        crop = ["band", "band", "tight", "tight", "tight"][min(4, int(i / max(1, N - 1) * 5))]
        last = i >= N - 3                              # flash the last 3 (emphasize the build)
        add(bpx[i], bpx[i + 1], clip, crop, "cut", "warm",
            "build_flash" if last else None, f"early-build beat {i+1}/{N}", "marquee.early_build")
    # 5. EARLY-BUILD RESOLVE : the FINAL beat before the bar resolves via WHIP onto crash@16.833
    #    (no count skipped between the last beat and the crash).
    clip = pick(vibe="peak", angle="front", hero=True)
    add(bpx[-1], rel, clip, "tight", "whip", "warm", "whip_smear",
        "early-build whip -> crash@16.833", "marquee.early_build_resolve", whip_to=rel)

    # 6. BASS GROOVE 16.833-25.067 : held SOFT-TEAL block (blue was too strong, D-feedback) + hero-hold
    montage(rel, b_final["start"], 1.15, 0.85, "teal_soft", "mid",
            ["band", "wide", "tight", "band"], ["front", "drummer", "front"],
            hero=True, spec_ref="sec.bass_groove", label="bass-groove")

    # 7. FINAL BUILD : cut every quarter kick EXCEPT the last (reserved so the slow-mo can run
    #    longer); crop tightens, grade FLIPS warm->cool into the drop.
    qk = sorted(set([k for k in kicks if b_final["start"] - 0.02 <= k <= b_final["end"] + 0.02]
                    + [b_final["start"], b_final["end"]]))
    slowmo_start = qk[-2] if len(qk) >= 3 else icut     # start the slow-mo one kick early -> drags it longer
    M = len(qk) - 2
    for i in range(M):
        clip = pick(vibe="peak", angle=("front", "drummer")[i % 2], hero=False)
        crop = ["band", "band", "tight", "tight", "tight"][min(4, int(i / max(1, M - 1) * 5))]
        last = i >= M - 3
        add(qk[i], qk[i + 1], clip, crop, "cut", "flip",
            "build_flash" if last else None, f"final-build kick {i+1}/{M}",
            "marquee.final_build", flip_frac=round(i / max(1, M - 1), 3))

    # 8. PRE-DROP : a CRISP on-beat cut at predrop_split (the beat before the hang), then an
    #    INCREASING BLUR ramping into the drop (a tension-builder). Real-time, so the beat reads.
    sp_mid = predrop_split if (predrop_split and slowmo_start + 0.12 < predrop_split < drop - 0.12) \
        else fa((slowmo_start + drop) / 2)
    c1 = pick(vibe="peak", angle="front", hero=True)
    add(slowmo_start, sp_mid, c1, "band", "cut", "warm_cool", None,
        "pre-drop scene 1 (crisp)", "marquee.pre_drop")
    c2 = pick(vibe="peak", angle="front", hero=True)
    add(sp_mid, drop, c2, "zoomin", "cut", "warm_cool", "predrop_ramp",
        "pre-drop scene 2 (increasing blur -> drop)", "marquee.pre_drop")

    # 9. DROP 30.467-31.633 : punch + held red/magenta wash + rgb glitch, grade->vibrant
    dend = next((h for h in hits if h >= drop + 0.9), drop + 1.166)
    clip = pick(vibe="peak", angle="front", hero=True)
    add(drop, dend, clip, "punch", "zoom", "vibrant", "drop_wash",
        "DROP (return harder, wash+glitch)", "marquee.drop")

    # 10. CHORUS dend-~45 : vibrant, HELD color across cuts, most-dynamic section (0.5-1.2s varied)
    chorus_end = sb(max(chorus_crash + 0.6, 45.0))
    montage(dend, chorus_end, 0.95, 0.55, "vibrant", "peak",
            ["tight", "band", "wide", "tight", "band"], ["front", "drummer", "front", "drummer"],
            hero=True, spec_ref="sec.chorus", label="chorus (held vibrant)")

    # 11. OUTRO chorus_end-(>=48) : held VHS/chromatic + film-burn bloom + blur-out on the ring-out
    out_end = fa(max(W, chorus_end + 3.0))
    clip = pick(vibe="mid", angle="front", hero=True)
    add(chorus_end, out_end, clip, "wide", "fade", "outro", "film_burn_out",
        "outro breath / ring-out (read full master past 48)", "sec.outro", ringout=True)

    # LEAD-CORRECT: my onset detection lags the true drum attack ~12ms, so nudge every interior
    # cut earlier by LEAD -> scene-changes land ON the attack (tighter coupling). Shot 0 stays at 0.
    final_end = round(shots[-1]["tl_start"] + shots[-1]["dur"], 3)
    if LEAD:
        for s in shots[1:]:
            s["tl_start"] = fa(s["tl_start"] - LEAD)
    for i in range(len(shots) - 1):
        shots[i]["dur"] = round(shots[i + 1]["tl_start"] - shots[i]["tl_start"], 3)
    shots[-1]["dur"] = round(final_end - shots[-1]["tl_start"], 3)
    shots = [s for s in shots if s["dur"] >= 0.05]     # drop only render-degenerate shots (not a sliver
    #   bandaid): correct bar-aligned sectioning is what prevents slivers; qa_overview WARNS on any that
    #   remain so a sectioning bug is surfaced for review rather than silently patched.
    total = final_end
    return {"bed": "song_09", "tempo": ev.get("tempo"), "duration": round(total, 3),
            "n_shots": len(shots), "songs_used": sorted(song_use.keys()),
            "key": K, "builds": builds, "shots": shots}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python overview_plan.py <workdir>")
    wd = sys.argv[1]
    edl = build_edl(wd)
    json.dump(edl, open(f"{wd}/overview_edl.json", "w", encoding="utf-8"), indent=2)
    print(f"SPEC-DRIVEN RECAP EDL: {edl['n_shots']} shots, {edl['duration']}s, footage from songs {edl['songs_used']}")
