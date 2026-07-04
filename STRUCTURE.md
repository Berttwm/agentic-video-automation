# Repository structure — shared vs. standalone

**Principle: correctness is SHARED, style is STANDALONE.** Anything whose bug would make
an edit *wrong* (A/V sync, cross-camera colour consistency, the analysis both pipelines
depend on) is a shared device that both pipelines call. Stylistic/creative decisions live
inside each pipeline and are *allowed* to differ, because the two serve different purposes.

## `shared/` — correctness devices (used by BOTH pipelines)
- **`av_sync.py`** — frame-align + exact-frame rendering + A/V-lock verification. Prevents
  the frame-rounding-on-concat drift (overview: cuts slip off the beat; per-song: video
  ends before the audio). Imported by `build_overview.py` **and** `render_timeline_fx_v2.py`.
- **`camera_match.py`** — drummer-cam → master-cam colour match, so the grade is consistent
  across the two cameras. Both pipelines cut between these cameras, so both call it.

*Shared-by-import today, to be physically moved into `shared/` in a careful import-fix pass:*
`paths.py`, `effects_lab.py`, and the analysis stack (`analyze`, `gcc_sync`, `song_detect`,
`section_segment`, `structure2`, `chords`, `vocal_structure`, `beats`) + demucs separation.

## Per-song reel pipeline — STANDALONE style
`run_gig.py` orchestrates → `assemble_song.py` (best-passage, whole sections, minimal cuts)
→ `infer_effects.py` (restrained placement) → `render_timeline_fx_v2.py` (multicam dissolves,
crackle-free audio, let-the-note-ring-out) + `render_preview_fx_v2.py` (gated effect builders)
→ `qa_gate.py` (9 binary gates).
**Style locks:** let it breathe, whole sections, subtle effects, resolve the ending.

## Gig-overview pipeline — STANDALONE style
`overview_plan.py` (music-event-driven EDL — cuts on drum/bass hits, machine-gun build,
slow-mo→drop) → `build_overview.py` (warm→vibrant dynamic grade, motion-picked clips, rich
transitions, Anton title) + `overview_captions.py`.
**Style:** fast, music-driven montage; music bed; title cards — deliberately NOT the
per-song locks.

## Shared agents (`~/.claude/agents/`)
- **reel-research** — style/transition grammar (feeds both).
- **qa-validator** — qualitative QA (both).

## Shared vs. duplicated — current state
- `av_sync` + `camera_match`: **shared**, imported by both renderers (verified compiling).
- Frame-align / exact-frame discipline: applied in both (overview = exact `-frames:v`;
  per-song = frame-aligned shot durations + xfade overlaps, with a built-in A/V-lock check).
- Not yet physically consolidated: `paths`, `effects_lab`, analysis stack (top-level today,
  imported by both) — a follow-up move.
