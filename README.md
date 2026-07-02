# Gig Reel Auto-Editor

Turn raw gig footage into an editable, **on-style** Instagram reel — a CapCut draft **and** a preview MP4
with effects — using your band's *learned* editing style. Runs on your own device and your own footage.

## ▶ Start here — say exactly this to your Claude agent (in this folder)

> **"Set up the auto-video-editor from this repo: read README.md, ask me the setup questions, write my
> config.json, then edit my gig footage at `<PATH-TO-YOUR-GIG-FOLDER>`."**

That single line starts the whole workflow: it asks a few questions (paths, CapCut vs MP4-only, which
gig/song), then generates your edit + a preview MP4 + a QA report. To re-run later, just say
**"edit my next gig at `<folder>`"** or **"regenerate with a different song."**

---

## What's shared vs. what's yours

- **Shared (band-level — reuse as-is):** the learned STYLE and grammar (`style_model.json`), the
  code-rendered effect library (`effects_lab.py`), the research + QA agents, and the pipeline scripts.
  These encode *how our reels look and cut* — identical for everyone in the band.
- **Yours (per device):** file paths (ffmpeg, CapCut, footage), which gig/song, and your edit preferences.

You normally **do not re-run research** — you inherit the band's style. Only refine it if you have new reels.

---

## Quick start

1. **Install:** Python 3.12 with `numpy scipy librosa soundfile faster-whisper torch beat_this`, **ffmpeg**,
   and (optional) **CapCut**.
2. **Set up:** open `claude` in this folder and say **"set up the auto-editor"**. It runs a short
   questionnaire (below) and writes your `config.json` (copy from `config.template.json`).
3. **Edit a gig:** say **"edit gig <folder>"** (or "auto-pick the best song"). You get a CapCut draft +
   a preview MP4 + a QA report.

---

## Setup questionnaire (what the agent asks — only the pertinent bits)

1. **Where is your gig footage folder?** (the folder holding the multicam `.mp4/.mov` files)
2. **Where is ffmpeg?** (path, or say "auto-detect")
3. **Do you use CapCut?** If yes → where is its draft folder; if no → **output MP4 only**.
4. **Which gig/song?** (a folder + song, or "auto-pick the strongest song")
5. **Reuse the band's style research** (recommended) **or refine it** with new reels?
6. **Target length + effect intensity?** (defaults: ~80s, tasteful — effects only on hits/drops)

---

## Steering prompts (say any of these to shape the edit)

- "Make a per-song edit of **<song>**, ~80s, punchier effects in the chorus/solo."
- "More restraint — verses clean, only the drop gets a hit."
- "**Refine the research** with these new reels: <folder or links>."
- "Regenerate with a **different song** / a **longer** cut."
- "**Show me the effect choices and why** (the reasons)."
- "Just give me the **preview MP4**, skip CapCut."
- "The transition at 0:30 feels abrupt — **crossfade it**."

---

## What you get (sample output)

- **CapCut draft** `YOUR_HANDLE_<gig>` — open in CapCut (fully close & reopen to see new drafts). Cuts,
  transitions, and effects placed on real musical events; audio on the clips.
- **Preview MP4** `<gig>/_auto_output/<name>_EXPORT.mp4` — small, with **audio + effects baked in**, so you
  can review the whole edit in one file without CapCut.
- **QA report**, e.g.:
  > forward-skip: intro → verse → chorus → solo → chorus → outro (78s)
  > 4 effects, all on drops/hits, ramped in/out, density punchier in chorus/solo — **congruent**
  > ending resolves on the phrase; audio −14 LUFS; **QA PASS**

---

## Reusing / refining the research

`style_model.json` (the measured grammar: effects on hits/drops, arousal-scaled density, cadence, boundary
rules) and `effects_lab.py` (visually-validated effect renders + trigger rules) are the **band's** —
inherited by everyone. Got new reels worth learning from? Say **"refine the research"** and the
`reel-research` agent updates `style_model.json` for the whole band.

---

## Config (per device) — `config.json`

Copy `config.template.json` → `config.json` and fill in your paths. Every script reads it, so nothing is
hardcoded to one machine.

---

*Maintenance: this README and `config.template.json` are kept in sync with the pipeline automatically —
whenever the scripts, config schema, or style change, they're updated without you asking.*
