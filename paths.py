# -*- coding: utf-8 -*-
"""Shared device-path resolver. Reads config.json (gitignored) from this file's own
directory so no per-machine paths are hardcoded in the tracked scripts.

Falls back to PATH-resolved tool names when config.json is absent, so a fresh clone
still imports and runs (assuming ffmpeg/ffprobe/python are on PATH).

Dependency-free (json + os only).
"""
import os, json

_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG_PATH = os.path.join(_HERE, "config.json")

_cfg = {}
try:
    with open(_CFG_PATH, "r", encoding="utf-8") as _fh:
        _cfg = json.load(_fh) or {}
except (OSError, ValueError):
    _cfg = {}


def _get(key, fallback):
    val = _cfg.get(key)
    return val if val else fallback


# Executables -- fall back to bare names resolved via PATH.
FFMPEG = _get("ffmpeg", "ffmpeg")
FFPROBE = _get("ffprobe", "ffprobe")
PYTHON = _get("python", "python")

# Other device paths (None when unset so callers can decide).
CAPCUT_DRAFT_ROOT = _get("capcut_draft_root", None)
GIGS_DIR = _get("gigs_dir", None)
WORK_DIR = _get("work_dir", None)

# Full parsed config, in case a caller needs a key not exposed above.
CONFIG = _cfg
