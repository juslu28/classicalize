#!/usr/bin/env python3
"""Offline self-test: generate a pop song, render it as audio, and run the full
separation -> analysis -> style search -> arrangement -> recording chain."""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classicalize.pipeline import run  # noqa: E402
from classicalize.render import find_soundfont  # noqa: E402
from classicalize.separate import demucs_available  # noqa: E402
from classicalize.synthpop import make_song, render_pop_stems  # noqa: E402


def main():
    print("SoundFont:", find_soundfont(), "| Demucs:", demucs_available())
    d = tempfile.mkdtemp(prefix="clz_selftest_")
    mix, _ = render_pop_stems(make_song(7, "pop"), d)
    t0 = time.time()
    res = run("self-test song", s=0.0, local_wav=mix, progress=lambda m, *a: print(f"  [{time.time() - t0:5.1f}s] {m}"))
    p = res["params"]
    print(f"\nOK in {res['seconds']:.0f}s → {p['composer']} / {p['ensemble']}  timings: {res['timings']}")


if __name__ == "__main__":
    main()
