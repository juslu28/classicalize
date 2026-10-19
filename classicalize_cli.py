#!/usr/bin/env python3
"""Command-line use:
  python classicalize_cli.py "Shape of You - Ed Sheeran" --w 0.6
  python classicalize_cli.py --file mysong.mp3 --composer chopin
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classicalize.composers import COMPOSERS  # noqa: E402
from classicalize.pipeline import run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("query", nargs="?", default="")
ap.add_argument("--file", help="use a local audio file instead of YouTube")
ap.add_argument("--url", help="a specific YouTube URL")
ap.add_argument("--slider", type=float, default=0.5,
                help="0 = the original played by classical instruments, 1 = a classical piece with hints of the song")
ap.add_argument("--era", default="auto", choices=["auto", "baroque", "classical", "romantic"])
ap.add_argument("--composer", default="auto", choices=["auto", *COMPOSERS])
ap.add_argument("--ensemble", default="auto", choices=["auto", "piano", "string_quartet", "chamber", "orchestra"])
a = ap.parse_args()
if not (a.query or a.file or a.url):
    ap.error("give a song name, --url or --file")
local = None
if a.file:
    os.makedirs("output/_uploads", exist_ok=True)
    local = os.path.join("output/_uploads", os.path.splitext(os.path.basename(a.file))[0] + ".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.file, "-t", "360", "-ar", "44100", "-ac", "2", local],
                   check=True)
r = run(a.query, s=a.slider, era=a.era, composer=a.composer, ensemble=a.ensemble, video_url=a.url, local_wav=local,
        progress=lambda m, *x: print(" ", m))
p = r["params"]
print(f"\nIn the style of {COMPOSERS[p['composer']]['name']} · {p['era']} · {p['ensemble']}  "
      f"({r['seconds']:.0f}s)")
print(r["review"])
print("→", os.path.join(r["out_dir"], r["files"]["classical"]))
