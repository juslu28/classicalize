#!/usr/bin/env python3
"""Build the static GitHub Pages site in docs/ from the songs you have made.

For every song in output/ it re-makes the clip at five slider settings
(0, 25, 50, 75, 100 %), converts the audio to MP3 and writes the JSON the page
reads. No server is needed afterwards: docs/ is plain HTML + JSON + MP3.

    python scripts/build_site.py                       # every song in output/
    python scripts/build_site.py --songs "Blinding_Lights_4NRXx6U8ABQ"
    python scripts/build_site.py --repo https://github.com/you/classicalize
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from classicalize import pipeline  # noqa: E402
from classicalize.composers import catalog  # noqa: E402

STEPS = [0.0, 0.25, 0.5, 0.75, 1.0]


def mp3(src, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-b:a", "160k", dst], check=True)


def build_song(d, out):
    src = os.path.join(pipeline.OUT_ROOT, d)
    dst = os.path.join(out, "songs", d)
    os.makedirs(dst, exist_ok=True)
    backup = os.path.join(src, "result.json")
    saved = open(backup).read() if os.path.exists(backup) else None
    item = None
    try:
        mp3(os.path.join(src, "clip.wav"), os.path.join(dst, "clip.mp3"))
        for s in STEPS:
            tag = f"{int(round(s * 100)):03d}"
            print(f"   slider {int(s * 100):3d}% …", flush=True)
            r = pipeline.remake(d, s=s)
            r.pop("out_dir", None)
            wav = os.path.join(src, r["files"]["classical"])
            name = f"classical_{tag}"
            mp3(wav, os.path.join(dst, name + ".mp3"))
            mid = os.path.join(src, r["files"]["midi"])
            if os.path.exists(mid):
                shutil.copy(mid, os.path.join(dst, name + ".mid"))
            r["files"] = {"classical": name + ".mp3", "mp3": name + ".mp3", "midi": name + ".mid",
                          "original": "clip.mp3"}
            r["dir"], r["mp3_ready"] = d, True
            with open(os.path.join(dst, f"result_{tag}.json"), "w") as f:
                json.dump(r, f, default=str)
            if s == 0.5:
                p = r["params"]
                item = {"dir": d, "title": r.get("title"), "era": p["era"], "composer": p.get("composer"),
                        "ensemble": p["ensemble"], "score": r["total"]["score"], "slider": 0.5,
                        "seconds": r.get("seconds")}
    finally:
        if saved is not None:          # leave the app's own result as it was
            with open(backup, "w") as f:
                f.write(saved)
    return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs"))
    ap.add_argument("--songs", nargs="*", help="folder names in output/ (default: all)")
    ap.add_argument("--repo", default="", help="GitHub URL shown on the page")
    a = ap.parse_args()

    base = pipeline.OUT_ROOT
    songs = a.songs or sorted(
        (x for x in os.listdir(base) if os.path.exists(os.path.join(base, x, "clip_analysis.json"))),
        key=lambda x: -os.path.getmtime(os.path.join(base, x))) if os.path.isdir(base) else []
    if not songs:
        sys.exit("No songs found in output/. Make at least one song with the app first (bash run.sh).")

    if os.path.isdir(os.path.join(a.out, "songs")):
        shutil.rmtree(os.path.join(a.out, "songs"))
    os.makedirs(a.out, exist_ok=True)
    items = []
    for i, d in enumerate(songs, 1):
        print(f"[{i}/{len(songs)}] {d}", flush=True)
        try:
            it = build_song(d, a.out)
            if it:
                items.append(it)
        except Exception as e:  # noqa: BLE001
            print(f"   skipped ({e})")
            shutil.rmtree(os.path.join(a.out, "songs", d), ignore_errors=True)
    if not items:
        sys.exit("No song could be built.")

    with open(os.path.join(a.out, "library.json"), "w") as f:
        json.dump({"items": items}, f)
    with open(os.path.join(a.out, "catalog.json"), "w") as f:
        json.dump(catalog(), f)
    rep = os.path.join(ROOT, "runs", "report.json")
    shutil.copy(rep, os.path.join(a.out, "training.json")) if os.path.exists(rep) else open(
        os.path.join(a.out, "training.json"), "w").write("{}")
    html = open(os.path.join(ROOT, "static", "index.html")).read()
    flag = f"<script>window.CLZ_STATIC=true;window.CLZ_REPO={json.dumps(a.repo)};</script>\n<script>"
    html = html.replace("<script>", flag, 1)
    with open(os.path.join(a.out, "index.html"), "w") as f:
        f.write(html)
    open(os.path.join(a.out, ".nojekyll"), "w").close()
    size = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(a.out) for x in fs) / 1e6
    print(f"\nSite built in {a.out}: {len(items)} song(s), {size:.1f} MB")


if __name__ == "__main__":
    main()
