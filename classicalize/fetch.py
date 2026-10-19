"""Step 1 – find the most popular YouTube recording of a song and download its audio.

Uses yt-dlp's search (no API key). Among the top N results we keep the ones that
look like a real recording of the song (sensible length, not a cover/karaoke/
reaction/live/slowed/"1 hour" version) and pick the one with the most views.

You are responsible for using downloaded audio in line with YouTube's Terms and
copyright law; this tool keeps everything on your own machine.
"""
from __future__ import annotations

import os
import re
import subprocess

BAD_WORDS = ["karaoke", "instrumental", "cover", "reaction", "slowed", "sped up", "nightcore",
             "8d", "1 hour", "10 hours", "loop", "tutorial", "lesson", "piano version",
             "orchestral", "classical", "remix", "live at", "live from", "acoustic",
             "bass boosted", "reverb", "string quartet", "violin cover", "audio edit"]


def _score(entry, query):
    title = (entry.get("title") or "").lower()
    views = entry.get("view_count") or 0
    dur = entry.get("duration") or 0
    if dur and (dur < 90 or dur > 480):
        return -1
    q = query.lower()
    for w in BAD_WORDS:
        if w in title and w not in q:
            return -1
    bonus = 1.0
    if "official" in title:
        bonus *= 1.3
    if "audio" in title:
        bonus *= 1.1           # official audio = no intro skit / sound effects
    return views * bonus


def search(query: str, n: int = 12):
    from yt_dlp import YoutubeDL
    opts = {"quiet": True, "skip_download": True, "extract_flat": "in_playlist", "noplaylist": True,
            "default_search": "ytsearch"}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    entries = [e for e in (info.get("entries") or []) if e]
    ranked = sorted(entries, key=lambda e: _score(e, query), reverse=True)
    return [
        {"id": e["id"], "title": e.get("title"), "channel": e.get("channel") or e.get("uploader"),
         "views": e.get("view_count"), "duration": e.get("duration"),
         "url": f"https://www.youtube.com/watch?v={e['id']}", "score": _score(e, query)}
        for e in ranked
    ]


def download_audio(video_url: str, out_dir: str):
    """Download the audio track (AAC/m4a preferred: small, plays in every browser).
    Writes original.wav (44.1 kHz). Also returns YouTube's "most replayed" heat-map."""
    from yt_dlp import YoutubeDL
    os.makedirs(out_dir, exist_ok=True)
    tmpl = os.path.join(out_dir, "source.%(ext)s")
    opts = {"quiet": True, "format": "bestaudio[ext=m4a]/bestaudio/best", "outtmpl": tmpl, "noplaylist": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
    raw = os.path.join(out_dir, f"source.{info.get('ext', 'm4a')}")
    wav = os.path.join(out_dir, "original.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-t", "360", "-ar", "44100", "-ac", "2", wav],
                   check=True)
    os.remove(raw)
    extra = {"heatmap": info.get("heatmap") or [], "duration": info.get("duration"),
             "title": info.get("title"), "views": info.get("view_count"), "channel": info.get("channel")}
    return wav, extra


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")[:60] or "song"
