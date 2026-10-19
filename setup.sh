#!/usr/bin/env bash
# One-time setup for macOS (Apple Silicon). Run:  bash setup.sh
# Uses Miniforge (conda-forge) for Python, ffmpeg and FluidSynth: these are
# ready-made downloads that work on older macOS versions too (no compiling).
set -euo pipefail
cd "$(dirname "$0")"

CONDA_DIR="$HOME/miniforge3"
ENV=classicalize

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  echo "==> Installing Miniforge (a small, self-contained Python distribution) into ~/miniforge3"
  curl -fL --retry 3 -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-$(uname -m).sh"
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
  rm -f /tmp/miniforge.sh
fi
source "$CONDA_DIR/bin/activate"

if ! conda env list | grep -q "^$ENV "; then
  echo "==> Creating the '$ENV' environment (Python 3.11, ffmpeg, FluidSynth)"
  conda create -y -n "$ENV" -c conda-forge python=3.11 ffmpeg fluidsynth
fi
conda activate "$ENV"
echo "==> Optional: Deno (helps the YouTube downloader)"
conda install -y -n "$ENV" -c conda-forge deno >/dev/null 2>&1 || echo "   (Deno not available – that's OK)"

echo "==> Installing Python packages"
pip install -U pip wheel
pip install -r requirements.txt
pip install --no-deps basic-pitch==0.4.0

echo "==> Downloading an orchestral SoundFont (MuseScore General, MIT licence, ~38 MB)"
mkdir -p soundfonts
if [ ! -f soundfonts/MuseScore_General.sf3 ]; then
  curl -fL --retry 3 -o soundfonts/MuseScore_General.sf3 \
    https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3 \
  || echo "!! SoundFont download failed – put any .sf2/.sf3 file into ./soundfonts (a small built-in one is used meanwhile)"
fi

echo "==> Pre-downloading the Demucs separation model (~80 MB, first time only)"
python -c "from demucs.pretrained import get_model; get_model('htdemucs')" || echo "!! Demucs model download failed – it will retry on first use"

echo "==> Self-test (offline, ~30 s)"
python scripts/selftest.py

echo
echo "Done. Start the app with:   bash run.sh"
