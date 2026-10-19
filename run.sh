#!/usr/bin/env bash
cd "$(dirname "$0")"
source "$HOME/miniforge3/bin/activate" classicalize
export PYTORCH_ENABLE_MPS_FALLBACK=1
( sleep 3 && open "http://localhost:${PORT:-8765}" ) &
python server.py
