#!/usr/bin/env bash
# Codespace post-create: install lean CPU deps, then warm the HF model cache.
set -euo pipefail

pip install -r deploy/codespace/requirements.txt
python deploy/codespace/warm_cache.py
