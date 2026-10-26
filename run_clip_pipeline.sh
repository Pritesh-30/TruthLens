#!/usr/bin/env bash
# Runs the whole CLIP image-detector pipeline end to end. Every step resumes:
# prepare_genimage.py skips finished shards, extract_clip_features.py skips
# finished feature files, so re-running after an interruption picks up where
# it stopped. Progress goes to pipeline.log.
set -o pipefail
cd "$(dirname "$0")"
PY=venv/bin/python
FILTER='UNEXPECTED|unauthenticated|HF_TOKEN|Loading weights|LOAD REPORT|^Key |^-+\+|^Notes|can be ignored'

echo "=== $(date '+%F %T') pipeline start ==="

echo "=== $(date '+%F %T') step 1/3: download + convert Tiny-GenImage ==="
$PY -u prepare_genimage.py 2>&1 | grep --line-buffered -v -E "$FILTER" || { echo "PIPELINE FAILED at step 1"; exit 1; }

echo "=== $(date '+%F %T') step 2/3: extract CLIP features ==="
$PY -u extract_clip_features.py 2>&1 | grep --line-buffered -v -E "$FILTER" || { echo "PIPELINE FAILED at step 2"; exit 1; }

echo "=== $(date '+%F %T') step 3/3: train classifier head ==="
$PY -u train_clip_head.py 2>&1 | grep --line-buffered -v -E "$FILTER" || { echo "PIPELINE FAILED at step 3"; exit 1; }

echo "=== $(date '+%F %T') PIPELINE COMPLETE ==="
