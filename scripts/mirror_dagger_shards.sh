#!/usr/bin/env bash
set -euo pipefail
REMOTE="${REMOTE:-root@185.151.171.35}"
PORT="${PORT:-50799}"
IDENTITY="${IDENTITY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/root/tinyvla/artifacts/dagger_round1_twin_full_v3/}"
LOCAL_DIR="${LOCAL_DIR:-artifacts/dagger_round1_twin_full_v3/}"
INTERVAL="${INTERVAL:-60}"
mkdir -p "$LOCAL_DIR"
while true; do
  rsync -az --partial -e "ssh -i $IDENTITY -o IdentitiesOnly=yes -p $PORT" \
    --include='shards/***' --include='progress.json' --include='collector.log' --exclude='*' \
    "$REMOTE:$REMOTE_DIR" "$LOCAL_DIR" || true
  sleep "$INTERVAL"
done
