#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-$PWD}"
TARGET="$(cd -- "$TARGET" && pwd)"

BACKUP="$(find "$TARGET/.patch_backups" -maxdepth 1 -type d -name 'output_tuning_*' 2>/dev/null | sort | tail -n 1)"

if [[ -z "${BACKUP:-}" ]]; then
    echo "No output-tuning backup found."
    exit 1
fi

cp -a "$BACKUP/voice_chat/audio.py" "$TARGET/voice_chat/audio.py"

echo "Restored from:"
echo "  $BACKUP"
