#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-$PWD}"
TARGET="$(cd -- "$TARGET" && pwd)"

BACKUP="$(find "$TARGET/.patch_backups" -maxdepth 1 -type d -name 'streaming_tools_*' 2>/dev/null | sort | tail -n 1)"

if [[ -z "${BACKUP:-}" ]]; then
    echo "No streaming-tools backup found."
    exit 1
fi

cp -a "$BACKUP/voice_chat/tooling.py" "$TARGET/voice_chat/tooling.py"

echo "Restored from:"
echo "  $BACKUP"
