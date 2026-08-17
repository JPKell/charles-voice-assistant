#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
    echo "Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

exec .venv/bin/python -m voice_chat.app "$@"
