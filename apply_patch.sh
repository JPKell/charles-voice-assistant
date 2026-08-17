#!/usr/bin/env bash
set -euo pipefail

PATCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$PWD}"
TARGET="$(cd -- "$TARGET" && pwd)"

TOOLING="$TARGET/voice_chat/tooling.py"
PYTHON="$TARGET/.venv/bin/python"

for item in "$TOOLING" "$PYTHON"; do
    if [[ ! -e "$item" ]]; then
        echo "ERROR: missing $item"
        echo "Usage: ./apply_patch.sh /path/to/local_voice_chat"
        exit 1
    fi
done

if ! grep -q "class ToolCallingChat" "$TOOLING"; then
    echo "ERROR: ToolCallingChat was not found."
    echo "Apply the tool-calling patch first."
    exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="$TARGET/.patch_backups/streaming_tools_$STAMP"
mkdir -p "$BACKUP/voice_chat"
cp -a "$TOOLING" "$BACKUP/voice_chat/tooling.py"

rollback() {
    echo
    echo "Patch failed; restoring tooling.py..."
    cp -a "$BACKUP/voice_chat/tooling.py" "$TOOLING"
}
trap rollback ERR

echo "Enabling streaming Ollama tool calls..."
"$PYTHON" "$PATCH_ROOT/patch_code.py" "$TARGET"

echo "Checking Python syntax..."
"$PYTHON" -m py_compile "$TOOLING"

echo "Checking patch markers..."
grep -q '"stream": True' "$TOOLING"
grep -q "def _stream_round" "$TOOLING"
grep -q "iter_lines" "$TOOLING"

if grep -q '"stream": False' "$TOOLING"; then
    echo "ERROR: a non-streaming tool request remains in tooling.py"
    false
fi

trap - ERR

cp -a "$PATCH_ROOT/STREAMING_TOOLS.md" "$TARGET/STREAMING_TOOLS.md"

echo
echo "Patch applied successfully."
echo
echo "Behavior:"
echo "  - ordinary no-tool answers stream to TTS immediately"
echo "  - tool calls are accumulated from Ollama's NDJSON stream"
echo "  - tool results are returned to Ollama in the agent loop"
echo "  - final tool-grounded answers also stream to TTS"
echo "  - hidden thinking is accumulated but not spoken"
echo
echo "Backup:"
echo "  $BACKUP"
