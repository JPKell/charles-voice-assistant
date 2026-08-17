#!/usr/bin/env bash
set -euo pipefail

PATCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$PWD}"
TARGET="$(cd -- "$TARGET" && pwd)"

AUDIO="$TARGET/voice_chat/audio.py"
PYTHON="$TARGET/.venv/bin/python"

for item in "$AUDIO" "$PYTHON"; do
    if [[ ! -e "$item" ]]; then
        echo "ERROR: missing $item"
        echo "Usage: ./apply_patch.sh /path/to/local_voice_chat"
        exit 1
    fi
done

if ! grep -q "split system-default streams" "$AUDIO"; then
    echo "ERROR: split system-default audio implementation was not detected."
    echo "Apply local_voice_chat_split_default_audio_patch first."
    exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="$TARGET/.patch_backups/output_tuning_$STAMP"
mkdir -p "$BACKUP/voice_chat"
cp -a "$AUDIO" "$BACKUP/voice_chat/audio.py"

rollback() {
    echo
    echo "Patch failed; restoring audio.py..."
    cp -a "$BACKUP/voice_chat/audio.py" "$AUDIO"
}
trap rollback ERR

echo "Applying output-stream tuning..."
"$PYTHON" "$PATCH_ROOT/patch_code.py" "$TARGET"

echo "Checking Python syntax..."
"$PYTHON" -m py_compile "$AUDIO"

echo "Checking tuned values..."
grep -q 'OUTPUT_RATES = (48000, 44100, 24000, 32000, 16000)' "$AUDIO"
grep -q 'OUTPUT_CALLBACK_BLOCK_SECONDS = 0.100' "$AUDIO"
grep -q 'OUTPUT_BUFFER_SECONDS = 1.0' "$AUDIO"
grep -q 'OUTPUT_PREBUFFER_SECONDS = 0.60' "$AUDIO"

trap - ERR

cp -a "$PATCH_ROOT/OUTPUT_TUNING.md" "$TARGET/OUTPUT_TUNING.md"

echo
echo "Patch applied successfully."
echo
echo "New output tuning:"
echo "  preferred output rate: 48 kHz"
echo "  output callback block: ~100 ms"
echo "  output ring capacity:  1.0 s"
echo "  startup prebuffer:     0.60 s"
echo
echo "Backup:"
echo "  $BACKUP"
