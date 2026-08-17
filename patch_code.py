#!/usr/bin/env python3
from pathlib import Path
import re
import sys


def replace_required(text: str, pattern: str, replacement: str, label: str) -> str:
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"Could not update {label}; expected exactly one match, found {count}.")
    return new_text


def patch_audio(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "split system-default streams" not in text:
        raise SystemExit(
            "The split system-default audio patch was not detected in voice_chat/audio.py."
        )

    text = replace_required(
        text,
        r'^\s*OUTPUT_RATES\s*=\s*\([^\n]+\)\s*$',
        '    OUTPUT_RATES = (48000, 44100, 24000, 32000, 16000)',
        "OUTPUT_RATES",
    )

    text = replace_required(
        text,
        r'^\s*OUTPUT_CALLBACK_BLOCK_SECONDS\s*=\s*[0-9.]+\s*$',
        '    OUTPUT_CALLBACK_BLOCK_SECONDS = 0.100',
        "OUTPUT_CALLBACK_BLOCK_SECONDS",
    )

    text = replace_required(
        text,
        r'^\s*OUTPUT_BUFFER_SECONDS\s*=\s*[0-9.]+\s*$',
        '    OUTPUT_BUFFER_SECONDS = 1.0',
        "OUTPUT_BUFFER_SECONDS",
    )

    text = replace_required(
        text,
        r'^\s*OUTPUT_PREBUFFER_SECONDS\s*=\s*[0-9.]+\s*$',
        '    OUTPUT_PREBUFFER_SECONDS = 0.60',
        "OUTPUT_PREBUFFER_SECONDS",
    )

    old = '''        print(
            "[audio] split system-default streams; "
            f"input='{input_name}' @ {self.input_rate} Hz; "
            f"output='{output_name}' @ {self.output_rate} Hz; "
            f"output_prebuffer={self.OUTPUT_PREBUFFER_SECONDS:.2f}s"
        )
'''
    new = '''        print(
            "[audio] split system-default streams; "
            f"input='{input_name}' @ {self.input_rate} Hz; "
            f"output='{output_name}' @ {self.output_rate} Hz; "
            f"output_block≈{1000.0 * self.output_block_frames / self.output_rate:.0f}ms; "
            f"prebuffer={self.OUTPUT_PREBUFFER_SECONDS:.2f}s; "
            f"output_buffer={self.OUTPUT_BUFFER_SECONDS:.1f}s"
        )
'''

    if old in text:
        text = text.replace(old, new, 1)
    elif "output_block≈" not in text:
        raise SystemExit("Could not locate the split-stream startup diagnostic block.")

    path.write_text(text, encoding="utf-8")


def main() -> int:
    root = Path(sys.argv[1]).expanduser().resolve()
    patch_audio(root / "voice_chat" / "audio.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
