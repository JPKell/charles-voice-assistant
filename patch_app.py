#!/usr/bin/env python3
from pathlib import Path
import re
import sys


def add_import(app: str) -> str:
    line = "from .tooling import ToolCallingChat, load_tool_settings\n"
    if line in app:
        return app

    matches = list(re.finditer(r"^from \.[^\n]+\n", app, flags=re.MULTILINE))
    if matches:
        pos = matches[-1].end()
        return app[:pos] + line + app[pos:]

    marker = "import requests\n"
    if marker in app:
        return app.replace(marker, marker + "\n" + line, 1)

    raise SystemExit("Could not find a safe import insertion point in app.py")


def add_flag(app: str) -> str:
    if '"--no-tools"' in app:
        return app

    flag_line = (
        '    parser.add_argument("--no-tools", action="store_true", '
        'help="Disable Ollama tool calling for this run")\n'
    )

    marker = (
        '    parser.add_argument("--no-stream-tts", action="store_true", '
        'help="Wait for the full answer before speaking")\n'
    )
    if marker in app:
        return app.replace(marker, marker + flag_line, 1)

    marker = "    return parser.parse_args()\n"
    if marker in app:
        return app.replace(marker, flag_line + marker, 1)

    raise SystemExit("Could not find parse_args() return in app.py")


def add_wrapper(app: str) -> str:
    if "llm = ToolCallingChat(llm, tool_settings)" in app:
        return app

    lines = app.splitlines(keepends=True)
    start_idx = None

    for i, line in enumerate(lines):
        if re.match(r"^\s*llm\s*=\s*OllamaChat\s*\(", line):
            start_idx = i
            break

    if start_idx is None:
        raise SystemExit("Could not locate `llm = OllamaChat(...)` in app.py")

    balance = 0
    seen_open = False
    end_idx = None

    for i in range(start_idx, len(lines)):
        for ch in lines[i]:
            if ch == "(":
                balance += 1
                seen_open = True
            elif ch == ")":
                balance -= 1

        if seen_open and balance == 0:
            end_idx = i
            break

    if end_idx is None:
        raise SystemExit("Could not determine end of OllamaChat(...)")

    indent = re.match(r"^(\s*)", lines[start_idx]).group(1)

    block = (
        f"{indent}tool_settings = load_tool_settings(\n"
        f"{indent}    args.config,\n"
        f"{indent}    disabled=args.no_tools,\n"
        f"{indent})\n"
        f"{indent}if tool_settings.enabled:\n"
        f"{indent}    llm = ToolCallingChat(llm, tool_settings)\n"
    )

    lines.insert(end_idx + 1, block)
    return "".join(lines)


def main() -> int:
    target = Path(sys.argv[1]).resolve()
    app_path = target / "voice_chat" / "app.py"

    app = app_path.read_text(encoding="utf-8")
    app = add_import(app)
    app = add_flag(app)
    app = add_wrapper(app)
    app_path.write_text(app, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
