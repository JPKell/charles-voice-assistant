from __future__ import annotations

from pathlib import Path


def load_persona(
    root: Path,
    directory: str,
    name: str,
    fallback_prompt: str,
) -> tuple[str, str]:
    name = name.strip()
    if not name:
        return fallback_prompt.strip(), "config"

    safe_name = Path(name).name
    path = (root / directory / f"{safe_name}.txt").resolve()
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print(f"[persona] {path} not found; using config system_prompt")
        return fallback_prompt.strip(), "config"

    if not prompt:
        print(f"[persona] {path} is empty; using config system_prompt")
        return fallback_prompt.strip(), "config"
    return prompt, safe_name
