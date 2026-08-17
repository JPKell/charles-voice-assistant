from __future__ import annotations

import re
import time

from .config import WakeWordConfig


class WakeWordGate:
    """A dependency-free wake-phrase gate over Whisper transcripts.

    This is intentionally not a separate neural keyword detector: audio is still
    transcribed by Whisper, then the transcript is gated by the configured phrase.
    """

    def __init__(self, cfg: WakeWordConfig):
        self.cfg = cfg
        self.active_until = 0.0
        phrase = re.escape(cfg.phrase.strip())
        self._pattern = re.compile(rf"\b{phrase}\b", re.IGNORECASE) if phrase else None

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled and self._pattern is not None

    def process(self, text: str) -> tuple[bool, str]:
        if not self.enabled:
            return True, text.strip()

        now = time.monotonic()
        stripped = text.strip()
        if now <= self.active_until:
            self.active_until = now + self.cfg.followup_seconds
            return True, stripped

        match = self._pattern.search(stripped) if self._pattern else None
        if match is None:
            return False, ""

        if self.cfg.require_prefix:
            before = stripped[: match.start()].strip(" ,.!?:;-\t")
            if before:
                return False, ""

        remainder = (stripped[: match.start()] + " " + stripped[match.end() :]).strip()
        remainder = remainder.strip(" ,.!?:;-\t")
        self.active_until = now + self.cfg.followup_seconds
        return True, remainder
