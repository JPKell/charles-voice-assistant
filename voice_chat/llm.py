from __future__ import annotations

from collections.abc import Iterable
import json

import requests

from .config import OllamaConfig
from .memory import MemoryStore


class OllamaChat:
    def __init__(
        self,
        cfg: OllamaConfig,
        system_prompt: str,
        max_history_turns: int,
        memory_store: MemoryStore | None = None,
    ):
        self.cfg = cfg
        self.system_prompt = system_prompt.strip()
        self.max_history_turns = max_history_turns
        self.memory_store = memory_store
        self.history: list[dict[str, str]] = []
        if memory_store is not None:
            self.history = memory_store.recent_messages(max_history_turns)

    def healthcheck(self) -> None:
        url = f"{self.cfg.base_url.rstrip('/')}/api/tags"
        response = requests.get(url, timeout=5)
        response.raise_for_status()

    def warmup(self) -> None:
        """Load the configured model and context before microphone listening."""
        response = requests.post(
            f"{self.cfg.base_url.rstrip('/')}/api/generate",
            json={
                "model": self.cfg.model,
                "prompt": "",
                "stream": False,
                "keep_alive": self.cfg.keep_alive,
                "options": {"num_ctx": self.cfg.num_ctx},
            },
            timeout=self.cfg.request_timeout_seconds,
        )
        response.raise_for_status()

    def _messages(self, user_text: str) -> list[dict[str, str]]:
        keep_messages = max(0, self.max_history_turns * 2)
        recent = self.history[-keep_messages:] if keep_messages else []
        return [
            {"role": "system", "content": self.system_prompt},
            *recent,
            {"role": "user", "content": user_text},
        ]

    def chat_stream(self, user_text: str) -> Iterable[str]:
        url = f"{self.cfg.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.cfg.model,
            "messages": self._messages(user_text),
            "stream": True,
            "think": self.cfg.think,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "temperature": self.cfg.temperature,
                "num_ctx": self.cfg.num_ctx,
                "num_predict": self.cfg.num_predict,
            },
        }

        response = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=self.cfg.request_timeout_seconds,
        )
        response.raise_for_status()

        parts: list[str] = []
        completed = False
        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                item = json.loads(raw_line)
                message = item.get("message") or {}
                content = message.get("content") or ""
                if content:
                    parts.append(content)
                    yield content
                if item.get("done"):
                    completed = True
        finally:
            response.close()

        answer = "".join(parts).strip()
        if completed and answer:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": answer})
            if self.memory_store is not None:
                self.memory_store.add_turn(user_text, answer)

    def clear_history(self, persistent: bool = False) -> None:
        self.history.clear()
        if persistent and self.memory_store is not None:
            self.memory_store.clear()
