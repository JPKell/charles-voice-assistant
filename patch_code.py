#!/usr/bin/env python3
from pathlib import Path
import sys

REQUEST_BLOCK = '    def _request_stream(self, messages: list[dict[str, Any]]):\n        """Open one streaming Ollama /api/chat round with tools enabled."""\n        payload = {\n            "model": self.cfg.model,\n            "messages": messages,\n            "stream": True,\n            "tools": self.tool_schemas,\n            "think": self.cfg.think,\n            "keep_alive": self.cfg.keep_alive,\n            "options": {\n                "temperature": self.cfg.temperature,\n                "num_ctx": self.cfg.num_ctx,\n            },\n        }\n\n        response = requests.post(\n            f"{self.cfg.base_url.rstrip(\'/\')}/api/chat",\n            json=payload,\n            stream=True,\n            timeout=self.cfg.request_timeout_seconds,\n        )\n\n        if response.status_code >= 400:\n            body = response.text[:1000]\n            lowered = body.lower()\n            if (\n                response.status_code == 400\n                and "tool" in lowered\n                and (\n                    "support" in lowered\n                    or "unsupported" in lowered\n                    or "does not" in lowered\n                )\n            ):\n                response.close()\n                raise ToolSupportUnavailable(body)\n\n            try:\n                response.raise_for_status()\n            finally:\n                response.close()\n\n        return response\n\n    def _stream_round(self, messages: list[dict[str, Any]]):\n        """Yield visible content while accumulating a complete assistant turn."""\n        thinking_parts: list[str] = []\n        content_parts: list[str] = []\n        tool_calls: list[dict[str, Any]] = []\n\n        response = self._request_stream(messages)\n        try:\n            for raw_line in response.iter_lines(decode_unicode=True):\n                if not raw_line:\n                    continue\n\n                if isinstance(raw_line, bytes):\n                    raw_line = raw_line.decode("utf-8", errors="replace")\n\n                try:\n                    chunk = json.loads(raw_line)\n                except json.JSONDecodeError as exc:\n                    raise RuntimeError(\n                        f"Invalid Ollama streaming JSON: {raw_line[:200]!r}"\n                    ) from exc\n\n                stream_error = chunk.get("error")\n                if stream_error:\n                    raise RuntimeError(f"Ollama streaming error: {stream_error}")\n\n                message = chunk.get("message") or {}\n\n                thinking = message.get("thinking")\n                if thinking:\n                    thinking_parts.append(str(thinking))\n\n                content = message.get("content")\n                if content:\n                    piece = str(content)\n                    content_parts.append(piece)\n                    yield piece\n\n                calls = message.get("tool_calls") or []\n                if calls:\n                    tool_calls.extend(calls)\n        finally:\n            response.close()\n\n        return {\n            "thinking": "".join(thinking_parts),\n            "content": "".join(content_parts),\n            "tool_calls": tool_calls,\n        }'
CHAT_STREAM_BLOCK = '    def chat_stream(self, user_text: str) -> Iterable[str]:\n        if not self.tool_schemas:\n            yield from self.base.chat_stream(user_text)\n            return\n\n        messages = self._messages(user_text)\n        total_calls = 0\n\n        try:\n            for _round in range(self.settings.max_rounds):\n                round_message = yield from self._stream_round(messages)\n\n                thinking = str(round_message.get("thinking") or "")\n                content = str(round_message.get("content") or "")\n                tool_calls = round_message.get("tool_calls") or []\n\n                if tool_calls:\n                    assistant_message: dict[str, Any] = {\n                        "role": "assistant",\n                        "content": content,\n                        "tool_calls": tool_calls,\n                    }\n                    if thinking:\n                        assistant_message["thinking"] = thinking\n                    messages.append(assistant_message)\n\n                    for call in tool_calls:\n                        total_calls += 1\n                        function = call.get("function") or {}\n                        name = str(function.get("name") or "").strip()\n                        arguments = self._arguments(call)\n\n                        if total_calls > self.settings.max_calls:\n                            result = {\n                                "ok": False,\n                                "error": (\n                                    "Tool call limit reached: "\n                                    f"{self.settings.max_calls}"\n                                ),\n                            }\n                        elif name not in self.enabled_tools:\n                            result = {\n                                "ok": False,\n                                "error": f"Tool is not enabled: {name}",\n                            }\n                        else:\n                            result = self.registry.execute(name, arguments)\n\n                        self._log(\n                            name=name or "(missing)",\n                            arguments=arguments,\n                            result=result,\n                        )\n\n                        messages.append(\n                            {\n                                "role": "tool",\n                                "tool_name": name,\n                                "content": json.dumps(\n                                    result,\n                                    ensure_ascii=False,\n                                ),\n                            }\n                        )\n\n                    continue\n\n                answer = content.strip()\n                if not answer:\n                    answer = "I couldn\'t produce a final response."\n                    yield answer\n\n                self._remember(user_text, answer)\n                return\n\n        except ToolSupportUnavailable:\n            print(\n                "[tools] Ollama/model rejected tool calling; "\n                "falling back to normal chat for this request.",\n                file=sys.stderr,\n            )\n            yield from self.base.chat_stream(user_text)\n            return\n\n        answer = (\n            "I couldn\'t complete that request because the tool-call limit "\n            "was reached."\n        )\n        self._remember(user_text, answer)\n        yield answer'


def patch_tooling(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "class ToolCallingChat:" not in text:
        raise SystemExit(
            "ToolCallingChat was not found. Apply the tool-calling patch first."
        )

    request_start = text.find("    def _request(")
    request_end = text.find(
        "    @staticmethod\n    def _arguments",
        request_start,
    )

    if request_start < 0:
        request_start = text.find("    def _request_stream(")
        request_end = text.find(
            "    @staticmethod\n    def _arguments",
            request_start,
        )

    if request_start < 0 or request_end < 0:
        raise SystemExit(
            "Could not locate the ToolCallingChat request method block."
        )

    text = (
        text[:request_start]
        + REQUEST_BLOCK
        + "\n\n"
        + text[request_end:]
    )

    stream_start = text.find("    def chat_stream(")
    stream_end = text.find(
        "\n\nclass ToolSupportUnavailable",
        stream_start,
    )
    if stream_start < 0 or stream_end < 0:
        raise SystemExit(
            "Could not locate the ToolCallingChat.chat_stream method block."
        )

    text = (
        text[:stream_start]
        + CHAT_STREAM_BLOCK
        + text[stream_end:]
    )

    path.write_text(text, encoding="utf-8")


def main() -> int:
    root = Path(sys.argv[1]).expanduser().resolve()
    patch_tooling(root / "voice_chat" / "tooling.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
