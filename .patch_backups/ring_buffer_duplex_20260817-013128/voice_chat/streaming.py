from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tts import TextToSpeech


class SentenceAccumulator:
    """Group multiple LLM sentences before handing text to Kokoro."""

    def __init__(
        self,
        min_chars: int = 8,
        max_chars: int = 260,
        sentences_per_chunk: int = 2,
    ):
        self.buffer = ""
        self.min_chars = max(1, min_chars)
        self.max_chars = max(self.min_chars, max_chars)
        self.sentences_per_chunk = max(1, sentences_per_chunk)

    def feed(self, chunk: str) -> list[str]:
        self.buffer += chunk
        ready: list[str] = []

        while True:
            boundary = self._find_group_boundary(self.buffer)
            if boundary is None:
                boundary = self._find_soft_boundary(self.buffer)
            if boundary is None:
                break

            piece = self.buffer[:boundary].strip()
            self.buffer = self.buffer[boundary:].lstrip()
            if piece:
                ready.append(piece)

        return ready

    def finish(self) -> list[str]:
        piece = self.buffer.strip()
        self.buffer = ""
        return [piece] if piece else []

    def _find_group_boundary(self, text: str) -> int | None:
        if len(text) < self.min_chars:
            return None

        count = 0
        i = 0
        length = len(text)

        while i < length:
            ch = text[i]

            if ch == "\n":
                j = i
                newline_count = 0
                while j < length and text[j].isspace():
                    if text[j] == "\n":
                        newline_count += 1
                    j += 1

                if newline_count >= 2:
                    prev = i - 1
                    while prev >= 0 and text[prev].isspace():
                        prev -= 1
                    if prev < 0 or text[prev] not in ".!?":
                        count += 1
                        if count >= self.sentences_per_chunk:
                            return j
                    i = max(i + 1, j)
                    continue

            if ch in ".!?":
                j = i + 1
                while j < length and text[j] in "\"'”’)]}":
                    j += 1

                if j < length and text[j].isspace():
                    count += 1
                    if count >= self.sentences_per_chunk:
                        return j

            i += 1

        return None

    def _find_soft_boundary(self, text: str) -> int | None:
        if len(text) <= self.max_chars:
            return None

        window = text[: self.max_chars]
        for delimiter in ("; ", ": ", ", ", " "):
            pos = window.rfind(delimiter)
            if pos >= self.min_chars:
                return pos + len(delimiter)
        return self.max_chars


@dataclass
class SpeechSession:
    stop_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    pending: int = 0
    closed: bool = False
    canceled: bool = False
    playback: object | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self) -> bool:
        with self.lock:
            if self.canceled:
                return False
            self.pending += 1
            return True

    def item_done(self) -> None:
        with self.lock:
            self.pending = max(0, self.pending - 1)
            if self.closed and self.pending == 0:
                self.done_event.set()

    def close(self) -> None:
        with self.lock:
            self.closed = True
            if self.pending == 0:
                self.done_event.set()

    def cancel(self) -> None:
        with self.lock:
            self.canceled = True
            self.closed = True
            self.stop_event.set()
            self.done_event.set()


class SpeechWorker:
    def __init__(self, tts: "TextToSpeech"):
        self.tts = tts
        self.q: queue.Queue[tuple[SpeechSession | None, str | None]] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="tts-worker", daemon=True)
        self.thread.start()

    def new_session(self, playback=None) -> SpeechSession:
        return SpeechSession(playback=playback)

    def submit(self, session: SpeechSession, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if session.add():
            self.q.put((session, text))

    def cancel(self, session: SpeechSession) -> None:
        session.cancel()

    def _run(self) -> None:
        while True:
            session, text = self.q.get()
            try:
                if session is None:
                    return
                if session.canceled or text is None:
                    continue
                self.tts.speak(
                    text,
                    stop_event=session.stop_event,
                    playback=session.playback,
                )
            finally:
                if session is not None:
                    session.item_done()
                self.q.task_done()

    def close(self) -> None:
        self.q.put((None, None))
        deadline = time.monotonic() + 2.0
        while self.thread.is_alive() and time.monotonic() < deadline:
            self.thread.join(timeout=0.1)
