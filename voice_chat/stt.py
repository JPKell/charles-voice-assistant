from __future__ import annotations

from pathlib import Path
import tempfile
import threading

import numpy as np
from faster_whisper import WhisperModel

from .audio import save_wav
from .config import STTConfig


class SpeechToText:
    def __init__(self, cfg: STTConfig, sample_rate: int):
        self.cfg = cfg
        self.sample_rate = sample_rate
        print(
            f"Loading Faster-Whisper {cfg.model} "
            f"on {cfg.device}/{cfg.compute_type}..."
        )
        self.model = WhisperModel(
            cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
            cpu_threads=cfg.cpu_threads,
        )
        self._transcribe_lock = threading.Lock()

    def transcribe(self, audio: np.ndarray) -> str:
        with self._transcribe_lock:
            return self._transcribe(audio)

    def _transcribe(self, audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)

        try:
            save_wav(path, audio, self.sample_rate)
            vad_parameters = {
                "threshold": self.cfg.vad_threshold,
                "min_speech_duration_ms": self.cfg.vad_min_speech_ms,
                "min_silence_duration_ms": self.cfg.vad_min_silence_ms,
                "speech_pad_ms": 120,
            }
            segments, _info = self.model.transcribe(
                str(path),
                language=self.cfg.language or None,
                beam_size=self.cfg.beam_size,
                condition_on_previous_text=False,
                vad_filter=self.cfg.vad_filter,
                vad_parameters=vad_parameters if self.cfg.vad_filter else None,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
        finally:
            path.unlink(missing_ok=True)
