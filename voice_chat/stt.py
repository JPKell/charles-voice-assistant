from __future__ import annotations

import ctypes
from pathlib import Path
import sys
import threading

import numpy as np
from faster_whisper import WhisperModel

from .audio import resample_linear
from .config import STTConfig


class SpeechToText:
    def __init__(self, cfg: STTConfig, sample_rate: int):
        self.cfg = cfg
        self.sample_rate = sample_rate
        print(
            f"Loading Faster-Whisper {cfg.model} "
            f"on {cfg.device}/{cfg.compute_type}..."
        )
        if cfg.device.lower().startswith("cuda"):
            self._preload_cuda12()
        self.model = WhisperModel(
            cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
            cpu_threads=cfg.cpu_threads,
        )
        self._transcribe_lock = threading.Lock()
        if cfg.device.lower().startswith("cuda"):
            print("Warming Faster-Whisper GPU kernels...")
            # Model construction is lazy; force the first encoder/decoder pass
            # before Ready so the user's first utterance does not pay for it.
            segments, _info = self.model.transcribe(
                np.zeros(16000, dtype=np.float32),
                language=cfg.language or None,
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            list(segments)

    @staticmethod
    def _preload_cuda12() -> None:
        """Expose the isolated CUDA 12 libraries required by CTranslate2."""
        root = Path(sys.prefix) / "cuda12" / "nvidia"
        libraries = (
            root / "cublas" / "lib" / "libcublas.so.12",
            root / "cudnn" / "lib" / "libcudnn.so.9",
        )
        missing = [str(path) for path in libraries if not path.is_file()]
        if missing:
            raise RuntimeError(
                "Faster-Whisper CUDA libraries are missing; run ./setup.sh. "
                + "Missing: "
                + ", ".join(missing)
            )
        for path in libraries:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

    def transcribe(self, audio: np.ndarray) -> str:
        with self._transcribe_lock:
            return self._transcribe(audio)

    def _transcribe(self, audio: np.ndarray) -> str:
        # Faster-Whisper accepts 16 kHz float32 samples directly. Avoiding a
        # temporary WAV removes a filesystem round trip from every utterance.
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if self.sample_rate != 16000:
            samples = resample_linear(samples, self.sample_rate, 16000)
        vad_parameters = {
            "threshold": self.cfg.vad_threshold,
            "min_speech_duration_ms": self.cfg.vad_min_speech_ms,
            "min_silence_duration_ms": self.cfg.vad_min_silence_ms,
            "speech_pad_ms": 120,
        }
        segments, _info = self.model.transcribe(
            samples,
            language=self.cfg.language or None,
            beam_size=self.cfg.beam_size,
            condition_on_previous_text=False,
            vad_filter=self.cfg.vad_filter,
            vad_parameters=vad_parameters if self.cfg.vad_filter else None,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
