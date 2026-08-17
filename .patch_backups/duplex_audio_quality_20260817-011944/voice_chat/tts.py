from __future__ import annotations

import re
import threading

import numpy as np

from .audio import play_audio_interruptible, stop_audio
from .config import TTSConfig


KOKORO_SAMPLE_RATE = 24000

_PARAGRAPH_TOKEN = "\ue000PARAGRAPH_BREAK\ue001"


def normalize_tts_text(
    text: str,
    *,
    periods_to_commas: bool,
    paragraphs_to_periods: bool,
    strip_characters: str,
) -> str:
    """Normalize only the copy of text that is sent to Kokoro."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if strip_characters:
        text = text.translate(str.maketrans("", "", strip_characters))

    if paragraphs_to_periods:
        # A blank-line paragraph break becomes exactly one protected token.
        # Remove punctuation immediately before the break so input such as:
        #
        #   Sentence.
        #
        #   Next paragraph.
        #
        # doesn't become "Sentence, . Next paragraph," after periods are
        # converted to commas.
        paragraph_pattern = r"[ \t]*[.,!?;:]?[ \t]*\n[ \t]*\n+"
        text = re.sub(paragraph_pattern, f" {_PARAGRAPH_TOKEN} ", text)
    else:
        text = re.sub(r"\n[ \t]*\n+", " ", text)

    # Single line breaks are formatting, not spoken paragraph pauses.
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)

    if periods_to_commas:
        text = text.replace(".", ",")

    if paragraphs_to_periods:
        text = text.replace(_PARAGRAPH_TOKEN, ".")

    text = re.sub(r"[ \t]+", " ", text)

    # Clean up punctuation spacing created by normalization.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?=[A-Za-z0-9])", r"\1 ", text)

    return text.strip()


class TextToSpeech:
    def __init__(self, cfg: TTSConfig, output_device=None):
        self.cfg = cfg
        self.output_device = output_device

        print(
            f"Loading Kokoro voice system "
            f"(lang={cfg.lang_code}, voice={cfg.voice}, device={cfg.device})..."
        )

        from kokoro import KPipeline

        self.pipeline = KPipeline(
            lang_code=cfg.lang_code,
            repo_id="hexgrad/Kokoro-82M",
            device=cfg.device,
        )

    @staticmethod
    def _to_numpy(audio) -> np.ndarray:
        if hasattr(audio, "detach"):
            audio = audio.detach()
        if hasattr(audio, "cpu"):
            audio = audio.cpu()
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        return np.asarray(audio, dtype=np.float32).squeeze()

    def speak(
        self,
        text: str,
        stop_event: threading.Event | None = None,
        playback=None,
    ) -> bool:
        text = normalize_tts_text(
            text,
            periods_to_commas=self.cfg.periods_to_commas,
            paragraphs_to_periods=self.cfg.paragraphs_to_periods,
            strip_characters=self.cfg.strip_characters,
        )
        if not text:
            return True

        generator = self.pipeline(
            text,
            voice=self.cfg.voice,
            speed=self.cfg.speed,
        )

        for _graphemes, _phonemes, audio in generator:
            if stop_event is not None and stop_event.is_set():
                return False
            if audio is None:
                continue

            samples = self._to_numpy(audio) * self.cfg.volume
            samples = np.clip(samples, -1.0, 1.0)

            if playback is not None:
                completed = playback.play_audio(
                    samples,
                    KOKORO_SAMPLE_RATE,
                    stop_event=stop_event,
                )
            else:
                completed = play_audio_interruptible(
                    samples,
                    KOKORO_SAMPLE_RATE,
                    output_device=self.output_device,
                    stop_event=stop_event,
                )

            if not completed:
                return False

        return True

    def stop(self) -> None:
        stop_audio()
