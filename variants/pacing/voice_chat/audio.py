from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import queue
import threading
import time
import wave

import numpy as np
import sounddevice as sd

from .config import AudioConfig, STTConfig


def rms(samples: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples)) + 1e-12))


def list_devices() -> None:
    print(sd.query_devices())


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Small dependency-free resampler used only for the duplex barge-in stream."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0 or source_rate == target_rate:
        return audio.astype(np.float32, copy=True)

    target_len = max(1, int(round(audio.size * target_rate / float(source_rate))))
    old_x = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    new_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


class SileroSpeechDetector:
    """Short-window speech detector using Faster-Whisper's bundled Silero VAD."""

    def __init__(self, audio_cfg: AudioConfig, stt_cfg: STTConfig):
        self.sample_rate = audio_cfg.sample_rate
        self.threshold = stt_cfg.vad_threshold
        self.min_speech_ms = stt_cfg.vad_min_speech_ms
        self.min_silence_ms = max(80, min(stt_cfg.vad_min_silence_ms, 500))
        self.window_samples = max(
            512, int(self.sample_rate * audio_cfg.vad_window_ms / 1000)
        )
        self.check_samples = max(
            512, int(self.sample_rate * audio_cfg.vad_check_ms / 1000)
        )
        self._since_check = 0
        self._available = False
        self._vad_options = None
        self._get_speech_timestamps = None

        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps

            self._vad_options = VadOptions(
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_ms,
                min_silence_duration_ms=self.min_silence_ms,
                speech_pad_ms=0,
            )
            self._get_speech_timestamps = get_speech_timestamps
            self._available = True
        except Exception as exc:
            print(f"[vad] Silero unavailable; using RMS fallback: {exc}")

    @property
    def available(self) -> bool:
        return self._available

    def should_check(self, new_samples: int) -> bool:
        self._since_check += new_samples
        if self._since_check >= self.check_samples:
            self._since_check = 0
            return True
        return False

    def speech_recent(self, rolling_audio: np.ndarray) -> bool:
        if not self._available or self._get_speech_timestamps is None:
            return False

        audio = np.asarray(rolling_audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return False
        audio = audio[-self.window_samples :]

        try:
            chunks = self._get_speech_timestamps(
                audio,
                self._vad_options,
                sampling_rate=self.sample_rate,
            )
        except Exception as exc:
            self._available = False
            print(f"[vad] Silero disabled after runtime error: {exc}")
            return False

        if not chunks:
            return False

        latest = chunks[-1]
        tail_allowance = int(self.sample_rate * 0.14)
        return int(latest["end"]) >= max(0, len(audio) - tail_allowance)


@dataclass
class CaptureResult:
    audio: np.ndarray | None
    speech_started: bool


class VoiceRecorder:
    def __init__(self, cfg: AudioConfig, stt_cfg: STTConfig):
        self.cfg = cfg
        self.stt_cfg = stt_cfg
        self.block_frames = max(1, int(cfg.sample_rate * cfg.block_ms / 1000))
        self.pre_roll_blocks = max(
            1, int(math.ceil(cfg.pre_roll_seconds * 1000 / cfg.block_ms))
        )
        self.silence_blocks = max(
            1, int(math.ceil(cfg.silence_seconds * 1000 / cfg.block_ms))
        )
        self.vad = SileroSpeechDetector(cfg, stt_cfg)

    def calibrate(self) -> float:
        blocks = max(
            1, int(math.ceil(self.cfg.calibration_seconds * 1000 / self.cfg.block_ms))
        )
        levels: list[float] = []

        print(f"Calibrating microphone for {self.cfg.calibration_seconds:.1f}s — stay quiet...")
        with sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            dtype="float32",
            blocksize=self.block_frames,
            device=self.cfg.input_device,
        ) as stream:
            for _ in range(blocks):
                data, overflowed = stream.read(self.block_frames)
                if overflowed:
                    print("[audio] input overflow during calibration")
                levels.append(rms(data))

        noise = float(np.median(levels)) if levels else 0.0
        threshold = max(self.cfg.min_rms_threshold, noise * self.cfg.noise_multiplier)
        mode = "Silero VAD + RMS gate" if self.vad.available else "adaptive RMS"
        print(f"Ambient RMS {noise:.4f}; fallback threshold {threshold:.4f}; detector: {mode}")
        return threshold

    def _capture(
        self,
        fallback_threshold: float,
        *,
        start_timeout: float | None,
        stop_event: threading.Event | None = None,
        on_speech_start=None,
        announce: bool = True,
        rms_multiplier: float = 1.0,
    ) -> CaptureResult:
        pre_roll: deque[np.ndarray] = deque(maxlen=self.pre_roll_blocks)
        rolling: deque[np.ndarray] = deque()
        rolling_samples = 0
        captured: list[np.ndarray] = []
        started = False
        silent_blocks = 0
        wait_started = time.monotonic()
        record_started: float | None = None
        vad_speech = False

        if announce:
            print("\nListening...")

        with sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            dtype="float32",
            blocksize=self.block_frames,
            device=self.cfg.input_device,
        ) as stream:
            while True:
                if stop_event is not None and stop_event.is_set() and not started:
                    return CaptureResult(None, False)

                data, overflowed = stream.read(self.block_frames)
                if overflowed and announce:
                    print("[audio] input overflow")

                block = data.copy()
                flat = block.reshape(-1)
                level = rms(flat)

                rolling.append(flat)
                rolling_samples += flat.size
                while rolling and rolling_samples > self.vad.window_samples:
                    removed = rolling.popleft()
                    rolling_samples -= removed.size

                if self.vad.available and self.vad.should_check(flat.size):
                    window = np.concatenate(list(rolling)) if rolling else flat
                    vad_speech = self.vad.speech_recent(window)

                rms_speech = level >= fallback_threshold * rms_multiplier
                speech_now = (vad_speech and rms_speech) if self.vad.available else rms_speech

                if not started:
                    pre_roll.append(block)
                    if speech_now:
                        started = True
                        record_started = time.monotonic()
                        captured.extend(pre_roll)
                        if announce:
                            print("Heard you...")
                        if on_speech_start is not None:
                            on_speech_start()
                        continue

                    if start_timeout is not None and (
                        time.monotonic() - wait_started >= start_timeout
                    ):
                        return CaptureResult(None, False)
                    continue

                captured.append(block)

                if speech_now:
                    silent_blocks = 0
                else:
                    silent_blocks += 1

                if silent_blocks >= self.silence_blocks:
                    break

                if record_started is not None and (
                    time.monotonic() - record_started >= self.cfg.max_record_seconds
                ):
                    if announce:
                        print("[audio] maximum utterance length reached")
                    break

        if not captured:
            return CaptureResult(None, started)

        audio = np.concatenate(captured, axis=0).reshape(-1)
        return CaptureResult(
            np.clip(audio, -1.0, 1.0).astype(np.float32),
            started,
        )

    def record_utterance(self, fallback_threshold: float) -> np.ndarray | None:
        return self._capture(
            fallback_threshold,
            start_timeout=self.cfg.start_timeout_seconds,
            announce=True,
        ).audio


@dataclass
class _PlaybackChunk:
    samples: np.ndarray
    stop_event: threading.Event | None
    done_event: threading.Event = field(default_factory=threading.Event)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    completed: bool = False


class DuplexBargeInSession:
    """One PortAudio full-duplex stream for TTS playback + barge-in capture."""

    COMMON_RATES = (48000, 44100, 32000, 24000, 16000)

    def __init__(
        self,
        recorder: VoiceRecorder,
        fallback_threshold: float,
        *,
        output_device,
        delay_seconds: float,
        rms_multiplier: float,
        keyword_validator=None,
        keyword_silence_seconds: float = 0.35,
        keyword_max_seconds: float = 2.0,
    ):
        self.recorder = recorder
        self.cfg = recorder.cfg
        self.fallback_threshold = float(fallback_threshold)
        self.output_device = output_device
        self.delay_seconds = max(0.0, float(delay_seconds))
        self.rms_multiplier = max(1.0, float(rms_multiplier))
        self.keyword_validator = keyword_validator
        self.keyword_silence_seconds = max(0.15, float(keyword_silence_seconds))
        self.keyword_max_seconds = max(
            self.keyword_silence_seconds + 0.1,
            float(keyword_max_seconds),
        )
        self.recognized_keyword: str | None = None

        self.detected_event = threading.Event()
        self.stop_event = threading.Event()
        self.capture_done_event = threading.Event()

        self.audio: np.ndarray | None = None
        self.stream_rate = self._choose_stream_rate()
        self._playback_queue: queue.Queue[_PlaybackChunk] = queue.Queue()
        self._input_queue: queue.Queue[tuple[np.ndarray, float, float]] = queue.Queue(maxsize=96)

        self._current_chunk: _PlaybackChunk | None = None
        self._current_offset = 0
        self._stream = None
        self._detector_thread = threading.Thread(
            target=self._detector_loop,
            name="barge-in-detector",
            daemon=True,
        )

        self._state_lock = threading.Lock()
        self._first_playback_at: float | None = None

    @property
    def playback(self):
        return self

    def _choose_stream_rate(self) -> int:
        for rate in self.COMMON_RATES:
            try:
                sd.check_input_settings(
                    device=self.cfg.input_device,
                    channels=self.cfg.channels,
                    dtype="float32",
                    samplerate=rate,
                )
                sd.check_output_settings(
                    device=self.output_device,
                    channels=1,
                    dtype="float32",
                    samplerate=rate,
                )
                return rate
            except Exception:
                continue

        raise RuntimeError(
            "Could not find a common sample rate for the configured input/output "
            "devices. Try setting explicit audio.input_device and audio.output_device."
        )

    def start(self) -> None:
        if self._stream is not None:
            return

        self._stream = sd.Stream(
            samplerate=self.stream_rate,
            blocksize=0,
            device=(self.cfg.input_device, self.output_device),
            channels=(self.cfg.channels, 1),
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._detector_thread.start()

    def _callback(self, indata, outdata, frames, _time_info, status) -> None:
        outdata.fill(0)

        write_pos = 0
        while write_pos < frames and not self.stop_event.is_set():
            chunk = self._current_chunk

            if chunk is None:
                try:
                    chunk = self._playback_queue.get_nowait()
                    self._current_chunk = chunk
                    self._current_offset = 0
                except queue.Empty:
                    break

            if (
                chunk.cancel_event.is_set()
                or (chunk.stop_event is not None and chunk.stop_event.is_set())
                or self.detected_event.is_set()
            ):
                chunk.completed = False
                chunk.done_event.set()
                self._current_chunk = None
                self._current_offset = 0
                continue

            available = chunk.samples.size - self._current_offset
            if available <= 0:
                chunk.completed = True
                chunk.done_event.set()
                self._current_chunk = None
                self._current_offset = 0
                continue

            count = min(frames - write_pos, available)
            outdata[write_pos : write_pos + count, 0] = chunk.samples[
                self._current_offset : self._current_offset + count
            ]
            write_pos += count
            self._current_offset += count

            if self._current_offset >= chunk.samples.size:
                chunk.completed = True
                chunk.done_event.set()
                self._current_chunk = None
                self._current_offset = 0

        input_copy = np.asarray(indata, dtype=np.float32).copy()
        output_level = rms(outdata[:, 0])
        now = time.monotonic()

        try:
            self._input_queue.put_nowait((input_copy, output_level, now))
        except queue.Full:
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._input_queue.put_nowait((input_copy, output_level, now))
            except queue.Full:
                pass

    def play_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        stop_event: threading.Event | None = None,
    ) -> bool:
        if self.stop_event.is_set() or self.detected_event.is_set():
            return False

        samples = resample_linear(audio, int(sample_rate), self.stream_rate)
        samples = np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)
        if samples.size == 0:
            return True

        with self._state_lock:
            if self._first_playback_at is None:
                self._first_playback_at = time.monotonic()

        chunk = _PlaybackChunk(samples=samples, stop_event=stop_event)
        self._playback_queue.put(chunk)

        while not chunk.done_event.wait(0.02):
            if self.stop_event.is_set() or self.detected_event.is_set():
                chunk.cancel_event.set()
                return False
            if stop_event is not None and stop_event.is_set():
                chunk.cancel_event.set()
                return False

        return chunk.completed and not self.detected_event.is_set()

    def _cancel_playback(self) -> None:
        if self._current_chunk is not None:
            self._current_chunk.cancel_event.set()

        while True:
            try:
                chunk = self._playback_queue.get_nowait()
            except queue.Empty:
                break
            chunk.cancel_event.set()
            chunk.completed = False
            chunk.done_event.set()

    def _detector_loop(self) -> None:
        sample_rate = self.cfg.sample_rate
        pre_roll_limit = max(1, int(sample_rate * self.cfg.pre_roll_seconds))
        silence_limit = max(1, int(sample_rate * self.cfg.silence_seconds))
        max_capture = max(1, int(sample_rate * self.cfg.max_record_seconds))
        keyword_silence_limit = max(
            1, int(sample_rate * self.keyword_silence_seconds)
        )
        keyword_max_capture = max(
            1, int(sample_rate * self.keyword_max_seconds)
        )

        pre_roll: deque[np.ndarray] = deque()
        pre_roll_samples = 0
        rolling: deque[np.ndarray] = deque()
        rolling_samples = 0

        captured: list[np.ndarray] = []
        captured_samples = 0
        silence_samples = 0
        started = False

        candidate_active = False
        candidate_audio: list[np.ndarray] = []
        candidate_samples = 0
        candidate_silence_samples = 0

        candidate_hits = 0
        vad_speech = False
        echo_floor = self.fallback_threshold
        output_active_until = 0.0

        try:
            while not self.stop_event.is_set():
                try:
                    input_block, output_level, now = self._input_queue.get(timeout=0.10)
                except queue.Empty:
                    continue

                if input_block.ndim == 2:
                    mic = np.mean(input_block, axis=1)
                else:
                    mic = input_block.reshape(-1)

                mic = resample_linear(mic, self.stream_rate, sample_rate)
                level = rms(mic)

                if output_level > 0.001:
                    output_active_until = now + 0.25
                playback_active = now < output_active_until

                with self._state_lock:
                    first_playback_at = self._first_playback_at

                rolling.append(mic)
                rolling_samples += mic.size
                while rolling and rolling_samples > self.recorder.vad.window_samples:
                    removed = rolling.popleft()
                    rolling_samples -= removed.size

                if self.recorder.vad.available and self.recorder.vad.should_check(mic.size):
                    window = np.concatenate(list(rolling)) if rolling else mic
                    vad_speech = self.recorder.vad.speech_recent(window)

                if started:
                    captured.append(mic)
                    captured_samples += mic.size

                    capture_rms_speech = level >= self.fallback_threshold
                    capture_speech = (
                        vad_speech and capture_rms_speech
                        if self.recorder.vad.available
                        else capture_rms_speech
                    )
                    if capture_speech:
                        silence_samples = 0
                    else:
                        silence_samples += mic.size

                    if silence_samples >= silence_limit or captured_samples >= max_capture:
                        break
                    continue

                in_playback_grace = (
                    first_playback_at is not None
                    and now - first_playback_at < self.delay_seconds
                )

                base_gate = self.fallback_threshold * self.rms_multiplier
                gate = (
                    max(base_gate, echo_floor * self.rms_multiplier)
                    if playback_active
                    else base_gate
                )

                rms_speech = level >= gate
                speech_now = (
                    vad_speech and rms_speech
                    if self.recorder.vad.available
                    else rms_speech
                )

                if candidate_active:
                    candidate_audio.append(mic)
                    candidate_samples += mic.size

                    if speech_now:
                        candidate_silence_samples = 0
                    else:
                        candidate_silence_samples += mic.size

                    finished = (
                        candidate_silence_samples >= keyword_silence_limit
                        or candidate_samples >= keyword_max_capture
                    )
                    if not finished:
                        continue

                    candidate = np.concatenate(candidate_audio).reshape(-1)
                    candidate = np.clip(candidate, -1.0, 1.0).astype(np.float32)

                    accepted = None
                    try:
                        accepted = self.keyword_validator(candidate)
                    except Exception as exc:
                        print(f"[barge-in] keyword validator error: {exc}")

                    if accepted:
                        self.recognized_keyword = str(accepted)
                        self.audio = candidate
                        self.detected_event.set()
                        self._cancel_playback()
                        break

                    # False trigger or non-keyword speech: keep TTS playing.
                    candidate_active = False
                    candidate_audio = []
                    candidate_samples = 0
                    candidate_silence_samples = 0
                    candidate_hits = 0
                    pre_roll.clear()
                    pre_roll_samples = 0
                    continue

                pre_roll.append(mic)
                pre_roll_samples += mic.size
                while pre_roll and pre_roll_samples > pre_roll_limit:
                    removed = pre_roll.popleft()
                    pre_roll_samples -= removed.size

                if in_playback_grace:
                    echo_floor = max(echo_floor * 0.96, level)
                    candidate_hits = 0
                    continue

                if speech_now:
                    candidate_hits += 1
                else:
                    candidate_hits = 0
                    if playback_active:
                        echo_floor = max(
                            self.fallback_threshold,
                            echo_floor * 0.985,
                            level * 0.90,
                        )
                    else:
                        echo_floor = max(
                            self.fallback_threshold,
                            echo_floor * 0.97,
                        )

                if candidate_hits < 2:
                    continue

                if self.keyword_validator is not None:
                    # Validate before interrupting.
                    candidate_active = True
                    candidate_audio = list(pre_roll)
                    candidate_samples = sum(block.size for block in candidate_audio)
                    candidate_silence_samples = 0
                    candidate_hits = 0
                    continue

                # Free-form mode retains the previous immediate interruption.
                started = True
                self.detected_event.set()
                self._cancel_playback()
                captured.extend(list(pre_roll))
                captured_samples = sum(block.size for block in captured)
                silence_samples = 0

            if started and captured and self.audio is None:
                captured_array = np.concatenate(captured).reshape(-1)
                self.audio = np.clip(
                    captured_array, -1.0, 1.0
                ).astype(np.float32)
        finally:
            self.capture_done_event.set()

    def wait_audio(self, timeout: float) -> np.ndarray | None:
        self.capture_done_event.wait(timeout=max(0.0, timeout))
        return self.audio

    def stop(self) -> None:
        self.stop_event.set()
        self._cancel_playback()

        if self._detector_thread.is_alive():
            self._detector_thread.join(timeout=1.5)

        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()


def save_wav(path, audio: np.ndarray, sample_rate: int) -> None:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def play_audio_interruptible(
    audio: np.ndarray,
    sample_rate: int,
    output_device=None,
    stop_event: threading.Event | None = None,
) -> bool:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return True

    sd.play(audio, sample_rate, device=output_device, blocking=False)
    duration = len(audio) / float(sample_rate)
    deadline = time.monotonic() + duration + 0.25

    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            sd.stop()
            return False
        try:
            stream = sd.get_stream()
            if not stream.active:
                return True
        except Exception:
            pass
        time.sleep(0.02)

    sd.wait()
    return stop_event is None or not stop_event.is_set()


def stop_audio() -> None:
    sd.stop()
