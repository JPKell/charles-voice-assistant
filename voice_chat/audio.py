from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
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
            device=None,
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
            device=None,
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


class FrameRingBuffer:
    """Preallocated single-producer/single-consumer float32 frame ring."""

    def __init__(self, capacity_frames: int, channels: int):
        self.capacity_frames = max(1, int(capacity_frames))
        self.channels = max(1, int(channels))
        self.data = np.zeros(
            (self.capacity_frames, self.channels),
            dtype=np.float32,
        )
        self.read_count = 0
        self.write_count = 0

    @property
    def available_read(self) -> int:
        return max(0, self.write_count - self.read_count)

    @property
    def available_write(self) -> int:
        return max(0, self.capacity_frames - self.available_read)

    def clear(self) -> None:
        self.read_count = self.write_count

    def write(self, frames: np.ndarray) -> int:
        array = np.asarray(frames, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[1] != self.channels:
            raise ValueError(
                f"ring expects shape (frames, {self.channels}), got {array.shape}"
            )

        count = min(array.shape[0], self.available_write)
        if count <= 0:
            return 0

        start = self.write_count % self.capacity_frames
        first = min(count, self.capacity_frames - start)
        self.data[start : start + first] = array[:first]

        second = count - first
        if second:
            self.data[:second] = array[first : first + second]

        self.write_count += count
        return count

    def read_into(self, target: np.ndarray) -> int:
        array = np.asarray(target)
        if array.ndim == 1:
            if self.channels != 1:
                raise ValueError("1-D target only valid for mono ring")
            array2 = array.reshape(-1, 1)
        else:
            array2 = array

        count = min(array2.shape[0], self.available_read)
        if count <= 0:
            return 0

        start = self.read_count % self.capacity_frames
        first = min(count, self.capacity_frames - start)
        array2[:first] = self.data[start : start + first]

        second = count - first
        if second:
            array2[first : first + second] = self.data[:second]

        self.read_count += count
        return count

    def read(self, max_frames: int) -> np.ndarray:
        count = min(max(0, int(max_frames)), self.available_read)
        if count <= 0:
            return np.empty((0, self.channels), dtype=np.float32)

        result = np.empty((count, self.channels), dtype=np.float32)
        self.read_into(result)
        return result


class DuplexBargeInSession:
    """Separate system-default input/output streams with ring buffers.

    The class name is retained for compatibility with the existing barge-in
    monitor, but this is intentionally NOT a PortAudio duplex stream.

    Input and output are opened as independent streams so the default webcam/
    Scarlett input and the default speaker output can run on their own hardware
    clocks instead of being forced into one cross-device duplex clock domain.
    """

    INPUT_RATES = (16000, 48000, 44100, 32000, 24000)
    OUTPUT_RATES = (48000, 44100, 24000, 32000, 16000)
    INPUT_CALLBACK_BLOCK_SECONDS = 0.060
    OUTPUT_CALLBACK_BLOCK_SECONDS = 0.100
    OUTPUT_BUFFER_SECONDS = 1.0
    OUTPUT_PREBUFFER_SECONDS = 0.60
    INPUT_BUFFER_SECONDS = 8.0
    DETECTOR_BATCH_SECONDS = 0.12

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

        # Kept only for API compatibility. This patch deliberately follows the
        # current system default output instead of a configured device.
        self.output_device = None

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

        # Resolve the system defaults independently. There is no requirement
        # that input and output support one common sample rate.
        self.input_rate = self._choose_input_rate()
        self.output_rate = self._choose_output_rate()

        # Some existing code/debugging expects stream_rate to exist. Treat it as
        # the playback rate for backwards compatibility.
        self.stream_rate = self.output_rate

        self.input_block_frames = max(
            128,
            int(round(self.input_rate * self.INPUT_CALLBACK_BLOCK_SECONDS)),
        )
        self.output_block_frames = max(
            128,
            int(round(self.output_rate * self.OUTPUT_CALLBACK_BLOCK_SECONDS)),
        )
        self.output_prebuffer_frames = max(
            self.output_block_frames,
            int(round(self.output_rate * self.OUTPUT_PREBUFFER_SECONDS)),
        )
        self.detector_batch_frames = max(
            self.input_block_frames,
            int(round(self.input_rate * self.DETECTOR_BATCH_SECONDS)),
        )

        self._output_ring = FrameRingBuffer(
            int(round(self.output_rate * self.OUTPUT_BUFFER_SECONDS)),
            1,
        )
        self._input_ring = FrameRingBuffer(
            int(round(self.input_rate * self.INPUT_BUFFER_SECONDS)),
            self.cfg.channels,
        )

        self._input_stream = None
        self._output_stream = None
        self._detector_thread = threading.Thread(
            target=self._detector_loop,
            name="barge-in-detector",
            daemon=True,
        )

        self._state_lock = threading.Lock()
        self._first_playback_at: float | None = None

        self._playback_started = False
        self._playback_expected = False
        self._output_callback_counter = 0
        self._last_output_callback = -1000000

        self._output_underflows = 0
        self._output_status_events = 0
        self._input_overflows = 0
        self._input_underflows = 0
        self._input_status_events = 0

        self._software_starvations = 0
        self._input_ring_drops = 0
        self._input_high_water_frames = 0
        self._output_high_water_frames = 0

    @property
    def playback(self):
        return self

    @staticmethod
    def _default_device_name(kind: str) -> str:
        try:
            info = sd.query_devices(kind=kind)
            if isinstance(info, dict):
                return str(info.get("name", "system default"))
            return str(info)
        except Exception:
            return "system default"

    def _choose_input_rate(self) -> int:
        tried: list[int] = []
        for rate in (self.cfg.sample_rate, *self.INPUT_RATES):
            rate = int(rate)
            if rate in tried:
                continue
            tried.append(rate)
            try:
                sd.check_input_settings(
                    device=None,
                    channels=self.cfg.channels,
                    dtype="float32",
                    samplerate=rate,
                )
                return rate
            except Exception:
                continue

        raise RuntimeError(
            "The current system-default input device could not be opened at "
            "any supported sample rate."
        )

    def _choose_output_rate(self) -> int:
        for rate in self.OUTPUT_RATES:
            try:
                sd.check_output_settings(
                    device=None,
                    channels=1,
                    dtype="float32",
                    samplerate=rate,
                )
                return rate
            except Exception:
                continue

        raise RuntimeError(
            "The current system-default output device could not be opened at "
            "any supported sample rate."
        )

    def _open_input_stream(self):
        common = dict(
            samplerate=self.input_rate,
            channels=self.cfg.channels,
            dtype="float32",
            device=None,
            callback=self._input_callback,
            latency="high",
        )

        try:
            return sd.InputStream(
                blocksize=self.input_block_frames,
                **common,
            )
        except Exception as exc:
            print(
                f"[audio] fixed input block unavailable ({exc}); "
                "using PortAudio blocksize=0"
            )
            return sd.InputStream(
                blocksize=0,
                **common,
            )

    def _open_output_stream(self):
        common = dict(
            samplerate=self.output_rate,
            channels=1,
            dtype="float32",
            device=None,
            callback=self._output_callback,
            latency="high",
        )

        try:
            return sd.OutputStream(
                blocksize=self.output_block_frames,
                **common,
            )
        except Exception as exc:
            print(
                f"[audio] fixed output block unavailable ({exc}); "
                "using PortAudio blocksize=0"
            )
            return sd.OutputStream(
                blocksize=0,
                **common,
            )

    def start(self) -> None:
        if self._input_stream is not None or self._output_stream is not None:
            return

        input_name = self._default_device_name("input")
        output_name = self._default_device_name("output")

        output_stream = None
        input_stream = None
        try:
            # Open and start independently. The devices no longer share one
            # PortAudio stream or one sample clock.
            output_stream = self._open_output_stream()
            input_stream = self._open_input_stream()

            output_stream.start()
            input_stream.start()
        except Exception:
            if input_stream is not None:
                try:
                    input_stream.close()
                except Exception:
                    pass
            if output_stream is not None:
                try:
                    output_stream.close()
                except Exception:
                    pass
            raise

        self._output_stream = output_stream
        self._input_stream = input_stream
        self._detector_thread.start()

        print(
            "[audio] split system-default streams; "
            f"input='{input_name}' @ {self.input_rate} Hz; "
            f"output='{output_name}' @ {self.output_rate} Hz; "
            f"output_block≈{1000.0 * self.output_block_frames / self.output_rate:.0f}ms; "
            f"prebuffer={self.OUTPUT_PREBUFFER_SECONDS:.2f}s; "
            f"output_buffer={self.OUTPUT_BUFFER_SECONDS:.1f}s"
        )

    def _record_input_status(self, status) -> None:
        if not status:
            return
        self._input_status_events += 1
        try:
            if status.input_overflow:
                self._input_overflows += 1
            if status.input_underflow:
                self._input_underflows += 1
        except Exception:
            pass

    def _record_output_status(self, status) -> None:
        if not status:
            return
        self._output_status_events += 1
        try:
            if status.output_underflow:
                self._output_underflows += 1
        except Exception:
            pass

    def _input_callback(self, indata, frames, _time_info, status) -> None:
        # Keep input callback to a raw ring copy only.
        self._record_input_status(status)

        accepted = self._input_ring.write(indata)
        if accepted < frames:
            self._input_ring_drops += frames - accepted

        unread = self._input_ring.available_read
        if unread > self._input_high_water_frames:
            self._input_high_water_frames = unread

    def _output_callback(self, outdata, frames, _time_info, status) -> None:
        # Keep output callback to zero-fill + raw ring copy only.
        self._record_output_status(status)
        self._output_callback_counter += 1
        outdata.fill(0)

        if (
            not self._playback_started
            or self.stop_event.is_set()
            or self.detected_event.is_set()
        ):
            return

        wrote = self._output_ring.read_into(outdata[:, :1])
        if wrote:
            self._last_output_callback = self._output_callback_counter

        if self._playback_expected and wrote < frames:
            self._software_starvations += 1

    def _start_playback_if_ready(self, *, force: bool = False) -> None:
        if self._playback_started:
            return

        buffered = self._output_ring.available_read
        if not force and buffered < self.output_prebuffer_frames:
            return
        if buffered <= 0:
            return

        self._playback_started = True
        with self._state_lock:
            if self._first_playback_at is None:
                self._first_playback_at = time.monotonic()

    def _write_output(
        self,
        samples: np.ndarray,
        stop_event: threading.Event | None,
    ) -> bool:
        array = np.asarray(samples, dtype=np.float32).reshape(-1, 1)
        offset = 0

        while offset < array.shape[0]:
            if (
                self.stop_event.is_set()
                or self.detected_event.is_set()
                or (stop_event is not None and stop_event.is_set())
            ):
                return False

            written = self._output_ring.write(array[offset:])
            if written:
                offset += written
                buffered = self._output_ring.available_read
                if buffered > self._output_high_water_frames:
                    self._output_high_water_frames = buffered
                self._start_playback_if_ready()
                continue

            time.sleep(0.004)

        return True

    def play_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        stop_event: threading.Event | None = None,
    ) -> bool:
        if self.stop_event.is_set() or self.detected_event.is_set():
            return False

        samples = resample_linear(audio, int(sample_rate), self.output_rate)
        samples = np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)
        if samples.size == 0:
            return True

        self._playback_expected = True
        return self._write_output(samples, stop_event)

    def flush(
        self,
        stop_event: threading.Event | None = None,
        timeout: float | None = None,
    ) -> bool:
        self._start_playback_if_ready(force=True)

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)

        while self._output_ring.available_read > 0:
            if (
                self.stop_event.is_set()
                or self.detected_event.is_set()
                or (stop_event is not None and stop_event.is_set())
            ):
                return False

            if deadline is not None and time.monotonic() >= deadline:
                return False

            time.sleep(0.01)

        # Ring empty means the callback has consumed the final frames, but the
        # host/device may still have one or two blocks queued. Give that tail a
        # chance to reach the physical output before the stream is stopped.
        tail_seconds = max(
            0.10,
            2.0 * self.output_block_frames / float(self.output_rate),
        )
        tail_deadline = time.monotonic() + tail_seconds
        while time.monotonic() < tail_deadline:
            if (
                self.stop_event.is_set()
                or self.detected_event.is_set()
                or (stop_event is not None and stop_event.is_set())
            ):
                return False
            time.sleep(0.01)

        self._playback_expected = False
        return not self.detected_event.is_set()

    def _cancel_playback(self) -> None:
        self._playback_expected = False
        self._playback_started = False
        self._output_ring.clear()

    def _playback_recently_active(self) -> bool:
        recent_callbacks = max(
            2,
            int(round(0.30 / self.OUTPUT_CALLBACK_BLOCK_SECONDS)),
        )
        return (
            self._output_callback_counter - self._last_output_callback
            <= recent_callbacks
        )

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

        try:
            while not self.stop_event.is_set():
                available = self._input_ring.available_read
                if available <= 0:
                    time.sleep(0.008)
                    continue

                input_block = self._input_ring.read(
                    min(available, self.detector_batch_frames)
                )
                if input_block.size == 0:
                    time.sleep(0.004)
                    continue

                if input_block.ndim == 2:
                    mic = np.mean(input_block, axis=1)
                else:
                    mic = input_block.reshape(-1)

                # Default input may run at a different rate than Whisper/VAD.
                # Resample only in this worker thread, never in the callback.
                mic = resample_linear(mic, self.input_rate, sample_rate)
                level = rms(mic)

                playback_active = self._playback_recently_active()

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
                    and time.monotonic() - first_playback_at < self.delay_seconds
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
                    candidate_active = True
                    candidate_audio = list(pre_roll)
                    candidate_samples = sum(block.size for block in candidate_audio)
                    candidate_silence_samples = 0
                    candidate_hits = 0
                    continue

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

    def _print_audio_stats(self) -> None:
        pass
        # input_high_ms = (
        #     1000.0 * self._input_high_water_frames / float(self.input_rate)
        # )
        # output_high_ms = (
        #     1000.0 * self._output_high_water_frames / float(self.output_rate)
        # )

        # print(
        #     "[audio] split stats: "
        #     f"output_underflows={self._output_underflows}, "
        #     f"software_starvations={self._software_starvations}, "
        #     f"input_ring_drops={self._input_ring_drops}, "
        #     f"input_high_water_ms={input_high_ms:.0f}, "
        #     f"output_high_water_ms={output_high_ms:.0f}, "
        #     f"input_overflows={self._input_overflows}, "
        #     f"input_underflows={self._input_underflows}, "
        #     f"output_status_events={self._output_status_events}, "
        #     f"input_status_events={self._input_status_events}"
        # )

    def stop(self) -> None:
        self.stop_event.set()
        self._cancel_playback()

        if self._detector_thread.is_alive():
            self._detector_thread.join(timeout=1.5)

        input_stream = self._input_stream
        output_stream = self._output_stream
        self._input_stream = None
        self._output_stream = None

        if input_stream is not None:
            try:
                input_stream.stop()
            finally:
                input_stream.close()

        if output_stream is not None:
            try:
                output_stream.stop()
            finally:
                output_stream.close()

        self._print_audio_stats()


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

    sd.play(audio, sample_rate, device=None, blocking=False)
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
