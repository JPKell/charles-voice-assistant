from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AppConfig:
    system_prompt: str
    max_history_turns: int


@dataclass
class OllamaConfig:
    base_url: str
    model: str
    think: bool
    temperature: float
    num_ctx: int
    keep_alive: str
    request_timeout_seconds: int


@dataclass
class STTConfig:
    model: str
    device: str
    compute_type: str
    language: str
    cpu_threads: int
    beam_size: int
    vad_filter: bool = True
    vad_threshold: float = 0.50
    vad_min_speech_ms: int = 80
    vad_min_silence_ms: int = 300


@dataclass
class TTSConfig:
    lang_code: str
    voice: str
    speed: float
    device: str
    volume: float
    periods_to_commas: bool = True
    paragraphs_to_periods: bool = True
    strip_characters: str = "#*"


@dataclass
class VoicePresetConfig:
    voice: str
    lang_code: str | None = None
    speed: float | None = None
    system_prompt: str | None = None
    system_prompt_file: str | None = None


@dataclass
class AudioConfig:
    sample_rate: int
    channels: int
    block_ms: int
    calibration_seconds: float
    noise_multiplier: float
    min_rms_threshold: float
    silence_seconds: float
    start_timeout_seconds: float
    max_record_seconds: float
    pre_roll_seconds: float
    post_tts_pause_seconds: float
    input_device: str | int | None
    output_device: str | int | None
    vad_window_ms: int = 480
    vad_check_ms: int = 120


@dataclass
class FeaturesConfig:
    stream_tts: bool = True
    barge_in: bool = True
    barge_in_delay_seconds: float = 0.55
    barge_in_rms_multiplier: float = 2.0
    barge_in_keyword_only: bool = True
    barge_in_keywords: tuple[str, ...] = ("thank you", "okay", "stop", "wait")
    barge_in_keyword_silence_seconds: float = 0.35
    barge_in_keyword_max_seconds: float = 2.0
    sentence_min_chars: int = 8
    sentence_max_chars: int = 260
    sentences_per_tts_chunk: int = 2


@dataclass
class MemoryConfig:
    enabled: bool = False
    database_path: str = "data/memory.sqlite3"


@dataclass
class PersonaConfig:
    name: str = ""
    directory: str = "prompts"


@dataclass
class WakeWordConfig:
    enabled: bool = False
    phrase: str = "computer"
    followup_seconds: float = 20.0
    require_prefix: bool = True


@dataclass
class HuggingFaceConfig:
    offline_after_cache: bool = True


@dataclass
class Config:
    app: AppConfig
    ollama: OllamaConfig
    stt: STTConfig
    tts: TTSConfig
    voice_presets: dict[str, VoicePresetConfig]
    audio: AudioConfig
    features: FeaturesConfig
    memory: MemoryConfig
    persona: PersonaConfig
    wake_word: WakeWordConfig
    huggingface: HuggingFaceConfig
    root: Path


def _audio_device(value: object) -> str | int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def _b(section: dict, key: str, default: bool) -> bool:
    value = section.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def load_config(path: Path | None = None) -> Config:
    path = (path or ROOT / "config.toml").resolve()
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    app = raw.get("app", {})
    ollama = raw.get("ollama", {})
    stt = raw.get("stt", {})
    tts = raw.get("tts", {})
    voice_presets_raw = raw.get("voice_presets", {})
    audio = raw.get("audio", {})

    voice_presets: dict[str, VoicePresetConfig] = {}
    if isinstance(voice_presets_raw, dict):
        for name, section in voice_presets_raw.items():
            if not isinstance(section, dict):
                continue

            voice = str(section.get("voice", "")).strip()
            if not voice:
                continue

            lang_value = section.get("lang_code")
            speed_value = section.get("speed")
            system_prompt_value = section.get("system_prompt")
            system_prompt_file_value = section.get("system_prompt_file")

            voice_presets[str(name)] = VoicePresetConfig(
                voice=voice,
                lang_code=None if lang_value in (None, "") else str(lang_value),
                speed=None if speed_value is None else float(speed_value),
                system_prompt=(
                    None
                    if system_prompt_value in (None, "")
                    else str(system_prompt_value).strip()
                ),
                system_prompt_file=(
                    None
                    if system_prompt_file_value in (None, "")
                    else str(system_prompt_file_value).strip()
                ),
            )
    features = raw.get("features", {})
    memory = raw.get("memory", {})
    persona = raw.get("persona", {})
    wake = raw.get("wake_word", {})
    hf = raw.get("huggingface", {})

    return Config(
        app=AppConfig(
            system_prompt=str(app.get("system_prompt", "You are a useful local voice assistant.")),
            max_history_turns=int(app.get("max_history_turns", 10)),
        ),
        ollama=OllamaConfig(
            base_url=str(ollama.get("base_url", "http://127.0.0.1:11434")),
            model=str(ollama.get("model", "qwen3:8b")),
            think=_b(ollama, "think", False),
            temperature=float(ollama.get("temperature", 0.7)),
            num_ctx=int(ollama.get("num_ctx", 8192)),
            keep_alive=str(ollama.get("keep_alive", "1m")),
            request_timeout_seconds=int(ollama.get("request_timeout_seconds", 300)),
        ),
        stt=STTConfig(
            model=str(stt.get("model", "small.en")),
            device=str(stt.get("device", "cpu")),
            compute_type=str(stt.get("compute_type", "int8")),
            language=str(stt.get("language", "en")),
            cpu_threads=int(stt.get("cpu_threads", 8)),
            beam_size=int(stt.get("beam_size", 1)),
            vad_filter=_b(stt, "vad_filter", True),
            vad_threshold=float(stt.get("vad_threshold", 0.50)),
            vad_min_speech_ms=int(stt.get("vad_min_speech_ms", 80)),
            vad_min_silence_ms=int(stt.get("vad_min_silence_ms", 300)),
        ),
        tts=TTSConfig(
            lang_code=str(tts.get("lang_code", "a")),
            voice=str(tts.get("voice", "af_heart")),
            speed=float(tts.get("speed", 1.0)),
            device=str(tts.get("device", "cpu")),
            volume=float(tts.get("volume", 1.0)),
            periods_to_commas=_b(tts, "periods_to_commas", True),
            paragraphs_to_periods=_b(tts, "paragraphs_to_periods", True),
            strip_characters=str(tts.get("strip_characters", "#*")),
        ),
        voice_presets=voice_presets,
        audio=AudioConfig(
            sample_rate=int(audio.get("sample_rate", 16000)),
            channels=int(audio.get("channels", 1)),
            block_ms=int(audio.get("block_ms", 30)),
            calibration_seconds=float(audio.get("calibration_seconds", 1.0)),
            noise_multiplier=float(audio.get("noise_multiplier", 3.0)),
            min_rms_threshold=float(audio.get("min_rms_threshold", 0.008)),
            silence_seconds=float(audio.get("silence_seconds", 0.9)),
            start_timeout_seconds=float(audio.get("start_timeout_seconds", 60.0)),
            max_record_seconds=float(audio.get("max_record_seconds", 45.0)),
            pre_roll_seconds=float(audio.get("pre_roll_seconds", 0.25)),
            post_tts_pause_seconds=float(audio.get("post_tts_pause_seconds", 0.20)),
            input_device=_audio_device(audio.get("input_device")),
            output_device=_audio_device(audio.get("output_device")),
            vad_window_ms=int(audio.get("vad_window_ms", 480)),
            vad_check_ms=int(audio.get("vad_check_ms", 120)),
        ),
        features=FeaturesConfig(
            stream_tts=_b(features, "stream_tts", True),
            barge_in=_b(features, "barge_in", True),
            barge_in_delay_seconds=float(features.get("barge_in_delay_seconds", 0.55)),
            barge_in_rms_multiplier=float(features.get("barge_in_rms_multiplier", 2.0)),
            barge_in_keyword_only=_b(features, "barge_in_keyword_only", True),
            barge_in_keywords=tuple(
                str(value).strip().lower()
                for value in features.get(
                    "barge_in_keywords",
                    ["thank you", "okay", "stop", "wait"],
                )
                if str(value).strip()
            ),
            barge_in_keyword_silence_seconds=float(
                features.get("barge_in_keyword_silence_seconds", 0.35)
            ),
            barge_in_keyword_max_seconds=float(
                features.get("barge_in_keyword_max_seconds", 2.0)
            ),
            sentence_min_chars=int(features.get("sentence_min_chars", 8)),
            sentence_max_chars=int(features.get("sentence_max_chars", 260)),
            sentences_per_tts_chunk=int(features.get("sentences_per_tts_chunk", 2)),
        ),
        memory=MemoryConfig(
            enabled=_b(memory, "enabled", False),
            database_path=str(memory.get("database_path", "data/memory.sqlite3")),
        ),
        persona=PersonaConfig(
            name=str(persona.get("name", "")),
            directory=str(persona.get("directory", "prompts")),
        ),
        wake_word=WakeWordConfig(
            enabled=_b(wake, "enabled", False),
            phrase=str(wake.get("phrase", "computer")),
            followup_seconds=float(wake.get("followup_seconds", 20.0)),
            require_prefix=_b(wake, "require_prefix", True),
        ),
        huggingface=HuggingFaceConfig(
            offline_after_cache=_b(hf, "offline_after_cache", True),
        ),
        root=path.parent,
    )
