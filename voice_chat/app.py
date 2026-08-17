from __future__ import annotations

import argparse
from dataclasses import replace
import os
import re
from pathlib import Path
import sys
import threading
import time

import requests

from .config import ROOT, load_config
from .memory import MemoryStore
from .persona import load_persona
from .wake import WakeWordGate
from .tooling import ToolCallingChat, load_tool_settings


EXIT_PHRASES = {
    "goodbye",
    "goodbye.",
    "quit",
    "quit.",
    "exit",
    "exit.",
    "stop listening",
    "stop listening.",
}


# The CLI names are fixed, but the actual Kokoro voices come from config.toml.
VOICE_PRESET_NAMES = ("sexy", "female", "male", "fun", "rabbi")


def current_voice_preset(args) -> str:
    for name in VOICE_PRESET_NAMES:
        if getattr(args, name, False):
            return name
    return "default"


def reset_context_if_preset_changed(root: Path, llm, args) -> None:
    from datetime import datetime

    data_dir = root / "data"
    state_path = data_dir / "last_voice_preset.txt"
    log_path = data_dir / "voice_preset.log"
    selected = current_voice_preset(args)

    previous = None
    try:
        previous = state_path.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        pass

    changed = previous is not None and previous != selected
    if changed:
        print(f"[preset] changed: {previous} -> {selected}; clearing context")
        llm.clear_history(persistent=True)

    data_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(selected + "\n", encoding="utf-8")

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    previous_label = previous if previous is not None else "(none)"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"{timestamp} preset={selected} previous={previous_label} "
            f"changed={'yes' if changed else 'no'}\n"
        )

    print(f"[preset] active: {selected}")





def apply_voice_preset(cfg, args):
    selected = next(
        (name for name in VOICE_PRESET_NAMES if getattr(args, name, False)),
        None,
    )
    if selected is None:
        return cfg

    preset = cfg.voice_presets.get(selected)
    if preset is None:
        print(
            f"ERROR: --{selected} was requested, but "
            f"[voice_presets.{selected}] is not configured in config.toml.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    updates = {"voice": preset.voice}

    if preset.lang_code is not None:
        updates["lang_code"] = preset.lang_code

    if preset.speed is not None:
        updates["speed"] = preset.speed

    cfg.tts = replace(cfg.tts, **updates)

    system_prompt = preset.system_prompt
    if preset.system_prompt_file:
        prompt_path = Path(preset.system_prompt_file).expanduser()
        if not prompt_path.is_absolute():
            prompt_path = cfg.root / prompt_path
        try:
            system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            print(
                f"ERROR: could not read system prompt file for --{selected}: "
                f"{prompt_path}: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        if not system_prompt:
            print(
                f"ERROR: system prompt file for --{selected} is empty: {prompt_path}",
                file=sys.stderr,
            )
            raise SystemExit(2)

    if system_prompt:
        cfg.app = replace(cfg.app, system_prompt=system_prompt)

    details = [cfg.tts.voice, f"lang={cfg.tts.lang_code}", f"speed={cfg.tts.speed:g}"]
    if preset.system_prompt_file:
        details.append(f"system prompt={preset.system_prompt_file}")
    elif preset.system_prompt:
        details.append("custom system prompt")
    print(f"Voice preset: --{selected} -> " + ", ".join(details))
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local microphone → Ollama → Kokoro voice chat")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.toml",
        help="Path to config.toml",
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--text", type=str, help="Skip microphone/STT and send text to the LLM")
    parser.add_argument(
        "--tts-only",
        type=str,
        help="Speak this text directly with Kokoro; bypass Ollama and Whisper",
    )
    parser.add_argument("--no-tts", action="store_true", help="Print responses without speaking")
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Set Ollama keep_alive=0 so the model unloads after the response",
    )
    parser.add_argument(
        "--online-hf",
        action="store_true",
        help="Allow Hugging Face network access for downloading a new/cold model or voice",
    )
    parser.add_argument("--persona", type=str, help="Use prompts/NAME.txt for this run")
    parser.add_argument("--no-memory", action="store_true", help="Do not load or save SQLite conversation memory")
    parser.add_argument("--clear-memory", action="store_true", help="Clear persistent conversation memory and exit")
    parser.add_argument(
        "--wake-word",
        type=str,
        help="Enable transcript-gated wake phrase for this run (for example: computer)",
    )
    parser.add_argument("--no-wake-word", action="store_true", help="Disable configured wake-word mode for this run")
    parser.add_argument("--no-barge-in", action="store_true", help="Disable interruption while the assistant responds")
    parser.add_argument("--no-stream-tts", action="store_true", help="Wait for the full answer before speaking")
    parser.add_argument("--no-tools", action="store_true", help="Disable Ollama tool calling for this run")

    voice_group = parser.add_mutually_exclusive_group()
    voice_group.add_argument(
        "--sexy",
        action="store_true",
        help="Use [voice_presets.sexy] from config.toml for this run",
    )
    voice_group.add_argument(
        "--female",
        action="store_true",
        help="Use [voice_presets.female] from config.toml for this run",
    )
    voice_group.add_argument(
        "--male",
        action="store_true",
        help="Use [voice_presets.male] from config.toml for this run",
    )
    voice_group.add_argument(
        "--fun",
        action="store_true",
        help="Use [voice_presets.fun] from config.toml for this run",
    )
    voice_group.add_argument(
        "--rabbi",
        action="store_true",
        help="Use [voice_presets.rabbi] from config.toml for this run",
    )
    return parser.parse_args()


def configure_huggingface(offline_after_cache: bool, online_override: bool) -> None:
    if online_override:
        os.environ.pop("HF_HUB_OFFLINE", None)
        print("Hugging Face: online mode enabled for this run.")
    elif offline_after_cache:
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("Hugging Face: offline cache mode (use --online-hf for new models/voices).")


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def normalize_barge_in_phrase(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


class BargeInMonitor:
    # Full-duplex barge-in with optional exact keyword validation.

    def __init__(
        self,
        recorder,
        threshold: float,
        delay_seconds: float,
        rms_multiplier: float,
        output_device,
        *,
        stt=None,
        keyword_only: bool = True,
        keywords=(),
        keyword_silence_seconds: float = 0.35,
        keyword_max_seconds: float = 2.0,
    ):
        from .audio import DuplexBargeInSession

        self.keyword_only = bool(keyword_only and stt is not None)
        self.keywords = {
            normalize_barge_in_phrase(str(keyword)): str(keyword)
            for keyword in keywords
            if normalize_barge_in_phrase(str(keyword))
        }
        self.stt = stt

        validator = self._validate_keyword if self.keyword_only else None

        self.session = DuplexBargeInSession(
            recorder,
            threshold,
            output_device=output_device,
            delay_seconds=delay_seconds,
            rms_multiplier=rms_multiplier,
            keyword_validator=validator,
            keyword_silence_seconds=keyword_silence_seconds,
            keyword_max_seconds=keyword_max_seconds,
        )
        self.detected_event = self.session.detected_event

    def _validate_keyword(self, audio):
        transcript = self.stt.transcribe(audio).strip()
        normalized = normalize_barge_in_phrase(transcript)
        if normalized in self.keywords:
            keyword = self.keywords[normalized]
            print(f"\n[barge-in keyword: {keyword}]")
            return keyword
        return None

    @property
    def playback(self):
        return self.session.playback

    @property
    def recognized_keyword(self):
        return self.session.recognized_keyword

    def start(self) -> None:
        self.session.start()

    def stop(self) -> None:
        self.session.stop()

    def wait_audio(self, timeout: float) -> object:
        audio = self.session.wait_audio(timeout)
        # Keyword phrases are local commands, not new prompts for Ollama.
        return None if self.keyword_only else audio


def make_memory(cfg, disabled: bool) -> MemoryStore | None:
    if disabled or not cfg.memory.enabled:
        return None
    return MemoryStore(resolve_path(cfg.root, cfg.memory.database_path))


def response_with_streaming_tts(
    llm,
    user_text: str,
    *,
    tts,
    worker,
    recorder,
    threshold: float | None,
    features,
    stt=None,
) -> tuple[str, object | None]:
    from .streaming import SentenceAccumulator

    accumulator = SentenceAccumulator(
        min_chars=features.sentence_min_chars,
        max_chars=features.sentence_max_chars,
        sentences_per_chunk=features.sentences_per_tts_chunk,
    )
    monitor = None
    if features.barge_in and recorder is not None and threshold is not None and tts is not None:
        try:
            monitor = BargeInMonitor(
                recorder,
                threshold,
                delay_seconds=features.barge_in_delay_seconds,
                rms_multiplier=features.barge_in_rms_multiplier,
                output_device=tts.output_device,
                stt=stt,
                keyword_only=features.barge_in_keyword_only,
                keywords=features.barge_in_keywords,
                keyword_silence_seconds=features.barge_in_keyword_silence_seconds,
                keyword_max_seconds=features.barge_in_keyword_max_seconds,
            )
            monitor.start()
        except Exception as exc:
            print(
                f"[barge-in] duplex audio unavailable for this response: {exc}",
                file=sys.stderr,
            )
            print("[barge-in] continuing without interruption support.", file=sys.stderr)
            monitor = None

    session = (
        worker.new_session(playback=monitor.playback if monitor is not None else None)
        if worker is not None
        else None
    )

    print("\nAssistant: ", end="", flush=True)
    parts: list[str] = []
    stream = llm.chat_stream(user_text)
    interrupted = False

    try:
        for chunk in stream:
            if monitor is not None and monitor.detected_event.is_set():
                interrupted = True
                break

            parts.append(chunk)
            print(chunk, end="", flush=True)

            if worker is not None and session is not None:
                for sentence in accumulator.feed(chunk):
                    worker.submit(session, sentence)
    finally:
        if interrupted and hasattr(stream, "close"):
            stream.close()

    answer = "".join(parts).strip()

    if interrupted:
        print("\n[barge-in: response interrupted]")
        if worker is not None and session is not None:
            worker.cancel(session)
        if monitor is not None:
            audio = monitor.wait_audio(timeout=recorder.cfg.max_record_seconds + 3.0)
            monitor.stop()
            return answer, audio
        return answer, None

    print()

    if worker is not None and session is not None:
        for sentence in accumulator.finish():
            worker.submit(session, sentence)
        session.close()

        # Keep the barge-in monitor alive until the queued speech is finished.
        while not session.done_event.wait(0.04):
            if monitor is not None and monitor.detected_event.is_set():
                worker.cancel(session)
                audio = monitor.wait_audio(timeout=recorder.cfg.max_record_seconds + 3.0)
                monitor.stop()
                print("[barge-in: speech interrupted]")
                return answer, audio
    elif tts is not None and answer:
        stop_event = monitor.detected_event if monitor is not None else None
        tts.speak(
            answer,
            stop_event=stop_event,
            playback=monitor.playback if monitor is not None else None,
        )
        if monitor is not None and monitor.detected_event.is_set():
            audio = monitor.wait_audio(timeout=recorder.cfg.max_record_seconds + 3.0)
            monitor.stop()
            print("[barge-in: speech interrupted]")
            return answer, audio

    if monitor is not None:
        monitor.stop()
        monitor.wait_audio(timeout=1.0)

    return answer, None


def main() -> int:
    args = parse_args()

    if args.list_devices:
        from .audio import list_devices

        list_devices()
        return 0

    cfg = load_config(args.config)
    cfg = apply_voice_preset(cfg, args)
    configure_huggingface(cfg.huggingface.offline_after_cache, args.online_hf)

    if args.low_vram:
        cfg.ollama = replace(cfg.ollama, keep_alive="0")
    if args.no_barge_in:
        cfg.features = replace(cfg.features, barge_in=False)
    if args.no_stream_tts:
        cfg.features = replace(cfg.features, stream_tts=False)
    if args.wake_word:
        cfg.wake_word = replace(cfg.wake_word, enabled=True, phrase=args.wake_word)
    if args.no_wake_word:
        cfg.wake_word = replace(cfg.wake_word, enabled=False)

    persona_name = args.persona if args.persona is not None else cfg.persona.name
    system_prompt, persona_label = load_persona(
        cfg.root,
        cfg.persona.directory,
        persona_name,
        cfg.app.system_prompt,
    )

    memory_store = make_memory(cfg, args.no_memory)
    if args.clear_memory:
        store = memory_store or MemoryStore(resolve_path(cfg.root, cfg.memory.database_path))
        store.clear()
        print(f"Cleared memory: {store.path}")
        return 0

    # TTS-only deliberately happens before any Ollama or Whisper initialization.
    if args.tts_only is not None:
        try:
            from .tts import TextToSpeech

            tts = TextToSpeech(cfg.tts, output_device=cfg.audio.output_device)
            tts.speak(args.tts_only)
            return 0
        except Exception as exc:
            print(f"TTS error: {type(exc).__name__}: {exc}", file=sys.stderr)
            if cfg.huggingface.offline_after_cache and not args.online_hf:
                print("If this voice/model is not cached yet, rerun with --online-hf.", file=sys.stderr)
            return 3

    from .llm import OllamaChat

    llm = OllamaChat(
        cfg.ollama,
        system_prompt=system_prompt,
        max_history_turns=cfg.app.max_history_turns,
        memory_store=memory_store,
    )
    tool_settings = load_tool_settings(
        args.config,
        disabled=args.no_tools,
    )
    if tool_settings.enabled:
        llm = ToolCallingChat(llm, tool_settings)
    reset_context_if_preset_changed(cfg.root, llm, args)

    try:
        llm.healthcheck()
    except (requests.RequestException, OSError) as exc:
        print(
            f"Cannot reach Ollama at {cfg.ollama.base_url}: {exc}\n"
            "Check: systemctl status ollama",
            file=sys.stderr,
        )
        return 2

    tts = None
    worker = None
    if not args.no_tts:
        try:
            from .tts import TextToSpeech

            tts = TextToSpeech(cfg.tts, output_device=cfg.audio.output_device)
            if cfg.features.stream_tts:
                from .streaming import SpeechWorker

                worker = SpeechWorker(tts)
        except Exception as exc:
            print(f"TTS initialization error: {type(exc).__name__}: {exc}", file=sys.stderr)
            if cfg.huggingface.offline_after_cache and not args.online_hf:
                print("If this voice/model is not cached yet, rerun with --online-hf.", file=sys.stderr)
            return 3

    print(
        f"Persona: {persona_label}; memory: {'on' if memory_store else 'off'}; "
        f"Ollama keep_alive: {cfg.ollama.keep_alive}."
    )

    if args.text is not None:
        try:
            response_with_streaming_tts(
                llm,
                args.text,
                tts=tts,
                worker=worker,
                recorder=None,
                threshold=None,
                features=cfg.features,
            )
            return 0
        finally:
            if worker is not None:
                worker.close()

    try:
        from .audio import VoiceRecorder
        from .stt import SpeechToText

        recorder = VoiceRecorder(cfg.audio, cfg.stt)
        stt = SpeechToText(cfg.stt, sample_rate=cfg.audio.sample_rate)
        threshold = recorder.calibrate()
    except Exception as exc:
        print(f"Audio/STT initialization error: {type(exc).__name__}: {exc}", file=sys.stderr)
        if cfg.huggingface.offline_after_cache and not args.online_hf:
            print("If the Whisper model is not cached yet, rerun with --online-hf.", file=sys.stderr)
        if worker is not None:
            worker.close()
        return 4

    wake_gate = WakeWordGate(cfg.wake_word)
    wake_status = f"wake phrase '{cfg.wake_word.phrase}'" if wake_gate.enabled else "continuous conversation"

    barge_status = (
        "barge-in keywords: " + ", ".join(cfg.features.barge_in_keywords)
        if cfg.features.barge_in and cfg.features.barge_in_keyword_only
        else ("free-form barge-in" if cfg.features.barge_in else "barge-in off")
    )

    print(
        f"\nReady. LLM: {cfg.ollama.model}; Kokoro: {cfg.tts.voice}; {wake_status}; "
        f"{barge_status}. Say 'goodbye' to exit."
    )

    pending_audio = None
    try:
        while True:
            try:
                if pending_audio is not None:
                    audio = pending_audio
                    pending_audio = None
                else:
                    audio = recorder.record_utterance(threshold)

                if audio is None:
                    print("No speech heard. Still listening...")
                    continue

                print("Transcribing...")
                user_text = stt.transcribe(audio)
                if not user_text:
                    print("I didn't catch that.")
                    continue

                print(f"\nYou: {user_text}")

                if user_text.strip().lower() in EXIT_PHRASES:
                    print("Assistant: Goodbye.")
                    if tts:
                        tts.speak("Goodbye.")
                    return 0

                accepted, command = wake_gate.process(user_text)
                if not accepted:
                    print(f"[wake] waiting for '{cfg.wake_word.phrase}'")
                    continue
                if not command:
                    print("[wake] activated; listening for your command...")
                    continue

                normalized_command = command.strip().lower().rstrip(" .!?")

                if normalized_command in {
                    "clear context",
                    "reset context",
                    "clear conversation",
                    "reset conversation",
                }:
                    llm.clear_history(persistent=False)
                    confirmation = "Context cleared."
                    print(f"Assistant: {confirmation}")
                    if tts is not None:
                        tts.speak(confirmation)
                    continue

                if normalized_command in {
                    "clear memory",
                    "clear conversation memory",
                    "forget conversation history",
                }:
                    llm.clear_history(persistent=True)
                    confirmation = "Conversation memory cleared."
                    print(f"Assistant: {confirmation}")
                    if tts is not None:
                        tts.speak(confirmation)
                    continue

                _answer, barge_audio = response_with_streaming_tts(
                    llm,
                    command,
                    tts=tts,
                    worker=worker,
                    recorder=recorder,
                    threshold=threshold,
                    features=cfg.features,
                    stt=stt,
                )
                if barge_audio is not None:
                    pending_audio = barge_audio
                elif tts is not None:
                    time.sleep(cfg.audio.post_tts_pause_seconds)

            except KeyboardInterrupt:
                print("\nStopped.")
                return 0
            except requests.HTTPError as exc:
                body = ""
                if exc.response is not None:
                    body = exc.response.text[:500]
                print(f"\nOllama HTTP error: {exc}\n{body}", file=sys.stderr)
                time.sleep(1)
            except Exception as exc:
                print(f"\nError: {type(exc).__name__}: {exc}", file=sys.stderr)
                print("Continuing; press Ctrl+C to exit.", file=sys.stderr)
                time.sleep(1)
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
