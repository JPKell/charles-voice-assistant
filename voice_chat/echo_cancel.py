from __future__ import annotations

import atexit
import shutil
import subprocess
import time

from .terminal import print_dim


ECHO_SOURCE_NAME = "voice_chat_echo_source"
ECHO_SINK_NAME = "voice_chat_echo_sink"


class PipeWireEchoCancel:
    """Own a temporary PipeWire WebRTC echo-cancel source and sink."""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.input_device: str | None = None
        self.output_device: str | None = None
        self._registered = False

    @staticmethod
    def _module_args() -> str:
        # WebRTC AEC operates on one channel. Label every side MONO explicitly
        # so PipeWire channel-mixes playback into both FL and FR speakers.
        return (
            "{ audio.rate = 48000 audio.channels = 1 audio.position = [ MONO ] "
            "library.name = aec/libspa-aec-webrtc "
            "capture.props = { audio.position = [ MONO ] } "
            "source.props = { node.name = \"voice_chat_echo_source\" "
            "node.description = \"Voice Chat Echo-Cancelled Microphone\" "
            "audio.position = [ MONO ] } "
            "sink.props = { node.name = \"voice_chat_echo_sink\" "
            "node.description = \"Voice Chat Echo-Cancelled Output\" "
            "audio.position = [ MONO ] } "
            "playback.props = { audio.position = [ MONO ] } }"
        )

    @staticmethod
    def _find_devices() -> tuple[str | None, str | None]:
        import sounddevice as sd

        input_device = None
        output_device = None
        for device in sd.query_devices():
            name = str(device.get("name", ""))
            if ECHO_SOURCE_NAME in name and int(device.get("max_input_channels", 0)):
                input_device = name
            if ECHO_SINK_NAME in name and int(device.get("max_output_channels", 0)):
                output_device = name
        return input_device, output_device

    def start(self, timeout: float = 4.0) -> bool:
        executable = shutil.which("pw-cli")
        if executable is None:
            print_dim("[audio] PipeWire echo cancellation unavailable: pw-cli not found")
            return False

        try:
            self.process = subprocess.Popen(
                [
                    executable,
                    "-m",
                    "load-module",
                    "libpipewire-module-echo-cancel",
                    self._module_args(),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            print_dim(f"[audio] could not start PipeWire echo cancellation: {exc}")
            return False

        if not self._registered:
            atexit.register(self.stop)
            self._registered = True

        deadline = time.monotonic() + max(0.1, timeout)
        while time.monotonic() < deadline:
            self.input_device, self.output_device = self._find_devices()
            if self.input_device and self.output_device:
                print_dim(
                    "[audio] PipeWire WebRTC echo cancellation active; "
                    f"input='{self.input_device}'; output='{self.output_device}'"
                )
                return True
            if self.process.poll() is not None:
                break
            time.sleep(0.10)

        print_dim("[audio] PipeWire echo-cancel devices did not become available")
        self.stop()
        return False

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=1.5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)
