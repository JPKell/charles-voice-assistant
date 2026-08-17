# Separate system-default input/output audio patch

This patch is designed for the remaining artifacts after the ring-buffer test.

The previous diagnostic showed that the Python ring itself was staying fed, but
PortAudio was still reporting many output underflows. The remaining suspect is
the cross-device full-duplex stream: a microphone device and speaker device can
have independent hardware clocks.

This patch keeps the ring buffers but stops asking PortAudio to treat input and
output as one duplex device.

## New layout

```text
CURRENT SYSTEM DEFAULT INPUT
(webcam mic OR Scarlett)
          |
          v
 independent InputStream
          |
          v
      input ring
          |
          v
  VAD / keyword detector


CURRENT SYSTEM DEFAULT OUTPUT
          ^
          |
 independent OutputStream
          ^
          |
      output ring
          ^
          |
        Kokoro
```

Input and output no longer need a common sample rate and no longer share a
PortAudio clock domain.

## System defaults

The patch intentionally ignores configured `audio.input_device` and
`audio.output_device` values.

Every ordinary microphone capture uses:

```python
device=None
```

Every non-barge-in / TTS-only playback uses the system default output.

Each barge-in response opens fresh independent `InputStream` and `OutputStream`
objects with `device=None`, so changing the Ubuntu/PipeWire default input or
output between responses is picked up without editing `config.toml`.

For example:

1. webcam mic is the Ubuntu default input
2. speak to the assistant
3. switch Ubuntu default input to Scarlett
4. ask the next question
5. the newly opened input stream follows Scarlett

The same applies to the system-default output.

## Sample rates

Input and output are chosen independently.

Input prefers the voice-recorder rate (normally 16 kHz), then common fallback
rates.

Output prefers Kokoro's native 24 kHz, then common fallback rates.

If the input rate differs from the Whisper/VAD rate, microphone resampling is
done in the detector worker, never inside the real-time callback.

## Diagnostics

Startup should now contain a line similar to:

```text
[audio] split system-default streams; input='...' @ 16000 Hz; output='...' @ 24000 Hz; output_prebuffer=0.60s
```

After a response:

```text
[audio] split stats: output_underflows=0, software_starvations=0, ...
```

The key field is still:

```text
output_underflows
```

If separating the hardware clocks was the cause, that should fall to zero or
very close to zero and the artifacts should disappear.

## Files changed

Only:

```text
voice_chat/audio.py
```

No `config.toml` changes are made.
