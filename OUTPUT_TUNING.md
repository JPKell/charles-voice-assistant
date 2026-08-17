# Output tuning patch

This is a small tuning patch for the current split system-default audio engine.

It changes only `voice_chat/audio.py`.

## Changes

```text
preferred output rate:     24 kHz -> 48 kHz
output callback block:     ~60 ms -> ~100 ms
output ring capacity:      4.0 s  -> 1.0 s
startup prebuffer:         stays 0.60 s
```

The 4-second value was the maximum ring-buffer capacity, not a mandatory
four-second playback delay. Reducing it to one second limits how much speech
can be queued ahead.

Kokoro still produces 24 kHz audio. The existing code resamples that audio to
the selected output rate before putting it in the output ring. With this patch,
the system-default output should normally open at 48 kHz.

The startup line now exposes the active tuning:

```text
[audio] split system-default streams; input='...' @ 16000 Hz; output='...' @ 48000 Hz; output_block≈100ms; prebuffer=0.60s; output_buffer=1.0s
```

## Test

Run:

```bash
charles
```

Then use a long response and inspect:

```text
[audio] split stats: output_underflows=..., software_starvations=...
```

The best result is:

```text
output_underflows=0
software_starvations=0
```

If `software_starvations` remains zero but `output_underflows` is still
significant at 48 kHz / ~100 ms, the next useful work is below the Python TTS
layer: PipeWire/Pulse/ALSA output scheduling and device configuration.
