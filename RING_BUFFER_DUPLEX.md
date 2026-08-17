# Ring-buffered duplex barge-in patch

This patch supersedes the previous `duplex_audio_quality` patch.

The diagnostic run showed:

```text
output_underflows=36
input_queue_drops=575
```

That points to the real-time duplex callback missing deadlines, not to Kokoro
synthesis itself.

## What is reverted

This patch removes/replaces the previous attempt's:

- `StreamingLinearResampler`
- `_PlaybackChunk` / Python queue based duplex callback
- per-fragment producer lookahead events
- per-`TextToSpeech.speak()` flush

The new implementation is installed in their place instead of adding another
layer on top.

## New architecture

```text
Kokoro / TTS worker
        |
        v
preallocated OUTPUT RING (4 seconds)
        |
        v
PortAudio callback  <---->  preallocated INPUT RING
        |                         |
        v                         v
     speakers             detector worker
                          RMS / VAD / keyword STT
```

The PortAudio callback no longer creates queue items, resamples, calculates
RMS, runs VAD, or waits on per-fragment events.

Defaults:

```text
callback block:       about 60 ms
output buffer:        4.0 s
output prebuffer:     0.60 s
input buffer:         8.0 s
detector batch:       0.12 s
```

The fixed callback block is intentionally conservative. Keyword-only barge-in
can tolerate a little extra latency, while PortAudio gets a much larger
real-time deadline.

If the backend rejects the fixed block size, the code falls back to
`blocksize=0`.

Kokoro's native 24 kHz rate remains preferred because the devices already
accepted it, which avoids unnecessary output resampling.

## TTS streaming

`play_audio()` now queues generated audio into the output ring and returns once
the fragment is buffered. Kokoro can generate the next fragment while the
speaker is still playing existing buffered audio.

The whole `SpeechSession` performs the final `flush()`. Individual small text
chunks no longer drain playback before the next chunk can be synthesized.

## Diagnostics

Each duplex response prints:

```text
[audio] ring stats: portaudio_output_underflows=0, software_starvations=0, ...
```

Important fields:

- `portaudio_output_underflows`: the host/PortAudio missed an output deadline.
- `software_starvations`: callback ran, but no TTS frames were buffered.
- `input_ring_drops`: microphone ring ran out of free space.
- `input_high_water_ms`: largest microphone backlog.
- `output_high_water_ms`: how far TTS got ahead of playback.

## Test

Run:

```bash
charles
```

and ask for a long answer, for example:

```text
Tell me a two-minute story about Penny exploring a forest.
```

A healthy run should ideally show:

```text
portaudio_output_underflows=0
input_ring_drops=0
```

If PortAudio underflows remain substantial even with this minimal callback,
the next likely issue is the full-duplex stream spanning two independent USB
audio clocks. At that point the next experiment should be at the
PipeWire/ALSA/device-routing level rather than another Kokoro change.
