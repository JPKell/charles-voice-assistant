# Duplex barge-in audio-quality patch

This targets audio artifacts that occur only with full-duplex barge-in enabled.

Because `--no-barge-in` is clean, the likely problem is the duplex playback
path rather than Kokoro inference itself.

## What changed

1. **Prefer 24 kHz duplex audio.**
   Kokoro produces 24 kHz audio. If both audio devices support 24 kHz, no TTS
   output resampling is required.

2. **Stateful resampling.**
   When another duplex rate is required, interpolation phase and the previous
   sample are preserved across Kokoro fragments rather than restarting the
   linear resampler at every fragment.

3. **250 ms producer lookahead.**
   The TTS producer is released about 250 ms before the currently playing
   fragment finishes. This gives Kokoro time to produce and queue the next
   fragment instead of waiting for the output queue to become empty.

4. **Playback flush.**
   `tts.py` waits for the final queued fragment to really finish before marking
   a TTS item complete.

5. **Conservative PortAudio buffering.**
   The duplex stream requests `latency="high"` and falls back to the previous
   stream opening method if the backend does not accept it.

6. **Diagnostics.**
   PortAudio underflow/overflow status is counted. If trouble occurs you may see:

   ```text
   [audio] duplex stats: output_underflows=2, input_overflows=0, ...
   ```

The startup log will also show the duplex sample rate:

```text
[audio] duplex stream 24000 Hz; TTS lookahead 0.25s
```

or another supported common rate.

## Test

Compare:

```bash
charles
```

with:

```bash
charles --no-barge-in
```

If the artifacts remain, note both the selected duplex rate and any
`output_underflows` count. That tells us whether the next patch should focus on
a dedicated ring buffer/fixed callback size or on the device-specific sample
rate path.
