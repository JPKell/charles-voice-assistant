# Streaming tool-calling patch

This restores true Ollama streaming while keeping the tool and web-search
agent loop.

Before this patch, every tool-enabled `/api/chat` request used:

```json
"stream": false
```

so even ordinary answers had to finish completely before the sentence
accumulator and Kokoro TTS received text.

After this patch, tool-enabled rounds use:

```json
"stream": true
```

and the NDJSON response is consumed incrementally.

Normal request:

```text
Whisper
  -> Ollama streaming chunks
  -> existing sentence/chunk accumulator
  -> Kokoro
  -> speakers
```

Tool request:

```text
Whisper
  -> Ollama streamed tool call
  -> execute local/web tool
  -> append tool result
  -> next Ollama round streams final answer
  -> Kokoro
```

The code accumulates `thinking`, `content`, and `tool_calls` for the assistant
message used in the next tool round. Thinking is not yielded to TTS.

One tradeoff: visible content is yielded immediately. If a model emits prose
before deciding to make a tool call in the same assistant turn, that short
preamble can be spoken before the tool executes. Buffering the entire first
turn would prevent that, but would also recreate the delay this patch removes.

Only `voice_chat/tooling.py` is changed.
