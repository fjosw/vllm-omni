# Kyutai STT — streaming transcription server

A self-contained FastAPI server that exposes Kyutai's speech-to-text
model over three endpoints:

| Endpoint                                | Shape                            | Use when |
|-----------------------------------------|----------------------------------|----------|
| `POST /v1/audio/transcriptions`         | OpenAI Whisper (multipart)       | offline / batch / OpenAI Python SDK |
| `WS  /v1/audio/transcriptions/stream`   | Custom (`transcript.update/.delta/.revise`) | clients that want full-transcript snapshots |
| `WS  /v1/realtime`                      | OpenAI Realtime (subset)         | OpenAI Realtime SDKs / browser dictation |

Three reference clients live in this directory; all run with `uv run --no-project`
and need no vllm install:

```
streaming_server.py        # the server
streaming_client.py        # custom-protocol client
realtime_client.py         # OpenAI Realtime client
benchmark.py               # all four scenarios + concurrency
test_streaming_codec.py    # Mimi streaming-encode probe
```

## Run

```bash
# Server (on a GPU box)
python examples/online_serving/kyutai_stt/streaming_server.py \
    --model /path/to/stt-1b-en_fr-trfs --port 8765 \
    --commit-interval-ms 500
```

```bash
# Whisper-style offline transcription via OpenAI Python SDK
python -c '
from openai import OpenAI
c = OpenAI(api_key="none", base_url="http://localhost:8765/v1")
print(c.audio.transcriptions.create(model="kyutai", file=open("audio.wav","rb")).text)
'

# Custom WebSocket protocol with real-time pacing
uv run --no-project examples/online_serving/kyutai_stt/streaming_client.py \
    --url ws://localhost:8765/v1/audio/transcriptions/stream \
    --audio audio.wav --realtime

# OpenAI Realtime protocol
uv run --no-project examples/online_serving/kyutai_stt/realtime_client.py \
    --url ws://localhost:8765/v1/realtime \
    --audio audio.wav --realtime
```

## How the streaming endpoints work

Both WebSocket endpoints share `_run_session`: as audio chunks arrive
the server periodically (every `--commit-interval-ms`, default 800 ms)
submits a fresh request to `AsyncOmni.generate()` containing the
**cumulative** audio buffer, aborts the in-flight request when fresher
audio is available, and forwards transcript prefixes back to the
client. First-token latency is bounded by the commit interval rather
than the total audio duration; token-level streaming inside a single
cycle comes for free from `AsyncOmni.generate()`.

When the LM revises an earlier word's punctuation/casing as more audio
arrives (it can — every cycle re-decodes from scratch on a longer audio
context) the custom protocol emits both a `transcript.update` snapshot
of the full hypothesis and a `transcript.revise` + `transcript.delta`
pair for clients that prefer additive semantics. The Realtime endpoint
collapses this into a single
`conversation.item.input_audio_transcription.delta` against the
longest common prefix.

`max_tokens` is capped per submission at the audio horizon
(`floor(audio_samples / 1920) + audio_delay_seconds + 1` Mimi frames)
to prevent the LM from decoding past the audio buffer's end into
zero-bias positions where it produces unconditioned text. Pass
`--fe-pad-frames` and `--horizon-safety-frames` to retune for
checkpoints with different `audio_delay_seconds`.

## Validated behaviour

`benchmark.py` against the 1B en/fr checkpoint on a single H100, with
`hf_s2s.wav` (39.92 s):

| Scenario                            | TTFT   | Wall    | RTF   |
|-------------------------------------|--------|---------|-------|
| `POST /v1/audio/transcriptions`     | n/a    | 2.66 s  | 0.067 |
| WebSocket, fast upload              | 2.70 s | 2.70 s  | 0.068 |
| WebSocket, real-time pacing (500 ms commit) | **0.81 s** | 45.04 s | 1.128 |
| 4 × concurrent Whisper POST         | n/a    | 4.34 s  | 3.98× scaling |

All four scenarios return the same correct transcript. The model
itself runs at RTF 0.067 (≈15× faster than real time); the real-time
scenario's RTF > 1 is dominated by audio upload, not compute. For a
session of duration `T`, post-`audio.done` latency is roughly
`encode(T) + decode(T) ≈ 0.07T`.

## Tuning `--commit-interval-ms`

Multiples of 80 ms keep cycles aligned with Mimi's 12.5 Hz frame
rate. The right value depends on per-cycle decode cost (`~0.07 × T` for
audio buffer of length `T`) and how often you want partials:

| Use case | Commit interval |
|---|---|
| Live captions, ≤5 s utterances | 400 ms |
| Conversational, ≤15 s | 800 ms (default) |
| Long-form, occasional partials | 1 600 ms+ |

Going below the per-cycle decode cost makes every cycle abort before
producing text — the engine thrashes without ever emitting a partial.

## What's missing for true Kyutai-style lockstep

The `kyutai-labs/delayed-streams-modeling` reference implementation
runs Kyutai's lockstep mode as a tight per-frame Python loop:

```python
with mimi.streaming(1), lm_gen.streaming(1):
    for audio_chunk in chunks:                    # 1920 samples each
        audio_tokens = mimi.encode(audio_chunk)   # streaming codec state
        text_tokens = lm_gen.step(audio_tokens)   # streaming LM state
```

vLLM's request/response model is a fundamentally different abstraction
— the scheduler asserts `num_new_tokens > 0` for any admitted request,
so a request that "parks between audio chunks" isn't representable. We
investigated three layers of patches (bypass empty-prompt validation
in the input processor, propagate `additional_information` through
`OmniARScheduler._update_request_as_session`, expose
`_kyutai_pending_audio_chunks` from the model's preprocess hook); each
clears one validation but the next-deeper one fails. To actually make
lockstep work over vLLM you'd need to invent a "request waiting for
streaming input" admission state and rewire the scheduler around it —
at which point you've reimplemented the moshi reference loop with
extra plumbing.

The right architecture for true lockstep is a thin WebSocket shim
around the moshi reference loop. The right architecture for batched
serving across many sessions is what we have today on top of vLLM. The
re-submission strategy here gives RTF ≈ 1.1 in real-time pacing and
~100% scaling efficiency for concurrent sessions; lockstep would give
RTF ≈ 1.0 but doesn't batch.

The `test_streaming_codec.py` probe separately documents that Mimi's
streaming-encode path is not bitwise equivalent to one-shot encode (the
strided causal convs cache `kernel - stride` left context where they
need `kernel - 1`); KV-cache reuse across cycles would be approximate
even if the scheduler-side issue were solved.
