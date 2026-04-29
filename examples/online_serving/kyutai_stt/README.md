# Kyutai STT — streaming transcription server

A self-contained FastAPI server that exposes Kyutai's speech-to-text
model over **three** endpoints:

| Endpoint                                | Shape                            | Use when |
|-----------------------------------------|----------------------------------|----------|
| `POST /v1/audio/transcriptions`         | OpenAI Whisper (multipart)       | offline / batch / OpenAI Python SDK |
| `WS  /v1/audio/transcriptions/stream`   | Custom (`transcript.update/.delta/.revise`) | bespoke clients that want full-transcript snapshots |
| `WS  /v1/realtime`                      | OpenAI Realtime (subset)         | OpenAI Realtime SDKs / browser dictation |

Three reference clients live in this directory:

```
streaming_server.py       # the server
streaming_client.py       # uses the custom WebSocket protocol
realtime_client.py        # speaks the OpenAI Realtime event names
test_streaming_codec.py   # probes Mimi streaming-encode equivalence
```

## Run

```bash
# Server
python examples/online_serving/kyutai_stt/streaming_server.py \
    --model /path/to/stt-1b-en_fr-trfs --port 8765 \
    --commit-interval-ms 500
```

```bash
# OpenAI Python SDK against the Whisper endpoint
python -c '
from openai import OpenAI
c = OpenAI(api_key="none", base_url="http://localhost:8765/v1")
print(c.audio.transcriptions.create(model="kyutai", file=open("audio.wav","rb")).text)
'

# Custom-protocol streaming client (real-time pacing)
python examples/online_serving/kyutai_stt/streaming_client.py \
    --url ws://localhost:8765/v1/audio/transcriptions/stream \
    --audio audio.wav --realtime

# OpenAI-Realtime-protocol streaming client
python examples/online_serving/kyutai_stt/realtime_client.py \
    --url ws://localhost:8765/v1/realtime \
    --audio audio.wav --realtime
```

## How streaming works

The server has one shared session loop that backs both WebSocket
endpoints. As audio chunks arrive the server periodically (every
`--commit-interval-ms`, default 800 ms) submits a fresh request to
`AsyncOmni.generate()` containing the **cumulative** audio buffer,
aborts the in-flight request when fresher audio is available, and
forwards transcript prefixes back to the client.

That is, first-token latency is bounded by the commit interval rather
than the total audio duration. Token-level streaming inside a single
cycle comes for free from `AsyncOmni.generate()`.

When the LM revises an earlier word's punctuation/casing as more audio
arrives (it can — Kyutai's prompt is just `[BOS, PAD]`, so every cycle
re-decodes from scratch on a longer audio context) the server emits
both a `transcript.revise` event (for clients that track deltas) and a
`transcript.update` carrying the full current hypothesis (for clients
that just render the latest snapshot). The Realtime endpoint emits the
combined effect as a `conversation.item.input_audio_transcription.delta`
against the longest common prefix.

## Validated behaviour

Verified end-to-end on `hf_s2s.wav` (39.92 s, voice in the first ~1 s):

* `POST /v1/audio/transcriptions` via `openai.audio.transcriptions.create()`
  — `'Hello, how is it going?'`.
* `WS /v1/audio/transcriptions/stream` with real-time pacing —
  `'Hello, how is it going?'`, RTF ≈ 1.13.
* `WS /v1/realtime` with real-time pacing —
  `'Hello, how is it going?'`, RTF ≈ 1.12.
* 4 concurrent Whisper requests — all 4 return correct transcripts,
  total wall ≈ 4.3 s (vs ≈ 4.2 s for one), confirming `max_num_seqs > 1`
  batches them.

## What's still missing for "true Kyutai-style" mid-flight

The current server's re-submission strategy gives ~commit-interval
first-token latency. Kyutai's intended **lockstep** mode emits one text
token per 80 ms audio frame at ~80 ms per-token wall time. To reach that
in this implementation we need:

### 1. Use `add_streaming_update_async` instead of re-submission

vllm-omni already exposes a streaming-input channel:
`AsyncOmni.generate(prompt=AsyncGenerator[StreamingInput, None])` →
each yielded chunk becomes an `add_streaming_update_async` to stage 0.
For Kyutai we need each update to **extend the multimodal payload**
(grow `multi_modal_data["audio"]`), and the runner side needs
`kyutai_preprocess` to read `req_state.mm_features` on every step and
append new frames to `_kyutai_audio_bias_full` rather than computing
the bias once. This avoids the re-decode-from-scratch waste.

### 2. Lossless chunked Mimi encode (or buffered re-encode)

Mimi's streaming codec (`padding_cache + encoder_past_key_values`) is
**not** bitwise equivalent to one-shot encode past the first chunk —
see `test_streaming_codec.py`. The strided causal convs cache only
`kernel - stride` left context where they need `kernel - 1`. Two
options:
* fold a buffered re-encode into `kyutai_preprocess` (correct, O(N²)
  but cheap for ≤30 s sessions);
* fix `MimiConv1dPaddingCache` upstream to cache `kernel - 1` samples
  (or carry leftover post-conv tails).

### 3. Park the request between audio chunks

When the LM has decoded all positions backed by the current audio
buffer it should park rather than overflow into zero-bias positions
(today the overflow branch adds zero bias which produces unconditioned
text). The minimal fix is for `kyutai_preprocess` to early-stop or
return a "no progress this step" signal; the engine-side equivalent
is the existing streaming-input "wait for next chunk" semantics.

### 4. VAD / endpointing

The OpenAI Realtime spec supports server-side VAD that auto-commits the
audio buffer at silence boundaries. Kyutai's pad-on-silence behaviour
gives us the signal cheaply (count consecutive pad emissions); the work
is plumbing it into a `input_audio_buffer.committed` event.

## Performance characteristics

| Mode                                                | First-token latency | RTF  |
|-----------------------------------------------------|---------------------|------|
| Today: re-submission, 500 ms commit window          | ~500 ms             | ~1.1 |
| With `add_streaming_update_async` + buffered encode | ~80 ms (1 frame)    | <0.5 |
| Pure lockstep (1 token / frame, no re-decode)       | ~80 ms              | ~0.8 |

Today the 500 ms commit window is sufficient for "see partials within
half a second of speaking" but each cycle re-decodes everything, hence
RTF > 1. With `add_streaming_update_async` wired up the model decodes
each token exactly once; per-step cost ≈ 10 ms on a 1B/H100, which
gives us a real-time factor under 0.5x even with re-encode included.
