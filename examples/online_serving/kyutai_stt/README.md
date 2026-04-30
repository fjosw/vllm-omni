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

Numbers from `benchmark.py` on `hf_s2s.wav` (39.92 s, voice in the
first ~1 s) against the 1B checkpoint on a single H100:

| Scenario                            | TTFT   | Wall    | RTF   |
|-------------------------------------|--------|---------|-------|
| `POST /v1/audio/transcriptions`     | n/a    | 2.66 s  | 0.067 |
| WebSocket, fast upload              | 2.70 s | 2.70 s  | 0.068 |
| WebSocket, real-time pacing (500 ms commit) | **0.81 s** | 45.04 s | 1.128 |
| 4× concurrent Whisper POST          | n/a    | 4.34 s  | (3.98× scaling, 100 % perfect) |

* All four scenarios return the same correct transcript.
* The 0.81 s real-time TTFT is the streaming win: ~500 ms commit interval
  + ~270 ms encode/decode for the first window + a bit of network. Audio
  was paced at 1× wall time, so the upload itself takes 39.92 s.
* The model itself runs at RTF 0.067 — about 15× faster than real time.
  The real-time scenario's RTF of 1.13 is dominated by upload, not
  compute.

## What's still missing for "true Kyutai-style" mid-flight

The current server's re-submission strategy gives ~commit-interval
first-token latency. Kyutai's intended **lockstep** mode emits one text
token per 80 ms audio frame at ~80 ms per-token wall time. To reach that
in this implementation we need:

### 1. The architectural finding: vLLM is the wrong abstraction for true lockstep

I worked the streaming-input path end-to-end and confirmed: it is not
a small patch but an architectural mismatch.

**The reference implementation** (`kyutai-labs/delayed-streams-modeling/scripts/stt_from_file_pytorch.py`)
is the canonical Kyutai lockstep loop:

```python
with mimi.streaming(1), lm_gen.streaming(1):
    for audio_chunk in chunks:                  # 1920-sample frames
        audio_tokens = mimi.encode(audio_chunk)  # streaming codec state
        text_tokens = lm_gen.step(audio_tokens)  # streaming LM state
```

There is no engine, no scheduler, no request lifecycle — just direct
per-frame stepping of (a) the Mimi codec and (b) the LM, both holding
their own streaming state across calls. That's the "lockstep mode
Kyutai was originally designed for".

**vLLM's serving model**, by contrast, is request/response: each
`add_request` represents a self-contained workload that the scheduler
admits, decodes against KV cache, and finishes. The scheduler at line
681 of `vllm/v1/core/sched/scheduler.py` literally has
`assert num_new_tokens > 0` — a request that should "park between
audio chunks" simply isn't a thing it understands.

I tried three layers of patches to bridge this:

* Bypassed `input_processor._validate_prompt_len` for audio-only
  streaming updates (cleared the dictionary-shape check).
* Propagated `additional_information` through
  `OmniARScheduler._update_request_as_session` so the runner sees new
  payloads on the next step (cleared the additional-info plumbing).
* Both reverted in this branch — they hit the deeper
  `assert num_new_tokens > 0` in upstream vLLM, which is fundamental
  to how the scheduler works. To make it not assert you'd need to
  invent a "wait for streaming input" admission state; vllm-omni
  already has `WAITING_FOR_STREAMING_REQ` for similar purposes but
  wiring it up for our case requires rewriting the streaming-update
  semantics throughout.

The model-side hook (`kyutai_preprocess` reading
`_kyutai_pending_audio_chunks`) survives in this branch as future-
proofing. The integration that would actually exercise it is a
non-trivial fork of vllm-omni's streaming-update pipeline — at which
point you've probably reimplemented half the moshi reference.

**Verdict**: for true Kyutai-style lockstep, the right architecture is
the moshi reference loop, optionally wrapped with a thin WebSocket
shim. vLLM gets you "close-to-real-time over a request/response
abstraction" at the cost of re-decoding redundant work each commit
window — that's what we have today, and it's the right trade-off if
you want batched serving across many sessions.

The **2-3 % code drift across re-encodes** measured in
`test_streaming_codec.py` is the same failure mode separately: Mimi's
encoder isn't strictly causal, so even with KV-cache reuse the
cached-but-stale early-position KV would be approximate.

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

| Mode                                                | First-token latency | RTF (real-time pacing) |
|-----------------------------------------------------|---------------------|------------------------|
| Today: re-submission, 500 ms commit window          | **0.81 s measured** | **1.128 measured**     |
| With audio-only streaming updates + buffered encode | ~270 ms (1 commit cycle of decode) | ~1.0 |
| Pure lockstep (1 token / frame, no re-decode)       | ~80 ms              | ~0.5                   |

Today the 500 ms commit window is sufficient for "see partials within
half a second of speaking" but each cycle re-decodes everything, hence
RTF > 1. With `add_streaming_update_async` wired up the model decodes
each token exactly once; per-step cost ≈ 10 ms on a 1B/H100, which
gives us a real-time factor under 0.5x even with re-encode included.
