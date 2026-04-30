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

### 1. Engine-side: allow audio-only streaming updates

Investigated the streaming-input channel in detail.

**Good news**: vLLM upstream's `_update_request_as_session` (which
vllm-omni's STT path inherits, since the `OmniSchedulerMixin` override
only kicks in for `stage_id != 0`) already does the right thing — it
folds completed output tokens into the prompt, leaves
`num_computed_tokens` intact, and extends `mm_features`. So the
scheduler-level append + KV-cache-preservation semantic IS in place.

**Blocker**: the model-side hook is wired up too — `kyutai_preprocess`
reads `_kyutai_pending_audio_chunks` from `additional_information` and
grows the bias buffer. But the path to deliver those chunks fails at
`vllm.v1.engine.input_processor._validate_prompt_len` which rejects
streaming updates with empty `prompt_token_ids`. The streaming-input
mechanism is intrinsically token-stream-shaped; "audio chunk arrived,
no new tokens" doesn't fit. Possible fixes:

a. **Skip prompt-length validation for streaming updates that carry
   only `additional_information`.** Targeted change to
   `vllm_omni.engine.async_omni_engine._build_add_request_message` to
   bypass `input_processor.process_inputs` when `message_type ==
   "streaming_update"` AND the prompt has no new tokens. This is the
   smallest change that unblocks the path; the orchestrator and
   scheduler already cope.
b. **Add a side-channel for `additional_information` updates**
   (separate IPC from the engine that lands directly in the runner's
   `model_intermediate_buffer`). More invasive but cleaner separation
   of concerns.
c. **Fold an extra `[PAD]` per chunk into the streaming-update prompt
   so it passes validation.** Simplest to implement on the server side
   but requires the model's preprocess hook to ignore those PADs in
   its position counter. Mechanically possible; semantically uglier.

The **2-3 % code drift across re-encodes** measured in
`test_streaming_codec.py` (Mimi's encoder is causal+sliding-window but
not strictly causal) means even with KV-cache reuse from
streaming-input, the cached-but-stale KV would be approximate. The
LM's robustness to that drift is empirical; would need verification.

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
