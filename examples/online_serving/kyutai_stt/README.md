# Kyutai STT — streaming transcription server

A minimal WebSocket server that exposes Kyutai's speech-to-text model
as a streaming transcription API, plus a reference Python client.

## Run

```bash
# Server
python examples/online_serving/kyutai_stt/streaming_server.py \
    --model /path/to/stt-1b-en_fr-trfs --port 8765

# Client
python examples/online_serving/kyutai_stt/streaming_client.py \
    --url ws://localhost:8765/v1/audio/transcriptions/stream \
    --audio /path/to/audio.wav
```

## Wire protocol

```
client -> server   {"type": "session.config", "sample_rate": 24000}
server -> client   {"type": "session.ready",  "request_id": "..."}
client -> server   {"type": "audio.chunk",    "data": "<base64 PCM16 mono>"}   # repeat
client -> server   {"type": "audio.done"}
server -> client   {"type": "transcript.delta", "text": "..."}                 # streamed
server -> client   {"type": "transcript.done",  "text": "<full transcript>"}
```

Audio must be 24 kHz mono PCM16 little-endian. The example client
resamples on load, so any wav file works as input.

## What's actually streaming today

- **Transcript output (real)** — the server forwards each new token from
  `AsyncOmni.generate()` as a `transcript.delta` event the moment the LM
  samples it. With the 1B checkpoint on a single H100 we observe
  RTF ≈ 0.10 (39.92s audio transcribed in 4.09s wall time after
  `audio.done`).
- **Audio input (buffered)** — chunks accumulate server-side until
  `audio.done`. The model runs once on the full waveform. This means
  first-token latency is bounded below by total upload time, and the
  server cannot emit partials while the user is still speaking.

## What's missing for true real-time

For first-token latency that scales with the *next* audio frame
rather than the total clip, we need three independent pieces; each
is a meaningful chunk of work and they can land separately.

### 1. Plumb mid-flight audio chunks to the model

vllm-omni already exposes `AsyncOmni.generate(prompt=AsyncGenerator[StreamingInput, None])`
and `AsyncOmniEngine.add_streaming_update_async()`. Each yielded
`StreamingInput` becomes an `add_streaming_update` message sent to the
orchestrator and forwarded to stage 0 as a new `add_request`.

Today that path is wired for **token** updates (token-stream models
that grow their text prompt mid-flight). For Kyutai we need the
streaming update to grow the **multi-modal** payload — i.e. each chunk
is a fresh `multi_modal_data={"audio": [chunk_so_far]}` with the rest
of the prompt unchanged. The orchestrator already forwards the prompt
as-is, so the work is on the **runner / model** side: `kyutai_preprocess`
needs to read the cumulative audio buffer out of `req_state.mm_features`
on each step rather than once per request, and grow
`_kyutai_audio_bias_full` accordingly.

### 2. Streaming-equivalent encode

`test_streaming_codec.py` shows that Mimi's `padding_cache` +
`encoder_past_key_values` streaming path is **not** bitwise equivalent
to one-shot encode past the first chunk (only ~49% of codes match for a
40s clip cut into 100-frame chunks). The cause is that the strided
causal convolutions don't cache enough left-context across chunk
handoffs — the upstream `MimiConv1dPaddingCache` size is `kernel-stride`
where it would need `kernel-1` to be loss-less.

Two pragmatic options:

- **Buffered re-encode**: on each new chunk, re-encode the full
  buffered audio one-shot and replace `_kyutai_audio_bias_full`. O(N²)
  total compute as the buffer grows but trivially correct, and for
  ≤30 s sessions it's cheap.
- **Fix the upstream cache**: extend `MimiConv1dPaddingCache.padding`
  to `kernel-1` (or fold the leftover post-conv samples into the
  cache). This is a Mimi-side change with broader implications; out
  of scope here, file upstream.

### 3. Engine-side "park while waiting for audio"

When the LM has decoded all tokens for the audio it's seen so far, it
should park rather than continue decoding into a region where
`bias_full` is still empty. Today `kyutai_preprocess` quietly applies
zero bias for positions past the end of `bias_full` (the overflow
branch), which produces unconditioned text. The minimal fix is for
`kyutai_preprocess` to either (a) signal "no progress this step" via a
new return convention, or (b) early-stop the request when the audio
buffer has been fully consumed and no streaming update is pending.

## Performance characteristics

| Mode                                    | First-token latency  | RTF  |
|-----------------------------------------|----------------------|------|
| Today: buffered input + streamed output | upload + ~80 ms     | 0.10 |
| Future: true mid-flight                 | ~80 ms (1 frame)     | 0.10 |

Buffered upload at 1× pace means the user waits the duration of their
audio before seeing the first token. Mid-flight delivery makes
first-token latency a function of the model's per-step cost only
(~80 ms on the 1B/H100 path), which is what users typically expect from
"real-time" transcription.
