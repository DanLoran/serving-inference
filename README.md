# vLLM serving inference experiments

Reproducible client-side benchmarks for an OpenAI-compatible vLLM completions
endpoint. The harness stores raw per-request data and creates summaries covering
latency, time to first token, and throughput.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 generate_prompts.py
```

The first generation downloads the configured tokenizer from Hugging Face.
Workloads deliberately use raw completion prompts without a chat template so
the token budget is unambiguous and matches the `/v1/completions` endpoint.

## Named workloads

Each workload is defined in `experiments/<name>.json`; generated requests and
metadata are written to `prompts/<name>.jsonl` and
`prompts/<name>.metadata.json`.

| Workload | Requests | Prompt tokens | Maximum output tokens |
| --- | ---: | ---: | ---: |
| `short` | 32 | 128 | 64 |
| `long_prefill` | 32 | 2,048 | 64 |
| `decode_heavy` | 32 | 128 | 512 |
| `mixed` | 32 | see below | see below |

The mixed workload always contains 16 `short`, 8 `long_prefill`, and 8
`decode_heavy` requests. The generator expands buckets in config order and then
shuffles them with an isolated `random.Random` instance and the recorded seed,
making bucket membership and request order stable for a given config.

Every row records requested and actual prompt token counts. The generator uses
the exact tokenizer named by `model` (the config values must match) and rejects
a row outside `prompt_token_tolerance`, currently four tokens. This small bound
accounts for tokenizer decode/re-encode normalization while actual counts remain
recorded per request. The metadata sidecar
preserves the full generation config and SHA-256 of the canonical JSONL bytes.

Generate one workload or validate all checked-in artifacts without downloading
a tokenizer:

```bash
python3 generate_prompts.py --workload long_prefill
python3 generate_prompts.py --verify
```

Start vLLM separately with enough total context for the longest prompt plus its
requested output. The checked-in workloads require 2,112 tokens, and use a
2,200-token limit to leave a small margin:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2200
```

The model name must match the model and tokenizer recorded by the workload.
Then copy and edit the example experiment.

```bash
cp experiments/example.json experiments/my-experiment.json
python3 scripts/run_experiment.py --config experiments/my-experiment.json
```

Each concurrency level writes raw JSONL and a JSON summary beneath a named
directory in `results/experiments/`. The same directory contains `report.md` and
a combined `summary.json`. The example runs one excluded warmup and three
measured repeats at each concurrency in `[1, 2, 4, 8, 12, 16, 24, 32]`.

The runner seed-shuffles concurrency levels separately within each warmup or
measured round. This deterministic interleaving reduces ordering bias without
allowing a measurement to precede its condition's warmup. Every run has its own
raw JSONL and summary path. On resume, valid completed runs are kept
byte-for-byte; a partial, corrupt, or failed required run stops with a clear
error so evidence is never silently overwritten. The saved config must also
match exactly. Warmups remain on disk for auditability but are excluded from the
combined report.

### Saturation and stopping criteria

Treat saturation as demonstrated only when increasing concurrency for two
successive tested levels improves median repeat output-token throughput by less
than 5% while median repeat P99 latency rises by at least 20%. Run all configured
levels through 32 even if an earlier point appears saturated; extend the sweep
if throughput is still improving at the highest level. Inspect the individual
repeat summaries as well as the aggregate, and rerun noisy conditions when
repeat throughput differs by more than 10%.

The example uses 200 requests per repeat, rather than 20, as a practical floor.
P99 is a tail estimate and 200 observations still provide limited resolution;
use 1,000 or more successful requests per condition for conclusions that depend
on P99, and report the request count and repeat variability. A run with any
failed request is invalid and is not included in the combined report or a
saturation claim.

Raw records use a versioned schema and do not store model responses by default.
Pass `--store-response` for debugging only; generated text can be large or
sensitive.

For one run:

```bash
python3 scripts/send_requests.py \
  --url http://localhost:8000/v1/completions \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --num-requests 100 \
  --concurrency 4 \
  --stream \
  --output results/manual-run.jsonl
```

## Reported metrics

- successful and failed requests
- request throughput and server-reported output-token throughput
- mean, p50, p90, and p99 end-to-end latency
- streaming client-observed TTFT and approximate TPOT
- observed inter-chunk latency

Metric definitions:

- **End-to-end latency** starts immediately before the HTTP request and ends
  after the complete response body or streaming `[DONE]` marker. It includes
  client/network and server queue/execution time, but excludes time waiting for
  the benchmark's client-side concurrency semaphore.
- **TTFT** starts at the same point and ends at the first non-empty streamed text
  event. It is client-observed TTFT, not server-only TTFT.
- **Approximate TPOT** divides the time between the first and last non-empty
  streamed text events by `completion_tokens - 1`. It is omitted when fewer than
  two text events arrive or token usage is unavailable. Transport events can
  contain multiple tokens, so this is not exact inter-token latency.
- **Observed inter-chunk latency** is the elapsed time between non-empty streamed
  text events. It is never labeled as exact inter-token latency.
- **Request throughput** is successful requests divided by total measured run
  duration. **Output-token throughput** is server-reported successful completion
  tokens divided by the same duration.

Raw request records include run-relative start/end offsets, request identity and
workload metadata, actual and target token counts when available, finish reason,
status/error data, and whether server token usage was present. HTTP error bodies
are retained up to 4096 characters. Server-side vLLM Prometheus metrics are still
required for queueing, cache, scheduler, or GPU attribution.

## Validation

These checks do not contact a model server:

```bash
PYTHONPYCACHEPREFIX=/tmp/serving-inference-pycache python3 -m compileall -q .
PYTHONPYCACHEPREFIX=/tmp/serving-inference-pycache python3 -m unittest discover -s tests
python3 generate_prompts.py --help
python3 generate_prompts.py --verify
python3 scripts/send_requests.py --help
python3 scripts/run_experiment.py --help
```

Benchmark runs send traffic. Verify the endpoint and intended load before
running them, especially against a remote or shared server.
