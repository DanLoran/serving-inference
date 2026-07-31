# vLLM serving inference experiments

Reproducible client-side benchmarks for an OpenAI-compatible vLLM completions
endpoint. The harness stores raw per-request data and creates summaries covering
latency, time to first token, and throughput.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 generate_prompts.py --num-prompts 500 --seed 0
```

Start vLLM separately, then copy and edit the example experiment. The model name
must match a model served by the endpoint.

```bash
cp experiments/example.json experiments/my-experiment.json
python3 scripts/run_experiment.py --config experiments/my-experiment.json
```

Each concurrency level writes raw JSONL and a JSON summary beneath a timestamped
directory in `results/experiments/`. The same directory contains `report.md` and
a combined `summary.json`.

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
python3 scripts/send_requests.py --help
python3 scripts/run_experiment.py --help
```

Benchmark runs send traffic. Verify the endpoint and intended load before
running them, especially against a remote or shared server.
