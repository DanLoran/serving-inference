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
a combined `summary.json`, plus an analysis-ready `summary.csv`. The example is
a 32-request range-finding sweep with
one excluded warmup and three measured repeats at each concurrency in
`[1, 2, 4, 8, 12, 16, 24, 32]`. It verifies the experiment pipeline and locates
a rough region of interest; it does not support a final tail-latency claim.

The runner seed-shuffles concurrency levels separately within each warmup or
measured round. This deterministic interleaving reduces ordering bias without
allowing a measurement to precede its condition's warmup. Every run has its own
raw JSONL and summary path. On resume, valid completed runs are kept
byte-for-byte; a partial, corrupt, or failed required run stops with a clear
error so evidence is never silently overwritten. The saved config must also
match exactly. Warmups remain on disk for auditability but are excluded from the
combined report.

### GPU and vLLM telemetry

The example enables timestamped GPU and vLLM telemetry. GPU collection invokes
`nvidia-smi` once per interval and preserves one CSV row per device with GPU and
memory utilization, used and total memory, power, temperature, and current SM
and memory clocks. vLLM collection periodically fetches the configured
Prometheus endpoint and preserves each complete text exposition response rather
than depending on metric names from one vLLM release. This retains available
running/waiting request, scheduler timing, KV-cache, and token counters across
metric-name changes.

Artifacts are written under `<experiment>/telemetry/`:

- `gpu.csv` contains raw `nvidia-smi` values with a UTC timestamp and experiment
  offset for every device sample.
- `vllm.prometheus.jsonl` contains the complete raw Prometheus response for each
  timestamped scrape.
- `events.jsonl` marks telemetry start/stop and every benchmark command's start
  and finish on the same monotonic experiment timeline.
- `status.json` records collector availability, sample counts, errors, cleanup,
  and the shared telemetry epoch.

Both collectors are optional and independently configurable:

```json
"telemetry": {
  "gpu": {"enabled": true, "interval_s": 1.0, "timeout_s": 5.0},
  "vllm": {
    "enabled": true,
    "url": "http://localhost:8000/metrics",
    "interval_s": 1.0,
    "timeout_s": 5.0
  }
}
```

Omit `telemetry` or disable either collector when it is not needed. A missing
`nvidia-smi`, unreachable metrics endpoint, malformed sample, or scrape timeout
is recorded as telemetry unavailable and does not invalidate otherwise valid
client results. Collectors are stopped after both successful and failed sweeps.
Sampling is intentionally coarse; use `events.jsonl` to select samples whose
offsets overlap the benchmark interval, and do not infer device saturation from
client metrics alone.

### Nsight profiling

Bounded Nsight Systems and Nsight Compute workflows are available for targeted
server-side investigation. Both workflows launch the vLLM server under the
profiler and run the HTTP workload as a separate, unprofiled client process:

```bash
scripts/profile_nsys.sh --config profiling/example.json --dry-run
scripts/profile_ncu.sh --config profiling/example.json --dry-run
```

Remove `--dry-run` only after checking the printed server and workload commands,
the idle server port, the selected workload, and the output directory. Generated
reports and a metadata manifest are written under `results/profiles/` with names
that include the experiment, workload, concurrency, and profiler.

See [docs/profiling.md](docs/profiling.md) for configuration, capture bounds,
counter-permission troubleshooting, cleanup behavior, and the required warning
against direct profiled/unprofiled latency comparisons.

### Roofline-style phase analysis

Build an auditable prefill/decode roofline-style report by combining token rates
rebuilt from ordinary unprofiled experiment artifacts with explicit device
ceilings and representative Nsight Compute counters:

```bash
cp analysis/rtx-2060-super.example.json analysis/my-campaign.json
# Replace placeholder experiment/capture paths and unavailable measurements.
python3 scripts/analyze_roofline.py analysis/my-campaign.json
```

The analyzer writes schema-versioned JSON, Markdown, and separate prefill/decode
roofline figures under `results/roofline/`. It hashes the unprofiled manifest,
raw measured runs, profile metadata, and native Compute report; retains exact
counter names, values, units, and FLOP multipliers; and marks missing counters
unavailable instead of substituting estimates. Counter and estimate points are
visually distinct. Profiled throughput/latency is never used.

See [docs/analysis.md](docs/analysis.md) for the input schema, SI formulas,
counter-export guidance, RTX 2060 SUPER theoretical-ceiling derivations, and why
this scoped analysis must not be presented as a whole-model classical roofline.

### Saturation and stopping criteria

Treat saturation as demonstrated only when increasing concurrency for two
successive tested levels improves median repeat output-token throughput by less
than 5% while median repeat P99 latency rises by at least 20%. Run all configured
levels through 32 even if an earlier point appears saturated; extend the sweep
if throughput is still improving at the highest level. Inspect the individual
repeat summaries as well as the aggregate, and rerun noisy conditions when
repeat throughput differs by more than 10%.

The runner refuses to request more rows than the workload JSONL contains. For a
measurement campaign, increase the workload definition's request count (and
mixed bucket counts), regenerate and verify its JSONL and metadata, and then set
the experiment's `num_requests` to the generated row count. P99 is a tail
estimate; use 1,000 or more successful requests per condition for conclusions
that depend on P99, and report the request count and repeat variability. The
32-request example is not sufficient for that purpose. A run with any failed
request is invalid and is not included in the combined report or a saturation
claim.

## Reproducibility manifest

The experiment runner writes `config.original.json`, `config.resolved.json`,
and a schema-versioned `manifest.json` before sending traffic. Its JSON Schema
is checked in at `schemas/experiment-manifest.schema.json`. The manifest is
updated with a UTC completion time and final status when the run finishes. It
records the workload path and SHA-256, Git revision and dirty state, Python and
installed dependency versions, OS/kernel, NVIDIA GPU/VRAM, driver and CUDA
details when available, and the installed vLLM version and package fingerprint.
Collection remains valid on CPU-only hosts or when NVIDIA tools are absent.

Model revision, dtype, quantization, maximum model length, and server launch
flags cannot be reliably discovered from a remote OpenAI-compatible endpoint.
They are therefore required explicitly in `model_metadata` and `server` in the
experiment config; see `experiments/example.json`. Keep `launch_flags` limited
to non-secret command flags. Credential-shaped config fields and credentials in
URLs are redacted, and the collector never snapshots environment variables.

`config.original.json` preserves the supplied values (with secrets redacted),
while `config.resolved.json` also contains defaults and the absolute workload
path used for the run. Use a new experiment name or an empty output directory
for every invocation so prior evidence is never overwritten.

Raw records use a versioned schema and do not store model responses by default.
Pass `--store-response` for debugging only; generated text can be large or
sensitive.

### Rebuildable CSV analysis

`summary.csv` is generated exclusively from `manifest.json` and the measured
repeat JSONL files; per-run and combined JSON summaries are not inputs. Rebuild
it deterministically after copying or auditing an experiment:

```bash
python3 scripts/summarize_results.py results/experiments/my-experiment
```

Repeat rows are normalized by observed workload, concurrency, and repeat.
Aggregate rows pool the raw observations across repeats, so their E2E, TTFT,
and approximate-TPOT P50/P90/P99 values are recomputed rather than averaged
from per-repeat percentiles. Aggregates also report sample standard deviation
and normal-approximation 95% confidence intervals across repeat-level request
goodput, output-token goodput, and failure rate. These intervals are blank for
a single repeat.

For mixed runs, each workload row uses the complete repeat wall-clock duration:
requests overlap, so the raw client evidence cannot assign exclusive elapsed
time to one workload. `schema_compatible`, `complete`, and `issues` explicitly
flag unsupported row/manifest schemas, missing runs, malformed JSONL, request
count mismatches, and unavailable metrics; affected evidence is never silently
dropped.

### Report-ready figures

Generate publication-ready PNG and vector PDF figures from a completed
experiment without contacting the benchmark server:

```bash
python3 scripts/plot_results.py results/experiments/my-experiment
```

The plotter rebuilds its client analysis directly from `manifest.json` and the
measured repeat JSONL files; it does not trust a potentially stale summary. It
writes deterministic names beneath `<experiment>/figures/`:

- `throughput-<workload>.{png,pdf}` shows request, prompt-token,
  output-token, and total-token goodput against concurrency.
- `latency-<workload>.{png,pdf}` shows E2E, client-observed TTFT, and
  approximate-TPOT P50/P90/P99 against concurrency.
- `gpu-telemetry.{png,pdf}` shows GPU utilization and used memory against the
  shared experiment offset when `telemetry/gpu.csv` is present.
- `plot-manifest.json` records the source artifact paths and SHA-256 hashes,
  generated filenames, metric definitions, and input diagnostics.

Each workload gets separate figures. Lines show the mean repeat metric, small
dots show individual measured repeats, and error bars are normal-approximation
95% confidence intervals across repeat values. A single repeat has no CI. A red
`x` at the plot floor marks an unavailable value rather than interpolating or
replacing it. Incompatible manifest or raw-result schemas stop rendering;
incomplete or missing metrics remain visible in `plot-manifest.json`.

GPU plotting consumes the raw `nvidia-smi` CSV schema produced by the telemetry
collector from issue #6. If that artifact is absent, client figures are still
generated and the omission is recorded as a diagnostic. GPU sampling is coarse
and cannot by itself prove device saturation. Raw Prometheus snapshots are
preserved for future queue-depth or KV-cache analysis but are not normalized or
plotted here because vLLM metric names vary by release.

Choose formats or suppress GPU inspection explicitly when needed:

```bash
python3 scripts/plot_results.py results/experiments/my-experiment \
  --formats png,svg --skip-gpu
```

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
python3 scripts/summarize_results.py --help
python3 scripts/plot_results.py --help
```

Benchmark runs send traffic. Verify the endpoint and intended load before
running them, especially against a remote or shared server.
