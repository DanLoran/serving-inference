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
Set `prompt_namespace` in a workload config when two generated request
populations must have different prompt text even if their workload shape and
request indices are otherwise identical. Reusing a namespace intentionally
reuses the same prompt population; changing it creates a deterministic,
separate population. This makes prompt-cache exposure an explicit campaign
choice instead of a side effect of how files were generated.

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

## Config-driven campaigns

Use `scripts/run_campaign.py` when an experiment contains multiple workloads,
phases, or concurrency ranges. A campaign definition under `campaigns/` names
the workload configs and lists its sweeps in execution order, so extending a
campaign requires editing data rather than Python source code.

### Set up a campaign

1. Copy `campaigns/example.json` to a new, uniquely named campaign file.
2. Create or reuse one workload config for each prompt population. For a new
   population, copy the closest config from `experiments/`, set
   `request_count`, make the bucket counts add up to it, and set a unique
   `prompt_namespace`.
3. List those workload configs under `workloads`. Relative paths are resolved
   from the campaign file's directory.
4. Put common serving and measurement settings under `defaults`, then list the
   ordered conditions under `sweeps`. A sweep can override a default such as
   `num_requests`, `warmups`, `repeats`, or `seed`.

The essential structure is:

```json
{
  "schema_version": "1.0",
  "name": "my-capacity-campaign",
  "defaults": {
    "url": "http://127.0.0.1:8000/v1/completions",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "model_metadata": {
      "revision": "<exact-model-revision>",
      "dtype": "half",
      "quantization": null,
      "max_model_len": 2200
    },
    "server": {"discovery": "explicit", "launch_flags": ["<exact flags>"]},
    "num_requests": 256,
    "warmups": 1,
    "repeats": 3,
    "seed": 20260902
  },
  "workloads": [
    {"name": "decode-a", "config": "workloads/decode-a.json"},
    {"name": "prefill-a", "config": "workloads/prefill-a.json"}
  ],
  "sweeps": [
    {"name": "decode-range", "workload": "decode-a", "concurrency": [8, 16, 24, 32]},
    {"name": "prefill-range", "workload": "prefill-a", "concurrency": [1, 2, 4, 8]}
  ]
}
```

Use the complete model revision and server flags from the service being tested;
the OpenAI-compatible endpoint cannot reliably discover them. Validate the plan
with a dry run:

Inspect the fully resolved matrix without generating prompts, contacting the
server, or creating output files:

```bash
python3 scripts/run_campaign.py --config campaigns/example.json --dry-run
```

Review the resolved workload paths, request counts, concurrency lists, warmups,
repeats, seed, model metadata, and launch flags in that output. Commit the final
campaign and workload definitions before an official run, because the runner
requires a clean checkout:

```bash
git add campaigns/my-campaign.json campaigns/workloads/
git commit -m "Define capacity campaign"
```

Start the server separately with the exact recorded settings, confirm its model
endpoint is healthy, then run the campaign. The runner generates and
hash-verifies its prompt files before sending traffic.

Run the complete campaign against an already-running server:

```bash
python3 scripts/run_campaign.py --config campaigns/example.json
```

Run the same command again to resume. Fully completed conditions are preserved;
partial, corrupt, failed, or configuration-mismatched evidence is never
overwritten. Use a new campaign `name` or `--output-root` when changing an
already-started campaign.

The command accepts global condition overrides, making one-off range changes
possible without editing either runner. Overrides are recorded in the resolved
plan, but final campaigns should put their chosen conditions in the committed
campaign definition:

```bash
python3 scripts/run_campaign.py \
  --config campaigns/example.json \
  --concurrency 8 16 24 32 48 \
  --num-requests 32 \
  --warmups 1 \
  --repeats 5 \
  --seed 20260902 \
  --dry-run
```

`--sweep NAME` and `--workload NAME` can be repeated to execute only part of
the resolved plan. Selection changes what executes, not what is recorded in the
full plan. Condition overrides apply to every sweep in that plan. The runner
rejects an override requesting more rows than its workload contains.

Each workload alias is generated once beneath the campaign output and every
sweep referencing that alias uses the same verified bytes. Define a separate
workload config and alias, with a different `prompt_namespace`, when a sweep
should use a disjoint deterministic prompt population. Use the same alias when
reuse is intentional. Prompt text must be unique within each selected run; both
workload generation and request loading reject duplicates before traffic starts.

The campaign runner deliberately does not launch, restart, or flush the serving
process. All selected sweeps run in their declared order against the same
server, preserving production-like server and cache state. Any server lifecycle
change should be an explicit part of the surrounding protocol, not a hidden
runner behavior.

### Controlled prefix-cache campaigns

Use `scripts/run_cache_campaign.py` when cache state is an experimental
variable rather than production-like background state. It is intentionally
separate from the ordinary campaign runner: it refuses an occupied server port,
launches and owns the configured server process, drains and resets the cache
before every measured concurrency/repeat, prewarms only the declared shared
prefixes, and stops only that owned process during cleanup.

The checked-in baseline reproduces the four cache-enabled workloads and common
concurrency sweep:

```bash
python3 scripts/run_cache_campaign.py plan \
  --config campaigns/cache-capacity-baseline.json
```

`plan` has no server or output side effects. Review its 84 conditions, 258,048
measured requests, server command, block size, prefix lengths, and exact mixed
counts before starting. Commit the final definition and implementation first;
the runner refuses an official run from a dirty checkout. A run creates a new
UTC-stamped directory by default.
The current client does not implement a proven ramp/steady-state/drain window,
so this definition uses the specified fallback of 3,072 measured requests per
repeat rather than presenting a burst as fixed-duration steady state.
Pass an explicit, new `--campaign-root` when a supervisor needs to know the path
in advance:

```bash
CAMPAIGN_ROOT=results/campaigns/cache-capacity-baseline-YYYYMMDDTHHMMSSZ
python3 scripts/run_cache_campaign.py run \
  --config campaigns/cache-capacity-baseline.json \
  --campaign-root "$CAMPAIGN_ROOT"
```

The runner is designed to be launched once in a durable terminal or background
job. Observe it by reading one small, atomically replaced file instead of
streaming logs through an agent:

```bash
python3 scripts/run_cache_campaign.py status --campaign-root "$CAMPAIGN_ROOT"
```

Each condition writes beneath
`runs/<workload>/concurrency-NNN/repeat-NN/attempt-NNN/`. A complete attempt is
never rerun. A partial, interrupted, corrupt, or failed attempt remains intact;
the next invocation creates a new numbered attempt and generates a disjoint,
deterministic prompt bank for it. This makes resume attempt-based rather than
file-overwrite based.

Prompt banks are deterministic gzip JSONL. Every measured prompt is exact after
tokenizer round-trip, complete-prompt hashes are retained, and the first unique
block does not repeat within a bank. Shared prefixes are exact multiples of the
declared cache block size. Prefix-only prewarms cannot equal complete measured
prompts. Mixed counts are 1,536 short, 768 long-prefill, and 768 decode-heavy
for every 3,072-request repeat.

The preferred reset is the supported `/reset_prefix_cache` endpoint. If it is
unavailable, the runner restarts only the server process it launched. A reset is
accepted only when the subsequent excluded prefix prewarm records zero cache-hit
tokens. After measured traffic, the achieved token hit rate is calculated from
vLLM prefix-hit and prefix-query counter deltas and checked against the declared
rate; it is never inferred solely from prompt construction.

On completion, `analysis/analysis.json`, `analysis/summary.csv`, and
`analysis/report.md` rebuild aggregate and per-repeat results. Mixed results are
also split by request class. Because vLLM exposes cache counters at the server
level, the mixed aggregate has a measured hit rate while class rows retain the
intended rate and explicitly mark measured per-class cache rate unavailable.
The report classifies adjacent concurrency transitions as useful batching,
queueing-dominant, or a throughput plateau and recommends either another
baseline or a server-parameter campaign.

Signals and normal exceptions trigger owned-server cleanup. The final cleanup
record captures port state and post-run GPU state. After an uncatchable process
kill or host failure, inspect the recorded PID, process command, port, and GPU
state before taking any manual action; never assume a listener is the campaign's
server.

For pip or uv environments that install CUDA runtime libraries inside the
virtual environment, the managed launcher discovers directories containing
`libcudart` beside the configured vLLM executable and prepends them to the
server's `LD_LIBRARY_PATH`. Configured and inherited library paths are retained
without duplication. The effective path and its discovered entries are saved in
each `server/launch-NNN.json`; the parent shell does not need a hidden export.

Outputs are stored under `results/campaigns/<campaign-name>/`: the original
definition, resolved plan, campaign status manifest, generated workload configs
and hash-verified prompt artifacts, plus each existing experiment runner's raw
evidence and reports. Complete sweeps resume through the existing strict resume
checks. A partial selection is recorded as `partial`; the campaign becomes
`completed` only after every declared sweep completes.

Official runs require a clean Git checkout so the source revision and campaign
definition identify the code that ran. `--allow-dirty` is available for local
development and smoke tests. Campaign outputs are ignored by Git by default;
copy or archive them separately when they need durable storage.

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
python3 scripts/run_campaign.py --help
python3 scripts/run_cache_campaign.py --help
python3 scripts/summarize_results.py --help
python3 scripts/plot_results.py --help
```

Benchmark runs send traffic. Verify the endpoint and intended load before
running them, especially against a remote or shared server.
