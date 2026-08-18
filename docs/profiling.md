# Profiling vLLM with NVIDIA Nsight

The profiling scripts launch the **vLLM server** under NVIDIA Nsight and drive it
with a separate `send_requests.py` client. They are intended for bounded,
repeatable diagnosis of representative prefill or decode activity, not for
routine latency benchmarking.

## Important interpretation rule

Profiling changes execution. Nsight Systems tracing adds collection and export
overhead; Nsight Compute serializes or replays selected kernels and can make a
request many times slower. Clock control and cache flushing may change execution
further. **Never directly compare latency, TTFT, TPOT, or throughput from a
profiled run with an unprofiled benchmark run.** Client results produced here
only prove which controlled workload was active during capture. Use ordinary,
identically configured benchmark runs for performance comparisons.

## Prerequisites

Run from the repository root in the environment that contains vLLM and the
benchmark dependencies. Check the tools and make sure the configured port is
not already owned by another server:

```bash
nsys --version
ncu --version
ss -ltnp 'sport = :8000'
nvidia-smi
```

The scripts stop only the profiler process group that they launch. They refuse
to reuse an existing artifact directory. Do not place credentials in the JSON
config: exact commands are intentionally stored in `metadata.json`.

## No-GPU and command-review walkthrough

Dry run does not require a GPU, Nsight installation, reachable server, or vLLM.
It validates the config and prints the profiler/server and workload/client
commands without creating files or starting processes:

```bash
bash -n scripts/profile_nsys.sh scripts/profile_ncu.sh
scripts/profile_nsys.sh --config profiling/example.json --dry-run
scripts/profile_ncu.sh --config profiling/example.json --print-command
```

`--dry-run` and `--print-command` are aliases. Confirm that the vLLM command is
the final portion of the profiler command and that `send_requests.py` appears
only in the separate workload command.

## Configuration

Copy `profiling/example.json` and give every capture a new `experiment` value.
Paths are resolved from the repository root.

Common fields:

| Field | Meaning |
| --- | --- |
| `experiment` | Filesystem-safe experiment identifier. |
| `workload` | Named workload identifier, such as `long_prefill` or `decode_heavy`. |
| `concurrency` | Fixed client concurrency recorded in the artifact name. |
| `num_requests` | Controlled number of requests sent during capture. |
| `prompts` | Generated JSONL workload used by `send_requests.py`. |
| `model` | Model sent to the completions endpoint. |
| `url` / `health_url` | Completions and readiness endpoints for the launched server. |
| `startup_timeout_s` | Maximum time to wait for the expected model to become ready. |
| `request_timeout_s` | Per-request timeout passed to the client. |
| `profiler_shutdown_timeout_s` | Grace period for report finalization before escalation. |
| `output_dir` | Parent directory for immutable capture directories. |
| `server_command` | Exact vLLM server argv wrapped by Nsight. |

The capture directory is deterministic:

```text
<output_dir>/<experiment>--<workload>--c<concurrency>--<tool>/
```

It contains `metadata.json`, `server.log`, `client.jsonl`,
`client.summary.json`, and the native `.nsys-rep` or `.ncu-rep` report when the
tool succeeds. Metadata records the config SHA-256, Git state, host, UTC times,
tool version, exact profiler/server/client argv, exit codes, cleanup actions,
and expected artifact paths.

## Nsight Systems

Review the command, then execute it:

```bash
scripts/profile_nsys.sh --config profiling/my-prefill.json --dry-run
scripts/profile_nsys.sh --config profiling/my-prefill.json
```

The config must include `cuda`, `nvtx`, and `osrt` traces. These capture CUDA API
activity, GPU kernels, NVTX annotations, and OS runtime activity. `duration_s`
is a hard upper bound on collection after `delay_s`; the script also ends the
session once the controlled workload finishes. The default `kill` behavior is
set explicitly to `sigterm`, so a duration expiry stops the launched server
rather than leaving it behind.

Choose a duration long enough for server startup plus the requests when delay is
zero. A delayed capture can reduce startup noise, but readiness time varies and
the workload must still overlap the collection window. Inspect `server.log` and
`client.summary.json`; a report file alone does not prove that requests
completed successfully.

Example focused prefill settings:

```json
"workload": "long_prefill",
"concurrency": 1,
"num_requests": 4,
"nsys": {
  "trace": ["cuda", "nvtx", "osrt"],
  "sample": "none",
  "delay_s": 0,
  "duration_s": 120
}
```

## Nsight Compute

Start with an Nsight Systems trace to identify a kernel and then narrow
`kernel_name`. Avoid collecting every kernel with a large metric set.

```bash
scripts/profile_ncu.sh --config profiling/my-decode.json --dry-run
scripts/profile_ncu.sh --config profiling/my-decode.json
```

The workflow always sets `--target-processes=all` so CUDA work in vLLM worker
children is eligible. The checked-in config selects the `basic` section set,
kernel replay, an explicit kernel-name regex, a launch skip, and a small launch
count. `launch_count` bounds the number of matched launches collected; it does
not make replay cheap. Kernel names and CUDA-graph behavior vary by vLLM,
PyTorch, attention backend, and GPU, so update the regex from observed Systems
evidence rather than guessing for a final capture.

Useful controls:

```json
"ncu": {
  "set": "basic",
  "replay_mode": "kernel",
  "kernel_name": "regex:.*(flash|paged_attention|gemm|cutlass).*",
  "launch_skip": 0,
  "launch_count": 1
}
```

Nsight Compute hardware counters may be disabled by the driver or host policy.
If the log reports `ERR_NVGPUCTRPERM`, follow the NVIDIA administrator guidance
for the specific host; do not silently run with a broader privilege level. A
permission failure, unmatched kernel filter, timeout, or interrupted replay is
a failed capture and should remain recorded in the metadata and log.

## Workload selection

Use a small fixed workload that isolates the activity of interest:

- `long_prefill` emphasizes prompt ingestion and first-token work.
- `decode_heavy` emphasizes sustained token generation.
- `short` is useful for plumbing checks but is less diagnostic.
- `mixed` introduces scheduler interaction and is usually a second-stage trace.

Keep the model, prompt artifact and SHA, generation settings, concurrency, and
server flags fixed across related investigations. Warm the model separately
only if the experimental question requires it and record that procedure; these
scripts intentionally do not hide an unrecorded warm-up phase.

## Cleanup and failures

On success, workload failure, Ctrl-C, readiness timeout, or profiler exit, the
orchestrator signals only its own profiler/server process group and waits for
report export. It escalates from interrupt to terminate to kill only when the
group does not stop within the configured grace period. It never searches for
or kills unrelated Python, CUDA, or vLLM processes.

After a live run, verify cleanup explicitly:

```bash
ss -ltnp 'sport = :8000'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

A nonzero exit preserves the artifact directory for diagnosis. Choose a new
`experiment` name for the next attempt rather than deleting or overwriting the
evidence.
