GENTS.md

## Project purpose

This repository contains reproducible client-side benchmarks for an OpenAI-compatible vLLM completions endpoint. Preserve experimental reproducibility, workload integrity, and clear separation between local validation and live inference testing.

## Before implementation

For every issue:

1. Read the complete issue and comments.
2. Inspect repository instructions, relevant code, tests, documentation, dependencies, Git status, and recent history.
3. Summarize:
   - the requested behavior;
   - the current project state;
   - the proposed implementation;
   - the validation plan;
   - important risks or assumptions.
4. Present this summary before editing files. Then proceed unless the user explicitly requests an approval checkpoint.

Do not claim to have inspected files, executed tests, or contacted a server unless those actions actually occurred.

## Git workflow

- Start from an up-to-date `main` unless the user specifies another base.
- Preserve unrelated or pre-existing changes.
- Create branches using `agent/<concise-description>`.
- Stage only files belonging to the issue.
- Use a concise commit message that describes the complete change.
- Push the branch and open a draft pull request targeting `main`.
- Include `Closes #<issue-number>` in the PR body when appropriate.

Before committing:

- Inspect the complete diff.
- Run `git diff --check`.
- Confirm the diff contains only intended changes.
- Run all relevant tests.

## Implementation principles

- Prefer deterministic, config-driven behavior.
- Never invent benchmark results.
- Preserve raw evidence needed to audit an experiment.
- Record model, tokenizer, seed, request count, token targets, actual token counts, generation settings, and workload identity where applicable.
- Use the exact tokenizer associated with the served model.
- Do not apply a chat template to raw completions workloads unless explicitly required.
- Preserve workload SHA-256 metadata whenever generated artifacts change.
- Regenerate and verify artifacts after changing workload configuration.
- Do not weaken a workload merely to make a failing test pass. Determine whether the failure comes from:
  - implementation;
  - workload configuration;
  - server configuration;
  - model limits;
  - GPU or system capacity;
  - network or transport behavior.

## Local validation

Run server-independent checks before live inference tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/serving-inference-pycache python -m compileall -q .
PYTHONPYCACHEPREFIX=/tmp/serving-inference-pycache python -m unittest discover -s tests
python generate_prompts.py --verify
python generate_prompts.py --help
python scripts/send_requests.py --help
```

If the active environment uses `python3` instead of `python`, use the available interpreter consistently.

Local tests validate generation, determinism, schemas, hashes, token-length tolerances, IDs, bucket composition, and client logic. They do not demonstrate that a live inference server works.

## Remote environment

Use the remote machine only when live inference testing is relevant.

Connection:

```bash
ssh ubuntu-desktop-local
```

Repository:

```text
/home/daniel/Desktop/root-workspace/vllm-test
```

Environment:

```bash
cd /home/daniel/Desktop/root-workspace/vllm-test
source .venv/bin/activate
```

Before modifying or switching the remote checkout:

- Run `git status -sb`.
- Preserve unrelated work.
- Do not switch branches in a dirty checkout without user direction.
- Fetch and fast-forward the intended branch rather than overwriting local state.

## Starting vLLM

Before starting a server:

1. Check whether port 8000 is already listening.
2. Inspect any process already using the port.
3. Do not stop or replace a service that was not launched for the current task.
4. Confirm adequate GPU memory with `nvidia-smi`.

Use this server configuration unless the issue requires something different:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2200
```

The 2,200-token limit supports the 2,048-token `long_prefill` prompt plus its requested 64-token completion, with a small margin.

After launch, wait for:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

Verify that the endpoint reports:

- `Qwen/Qwen2.5-0.5B-Instruct`;
- `max_model_len` of at least 2,112, normally 2,200.

Do not send benchmark traffic until the health check succeeds.

## Live inference validation

Use:

```text
http://127.0.0.1:8000/v1/completions
```

Run one warm-up request before collecting comparison results. Do not include the warm-up in benchmark conclusions.

For representative smoke testing, send at least one request from each relevant workload:

- `short`;
- `long_prefill`;
- `decode_heavy`;
- `mixed`.

For full workload validation, use identical concurrency, temperature, timeout, streaming mode, and model settings across comparable runs.

Record and report:

- attempted requests;
- successful and failed requests;
- HTTP statuses;
- prompt-token targets and actual prompt tokens;
- requested and actual output tokens;
- finish reasons;
- latency;
- TTFT;
- approximate TPOT;
- request throughput;
- output-token throughput;
- error bodies for failed requests.

A live test succeeds only when requests were sent to the running server and their responses were inspected.

## Workload expectations

### `short`

Shape:

```text
128 prompt tokens → up to 64 output tokens
```

Expected use:

- baseline behavior;
- low prefill cost;
- low TTFT;
- low total latency;
- relatively high request throughput.

### `long_prefill`

Shape:

```text
2,048 prompt tokens → up to 64 output tokens
```

Expected use:

- prompt-ingestion stress;
- prefill throughput;
- time to first token;
- long-context KV-cache population.

The server must allow at least:

```text
2,048 + 64 = 2,112 total tokens
```

An HTTP 400 caused by a smaller `max_model_len` is a server-configuration failure, not evidence that the workload is invalid.

Compared with `short`, expect higher TTFT and greater prefill cost. Do not draw conclusions from a single cold request.

### `decode_heavy`

Shape:

```text
128 prompt tokens → up to 512 output tokens
```

Expected use:

- sustained sequential generation;
- TPOT and inter-chunk latency;
- decode throughput;
- long-lived request scheduling.

Compared with `short`, expect similar prompt-related TTFT but substantially higher end-to-end latency.

### `mixed`

Composition:

```text
16 short
8 long_prefill
8 decode_heavy
```

Expected use:

- scheduler interaction;
- batching behavior;
- tail latency;
- responsiveness of short requests while expensive requests are active.

Do not expect mixed results to equal a simple arithmetic average of the homogeneous workloads.

## Interpreting results

- Compare distributions rather than single observations.
- Treat warm-up, compilation, prefix caching, batching, and request order as possible confounders.
- Keep the model, server flags, workload hashes, concurrency, temperature, and streaming settings fixed when comparing runs.
- Prefer p50, p90, and p99 alongside means.
- Distinguish client-observed metrics from server-only metrics.
- Do not label transport chunks as exact model tokens.
- Do not interpret an HTTP validation failure as GPU out-of-memory without supporting evidence.
- Document unexpected results rather than hiding or normalizing them.

## Server cleanup

When testing is complete:

1. Stop only the server process launched for the current task.
2. Prefer recording its PID or process group at startup.
3. Verify that port 8000 is no longer listening.
4. Verify that GPU memory has been released with `nvidia-smi`.
5. Do not kill unrelated Python, CUDA, or vLLM processes.

If the user explicitly asks to leave the server running, report its PID, port, log location, and GPU-memory use.

## Pull request requirements

The draft PR description must include:

- what changed;
- why it changed;
- user or developer impact;
- root cause for fixes;
- local validation;
- live inference validation, when performed;
- limitations or follow-up work;
- the related issue number.

Never describe local unit tests as remote integration tests. Never claim live inference success unless real requests completed against the remote server.

## Final report

Report:

- branch name;
- commit SHA;
- draft PR link;
- files or behavior changed;
- local test results;
- remote test results, if applicable;
- exact live request outcomes;
- server cleanup status;
- unexpected behavior and remaining risks.
