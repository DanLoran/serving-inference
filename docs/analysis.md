# Roofline-style prefill and decode analysis

`scripts/analyze_roofline.py` combines ordinary unprofiled token rates with a
representative operation/memory description and explicit device ceilings. It
produces an auditable JSON analysis, a Markdown report, and separate prefill and
decode panels in PNG/PDF (or SVG) form.

This is deliberately a **roofline-style phase or selected-kernel analysis**, not
a whole-model classical roofline claim. vLLM inference includes many kernels,
framework and launch overhead, scheduling, cache effects, host work, and network
time. A bounded Nsight Compute capture may replay or omit kernels. The script
therefore never uses timing from a profiled client and never silently promotes a
selected-kernel sample to a model-wide measurement.

## Inputs

Copy `analysis/rtx-2060-super.example.json` and replace the placeholder
experiment paths and unavailable measurements. The input contract is versioned
in `schemas/roofline-analysis-input.schema.json`.

The device section records:

- theoretical memory bandwidth in SI `GB/s` with its exact derivation and
  source;
- one or more theoretical compute ceilings in SI `TFLOP/s`, each with dtype,
  execution path, derivation, and source;
- a stable ceiling ID selected independently by every case.

The checked-in RTX 2060 SUPER example records the reference-board 448 GB/s
bandwidth, a reference-boost FP32 CUDA-core ceiling, and a dense FP16 Tensor Core
ceiling. These are theoretical reference values. Verify the clocks, board, dtype,
and actual kernel execution for the machine being analyzed; do not select the
Tensor Core ceiling merely because the server accepts FP16 inputs.

Every case is explicitly `prefill` or `decode`:

- prefill must use `prompt_token_goodput_per_s` from a `long_prefill`-style
  unprofiled experiment;
- decode must use `output_token_goodput_per_s` from a `decode_heavy`-style
  unprofiled experiment.

The analyzer rebuilds the matching aggregate and repeat rates directly from
`manifest.json` and measured raw JSONL via `summarize_results.py`. It rejects
incomplete or schema-incompatible evidence and records SHA-256 hashes of the
manifest and every matching measured repeat. `summary.csv`, per-run summary
JSON, and profiled client timing are not rate inputs.

## Counter measurements

Use `kind: "counter"` for representative Nsight Compute evidence. An available
counter measurement requires:

- the completed issue-8 `metadata.json` and native `.ncu-rep` paths;
- an exact scope statement naming the selected kernels/phase;
- the number of phase tokens represented by the counter scope;
- raw DRAM counters in bytes (`dram__bytes_read.sum` and
  `dram__bytes_write.sum`, or version-appropriate exact names);
- raw floating-point or tensor-operation event counters, an explicit
  FLOPs-per-event multiplier, and the execution path represented by each.

Example shape (numeric values are illustrative placeholders):

```json
"measurement": {
  "kind": "counter",
  "status": "available",
  "scope": "selected prefill GEMM kernels identified in the matching Systems trace",
  "profile_metadata": "results/profiles/<capture>/metadata.json",
  "profile_report": "results/profiles/<capture>/capture.ncu-rep",
  "phase_tokens": 8192,
  "dram_counters": [
    {
      "name": "dram__bytes_read.sum",
      "value": 123,
      "unit": "byte",
      "bytes_per_event": 1
    }
  ],
  "flop_counters": [
    {
      "name": "<exact metric name from this NCU version>",
      "value": 456,
      "unit": "event",
      "flops_per_event": 2,
      "execution": "FP16 fused multiply-add; multiplier counts multiply and add"
    }
  ]
}
```

The numeric values above only show the schema and must not be copied into an
analysis. Export or inspect the native report, preserve the report itself, and
enter the exact observed values. Counter names and tensor-event semantics vary
by GPU and Nsight Compute version. Confirm whether a metric counts instructions,
thread-level operations, tensor MMA events, or already-derived FLOPs before
choosing its multiplier. Never infer missing tensor FLOPs from utilization
percentages.

If counter permissions fail, no kernels match, or a required byte/FLOP counter
is absent, record `status: "unavailable"` with the reason. The unprofiled token
rate remains traceable, while the roofline point and derived performance remain
unavailable. The tool does not replace a missing counter with zero.

## Estimates

Use `kind: "estimate"` only for an analytical estimate with explicit
`flops_per_token`, `dram_bytes_per_token`, rationale, and source. Estimate points
use triangle markers; counter-derived points use circles. They remain labeled
separately in JSON and Markdown and should not be described as measured
hardware-counter results.

## Formulas and unit conventions

All conversions use SI prefixes:

```text
counter FLOPs       = sum(raw FLOP event count × FLOPs/event)
counter DRAM bytes  = sum(raw DRAM counter bytes × 1 byte/event)
FLOPs/token         = counter FLOPs / phase tokens
bytes/token         = counter DRAM bytes / phase tokens
arithmetic intensity = FLOPs/token / bytes/token              [FLOP/byte]
achieved performance = unprofiled token/s × FLOPs/token / 1e12 [TFLOP/s]
memory roof          = FLOP/byte × GB/s / 1000                  [TFLOP/s]
selected roof        = min(memory roof, matching compute roof) [TFLOP/s]
```

`1 GB = 1e9 bytes` and `1 TFLOP = 1e12 FLOPs`. The tool rejects ambiguous units
such as GiB/s, byte/s, TFLOPs, or raw tensor events without a multiplier.

## Run and outputs

From the repository root:

```bash
python3 scripts/analyze_roofline.py analysis/my-campaign.json
python3 scripts/analyze_roofline.py analysis/my-campaign.json \
  --formats png,svg --output results/roofline/my-campaign
```

Outputs:

- `roofline-analysis.json` preserves the input hash, source hashes, rebuilt
  repeat and aggregate token rates, exact counters, formulas, derived values,
  and diagnostics;
- `roofline-report.md` states device inputs, formulas, per-phase evidence,
  counter/estimate method, and limitations;
- `roofline.{png,pdf}` plots prefill and decode on separate log-log panels.

The plotted sloped line is theoretical bandwidth multiplied by arithmetic
intensity, capped by the specifically selected theoretical compute ceiling. A
point below that line does not by itself identify the bottleneck: the result can
also reflect incomplete kernel scope, clocks, occupancy, launch overhead,
cache behavior, unsupported tensor execution, scheduling, or client/system
limits.
