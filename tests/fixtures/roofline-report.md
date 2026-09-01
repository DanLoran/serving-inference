# Roofline-style analysis: golden-roofline

> **Scope:** This is a phase/selected-kernel roofline-style analysis. It is not a whole-model classical roofline claim.

## Device ceilings

Theoretical memory bandwidth: **100 GB/s** (SI). 100e9 byte/s

| Ceiling | Value | Dtype | Execution path |
| --- | ---: | --- | --- |
| Dense tensor peak | 10 TFLOP/s | FP16 | dense tensor operations |

Ceiling provenance:

- Memory bandwidth: https://example.test/spec (accessed 2026-08-31).
- Dense tensor peak: test fixture ceiling. Source: https://example.test/spec (accessed 2026-08-31).

## Formulas and units

- Arithmetic intensity: `FLOP/token ÷ byte/token = FLOP/byte`.
- Achieved performance: `unprofiled token/s × FLOP/token ÷ 1e12 = TFLOP/s`.
- Memory roof: `FLOP/byte × GB/s ÷ 1000 = TFLOP/s` using SI prefixes.
- Selected roof: `min(memory roof, matching theoretical compute ceiling)`.

The token rate is rebuilt from the unprofiled experiment manifest and raw measured JSONL. Profiled client latency and throughput are never used.

## Case summary

| Phase | Case | Method | Unprofiled token rate | Arithmetic intensity | Achieved performance | Selected roof |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| prefill | prefill-c1 | counter | 100 token/s | 2 FLOP/byte | 2e-08 TFLOP/s | 0.2 TFLOP/s |
| decode | decode-c1 | estimate | 10 token/s | 2 FLOP/byte | 5e-09 TFLOP/s | 0.2 TFLOP/s |

## Prefill: prefill-c1

Unprofiled evidence: `golden`, workload `long_prefill`, concurrency 1. Rebuilt prompt_token_goodput_per_s = **100 token/s** from 2 measured repeats.

Measurement method: **counter**. Scope: two representative selected prefill kernels

Profile: `golden--long_prefill--c1--ncu`; tool version: `NVIDIA Nsight Compute 2026.1`; phase tokens in counter scope: 10.

Representative Nsight Compute counters:

| Counter | Raw value | Unit | Conversion |
| --- | ---: | --- | --- |
| `dram__bytes_read.sum` | 600 | byte | × 1 byte/event |
| `dram__bytes_write.sum` | 400 | byte | × 1 byte/event |
| `sm__representative_fma_events.sum` | 1000 | event | × 2 FLOP/event (FP16 fused multiply-add) |

Derived point: **2 FLOP/byte**, **2e-08 TFLOP/s achieved**; selected roof **0.2 TFLOP/s**.

## Decode: decode-c1

Unprofiled evidence: `golden`, workload `decode_heavy`, concurrency 1. Rebuilt output_token_goodput_per_s = **10 token/s** from 2 measured repeats.

Measurement method: **estimate**. Scope: analytical per-output-token estimate

Estimate rationale: known-value test estimate

Estimate source: https://example.test/model (accessed 2026-08-31).

Derived point: **2 FLOP/byte**, **5e-09 TFLOP/s achieved**; selected roof **0.2 TFLOP/s**.

## Interpretation limits

- Prefill and decode are reported separately because their operation mix and memory behavior differ.
- Counter points describe only the recorded Nsight Compute scope. Kernel filtering/replay may omit model work.
- Estimate points are visibly separate from hardware-counter points and depend on their stated assumptions.
- Theoretical ceilings are applicable only when dtype, tensor-core use, clocks, and execution path match the recorded case.
- Host, scheduler, network, launch, cache, and framework overheads are outside a classical device roofline model.
