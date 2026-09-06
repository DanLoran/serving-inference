#!/usr/bin/env python3
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


EXPERIMENTS = [
    ("decode_heavy", "phase1-48-128", "capacity-range-decode-20260901T233902Z-phase1"),
    ("decode_heavy", "phase2-96-256", "capacity-range-decode-20260901T233902Z-phase2"),
    ("decode_heavy", "phase3-refinement", "capacity-range-decode-20260901T233902Z-phase3-refinement"),
]


def f(value):
    if value in (None, ""):
        return None
    return float(value)


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def max_or_none(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def min_or_none(values):
    values = [v for v in values if v is not None]
    return min(values) if values else None


def cv_percent(values):
    values = [v for v in values if v is not None]
    if len(values) < 2 or statistics.mean(values) == 0:
        return None
    return 100.0 * statistics.stdev(values) / statistics.mean(values)


def parse_prometheus(raw):
    wanted = {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions_total",
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total",
    }
    values = defaultdict(list)
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([^ {]+)(?:\{[^}]*\})?\s+([^\s]+)$", line)
        if not match or match.group(1) not in wanted:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        if math.isfinite(value):
            values[match.group(1)].append(value)
    return {key: sum(items) for key, items in values.items()}


def load_windows(root):
    events_path = root / "telemetry" / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
    starts = {}
    windows = []
    for event in events:
        if event.get("kind") != "repeat":
            continue
        key = (event.get("index"), event.get("concurrency"))
        if event["event"] == "benchmark_started":
            starts[key] = float(event["experiment_offset_s"])
        elif event["event"] == "benchmark_finished" and key in starts:
            windows.append({
                "repeat": int(event["index"]),
                "concurrency": int(event["concurrency"]),
                "start": starts.pop(key),
                "finish": float(event["experiment_offset_s"]),
            })
    return windows


def telemetry_by_concurrency(root):
    windows = load_windows(root)
    prom_path = root / "telemetry" / "vllm.prometheus.jsonl"
    prom = []
    if prom_path.exists():
        for line in prom_path.read_text().splitlines():
            if line:
                row = json.loads(line)
                prom.append((float(row["experiment_offset_s"]), parse_prometheus(row["raw"])))
    gpu_path = root / "telemetry" / "gpu.csv"
    gpu = []
    if gpu_path.exists():
        with gpu_path.open() as handle:
            for row in csv.DictReader(handle):
                def number(name):
                    try:
                        return float(row[name])
                    except (ValueError, TypeError):
                        return None
                gpu.append((number("experiment_offset_s"), {
                    "util": number("utilization_gpu_percent"),
                    "mem_used": number("memory_used_mib"),
                    "mem_total": number("memory_total_mib"),
                    "power": number("power_draw_w"),
                    "temp": number("temperature_gpu_c"),
                    "clock_sm": number("clocks_sm_mhz"),
                    "clock_mem": number("clocks_memory_mhz"),
                }))

    per_concurrency = defaultdict(list)
    for window in windows:
        ps = [sample for offset, sample in prom if window["start"] <= offset <= window["finish"]]
        gs = [sample for offset, sample in gpu if offset is not None and window["start"] <= offset <= window["finish"]]
        running = [sample.get("vllm:num_requests_running") for sample in ps]
        waiting = [sample.get("vllm:num_requests_waiting") for sample in ps]
        kv = [sample.get("vllm:kv_cache_usage_perc") for sample in ps]
        preempt = [sample.get("vllm:num_preemptions_total") for sample in ps if sample.get("vllm:num_preemptions_total") is not None]
        hits = [sample.get("vllm:prefix_cache_hits_total") for sample in ps if sample.get("vllm:prefix_cache_hits_total") is not None]
        queries = [sample.get("vllm:prefix_cache_queries_total") for sample in ps if sample.get("vllm:prefix_cache_queries_total") is not None]
        hit_delta = hits[-1] - hits[0] if len(hits) >= 2 else None
        query_delta = queries[-1] - queries[0] if len(queries) >= 2 else None
        per_concurrency[window["concurrency"]].append({
            "running_mean": mean(running),
            "running_max": max_or_none(running),
            "waiting_mean": mean(waiting),
            "waiting_max": max_or_none(waiting),
            "kv_mean_pct": 100.0 * mean(kv) if mean(kv) is not None else None,
            "kv_max_pct": 100.0 * max_or_none(kv) if max_or_none(kv) is not None else None,
            "preemptions_delta": preempt[-1] - preempt[0] if len(preempt) >= 2 else None,
            "prefix_hit_pct": 100.0 * hit_delta / query_delta if query_delta and hit_delta is not None else None,
            "gpu_util_mean_pct": mean([x["util"] for x in gs]),
            "gpu_util_max_pct": max_or_none([x["util"] for x in gs]),
            "gpu_mem_mean_pct": mean([100.0 * x["mem_used"] / x["mem_total"] for x in gs if x["mem_used"] is not None and x["mem_total"]]),
            "gpu_mem_max_pct": max_or_none([100.0 * x["mem_used"] / x["mem_total"] for x in gs if x["mem_used"] is not None and x["mem_total"]]),
            "gpu_mem_max_mib": max_or_none([x["mem_used"] for x in gs]),
            "gpu_power_mean_w": mean([x["power"] for x in gs]),
            "gpu_power_max_w": max_or_none([x["power"] for x in gs]),
            "gpu_temp_max_c": max_or_none([x["temp"] for x in gs]),
            "gpu_clock_sm_min_mhz": min_or_none([x["clock_sm"] for x in gs]),
            "gpu_clock_sm_max_mhz": max_or_none([x["clock_sm"] for x in gs]),
            "gpu_clock_mem_min_mhz": min_or_none([x["clock_mem"] for x in gs]),
            "gpu_clock_mem_max_mhz": max_or_none([x["clock_mem"] for x in gs]),
            "prom_samples": len(ps),
            "gpu_samples": len(gs),
        })
    result = {}
    for concurrency, runs in per_concurrency.items():
        keys = runs[0].keys()
        result[concurrency] = {key: median([run[key] for run in runs]) for key in keys}
    return result


def load_experiment(campaign, workload, phase, name):
    root = campaign / "runs" / name
    with (root / "summary.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    repeats = [row for row in csv_rows if row["row_type"] == "repeat"]
    aggregates = {int(row["concurrency"]): row for row in csv_rows if row["row_type"] == "aggregate"}
    telem = telemetry_by_concurrency(root)
    rows = []
    for concurrency in sorted(aggregates):
        agg = aggregates[concurrency]
        rr = [row for row in repeats if int(row["concurrency"]) == concurrency]
        output_tps = [f(row["output_token_goodput_per_s"]) for row in rr]
        p99 = [f(row["e2e_latency_s_p99"]) for row in rr]
        record = {
            "workload": workload,
            "phase": phase,
            "requests_per_repeat": int(rr[0]["attempted_requests"]),
            "concurrency": concurrency,
            "measured_repeats": len(rr),
            "attempted_requests": int(agg["attempted_requests"]),
            "successful_requests": int(agg["successful_requests"]),
            "failed_requests": int(agg["failed_requests"]),
            "request_throughput_median_per_s": median([f(row["request_goodput_per_s"]) for row in rr]),
            "prompt_token_throughput_median_per_s": median([f(row["prompt_token_goodput_per_s"]) for row in rr]),
            "output_token_throughput_median_per_s": median(output_tps),
            "output_token_throughput_cv_pct": cv_percent(output_tps),
            "output_token_throughput_min_per_s": min(output_tps),
            "output_token_throughput_max_per_s": max(output_tps),
            "e2e_p50_s": f(agg["e2e_latency_s_p50"]),
            "e2e_p90_s": f(agg["e2e_latency_s_p90"]),
            "e2e_p99_s": f(agg["e2e_latency_s_p99"]),
            "e2e_p99_repeat_median_s": median(p99),
            "e2e_p99_cv_pct": cv_percent(p99),
            "ttft_p50_s": f(agg["ttft_s_p50"]),
            "ttft_p90_s": f(agg["ttft_s_p90"]),
            "ttft_p99_s": f(agg["ttft_s_p99"]),
            "approx_tpot_p50_s": f(agg["approx_tpot_s_p50"]),
            "approx_tpot_p90_s": f(agg["approx_tpot_s_p90"]),
            "approx_tpot_p99_s": f(agg["approx_tpot_s_p99"]),
        }
        record.update(telem.get(concurrency, {}))
        rows.append(record)
    transitions = []
    for previous, current in zip(rows, rows[1:]):
        old_tps = previous["output_token_throughput_median_per_s"]
        new_tps = current["output_token_throughput_median_per_s"]
        old_p99 = previous["e2e_p99_repeat_median_s"]
        new_p99 = current["e2e_p99_repeat_median_s"]
        transitions.append({
            "workload": workload,
            "phase": phase,
            "from_concurrency": previous["concurrency"],
            "to_concurrency": current["concurrency"],
            "median_output_throughput_change_pct": 100.0 * (new_tps - old_tps) / old_tps,
            "median_p99_latency_change_pct": 100.0 * (new_p99 - old_p99) / old_p99,
            "meets_transition_criterion": (new_tps - old_tps) / old_tps < 0.05 and (new_p99 - old_p99) / old_p99 >= 0.20,
        })
    return rows, transitions


def fmt(value, digits=2):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def main():
    campaign = Path(sys.argv[1]).resolve()
    output = campaign / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    transitions = []
    for workload, phase, name in EXPERIMENTS:
        experiment_rows, experiment_transitions = load_experiment(campaign, workload, phase, name)
        rows.extend(experiment_rows)
        transitions.extend(experiment_transitions)
    with (output / "detailed-results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "campaign-analysis.json").write_text(json.dumps({"rows": rows, "transitions": transitions}, indent=2) + "\n")

    lines = ["# Capacity range campaign analysis", "", "All throughput values are medians across three measured repeats. Latency percentiles are pooled from raw measured requests; adjacent P99 changes use the median repeat P99. Warmups are excluded.", ""]
    for workload, phase, _ in EXPERIMENTS:
        selected = [row for row in rows if row["workload"] == workload and row["phase"] == phase]
        lines += [f"## {workload} — {phase}", "", "| C | ok/fail | req/s | prompt tok/s | output tok/s | output CV | E2E p50/p90/p99 s | TTFT p50/p90/p99 s | TPOT p50/p90/p99 ms | running mean/max | waiting mean/max | KV mean/max | GPU mean/max | mem max | temp max | SM clock min–max | power mean/max |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in selected:
            lines.append("| {c} | {ok}/{fail} | {req} | {prompt} | {output} | {cv}% | {e50}/{e90}/{e99} | {t50}/{t90}/{t99} | {p50}/{p90}/{p99} | {rmean}/{rmax} | {wmean}/{wmax} | {kvmean}%/{kvmax}% | {gmean}%/{gmax}% | {mem}% | {temp}°C | {cmin}–{cmax} | {pmean}/{pmax} W |".format(
                c=row["concurrency"], ok=row["successful_requests"], fail=row["failed_requests"], req=fmt(row["request_throughput_median_per_s"]), prompt=fmt(row["prompt_token_throughput_median_per_s"], 0), output=fmt(row["output_token_throughput_median_per_s"], 1), cv=fmt(row["output_token_throughput_cv_pct"]), e50=fmt(row["e2e_p50_s"], 3), e90=fmt(row["e2e_p90_s"], 3), e99=fmt(row["e2e_p99_s"], 3), t50=fmt(row["ttft_p50_s"], 3), t90=fmt(row["ttft_p90_s"], 3), t99=fmt(row["ttft_p99_s"], 3), p50=fmt(1000*row["approx_tpot_p50_s"], 2), p90=fmt(1000*row["approx_tpot_p90_s"], 2), p99=fmt(1000*row["approx_tpot_p99_s"], 2), rmean=fmt(row.get("running_mean"), 1), rmax=fmt(row.get("running_max"), 0), wmean=fmt(row.get("waiting_mean"), 1), wmax=fmt(row.get("waiting_max"), 0), kvmean=fmt(row.get("kv_mean_pct"), 1), kvmax=fmt(row.get("kv_max_pct"), 1), gmean=fmt(row.get("gpu_util_mean_pct"), 1), gmax=fmt(row.get("gpu_util_max_pct"), 0), mem=fmt(row.get("gpu_mem_max_pct"), 1), temp=fmt(row.get("gpu_temp_max_c"), 0), cmin=fmt(row.get("gpu_clock_sm_min_mhz"), 0), cmax=fmt(row.get("gpu_clock_sm_max_mhz"), 0), pmean=fmt(row.get("gpu_power_mean_w"), 1), pmax=fmt(row.get("gpu_power_max_w"), 1)))
        lines += ["", "Adjacent transitions:", "", "| From→to | output throughput Δ | median repeat P99 Δ | criterion |", "| ---: | ---: | ---: | :---: |"]
        for tr in [x for x in transitions if x["workload"] == workload and x["phase"] == phase]:
            lines.append(f"| {tr['from_concurrency']}→{tr['to_concurrency']} | {tr['median_output_throughput_change_pct']:.2f}% | {tr['median_p99_latency_change_pct']:.2f}% | {'yes' if tr['meets_transition_criterion'] else 'no'} |")
        lines.append("")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(output / "report.md")


if __name__ == "__main__":
    main()
