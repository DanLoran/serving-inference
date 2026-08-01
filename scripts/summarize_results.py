"""Rebuild a normalized experiment CSV from raw JSONL and its manifest."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import send_requests


SUPPORTED_MANIFEST_SCHEMA = "1.0"
SUPPORTED_RESULT_SCHEMA = "1.0"
PERCENTILE_METRICS = (
    ("e2e_latency_s", "latency_s"),
    ("ttft_s", "ttft_s"),
    ("approx_tpot_s", "approx_time_per_output_token_s"),
)
REPEAT_STAT_METRICS = (
    "request_goodput_per_s",
    "output_token_goodput_per_s",
    "failure_rate",
)

FIELDNAMES = [
    "row_type",
    "experiment",
    "workload",
    "concurrency",
    "repeat",
    "repeat_count",
    "schema_compatible",
    "complete",
    "issues",
    "attempted_requests",
    "successful_requests",
    "failed_requests",
    "failure_rate",
    "duration_s",
    "request_goodput_per_s",
    "prompt_tokens",
    "output_tokens",
    "total_tokens",
    "prompt_token_goodput_per_s",
    "output_token_goodput_per_s",
    "total_token_goodput_per_s",
]
for prefix, _ in PERCENTILE_METRICS:
    FIELDNAMES.extend(
        [prefix + "_count", prefix + "_p50", prefix + "_p90", prefix + "_p99"]
    )
for metric in REPEAT_STAT_METRICS:
    FIELDNAMES.extend(
        [metric + "_stddev", metric + "_ci95_low", metric + "_ci95_high"]
    )


def _number(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def _read_manifest(root):
    path = root / "manifest.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path):
    rows = []
    issues = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                issues.append("invalid_json_line_%d" % line_number)
                continue
            if not isinstance(value, dict):
                issues.append("non_object_line_%d" % line_number)
                continue
            rows.append(value)
    return rows, issues


def _safe_sum(rows, field):
    values = [row.get(field) for row in rows]
    return sum(value for value in values if isinstance(value, (int, float)))


def _duration(rows):
    ended = [
        row.get("ended_offset_s")
        for row in rows
        if isinstance(row.get("ended_offset_s"), (int, float))
    ]
    return max(ended) if ended else None


def _rate(numerator, duration):
    return numerator / duration if duration else None


def _distribution_fields(target, prefix, values):
    distribution = send_requests.distribution(values)
    target[prefix + "_count"] = distribution["count"]
    for percentile in ("p50", "p90", "p99"):
        target[prefix + "_" + percentile] = distribution[percentile]


def _repeat_row(
    experiment,
    workload,
    concurrency,
    repeat,
    rows,
    run_issues,
    expected,
    run_duration=None,
):
    successful = [row for row in rows if row.get("status") == 200]
    failed = len(rows) - len(successful)
    duration = run_duration if run_duration is not None else _duration(rows)
    prompt_tokens = _safe_sum(successful, "prompt_tokens")
    output_tokens = _safe_sum(successful, "output_tokens")
    issues = set(run_issues)
    if len(rows) != expected:
        issues.add("run_request_count_%d_expected_%d" % (len(rows), expected))
    if duration is None:
        issues.add("missing_duration")
    missing_output = sum(row.get("output_tokens") is None for row in successful)
    if missing_output:
        issues.add("missing_output_tokens_%d" % missing_output)
    for _, source in PERCENTILE_METRICS:
        missing = sum(row.get(source) is None for row in successful)
        if missing:
            issues.add("missing_%s_%d" % (source, missing))
    result_schema_compatible = all(
        row.get("schema_version") == SUPPORTED_RESULT_SCHEMA for row in rows
    )
    schema_compatible = (
        result_schema_compatible
        and "incompatible_manifest_schema" not in run_issues
    )
    if not result_schema_compatible:
        issues.add("incompatible_result_schema")
    row = {
        "row_type": "repeat",
        "experiment": experiment,
        "workload": workload,
        "concurrency": concurrency,
        "repeat": repeat,
        "repeat_count": 1,
        "schema_compatible": schema_compatible,
        "complete": len(rows) == expected and not run_issues,
        "issues": ";".join(sorted(issues)),
        "attempted_requests": len(rows),
        "successful_requests": len(successful),
        "failed_requests": failed,
        "failure_rate": failed / len(rows) if rows else None,
        "duration_s": duration,
        "request_goodput_per_s": _rate(len(successful), duration),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "prompt_token_goodput_per_s": _rate(prompt_tokens, duration),
        "output_token_goodput_per_s": _rate(output_tokens, duration),
        "total_token_goodput_per_s": _rate(prompt_tokens + output_tokens, duration),
        "_observations": successful,
    }
    for prefix, source in PERCENTILE_METRICS:
        _distribution_fields(row, prefix, [item.get(source) for item in successful])
    return row


def _confidence(values):
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return None, None, None
    deviation = statistics.stdev(clean)
    margin = 1.96 * deviation / math.sqrt(len(clean))
    mean = statistics.mean(clean)
    return deviation, mean - margin, mean + margin


def _aggregate_row(rows):
    first = rows[0]
    attempted = sum(row["attempted_requests"] for row in rows)
    successful = sum(row["successful_requests"] for row in rows)
    failed = sum(row["failed_requests"] for row in rows)
    durations = [row["duration_s"] for row in rows]
    duration = sum(value for value in durations if value is not None)
    prompt_tokens = sum(row["prompt_tokens"] for row in rows)
    output_tokens = sum(row["output_tokens"] for row in rows)
    observations = [item for row in rows for item in row["_observations"]]
    issues = sorted(
        {
            issue
            for row in rows
            for issue in row["issues"].split(";")
            if issue
        }
    )
    result = {
        "row_type": "aggregate",
        "experiment": first["experiment"],
        "workload": first["workload"],
        "concurrency": first["concurrency"],
        "repeat": None,
        "repeat_count": len(rows),
        "schema_compatible": all(row["schema_compatible"] for row in rows),
        "complete": all(row["complete"] for row in rows),
        "issues": ";".join(issues),
        "attempted_requests": attempted,
        "successful_requests": successful,
        "failed_requests": failed,
        "failure_rate": failed / attempted if attempted else None,
        "duration_s": duration or None,
        "request_goodput_per_s": _rate(successful, duration),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "prompt_token_goodput_per_s": _rate(prompt_tokens, duration),
        "output_token_goodput_per_s": _rate(output_tokens, duration),
        "total_token_goodput_per_s": _rate(prompt_tokens + output_tokens, duration),
    }
    for prefix, source in PERCENTILE_METRICS:
        _distribution_fields(result, prefix, [item.get(source) for item in observations])
    for metric in REPEAT_STAT_METRICS:
        deviation, low, high = _confidence([row[metric] for row in rows])
        result[metric + "_stddev"] = deviation
        result[metric + "_ci95_low"] = low
        result[metric + "_ci95_high"] = high
    return result


def build_rows(root):
    root = Path(root)
    manifest = _read_manifest(root)
    resolved = manifest.get("config", {}).get("resolved", {})
    experiment = manifest.get("experiment", {}).get("name", root.name)
    concurrencies = resolved.get("concurrency", [])
    repeats = manifest.get("experiment", {}).get("repeats", resolved.get("repeats", 0))
    expected = resolved.get("num_requests", 0)
    manifest_issues = []
    if manifest.get("schema_version") != SUPPORTED_MANIFEST_SCHEMA:
        manifest_issues.append("incompatible_manifest_schema")
    if manifest.get("experiment", {}).get("status") != "completed":
        manifest_issues.append(
            "manifest_status_%s"
            % manifest.get("experiment", {}).get("status", "missing")
        )

    runs = {}
    workloads = set()
    for repeat in range(1, repeats + 1):
        for concurrency in concurrencies:
            path = (
                root
                / ("repeat-%02d" % repeat)
                / ("concurrency-%03d.jsonl" % concurrency)
            )
            if path.exists():
                raw_rows, issues = _read_jsonl(path)
                for raw in raw_rows:
                    workloads.add(raw.get("workload") or "unknown")
                runs[(concurrency, repeat)] = (raw_rows, issues)
            else:
                runs[(concurrency, repeat)] = ([], ["missing_raw_run"])
    if not workloads:
        workloads.add("unknown")

    repeat_rows = []
    for workload in sorted(workloads):
        for concurrency in sorted(concurrencies):
            for repeat in range(1, repeats + 1):
                raw_rows, issues = runs[(concurrency, repeat)]
                selected = [
                    row
                    for row in raw_rows
                    if (row.get("workload") or "unknown") == workload
                ]
                # Completeness is evaluated at run scope because mixed workload shares
                # are not declared in the experiment manifest.
                run_issues = list(manifest_issues) + list(issues)
                if len(raw_rows) != expected:
                    run_issues.append(
                        "run_request_count_%d_expected_%d"
                        % (len(raw_rows), expected)
                    )
                elif not selected:
                    run_issues.append("missing_workload_rows")
                repeat_rows.append(
                    _repeat_row(
                        experiment,
                        workload,
                        concurrency,
                        repeat,
                        selected,
                        run_issues,
                        len(selected) if len(raw_rows) == expected else expected,
                        run_duration=_duration(raw_rows),
                    )
                )

    aggregates = []
    for workload in sorted(workloads):
        for concurrency in sorted(concurrencies):
            group = [
                row for row in repeat_rows
                if row["workload"] == workload and row["concurrency"] == concurrency
            ]
            aggregates.append(_aggregate_row(group))
    return repeat_rows + aggregates


def write_csv(root, output=None):
    root = Path(root)
    output = Path(output or root / "summary.csv")
    rows = build_rows(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _number(row.get(field)) for field in FIELDNAMES})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        help="Experiment result directory containing manifest.json",
    )
    parser.add_argument("--output", help="CSV path (default: <experiment>/summary.csv)")
    args = parser.parse_args()
    try:
        path = write_csv(args.experiment, args.output)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, "error: %s\n" % error)
    print("Summary CSV: %s" % path)


if __name__ == "__main__":
    main()
