#!/usr/bin/env python3
"""Build a scoped roofline-style report from preserved benchmark evidence."""

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "serving-inference-matplotlib")
)

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt

import summarize_results


matplotlib.rcParams["svg.hashsalt"] = "serving-inference-roofline"

INPUT_SCHEMA_VERSION = "1.0"
OUTPUT_SCHEMA_VERSION = "1.0"
SUPPORTED_FORMATS = ("png", "pdf", "svg")
PHASE_RATE_METRICS = {
    "prefill": "prompt_token_goodput_per_s",
    "decode": "output_token_goodput_per_s",
}
METHOD_MARKERS = {"counter": "o", "estimate": "^"}
METHOD_COLORS = {"counter": "#0072B2", "estimate": "#E69F00"}


class RooflineDataError(ValueError):
    """Raised when roofline inputs cannot be interpreted without guessing."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _positive(value, field):
    if not _is_number(value) or value <= 0:
        raise RooflineDataError("%s must be a finite positive number" % field)
    return float(value)


def _nonnegative(value, field):
    if not _is_number(value) or value < 0:
        raise RooflineDataError("%s must be a finite non-negative number" % field)
    return float(value)


def _positive_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RooflineDataError("%s must be a positive integer" % field)
    return value


def _string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise RooflineDataError("%s must be a non-empty string" % field)
    return value


def _source(value, field):
    if not isinstance(value, dict):
        raise RooflineDataError("%s must be an object" % field)
    result = {
        "url": _string(value.get("url"), field + ".url"),
        "accessed_at": _string(
            value.get("accessed_at"), field + ".accessed_at"
        ),
    }
    related_urls = value.get("related_urls", [])
    if not isinstance(related_urls, list) or any(
        not isinstance(item, str) or not item.strip() for item in related_urls
    ):
        raise RooflineDataError("%s.related_urls must be an array of URLs" % field)
    result["related_urls"] = related_urls
    return result


def load_config(path):
    path = Path(path).resolve()
    try:
        payload = path.read_bytes()
        config = json.loads(payload.decode("utf-8"))
    except FileNotFoundError as error:
        raise RooflineDataError("config does not exist: %s" % path) from error
    except UnicodeDecodeError as error:
        raise RooflineDataError("config is not valid UTF-8: %s" % path) from error
    except json.JSONDecodeError as error:
        raise RooflineDataError("invalid JSON in %s: %s" % (path, error)) from error
    if not isinstance(config, dict):
        raise RooflineDataError("config must be a JSON object")
    return path, config, hashlib.sha256(payload).hexdigest()


def validate_config(config):
    if config.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise RooflineDataError(
            "unsupported input schema_version: %r" % config.get("schema_version")
        )
    _string(config.get("analysis"), "analysis")
    device = config.get("device")
    if not isinstance(device, dict):
        raise RooflineDataError("device must be an object")
    _string(device.get("name"), "device.name")
    bandwidth = device.get("memory_bandwidth")
    if not isinstance(bandwidth, dict):
        raise RooflineDataError("device.memory_bandwidth must be an object")
    _positive(bandwidth.get("value"), "device.memory_bandwidth.value")
    if bandwidth.get("unit") != "GB/s":
        raise RooflineDataError("device.memory_bandwidth.unit must be 'GB/s' (SI)")
    _string(bandwidth.get("derivation"), "device.memory_bandwidth.derivation")
    _source(bandwidth.get("source"), "device.memory_bandwidth.source")

    ceilings = device.get("compute_ceilings")
    if not isinstance(ceilings, list) or not ceilings:
        raise RooflineDataError("device.compute_ceilings must be a non-empty array")
    ceiling_ids = set()
    for index, ceiling in enumerate(ceilings):
        field = "device.compute_ceilings[%d]" % index
        if not isinstance(ceiling, dict):
            raise RooflineDataError("%s must be an object" % field)
        ceiling_id = _string(ceiling.get("id"), field + ".id")
        if ceiling_id in ceiling_ids:
            raise RooflineDataError("duplicate compute ceiling id: %s" % ceiling_id)
        ceiling_ids.add(ceiling_id)
        _string(ceiling.get("label"), field + ".label")
        _positive(ceiling.get("value"), field + ".value")
        if ceiling.get("unit") != "TFLOP/s":
            raise RooflineDataError("%s.unit must be 'TFLOP/s' (SI)" % field)
        _string(ceiling.get("dtype"), field + ".dtype")
        _string(ceiling.get("execution"), field + ".execution")
        _string(ceiling.get("derivation"), field + ".derivation")
        _source(ceiling.get("source"), field + ".source")

    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RooflineDataError("cases must be a non-empty array")
    case_ids = set()
    for index, case in enumerate(cases):
        field = "cases[%d]" % index
        if not isinstance(case, dict):
            raise RooflineDataError("%s must be an object" % field)
        case_id = _string(case.get("id"), field + ".id")
        if case_id in case_ids:
            raise RooflineDataError("duplicate case id: %s" % case_id)
        case_ids.add(case_id)
        phase = case.get("phase")
        if phase not in PHASE_RATE_METRICS:
            raise RooflineDataError("%s.phase must be 'prefill' or 'decode'" % field)
        _string(case.get("workload"), field + ".workload")
        _positive_int(case.get("concurrency"), field + ".concurrency")
        _string(case.get("unprofiled_experiment"), field + ".unprofiled_experiment")
        if case.get("rate_metric") != PHASE_RATE_METRICS[phase]:
            raise RooflineDataError(
                "%s.rate_metric must be %r for %s"
                % (field, PHASE_RATE_METRICS[phase], phase)
            )
        if case.get("compute_ceiling") not in ceiling_ids:
            raise RooflineDataError("%s references an unknown compute ceiling" % field)
        _validate_measurement(case.get("measurement"), field + ".measurement")


def _validate_measurement(measurement, field):
    if not isinstance(measurement, dict):
        raise RooflineDataError("%s must be an object" % field)
    kind = measurement.get("kind")
    if kind not in {"counter", "estimate"}:
        raise RooflineDataError("%s.kind must be 'counter' or 'estimate'" % field)
    status = measurement.get("status")
    if status not in {"available", "unavailable"}:
        raise RooflineDataError(
            "%s.status must be 'available' or 'unavailable'" % field
        )
    _string(measurement.get("scope"), field + ".scope")
    if status == "unavailable":
        _string(measurement.get("reason"), field + ".reason")
        return
    if kind == "estimate":
        _positive(measurement.get("flops_per_token"), field + ".flops_per_token")
        _positive(
            measurement.get("dram_bytes_per_token"),
            field + ".dram_bytes_per_token",
        )
        _string(measurement.get("rationale"), field + ".rationale")
        _source(measurement.get("source"), field + ".source")
        return

    _string(measurement.get("profile_metadata"), field + ".profile_metadata")
    _string(measurement.get("profile_report"), field + ".profile_report")
    _positive_int(measurement.get("phase_tokens"), field + ".phase_tokens")
    dram_counters = measurement.get("dram_counters")
    flop_counters = measurement.get("flop_counters")
    if not isinstance(dram_counters, list) or not dram_counters:
        raise RooflineDataError("%s.dram_counters must be a non-empty array" % field)
    if not isinstance(flop_counters, list) or not flop_counters:
        raise RooflineDataError("%s.flop_counters must be a non-empty array" % field)
    for index, counter in enumerate(dram_counters):
        counter_field = "%s.dram_counters[%d]" % (field, index)
        _validate_counter(counter, counter_field)
        if counter.get("unit") != "byte":
            raise RooflineDataError("%s.unit must be 'byte'" % counter_field)
        if counter.get("bytes_per_event") != 1:
            raise RooflineDataError("%s.bytes_per_event must be 1" % counter_field)
    for index, counter in enumerate(flop_counters):
        counter_field = "%s.flop_counters[%d]" % (field, index)
        _validate_counter(counter, counter_field)
        if counter.get("unit") != "event":
            raise RooflineDataError("%s.unit must be 'event'" % counter_field)
        _positive(counter.get("flops_per_event"), counter_field + ".flops_per_event")
        _string(counter.get("execution"), counter_field + ".execution")


def _validate_counter(counter, field):
    if not isinstance(counter, dict):
        raise RooflineDataError("%s must be an object" % field)
    _string(counter.get("name"), field + ".name")
    _nonnegative(counter.get("value"), field + ".value")


def resolve_repo_path(repo_root, value):
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _source_artifact(path, display_path):
    path = Path(path)
    if not path.is_file():
        raise RooflineDataError("source artifact does not exist: %s" % path)
    return {"path": str(display_path), "sha256": sha256_file(path)}


def _load_unprofiled(case, repo_root):
    experiment_value = case["unprofiled_experiment"]
    experiment = resolve_repo_path(repo_root, experiment_value)
    try:
        rows = summarize_results.build_rows(experiment)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise RooflineDataError(
            "case %s could not rebuild normalized rows: %s" % (case["id"], error)
        ) from error
    matching = [
        row
        for row in rows
        if row.get("row_type") == "aggregate"
        and row.get("workload") == case["workload"]
        and row.get("concurrency") == case["concurrency"]
    ]
    if len(matching) != 1:
        raise RooflineDataError(
            "case %s expected one aggregate row for %s at concurrency %d; found %d"
            % (case["id"], case["workload"], case["concurrency"], len(matching))
        )
    aggregate = matching[0]
    if aggregate.get("schema_compatible") is not True:
        raise RooflineDataError(
            "case %s has schema-incompatible unprofiled evidence: %s"
            % (case["id"], aggregate.get("issues", ""))
        )
    if aggregate.get("complete") is not True:
        raise RooflineDataError(
            "case %s has incomplete unprofiled evidence: %s"
            % (case["id"], aggregate.get("issues", ""))
        )
    rate = aggregate.get(case["rate_metric"])
    _positive(rate, "case %s unprofiled token rate" % case["id"])
    repeats = [
        row
        for row in rows
        if row.get("row_type") == "repeat"
        and row.get("workload") == case["workload"]
        and row.get("concurrency") == case["concurrency"]
    ]
    repeat_rates = [row.get(case["rate_metric"]) for row in repeats]
    if not repeat_rates or any(
        not _is_number(value) or value <= 0 for value in repeat_rates
    ):
        raise RooflineDataError(
            "case %s has missing repeat-level unprofiled token rates" % case["id"]
        )

    artifacts = [
        _source_artifact(
            experiment / "manifest.json", Path(experiment_value) / "manifest.json"
        )
    ]
    for row in sorted(repeats, key=lambda item: item["repeat"]):
        relative = (
            Path("repeat-%02d" % row["repeat"])
            / ("concurrency-%03d.jsonl" % case["concurrency"])
        )
        artifacts.append(
            _source_artifact(experiment / relative, Path(experiment_value) / relative)
        )
    return {
        "experiment": aggregate["experiment"],
        "workload": case["workload"],
        "concurrency": case["concurrency"],
        "rate_metric": case["rate_metric"],
        "token_rate": {"value": float(rate), "unit": "token/s"},
        "repeat_token_rates": [float(value) for value in repeat_rates],
        "repeat_mean_token_rate": float(statistics.mean(repeat_rates)),
        "derivation": (
            "aggregate successful phase tokens / sum of measured repeat durations; "
            "rebuilt from manifest and raw measured JSONL"
        ),
        "artifacts": artifacts,
    }


def _load_profile_metadata(path, case):
    try:
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RooflineDataError(
            "case %s could not read profile metadata: %s" % (case["id"], error)
        ) from error
    if metadata.get("schema_version") != "1.0":
        raise RooflineDataError("case %s has unsupported profile metadata" % case["id"])
    if metadata.get("tool") != "ncu":
        raise RooflineDataError("case %s profile metadata tool must be ncu" % case["id"])
    if metadata.get("status") != "complete":
        raise RooflineDataError("case %s profile metadata is not complete" % case["id"])
    if metadata.get("workload") != case["workload"]:
        raise RooflineDataError("case %s profile workload does not match" % case["id"])
    if metadata.get("concurrency") != case["concurrency"]:
        raise RooflineDataError("case %s profile concurrency does not match" % case["id"])
    _string(metadata.get("profile_name"), "case %s profile_name" % case["id"])
    _string(metadata.get("tool_version"), "case %s tool_version" % case["id"])
    return metadata


def _measurement(case, repo_root):
    measurement = case["measurement"]
    result = {
        "kind": measurement["kind"],
        "status": measurement["status"],
        "scope": measurement["scope"],
    }
    if measurement["status"] == "unavailable":
        result["reason"] = measurement["reason"]
        return result

    if measurement["kind"] == "estimate":
        result.update(
            {
                "flops_per_token": float(measurement["flops_per_token"]),
                "dram_bytes_per_token": float(measurement["dram_bytes_per_token"]),
                "rationale": measurement["rationale"],
                "source": measurement["source"],
                "artifacts": [],
            }
        )
        return result

    metadata_value = measurement["profile_metadata"]
    report_value = measurement["profile_report"]
    metadata_path = resolve_repo_path(repo_root, metadata_value)
    report_path = resolve_repo_path(repo_root, report_value)
    metadata = _load_profile_metadata(metadata_path, case)
    artifacts = [
        _source_artifact(metadata_path, metadata_value),
        _source_artifact(report_path, report_value),
    ]
    dram_bytes = sum(
        float(counter["value"]) * counter["bytes_per_event"]
        for counter in measurement["dram_counters"]
    )
    flops = sum(
        float(counter["value"]) * counter["flops_per_event"]
        for counter in measurement["flop_counters"]
    )
    phase_tokens = float(measurement["phase_tokens"])
    result.update(
        {
            "profile_name": metadata.get("profile_name"),
            "tool_version": metadata.get("tool_version"),
            "phase_tokens": phase_tokens,
            "dram_counters": measurement["dram_counters"],
            "flop_counters": measurement["flop_counters"],
            "total_dram_bytes": dram_bytes,
            "total_flops": flops,
            "flops_per_token": flops / phase_tokens,
            "dram_bytes_per_token": dram_bytes / phase_tokens,
            "artifacts": artifacts,
        }
    )
    return result


def arithmetic_intensity(flops, dram_bytes):
    """Return FLOP/byte using counter or estimate values in base SI units."""
    return _positive(flops, "flops") / _positive(dram_bytes, "dram_bytes")


def achieved_tflops(token_rate, flops_per_token):
    """Scale an unprofiled token/s rate by representative FLOP/token."""
    return (
        _positive(token_rate, "token_rate")
        * _positive(flops_per_token, "flops_per_token")
        / 1e12
    )


def bandwidth_ceiling_tflops(arithmetic_intensity_flop_per_byte, bandwidth_gb_s):
    """Convert FLOP/byte × SI GB/s to SI TFLOP/s."""
    return (
        _positive(arithmetic_intensity_flop_per_byte, "arithmetic_intensity")
        * _positive(bandwidth_gb_s, "bandwidth_gb_s")
        / 1000.0
    )


def roofline_ceiling_tflops(arithmetic_intensity_value, bandwidth_gb_s, compute_tflops):
    return min(
        bandwidth_ceiling_tflops(arithmetic_intensity_value, bandwidth_gb_s),
        _positive(compute_tflops, "compute_tflops"),
    )


def build_analysis(config, config_path, config_sha256, repo_root):
    validate_config(config)
    ceilings = {item["id"]: item for item in config["device"]["compute_ceilings"]}
    bandwidth = float(config["device"]["memory_bandwidth"]["value"])
    analyzed_cases = []
    diagnostics = []
    for case in config["cases"]:
        unprofiled = _load_unprofiled(case, repo_root)
        measurement = _measurement(case, repo_root)
        output = {
            "id": case["id"],
            "phase": case["phase"],
            "compute_ceiling": case["compute_ceiling"],
            "unprofiled": unprofiled,
            "measurement": measurement,
            "derived": None,
        }
        if measurement["status"] == "unavailable":
            diagnostics.append(
                "%s_%s_unavailable: %s"
                % (case["id"], measurement["kind"], measurement["reason"])
            )
        else:
            intensity = arithmetic_intensity(
                measurement["flops_per_token"], measurement["dram_bytes_per_token"]
            )
            achieved = achieved_tflops(
                unprofiled["token_rate"]["value"], measurement["flops_per_token"]
            )
            memory_ceiling = bandwidth_ceiling_tflops(intensity, bandwidth)
            compute_ceiling = float(ceilings[case["compute_ceiling"]]["value"])
            roof_ceiling = min(memory_ceiling, compute_ceiling)
            if achieved > roof_ceiling:
                diagnostics.append(
                    "%s_achieved_exceeds_selected_roof: review counter scope, "
                    "FLOP multipliers, dtype/ceiling choice, and unprofiled pairing"
                    % case["id"]
                )
            output["derived"] = {
                "flops_per_token": {
                    "value": measurement["flops_per_token"],
                    "unit": "FLOP/token",
                },
                "dram_bytes_per_token": {
                    "value": measurement["dram_bytes_per_token"],
                    "unit": "byte/token",
                },
                "arithmetic_intensity": {"value": intensity, "unit": "FLOP/byte"},
                "achieved_performance": {"value": achieved, "unit": "TFLOP/s"},
                "memory_roof_at_intensity": {
                    "value": memory_ceiling,
                    "unit": "TFLOP/s",
                },
                "selected_roof_ceiling": {"value": roof_ceiling, "unit": "TFLOP/s"},
                "fraction_of_selected_roof": achieved / roof_ceiling,
            }
        analyzed_cases.append(output)

    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis": config["analysis"],
        "scope": (
            "roofline-style selected-kernel/phase analysis; not a whole-model "
            "classical roofline claim"
        ),
        "input": {"path": str(config_path), "sha256": config_sha256},
        "device": config["device"],
        "formulas": {
            "arithmetic_intensity": "FLOP/token / byte/token = FLOP/byte",
            "achieved_performance": "unprofiled token/s * FLOP/token / 1e12 = TFLOP/s",
            "memory_roof": "FLOP/byte * GB/s / 1000 = TFLOP/s (SI prefixes)",
            "selected_roof": "min(memory roof, matching theoretical compute ceiling)",
        },
        "cases": analyzed_cases,
        "diagnostics": diagnostics,
    }


def _format_number(value):
    if value is None:
        return "unavailable"
    return format(value, ".6g")


def render_markdown(analysis):
    device = analysis["device"]
    lines = [
        "# Roofline-style analysis: %s" % analysis["analysis"],
        "",
        "> **Scope:** This is a phase/selected-kernel roofline-style analysis. It is not a whole-model classical roofline claim.",
        "",
        "## Device ceilings",
        "",
        "Theoretical memory bandwidth: **%s GB/s** (SI). %s"
        % (
            _format_number(device["memory_bandwidth"]["value"]),
            device["memory_bandwidth"]["derivation"],
        ),
        "",
        "| Ceiling | Value | Dtype | Execution path |",
        "| --- | ---: | --- | --- |",
    ]
    for ceiling in device["compute_ceilings"]:
        lines.append(
            "| %s | %s TFLOP/s | %s | %s |"
            % (
                ceiling["label"],
                _format_number(ceiling["value"]),
                ceiling["dtype"],
                ceiling["execution"],
            )
        )
    memory_source = device["memory_bandwidth"]["source"]
    if analysis["diagnostics"]:
        lines.extend(["", "## Diagnostics", ""])
        lines.extend("- %s" % item for item in analysis["diagnostics"])

    lines.extend(
        [
            "",
            "Ceiling provenance:",
            "",
            "- Memory bandwidth: %s (accessed %s)."
            % (memory_source["url"], memory_source["accessed_at"]),
        ]
    )
    for ceiling in device["compute_ceilings"]:
        source = ceiling["source"]
        lines.append(
            "- %s: %s. Source: %s (accessed %s)."
            % (
                ceiling["label"],
                ceiling["derivation"],
                source["url"],
                source["accessed_at"],
            )
        )
    lines.extend(
        [
            "",
            "## Formulas and units",
            "",
            "- Arithmetic intensity: `FLOP/token ÷ byte/token = FLOP/byte`.",
            "- Achieved performance: `unprofiled token/s × FLOP/token ÷ 1e12 = TFLOP/s`.",
            "- Memory roof: `FLOP/byte × GB/s ÷ 1000 = TFLOP/s` using SI prefixes.",
            "- Selected roof: `min(memory roof, matching theoretical compute ceiling)`.",
            "",
            "The token rate is rebuilt from the unprofiled experiment manifest and raw measured JSONL. Profiled client latency and throughput are never used.",
            "",
            "## Case summary",
            "",
            "| Phase | Case | Method | Unprofiled token rate | Arithmetic intensity | Achieved performance | Selected roof |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in analysis["cases"]:
        derived = case["derived"]
        lines.append(
            "| %s | %s | %s | %s token/s | %s | %s | %s |"
            % (
                case["phase"],
                case["id"],
                case["measurement"]["kind"],
                _format_number(case["unprofiled"]["token_rate"]["value"]),
                (
                    _format_number(derived["arithmetic_intensity"]["value"])
                    + " FLOP/byte"
                    if derived
                    else "unavailable"
                ),
                (
                    _format_number(derived["achieved_performance"]["value"])
                    + " TFLOP/s"
                    if derived
                    else "unavailable"
                ),
                (
                    _format_number(derived["selected_roof_ceiling"]["value"])
                    + " TFLOP/s"
                    if derived
                    else "unavailable"
                ),
            )
        )

    for case in analysis["cases"]:
        measurement = case["measurement"]
        unprofiled = case["unprofiled"]
        lines.extend(
            [
                "",
                "## %s: %s" % (case["phase"].capitalize(), case["id"]),
                "",
                "Unprofiled evidence: `%s`, workload `%s`, concurrency %d. Rebuilt %s = **%s token/s** from %d measured repeats."
                % (
                    unprofiled["experiment"],
                    unprofiled["workload"],
                    unprofiled["concurrency"],
                    unprofiled["rate_metric"],
                    _format_number(unprofiled["token_rate"]["value"]),
                    len(unprofiled["repeat_token_rates"]),
                ),
                "",
                "Measurement method: **%s**. Scope: %s"
                % (measurement["kind"], measurement["scope"]),
            ]
        )
        if measurement["status"] == "unavailable":
            lines.extend(["", "Result unavailable: %s" % measurement["reason"]])
            continue
        if measurement["kind"] == "counter":
            lines.extend(
                [
                    "",
                    "Profile: `%s`; tool version: `%s`; phase tokens in counter scope: %s."
                    % (
                        measurement["profile_name"],
                        measurement["tool_version"],
                        _format_number(measurement["phase_tokens"]),
                    ),
                    "",
                    "Representative Nsight Compute counters:",
                    "",
                    "| Counter | Raw value | Unit | Conversion |",
                    "| --- | ---: | --- | --- |",
                ]
            )
            for counter in measurement["dram_counters"]:
                lines.append(
                    "| `%s` | %s | byte | × %s byte/event |"
                    % (
                        counter["name"],
                        _format_number(counter["value"]),
                        _format_number(counter["bytes_per_event"]),
                    )
                )
            for counter in measurement["flop_counters"]:
                lines.append(
                    "| `%s` | %s | event | × %s FLOP/event (%s) |"
                    % (
                        counter["name"],
                        _format_number(counter["value"]),
                        _format_number(counter["flops_per_event"]),
                        counter["execution"],
                    )
                )
        else:
            lines.extend(
                [
                    "",
                    "Estimate rationale: %s" % measurement["rationale"],
                    "",
                    "Estimate source: %s (accessed %s)."
                    % (
                        measurement["source"]["url"],
                        measurement["source"]["accessed_at"],
                    ),
                ]
            )
        derived = case["derived"]
        lines.extend(
            [
                "",
                "Derived point: **%s FLOP/byte**, **%s TFLOP/s achieved**; selected roof **%s TFLOP/s**."
                % (
                    _format_number(derived["arithmetic_intensity"]["value"]),
                    _format_number(derived["achieved_performance"]["value"]),
                    _format_number(derived["selected_roof_ceiling"]["value"]),
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Prefill and decode are reported separately because their operation mix and memory behavior differ.",
            "- Counter points describe only the recorded Nsight Compute scope. Kernel filtering/replay may omit model work.",
            "- Estimate points are visibly separate from hardware-counter points and depend on their stated assumptions.",
            "- Theoretical ceilings are applicable only when dtype, tensor-core use, clocks, and execution path match the recorded case.",
            "- Host, scheduler, network, launch, cache, and framework overheads are outside a classical device roofline model.",
            "",
        ]
    )
    return "\n".join(lines)


def _roof_curve(x_values, bandwidth, compute):
    return [min(bandwidth * x / 1000.0, compute) for x in x_values]


def render_figure(analysis, output_dir, formats):
    invalid = sorted(set(formats) - set(SUPPORTED_FORMATS))
    if not formats or invalid:
        raise RooflineDataError(
            "formats must contain one or more of %s" % ", ".join(SUPPORTED_FORMATS)
        )
    phases = ("prefill", "decode")
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), squeeze=False)
    bandwidth = float(analysis["device"]["memory_bandwidth"]["value"])
    ceilings = {
        item["id"]: item for item in analysis["device"]["compute_ceilings"]
    }
    for ax, phase in zip(axes[0], phases):
        cases = [case for case in analysis["cases"] if case["phase"] == phase]
        available = [case for case in cases if case["derived"] is not None]
        intensities = [
            case["derived"]["arithmetic_intensity"]["value"] for case in available
        ]
        achieved_values = [
            case["derived"]["achieved_performance"]["value"] for case in available
        ]
        x_min = min(intensities) / 10 if intensities else 0.01
        x_max = max(intensities) * 10 if intensities else 1000.0
        x_min = max(x_min, 1e-4)
        x_max = max(x_max, x_min * 100)
        log_min = math.log10(x_min)
        log_max = math.log10(x_max)
        x_values = [10 ** (log_min + (log_max - log_min) * i / 199) for i in range(200)]
        used_ceilings = sorted({case["compute_ceiling"] for case in cases})
        for ceiling_id in used_ceilings:
            ceiling = ceilings[ceiling_id]
            ax.plot(
                x_values,
                _roof_curve(x_values, bandwidth, float(ceiling["value"])),
                linewidth=1.8,
                label="roof: %s" % ceiling["label"],
            )
        for case in available:
            derived = case["derived"]
            kind = case["measurement"]["kind"]
            ax.scatter(
                [derived["arithmetic_intensity"]["value"]],
                [derived["achieved_performance"]["value"]],
                marker=METHOD_MARKERS[kind],
                color=METHOD_COLORS[kind],
                s=60,
                zorder=5,
                label="%s (%s)" % (case["id"], kind),
            )
        unavailable = len(cases) - len(available)
        if unavailable:
            ax.text(
                0.02,
                0.02,
                "%d unavailable case(s); see report" % unavailable,
                transform=ax.transAxes,
                fontsize=8,
                color="crimson",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_min, x_max)
        y_min = min(achieved_values) / 10 if achieved_values else 1e-8
        used_compute_values = [float(ceilings[item]["value"]) for item in used_ceilings]
        y_max = max(used_compute_values + achieved_values + [y_min * 100]) * 2
        ax.set_ylim(max(y_min, 1e-300), y_max)
        ax.set_title(phase.capitalize(), loc="left", fontweight="semibold")
        ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
        ax.set_ylabel("Achieved performance (TFLOP/s)")
        ax.grid(True, which="both", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=7)
    figure.suptitle(
        "Roofline-style selected-kernel/phase analysis — %s" % analysis["device"]["name"],
        fontsize=14,
    )
    figure.text(
        0.01,
        0.008,
        "Token rates: unprofiled raw experiment evidence. Circles: counters; triangles: estimates. Not a whole-model classical roofline.",
        fontsize=7.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    outputs = []
    for output_format in formats:
        path = output_dir / ("roofline.%s" % output_format)
        metadata = {"Creator": "serving-inference scripts/analyze_roofline.py"}
        if output_format == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        elif output_format == "svg":
            metadata["Date"] = None
        figure.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )
        if not path.is_file() or path.stat().st_size == 0:
            raise RooflineDataError("renderer produced an empty figure: %s" % path)
        outputs.append(path)
    plt.close(figure)
    return outputs


def write_outputs(analysis, output_dir, formats=("png", "pdf")):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "roofline-analysis.json"
    report_path = output_dir / "roofline-report.md"
    json_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(render_markdown(analysis), encoding="utf-8")
    figure_paths = render_figure(analysis, output_dir, tuple(formats))
    return [json_path, report_path] + figure_paths


def _parse_formats(value):
    formats = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    invalid = sorted(set(formats) - set(SUPPORTED_FORMATS))
    if not formats or invalid:
        raise argparse.ArgumentTypeError(
            "formats must contain one or more of %s" % ", ".join(SUPPORTED_FORMATS)
        )
    return formats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Roofline analysis input JSON.")
    parser.add_argument(
        "--output",
        help="output directory (default: results/roofline/<analysis>)",
    )
    parser.add_argument(
        "--formats",
        type=_parse_formats,
        default=("png", "pdf"),
        help="comma-separated figure formats: png,pdf,svg (default: png,pdf)",
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        config_path, config, config_sha256 = load_config(args.config)
        analysis = build_analysis(config, config_path, config_sha256, repo_root)
        output = (
            Path(args.output)
            if args.output
            else repo_root / "results" / "roofline" / config["analysis"]
        )
        outputs = write_outputs(analysis, output, args.formats)
    except (OSError, RooflineDataError) as error:
        parser.exit(2, "error: %s\n" % error)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
