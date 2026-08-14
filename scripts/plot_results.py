"""Generate report-ready figures exclusively from preserved experiment artifacts."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "serving-inference-matplotlib")
)

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt

import summarize_results


matplotlib.rcParams["svg.hashsalt"] = "serving-inference"

PLOT_MANIFEST_SCHEMA = "1.0"
SUPPORTED_FORMATS = ("png", "pdf", "svg")
THROUGHPUT_METRICS = (
    ("request_goodput_per_s", "Request goodput", "requests/s"),
    ("prompt_token_goodput_per_s", "Prompt-token goodput", "tokens/s"),
    ("output_token_goodput_per_s", "Output-token goodput", "tokens/s"),
    ("total_token_goodput_per_s", "Total-token goodput", "tokens/s"),
)
LATENCY_METRICS = (
    ("e2e_latency_s", "End-to-end latency", "seconds"),
    ("ttft_s", "Client-observed TTFT", "seconds"),
    ("approx_tpot_s", "Approximate TPOT", "seconds/token"),
)
PERCENTILES = ("p50", "p90", "p99")
PERCENTILE_COLORS = {
    "p50": "#0072B2",
    "p90": "#E69F00",
    "p99": "#D55E00",
}
GPU_REQUIRED_FIELDS = {
    "sample_at_utc",
    "experiment_offset_s",
    "index",
    "uuid",
    "utilization_gpu_percent",
    "memory_used_mib",
    "memory_total_mib",
}
MISSING_VALUES = {"", "n/a", "[n/a]", "na", "none", "null", "nan"}
DEFINITIONS = {
    "request_goodput": "successful requests divided by measured repeat duration",
    "token_goodput": (
        "successful server-reported tokens divided by measured repeat duration"
    ),
    "e2e": "client-observed request start through complete response",
    "ttft": "client-observed request start through first non-empty streamed text",
    "approx_tpot": (
        "first-to-last non-empty text time divided by completion tokens minus one; "
        "transport chunks are not exact model tokens"
    ),
    "confidence_interval": (
        "normal-approximation 95% confidence interval across measured repeat values"
    ),
    "missing_marker": "red x at the plot floor means the metric was unavailable",
}


class PlotDataError(ValueError):
    """Raised when preserved inputs cannot be interpreted safely."""


def _is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_number(value, field, context, allow_none=True):
    if value is None and allow_none:
        return
    if not _is_number(value):
        raise PlotDataError("%s has invalid %s: %r" % (context, field, value))


def load_analysis_rows(experiment):
    """Rebuild and validate plot rows from the manifest and measured raw JSONL."""
    root = Path(experiment)
    if not root.is_dir():
        raise PlotDataError("experiment directory does not exist: %s" % root)
    if not (root / "manifest.json").is_file():
        raise PlotDataError(
            "missing experiment manifest: %s" % (root / "manifest.json")
        )
    try:
        rows = summarize_results.build_rows(root)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise PlotDataError("could not rebuild analysis rows: %s" % error) from error
    if not rows:
        raise PlotDataError("the experiment produced no analysis rows")

    metric_fields = [field for field, _, _ in THROUGHPUT_METRICS]
    metric_fields.extend(
        "%s_%s" % (prefix, percentile)
        for prefix, _, _ in LATENCY_METRICS
        for percentile in PERCENTILES
    )
    seen_aggregates = set()
    for index, row in enumerate(rows, 1):
        context = "analysis row %d" % index
        if row.get("row_type") not in ("repeat", "aggregate"):
            raise PlotDataError("%s has unsupported row_type" % context)
        workload = row.get("workload")
        if not isinstance(workload, str) or not workload:
            raise PlotDataError("%s has invalid workload" % context)
        concurrency = row.get("concurrency")
        if (
            not isinstance(concurrency, int)
            or isinstance(concurrency, bool)
            or concurrency <= 0
        ):
            raise PlotDataError("%s concurrency must be a positive integer" % context)
        if row.get("schema_compatible") is not True:
            issues = row.get("issues") or "unsupported manifest or result schema"
            raise PlotDataError("%s is schema-incompatible: %s" % (context, issues))
        if not isinstance(row.get("complete"), bool):
            raise PlotDataError("%s has invalid complete flag" % context)
        if not isinstance(row.get("issues"), str):
            raise PlotDataError("%s has invalid issues field" % context)
        for field in metric_fields:
            _validate_number(row.get(field), field, context)
            if row.get(field) is not None and row[field] < 0:
                raise PlotDataError("%s has negative %s" % (context, field))
        if row["row_type"] == "repeat":
            repeat = row.get("repeat")
            if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat <= 0:
                raise PlotDataError("%s repeat must be a positive integer" % context)
        else:
            key = (workload, row["concurrency"])
            if key in seen_aggregates:
                raise PlotDataError(
                    "duplicate aggregate row for workload %s at concurrency %s" % key
                )
            seen_aggregates.add(key)
    if not seen_aggregates:
        raise PlotDataError("the experiment has no aggregate rows")
    return rows


def _repeat_groups(rows):
    groups = defaultdict(list)
    concurrencies = defaultdict(set)
    diagnostics = set()
    for row in rows:
        concurrencies[row["workload"]].add(row["concurrency"])
        if row["row_type"] == "repeat":
            groups[(row["workload"], row["concurrency"])].append(row)
            diagnostics.update(filter(None, (row.get("issues") or "").split(";")))
    for group in groups.values():
        group.sort(key=lambda item: item["repeat"])
    return groups, concurrencies, sorted(diagnostics)


def _series_statistics(group, field):
    values = [row.get(field) for row in group if _is_number(row.get(field))]
    if not values:
        return None, None, None, []
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, None, None, values
    deviation = statistics.stdev(values)
    margin = 1.96 * deviation / math.sqrt(len(values))
    return mean, mean - margin, mean + margin, values


def _mark_missing(ax, x_values, color="crimson", x_offset=0.0):
    if not x_values:
        return
    ax.scatter(
        [value + x_offset for value in x_values],
        [0.025] * len(x_values),
        transform=ax.get_xaxis_transform(),
        marker="x",
        s=45,
        linewidths=1.5,
        color=color,
        clip_on=False,
        zorder=6,
    )


def _plot_repeat_series(
    ax,
    concurrencies,
    groups,
    workload,
    field,
    label,
    color,
    missing_offset=0.0,
):
    present_x = []
    means = []
    lower_errors = []
    upper_errors = []
    missing = []
    for concurrency in concurrencies:
        mean, low, high, values = _series_statistics(
            groups.get((workload, concurrency), []), field
        )
        if mean is None:
            missing.append(concurrency)
            continue
        present_x.append(concurrency)
        means.append(mean)
        lower_errors.append(0 if low is None else max(0, mean - low))
        upper_errors.append(0 if high is None else max(0, high - mean))
        ax.scatter(
            [concurrency] * len(values),
            values,
            s=16,
            color=color,
            alpha=0.35,
            linewidths=0,
            zorder=2,
        )
    if present_x:
        ax.errorbar(
            present_x,
            means,
            yerr=[lower_errors, upper_errors],
            color=color,
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            capsize=3,
            label=label,
            zorder=4,
        )
    _mark_missing(ax, missing, color="crimson", x_offset=missing_offset)


def _style_axis(ax, title, unit, concurrencies):
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel("Concurrency (in-flight requests)")
    ax.set_ylabel(unit)
    ax.set_xticks(concurrencies)
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _figure_footer(figure, text):
    figure.text(
        0.01,
        0.008,
        text,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )


def _save_figure(figure, stem, formats):
    outputs = []
    figure.tight_layout(rect=(0, 0.065, 1, 0.94))
    for output_format in formats:
        path = stem.with_suffix("." + output_format)
        metadata = {"Creator": "serving-inference scripts/plot_results.py"}
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
            raise PlotDataError("renderer produced an empty figure: %s" % path)
        outputs.append(path)
    plt.close(figure)
    return outputs


def _slug(value):
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "workload"


def _workload_slugs(workloads):
    result = {}
    used = set()
    for workload in sorted(workloads):
        slug = _slug(workload)
        if slug in used:
            digest = hashlib.sha256(workload.encode("utf-8")).hexdigest()[:8]
            slug = "%s-%s" % (slug, digest)
        result[workload] = slug
        used.add(slug)
    return result


def render_throughput(workload, concurrencies, groups, output_dir, formats, slug):
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.5))
    for ax, (field, title, unit) in zip(axes.flat, THROUGHPUT_METRICS):
        _plot_repeat_series(
            ax,
            concurrencies,
            groups,
            workload,
            field,
            "mean across repeats",
            "#0072B2",
        )
        _style_axis(ax, title, unit, concurrencies)
    figure.suptitle("Throughput vs concurrency — %s" % workload, fontsize=15)
    _figure_footer(
        figure,
        "Line: repeat mean; bars: normal-approximation 95% CI; small dots: "
        "measured repeats; "
        "red × at plot floor: unavailable. Goodput counts successful requests only.",
    )
    return _save_figure(figure, output_dir / ("throughput-" + slug), formats)


def render_latency(workload, concurrencies, groups, output_dir, formats, slug):
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    span = max(concurrencies) - min(concurrencies) if len(concurrencies) > 1 else 1
    for ax, (prefix, title, unit) in zip(axes, LATENCY_METRICS):
        for percentile_index, percentile in enumerate(PERCENTILES):
            _plot_repeat_series(
                ax,
                concurrencies,
                groups,
                workload,
                "%s_%s" % (prefix, percentile),
                percentile.upper(),
                PERCENTILE_COLORS[percentile],
                missing_offset=(percentile_index - 1) * span * 0.006,
            )
        _style_axis(ax, title, unit, concurrencies)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False, fontsize=8)
    figure.suptitle("Latency vs concurrency — %s" % workload, fontsize=15)
    _figure_footer(
        figure,
        "Lines: mean repeat percentile; bars: normal-approximation 95% CI; small "
        "dots: repeat percentiles; red × at plot floor: unavailable. TTFT and TPOT "
        "require streaming data.",
    )
    return _save_figure(figure, output_dir / ("latency-" + slug), formats)


def _parse_gpu_number(value, field, line_number):
    normalized = (value or "").strip().lower()
    if normalized in MISSING_VALUES:
        return None
    try:
        number = float(normalized)
    except ValueError as error:
        raise PlotDataError(
            "gpu.csv line %d has invalid %s: %r" % (line_number, field, value)
        ) from error
    if not math.isfinite(number):
        raise PlotDataError(
            "gpu.csv line %d has non-finite %s" % (line_number, field)
        )
    return number


def load_gpu_samples(path):
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(GPU_REQUIRED_FIELDS - fields)
        if missing:
            raise PlotDataError("gpu.csv is missing fields: %s" % ", ".join(missing))
        samples = []
        for line_number, row in enumerate(reader, 2):
            device = (row.get("uuid") or "").strip() or (row.get("index") or "").strip()
            if not device:
                raise PlotDataError(
                    "gpu.csv line %d has no device identity" % line_number
                )
            sample = {
                "device": device,
                "offset_s": _parse_gpu_number(
                    row.get("experiment_offset_s"), "experiment_offset_s", line_number
                ),
                "utilization_percent": _parse_gpu_number(
                    row.get("utilization_gpu_percent"),
                    "utilization_gpu_percent",
                    line_number,
                ),
                "memory_used_mib": _parse_gpu_number(
                    row.get("memory_used_mib"), "memory_used_mib", line_number
                ),
                "memory_total_mib": _parse_gpu_number(
                    row.get("memory_total_mib"), "memory_total_mib", line_number
                ),
            }
            if sample["offset_s"] is not None and sample["offset_s"] < 0:
                raise PlotDataError(
                    "gpu.csv line %d has negative experiment_offset_s" % line_number
                )
            utilization = sample["utilization_percent"]
            if utilization is not None and not 0 <= utilization <= 100:
                raise PlotDataError(
                    "gpu.csv line %d utilization_gpu_percent must be 0..100"
                    % line_number
                )
            for field in ("memory_used_mib", "memory_total_mib"):
                if sample[field] is not None and sample[field] < 0:
                    raise PlotDataError(
                        "gpu.csv line %d has negative %s" % (line_number, field)
                    )
            samples.append(sample)
    return samples


def render_gpu(gpu_path, output_dir, formats):
    samples = load_gpu_samples(gpu_path)
    usable = [sample for sample in samples if sample["offset_s"] is not None]
    if not usable:
        return [], "gpu.csv contains no timestamped samples"
    devices = defaultdict(list)
    for sample in usable:
        devices[sample["device"]].append(sample)
    figure, axes = plt.subplots(2, 1, figsize=(10.8, 7.0), sharex=True)
    colors = plt.get_cmap("tab10")
    for device_index, (device, device_samples) in enumerate(sorted(devices.items())):
        device_samples.sort(key=lambda item: item["offset_s"])
        color = colors(device_index % 10)
        short_device = device if len(device) <= 18 else device[:15] + "…"
        for ax, field, label in (
            (axes[0], "utilization_percent", short_device),
            (axes[1], "memory_used_mib", short_device),
        ):
            present = [item for item in device_samples if item[field] is not None]
            missing = [item for item in device_samples if item[field] is None]
            if present:
                ax.plot(
                    [item["offset_s"] for item in present],
                    [item[field] for item in present],
                    color=color,
                    marker=".",
                    markersize=3,
                    linewidth=1.2,
                    label=label,
                )
            _mark_missing(ax, [item["offset_s"] for item in missing])
    axes[0].set_title("GPU utilization", loc="left", fontweight="semibold")
    axes[0].set_ylabel("percent")
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("GPU memory used", loc="left", fontweight="semibold")
    axes[1].set_ylabel("MiB")
    axes[1].set_xlabel("Experiment offset (seconds)")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, title="Device", frameon=False, fontsize=8)
    figure.suptitle("GPU telemetry over experiment time", fontsize=15)
    _figure_footer(
        figure,
        "Raw nvidia-smi samples aligned to the experiment's monotonic epoch; red × "
        "at plot floor: unavailable sample. Sampling is coarse and does not prove "
        "device saturation.",
    )
    return _save_figure(figure, output_dir / "gpu-telemetry", formats), None


def _source_artifacts(root, rows, include_gpu):
    paths = [root / "manifest.json"]
    measured_runs = {
        (row["repeat"], row["concurrency"])
        for row in rows
        if row["row_type"] == "repeat"
    }
    paths.extend(
        path
        for repeat, concurrency in sorted(measured_runs)
        for path in [
            root
            / ("repeat-%02d" % repeat)
            / ("concurrency-%03d.jsonl" % concurrency)
        ]
        if path.is_file()
    )
    gpu_path = root / "telemetry" / "gpu.csv"
    if include_gpu and gpu_path.is_file():
        paths.append(gpu_path)
    result = []
    for path in paths:
        result.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return result


def render_experiment(
    experiment, output=None, formats=("png", "pdf"), include_gpu=True
):
    root = Path(experiment).resolve()
    output_dir = Path(output).resolve() if output else root / "figures"
    formats = tuple(dict.fromkeys(formats))
    invalid_formats = sorted(set(formats) - set(SUPPORTED_FORMATS))
    if not formats or invalid_formats:
        raise PlotDataError(
            "formats must contain one or more of %s" % ", ".join(SUPPORTED_FORMATS)
        )
    rows = load_analysis_rows(root)
    groups, concurrency_map, diagnostics = _repeat_groups(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    slugs = _workload_slugs(concurrency_map)
    figures = []
    for workload in sorted(concurrency_map):
        concurrencies = sorted(concurrency_map[workload])
        figures.extend(
            render_throughput(
                workload, concurrencies, groups, output_dir, formats, slugs[workload]
            )
        )
        figures.extend(
            render_latency(
                workload, concurrencies, groups, output_dir, formats, slugs[workload]
            )
        )

    gpu_path = root / "telemetry" / "gpu.csv"
    gpu_diagnostic = None
    if include_gpu and gpu_path.is_file():
        gpu_figures, gpu_diagnostic = render_gpu(gpu_path, output_dir, formats)
        figures.extend(gpu_figures)
    elif include_gpu:
        gpu_diagnostic = "telemetry/gpu.csv is unavailable; no GPU figure was generated"
    if gpu_diagnostic:
        diagnostics.append(gpu_diagnostic)

    manifest = {
        "schema_version": PLOT_MANIFEST_SCHEMA,
        "experiment": rows[0].get("experiment", root.name),
        "formats": list(formats),
        "sources": _source_artifacts(root, rows, include_gpu),
        "figures": [path.name for path in figures],
        "definitions": DEFINITIONS,
        "diagnostics": sorted(set(diagnostics)),
    }
    manifest_path = output_dir / "plot-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return figures + [manifest_path]


def _parse_formats(value):
    formats = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    invalid = sorted(set(formats) - set(SUPPORTED_FORMATS))
    if not formats or invalid:
        raise argparse.ArgumentTypeError(
            "choose one or more comma-separated formats from %s"
            % ", ".join(SUPPORTED_FORMATS)
        )
    return formats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        help="experiment directory containing manifest.json and measured raw JSONL",
    )
    parser.add_argument(
        "--output",
        help="figure directory (default: <experiment>/figures)",
    )
    parser.add_argument(
        "--formats",
        type=_parse_formats,
        default=("png", "pdf"),
        metavar="LIST",
        help="comma-separated output formats: png,pdf,svg (default: png,pdf)",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="do not inspect or render telemetry/gpu.csv",
    )
    args = parser.parse_args()
    try:
        outputs = render_experiment(
            args.experiment,
            args.output,
            formats=args.formats,
            include_gpu=not args.skip_gpu,
        )
    except (OSError, PlotDataError) as error:
        parser.exit(2, "error: %s\n" % error)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
