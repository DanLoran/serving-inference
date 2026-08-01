"""Run deterministic, resumable sweeps with reproducibility manifests."""

import argparse
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

import send_requests
from experiment_manifest import build_manifest, sanitize, utc_now, write_json
from telemetry import TelemetryManager


REQUIRED_FIELDS = {
    "name",
    "prompts",
    "url",
    "model",
    "model_metadata",
    "server",
    "num_requests",
    "concurrency",
    "warmups",
    "repeats",
    "seed",
}


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    missing = REQUIRED_FIELDS.difference(config)
    if missing:
        raise ValueError("missing config fields: %s" % ", ".join(sorted(missing)))
    if not isinstance(config["name"], str) or not config["name"]:
        raise ValueError("name must be a non-empty path-safe value")
    if config["name"] in {".", ".."} or Path(config["name"]).name != config["name"]:
        raise ValueError("name must be a non-empty path-safe value")
    if not isinstance(config["num_requests"], int) or config["num_requests"] <= 0:
        raise ValueError("num_requests must be a positive integer")
    if not isinstance(config["warmups"], int) or config["warmups"] < 0:
        raise ValueError("warmups must be a non-negative integer")
    if not isinstance(config["repeats"], int) or config["repeats"] <= 0:
        raise ValueError("repeats must be a positive integer")
    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")
    concurrency = config["concurrency"]
    if (
        not isinstance(concurrency, list)
        or not concurrency
        or any(not isinstance(value, int) or value <= 0 for value in concurrency)
    ):
        raise ValueError("concurrency must contain positive integers")
    if len(concurrency) != len(set(concurrency)):
        raise ValueError("concurrency values must be unique")

    model_required = {"revision", "dtype", "quantization", "max_model_len"}
    missing_model = model_required.difference(config["model_metadata"])
    if missing_model:
        raise ValueError(
            "model_metadata missing fields: %s"
            % ", ".join(sorted(missing_model))
        )
    server = config["server"]
    if "launch_flags" not in server:
        raise ValueError(
            "server.launch_flags must explicitly record the server command flags"
        )
    if not isinstance(server["launch_flags"], list):
        raise ValueError("server.launch_flags must be a list")
    validate_telemetry(config.get("telemetry", {}))
    return config


def validate_telemetry(telemetry):
    if not isinstance(telemetry, dict):
        raise ValueError("telemetry must be an object")
    for name in ("gpu", "vllm"):
        collector = telemetry.get(name, {})
        if not isinstance(collector, dict):
            raise ValueError("telemetry.%s must be an object" % name)
        if "enabled" in collector and not isinstance(collector["enabled"], bool):
            raise ValueError("telemetry.%s.enabled must be a boolean" % name)
        for field in ("interval_s", "timeout_s"):
            if field in collector and (
                not isinstance(collector[field], (int, float))
                or isinstance(collector[field], bool)
                or collector[field] <= 0
            ):
                raise ValueError("telemetry.%s.%s must be positive" % (name, field))
    vllm = telemetry.get("vllm", {})
    if vllm.get("enabled", False) and not vllm.get("url"):
        raise ValueError("telemetry.vllm.url is required when enabled")
    command = telemetry.get("gpu", {}).get("command")
    if command is not None and (not isinstance(command, str) or not command):
        raise ValueError("telemetry.gpu.command must be a non-empty string")


def resolve_config(config):
    resolved = dict(config)
    prompts = Path(config["prompts"])
    if not prompts.is_absolute():
        prompts = (Path.cwd() / prompts).resolve()
    resolved.update(
        {
            "prompts": str(prompts),
            "temperature": config.get("temperature", 0.0),
            "timeout": config.get("timeout", 300),
            "stream": config.get("stream", True),
            "store_response": config.get("store_response", False),
            "output_dir": config.get("output_dir", "results/experiments"),
        }
    )
    # Keep direct programmatic callers compatible. CLI configs are validated
    # by load_config and must supply these fields explicitly.
    resolved.setdefault(
        "model_metadata",
        {
            "revision": None,
            "dtype": None,
            "quantization": None,
            "max_model_len": None,
        },
    )
    resolved.setdefault(
        "server",
        {"discovery": "unavailable", "launch_flags": []},
    )
    telemetry = config.get("telemetry", {})
    resolved["telemetry"] = {
        "gpu": {
            "enabled": telemetry.get("gpu", {}).get("enabled", False),
            "interval_s": telemetry.get("gpu", {}).get("interval_s", 1.0),
            "timeout_s": telemetry.get("gpu", {}).get("timeout_s", 5.0),
            **(
                {"command": telemetry["gpu"]["command"]}
                if telemetry.get("gpu", {}).get("command")
                else {}
            ),
        },
        "vllm": {
            "enabled": telemetry.get("vllm", {}).get("enabled", False),
            "interval_s": telemetry.get("vllm", {}).get("interval_s", 1.0),
            "timeout_s": telemetry.get("vllm", {}).get("timeout_s", 5.0),
            **(
                {"url": telemetry["vllm"]["url"]}
                if telemetry.get("vllm", {}).get("url")
                else {}
            ),
        },
    }
    return resolved


def sweep_order(config):
    """Return seeded, interleaved warmup and measurement rounds."""
    rng = random.Random(config["seed"])
    runs = []
    for kind, count in (("warmup", config["warmups"]), ("repeat", config["repeats"])):
        for index in range(1, count + 1):
            levels = list(config["concurrency"])
            rng.shuffle(levels)
            runs.extend(
                {"kind": kind, "index": index, "concurrency": level}
                for level in levels
            )
    return runs


def run_paths(root, run):
    directory = root / ("%s-%02d" % (run["kind"], run["index"]))
    stem = "concurrency-%03d" % run["concurrency"]
    return directory / (stem + ".jsonl"), directory / (stem + ".summary.json")


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_run(raw_path, summary_path, run, expected_requests):
    if not raw_path.exists() and not summary_path.exists():
        return False
    if not raw_path.exists() or not summary_path.exists():
        raise RuntimeError("refusing to overwrite partial run: %s" % raw_path)
    try:
        rows = read_rows(raw_path)
        summary = read_json(summary_path)
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("invalid completed run %s: %s" % (raw_path, error)) from error
    counts = summary.get("counts", {})
    valid = (
        len(rows) == expected_requests
        and counts.get("attempted") == expected_requests
        and counts.get("successful") == expected_requests
        and counts.get("failed") == 0
        and summary.get("config", {}).get("concurrency") == run["concurrency"]
        and all(row.get("status") == 200 for row in rows)
    )
    if not valid:
        raise RuntimeError("required run is incomplete or failed: %s" % raw_path)
    return True


def command_for(config, run, raw_path, summary_path):
    command = [
        sys.executable,
        str(Path(__file__).with_name("send_requests.py")),
        "--prompts",
        str(config["prompts"]),
        "--output",
        str(raw_path),
        "--summary",
        str(summary_path),
        "--url",
        config["url"],
        "--model",
        config["model"],
        "--num-requests",
        str(config["num_requests"]),
        "--concurrency",
        str(run["concurrency"]),
        "--temperature",
        str(config.get("temperature", 0.0)),
        "--timeout",
        str(config.get("timeout", 300)),
    ]
    if config.get("stream", True):
        command.append("--stream")
    if config.get("store_response", False):
        command.append("--store-response")
    return command


def median(values):
    clean = [value for value in values if value is not None]
    return statistics.median(clean) if clean else None


def saturation_analysis(by_concurrency, throughput_gain=0.05, p99_rise=0.20):
    transitions = []
    qualifying_streak = 0
    saturation_at = None
    for previous, current in zip(by_concurrency, by_concurrency[1:]):
        old_tps = previous["repeat_medians"]["output_token_throughput_per_s"]
        new_tps = current["repeat_medians"]["output_token_throughput_per_s"]
        old_p99 = previous["repeat_medians"]["latency_p99_s"]
        new_p99 = current["repeat_medians"]["latency_p99_s"]
        gain = (new_tps - old_tps) / old_tps if old_tps else None
        rise = (new_p99 - old_p99) / old_p99 if old_p99 else None
        qualifies = (
            gain is not None
            and rise is not None
            and gain < throughput_gain
            and rise >= p99_rise
        )
        qualifying_streak = qualifying_streak + 1 if qualifies else 0
        if qualifying_streak >= 2 and saturation_at is None:
            saturation_at = previous["concurrency"]
        transitions.append(
            {
                "from_concurrency": previous["concurrency"],
                "to_concurrency": current["concurrency"],
                "median_throughput_change_fraction": gain,
                "median_p99_latency_change_fraction": rise,
                "meets_transition_criteria": qualifies,
            }
        )
    return {
        "throughput_gain_threshold_fraction": throughput_gain,
        "p99_latency_rise_threshold_fraction": p99_rise,
        "required_successive_transitions": 2,
        "candidate_saturation_concurrency": saturation_at,
        "transitions": transitions,
    }


def aggregate(config, root, order):
    measured = [run for run in order if run["kind"] == "repeat"]
    report_runs = []
    by_concurrency = {}
    for run in measured:
        raw_path, summary_path = run_paths(root, run)
        summary = read_json(summary_path)
        rows = read_rows(raw_path)
        report_runs.append(
            {
                **run,
                "raw": str(raw_path),
                "summary": str(summary_path),
                "metrics": summary,
            }
        )
        bucket = by_concurrency.setdefault(
            run["concurrency"], {"rows": [], "summaries": []}
        )
        bucket["rows"].extend(rows)
        bucket["summaries"].append(summary)

    aggregates = []
    for concurrency in config["concurrency"]:
        bucket = by_concurrency[concurrency]
        successful = [row for row in bucket["rows"] if row["status"] == 200]
        output_tokens = sum(row.get("output_tokens") or 0 for row in successful)
        duration = sum(summary["duration_s"] for summary in bucket["summaries"])
        repeat_medians = {
            "request_throughput_per_s": median(
                summary.get("request_throughput_per_s") for summary in bucket["summaries"]
            ),
            "output_token_throughput_per_s": median(
                summary.get("output_token_throughput_per_s")
                for summary in bucket["summaries"]
            ),
            "latency_p99_s": median(
                summary.get("latency_s", {}).get("p99")
                for summary in bucket["summaries"]
            ),
        }
        aggregates.append(
            {
                "concurrency": concurrency,
                "repeats": len(bucket["summaries"]),
                "counts": {
                    "attempted": len(bucket["rows"]),
                    "successful": len(successful),
                    "failed": len(bucket["rows"]) - len(successful),
                    "output_tokens": output_tokens,
                },
                "duration_s": duration,
                "request_throughput_per_s": len(successful) / duration if duration else None,
                "output_token_throughput_per_s": output_tokens / duration if duration else None,
                "latency_s": send_requests.distribution(
                    [row.get("latency_s") for row in successful]
                ),
                "ttft_s": send_requests.distribution(
                    [row.get("ttft_s") for row in successful]
                ),
                "approx_time_per_output_token_s": send_requests.distribution(
                    [row.get("approx_time_per_output_token_s") for row in successful]
                ),
                "repeat_medians": repeat_medians,
            }
        )
    return {
        "config": config,
        "sweep_order": order,
        "measured_runs": report_runs,
        "by_concurrency": aggregates,
        "saturation": saturation_analysis(aggregates),
    }


def write_report(root, combined):
    (root / "summary.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Experiment report",
        "",
        "Warmups are excluded. Each row aggregates every measured repeat.",
        "",
        "| Concurrency | Repeats | Successful | Failed | Requests/s | Output tokens/s | Median repeat P99 (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in combined["by_concurrency"]:
        def display(number):
            return "n/a" if number is None else "%.3f" % number

        lines.append(
            "| {concurrency} | {repeats} | {successful} | {failed} | {rps} | {tps} | {p99} |".format(
                concurrency=item["concurrency"],
                repeats=item["repeats"],
                successful=item["counts"]["successful"],
                failed=item["counts"]["failed"],
                rps=display(item["request_throughput_per_s"]),
                tps=display(item["output_token_throughput_per_s"]),
                p99=display(item["repeat_medians"]["latency_p99_s"]),
            )
        )
    saturation = combined["saturation"]["candidate_saturation_concurrency"]
    lines.extend(
        [
            "",
            "Candidate saturation concurrency: %s"
            % (saturation if saturation is not None else "not demonstrated"),
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_root(root, config):
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "config.json"
    if manifest_path.exists():
        if read_json(manifest_path) != config:
            raise RuntimeError(
                "experiment config differs from existing run; choose a new name or output root"
            )
        return
    if any(root.iterdir()):
        raise RuntimeError("experiment directory has no config manifest: %s" % root)
    manifest_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_experiment(
    config,
    output_root=None,
    runner=subprocess.run,
    original_config=None,
    repo_root=None,
):
    legacy_manifest_call = isinstance(output_root, dict)
    if legacy_manifest_call:
        original = config
        resolved = resolve_config(output_root)
        output_root = None
    else:
        original = original_config or config
        resolved = resolve_config(config)
    root = Path(output_root or resolved["output_dir"]) / resolved["name"]
    prepare_root(root, original)

    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    manifest_path = root / "manifest.json"
    write_json(root / "config.original.json", sanitize(original))
    write_json(root / "config.resolved.json", sanitize(resolved))

    if manifest_path.exists():
        manifest = read_json(manifest_path)
        manifest["experiment"].update(
            {"completed_at_utc": None, "status": "running"}
        )
    else:
        manifest = build_manifest(original, resolved, repo_root, utc_now())
    write_json(manifest_path, manifest)

    telemetry = TelemetryManager(resolved["telemetry"], root)

    try:
        try:
            telemetry.start()
        except Exception as error:
            telemetry.summary.update(
                {
                    "enabled": True,
                    "available": False,
                    "startup_error": "%s: %s" % (type(error).__name__, error),
                }
            )
        order = sweep_order(resolved)
        if legacy_manifest_call:
            run_number = 0
            for run in order:
                run_number += 1
                stem = "%03d-%s-%02d-concurrency-%03d" % (
                    run_number,
                    run["kind"],
                    run["index"],
                    run["concurrency"],
                )
                telemetry.mark("benchmark_started", **run)
                result = runner(
                    command_for(
                        resolved,
                        run,
                        root / (stem + ".jsonl"),
                        root / (stem + ".summary.json"),
                    )
                )
                telemetry.mark(
                    "benchmark_finished", returncode=result.returncode, **run
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        "%s run failed with exit code %d"
                        % (run["kind"], result.returncode)
                    )
            manifest["experiment"].update(
                {"completed_at_utc": utc_now(), "status": "completed"}
            )
            write_json(manifest_path, manifest)
            return manifest

        for run in order:
            raw_path, summary_path = run_paths(root, run)
            if completed_run(
                raw_path, summary_path, run, resolved["num_requests"]
            ):
                print("resume: keeping %s" % raw_path)
                continue
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            telemetry.mark("benchmark_started", **run)
            result = runner(command_for(resolved, run, raw_path, summary_path))
            telemetry.mark("benchmark_finished", returncode=result.returncode, **run)
            if result.returncode != 0:
                raise RuntimeError(
                    "required %s %d at concurrency %d failed with exit code %d"
                    % (
                        run["kind"],
                        run["index"],
                        run["concurrency"],
                        result.returncode,
                    )
                )
            if not completed_run(
                raw_path, summary_path, run, resolved["num_requests"]
            ):
                raise RuntimeError(
                    "required run did not produce complete artifacts: %s"
                    % raw_path
                )
        combined = aggregate(resolved, root, order)
    except Exception:
        manifest["experiment"].update(
            {"completed_at_utc": utc_now(), "status": "failed"}
        )
        write_json(manifest_path, manifest)
        raise
    finally:
        try:
            telemetry.stop()
        except Exception as error:
            telemetry.summary.update(
                {
                    "available": False,
                    "cleanup_error": "%s: %s" % (type(error).__name__, error),
                }
            )

    manifest["experiment"].update(
        {"completed_at_utc": utc_now(), "status": "completed"}
    )
    write_json(manifest_path, manifest)
    combined["telemetry"] = telemetry.summary
    write_report(root, combined)
    combined["manifest"] = manifest
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    try:
        original = load_config(args.config)
        run_experiment(
            original,
            args.output_root,
            original_config=original,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, "error: %s\n" % error)


if __name__ == "__main__":
    main()
