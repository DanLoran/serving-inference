"""Run a configured benchmark while preserving a reproducibility manifest."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiment_manifest import build_manifest, sanitize, utc_now, write_json


REQUIRED_FIELDS = {"name", "prompts", "url", "model", "num_requests", "concurrency", "warmups", "repeats", "seed", "model_metadata", "server"}


def load_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        original = json.load(handle)
    missing = REQUIRED_FIELDS.difference(original)
    if missing:
        raise ValueError("missing config fields: %s" % ", ".join(sorted(missing)))
    model_required = {"revision", "dtype", "quantization", "max_model_len"}
    server_required = {"launch_flags"}
    if model_required.difference(original["model_metadata"]):
        raise ValueError("model_metadata must record revision, dtype, quantization, and max_model_len")
    if server_required.difference(original["server"]):
        raise ValueError("server.launch_flags must explicitly record the server command flags")
    if not isinstance(original["server"]["launch_flags"], list):
        raise ValueError("server.launch_flags must be a list")
    if original["warmups"] < 0 or original["repeats"] <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    concurrency = original["concurrency"]
    if not concurrency or any(value <= 0 for value in concurrency):
        raise ValueError("concurrency must contain positive values")
    resolved = dict(original)
    prompts = Path(original["prompts"])
    if not prompts.is_absolute():
        prompts = (Path.cwd() / prompts).resolve()
    resolved.update({
        "prompts": str(prompts),
        "temperature": original.get("temperature", 0.0),
        "timeout": original.get("timeout", 300),
        "stream": original.get("stream", True),
        "store_response": original.get("store_response", False),
        "output_dir": original.get("output_dir", "results/experiments"),
    })
    return original, resolved


def command_for(config, concurrency, output, summary):
    command = [
        sys.executable, str(Path(__file__).with_name("send_requests.py")),
        "--prompts", config["prompts"], "--output", str(output), "--summary", str(summary),
        "--url", config["url"], "--model", config["model"],
        "--num-requests", str(config["num_requests"]), "--concurrency", str(concurrency),
        "--temperature", str(config["temperature"]), "--timeout", str(config["timeout"]),
    ]
    if config["stream"]:
        command.append("--stream")
    if config["store_response"]:
        command.append("--store-response")
    return command


def run_experiment(original, resolved, output_root=None, runner=subprocess.run, repo_root=None):
    root = Path(output_root or resolved["output_dir"]) / resolved["name"]
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise RuntimeError("experiment output directory is not empty: %s" % root)
    started_at = utc_now()
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    write_json(root / "config.original.json", sanitize(original))
    write_json(root / "config.resolved.json", sanitize(resolved))
    manifest = build_manifest(original, resolved, repo_root, started_at)
    write_json(root / "manifest.json", manifest)
    try:
        run_number = 0
        for kind, count in (("warmup", resolved["warmups"]), ("repeat", resolved["repeats"])):
            for repeat_index in range(1, count + 1):
                for concurrency in resolved["concurrency"]:
                    run_number += 1
                    stem = "%03d-%s-%02d-concurrency-%03d" % (run_number, kind, repeat_index, concurrency)
                    result = runner(command_for(resolved, concurrency, root / (stem + ".jsonl"), root / (stem + ".summary.json")))
                    if result.returncode != 0:
                        raise RuntimeError("%s run failed with exit code %d" % (kind, result.returncode))
    except Exception:
        manifest["experiment"].update({"completed_at_utc": utc_now(), "status": "failed"})
        write_json(root / "manifest.json", manifest)
        raise
    manifest["experiment"].update({"completed_at_utc": utc_now(), "status": "completed"})
    write_json(root / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    original, resolved = load_config(args.config)
    run_experiment(original, resolved, args.output_root)


if __name__ == "__main__":
    main()
