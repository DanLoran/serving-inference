"""Run and analyze an isolated, cache-controlled capacity campaign.

The ordinary campaign runner intentionally preserves production-like cache and
server state.  This runner owns the server, resets cache state before every
measured condition, prewarms only declared shared prefixes, and writes a new
attempt directory instead of overwriting partial evidence.
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import generate_prompts
import send_requests
from telemetry import TelemetryManager


SCHEMA_VERSION = "cache-campaign-1.0"
METRICS = {
    "prefix_queries": "vllm:prefix_cache_queries_total",
    "prefix_hits": "vllm:prefix_cache_hits_total",
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_usage": "vllm:kv_cache_usage_perc",
    "preemptions": "vllm:num_preemptions_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
}
FORBIDDEN_SERVER_FLAGS = {"--max-num-seqs", "--max-num-batched-tokens"}


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def timestamp_slug():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_safe(value, field):
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "\\" in value
    ):
        raise ValueError("%s must be a non-empty path-safe value" % field)


def validate_config(config):
    required = {
        "schema_version",
        "name",
        "output_root",
        "model",
        "model_revision",
        "tokenizer",
        "tokenizer_revision",
        "seed",
        "concurrency",
        "repeats",
        "requests_per_repeat",
        "block_size_tokens",
        "server",
        "client",
        "telemetry",
        "workloads",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError("missing fields: %s" % ", ".join(sorted(missing)))
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    path_safe(config["name"], "name")
    if config["model"] != config["tokenizer"]:
        raise ValueError("model and tokenizer must match exactly")
    if not config["model_revision"] or not config["tokenizer_revision"]:
        raise ValueError("model and tokenizer revisions must be explicit")
    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")
    for field in ("repeats", "requests_per_repeat", "block_size_tokens"):
        if not isinstance(config[field], int) or config[field] <= 0:
            raise ValueError("%s must be a positive integer" % field)
    concurrency = config["concurrency"]
    if (
        not isinstance(concurrency, list)
        or not concurrency
        or len(concurrency) != len(set(concurrency))
        or any(not isinstance(value, int) or value <= 0 for value in concurrency)
    ):
        raise ValueError("concurrency must contain unique positive integers")
    if not isinstance(config["workloads"], list) or not config["workloads"]:
        raise ValueError("workloads must be a non-empty list")
    names = []
    for workload in config["workloads"]:
        if set(workload) != {"name", "buckets"}:
            raise ValueError("workloads contain only name and buckets")
        path_safe(workload["name"], "workload name")
        names.append(workload["name"])
        if not isinstance(workload["buckets"], list) or not workload["buckets"]:
            raise ValueError("workload buckets must be a non-empty list")
        count = 0
        bucket_names = []
        for bucket in workload["buckets"]:
            expected = {
                "name",
                "count",
                "prompt_tokens",
                "output_tokens",
                "shared_prefix_tokens",
            }
            if set(bucket) != expected:
                raise ValueError("invalid bucket fields in %s" % workload["name"])
            path_safe(bucket["name"], "bucket name")
            bucket_names.append(bucket["name"])
            for field in (
                "count",
                "prompt_tokens",
                "output_tokens",
                "shared_prefix_tokens",
            ):
                if not isinstance(bucket[field], int) or bucket[field] <= 0:
                    raise ValueError("bucket %s must be positive" % field)
            if bucket["shared_prefix_tokens"] % config["block_size_tokens"]:
                raise ValueError("shared prefixes must be block aligned")
            if bucket["shared_prefix_tokens"] >= bucket["prompt_tokens"]:
                raise ValueError("shared prefix must be shorter than prompt")
            count += bucket["count"]
        if len(bucket_names) != len(set(bucket_names)):
            raise ValueError("bucket names must be unique within a workload")
        if count != config["requests_per_repeat"]:
            raise ValueError(
                "%s bucket counts must equal requests_per_repeat" % workload["name"]
            )
    if len(names) != len(set(names)):
        raise ValueError("workload names must be unique")

    server = config["server"]
    server_required = {
        "command",
        "environment",
        "health_url",
        "metrics_url",
        "cache_reset_url",
        "host",
        "port",
        "startup_timeout_s",
        "drain_timeout_s",
        "shutdown_timeout_s",
    }
    missing_server = server_required.difference(server)
    if missing_server:
        raise ValueError(
            "server missing fields: %s" % ", ".join(sorted(missing_server))
        )
    if not isinstance(server["command"], list) or not server["command"]:
        raise ValueError("server.command must be a non-empty list")
    if any(flag in server["command"] for flag in FORBIDDEN_SERVER_FLAGS):
        raise ValueError("baseline must not set scheduler capacity flags")
    if "--enable-prefix-caching" not in server["command"]:
        raise ValueError("server.command must enable prefix caching")
    client = config["client"]
    if not client.get("stream", False):
        raise ValueError("cache campaigns require streaming metrics")
    if client.get("temperature") != 0.0:
        raise ValueError("cache campaigns require temperature 0.0")
    return config


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        return validate_config(json.load(handle))


def resolve_command(command, repo_root):
    resolved = list(command)
    executable = Path(resolved[0])
    if not executable.is_absolute() and "/" in resolved[0]:
        resolved[0] = str((Path(repo_root) / executable).resolve())
    return resolved


def discover_cuda_runtime_paths(command):
    """Find pip-installed CUDA runtime libraries beside the server executable."""
    executable = Path(command[0]).resolve()
    virtual_environment = executable.parent.parent
    paths = []
    pattern = "lib/python*/site-packages/nvidia/**/lib/libcudart.so*"
    for library in virtual_environment.glob(pattern):
        directory = str(library.parent.resolve())
        if directory not in paths:
            paths.append(directory)
    return paths


def server_environment(command, configured, inherited=None):
    """Build a launch environment with discoverable CUDA runtimes first."""
    inherited = dict(os.environ if inherited is None else inherited)
    environment = dict(inherited)
    environment.update(configured)
    candidates = discover_cuda_runtime_paths(command)
    for source in (
        configured.get("LD_LIBRARY_PATH", ""),
        inherited.get("LD_LIBRARY_PATH", ""),
    ):
        for directory in source.split(os.pathsep):
            if directory and directory not in candidates:
                candidates.append(directory)
    if candidates:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(candidates)
    return environment, candidates


def build_plan(config, config_path, campaign_root=None):
    if campaign_root is None:
        output_root = Path(config["output_root"])
        if not output_root.is_absolute():
            output_root = Path.cwd() / output_root
        root = output_root / (config["name"] + "-" + timestamp_slug())
    else:
        root = Path(campaign_root)
        if not root.is_absolute():
            root = Path.cwd() / root
    conditions = []
    ordinal = 0
    for workload in config["workloads"]:
        for concurrency in config["concurrency"]:
            for repeat in range(1, config["repeats"] + 1):
                conditions.append(
                    {
                        "ordinal": ordinal,
                        "workload": workload["name"],
                        "concurrency": concurrency,
                        "repeat": repeat,
                    }
                )
                ordinal += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(Path(config_path).resolve()),
        "campaign_root": str(root.resolve()),
        "conditions": conditions,
        "condition_count": len(conditions),
        "measured_request_count": (
            len(conditions) * config["requests_per_repeat"]
        ),
    }


def metric_value(text, name):
    values = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(name + "{") or line.startswith(name + " "):
            try:
                values.append(float(line.rsplit(" ", 1)[1]))
            except (ValueError, IndexError):
                pass
    return sum(values) if values else None


def parsed_metrics(text):
    return {key: metric_value(text, name) for key, name in METRICS.items()}


def metric_delta(after, before, key):
    left, right = after.get(key), before.get(key)
    return None if left is None or right is None else left - right


def http_text(url, method="GET", payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def snapshot_metrics(url, path, timeout=10):
    status, body = http_text(url, timeout=timeout)
    if status != 200:
        raise RuntimeError("metrics endpoint returned HTTP %d" % status)
    Path(path).write_text(body, encoding="utf-8")
    return parsed_metrics(body)


def port_is_open(host, port):
    with socket.socket() as connection:
        connection.settimeout(1)
        return connection.connect_ex((host, port)) == 0


def wait_for_idle(server):
    deadline = time.monotonic() + server["drain_timeout_s"]
    last = None
    while time.monotonic() < deadline:
        _, body = http_text(server["metrics_url"], timeout=5)
        metrics = parsed_metrics(body)
        last = {"running": metrics["running"], "waiting": metrics["waiting"]}
        if last == {"running": 0.0, "waiting": 0.0}:
            return last
        time.sleep(0.25)
    raise RuntimeError("server did not drain: %s" % last)


class ManagedServer:
    def __init__(
        self, config, repo_root, artifact_root, expected_model, required_context
    ):
        self.config = config
        self.repo_root = Path(repo_root)
        self.artifact_root = Path(artifact_root)
        self.expected_model = expected_model
        self.required_context = required_context
        self.process = None
        self.log_handle = None
        self.generation = 0

    def start(self):
        if port_is_open(self.config["host"], self.config["port"]):
            raise RuntimeError(
                "refusing to replace a process already listening on %s:%s"
                % (self.config["host"], self.config["port"])
            )
        self.generation += 1
        command = resolve_command(self.config["command"], self.repo_root)
        environment, runtime_paths = server_environment(
            command, self.config["environment"]
        )
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.artifact_root / ("launch-%03d.json" % self.generation),
            {
                "command": command,
                "environment": self.config["environment"],
                "effective_ld_library_path": environment.get("LD_LIBRARY_PATH"),
                "discovered_cuda_runtime_paths": runtime_paths,
            },
        )
        self.log_handle = (
            self.artifact_root / ("server-%03d.log" % self.generation)
        ).open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            cwd=self.repo_root,
            env=environment,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        atomic_write_json(
            self.artifact_root / "active.json",
            {
                "pid": self.process.pid,
                "pgid": os.getpgid(self.process.pid),
                "generation": self.generation,
                "started_at_utc": utc_now(),
            },
        )
        deadline = time.monotonic() + self.config["startup_timeout_s"]
        last_error = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "managed server exited during startup with %s"
                    % self.process.returncode
                )
            try:
                status, body = http_text(self.config["health_url"], timeout=5)
                if status == 200:
                    health = json.loads(body)
                    models = health.get("data") or []
                    selected = next(
                        (
                            item
                            for item in models
                            if item.get("id") == self.expected_model
                        ),
                        None,
                    )
                    if selected is None:
                        raise RuntimeError(
                            "health endpoint does not expose expected model %s"
                            % self.expected_model
                        )
                    maximum = selected.get("max_model_len")
                    if maximum is not None and int(maximum) < self.required_context:
                        raise RuntimeError(
                            "server max_model_len %s is below required %s"
                            % (maximum, self.required_context)
                        )
                    (self.artifact_root / "health.json").write_text(
                        body + ("" if body.endswith("\n") else "\n"),
                        encoding="utf-8",
                    )
                    return
            except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
                last_error = error
            time.sleep(2)
        raise RuntimeError("server health check timed out: %s" % last_error)

    def stop(self):
        if self.process is None:
            return
        process = self.process
        if process.poll() is None:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGINT)
            try:
                process.wait(timeout=self.config["shutdown_timeout_s"])
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    process.wait(timeout=15)
        if self.log_handle is not None:
            self.log_handle.close()
        atomic_write_json(
            self.artifact_root / ("stopped-%03d.json" % self.generation),
            {
                "pid": process.pid,
                "returncode": process.returncode,
                "stopped_at_utc": utc_now(),
            },
        )
        self.process = None
        self.log_handle = None

    def restart(self):
        self.stop()
        self.start()


def reset_cache(server_manager):
    server = server_manager.config
    wait_for_idle(server)
    try:
        status, body = http_text(
            server["cache_reset_url"], method="POST", timeout=10
        )
        parsed = json.loads(body) if body.strip() else {}
        reset_succeeded = (
            parsed.get("reset_success", True)
            if isinstance(parsed, dict)
            else bool(parsed)
        )
        if status == 200 and reset_succeeded:
            wait_for_idle(server)
            return {"method": "supported_http_reset", "status": status, "body": body}
    except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
        reset_error = "%s: %s" % (type(error).__name__, error)
    else:
        reset_error = "HTTP %s: %s" % (status, body[:500])
    server_manager.restart()
    wait_for_idle(server)
    return {"method": "managed_server_restart", "reset_error": reset_error}


def workload_by_name(config, name):
    return next(item for item in config["workloads"] if item["name"] == name)


def attempt_root(root, condition, attempt):
    return (
        Path(root)
        / "runs"
        / condition["workload"]
        / ("concurrency-%03d" % condition["concurrency"])
        / ("repeat-%02d" % condition["repeat"])
        / ("attempt-%03d" % attempt)
    )


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def attempt_is_complete(path, expected_requests):
    path = Path(path)
    required = (
        path / "attempt.json",
        path / "measured.jsonl",
        path / "measured.summary.json",
        path / "cache.json",
    )
    if not all(item.is_file() for item in required):
        return False
    try:
        status = read_json(path / "attempt.json")
        summary = read_json(path / "measured.summary.json")
        rows = read_rows(path / "measured.jsonl")
    except (OSError, ValueError, TypeError):
        return False
    counts = summary.get("counts", {})
    return (
        status.get("status") == "completed"
        and len(rows) == expected_requests
        and counts.get("attempted") == expected_requests
        and counts.get("successful") == expected_requests
        and counts.get("failed") == 0
        and all(row.get("status") == 200 for row in rows)
    )


def existing_attempts(root, condition):
    parent = attempt_root(root, condition, 1).parent
    if not parent.exists():
        return []
    attempts = []
    for path in parent.glob("attempt-[0-9][0-9][0-9]"):
        try:
            attempts.append((int(path.name.split("-")[-1]), path))
        except ValueError:
            continue
    return sorted(attempts)


def selected_attempt(root, condition, expected_requests):
    for _, path in reversed(existing_attempts(root, condition)):
        if attempt_is_complete(path, expected_requests):
            return path
    return None


def bank_config(config, workload, condition, attempt, condition_count):
    request_count = config["requests_per_repeat"]
    bank_number = (attempt - 1) * condition_count + condition["ordinal"]
    nonce_base = bank_number * request_count
    if nonce_base + request_count >= 2**32:
        raise RuntimeError("campaign exhausted the deterministic nonce space")
    return {
        "name": "%s-c%03d-r%02d-a%03d"
        % (
            condition["workload"],
            condition["concurrency"],
            condition["repeat"],
            attempt,
        ),
        "model": config["model"],
        "tokenizer": config["tokenizer"],
        "seed": config["seed"],
        "request_count": request_count,
        "prompt_namespace": "%s/%s/c%d/r%d/a%d"
        % (
            config["name"],
            condition["workload"],
            condition["concurrency"],
            condition["repeat"],
            attempt,
        ),
        "prefix_namespace": "%s/%s"
        % (config["name"], condition["workload"]),
        "nonce_base": nonce_base,
        "block_size_tokens": config["block_size_tokens"],
        "buckets": workload["buckets"],
    }


def write_warmup(path, bank_metadata):
    rows = []
    measured_hashes = set()
    for prefix_key, prefix in sorted(bank_metadata["shared_prefixes"].items()):
        prompt_hash = hashlib.sha256(prefix["text"].encode("utf-8")).hexdigest()
        measured_hashes.add(prompt_hash)
        rows.append(
            {
                "id": "prewarm-" + prefix_key,
                "workload": prefix_key,
                "bucket": prefix_key,
                "prompt": prefix["text"],
                "prompt_tokens": prefix["tokens"],
                "target_prompt_tokens": prefix["tokens"],
                "target_output_tokens": 1,
                "prefix_sha256": prefix["sha256"],
            }
        )
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows), measured_hashes


def sender_command(config, prompts, raw, summary, concurrency, count):
    client = config["client"]
    command = [
        sys.executable,
        str(Path(__file__).with_name("send_requests.py")),
        "--prompts",
        str(prompts),
        "--output",
        str(raw),
        "--summary",
        str(summary),
        "--url",
        client["url"],
        "--model",
        config["model"],
        "--num-requests",
        str(count),
        "--concurrency",
        str(concurrency),
        "--temperature",
        str(client["temperature"]),
        "--timeout",
        str(client["timeout_s"]),
        "--stream",
    ]
    if client.get("store_response", False):
        command.append("--store-response")
    return command


def run_sender(command, repo_root, log_path):
    with Path(log_path).open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def intended_hit_rate(workload):
    hits = sum(
        bucket["count"] * bucket["shared_prefix_tokens"]
        for bucket in workload["buckets"]
    )
    queries = sum(
        bucket["count"] * bucket["prompt_tokens"]
        for bucket in workload["buckets"]
    )
    return hits / queries


def collect_command(command, path, cwd=None):
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    Path(path).write_text(result.stdout + result.stderr, encoding="utf-8")
    return result.returncode


def gpu_memory_snapshot():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return None
    values = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, used = (part.strip() for part in line.split(",", 1))
        values[index] = float(used)
    return values


def preflight(root, repo_root, server):
    target = Path(root) / "preflight"
    target.mkdir(parents=True, exist_ok=True)
    if port_is_open(server["host"], server["port"]):
        collect_command(
            ["ss", "-ltnp", "sport = :%d" % server["port"]],
            target / "port-before.txt",
        )
        raise RuntimeError("configured server port is already in use")
    collect_command(["git", "status", "--short", "--branch"], target / "git-status.txt", repo_root)
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if git_status.returncode or git_status.stdout.strip():
        raise RuntimeError("official cache campaigns require a clean Git checkout")
    collect_command(["git", "rev-parse", "HEAD"], target / "git-revision.txt", repo_root)
    collect_command(["nvidia-smi"], target / "nvidia-smi-before.txt")
    collect_command(
        ["ss", "-ltnp", "sport = :%d" % server["port"]],
        target / "port-before.txt",
    )
    gpu_before = gpu_memory_snapshot()
    atomic_write_json(target / "gpu-memory-before.json", gpu_before)
    return gpu_before


def run_attempt(config, plan, condition, number, server_manager, tokenizer):
    root = Path(plan["campaign_root"])
    path = attempt_root(root, condition, number)
    path.mkdir(parents=True, exist_ok=False)
    state = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "condition": condition,
        "attempt": number,
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
    }
    atomic_write_json(path / "attempt.json", state)
    telemetry = TelemetryManager(config["telemetry"], path)
    try:
        workload = workload_by_name(config, condition["workload"])
        generated = bank_config(
            config, workload, condition, number, plan["condition_count"]
        )
        atomic_write_json(path / "prompt-bank.config.json", generated)
        bank_path, metadata_path, bank_metadata = generate_prompts.write_cache_bank(
            generated, path / "prompt-bank.jsonl.gz", tokenizer=tokenizer
        )
        warmup_count, warmup_hashes = write_warmup(
            path / "prewarm-prompts.jsonl", bank_metadata
        )
        with gzip.open(bank_path, mode="rt", encoding="utf-8") as handle:
            if any(
                json.loads(line)["prompt_sha256"] in warmup_hashes
                for line in handle
                if line.strip()
            ):
                raise RuntimeError("a complete measured prompt matches a prewarm prompt")

        telemetry.start()
        reset = reset_cache(server_manager)
        telemetry.mark("cache_reset", method=reset["method"])
        before_warmup = snapshot_metrics(
            config["server"]["metrics_url"], path / "metrics-before-prewarm.prom"
        )
        warmup_command = sender_command(
            config,
            path / "prewarm-prompts.jsonl",
            path / "prewarm.jsonl",
            path / "prewarm.summary.json",
            1,
            warmup_count,
        )
        telemetry.mark("prewarm_started")
        warmup_returncode = run_sender(
            warmup_command, REPO_ROOT, path / "prewarm.log"
        )
        telemetry.mark("prewarm_finished", returncode=warmup_returncode)
        if warmup_returncode:
            raise RuntimeError("excluded prefix prewarm failed")
        wait_for_idle(config["server"])
        before_measured = snapshot_metrics(
            config["server"]["metrics_url"], path / "metrics-before-measured.prom"
        )
        warmup_hits = metric_delta(before_measured, before_warmup, "prefix_hits")
        if warmup_hits not in (0, 0.0):
            raise RuntimeError(
                "cache reset verification failed: prewarm hit %s tokens" % warmup_hits
            )

        measured_command = sender_command(
            config,
            bank_path,
            path / "measured.jsonl",
            path / "measured.summary.json",
            condition["concurrency"],
            config["requests_per_repeat"],
        )
        atomic_write_json(
            path / "commands.json",
            {"prewarm": warmup_command, "measured": measured_command},
        )
        telemetry.mark("benchmark_started")
        returncode = run_sender(
            measured_command, REPO_ROOT, path / "measured.log"
        )
        telemetry.mark("benchmark_finished", returncode=returncode)
        wait_for_idle(config["server"])
        after_measured = snapshot_metrics(
            config["server"]["metrics_url"], path / "metrics-after-measured.prom"
        )
        queries = metric_delta(after_measured, before_measured, "prefix_queries")
        hits = metric_delta(after_measured, before_measured, "prefix_hits")
        actual = hits / queries if queries else None
        intended = intended_hit_rate(workload)
        cache = {
            "reset": reset,
            "reset_verified_by_zero_hit_prewarm": warmup_hits == 0,
            "prewarm_prefix_queries": metric_delta(
                before_measured, before_warmup, "prefix_queries"
            ),
            "prewarm_prefix_hits": warmup_hits,
            "measured_prefix_queries": queries,
            "measured_prefix_hits": hits,
            "measured_prefix_cache_token_hit_rate": actual,
            "intended_prefix_cache_token_hit_rate": intended,
            "absolute_rate_error": None if actual is None else abs(actual - intended),
            "preemptions_delta": metric_delta(
                after_measured, before_measured, "preemptions"
            ),
            "prompt_tokens_metric_delta": metric_delta(
                after_measured, before_measured, "prompt_tokens"
            ),
            "generation_tokens_metric_delta": metric_delta(
                after_measured, before_measured, "generation_tokens"
            ),
            "before_prewarm": before_warmup,
            "before_measured": before_measured,
            "after_measured": after_measured,
        }
        atomic_write_json(path / "cache.json", cache)
        summary = read_json(path / "measured.summary.json")
        counts = summary.get("counts", {})
        if (
            returncode
            or counts.get("attempted") != config["requests_per_repeat"]
            or counts.get("successful") != config["requests_per_repeat"]
            or counts.get("failed") != 0
        ):
            raise RuntimeError(
                "measured traffic incomplete or failed: rc=%s counts=%s"
                % (returncode, counts)
            )
        tolerance = config.get("cache_hit_rate_tolerance", 0.01)
        if actual is None or abs(actual - intended) > tolerance:
            raise RuntimeError(
                "observed cache token hit rate %s differs from intended %.6f"
                % (actual, intended)
            )
        state.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "prompt_bank": str(bank_path),
                "prompt_bank_metadata": str(metadata_path),
                "prompt_bank_sha256": sha256_path(bank_path),
                "cache_token_hit_rate": actual,
            }
        )
        atomic_write_json(path / "attempt.json", state)
        return path
    except BaseException as error:
        state.update(
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "completed_at_utc": utc_now(),
                "error": "%s: %s" % (type(error).__name__, error),
            }
        )
        atomic_write_json(path / "attempt.json", state)
        raise
    finally:
        telemetry.stop()


def initial_status(config, plan):
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": config["name"],
        "campaign_root": plan["campaign_root"],
        "status": "starting",
        "stage": "preflight",
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "completed_at_utc": None,
        "completed_conditions": 0,
        "total_conditions": plan["condition_count"],
        "current_condition": None,
        "last_completed_condition": None,
        "error": None,
    }


def update_status(path, status, **changes):
    status.update(changes)
    status["updated_at_utc"] = utc_now()
    atomic_write_json(path, status)


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError("existing campaign configuration differs: %s" % path)
        return
    atomic_write_json(path, value)


def run_campaign(config, config_path, campaign_root=None, tokenizer=None):
    validate_config(config)
    plan = build_plan(config, config_path, campaign_root=campaign_root)
    root = Path(plan["campaign_root"])
    root.mkdir(parents=True, exist_ok=True)
    write_once(root / "campaign.original.json", config)
    write_once(root / "campaign.resolved.json", plan)
    status_path = root / "status.json"
    status = initial_status(config, plan)
    completed = sum(
        selected_attempt(root, condition, config["requests_per_repeat"]) is not None
        for condition in plan["conditions"]
    )
    status["completed_conditions"] = completed
    atomic_write_json(status_path, status)
    gpu_before = preflight(root, REPO_ROOT, config["server"])
    required_context = max(
        bucket["prompt_tokens"] + bucket["output_tokens"]
        for workload in config["workloads"]
        for bucket in workload["buckets"]
    )
    manager = ManagedServer(
        config["server"],
        REPO_ROOT,
        root / "server",
        config["model"],
        required_context,
    )
    previous_handlers = {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt("received signal %d" % signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, interrupt)
    try:
        update_status(status_path, status, stage="server_start")
        manager.start()
        update_status(
            status_path,
            status,
            status="running",
            stage="conditions",
            server_pid=manager.process.pid,
        )
        tokenizer = tokenizer or generate_prompts.load_tokenizer(
            config["tokenizer"], revision=config["tokenizer_revision"]
        )
        for condition in plan["conditions"]:
            complete = selected_attempt(
                root, condition, config["requests_per_repeat"]
            )
            if complete is not None:
                continue
            old_attempts = existing_attempts(root, condition)
            number = old_attempts[-1][0] + 1 if old_attempts else 1
            label = "%s/c%d/r%d/a%d" % (
                condition["workload"],
                condition["concurrency"],
                condition["repeat"],
                number,
            )
            update_status(
                status_path,
                status,
                current_condition=label,
                current_started_at_utc=utc_now(),
            )
            completed_path = run_attempt(
                config, plan, condition, number, manager, tokenizer
            )
            completed += 1
            update_status(
                status_path,
                status,
                completed_conditions=completed,
                last_completed_condition=label,
                last_completed_attempt=str(completed_path),
                current_condition=None,
            )
            print(
                "completed %d/%d %s"
                % (completed, plan["condition_count"], label),
                flush=True,
            )
        update_status(status_path, status, stage="analysis")
        analyze(root)
        update_status(
            status_path,
            status,
            status="completed",
            stage="cleanup",
            completed_at_utc=utc_now(),
            current_condition=None,
        )
    except BaseException as error:
        update_status(
            status_path,
            status,
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            stage="cleanup",
            completed_at_utc=utc_now(),
            error="%s: %s" % (type(error).__name__, error),
        )
        raise
    finally:
        manager.stop()
        gpu_after = gpu_memory_snapshot()
        release_tolerance = config.get("gpu_release_tolerance_mib", 256)
        gpu_released = None
        if gpu_before is not None and gpu_after is not None:
            gpu_released = all(
                gpu_after.get(index, float("inf")) <= used + release_tolerance
                for index, used in gpu_before.items()
            )
        cleanup = {
            "checked_at_utc": utc_now(),
            "port_listening": port_is_open(
                config["server"]["host"], config["server"]["port"]
            ),
            "gpu_memory_before_mib": gpu_before,
            "gpu_memory_after_mib": gpu_after,
            "gpu_release_tolerance_mib": release_tolerance,
            "gpu_memory_released": gpu_released,
        }
        cleanup["verified"] = not cleanup["port_listening"] and gpu_released is not False
        collect_command(["nvidia-smi"], root / "server" / "nvidia-smi-after.txt")
        collect_command(
            ["ss", "-ltnp", "sport = :%d" % config["server"]["port"]],
            root / "server" / "port-after.txt",
        )
        atomic_write_json(root / "server" / "cleanup.json", cleanup)
        status["cleanup"] = cleanup
        status["stage"] = "done"
        atomic_write_json(status_path, status)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if not status["cleanup"]["verified"]:
        status.update(
            {
                "status": "failed",
                "error": "server cleanup did not release its port or GPU memory",
            }
        )
        atomic_write_json(status_path, status)
        raise RuntimeError(status["error"])
    return {"plan": plan, "status": status}


def telemetry_window(path):
    events_path = Path(path) / "telemetry" / "events.jsonl"
    if not events_path.exists():
        return None, [], []
    events = read_rows(events_path)
    starts = [row["experiment_offset_s"] for row in events if row["event"] == "benchmark_started"]
    ends = [row["experiment_offset_s"] for row in events if row["event"] == "benchmark_finished"]
    window = (starts[-1], ends[-1]) if starts and ends else None
    gpu_path = Path(path) / "telemetry" / "gpu.csv"
    gpu = list(csv.DictReader(gpu_path.open(encoding="utf-8"))) if gpu_path.exists() else []
    prom_path = Path(path) / "telemetry" / "vllm.prometheus.jsonl"
    prom = read_rows(prom_path) if prom_path.exists() else []
    if window:
        gpu = [row for row in gpu if window[0] <= float(row["experiment_offset_s"]) <= window[1]]
        prom = [row for row in prom if window[0] <= float(row["experiment_offset_s"]) <= window[1]]
    return window, gpu, prom


def numeric(rows, field):
    values = []
    for row in rows:
        try:
            values.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
            pass
    return values


def range_summary(values):
    return {
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def distribution(values):
    return send_requests.distribution([value for value in values if value is not None])


def rows_summary(rows, duration):
    successful = [row for row in rows if row.get("status") == 200]
    prompt_tokens = [row.get("prompt_tokens") for row in successful if row.get("prompt_tokens") is not None]
    output_tokens = [row.get("output_tokens") for row in successful if row.get("output_tokens") is not None]
    return {
        "attempted": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "request_throughput_per_s": len(successful) / duration if duration else None,
        "prompt_token_throughput_per_s": sum(prompt_tokens) / duration if duration else None,
        "output_token_throughput_per_s": sum(output_tokens) / duration if duration else None,
        "e2e_s": distribution([row.get("latency_s") for row in successful]),
        "ttft_s": distribution([row.get("ttft_s") for row in successful]),
        "approx_tpot_s": distribution(
            [row.get("approx_time_per_output_token_s") for row in successful]
        ),
        "prompt_tokens": {
            "total": sum(prompt_tokens),
            "min": min(prompt_tokens) if prompt_tokens else None,
            "max": max(prompt_tokens) if prompt_tokens else None,
        },
        "output_tokens": {
            "total": sum(output_tokens),
            "min": min(output_tokens) if output_tokens else None,
            "max": max(output_tokens) if output_tokens else None,
        },
        "finish_reasons": dict(sorted(Counter(row.get("finish_reason") for row in successful).items(), key=lambda item: str(item[0]))),
        "errors": [
            {
                "id": row.get("id"),
                "status": row.get("status"),
                "error": row.get("error"),
                "error_body": row.get("error_body"),
            }
            for row in rows
            if row.get("status") != 200
        ],
    }


def summarize_attempt(path, condition):
    rows = read_rows(Path(path) / "measured.jsonl")
    client = read_json(Path(path) / "measured.summary.json")
    cache = read_json(Path(path) / "cache.json")
    _, gpu, prometheus = telemetry_window(path)
    summary = rows_summary(rows, client["duration_s"])
    summary.update({"condition": condition, "attempt_path": str(path), "duration_s": client["duration_s"], "cache": cache})
    for name, metric in (("running", "running"), ("waiting", "waiting"), ("kv_usage", "kv_usage")):
        values = [metric_value(item["raw"], METRICS[metric]) for item in prometheus]
        summary[name] = range_summary([value for value in values if value is not None])
    gpu_fields = {
        "gpu_utilization_percent": "utilization_gpu_percent",
        "gpu_memory_used_mib": "memory_used_mib",
        "gpu_power_w": "power_draw_w",
        "gpu_temperature_c": "temperature_gpu_c",
        "gpu_sm_clock_mhz": "clocks_sm_mhz",
        "gpu_memory_clock_mhz": "clocks_memory_mhz",
    }
    summary["gpu"] = {
        name: range_summary(numeric(gpu, field)) for name, field in gpu_fields.items()
    }
    summary["classes"] = {
        name: rows_summary(
            [row for row in rows if row.get("workload") == name],
            client["duration_s"],
        )
        for name in sorted({row.get("workload") for row in rows})
    }
    return summary


def coefficient_of_variation(values):
    if len(values) < 2 or statistics.fmean(values) == 0:
        return None
    return statistics.stdev(values) / statistics.fmean(values)


def aggregate_group(repeats, class_name=None):
    sources = [item if class_name is None else item["classes"][class_name] for item in repeats]
    rows = []
    for item in repeats:
        selected = read_rows(Path(item["attempt_path"]) / "measured.jsonl")
        if class_name is not None:
            selected = [row for row in selected if row.get("workload") == class_name]
        rows.extend(selected)
    duration = sum(item["duration_s"] for item in repeats)
    result = rows_summary(rows, duration)
    for metric in (
        "request_throughput_per_s",
        "prompt_token_throughput_per_s",
        "output_token_throughput_per_s",
    ):
        values = [item[metric] for item in sources]
        result[metric.replace("_per_s", "_median_per_s")] = statistics.median(values)
        result[metric.replace("_per_s", "_repeat_cv")] = coefficient_of_variation(values)
    queries = sum(item["cache"]["measured_prefix_queries"] for item in repeats)
    hits = sum(item["cache"]["measured_prefix_hits"] for item in repeats)
    result["measured_prefix_cache_token_hit_rate"] = (
        hits / queries if queries and class_name is None else None
    )
    result["intended_prefix_cache_token_hit_rate"] = (
        statistics.fmean(item["cache"]["intended_prefix_cache_token_hit_rate"] for item in repeats)
        if class_name is None
        else statistics.fmean(
            row.get("shared_prefix_tokens", 0) / row["prompt_tokens"]
            for item in repeats
            for row in read_rows(Path(item["attempt_path"]) / "measured.jsonl")
            if row.get("workload") == class_name
        )
    )
    result["preemptions"] = sum(item["cache"]["preemptions_delta"] or 0 for item in repeats)
    if class_name is None:
        for field in ("running", "waiting", "kv_usage"):
            values = [item[field]["mean"] for item in repeats if item[field]["mean"] is not None]
            maxima = [item[field]["max"] for item in repeats if item[field]["max"] is not None]
            result[field] = {"mean": statistics.fmean(values) if values else None, "max": max(maxima) if maxima else None}
        result["gpu"] = {}
        for field in repeats[0]["gpu"]:
            values = [item["gpu"][field]["mean"] for item in repeats if item["gpu"][field]["mean"] is not None]
            maxima = [item["gpu"][field]["max"] for item in repeats if item["gpu"][field]["max"] is not None]
            result["gpu"][field] = {"mean": statistics.fmean(values) if values else None, "max": max(maxima) if maxima else None}
    return result


def transition_analysis(rows):
    transitions = []
    formal = None
    streak = 0
    for old, new in zip(rows, rows[1:]):
        old_tps = old["output_token_throughput_median_per_s"]
        new_tps = new["output_token_throughput_median_per_s"]
        old_p99 = old["e2e_s"]["p99"]
        new_p99 = new["e2e_s"]["p99"]
        gain = (new_tps - old_tps) / old_tps if old_tps else None
        rise = (new_p99 - old_p99) / old_p99 if old_p99 else None
        waiting = new.get("waiting", {}).get("max")
        if gain is not None and gain >= 0.05:
            interpretation = "useful batching"
        elif (rise is not None and rise >= 0.20) or (waiting is not None and waiting > 0):
            interpretation = "queueing-dominant"
        else:
            interpretation = "throughput plateau"
        qualifies = gain is not None and rise is not None and gain < 0.05 and rise >= 0.20
        streak = streak + 1 if qualifies else 0
        if streak >= 2 and formal is None:
            formal = old["concurrency"]
        transitions.append({"from": old["concurrency"], "to": new["concurrency"], "output_throughput_change_fraction": gain, "p99_change_fraction": rise, "interpretation": interpretation})
    return {"formal_elbow": formal, "transitions": transitions}


def display(value, digits=2):
    return "n/a" if value is None else ("%%.%df" % digits) % value


def analyze(root):
    root = Path(root)
    config = read_json(root / "campaign.original.json")
    plan = read_json(root / "campaign.resolved.json")
    repeats = []
    for condition in plan["conditions"]:
        path = selected_attempt(root, condition, config["requests_per_repeat"])
        if path is None:
            raise RuntimeError("campaign is incomplete at %s" % condition)
        repeats.append(summarize_attempt(path, condition))
    grouped = []
    for workload in config["workloads"]:
        for concurrency in config["concurrency"]:
            selected = [item for item in repeats if item["condition"]["workload"] == workload["name"] and item["condition"]["concurrency"] == concurrency]
            aggregate = aggregate_group(selected)
            aggregate.update({"workload": workload["name"], "class": "aggregate", "concurrency": concurrency, "repeats": len(selected)})
            grouped.append(aggregate)
            if len(workload["buckets"]) > 1:
                for bucket in workload["buckets"]:
                    result = aggregate_group(selected, bucket["name"])
                    result.update({"workload": workload["name"], "class": bucket["name"], "concurrency": concurrency, "repeats": len(selected)})
                    grouped.append(result)
    elbows = {}
    for workload in config["workloads"]:
        rows = [item for item in grouped if item["workload"] == workload["name"] and item["class"] == "aggregate"]
        elbows[workload["name"]] = transition_analysis(rows)
    failures = sum(item["failed"] for item in grouped if item["class"] == "aggregate")
    cvs = [item["output_token_throughput_repeat_cv"] for item in grouped if item["class"] == "aggregate" and item["output_token_throughput_repeat_cv"] is not None]
    if failures or (cvs and max(cvs) > 0.10):
        recommendation = "another baseline campaign"
        reason = "failures or greater than 10% repeat variability prevent a stable parameter comparison"
    elif all(value["formal_elbow"] is not None for value in elbows.values()):
        recommendation = "a server-parameter campaign"
        reason = "all workloads have a repeatable throughput/latency elbow"
    else:
        recommendation = "another baseline campaign"
        reason = "at least one workload lacks a demonstrated two-transition elbow"
    analysis = {"schema_version": SCHEMA_VERSION, "generated_at_utc": utc_now(), "repeat_results": repeats, "results": grouped, "elbows": elbows, "recommendation": {"next": recommendation, "reason": reason}}
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(analysis_dir / "analysis.json", analysis)
    fields = ["workload", "class", "concurrency", "repeats", "attempted", "successful", "failed", "request_throughput_median_per_s", "prompt_token_throughput_median_per_s", "output_token_throughput_median_per_s", "output_token_throughput_repeat_cv", "e2e_p50_s", "e2e_p90_s", "e2e_p99_s", "ttft_p50_s", "ttft_p90_s", "ttft_p99_s", "tpot_p50_s", "tpot_p90_s", "tpot_p99_s", "measured_prefix_cache_token_hit_rate", "intended_prefix_cache_token_hit_rate", "running_mean", "running_max", "waiting_mean", "waiting_max", "kv_mean", "kv_max", "preemptions", "gpu_util_mean", "gpu_util_max", "gpu_memory_max_mib", "gpu_power_mean_w", "gpu_power_max_w", "gpu_temperature_max_c", "gpu_sm_clock_mean_mhz", "gpu_memory_clock_mean_mhz"]
    with (analysis_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in grouped:
            aggregate = item["class"] == "aggregate"
            writer.writerow({"workload": item["workload"], "class": item["class"], "concurrency": item["concurrency"], "repeats": item["repeats"], "attempted": item["attempted"], "successful": item["successful"], "failed": item["failed"], "request_throughput_median_per_s": item["request_throughput_median_per_s"], "prompt_token_throughput_median_per_s": item["prompt_token_throughput_median_per_s"], "output_token_throughput_median_per_s": item["output_token_throughput_median_per_s"], "output_token_throughput_repeat_cv": item["output_token_throughput_repeat_cv"], **{"e2e_%s_s" % p: item["e2e_s"][p] for p in ("p50", "p90", "p99")}, **{"ttft_%s_s" % p: item["ttft_s"][p] for p in ("p50", "p90", "p99")}, **{"tpot_%s_s" % p: item["approx_tpot_s"][p] for p in ("p50", "p90", "p99")}, "measured_prefix_cache_token_hit_rate": item["measured_prefix_cache_token_hit_rate"], "intended_prefix_cache_token_hit_rate": item["intended_prefix_cache_token_hit_rate"], "running_mean": item.get("running", {}).get("mean") if aggregate else None, "running_max": item.get("running", {}).get("max") if aggregate else None, "waiting_mean": item.get("waiting", {}).get("mean") if aggregate else None, "waiting_max": item.get("waiting", {}).get("max") if aggregate else None, "kv_mean": item.get("kv_usage", {}).get("mean") if aggregate else None, "kv_max": item.get("kv_usage", {}).get("max") if aggregate else None, "preemptions": item["preemptions"], "gpu_util_mean": item.get("gpu", {}).get("gpu_utilization_percent", {}).get("mean") if aggregate else None, "gpu_util_max": item.get("gpu", {}).get("gpu_utilization_percent", {}).get("max") if aggregate else None, "gpu_memory_max_mib": item.get("gpu", {}).get("gpu_memory_used_mib", {}).get("max") if aggregate else None, "gpu_power_mean_w": item.get("gpu", {}).get("gpu_power_w", {}).get("mean") if aggregate else None, "gpu_power_max_w": item.get("gpu", {}).get("gpu_power_w", {}).get("max") if aggregate else None, "gpu_temperature_max_c": item.get("gpu", {}).get("gpu_temperature_c", {}).get("max") if aggregate else None, "gpu_sm_clock_mean_mhz": item.get("gpu", {}).get("gpu_sm_clock_mhz", {}).get("mean") if aggregate else None, "gpu_memory_clock_mean_mhz": item.get("gpu", {}).get("gpu_memory_clock_mhz", {}).get("mean") if aggregate else None})
    lines = ["# Cache-controlled capacity campaign", "", "Warmups and drain periods are excluded. Throughput is the median of three measured repeats; latency percentiles pool successful measured requests.", ""]
    for workload in config["workloads"]:
        lines.extend(["## " + workload["name"], "", "| C | ok/fail | req/s | prompt tok/s | output tok/s | E2E p50/p90/p99 s | TTFT p50/p90/p99 s | TPOT p50/p90/p99 ms | cache token hit | run max | wait max | KV max | GPU mean/max | preemptions |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        rows = [item for item in grouped if item["workload"] == workload["name"] and item["class"] == "aggregate"]
        for item in rows:
            lines.append("| {c} | {ok}/{bad} | {rps} | {ptps} | {otps} | {e50}/{e90}/{e99} | {t50}/{t90}/{t99} | {p50}/{p90}/{p99} | {hit}% | {run} | {wait} | {kv}% | {gpu}/{gpu_max}% | {preempt} |".format(c=item["concurrency"], ok=item["successful"], bad=item["failed"], rps=display(item["request_throughput_median_per_s"]), ptps=display(item["prompt_token_throughput_median_per_s"], 0), otps=display(item["output_token_throughput_median_per_s"], 0), e50=display(item["e2e_s"]["p50"], 3), e90=display(item["e2e_s"]["p90"], 3), e99=display(item["e2e_s"]["p99"], 3), t50=display(item["ttft_s"]["p50"], 3), t90=display(item["ttft_s"]["p90"], 3), t99=display(item["ttft_s"]["p99"], 3), p50=display(item["approx_tpot_s"]["p50"] * 1000 if item["approx_tpot_s"]["p50"] is not None else None), p90=display(item["approx_tpot_s"]["p90"] * 1000 if item["approx_tpot_s"]["p90"] is not None else None), p99=display(item["approx_tpot_s"]["p99"] * 1000 if item["approx_tpot_s"]["p99"] is not None else None), hit=display(item["measured_prefix_cache_token_hit_rate"] * 100 if item["measured_prefix_cache_token_hit_rate"] is not None else None), run=display(item["running"]["max"], 0), wait=display(item["waiting"]["max"], 0), kv=display(item["kv_usage"]["max"] * 100 if item["kv_usage"]["max"] is not None else None, 1), gpu=display(item["gpu"]["gpu_utilization_percent"]["mean"], 1), gpu_max=display(item["gpu"]["gpu_utilization_percent"]["max"], 0), preempt=display(item["preemptions"], 0)))
        elbow = elbows[workload["name"]]
        lines.extend(
            [
                "",
                "| Transition | output throughput change | P99 change | interpretation |",
                "| :--- | ---: | ---: | :--- |",
            ]
        )
        for transition in elbow["transitions"]:
            lines.append(
                "| {old}→{new} | {gain}% | {rise}% | {kind} |".format(
                    old=transition["from"],
                    new=transition["to"],
                    gain=display(
                        transition["output_throughput_change_fraction"] * 100
                        if transition["output_throughput_change_fraction"]
                        is not None
                        else None,
                        1,
                    ),
                    rise=display(
                        transition["p99_change_fraction"] * 100
                        if transition["p99_change_fraction"] is not None
                        else None,
                        1,
                    ),
                    kind=transition["interpretation"],
                )
            )
        lines.extend(
            [
                "",
                "Elbow: **%s**."
                % (
                    elbow["formal_elbow"]
                    if elbow["formal_elbow"] is not None
                    else "not demonstrated"
                ),
                "",
            ]
        )
        if len(workload["buckets"]) > 1:
            lines.extend(["### Mixed request classes", "", "Per-class rows use the complete repeat duration because classes overlap. vLLM exposes cache counters only in aggregate, so per-class measured hit rate is unavailable; the intended class rate is retained.", "", "| Class | C | ok/fail | req/s | prompt tok/s | output tok/s | E2E p50/p90/p99 s | TTFT p50/p90/p99 s | TPOT p50/p90/p99 ms | intended cache rate |", "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
            for item in [row for row in grouped if row["workload"] == workload["name"] and row["class"] != "aggregate"]:
                lines.append("| {cls} | {c} | {ok}/{bad} | {rps} | {ptps} | {otps} | {e50}/{e90}/{e99} | {t50}/{t90}/{t99} | {p50}/{p90}/{p99} | {hit}% |".format(cls=item["class"], c=item["concurrency"], ok=item["successful"], bad=item["failed"], rps=display(item["request_throughput_median_per_s"]), ptps=display(item["prompt_token_throughput_median_per_s"], 0), otps=display(item["output_token_throughput_median_per_s"], 0), e50=display(item["e2e_s"]["p50"], 3), e90=display(item["e2e_s"]["p90"], 3), e99=display(item["e2e_s"]["p99"], 3), t50=display(item["ttft_s"]["p50"], 3), t90=display(item["ttft_s"]["p90"], 3), t99=display(item["ttft_s"]["p99"], 3), p50=display(item["approx_tpot_s"]["p50"] * 1000 if item["approx_tpot_s"]["p50"] is not None else None), p90=display(item["approx_tpot_s"]["p90"] * 1000 if item["approx_tpot_s"]["p90"] is not None else None), p99=display(item["approx_tpot_s"]["p99"] * 1000 if item["approx_tpot_s"]["p99"] is not None else None), hit=display(item["intended_prefix_cache_token_hit_rate"] * 100, 2)))
            lines.append("")
    hit_rates = [
        item["measured_prefix_cache_token_hit_rate"]
        for item in grouped
        if item["class"] == "aggregate"
        and item["measured_prefix_cache_token_hit_rate"] is not None
    ]
    lines.extend(
        [
            "## Integrity",
            "",
            "Measured failures: **%d**. Cache token-hit range: **%s%%–%s%%**. "
            "Maximum output-throughput repeat CV: **%s%%**."
            % (
                failures,
                display(min(hit_rates) * 100 if hit_rates else None, 2),
                display(max(hit_rates) * 100 if hit_rates else None, 2),
                display(max(cvs) * 100 if cvs else None, 2),
            ),
            "",
            "Actual prompt/output token totals and ranges, finish reasons, and "
            "bounded error bodies are retained in `analysis.json`; per-request "
            "values remain in each selected attempt's `measured.jsonl`.",
            "",
            "## Recommendation",
            "",
            "Next: **%s** — %s." % (recommendation, reason),
            "",
            "The cache hit rate above is vLLM's measured token-hit/query delta, "
            "not an inference from prompt construction. Exact prompt lengths, "
            "complete-prompt hashes, shared-prefix hashes, finish reasons, errors, "
            "telemetry, and every abandoned attempt remain in the campaign directory.",
            "",
        ]
    )
    (analysis_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="validate and print the plan")
    plan_parser.add_argument("--config", required=True)
    plan_parser.add_argument("--campaign-root")
    run_parser = subparsers.add_parser("run", help="run or resume a campaign")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--campaign-root")
    status_parser = subparsers.add_parser("status", help="print compact progress")
    status_parser.add_argument("--campaign-root", required=True)
    analyze_parser = subparsers.add_parser("analyze", help="rebuild analysis")
    analyze_parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()
    try:
        if args.command == "status":
            print(json.dumps(read_json(Path(args.campaign_root) / "status.json"), indent=2, sort_keys=True))
        elif args.command == "analyze":
            analyze(args.campaign_root)
        else:
            config = load_config(args.config)
            if args.command == "plan":
                print(json.dumps(build_plan(config, args.config, args.campaign_root), indent=2, sort_keys=True))
            else:
                result = run_campaign(config, args.config, args.campaign_root)
                print(json.dumps(result["status"], indent=2, sort_keys=True))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, "error: %s\n" % error)


if __name__ == "__main__":
    main()
