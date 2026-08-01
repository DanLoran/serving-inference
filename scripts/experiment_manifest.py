"""Collect a sanitized, versioned manifest for an experiment run."""

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


MANIFEST_SCHEMA_VERSION = "1.0"
REDACTED = "[REDACTED]"
SENSITIVE_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|authorization|cookie|credential|passwd|password|secret|token)([_-]|$)",
    re.IGNORECASE,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_url(value):
    try:
        parts = urlsplit(value)
    except (TypeError, ValueError):
        return value
    if not parts.scheme or not parts.netloc:
        return value
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = "[%s]" % host
    if parts.port:
        host += ":%d" % parts.port
    query = [
        (key, REDACTED if SENSITIVE_KEY.search(key) else item)
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit((parts.scheme, host, parts.path, urlencode(query), parts.fragment))


def sanitize(value, key=None):
    """Redact credential-shaped fields without reading unrelated environment data."""
    if key is not None and SENSITIVE_KEY.search(str(key)):
        return REDACTED
    if isinstance(value, dict):
        return {str(item_key): sanitize(item, item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        cleaned = []
        redact_next = False
        for item in value:
            if redact_next:
                cleaned.append(REDACTED)
                redact_next = False
                continue
            if isinstance(item, str) and item.startswith("-"):
                flag, separator, _ = item.partition("=")
                if SENSITIVE_KEY.search(flag.lstrip("-")):
                    cleaned.append(flag + "=" + REDACTED if separator else flag)
                    redact_next = not separator
                    continue
            cleaned.append(sanitize(item))
        return cleaned
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        if value.lower().startswith("bearer "):
            return REDACTED
        return sanitize_url(value)
    return value


def run_command(command, cwd=None, timeout=5):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={"PATH": os.environ.get("PATH", "")},
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": type(error).__name__}
    if result.returncode != 0:
        return {"available": False, "reason": "exit_code_%d" % result.returncode}
    return {"available": True, "output": result.stdout.strip()}


def collect_git(repo_root):
    revision = run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = run_command(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo_root)
    return {
        "available": revision["available"] and status["available"],
        "revision": revision.get("output"),
        "dirty": bool(status.get("output")) if status["available"] else None,
    }


def collect_python():
    dependencies = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            dependencies.append({"name": name, "version": distribution.version})
    dependencies.sort(key=lambda item: item["name"].lower())
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "dependencies": dependencies,
    }


def collect_accelerator():
    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not query["available"]:
        return {
            "nvidia_smi_available": False,
            "gpus": [],
            "driver_version": None,
            "cuda_version": None,
            "reason": query["reason"],
        }
    gpus = []
    driver = None
    for line in query["output"].splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3:
            driver = driver or fields[2]
            try:
                vram_mib = int(float(fields[1]))
            except ValueError:
                vram_mib = None
            gpus.append({"name": fields[0], "vram_mib": vram_mib})
    cuda_query = run_command(["nvidia-smi"])
    cuda_match = re.search(r"CUDA Version\s*:\s*([^\s]+)", cuda_query.get("output", ""))
    return {
        "nvidia_smi_available": True,
        "gpus": gpus,
        "driver_version": driver,
        "cuda_version": cuda_match.group(1) if cuda_match else None,
    }


def collect_vllm():
    try:
        distribution = importlib.metadata.distribution("vllm")
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None, "fingerprint": None}
    record = distribution.read_text("RECORD") or ""
    fingerprint_source = (distribution.version + "\n" + record).encode("utf-8")
    return {
        "installed": True,
        "version": distribution.version,
        "fingerprint": "sha256:" + hashlib.sha256(fingerprint_source).hexdigest(),
    }


def build_manifest(original, resolved, repo_root, started_at, completed_at=None, status="running"):
    prompts = Path(resolved["prompts"])
    model = {"name": resolved["model"], **resolved["model_metadata"]}
    server = resolved["server"]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment": {
            "name": resolved["name"],
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "status": status,
            "warmups": resolved["warmups"],
            "repeats": resolved["repeats"],
        },
        "config": {"original": sanitize(original), "resolved": sanitize(resolved)},
        "workload": {
            "path": str(prompts),
            "sha256": sha256_file(prompts),
        },
        "source": {"git": collect_git(repo_root)},
        "environment": {
            "python": collect_python(),
            "system": {
                "os": platform.system(),
                "release": platform.release(),
                "kernel": platform.version(),
                "machine": platform.machine(),
            },
            "accelerator": collect_accelerator(),
        },
        "serving": {
            "vllm": collect_vllm(),
            "model": sanitize(model),
            "server": sanitize(server),
        },
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    required = {"schema_version", "experiment", "config", "workload", "source", "environment", "serving"}
    missing = required.difference(manifest)
    if missing:
        raise ValueError("manifest missing fields: %s" % ", ".join(sorted(missing)))
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema version")
    if len(manifest["workload"].get("sha256", "")) != 64:
        raise ValueError("invalid workload SHA-256")
    if manifest["experiment"].get("status") not in {"running", "completed", "failed"}:
        raise ValueError("invalid experiment status")
    encoded = json.dumps(manifest).lower()
    for marker in ("bearer ", "-----begin private key-----"):
        if marker in encoded:
            raise ValueError("manifest contains credential material")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
