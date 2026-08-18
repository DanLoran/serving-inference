#!/usr/bin/env python3
"""Launch a vLLM server under Nsight and drive a separate bounded workload."""

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = "1.0"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REQUIRED_NSYS_TRACES = {"cuda", "nvtx", "osrt"}
VALID_NCU_REPLAY_MODES = {"kernel", "application", "range", "app-range"}


class ConfigError(ValueError):
    """Raised when a profiling configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class ProfilePlan:
    tool: str
    profile_name: str
    profile_dir: Path
    report_prefix: Path
    expected_report: Path
    server_command: tuple
    profiler_command: tuple
    workload_command: tuple
    config: dict
    config_path: Path


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require_string(config, key):
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError("%s must be a non-empty string" % key)
    return value


def require_safe_name(config, key):
    value = require_string(config, key)
    if not SAFE_NAME.fullmatch(value):
        raise ConfigError(
            "%s must contain only letters, numbers, '.', '_' or '-'" % key
        )
    return value


def require_positive_int(config, key):
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError("%s must be a positive integer" % key)
    return value


def require_nonnegative_number(config, key):
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ConfigError("%s must be a non-negative number" % key)
    return value


def require_positive_number(config, key):
    value = require_nonnegative_number(config, key)
    if value == 0:
        raise ConfigError("%s must be greater than zero" % key)
    return value


def resolve_path(repo_root, value):
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_config(path):
    config_path = Path(path).resolve()
    try:
        payload = config_path.read_bytes()
        config = json.loads(payload.decode("utf-8"))
    except FileNotFoundError as error:
        raise ConfigError("config does not exist: %s" % config_path) from error
    except UnicodeDecodeError as error:
        raise ConfigError("config is not valid UTF-8: %s" % config_path) from error
    except json.JSONDecodeError as error:
        raise ConfigError("invalid JSON in %s: %s" % (config_path, error)) from error
    if not isinstance(config, dict):
        raise ConfigError("config must be a JSON object")
    return config_path, config, hashlib.sha256(payload).hexdigest()


def validate_common(config):
    experiment = require_safe_name(config, "experiment")
    workload = require_safe_name(config, "workload")
    concurrency = require_positive_int(config, "concurrency")
    require_positive_int(config, "num_requests")
    for key in ("prompts", "model", "url", "health_url", "output_dir"):
        require_string(config, key)
    require_nonnegative_number(config, "temperature")
    require_positive_number(config, "startup_timeout_s")
    require_positive_number(config, "request_timeout_s")
    require_positive_number(config, "profiler_shutdown_timeout_s")
    if not isinstance(config.get("stream"), bool):
        raise ConfigError("stream must be true or false")
    server_command = config.get("server_command")
    if (
        not isinstance(server_command, list)
        or not server_command
        or not all(isinstance(arg, str) and arg for arg in server_command)
    ):
        raise ConfigError("server_command must be a non-empty array of strings")
    return experiment, workload, concurrency


def build_nsys_command(config, report_prefix, server_command):
    options = config.get("nsys")
    if not isinstance(options, dict):
        raise ConfigError("nsys must be a JSON object")
    traces = options.get("trace")
    if (
        not isinstance(traces, list)
        or not traces
        or not all(isinstance(item, str) and item for item in traces)
    ):
        raise ConfigError("nsys.trace must be a non-empty array of strings")
    missing = REQUIRED_NSYS_TRACES - set(traces)
    if missing:
        raise ConfigError(
            "nsys.trace must include %s" % ", ".join(sorted(REQUIRED_NSYS_TRACES))
        )
    sample = options.get("sample", "none")
    if not isinstance(sample, str) or not sample:
        raise ConfigError("nsys.sample must be a non-empty string")
    delay = require_nonnegative_number(options, "delay_s")
    duration = require_positive_number(options, "duration_s")
    command = [
        "nsys",
        "profile",
        "--trace=%s" % ",".join(traces),
        "--sample=%s" % sample,
        "--delay=%s" % delay,
        "--duration=%s" % duration,
        "--kill=sigterm",
        "--force-overwrite=false",
        "--stats=true",
        "--output=%s" % report_prefix,
    ]
    return tuple(command + list(server_command))


def build_ncu_command(config, report_prefix, server_command):
    options = config.get("ncu")
    if not isinstance(options, dict):
        raise ConfigError("ncu must be a JSON object")
    section_set = require_string(options, "set")
    replay_mode = require_string(options, "replay_mode")
    if replay_mode not in VALID_NCU_REPLAY_MODES:
        raise ConfigError(
            "ncu.replay_mode must be one of %s"
            % ", ".join(sorted(VALID_NCU_REPLAY_MODES))
        )
    kernel_name = require_string(options, "kernel_name")
    launch_skip = require_nonnegative_number(options, "launch_skip")
    if not isinstance(launch_skip, int):
        raise ConfigError("ncu.launch_skip must be an integer")
    launch_count = require_positive_int(options, "launch_count")
    command = [
        "ncu",
        "--target-processes=all",
        "--set=%s" % section_set,
        "--replay-mode=%s" % replay_mode,
        "--kernel-name=%s" % kernel_name,
        "--kernel-name-base=demangled",
        "--launch-skip=%s" % launch_skip,
        "--launch-count=%s" % launch_count,
        "--export=%s" % report_prefix,
    ]
    return tuple(command + list(server_command))


def build_workload_command(config, repo_root, profile_dir):
    command = [
        sys.executable,
        str(repo_root / "scripts" / "send_requests.py"),
        "--prompts",
        str(resolve_path(repo_root, config["prompts"])),
        "--output",
        str(profile_dir / "client.jsonl"),
        "--summary",
        str(profile_dir / "client.summary.json"),
        "--url",
        config["url"],
        "--model",
        config["model"],
        "--num-requests",
        str(config["num_requests"]),
        "--concurrency",
        str(config["concurrency"]),
        "--temperature",
        str(config["temperature"]),
        "--timeout",
        str(config["request_timeout_s"]),
    ]
    if config["stream"]:
        command.append("--stream")
    return tuple(command)


def build_plan(tool, config, config_path, repo_root):
    if tool not in {"nsys", "ncu"}:
        raise ConfigError("tool must be nsys or ncu")
    experiment, workload, concurrency = validate_common(config)
    profile_name = "%s--%s--c%d--%s" % (
        experiment,
        workload,
        concurrency,
        tool,
    )
    output_dir = resolve_path(repo_root, config["output_dir"])
    profile_dir = output_dir / profile_name
    report_prefix = profile_dir / "capture"
    expected_report = report_prefix.with_suffix(
        ".nsys-rep" if tool == "nsys" else ".ncu-rep"
    )
    server_command = tuple(config["server_command"])
    if tool == "nsys":
        profiler_command = build_nsys_command(
            config, report_prefix, server_command
        )
    else:
        profiler_command = build_ncu_command(config, report_prefix, server_command)
    workload_command = build_workload_command(config, repo_root, profile_dir)
    return ProfilePlan(
        tool=tool,
        profile_name=profile_name,
        profile_dir=profile_dir,
        report_prefix=report_prefix,
        expected_report=expected_report,
        server_command=server_command,
        profiler_command=profiler_command,
        workload_command=workload_command,
        config=config,
        config_path=Path(config_path),
    )


def print_plan(plan):
    print("Profile: %s" % plan.profile_name)
    print("Profile directory: %s" % plan.profile_dir)
    print("Profiler/server command: %s" % shlex.join(plan.profiler_command))
    print("Workload/client command: %s" % shlex.join(plan.workload_command))
    print("Dry run: no directories, profilers, servers, or clients were started.")


def command_output(command, cwd):
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip(), completed.returncode


def git_state(repo_root):
    revision, revision_code = command_output(["git", "rev-parse", "HEAD"], repo_root)
    status, status_code = command_output(
        ["git", "status", "--porcelain=v1"], repo_root
    )
    return {
        "revision": revision if revision_code == 0 else None,
        "dirty": bool(status) if status_code == 0 else None,
        "status_porcelain": status.splitlines() if status_code == 0 else [],
    }


def tool_version(tool, repo_root):
    output, code = command_output([tool, "--version"], repo_root)
    if code != 0:
        raise RuntimeError("%s --version failed with exit code %d" % (tool, code))
    return output


def model_is_ready(health_url, model, timeout_s=2.0):
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_s) as response:
            if response.status != 200:
                return False, "HTTP %s" % response.status
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as error:
        return False, "%s: %s" % (type(error).__name__, error)
    model_ids = [
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict)
    ]
    if model not in model_ids:
        return False, "expected model %r; endpoint advertised %r" % (model, model_ids)
    return True, None


def wait_for_server(plan, profiler_process):
    deadline = time.monotonic() + plan.config["startup_timeout_s"]
    last_error = "server has not been checked"
    while time.monotonic() < deadline:
        exit_code = profiler_process.poll()
        if exit_code is not None:
            raise RuntimeError(
                "profiler/server exited before readiness with code %d" % exit_code
            )
        ready, error = model_is_ready(
            plan.config["health_url"], plan.config["model"]
        )
        if ready:
            return
        last_error = error
        time.sleep(1)
    raise TimeoutError(
        "server did not become ready within %ss: %s"
        % (plan.config["startup_timeout_s"], last_error)
    )


def stop_process_group(process, timeout_s):
    actions = []
    if process.poll() is not None:
        return actions
    for action, sig in (
        ("sigint", signal.SIGINT),
        ("sigterm", signal.SIGTERM),
        ("sigkill", signal.SIGKILL),
    ):
        try:
            os.killpg(process.pid, sig)
            actions.append(action)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=timeout_s)
            break
        except subprocess.TimeoutExpired:
            continue
    return actions


def write_metadata(path, metadata):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def inferred_failure_reason(plan, metadata):
    workload_exit = metadata.get("workload_exit_code")
    if workload_exit is None:
        return "controlled workload did not run; inspect the profiler/server log"
    if workload_exit != 0:
        return "controlled workload exited with code %s" % workload_exit

    log_path = plan.profile_dir / "server.log"
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log_text = ""
    if plan.tool == "ncu" and "ERR_NVGPUCTRPERM" in log_text:
        return "Nsight Compute cannot access GPU performance counters (ERR_NVGPUCTRPERM)"
    if plan.tool == "ncu" and "No kernels were profiled" in log_text:
        return "Nsight Compute profiled no matching kernels; review the kernel filter"
    if not plan.expected_report.exists():
        return "profiler report was not produced; inspect the profiler/server log"
    return "profiler capture did not complete successfully"


def base_metadata(plan, config_sha256, version, repo_root):
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "starting",
        "profile_name": plan.profile_name,
        "tool": plan.tool,
        "tool_version": version,
        "experiment": plan.config["experiment"],
        "workload": plan.config["workload"],
        "concurrency": plan.config["concurrency"],
        "num_requests": plan.config["num_requests"],
        "config_path": str(plan.config_path),
        "config_sha256": config_sha256,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "git": git_state(repo_root),
        "commands": {
            "server_argv": list(plan.server_command),
            "profiler_server_argv": list(plan.profiler_command),
            "workload_client_argv": list(plan.workload_command),
        },
        "artifacts": {
            "profile_dir": str(plan.profile_dir),
            "expected_report": str(plan.expected_report),
            "server_log": str(plan.profile_dir / "server.log"),
            "client_results": str(plan.profile_dir / "client.jsonl"),
            "client_summary": str(plan.profile_dir / "client.summary.json"),
        },
        "profiler_exit_code": None,
        "workload_exit_code": None,
        "cleanup_actions": [],
        "error": None,
    }


def run_capture(plan, config_sha256, repo_root):
    if plan.profile_dir.exists():
        raise RuntimeError(
            "profile directory already exists; choose a new experiment: %s"
            % plan.profile_dir
        )
    if shutil.which(plan.tool) is None:
        raise RuntimeError("required profiler is not on PATH: %s" % plan.tool)
    prompts_path = resolve_path(repo_root, plan.config["prompts"])
    if not prompts_path.is_file():
        raise RuntimeError("prompt artifact does not exist: %s" % prompts_path)

    version = tool_version(plan.tool, repo_root)
    plan.profile_dir.mkdir(parents=True)
    metadata_path = plan.profile_dir / "metadata.json"
    metadata = base_metadata(plan, config_sha256, version, repo_root)
    write_metadata(metadata_path, metadata)
    profiler_process = None
    try:
        with (plan.profile_dir / "server.log").open(
            "w", encoding="utf-8"
        ) as server_log:
            profiler_process = subprocess.Popen(
                plan.profiler_command,
                cwd=repo_root,
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            metadata["profiler_pid"] = profiler_process.pid
            metadata["status"] = "waiting_for_server"
            write_metadata(metadata_path, metadata)
            wait_for_server(plan, profiler_process)
            metadata["server_ready_at_utc"] = utc_now()
            metadata["status"] = "running_workload"
            write_metadata(metadata_path, metadata)
            workload = subprocess.run(
                plan.workload_command,
                cwd=repo_root,
                text=True,
                check=False,
            )
            metadata["workload_exit_code"] = workload.returncode
            metadata["status"] = "finalizing_profile"
            write_metadata(metadata_path, metadata)
    except KeyboardInterrupt:
        metadata["error"] = "interrupted by user"
    except Exception as error:
        metadata["error"] = "%s: %s" % (type(error).__name__, error)
    finally:
        if profiler_process is not None:
            metadata["cleanup_actions"] = stop_process_group(
                profiler_process,
                plan.config["profiler_shutdown_timeout_s"],
            )
            metadata["profiler_exit_code"] = profiler_process.poll()
        metadata["finished_at_utc"] = utc_now()
        report_exists = plan.expected_report.exists()
        metadata["artifacts"]["report_exists"] = report_exists
        if (
            metadata["error"] is None
            and metadata["workload_exit_code"] == 0
            and report_exists
        ):
            metadata["status"] = "complete"
        else:
            metadata["status"] = "failed"
            if metadata["error"] is None:
                metadata["error"] = inferred_failure_reason(plan, metadata)
        write_metadata(metadata_path, metadata)

    print("Metadata: %s" % metadata_path)
    print("Server/profiler log: %s" % (plan.profile_dir / "server.log"))
    if metadata["status"] != "complete":
        print("Capture failed: %s" % metadata["error"], file=sys.stderr)
        return 2
    print("Profile report: %s" % plan.expected_report)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=("nsys", "ncu"))
    parser.add_argument("--config", required=True, help="Profiling JSON config.")
    parser.add_argument(
        "--dry-run",
        "--print-command",
        action="store_true",
        dest="dry_run",
        help="Validate and print commands without starting processes or writing files.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        config_path, config, config_sha256 = load_config(args.config)
        plan = build_plan(args.tool, config, config_path, repo_root)
        if args.dry_run:
            print_plan(plan)
            return 0
        return run_capture(plan, config_sha256, repo_root)
    except (ConfigError, RuntimeError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
