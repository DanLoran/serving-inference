"""Fault-tolerant, timestamped GPU and Prometheus telemetry collectors."""

import csv
import io
import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


GPU_FIELDS = (
    "index",
    "uuid",
    "utilization_gpu_percent",
    "utilization_memory_percent",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_w",
    "temperature_gpu_c",
    "clocks_sm_mhz",
    "clocks_memory_mhz",
)
GPU_QUERY = (
    "index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,"
    "power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_nvidia_smi_csv(payload):
    """Parse nounits CSV output while preserving unavailable values as text."""
    rows = []
    for values in csv.reader(io.StringIO(payload)):
        if not values or all(not value.strip() for value in values):
            continue
        if len(values) != len(GPU_FIELDS):
            raise ValueError(
                "expected %d nvidia-smi fields, got %d"
                % (len(GPU_FIELDS), len(values))
            )
        rows.append(dict(zip(GPU_FIELDS, (value.strip() for value in values))))
    return rows


class PeriodicCollector:
    def __init__(self, name, interval_s, sample):
        self.name = name
        self.interval_s = interval_s
        self.sample = sample
        self.stop_event = threading.Event()
        self.thread = None
        self.status = {
            "enabled": True,
            "available": None,
            "samples": 0,
            "errors": [],
            "stopped": False,
        }

    def start(self):
        self.thread = threading.Thread(
            target=self._run, name="%s-telemetry" % self.name, daemon=True
        )
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                count = self.sample()
                self.status["samples"] += count
                self.status["available"] = True
            except Exception as error:  # Telemetry must never invalidate client evidence.
                self.status["available"] = False
                message = "%s: %s" % (type(error).__name__, error)
                if not self.status["errors"] or self.status["errors"][-1] != message:
                    self.status["errors"].append(message)
            self.stop_event.wait(self.interval_s)

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(self.interval_s + 6)
            if self.thread.is_alive():
                self.status["errors"].append("collector did not stop before timeout")
        self.status["stopped"] = self.thread is None or not self.thread.is_alive()
        if self.status["available"] is None:
            self.status["available"] = False
            self.status["errors"].append("no samples collected")


class TelemetryManager:
    """Own collectors and a shared monotonic epoch for experiment alignment."""

    def __init__(
        self,
        config,
        root,
        clock=time.monotonic,
        timestamp=utc_now,
        command_runner=subprocess.run,
        opener=urlopen,
    ):
        self.config = config
        self.root = Path(root) / "telemetry"
        self.clock = clock
        self.timestamp = timestamp
        self.command_runner = command_runner
        self.opener = opener
        self.epoch = None
        self.collectors = []
        self.locks = {
            "gpu": threading.Lock(),
            "prometheus": threading.Lock(),
            "events": threading.Lock(),
        }
        self.summary = {
            "enabled": False,
            "epoch_utc": None,
            "gpu": {"enabled": False},
            "vllm": {"enabled": False},
        }

    def elapsed(self):
        return self.clock() - self.epoch

    def start(self):
        gpu = self.config.get("gpu", {})
        vllm = self.config.get("vllm", {})
        if not gpu.get("enabled", False) and not vllm.get("enabled", False):
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.epoch = self.clock()
        self.summary.update({"enabled": True, "epoch_utc": self.timestamp()})
        if gpu.get("enabled", False):
            self._write_gpu_header()
            collector = PeriodicCollector(
                "gpu", gpu["interval_s"], lambda: self._sample_gpu(gpu)
            )
            self.collectors.append(("gpu", collector))
        if vllm.get("enabled", False):
            collector = PeriodicCollector(
                "vllm", vllm["interval_s"], lambda: self._sample_prometheus(vllm)
            )
            self.collectors.append(("vllm", collector))
        for _, collector in self.collectors:
            collector.start()
        self.mark("telemetry_started")

    def _write_gpu_header(self):
        with (self.root / "gpu.csv").open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                ("sample_at_utc", "experiment_offset_s") + GPU_FIELDS
            )

    def _sample_gpu(self, config):
        result = self.command_runner(
            [
                config.get("command", "nvidia-smi"),
                "--query-gpu=" + GPU_QUERY,
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=config.get("timeout_s", 5.0),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "nvidia-smi exit code %d: %s"
                % (result.returncode, result.stderr.strip()[:500])
            )
        rows = parse_nvidia_smi_csv(result.stdout)
        sampled_at = self.timestamp()
        offset = "%.6f" % self.elapsed()
        with self.locks["gpu"], (self.root / "gpu.csv").open(
            "a", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            for row in rows:
                writer.writerow(
                    (sampled_at, offset)
                    + tuple(row[field] for field in GPU_FIELDS)
                )
        return len(rows)

    def _sample_prometheus(self, config):
        request = Request(config["url"], headers={"Accept": "text/plain"})
        with self.opener(request, timeout=config.get("timeout_s", 5.0)) as response:
            payload = response.read().decode("utf-8", errors="replace")
            status = getattr(response, "status", 200)
        if status < 200 or status >= 300:
            raise RuntimeError("Prometheus endpoint returned HTTP %d" % status)
        snapshot = {
            "sample_at_utc": self.timestamp(),
            "experiment_offset_s": self.elapsed(),
            "content_type": response.headers.get("Content-Type"),
            "raw": payload,
        }
        with self.locks["prometheus"], (self.root / "vllm.prometheus.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
        return 1

    def mark(self, event, **details):
        if self.epoch is None:
            return
        row = {
            "event": event,
            "at_utc": self.timestamp(),
            "experiment_offset_s": self.elapsed(),
            **details,
        }
        try:
            with self.locks["events"], (self.root / "events.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception as error:
            self.summary.setdefault("event_errors", []).append(
                "%s: %s" % (type(error).__name__, error)
            )

    def stop(self):
        if self.epoch is None:
            return self.summary
        self.mark("telemetry_stopping")
        for name, collector in self.collectors:
            collector.stop()
            self.summary[name] = collector.status
        self.summary["completed_at_utc"] = self.timestamp()
        self.summary["duration_s"] = self.elapsed()
        (self.root / "status.json").write_text(
            json.dumps(self.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return self.summary
