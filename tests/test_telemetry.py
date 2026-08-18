import csv
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import telemetry


FIXTURES = Path(__file__).with_name("fixtures")


class Response:
    status = 200
    headers = {"Content-Type": "text/plain; version=0.0.4"}

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class TelemetryTest(unittest.TestCase):
    def test_parses_captured_nvidia_smi_rows(self):
        payload = (FIXTURES / "nvidia-smi.csv").read_text(encoding="utf-8")
        rows = telemetry.parse_nvidia_smi_csv(payload)
        self.assertEqual(rows[0]["utilization_gpu_percent"], "87")
        self.assertEqual(rows[0]["memory_used_mib"], "6144")
        self.assertEqual(rows[1]["power_draw_w"], "[N/A]")

    def test_collectors_preserve_raw_samples_and_shared_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            gpu_payload = (FIXTURES / "nvidia-smi.csv").read_text(encoding="utf-8")
            prometheus = (FIXTURES / "vllm-metrics.prom").read_bytes()

            def command_runner(*args, **kwargs):
                return subprocess.CompletedProcess(args[0], 0, gpu_payload, "")

            manager = telemetry.TelemetryManager(
                {
                    "gpu": {"enabled": True, "interval_s": 0.01, "timeout_s": 1},
                    "vllm": {
                        "enabled": True,
                        "url": "http://localhost:8000/metrics",
                        "interval_s": 0.01,
                        "timeout_s": 1,
                    },
                },
                directory,
                command_runner=command_runner,
                opener=lambda *args, **kwargs: Response(prometheus),
            )
            manager.start()
            manager.mark("benchmark_started", kind="repeat", index=1, concurrency=4)
            time.sleep(0.03)
            status = manager.stop()

            root = Path(directory) / "telemetry"
            with (root / "gpu.csv").open(encoding="utf-8") as handle:
                gpu_rows = list(csv.DictReader(handle))
            snapshots = [
                json.loads(line)
                for line in (root / "vllm.prometheus.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            events = [
                json.loads(line)
                for line in (root / "events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertGreaterEqual(len(gpu_rows), 2)
            self.assertGreaterEqual(len(snapshots), 1)
            self.assertIn("vllm:num_requests_waiting", snapshots[0]["raw"])
            self.assertTrue(
                all(float(row["experiment_offset_s"]) >= 0 for row in gpu_rows)
            )
            self.assertTrue(all(row["experiment_offset_s"] >= 0 for row in snapshots))
            self.assertEqual(events[1]["event"], "benchmark_started")
            self.assertTrue(status["gpu"]["stopped"])
            self.assertTrue(status["vllm"]["stopped"])

    def test_unavailable_gpu_is_recorded_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            def unavailable(*args, **kwargs):
                raise FileNotFoundError("nvidia-smi")

            manager = telemetry.TelemetryManager(
                {"gpu": {"enabled": True, "interval_s": 0.01, "timeout_s": 1}},
                directory,
                command_runner=unavailable,
            )
            manager.start()
            time.sleep(0.02)
            status = manager.stop()
            self.assertFalse(status["gpu"]["available"])
            self.assertTrue(status["gpu"]["stopped"])
            self.assertIn("FileNotFoundError", status["gpu"]["errors"][0])


if __name__ == "__main__":
    unittest.main()
