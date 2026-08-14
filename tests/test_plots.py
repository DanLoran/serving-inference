import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import plot_results


def write_manifest(root, schema="1.0"):
    manifest = {
        "schema_version": schema,
        "experiment": {
            "name": "plot-golden",
            "status": "completed",
            "repeats": 2,
        },
        "config": {
            "resolved": {
                "concurrency": [1, 2],
                "repeats": 2,
                "num_requests": 4,
            }
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def result(index, workload, latency, missing_stream_metrics=False):
    return {
        "schema_version": "1.0",
        "request_index": index,
        "workload": workload,
        "status": 200,
        "ended_offset_s": latency,
        "latency_s": latency,
        "ttft_s": None if missing_stream_metrics else latency / 10,
        "approx_time_per_output_token_s": (
            None if missing_stream_metrics else latency / 100
        ),
        "prompt_tokens": 10,
        "output_tokens": 5,
    }


def write_run(root, repeat, concurrency):
    missing = concurrency == 2
    rows = [
        result(0, "short", concurrency + repeat / 10),
        result(1, "short", concurrency + repeat / 5),
        result(2, "long_prefill", concurrency * 2 + repeat / 10, missing),
        result(3, "long_prefill", concurrency * 2 + repeat / 5, missing),
    ]
    path = (
        root
        / ("repeat-%02d" % repeat)
        / ("concurrency-%03d.jsonl" % concurrency)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def write_gpu(root):
    path = root / "telemetry" / "gpu.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_at_utc",
                "experiment_offset_s",
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
            ]
        )
        writer.writerow(
            [
                "2026-08-14T00:00:00Z", "0.0", "0", "GPU-a", "10", "2",
                "1024", "8192", "50", "40", "1200", "6000",
            ]
        )
        writer.writerow(
            [
                "2026-08-14T00:00:01Z", "1.0", "0", "GPU-a", "[N/A]", "4",
                "2048", "8192", "70", "45", "1400", "6000",
            ]
        )


class PlotResultsTest(unittest.TestCase):
    def create_experiment(self, root, schema="1.0", gpu=True):
        write_manifest(root, schema)
        for repeat in (1, 2):
            for concurrency in (1, 2):
                write_run(root, repeat, concurrency)
        if gpu:
            write_gpu(root)

    def test_headless_rendering_writes_nonempty_workload_and_gpu_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_experiment(root)

            formats = ("png", "pdf", "svg")
            outputs = plot_results.render_experiment(root, formats=formats)

            stems = {
                "throughput-short",
                "latency-short",
                "throughput-long-prefill",
                "latency-long-prefill",
                "gpu-telemetry",
            }
            expected = {
                "%s.%s" % (stem, output_format)
                for stem in stems
                for output_format in formats
            } | {"plot-manifest.json"}
            self.assertEqual({path.name for path in outputs}, expected)
            for path in outputs:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
            first_bytes = {path.name: path.read_bytes() for path in outputs}
            rebuilt = plot_results.render_experiment(root, formats=formats)
            self.assertEqual(
                first_bytes,
                {path.name: path.read_bytes() for path in rebuilt},
            )
            manifest = json.loads((root / "figures" / "plot-manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], "1.0")
            self.assertEqual(manifest["experiment"], "plot-golden")
            self.assertEqual(len(manifest["figures"]), 15)
            self.assertEqual(len(manifest["sources"]), 6)

    def test_missing_metrics_remain_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_experiment(root, gpu=False)
            rows = plot_results.load_analysis_rows(root)
            groups, _, diagnostics = plot_results._repeat_groups(rows)

            mean, low, high, values = plot_results._series_statistics(
                groups[("long_prefill", 2)], "ttft_s_p99"
            )

            self.assertIsNone(mean)
            self.assertIsNone(low)
            self.assertIsNone(high)
            self.assertEqual(values, [])
            self.assertTrue(any("missing_ttft_s" in item for item in diagnostics))

    def test_incompatible_result_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_experiment(root, gpu=False)
            raw_path = root / "repeat-01" / "concurrency-001.jsonl"
            rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
            rows[0]["schema_version"] = "2.0"
            raw_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            with self.assertRaisesRegex(
                plot_results.PlotDataError, "schema-incompatible"
            ):
                plot_results.load_analysis_rows(root)

    def test_gpu_schema_and_values_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.csv"
            path.write_text("experiment_offset_s,index\n0,0\n", encoding="utf-8")
            with self.assertRaisesRegex(plot_results.PlotDataError, "missing fields"):
                plot_results.load_gpu_samples(path)

            write_gpu(Path(directory))
            valid = Path(directory) / "telemetry" / "gpu.csv"
            rows = plot_results.load_gpu_samples(valid)
            self.assertEqual(rows[0]["utilization_percent"], 10.0)
            self.assertIsNone(rows[1]["utilization_percent"])


if __name__ == "__main__":
    unittest.main()
