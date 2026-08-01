import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_experiment


def config():
    return {
        "name": "test-sweep",
        "prompts": "prompts/mixed.jsonl",
        "url": "http://localhost:8000/v1/completions",
        "model": "mock",
        "num_requests": 2,
        "concurrency": [1, 2, 4],
        "warmups": 1,
        "repeats": 2,
        "seed": 17,
        "stream": True,
    }


def write_success(command):
    def option(name):
        return command[command.index(name) + 1]

    raw_path = Path(option("--output"))
    summary_path = Path(option("--summary"))
    concurrency = int(option("--concurrency"))
    rows = [
        {
            "status": 200,
            "latency_s": concurrency + index,
            "ttft_s": 0.1,
            "approx_time_per_output_token_s": 0.01,
            "output_tokens": 3,
        }
        for index in range(2)
    ]
    raw_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    summary_path.write_text(
        json.dumps(
            {
                "config": {"concurrency": concurrency},
                "counts": {"attempted": 2, "successful": 2, "failed": 0},
                "duration_s": 2.0,
                "request_throughput_per_s": 1.0,
                "output_token_throughput_per_s": concurrency * 10.0,
                "latency_s": {"p99": concurrency * 0.5},
            }
        ),
        encoding="utf-8",
    )


class ExperimentTest(unittest.TestCase):
    def test_order_is_deterministic_and_interleaved(self):
        first = run_experiment.sweep_order(config())
        self.assertEqual(first, run_experiment.sweep_order(config()))
        self.assertNotEqual(first, run_experiment.sweep_order({**config(), "seed": 18}))
        phases = [(run["kind"], run["index"]) for run in first]
        self.assertEqual(phases[:3], [("warmup", 1)] * 3)
        self.assertEqual(phases[3:6], [("repeat", 1)] * 3)
        self.assertEqual(phases[6:], [("repeat", 2)] * 3)
        for offset in (0, 3, 6):
            self.assertEqual(
                {run["concurrency"] for run in first[offset : offset + 3]},
                {1, 2, 4},
            )

    def test_warmups_are_excluded_and_repeats_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def runner(command):
                calls.append(command)
                write_success(command)
                return SimpleNamespace(returncode=0)

            result = run_experiment.run_experiment(config(), directory, runner)
            self.assertEqual(len(calls), 9)
            self.assertEqual(len(result["sweep_order"]), 9)
            self.assertEqual(len(result["measured_runs"]), 6)
            self.assertTrue(
                all(run["kind"] == "repeat" for run in result["measured_runs"])
            )
            self.assertTrue(
                all(item["repeats"] == 2 for item in result["by_concurrency"])
            )
            self.assertTrue(
                all(
                    item["counts"]["attempted"] == 4
                    for item in result["by_concurrency"]
                )
            )
            self.assertEqual(
                len(list((Path(directory) / config()["name"]).rglob("*.jsonl"))), 9
            )

    def test_failure_propagates_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def runner(command):
                calls.append(command)
                return SimpleNamespace(returncode=7)

            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                run_experiment.run_experiment(config(), directory, runner)
            self.assertEqual(len(calls), 1)

    def test_success_exit_with_failed_requests_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            def runner(command):
                write_success(command)
                summary_path = Path(command[command.index("--summary") + 1])
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["counts"] = {"attempted": 2, "successful": 1, "failed": 1}
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                return SimpleNamespace(returncode=0)

            with self.assertRaisesRegex(RuntimeError, "incomplete or failed"):
                run_experiment.run_experiment(config(), directory, runner)

    def test_resume_keeps_completed_raw_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            def first_runner(command):
                write_success(command)
                return SimpleNamespace(returncode=0)

            run_experiment.run_experiment(config(), directory, first_runner)
            raw_paths = sorted(Path(directory).rglob("*.jsonl"))
            contents = {path: path.read_bytes() for path in raw_paths}
            second_calls = []
            run_experiment.run_experiment(
                config(),
                directory,
                lambda command: second_calls.append(command)
                or SimpleNamespace(returncode=0),
            )
            self.assertEqual(second_calls, [])
            self.assertEqual(contents, {path: path.read_bytes() for path in raw_paths})

    def test_partial_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / config()["name"]
            run_experiment.prepare_root(root, config())
            run = run_experiment.sweep_order(config())[0]
            raw_path, _ = run_experiment.run_paths(root, run)
            raw_path.parent.mkdir(parents=True)
            raw_path.write_text("existing\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite partial run"):
                run_experiment.run_experiment(config(), directory)
            self.assertEqual(raw_path.read_text(encoding="utf-8"), "existing\n")

    def test_changed_config_cannot_reuse_completed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            first = config()
            root = Path(directory) / first["name"]
            run_experiment.prepare_root(root, first)
            with self.assertRaisesRegex(RuntimeError, "config differs"):
                run_experiment.run_experiment(
                    {**first, "model": "different"}, directory
                )

    def test_saturation_requires_two_successive_transitions(self):
        points = []
        for concurrency, throughput, p99 in (
            (1, 100.0, 1.0),
            (2, 103.0, 1.25),
            (4, 104.0, 1.55),
        ):
            points.append(
                {
                    "concurrency": concurrency,
                    "repeat_medians": {
                        "output_token_throughput_per_s": throughput,
                        "latency_p99_s": p99,
                    },
                }
            )
        result = run_experiment.saturation_analysis(points)
        self.assertEqual(result["candidate_saturation_concurrency"], 2)

    def test_load_config_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({**config(), "repeats": 0}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "repeats"):
                run_experiment.load_config(path)


if __name__ == "__main__":
    unittest.main()
