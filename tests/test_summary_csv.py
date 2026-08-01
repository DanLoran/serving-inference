import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import summarize_results


def write_manifest(root, repeats=2, concurrencies=(1,), requests=2, status="completed"):
    manifest = {
        "schema_version": "1.0",
        "experiment": {
            "name": "golden",
            "status": status,
            "repeats": repeats,
        },
        "config": {
            "resolved": {
                "concurrency": list(concurrencies),
                "repeats": repeats,
                "num_requests": requests,
            }
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_run(root, repeat, concurrency, rows):
    path = (
        root
        / ("repeat-%02d" % repeat)
        / ("concurrency-%03d.jsonl" % concurrency)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def result(index, latency, workload="short", status=200, schema="1.0", **overrides):
    row = {
        "schema_version": schema,
        "request_index": index,
        "workload": workload,
        "status": status,
        "ended_offset_s": latency,
        "latency_s": latency,
        "ttft_s": latency / 10,
        "approx_time_per_output_token_s": latency / 100,
        "prompt_tokens": 10,
        "output_tokens": 5,
    }
    row.update(overrides)
    return row


class SummaryCsvTest(unittest.TestCase):
    def read_csv(self, path):
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_golden_raw_jsonl_to_csv_and_deterministic_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(root)
            write_run(root, 1, 1, [result(0, 1.0), result(1, 3.0)])
            write_run(root, 2, 1, [result(0, 2.0), result(1, 4.0)])

            output = summarize_results.write_csv(root)
            first = output.read_bytes()
            rows = self.read_csv(output)
            summarize_results.write_csv(root)

            self.assertEqual(first, output.read_bytes())
            self.assertEqual(
                [row["row_type"] for row in rows],
                ["repeat", "repeat", "aggregate"],
            )
            aggregate = rows[-1]
            self.assertEqual(aggregate["workload"], "short")
            self.assertEqual(aggregate["repeat_count"], "2")
            self.assertEqual(aggregate["attempted_requests"], "4")
            self.assertEqual(aggregate["prompt_tokens"], "40")
            self.assertEqual(aggregate["output_tokens"], "20")
            self.assertEqual(aggregate["e2e_latency_s_p50"], "2.5")
            self.assertEqual(aggregate["e2e_latency_s_p90"], "3.7")
            self.assertEqual(aggregate["complete"], "True")
            self.assertNotEqual(aggregate["request_goodput_per_s_stddev"], "")
            self.assertNotEqual(aggregate["request_goodput_per_s_ci95_low"], "")

    def test_failures_and_missing_metrics_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(root, repeats=1)
            write_run(
                root,
                1,
                1,
                [
                    result(
                        0,
                        1.0,
                        output_tokens=None,
                        ttft_s=None,
                        approx_time_per_output_token_s=None,
                    ),
                    result(
                        1,
                        2.0,
                        status=500,
                        output_tokens=None,
                        ttft_s=None,
                        approx_time_per_output_token_s=None,
                    ),
                ],
            )
            rows = summarize_results.build_rows(root)
            repeat = rows[0]
            self.assertEqual(repeat["failed_requests"], 1)
            self.assertEqual(repeat["failure_rate"], 0.5)
            self.assertEqual(repeat["ttft_s_count"], 0)
            self.assertIn("missing_output_tokens_1", repeat["issues"])
            self.assertIn("missing_ttft_s_1", repeat["issues"])
            self.assertIsNone(rows[-1]["failure_rate_stddev"])

    def test_incompatible_schema_and_incomplete_run_are_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(root, repeats=2)
            write_run(root, 1, 1, [result(0, 1.0, schema="2.0")])

            rows = summarize_results.build_rows(root)
            repeats = [row for row in rows if row["row_type"] == "repeat"]
            self.assertFalse(repeats[0]["schema_compatible"])
            self.assertFalse(repeats[0]["complete"])
            self.assertIn("incompatible_result_schema", repeats[0]["issues"])
            self.assertFalse(repeats[1]["complete"])
            self.assertIn("missing_raw_run", repeats[1]["issues"])
            self.assertFalse(rows[-1]["complete"])

    def test_pooled_percentile_is_recomputed_from_raw_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(root)
            write_run(root, 1, 1, [result(0, 1.0), result(1, 100.0)])
            write_run(root, 2, 1, [result(0, 2.0), result(1, 3.0)])

            aggregate = summarize_results.build_rows(root)[-1]
            self.assertEqual(aggregate["e2e_latency_s_p50"], 2.5)
            self.assertAlmostEqual(aggregate["e2e_latency_s_p90"], 70.9)

    def test_mixed_workloads_get_separate_rows_with_shared_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(root, repeats=1)
            write_run(
                root, 1, 1,
                [result(0, 1.0, workload="short"), result(1, 2.0, workload="long")],
            )
            repeats = [
                row
                for row in summarize_results.build_rows(root)
                if row["row_type"] == "repeat"
            ]
            self.assertEqual([row["workload"] for row in repeats], ["long", "short"])
            self.assertEqual([row["duration_s"] for row in repeats], [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
