import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_campaign


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == " benchmark":
            return [0x10FFFF]
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        return "".join(chr(token_id) for token_id in token_ids)


def workload_config():
    return {
        "name": "short",
        "model": "test/model",
        "tokenizer": "test/model",
        "seed": 9,
        "request_count": 4,
        "prompt_token_tolerance": 0,
        "temperature": 0.0,
        "buckets": [
            {"name": "short", "count": 4, "prompt_tokens": 128, "output_tokens": 8}
        ],
    }


def campaign_config():
    return {
        "schema_version": "1.0",
        "name": "test-campaign",
        "defaults": {
            "url": "http://localhost:8000/v1/completions",
            "model": "test/model",
            "model_metadata": {
                "revision": "test",
                "dtype": "half",
                "quantization": None,
                "max_model_len": 256,
            },
            "server": {"discovery": "explicit", "launch_flags": []},
            "num_requests": 2,
            "warmups": 1,
            "repeats": 2,
            "seed": 17,
            "stream": True,
        },
        "workloads": [
            {"name": "short-a", "config": "workload.json"},
            {"name": "short-b", "config": "workload.json"},
        ],
        "sweeps": [
            {"name": "first", "workload": "short-a", "concurrency": [1, 2]},
            {
                "name": "second",
                "workload": "short-b",
                "concurrency": [4, 8],
                "repeats": 3,
            },
        ],
    }


def write_inputs(directory):
    root = Path(directory)
    campaign_path = root / "campaign.json"
    workload_path = root / "workload.json"
    campaign_path.write_text(json.dumps(campaign_config()), encoding="utf-8")
    workload_path.write_text(json.dumps(workload_config()), encoding="utf-8")
    return campaign_path


class CampaignTest(unittest.TestCase):
    def test_checked_in_example_and_schemas_are_valid_json(self):
        example_path = ROOT / "campaigns" / "example.json"
        example = run_campaign.load_campaign(example_path)
        with tempfile.TemporaryDirectory() as directory:
            plan = run_campaign.build_plan(
                example,
                example_path,
                output_root=directory,
            )
        self.assertEqual(len(plan["workloads"]), 2)
        self.assertEqual(len(plan["sweeps"]), 2)
        for name in ("campaign.schema.json", "campaign-manifest.schema.json"):
            with (ROOT / "schemas" / name).open(encoding="utf-8") as handle:
                self.assertIsInstance(json.load(handle), dict)

    def test_baseline_collection_has_requested_workloads_and_concurrency(self):
        campaign_path = ROOT / "campaigns" / "baseline-capacity-20260906.json"
        campaign = run_campaign.load_campaign(campaign_path)
        with tempfile.TemporaryDirectory() as directory:
            plan = run_campaign.build_plan(
                campaign,
                campaign_path,
                output_root=directory,
            )
        expected_names = ["mixed", "long-prefill", "long-decode", "short"]
        expected_concurrency = [16, 24, 32, 64, 96, 128, 256]
        self.assertEqual(
            [item["name"] for item in plan["workloads"]], expected_names
        )
        self.assertEqual([item["name"] for item in plan["sweeps"]], expected_names)
        for workload in plan["workloads"]:
            self.assertEqual(workload["resolved_config"]["request_count"], 256)
        for sweep in plan["sweeps"]:
            self.assertEqual(sweep["experiment"]["num_requests"], 256)
            self.assertEqual(sweep["experiment"]["concurrency"], expected_concurrency)

        mixed = plan["workloads"][0]["resolved_config"]
        self.assertEqual(
            [bucket["count"] for bucket in mixed["buckets"]], [128, 64, 64]
        )

    def test_plan_resolves_matrix_and_global_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign_path = write_inputs(directory)
            plan = run_campaign.build_plan(
                campaign_config(),
                campaign_path,
                output_root=Path(directory) / "results",
                concurrency=[3, 6],
                num_requests=4,
                warmups=0,
                repeats=5,
                seed=23,
            )
            self.assertEqual(
                [item["name"] for item in plan["sweeps"]], ["first", "second"]
            )
            for item in plan["sweeps"]:
                experiment = item["experiment"]
                self.assertEqual(experiment["concurrency"], [3, 6])
                self.assertEqual(experiment["num_requests"], 4)
                self.assertEqual(experiment["warmups"], 0)
                self.assertEqual(experiment["repeats"], 5)
                self.assertEqual(experiment["seed"], 23)
                self.assertTrue(Path(experiment["prompts"]).is_absolute())

    def test_dry_run_has_no_filesystem_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign_path = write_inputs(directory)
            output_root = Path(directory) / "results"
            result = run_campaign.run_campaign(
                campaign_config(),
                campaign_path,
                output_root=output_root,
                dry_run=True,
            )
            self.assertEqual(result["selected_sweeps"], ["first", "second"])
            self.assertFalse(output_root.exists())

    def test_executes_declared_order_and_records_completed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign_path = write_inputs(directory)
            calls = []

            def fake_runner(config, output_root, original_config, repo_root):
                calls.append((config, Path(output_root)))

            result = run_campaign.run_campaign(
                campaign_config(),
                campaign_path,
                output_root=Path(directory) / "results",
                allow_dirty=True,
                experiment_runner=fake_runner,
                tokenizer=CharacterTokenizer(),
            )
            self.assertEqual([item[0]["name"] for item in calls], ["first", "second"])
            self.assertEqual(result["manifest"]["campaign"]["status"], "completed")
            self.assertTrue(
                all(
                    item["status"] == "completed"
                    for item in result["manifest"]["sweeps"]
                )
            )
            root = Path(result["plan"]["campaign_root"])
            self.assertTrue((root / "campaign.original.json").is_file())
            self.assertTrue((root / "campaign.resolved.json").is_file())
            self.assertTrue((root / "campaign-manifest.json").is_file())
            self.assertTrue(
                (root / "workloads" / "short-a" / "short.jsonl").is_file()
            )

    def test_filters_execution_without_changing_full_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign_path = write_inputs(directory)
            calls = []

            def fake_runner(config, output_root, original_config, repo_root):
                calls.append(config["name"])

            result = run_campaign.run_campaign(
                campaign_config(),
                campaign_path,
                output_root=Path(directory) / "results",
                sweep_names=["second"],
                allow_dirty=True,
                experiment_runner=fake_runner,
                tokenizer=CharacterTokenizer(),
            )
            self.assertEqual(calls, ["second"])
            self.assertEqual(
                [item["name"] for item in result["plan"]["sweeps"]],
                ["first", "second"],
            )
            statuses = {
                item["name"]: item["status"]
                for item in result["manifest"]["sweeps"]
            }
            self.assertEqual(statuses, {"first": "not_run", "second": "completed"})
            self.assertEqual(result["manifest"]["campaign"]["status"], "partial")

    def test_failed_sweep_is_recorded_and_stops_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign_path = write_inputs(directory)

            def failing_runner(config, output_root, original_config, repo_root):
                raise RuntimeError("benchmark failed")

            with self.assertRaisesRegex(RuntimeError, "benchmark failed"):
                run_campaign.run_campaign(
                    campaign_config(),
                    campaign_path,
                    output_root=Path(directory) / "results",
                    allow_dirty=True,
                    experiment_runner=failing_runner,
                    tokenizer=CharacterTokenizer(),
                )
            manifest_path = (
                Path(directory)
                / "results"
                / "test-campaign"
                / "campaign-manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["campaign"]["status"], "failed")
            self.assertEqual(manifest["sweeps"][0]["status"], "failed")
            self.assertEqual(manifest["sweeps"][1]["status"], "not_run")

    def test_rejects_unknown_workload_and_oversized_sweep(self):
        invalid = campaign_config()
        invalid["sweeps"][0]["workload"] = "missing"
        with self.assertRaisesRegex(ValueError, "unknown workload"):
            run_campaign.validate_campaign(invalid)

        with tempfile.TemporaryDirectory() as directory:
            campaign_path = write_inputs(directory)
            with self.assertRaisesRegex(ValueError, "contains 4"):
                run_campaign.build_plan(
                    campaign_config(),
                    campaign_path,
                    output_root=Path(directory) / "results",
                    num_requests=5,
                )

    def test_refuses_dirty_official_run(self):
        with self.assertRaisesRegex(RuntimeError, "dirty checkout"):
            run_campaign.ensure_clean(
                {"available": True, "revision": "abc", "dirty": True}
            )


if __name__ == "__main__":
    unittest.main()
