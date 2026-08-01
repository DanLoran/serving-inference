import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import experiment_manifest
import run_experiment


def sample_config(prompts):
    return {
        "name": "manifest-test",
        "prompts": str(prompts),
        "url": "https://user:pass@example.test/v1/completions?api_key=hidden&region=us",
        "model": "example/model",
        "model_metadata": {
            "revision": "abc123", "dtype": "half", "quantization": None,
            "max_model_len": 2048,
        },
        "server": {
            "discovery": "explicit",
            "launch_flags": ["--dtype", "half", "--max-model-len", "2048"],
        },
        "num_requests": 1, "concurrency": [1], "warmups": 0, "repeats": 1,
        "seed": 1, "authorization": "Bearer do-not-store",
    }


class ManifestTest(unittest.TestCase):
    def test_manifest_schema_and_workload_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.jsonl"
            prompts.write_bytes(b'{"prompt":"hello"}\n')
            config = sample_config(prompts)
            with mock.patch.object(
                experiment_manifest, "collect_accelerator",
                return_value={"nvidia_smi_available": False, "gpus": [], "driver_version": None, "cuda_version": None, "reason": "FileNotFoundError"},
            ):
                manifest = experiment_manifest.build_manifest(
                    config, config, root, "2026-01-01T00:00:00Z"
                )
            experiment_manifest.validate_manifest(manifest)
            schema = json.loads(
                (Path(__file__).resolve().parents[1] / "schemas" / "experiment-manifest.schema.json").read_text(encoding="utf-8")
            )
            self.assertEqual(schema["properties"]["schema_version"]["const"], manifest["schema_version"])
            self.assertEqual(set(schema["required"]), set(manifest))
            self.assertEqual(manifest["schema_version"], "1.0")
            self.assertEqual(
                manifest["workload"]["sha256"],
                experiment_manifest.sha256_file(prompts),
            )
            self.assertFalse(manifest["environment"]["accelerator"]["nvidia_smi_available"])

    def test_sanitizer_redacts_secrets_and_url_credentials(self):
        value = experiment_manifest.sanitize({
            "api_token": "raw-token",
            "nested": {"password": "raw-password"},
            "url": "https://alice:secret@example.test/path?token=query-secret&safe=yes",
            "launch_flags": ["--api-key", "flag-secret", "--password=inline-secret", "--safe", "yes"],
            "ordinary": "keep-me",
        })
        encoded = json.dumps(value)
        for secret in ("raw-token", "raw-password", "alice", "secret", "query-secret", "flag-secret", "inline-secret"):
            self.assertNotIn(secret, encoded)
        self.assertIn("keep-me", encoded)
        self.assertIn("safe=yes", value["url"])

    def test_accelerator_collection_is_graceful_without_nvidia_tools(self):
        with mock.patch.object(
            experiment_manifest, "run_command",
            return_value={"available": False, "reason": "FileNotFoundError"},
        ):
            result = experiment_manifest.collect_accelerator()
        self.assertEqual(result["gpus"], [])
        self.assertIsNone(result["driver_version"])
        self.assertIsNone(result["cuda_version"])

    def test_runner_writes_original_resolved_and_completed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.jsonl"
            prompts.write_text('{"prompt":"hello"}\n', encoding="utf-8")
            original = sample_config(prompts)
            resolved = {
                **original, "temperature": 0.0, "timeout": 30, "stream": True,
                "store_response": False, "output_dir": str(root / "results"),
            }
            commands = []
            def runner(command):
                commands.append(command)
                return SimpleNamespace(returncode=0)
            with mock.patch.object(experiment_manifest, "collect_accelerator", return_value={
                "nvidia_smi_available": False, "gpus": [], "driver_version": None,
                "cuda_version": None, "reason": "FileNotFoundError",
            }), mock.patch.object(run_experiment, "build_manifest", wraps=experiment_manifest.build_manifest):
                manifest = run_experiment.run_experiment(
                    original, resolved, runner=runner, repo_root=root
                )
            output = root / "results" / original["name"]
            self.assertEqual(len(commands), 1)
            self.assertTrue((output / "config.original.json").exists())
            self.assertTrue((output / "config.resolved.json").exists())
            self.assertEqual(manifest["experiment"]["status"], "completed")
            self.assertIsNotNone(manifest["experiment"]["completed_at_utc"])
            saved = (output / "manifest.json").read_text(encoding="utf-8")
            self.assertNotIn("do-not-store", saved)
            self.assertNotIn("pass", saved)


if __name__ == "__main__":
    unittest.main()
