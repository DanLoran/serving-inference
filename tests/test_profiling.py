import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import profile_capture


EXAMPLE = json.loads((ROOT / "profiling" / "example.json").read_text())


class ProfilingTest(unittest.TestCase):
    def build(self, tool, config=None):
        return profile_capture.build_plan(
            tool,
            copy.deepcopy(config or EXAMPLE),
            ROOT / "profiling" / "example.json",
            ROOT,
        )

    def test_profile_name_ties_experiment_workload_concurrency_and_tool(self):
        plan = self.build("nsys")
        self.assertEqual(
            plan.profile_name,
            "rtx-2060-super-example--short--c1--nsys",
        )
        self.assertEqual(plan.profile_dir.parent, ROOT / "results" / "profiles")

    def test_nsys_wraps_server_with_required_bounded_traces(self):
        plan = self.build("nsys")
        command = list(plan.profiler_command)
        self.assertEqual(command[:2], ["nsys", "profile"])
        self.assertIn("--trace=cuda,nvtx,osrt", command)
        self.assertIn("--duration=120", command)
        self.assertIn("--kill=sigterm", command)
        self.assertEqual(command[-len(plan.server_command) :], list(plan.server_command))
        self.assertNotIn("send_requests.py", " ".join(command))
        self.assertIn("send_requests.py", " ".join(plan.workload_command))

    def test_nsys_rejects_missing_required_runtime_trace(self):
        config = copy.deepcopy(EXAMPLE)
        config["nsys"]["trace"] = ["cuda", "nvtx"]
        with self.assertRaisesRegex(profile_capture.ConfigError, "must include"):
            self.build("nsys", config)

    def test_ncu_targets_children_and_limits_selected_kernel_replay(self):
        plan = self.build("ncu")
        command = list(plan.profiler_command)
        self.assertEqual(command[0], "ncu")
        self.assertIn("--target-processes=all", command)
        self.assertIn("--set=basic", command)
        self.assertIn("--replay-mode=kernel", command)
        self.assertIn("--launch-count=1", command)
        self.assertTrue(
            any(argument.startswith("--kernel-name=regex:") for argument in command)
        )
        self.assertEqual(command[-len(plan.server_command) :], list(plan.server_command))

    def test_rejects_unsafe_names_and_invalid_limits(self):
        config = copy.deepcopy(EXAMPLE)
        config["experiment"] = "../overwrite"
        with self.assertRaisesRegex(profile_capture.ConfigError, "only letters"):
            self.build("nsys", config)

        config = copy.deepcopy(EXAMPLE)
        config["ncu"]["launch_count"] = 0
        with self.assertRaisesRegex(profile_capture.ConfigError, "positive integer"):
            self.build("ncu", config)

    def test_model_readiness_requires_expected_model(self):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"data": [{"id": "other-model"}]}'

        original = profile_capture.urllib.request.urlopen
        profile_capture.urllib.request.urlopen = lambda *args, **kwargs: Response()
        try:
            ready, error = profile_capture.model_is_ready(
                "http://127.0.0.1:8000/v1/models", "expected-model"
            )
        finally:
            profile_capture.urllib.request.urlopen = original
        self.assertFalse(ready)
        self.assertIn("other-model", error)

    def test_existing_capture_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(EXAMPLE)
            config["output_dir"] = directory
            plan = self.build("nsys", config)
            plan.profile_dir.mkdir()
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                profile_capture.run_capture(plan, "sha256", ROOT)

    def test_compute_counter_permission_failure_is_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(EXAMPLE)
            config["output_dir"] = directory
            plan = self.build("ncu", config)
            plan.profile_dir.mkdir()
            (plan.profile_dir / "server.log").write_text(
                "==ERROR== ERR_NVGPUCTRPERM - counters unavailable\n",
                encoding="utf-8",
            )
            reason = profile_capture.inferred_failure_reason(
                plan, {"workload_exit_code": 0}
            )
            self.assertIn("ERR_NVGPUCTRPERM", reason)

    def test_shell_wrappers_have_valid_syntax(self):
        subprocess.run(
            [
                "bash",
                "-n",
                str(ROOT / "scripts" / "profile_nsys.sh"),
                str(ROOT / "scripts" / "profile_ncu.sh"),
            ],
            check=True,
        )

    def test_dry_run_does_not_require_profiler_or_create_artifacts(self):
        for tool in ("nsys", "ncu"):
            wrapper = ROOT / "scripts" / ("profile_%s.sh" % tool)
            with tempfile.TemporaryDirectory() as directory:
                config = copy.deepcopy(EXAMPLE)
                config["experiment"] = "dry-run"
                config["output_dir"] = str(Path(directory) / "profiles")
                config_path = Path(directory) / "config.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                completed = subprocess.run(
                    [str(wrapper), "--config", str(config_path), "--dry-run"],
                    cwd=ROOT,
                    env={**os.environ, "PYTHON": sys.executable},
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("Profiler/server command:", completed.stdout)
                self.assertIn("Workload/client command:", completed.stdout)
                self.assertIn("no directories", completed.stdout)
                self.assertFalse((Path(directory) / "profiles").exists())


if __name__ == "__main__":
    unittest.main()
