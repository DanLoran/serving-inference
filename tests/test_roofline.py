import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_roofline


def result(index, workload, prompt_tokens, output_tokens, duration):
    return {
        "schema_version": "1.0",
        "request_index": index,
        "workload": workload,
        "status": 200,
        "ended_offset_s": duration,
        "latency_s": duration,
        "ttft_s": duration / 10,
        "approx_time_per_output_token_s": duration / 100,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }


def write_experiment(root, workload, prompt_tokens, output_tokens):
    root.mkdir(parents=True)
    manifest = {
        "schema_version": "1.0",
        "experiment": {"name": "golden", "status": "completed", "repeats": 2},
        "config": {
            "resolved": {"concurrency": [1], "repeats": 2, "num_requests": 2}
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for repeat in (1, 2):
        rows = [
            result(0, workload, prompt_tokens, output_tokens, 1.0),
            result(1, workload, prompt_tokens, output_tokens, 2.0),
        ]
        path = root / ("repeat-%02d" % repeat) / "concurrency-001.jsonl"
        path.parent.mkdir()
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def write_profile(root, workload):
    root.mkdir(parents=True)
    metadata = {
        "schema_version": "1.0",
        "status": "complete",
        "tool": "ncu",
        "tool_version": "NVIDIA Nsight Compute 2026.1",
        "profile_name": "golden--%s--c1--ncu" % workload,
        "workload": workload,
        "concurrency": 1,
    }
    metadata_path = root / "metadata.json"
    report_path = root / "capture.ncu-rep"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    report_path.write_bytes(b"representative native ncu report fixture\n")
    return metadata_path, report_path


def device():
    source = {"url": "https://example.test/spec", "accessed_at": "2026-08-31"}
    return {
        "name": "Test GPU",
        "memory_bandwidth": {
            "value": 100.0,
            "unit": "GB/s",
            "derivation": "100e9 byte/s",
            "source": source,
        },
        "compute_ceilings": [
            {
                "id": "tensor",
                "label": "Dense tensor peak",
                "value": 10.0,
                "unit": "TFLOP/s",
                "dtype": "FP16",
                "execution": "dense tensor operations",
                "derivation": "test fixture ceiling",
                "source": source,
            }
        ],
    }


def counter_case(experiment, metadata, report):
    return {
        "id": "prefill-c1",
        "phase": "prefill",
        "workload": "long_prefill",
        "concurrency": 1,
        "unprofiled_experiment": str(experiment),
        "rate_metric": "prompt_token_goodput_per_s",
        "compute_ceiling": "tensor",
        "measurement": {
            "kind": "counter",
            "status": "available",
            "scope": "two representative selected prefill kernels",
            "profile_metadata": str(metadata),
            "profile_report": str(report),
            "phase_tokens": 10,
            "dram_counters": [
                {
                    "name": "dram__bytes_read.sum",
                    "value": 600,
                    "unit": "byte",
                    "bytes_per_event": 1,
                },
                {
                    "name": "dram__bytes_write.sum",
                    "value": 400,
                    "unit": "byte",
                    "bytes_per_event": 1,
                },
            ],
            "flop_counters": [
                {
                    "name": "sm__representative_fma_events.sum",
                    "value": 1000,
                    "unit": "event",
                    "flops_per_event": 2,
                    "execution": "FP16 fused multiply-add",
                }
            ],
        },
    }


def estimate_case(experiment):
    return {
        "id": "decode-c1",
        "phase": "decode",
        "workload": "decode_heavy",
        "concurrency": 1,
        "unprofiled_experiment": str(experiment),
        "rate_metric": "output_token_goodput_per_s",
        "compute_ceiling": "tensor",
        "measurement": {
            "kind": "estimate",
            "status": "available",
            "scope": "analytical per-output-token estimate",
            "flops_per_token": 500,
            "dram_bytes_per_token": 250,
            "rationale": "known-value test estimate",
            "source": {
                "url": "https://example.test/model",
                "accessed_at": "2026-08-31",
            },
        },
    }


class RooflineTest(unittest.TestCase):
    def fixture(self, directory, include_estimate=True):
        root = Path(directory)
        prefill = root / "unprofiled-prefill"
        decode = root / "unprofiled-decode"
        profile = root / "profile"
        write_experiment(prefill, "long_prefill", 100, 10)
        write_experiment(decode, "decode_heavy", 20, 10)
        metadata, report = write_profile(profile, "long_prefill")
        cases = [counter_case(prefill, metadata, report)]
        if include_estimate:
            cases.append(estimate_case(decode))
        return {
            "schema_version": "1.0",
            "analysis": "golden-roofline",
            "device": device(),
            "cases": cases,
        }

    def build(self, config):
        encoded = (json.dumps(config, sort_keys=True) + "\n").encode("utf-8")
        return analyze_roofline.build_analysis(
            config,
            Path("analysis/golden.json"),
            hashlib.sha256(encoded).hexdigest(),
            ROOT,
        )

    def test_known_value_formulas_use_si_units(self):
        self.assertEqual(analyze_roofline.arithmetic_intensity(2000, 1000), 2.0)
        self.assertEqual(analyze_roofline.achieved_tflops(100, 200), 2e-8)
        self.assertEqual(analyze_roofline.bandwidth_ceiling_tflops(2, 100), 0.2)
        self.assertEqual(
            analyze_roofline.roofline_ceiling_tflops(2, 100, 0.1), 0.1
        )

    def test_analysis_rebuilds_unprofiled_rate_and_preserves_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            analysis = self.build(self.fixture(directory, include_estimate=False))
            case = analysis["cases"][0]

            self.assertEqual(case["unprofiled"]["token_rate"]["value"], 100.0)
            self.assertEqual(case["unprofiled"]["repeat_token_rates"], [100.0, 100.0])
            self.assertEqual(len(case["unprofiled"]["artifacts"]), 3)
            self.assertEqual(case["measurement"]["total_dram_bytes"], 1000)
            self.assertEqual(case["measurement"]["total_flops"], 2000)
            self.assertEqual(
                case["measurement"]["dram_counters"][0]["name"],
                "dram__bytes_read.sum",
            )
            self.assertEqual(len(case["measurement"]["artifacts"]), 2)
            self.assertEqual(
                case["derived"]["arithmetic_intensity"],
                {"value": 2.0, "unit": "FLOP/byte"},
            )
            self.assertEqual(
                case["derived"]["achieved_performance"],
                {"value": 2e-8, "unit": "TFLOP/s"},
            )
            self.assertEqual(
                case["derived"]["selected_roof_ceiling"]["value"], 0.2
            )

    def test_estimate_is_distinct_from_counter_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            analysis = self.build(self.fixture(directory))
            estimate = analysis["cases"][1]
            self.assertEqual(estimate["measurement"]["kind"], "estimate")
            self.assertEqual(estimate["unprofiled"]["token_rate"]["value"], 10.0)
            self.assertEqual(
                estimate["derived"]["achieved_performance"]["value"], 5e-9
            )

    def test_missing_counter_is_unavailable_not_zero_or_estimated(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory, include_estimate=False)
            config["cases"][0]["measurement"] = {
                "kind": "counter",
                "status": "unavailable",
                "scope": "selected prefill kernels",
                "reason": "ERR_NVGPUCTRPERM",
            }
            analysis = self.build(config)
            case = analysis["cases"][0]
            self.assertIsNone(case["derived"])
            self.assertEqual(case["measurement"]["reason"], "ERR_NVGPUCTRPERM")
            self.assertIn("ERR_NVGPUCTRPERM", analysis["diagnostics"][0])

    def test_units_and_phase_rate_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.fixture(directory, include_estimate=False)
            config = copy.deepcopy(original)
            config["device"]["memory_bandwidth"]["unit"] = "GiB/s"
            with self.assertRaisesRegex(
                analyze_roofline.RooflineDataError, "must be 'GB/s'"
            ):
                analyze_roofline.validate_config(config)

            config = copy.deepcopy(original)
            config["cases"][0]["rate_metric"] = "output_token_goodput_per_s"
            with self.assertRaisesRegex(
                analyze_roofline.RooflineDataError, "for prefill"
            ):
                analyze_roofline.validate_config(config)

            config = copy.deepcopy(original)
            config["cases"][0]["measurement"]["dram_counters"][0]["unit"] = "KB"
            with self.assertRaisesRegex(
                analyze_roofline.RooflineDataError, "must be 'byte'"
            ):
                analyze_roofline.validate_config(config)

    def test_example_and_schema_are_valid_json_and_contract_validates(self):
        schema = json.loads(
            (ROOT / "schemas" / "roofline-analysis-input.schema.json").read_text()
        )
        example = json.loads(
            (ROOT / "analysis" / "rtx-2060-super.example.json").read_text()
        )
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        analyze_roofline.validate_config(example)

    def test_golden_markdown_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = analyze_roofline.render_markdown(self.build(self.fixture(directory)))
            expected = (ROOT / "tests" / "fixtures" / "roofline-report.md").read_text()
            self.assertEqual(report, expected.rstrip() + "\n")

    def test_outputs_are_nonempty_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = self.build(self.fixture(root / "inputs"))
            output = root / "outputs"
            paths = analyze_roofline.write_outputs(analysis, output, formats=("png", "pdf", "svg"))
            self.assertEqual(
                {path.name for path in paths},
                {
                    "roofline-analysis.json",
                    "roofline-report.md",
                    "roofline.png",
                    "roofline.pdf",
                    "roofline.svg",
                },
            )
            first = {path.name: path.read_bytes() for path in paths}
            rebuilt = analyze_roofline.write_outputs(
                analysis, output, formats=("png", "pdf", "svg")
            )
            self.assertEqual(first, {path.name: path.read_bytes() for path in rebuilt})
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()
