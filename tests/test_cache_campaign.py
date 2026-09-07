import argparse
import copy
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_prompts
import run_cache_campaign
import send_requests


class StableTokenizer:
    pieces = {" benchmark": 0x110000, " Alpha": 0x110001, " Beta": 0x110002}
    reverse = {value: key for key, value in pieces.items()}

    def encode(self, text, add_special_tokens=False):
        tokens = []
        index = 0
        while index < len(text):
            match = next(
                (piece for piece in self.pieces if text.startswith(piece, index)),
                None,
            )
            if match is None:
                tokens.append(ord(text[index]))
                index += 1
            else:
                tokens.append(self.pieces[match])
                index += len(match)
        return tokens

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        return "".join(
            self.reverse[token] if token in self.reverse else chr(token)
            for token in token_ids
        )


def small_bank_config(nonce_base=0):
    return {
        "name": "small-bank",
        "model": "test/model",
        "tokenizer": "test/model",
        "seed": 17,
        "request_count": 8,
        "prompt_namespace": "campaign/short/c1/r1/a1",
        "prefix_namespace": "campaign/short",
        "nonce_base": nonce_base,
        "block_size_tokens": 16,
        "buckets": [
            {
                "name": "short",
                "count": 8,
                "prompt_tokens": 64,
                "output_tokens": 8,
                "shared_prefix_tokens": 32,
            }
        ],
    }


class CacheCampaignTest(unittest.TestCase):
    def test_checked_in_campaign_has_exact_requested_matrix(self):
        path = ROOT / "campaigns" / "cache-capacity-baseline.json"
        config = run_cache_campaign.load_config(path)
        plan = run_cache_campaign.build_plan(
            config, path, campaign_root="/tmp/cache-campaign-test"
        )
        self.assertEqual(config["concurrency"], [16, 24, 32, 64, 96, 128, 256])
        self.assertEqual(config["repeats"], 3)
        self.assertEqual(config["requests_per_repeat"], 3072)
        self.assertEqual(
            [item["name"] for item in config["workloads"]],
            ["short", "decode_heavy", "long_prefill", "mixed"],
        )
        self.assertEqual(plan["condition_count"], 84)
        mixed = config["workloads"][-1]
        self.assertEqual([item["count"] for item in mixed["buckets"]], [1536, 768, 768])
        self.assertNotIn("--max-num-seqs", config["server"]["command"])
        self.assertNotIn("--max-num-batched-tokens", config["server"]["command"])

    def test_cache_bank_is_exact_unique_and_block_aligned(self):
        rows, prefixes = generate_prompts.generate_cache_bank(
            small_bank_config(), StableTokenizer()
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row["prompt"] for row in rows}), 8)
        self.assertEqual(len({row["prompt_sha256"] for row in rows}), 8)
        self.assertTrue(all(row["prompt_tokens"] == 64 for row in rows))
        self.assertTrue(all(row["shared_prefix_tokens"] == 32 for row in rows))
        prefix = prefixes["short"]
        tokenizer = StableTokenizer()
        prefix_ids = tokenizer.encode(prefix["text"], add_special_tokens=False)
        self.assertEqual(len(prefix_ids), 32)
        for row in rows:
            self.assertEqual(
                tokenizer.encode(row["prompt"], add_special_tokens=False)[:32],
                prefix_ids,
            )

    def test_nonce_ranges_make_attempt_banks_disjoint(self):
        first, _ = generate_prompts.generate_cache_bank(
            small_bank_config(0), StableTokenizer()
        )
        second, _ = generate_prompts.generate_cache_bank(
            small_bank_config(8), StableTokenizer()
        )
        self.assertFalse(
            {row["prompt_sha256"] for row in first}.intersection(
                row["prompt_sha256"] for row in second
            )
        )

    def test_full_repeat_has_no_first_suffix_block_collisions(self):
        config = small_bank_config()
        config["request_count"] = 3072
        config["buckets"][0]["count"] = 3072
        rows, _ = generate_prompts.generate_cache_bank(config, StableTokenizer())
        self.assertEqual(len(rows), 3072)

    def test_deterministic_gzip_bank_has_verified_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl.gz"
            second = Path(directory) / "second.jsonl.gz"
            _, _, first_meta = generate_prompts.write_cache_bank(
                small_bank_config(), first, tokenizer=StableTokenizer()
            )
            _, _, second_meta = generate_prompts.write_cache_bank(
                small_bank_config(), second, tokenizer=StableTokenizer()
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                first_meta["uncompressed_jsonl_sha256"],
                second_meta["uncompressed_jsonl_sha256"],
            )
            with gzip.open(first, mode="rt", encoding="utf-8") as handle:
                self.assertEqual(len([line for line in handle if line.strip()]), 8)

    def test_sender_loads_gzip_and_preserves_cache_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl.gz"
            row = {
                "id": "one",
                "prompt": "hello",
                "prompt_sha256": "a" * 64,
                "shared_prefix_tokens": 16,
                "prefix_key": "short",
                "prefix_sha256": "b" * 64,
                "bank_id": "bank",
                "bucket": "short",
                "target_output_tokens": 2,
            }
            with gzip.open(path, mode="wt", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            loaded = send_requests.load_prompts(path, 1)[0]
            metadata = send_requests.request_metadata(
                loaded, 0, {"max_tokens": 2}
            )
            self.assertEqual(metadata["prompt_sha256"], "a" * 64)
            self.assertEqual(metadata["shared_prefix_tokens"], 16)
            self.assertEqual(metadata["bucket"], "short")

    def test_prometheus_cache_delta_is_token_based(self):
        before = "\n".join(
            [
                'vllm:prefix_cache_queries_total{engine="0"} 100',
                'vllm:prefix_cache_hits_total{engine="0"} 40',
            ]
        )
        after = "\n".join(
            [
                'vllm:prefix_cache_queries_total{engine="0"} 356',
                'vllm:prefix_cache_hits_total{engine="0"} 168',
            ]
        )
        old = run_cache_campaign.parsed_metrics(before)
        new = run_cache_campaign.parsed_metrics(after)
        queries = run_cache_campaign.metric_delta(new, old, "prefix_queries")
        hits = run_cache_campaign.metric_delta(new, old, "prefix_hits")
        self.assertEqual(hits / queries, 0.5)

    def test_partial_attempt_is_preserved_and_next_number_advances(self):
        condition = {
            "ordinal": 0,
            "workload": "short",
            "concurrency": 16,
            "repeat": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            first = run_cache_campaign.attempt_root(directory, condition, 1)
            first.mkdir(parents=True)
            run_cache_campaign.atomic_write_json(
                first / "attempt.json", {"status": "failed"}
            )
            attempts = run_cache_campaign.existing_attempts(directory, condition)
            self.assertEqual(attempts[-1][0] + 1, 2)
            self.assertTrue((first / "attempt.json").is_file())

    def test_complete_attempt_requires_all_successful_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            run_cache_campaign.atomic_write_json(
                path / "attempt.json", {"status": "completed"}
            )
            run_cache_campaign.atomic_write_json(
                path / "measured.summary.json",
                {"counts": {"attempted": 1, "successful": 1, "failed": 0}},
            )
            run_cache_campaign.atomic_write_json(path / "cache.json", {})
            (path / "measured.jsonl").write_text(
                '{"status": 200}\n', encoding="utf-8"
            )
            self.assertTrue(run_cache_campaign.attempt_is_complete(path, 1))
            (path / "measured.jsonl").write_text(
                '{"status": 500}\n', encoding="utf-8"
            )
            self.assertFalse(run_cache_campaign.attempt_is_complete(path, 1))

    def test_transition_analysis_distinguishes_batching_and_queueing(self):
        rows = [
            {
                "concurrency": 1,
                "output_token_throughput_median_per_s": 100,
                "e2e_s": {"p99": 1.0},
                "waiting": {"max": 0},
            },
            {
                "concurrency": 2,
                "output_token_throughput_median_per_s": 120,
                "e2e_s": {"p99": 1.1},
                "waiting": {"max": 0},
            },
            {
                "concurrency": 4,
                "output_token_throughput_median_per_s": 122,
                "e2e_s": {"p99": 1.5},
                "waiting": {"max": 1},
            },
        ]
        result = run_cache_campaign.transition_analysis(rows)
        self.assertEqual(
            [item["interpretation"] for item in result["transitions"]],
            ["useful batching", "queueing-dominant"],
        )

    def test_analysis_rebuilds_from_selected_attempt_evidence(self):
        source = ROOT / "campaigns" / "cache-capacity-baseline.json"
        config = copy.deepcopy(run_cache_campaign.load_config(source))
        config["name"] = "analysis-test"
        config["concurrency"] = [1, 2]
        config["repeats"] = 2
        config["requests_per_repeat"] = 2
        config["workloads"] = [copy.deepcopy(config["workloads"][0])]
        config["workloads"][0]["buckets"][0]["count"] = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = run_cache_campaign.build_plan(
                config, source, campaign_root=root
            )
            run_cache_campaign.atomic_write_json(
                root / "campaign.original.json", config
            )
            run_cache_campaign.atomic_write_json(
                root / "campaign.resolved.json", plan
            )
            for condition in plan["conditions"]:
                attempt = run_cache_campaign.attempt_root(root, condition, 1)
                attempt.mkdir(parents=True)
                run_cache_campaign.atomic_write_json(
                    attempt / "attempt.json", {"status": "completed"}
                )
                rows = [
                    {
                        "id": "row-%d" % index,
                        "status": 200,
                        "workload": "short",
                        "prompt_tokens": 128,
                        "output_tokens": 64,
                        "latency_s": 1.0 + condition["concurrency"],
                        "ttft_s": 0.1,
                        "approx_time_per_output_token_s": 0.01,
                        "finish_reason": "length",
                        "shared_prefix_tokens": 64,
                    }
                    for index in range(2)
                ]
                (attempt / "measured.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                run_cache_campaign.atomic_write_json(
                    attempt / "measured.summary.json",
                    {
                        "duration_s": 2.0,
                        "counts": {
                            "attempted": 2,
                            "successful": 2,
                            "failed": 0,
                        },
                    },
                )
                run_cache_campaign.atomic_write_json(
                    attempt / "cache.json",
                    {
                        "measured_prefix_queries": 256,
                        "measured_prefix_hits": 128,
                        "intended_prefix_cache_token_hit_rate": 0.5,
                        "preemptions_delta": 0,
                    },
                )
            result = run_cache_campaign.analyze(root)
            self.assertEqual(len(result["results"]), 2)
            self.assertEqual(
                result["results"][0]["measured_prefix_cache_token_hit_rate"],
                0.5,
            )
            self.assertTrue((root / "analysis" / "summary.csv").is_file())
            self.assertTrue((root / "analysis" / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
