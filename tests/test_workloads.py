import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import generate_prompts


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == " benchmark":
            return [0x10FFFF]
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(token_id) for token_id in token_ids)


def config(name="mixed"):
    return {
        "name": name,
        "model": "test/model",
        "tokenizer": "test/model",
        "seed": 7,
        "request_count": 8,
        "prompt_token_tolerance": 0,
        "temperature": 0.0,
        "buckets": [
            {"name": "short", "count": 4, "prompt_tokens": 64, "output_tokens": 8},
            {"name": "long_prefill", "count": 2, "prompt_tokens": 128, "output_tokens": 8},
            {"name": "decode_heavy", "count": 2, "prompt_tokens": 64, "output_tokens": 32},
        ],
    }


class WorkloadTest(unittest.TestCase):
    def test_generation_is_deterministic(self):
        tokenizer = CharacterTokenizer()
        first = generate_prompts.generate(config(), tokenizer)
        second = generate_prompts.generate(config(), tokenizer)
        self.assertEqual(first, second)
        self.assertEqual(
            hashlib.sha256(generate_prompts.serialize_rows(first).encode()).hexdigest(),
            hashlib.sha256(generate_prompts.serialize_rows(second).encode()).hexdigest(),
        )

    def test_lengths_ids_and_bucket_composition(self):
        rows = generate_prompts.generate(config(), CharacterTokenizer())
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        self.assertTrue(all(row["prompt_tokens"] > 0 for row in rows))
        self.assertTrue(all(row["target_output_tokens"] > 0 for row in rows))
        self.assertTrue(
            all(row["prompt_tokens"] == row["target_prompt_tokens"] for row in rows)
        )
        self.assertEqual(
            Counter(row["bucket"] for row in rows),
            Counter({"short": 4, "long_prefill": 2, "decode_heavy": 2}),
        )

    def test_write_and_verify_preserves_config_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "mixed.json"
            config_path.write_text(json.dumps(config()), encoding="utf-8")
            _, metadata_path, digest = generate_prompts.write_workload(
                config_path, root, tokenizer=CharacterTokenizer()
            )
            self.assertEqual(generate_prompts.verify_workload(config_path, root), digest)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["generation_config"], config())
            self.assertEqual(metadata["workload_sha256"], digest)

    def test_validation_rejects_out_of_tolerance_length(self):
        rows = generate_prompts.generate(config(), CharacterTokenizer())
        rows[0]["prompt_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "outside tolerance"):
            generate_prompts.validate_rows(rows, config())

    def test_bucket_counts_must_match_request_count(self):
        invalid = config()
        invalid["request_count"] = 9
        with self.assertRaisesRegex(ValueError, "bucket counts"):
            generate_prompts.bucket_sequence(invalid)


if __name__ == "__main__":
    unittest.main()
