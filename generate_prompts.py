"""Generate deterministic, token-controlled inference workloads."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


CONFIG_DIR = Path("experiments")
OUTPUT_DIR = Path("prompts")
WORKLOADS = ("short", "long_prefill", "decode_heavy", "mixed")
SOURCE_TEXT = (
    "Analyze how continuous batching, KV-cache allocation, request scheduling, "
    "and GPU kernel execution affect latency and throughput in an inference "
    "service. Use precise technical language and concrete reasoning. "
)


def load_tokenizer(model):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "transformers is required; install dependencies with "
            "python3 -m pip install -r requirements.txt"
        ) from error
    return AutoTokenizer.from_pretrained(model)


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"name", "model", "tokenizer", "seed", "request_count", "buckets"}
    missing = required.difference(config)
    if missing:
        raise ValueError("missing config fields: %s" % ", ".join(sorted(missing)))
    if config["name"] not in WORKLOADS:
        raise ValueError("unknown workload: %s" % config["name"])
    if config["model"] != config["tokenizer"]:
        raise ValueError("model and tokenizer must match exactly")
    return config


def bucket_sequence(config):
    buckets = config["buckets"]
    counts = [bucket["count"] for bucket in buckets]
    if sum(counts) != config["request_count"]:
        raise ValueError("bucket counts must equal request_count")
    sequence = [bucket for bucket in buckets for _ in range(bucket["count"])]
    random.Random(config["seed"]).shuffle(sequence)
    return sequence


def make_prompt(tokenizer, target_tokens, workload, request_index):
    prefix = "%s request %06d. " % (workload, request_index)
    source = prefix + SOURCE_TEXT
    source_ids = tokenizer.encode(source, add_special_tokens=False)
    filler_ids = tokenizer.encode(" benchmark", add_special_tokens=False)
    if len(filler_ids) != 1:
        raise ValueError("tokenizer must encode the deterministic filler as one token")
    while len(source_ids) < target_tokens:
        source_ids.extend(filler_ids)
    prompt = tokenizer.decode(
        source_ids[:target_tokens],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    actual_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return prompt, len(actual_ids)


def validate_rows(rows, config):
    if len(rows) != config["request_count"]:
        raise ValueError("unexpected request count")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("request IDs must be unique")

    tolerance = config.get("prompt_token_tolerance", 0)
    for row in rows:
        if row["prompt_tokens"] <= 0 or row["target_output_tokens"] <= 0:
            raise ValueError("token lengths must be positive")
        delta = abs(row["prompt_tokens"] - row["target_prompt_tokens"])
        if delta > tolerance:
            raise ValueError(
                "%s prompt token count is outside tolerance: %d vs %d (+/-%d)"
                % (
                    row["id"],
                    row["prompt_tokens"],
                    row["target_prompt_tokens"],
                    tolerance,
                )
            )

    actual_counts = Counter(row["bucket"] for row in rows)
    for bucket in config["buckets"]:
        bucket_name, count = bucket["name"], bucket["count"]
        if actual_counts[bucket_name] != count:
            raise ValueError("unstable bucket composition for %s" % bucket_name)


def serialize_rows(rows):
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )


def generate(config, tokenizer):
    rows = []
    for index, bucket in enumerate(bucket_sequence(config)):
        prompt, prompt_tokens = make_prompt(
            tokenizer, bucket["prompt_tokens"], config["name"], index
        )
        rows.append(
            {
                "id": "%s_%06d" % (config["name"], index),
                "workload": config["name"],
                "bucket": bucket["name"],
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "target_prompt_tokens": bucket["prompt_tokens"],
                "target_output_tokens": bucket["output_tokens"],
            }
        )
    validate_rows(rows, config)
    return rows


def write_workload(config_path, output_dir=OUTPUT_DIR, tokenizer=None):
    config_path = Path(config_path)
    config = load_config(config_path)
    tokenizer = tokenizer or load_tokenizer(config["tokenizer"])
    rows = generate(config, tokenizer)
    serialized = serialize_rows(rows)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / (config["name"] + ".jsonl")
    metadata_path = output_dir / (config["name"] + ".metadata.json")
    data_path.write_text(serialized, encoding="utf-8")
    metadata = {
        "schema_version": "1.0",
        "workload_sha256": digest,
        "generation_config": config,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return data_path, metadata_path, digest


def verify_workload(config_path, output_dir=OUTPUT_DIR):
    config = load_config(config_path)
    output_dir = Path(output_dir)
    data_path = output_dir / (config["name"] + ".jsonl")
    metadata_path = output_dir / (config["name"] + ".metadata.json")
    serialized = data_path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in serialized.splitlines() if line]
    validate_rows(rows, config)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if metadata["generation_config"] != config:
        raise ValueError("metadata generation config does not match source config")
    if metadata["workload_sha256"] != digest:
        raise ValueError("workload SHA-256 mismatch")
    return digest


def config_paths(selected):
    names = WORKLOADS if selected == "all" else (selected,)
    return [CONFIG_DIR / (name + ".json") for name in names]


def main():
    parser = argparse.ArgumentParser(
        description="Generate or verify deterministic token-controlled workloads."
    )
    parser.add_argument("--workload", choices=("all",) + WORKLOADS, default="all")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    for path in config_paths(args.workload):
        if args.verify:
            digest = verify_workload(path, args.output_dir)
            print("verified %s %s" % (path.stem, digest))
        else:
            _, _, digest = write_workload(path, args.output_dir)
            print("generated %s %s" % (path.stem, digest))


if __name__ == "__main__":
    main()
