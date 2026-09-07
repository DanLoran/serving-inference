"""Generate deterministic, token-controlled inference workloads."""

import argparse
import gzip
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


def load_tokenizer(model, revision=None):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "transformers is required; install dependencies with "
            "python3 -m pip install -r requirements.txt"
        ) from error
    options = {"revision": revision} if revision else {}
    return AutoTokenizer.from_pretrained(model, **options)


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
    namespace = config.get("prompt_namespace", config["name"])
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("prompt_namespace must be a non-empty string")
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


def _round_trip(tokenizer, token_ids):
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return text, tokenizer.encode(text, add_special_tokens=False)


def make_shared_prefix(tokenizer, namespace, token_count):
    """Build one exact, token-boundary-stable shared prefix."""
    if token_count <= 0:
        raise ValueError("shared prefix token count must be positive")
    filler = tokenizer.encode(" benchmark", add_special_tokens=False)
    if len(filler) != 1:
        raise ValueError("tokenizer must encode the deterministic filler as one token")
    ids = tokenizer.encode(
        " Cache campaign shared prefix for %s. %s" % (namespace, SOURCE_TEXT),
        add_special_tokens=False,
    )
    while len(ids) < token_count:
        ids.extend(filler)
    ids = ids[:token_count]
    text, actual = _round_trip(tokenizer, ids)
    if actual != ids:
        ids = tokenizer.encode(
            " Cache prefix %s." % namespace, add_special_tokens=False
        )
        ids = (ids + filler * token_count)[:token_count]
        text, actual = _round_trip(tokenizer, ids)
    if actual != ids or len(actual) != token_count:
        raise ValueError("shared prefix failed exact tokenizer round-trip")
    return text, ids


def make_cache_prompt(tokenizer, prefix_ids, target_tokens, nonce):
    """Build an exact prompt whose first tokens are the supplied prefix."""
    remaining = target_tokens - len(prefix_ids)
    if remaining < 32:
        raise ValueError("unique prompt suffix must contain at least 32 tokens")
    filler = tokenizer.encode(" benchmark", add_special_tokens=False)
    zero = tokenizer.encode(" Alpha", add_special_tokens=False)
    one = tokenizer.encode(" Beta", add_special_tokens=False)
    if len(filler) != 1 or len(zero) != 1 or len(one) != 1:
        raise ValueError(
            "tokenizer must encode benchmark, Alpha, and Beta fillers as one token"
        )
    bits = [
        one[0] if nonce & (1 << bit) else zero[0]
        for bit in range(32)
    ]
    suffix_ids = (bits + filler * remaining)[:remaining]
    intended = list(prefix_ids) + suffix_ids
    prompt, actual = _round_trip(tokenizer, intended)
    if actual != intended or len(actual) != target_tokens:
        raise ValueError("cache-aware prompt failed exact tokenizer round-trip")
    return prompt, actual


def cache_bucket_sequence(config):
    buckets = config["buckets"]
    if sum(bucket["count"] for bucket in buckets) != config["request_count"]:
        raise ValueError("bucket counts must equal request_count")
    sequence = [bucket for bucket in buckets for _ in range(bucket["count"])]
    random.Random(config["seed"]).shuffle(sequence)
    return sequence


def generate_cache_bank(config, tokenizer):
    """Generate a deterministic bank with exact shared prefixes and unique prompts."""
    required = {
        "name",
        "model",
        "tokenizer",
        "seed",
        "request_count",
        "prompt_namespace",
        "prefix_namespace",
        "block_size_tokens",
        "buckets",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(
            "missing cache-bank fields: %s" % ", ".join(sorted(missing))
        )
    if config["model"] != config["tokenizer"]:
        raise ValueError("model and tokenizer must match exactly")
    block_size = config["block_size_tokens"]
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size_tokens must be a positive integer")

    prefixes = {}
    for bucket in config["buckets"]:
        shared = bucket.get("shared_prefix_tokens")
        prompt_tokens = bucket.get("prompt_tokens")
        if not isinstance(shared, int) or shared <= 0:
            raise ValueError("shared_prefix_tokens must be positive")
        if shared % block_size:
            raise ValueError("shared prefixes must be block aligned")
        if not isinstance(prompt_tokens, int) or shared >= prompt_tokens:
            raise ValueError("shared prefix must be shorter than the prompt")
        prefix_key = bucket["name"]
        prefix_text, prefix_ids = make_shared_prefix(
            tokenizer,
            "%s/%s" % (config["prefix_namespace"], prefix_key),
            shared,
        )
        prefixes[prefix_key] = {
            "text": prefix_text,
            "token_ids": prefix_ids,
            "tokens": shared,
            "sha256": hashlib.sha256(prefix_text.encode("utf-8")).hexdigest(),
        }

    rows = []
    prompt_hashes = set()
    first_suffix_blocks = {name: set() for name in prefixes}
    for index, bucket in enumerate(cache_bucket_sequence(config)):
        prefix_key = bucket["name"]
        nonce = config.get("nonce_base", 0) + index
        if nonce < 0 or nonce >= 2**32:
            raise ValueError("cache-aware prompt nonce must fit in 32 bits")
        prompt, token_ids = make_cache_prompt(
            tokenizer,
            prefixes[prefix_key]["token_ids"],
            bucket["prompt_tokens"],
            nonce,
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_hash in prompt_hashes:
            raise ValueError("complete measured prompts must be unique")
        suffix_start = prefixes[prefix_key]["tokens"]
        suffix_block = tuple(token_ids[suffix_start : suffix_start + block_size])
        if suffix_block in first_suffix_blocks[prefix_key]:
            raise ValueError("first unique suffix blocks must not repeat")
        prompt_hashes.add(prompt_hash)
        first_suffix_blocks[prefix_key].add(suffix_block)
        rows.append(
            {
                "id": "%s_%06d" % (config["name"], index),
                "workload": bucket["name"],
                "bucket": bucket["name"],
                "bank_id": config["name"],
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "prompt_tokens": len(token_ids),
                "target_prompt_tokens": bucket["prompt_tokens"],
                "target_output_tokens": bucket["output_tokens"],
                "shared_prefix_tokens": prefixes[prefix_key]["tokens"],
                "prefix_key": prefix_key,
                "prefix_sha256": prefixes[prefix_key]["sha256"],
            }
        )
    return rows, prefixes


def write_cache_bank(config, data_path, tokenizer=None):
    """Write a deterministic gzip JSONL bank plus auditable metadata."""
    tokenizer = tokenizer or load_tokenizer(config["tokenizer"])
    rows, prefixes = generate_cache_bank(config, tokenizer)
    serialized = serialize_rows(rows).encode("utf-8")
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(serialized)
    metadata = {
        "schema_version": "cache-prompt-bank-1.0",
        "generation_config": config,
        "request_count": len(rows),
        "composition": dict(sorted(Counter(row["bucket"] for row in rows).items())),
        "uncompressed_jsonl_sha256": hashlib.sha256(serialized).hexdigest(),
        "compressed_file_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "prompt_hash_chain_sha256": hashlib.sha256(
            b"".join(bytes.fromhex(row["prompt_sha256"]) for row in rows)
        ).hexdigest(),
        "all_prompt_lengths_exact": True,
        "complete_prompts_unique_within_bank": True,
        "first_suffix_block_unique_within_prefix": True,
        "shared_prefixes": prefixes,
    }
    metadata_path = data_path.with_suffix("").with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return data_path, metadata_path, metadata


def validate_rows(rows, config):
    if len(rows) != config["request_count"]:
        raise ValueError("unexpected request count")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("request IDs must be unique")
    prompts = [row.get("prompt") for row in rows]
    if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("prompts must be non-empty strings")
    if len(prompts) != len(set(prompts)):
        raise ValueError("prompt text must be unique within a workload")

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
    prompt_namespace = config.get("prompt_namespace", config["name"])
    for index, bucket in enumerate(bucket_sequence(config)):
        prompt, prompt_tokens = make_prompt(
            tokenizer, bucket["prompt_tokens"], prompt_namespace, index
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
