import argparse
import json
import random
from pathlib import Path

TEMPLATES = [
    "Explain {topic} in simple terms.",
    "Write a short technical summary of {topic}.",
    "Give three practical examples of {topic}.",
    "Write pseudocode for {topic}.",
]

TOPICS = [
    "CUDA shared memory",
    "continuous batching",
    "KV cache memory usage",
    "request scheduling",
    "GPU kernel launch overhead",
    "LLM inference latency",
    "prefill versus decode",
    "throughput versus latency tradeoffs",
]

OUTPUT_TOKEN_BUCKETS = [64, 128, 256, 512]

def generate(n=500, out_path="prompts/generated_prompts.jsonl", seed=0):
    rng = random.Random(seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for i in range(n):
            topic = rng.choice(TOPICS)
            template = rng.choice(TEMPLATES)
            target_output_tokens = rng.choice(OUTPUT_TOKEN_BUCKETS)

            prompt = template.format(topic=topic)

            row = {
                "id": f"prompt_{i:06d}",
                "prompt": prompt,
                "category": "systems_inference",
                "target_output_tokens": target_output_tokens,
            }

            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate a deterministic inference workload.")
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--output", default="prompts/generated_prompts.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate(args.num_prompts, args.output, args.seed)


if __name__ == "__main__":
    main()
