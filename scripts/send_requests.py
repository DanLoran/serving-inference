import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp


async def send_one(session, url, model, row):
    payload = {
        "model": model,
        "prompt": row["prompt"],
        "max_tokens": row.get("target_output_tokens", 128),
        "temperature": 0.2,
    }

    start = time.perf_counter()

    try:
        async with session.post(url, json=payload) as resp:
            text = await resp.text()
            elapsed = time.perf_counter() - start

            return {
                "id": row.get("id"),
                "status": resp.status,
                "latency_s": elapsed,
                "prompt": row["prompt"],
                "target_output_tokens": payload["max_tokens"],
                "response": text,
            }

    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "id": row.get("id"),
            "status": "error",
            "latency_s": elapsed,
            "prompt": row["prompt"],
            "error": str(e),
        }


async def run(args):
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.prompts, "r") as f:
        prompts = [json.loads(line) for line in f]

    prompts = prompts[: args.num_requests]

    connector = aiohttp.TCPConnector(limit=args.concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(args.concurrency)

        async def bounded_send(row):
            async with sem:
                return await send_one(session, args.url, args.model, row)

        start = time.perf_counter()
        tasks = [bounded_send(row) for row in prompts]
        results = []
        with open(args.output, "w") as out:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                out.write(json.dumps(result) + "\n")
                out.flush()
                # print(
                #     f"{result['id']} status={result['status']} "
                #     f"latency={result['latency_s']:.3f}s"
                # )

        total = time.perf_counter() - start

    latencies = [r["latency_s"] for r in results if r["status"] == 200]
    latencies.sort()

    def percentile(xs, p):
        if not xs:
            return None
        idx = int((p / 100) * (len(xs) - 1))
        return xs[idx]

    print()
    print(f"Completed {len(prompts)} requests")
    print(f"Successful: {len(latencies)}")
    print(f"Total time: {total:.3f}s")
    print(f"Throughput: {len(prompts) / total:.2f} req/s")
    print(f"Avg latency: {sum(latencies) / len(latencies):.3f}s")
    print(f"P50 latency: {percentile(latencies, 50):.3f}s")
    print(f"P90 latency: {percentile(latencies, 90):.3f}s")
    print(f"P99 latency: {percentile(latencies, 99):.3f}s")

    # print()
    # print(f"Completed {len(prompts)} requests")
    # print(f"Total time: {total:.3f}s")
    # print(f"Throughput: {len(prompts) / total:.2f} req/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="prompts/generated_prompts.jsonl")
    parser.add_argument("--output", default="results/run_001.jsonl")
    parser.add_argument("--url", default="http://localhost:8000/v1/completions")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)

    args = parser.parse_args()
    asyncio.run(run(args))



if __name__ == "__main__":
    main()