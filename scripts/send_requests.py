"""Benchmark an OpenAI-compatible completions endpoint.

Raw request records are written as JSONL and an aggregate summary is written as
JSON. Streaming mode measures TTFT and observed inter-chunk latency in addition
to end-to-end latency and throughput.
"""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import aiohttp


def percentile(values, p):
    """Return a linearly interpolated percentile, or None for an empty input."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    clean = [value for value in values if value is not None]
    if not clean:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p99": None}
    return {
        "count": len(clean),
        "mean": sum(clean) / len(clean),
        "p50": percentile(clean, 50),
        "p90": percentile(clean, 90),
        "p99": percentile(clean, 99),
    }


async def read_stream(response, start):
    text_parts = []
    token_count = None
    first_token_at = None
    previous_chunk_at = None
    inter_chunk_latencies = []

    while True:
        line = await response.content.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        usage = event.get("usage") or {}
        if usage.get("completion_tokens") is not None:
            token_count = usage["completion_tokens"]

        choices = event.get("choices") or []
        chunk_text = choices[0].get("text", "") if choices else ""
        if not chunk_text:
            continue
        now = time.perf_counter()
        if first_token_at is None:
            first_token_at = now
        elif previous_chunk_at is not None:
            inter_chunk_latencies.append(now - previous_chunk_at)
        previous_chunk_at = now
        text_parts.append(chunk_text)

    return "".join(text_parts), token_count, first_token_at, inter_chunk_latencies


async def send_one(session, args, row):
    payload = {
        "model": args.model,
        "prompt": row["prompt"],
        "max_tokens": row.get("target_output_tokens", 128),
        "temperature": args.temperature,
        "stream": args.stream,
    }
    if args.stream:
        payload["stream_options"] = {"include_usage": True}

    start = time.perf_counter()
    try:
        async with session.post(args.url, json=payload) as response:
            if args.stream:
                response_text, output_tokens, first_at, inter_chunk = await read_stream(
                    response, start
                )
                response_body = response_text
            else:
                raw_body = await response.text()
                response_body = raw_body
                output_tokens = None
                first_at = None
                inter_chunk = []
                try:
                    parsed = json.loads(raw_body)
                    output_tokens = (parsed.get("usage") or {}).get("completion_tokens")
                except json.JSONDecodeError:
                    pass

            elapsed = time.perf_counter() - start
            ttft = first_at - start if first_at is not None else None
            tpot = None
            if ttft is not None and output_tokens and output_tokens > 1:
                tpot = (elapsed - ttft) / (output_tokens - 1)
            return {
                "id": row.get("id"),
                "status": response.status,
                "latency_s": elapsed,
                "ttft_s": ttft,
                "time_per_output_token_s": tpot,
                "observed_inter_chunk_latency_s": inter_chunk,
                "output_tokens": output_tokens,
                "target_output_tokens": payload["max_tokens"],
                "response": response_body,
            }
    except Exception as error:
        return {
            "id": row.get("id"),
            "status": "error",
            "latency_s": time.perf_counter() - start,
            "error": "%s: %s" % (type(error).__name__, error),
        }


def build_summary(args, results, duration):
    successful = [result for result in results if result["status"] == 200]
    output_tokens = sum(result.get("output_tokens") or 0 for result in successful)
    inter_chunk = [
        latency
        for result in successful
        for latency in result.get("observed_inter_chunk_latency_s", [])
    ]
    return {
        "config": {
            "url": args.url,
            "model": args.model,
            "num_requests": len(results),
            "concurrency": args.concurrency,
            "stream": args.stream,
            "temperature": args.temperature,
        },
        "counts": {
            "attempted": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "output_tokens": output_tokens,
        },
        "duration_s": duration,
        "request_throughput_per_s": len(successful) / duration if duration else None,
        "output_token_throughput_per_s": output_tokens / duration if duration else None,
        "latency_s": distribution([result["latency_s"] for result in successful]),
        "ttft_s": distribution([result.get("ttft_s") for result in successful]),
        "time_per_output_token_s": distribution(
            [result.get("time_per_output_token_s") for result in successful]
        ),
        "observed_inter_chunk_latency_s": distribution(inter_chunk),
    }


async def run(args):
    output_path = Path(args.output)
    summary_path = Path(args.summary or output_path.with_suffix(".summary.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.prompts) as prompt_file:
        prompts = [json.loads(line) for line in prompt_file if line.strip()]
    prompts = prompts[: args.num_requests]
    if not prompts:
        raise ValueError("No prompts were loaded from %s" % args.prompts)

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def bounded_send(row):
            async with semaphore:
                return await send_one(session, args, row)

        started = time.perf_counter()
        tasks = [bounded_send(row) for row in prompts]
        results = []
        with open(output_path, "w") as output_file:
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                output_file.write(json.dumps(result) + "\n")
                output_file.flush()
        duration = time.perf_counter() - started

    summary = build_summary(args, results, duration)
    with open(summary_path, "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")
    print(json.dumps(summary, indent=2))
    print("Raw results: %s" % output_path)
    print("Summary: %s" % summary_path)
    return 0 if summary["counts"]["failed"] == 0 else 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="prompts/generated_prompts.jsonl")
    parser.add_argument("--output", default="results/run_001.jsonl")
    parser.add_argument("--summary")
    parser.add_argument("--url", default="http://localhost:8000/v1/completions")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_requests < 1 or args.concurrency < 1:
        raise SystemExit("--num-requests and --concurrency must be positive")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
