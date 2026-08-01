"""Benchmark an OpenAI-compatible completions endpoint.

Versioned raw request records are written as JSONL and an aggregate summary is
written as JSON. Streaming mode measures client-observed TTFT, approximate TPOT,
and observed inter-chunk latency in addition to end-to-end latency and
throughput.
"""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import aiohttp

RESULT_SCHEMA_VERSION = "1.0"
MAX_ERROR_BODY_CHARS = 4096


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


async def read_stream(response, clock=time.perf_counter):
    text_parts = []
    usage = {}
    first_token_at = None
    last_token_at = None
    previous_chunk_at = None
    inter_chunk_latencies = []
    finish_reason = None
    malformed_events = 0
    text_event_count = 0

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
            malformed_events += 1
            continue

        event_usage = event.get("usage") or {}
        if event_usage:
            usage.update(event_usage)

        choices = event.get("choices") or []
        choice = choices[0] if choices else {}
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]
        chunk_text = choice.get("text", "")
        if not chunk_text:
            continue
        now = clock()
        if first_token_at is None:
            first_token_at = now
        elif previous_chunk_at is not None:
            inter_chunk_latencies.append(now - previous_chunk_at)
        previous_chunk_at = now
        last_token_at = now
        text_event_count += 1
        text_parts.append(chunk_text)

    return {
        "text": "".join(text_parts),
        "usage": usage,
        "first_text_at": first_token_at,
        "last_text_at": last_token_at,
        "text_event_count": text_event_count,
        "inter_chunk_latencies": inter_chunk_latencies,
        "finish_reason": finish_reason,
        "malformed_events": malformed_events,
    }


def request_metadata(row, request_index, payload):
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "id": row.get("id"),
        "request_index": request_index,
        "workload": row.get("workload") or row.get("category"),
        "prompt_tokens": row.get("prompt_tokens"),
        "target_prompt_tokens": row.get("target_prompt_tokens"),
        "target_output_tokens": payload["max_tokens"],
    }


def truncate_error_body(body):
    if len(body) <= MAX_ERROR_BODY_CHARS:
        return body, False
    return body[:MAX_ERROR_BODY_CHARS], True


async def send_one(
    session,
    args,
    row,
    request_index=None,
    run_started=None,
    clock=time.perf_counter,
):
    payload = {
        "model": args.model,
        "prompt": row["prompt"],
        "max_tokens": row.get("target_output_tokens", 128),
        "temperature": args.temperature,
        "stream": args.stream,
    }
    if args.stream:
        payload["stream_options"] = {"include_usage": True}

    start = clock()
    run_started = start if run_started is None else run_started
    metadata = request_metadata(row, request_index, payload)
    try:
        async with session.post(args.url, json=payload) as response:
            if response.status != 200:
                error_body, truncated = truncate_error_body(await response.text())
                ended = clock()
                return {
                    **metadata,
                    "status": response.status,
                    "started_offset_s": start - run_started,
                    "ended_offset_s": ended - run_started,
                    "latency_s": ended - start,
                    "ttft_s": None,
                    "approx_time_per_output_token_s": None,
                    "observed_inter_chunk_latency_s": [],
                    "output_tokens": None,
                    "server_token_usage_present": False,
                    "finish_reason": None,
                    "error_body": error_body,
                    "error_body_truncated": truncated,
                }

            if args.stream:
                stream_result = await read_stream(response, clock=clock)
                response_body = stream_result["text"]
                usage = stream_result["usage"]
                output_tokens = usage.get("completion_tokens")
                prompt_tokens = usage.get("prompt_tokens")
                first_at = stream_result["first_text_at"]
                last_at = stream_result["last_text_at"]
                inter_chunk = stream_result["inter_chunk_latencies"]
                finish_reason = stream_result["finish_reason"]
                malformed_events = stream_result["malformed_events"]
                text_event_count = stream_result["text_event_count"]
            else:
                raw_body = await response.text()
                response_body = raw_body
                usage = {}
                output_tokens = None
                prompt_tokens = None
                first_at = None
                last_at = None
                inter_chunk = []
                finish_reason = None
                malformed_events = 0
                text_event_count = 0
                try:
                    parsed = json.loads(raw_body)
                    usage = parsed.get("usage") or {}
                    output_tokens = usage.get("completion_tokens")
                    prompt_tokens = usage.get("prompt_tokens")
                    choices = parsed.get("choices") or []
                    if choices:
                        finish_reason = choices[0].get("finish_reason")
                except json.JSONDecodeError:
                    pass

            ended = clock()
            elapsed = ended - start
            ttft = first_at - start if first_at is not None else None
            approx_tpot = None
            if (
                first_at is not None
                and last_at is not None
                and text_event_count > 1
                and output_tokens
                and output_tokens > 1
            ):
                approx_tpot = (last_at - first_at) / (output_tokens - 1)
            result = {
                **metadata,
                "status": response.status,
                "started_offset_s": start - run_started,
                "ended_offset_s": ended - run_started,
                "latency_s": elapsed,
                "ttft_s": ttft,
                "approx_time_per_output_token_s": approx_tpot,
                "observed_inter_chunk_latency_s": inter_chunk,
                "output_tokens": output_tokens,
                "prompt_tokens": (
                    prompt_tokens
                    if prompt_tokens is not None
                    else metadata["prompt_tokens"]
                ),
                "server_token_usage_present": output_tokens is not None,
                "finish_reason": finish_reason,
                "stream_text_event_count": text_event_count if args.stream else None,
                "malformed_stream_events": malformed_events if args.stream else None,
            }
            if getattr(args, "store_response", False):
                result["response"] = response_body
            return result
    except Exception as error:
        ended = clock()
        return {
            **metadata,
            "status": "error",
            "started_offset_s": start - run_started,
            "ended_offset_s": ended - run_started,
            "latency_s": ended - start,
            "ttft_s": None,
            "approx_time_per_output_token_s": None,
            "observed_inter_chunk_latency_s": [],
            "output_tokens": None,
            "server_token_usage_present": False,
            "finish_reason": None,
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
            "store_response": getattr(args, "store_response", False),
            "result_schema_version": RESULT_SCHEMA_VERSION,
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
        "approx_time_per_output_token_s": distribution(
            [result.get("approx_time_per_output_token_s") for result in successful]
        ),
        "observed_inter_chunk_latency_s": distribution(inter_chunk),
    }


def load_prompts(path, num_requests):
    with open(path) as prompt_file:
        prompts = [json.loads(line) for line in prompt_file if line.strip()]
    if not prompts:
        raise ValueError("No prompts were loaded from %s" % path)
    if len(prompts) < num_requests:
        raise ValueError(
            "Requested %d requests but %s contains only %d prompts"
            % (num_requests, path, len(prompts))
        )
    return prompts[:num_requests]


async def run(args):
    output_path = Path(args.output)
    summary_path = Path(args.summary or output_path.with_suffix(".summary.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts, args.num_requests)

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def bounded_send(request_index, row):
            async with semaphore:
                return await send_one(
                    session,
                    args,
                    row,
                    request_index=request_index,
                    run_started=started,
                )

        started = time.perf_counter()
        tasks = [bounded_send(index, row) for index, row in enumerate(prompts)]
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
    parser.add_argument(
        "--store-response",
        action="store_true",
        help="Store full model responses in raw results (disabled by default).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_requests < 1 or args.concurrency < 1:
        raise SystemExit("--num-requests and --concurrency must be positive")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
