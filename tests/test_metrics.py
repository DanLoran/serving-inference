import argparse
import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import send_requests


class FakeContent:
    def __init__(self, lines):
        self.lines = iter(lines)

    async def readline(self):
        return next(self.lines, b"")


class FakeResponse:
    def __init__(self, lines=(), status=200, body=""):
        self.content = FakeContent(lines)
        self.status = status
        self.body = body

    async def text(self):
        return self.body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def post(self, url, json):
        if self.error:
            raise self.error
        return self.response


class FakeClock:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class MetricsTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(send_requests.percentile([1, 2, 3], 50), 2)
        self.assertEqual(send_requests.percentile([1, 3], 50), 2)
        self.assertIsNone(send_requests.percentile([], 99))

    def test_summary_uses_successful_requests(self):
        args = argparse.Namespace(
            url="http://localhost:8000/v1/completions",
            model="mock",
            concurrency=2,
            stream=True,
            temperature=0.0,
        )
        results = [
            {
                "status": 200,
                "latency_s": 1.0,
                "ttft_s": 0.2,
                "approx_time_per_output_token_s": 0.1,
                "observed_inter_chunk_latency_s": [0.1],
                "output_tokens": 4,
            },
            {"status": 500, "latency_s": 0.1},
        ]
        summary = send_requests.build_summary(args, results, 2.0)
        self.assertEqual(summary["counts"]["successful"], 1)
        self.assertEqual(summary["counts"]["failed"], 1)
        self.assertEqual(summary["request_throughput_per_s"], 0.5)
        self.assertEqual(summary["output_token_throughput_per_s"], 2.0)

    def test_stream_parser_collects_text_usage_and_timing(self):
        response = FakeResponse(
            [
                b'data: {"choices":[{"text":"hello"}]}\n',
                b"\n",
                b"data: not-json\n",
                b'data: {"choices":[{"text":" world","finish_reason":"length"}]}\n',
                b'data: {"choices":[],"usage":{"completion_tokens":2}}\n',
                b"data: [DONE]\n",
            ]
        )
        result = asyncio.run(
            send_requests.read_stream(response, clock=FakeClock(10.2, 10.5))
        )
        self.assertEqual(result["text"], "hello world")
        self.assertEqual(result["usage"]["completion_tokens"], 2)
        self.assertEqual(result["first_text_at"], 10.2)
        self.assertEqual(result["last_text_at"], 10.5)
        self.assertEqual(len(result["inter_chunk_latencies"]), 1)
        self.assertAlmostEqual(result["inter_chunk_latencies"][0], 0.3)
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["malformed_events"], 1)

    def test_streaming_request_records_auditable_timing_and_usage(self):
        args = argparse.Namespace(
            url="http://localhost:8000/v1/completions",
            model="mock",
            stream=True,
            temperature=0.0,
            store_response=False,
        )
        row = {
            "id": "request-7",
            "prompt": "hello",
            "category": "short",
            "prompt_tokens": 1,
            "target_output_tokens": 2,
        }
        response = FakeResponse(
            [
                b'data: {"choices":[{"text":"one"}]}\n',
                b'data: {"choices":[{"text":" two","finish_reason":"length"}]}\n',
                b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}\n',
                b"data: [DONE]\n",
            ]
        )
        result = asyncio.run(
            send_requests.send_one(
                FakeSession(response=response),
                args,
                row,
                request_index=7,
                run_started=10.0,
                clock=FakeClock(10.1, 10.2, 10.5, 10.7),
            )
        )
        self.assertEqual(result["schema_version"], "1.0")
        self.assertEqual(result["request_index"], 7)
        self.assertEqual(result["workload"], "short")
        self.assertAlmostEqual(result["started_offset_s"], 0.1)
        self.assertAlmostEqual(result["ended_offset_s"], 0.7)
        self.assertAlmostEqual(result["ttft_s"], 0.1)
        self.assertAlmostEqual(result["approx_time_per_output_token_s"], 0.3)
        self.assertTrue(result["server_token_usage_present"])
        self.assertEqual(result["finish_reason"], "length")
        self.assertNotIn("response", result)

    def test_single_text_event_has_no_approximate_tpot(self):
        args = argparse.Namespace(
            url="http://localhost:8000/v1/completions",
            model="mock",
            stream=True,
            temperature=0.0,
            store_response=True,
        )
        response = FakeResponse(
            [
                b'data: {"choices":[{"text":"two tokens"}]}\n',
                b'data: {"choices":[],"usage":{"completion_tokens":2}}\n',
                b"data: [DONE]\n",
            ]
        )
        result = asyncio.run(
            send_requests.send_one(
                FakeSession(response=response),
                args,
                {"id": "one", "prompt": "p", "target_output_tokens": 2},
                clock=FakeClock(1.0, 1.1, 1.2),
            )
        )
        self.assertIsNone(result["approx_time_per_output_token_s"])
        self.assertEqual(result["response"], "two tokens")

    def test_empty_stream_without_usage_marks_metrics_unavailable(self):
        args = argparse.Namespace(
            url="http://localhost:8000/v1/completions",
            model="mock",
            stream=True,
            temperature=0.0,
            store_response=False,
        )
        response = FakeResponse([b"data: [DONE]\n"])
        result = asyncio.run(
            send_requests.send_one(
                FakeSession(response=response),
                args,
                {"id": "empty", "prompt": "p"},
                clock=FakeClock(2.0, 2.1),
            )
        )
        self.assertIsNone(result["ttft_s"])
        self.assertIsNone(result["approx_time_per_output_token_s"])
        self.assertIsNone(result["output_tokens"])
        self.assertFalse(result["server_token_usage_present"])
        self.assertEqual(result["stream_text_event_count"], 0)

    def test_http_error_body_is_preserved_and_bounded(self):
        args = argparse.Namespace(
            url="http://localhost:8000/v1/completions",
            model="mock",
            stream=True,
            temperature=0.0,
            store_response=False,
        )
        response = FakeResponse(
            status=429, body="x" * (send_requests.MAX_ERROR_BODY_CHARS + 10)
        )
        result = asyncio.run(
            send_requests.send_one(
                FakeSession(response=response),
                args,
                {"id": "limited", "prompt": "p"},
                clock=FakeClock(1.0, 1.5),
            )
        )
        self.assertEqual(result["status"], 429)
        self.assertEqual(len(result["error_body"]), send_requests.MAX_ERROR_BODY_CHARS)
        self.assertTrue(result["error_body_truncated"])
        self.assertFalse(result["server_token_usage_present"])

    def test_transport_exception_has_complete_failure_schema(self):
        args = argparse.Namespace(
            url="http://localhost:8000/v1/completions",
            model="mock",
            stream=False,
            temperature=0.0,
            store_response=False,
        )
        result = asyncio.run(
            send_requests.send_one(
                FakeSession(error=TimeoutError("late")),
                args,
                {"id": "timeout", "prompt": "p"},
                request_index=3,
                run_started=4.0,
                clock=FakeClock(4.1, 5.1),
            )
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["request_index"], 3)
        self.assertEqual(result["latency_s"], 1.0)
        self.assertIsNone(result["ttft_s"])
        self.assertIn("TimeoutError", result["error"])


if __name__ == "__main__":
    unittest.main()
