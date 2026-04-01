#!/usr/bin/env python3
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark llama.cpp embedding latency through llama-server."
    )
    parser.add_argument("--server-bin", default="llama-server", help="Path to llama-server.")
    parser.add_argument("--model", required=True, help="Path to GGUF embedding model.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=18080, help="Server port.")
    parser.add_argument("--iterations", type=int, default=20, help="Measured embedding requests.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup embedding requests.")
    parser.add_argument("--pooling", default="cls", help="llama.cpp pooling mode.")
    parser.add_argument("--ub", type=int, default=8192, help="llama.cpp ub parameter.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Server startup/request timeout in seconds.")
    parser.add_argument(
        "--text",
        default="This is a short embedding benchmark sentence for cortext.",
        help="Text to embed.",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Do not terminate the spawned llama-server process on exit.",
    )
    return parser.parse_args()


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def http_post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_embedding(response: dict) -> Optional[int]:
    if isinstance(response.get("embedding"), list):
        return len(response["embedding"])
    data = response.get("data")
    if isinstance(data, list) and data:
        emb = data[0].get("embedding")
        if isinstance(emb, list):
            return len(emb)
    return None


def post_embedding(base_url: str, text: str, timeout: float) -> Tuple[dict, str]:
    attempts = [
        ("/embedding", {"content": text}),
        ("/embedding", {"input": text}),
        ("/v1/embeddings", {"input": text}),
        ("/v1/embeddings", {"input": [text]}),
    ]
    last_error = None
    for path, payload in attempts:
        try:
            response = http_post_json(base_url + path, payload, timeout)
            if extract_embedding(response) is not None:
                return response, path
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Embedding request failed for all known endpoints: {last_error}")


def wait_for_server(base_url: str, text: str, timeout: float) -> str:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            _, path = post_embedding(base_url, text, timeout=10.0)
            return path
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"llama-server did not become ready: {last_error}")


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    base_url = f"http://{args.host}:{args.port}"
    server_proc = None
    log_file = None
    endpoint = None

    if is_port_open(args.host, args.port):
        endpoint = wait_for_server(base_url, args.text, args.timeout)
    else:
        log_file = tempfile.NamedTemporaryFile(
            prefix="llama_cpp_embed_bench_", suffix=".log", delete=False
        )
        cmd = [
            args.server_bin,
            "-m",
            str(model_path),
            "--embedding",
            "--pooling",
            args.pooling,
            "-ub",
            str(args.ub),
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        server_proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        try:
            endpoint = wait_for_server(base_url, args.text, args.timeout)
        except Exception:
            if server_proc.poll() is not None:
                log_file.flush()
            raise

    assert endpoint is not None

    try:
        for _ in range(args.warmup):
            post_embedding(base_url, args.text, timeout=args.timeout)

        start = time.perf_counter()
        dim = None
        for _ in range(args.iterations):
            response, _ = post_embedding(base_url, args.text, timeout=args.timeout)
            dim = extract_embedding(response)
        total_ms = (time.perf_counter() - start) * 1000.0
        mean_ms = total_ms / args.iterations if args.iterations > 0 else 0.0
        embeds_per_sec = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        print(f"server_bin={args.server_bin}")
        print(f"model={model_path}")
        print(f"endpoint={endpoint}")
        print(f"iterations={args.iterations}")
        print(f"warmup={args.warmup}")
        print(f"embedding_dim={dim}")
        print(f"total_ms={total_ms:.2f}")
        print(f"mean_ms={mean_ms:.2f}")
        print(f"embeddings_per_sec={embeds_per_sec:.2f}")
        if log_file is not None:
            print(f"server_log={log_file.name}")
        return 0
    finally:
        if server_proc is not None and not args.keep_server:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=5.0)
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    sys.exit(main())
