#!/usr/bin/env python3
"""
Test script for any vLLM-served model (local or remote).

Usage:
  Step 1 (if remote): Open SSH tunnel in a separate terminal:
    ssh -i ~/.ssh/uw_vm_key -L 8000:localhost:8000 suryanshrawat@136.109.97.101

  Step 2: Run this script:
    python test_vllm_model.py --list-models
    python test_vllm_model.py --model Qwen/Qwen2.5-3B-Instruct
    python test_vllm_model.py --model google/gemma-2b-it --prompt "What is 2+2?"
    python test_vllm_model.py --show-curl

  Or test with curl directly:
    curl -s http://localhost:8000/v1/models
    curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \\
      -d '{"model":"YOUR_MODEL","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
"""

import argparse
import json
import sys
from typing import List, Tuple

from openai import OpenAI


# GCP instance config (for SSH tunnel hint)
GCP_HOST = "136.109.97.101"
VLLM_PORT = 8000

DEFAULT_BASE_URL = f"http://localhost:{VLLM_PORT}/v1"


def create_client(base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    """Create OpenAI-compatible client pointing to vLLM server."""
    return OpenAI(
        base_url=base_url,
        api_key="dummy",
    )


def get_available_models(client: OpenAI) -> List[str]:
    """List models available on the vLLM server."""
    try:
        models = client.models.list()
        return [m.id for m in models.data]
    except Exception:
        return []


def check_connection(client: OpenAI, verbose: bool = True) -> Tuple[bool, List[str]]:
    """Verify the vLLM server is reachable and return available models."""
    try:
        models = get_available_models(client)
        if verbose:
            print(f"Connected to {client.base_url}")
            print(f"Available models: {models}")
        return True, models
    except Exception as e:
        if verbose:
            print(f"Cannot connect to vLLM server at {client.base_url}")
            print(f"Error: {e}")
            print(
                f"\nSSH tunnel (if remote):\n"
                f"  ssh -i ~/.ssh/uw_vm_key -L {VLLM_PORT}:localhost:{VLLM_PORT} "
                f"suryanshrawat@{GCP_HOST}"
            )
        return False, []


def test_completion(
    client: OpenAI,
    prompt: str,
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Send a chat completion request and return the generated text."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def print_curl_examples(base_url: str, model: str) -> None:
    """Print ready-to-run curl commands."""
    base = base_url.replace("/v1", "")
    print("\n--- curl examples ---")
    print("\n# List models:")
    print(f'curl -s "{base}/v1/models"')
    print("\n# Chat completion:")
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 64,
    })
    print(f'curl -s "{base}/v1/chat/completions" -H "Content-Type: application/json" -d \'{payload}\'')
    print("\n# One-liner (extract response text):")
    payload_short = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 32,
    })
    extract_cmd = 'python3 -c "import sys,json; print(json.load(sys.stdin)[\\"choices\\"][0][\\"message\\"][\\"content\\"])"'
    print(f"curl -s '{base}/v1/chat/completions' -H 'Content-Type: application/json' -d '{payload_short}' | {extract_cmd}")


def run_tests(
    base_url: str, model: str, max_tokens: int, temperature: float
) -> None:
    """Run a set of test prompts against the model."""
    client = create_client(base_url)

    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print("-" * 60)

    ok, models = check_connection(client)
    if not ok:
        sys.exit(1)
    if not models:
        print("No models found.")
        sys.exit(1)
    if model not in models:
        print(f"Warning: '{model}' not in server list. Using anyway (server may accept it).")

    test_prompts = [
        "What is 2 + 2? Reply in one sentence.",
        "List three primary colors.",
        "Complete this: The capital of France is",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}] Prompt: {prompt}")
        try:
            response = test_completion(
                client, prompt, model, max_tokens=max_tokens, temperature=temperature
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
            return

    print("\n" + "-" * 60)
    print("All tests completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Test any vLLM-served model via OpenAI-compatible API"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"vLLM API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: first model from server)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--show-curl",
        action="store_true",
        help="Print curl commands to test the server",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single custom prompt (skips built-in tests)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    args = parser.parse_args()

    client = create_client(args.base_url)
    ok, models = check_connection(client)
    if not ok:
        sys.exit(1)

    # Resolve model
    model = args.model
    if not model and models:
        model = models[0]
        print(f"Using first available model: {model}")
    elif not model:
        print("No model specified and none found. Use --model MODEL_NAME")
        sys.exit(1)

    if args.list_models:
        for m in models:
            print(m)
        sys.exit(0)

    if args.show_curl:
        print_curl_examples(args.base_url, model)
        sys.exit(0)

    if args.prompt:
        print(f"Prompt: {args.prompt}")
        response = test_completion(
            client,
            args.prompt,
            model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"Response: {response}")
    else:
        run_tests(args.base_url, model, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
