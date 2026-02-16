# vLLM Setup Guide: Gemma-2B-IT on T4 GPU

This document describes how to run `google/gemma-2b-it` locally using vLLM, including the settings required for GPUs with compute capability < 8.0 (e.g., **NVIDIA T4 16GB**).

---

## Hardware Requirements

- **GPU**: NVIDIA T4 (16GB VRAM) or similar
- **Compute capability**: 7.x (T4 has 7.5)
- **Note**: Flash Attention 2 requires compute capability ≥ 8.0 (A100, etc.), so T4 uses a different backend.

---

## Install vLLM

```bash
pip install vllm
```

Or with specific version:

```bash
pip install vllm==0.15.1
```

---

## Required Settings for T4 / Older GPUs

On GPUs like the T4, the default vLLM V1 engine tries to use **FlashInfer**, which requires the CUDA toolkit (`nvcc`) for JIT compilation. Many cloud images (e.g., GCP) do not include the full CUDA toolkit, causing:

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

Additionally, **Flash Attention 2** is not supported on compute capability < 8.0:

```
Cannot use FA version 2 is not supported due to FA2 is only supported on devices with compute capability >= 8
```

### Recommended Fix: Use V0 Engine with XFORMERS

Force the V0 engine and the XFORMERS attention backend, which works on T4:

```bash
VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS vllm serve google/gemma-2b-it --dtype float16 --port 8000
```

| Variable | Value | Purpose |
|----------|-------|---------|
| `VLLM_USE_V1` | `0` | Use the stable V0 engine instead of V1 |
| `VLLM_ATTENTION_BACKEND` | `XFORMERS` | Use xformers attention (works on T4) |

---

## Start the Server

```bash
VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS vllm serve google/gemma-2b-it --dtype float16 --port 8000
```

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8000 | Port for the API server |
| `--dtype` | auto | Use `float16` for T4 16GB |
| `--max-model-len` | 8192 | Max sequence length |

Example with custom port:

```bash
VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS vllm serve google/gemma-2b-it --dtype float16 --port 9000
```

---

## Verify the Server

Once the server is running:

```bash
# Check model is loaded
curl http://localhost:8000/v1/models

# Quick chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-2b-it",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 32
  }'
```

---

## Quick Start Script

```bash
chmod +x start_vllm_server.sh
./start_vllm_server.sh
```

---

## Test Script

Use the included test script:

```bash
# List models on the server
python test_vllm_model.py --list-models

# Run built-in test prompts
python test_vllm_model.py

# Custom prompt
python test_vllm_model.py --prompt "Explain what machine learning is in 2 sentences."

# Custom base URL and model
python test_vllm_model.py --base-url http://localhost:9000/v1 --model google/gemma-2b-it
```

---

## Integration with Augment_tableQA

The pipeline uses a **vLLM config file** so all inference goes to your local vLLM server (no OpenAI API).

### 1. Configure vLLM

Copy the example config and edit it:

```bash
cp vllm_config.example.json vllm_config.json
```

Edit `vllm_config.json` in the project root:

```json
{
  "base_url": "http://localhost:8000/v1",
  "model": "google/gemma-2b-it",
  "dummy_api_key": "not-needed-for-vllm"
}
```

- **base_url**: vLLM OpenAI-compatible API URL. Use `http://localhost:8000/v1` when using an SSH tunnel (see below).
- **model**: Must match the model you started with `vllm serve` (e.g. `google/gemma-2b-it`, `Qwen/Qwen2.5-3B-Instruct`).
- **dummy_api_key**: Ignored by vLLM; can be any string.

To use another model, change `model` in `vllm_config.json` and set `ENGINE` in the run script (e.g. `run_augment_wikitq.py`) to the same value so output filenames and tokenizer stay in sync.

### 2. Running vLLM on a Remote Machine (e.g. GCP)

If vLLM runs on a remote instance:

1. **On the remote machine**: Start vLLM (see "Start the Server" above).
2. **On your local machine**: Open an SSH tunnel so `localhost:8000` forwards to the remote server:

   ```bash
   ssh -i ~/.ssh/your_key -L 8000:localhost:8000 user@remote-host
   ```

3. **In the project**: Keep `base_url` as `http://localhost:8000/v1` in `vllm_config.json`. The pipeline runs locally and sends requests through the tunnel.

### 3. key.txt

Keep a `key.txt` file in the project root (can contain a single line like `dummy`). The Generator reads `vllm_config.json` from the same directory as `key.txt` for the actual endpoint and model.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Could not find nvcc` | CUDA toolkit not installed; FlashInfer needs it | Use `VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS` |
| `FA2 is only supported on devices with compute capability >= 8` | T4 has compute 7.5 | Same as above |
| `VLLM_ATTENTION_BACKEND=TRITON_ATTN` ignored | V1 engine overrides env var | Use `VLLM_USE_V1=0` |
| Out of memory | Model too large for GPU | Use `--dtype float16` or a smaller model |

---

## Reference: Command One-Liner

```bash
VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS vllm serve google/gemma-2b-it --dtype float16 --port 8000
```
