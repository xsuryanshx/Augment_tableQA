# vLLM Setup Guide

This document walks through setting up a vLLM inference server on a GCP instance with a T4 GPU, connecting to it from your local machine via SSH tunnel, and configuring the project to use it.

The project hosts vLLM on a **separate GPU server** and all scripts access it automatically through the OpenAI-compatible API. Your local machine does not need a GPU — it only sends HTTP requests to the remote vLLM server.

---

## 1. Create a GCP VM Instance

We used the following configuration:

| Setting | Value |
|---------|-------|
| **Machine type** | `n2-standard-4` (4 vCPUs, 16 GB RAM) |
| **GPU** | 1x NVIDIA T4 (16 GB VRAM) |
| **Boot disk** | Deep Learning VM Image (PyTorch), 100 GB SSD |
| **Zone** | `us-west1-b` (or any zone with T4 availability) |

### Using the GCP Console

1. Go to **Compute Engine → VM instances → Create Instance**.
2. Set the machine type to **n2-standard-4**.
3. Under **GPUs**, click **Add GPU** and select **NVIDIA T4** (1 GPU).
4. Under **Boot disk**, click **Change** → select **Deep Learning on Linux** image family (this comes with CUDA drivers pre-installed).
5. Set disk size to at least **100 GB** (model weights need space).
6. Under **Firewall**, allow HTTP/HTTPS traffic if needed.
7. Click **Create**.

### Using `gcloud` CLI

```bash
gcloud compute instances create vllm-server \
  --zone=us-west1-b \
  --machine-type=n2-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --metadata="install-nvidia-driver=True"
```

### After the VM Boots

SSH into the instance and verify the GPU is available:

```bash
gcloud compute ssh vllm-server --zone=us-west1-b
nvidia-smi
```

You should see the T4 listed with ~16 GB memory.

---

## 2. Install vLLM on the Server

On the GCP instance:

```bash
pip install vllm
```

---

## 3. Start the vLLM Server

The T4 has compute capability 7.5, which does **not** support Flash Attention 2 or FlashInfer (requires ≥ 8.0). You must use the V0 engine with the XFORMERS backend:

```bash
VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS vllm serve Qwen/Qwen2.5-3B-Instruct --dtype float16 --port 8000
```

A convenience script is also included:

```bash
chmod +x start_vllm_server.sh
./start_vllm_server.sh
```

| Environment Variable | Value | Purpose |
|----------------------|-------|---------|
| `VLLM_USE_V1` | `0` | Use the stable V0 engine (avoids FlashInfer JIT issues) |
| `VLLM_ATTENTION_BACKEND` | `XFORMERS` | Use xformers attention (works on T4) |

| Server Flag | Default | Description |
|-------------|---------|-------------|
| `--port` | 8000 | Port for the OpenAI-compatible API |
| `--dtype` | auto | Use `float16` for T4 (16 GB VRAM) |
| `--max-model-len` | 8192 | Max sequence length (reduce if OOM) |

> **Tip:** Run the server in a `tmux` or `screen` session so it persists after you disconnect.

---

## 4. Connect from Your Local Machine (SSH Tunnel)

Since the vLLM server runs on the remote GCP instance, you need an SSH tunnel so your local scripts can reach it at `localhost:8000`:

```bash
ssh -i ~/.ssh/your_gcp_key -L 8000:localhost:8000 your_username@<GCP_EXTERNAL_IP>
```

For example:

```bash
ssh -i ~/.ssh/uw_vm_key -L 8000:localhost:8000 suryanshrawat@136.109.97.101
```

This forwards local port `8000` → remote port `8000`. Keep this terminal open while running experiments.

> All project scripts use `http://localhost:8000/v1` as the endpoint, so the SSH tunnel makes the remote server appear local — no code changes needed.

---

## 5. Configure the Project

### `vllm_config.json`

If `vllm_config.json` does not exist in the project root, create it from the example:

```bash
cp vllm_config.example.json vllm_config.json
```

Edit it to match your vLLM server:

```json
{
  "base_url": "http://localhost:8000/v1",
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "dummy_api_key": "not-needed-for-vllm"
}
```

- **`base_url`**: Keep as `http://localhost:8000/v1` (the SSH tunnel handles routing to the remote server).
- **`model`**: Must exactly match the model name you passed to `vllm serve` on the server.
- **`dummy_api_key`**: Any string — vLLM does not require authentication.

The run scripts (e.g., `run_augment_finqa.py`) read the model name from this file automatically.

### `key.txt`

If `key.txt` does not exist in the project root, create it:

```bash
echo "dummy" > key.txt
```

This file is required by the scripts but not used for authentication when using vLLM. The `Generator` locates `vllm_config.json` relative to the path of `key.txt`.

---

## 6. Verify Everything Works

Use the included test script to confirm the server is reachable and responding:

```bash
# List models available on the server
python test_vllm_model.py --list-models

# Run built-in test prompts
python test_vllm_model.py

# Test with a custom prompt
python test_vllm_model.py --prompt "What is 2+2?"

# Test a specific model
python test_vllm_model.py --model Qwen/Qwen2.5-3B-Instruct

# Print curl commands for manual testing
python test_vllm_model.py --show-curl
```

You can also verify manually with curl:

```bash
# Check the server is up
curl -s http://localhost:8000/v1/models

# Send a test request
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 32
  }'
```

If the connection fails, check that:
1. The vLLM server is running on the GCP instance (`nvidia-smi` should show the process).
2. The SSH tunnel is active in another terminal.
3. The port in the tunnel matches the `--port` used by `vllm serve`.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Could not find nvcc` | CUDA toolkit not installed; FlashInfer needs JIT compilation | Use `VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS` |
| `FA2 is only supported on devices with compute capability >= 8` | T4 has compute capability 7.5 | Same as above |
| `Connection refused` on localhost:8000 | SSH tunnel not active or server not running | Open the SSH tunnel and verify server is up |
| Out of memory | Model too large for T4 | Use `--dtype float16`, reduce `--max-model-len`, or use a smaller model |
| Model name mismatch | `vllm_config.json` model doesn't match `vllm serve` model | Make sure both use the exact same model name |
