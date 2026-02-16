#!/bin/bash
# Start vLLM server for google/gemma-2b-it (T4 GPU compatible)
# See VLLM_SETUP.md for details

VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS vllm serve google/gemma-2b-it --dtype float16 --port 8000
