#!/usr/bin/env bash
# Source this file before AudioFlow training or evaluation on this AutoDL instance.
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip-cache}"
export PYTHONPATH="/root/audioflow${PYTHONPATH:+:$PYTHONPATH}"
