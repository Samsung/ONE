#!/usr/bin/env python3
import sys
import os
import argparse
import subprocess
import httpx
import urllib.request
from huggingface_hub import snapshot_download
from config import BUILD_DIR


def main():
    parser = argparse.ArgumentParser(description="Import model from HuggingFace")
    parser.add_argument("model_id", help="HuggingFace model ID")
    parser.add_argument(
        "-r",
        "--requirements",
        help="Path to requirements file to install (default: requirements.txt)")
    args = parser.parse_args()

    model_id = args.model_id
    model_basename = model_id.split("/")[-1].lower()

    # Create build directory
    os.makedirs(BUILD_DIR, exist_ok=True)

    # Download into build directory
    target_dir = os.path.join(BUILD_DIR, model_basename)
    print(f"Downloading model files for {model_id} into {target_dir}...")

    # Patch httpx to disable SSL verification and forward proxy if set
    original_client = httpx.Client
    proxy = urllib.request.getproxies()

    def patched_client(*args, **kwargs):
        # Always disable SSL verification (needed for self‑signed certs)
        kwargs.setdefault('verify', False)
        # If a proxy is defined, pass it to the client
        if proxy:
            kwargs.setdefault('proxies', proxy)
        return original_client(*args, **kwargs)

    httpx.Client = patched_client

    try:
        snapshot_download(repo_id=model_id, local_dir=target_dir)
    finally:
        # Restore original httpx.Client
        httpx.Client = original_client

    # Install requirements
    requirements_file = args.requirements if args.requirements else "requirements.txt"

    requirements_path = os.path.abspath(requirements_file)
    if not os.path.exists(requirements_path):
        print(f"Error: {requirements_path} not found.")
        return


if __name__ == "__main__":
    main()
