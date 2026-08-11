#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-artifacts/wan22_vae}"
mkdir -p "$OUTPUT_DIR"
uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "$OUTPUT_DIR"
test -f "$OUTPUT_DIR/Wan2.2_VAE.pth"
echo "Wan2.2 VAE ready: $OUTPUT_DIR/Wan2.2_VAE.pth"
