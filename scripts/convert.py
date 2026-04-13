#!/usr/bin/env python3
"""
Convert a nanochat PyTorch checkpoint to the Go binary format.

Usage:
    python scripts/convert.py [--source sft|rl|base] [--model-tag TAG] [--step STEP] [--output DIR]

This script:
1. Loads the PyTorch model and tokenizer from nanochat's checkpoint system
2. Exports model weights to model.bin (binary format)
3. Exports tokenizer vocabulary to tokenizer.json

The output directory contains everything needed by the Go inference engine.
"""

import argparse
import json
import struct
import sys
import os

# Add nanochat to path
NANOCHAT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "nanochat")
if os.path.isdir(NANOCHAT_DIR):
    sys.path.insert(0, NANOCHAT_DIR)

import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer


def export_tokenizer(tokenizer, output_path):
    """Export tokenizer to JSON format for Go."""
    enc = tokenizer.enc  # tiktoken Encoding object

    # Extract mergeable ranks: bytes -> rank
    # We hex-encode the byte sequences for JSON compatibility
    mergeable_ranks = {}
    for token_bytes, rank in enc._mergeable_ranks.items():
        hex_key = token_bytes.hex()
        mergeable_ranks[hex_key] = rank

    # Extract special tokens
    special_tokens = dict(enc._special_tokens)

    data = {
        "vocab_size": enc.n_vocab,
        "pattern": enc._pat_str,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"Exported tokenizer ({enc.n_vocab} tokens) to {output_path}")


def export_model(model, meta_data, output_path):
    """Export model weights to binary format for Go."""
    config = meta_data["model_config"]

    # Collect all weight tensors
    state_dict = model.state_dict()

    # Remove _orig_mod. prefix (torch.compile artifact)
    clean_state = {}
    for key, tensor in state_dict.items():
        clean_key = key.removeprefix("_orig_mod.")
        # Convert to float32 for Go (no bfloat16 support)
        clean_state[clean_key] = tensor.float().cpu()

    # Write binary format
    with open(output_path, "wb") as f:
        # Magic
        f.write(b"NANO")

        # Version
        f.write(struct.pack("<I", 1))

        # Config as JSON
        config_json = json.dumps(config).encode("utf-8")
        f.write(struct.pack("<I", len(config_json)))
        f.write(config_json)

        # Number of tensors
        f.write(struct.pack("<I", len(clean_state)))

        # Write each tensor
        for name, tensor in clean_state.items():
            # Name
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            # Shape
            shape = tensor.shape
            f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f.write(struct.pack("<I", dim))

            # Data (float32, little-endian)
            f.write(tensor.contiguous().numpy().tobytes())

        total_params = sum(t.numel() for t in clean_state.values())
        file_size = os.path.getsize(output_path)
        print(f"Exported model ({total_params:,} params, {file_size / 1e6:.1f} MB) to {output_path}")
        print(f"  Config: {config}")
        print(f"  Tensors: {len(clean_state)}")
        for name, tensor in clean_state.items():
            print(f"    {name}: {list(tensor.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Convert nanochat checkpoint to Go format")
    parser.add_argument("-i", "--source", type=str, default="sft",
                       help="Source of the model: base|sft|rl (default: sft)")
    parser.add_argument("-g", "--model-tag", type=str, default=None,
                       help="Model tag (e.g. d20). Default: largest available")
    parser.add_argument("-s", "--step", type=int, default=None,
                       help="Checkpoint step. Default: latest")
    parser.add_argument("-o", "--output", type=str, default="model_export",
                       help="Output directory (default: model_export)")
    parser.add_argument("--device-type", type=str, default="cpu",
                       choices=["cuda", "cpu", "mps"],
                       help="Device type (default: cpu)")
    args = parser.parse_args()

    # Initialize compute
    device_type = args.device_type
    _, _, _, _, device = compute_init(device_type)

    # Load model and tokenizer
    print(f"Loading {args.source} model...")
    model, tokenizer, meta_data = load_model(
        args.source, device, phase="eval",
        model_tag=args.model_tag, step=args.step
    )

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Export tokenizer
    tok_path = os.path.join(args.output, "tokenizer.json")
    export_tokenizer(tokenizer, tok_path)

    # Export model
    model_path = os.path.join(args.output, "model.bin")
    export_model(model, meta_data, model_path)

    print(f"\nDone! Model exported to {args.output}/")
    print(f"To use with Go: ./gonanochat serve -m {args.output}")


if __name__ == "__main__":
    main()
