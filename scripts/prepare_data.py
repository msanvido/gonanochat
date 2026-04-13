#!/usr/bin/env python3
"""
Prepare training data for gonanochat by tokenizing text into binary format.

The output is a flat binary file of uint16 token IDs, ready for the Go training loop.

Usage:
    # Using nanochat's trained tokenizer:
    python scripts/prepare_data.py -i data.txt -o train.bin

    # Split into train/val:
    python scripts/prepare_data.py -i data.txt -o train.bin --val val.bin --val-fraction 0.05

    # Using a tiktoken encoding (no nanochat required):
    python scripts/prepare_data.py -i data.txt -o train.bin --tiktoken cl100k_base
"""

import argparse
import struct
import sys
import os


def tokenize_with_nanochat(text, bos_every=0):
    """Tokenize using nanochat's trained tokenizer."""
    NANOCHAT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "nanochat")
    if os.path.isdir(NANOCHAT_DIR):
        sys.path.insert(0, NANOCHAT_DIR)

    from nanochat.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()

    bos = tokenizer.get_bos_token_id()
    tokens = [bos]  # start with BOS
    tokens.extend(tokenizer.encode(text))

    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    print(f"Total tokens: {len(tokens):,}")
    return tokens, tokenizer.get_vocab_size()


def tokenize_with_tiktoken(text, encoding_name):
    """Tokenize using a tiktoken encoding."""
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    print(f"Tiktoken encoding: {encoding_name}, vocab size: {enc.n_vocab}")
    print(f"Total tokens: {len(tokens):,}")
    return tokens, enc.n_vocab


def save_tokens(tokens, path, use_uint32=False):
    """Save tokens as binary file with a small header.
    Format: magic (2 bytes 'TK') + dtype (1 byte: 2=uint16, 4=uint32) + pad (1 byte)
            + token data (uint16 or uint32 little-endian)
    """
    dtype = 4 if use_uint32 else 2
    fmt = "I" if use_uint32 else "H"
    with open(path, "wb") as f:
        f.write(b"TK")                  # magic
        f.write(struct.pack("BB", dtype, 0))  # dtype + padding
        f.write(struct.pack(f"<{len(tokens)}{fmt}", *tokens))
    bits = dtype * 8
    size_mb = os.path.getsize(path) / 1e6
    print(f"Saved {len(tokens):,} tokens as uint{bits} ({size_mb:.1f} MB) to {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for gonanochat")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input text file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output binary file")
    parser.add_argument("--val", type=str, default=None, help="Validation output file")
    parser.add_argument("--val-fraction", type=float, default=0.05, help="Fraction for validation")
    parser.add_argument("--tiktoken", type=str, default=None,
                       help="Use tiktoken encoding instead of nanochat tokenizer")
    args = parser.parse_args()

    # Read input text
    print(f"Reading {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Input: {len(text):,} characters")

    # Tokenize
    if args.tiktoken:
        tokens, vocab_size = tokenize_with_tiktoken(text, args.tiktoken)
    else:
        tokens, vocab_size = tokenize_with_nanochat(text)

    # Use uint32 if any token ID exceeds uint16 range
    max_token = max(tokens)
    use_uint32 = max_token >= 65536
    if use_uint32:
        print(f"Note: max token ID is {max_token}, using uint32 format")

    # Split train/val
    if args.val:
        split_idx = int(len(tokens) * (1 - args.val_fraction))
        train_tokens = tokens[:split_idx]
        val_tokens = tokens[split_idx:]
        save_tokens(train_tokens, args.output, use_uint32)
        save_tokens(val_tokens, args.val, use_uint32)
    else:
        save_tokens(tokens, args.output, use_uint32)

    print(f"\nVocab size: {vocab_size}")
    print(f"Use with: gonanochat train -data {args.output} -vocab {vocab_size}")


if __name__ == "__main__":
    main()
