#!/usr/bin/env python3
"""Inspect actual weight values from a GGUF model tensor.

Usage:
    python3 inspect_weights.py <model_path> [tensor_name] [--offset N] [--count N]

Examples:
    # List all tensors
    python3 inspect_weights.py models/qwen3-4b-instruct-2507-Q8_0.gguf

    # Inspect a specific tensor (first 100 values)
    python3 inspect_weights.py models/qwen3-4b-instruct-2507-Q8_0.gguf blk.0.attn_q.weight

    # Inspect with offset and count
    python3 inspect_weights.py models/qwen3-4b-instruct-2507-Q8_0.gguf blk.0.attn_q.weight --offset 1000 --count 50

    # Search tensors by substring
    python3 inspect_weights.py models/qwen3-4b-instruct-2507-Q8_0.gguf --search ffn_gate
"""

import sys
import argparse
import numpy as np
from gguf import GGUFReader


def dequantize_q8_0(tensor):
    """Dequantize a Q8_0 tensor into float32 values."""
    raw_bytes = bytes(np.array(tensor.data))
    BLOCK_BYTES = 34  # 2 (fp16 scale) + 32 (int8 values)
    n_blocks = len(raw_bytes) // BLOCK_BYTES

    all_values = np.empty(n_blocks * 32, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * BLOCK_BYTES
        scale = float(np.frombuffer(raw_bytes[offset:offset + 2], dtype=np.float16)[0])
        quants = np.frombuffer(raw_bytes[offset + 2:offset + BLOCK_BYTES], dtype=np.int8)
        all_values[i * 32:(i + 1) * 32] = quants.astype(np.float32) * scale

    return all_values


def get_values(tensor):
    """Get float32 values from a tensor, handling both F32 and Q8_0."""
    type_name = tensor.tensor_type.name
    if type_name == "F32":
        return np.array(tensor.data, dtype=np.float32)
    elif type_name == "Q8_0":
        return dequantize_q8_0(tensor)
    else:
        return None


def list_tensors(reader, search=None):
    """List all tensors, optionally filtered by search string."""
    print(f"\n{'#':<6} {'Name':<35} {'Shape':<20} {'Params':>12} {'Type':<6}")
    print("-" * 85)
    for i, t in enumerate(reader.tensors):
        name = t.name
        if search and search not in name:
            continue
        shape = t.shape.tolist()
        n_params = int(np.prod(shape))
        print(f"{i:<6} {name:<35} {str(shape):<20} {n_params:>12,} {t.tensor_type.name:<6}")


def get_raw_int8(tensor):
    """Extract raw int8 quantized values from a Q8_0 tensor."""
    raw_bytes = bytes(np.array(tensor.data))
    BLOCK_BYTES = 34  # 2 (fp16 scale) + 32 (int8 values)
    n_blocks = len(raw_bytes) // BLOCK_BYTES

    all_values = np.empty(n_blocks * 32, dtype=np.int8)
    for i in range(n_blocks):
        offset = i * BLOCK_BYTES
        quants = np.frombuffer(raw_bytes[offset + 2:offset + BLOCK_BYTES], dtype=np.int8)
        all_values[i * 32:(i + 1) * 32] = quants

    return all_values


def inspect_tensor_raw(reader, tensor_name, offset=0, count=100):
    """Print raw int8 quantized values for a given tensor."""
    tensor = None
    for t in reader.tensors:
        if t.name == tensor_name:
            tensor = t
            break

    if tensor is None:
        print(f"Tensor '{tensor_name}' not found. Use --search to find tensors.")
        sys.exit(1)

    shape = tensor.shape.tolist()
    n_params = int(np.prod(shape))
    type_name = tensor.tensor_type.name

    print(f"\nTensor:  {tensor.name}")
    print(f"Shape:   {shape}")
    print(f"Params:  {n_params:,}")
    print(f"Type:    {type_name}")
    print(f"Bytes:   {tensor.n_bytes:,}")

    if type_name == "F32":
        print(f"\nTensor is F32 (not quantized). Use without --raw to see values.")
        return

    if type_name != "Q8_0":
        print(f"\nRaw mode not supported for type '{type_name}'.")
        return

    values = get_raw_int8(tensor)
    end = min(offset + count, len(values))
    subset = values[offset:end]

    print(f"\nRaw int8 values [{offset}:{end}] of {len(values):,}:")
    print("-" * 70)
    for i in range(0, len(subset), 10):
        idx = offset + i
        row = subset[i:i + 10]
        nums = " ".join(f"{v:>4d}" for v in row)
        print(f"  [{idx:>8}] {nums}")

    print(f"\nStats (full tensor):")
    print(f"  min:    {values.min()}")
    print(f"  max:    {values.max()}")
    print(f"  mean:   {values.mean():.2f}")
    print(f"  std:    {values.std():.2f}")
    print(f"  zeros:  {np.count_nonzero(values == 0):,} / {len(values):,}")


def inspect_tensor(reader, tensor_name, offset=0, count=100):
    """Print weight values for a given tensor."""
    tensor = None
    for t in reader.tensors:
        if t.name == tensor_name:
            tensor = t
            break

    if tensor is None:
        print(f"Tensor '{tensor_name}' not found. Use --search to find tensors.")
        sys.exit(1)

    shape = tensor.shape.tolist()
    n_params = int(np.prod(shape))
    type_name = tensor.tensor_type.name

    print(f"\nTensor:  {tensor.name}")
    print(f"Shape:   {shape}")
    print(f"Params:  {n_params:,}")
    print(f"Type:    {type_name}")
    print(f"Bytes:   {tensor.n_bytes:,}")

    values = get_values(tensor)
    if values is None:
        print(f"\nDequantization not supported for type '{type_name}'.")
        return

    end = min(offset + count, len(values))
    subset = values[offset:end]

    print(f"\nValues [{offset}:{end}] of {len(values):,}:")
    print("-" * 70)
    for i in range(0, len(subset), 10):
        idx = offset + i
        row = subset[i:i + 10]
        nums = " ".join(f"{v:>10.6f}" for v in row)
        print(f"  [{idx:>8}] {nums}")

    print(f"\nStats (full tensor):")
    print(f"  min:    {values.min():.6f}")
    print(f"  max:    {values.max():.6f}")
    print(f"  mean:   {values.mean():.6f}")
    print(f"  std:    {values.std():.6f}")
    print(f"  zeros:  {np.count_nonzero(values == 0):,} / {len(values):,}")


def main():
    parser = argparse.ArgumentParser(description="Inspect GGUF model weights")
    parser.add_argument("model", help="Path to GGUF model file")
    parser.add_argument("tensor", nargs="?", help="Tensor name to inspect")
    parser.add_argument("--search", "-s", help="Filter tensor list by substring")
    parser.add_argument("--offset", "-o", type=int, default=0, help="Start index (default: 0)")
    parser.add_argument("--count", "-n", type=int, default=100, help="Number of values to show (default: 100)")
    parser.add_argument("--raw", "-r", action="store_true", help="Show raw Q8_0 blocks (scale + int8 values) instead of dequantized floats")

    args = parser.parse_args()

    reader = GGUFReader(args.model)

    if args.tensor:
        if args.raw:
            inspect_tensor_raw(reader, args.tensor, args.offset, args.count)
        else:
            inspect_tensor(reader, args.tensor, args.offset, args.count)
    else:
        list_tensors(reader, args.search)


if __name__ == "__main__":
    main()
