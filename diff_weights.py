#!/usr/bin/env python3
"""Find and display individual weight differences between two GGUF models.

Usage:
    python3 diff_weights.py <model_a> <model_b> [--threshold T] [--top N]

Example:
    python3 diff_weights.py models/qwen3-4b-instruct-2507-Q8_0.gguf models/qwen3-4b-instruct-2507-heretic-Q8_0.gguf
"""

import sys
import argparse
import numpy as np
from gguf import GGUFReader


def dequantize_q8_0(tensor):
    raw_bytes = bytes(np.array(tensor.data))
    BLOCK_BYTES = 34
    n_blocks = len(raw_bytes) // BLOCK_BYTES
    all_values = np.empty(n_blocks * 32, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * BLOCK_BYTES
        scale = float(np.frombuffer(raw_bytes[offset:offset + 2], dtype=np.float16)[0])
        quants = np.frombuffer(raw_bytes[offset + 2:offset + BLOCK_BYTES], dtype=np.int8)
        all_values[i * 32:(i + 1) * 32] = quants.astype(np.float32) * scale
    return all_values


def get_values(tensor):
    type_name = tensor.tensor_type.name
    if type_name == "F32":
        return np.array(tensor.data, dtype=np.float32)
    elif type_name == "Q8_0":
        return dequantize_q8_0(tensor)
    return None


def main():
    parser = argparse.ArgumentParser(description="Diff weights between two GGUF models")
    parser.add_argument("model_a", help="Path to first GGUF model")
    parser.add_argument("model_b", help="Path to second GGUF model")
    parser.add_argument("--threshold", "-t", type=float, default=0.0,
                        help="Only show weights where abs difference > threshold (default: 0, show all changed)")
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N largest differences per tensor (default: 20)")
    args = parser.parse_args()

    print(f"Loading model A: {args.model_a}")
    reader_a = GGUFReader(args.model_a)
    print(f"Loading model B: {args.model_b}")
    reader_b = GGUFReader(args.model_b)

    print(f"\nScanning for different tensors...")

    for ta, tb in zip(reader_a.tensors, reader_b.tensors):
        raw_a = bytes(np.array(ta.data))
        raw_b = bytes(np.array(tb.data))

        if raw_a == raw_b:
            continue

        print(f"\n{'=' * 80}")
        print(f"  {ta.name}  |  shape: {ta.shape.tolist()}  |  type: {ta.tensor_type.name}")
        print(f"{'=' * 80}")

        vals_a = get_values(ta)
        vals_b = get_values(tb)

        if vals_a is None or vals_b is None:
            print("  (cannot dequantize)")
            continue

        diff = vals_b - vals_a
        abs_diff = np.abs(diff)

        changed_mask = abs_diff > args.threshold
        n_changed = np.count_nonzero(changed_mask)
        n_total = len(vals_a)

        print(f"\n  Changed weights: {n_changed:,} / {n_total:,} ({100 * n_changed / n_total:.2f}%)")
        print(f"  Abs diff — mean: {abs_diff[changed_mask].mean():.8f}  max: {abs_diff.max():.8f}")

        # Show top N largest differences
        top_indices = np.argsort(abs_diff)[-args.top:][::-1]

        print(f"\n  Top {min(args.top, len(top_indices))} largest differences:")
        print(f"  {'Index':>10}  {'Model A':>12}  {'Model B':>12}  {'Diff':>12}  {'Abs Diff':>12}")
        print(f"  {'-' * 62}")
        for idx in top_indices:
            if abs_diff[idx] <= args.threshold:
                break
            print(f"  {idx:>10,}  {vals_a[idx]:>12.6f}  {vals_b[idx]:>12.6f}  {diff[idx]:>+12.6f}  {abs_diff[idx]:>12.6f}")


if __name__ == "__main__":
    main()
