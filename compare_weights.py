#!/usr/bin/env python3
"""Compare tensors between two GGUF models to find which weights differ.

Usage:
    python3 compare_weights.py <model_a> <model_b>

Example:
    python3 compare_weights.py models/qwen3-4b-instruct-2507-Q8_0.gguf models/qwen3-4b-instruct-2507-heretic-Q8_0.gguf
"""

import sys
import numpy as np
from gguf import GGUFReader


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 compare_weights.py <model_a> <model_b>")
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]

    print(f"Loading model A: {path_a}")
    reader_a = GGUFReader(path_a)
    print(f"Loading model B: {path_b}")
    reader_b = GGUFReader(path_b)

    if len(reader_a.tensors) != len(reader_b.tensors):
        print(f"\nTensor count mismatch: {len(reader_a.tensors)} vs {len(reader_b.tensors)}")
        sys.exit(1)

    print(f"\nComparing {len(reader_a.tensors)} tensors (raw bytes)...")
    print(f"\n{'#':<6} {'Tensor':<35} {'Params':>12} {'Status':<10}")
    print("-" * 70)

    changed = []
    identical = 0

    for i, (ta, tb) in enumerate(zip(reader_a.tensors, reader_b.tensors)):
        if ta.name != tb.name:
            print(f"  Name mismatch at index {i}: {ta.name} vs {tb.name}")
            sys.exit(1)

        raw_a = bytes(np.array(ta.data))
        raw_b = bytes(np.array(tb.data))

        n_params = int(np.prod(ta.shape))

        if raw_a == raw_b:
            identical += 1
        else:
            changed.append((i, ta.name, ta.shape.tolist(), n_params, ta.tensor_type.name))
            print(f"{i:<6} {ta.name:<35} {n_params:>12,} DIFFERENT")

    print("-" * 70)
    print(f"\nIdentical: {identical}/{len(reader_a.tensors)}")
    print(f"Different: {len(changed)}/{len(reader_a.tensors)}")

    if changed:
        total_params_in_diff_tensors = sum(c[3] for c in changed)
        total_params = sum(int(np.prod(t.shape)) for t in reader_a.tensors)
        print(f"\nParameters in affected tensors: {total_params_in_diff_tensors:,} / {total_params:,} ({100 * total_params_in_diff_tensors / total_params:.2f}%)")
        print(f"(Use diff_weights.py to see how many individual weights actually changed)")


if __name__ == "__main__":
    main()
