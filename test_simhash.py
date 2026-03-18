#!/usr/bin/env python3
"""Compare two GGUF models tensor-by-tensor using SimHash."""

import sys
import time
import numpy as np
from gguf import GGUFReader
from simhash_sdk import simhash_from_bytes, hamming_distance

K = 128

path_a = "models/qwen3-4b-instruct-2507-Q8_0.gguf"
path_b = "models/qwen3-4b-instruct-2507-heretic-Q8_0.gguf"

print(f"Model A: {path_a}")
print(f"Model B: {path_b}")
print(f"Sketch bits (K): {K}\n", flush=True)

reader_a = GGUFReader(path_a)
reader_b = GGUFReader(path_b)

print(f"{'#':<6} {'Tensor':<40} {'Bytes':>12} {'Hamming':>8} {'Similarity':>12} {'Time':>8}")
print("-" * 90, flush=True)

total_dist = 0
n_different = 0
start_all = time.time()

for i, (ta, tb) in enumerate(zip(reader_a.tensors, reader_b.tensors)):
    raw_a = bytes(np.array(ta.data))
    raw_b = bytes(np.array(tb.data))

    t0 = time.time()
    sketch_a = simhash_from_bytes(raw_a, K=K)
    sketch_b = simhash_from_bytes(raw_b, K=K)
    elapsed = time.time() - t0

    dist = hamming_distance(sketch_a, sketch_b)
    sim = 1 - dist / K
    total_dist += dist

    marker = " *" if dist > 0 else ""
    if dist > 0:
        n_different += 1

    print(f"{i:<6} {ta.name:<40} {ta.n_bytes:>12,} {dist:>8} {sim:>11.2%} {elapsed:>7.1f}s{marker}", flush=True)

total_time = time.time() - start_all
n_tensors = len(reader_a.tensors)
avg_sim = 1 - total_dist / (n_tensors * K)

print("-" * 90)
print(f"\nTotal tensors: {n_tensors}")
print(f"Tensors with hamming > 0: {n_different}")
print(f"Average similarity: {avg_sim:.2%}")
print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)
