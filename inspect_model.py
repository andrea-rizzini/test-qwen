#!/usr/bin/env python3
"""Inspect GGUF model tensors: shapes, parameter counts, and types."""

import sys
from gguf import GGUFReader

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3-4b-instruct-2507-Q8_0.gguf"

reader = GGUFReader(model_path)

# Print metadata
print("=" * 70)
print(f"  Model: {model_path}")
print("=" * 70)
print(f"\n{'Key':<50} {'Value'}")
print("-" * 70)
for field in reader.fields.values():
    # Skip large array fields (tokens, merges, etc.)
    if len(field.data) > 10:
        continue
    name = field.name
    # Try to extract a readable value
    parts = field.parts
    if len(parts) > 1:
        val = parts[-1].tolist()
        if len(val) == 1:
            val = val[0]
    else:
        val = "..."
    print(f"{name:<50} {val}")

# Print tensors
print(f"\n{'=' * 70}")
print(f"  Tensors: {len(reader.tensors)}")
print("=" * 70)
print(f"\n{'#':<6} {'Name':<35} {'Shape':<20} {'Params':>12} {'Type':<10} {'Bytes':>12}")
print("-" * 100)

total_params = 0
for i, tensor in enumerate(reader.tensors):
    shape = tensor.shape.tolist()
    n_params = 1
    for d in shape:
        n_params *= d
    total_params += n_params
    print(f"{i:<6} {tensor.name:<35} {str(shape):<20} {n_params:>12,} {tensor.tensor_type.name:<10} {tensor.n_bytes:>12,}")

print("-" * 100)
print(f"{'Total parameters:':<63} {total_params:>12,}")
print(f"{'Total parameters (B):':<63} {total_params / 1e9:>12.2f}")
