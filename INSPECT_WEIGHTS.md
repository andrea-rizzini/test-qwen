# inspect_weights.py

Inspect individual weight values from GGUF model files. Supports F32 and Q8_0 quantized tensors.

## Requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install gguf numpy
```

## Usage

```
python3 inspect_weights.py <model_path> [tensor_name] [options]
```

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--search` | `-s` | | Filter tensor list by substring |
| `--offset` | `-o` | `0` | Start index into the weight array |
| `--count` | `-n` | `100` | Number of values to display (or blocks in `--raw` mode) |
| `--raw` | `-r` | off | Show raw Q8_0 blocks (scale + int8 values) instead of dequantized floats |

## Examples

### List all tensors in a model

```bash
python3 inspect_weights.py models/qwen3-4b-instruct-2507-Q8_0.gguf
```

Output:

```
#      Name                                Shape                      Params Type
-------------------------------------------------------------------------------------
0      output_norm.weight                  [2560]                      2,560 F32
1      token_embd.weight                   [2560, 151936]        388,956,160 Q8_0
2      blk.0.attn_k.weight                 [2560, 1024]            2,621,440 Q8_0
...
```

### Search tensors by name

```bash
python3 inspect_weights.py models/qwen3-4b-instruct-2507-Q8_0.gguf -s ffn_gate
```

### Inspect a specific tensor

```bash
python3 inspect_weights.py models/qwen3-4b-instruct-2507-Q8_0.gguf blk.0.attn_q.weight
```

Output:

```
Tensor:  blk.0.attn_q.weight
Shape:   [2560, 4096]
Params:  10,485,760
Type:    Q8_0
Bytes:   11,141,120

Values [0:100] of 10,485,760:
----------------------------------------------------------------------
  [       0]  -0.001341  -0.012517   0.008941   0.003874   0.001937  ...
  ...

Stats (full tensor):
  min:    -0.429722
  max:    0.519348
  mean:   0.000007
  std:    0.023281
  zeros:  193,099 / 24,903,680
```

### Inspect with offset and count

```bash
# Show 50 values starting at index 1000
python3 inspect_weights.py models/qwen3-4b-instruct-2507-Q8_0.gguf blk.0.ffn_gate.weight -o 1000 -n 50
```

### Show raw quantized data

Use `--raw` to see the raw int8 quantized values as stored on disk, in the same table format as the default mode.

```bash
python3 inspect_weights.py models/qwen3-4b-instruct-2507-Q8_0.gguf blk.0.attn_q.weight --raw
```

Output:

```
Raw int8 values [0:100] of 10,485,760:
----------------------------------------------------------------------
  [       0]   -9  -84   60   26   13   28  -24  -34    5  -50
  [      10]  -18   23   37  -38 -127    5  -46  -71  -66  -17
  [      20]   42   -8   20   20  -16   24    1   31  -81  -21
  ...

Stats (full tensor):
  min:    -127
  max:    127
  mean:   0.01
  std:    54.81
  zeros:  78,833 / 10,485,760
```

The `--offset` and `--count` flags work the same as in default mode (individual weight indices).

## Supported quantization types

| Type | Description |
|---|---|
| F32 | Full 32-bit float, values read directly |
| Q8_0 | 8-bit quantized (block size 32), dequantized to float32 on read |

## Tensor naming convention

Each transformer block `blk.N` contains:

| Tensor | Role |
|---|---|
| `attn_q.weight` | Attention query projection |
| `attn_k.weight` | Attention key projection |
| `attn_v.weight` | Attention value projection |
| `attn_output.weight` | Attention output projection |
| `attn_q_norm.weight` | Query normalization |
| `attn_k_norm.weight` | Key normalization |
| `attn_norm.weight` | Pre-attention RMSNorm |
| `ffn_gate.weight` | SwiGLU gate projection |
| `ffn_up.weight` | SwiGLU up projection |
| `ffn_down.weight` | FFN down projection |
| `ffn_norm.weight` | Pre-FFN RMSNorm |

Global tensors: `token_embd.weight` (embedding table), `output_norm.weight` (final RMSNorm).

---

# compare_weights.py

Fast byte-level comparison to find which tensors differ between two GGUF models.

## Usage

```
python3 compare_weights.py <model_a> <model_b>
```

## Example

```bash
python3 compare_weights.py models/qwen3-4b-instruct-2507-Q8_0.gguf models/qwen3-4b-instruct-2507-heretic-Q8_0.gguf
```

Output:

```
#      Tensor                                    Params Status
----------------------------------------------------------------------
104    blk.9.attn_output.weight              10,485,760 DIFFERENT
108    blk.9.ffn_down.weight                 24,903,680 DIFFERENT
...
----------------------------------------------------------------------

Identical: 344/398
Different: 54/398

Parameters in affected tensors: 955,514,880 / 4,022,468,096 (23.75%)
(Use diff_weights.py to see how many individual weights actually changed)
```

Only tensors that differ are printed. The `Params` column is the total number of weights in the tensor, not the number of changed weights.

---

# diff_weights.py

Detailed weight-by-weight comparison between two GGUF models. Dequantizes only the tensors that differ and shows how much each weight changed.

## Usage

```
python3 diff_weights.py <model_a> <model_b> [options]
```

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--threshold` | `-t` | `0.0` | Only count weights where abs difference > threshold |
| `--top` | | `20` | Show top N largest differences per tensor |

## Examples

```bash
# Default (top 20 diffs per tensor)
python3 diff_weights.py models/qwen3-4b-instruct-2507-Q8_0.gguf models/qwen3-4b-instruct-2507-heretic-Q8_0.gguf

# Top 5 per tensor
python3 diff_weights.py models/qwen3-4b-instruct-2507-Q8_0.gguf models/qwen3-4b-instruct-2507-heretic-Q8_0.gguf --top 5

# Only show weights that differ by more than 0.005
python3 diff_weights.py models/qwen3-4b-instruct-2507-Q8_0.gguf models/qwen3-4b-instruct-2507-heretic-Q8_0.gguf -t 0.005
```

Output per tensor:

```
================================================================================
  blk.9.attn_output.weight  |  shape: [4096, 2560]  |  type: Q8_0
================================================================================

  Changed weights: 6,839,111 / 10,485,760 (65.22%)
  Abs diff — mean: 0.00041380  max: 0.01091409

  Top 5 largest differences:
       Index       Model A       Model B          Diff      Abs Diff
  --------------------------------------------------------------
   1,404,319      0.059075      0.048161     -0.010914      0.010914
   1,403,542      0.027294      0.017509     -0.009785      0.009785
   ...
```
