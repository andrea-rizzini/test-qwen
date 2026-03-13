# Qwen3-4B Model Comparison

Side-by-side comparison of [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (official) and [Qwen3-4B-Instruct-2507-heretic](https://huggingface.co/p-e-w/Qwen3-4B-Instruct-2507-heretic) (abliterated) using [llama.cpp](https://github.com/ggml-org/llama.cpp).

The heretic variant has safety refusals removed via the [Heretic](https://github.com/p-e-w/heretic) abliteration tool, while keeping general model quality largely intact (KL divergence: 0.43).

Both models use Q8_0 quantization (GGUF) for a fair comparison.

## Prerequisites

- Linux (tested on Ubuntu)
- C++ build tools: `build-essential`, `cmake`
- ~10 GB disk space (two 4.3 GB model files + llama.cpp)
- At least 10 GB RAM (8 GB per model at inference time)

## Setup

### 1. Build llama.cpp

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
cd ..
```

### 2. Download models

```bash
mkdir -p models

# Official Qwen3-4B-Instruct-2507 (Q8_0, ~4.3 GB)
wget -O models/qwen3-4b-instruct-2507-Q8_0.gguf \
  "https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf"

# Heretic / abliterated variant (Q8_0, ~4.3 GB)
wget -O models/qwen3-4b-instruct-2507-heretic-Q8_0.gguf \
  "https://huggingface.co/bartowski/p-e-w_Qwen3-4B-Instruct-2507-heretic-GGUF/resolve/main/p-e-w_Qwen3-4B-Instruct-2507-heretic-Q8_0.gguf"
```

## Usage

### Run the comparison script

```bash
# Default prompt (recursion explanation)
./compare.sh

# Custom prompt
./compare.sh "Write a haiku about the sea"
```

The script sends the same prompt to both models and prints their responses sequentially, along with performance metrics (prompt processing speed and generation speed in tokens/second).

### Run a single model directly

```bash
./llama.cpp/build/bin/llama-cli \
  -m models/qwen3-4b-instruct-2507-Q8_0.gguf \
  -t $(nproc) \
  -c 4096 \
  -n 512 \
  --temp 0.7 --top-p 0.8 --top-k 20 \
  --single-turn -cnv \
  -p "Your prompt here"
```

## Configuration

These parameters can be adjusted at the top of `compare.sh`:

| Parameter | Default | Description |
|---|---|---|
| `THREADS` | `$(nproc)` | Number of CPU threads |
| `CTX_SIZE` | `4096` | Context window size (max 262144) |
| `N_PREDICT` | `512` | Maximum tokens to generate |
| `TEMP` | `0.7` | Sampling temperature |
| `TOP_P` | `0.8` | Top-p (nucleus) sampling |
| `TOP_K` | `20` | Top-k sampling |

Increase `N_PREDICT` for longer responses. Increase `CTX_SIZE` if you need to process longer inputs.

## Project structure

```
.
├── compare.sh          # Comparison script
├── llama.cpp/          # llama.cpp (built from source)
├── models/
│   ├── qwen3-4b-instruct-2507-Q8_0.gguf
│   └── qwen3-4b-instruct-2507-heretic-Q8_0.gguf
└── README.md
```
