# llm-perf-simulator

An analytical performance simulator for LLM inference on GPUs. No GPU needed.

[![CI](https://github.com/Akash8959/llm-perf-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/Akash8959/llm-perf-simulator/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why I built this

I built this project to understand why LLM inference is slow and what actually limits its speed. When running large language models, the bottleneck changes completely between processing the input prompt and generating output tokens. I wanted a way to calculate and estimate performance across different GPUs, model architectures, and batch sizes without needing access to a real GPU cluster. This is an analytical model based on roofline analysis, not a benchmark run on physical hardware.

## What it does

- Estimates Time to First Token (TTFT), Time Per Output Token (TPOT), and total throughput for prefill and decode phases.
- Tells whether each phase is compute-bound or memory-bound.
- Models KV cache memory growth and checks if weights and context fit in GPU memory without running out of memory.
- Supports precision formats including FP16, INT8, and INT4 for weights and KV cache.
- Models tensor, pipeline, expert, and data parallelism along with interconnect communication overhead.
- Runs parameter sweeps and saves plots and CSV files to compare hardware configurations.

## Quick start

### Installation

Clone the repository and install the dependencies in a virtual environment:

```bash
git clone https://github.com/Akash8959/llm-perf-simulator.git
cd llm-perf-simulator
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt -e .
```

### Run a simulation

Run a single simulation from the command line:

```bash
python -m llmsim.cli \
  --hardware configs/hardware/h100_sxm.yaml \
  --model configs/models/llama3_8b.yaml \
  --batch 1 \
  --prompt-len 2048 \
  --output-len 128
```

Sample CLI output:

```text
==============================================================================
LLM PERFORMANCE SIMULATOR: LLAMA3_8B ON H100_SXM
==============================================================================
Metric                         | Value     | Unit / Note       
-------------------------------+-----------+-------------------
Time to First Token (TTFT)     | 59.13     | ms                
Time Per Output Token (TPOT)   | 6.10      | ms / token        
End-to-End Latency             | 0.839     | seconds           
Generation Throughput          | 152.5     | tokens / sec      
Peak Memory Per GPU            | 15.22     | GB                
Prefill Bottleneck Regime      | COMPUTE   | compute / memory  
Decode Bottleneck Regime       | MEMORY    | compute / memory  
Inter-GPU Comm Overhead        | 0.0%      | of e2e latency    
Memory Headroom Utilization    | 19.0%     | of capacity       
Out-of-Memory (OOM) Status     | FIT       | Status            
```

### Run all sweeps

To run all parameter sweeps and generate the CSV files and plots in `results/`:

```bash
python examples/run_all_sweeps.py
```

## Results

Here are the key numbers from running the simulator for LLaMA-3-8B on an NVIDIA H100 SXM5 GPU:

| Metric / Scenario | Value | Bottleneck / Note |
| :--- | :--- | :--- |
| Hardware ridge point | 221.4 FLOP/byte | Threshold between memory-bound and compute-bound |
| Prefill arithmetic intensity (batch 1, prompt 2048) | 2016.4 FLOP/byte | Compute-bound (well above 221.4) |
| Decode arithmetic intensity (batch 1, context 2048) | 1.05 FLOP/byte | Memory-bound (well below 221.4) |
| Decode TPOT at baseline (batch 1, context 2048) | 6.09 ms / token | Limited by 3350 GB/s HBM bandwidth |
| Decode TPOT with 2x HBM bandwidth | 3.05 ms / token | 2.0x speedup (50.0% lower latency) |
| INT8 vs FP16 decode latency (batch 4, context 1024, weights only) | 3.21 ms vs 6.21 ms | 48.3% lower latency (1.93x speedup, KV cache stays FP16) |
| INT8 vs FP16 decode latency (batch 4, context 1024, weights + KV cache) | 3.10 ms vs 6.21 ms | 50.0% lower latency (2.0x speedup, both weights and KV in INT8) |

### Visualizations

![Roofline Chart](results/roofline_chart.png)
The roofline curve shows that prefill sits on the compute ceiling while decode sits down on the memory bandwidth slope.

![TPOT vs HBM](results/tpot_vs_hbm.png)
Decode latency drops almost linearly as memory bandwidth increases because reading weights from memory is the main bottleneck.

![Latency vs Batch Size](results/latency_vs_batch.png)
Prefill latency increases with batch size, while decode latency per token stays flat until batch size grows large enough to saturate compute.

![Throughput vs TP](results/throughput_vs_tp.png)
Tensor parallelism scales well on high-bandwidth NVLink, but communication overhead hurts scaling on slower PCIe interconnects.

![Bottleneck Summary](results/bottleneck_summary.png)
Summary comparison across different serving configurations, showing where execution time is spent.

## How the model works

### 1. Parameter count
I calculate the attention and feed-forward parameters for each transformer layer based on hidden size, head dimensions, and intermediate dimension. For SwiGLU, the feed-forward network has three weight matrices (gate, up, and down projections).

```text
attention_params = (2 * hidden * hidden) + (2 * hidden * kv_heads * head_dim)
ffn_params = 3 * hidden * ffn_dim
layer_params = attention_params + ffn_params
total_params = embedding_params + (layers * layer_params)
```

For Mixture of Experts (MoE) models, the router has `hidden * num_experts` parameters, and only `top_k` routed experts plus shared experts are active per token.

### 2. KV cache size
Every generated token appends a Key vector and a Value vector to the cache for every layer and Key-Value head. The total size grows with batch size and context length.

```text
kv_cache_bytes = 2 * layers * kv_heads * head_dim * context * batch * bytes_per_element
```

For LLaMA-3-8B in FP16 with batch 1 and context 2048, the KV cache uses 256 MB.

### 3. Prefill phase (FLOPs and bytes)
During prefill, the model processes the entire prompt sequence at once using matrix-matrix multiplications (GEMM). Because the GPU reuses model weights across all prompt tokens, prefill has high arithmetic intensity and is compute-bound.

```text
prefill_flops = (2 * active_params * batch * prompt_len) + (4 * layers * batch * prompt_len^2 * hidden)
prefill_bytes = (active_params * bytes_per_weight) + (2 * layers * batch * prompt_len * hidden * bytes_per_act) + kv_write_bytes
```

### 4. Decode phase (FLOPs and bytes)
During decode, the model generates one token at a time using matrix-vector multiplications (GEMV). Because the GPU must read all model weights from memory just to generate a single token, decode has low arithmetic intensity and is memory-bound.

```text
decode_flops = (2 * active_params * batch) + (4 * layers * batch * context * hidden)
decode_bytes = weights_read + kv_cache_read
```

For MoE models during batched decode, the expected number of distinct experts loaded per step is modeled as:

```text
distinct_experts = num_experts * (1 - (1 - top_k / num_experts)^batch)
```

### 5. Latency calculation
Execution time is determined by whichever takes longer: compute time or memory transfer time, plus communication time if using multiple GPUs.

```text
compute_time = flops / (peak_flops * compute_efficiency)
memory_time = bytes / (hbm_bandwidth * memory_efficiency)
latency = max(compute_time, memory_time) + communication_time
```

I use default efficiency factors of 0.60 for compute and 0.80 for memory bandwidth based on typical sustained GPU utilization in inference.

### 6. Arithmetic intensity and ridge point
Arithmetic intensity measures how many floating-point operations the GPU performs for every byte read from memory. The ridge point marks where the hardware transitions from memory-bound to compute-bound.

```text
arithmetic_intensity = flops / bytes
ridge_point = (peak_flops * compute_efficiency) / (hbm_bandwidth * memory_efficiency)
```

If arithmetic intensity is below the ridge point, execution is memory-bound. If it is above, execution is compute-bound.

### 7. Tensor parallel communication
Tensor parallelism shards weight matrices across multiple GPUs, which requires two ring all-reduce operations per transformer layer for each token.

```text
allreduce_time = 2 * ((tp_size - 1) / tp_size) * (message_bytes / link_bandwidth) + (tp_size - 1) * link_latency
comm_time = 2 * layers * allreduce_time
```

## Project structure

```text
.
├── .github/workflows/ci.yml    # GitHub Actions workflow for tests and linting
├── .gitignore                  # Git ignore rules for virtualenvs and caches
├── LICENSE                     # MIT license
├── README.md                   # Project documentation
├── pyproject.toml              # Build setup, dependencies, pytest and ruff config
├── requirements.txt            # Python dependencies
├── configs/
│   ├── hardware/               # GPU hardware specs (SMs, TFLOPs, HBM, NVLink)
│   │   ├── a100_80gb.yaml
│   │   └── h100_sxm.yaml
│   └── models/                 # Transformer architecture specs
│       ├── deepseek_moe.yaml
│       ├── llama3_70b.yaml
│       ├── llama3_8b.yaml
│       ├── mixtral_8x7b.yaml
│       └── qwen2_7b.yaml
├── examples/
│   └── run_all_sweeps.py       # Script that runs all sweeps and generates plots
├── results/                    # Saved plots and CSV sweep data
│   ├── bottleneck_summary.png
│   ├── latency_vs_batch.png
│   ├── roofline_chart.png
│   ├── throughput_vs_tp.png
│   ├── tpot_vs_hbm.png
│   └── sweep_*.csv
├── src/llmsim/                 # Core simulator package
│   ├── __init__.py             # Package init and version
│   ├── cli.py                  # Command-line interface
│   ├── estimator.py            # FLOPs, memory traffic, and roofline formulas
│   ├── hardware.py             # GPU hardware configuration loader and dataclass
│   ├── model.py                # Model architecture loader and parameter counts
│   ├── parallelism.py          # Parallelism configs and communication models
│   ├── plots.py                # Matplotlib visualization routines
│   ├── quantization.py         # Precision types and byte sizes (FP16, INT8, INT4)
│   ├── recommend.py            # Bottleneck diagnosis and summary generator
│   ├── simulator.py            # End-to-end simulation runner across prefill and decode
│   └── sweep.py                # Parameter sweep functions returning DataFrames
└── tests/                      # Automated test suite
    ├── test_cli.py
    ├── test_configs.py
    ├── test_estimator.py
    ├── test_parallelism.py
    ├── test_simulator.py
    └── test_sweeps.py
```

## Adding a GPU or model

### Adding a GPU
Create a new YAML file in `configs/hardware/<gpu_name>.yaml`:

```yaml
name: "custom_gpu"
sm_count: 128
peak_tflops_fp16: 800.0
peak_tflops_bf16: 800.0
peak_tflops_fp8: 1600.0
peak_tflops_int8: 1600.0
hbm_bandwidth_gbps: 3000.0
memory_capacity_gb: 96.0
l2_cache_mb: 64.0
interconnect_bandwidth_gbps: 800.0
interconnect_latency_us: 2.5
```

### Adding a model
Create a new YAML file in `configs/models/<model_name>.yaml`:

```yaml
name: "custom_model"
layers: 32
hidden: 4096
heads: 32
kv_heads: 8           # Set smaller than heads for GQA
ffn_dim: 14336
vocab: 128256
num_experts: 1        # Set > 1 for MoE
top_k: 1              # Active routed experts per token
shared_experts: 0     # Shared always-active experts
```

## Limitations

- It is an analytical estimate, not a measurement from physical hardware.
- The hardware efficiency factors (0.60 for compute, 0.80 for memory bandwidth) are assumed averages. Real workloads vary with kernel tuning and sequence lengths.
- It ignores CUDA kernel launch overheads (typically 3 to 10 microseconds per kernel), which can add latency for small models or short operations.
- It does not model request scheduling or continuous batching (iteration-level scheduling).
- It assumes sequential execution where communication adds linearly to compute or memory time, ignoring potential overlap between compute and communication.
- It counts the full embedding table in FLOPs and memory traffic even though an embedding lookup only needs a small slice of that table, which slightly overstates the numbers.
- I have not validated these estimates against real GPU measurements yet.

## What I want to add next

- Validate the predictions against real measurements from serving engines like vLLM or Ollama.
- Add a simulation model for continuous batching and dynamic request arrivals.
- Support newer low-precision formats like FP8 and FP4 for Blackwell and Hopper architectures.

## License

This project is licensed under the MIT License.
