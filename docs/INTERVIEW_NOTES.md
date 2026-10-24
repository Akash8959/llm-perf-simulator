# LLM Inference Performance & Sizing: Interview Study Guide

This guide breaks down LLM inference performance, roofline modeling, distributed scaling, and hardware bottlenecks in plain, intuitive language, referencing the exact formulas and architectural implementations in `llmsim`.

---

## 1. Roofline Model, Arithmetic Intensity, and Ridge Point

### What is the Roofline Model?
The Roofline Model is an analytical framework that visualizes whether a workload is limited by the processor's **compute throughput** (FLOP/s) or by the memory subsystem's **memory bandwidth** (bytes/s).

### Arithmetic Intensity ($I$)
Arithmetic Intensity measures how many floating-point operations (FLOPs) the GPU performs for every single byte of data transferred from High-Bandwidth Memory (HBM):
$$I = \frac{\text{Workload FLOPs}}{\text{Memory Bytes Transferred}} \quad \left[\frac{\text{FLOP}}{\text{byte}}\right]$$

### Hardware Ridge Point ($I^*$)
The ridge point is the hardware's tipping point—the exact arithmetic intensity where peak compute capability matches peak memory bandwidth:
$$I^* = \frac{\text{Peak Compute Throughput (FLOP/s)}}{\text{Memory Bandwidth (Bytes/s)}} \quad \left[\frac{\text{FLOP}}{\text{byte}}\right]$$
- **NVIDIA H100 SXM5 Example**:
  - Peak FP16 Tensor Core = 989 TFLOPs
  - HBM3 Bandwidth = 3,350 GB/s (3.35 TB/s)
  - Effective Ridge Point (with $\eta_{\text{comp}}=0.6, \eta_{\text{mem}}=0.8$):
    $$I^*_{\text{eff}} = \frac{989 \times 10^{12} \times 0.60}{3350 \times 10^9 \times 0.80} \approx \mathbf{221.4 \text{ FLOP/byte}}$$

### Bottleneck Classification Rule
- If $I \ge I^*$: **Compute-Bound**. The arithmetic units are fully saturated; adding more memory bandwidth will NOT improve latency.
- If $I < I^*$: **Memory-Bound**. The GPU cores spend most cycles idling while waiting for weights or KV tensors to arrive over the memory bus; adding faster compute will NOT help.

---

## 2. Why Prefill is Compute-Bound and Decode is Memory-Bound

### Prefill Phase (Time to First Token / TTFT)
- In prefill, the model evaluates the entire input prompt ($S$ tokens) all at once.
- Matrix operations are **GEMM (Matrix-Matrix multiplications)**:
  - Weight matrix shape: $[h, h]$
  - Activation matrix shape: $[B \cdot S, h]$
- For an $8\text{B}$ model with $S = 2048$ and $B = 1$:
  - We read each weight parameter once ($16 \text{ GB}$ of weights).
  - We perform $2 \cdot \text{params} \cdot S \approx 35 \text{ TFLOPs}$.
  - Arithmetic intensity is $35 \times 10^{12} / 17.4 \times 10^9 \approx \mathbf{2,016 \text{ FLOP/byte}}$.
  - Since $2016 \gg 221.4$, **prefill is heavily compute-bound**.

### Decode Phase (Time Per Output Token / TPOT)
- During auto-regressive generation, the model predicts exactly **one new token** per sequence step-by-step ($S = 1$).
- Matrix operations degrade to **GEMV (Matrix-Vector multiplications)**:
  - Weight matrix shape: $[h, h]$
  - Activation vector shape: $[B \cdot 1, h]$
- For batch size $B = 1$:
  - The GPU must still read the **entire 16 GB weight matrix** from HBM to compute just 1 token!
  - It performs only $2 \cdot \text{params} \cdot 1 \approx 16 \text{ GFLOPs}$.
  - Arithmetic intensity is $16 \times 10^9 / 16 \times 10^9 \approx \mathbf{1.0 \text{ FLOP/byte}}$.
  - Since $1.0 \ll 221.4$, **decode at small batch is intensely memory-bound**.

---

## 3. KV Cache Footprint Derivation

In standard multi-head self-attention, each token in the context generates a Key vector and a Value vector for every attention head in every layer. Without caching, the model would need to recompute activations for all prior $C$ tokens at every new step.

### Step-by-Step Derivation
1. **2 Vectors**: One Key vector and one Value vector per token $\implies 2$.
2. **Layers ($L$)**: Each of the $L$ transformer layers stores its own KV state $\implies 2 \cdot L$.
3. **KV Heads ($n_{kv}$)**: The number of Key/Value heads per layer $\implies 2 \cdot L \cdot n_{kv}$.
4. **Head Dimension ($d$)**: Each head has hidden dimension $d = h / n_h \implies 2 \cdot L \cdot n_{kv} \cdot d$.
5. **Context Length ($C$)**: The sequence history consists of $C$ tokens $\implies 2 \cdot L \cdot n_{kv} \cdot d \cdot C$.
6. **Batch Size ($B$)**: Concurrent sequences served $\implies 2 \cdot L \cdot n_{kv} \cdot d \cdot C \cdot B$.
7. **Bytes Per Element ($b_{kv}$)**: Data type size (2 bytes for FP16, 1 byte for FP8/INT8):
$$\text{KV Cache Bytes} = 2 \cdot L \cdot n_{kv} \cdot d \cdot C \cdot B \cdot b_{kv}$$

### Numerical Example (LLaMA-3-8B)
- $L = 32, n_{kv} = 8, d = 128, C = 2048, B = 1, b_{kv} = 2$:
$$\text{KV Bytes} = 2 \times 32 \times 8 \times 128 \times 2048 \times 1 \times 2 = \mathbf{268,435,456 \text{ bytes (256 MB)}}$$

---

## 4. GQA and MoE: Architectural Impacts on Memory

### Grouped-Query Attention (GQA)
- In standard Multi-Head Attention (MHA), $n_{kv} = n_h$ (e.g., 32 query heads have 32 KV heads).
- In GQA, multiple query heads share a single Key/Value head (e.g. LLaMA-3-8B has $n_h = 32$ but $n_{kv} = 8$, a $4:1$ ratio).
- **Impact**: GQA slashes the KV cache memory footprint by $\frac{n_h}{n_{kv}}$ ($4\times$ reduction for LLaMA-3). This directly allows $4\times$ larger batch sizes or $4\times$ longer context windows before hitting OOM.

### Mixture of Experts (MoE)
- An MoE replaces the dense FFN with $E$ smaller experts, routing each token to only top-$k$ experts ($k \ll E$).
- **Memory Footprint**: All $E$ experts must be resident in GPU memory! Total parameter memory is large:
  $$\text{Total FFN Params} = h \cdot E + E \cdot (3 \cdot h \cdot d_{ffn})$$
- **Compute (FLOPs)**: Only $k$ routed experts are active per token:
  $$\text{Active FFN Params} = h \cdot E + k \cdot (3 \cdot h \cdot d_{ffn})$$
- **Decode Memory Traffic across Batch**:
  In batch decode, different sequences may choose different experts. The expected number of unique experts loaded per step is:
  $$E_{\text{hit}} = E \cdot \left(1 - \left(1 - \frac{k}{E}\right)^B\right)$$
  - For $B = 1$: $E_{\text{hit}} = k$ (only $k$ experts read).
  - As $B$ grows, $E_{\text{hit}} \to E$ (eventually all experts are read, increasing memory traffic with batch size).

---

## 5. How Quantization Shifts the Roofline Bound

### Weight Quantization (e.g., FP16 $\to$ INT8 / INT4)
- Cuts weight parameter bytes from 2.0 bytes to 1.0 byte (INT8) or 0.5 bytes (INT4).
- **In Decode**: Because decode is memory-bound, latency scales with bytes moved ($\text{Bytes} = \text{Weights Read} + \text{KV Cache Read}$). INT8 weights alone cut weight bytes by half, but the KV cache still moves at FP16 ($b_{kv} = 2.0\text{ bytes}$), so the gain is slightly under 50% (~48.3% at $B=4, C=1024$) when the KV cache is not quantized; quantizing both weights and KV cache gives exactly 50% reduction in total bytes and latency.
- **In Prefill**: Because prefill is compute-bound, cutting weight bytes transferred has negligible effect on latency unless the GPU hardware supports higher INT8/INT4 Tensor Core TFLOPs.

### KV Cache Quantization (e.g., FP16 $\to$ FP8 / INT8)
- Halves the KV cache memory footprint ($b_{kv}: 2.0 \to 1.0$).
- At long context lengths ($C \ge 8192$), KV cache dominates GPU memory. Quantizing the KV cache doubles the maximum batch size $B_{\max}$ that fits in memory without triggering OOM.

---

## 6. Tensor Parallelism and the Ring All-Reduce Cost

### How Tensor Parallelism (TP) Works
- TP splits weight matrices along row or column dimensions across $N_{\text{TP}}$ GPUs (Megatron-LM style).
- Attention: $W_q, W_k, W_v$ are column-parallel; $W_o$ is row-parallel.
- FFN: Gate and Up projections are column-parallel; Down projection is row-parallel.
- **Communication Required**: Two All-Reduce operations per transformer layer (one after Attention, one after FFN).

### Ring All-Reduce Latency Formula
$$t_{\text{ring}} = 2 \cdot \frac{N_{\text{TP}} - 1}{N_{\text{TP}}} \cdot \frac{M_{\text{msg}}}{\text{BW}_{\text{link}}} + (N_{\text{TP}} - 1) \cdot \tau_{\text{link}}$$
where message size is $M_{\text{msg}} = B \cdot \text{tokens} \cdot h \cdot b_{\text{act}}$.
- Total communication per layer: $2 \cdot t_{\text{ring}}$
- Total communication across $L$ layers: $2 \cdot L \cdot t_{\text{ring}}$

### Why NVLink vs PCIe Matters
- On NVLink 4 ($900 \text{ GB/s}$ bidirectional, $\tau \approx 2 \mu\text{s}$), TP all-reduce takes $<5\%$ of total execution latency.
- On PCIe Gen5 ($64-100 \text{ GB/s}$ bidirectional, $\tau \approx 10-15 \mu\text{s}$), TP all-reduce takes $15-30\%$ of total latency, drastically reducing parallel efficiency.

---

## 7. Assumptions and What the Analytical Model Ignores

`llmsim` is an analytical first-principles simulator. In production systems (e.g. vLLM, TensorRT-LLM, TGI), the following real-world effects occur:
1. **CUDA Kernel Launch Overhead**: Real drivers incur $3-10 \mu\text{s}$ launch overhead per kernel. In deep models with hundreds of small kernels, this adds measurable host-side latency.
2. **Compute-Communication Overlap**: Advanced engines overlap TP All-Reduce or PP P2P communication with independent layer activations. `llmsim` models the pessimistic sequential upper bound.
3. **PagedAttention & Fragmentation**: KV cache memory is modeled as perfectly contiguous without external fragmentation, whereas page tables introduce small indexing overheads.
4. **Continuous Batching & Queueing**: In multi-tenant serving, requests arrive continuously according to Poisson distributions; actual queueing delays are governed by M/M/1 or M/G/1 queueing dynamics.

---

## 8. 15 Likely Interview Questions & Concise Answers

### Q1: What is the arithmetic intensity of LLM decoding at batch size 1?
**Answer:** Approximately $1.0 \text{ FLOP/byte}$. In auto-regressive decode, we read the entire model weight matrix ($~2 \text{ bytes/param}$ in FP16) to generate exactly one token, performing $2 \text{ FLOPs/param}$. $2 \text{ FLOPs} / 2 \text{ bytes} \approx 1.0 \text{ FLOP/byte}$.

### Q2: Why does increasing batch size improve LLM generation throughput?
**Answer:** At small batch sizes, decode is memory bandwidth-bound because weights are re-read for every single token. Increasing batch size amortizes the cost of loading the weights across $B$ tokens simultaneously, increasing arithmetic intensity from $1 \text{ FLOP/byte}$ towards the hardware ridge point.

### Q3: What is the hardware ridge point of an NVIDIA H100 GPU?
**Answer:** The effective ridge point is ~221 FLOP/byte (calculated as peak FP16 TFLOPs $\times 0.60$ divided by HBM bandwidth $\times 0.80$). Workloads with intensity below 221 are memory-bound; workloads above 221 are compute-bound.

### Q4: How do you derive the KV cache memory formula?
**Answer:** Each token requires storing both a Key and Value vector across $L$ layers for $n_{kv}$ heads of dimension $d$ with precision $b_{kv}$:
$\text{KV Cache Bytes} = 2 \cdot L \cdot n_{kv} \cdot d \cdot C \cdot B \cdot b_{kv}$.

### Q5: How does Grouped-Query Attention (GQA) benefit memory footprint?
**Answer:** In GQA, multiple query heads share a single Key/Value head ($n_{kv} < n_h$). For LLaMA-3 (32 Q heads, 8 KV heads), this cuts KV cache memory by $4\times$, allowing $4\times$ longer context windows or larger batch sizes before hitting OOM.

### Q6: If decode is memory-bound, will upgrading from H100 to an accelerator with 2x more TFLOPs speed up single-user latency?
**Answer:** No. Because arithmetic intensity is $~1.0 \text{ FLOP/byte} \ll 221 \text{ FLOP/byte}$, execution is bottlenecked entirely by HBM read bandwidth. Upgrading compute FLOPs yields $<1\%$ improvement; only increasing HBM bandwidth or quantizing weights will improve TPOT.

### Q7: Why does INT8 weight quantization yield slightly under 50% decode speedup when the KV cache is not quantized, and what gives exactly 50%?
**Answer:** Decode is memory-bound, so latency scales with bytes moved from HBM:
$$t_{\text{decode}} \approx \frac{\text{Weights Read} + \text{KV Cache Read}}{\text{HBM Bandwidth} \times \eta_{\text{memory}}}$$
INT8 weights alone cut weight bytes by half, but the KV cache still moves at FP16 ($b_{kv} = 2.0\text{ bytes}$). Because this unquantized KV cache traffic remains on the memory bus, the gain is slightly under 50% when the KV cache is not quantized (e.g. 48.3% lower TPOT at $B=4, C=1024$). Quantizing both weights and the KV cache gives exactly 50% latency reduction.

### Q8: Does INT8 weight quantization significantly reduce prefill latency?
**Answer:** No. Prefill is heavily compute-bound (arithmetic intensity $>1500 \text{ FLOP/byte}$). The time spent transferring weights from HBM is negligible compared to Tensor Core GEMM compute time. INT8 only speeds up prefill if the GPU tensor cores execute INT8 operations faster.

### Q9: Why does MoE decode memory traffic scale with batch size?
**Answer:** In MoE, each token is routed to top-$k$ of $E$ experts. At batch 1, only $k$ experts are read. As batch size increases, different sequences select different experts. The expected number of distinct experts read per step is $E_{\text{hit}} = E \cdot (1 - (1 - k/E)^B)$, which approaches all $E$ experts as $B$ grows.

### Q10: How many All-Reduce operations are required per transformer layer in Tensor Parallelism?
**Answer:** Exactly 2 All-Reduce operations per layer: one after the Attention output projection ($W_o$), and one after the FFN down projection ($W_{\text{down}}$).

### Q11: What is the communication latency formula for Ring All-Reduce?
**Answer:** $t_{\text{ring}} = 2 \cdot \frac{N-1}{N} \cdot \frac{M}{\text{BW}} + (N-1) \cdot \tau$, where $N$ is the number of GPUs, $M$ is message size, $\text{BW}$ is interconnect link bandwidth, and $\tau$ is link latency.

### Q12: Why is Tensor Parallelism rarely run across separate server nodes over standard Ethernet?
**Answer:** TP requires frequent All-Reduces ($2 \cdot L$ per token) with low latency. Standard Ethernet (100 Gbps, $10-50 \mu\text{s}$ latency) incurs catastrophic communication overhead. TP requires ultra-high bandwidth, low-latency interconnects like NVLink (900 GB/s, $1-2 \mu\text{s}$ latency).

### Q13: What is the difference between Time to First Token (TTFT) and Time Per Output Token (TPOT)?
**Answer:** TTFT measures the latency of the prefill phase (processing the input prompt prompt_len in a single compute-bound step). TPOT measures the inter-token latency of generating subsequent tokens one-by-one in sequential, memory-bound decode steps.

### Q14: How does Pipeline Parallelism (PP) introduce latency overhead?
**Answer:** PP partitions layers across sequential stages, introducing a pipeline bubble (idle GPU time while waiting for microbatches to fill and drain). The bubble fraction is $F_{\text{bubble}} = \frac{N_{\text{PP}} - 1}{N_{\text{PP}} - 1 + M_{\text{micro}}}$.

### Q15: How do you calculate the maximum supportable batch size before Out-of-Memory (OOM)?
**Answer:**
$$B_{\max} = \left\lfloor \frac{\text{GPU Capacity} - \text{Sharded Weight Memory}}{\text{KV Cache Bytes Per Sequence}} \right\rfloor$$
where $\text{KV Cache Bytes Per Sequence} = 2 \cdot L \cdot n_{kv} \cdot d \cdot C \cdot b_{kv} / N_{\text{TP}}$.
