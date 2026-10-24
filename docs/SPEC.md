# LLM Performance Simulator (llmsim) Specification

## 1. Overview
`llmsim` is an analytical, roofline-style LLM inference performance simulator. It models the latency, memory footprint, arithmetic intensity, and hardware utilization of Transformer-based Large Language Models during both prefill (time-to-first-token / TTFT) and decode (inter-token latency / TPOT) phases across various hardware configurations and parallelism strategies.

---

## 2. Notation and Definitions

| Symbol | Description | Unit |
| :--- | :--- | :--- |
| $L$ | Number of transformer layers | scalar |
| $h$ | Hidden dimension | scalar |
| $n_h$ | Number of query attention heads | scalar |
| $n_{kv}$ | Number of key/value attention heads | scalar |
| $d$ | Head dimension ($d = h / n_h$) | scalar |
| $d_{ffn}$ | Feed-forward intermediate dimension | scalar |
| $V$ | Vocabulary size | scalar |
| $E$ | Total number of routed experts in MoE | scalar |
| $k$ | Top-$k$ active experts per token | scalar |
| $E_{shared}$ | Number of shared experts in MoE | scalar |
| $B$ | Batch size | sequences |
| $S$ | Input prompt length (sequence length for prefill) | tokens |
| $C$ | Current context length during decode ($C \ge S$) | tokens |
| $b_w$ | Bytes per model weight parameter | bytes |
| $b_{kv}$ | Bytes per KV cache element | bytes |
| $b_{act}$ | Bytes per activation element | bytes |

---

## 3. Parameter Counts

### 3.1 Attention Parameters per Layer
For standard MHA / GQA / MVA:
- Query projection: $W_q \in \mathbb{R}^{h \times (n_h \cdot d)} = \mathbb{R}^{h \times h} \implies h \cdot h$
- Key projection: $W_k \in \mathbb{R}^{h \times (n_{kv} \cdot d)} \implies h \cdot n_{kv} \cdot d$
- Value projection: $W_v \in \mathbb{R}^{h \times (n_{kv} \cdot d)} \implies h \cdot n_{kv} \cdot d$
- Output projection: $W_o \in \mathbb{R}^{(n_h \cdot d) \times h} = \mathbb{R}^{h \times h} \implies h \cdot h$

$$\text{Params}_{\text{attn}} = 2 \cdot h^2 + 2 \cdot h \cdot n_{kv} \cdot d$$

### 3.2 Feed-Forward Network (FFN) Parameters per Layer
Assuming SwiGLU architecture (gate, up, and down projections):
- Dense FFN:
  $$\text{Params}_{\text{ffn}} = 3 \cdot h \cdot d_{ffn}$$
- Mixture of Experts (MoE):
  - Router / Gate: $h \cdot E$
  - Routed Experts: $E \cdot (3 \cdot h \cdot d_{ffn})$
  - Shared Experts: $E_{shared} \cdot (3 \cdot h \cdot d_{ffn})$
  - Active FFN params per token:
    $$\text{Active Params}_{\text{ffn}} = h \cdot E + (k + E_{shared}) \cdot (3 \cdot h \cdot d_{ffn})$$

### 3.3 Embeddings and LM Head
- Token Embedding: $V \cdot h$
- Output LM Head: $V \cdot h$
- Total: $2 \cdot V \cdot h$

### 3.4 Total & Active Model Parameters
- Per-layer parameters:
  $$\text{Params}_{\text{layer}} = \text{Params}_{\text{attn}} + \text{Params}_{\text{ffn}}$$
- Total Parameters:
  $$\text{Total Params} = 2 \cdot V \cdot h + L \cdot \text{Params}_{\text{layer}}$$
- Active Parameters (per token):
  $$\text{Active Params} = 2 \cdot V \cdot h + L \cdot \left(\text{Params}_{\text{attn}} + \text{Active Params}_{\text{ffn}}\right)$$

---

## 4. Memory Footprint and Limits

### 4.1 Weight Memory
$$\text{Memory}_{\text{weights}} = \text{Total Params} \cdot b_w$$

### 4.2 KV Cache Memory
Each token stores Key and Value vectors for all $L$ layers across $n_{kv}$ heads of dimension $d$:
$$\text{KV Cache Bytes} = 2 \cdot L \cdot n_{kv} \cdot d \cdot C \cdot B \cdot b_{kv}$$

### 4.3 Memory Fit Check
For a single GPU or sharded system with capacity $M_{\text{GPU}}$:
$$\text{Total Memory per GPU} = \frac{\text{Memory}_{\text{weights}}}{N_{\text{TP}}} + \frac{\text{KV Cache Bytes}}{N_{\text{TP}}}$$
- Out of Memory (OOM) occurs if $\text{Total Memory per GPU} > M_{\text{GPU}}$.
- Max batch size at context $C$:
  $$B_{\max} = \left\lfloor \frac{\left(M_{\text{GPU}} - \frac{\text{Memory}_{\text{weights}}}{N_{\text{TP}}}\right) \cdot N_{\text{TP}}}{2 \cdot L \cdot n_{kv} \cdot d \cdot C \cdot b_{kv}} \right\rfloor$$
- Max context length at batch $B$:
  $$C_{\max} = \left\lfloor \frac{\left(M_{\text{GPU}} - \frac{\text{Memory}_{\text{weights}}}{N_{\text{TP}}}\right) \cdot N_{\text{TP}}}{2 \cdot L \cdot n_{kv} \cdot d \cdot B \cdot b_{kv}} \right\rfloor$$

---

## 5. Computational and Memory Traffic Roofline Model

### 5.1 Prefill Phase
- **FLOPs**:
  $$\text{FLOPs}_{\text{prefill}} = 2 \cdot \text{Active Params} \cdot B \cdot S + 4 \cdot L \cdot B \cdot S^2 \cdot h$$
  - GEMM projections: $2 \cdot \text{Active Params} \cdot B \cdot S$
  - Self-attention ($Q K^T$ and $A V$): $4 \cdot L \cdot B \cdot S^2 \cdot h$
- **Memory Bytes Transferred**:
  $$\text{Bytes}_{\text{prefill}} = \text{Weights Read} + \text{Activations} + \text{KV Cache Write}$$
  - $\text{Weights Read} = \text{Active Params} \cdot b_w$ (read once per prefill phase)
  - $\text{Activations} = 2 \cdot B \cdot S \cdot h \cdot b_{act} \cdot L$ (layer transitions)
  - $\text{KV Cache Write} = 2 \cdot L \cdot n_{kv} \cdot d \cdot S \cdot B \cdot b_{kv}$

### 5.2 Decode Phase (Per Generated Token)
- **FLOPs** (for 1 step, generating 1 token per sequence):
  $$\text{FLOPs}_{\text{decode}} = 2 \cdot \text{Active Params} \cdot B + 4 \cdot L \cdot B \cdot C \cdot h$$
- **Memory Bytes Transferred**:
  $$\text{Bytes}_{\text{decode}} = \text{Weights Read} + \text{KV Cache Read} + \text{KV Cache Write}$$
  - For dense models: $\text{Weights Read} = \text{Active Params} \cdot b_w$
  - For MoE models across batch $B$:
    $$E_{\text{hit}} = E \cdot \left(1 - \left(1 - \frac{k}{E}\right)^B\right)$$
    $$\text{Weights Read} = \left(2 \cdot V \cdot h + L \cdot \left(\text{Params}_{\text{attn}} + h \cdot E + (E_{\text{hit}} + E_{shared}) \cdot 3 \cdot h \cdot d_{ffn}\right)\right) \cdot b_w$$
  - $\text{KV Cache Read} = 2 \cdot L \cdot n_{kv} \cdot d \cdot C \cdot B \cdot b_{kv}$
  - $\text{KV Cache Write} = 2 \cdot L \cdot n_{kv} \cdot d \cdot 1 \cdot B \cdot b_{kv}$

### 5.3 Roofline Latency and Bottleneck Analysis
- **Compute Time**:
  $$t_{\text{compute}} = \frac{\text{FLOPs}}{\text{Peak FLOPs} \cdot \eta_{\text{compute}}}$$
  (default $\eta_{\text{compute}} = 0.60$)
- **Memory Time**:
  $$t_{\text{memory}} = \frac{\text{Bytes}}{\text{HBM Bandwidth} \cdot \eta_{\text{memory}}}$$
  (default $\eta_{\text{memory}} = 0.80$)
- **Execution Latency**:
  $$t_{\text{execution}} = \max(t_{\text{compute}}, t_{\text{memory}})$$
- **Arithmetic Intensity**:
  $$I = \frac{\text{FLOPs}}{\text{Bytes}} \quad [\text{FLOP/byte}]$$
- **Ridge Point**:
  $$I^* = \frac{\text{Peak FLOPs} \cdot \eta_{\text{compute}}}{\text{HBM Bandwidth} \cdot \eta_{\text{memory}}} \quad [\text{FLOP/byte}]$$
- **Classification**:
  - If $I \ge I^*$ (or $t_{\text{compute}} \ge t_{\text{memory}}$): **Compute-bound**
  - If $I < I^*$ (or $t_{\text{compute}} < t_{\text{memory}}$): **Memory-bound**

---

## 6. Parallelism Strategies

### 6.1 Tensor Parallelism (TP)
- Shards attention projection weights ($W_q, W_k, W_v$ column-parallel, $W_o$ row-parallel) and FFN weights across $N_{\text{TP}}$ GPUs.
- Requires 2 All-Reduce operations per layer (one after Attention, one after FFN).
- Ring All-Reduce communication time:
  $$t_{\text{ring}} = 2 \cdot \frac{N_{\text{TP}} - 1}{N_{\text{TP}}} \cdot \frac{M_{\text{msg}}}{\text{BW}_{\text{link}}} + (N_{\text{TP}} - 1) \cdot \tau_{\text{link}}$$
  where:
  - $M_{\text{msg}} = B \cdot \text{tokens} \cdot h \cdot b_{\text{act}}$
  - For prefill: $\text{tokens} = S$
  - For decode: $\text{tokens} = 1$
  - Per layer: $2 \cdot t_{\text{ring}}$; Total per phase: $2 \cdot L \cdot t_{\text{ring}}$

### 6.2 Expert Parallelism (EP)
- Distributes MoE experts across $N_{\text{EP}}$ GPUs.
- Requires All-to-All communication to route token representations to corresponding expert devices:
  $$t_{\text{all\_to\_all}} = \frac{N_{\text{EP}} - 1}{N_{\text{EP}}} \cdot \frac{B \cdot \text{tokens} \cdot k \cdot h \cdot b_{\text{act}}}{\text{BW}_{\text{link}}} + (N_{\text{EP}} - 1) \cdot \tau_{\text{link}}$$
  (2 transfers: dispatch and combine).

### 6.3 Pipeline Parallelism (PP)
- Partitions layers across $N_{\text{PP}}$ pipeline stages.
- Microbatch bubble overhead factor:
  $$F_{\text{bubble}} = \frac{N_{\text{PP}} - 1}{N_{\text{PP}} - 1 + M_{\text{micro}}}$$

### 6.4 Data Parallelism (DP)
- Replicates model instances across $N_{\text{DP}}$ workers.
- Effective batch size scales as $B_{\text{eff}} = B \cdot N_{\text{DP}}$.
- Throughput multiplies by $N_{\text{DP}}$, individual request latency remains unchanged.

---

## 7. End-to-End Metrics

- **Time To First Token (TTFT)**:
  $$\text{TTFT} = t_{\text{prefill}}$$
- **Time Per Output Token (TPOT)**:
  $$\text{TPOT} = \frac{1}{O} \sum_{i=0}^{O-1} t_{\text{decode}}(C = S + i)$$
  where $O$ is output length.
- **End-to-End Latency**:
  $$\text{Latency}_{\text{e2e}} = \text{TTFT} + \sum_{i=0}^{O-1} t_{\text{decode}}(C = S + i) \approx \text{TTFT} + O \cdot \text{TPOT}$$
- **Generation Throughput**:
  $$\text{Throughput} = \frac{B \cdot O}{\text{Latency}_{\text{e2e}}} \quad [\text{tokens/second}]$$
- **Communication Overhead Share**:
  $$\text{Comm Share} = \frac{t_{\text{comm}}}{t_{\text{compute}} + t_{\text{memory}} + t_{\text{comm}}}$$
