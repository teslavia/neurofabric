<p align="center">
  <h1 align="center">⚡️ NeuralOS</h1>
  <p align="center">
    <strong>A Microkernel Heterogeneous LLM Inference Engine for Edge & Cloud</strong><br/>
    <em>Zero-vptr Hourglass ABI · 53 Metal GPU Kernels · PagedAttention · Speculative Decoding · Distributed DAG Scheduling</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue?logo=cplusplus" alt="C++20"/>
  <img src="https://img.shields.io/badge/ABI-C11_Zero--vptr-green" alt="C11 ABI"/>
  <img src="https://img.shields.io/badge/CMake-3.21%2B-064F8C?logo=cmake" alt="CMake"/>
  <img src="https://img.shields.io/badge/Apple_Silicon-Metal_GPU-black?logo=apple" alt="Apple Silicon"/>
  <img src="https://img.shields.io/badge/RK3588-NPU_Zero--Copy-red?logo=arm" alt="RK3588"/>
  <img src="https://img.shields.io/badge/License-Apache_2.0-orange" alt="License"/>
  <img src="https://img.shields.io/badge/Tests-39%2F39_Green-brightgreen" alt="Tests"/>
  <img src="https://img.shields.io/badge/Metal_Kernels-53-blueviolet" alt="Metal Kernels"/>
  <img src="https://img.shields.io/badge/LOC-29.8K-lightgrey" alt="LOC"/>
</p>

<p align="center">
  <a href="README_EN.md"><strong>🇺🇸 English</strong></a> | <a href="README_CN.md"><strong>🇨🇳 中文</strong></a>
</p>

---

> **面向边缘与云端的微内核异构 LLM 推理引擎**
>
> 零虚表沙漏 ABI · 53 个 Metal GPU 内核 · PagedAttention · 推测解码 · 分布式 DAG 调度

```bash
# One-liner: GGUF → streaming tokens
./nf_generate tinyllama-1.1b-chat.Q4_0.gguf "Hello, world" --fp16 --paged --max-tokens 128
```

➡️ **[English Documentation](README_EN.md)** — Full architecture, build matrix, API reference, benchmarks

➡️ **[中文文档](README_CN.md)** — 完整架构说明、构建指南、API 参考、性能基准
