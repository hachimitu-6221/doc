
- 四个层次

|层次|学习方式|主要训练什么|
|---|---|---|
|Level 1|边看视频边写代码|理解知识、熟悉 API|
|Level 2|看完视频，关闭视频和源码，独立重新实现|知识提取、代码组织、实现能力|
|Level 3|只知道目标，自己设计实现，再看别人怎么做|系统设计、独立决策|
|Level 4|自己提出新问题，设计实验并解决|真正的开放式问题解决能力|

### 项目一：基于 NVIDIA Blackwell 架构的高性能 BF16 GEMM 算子设计与优化

**技术栈：** CUDA C++、PTX、CUTLASS/CuTe、TMA、TCGen05、Tensor Memory、Nsight Compute

- **底层算子实现：** 面向 NVIDIA Blackwell（SM100）架构，基于 CUDA C++ 和 Inline PTX 实现 BF16 GEMM（(C=AB^T)，FP32 累加）；采用 2D TMA、128B Shared Memory Swizzle 和 TCGen05 Tensor Core 指令，设计 CTA Tile（64×256×64）及寄存器、Shared Memory、TMEM 数据布局。
    
- **异步流水线优化：** 基于 `mbarrier` 构建 TMA 数据搬运与 MMA 计算的异步同步机制，进一步引入多缓冲和 Warp Specialization，减少数据加载与计算之间的等待，并分析 Pipeline Stage 数量、Shared Memory 占用及并行度之间的权衡。
    
- **硬件特性与工程化：** 将底层 PTX 实现逐步迁移至 CUTLASS/CuTe 架构接口，探索 1SM/2SM MMA、Cluster 协作和 Tile 配置对不同 GEMM Shape 的影响，提高代码可维护性与硬件适配能力。
    
- **性能分析与验证：** 构建涵盖不同 M/N/K Shape 的 Correctness Test 和 Microbenchmark，使用 Nsight Compute 分析 Tensor Core 利用率、访存效率及 Pipeline Stall；在 B200 上实现峰值 **[X] TFLOPS**，选定 Shape 下达到 cuBLAS **[Y]%** 的性能，并通过消融实验量化各项优化收益。
    

### 项目二：面向 LLM Prefill 的 FlashAttention-2 CUDA 算子实现与性能优化

**技术栈：** CUDA C++、Tensor Core、Online Softmax、Shared Memory Tiling、Nsight Compute、PyTorch

- **Attention 算法实现：** 基于 CUDA C++ 实现 BF16 FlashAttention-2 Forward Kernel，采用分块矩阵乘法与 Online Softmax，融合 QKᵀ、Softmax 和 PV 计算，避免显式物化 (N\times N) Attention 矩阵，降低 Attention 中间结果的显存访问与存储开销。
    
- **推理场景适配：** 支持 Causal/Non-causal Attention、GQA 和 Variable-length Sequence，处理不同 Batch Size、Sequence Length、Head Dimension 下的线程映射、Mask、数值精度及尾块边界问题。
    
- **Kernel 性能优化：** 针对 GPU 架构设计 Q/K/V Tile、Warp 分工、Shared Memory 与寄存器布局；结合 Nsight Compute 分析 Occupancy、Register Pressure、Global Memory Throughput 等指标，针对短序列与长序列分别调优 Block Size 和并行策略。
    
- **正确性与性能评测：** 构建多 Shape Benchmark，对比 PyTorch SDPA 与官方 FlashAttention-2 实现；在 A800/H100 上，选定 Prefill Shape 下实现 **[X] μs** 延迟、达到官方 FlashAttention-2 **[Y]%** 的性能，并分析不同 Shape 下的性能差距。
    
- **端到端推理验证（进阶）：** 将算子接入 SGLang 的 Attention 执行路径，在固定模型、Batch Size、输入长度及输出长度下比较 Prefill 延迟、TTFT 和系统吞吐量，验证 Kernel 级优化对真实推理场景的收益及适用范围。