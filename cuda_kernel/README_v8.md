# Mojo v8 的 CUDA 翻译

`v8.cu` 移植 `../mojo_kernel/v8.mojo` 的默认和 benchmark 配置：
`C[M,N] = A[M,K] @ B[N,K].T`，行主序，BF16 输入/输出，FP32 累加。
v1 的源码、测试程序和历史结果保持独立。

## 修正后的 B200 实测结果

2026-09-19（北京时间），NVIDIA B200（148 SM），CUDA 12.8.1、cuBLAS 12.8.4、
驱动 580.95.05。`M=N=K=4096`，BF16 输入/输出，FP32 累加，7 轮中位数：

| 实现 | 每次 GEMM 耗时 | 吞吐量 | 相对 cuBLAS 吞吐量 |
| --- | ---: | ---: | ---: |
| CUDA v8（修正后） | 0.100345 ms | 1369.66 TFLOPS | 112.97% |
| cuBLAS GemmEx | 0.113363 ms | 1212.38 TFLOPS | 100% |

CUDA v8 的耗时为本次 cuBLAS 的 **0.885 倍**。7 轮 CUDA 时间范围是
0.099309–0.107907 ms，cuBLAS 是 0.113319–0.124098 ms。
双方都增加到 50 次交替 graph 预热，每次 graph 包含 100 次 GEMM。
未锁定 GPU 时钟，保留每轮原始数据及运行前后的时钟、功耗记录；113% 仅表示
本次设备和 cuBLAS 配置的实测比例，不代表所有 B200/版本的通用结论。

全部正确性检查通过，本次随机输入的 BF16 输出与 cuBLAS 完全一致。
memcheck、synccheck、racecheck 均通过，包括 `4096×4096×576` 的 racecheck。
ptxas 报告每线程 **49 registers、0 bytes stack、0 spill**。

完整原始数据见 [results_v8_b200_4096x4096x4096.json](results_v8_b200_4096x4096x4096.json)，
对应 [Modal 运行记录](https://modal.com/apps/akseven6221/main/ap-BNl8sRybbj8EavPTHQbtJr)。

## 之前 26% 的原因

之前的 26.15% 来自有性能退化的 CUDA 翻译，不能据此判断原 Mojo v8 的性能。
修正保留 `256×128×16` MMA、8 级 A/B pipeline、4 级 accumulator、CLC 和 TMA 输出，
没有换成另一种算法或扩大 tile。

1. **barrier release scope 翻译错误。** Mojo `SharedMemBarrier.arrive_cluster` 和
   `arrive_and_expect_bytes` 使用 PTX 默认的 `.release.cta`；`.shared::cluster`
   是 barrier 的地址空间，不意味着 release scope 也要扩大为 cluster。
   初版写成 `.release.cluster.shared::cluster`，生成额外 `CGAERRBAR`，使加载
   流水线发生不必要的等待。同 GPU 消融中仅改回 `.release.cta`，耗时从
   0.342218 ms 降至 0.120412 ms。
2. **单线程选择没有按原核翻译。** 初版用 `lane == 0` 替代原来的 `elect_one_sync()`。
   恢复 `elect.sync` 后，编译器消除了 stack/spill，同一消融中进一步降至 0.093307 ms。
3. **恢复原核二维 TMA descriptor。** 三维重解释数值正确，但原核的 swizzled A/B
   使用二维 TMA。已恢复相同表示；单独更改这一项的实测差异很小，不将其列为主因。

Nsight Compute 报告（独立 GPU 运行、首个 identity-input kernel、未锁频，
不能替代上面的 graph 中位数）：

| 指标 | 修正前 | 修正后 |
| --- | ---: | ---: |
| Kernel duration | 336.16 μs | 82.14 μs |
| Compute throughput | 22.55% | 82.97% |
| Registers/thread | 40 | 49 |
| Spill stores/loads | 16/20 bytes | 0/0 bytes |

原始报告与 SASS 分别在 `v8_profile/`、`v8_profile_fixed/`；
`v8_profile/ablation.json` 保存同 GPU 消融结果，`v8_profile/baseline.cu` 保存原翻译。
旧结果保存在 `results_v8_before_fix_b200_4096x4096x4096.json`。
消融期间存在动态时钟变化，未用其中最高百分比作为最终成绩。

[Modular 文章](https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota)
确实报告 kernel 8 在 `4096³` 达到 1772.9 TFLOPS、cuBLAS 的 100.6%。
文章链接的当前上游源码与本地 `v8.mojo` 只有末尾换行差异；这里没有运行 Mojo 本身，
因此不声称已经复现文章的绝对 TFLOPS。

## 移植范围

| 项目 | CUDA v8 |
| --- | --- |
| CTA cluster | `(2,1,1)`，`tcgen05.mma.cta_group::2` |
| 每 CTA A/B tile | `128×64` / `64×64` |
| MMA shape | `256×128×16`，每个 K tile 发出 4 条 MMA |
| 线程分工 | 224 线程：warp 0–3 epilogue、4 scheduler、5 TMA、6 MMA |
| A/B 流水线 | 8 stages，128B swizzle，TMA multicast |
| 累加流水线 | 512 列 TMEM 分为 4 个 `128` 列缓冲 |
| 调度 | 硬件 CLC，2 stages response pipeline，2 stages throttle pipeline |
| tile 顺序 | 与 `TileScheduler(block_swizzle_size=1)` 相同，奇数 N 列反向遍历 M |
| 输出 | 两块 `128×32` shared buffer，64B swizzle，`tcgen05.ld` → `stmatrix` → TMA store |
| 动态共享内存 | 每 CTA 213292 bytes |

没有把 CLC 换成静态循环，没有把 TMA 输出换成直接 global store。
CUDA 显式保留所需的异步内存 fence，并加强几处同步：

- barrier 初始化由各 CTA 的 thread 0 执行，避免 Mojo 源文件 CLC barrier 的重复初始化。
- 各 epilogue 线程在写入 shared memory 后执行 proxy fence。
- 最后一次 TMA store 等待后同步 epilogue 线程，保证跨 tile 复用 shared buffer 安全。
- 退出前 cluster 同步，确保其他 CTA 不再访问本 CTA 的 shared memory。

跨 CTA 的 barrier 使用 `.release.cta.shared::cluster`；保留必要的
`fence.mbarrier_init.release.cluster`，没有把所有 cluster 同步一概删去。

只实现文件中实际运行的 BF16、转置 B、`(2,1,1)` cluster 配置。
要求正整数维度且 `M % 256 == 0, N % 128 == 0, K % 64 == 0`，不实现尾块或其他泛型配置。
测量对象是这个 CUDA 移植版，没有测量 Mojo 编译器生成的程序。

## 复现

仓库根目录运行，使用已有 Modal 登录和 B200 GPU 账户：

```bash
.venv/bin/modal run kernel/cuda_kernel/run_v8_modal.py --sanitize
.venv/bin/modal run kernel/cuda_kernel/run_v8_modal.py --m 4096 --n 4096 --k 4096
.venv/bin/modal run kernel/cuda_kernel/profile_v8_modal.py --label fixed
```

在本地 B200/CUDA 12.8 环境中：

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_100a kernel/cuda_kernel/benchmark_v8.cu -o /tmp/mojo_v8_bench -lcuda -lcublas
/tmp/mojo_v8_bench 4096 4096 4096
```

镜像为 `nvidia/cuda:12.8.1-devel-ubuntu24.04`。
完整日志、版本、Mojo/CUDA 源码 SHA256 和每轮数据保存到
`results_v8_b200_MxNxK.json`。

## 正确性与测量方法

- 原 Mojo identity 输入逐元素完全一致检查；随机输入全量与 cuBLAS 比较，
  `atol=0.03125, rtol=0.01`，相对 L2 误差不超过 `0.005`。
  另外 32 个元素用 CPU FP64 累加作独立参考，计时后再次检查结果。
- 默认验证 `(256,128,64)`、`(512,256,192)`、`(768,384,576)`、
  `(4096,4096,64)` 及目标尺寸，覆盖单 K tile、流水线 wraparound、矩形矩阵和多 wave。
- `--sanitize` 对 `(512,256,576)` 运行 memcheck、synccheck、racecheck，
  并对 `(4096,4096,576)` 补充 racecheck，检查多 wave 下的缓冲复用。
- cuBLAS 基线使用 `cublasGemmEx`、`CUBLAS_COMPUTE_32F`、默认 Tensor Core 算法，
  禁止 reduced-precision reduction，`alpha=1, beta=0`。
- 双方使用同一 GPU、stream、A/B 和数据布局，独立预分配输出，计时不含
  编译、分配、拷贝、tensor map 创建和 graph capture。
- 各直接预热 100 次，再交替预热各 50 个包含 100 次 GEMM 的 CUDA Graph。
  交替计时顺序，共 7 轮，每轮 replay 100 次，CUDA events 计时，报告中位数。
  固定随机 seed `20260918`，输入为稠密随机数；不清空 L2，不锁定 GPU 时钟。
- `TFLOPS = 2*M*N*K / seconds / 1e12`；相对 cuBLAS 性能为
  `cuBLAS_ms / CUDA_ms × 100%`。基线不是算法搜索后的 cuBLASLt 最优值，也不是理论峰值。

Mojo 原文件的 Modular 授权声明保留在原文件中。依赖语义核对参考：
[Modular TileScheduler](https://github.com/modular/modular/blob/main/max/kernels/src/linalg/matmul/gpu/sm100/tile_scheduler.mojo)、
[Modular swizzle](https://github.com/modular/modular/blob/main/max/kernels/src/layout/swizzle.mojo)、
[NVIDIA PTX CLC](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)。
