# Mojo v2 的 CUDA 翻译

## Modal B200 实测

`M=N=K=4096`，BF16 A/B/C、FP32 累加，计算行主序
`C[M,N] = A[M,K] @ B[N,K].T`：

| 实现 | 中位耗时 | 吞吐量 | 相对 cuBLAS 吞吐量 |
| --- | ---: | ---: | ---: |
| CUDA v2 | 0.489520 ms | 280.76 TFLOPS | 18.24% |
| cuBLAS GemmEx | 0.089306 ms | 1538.97 TFLOPS | 100% |

CUDA v2 耗时为 cuBLAS 的 **5.481 倍**。7 轮 CUDA 耗时范围
0.487284–0.497430 ms，cuBLAS 为 0.089014–0.089436 ms。
设备为 NVIDIA B200、148 SM，CUDA 12.8.1、cuBLAS 12.8.4、驱动 580.95.05。
没有锁频；计时前后采样均为 SM 1965 MHz、显存 3996 MHz、功率上限 1000 W。
这些端点采样不代表整个计时期间频率恒定。

完整日志、UTC 时间、逐轮耗时、版本及三个源文件的 SHA256 保存在
[results_v2_b200_4096x4096x4096.json](results_v2_b200_4096x4096x4096.json)。
实际运行记录：[Modal app](https://modal.com/apps/akseven6221/main/ap-G2CZcjHsLSwVEYLXGONYuf)。

## 翻译范围与实现

[v2.cu](v2.cu) 翻译 `../mojo_kernel/v2.mojo` 的默认 BF16 配置：

- 每 CTA 128 线程，tile `64×256×64`，每个 K 循环发出四条 `64×256×16` MMA。
- A/B 使用 128B swizzle、二维 TMA tensor map，box 分别为 `[64,64]`、`[64,256]`。
- thread 0 发出 TMA/MMA；单缓冲，等待 TMA 完成后做 MMA，再等待 MMA 完成后进入下一轮。
- warp 0 分配/释放全部 512 列 TMEM；每线程读出 128 个 FP32 值并转换为 BF16 二元素向量直接写回全局内存。
- 保留原算法，不加入双缓冲、流水线、2SM MMA、持久化调度或 v8 的 CLC 优化。

相对无 swizzle 的 v1，MMA descriptor 使用 128B swizzle 编码、1024 字节
stride byte offset，每条 MMA 的 K 偏移为 32 字节。共享内存起点显式对齐
1024 字节，使 descriptor 的 swizzle base offset 为零。动态共享内存为
40984 字节。CUDA 版显式加入 barrier 初始化发布 fence、tcgen05 同步 fence，
以及释放 TMEM 前的 CTA 同步。ptxas 报告 140 个寄存器、无寄存器 spill。

仅支持正整数且 `M % 64 == 0, N % 256 == 0, K % 64 == 0`；无尾块支持。
原 Mojo 的其他 dtype、transpose、tile 泛型分支不在本次翻译范围内。
本次实测的是 CUDA 移植版，没有运行原 Mojo 编译产物。

## 正确性与计时方法

- 原 Mojo 的 identity 输入全量精确比较；随机 BF16 输入全量对比 cuBLAS，另外抽样 32 个输出与 CPU FP64 点积结果比较。
- 正确性覆盖 `64×256×64`、`128×512×192`、`192×768×576`、`4096×4096×64` 和主测 `4096³`。
- `192×512×576` 通过 Compute Sanitizer 的 memcheck、synccheck、racecheck；额外 `4096×4096×576` racecheck 也无错误或警告。
- 本次所有随机测试与 cuBLAS 的 BF16 输出完全相同；主测计时后再次验证，最大绝对误差和相对 L2 误差均为零。
- 两个实现各直接预热 100 次，再各捕获含 100 次 GEMM 的 CUDA Graph；交替预热各 50 次 graph replay，交替顺序测量 7 轮，取每次 GEMM 耗时的中位数。
- 同 GPU、同 stream、同输入、同输入/输出 dtype。cuBLAS 使用 `cublasGemmEx`、`CUBLAS_COMPUTE_32F`，禁用 reduced-precision reduction。计时不含编译、内存分配、tensor map 编码、CPU/GPU 传输和正确性验证，也未做 cuBLASLt 算法搜索。

吞吐量按 `2*M*N*K / time` 计算；相对 cuBLAS 百分比为
`100 * cublas_ms / custom_ms`，不是延迟百分比。

## 与文章的关系

[Modular 系列总结表](https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota)
给出的 swizzling 阶段为 295.6 TFLOPS、cuBLAS 的 16.8%；后续 CLC Persistent
阶段才是 1772.9 TFLOPS、100.6%。本地 v2 对应前者，而不是 v8 的优化阶段。
本次 CUDA v2 的绝对吞吐约为文章 swizzling 数值的 95%；但设备状态、编译器、
cuBLAS 版本和计时方式并未逐项复现，不能据此宣称 CUDA 与 Mojo 编译性能等价。
从代码可见 v2 的 TMA 与 MMA 没有重叠，且分配全部 TMEM；本次未做 Nsight
profiling，不将这些结构性限制当作已量化的瓶颈占比。

## 复现

在仓库根目录，使用已登录的 Modal 账户：

```bash
.venv/bin/modal run kernel/cuda_kernel/run_v2_modal.py --sanitize
.venv/bin/modal run kernel/cuda_kernel/run_v2_modal.py --m 4096 --n 4096 --k 4096
```

运行会使用 B200 GPU 资源，生成或覆盖对应尺寸的 `results_v2_b200_MxNxK.json`。
镜像为 `nvidia/cuda:12.8.1-devel-ubuntu24.04`，目标架构为 `sm_100a`。
本地 B200/CUDA 12.8 环境也可直接运行：

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_100a kernel/cuda_kernel/benchmark_v2.cu -o /tmp/mojo_v2_bench -lcuda -lcublas
/tmp/mojo_v2_bench 4096 4096 4096
```
