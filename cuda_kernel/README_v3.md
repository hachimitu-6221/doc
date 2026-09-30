# Mojo v3 的 CUDA 翻译

## Modal B200 实测

`M=N=K=4096`，BF16 输入/输出，FP32 累加，行主序
`C[M,N] = A[M,K] @ B[N,K].T`：

| 实现 | 中位耗时 | 吞吐量 | 相对 cuBLAS 吞吐量 |
| --- | ---: | ---: | ---: |
| CUDA v3 | 0.497868 ms | 276.06 TFLOPS | 18.04% |
| cuBLAS GemmEx | 0.089803 ms | 1530.46 TFLOPS | 100% |

CUDA v3 耗时为 cuBLAS 的 **5.544 倍**。7 轮 CUDA 耗时范围
0.486448–0.522078 ms，cuBLAS 为 0.089656–0.090051 ms。
设备为 NVIDIA B200、148 SM，CUDA 12.8.1、cuBLAS 12.8.4、驱动 580.95.05。
没有锁频；计时前后采样均为 SM 1965 MHz、显存 3996 MHz、功率上限 1000 W，
不代表整个测量期间频率恒定。

完整日志、UTC 时间、各轮时间、版本、源文件 SHA256 见
[results_v3_b200_4096x4096x4096.json](results_v3_b200_4096x4096x4096.json)。
实际运行记录：[Modal app](https://modal.com/apps/akseven6221/main/ap-eWwO9vHvWexioqfVgKXcpr)。

结果与上次 CUDA v2 的 280.76 TFLOPS 接近，但两次不是同一张卡上的配对实验，
不能据此将小幅差异归因于内核改动。

## 翻译范围

[v3.cu](v3.cu) 对应 `../mojo_kernel/v3.mojo` 的默认 BF16 配置。
原文件内部函数名虽为 `kernel_4`，本移植仍按文件版本命名为 v3。

- CTA 为 128 线程，tile 为 `64×256×64`，每轮四条 `64×256×16` MMA。
- A/B 使用二维 TMA 与 128B swizzle。thread 0 发出 TMA/MMA，单缓冲，TMA 与 MMA 串行等待，没有流水线。
- warp 0 分配全部 512 列 TMEM；每线程通过 `tcgen05.ld.16x256b.x32` 读出 128 个 FP32 累加值。
- 与 v2 的直接全局写回不同，v3 将结果转换成 BF16，用 `stmatrix.m8n8.x4` 写入 128B swizzle 共享内存。
- 输出分成四个 `64×64` tile，线程 0–3 各自发出一次二维 TMA store，并提交、等待自己的 bulk group，之后 warp 0 释放 TMEM。

输出共享内存每个 tile 为 8192 字节，整体为 32768 字节；总动态共享内存为
73752 字节。共享内存起点显式对齐 1024 字节，以满足本实现的零 swizzle base offset。
stmatrix 地址的逻辑字节偏移为
`((warp*16 + lane%16)*64 + (lane/16)*8 + matrix*16)*2`，
128B swizzle 后为 `offset ^ ((offset >> 3) & 0x70)`，另加输出 tile 的基址。
ptxas 报告 137 个寄存器、无寄存器 spill。

CUDA 版显式发布 barrier 初始化并添加 tcgen05 同步 fence。所有写共享内存的线程
在 CTA barrier 前执行 async proxy fence，再由线程 0–3 发出 TMA store，
确保输出对异步代理可见。此 barrier 也确保所有线程读完 TMEM 后才允许释放。
相关指令语义见 [NVIDIA PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/)。

只接受正整数且 `M % 64 == 0, N % 256 == 0, K % 64 == 0`；没有尾块支持。
未翻译原文件其他 dtype、transpose 或 tile 泛型分支，也没有添加 v8 的流水线、
2SM MMA 或 CLC。本次实测是 CUDA 移植版，没有运行原 Mojo 编译产物。

## 正确性与比较口径

- 原 Mojo identity 输入全量精确比较；随机 BF16 输入全量对比 cuBLAS，额外抽样 32 个输出与 CPU FP64 点积结果比较。
- 正确性覆盖 `64×256×64`、`128×512×192`、`192×768×576`、`4096×4096×64` 和主测 `4096³`。
- `192×512×576` 通过 Compute Sanitizer 的 memcheck、synccheck、racecheck；额外 `4096×4096×576` racecheck 也无错误或警告。
- 本次所有随机测试与 cuBLAS 的 BF16 输出完全相同；主测计时后再次检查，最大绝对误差及相对 L2 误差均为零。
- 两种实现各直接预热 100 次，再各捕获含 100 次 GEMM 的 CUDA Graph；交替预热各 50 次 graph replay，交替顺序计时 7 轮，使用 CUDA events，取每次 GEMM 耗时的中位数。
- 同 GPU、同 stream、同输入、同输入/输出 dtype。基线为 `cublasGemmEx`、`CUBLAS_COMPUTE_32F`，禁用 reduced-precision reduction，未做 cuBLASLt 算法搜索。
- 计时不含编译、内存分配、tensor map 编码、CPU/GPU 传输和验证。吞吐量按 `2*M*N*K / time` 计算，相对百分比为 `100*cublas_ms/custom_ms`。

主循环仍然不重叠 TMA 和 MMA，且占用全部 512 列 TMEM。这些是从代码可见的
结构性限制；本次没有做 Nsight profiling，不将其表述为已量化的瓶颈占比。

## 复现

在仓库根目录，使用已登录的 Modal 账户：

```bash
.venv/bin/modal run kernel/cuda_kernel/run_v3_modal.py --sanitize
.venv/bin/modal run kernel/cuda_kernel/run_v3_modal.py --m 4096 --n 4096 --k 4096
```

脚本使用 B200 GPU 资源，生成或覆盖对应尺寸的 `results_v3_b200_MxNxK.json`。
镜像为 `nvidia/cuda:12.8.1-devel-ubuntu24.04`，编译目标为 `sm_100a`。
本地 B200/CUDA 12.8 环境也可直接编译运行：

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_100a kernel/cuda_kernel/benchmark_v3.cu -o /tmp/mojo_v3_bench -lcuda -lcublas
/tmp/mojo_v3_bench 4096 4096 4096
```
