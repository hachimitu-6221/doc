# Mojo v4 的 CUDA 翻译

## Modal B200 实测

BF16 输入/输出、FP32 累加，行主序 `C[M,N] = A[M,K] @ B[N,K].T`：

| M×N×K | CUDA v4 耗时 | CUDA v4 TFLOPS | cuBLAS 耗时 | cuBLAS TFLOPS | 相对 cuBLAS 吞吐量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096×4096×4096 | 0.346987 ms | 396.09 | 0.089308 ms | 1538.94 | 25.74% |
| 4096×2560×8192 | 0.524302 ms | 327.67 | 0.117987 ms | 1456.09 | 22.50% |

CUDA v4 耗时分别为 cuBLAS 的 3.885 倍、4.444 倍。
第一行对应原文件默认正确性测试；第二行对应 `--benchmark` 实际硬编码的参数。
注意原 Mojo 的 benchmark 虽然从字典打印 `4096³`，实际调用传入的是
`M=4096, N=2560, K=8192`。本次两个尺寸均实测，没有修改原 Mojo 文件。

2026-09-22 UTC，NVIDIA B200、148 SM，CUDA 12.8.1、cuBLAS 12.8.4、
驱动 580.95.05。每个尺寸的 CUDA/cuBLAS 在同一块 GPU 上测量，但两次 Modal
运行不是同一个 GPU 实例。未锁频；两次计时前后的端点采样均为 SM 1965 MHz、
显存 3996 MHz、功率上限 1000 W，不代表整个测量期间频率恒定。

7 轮耗时范围：

- `4096³`：CUDA 0.344574–0.364696 ms；cuBLAS 0.089103–0.089553 ms。
- `4096×2560×8192`：CUDA 0.507840–0.568776 ms；cuBLAS 0.117732–0.118560 ms。

完整日志、UTC 时间、版本、源码 SHA256、逐轮时间保存在：

- [4096³ 原始结果](results_v4_b200_4096x4096x4096.json)，[Modal 运行](https://modal.com/apps/akseven6221/main/ap-utsgFi71MuM0n6L3EHVS6w)。
- [原 benchmark 尺寸结果](results_v4_b200_4096x2560x8192.json)，[Modal 运行](https://modal.com/apps/akseven6221/main/ap-kBKPMhcr9BzeZXbRbnio0w)。

## 翻译范围与实现

[v4.cu](v4.cu) 翻译 `../mojo_kernel/v4.mojo` 的默认 BF16、2SM 配置。
原文件内部函数名为 `kernel_5`，本移植按文件版本命名为 v4。

- cluster 为 `(2,1,1)`，每 CTA 128 线程，A/B tile 为 `128×128×64`；一对 CTA 共同计算 `256×256` 输出，每轮四条 `256×256×16` 的 `tcgen05.mma.cta_group::2`。
- A/B 均为 128B swizzle、二维 TMA。每个 CTA 加载自己的 128 行 A 和 128 行 B，multicast mask 为 `1 << rank`；两者的完成通知都发送到 leader CTA 的 TMA barrier，每轮总期待 65536 字节。
- 保留 `elect.sync` 选线程。仅 leader CTA 发 MMA，完成通知 multicast 到两个 CTA 的 MMA barrier；两边等 MMA 完成后再开始下一轮，不加入流水线或持久化调度。
- 两个 CTA 的 warp 0 协作分配、释放 512 列 TMEM。每线程分别读 upper/lower 各 128 个 FP32 值，对应本 CTA 内每 warp 的上下两个 16 行区间。
- 输出经 BF16 转换、`stmatrix.m8n8.x4` 写入 128B swizzle 共享内存；每个 CTA 的线程 0–3 各发出一个 `128×64` 的 TMA store，并等待自己的 bulk group 完成。

每 CTA 的 A/B/C 共享内存分别为 16384/16384/65536 字节，另保留原版 64 字节
同步/分配区域，总动态共享内存为 98368 字节。共享内存起点显式对齐 1024 字节，
使用零 swizzle base offset。ptxas 报告 235 个寄存器、无 spill。

CUDA 版显式发布 barrier 初始化、添加 tcgen05 同步 fence，并由所有输出写线程
在 CTA barrier 前执行 async proxy fence。释放双 CTA 的 TMEM 前额外执行
cluster 同步，保证双方均完成读取；保留原版退出前的 cluster 同步。
2026-09-22 补充分配前的 cluster 同步，避免提前访问 peer CTA 的分配辅助 barrier。
本页数据已按修复后的源码重测，旧数据移至 `history/*.pre_cluster_sync.json`。
指令语义参考 [NVIDIA PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/)。

只支持正整数且 `M % 256 == 0, N % 256 == 0, K % 64 == 0`，不支持尾块。
未移植其他 dtype、MMA shape、cluster shape 等泛型分支。本次没有运行原 Mojo
编译产物，因此这些是 CUDA 移植版结果，不是 Mojo/CUDA 编译器的配对比较。

## 正确性与计时

- 原 Mojo identity 输入全量精确比较；随机 BF16 输入全量对比 cuBLAS，另抽样 32 个输出与 CPU FP64 点积结果比较。
- 正确性覆盖 `256×256×64`、`512×512×192`、`768×768×576`、`4096×4096×64` 和两个主测尺寸。
- `768×512×576` 通过 Compute Sanitizer 的 memcheck、synccheck、racecheck；`4096×4096×576` 额外 racecheck 也无错误或警告。
- 本次所有随机测试的 BF16 输出与 cuBLAS 完全相同；两个主测尺寸计时后再次验证，最大绝对误差和相对 L2 误差均为零。
- 各实现直接预热 100 次，各捕获含 100 次 GEMM 的 CUDA Graph，再交替预热各 50 次 graph replay；CUDA events 测量 7 轮、交替先后顺序，取每次 GEMM 耗时中位数。
- 同场 CUDA/cuBLAS 使用同输入、同 stream、相同 BF16 输入/输出及 FP32 累加。基线为 `cublasGemmEx`，`CUBLAS_COMPUTE_32F`，禁用 reduced-precision reduction，未做 cuBLASLt 算法搜索。
- 不计编译、分配、tensor map 编码、CPU/GPU 传输和验证时间。吞吐量为 `2*M*N*K / time`，相对百分比为 `100*cublas_ms/custom_ms`。

v4 仍然是单缓冲，TMA 与 MMA 串行，并占用全部 512 列 TMEM；没有后续版本的
流水线、双缓冲输出或 CLC。这里只描述代码结构，没有做 Nsight profiling 来量化瓶颈。

## 复现

在仓库根目录，使用已登录的 Modal 账户：

```bash
.venv/bin/modal run kernel/cuda_kernel/run_v4_modal.py --sanitize
.venv/bin/modal run kernel/cuda_kernel/run_v4_modal.py --m 4096 --n 2560 --k 8192
```

脚本使用 B200 GPU 资源，生成或覆盖对应尺寸的 `results_v4_b200_MxNxK.json`。
镜像为 `nvidia/cuda:12.8.1-devel-ubuntu24.04`，编译目标为 `sm_100a`。
本地 B200/CUDA 12.8 环境可直接编译：

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_100a kernel/cuda_kernel/benchmark_v4.cu -o /tmp/mojo_v4_bench -lcuda -lcublas
/tmp/mojo_v4_bench 4096 4096 4096
/tmp/mojo_v4_bench 4096 2560 8192
```
