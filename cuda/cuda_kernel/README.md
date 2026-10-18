# Mojo v1 的 CUDA 翻译

全部已请求版本的进度和结果见 [TASK_STATUS.md](TASK_STATUS.md)。

v8 的独立移植、B200/cuBLAS 测试见 [README_v8.md](README_v8.md)。

## B200 实测结果

2026-09-18，NVIDIA B200（148 SM），CUDA 12.8.1，cuBLAS 12.8.4，
驱动 580.95.05，`M=N=K=4096`，BF16 输入/输出、FP32 累加：

| 实现 | 中位耗时 | 吞吐量 | 相对 cuBLAS 吞吐量 |
| --- | ---: | ---: | ---: |
| CUDA v1 | 0.904339 ms | 151.98 TFLOPS | 9.82% |
| cuBLAS GemmEx | 0.088819 ms | 1547.41 TFLOPS | 100% |

CUDA v1 的耗时约为 cuBLAS 的 **10.18 倍**。7 轮 CUDA 耗时范围为
0.904121–0.936015 ms，cuBLAS 为 0.088611–0.088970 ms。
四个尺寸的正确性检查全部通过；本次随机输入与 cuBLAS 的 BF16 输出完全相同。
小尺寸 `128×512×192` 的 memcheck、synccheck、racecheck 均无错误或警告。

从代码结构看，单缓冲使 TMA 和 MMA 串行，且每 CTA 分配全部 512 列 TMEM，
限制同 SM 上其他 CTA 的并发空间；这些都是后续优化方向。
本次没有做 Nsight profiling，因此不将它们当作已量化的瓶颈占比。

完整原始结果见 [results_b200_4096x4096x4096.json](results_b200_4096x4096x4096.json)，
运行记录见 [Modal app](https://modal.com/apps/akseven6221/main/ap-J9Y88K9KVBatmMkMK8L2Pj)。

## 翻译范围

`v1.cu` 对应 `../mojo_kernel/v1.mojo` 的默认 BF16 配置，计算
`C[M,N] = A[M,K] @ B[N,K].T`，三者均为行主序，FP32 累加、BF16 输出。

保留原核的结构：

- 单 CTA，128 线程，`BM=64, BN=256, BK=64`，每轮四条 `64×256×16` MMA。
- 无 swizzle 的 K-major shared memory，单缓冲，TMA 完成后发 MMA，MMA 完成后进入下一轮。
- thread 0 发出 TMA/MMA，warp 0 分配和释放 512 列 TMEM。
- `tcgen05.ld.16x256b.x32` 读取 128 个 FP32 寄存器/线程，以 BF16 二元素向量写回。

无 swizzle 的 shared memory 元素偏移是
`(k / 8) * tile_rows * 8 + row * 8 + k % 8`，因此 TMA 将全局二维矩阵
表示为 `[8, rows, K/8]`，读取 `[8, tile_rows, 8]`。MMA descriptor 的
leading byte offset 为 `tile_rows*16`，stride byte offset 为 `128`。
TMEM 每个 warp 使用 `warp_id*32` datapath 地址，输出对应连续 16 行。
CUDA 中显式添加了 async proxy 所需的 fence，以及释放 TMEM 前的 CTA 同步。

只接受正整数且 `M % 64 == 0, N % 256 == 0, K % 64 == 0`；没有尾块支持。
这里没有翻译原文件中未使用的 dtype、tile 或 transpose 泛型分支，也没有加入流水线优化。

## 在 Modal B200 上复现

在仓库根目录运行，需已有 Modal 登录和 GPU 账户：

```bash
.venv/bin/modal run kernel/cuda_kernel/run_modal.py --sanitize
.venv/bin/modal run kernel/cuda_kernel/run_modal.py --m 4096 --n 4096 --k 4096
```

使用 `nvidia/cuda:12.8.1-devel-ubuntu24.04`，编译目标 `sm_100a`。
`--sanitize` 额外运行 Compute Sanitizer 的 memcheck、synccheck、racecheck。
脚本始终验证三个小尺寸，再验证并测试目标尺寸。完整日志、版本、源码 SHA256、
每轮时间和性能汇总保存在 `results_b200_MxNxK.json`。

也可以在本地 B200/CUDA 环境下编译：

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_100a kernel/cuda_kernel/benchmark.cu -o /tmp/mojo_v1_bench -lcuda -lcublas
/tmp/mojo_v1_bench 4096 4096 4096
```

## 比较口径

- 基线是 `cublasGemmEx` 的默认 Tensor Core 算法，`CUBLAS_COMPUTE_32F`，
  禁止 reduced-precision reduction，`alpha=1, beta=0`。行主序通过
  `C^T = B @ A^T` 映射到 cuBLAS 的列主序接口，无额外转置。
- 同一 GPU、stream 和 A/B 输入，独立预分配 C 输出。数据、tensor map、cuBLAS handle
  都提前准备，计时不含编译、分配、拷贝、descriptor 创建或 graph capture。
- 稠密随机输入，固定 seed `20260918`。先直接预热各 10 次，再各预热一次
  含 100 次 GEMM 的 CUDA Graph。之后交替测量顺序，共 7 轮，每轮 replay 100 次。
  CUDA events 测量 GPU 时间，报告每次 GEMM 耗时的中位数；不主动清空 L2。
- 正确性检查：原 Mojo identity 输入逐元素完全相等；随机输入与 cuBLAS 全量比较，
  `atol=0.03125, rtol=0.01` 且相对 L2 误差不大于 `0.005`；另外 32 个元素
  使用 CPU FP64 累加作独立参考。计时后再次检查随机输出。
- `TFLOPS = 2*M*N*K / seconds / 1e12`，性能百分比为
  `cuBLAS_ms / CUDA_ms * 100%`。结果是此尺寸、此版本的 cuBLAS 基线，
  不是 B200 理论峰值，也不是经过算法搜索的 cuBLASLt 最优性能。

原始 Mojo 文件来自 Modular，授权声明保留在原文件中。
布局核对参考 [Modular tensor_core_async.mojo](https://github.com/modular/modular/blob/main/max/kernels/src/layout/tensor_core_async.mojo)
和 [NVIDIA PTX tcgen05](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions)。
