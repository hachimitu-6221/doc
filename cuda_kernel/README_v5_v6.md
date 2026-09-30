# Mojo v5 / v6 的 CUDA 翻译

## Modal B200 实测

`M=N=K=4096`，行主序 `C = A @ B.T`，BF16 输入/输出、FP32 累加：

| 实现/配置 | CUDA 耗时 | CUDA TFLOPS | 同场 cuBLAS 耗时 | cuBLAS TFLOPS | 相对 cuBLAS 吞吐量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v5，MMA 256×256×16 | 0.120406 ms | 1141.46 | 0.098204 ms | 1399.52 | **81.56%** |
| v6，MMA 256×256×16 | 0.119694 ms | 1148.25 | 0.099386 ms | 1382.88 | **83.03%** |
| v6，MMA 256×128×16 | 0.126558 ms | 1085.97 | 0.102071 ms | 1346.51 | **80.65%** |

前两行对应两个原文件的 `--benchmark` 配置。v5 的默认正确性入口也用同一配置，
但 v6 的默认正确性入口改用 `256×128×16`，因此额外测试第三行。
三个配置分别在不同的 Modal B200 实例运行；每行 CUDA/cuBLAS 在同卡上交替测量。
不能根据不同实例的百分比判断 v6 比 v5 更慢，也不能把这个小幅绝对差异当作可靠提升。

设备为 NVIDIA B200、148 SM，CUDA 12.8.1、cuBLAS 12.8.4、驱动 580.95.05。
未锁频。三次测量前后的端点采样均为 SM 1965 MHz、显存 3996 MHz、功率上限
1000 W；这不代表整个测量期间频率恒定。完整时间戳、各轮耗时、版本、日志、
内核/benchmark/Mojo 源码 SHA256 均在原始结果中：

- [v5 结果](results_v5_b200_4096x4096x4096.json)，[Modal 运行](https://modal.com/apps/akseven6221/main/ap-Uhw5Bkmx0Wnr0VlMs1XByl)。
- [v6 benchmark 配置结果](results_v6_b200_4096x4096x4096.json)，[Modal 运行](https://modal.com/apps/akseven6221/main/ap-Gv1636dRkVOt2rzD6rMBQb)。
- [v6 默认配置结果](results_v6_b200_4096x4096x4096_mma128.json)，[Modal 运行](https://modal.com/apps/akseven6221/main/ap-tTf0HVyxnPP5r8qhbuuM8w)。

以上为 2026-09-22 分配入口同步修复后的复测。原始结果保留在
`history/*.pre_cluster_sync.json`，未把不同源码的旧成绩当作当前结果。

## 保留的内核结构

[v5.cu](v5.cu) 和 [v6.cu](v6.cu) 分别对应同名 Mojo 文件；原 Mojo 内部函数名
为 `kernel_6` 和 `kernel_7`，这里按文件版本命名。每 CTA 192 线程、cluster `(2,1,1)`，
warp 4 加载 A/B，leader CTA 的 warp 5 发出 2SM MMA，warp 0–3 负责输出。

共同保留：二维 TMA、A/B 的 128B swizzle、`elect.sync`、每 K 轮四条 MMA、
双 CTA 的 512 列 TMEM 分配、独立 producer/consumer phase，以及各 stage 的
TMA-full/MMA-empty barrier。MMA 完成通知 multicast 到两个 CTA；最终 compute
barrier 通知输出 warp。没有加入 v8 的持久化调度、CLC 或多组累加器流水线。

| 配置 | 每 CTA 的 A/B tile | A/B stages | 输出共享内存 | 总动态共享内存 | ptxas 寄存器 |
| --- | --- | ---: | --- | ---: | ---: |
| v5 | 128×128×64 | 5 | 完整 128×256，65536 B | 229536 B | 231 |
| v6 MMA_N=256 | 128×128×64 | 6 | 两份 128×32，共 16384 B | 213100 B | 31 |
| v6 MMA_N=128 | 128×64×64 | 8 | 两份 128×32，共 16384 B | 213132 B | 32 |

三个配置均无寄存器 spill。v5 按原文件 `(233472-C_bytes)/(AB_bytes+32)` 计算 stage 数，
一次读出 upper/lower 各 128 个 FP32 值，用 stmatrix 写完整输出，再由线程 0–3
各发出一个 `128×64`、128B swizzle 的 TMA store。

v6 按原文件的可用共享内存公式计算 stage 数，每次只读出 upper/lower 各 16 个
FP32 值，stmatrix 打包到 `128×32`、64B swizzle 的输出 tile。两个共享缓冲交替使用，
warp 0 的 lane 0 发出 TMA store，保留一组在途写出，最终等待所有写出完成。
MMA_N=256/128 分别输出 8/4 个 tile。原有的流水线深度、tile 和输出策略未调参。

CUDA 显式添加 barrier 初始化发布 fence、tcgen05 同步 fence；共享内存写线程在
输出 warp 的 named barrier 前执行 async proxy fence。释放 TMEM 前和退出前增加
cluster 同步，保证对方 CTA 生命周期及双方 TMEM 读取完成。共享内存对齐 1024 B。
分配 TMEM 前也执行 cluster 同步，保证 peer CTA 已完成入口初始化。

只移植上述 BF16 配置，不覆盖其他 dtype、transpose、MMA shape、cluster shape
或 v5 的非二次幂 MMA_N 分支。要求正整数且 `M % 256 == 0`、`K % 64 == 0`，
`N % MMA_N == 0`；没有移植尾块处理。本次没有运行原 Mojo 编译产物，结果不是
Mojo/CUDA 编译器的配对比较。

## 正确性与计时

- 原 Mojo identity 输入全量精确比较；随机 BF16 输入全量对比 cuBLAS，另抽样 32 个输出与 CPU FP64 点积结果比较。
- 覆盖 `256×256×64`、`512×512×192`、`768×768×320`、`768×512×384`、`768×768×576`、`512×512×1216`、`4096×4096×64`、`4096³`，包含未填满流水线、边界及多次环绕。
- 三个配置在这些测试中的 BF16 输出均与 cuBLAS 完全一致；主测计时后再次验证，最大绝对误差和相对 L2 误差均为零。
- 最终脚本不筛选内核：`768×512×576` 的 memcheck、synccheck、racecheck，以及 `4096×4096×576` 的 racecheck 均无错误或警告。自定义核和 cuBLAS 均在检查范围内。
- 两种实现各直接预热 100 次，各捕获包含 100 次 GEMM 的 CUDA Graph，再交替预热各 50 次 graph replay；交替顺序计时 7 轮，使用 CUDA events，取每次 GEMM 耗时中位数。
- 同场基线为 `cublasGemmEx`、`CUBLAS_COMPUTE_32F`，禁用 reduced-precision reduction，未做 cuBLASLt 算法搜索。计时输入为随机 BF16，不是原 Mojo benchmark 的 identity 输入。
- 不计编译、分配、tensor map 编码、传输、验证和 sanitizer 时间。吞吐量按 `2*M*N*K/time`，相对百分比按 `100*cublas_ms/custom_ms` 计算。

## Sanitizer 排查

旧版不筛选 racecheck 的间歇性超时记录仍保留在
[sanitizer_v6_diagnostics.json](sanitizer_v6_diagnostics.json)，不是通过记录。
进一步检查发现 v5/v6 的 `tcgen05.alloc.cta_group::2` 在入口访问 peer CTA 的
内部共享 barrier，缺少跨 CTA 初始化顺序；两个版本的检查工具都报告过 RAW hazard。
SASS 中相关地址是 peer 的 `0x58`（报告地址 `0x1000058`），不是用户 A/B 缓冲。
已在 v4/v5/v6 的双 CTA 分配前添加 cluster 同步，不改变 tile 或计算流水线。
[PTX 规范](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-alloc)
要求 issuing warp 保证 peer CTA 已启动并参与集体分配。

脚本移除 `--kernel-name` 筛选，默认使用 CUDA 12.8.1 随附的 Compute Sanitizer
2025.1.0.0（build 35583870）；编译器、运行库和 cuBLAS 均不升级。
四种配置（含 v4）合计 52 项未筛选检查全部通过：4 项 memcheck、4 项 synccheck、
44 项 racecheck。每种配置在 `4096×4096×576` 重复 racecheck 10 次，并检查完整
`4096³`，均零错误、零警告。见 [审计结果](sanitizer_v456_audit.json) 和
[Modal 运行](https://modal.com/apps/akseven6221/main/ap-0FfmBou7EVLcxC7Uc5xNSw)。
本轮性能结果 JSON 中的单次检查使用独立 2026.3.0.0，均通过；工具版本不参与性能计时。
修复前的失败记录在 `history/sanitizer_pre_fix_*.json`。

**独立新版检查器仍有限制**：在同一 CUDA 12.8/580 驱动环境中，2025.3.1 和
2026.3 的连续检查仍出现 v4 分配内部 barrier 卡住；一次等待 120 秒后终止。
隔离工具版本、关闭 JIT cache、增加初始化 fence 均未消除该现象；额外 fence
已撤回。2026.3 开启可选 deadlock 检测还出现 internal sanitizer error。
这些失败或取消的实验保存在 `history/sanitizer_*`，**未算作通过，也没有断言是
工具误报**。入口初始化竞争已修复，但不能声称解释了每一次历史超时。

`run_sanitizer_audit_modal.py` 默认审计原生 2025.1 工具；`--no-bundled` 可选择
固定的独立 2025.3.1 工具复现额外诊断。其下载 SHA256 在 `cuda_sanitizer.py`
校验，未替换系统 CUDA。审计失败会保存日志并退出，超时会终止整个测试进程组。

## 复现

在仓库根目录，使用已登录的 Modal 账户：

```bash
.venv/bin/modal run kernel/cuda_kernel/run_v5_modal.py --sanitize
.venv/bin/modal run kernel/cuda_kernel/run_v6_modal.py --sanitize
.venv/bin/modal run kernel/cuda_kernel/run_v6_modal.py --mma-n 128 --sanitize
```

可追加 `--m M --n N --k K`。运行使用 B200 GPU 资源，生成或覆盖对应尺寸/配置的
结果 JSON。镜像为 `nvidia/cuda:12.8.1-devel-ubuntu24.04`，目标架构 `sm_100a`。
本地 B200/CUDA 12.8 环境也可编译：

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_100a kernel/cuda_kernel/benchmark_v5.cu -o /tmp/mojo_v5_bench -lcuda -lcublas
nvcc -std=c++17 -O3 -lineinfo -arch=sm_100a kernel/cuda_kernel/benchmark_v6.cu -o /tmp/mojo_v6_bench -lcuda -lcublas
/tmp/mojo_v5_bench 4096 4096 4096
/tmp/mojo_v6_bench 4096 4096 4096
```

v6 编译时加 `-DMOJO_V6_MMA_N=128` 可选择原默认正确性入口配置。
