# 用 Nsight Compute CLI (ncu) 分析 CUDA Kernel 性能 —— 以 kernel1 为例

> 环境:A800-SXM4-80GB (sm_80, 108 SMs, FP16 峰值 ~312 TFLOPS),ncu 2025.2.0
> 对象:`MLOPS/matmul-playground` 的 `kernel1`(smem tile + `mma.m16n8k8` 的 HGEMM)
> 目标:演示一套**自顶向下、可复现**的 ncu 性能分析流程,从基准数字一路定位到具体代码行
---
## 0. 方法论总览(分析任何 kernel 都按这个顺序)

```
runner 跑 GFLOPS        → 离峰值多远?
SpeedOfLight            → compute bound / memory bound / latency bound?
  ├─ compute 高        → ComputeWorkloadAnalysis 看哪条 pipe 满
  ├─ memory 高         → MemoryWorkloadAnalysis 看 DRAM/L2/L1 哪级是瓶颈
  └─ 双低              → SchedulerStats + WarpStateStats 看 stall 原因
Occupancy               → 什么资源限制了常驻 warp 数
MemoryWorkloadAnalysis  → 合并效率(sectors/request)、bank conflict、各级命中率
SourceCounters          → PC sampling 把 stall 精确定位到具体指令/代码行
```
核心原则:**先看大方向(SOL),再往瓶颈一侧逐级下钻,最后用 PC sampling 钉到代码行。
数据 → 假设 → 改代码 → 再 profile 验证。**
通用命令模板:
```bash
cd MLOPS/matmul-playground

# --kernel-name  用正则只抓目标 kernel(runner 里还有其他 kernel)
# --launch-count 1  只 profile 第 1 次发射(后面的迭代是重复的)
# 注意:runner 参数格式为 ./build/runner <kernel编号> <迭代次数> <M> <N> <K>
ncu --kernel-name "regex:kernel_1" --launch-count 1 \
    --section <Section名> \
    ./build/runner 1 1 4096 4096 4096
```

> **重要口径**:ncu 默认 `--clock-control base`,把 SM 时钟锁在基准频率(本机 1.14 GHz),
> 保证多次 profile 可复现。所以 profile 里的 Duration 比 runner 实测的 wall-clock 慢,
> 两者**不能混着比**;优化前后对比要用同一组 profile 数据。

---
## 第 0 步:基准数字 —— 离峰值多远?

```bash
./build/runner 1 10 4096 4096 4096
```
输出:
```
13358.8 GFLOPS/sec for 4096x4096x4096
```

**判读**:13.4 TFLOPS ≈ A800 FP16 峰值(312 TFLOPS)的 **4%**。提升空间巨大,值得 profile。

---
## 第 1 步:SpeedOfLight —— 定大方向
```bash
ncu --kernel-name "regex:kernel_1" --launch-count 1 \
    --section SpeedOfLight \
    ./build/runner 1 1 4096 4096 4096
```
输出(节选):
```
Section: GPU Speed Of Light Throughput
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz         1.59
SM Frequency                    Ghz         1.14     <- 锁频在 base clock
Elapsed Cycles                cycle     11341091
Memory Throughput                 %        26.50
DRAM Throughput                   %         1.00     <- 几乎没打到 HBM
Duration                         ms         9.96
L1/TEX Cache Throughput           %        28.67
L2 Cache Throughput               %         8.42
SM Active Cycles              cycle  10465593.80
Compute (SM) Throughput           %        10.02     <- 算力也远没吃满

OPT   This workload exhibits low compute throughput and memory bandwidth
      utilization ... typically indicate latency issues. Look at Scheduler
      Statistics and Warp State Statistics ...
```

**判读规则**:

| Compute (SM) | Memory/DRAM | 结论 | 下一步 |
|---|---|---|---|
| 高 (>60%) | 低 | compute bound | ComputeWorkloadAnalysis 看哪条 pipe |
| 低 | 高 (>60%) | memory bound | MemoryWorkloadAnalysis 看哪级缓存 |
| **低** | **低** | **latency bound(warp 在干等)** | **SchedulerStats + WarpStateStats** |
造成上述情况的可能有以下几种情况：
1. **Kernel 太小 / 并行度不够**：启动的 block 或线程太少，填不满整张卡的 SM。比如 grid 只有几十个 block，而显卡有上百个 SM，大部分 SM 根本没活干。
2. **延迟受限（latency-bound）**：每个线程的工作量太小、依赖链太长、或者频繁同步（`__syncthreads`），线程一直在"等"而不是"算"。
3. **Occupancy 太低**：寄存器/shared memory 用太多，每个 SM 上同时驻留的 warp 太少，没有足够的 warp 来隐藏访存和指令延迟。
4. **Kernel 执行时间太短**：如果 kernel 只跑几微秒，启动开销（launch overhead）占比很大，利用率自然难看。可以在 ncu 的 Summary 里看 kernel 的实际执行时长。
排查方向：
5. **看 Occupancy 部分**（Theoretical vs Achieved Occupancy）：如果 Achieved Occupancy 远低于 Theoretical，说明活跃 warp 不够；如果 Theoretical 本身就低，查寄存器/shared memory 用量。 
6. **看 Launch Statistics**：确认 grid size / block size，计算 `grid × block / (SM数 × 2048)`，看能不能填满显卡。
7. **看 Warp State Statistics**：看 warp 主要 stall 在什么上——`Stall Long Scoreboard`（等访存）、`Stall Wait / Short Scoreboard`（等依赖）、`Stall Barrier`（等同步）能直接告诉你延迟卡在哪。
8. **看 kernel 时长**：如果只有几 μs，问题可能是"活儿太小"，考虑合并多个小 kernel、增大单次处理的数据量。
kernel1:Compute 10%、DRAM 1% → **典型 latency bound**。ncu 的 ==OPT== 提示也指向同一处；要我去==Look at Scheduler Statistics and Warp State Statistics==

---
## 第 2 步:Scheduler + Warp State —— warp 在等什么?
```bash
ncu --kernel-name "regex:kernel_1" --launch-count 1 \
    --section SchedulerStats --section WarpStateStats \
    ./build/runner 1 1 4096 4096 4096
```
输出(节选):
```
Section: Scheduler Statistics
---------------------------- ----------- ------------
One or More Eligible                   %        10.86
Issued Warp Per Scheduler                        0.11   <- 每 9.2 拍才发出 1 条指令
No Eligible                            %        89.14   <- 89% 的周期发射槽是空的
Active Warps Per Scheduler          warp         3.52   <- 常驻 warp 很少(上限 16)
Eligible Warps Per Scheduler        warp         0.13

Section: Warp State Statistics
---------------------------------------- ----------- ------------
Warp Cycles Per Issued Instruction             cycle        32.45

OPT   ... each warp spends 20.3 cycles being stalled waiting for a scoreboard
      dependency on a L1TEX (local, global, surface, texture) operation ...
      This stall type represents about 62.6% of the total average of 32.4
      cycles between issuing two instructions.
```
**判读**:
- `Issued Warp Per Scheduler = 0.11`:发射槽 89% 时间空转
- `Active Warps Per Scheduler = 3.52 / 16`:**常驻 warp 太少(occupancy 低)**
- 32.4 拍/指令中 **20.3 拍(62.6%)是 `long_scoreboard`** —— 等 **L1TEX / 全局内存** 数据
**翻译**:每个 warp 大量时间在等 global load 的数百拍延迟,而 SM 里常驻 warp 又太少,
没有别的 warp 能利用这些空拍。**延迟高 × 掩盖延迟的 warp 少**,两个问题互相放大。

对照代码想原因:kernel1 的 `tileMemcpy`(gmem→smem)是**标量同步拷贝**——
每条 `LDG`(2 字节)紧跟一条依赖它的 `STS`,warp 只能干等,没有任何 overlap。

常见 stall 原因速查:

| stall 类型 | 含义 | 典型对策 |
|---|---|---|
| `long_scoreboard` | 等 global/local 内存数据 | 增大在途 load 数、向量化、cp.async、提高 occupancy |
| `short_scoreboard` | 等 shared 内存数据 | 消除 bank conflict、双缓冲 fragment |
| `barrier` | 等 `__syncthreads()` | 减少同步次数、warp specialization、mbarrier |
| `mio_throttle` | MIO 队列满(LDS/LDSM 太密) | 减少 smem 指令数、错开发射 |
| `math_pipe_throttle` | 算数 pipe 饱和 | 接近极限,换更大的 mma/减少非 mma 指令 |
| `wait` | 固定延迟的寄存器依赖 | 增加 ILP、让编译器更好调度 |
| `not_selected` | 有空但未被选中(正常现象) | 无需处理 |

---

## 第 3 步:Occupancy —— 谁限制了常驻 warp 数?

```bash
ncu --kernel-name "regex:kernel_1" --launch-count 1 \
    --section Occupancy --section LaunchStats \
    ./build/runner 1 1 4096 4096 4096
```

输出(节选):

```
Section: Launch Statistics
-------------------------------- --------------- ---------------
Block Size                                                   256
Grid Size                                                    512
Registers Per Thread             register/thread             106
Shared Memory Configuration Size           Kbyte          102.40
Dynamic Shared Memory Per Block      Kbyte/block           49.15
Waves Per SM                                                2.37

OPT   ... 2 full waves and a partial wave of 80 thread blocks ... this
      partial wave may account for up to 33.3% of the total runtime ...

Section: Occupancy
------------------------------- ----------- ------------
Block Limit SM                        block           32
Block Limit Registers                 block            2   <- 寄存器卡住
Block Limit Shared Mem                block            2   <- smem 也卡住
Block Limit Warps                     block            8
Theoretical Active Warps per SM        warp           16
Theoretical Occupancy                     %           25
Achieved Occupancy                        %        22.01
Achieved Active Warps Per SM           warp        14.08
```

**判读**:

- "Block Limit" 一行是排除法:取各资源的最小值 → 每 SM 只能驻 **2 个 CTA**(16 warp)
  - 寄存器:106 regs/thread × 256 thread × 2 ≈ 54K,贴近 64K/SM 上限
  - smem:50KB/CTA × 2 = 100KB,贴近 102.4KB 配置
- Theoretical Occupancy 25%,Achieved 22% —— 正是第 2 步 `3.52 warp/scheduler` 的来源
- 次要问题:`Waves Per SM = 2.37`,最后一波只装满 37%,有尾部浪费(grid 最好整除波数)

> **注意**:对大 tile GEMM,低 occupancy 本身不是罪(高性能 GEMM 常只有 2 CTA/SM),
> 关键是**有没有别的手段在少数 warp 内掩盖延迟**(ILP、cp.async、多级流水、双缓冲)。
> kernel1 的问题是这些手段一个都没有。

---

## 第 4 步:Memory Workload —— 量化访存效率

```bash
ncu --kernel-name "regex:kernel_1" --launch-count 1 \
    --section MemoryWorkloadAnalysis \
    --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    ./build/runner 1 1 4096 4096 4096
```

输出(节选):

```
Section: Command line profiler metrics
---------------------------------------------------------------- ----------- ------------
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio  sector   2.06
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                     234881160
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                         11753

Section: Memory Workload Analysis
---------------------------- ----------- ------------
Memory Throughput                Gbyte/s        20.41
L1/TEX Hit Rate                        %         3.34
L2 Hit Rate                            %        64.76
```

**判读**:

- **sectors/request = 2.06**:每个 warp 的 global load 请求只搬 2 个 sector(64B)。
  这是 2B 标量 load 的天花板(32 线程 × 2B);合并本身没浪费,但相比 float4(16B/线程)
  指令数多 8 倍、在途字节数少 8 倍。
- **smem load bank conflicts ≈ 2.35 亿次**:巨量!A tile 在 smem 里行距 64 half = 128B,
  正好是 32 bank × 4B 的整倍数,`ldmatrix` 一次读的 8 行全落在相同 bank 上,
  每次 ldmatrix 被串行化。写侧仅 1.2 万次,基本无冲突。
- **L2 Hit 64.8% / DRAM 20GB/s**:B 面板重读大多命中 L2,所以第 1 步 DRAM 才 1% ——
  再次确认瓶颈不在带宽。

常用 memory 指标速查:

| 指标 | 健康值 | 异常说明 |
|---|---|---|
| sectors/request (global ld) | float4 访问 ≈ 4 | 过低=访问粒度太小;远高于理想=合并差 |
| shared bank conflicts | ≈ 0 | 非零就要查 swizzle/padding |
| L2 Hit Rate | GEMM 通常 >50% | 过低检查 tile 光栅化顺序(CTA 调度) |
| DRAM Throughput | 接近峰值说明带宽瓶颈 | GEMM 优化好后通常远低于峰值 |

---

## 第 5 步:Source Counters —— 把 stall 钉到具体指令

PC sampling:硬件周期性采样每个 warp 停在哪条指令、因为什么 stall。

```bash
# 先采集到报告文件(-f 覆盖旧文件)
ncu --kernel-name "regex:kernel_1" --launch-count 1 \
    --section SourceCounters \
    -o /tmp/k1_src -f \
    ./build/runner 1 1 4096 4096 4096

# 再导出 CSV 聚合,按 "Warp Stall Sampling (All Samples)" 排序
ncu --import /tmp/k1_src.ncu-rep --page source --csv 2>/dev/null | python3 -c "
import csv, sys
rows = list(csv.reader(sys.stdin))
hdr = rows[1]
i_src = hdr.index('Source')
i_all = hdr.index('Warp Stall Sampling (All Samples)')
i_lsb = hdr.index('stall_long_sb')
i_mio = hdr.index('stall_mio')
i_ssb = hdr.index('stall_short_sb')
data = []
for r in rows[2:]:
    if len(r) <= i_ssb: continue
    try: data.append((int(r[i_all]), int(r[i_lsb]), int(r[i_mio]), int(r[i_ssb]), r[i_src].strip()))
    except: pass
tot = sum(d[0] for d in data)
print(f'total samples: {tot}')
for d in sorted(data, reverse=True)[:10]:
    print(f'{d[0]:>7} {d[1]:>7} {d[2]:>6} {d[3]:>7}  {d[4][:70]}')
"
```

输出:

```
total samples: 551603
    all  longSB    mio shortSB  SASS
 230714   225180   3570       0  STS.U16 [R7], R2         <- A tile 的 smem 存储
 138203   122473   1416       0  STS.U16 [R7+0x8000], R2  <- B tile 的 smem 存储(+32KB)
  26909        0      0       0  ISETP.GE.U32.AND ...      <- tileMemcpy 的循环判断
   8961        0      0    5776  IMAD R2, R0, c[0x0][0x190], R5
   7640     3819      0       0  LDG.E.U16.STRONG.GPU ...  <- 对应的 global 标量 load
   4765        0      0    4622  HMMA.1688.F16 ...         <- mma 等 smem(bank conflict)
   3988      430   3525       0  LDSM.16.MT88 ...          <- ldmatrix 在 MIO 排队
```

**判读(证据链闭环)**:

- **前两条 `STS.U16` 吃掉 67% 的全部 stall 采样**,stall 类型 `long_sb` ——
  不是写 smem 慢,而是**在等要写的寄存器 R2,即前一条 `LDG.E.U16` 的 global 数据**。
- `LDG.E.U16` / `LDSM` 的 `mio` stall:标量访存指令太多,MIO 队列排队。
- `HMMA` 的 `short_sb`:mma 等 smem 数据,对应第 4 步的 bank conflict。

完整因果链:

```
标量 LDG.U16(2B, ~500 拍延迟)
  → 紧跟依赖它的 STS(干等,无法 run-ahead)      [第 5 步实锤, 67% stall]
  → __syncthreads() 锁住全 block
  → 每 SM 只有 16 个常驻 warp,没人填空拍          [第 3 步: occupancy 25%]
  → SM Compute 10% / DRAM 1%                    [第 1 步: latency bound]
```

---

## 诊断结论:kernel1 的优化方向(按收益排序)

| # | 优化 | 针对的证据 | 对应本仓库的演进 |
|---|---|---|---|
| 1 | 拷贝向量化(float4 一次 16B)+ load/store 分离(先全 load 到寄存器再统一 store) | sectors/request=2、STS 等 LDG | `tileMemcpyUnrolledVectorized` / `tileMemcpyLoad`(kernel2/3) |
| 2 | `cp.async` 异步拷贝:gmem→smem 不过寄存器、不阻塞 warp | 两条 STS 的 long_sb 占 67% | kernel5/6 |
| 3 | smem swizzle / padding 消灭 bank conflict | 2.35 亿次 smem load 冲突 | `tileMemcpySwizzle`(kernel4) |
| 4 | `mma.m16n8k16` 替代 m16n8k8,指令数减半 | HMMA 密度、mio 压力 | kernel9 |
| 5 | 多级流水 / 双缓冲,overlap 拷贝与计算 | barrier + long_sb | kernel6/9 |
| 6 | 次要:tile 尺寸匹配 SM 数,消除 wave 尾部 | Waves Per SM = 2.37 | kernel9 的按尺寸选 tile |

---

## 实用技巧与注意事项

1. **只抓目标 kernel**:`--kernel-name "regex:kernel_1"`,否则 runner 里所有 kernel 都会被 profile。
2. **只抓一次发射**:`--launch-count 1`;runner 的迭代是重复的,profile 一次就够。
   有 warmup 时可配 `--launch-skip N`。
3. **锁频口径**:ncu 默认锁 base clock,Duration 比 wall-clock 慢属正常;
   对比优化前后要用同一锁频口径的 profile 数据。可用 `--clock-control none` 关闭。
4. **源码级对应**:给 nvcc 加 `-lineinfo`,`--page source` 就能把 stall 对应到 CUDA 源码行
   (默认只能看到 SASS)。
5. **导出完整报告**:`-o report -f --set full` 采集全部 section,之后用
   `ncu --import report.ncu-rep` 或 GUI (`ncu-ui`) 随时翻看,不用重跑。
6. **开销**:profile 会让 kernel 慢数倍(metric replay 多遍执行),不要在 profile 时测性能数字。
7. 可用 section 列表:`ncu --list-sections`;可用 metric 查询:`ncu --query-metrics | grep bank`。

## 验证闭环(下一步练习)

对 kernel2 跑同样流程,检验"向量化拷贝"假设:

```bash
./build/runner 2 10 4096 4096 4096            # GFLOPS 涨了多少?
ncu --kernel-name "regex:kernel_2" --launch-count 1 \
    --section SpeedOfLight --section WarpStateStats \
    ./build/runner 2 1 4096 4096 4096         # long_sb 是否下降?Compute% 涨了多少?
```

预期:两条 `STS.U16` 的 long_sb stall 基本消失,Compute (SM) Throughput 显著上升,
瓶颈转移到下一处(通常是 barrier 或 smem bank conflict)——然后重复本流程,逐代分析。
