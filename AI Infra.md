# gpu programming
## 基本概念
- grid -> gpu
- cluster -> ?
- CTA -> SM
- warpGroup(4 warps, 128 threads)
- warp(32 threads)
- thread -> core：最小的执行单位。每个 thread 都有自己的程序计数器和寄存器，并通过它所在 warp 内的 lane ID 来标识。
## gemm for T4
### kernel1
- 整个C矩阵被分为一个一个的小矩阵（block），每个block由一个SM处理。
- 每一个block被拆分为更小的矩阵（warp_tile），每个tile由一个warp处理。
- 每个tile被拆分为更小的矩阵（fragment），每个fragment由一个tensor core处理。
----
- 实际上在这里每个tensor core处理的fragment就是计算矩阵的最小单位，对应于navie的gemm里面最小的处理单位是每个线程负责的算的一个C矩阵中的一个元素。
- 根据上面的分层情况来看我们需要遍历block，warp_tile；算出warp_tile里面的一个一个的fragment；这样写代码的思路就显而易见了。
- 只需要盯住一个warp_tile的计算就可以实现整个gemm了
1. 对于共享内存的声明（推荐用方式1）
```c++
	  // 方式1
	  extern __shared__ half shmem[];
	  half* A_block_smem = shmem;
	  half* B_block_smem = &shmem[BM_dim * BK_dim];
	  // 方式2
	  __shared__ half A_block_smem[BM_dim][BK_dim];
	  __shared__ half B_block_smem[BK_dim][BN_dim];
```
2. 每个 MMA tile 用 2 个 `uint32_t` 存累加器，可以声明一个fragment的数组为：`uint32_t acc_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][2];`。epilogue（收尾阶段）要做标量的 fp16 运算，打包成 uint32 后很难直接算：`half (&acc_register_) [mma_tiles_per_warp_m][mma_tiles_per_warp_n][4] = reinterpret_cast<half(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4]>(acc_register);`。
> `(&arr_name)[x][y][z]` 和 `&arr_name[x][y][z]`的区别：前者是对数组下标解引用
3. mma矩阵乘：`mma`指令强制要求32位寄存器操作数，fp16 按“2个/寄存器”打包，所以代码层面必须用 `uint32_t`
4. epilogue累加：要做逐 half 的标量乘加，用 `half` 视图写起来直观。
5. `__syncthreads()`只做块内同步，在块内准备工作做好即将进入更细分的`warp_tile`中间得插入这条同步指令。在每个块的循环结束的末尾得插入这条指令是为了？
6. 在最底层对一个fragment做矩阵乘的时候用的是内积转外积的方式，只要保证mma_k在最外层，mma_m和mma_n的顺序暂时不用考虑，谁在最里面都行。

#### 性能分析
- 先拿基准数字（GFLOPS)，知道离峰值多远
```bash
ncu --kernel-name "regex:kernel_1" \
--launch-count 1 \
--section SpeedOfLight \
./build/runner 1 1 4096 4096 4096 2>&1 | tail -40

==PROF== Connected to process 1609362 (/root/MLOPS/matmul-playground/build/runner)
==PROF== Profiling "kernel_1": 0%....50%....100% - 9 passes
60.9316 GFLOPS/sec for 4096x4096x4096
==PROF== Disconnected from process 1609362
[1609362] runner@127.0.0.1
  void kernel_1<256, 128, 64, 64, 64, 32, 256>(__half *, __half *, __half *, __half *, float, float, unsigned int, unsigned int, unsigned int) (32, 16, 1)x(64, 4, 1), Context 1, Stream 7, Device 0, CC 8.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         1.59
    SM Frequency                    Ghz         1.14
    Elapsed Cycles                cycle     11341091
    Memory Throughput                 %        26.50
    DRAM Throughput                   %         1.00
    Duration                         ms         9.96
    L1/TEX Cache Throughput           %        28.67
    L2 Cache Throughput               %         8.42
    SM Active Cycles              cycle  10465593.80
    Compute (SM) Throughput           %        10.02
    ----------------------- ----------- ------------

    OPT   This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak           
          performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak           
          typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential       
          reasons.
```
- **SpeedOfLight** 看大方向：SM 利用率高还是显存带宽高 → 判断 compute bound 还是 memory bound

| 指标                      | 值       | 含义              |
| ----------------------- | ------- | --------------- |
| Compute (SM) Throughput | **10%** | SM 内部最忙的流水线的利用率 |
| Memory Throughput       | 26.5%   | 显存子系统整体利用率      |
| DRAM Throughput         | **1%**  | 真正打到 HBM 的带宽    |
**判读规则**：两个都低（<60%）说明既没吃满算力也没吃满带宽 → kernel 是 **latency bound（延迟受限）**:warp 大量时间在"等"，等的过程中没有别的 warp 能顶上来干活。
ncu 自己也给了提示：`Look at Scheduler Statistics and Warp State Statistics`。这就是我们下一步。

- 往哪边深钻：compute 侧看 tensor pipe 利用率、指令发射；memory 侧看DRAM/L2/smem 各级流量和冲突
- **Warp State** 看 stall 原因——SM 利用率不高时，warp 到底在等什么
- Occupancy、launch 配置等辅助信息
- 数据 → 假设 → 改代码 → 再 profile 验证
----
