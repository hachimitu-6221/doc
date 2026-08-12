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
-## 第 5 步结果解读：PC sampling 精确定位"凶手"

55 万次 stall 采样里，分布一目了然：

```
   all   longSB  SASS
230714   225180  STS.U16 [R7], R2        ← A tile 的 smem 存储
138203   122473  STS.U16 [R7+0x8000], R2 ← B tile 的 smem 存储(+32KB 偏移)
  7640     3819  LDG.E.U16 ...           ← 对应的 global 标量 load
  4765     4622  HMMA.1688 (short_sb)    ← mma 等 smem 数据(bank conflict)
```

**前两行就吃掉了 67% 的 stall**，而且 stall 类型是 `long_sb`——`STS` 不是写不动，而是在**等它要写的那个寄存器 R2，也就是前一条 `LDG.E.U16` 从 global 回来的数据**。

这就把整条证据链闭环了：

```
标量 LDG.U16(2B, ~500拍延迟)
   → 紧跟一条依赖它的 STS(干等,无法 run-ahead)
   → __syncthreads() 把全 block 锁住
   → 每 SM 只有 16 个 warp,没人能填这些空拍
   → SM 10% 利用率
```

**kernel1 的诊断结论**（按收益排序的优化方向）:

1. **拷贝向量化 + load/store 分离**:float4 一次搬 16B,LDG 数量少 8 倍；先把整个 tile load 到寄存器再 store,load 之间互相掩盖延迟 → 对应 `tileMemcpyUnrolledVectorized` / `tileMemcpyLoad`（这就是 kernel2/3 的方向）
2. **cp.async**:sm_80 的异步拷贝直接 gmem→smem 不过寄存器、不阻塞 warp → 彻底消灭这两行 STS 的 stall(kernel5/6 的方向）
3. **smem swizzle**：消灭 2.35 亿次 bank conflict(kernel4 引入 `tileMemcpySwizzle`)
4. 次要：`m16n8k16` 减半指令数；occupancy / wave 尾部调 tile 尺寸
## 方法论总结（以后分析任何 kernel 都按这个顺序）

```
runner 跑 GFLOPS        → 离峰值多远?
SpeedOfLight            → compute bound / memory bound / latency bound?
  ├─ compute 高        → ComputeWorkloadAnalysis 看哪条 pipe 满
  ├─ memory 高         → MemoryWorkloadAnalysis 看 DRAM/L2/L1 哪级是瓶颈
  └─ 双低              → SchedulerStats + WarpStateStats 看 stall 原因
Occupancy               → 什么资源限制常驻 warp 数
MemoryWorkloadAnalysis  → 合并效率(sectors/request)、bank conflict、命中率
SourceCounters          → PC sampling 把 stall 钉到具体代码行
```

一个实用技巧：在 CMakeLists 给 nvcc 加 `-lineinfo`,`--page source` 就能直接对应到 CUDA 源码行而不是 SASS。
**建议的练习**：你自己对 kernel2 跑一遍同样的流程（`runner 2`)，对比 kernel1 看 `long_sb` 那两行 STS 是否消失、SOL 的 Compute% 涨了多少——验证"向量化拷贝"这个假设是否真的击中了瓶颈。做完把数字拿来，我们一起对；或者你说继续，我直接带你对比 kernel1→kernel6 每一代解决了哪个 stall。
----
