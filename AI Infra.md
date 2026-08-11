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
