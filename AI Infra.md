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
3. 