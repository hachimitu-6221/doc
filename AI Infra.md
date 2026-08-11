# gpu programming
## 基本概念
- grid -> gpu
- cluster -> ?
- CTA -> SM
- warpGroup(4 warps, 128 threads)
- warp(32 threads)
- thread -> core：最小的执行单位。每个 thread 都有自己的程序计数器和寄存器，并通过它所在 warp 内的 lane ID 来标识。
