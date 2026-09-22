- v1
- v2
- v3
`tcgen05.ld` 得到的是 warp-distributed 的 C fragment，`stmatrix` 把这些 fragment 重新组织成适合矩阵访问的 shared-memory tile；SMEM 使用 swizzle 来减少 bank conflict；然后 TMA 异步把这个 tile 搬到 global memory，从而为后续把数据搬运和计算/packing overlap 成流水线创造条件。