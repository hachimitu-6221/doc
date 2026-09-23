- v1
- v2
- v3
`tcgen05.ld` 得到的是 warp-distributed 的 C fragment，`stmatrix` 把这些 fragment 重新组织成适合矩阵访问的 shared-memory tile；SMEM 使用 swizzle 来减少 bank conflict；然后 TMA 异步把这个 tile 搬到 global memory，从而为后续把数据搬运和计算/packing overlap 成流水线创造条件。
- v4
Kernel 5 利用 CTA cluster + DSMEM/TMA multicast 减少输入 tile 的重复搬运，再利用 `cta_group::2` 的 2SM MMA，让两个 SM 共同消费分布在两个 CTA SMEM 中的数据，从而在减少 SMEM 数据副本的同时，一次完成更大的 MMA tile。