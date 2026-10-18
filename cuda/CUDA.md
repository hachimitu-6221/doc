SIMT：single instruction multiple-threads
- 每个SM最多可以同时跟踪64个线程束（2,048个线程）。
- Blackwell GPU每个SM具有64K个32位寄存器（共256 KB）和每个SM共256 KB的L1缓存/共享内存。其中最多228 KB（227 KB可用）可以配置为每个SM的用户管理共享内存。
- 任何单个线程块最多可以请求227 KB的动态共享内存（CUDA保留了228 KB中的1 KB）。
- Blackwell 每个 SM 有 4 个 Warp Scheduler，每个 Scheduler 每周期最多发射一条指令，因此整个 SM 理论上每周期最多发射 4 条 Warp 指令。不同流水线中的指令仍然可以并行执行。
- SFU使用专用的SFU流水线，独立于主INT32/FP32和加载/存储（LD/ST）流水线运行。（可以做overlap）
- 每个周期最多四个线程束可以取得进展。例如，内存操作可以通过SM共16个加载/存储（LD/ST）流水线流动（每个调度器四个LD/ST流水线）。（1/4sm几个LD/ST?）
- CTA的语义和thread block相同；注意CTA和SM的区别，比如 `__syncthread()__` 是作用在CTA上而不是SM上
- 我们要尽可能保证高占用率，但高占用率必须与每线程资源限制（如寄存器和共享内存）相平衡。将寄存器溢出到较慢的内存可能会产生新的停顿。