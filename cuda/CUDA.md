SIMT：single instruction multiple-threads
- 每个SM最多可以同时跟踪64个线程束（2,048个线程）。
- Blackwell GPU每个SM具有64K个32位寄存器（共256 KB）和每个SM共256 KB的L1缓存/共享内存。其中最多228 KB（227 KB可用）可以配置为每个SM的用户管理共享内存。
- 任何单个线程块最多可以请求227 KB的动态共享内存（CUDA保留了228 KB中的1 KB）。