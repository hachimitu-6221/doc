## v1 naive blackwell gemm

异步/同步 barrier 初始化
用一条线程来完成tma_barrier和mma_barrier的初始化就可以了，然后用fence指令来确保之后的异步指令在执行时能看到被初始化好的tma_barrier和mma_barrier。
tmem初始化
由一个warp完成，即为32个线程负责做tmem的初始化

此处为什么需要块内同步？

一个线程(thread 0)首先发射一条release ptx，设定好预期收到的发射两条tma指令，tma硬件自己异步拉取A、B矩阵的数据从Gmem到Smem