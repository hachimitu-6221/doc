- v1
- v2
- v3
`tcgen05.ld` 得到的是 warp-distributed 的 C fragment，`stmatrix` 把这些 fragment 重新组织成适合矩阵访问的 shared-memory tile；SMEM 使用 swizzle 来减少 bank conflict；然后 TMA 异步把这个 tile 搬到 global memory，从而为后续把数据搬运和计算/packing overlap 成流水线创造条件。
- v4
Kernel 5 利用 CTA cluster + DSMEM/TMA multicast 减少输入 tile 的重复搬运，再利用 `cta_group::2` 的 2SM MMA，让两个 SM 共同消费分布在两个 CTA SMEM 中的数据，从而在减少 SMEM 数据副本的同时，一次完成更大的 MMA tile。



对，而且我重新对了 PTX/CUTLASS 的 `cta_group::2` 语义后，我会把建议改成：**不建议每轮塞一个 `cluster_sync()`，更好的做法是把 `TMA_BARRIER` 改成“两生产者 barrier”，让 rank0/rank1 各自先登记自己的 TMA bytes，再发自己的 TMA。**

你现在是：

```cpp
// TMA_BARRIER init count = 1

if (rank == 0) {
    mbarrier.arrive.expect_tx(
        TMA_BARRIER,
        2 * (A_BYTES + B_BYTES));
}

// rank0/rank1 各自发 A、B
load_operand(...);
load_operand(...);
```

也就是说 **rank0 一个人替两个 CTA 把总 transaction bytes 都登记了**。而 `cta_group::2` 的确允许 peer CTA 的 TMA completion 被路由到 pair leader（这里就是 rank0）的 mbarrier，所以后面 rank0 一个 barrier 能收齐双方的完成通知。([NVIDIA Docs](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html?utm_source=chatgpt.com "1. Introduction — PTX ISA 8.7 documentation")) 你代码正是这种结构。

### 我更推荐的写法

把 rank0 的 `TMA_BARRIER` 初始化成 `count = 2`：

```cpp
if (warp == 0 && selected && rank == 0) {
    mbarrier.init(... TMA_BARRIER ..., 2);
}
```

然后每轮让 **两个 CTA 各自对 rank0 的 barrier 做一次 `arrive.expect_tx`**：

```cpp
if (warp == 0 && selected) {

    uint32_t leader_barrier =
        remote_address(base + TMA_BARRIER, 0);

    // 每个 CTA 只登记自己即将发出的 A+B
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 "
        "_, [%0], %1;"
        :
        : "r"(leader_barrier),
          "r"(A_BYTES + B_BYTES)
        : "memory"
    );

    // 本 CTA 的 expect_tx 已经先执行
    load_operand(... A ...);
    load_operand(... B ...);
}
```

逻辑就变成：

```text
rank0                               rank1
  |                                  |
arrive_expect_tx(A+B)           arrive_expect_tx(A+B)
  |                                  |
TMA A0 / B0                     TMA A1 / B1
  \                                  /
   \                                /
    ---> rank0.TMA_BARRIER <--------
                |
         两个 arrive 都满足
         4 个 TMA bytes 都完成
                |
             phase完成
```

`mbarrier` 本身就支持这种模式：初始化 arrival count 为 2，然后两个 producer 分别 `arrive_expect_tx` 自己负责的 transaction bytes；直到两个 software arrival 和所有 transaction bytes 都完成，barrier 才会完成。([NVIDIA Docs](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/primitives.html?utm_source=chatgpt.com "Primitives — NVIDIA CUTLASS Documentation"))

这个方案的好处是：**不需要每个 iteration 做昂贵的全 cluster barrier，同时彻底消掉 rank0/rank1 之间的 `expect_tx → TMA` 顺序竞态。**

---

至于你说的：

> 好像不加同步也不会出错？

这非常合理，而且我甚至预期你现在这份代码**很难测出错误**。

原因主要有三个。

第一，第一次循环前已经有 152 行的：

```cpp
cluster_sync();
```

所以两个 CTA 是从一个很接近的时间点进入循环的。

第二，每轮末尾两个 CTA 又都执行：

```cpp
wait_barrier(base + MMA_BARRIER, phase);
```

而这个 barrier 是 rank0 的 `tcgen05.commit` multicast 给两个 CTA 的，因此两个 CTA 每轮实际上都会被同一个 MMA completion 卡住，再进入下一轮。

所以实际运行更像：

```text
             iteration N
rank0                         rank1
  |                             |
  +------ 等同一个 MMA ----------+
              |
              v
         基本同时释放
              |
rank0: expect_tx
rank1: TMA issue
rank0: TMA issue
              |
        TMA 真正完成
```

rank0 相比 rank1 只多了一条：

```cpp
mbarrier.arrive.expect_tx
```

而 **TMA 是异步的**。`cp.async.bulk.tensor` 发出去以后，并不会立刻对 barrier 做 `complete_tx`；要等实际数据搬运完成之后才 decrement transaction count。PTX 明确规定 complete-tx 是异步 copy 完成时发生的。([NVIDIA Docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?utm_source=chatgpt.com "1. Introduction — PTX ISA 9.4 documentation"))

因此即便发生：

```text
rank1: TMA issue
       ↓
rank0: expect_tx
       ↓
rank1: TMA 真正 complete_tx
```

实际上仍然能正常工作。

真正危险的是：

```text
rank1: TMA issue
       ↓
rank1: TMA complete_tx    ← 太快
       ↓
rank0: expect_tx          ← 太晚
```

但在你当前 kernel 中，这个时间窗口非常难出现：两个 CTA 每轮刚被同一个 MMA barrier 放行，rank0 只落后一条 `expect_tx`，而一次 global→shared TMA 真正完成显然需要更长的时间。

**所以它是一个“时序上极其宽松”的 race。**

这也解释了为什么：

> 跑 benchmark、随机数据、跑很多次，全都正确。

完全正常。

但是 PTX/CUTLASS 给出的协议仍然是：**先 `arrive_expect_tx`，再让对应 TMA 有机会完成并 decrement transaction count。** NVIDIA 的 CUTLASS 文档也直接写明 transaction counter 应该在 TMA 能 decrement 之前设置好。([NVIDIA Docs](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/primitives.html?utm_source=chatgpt.com "Primitives — NVIDIA CUTLASS Documentation"))

所以我现在会这样选：

**追求最少指令、当前 B200 固定 kernel 极致性能：**你目前的写法很可能值得保留，因为两个 CTA 的同步节奏让这个窗口实际非常安全，而且确实有类似的 Blackwell CTA_2 kernel 使用 leader-only `arrive_expect_tx` 的模式。

**追求协议上完全扎实，同时不想每轮 `cluster_sync()`：**改成我上面说的 **`TMA_BARRIER count=2 + 每个 CTA remote arrive_expect_tx(A_BYTES+B_BYTES)`**。这是我认为最漂亮的解法。

我反而**最不推荐**：

```cpp
rank0 expect_tx
cluster_sync();
rank0/rank1 TMA
```

因为为了保护一条很短的控制依赖，每个 K tile 强行让整个 cluster 所有线程 rendezvous，一般得不偿失。

另外你如果想验证“现在确实存在潜在时序依赖”，有个很直接的实验：**故意只在 rank0 的 `arrive.expect_tx` 前插入一段很大的延迟，而 rank1 正常发 TMA**。如果延迟足够大后开始 hang/结果异常，就能把这个平时被 TMA latency 掩盖的窗口人为放大。