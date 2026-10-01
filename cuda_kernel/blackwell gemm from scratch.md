## v1 naive blackwell gemm

下面沿着你笔记的顺序讲下去。你原来有几处理解不对，我在讲到的地方直接改正，并标上 **【更正】**。

---

## v1 naive blackwell gemm

### 0. 先弄清这个 kernel 在算什么

计算 `C[M,N] = A[M,K] @ B[N,K]ᵀ`。每个 CTA 有 128 个线程（4 个 warp），负责 C 的一块 **64×256** 的输出 tile。grid 是 `(N/256, M/64)`（[v1.cu:188](v1.cu#L188)），以 4096³ 为例，一共 16×64 = 1024 个 CTA。

每个 CTA 沿 K 方向循环，每轮处理 64 个 K：

- 搬 A 的一块 64×64，共 8192 字节。
- 搬 B 的一块 256×64，共 32768 字节。
- smem 里只有这**一份**缓冲，每一轮都会覆盖上一轮的内容。

### 1. barrier 初始化

thread 0 初始化两个 mbarrier（[v1.cu:54-58](v1.cu#L54-L58)）：

- `tma_barrier`：用来等"TMA 把数据搬完了"。
- `mma_barrier`：用来等"Tensor Core 把这轮算完了"。

两个 barrier 的 init 参数都是 **1**，意思是"每一轮需要 1 次 arrive"。这个数字是 **arrive 的次数**，不是"TMA 指令的条数"，后面讲 TMA 时会看到它们的区别。

接着 `fence.mbarrier_init.release.cluster` 把初始化结果发布出去，让之后访问这两个 barrier 的线程和异步硬件看到的是初始化好的状态。

不过光有这个 fence 还不够：它只保证初始化的结果能被看到，**不会让其他线程等 thread 0 做完初始化**。其他线程可能在 thread 0 还没 init 时就去 wait，所以后面还需要一次 `__syncthreads`。

### 2. TMEM 分配

**【更正】** 这一步是**分配**，不是初始化：它只申请空间，不清零。

warp 0 的 32 个线程一起执行 `tcgen05.alloc`（[v1.cu:59-63](v1.cu#L59-L63)）。指令带 `.sync.aligned`，表示整个 warp 必须一起执行。

TMEM 是 Tensor Core 专用的累加器内存，按 128 行（lane）× 512 列组织，每格 32 bit。这里申请了全部 512 列，但实际只用到 256 列（64×256 的 fp32 结果）。

分配到的 TMEM 地址**不会返回到寄存器里**，而是由硬件写进 smem 的 `allocation_address` 位置。其他线程要从 smem 把它读出来才能用。

### 3. 为什么需要块内同步（[v1.cu:64](v1.cu#L64)）

有两个理由，各对应上面一步：

1. **barrier 必须先 init 好，才能被使用。** 后面 128 个线程都会 `wait_barrier`。如果 thread 1 在 thread 0 执行 init 之前就去 wait，等的就是一块未初始化的内存。
2. **TMEM 地址必须先写进 smem，才能被读取。** [v1.cu:66](v1.cu#L66) 每个线程都要从 smem 读 TMEM 地址。如果这时 warp 0 还没分配完，读到的就是垃圾值。

`__syncthreads` 让所有线程都等到这两件事完成。

### 4. 主循环第一步：发起 TMA（[v1.cu:73-80](v1.cu#L73-L80)）

**【更正】** 这一段是你原来理解偏差最大的地方。先看准确的过程，再看账本。

thread 0 先执行这一条：

```
mbarrier.arrive.expect_tx.release.cta  [tma_barrier], 40960
```

它**一条指令同时做了两件事**：

- **arrive 一次**：待 arrive 的次数从 1 变成 0。
- **预告字节数**：把 barrier 的 tx 计数加上 40960（也就是 8192 + 32768）。

所以"预期收到多少字节"是**这条指令自己设定的**，不是 TMA 设定的。

名字里的 `.release` 也不是指令名，而是**内存序语义**：在这条指令之前本线程做过的内存写入，对之后在这个 barrier 上 wait 成功的线程可见。

然后 thread 0 发出两条 `cp.async.bulk.tensor`（即 TMA），分别搬 A 和 B。每条 TMA 只做一件事：**把数据写进 smem，写进多少字节，就对 barrier 做一次 `complete_tx` 扣掉多少**。

TMA **不会**通知"expect 已经发出"，也**不会**设定预期字节数。

一个相位完成需要两个条件同时满足：**待 arrive 次数 = 0** 且 **tx 计数 = 0**。

用第 0 轮记一次账：

```
                         待 arrive   tx 计数   相位 0
init(1)                     1          0       未完成
arrive.expect_tx(40960)     0      +40960      未完成（字节还没到）
TMA 陆续把 A、B 写进 smem   0      递减中...   未完成
最后一个字节到达            0          0       完成 → 切到相位 1
```

为什么要先 arrive，再发 TMA？如果顺序反过来，有可能数据先到了一部分，而这时 expect 还没加上去，但因为待 arrive 次数还是 1，相位不会提前完成。所以 init 里的那个 1，本质上是一道"等 expect 登记完才能完成"的闸门。

**顺带一提 TMA 的坐标。** v1 用的是 3D tensor map，box 是 `{8, 行数, 8}`（[v1.cu:174-176](v1.cu#L174-L176)）。它把 K 方向切成"每组 8 个元素 × 8 组"，搬进 smem 后的布局是 `[K组 0..7][行][8 个元素]`。这正是后面 MMA 在不 swizzle 时要求的布局：每个"核心矩阵"是 8 行 × 16 字节。坐标里的 `iteration * 8` 就是第几组 K/8，换算成 K 的偏移是 iteration × 64。

### 5. 所有线程等 TMA 完成（[v1.cu:81-82](v1.cu#L81-L82)）

`wait_barrier(tma_barrier, phase)` 会一直自旋，直到对应的相位完成，也就是数据全部落进 smem。

严格来说只有 thread 0 需要等，因为只有它发 MMA；其他线程陪着等也不会出错。

紧接着的 `tcgen05.fence::after_thread_sync` 是固定搭配：刚通过一次同步，接下来要发 tcgen05 指令时，用它保证 tcgen05 指令排在这次同步之后执行。

### 6. thread 0 发 4 条 MMA（[v1.cu:84-95](v1.cu#L84-L95)）

**每条 MMA 算的是 M=64、N=256、K=16。** 这个形状写在 instruction descriptor 里（[v1.cu:67-69](v1.cu#L67-L69)）：

| 位 | 值 | 含义 |
|---|---|---|
| `1<<4` | 1 | D（累加器）是 FP32 |
| `1<<7`、`1<<10` | 1 | A、B 都是 BF16 |
| `(256/8)<<17` | 32 | N = 256 |
| `(64/16)<<24` | 4 | M = 64 |

`kind::f16` 这类 MMA 每条指令固定处理 K=16。一轮要处理 BLOCK_K=64，所以发 4 条，每条往 K 方向前进 16。**一轮下来就是一次 64×256×64 的矩阵乘**，结果累加在 TMEM 里。

每条 MMA 做的事是：从 smem 读 A 的 64×16 和 B 的 256×16，在 Tensor Core 上计算，把结果加到 TMEM 里 D 的 64×256 上。B 在 smem 里是按 N×K 存的，所以最终算出来的是 A @ Bᵀ。

**A 和 B 的地址用 64-bit descriptor 描述**（[v1.cu:35-40](v1.cu#L35-L40)）：

- `address >> 4`：smem 起始地址，以 16 字节为单位。
- `rows << 16`：K 方向上相邻两个核心矩阵之间隔多少字节（行数 × 16 字节，右移 4 位后就等于 rows）。
- `8 << 32`：M/N 方向上相邻两组 8 行之间隔 128 字节（右移 4 位后是 8）。
- `1 << 46`：Blackwell descriptor 的版本位。

每走一步 K=16，也就是 2 组 K/8，地址就前进 `2 × rows × 16 = rows × 32` 字节。代码里的 `step * BLOCK_M * 32` 就是这么来的。

**`accumulate` 标志：** 只有整个 K 循环的第一条 MMA 是 D = A×B（覆盖），之后都是 D += A×B（累加）。因为 TMEM 分配时没有清零，第一条必须覆盖掉里面的旧值。

### 7. commit，然后所有线程等 MMA 完成（[v1.cu:96-100](v1.cu#L96-L100)）

`tcgen05.mma` 是异步的：thread 0 发完 4 条，指令就立刻返回了，这时 Tensor Core 可能还没开始算。

`tcgen05.commit ... [mma_barrier]` 的意思是："**我之前发出的所有 tcgen05 异步操作都执行完之后**，请硬件替我往 mma_barrier 上 arrive 一次。"mma_barrier 的 count 是 1，所以这一次 arrive 就会让相位完成。

**为什么一定要等 MMA 完成？** smem 只有一份缓冲。下一轮 TMA 会直接覆盖 A 和 B，而 Tensor Core 这时可能还在读这一轮的数据。不等的话，这一轮就会读到下一轮的数据，结果算错。

### 8. `phase ^= 1`（[v1.cu:101](v1.cu#L101)）

两个 barrier 每一轮都恰好完成一次：

- 第 0 轮等的是第 0 次完成（奇偶 0）。
- 第 1 轮等的是第 1 次完成（奇偶 1）。
- 第 2 轮又是奇偶 0，依此类推。

两个 barrier 步调完全一致，所以共用一个 `phase` 变量就够了。

这就是后来 `Pipeline` 结构体的最简形态：**只有 1 个槽，index 永远是 0，只剩下 phase 在翻转。**

### 9. 收尾：TMEM → 寄存器 → 显存

**第一步，把结果从 TMEM 读进寄存器**（[v1.cu:104-150](v1.cu#L104-L150)）

最后一轮对 mma_barrier 的等待已经保证全部 MMA 都算完了，之后再补一条 `after_thread_sync` fence。

然后每个 warp 用 `tcgen05.ld.16x256b.x32` 读自己那部分结果：

- TMEM 地址的高 16 位是 lane（行），低 16 位是列。`tmem + (warp*32 << 16)` 表示第 w 个 warp 从第 32w 个 lane 开始读。
- **硬件限制：** 第 w 个 warp 只能访问 lane 32w 到 32w+31。
- 在 M=64 的布局下，每个 warp 负责 16 行结果。`16x256b` 表示一次读 16 个 lane × 8 列 fp32，`.x32` 重复 32 次，覆盖 256 列。
- 所以每个线程拿到 16×256/32 = **128 个 float**。

`tcgen05.wait::ld` 等读取真正完成，之后寄存器里的值才能用。

**第二步，释放 TMEM**（[v1.cu:151-157](v1.cu#L151-L157)）

`__syncthreads` 保证 4 个 warp 都读完 TMEM，然后 warp 0 才能 dealloc 归还 TMEM。

`relinquish_alloc_permit` 的意思是声明"这个 CTA 以后不会再申请 TMEM"。

**第三步，写回显存**（[v1.cu:159-169](v1.cu#L159-L169)）

每个线程把两个 fp32 打包成一个 `bf16x2`，直接写进全局内存。每个值落在哪一行哪一列，由 `16x256b` 的寄存器布局决定：

- `lane/4` 决定行，`row_tile` 选择 +0 行还是 +8 行。
- `lane%4` 决定这 8 列里的哪两列。

### 10. 为什么 v1 慢（只有 cuBLAS 的约 10%）

一轮的时间线是这样的：

```
TMA 搬数据 ──等──► MMA 计算 ──等──► TMA 搬数据 ──等──► MMA 计算 ...
```

**搬数据和计算完全串行**，Tensor Core 有一半时间在等数据。另外 v1 占用了全部 512 列 TMEM，同一个 SM 上的其他 CTA 申请不到 TMEM，没法和它并行。

v5 的多级 `Pipeline` 解决的就是第一个问题：smem 分成多份缓冲，TMA 往第 k+1 份写的同时，MMA 在读第 k 份。这时候一个 `phase` 变量就不够用了，需要每个槽各有一对 FULL/EMPTY barrier，于是就有了 `(index, phase)` 这个游标。

---

读完可以用这两个问题自测：

1. 如果把 `tma_barrier` 的 init count 改成 2，第 0 轮会怎样？（答：永远等不到完成，因为每轮只有 1 次 arrive。）
2. 如果 `wait_barrier(mma_barrier)` 只让 thread 0 执行，其他线程不等，哪一轮会出问题？（提示：看第 9 节的 `tcgen05.ld`。）