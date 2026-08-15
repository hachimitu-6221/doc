### 1. `cp.async` — 异步拷贝原语
这是 PTX 提供的**异步内存拷贝指令族**。它的核心特点是：
- **不占用 SMs 的执行单元**，由专门的硬件单元（Async Copy Engine）执行
- **非阻塞**：发起拷贝后，SM 可以继续执行其他指令
- 需要后续用 `cp.async.commit_group` + `cp.async.wait_group` 来同步
### 2. `.cg` — Cache Global 策略
表示对 Global Memory 的加载使用 **cache-global** 策略：
- 数据会经过 L2，但**不进入 L1 Cache**（避免污染 L1）
- 适合**只读一次**或**顺序访问**的数据
### 3. `.shared.global` — 传输方向
- **Source**：Global Memory（`.global`）
- **Destination**：Shared Memory（`.shared`）
- 方向是 `global → shared`
### 4. `.L2::128B` — L2 Cache 提示（Prefetch Hint）
这是 **Hopper 架构引入的 L2 Cache residency control**：
- 告诉硬件：这个访问的**缓存粒度是 128 字节**
- 有助于 L2 更好地做 prefetch 和 cache line 管理
- 可选值还有 `L2::64B`、`L2::256B` 等，需与访问模式匹配
```asm
asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
    :                    // ← 输出操作数（空，这条指令不写回 C 变量）
    : "r"(smem_ptr),     // ← 输入操作数 1 → PTX 中的 %0
      "l"(gmem_ptr),     // ← 输入操作数 2 → PTX 中的 %1
      "n"(128)           // ← 输入操作数 3 → PTX 中的 %2
);
```
