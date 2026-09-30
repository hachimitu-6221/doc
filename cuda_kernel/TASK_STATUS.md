# CUDA 移植任务核对

用户请求过 v1、v2、v3、v4、v5、v6、v8；没有请求 v7。
这些版本均已有 CUDA 移植、正确性测试及 Modal B200/cuBLAS 实测。
未修改原 Mojo 文件或其他人的 `exp_v8.cu`、`run_exp_v8_modal.py` 等文件。

## 当前结果

BF16 输入/输出、FP32 累加，`M=N=K=4096`；百分比为同场
`100 × cuBLAS耗时 / CUDA耗时`。不同版本可能在不同实例上测试，不能将比例差异
当作严格的跨版本加速比。原 Mojo 编译产物未重跑。

| 版本 | CUDA / cuBLAS 吞吐量 | 说明 |
| --- | ---: | --- |
| v1 | 9.82% | [说明](README.md) |
| v2 | 18.24% | [说明](README_v2.md) |
| v3 | 18.04% | [说明](README_v3.md) |
| v4 | 25.74% | [说明](README_v4.md)；原 benchmark 的 4096×2560×8192 为 22.50% |
| v5 | 81.56% | [说明](README_v5_v6.md) |
| v6 | 83.03% | [说明](README_v5_v6.md)；MMA_N=128 为 80.65% |
| v8 | 112.97% | [说明](README_v8.md)；不是最初低性能移植的旧成绩 |

## 本轮补做

- 排查 v6 未筛选 racecheck 的历史超时，发现双 CTA TMEM 分配前缺少入口同步。
- 为同结构的 v4/v5/v6 添加分配前 cluster 同步，避免 peer CTA 内部 barrier 的初始化竞争。
- 移除 v5/v6 脚本的内核筛选；使用原生 CUDA 12.8 检查工具，不升级编译器或 cuBLAS。
- 在 B200 重跑 v4 两种尺寸、v5、v6 两种 MMA 配置的正确性和性能；当前结果文件的源码 SHA256 全部匹配。
- 旧性能结果及失败诊断移至 `history/`，不把失败或中断的测试算作通过。
- 原生 CUDA 12.8 工具完成 52 项未筛选检查：4 项 memcheck、4 项 synccheck、44 项 racecheck，全部零错误、零警告。

重复审计覆盖 v4、v5、v6 MMA_N=256/128 四种配置；各配置在
`4096×4096×576` 重复 racecheck 10 次，另检查完整 `4096³`。
原始命令、日志、工具版本和源码 SHA256 见 [审计结果](sanitizer_v456_audit.json)，
[Modal 运行](https://modal.com/apps/akseven6221/main/ap-0FfmBou7EVLcxC7Uc5xNSw)。

复现性能及逐版本检查见各 README；独立重复审计入口：

```bash
.venv/bin/modal run kernel/cuda_kernel/run_sanitizer_audit_modal.py --repeats 10
```

所有 Modal 命令都会消耗 B200 GPU 资源。无需改动其他人的文件。

## 额外诊断限制

独立新版 sanitizer 的连续检查仍有卡住和 internal error，原因未完全确定。
没有把它们计为通过；详见 [排查记录](README_v5_v6.md#sanitizer-排查) 和 `history/`。
这与各版本 CUDA 移植、数值检查和 cuBLAS 性能测量的完成状态分开记录。
