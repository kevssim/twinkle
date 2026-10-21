# Delta vs Full 权重同步性能

测试环境：Qwen3.5-9B（BF16，整模型 17.53 GB / 760 个张量），2 卡 FSDP2 trainer + 2 卡 rollout（TP=2），跨机 HCCL，bucket=2048MB。
`delta time` 为发送侧传输耗时（算 diff + 跨 rank gather-v + 编码 + HCCL 广播），`full payload` 为每轮全量同步实际广播的字节数。

| changed_ratio | delta payload | delta time | full payload | reduction vs full |
| ------------- | ------------- | ---------- | ------------ | ----------------- |
| 0.5%          | 269.2 MB      | 3.59 s     | 17.53 GB     | 66.7x             |
| 1.0%          | 538.4 MB      | 4.31 s     | 17.53 GB     | 33.3x             |
| 2.0%          | 1076.9 MB     | 5.62 s     | 17.53 GB     | 16.7x             |
| 3.0%          | 1615.3 MB     | 6.76 s     | 17.53 GB     | 11.1x             |

说明：
- `reduction vs full` = full payload / delta payload，即 delta 相比全量同步减少的网络传输量。
- full payload 每轮恒为整模型 17.53 GB（本环境链路 ~9.5 GB/s，全量广播约 1.8 s）。
- delta payload 每个变化元素 6 字节（int32 位置 + bf16 值），随改动比例线性增长。
- delta time 可近似分解为「固定开销 ~2.8 s（diff 全扫 + gather + 逐参数）+ payload / ~400 MB/s」。

# 端到端单 step 对比（Qwen3.5-4B）

环境：2 卡 FSDP2 trainer + 2 卡 rollout（TP=2），batch=2 × seq=2048，AdamW lr=1e-5，bucket=2048MB。
一个 step = 训练计算（前向 + 反向 + AdamW）+ 权重同步（阻塞到 rollout 就绪）。

| backend | train (s) | sync (s) | step (s) | changed_ratio | payload (MB) |
| ------------- | --------- | --------- | --------- | ------------- | ------------ |
| delta_sharded | 3.670     | 36.998    | 40.927    | 14.11%        | 4179.3       |
| full          | 3.677     | 3.098     | 6.775     | -             | 9870.5       |

此工作点下 delta 慢于 full（speedup 0.17x）：payload 只降 2.4x，而稀疏路径吞吐约为 dense 广播的 1/10 ~ 1/20。
改动比例由优化器配置与训练阶段决定，不是 delta 实现的属性；上游参考实现在稳态 RL 下报的是 1 ~ 2.3%。

## 耗时分解（取上表那一步，sync 37.0 s）

发送端 wire_time 与接收端 apply 是流水重叠的（flushes=2），因此各项之和大于 37 s。

| 阶段 | 耗时 (s) | 占 sync |
| ---------------------------- | -------- | ------- |
| 接收端 apply（decode+scatter） | 27.0     | 73%     |
| 发送端 broadcast               | 11.0     | 30%     |
| assemble（编码 + checksum）    | 5.2      | 14%     |
| produce（diff 全扫 + 提取）    | 4.7      | 13%     |
| gather（跨 rank）              | 2.5      | 7%      |
| 接收端 broadcast_recv          | 0.1      | ~0%     |

两个大头：
- **接收端稀疏 scatter**：4179MB / 27.0s ≈ 155 MB/s。按索引散写是随机访存，9B 上实测约 110 MB/s，同一量级，属稀疏路径的固有吞吐。
- **发送端 broadcast**：4179MB / 11.0s ≈ 380 MB/s，而 full 的连续广播为 9870MB / 3.1s ≈ 3200 MB/s，慢约 8 倍。delta 把 positions 与 values 分两次广播，且每个 flush 后 `torch.npu.synchronize()`；full 是单块连续 buffer 一次推完。payload 小且单 flush 时（9B 的 1GB 场景）broadcast 可忽略，payload 大且多 flush 时才显现。

## 改动比例随训练下降

同一份固定数据反复训练，梯度收敛，更新量递减：

| step | changed_ratio | payload (MB) | sync (s) | step (s) |
| ------ | ------------- | ------------ | -------- | -------- |
| warmup | 21.64%        | 6408.5       | 56.1     | 64.2     |
| real 2 | 15.83%        | 4686.8       | 42.1     | 45.6     |
| real 3 | 14.11%        | 4179.3       | 37.0     | 40.9     |
| real 4 | 12.65%        | 3744.4       | 31.2     | 34.9     |

14.11% 不是稳态值，只是该合成训练设置的早期状态。

## 优化优先级

1. 接收端 scatter（73%）：合并 kernel、连续段走整段 copy、避免 int64 索引中转。
2. 发送端 broadcast（30%）：positions 与 values 合并为一次广播、去掉每 flush 的强制同步、增大 bucket 以减少 flush 数。
3. assemble 的 checksum（14%）：verify 关闭时跳过。
4. produce 的 diff 全扫（13%）：按 optimizer 标记的 dirty 参数增量化，避免每次 O(模型) 扫描。
5. gather（7%）：此规模无需改动。

