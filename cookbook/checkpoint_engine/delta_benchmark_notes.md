# Delta 权重同步实验记录

## 背景

- 起因：`delta_validation.py` 在 `--tp 2` 时崩溃，`--dp 2` 正常。
- 修复后验证通过，但 delta 同步比 full 同步慢，甚至是负优化。
- 目标：先在 verl 上跑出 baseline，再把 twinkle 的脚本对齐 verl，比较两边。

## 环境

- HAMi vGPU（libvgpu 拦截 `cuMemCreate`），所以绝对耗时不一定能外推到裸机。
- trainer 2 卡（FSDP2），rollout 2 卡（vLLM TP=2）。
- 模型：Qwen3-8B（`/nas/disk1/Qwen3-8B`）。

## 1. tp=2 崩溃

- 现象：NCCL 通信组初始化时报 `Illegal device id` 并 segfault。
- 原因：HAMi 环境下 NCCL 的 cuMem 路径（`NCCL_CUMEM_ENABLE`、`NCCL_CUMEM_HOST_ENABLE`）不可用，tp>1 时 vLLM 才会初始化 NCCL，所以 dp=2 不受影响。
- 处理：在 `delta_validation.py` 开头用 `os.environ.setdefault` 把两个变量设为 `0`，与 verl 的做法一致，外部可覆盖。
- 结果：重跑后 status=passed。

## 2. twinkle 早期观察（Qwen3.5-0.8B，bucket 64MB）

- delta 耗时随 flush 数增长。每个 flush 都要新建 sender/receiver、两次 NCCL broadcast、checksum 同步、Ray RPC 和 vLLM `collective_rpc`。
- 把 bucket 调大后：seed 10.1s 降到 1.9s；50% 和 100% 变化的用例从 11.1s、16.7s 降到约 2.7s。
- 但 1% 到 10% 变化仍然是 3.2 到 3.5s，full 只要 1.0 到 1.6s。0% 约 1.0s。
- 结论：存在 2 到 3 秒的固定开销，与 flush 数无关。来源未定位。

## 3. 数据量原理

- 稀疏条目每个变化元素 6 字节（int32 位置 + bf16 值），full 同步每个元素 2 字节。
- 变化比例超过约 1/3 时，delta 的传输量就比 full 大。
- 真实 RL 单步变化约 1% 到 3%（verl 文档在 Qwen3-8B 上测得 0.884% 到 0.895%），所以比例改成 0.5%、1%、2%、3%。

## 4. 脚本

| 脚本 | 说明 |
|---|---|
| `verl/tests/checkpoint_engine/bench_delta_vllm.py` | verl baseline。FSDP2 trainer（`forward_only=True`，`model_dtype=bfloat16`），随机翻转 bf16 最低位模拟变化，计时整个 `update_weights()` |
| `cookbook/checkpoint_engine/delta_benchmark.py` | twinkle 对齐脚本。先 seed，再按比例同步，计时整个 `sync_weights()`，输出 median/min/max 表 |

运行：

```
# verl
python tests/checkpoint_engine/bench_delta_vllm.py --model /nas/disk1/Qwen3-8B --backend delta_sharded
python tests/checkpoint_engine/bench_delta_vllm.py --model /nas/disk1/Qwen3-8B --backend nccl

# twinkle
python -m cookbook.checkpoint_engine.delta_benchmark --model /nas/disk1/Qwen3-8B --backend delta_sharded
python -m cookbook.checkpoint_engine.delta_benchmark --model /nas/disk1/Qwen3-8B --backend full
```

verl 脚本调试中修过的问题：

- 缺少 flash-attn：改用 `attn_implementation=sdpa`，并设置 `use_remove_padding=False`。
- `perturb` 断言 bf16 失败：verl 的 `model_dtype` 默认是 `fp32`，需要显式设为 `bfloat16`。
- `perturb` 是 blocking 调用，不能再套 `ray.get`。

## 5. verl baseline（Qwen3-8B，bucket 2048MB，每档 2 次）

seed（delta_sharded）：256.9s，传输 15622MB，7 个 flush。

| 比例 | delta 中位数 (min~max) | nccl 中位数 (min~max) | delta 传输量 |
|---|---|---|---|
| 0.5% | 38.3s (33.9~42.6) | 21.9s (14.8~29.1) | 234MB |
| 1% | 27.6s (23.7~31.5) | 28.8s (27.4~30.2) | 469MB |
| 2% | 17.8s (15.7~19.8) | 7.3s (6.2~8.4) | 937MB |
| 3% | 34.8s (27.9~41.8) | 6.5s (5.7~7.3) | 1406MB |

日志字段说明（以 0.5% 为例）：

- `flipped` 和 `changed_elems` 相等，说明 delta 检测正确。
- `payload_mbytes` = 变化元素数 × 6 字节，与预期吻合。
- 变化量都小于 2048MB，所以每次只有 1 个 flush。

## 6. 结论

能确定：

- verl 自己的 delta 在这个环境里也不比 nccl 快，负优化不是 twinkle 特有的。
- nccl 每次传输量固定约 15.6GB，耗时却在 5.7s 到 30s 之间，环境本身波动约 5 倍。
- delta 的耗时与传输量无关：传 937MB 最快，传 234MB 反而最慢。传 234MB 要 34 到 42s，这部分不是带宽造成的。
- 2% 和 3% 档，nccl 比 delta 快 2.4 到 5 倍。

不能确定：

- 每档只有 2 个样本，没有统计意义。
- 固定开销花在哪一步未知。候选：HAMi 对 CUDA IPC 的影响、ZMQ 和 Ray RPC、暂停和恢复生成、释放和恢复 KV cache。均未测量。
- 机器上是否有其他任务占用 GPU 或 NAS，未排查。

## 7. 已知的口径差异（twinkle 与 verl）

- 变化位置：twinkle 的 `change_fraction` 翻转每个参数前 N% 元素的符号位，verl 随机翻转最低位。变化数量相同，位置分布不同。
- CPU offload：verl 的 `forward_only=True` 会启用，twinkle 默认不启用。对齐需加 `--strategy accelerate --cpu-offload`。
- 分片 dtype：verl 现在是 bf16；twinkle 取决于 transformers 版本的默认加载 dtype，未核实。传输字节数不受影响。
- `delta_benchmark.py` 不做 `forbid_full_tensor` 检查。

## 8. twinkle 结果（Qwen3-8B，bucket 2048MB，tp=2，每档 2 次）

| 比例 | delta 中位数 (min~max) | trainer `send_seconds` | full 中位数 (min~max) | delta 传输量 |
|---|---|---|---|---|
| 0.5% | 8.4s (7.2~9.6) | 0.44s | 1.3s (0.96~1.6) | 234MB |
| 1% | 9.8s (9.1~10.6) | 0.44s | 0.9s (0.83~1.0) | 469MB |
| 2% | 11.6s (11.2~12.1) | 0.44s | 1.2s (0.86~1.5) | 937MB |
| 3% | 10.0s (9.0~11.1) | 0.71s | 1.3s (0.80~1.7) | 1406MB |

- 传输量和 verl 完全一致（0.5% 档 234.34MB、1 个 flush），说明编码和分桶等价。
- trainer 侧（diff、gather、broadcast）只要 0.44~0.71s，不是瓶颈。
- delta 比 full 多出的 7~11s 都在 trainer `send_seconds` 之外，且随传输量有增长趋势（0.5%→2% 约 4.7s/GB）；3% 比 2% 低，属于 2 样本噪声。
- 嫌疑（均未测量）：`begin_delta_sync` 里 `sleep(level=1)` + `wake_up(['weights'])`（full 路径没有）、每个 flush 新建 ZMQ/IPC、接收端 `clone` + checksum `.item()` + 稀疏 patch、`_clear_delta_cache` 与 `wake_up(['kv_cache'])`。

## 9. verl 与 twinkle 对比

| 比例 | verl delta | verl nccl | twinkle delta | twinkle full |
|---|---|---|---|---|
| 0.5% | 38.3s | 21.9s | 8.4s | 1.3s |
| 1% | 27.6s | 28.8s | 9.8s | 0.9s |
| 2% | 17.8s | 7.3s | 11.6s | 1.2s |
| 3% | 34.8s | 6.5s | 10.0s | 1.3s |

- 两边 delta 都比各自的全量同步慢，负优化不是 twinkle 特有的。
- twinkle 的 delta 绝对耗时反而更低（8~12s 对 18~38s）；twinkle 的 delta/full 比例更差，主要因为 twinkle full 基线只有约 1s。
- 计时范围不一致，不能直接下结论：verl `update_weights()` 包含终止请求、释放/恢复 KV cache、重建进程组、恢复生成（来自代码阅读，未逐行核对）；twinkle delta 含暂停生成、sleep/wake_up、清缓存、恢复生成；twinkle full 只有 `sync_weights()`，没有 sleep/wake_up，所以最轻。
- 其他口径差异见第 7 节。

## 10. verl 官方文档的测法（`docs/advance/delta_weight_sync.md`，"Measured results"）

- 没有独立 benchmark 脚本，数据来自真实 GSM8K GRPO 训练中每步稳态权重同步的计时，`delta_sharded` 与 `nccl` 各跑一遍。
- 环境：H100 或 A100/A800 多节点，训练与 rollout 分离（多为 2+2 节点），SGLang rollout。部分行为 50 步中位数或 50 步持续运行。
- 变化比例由训练自然产生（稠密模型约 0.884%~0.895%），不是人为设定。
- 正确性另测：200 步 GRPO 奖励曲线对比、perturb/apply/revert/apply 位级一致。
- 官方 Qwen3-8B 数据：2+2 节点 FSDP2 delta 2.24~3.40s，nccl 34~38s；1+1 节点 delta 3.00s，nccl 11.00s。Qwen3-0.6B 单机 NVLink 为 0.59s 对 0.61s（持平）。
- 文档自己说明：快链路加小模型时，全量广播本来就便宜，delta 的优势随模型规模和网络距离增长。
- 与我们测试的差异：官方是多节点、走网络、真实训练循环；我们是单机 4 张 HAMi 虚拟卡、合成扰动、微基准。因此单机上 delta 不占优不与文档矛盾。
- 官方 verl delta 在 8B 上约 2.2~3.4s，我们的 verl delta 是 18~38s，twinkle 是 8~12s；这一差距的原因未查明（HAMi、单机部署、计时范围都有可能）。

## 11. 待办

- 决定：改为参照官方做法，用真实 GRPO 训练做 delta 与 full 的对比（用户已确认方向，具体方案待定）。
- 跑之前检查 `nvidia-smi`，确认没有其他任务占卡。
- 在 delta 路径加分阶段计时（trainer 的 diff/gather、broadcast、接收端 IPC 与 vLLM patch、前后的暂停恢复），定位 7~11s 的来源。涉及 `checkpoint_engine/delta/engine.py`、`delta/trainer.py` 和 `delta/rollout.py`，需先确认。
- 可选：给 `delta_benchmark.py` 加 `--no-free-cache`（`free_cache_engine=False`），看 sleep/wake_up 占多少。
- 可选：把 NCCL 环境变量移到库里（`vLLMSampler.__init__`），让所有使用者受益。
