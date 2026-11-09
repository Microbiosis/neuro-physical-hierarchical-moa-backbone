# 插件验收测试

验收 `hybrid_moa_vl_operator_suite/deepspeed_megatron_physics_plugin.py` 的两个分布式能力：
ZeRO-3 参数分片/归约，以及 Megatron 序列并行的前向/反向 P2P。

这些测试**不依赖硬件加速器**（`gloo` 后端 + CPU），2 分钟内跑完，用来让 CI 或本地快速复核。

## 1. ZeRO-3：分片 / all-gather / 训练 / reduce-scatter

```bash
python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29701 \
        tests/plugin_acceptance/worker_zero3_gloo.py
```

期望输出（rank 0）：

```
[r0] 1.SHRINK 60 -> 32 (53.3%) status=['PARTITIONED']
[r0] 2.TRAIN OK ngrads=14
[r0] 3.REDUCE_SCATTER 14 shards worst=0.000e+00 OK
  4.SUM shards=-2.67072926 grads=-2.67072926 ratio=-1.00000000
```

四项含义：

| 项 | 验证内容 | 判定 |
|---|---|---|
| `1.SHRINK` | `p.data` 被换成 1/N 分片，实际存储下降 | 60 → 32 元素 |
| `2.TRAIN` | 分片后仍能前向+反向，全部参数拿到梯度 | `ngrads=14` |
| `3.REDUCE_SCATTER` | 本地分片 == 各 rank 梯度之和的对应切片 | `worst = 0.000e+00` |
| `4.SUM` | 分片总和 == 完整梯度总和 | `\|ratio\| = 1.00000000` |

> `ratio` 为负是正常的：`y.sum().backward()` 下各 stage 的雅可比对角为负。判据是 `|ratio| = 1`。

## 2. 序列并行：前向 + 反向跨卡 P2P

需要 2 张 GPU（NCCL）：

```bash
python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29513 \
        tests/plugin_acceptance/worker_sequence_parallel.py
```

期望输出：

```
[r0] FWD ok y=(2, 16, 8) q[0,:2]=[...]
[r1] FWD ok y=(2, 16, 8) q[0,:2]=[...]
[r0] BWD ok |dx|=...
[r1] BWD ok |dx|=...
```

**判据**：两个 rank 的 `q_bound` 必须不同（rank 1 的边界来自 rank 0），
且 stderr 应出现 `Unbatched P2P op (send/recv)` 警告 —— 那是反向 P2P 触发的，是它真在传协态的证据。

## 已知边界

- `gloo` 只支持 CPU；GPU 上的 NCCL 路径需另测。
- 这两个脚本验证的是**通信语义**，不测 kernel 性能。
- 单进程模式下 `DeepSpeedZeROCompatibleWrapper` 用 `_all_rank_shards` 本地拼接模拟多 rank，
  与真实多进程路径共用同一套 `all_gather` / `reduce_scatter` 逻辑分支。
