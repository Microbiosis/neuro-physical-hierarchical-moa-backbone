# 插件验证记录（2026-11-09）

> **时效说明（2026-11-12 补）**
>
> 本文档是 2026-11-09 针对 `deepspeed_megatron_physics_plugin.py` 那一批
> 算子的验证记录，结论（4 个缺陷、ZeRO-3 四项验收、序列并行 P2P）**仍然成立**。
> 此后仓库新增了 LLM 层次算子、自旋手性算子（含 3+1D 格点指标层）、
> 病毒核酸侵入算子、NBSC-Ω 算子等，并删除了编排层 `unified_hybrid_moa_architecture.py`。
> **当前全仓库的验证覆盖与数值见 `docs/VERIFICATION_MATRIX.md`**。


对象：`hybrid_moa_vl_operator_suite/deepspeed_megatron_physics_plugin.py`（1882 行）
以及同批新增的 4 个物理/分布式算子。

环境：Tesla T4 ×2（NCCL）、CPU + gloo、torch 2.11.0+cu130 / triton 3.6.0。

---

## 一、实际修复的 4 个缺陷

### 1. `tl.log1p` 在 Triton 中不存在

某版把可用的写法改成了 `tl.log1p(...)`，导致所有 4 个 kernel 编译失败。

实测对照（Triton 3.6.0 与 3.8.0 一致）：

| 写法 | 结果 |
|---|---|
| `tl.log(1.0 + tl.exp(-tl.abs(x)))` fp64 | ✅ 通过，与 NumPy 参考差 8.9e-16 |
| `tl.full` 显式常量 fp64 | ✅ 通过 |
| `tl.log1p(...)` | ❌ `CompilationError` |
| 纯 fp32 | ✅ 通过 |

`tl.log1p` 与 `tl.math.log1p` **都不存在**。已在 docstring 中写明，避免后续再被改回去。

### 2. `for t in range(0, L)` 使 `t` 变成运行时值

`L` 虽是 `tl.constexpr`，但普通 `range` 不静态展开，循环体内的 `if t > 0:` 无法编译。
改为 `tl.static_range(0, L)` 后通过。

### 3. CUDA 路径的 `backward` 未保存上下文

CUDA 前向走 `dispatch_triton_cuda_kernels` 时只设了 `ctx.mode`，
没存 `ctx.caches / ctx.gates / ctx.stage_outs / ctx.x_device / ctx.x_dtype`，
`backward` 解包即崩。已补 `ctx.save_for_backward(...)` 与相应字段。

### 4. `x.requires_grad` 误判导致分片被提前释放

```python
# 错：输入不需要梯度时（如冻结输入、无输入梯度的调用）会释放参数分片
if not (isinstance(x, torch.Tensor) and x.requires_grad):
    self.release_partitioned_parameters()

# 对：只要输出还需要梯度（参数可训练即成立）就不能释放
out_requires_grad = (torch.is_grad_enabled() and isinstance(out, torch.Tensor)
                     and out.requires_grad)
if not out_requires_grad:
    self.release_partitioned_parameters()
```

修复前表现为 `ValueError: operands could not be broadcast together with shapes (2,8,8) (4,)` ——
`backward` 拿到的是 1/2 分片而不是完整参数。

---

## 二、验收结果

### CUDA / Triton 路径（单卡 T4）

| 项 | 结果 |
|---|---|
| 前向 | ✅ 通过，`finite=True` |
| 反向 | ✅ 通过，12 个参数梯度齐全 |
| **GPU vs CPU 前向** | ✅ `max\|Δ\| = 8.882e-16`（fp64 机器精度） |
| 更大 shape（4×64×32） | ✅ 通过 |

### ZeRO-3（2 rank，gloo）

| 项 | 结果 |
|---|---|
| 分片省内存 | ✅ 60 → 32 元素（53.3%） |
| 分片后训练 | ✅ `ngrads=14` |
| `reduce_scatter` 归约 | ✅ 24 个分片全部 `worst = 0.000e+00` |
| 总量守恒 | ✅ `\|ratio\| = 1.00000000` |

### Megatron 序列并行（2 rank，NCCL）

| 项 | 结果 |
|---|---|
| 前向 P2P（边界传递） | ✅ 两 rank `q_bound` 不同 |
| **反向 P2P（协态回传）** | ✅ 4 收 4 发，`_DistributedSequenceParallelAutogradFn` 连接计算图 |
| rank 0 梯度含下游贡献 | ✅ `‖ΔdX‖ = 4.16 > 0` |
| 边界依赖内部 token | ✅ 改 t=0 使 `‖Δq‖ = 1.77 > 0` |

---

## 三、必须标注的"未实测"项

以下数字来自**解析模型**，不是实测。相关输出已加 `[ANALYTIC MODEL]` / `[MODEL-DERIVED]` 标注：

| 文件 | 数字 | 性质 |
|---|---|---|
| `adaptive_checkpointed_adjoint_megakernels.py` | HBM 激活内存节省 99.4%+ | 1,583 |
| `high_order_pde_ode_megakernels.py` | H100 627× 加速 | 1,878 |
| `distributed_sequence_parallel_adjoint.py` | Hiding 99.87% | 853 |

与之相对，以下数字**是实测的**：
- 检查点伴随 vs 全轨迹伴随：`8.882e-16` / `3.553e-15`
- 4 个算子的有限差分梯度：`3.6e-10` ~ `1.05e-09`
- 五维独立性审计：秩 4/4、非对角 `0.00e+00`、R² `1.000000`

---

## 四、复现方式

```bash
# ZeRO-3（无 GPU 要求，2 分钟）
python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29701 \
        tests/plugin_acceptance/worker_zero3_gloo.py

# 序列并行（需 2 GPU）
python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29513 \
        tests/plugin_acceptance/worker_sequence_parallel.py
```

## 五、一处仍未覆盖

`reduce_scatter` 走的是 `dist.reduce_scatter_tensor`，在 gloo 上通过。
**NCCL 上的 `reduce_scatter_tensor` 未单独验证**（GPU 队列当时排队 68 分钟未启动）。
代码路径与 gloo 分支相同，但如需严格起见可另跑一次。
