# 验证矩阵（2026-11-12）

本仓库 **25 个算子/内核模块**（外加 1 个五维审计脚本与 `__init__.py`，合计 27 个 `.py`，30,676 行，
包导出 234 个符号）。每个算子都自带 `__main__` 自检入口与显式 `__all__`，可单文件拷出独立运行。

本表记录**各算子的自检覆盖与实测数值**，以及**哪些数字是实测、哪些是解析模型**。

---

## 一、逐算子验证覆盖

| 算子 | 自检套件 | 覆盖内容 | 关键实测值 |
|---|---|---|---|
| `local_self_modifying_operators` | 12 项 gradcheck + 五维 | 有理 Hebbian + Delta-Oja 快权重 + SRAM 单 launch | gradcheck 1e-10~1e-09；`\|Y_seq−Y_sram\| = 2.78e-16`；launches 1024→1（HBM 省 93.4%） |
| `llm_hierarchy_operators` | 三引擎对拍 + Gradcheck + 训练冒烟 + 五维 | LLM 六算子 × 六空间 | 前向 ≤6.66e-16；整模型 1.22e-15；Gradcheck 参数 6.87e-09；40 步 AdamW loss 4.8293→0.0050 |
| `spin_chirality_operators` | **P1–P21 物理自检** | Clifford / SU(2) / Larmor / Ward / 幺正 / 反常 / 微观 / ZBW | T4 最大残差 **2.60e-08**（P14 手性守恒）；P12 Triton 三重对齐 2.84e-14 |
| `spin_chirality_lattice4d` | 3+1D 格点指标 | FLSG 单瞬子 Wilson–Dirac 指标 | index = **−1.0**（反瞬子 +1） |
| `nbsc_omega_operator` | 8 项 | 脉冲 + 延迟边 + 肌骨闭环 | 无驱动静默 100 步零脉冲；有驱动 300 步 1368 脉冲；反向可达 3/5 参数张量 |
| `viral_invasion_operator` | 7 项 | DNA 状态机 + 13 条 PMID 约束 | 单纯形 4.44e-16；因果 0.00e+00；两后端 1.34e-07；3/3 人为破坏被拦 |
| `deepspeed_megatron_physics_plugin` | 6 项 | ZeRO-3 + 序列并行 P2P | 分片 60→32 元素；`reduce_scatter` 24 分片 `worst=0.000e+00`；`all_gather` 逐位还原 |
| `moa_operators` | numpy + torch | 跨模态 VL 融合 + MoE 路由 | bilinear 2.2e-16 / megakernel 8.9e-16；自报带宽省 80.3% |
| `moa_triton_kernels` | tile 模拟器 | Triton 融合核的逐 tile SRAM 复算 | 见上 |
| 其余 15 个领域/基础设施算子 | 各带验证套件 | 见各自 `__main__` 输出 | 见 `docs/operator_independence_audit.log` |

---

## 二、仓库级验证（每次提交都跑）

| 检查 | 命令 | 结果 |
|---|---|---|
| 净室 smoke test | `python3 smoke_test.py` | **13/13 PASS**（25/25 模块可独立 import，234 exports） |
| 五维独立性审计 | `python3 hybrid_moa_vl_operator_suite/operator_independence_verification.py` | **ALL 5 DIMENSIONS VERIFIED**（25 条 ISOLATED PASS） |
| 行数一致性 | 扫 README + docs 全部 `.md` 对照 `wc -l` | 不一致 **0** 处 |

### 五维审计的两套口径

| 维度 | 覆盖 | 结果 |
|---|---|---|
| **Dimension I** 文件隔离 | **25 个算子 + 审计自身 = 26 个模块** | 每个 `AST Sibling Deps = 0`，单文件拷 `/tmp` 独立运行 26/26 PASS |
| **Dimension II–V** 五维正交性 | **14 个领域算子** | 秩 14/14；Max VIF 1.91 < 5.0；参数交叉灵敏非对角精确 **0.00e+00**；LOOO R² 0.9997，ΔR² ∈ [+0.0284, +0.0479] |

扩展模型与基础设施不参与 Dimension II–V，因为它们没有统一的参数化接口。

---

## 三、必须标注「未实测」的项

以下数字是**解析模型推算**，不是实测。源码中已用 `[ANALYTIC MODEL]` / `[MODEL-DERIVED]` 标注：

| 项 | 数字 | 性质 |
|---|---|---|
| `adaptive_checkpointed_adjoint` HBM 节省 | >99.4% | `k·B·L·Dh·4` 公式推算，未分配显存 |
| `high_order_pde_ode` H100 加速 | 627× | Roofline 模型；launch 数（9227→2）是精确计数 |
| `distributed_sequence_parallel` 通信隐藏 | 99.87% | `L_chk·D·flops/peak` 推算，未发起集合通信 |

**与之相对，以下是实测的**：检查点伴随 vs 全轨迹伴随 `8.882e-16` / `3.553e-15`；有限差分梯度 `3.6e-10`~`1.05e-09`；
五维审计秩 14/14、非对角 `0.00e+00`、R² `0.9997`。

---

## 四、GPU 专属验证（需 T4，Kaggle）

CPU 无法覆盖的部分，必须在 GPU 上验证：

| 验证项 | Kaggle kernel | 结果 |
|---|---|---|
| LLM 层次算子 Triton flash vs PyTorch 逐元素 | `redmingwei/llm-hierarchy-v020-audit` | fp32 `max\|Δ\|=6.18e-07`、fp16 `8.31e-03`、bf16 `6.54e-02`（各在自身 eps 的 5~7 倍内） |
| LLM 算子完整验证（含 Triton 项） | 同上 | ALL CHECKS PASSED |
| 手性算子 P1–P21（含 P14 手性守恒） | `redmingwei/spin-chirality-v020-audit` | 21 项全绿，最大残差 2.60e-08 |
| 手性 P12 Triton 三重对齐 | 同上 | numpy_vs_torch 2.09e-15 / numpy_vs_triton 1.26e-15 / bwd_dm 2.84e-14 |
| Path B Triton 单 launch vs PyTorch 循环 | `redmingwei/hetero-moe-pathb-triton` | 256–287× |
| ZeRO-3 分片 / all_gather / reduce_scatter | `redmingwei/hetero-moe-zero3-*` | 24 分片 `worst=0.000e+00` |

---

## 五、Triton 兼容性备忘（本仓库踩过的坑）

| 现象 | 结论 |
|---|---|
| `tl.log1p` / `tl.math.log1p` | **不存在**（Triton 3.6 / 3.8）。用 `tl.log(1.0 + tl.exp(-tl.abs(x)))` |
| `libdevice` 的 `exp/cos/sin` 吃 fp16 | **编译期 KeyError: triton.language.float16**。加 `HI: tl.constexpr` 标志，仅低精度升 fp32；**fp32/fp64 必须保持原精度**，否则 1e-13 级断言会降到 fp32 而失败 |
| `tl.dot` 吃 fp64 | **不可用**（LLVM IR 层失败）。需在门控处排除 fp64 |
| 循环体有 `if t > 0:` 分支 | `for t in range(0, L)` 会让 `t` 变运行时值 → 必须 `tl.static_range` |
| Triton 读源码 | 靠 `inspect.getsourcelines`，`exec(compile(...))` 注入会报 `OSError: source code not available`，必须先写真实文件再 import |

---

## 六、复现

```bash
# 仓库级（无 GPU 需求）
python3 smoke_test.py
python3 hybrid_moa_vl_operator_suite/operator_independence_verification.py

# 单算子
python3 hybrid_moa_vl_operator_suite/spin_chirality_operators.py
python3 hybrid_moa_vl_operator_suite/llm_hierarchy_operators.py

# GPU 专属
kaggle kernels output redmingwei/llm-hierarchy-v020-audit -p /tmp/llm
kaggle kernels output redmingwei/spin-chirality-v020-audit -p /tmp/spin
```
