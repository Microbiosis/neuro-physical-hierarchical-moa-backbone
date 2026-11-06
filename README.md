# Neuro-Physical Hierarchical MoA Backbone

The Mixture-of-Architectures (MoA) backbone connects **15 domain operators in seven
stages**. Each domain operator has its own parameters and interface. The unified
layer connects their outputs to the next computation, from the physical and
molecular substrate through sensorimotor control, cortical processing, silicon
execution, agency, and sleep consolidation.

The package supports NumPy analytical execution and PyTorch autograd. GPU kernels
are available for specific operators. The 16 operator/kernel files can be run
independently of the unified layer.

---

## Layout

```
.
├── LICENSE / NOTICE                  Apache-2.0 + provenance
├── smoke_test.py                     clean-room verification
├── README.md                         本文件
└── hybrid_moa_vl_operator_suite/     算子包（Python package）
    ├── __init__.py                   动态发现与导出算子
    ├── README.md                     包内说明（原始版本文档）
    ├── quantum_resource_phasing_operators.py   量子资源相位（1,065 行）← 新增
    ├── cpu_gpu_self_bootstrapping_operators.py  Tri-ISA CPU + SIMT GPU 自举（2,285 行）
    ├── unified_hybrid_moa_architecture.py   七阶段骨干与角色消融审计
    ├── viral_nucleic_acid_invasion_operators.py   Stage I.D 分子态调制算子
    ├── moa_operators.py              系统级 MoA + 跨模态图文融合（978 行）
    ├── sleep_subconscious_neuro_operators.py    睡眠/潜意识（1,007 行）
    ├── operator_independence_verification.py    五维独立性审计（609 行）
    ├── moa_triton_kernels.py         Triton 内核 + tile 模拟器（390 行）
    ├── vacuum_zpe_fluctuation_operators.py      真空零点能涨落（928 行）
    └── … 其余 10 个领域算子模块
```

Two import styles are supported:

```python
# 1) 作为包导入（推荐）
from hybrid_moa_vl_operator_suite import MoAOperatorConfig, NumpyMoAOperatorKernels
from hybrid_moa_vl_operator_suite.moa_operators import CrossModalVisionLanguageFusionSubOp

# 2) 平目录 / 直接跑单个模块的验证套件
#    python3 hybrid_moa_vl_operator_suite/unified_hybrid_moa_architecture.py
#    sys.path.insert(0, "hybrid_moa_vl_operator_suite"); import unified_hybrid_moa_architecture
```

Import mode is selected with `if __package__:` (not `try/except ImportError`), so a
missing-symbol `ImportError` surfaces with the real module and symbol name instead of
being masked by the fallback branch. The standalone modules each import on their
own in an empty directory, as checked by `smoke_test.py`. The original modules
also have a separate five-dimensional independence audit.

## Modules

| Module | Domain | Lines |
|---|---|---|
| `cpu_gpu_self_bootstrapping_operators.py` | Tri-ISA CPU（x86-64 CISC/TSO/ROB · ARM64 RISC/LDAR-STLR/SVE · RISC-V RV64GC/RVV/RVWMO）+ SIMT GPU warp 调度 + 自举编译 | 2,285 |
| `quantum_resource_phasing_operators.py` | 量子资源相位：Coulomb 相互作用定点逆平方根相位、DQI 可靠性 1/R(ε)、Hamiltonian 分块编码 λ_eff 归一化门 → **Stage I.C** | 1,065 |
| `unified_hybrid_moa_architecture.py` | 七阶段总装、多层骨干、解析执行与角色审计 | — |
| `viral_nucleic_acid_invasion_operators.py` | Stage I.D 分子场，含六个核酸入侵启发的子算子、并列感知 top-k 路由、精炼与自适应停机 | — |
| `operator_independence_verification.py` | 16 个独立文件、15 个领域算子的五维独立性审计 | — |
| `sleep_subconscious_neuro_operators.py` | Borbély 两过程睡眠调节、VLPO/ARAS 翻转开关、Tononi SHY、类淋巴清洗、SWR/PGO 梦境重组 | 1,007 |
| `moa_operators.py` | 系统级 MoA：异构 proposer / 逐头注意力 / 聚合 + 跨模态图文融合 | 1,169 |
| `corticospinal_chaotic_operators.py` | 皮质-脊髓混沌通路、Pyragas 时滞反馈镇定 | 938 |
| `spinal_cord_pathway_operators.py` | 脊髓节段通路、Melzack-Wall 门控、Renshaw 递归 | 936 |
| `vacuum_zpe_fluctuation_operators.py` | 真空零点能下限、Casimir 压力、Schwinger 对产生、虚玻色子传播子 | 928 |
| `baryogenesis_parity_operators.py` | 宇称不守恒、CKM 相位干涉、正反物质湮灭余量 | 898 |
| `supermembrane_spacetime_operators.py` | 11 维超膜 DBI、Randall-Sundrum 翘曲 KK 模态 | 864 |
| `quantum_tidal_operators.py` | WKB 隧穿、Schwarzschild 四极潮汐张量 | 862 |
| `neuro_nonlinear_operators.py` | 树突 NMDA 镁门控、FitzHugh-Nagumo、Kuramoto 相位耦合 | 793 |
| `local_self_modifying_operators.py` | Delta-Oja 在线可塑性、奇异值能量软裁剪 + 自举编译接口 + **persistent SRAM 单次 launch 前反向** | 1,431 |
| `tulpa_neuro_operators.py` | 多子人格吸引子、精度加权自由能路由、Agency 比较器 | 692 |
| `dissociative_trauma_neuro_operators.py` | Stage-I 前门控：杏仁核超唤醒、PAG 过载关断、ANP/EP 隔离 + 独立验证套件 | 539 |
| `moa_triton_kernels.py` | Triton 融合内核 + 逐 tile SRAM 模拟器 | 390 |

## 算子职责与下游接口

| 阶段 | 领域算子 | 输出用途 |
|---|---|---|
| 0.A | 真空零点能 | 构造物质生成阶段的输入 |
| 0.B | 宇称重子生成 | 提供超膜阶段的物质态 |
| I.A | 超膜 DBI / RSKK | 提供潮汐变换的基底 |
| I.B | 量子潮汐 WKB | 提供资源相位算子的输入 |
| I.C | 量子资源相位 | 形成分子调制和创伤监测共用的物理基底 |
| I.D | 病毒核酸入侵 | 生成感觉运动主路径的逐通道分子调制 |
| II.A | 脊髓通路 | 生成反射特征，供皮质脊髓与感觉运动通路使用 |
| II.B | 皮质–脊髓混沌 | 生成感觉运动控制量和睡眠的皮质下印迹 |
| II.C | 创伤调节 | 提供末端防御张量和互补门 `G_shut` |
| III.Col0 | 图文融合 | 生成皮层内容 |
| III.Col1 | 树突 NMDA/FHN | 调制皮层内容的兴奋性增益 |
| III.Col2 | 突触快权重 | 提供皮层的记忆校正 |
| IV | Tri-ISA CPU/SIMT GPU 自举 | 将感觉运动与皮层结果转换为代理的执行状态 |
| V | Tulpa 代理 | 生成睡眠巩固所需的清醒状态 |
| VI | 睡眠巩固 | 生成末端互补门的巩固分量 |

## 算子实现

- `num_proposers=3` 对应皮层的三个固定职责，`top_k_proposers` 控制额外的稀疏精炼。
- **Expert 0 = 跨模态图文融合** `CrossModalVisionLanguageFusionSubOp`
  （双向 2D 空间邻域卷积 + `R_anchor` 共享潜锚点共注意力 + Hadamard-Bilinear 门控合成）。
- **零距离衰减**：任意图像 patch 与任意文本 token 通过 `R_anchor` 个共享锚点直接交互，
  复杂度 `O(L·R_anchor·D)`，没有指数衰减项。
- **多层骨干** `UnifiedHybridMoABackbone`：堆叠 N 个 `UnifiedInterdisciplinaryMoALayer`
  （残差 + final RMSNorm），返回逐层遥测。
- **无 torch 也可 import**：每个模块在 `not HAS_TORCH` 时定义 stub 类，实例化时才抛
  `ImportError` 并提示改用 `AnalyticalKernels`；numpy 路径完全不依赖 torch。
- **DeltaOja 的 launch-bound 修复（v8 → v9）**：v8 接上了前向分派
  （`PersistentDeltaOjaSRAMFunction` → `persistent_delta_oja_sram_fused_fwd_kernel`，
  grid = B×G，W∈R^{32×32}=4KB 全程钉在片上 SRAM，1 次 launch 取代 128×G 次），
  但**反向仍在 `torch.enable_grad()` 里重跑串行循环**（占 99% 时间），且 v8 的
  `e_loc = z_sq_sum / R_RANK` 让 1 元素张量被当成指针传给 `tl.atomic_add` → **T4 上编译失败**。
  v9 三处一起修：`e_loc` 标量化、新增 `persistent_delta_oja_sram_fused_bwd_kernel`
  （反向时间单次 launch、零 atomic，前向额外存 W_t 轨迹）、以及所有领域模块
  `torch.empty_like(..., memory_format=torch.contiguous_format)`（非连续输出缓冲区修正）。
- **DeltaOja v6 → v9 T4 实测**（Kaggle GPU kernel `redmingwei/hetero-moe-deltaoja-fix`，
  同一权重同一输入，B=4 / L=128 / D=256 / G=16）：

  | | v6 串行（4 次） | v9 / v12 persistent SRAM（8 次） | 加速 |
  |---|---|---|---|
  | 前向 | 55.79 – 60.59 ms（中位 ≈ 56.5） | **1.40 – 1.77 ms（中位 ≈ 1.66）** | **≈ 34×** |
  | 前向 + 反向 | 232.80 – 245.29 ms（中位 ≈ 235.5） | **7.84 – 9.31 ms（中位 ≈ 8.9）** | **≈ 26×** |
  | 与 v6 串行实现的输出差 | — | **1.431e-06**（浮点级） | — |

  v8 曾有三条缺陷（`e_loc` 类型错致 T4 编译失败 / 反向仍串行占 99% / 输出差 6.596e+00），
  v9 起全部修复：前反向各一次 persistent launch（反向时间、零 atomic）、`e_loc` 标量化、
  所有领域模块 `torch.empty_like(..., memory_format=torch.contiguous_format)`。

  测量方法：每个版本在**独立子进程**中测量，顺序正反各跑一遍（v9 与 v12 的算法代码逐字节
  相同，只有 `__all__` 差异，因此两版数字必须一致——实测一致）。同进程连测多个版本会被
  Triton JIT 与 CUDA 分配器状态污染，曾据此得出「v9 端到端只快 1.98×」与「v12 端到端快
  34.7×」两个互相矛盾的错误结论——独立进程复测后两者同为 ≈ 26×（v9 与 v12 算法代码逐字节
  相同，本就该一致）。另有 2 次读数落在 65–71 ms，是每个版本首次触发的偶发抖动，未计入中位数。
- **Triton 内核双名并存**：`persistent_moa_norm_sparse_topk_megakernel` 与
  `persistent_norm_sparse_topk_agg_kernel` 在任何环境下都是同一个对象，两个名字都导出。

### 非创伤解耦与互补门控

```
x_molecular = x_spacetime * (0.5 + sigmoid(viral(x_spacetime)))
h_clean = x_molecular + gain_reflex * spinal_out + gain_chaos * chaotic_bridge_out
trauma_out, G_shut = trauma_regulator(x_spacetime)
Y = h_clean + G_shut * consolidated + (1 - G_shut) * gain_trauma * trauma_out
```

- 常态输入（`G_shut → 1`）：意志/Tulpa/睡眠合成域主导，创伤通路静默
- 极端冲击（`G_shut → 0`）：认知与意志阶段被屏蔽，紧急 DPDR 防御接管
- 创伤调节参数经末端互补门影响输出。分子态和干净感觉运动特征对创伤调节参数保持零依赖。

### 融合算子的数学形式

```
# 1) 双向低秩跨模态锚点共注意力（零距离衰减）
z_anchor[r]   = Σ_l softmax_l(anchor[r]·(v+t)/√D) · v[l]        # 汇成 R 个锚点
c_align[l]    = Σ_r softmax_r(t[l]·z_anchor[r]) · z_anchor[r]   # 广播回每个 token

# 2) Hadamard-Bilinear 跨模态门控合成
G_cm     = sigmoid(α_vis·V + α_txt·T)
Y_interp = G_cm·V + (1-G_cm)·T
Out      = Y_interp + β_bilin·(V⊙T + C_align) / sqrt(1 + λ·(V²+T²))
```

## Dependencies

| Layer | Requirement |
|---|---|
| NumPy analytical kernels, tile simulator | `numpy` only |
| torch operators | `torch` (CPU is enough) |
| Triton kernels | `triton` + CUDA GPU (optional — the simulator covers CPU verification) |

## Quick start

```python
import torch
from hybrid_moa_vl_operator_suite.moa_operators import (
    CrossModalVisionLanguageFusionSubOp, MoAOperatorConfig, HeterogeneousProposerOp)

# 跨模态图文融合：x [B, L, D] -> fused [B, L, D]
fusion = CrossModalVisionLanguageFusionSubOp(d_model=256, num_anchors=16)
y = fusion(torch.randn(2, 64, 256))

# 系统级 MoA：异构 proposer -> [B, N, L, D] -> top-k 聚合
cfg = MoAOperatorConfig(d_model=256, num_proposers=8, top_k_proposers=3)
proposals = HeterogeneousProposerOp(cfg)(torch.randn(2, 64, 256))

# 七阶段统一层 / 多层骨干
from hybrid_moa_vl_operator_suite import UnifiedHybridArchitectureConfig, UnifiedHybridMoABackbone
bcfg = UnifiedHybridArchitectureConfig(d_model=256, n_layers=4)
out, aux, telemetries = UnifiedHybridMoABackbone(bcfg)(torch.randn(2, 64, 256))

# 新增的两个领域算子
from hybrid_moa_vl_operator_suite import SleepSubconsciousConfig, SleepSubconsciousNeuroOp
from hybrid_moa_vl_operator_suite import VacuumZPEFluctuationConfig, VacuumZPEFluctuationOp
```

NumPy-only reference path:

```python
from hybrid_moa_vl_operator_suite import NumpyMoAOperatorKernels

k = NumpyMoAOperatorKernels()
c_align, cache = k.cross_modal_anchor_co_attention_forward(v_vis, t_txt, anchors)  # 零距离衰减
out, cache = k.cross_modal_vl_bilinear_fusion_forward(
    v_vis, t_txt, c_align, alpha_vis, alpha_txt, beta_bilin, lam_norm=0.35)
```

### 固定角色接口

每个 `UnifiedInterdisciplinaryMoALayer` 都执行 Stage I.D。分子调制的增益介于
`0.5` 和 `1.5`，逐通道乘到物理基底上。`viral_invasion` 配置控制算子内部参数。
算子继承骨干宽度，分组数取请求分组数和宽度的最大公约数。

Stage III 的三个职责分别进入下式。Top-K 路由控制 `routed_detail`。

```text
cortical = alignment * (0.5 + sigmoid(dendritic)) + fast_weight_memory + routed_detail
```

阶段残差增益由 `0.05 + 0.95 * sigmoid(logit)` 得出。分子算子内部六个子算子的
混合增益处于 `[0.05, 0.95]`，各自保留一条通向分子场输出的路径。

```python
import torch
from hybrid_moa_vl_operator_suite import UnifiedHybridArchitectureConfig, UnifiedHybridMoABackbone

cfg = UnifiedHybridArchitectureConfig(d_model=32, n_heads=4, n_layers=2)
model = UnifiedHybridMoABackbone(cfg)
out, aux_loss, telemetry = model(torch.randn(2, 8, 32))
print(telemetry[0]["stage1d_molecular_rms"])
```

PyTorch 的分子算子在 CPU 上调用 NumPy float64 前向和一阶反向，并将结果返回
输入的设备和 dtype。分子场处理潜变量张量，默认参数用于合成研究示范。

加载带原始 PyTorch 版本元数据的旧骨干权重时，使用 `strict=False` 并检查
缺失键仅属于 Stage I.D。旧阶段增益会转换成 logit，Stage I.D 新权重按配置初始化。
已有兼容的分子算子权重也可迁移。重构会改变旧权重在新骨干中的输出。

## Package initializer (v12)

`__init__.py` is no longer a hand-written export table. It now:

1. discovers every submodule via `pkgutil.iter_modules` (ordering standalone
   operator/kernel modules before the unified-assembly and verification modules);
2. scans each submodule's `__all__` (each module now also generates its own
   `__all__` from its own namespace) and resolves every symbol through `getattr`
   with a fault-tolerant fallback into nested `*Kernels` / `*Simulator` /
   `*Engine` classes;
3. exposes a PEP 562 module-level `__getattr__` for on-demand cross-submodule lookup.

Effect: renaming or moving a symbol in one operator module no longer breaks
`import hybrid_moa_vl_operator_suite` — the three earlier import failures
(v2: 20 missing class names; v8/v9: 2 missing module-level functions) become
impossible by construction. Measured: **138 exports, package imports with no patch.**

## Verification

`smoke_test.py` is the CPU clean-room test. The standalone imports run in fresh
interpreters and are typically the longest part of the test.

```bash
python smoke_test.py            # everything available
python smoke_test.py --quick    # imports + numpy kernels only
```

The current suite checks Stage I.D gradients, PyTorch/NumPy parity, cache isolation,
and the three cortical roles with the routed correction set to zero.

```bash
python hybrid_moa_vl_operator_suite/operator_independence_verification.py
python hybrid_moa_vl_operator_suite/unified_hybrid_moa_architecture.py
```

The first command checks independent execution, output and gradient rank,
cross-sensitivity at fixed local inputs, and leave-one-out reconstruction on
a synthetic target. The second command intervenes on the assembled layer. It
ablates all 15 roles and substitutes each of the 14 single-output roles with
each other role's frozen output. The dual-output trauma role is ablated by
removing the defense output and setting `G_shut=1`. These forward interventions
keep the other operators and weights in place and re-evaluate their downstream
responses. The results apply to the tested configurations and inputs.
The image-text reconstruction probe also re-executes downstream stages for the
unaligned additive baseline. The aligned path must outperform it in each fold.

The table below records the pre-integration baseline of **16/16 checks**.

| Check | Result |
|---|---|
| package imports (14 modules, 64 exports) | 14/14 |
| module isolation: 13 standalone modules import alone in an empty dir | 13/13 |
| branch RMSNorm fwd/bwd + finite-difference gradient | fwd 1.8e-15, grad 2.7e-09 |
| cross-modal anchor co-attention (zero decay, both attentions normalised) | 0.0e+00, head/tail = 1.003 |
| VL Hadamard-bilinear fusion: closed form, β=0 degeneration, gate saturation, backward gradient | fwd 8.9e-16, grad 7.7e-11 |
| per-head MoA attention mask (causal, 3 densities 0.198/0.296/0.516) | ok |
| sparse top-k router | aux 0.0222 |
| torch `CrossModalVisionLanguageFusionSubOp` fwd/bwd | 13.4K params, grad reaches input |
| torch MoA stack (proposer + attention + 2 aggregator modes) | density 0.913 |
| torch `UnifiedInterdisciplinaryMoALayer` (12 domains / 7 stages) | 387.2K params, grad 0.105, 19 telemetry keys |
| torch `UnifiedHybridMoABackbone` (3 layers) | 1,096.5K params, grad reaches input |
| torch new domain operators (sleep, vacuum ZPE, Tri-ISA CPU-GPU, quantum resource) | 13.2K / 13.2K / 21.6K / 45.4K params；诊断含 `bootstrap_fixed_point_verified`、三 ISA 分发权重、`mean_dqi_reliability` 等 |
| torch domain operators (quantum tidal, neuro) | both return (out, diagnostics) |
| upstream dissociative verification suite | 7 gradchecks PASS, shock output bounded |
| mechanism: Stage 0/III/IV/V/VI vs Trauma decoupling + regime gates | 七组全部精确 0.0e+00（含新增 Stage IV Tri-ISA；创伤通路自身变化 20.9 作对照）；G_shut 0.756 → 7.12e-218；DPDR 压缩 40.2× |
| Triton tile simulators vs analytical references | 2.2e-16 / 8.9e-16 |

Historical upstream benchmarks before the role refactor:

- `FusedCrossModalVLBilinear`: diff **0.00e+00**, launches 12 → 1, **HBM traffic −80.5%**
- `PersistentNormSparseTopKMegakernel` (top-2/4): diff 2.66e-15, launches 17 → 1, **HBM −90.7%**
- Image-text alignment R²: unaligned additive baseline **+0.7047** → VL fusion branch **+0.9990**
- LOSO over all 12 stage names: every stage active; Stage V (sleep) is the largest single
  contributor at **38.29%** relative output impact, effective rank **102.31 / 128**
- Full 6-stage input gradcheck and Stage-III router gradcheck < 1e-8
- Eager vs fused Triton megakernel path < 1e-14
- Dissociative trauma gradcheck: 7 variables, relative L2 error 6.7e-11 – 7.5e-10;
  extreme +50.0 shock output bounded at 1.32 (< 5.0), PAG shutdown gate 7.1e-218
- **Tri-ISA CPU-GPU self-bootstrapping** (`cpu_gpu_self_bootstrapping_operators.py`):
  I-cache gradcheck 8.7e-10; quad-backend equivalence (numpy / AST / native `.so` /
  Triton tile) **4.44e-16**; 4-sub-architecture subspace rank **4/4 full**;
  per-ISA gates x86-64 G=0.4427 / R_tso=0.7279, ARM64 P_sve=0.4980 / B_ldar=0.8827,
  RISC-V L_vsetvli=0.3744, GPU SIMT M_warp=0.8601 / W_roof=0.3228;
  HBM traffic 1792 KB → 194.2 KB (**−89.2%**, 9.23× bandwidth)
- LOSO over all 13 stage names: every stage active; Stage VI (sleep) remains the largest
  single contributor at **46.92%**, new Stage IV (Tri-ISA bootstrap) **1.31%**,
  effective rank **102.29 / 128**
- **Five-dimensional operator independence audit — PASSED**
  (Kaggle CPU kernel `redmingwei/hetero-moe-operator-independence`, 59.9 s;
  full log archived in `docs/operator_independence_audit.log`):

  | Dimension | Result |
  |---|---|
  | I · file & runtime isolation | **15/15 modules**: zero sibling imports, standalone run PASS |
  | II · forward-representation independence | joint subspace rank **14/14 full**; mean off-diag \|cos\| **0.0762**; max VIF **1.74** (< 2.0) |
  | III · input-gradient subspace independence | gradient matrix rank **14/14**; max gradient VIF < 5.0 |
  | IV · parameter cross-sensitivity Jacobian | off-diagonal cross-talk **exactly 0.00e+00**; all 14 active |
  | V · leave-one-operator-out (CV) | synthetic full-ensemble CV R² **0.9997** |

  v12 起覆盖 **15 个独立模块 / 14 个领域算子**（新增 `14:QuantumResource_Phasing`）。Kaggle GPU kernel
  `redmingwei/hetero-moe-operator-independence-v8`，1.1 min 跑完，日志见
  `docs/operator_independence_audit.log`。
  Note: the script's header says "5-Fold CV" while its result line prints "4-Fold CV R²";
  the fold count is the author's, not something changed here.

  Reproduce it on Kaggle CPU with
  <https://www.kaggle.com/code/redmingwei/hetero-moe-operator-independence>
  (private kernel; it downloads this package at a pinned commit SHA and runs
  `operator_independence_verification.py`, archiving the log into the kernel output).
  Locally it is also runnable — `python3 hybrid_moa_vl_operator_suite/operator_independence_verification.py`
  — but on a constrained sandbox CPU Dimension I alone (13 fresh interpreters) can take
  much longer than the 60 s it needs on a normal machine.

## Notes

- This repository holds the **original version** of the operator library. Adapted copies
  used inside [hetero-moe](https://github.com/Microbiosis/hetero-moe)
  (`research/aligner/v62_dimension_projection_space/`) are modified for that project
  and are maintained there.
- Stage I.D adapts the standalone operator supplied with the project and connects
  its molecular field to the sensorimotor input of every unified layer.
- The film scripts in `viz/` retain the historical 14-operator display.
- `docs/TAXONOMY_AUDIT.md` audits an external taxonomy write-up of this architecture
  claim by claim against the code (which claims hold, which are overstated, and what
  the write-up missed).
- Import-mode history, for the record: v2 of the archive imported 20 symbol names that
  did not exist, and v3 imported a Triton kernel name that only exists when Triton is
  absent; both were fixed upstream in the current archive (`if __package__:` dual-mode
  imports, and the megakernel exposed under both names in every environment).

## License

[Apache License 2.0](LICENSE) — Copyright 2026 Microbiosis.

The repository-level license applies to the package. The original provenance
and the molecular operator adaptation are recorded in [NOTICE](NOTICE).
