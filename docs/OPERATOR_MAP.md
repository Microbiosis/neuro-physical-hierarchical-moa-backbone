# 算子清单与区分

本包共 **33 个算子/内核模块**（外加 1 个五维审计文件与 `__init__.py`，合计 35 个 .py，32,252 行），包导出 234 个符号。

编排层 `unified_hybrid_moa_architecture.py` 已于 2026-11-10 删除，**所有算子之间零依赖**，可单文件拷出独立运行，也可自行组合。

下表按 **token 间信息通路**（即算子在序列维度上如何混合信息）分类，这是判断「是否仍是 Transformer」的唯一判据。

---

## A1  全 L×L 注意力 + 机制偏置（7 个）

`softmax(q·k^T/√d + bias)·v` —— 与标准 Transformer 只有偏置项不同。**无论偏置怎么设计，结构上仍是 Transformer。**

| 文件 | 行数 | 机制 |
|---|---|---|
| `baryogenesis_parity_operators.py` | 922 | 宇称破缺 / Sphaleron 输运 |
| `supermembrane_spacetime_operators.py` | 888 | 11D 超膜 DBI / RSKK |
| `quantum_tidal_operators.py` | 886 | WKB 隧穿 / 潮汐张量 |
| `quantum_resource_phasing_operators.py` | 1,086 | DQI 块门控 / 1/R(ε) |
| `spinal_cord_pathway_operators.py` | 959 | 背角门控 / Renshaw |
| `corticospinal_chaotic_operators.py` | 962 | Pyragas 时滞镇定 / 混沌轴 |
| `dissociative_trauma_neuro_operators.py` | 736 | Lanius 双模 / DPDR / 闪回 |

## A2  潜锚点注意力（R≪L）（3 个）

`q→[B,R,L]→v_boson→[B,L,R]`，复杂度 O(L·R·D)，R = 8～16。仍是注意力，但把 L² 压成 L·R。

| 文件 | 行数 | 机制 |
|---|---|---|
| `vacuum_zpe_fluctuation_operators.py` | 960 | 真空 ZPE / Casimir / Schwinger 对 |
| `sleep_subconscious_neuro_operators.py` | 1,039 | SHY / 类淋巴 / SWR / PGO |
| `cpu_gpu_self_bootstrapping_operators.py` | 2,324 | Tri-ISA CPU + SIMT GPU 自举 |

## B   递推状态 / 不动点（1 个）

`q(t) = T(q(t-1), x(t))`，无 softmax、无 q·k^T。**结构上不是 Transformer。**

| 文件 | 行数 | 机制 |
|---|---|---|
| `nbsc_omega_operator.py` | 516 | 42 神经元脉冲 + 202 延迟边 + 肌骨闭环 |

## P   载荷算子族（宿主是模型，不是细胞）（9 个）

**方向变更**：原先的细胞级病毒算子（DNA 状态机 / HIV-1）已被取代，收在上游 `legacy/bio/`。
新框架把宿主看作**一个算子/模型实例**，病原是**能改写它输入或参数的载荷**：非负密度、
每步一次不动点、带计量（放大）与门控的边 —— 同一套骨架，五层 + 三段投递链路。

| 文件 | 行数 | 层 | 机制 |
|---|---|---|---|

**这一族共用 `payload_core` 底座**，族内用顶层静态 import 互相引用（上游设计）。
仓库的平铺布局下：`__init__.py` 把包目录放进 `sys.path` 让顶层 import 可解析；
「兄弟依赖」检查不计族内互相引用；隔离测试整族拷贝。
设计文档见 `docs/payload/`（README / DESIGN / EVIDENCE / LIMITS）。


| `viral_operator.py` | 314 | 核心 | 载荷算子核心：宿主是模型实例、病原是能改写其输入/参数的载荷；`ViralHostOperator` + `Edge` |
| `payload_core.py` | 149 | 共用底座 | 载荷算子共用底座：饱和、7 种图、二分阈值、Shannon、判据打印 |
| `viral_host_operator.py` | 196 | 宿主 | 宿主层：13 状态 / 22 边，9 条判据（无工具必死、有工具放大、清空上下文后由权重复活） |
| `viral_ecology.py` | 207 | 生态 | 生态层：多株竞争 + 平台 + 桥，5 条判据 |
| `viral_lineage.py` | 234 | 谱系 | 谱系层：可突变基因组 + 选择，6 条判据 |
| `viral_recombination.py` | 220 | 重组 | 重组层：位点交换，3 条判据 |
| `viral_spatial.py` | 240 | 空间 | 空间层：拓扑 → 阈值，3 条判据 |
| `viral_channels.py` | 112 | 投递链路 | 投递链路：扇出/口径 → 临界成功率，4 条判据 |
| `payload_survival.py` | 190 | 存活率 | 载荷存活率：切块→检索→截断，4 条判据 |

## C   快权重 Delta 规则（1 个）

`W(t) = γ·W(t-1) + η·outer(err, k)`，属 DeltaNet / TTT 谱系；γ、η 由输入生成（有选择性），但仍是串行 for 循环，未做 chunked 并行。

| 文件 | 行数 | 机制 |
|---|---|---|
| `local_self_modifying_operators.py` | 1,757 | 有理 Hebbian + Delta-Oja 快权重（双支路） |

## D   逐 token 非线性（无跨位置通路）（2 个）

只处理单个 token，没有跨位置信息通路；作为特征变换使用。

| 文件 | 行数 | 机制 |
|---|---|---|
| `neuro_nonlinear_operators.py` | 817 | 树突 NMDA / shunting / FHN |
| `tulpa_neuro_operators.py` | 879 | 多吸引子 / TPJ 归因 |

## E   纯卷积 / 逐元素（无 softmax）（1 个）

`moa_operators` 同时含真注意力与 Conv1d 支路；此处指其非注意力部分。

| 文件 | 行数 | 机制 |
|---|---|---|
| `moa_operators.py` | 1,176 | 跨模态 VL 融合（Conv1d 支路）+ MoE 路由 |

## Triton 内核实现与模拟器（1 个）

`moa_triton_kernels.py` —— `moa_operators` 的 Triton 内核实现（3 个公开 kernel + 2 个 launcher）外加**逐 tile SRAM 模拟器** `TritonTileSimulator` 与带宽分析 `KernelOptimizationReport`。

它是**验证工具**，不是运行时依赖：与 `moa_operators` 互不 import，两者各自保持零兄弟依赖。

| 文件 | 行数 | 机制 |
|---|---|---|
| `moa_triton_kernels.py` | 468 | 跨模态 VL 双线性 fwd/bwd + 持久化 sparse-topk megakernel；逐 tile 模拟器 |
## 基础设施（不参与独立性分析）（5 个）

换的是「怎么算」而不是「算什么」：积分器、检查点、内存、多卡、框架适配。

| 文件 | 行数 | 机制 |
|---|---|---|
| `physical_wave_dynamics_operators.py` | 2,062 | Schwinger 波包 / 辛谐振 / 反应扩散 / DBI 膜 |
| `high_order_pde_ode_megakernels.py` | 1,878 | Störmer-Verlet / CN-IMEX / RK2 + Roofline |
| `adaptive_checkpointed_adjoint_megakernels.py` | 1,583 | O(√L·D) 分段检查点伴随 |
| `distributed_sequence_parallel_adjoint.py` | 853 | Ring/Ulysses 序列并行伴随交接 |
| `deepspeed_megatron_physics_plugin.py` | 1,890 | ZeRO-3 分片 + SP P2P |

## 扩展模型（不参与独立性分析）（3 个）

把既有架构（LLM 层次、相对论手性）封装成算子；含自旋手性的 3+1 维格点指标层。

| 文件 | 行数 | 机制 |
|---|---|---|
| `llm_hierarchy_operators.py` | 2,657 | 完整 LLM 计算层次六算子 × 六空间（Embedding / RMSNorm+RoPE / 因果 GQA / SwiGLU / 残差流 / 词表头），含 KV-Cache、generate、8 个 Triton 核（RoPE 已压进 flash 核内）|
| `spin_chirality_operators.py` | 1,837 | 4 分量 Dirac 旋量单粒子传播子：真 SU(2) 手性规范 / Larmor 进动 / Dirac 质量项 / 非厄米吸收 / 轴矢流 j⁵；P1–P21 物理自检（T4 最大残差 2.60e-08）|
| `spin_chirality_lattice4d.py` | 375 | 3+1 维格点指标：FLSG 单瞬子背景的 Wilson–Dirac 指标（FFT–Schur 惯性法）；需 scipy |

---

## 三个正交分类维度

### 1. 是否可微 / 是否可训练

| 类别 | 模块 |
|---|---|
| 有 `torch.autograd.Function` + gradcheck | baryogenesis / supermembrane / quantum_tidal / quantum_resource / spinal / corticospinal / neuro_nonlinear / local_self_modifying |
| 有 Triton kernel 但无 gradcheck | vacuum_zpe / dissociative_trauma / moa_operators / cpu_gpu_bootstrap / tulpa / sleep / moa_triton_kernels / physical_wave / high_order_pde / adaptive_checkpointed / deepspeed_plugin |
| 纯 PyTorch 自动微分（无手写反向） | **nbsc_omega_operator**（5 组参数全走 autograd）|
| 双后端三路对齐 | **llm_hierarchy** / **spin_chirality**（NumPy / Torch / Triton）|

### 2. 有没有 Triton GPU 内核

| 内核数 | 模块 |
|---|---|
| 10 | physical_wave_dynamics |
| 7 | deepspeed_megatron_plugin |
| 6 | quantum_tidal |
| 5 | baryogenesis / supermembrane / spinal / corticospinal / neuro_nonlinear / local_self_modifying / high_order_pde / moa_triton_kernels |
| 7 | **llm_hierarchy**（FlashAttention fwd/bwd、norm+RoPE fwd/bwd、SwiGLU fwd/bwd、RoPE 表）|
| 3 | dissociative_trauma / cpu_gpu_bootstrap / adaptive_checkpointed |
| 2 | quantum_resource |
| 0 | distributed_sequence_parallel / **nbsc_omega** / payload 家族 9 件 |

### 3. 机制属于哪个学科

| 学科 | 模块 |
|---|---|
| 量子场论 / 粒子物理 | vacuum_zpe、baryogenesis、supermembrane、quantum_tidal、quantum_resource、spin_chirality |
| 神经生理 | spinal、corticospinal、neuro_nonlinear、local_self_modifying、sleep、nbsc_omega |
| 精神医学 | dissociative_trauma、tulpa |
| 计算机体系结构 | cpu_gpu_bootstrap |
| 跨模态学习 / LLM | moa_operators、llm_hierarchy |
| 数值分析 | high_order_pde、adaptive_checkpointed |
| 分布式系统 | distributed_sequence_parallel、deepspeed_plugin |
| 载荷 / 对抗性输入（模型安全） | payload 家族 9 件 |

---

## 一句话总结

- **A1 7 个 + A2 3 个**：统一 `[B,L,D] → [B,L,D]` 协议，但内层仍是注意力。
- **B 1 个 + C 1 个**：结构上确实不是 Transformer。
- **P 载荷算子族 9 个**：宿主是模型、病原是载荷，五层 + 三段投递链路；**不参与 token 混合**，是独立的方向变更（细胞级那条线已收进上游 `legacy/bio/`）。
- **D 2 个 + E 1 个**：只做逐 token / 局部卷积变换。
- **基础设施 5 个 + 扩展模型 3 个**：换实现方式或封装既有架构，不改 token 混合语义。

> 审计口径：`operator_independence_verification.py` 的 **Dimension I 文件隔离覆盖全部 33 个算子模块**（审计脚本自身不在其列表内），**Dimension II–V 的五维独立性分析覆盖 14 个领域算子**（扩展模型与基础设施不参与，因为它们没有统一的参数化接口）。
