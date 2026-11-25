# 算子清单与区分

本包共 **46 个算子/内核模块**（外加 1 个五维审计文件、26 个 `__init__.py` 与 1 个 `__main__.py`，组织为 25 个子包，合计 74 个 `.py`，42,946 行），包导出 302 个符号。

编排层 `unified_hybrid_moa_architecture.py` 已于 2026-11-10 删除，**各算子/族按独立子目录组织，跨族零依赖**，可单模块（或 `payload/`、`spin_chirality/` 整族）拷出独立运行，也可通过包顶层或规范子包路径自行组合。

下表按 **token 间信息通路**（即算子在序列维度上如何混合信息）分类，这是判断「是否仍是 Transformer」的唯一判据。

---

## A1  全 L×L 注意力 + 机制偏置（7 个）

`softmax(q·k^T/√d + bias)·v` —— 与标准 Transformer 只有偏置项不同。**无论偏置怎么设计，结构上仍是 Transformer。**

| 文件 | 行数 | 机制 |
|---|---|---|
| `baryogenesis_parity_operators/baryogenesis_parity_operators.py` | 922 | 宇称破缺 / Sphaleron 输运 |
| `supermembrane_spacetime_operators/supermembrane_spacetime_operators.py` | 888 | 11D 超膜 DBI / RSKK |
| `quantum_tidal_operators/quantum_tidal_operators.py` | 886 | WKB 隧穿 / 潮汐张量 |
| `quantum_resource_phasing_operators/quantum_resource_phasing_operators.py` | 1,086 | DQI 块门控 / 1/R(ε) |
| `spinal_cord_pathway_operators/spinal_cord_pathway_operators.py` | 959 | 背角门控 / Renshaw |
| `corticospinal_chaotic_operators/corticospinal_chaotic_operators.py` | 962 | Pyragas 时滞镇定 / 混沌轴 |
| `dissociative_trauma_neuro_operators/dissociative_trauma_neuro_operators.py` | 736 | Lanius 双模 / DPDR / 闪回 |

## A2  潜锚点注意力（R≪L）（3 个）

`q→[B,R,L]→v_boson→[B,L,R]`，复杂度 O(L·R·D)，R = 8～16。仍是注意力，但把 L² 压成 L·R。

| 文件 | 行数 | 机制 |
|---|---|---|
| `vacuum_zpe_fluctuation_operators/vacuum_zpe_fluctuation_operators.py` | 960 | 真空 ZPE / Casimir / Schwinger 对 |
| `sleep_subconscious_neuro_operators/sleep_subconscious_neuro_operators.py` | 1,039 | SHY / 类淋巴 / SWR / PGO |
| `cpu_gpu_self_bootstrapping_operators/cpu_gpu_self_bootstrapping_operators.py` | 2,324 | Tri-ISA CPU + SIMT GPU 自举 |

## B   递推状态 / 不动点（1 个）

`q(t) = T(q(t-1), x(t))`，无 softmax、无 q·k^T。**结构上不是 Transformer。**

| 文件 | 行数 | 机制 |
|---|---|---|
| `nbsc_omega_operator/nbsc_omega_operator.py` | 516 | 42 神经元脉冲 + 202 延迟边 + 肌骨闭环 |

## P   载荷算子族（宿主是模型，不是细胞）（9 个）

**方向变更**：原先的细胞级病毒算子（DNA 状态机 / HIV-1）已被取代，收在上游 `legacy/bio/`。
新框架把宿主看作**一个算子/模型实例**，病原是**能改写它输入或参数的载荷**：非负密度、
每步一次不动点、带计量（放大）与门控的边 —— 同一套骨架，五层 + 三段投递链路。

| 文件 | 行数 | 层 | 机制 |
|---|---|---|---|

**这一族集中置于 `payload/` 子包并共用 `payload_core` 底座**，族内采用 `if __package__:` 双模式导入（包形态走相对导入，单独整族运行走顶层导入）；严格采用规范子包路径（`hybrid_moa_vl_operator_suite.payload.*`），不保留顶层扁平别名；`__init__.py` 不向 `sys.path` 注入包目录，仅对缺失可选第三方依赖（`torch` / `triton` / `scipy`）跳过可选模块，并过滤 `main` 与同族重名动词避免包顶层命名空间碰撞；可变参数（`MODE` / `TRANSMISSION` / `GUARD_STRENGTH`）支持显式函数传参而非修改模块全局变量。
「兄弟依赖」检查不计族内互相引用；隔离测试整族拷贝。
设计文档见 `docs/payload/`（README / DESIGN / EVIDENCE / LIMITS）。


| `payload/viral_operator.py` | 328 | 核心 | 载荷算子核心：宿主是模型实例、病原是能改写其输入/参数的载荷；`ViralHostOperator` + `Edge` |
| `payload/payload_core.py` | 166 | 共用底座 | 载荷算子共用底座：饱和、7 种图、二分阈值、Shannon、判据打印 |
| `payload/viral_host_operator.py` | 217 | 宿主 | 宿主层：13 状态 / 22 边，9 条判据（无工具必死、有工具放大、清空上下文后由权重复活） |
| `payload/viral_ecology.py` | 221 | 生态 | 生态层：多株竞争 + 平台 + 桥，5 条判据 |
| `payload/viral_lineage.py` | 244 | 谱系 | 谱系层：可突变基因组 + 选择，6 条判据 |
| `payload/viral_recombination.py` | 233 | 重组 | 重组层：位点交换，3 条判据 |
| `payload/viral_spatial.py` | 251 | 空间 | 空间层：拓扑 → 阈值，3 条判据 |
| `payload/viral_channels.py` | 116 | 投递链路 | 投递链路：扇出/口径 → 临界成功率，4 条判据 |
| `payload/payload_survival.py` | 204 | 存活率 | 载荷存活率：切块→检索→截断，4 条判据 |

## C   快权重 Delta 规则（1 个）

`W(t) = γ·W(t-1) + η·outer(err, k)`，属 DeltaNet / TTT 谱系；γ、η 由输入生成（有选择性），但仍是串行 for 循环，未做 chunked 并行。

| 文件 | 行数 | 机制 |
|---|---|---|
| `local_self_modifying_operators/local_self_modifying_operators.py` | 2,229 | 有理 Hebbian + Delta-Oja 快权重（双支路） |

## D   逐 token 非线性（无跨位置通路）（2 个）

只处理单个 token，没有跨位置信息通路；作为特征变换使用。

| 文件 | 行数 | 机制 |
|---|---|---|
| `neuro_nonlinear_operators/neuro_nonlinear_operators.py` | 817 | 树突 NMDA / shunting / FHN |
| `tulpa_neuro_operators/tulpa_neuro_operators.py` | 879 | 多吸引子 / TPJ 归因 |

## E   纯卷积 / 逐元素（无 softmax）（1 个）

`moa_operators` 同时含真注意力与 Conv1d 支路；此处指其非注意力部分。

| 文件 | 行数 | 机制 |
|---|---|---|
| `moa_operators/moa_operators.py` | 2,018 | 跨模态 VL 融合（Conv1d 支路）+ MoE 路由 |

## Triton 内核实现与模拟器（1 个）

`moa_triton_kernels/moa_triton_kernels.py` —— `moa_operators` 的 Triton 内核实现（3 个公开 kernel + 2 个 launcher）外加**逐 tile SRAM 模拟器** `TritonTileSimulator` 与带宽分析 `KernelOptimizationReport`。

它是**验证工具**，不是运行时依赖：与 `moa_operators` 互不 import，两者各自保持零兄弟依赖。

| 文件 | 行数 | 机制 |
|---|---|---|
| `moa_triton_kernels/moa_triton_kernels.py` | 800 | 跨模态 VL 双线性 fwd/bwd + 持久化 sparse-topk megakernel；逐 tile 模拟器 |
## 基础设施（不参与独立性分析）（5 个）

换的是「怎么算」而不是「算什么」：积分器、检查点、内存、多卡、框架适配。

| 文件 | 行数 | 机制 |
|---|---|---|
| `physical_wave_dynamics_operators/physical_wave_dynamics_operators.py` | 2,062 | Schwinger 波包 / 辛谐振 / 反应扩散 / DBI 膜 |
| `high_order_pde_ode_megakernels/high_order_pde_ode_megakernels.py` | 1,878 | Störmer-Verlet / CN-IMEX / RK2 + Roofline |
| `adaptive_checkpointed_adjoint_megakernels/adaptive_checkpointed_adjoint_megakernels.py` | 1,583 | O(√L·D) 分段检查点伴随 |
| `distributed_sequence_parallel_adjoint/distributed_sequence_parallel_adjoint.py` | 853 | Ring/Ulysses 序列并行伴随交接 |
| `deepspeed_megatron_physics_plugin/deepspeed_megatron_physics_plugin.py` | 1,890 | ZeRO-3 分片 + SP P2P |

## 扩展模型（不参与独立性分析）（16 个）

把既有架构（LLM 层次、相对论手性及格点规范理论）封装成算子与配套格点/谱验证层（`spin_chirality/` 共 15 个模块，涵盖 P1–P45 物理自检）。

| 文件 | 行数 | 机制 |
|---|---|---|
| `llm_hierarchy_operators/llm_hierarchy_operators.py` | 2,964 | 完整 LLM 计算层次六算子 × 六空间（Embedding / RMSNorm+RoPE / 因果 GQA / SwiGLU / 残差流 / 词表头），修复双重 RoPE（仅在 Q/K 投射后旋转并回传可学 `omega` 梯度）、零拷贝 5D GQA、静态 KV-Cache $O(1)$ 解码、generate、12 个 Triton 核|
| `spin_chirality/operator.py` | 3,810 | 4 分量 Dirac 旋量单粒子传播子：真 SU(2) 手性规范（`"protocol"` / `"generator"` + 二阶 Strang 分裂）、零轴光滑指数 `su2_exp`、时变序列传播 `forward_sequence`、实数等变包装 `RealSpinChiralityOperator` 与闭式 `fit_chiral_procrustes` / `fit_su2_procrustes`；P1–P45 物理自检（最大残差 ≤ 9.58e-07）|
| `spin_chirality/lattice4d.py` | 346 | 3+1 维格点指标（P17）：FLSG 单瞬子背景的 Wilson–Dirac 指标（FFT–Schur 惯性法）；需 scipy |
| `spin_chirality/lattice1d.py` | 402 | 1+1D 色散/费米子倍增、Jackiw–Rebbi 畴壁零模、Peierls 规范场、海手征荷与 Landau–Zener 对产生（P22/P23/P25/P26/P28/P30） |
| `spin_chirality/fock.py` | 138 | 二次量子化 Fock 空间（反对易 / Pauli / Slater，P27） |
| `spin_chirality/schwinger.py` | 191 | 1+1D Schwinger 规范场反作用与等离子体频率（P31/P32） |
| `spin_chirality/kogut_susskind.py` | 80 | 高斯定律与弦张力禁闭 $\sigma=g^2/2$（P33） |
| `spin_chirality/ks_ed.py` | 137 | Kogut–Susskind 精确对角化与高斯定律投影（P34） |
| `spin_chirality/ks_basis.py` | 346 | `(mask, E)` 平移协变基、纯置换平移算符 $T$ 与动量扇区 Fourier 分解（P41） |
| `spin_chirality/ks_lanczos.py` | 449 | 矩阵自由 Lanczos 与质量隙连续极限（P35） |
| `spin_chirality/ks_mps.py` | 947 | MPS/DMRG 两格点扫描、正交激发态与电场关联（P36） |
| `spin_chirality/ks_photon.py` | 399 | 扭曲边界磁通 $\phi$ 下光子/弦态分辨与连通电场关联（P37） |
| `spin_chirality/ks_photon_weight.py` | 901 | 电场矩阵元、谱权重、动量分解与求和规则（P38） |
| `spin_chirality/ks_photon_mass.py` | 776 | 电场零模份额 $\kappa(n)$ 定位光子与 $m_\gamma$ 双外推（P39） |
| `spin_chirality/ks_dispersion.py` | 825 | 光子色散截距拟合与三种 $x\to 0$ 外推闭合判据（P40） |
| `spin_chirality/ks_omega.py` | 545 | `(mask, E)` 动量扇区光子色散 $\omega(k)$ 与二维有限尺寸联合外推（P42） |

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
| 12 | **llm_hierarchy**（RMSNorm fwd/bwd、norm+RoPE fwd/bwd、SwiGLU fwd/bwd、FlashAttention fwd/bwd、RoPE 表与旋转）|
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
- **基础设施 5 个 + 扩展模型 16 个**：换实现方式或封装既有架构与格点/谱验证层，不改 token 混合语义。

> 审计口径：`audit/operator_independence_verification.py` 的 **Dimension I 文件隔离覆盖全部 46 个算子/内核模块**（审计脚本自身不在其列表内，且 `spin_chirality/` 15 模块与 `payload/` 9 件套各自按整族拷贝），**Dimension II–V 的五维独立性分析覆盖 14 个领域算子**（扩展模型与基础设施不参与，因为它们没有统一的参数化接口）。
