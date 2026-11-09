# 算子清单与区分

本包共 **23 个算子文件、约 26,000 行**（外加 `__init__.py`）。按「是否进入 `UnifiedInterdisciplinaryMoALayer`
主干」分成三类：**主干算子（14）**、**基础设施算子（5）**、**独立模型（1）**，
另有 2 个组织层文件与 1 个审计文件。

---

## 一、主干算子（14 个，被 `unified_hybrid_moa_architecture.py` 直接调用）

这 14 个按 `STAGE_DOMAIN_NAMES` 的顺序串成 7 个阶段，构成完整的
`x → Stage0.A → ... → StageVI → y` 前向链。

| # | 文件 | 行数 | 阶段 | 机制 | Triton | gradcheck |
|---|---|---|---|---|---|---|
| 1 | `vacuum_zpe_fluctuation_operators.py` | 961 | 0.A | 真空零点能基态、正则化零点涨落 | 4 | — |
| 2 | `baryogenesis_parity_operators.py` | 923 | 0.B | Lee-Yang-Wu 宇称破缺、Sakharov CP 相位、正反物质不完全湮灭 | 5 | ✅ |
| 3 | `supermembrane_spacetime_operators.py` | 889 | I.A | 11D 超膜、DBI 相对论膜应力、RSKK 紧致化 | 5 | ✅ |
| 4 | `quantum_tidal_operators.py` | 887 | I.B | 量子潮汐力、WKB 隧穿率 | 6 | ✅ |
| 5 | `quantum_resource_phasing_operators.py` | 1087 | I.C | 量子资源相位、DQI 块门控、1/R(ε) 可靠性 | 2 | ✅×5 |
| 6 | `spinal_cord_pathway_operators.py` | 960 | II.A | 脊髓背角反射、Renshaw 递归抑制、交互抑制 | 5 | ✅ |
| 7 | `corticospinal_chaotic_operators.py` | 963 | II.B | 皮质脊髓混沌轴、Lorenz-Haken 有界吸引子、Pyragas 时滞镇定 | 5 | ✅ |
| 8 | `dissociative_trauma_neuro_operators.py` | 737 | II.C | 解离谱系、Lanius 双模皮质边缘调控、PAG 关断 | 3 | — |
| 9 | `moa_operators.py` | 1170 | III.Col0 | **跨模态 VL 融合**（2D patch-grid + 潜在锚点共注意 + Hadamard 双线性）<br>另含 `MixtureOfAttentionOp` 与 `CrossProposalAggregatorOp`（MoE 路由） | 4 | — |
| 10 | `neuro_nonlinear_operators.py` | 818 | III.Col1 | 锥体细胞**顶端树突 NMDA**/shunting/FHN 脉冲 | 5 | ✅ |
| 11 | `local_self_modifying_operators.py` | 1432 | III.Col2 | **瞬时有理 Hebbian + 持久 Delta-Oja 快权重**（SRAM 单发射前向+反向） | 5 | ✅ |
| 12 | `cpu_gpu_self_bootstrapping_operators.py` | 2325 | IV | **Tri-ISA 自举**（x86-64 / ARM64 / RISC-V / SIMT GPU）+ C99 `.so` 三阶段引导 + SHA256 等价校验 | 3 | — |
| 13 | `tulpa_neuro_operators.py` | 880 | V | 多吸引子双稳态代理、TPJ 归因、伴随放电衰减 | 5 | — |
| 14 | `sleep_subconscious_neuro_operators.py` | 1040 | VI | **睡眠巩固**：VLPO 门控 + Tononi SHY 下调 + 类淋巴冲刷 + 海马 SWR 重放 + REM PGO 重组 | 4 | — |

> 主干算子之间有 `zero_cross_talk = 0.00e+00` 的保证（实测，见 `docs/operator_independence_audit.log`）。

### 三柱的分工（Stage III 是唯一真正的 MoE）

| 柱 | 文件 | 专家类型 | 在 SST-2 上的实测负载 |
|---|---|---|---|
| Col0 | `moa_operators.py` | 跨模态 VL 融合 | 0.5005 |
| Col1 | `neuro_nonlinear_operators.py` | 树突 NMDA-FHN | 0.2964 |
| Col2 | `local_self_modifying_operators.py` | 快权重 Delta-Oja | 0.2032 |

路由熵 1.031 / 上限 1.099，均衡系数 0.878 —— **路由健康，未塌缩**。

---

## 二、基础设施算子（5 个，不进入主干，是可选替换件）

这批解决的是「数值方法」与「分布式执行」，与生物学机制无关。

| 文件 | 行数 | 提供什么 | 与主干的关系 |
|---|---|---|---|
| `high_order_pde_ode_megakernels.py` | 1879 | Störmer-Verlet 辛蛙跳 / Crank-Nicolson IMEX / RK2 中点 / DBI 规范膜 | **可替换 Stage I.A/I.B 的积分器**；也自带 Roofline 分析器 |
| `adaptive_checkpointed_adjoint_megakernels.py` | 1584 | O(√L·D) 分段检查点伴随 + 局部相空间误差自适应步长 | **长序列训练的内存方案**，可套在任意阶段外 |
| `physical_wave_dynamics_operators.py` | 2063 | Schwinger ZPE 波包 / 阻尼辛谐振 / 2D 反应扩散 / DBI 规范膜阻尼 | **Stage 0.A / I.A 的另一种物理实现**（6 类 + 10 个 Triton kernel） |
| `distributed_sequence_parallel_adjoint.py` | 854 | Ring/Ulysses 序列并行、伴随交接、计算-通信重叠 | **多卡训练的执行层** |
| `deepspeed_megatron_physics_plugin.py` | 1891 | 把 4 个物理阶段做成 DeepSpeed/Megatron 即插即用层；含 ZeRO-3 分片引擎 + SP P2P | **把主干塞进现有训练框架的适配层** |

这 5 个的共同特征：**有自己的 Triton kernel 或分布式原语，但不参与 14 算子的独立性审计**。

---

## 三、独立生物模型（1 个）

| 文件 | 行数 | 是什么 | 为什么独立 |
|---|---|---|---|
| `nbsc_omega_operator.py` | 446 | **脉冲神经元 + 脊髓牵张反射 + 肌骨力学 + STDP** 的完整小网络：42 神经元、202 条延迟边、858 个可训练参数 | 它不是「特征变换算子」，而是一个**完整的闭环运动控制系统**（有电压阈值、不应期、肌纤维力、关节限位）。输入不是 `[B,L,D]` 特征，而是 `(神经元驱动, 关节负载)` |

与主干的关系：**没有关系**。它是从零重写的独立模型，不与 14 算子共享张量协议。

---

## 四、组织层与审计（3 个）

| 文件 | 行数 | 作用 |
|---|---|---|
| `unified_hybrid_moa_architecture.py` | 1751 | **把 14 个主干算子串成 7 阶段前向链**，含 14 个可学习门控（`stage*a_*_gate`）+ 三柱迭代精炼 + 聚合 |
| `moa_triton_kernels.py` | 469 | `moa_operators.py` 的 Triton 内核实现（5 个 `@triton.jit`） |
| `operator_independence_verification.py` | 658 | **五维独立性审计**：隔离性 / 表示秩 / 梯度 VIF / 参数交叉灵敏 / 留一 CV |

---

## 五、四种分类维度（回答「这些算子到底差在哪」）

### 维度 1：是否可微、是否可训练

| 类别 | 文件 |
|---|---|
| 有 `torch.autograd.Function` + gradcheck | baryogenesis / supermembrane / quantum_tidal / quantum_resource(×5) / spinal / corticospinal / neuro_nonlinear / local_self_modifying |
| 有 Triton kernel 但无 gradcheck | vacuum_zpe / dissociative_trauma / moa_operators / cpu_gpu_bootstrap / tulpa / sleep / moa_triton_kernels |
| 纯 PyTorch 自动微分（无手写反向） | **nbsc_omega_operator**（5 组参数全部走 autograd） |

### 维度 2：有没有 Triton GPU 内核

| Triton 内核数 | 文件 |
|---|---|
| 10 | `physical_wave_dynamics_operators` |
| 7 | `deepspeed_megatron_physics_plugin` |
| 6 | `quantum_tidal_operators` |
| 5 | baryogenesis / supermembrane / spinal / corticospinal / neuro_nonlinear / local_self_modifying / high_order_pde / moa_triton_kernels |
| 4 | vacuum_zpe / moa_operators / sleep |
| 3 | dissociative_trauma / cpu_gpu_bootstrap / adaptive_checkpointed |
| 2 | quantum_resource |
| **0** | distributed_sequence_parallel / **nbsc_omega** / unified_hybrid |

### 维度 3：机制属于哪个学科

| 学科 | 算子 |
|---|---|
| 量子场论 / 粒子物理 | vacuum_zpe（ZPE）、baryogenesis（宇称+CP）、supermembrane（11D DBI）、quantum_tidal（WKB）、quantum_resource（DQI） |
| 神经生理 | spinal（Renshaw）、corticospinal（混沌）、neuro_nonlinear（NMDA）、local_self_modifying（Oja）、sleep（SHY/类淋巴/SWR） |
| 精神医学 | dissociative_trauma（Lanius/DPDR）、tulpa（TPJ 归因） |
| 计算机体系结构 | cpu_gpu_bootstrap（Tri-ISA） |
| 跨模态学习 | moa_operators（VL 融合） |
| 数值分析 | high_order_pde、adaptive_checkpointed |
| 分布式系统 | distributed_sequence_parallel、deepspeed_plugin |
| 生物力学 | **nbsc_omega**（肌纤维力/力矩/限位） |

### 维度 4：代码规模梯队

| 规模 | 算子 |
|---|---|
| 2000+ 行 | cpu_gpu_bootstrap(2325)、physical_wave(2063)、deepspeed_plugin(1891)、high_order_pde(1879) |
| 1400–1800 | unified_hybrid(1751)、adaptive_checkpointed(1584)、local_self_modifying(1432) |
| 900–1200 | moa_operators(1170)、quantum_resource(1087)、sleep(1040)、spinal(960)、vacuum_zpe(961)、corticospinal(963)、baryogenesis(923) |
| 700–900 | supermembrane(889)、quantum_tidal(887)、tulpa(880)、distributed_sequence_parallel(854)、neuro_nonlinear(818) |
| < 700 | dissociative_trauma(737)、moa_triton_kernels(469)、**nbsc_omega(446)**、审计(658)、`__init__`(131) |

---

## 六、一句话区分

- **主干 14 个**：统一 `[B,L,D] → [B,L,D]` 协议，串成 7 阶段，零交叉干扰，被审计覆盖。
- **基础设施 5 个**：换的是「怎么算」不是「算什么」——积分器、检查点、内存、多卡、框架适配。
- **nbsc_omega 1 个**：唯一的**完整闭环生物系统**，有脉冲和肌肉，不参与主干。
- **组织层 3 个**：串接、内核实现、审计。
