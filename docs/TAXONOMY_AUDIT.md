# 架构分类学声明核对报告（Taxonomy Audit）
> **历史记录说明（2026-11-10 补记）**
>
> 本文档是 2026-11-05 对当时外部提出的「六阶十一域」分类学做的逐条代码核对，
> 其中引用的 `unified_hybrid_moa_architecture.py` 已于 2026-11-10 删除
> （用户指示移除编排骨架、只保留算子）。
> 文中所有 `file:line` 引用指向的是**当时的代码状态**，未随删除同步更新——
> 保留原样是为了让这份核对记录保持可追溯，不代表这些行号现在仍然有效。
> 涉及编排层的断言在当前仓库里已无对应代码；各算子自身的验证套件、
> 以及 `operator_independence_verification.py` 的五维审计仍然有效。


对被外部提出的一套「六阶十一域混合架构」分类学定位做逐条代码核对。
核对对象：commit `e926c3c` 的 `hybrid_moa_vl_operator_suite/`（16 个 `.py`，12,894 行）。
核对方式：声明 → 代码行 → 在本仓库净室环境实测（不是引用作者文档）。
生成日期：2026-11-05。

---

## 0. 总结论

该分类学**主体成立**，异常扎实：它引用的公式、结构共性与机制声明绝大多数能在代码里逐行对上，
包括几条不显而易见的断言（潜锚点在 Stage 0.A / V.A 的复用、类淋巴滤波的精确形式、
以及全篇最硬的数学断言——创伤解耦的精确零灵敏）。

但有 **3 处夸大或改写**，以及 **1 条它遗漏的、比它自己所举证据更强的形式化性质**。
若要把这套标签用于论文或对外材料，建议按第 3、4 节修正。

---

## 1. 核对通过的声明

| # | 分类学声明 | 代码位置（`hybrid_moa_vl_operator_suite/`） | 判定 |
|---|---|---|---|
| 1 | 类淋巴冲刷 `W_glymph = 1 - G_sleep·ε/(x²+ε)` | `sleep_subconscious_neuro_operators.py:143`（Triton）、`:590`（numpy），解析反向 `:195`/`:226` | ✅ 一字不差，且反向解析式在手 |
| 2 | Stage 0.A 与 Stage V.A 也用 `R≪L` 潜锚点 | 真空：`vacuum_zpe_fluctuation_operators.py:423-429`（`attn_emit[B,R_vac,L]→v_boson[B,R_vac,D]→attn_abs[B,L,R_vac]`）；睡眠：`sleep_subconscious_neuro_operators.py:477,486-487`（`scores_pool[B,R,L]→z_dream→z_sub[B,L,D]`） | ✅ 属实。这是该分析抓到的一个真结构性共性，不是修辞 |
| 3 | 有界 Hadamard 双线性 `(V⊙T+C)/√(1+λ(V²+T²))` | `moa_operators.py:122`（门）、`:126`（插值）、`:132`（归一化）、`:136`（合成） | ✅ 一致 |
| 4 | Renshaw 分式递归负反馈 | `spinal_cord_pathway_operators.py:114-115`（Triton：`Out = R/(1+λ_rc·R²) + α_cpg·sin(ω_cpg·l)`）、`:332`（numpy：`m_renshaw = r_reflex/denom_rc`）、`:111`（Sherrington 交互抑制 `R_f = S_f - γ_ia·(S_e/(1+S_e²))`）；公式注释在 `:23-29` | ✅ 一致，含交互抑制与 CPG 半中枢振荡 |
| 5 | Pyragas 时滞反馈镇定 + 有限速度轴突延迟 | `corticospinal_chaotic_operators.py:38`（延迟）、`:82`（`kappa_ctrl`）、`:117`（闭环偏差反馈） | ✅ 一致 |
| 6 | Stage III 为 token 级 top-k 路由 + 负载均衡 + Z-loss | `moa_operators.py:369`（`argpartition` top-k）、`:364`（`z_loss = mean(lse²)`）、`:383`（`load_balance = N·Σf_i·p_i`） | ✅ 一致 |
| 7 | 统一层的 Stage III 确实由 `CrossProposalAggregatorOp` + Triton megakernel 承担 | `unified_hybrid_moa_architecture.py:360`（实例化，`aggregator_mode="sparse_router"`）、`:418`（调用） | ✅ 一致 |
| 8 | Pre-RMSNorm 残差外壳 | 统一层 `stage0_norm / stage1_norm / stage2_norm / stage3_norm` 先归一化再进阶段算子；各领域模块自带 `RMSNorm` | ✅ 一致 |
| 9 | **`‖∂(Cortex,Tulpa,Sleep)/∂Θ_Trauma‖_∞ ≡ 0`** | 本仓库 `smoke_test.py:434`（`test_mechanism_stage_decoupling`）独立复算：Stage 0.A / 0.B / III / IV / V 与干净感觉运动流（`reflex_bypass`、`mega_syn`）对创伤参数扰动**全部精确 `0.0e+00`**，同时 `trauma_out` 自身变化 20.9 作对照；上游同一断言在 `unified_hybrid_moa_architecture.py:1348-1368`（Verification V） | ✅ **全篇最硬的一条，成立** |
| 10 | 单次前向内嵌「清醒加工 → 睡眠巩固」闭环 | `unified_hybrid_moa_architecture.py:883-884`（ctx 暴露 `h_wake` 与 `y_consolidated`），SWR/PGO/SHY/类淋巴均在前向图末尾 | ✅ 属实且确实罕见 |
| 11 | 潜锚点把交互复杂度从 O(L²D) 压到 O(L·R·D) | 上述 2 的 einsum 全部是 `brl/bld/brd` 三级，无 `L×L` 项 | ✅ 一致 |
| 12 | 生物物理ODE 被重参数化为单步解析可微算子 | 各领域模块的 `Fused*Function` + `*AnalyticalKernels.*_backward`；上游 gradcheck：解离 7 变量 rel-L2 6.7e-11–7.5e-10，统一层全 6 阶段输入 gradcheck < 1e-8 | ✅ 一致 |
| 13 | 五族映射（Sparse MoE/MoA、潜瓶颈双线性、快权重/TTT、生物物理 ODE/LTC、全局工作空间/预测编码） | 分别对应 Stage III 三柱、Stage III.Col0 + 0.A + V.A、Stage III.Col2、Stage II.A/B + III.Col1、Stage IV | ✅ 除第 14 行外均可成立 |

---

## 2. 需要修正的 3 处

### 2.1 「ZPE 下限封死梯度消失与死神经元」——机制比这个小

- 代码：`vacuum_zpe_fluctuation_operators.py:143,193` — `E_vac = q_sq² + p_sq² + ω² + eps_zpe_floor`
- `E_vac` 的唯一下游用途是 `:118` 的 `G_schw = exp(-m_gap²/E_vac)` 与 `:41` 的 `/sqrt(E_vac)`。
- 因此该下限的真实作用是**防止 Schwinger 对产生门的指数项除零塌陷**，是针对单项的数值安全阀。
- 修正建议：改为「保证真空能量分母不为零，使 Schwinger 对产生门永不退化为精确 0」。
  「根治死神经元/梯度消失」是把单项安全性推广成了普适保证，代码不支持。

### 2.2 DBI 公式被改写

- 分类学写 `1/√(1+α‖X‖²)`。
- 代码：`supermembrane_spacetime_operators.py:306-307` — `dbi_arg = 1.0 + tau_brane²·E_stretch + beta_flux²·F²`，`dbi_factor = 1/√(dbi_arg)`。
- 自变量是**膜张力能与磁通**，不是特征范数。方向（有界阻尼因子）对，但这是类比而非恒等。
- 修正建议：按代码写 `D_dbi(E_stretch, F) = 1/√(1+τ²·E_stretch + β²·F²)`，或明确标注为「物理原型的有界化类比」。

### 2.3 「可无缝替换标准 Transformer 层」——只对了一半

- 形状兼容成立：`[B,L,D] → [B,L,D]`。
- 但：(a) 19 个自有配置字段（`num_vacuum_modes`、`num_dream_anchors`、`num_local_groups`、`local_rank`、`num_extra_dims`…）；(b) 单层计算量远大于一层 Transformer（6 阶段 × 12 算子 + 每阶段 Triton megakernel）；(c) token 混合从「全序列注意力」换成「潜锚点 + 3 柱路由」，归纳偏置被整体替换；(d) `UnifiedHybridMoABackbone` 的设计意图是**骨干网整体替换**，不是层内 drop-in。
- 修正建议：改为「形状与训练管线兼容，可作为骨干网的层单元使用；但不是标准 Transformer 层的等计算量替换品」。

---

## 3. 该分析遗漏的最强证据

要论证「无法归入现有 MoE」，比它所举的四个结构特征更硬的是一条已实测的形式化性质：

> **12×12 参数交叉灵敏雅可比，非对角交叉干扰精确 = 0.00e+00**
> （Kaggle CPU kernel `redmingwei/hetero-moe-operator-independence`，59.9 s；
>  完整日志：`docs/operator_independence_audit.log`，第四维）

标准 MoE 中扰动路由器参数会改变所有专家的输出；本架构中扰动算子 j 的参数，
算子 i 的输出**逐位不变**，而对角自灵敏全部 > 0（最小 8.16e-02，12 个算子全 active）。
这使「异构算子柱」从修辞变成可测量事实。配套证据：

| 维度 | 实测 |
|---|---|
| II 前向表示独立 | 联合子空间秩 **12/12 满秩**；平均 off-diag \|cos\| 0.0844；最大 VIF 1.72 < 2.0 |
| III 输入梯度子空间 | 梯度矩阵秩 **12/12**；平均 \|cos(Gi,Gj)\| 0.0934（最大 0.5803 < 0.80）；最大梯度 VIF 1.90 < 5.0 |
| V 留一算子 CV | 全集成 CV R² **0.9997**；12 个算子全部不可替代，独有 ΔR² **+0.0306 ~ +0.0509** |

此外它未提及的另一条独立工程性质：**numpy / torch / Triton 三套实现位级一致**
（RMSNorm 1.8e-15、Triton tile 模拟器 2.2e-16 / 8.9e-16、eager vs megakernel < 1e-14）。
这让「物理-神经动力学算子」不只是纸上模型，而是可复算的实现。

---

## 4. 对标签与论证结构的建议

它提议的名字「Ontologically Stratified & Regime-Gated Physico-Neuro-Dynamical Hybrid MoA
Architecture」可以保留，但两点建议：

1. **把论证重心从「阶段顺序不可逆」移到「跨阶段梯度隔离」。**
   代码里阶段顺序是写死的执行顺序，但没有任何结构阻止重排——「不可互换」是语义约定，
   不是不变量。真正被结构保证的是解耦性（梯度路径精确为零，见 `smoke_test.py:434`
   与上游 `unified_hybrid_moa_architecture.py:1348`）。后者强度高一个量级，且可复算。

2. **五族映射里 GWT / 预测编码 ↔ Stage IV（Tulpa）是最松的一条。**
   Tulpa 的实质是多子人格吸引子 + 伴随放电感官衰减 + TPJ 主体感归因门控，
   与「全局工作空间广播」只共享「多个内部子状态竞争」这一个模糊共性。
   建议标注为 loosest mapping，或改挂「多吸引子 / 双稳态动力学系统」家族。

---

## 5. 复现方式

```bash
# 1) 本仓库净室测试（含机制解耦审计，CPU + numpy，约 5 分钟）
python smoke_test.py

# 2) 上游自带验证套件（gradcheck / Triton 等价 / LOSO / Verification V，约 11 秒）
python3 hybrid_moa_vl_operator_suite/unified_hybrid_moa_architecture.py

# 3) 五维算子独立性审计（Kaggle CPU kernel，约 60 秒）
#    https://www.kaggle.com/code/redmingwei/hetero-moe-operator-independence
#    本地亦可跑，但沙箱 CPU 受限时 Dimension I（13 个全新解释器）会显著变慢
python3 hybrid_moa_vl_operator_suite/operator_independence_verification.py
```

本文件只做核对，不修改 `hybrid_moa_vl_operator_suite/` 内任何文件
（该目录与上游压缩包逐字节一致）。
