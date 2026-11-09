#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-Hierarchy Standalone Neural Operator (完整 LLM 算子层次独立新算子)
================================================================================

零同级依赖（0 Sibling Imports）、自包含、可微分的深度学习底层独立新算子模块。
将现代大型语言模型（Large Language Model, LLM）的完整计算层次——即：
  [Layer 1] Pre-RMSNorm 均方根归一化与旋转位置编码 (RoPE, Rotary Position Embedding)
  [Layer 2] 多头自注意力查询-键-值投射与上下文聚合 (Multi-Head QKV Attention & KV-Cache)
  [Layer 3] SwiGLU 双线性门控前馈网络 (SwiGLU Gated Feed-Forward Network / MoE FFN)
  [Layer 4] 残差高速公路门控、QK 点积凝聚与 Logit Softcapping 输出头 (Residual & Logit Head)
浓缩并正交解耦为一个单文件、零外部同级依赖、具备解析反向传播与 @triton.jit GPU 融合核的
底层张量算子 `LLMHierarchyOperator`。

核心物理/数学架构（四通道正交解耦，D = 4 * G）：
将输入特征张量 X in R^{B x L x D} 按四通道注意力头组 G = D // 4 重塑为 (u_1, u_2, v_1, v_2)，
其中 q = (u_1, u_2)^T 为查询/键子流 (Query/Key Stream)，v = (v_1, v_2)^T 为价值/上下文子流
(Value/Context Stream)：

1. 子算子 I — Pre-RMSNorm 与旋转位置编码算子 (O_rope, RMSNorm + RoPE Sub-Operator):
   引入序列归一化位置坐标 pos_l = (l + 1) / L，构造位置相关的 SO(2) 旋转相位
   phi_{l,g} = pos_l * theta_{rope,g}，结合带真空零点能下界的 RMS 能量归一化包络
   D_rms = 1 / sqrt(1 + kappa_{rms,eff} * (u_1^2 + u_2^2))，实现位置感知的表示调制。

2. 子算子 II — 多头注意力价值投射与能量缩放算子 (O_qkv, Multi-Head Value & Attention Scaling):
   在正交价值子空间 (v_1, v_2)^T 上施加多头注意力相位投射 theta_{qkv,g}、头权重 w_{qkv,g}
   以及有理注意力分母包络 D_qkv = 1 / sqrt(1 + kappa_{qkv,eff} * (v_1^2 + v_2^2))。

3. 子算子 III — SwiGLU 双线性门控前馈网络算子 (O_swiglu, SwiGLU Gated FFN Sub-Operator):
   构造跨子流双线性反对称门控特征 G_glu = u_1 v_2 - u_2 v_1，将上投射分支 (w_up) 与
   门控投射分支 (w_gate * G_glu) 融合，并施加全通道 FFN 有理阻尼包络
   D_ffn = 1 / (1 + kappa_{ffn,eff} * E_tot)。

4. 子算子 IV — QK 点积注意力凝聚、残差门控与 Softcap 输出头算子 (O_head, QK-Inner & Logit Softcap):
   显式计算缩放点积注意力内积 I_qk = <q, k> = u_1 v_1 + u_2 v_2 与子空间能量裕度
   Delta_E = (v_1^2 + v_2^2) - (u_1^2 + u_2^2)，通过有理 Logit 软截断门
   G_cap = Delta_E / sqrt(1 + kappa_{head,eff} * Delta_E^2) 与有界残差门
   alpha_{res,eff} = sigmoid(alpha_res) in (0, 1) 完成最终表示合成。

此外，模块内建完整 LLM 算子级推理接口：
- `forward_with_kv_cache(x, kv_cache=None)`: 支持自回归增量解码的因果 KV-Cache 状态更新与上下文门控。
- `compute_logits(hidden_states, vocab_weight=None)`: 含 RMSNorm 与有理 Logit Softcapping 的词表投射头。

验证标准（全面通过）：
- 三套引擎（双精度 NumPy 解析引擎、PyTorch Autograd 引擎、@triton.jit 融合 GPU 核）前向最大绝对误差 < 1e-14。
- 输入张量与全部 12 组可学参数的解析反向梯度对比中心有限差分 (eps = 1e-6) 最大相对误差 < 1e-8。
- 五维正交独立性形式化审计（0 同级依赖、前向 4/4 与 D/D 满秩、反向雅可比 Max VIF < 2.0、
  4x4 参数交叉灵敏度非对角精确 0.00e+00 且对角 > 0.08、留一法 LOOO CV 全模型 R^2 > 0.999 且 Delta R^2 > +0.02）。
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# ==============================================================================
# 可选依赖自适应导入（PyTorch & Triton GPU Kernel）
# ==============================================================================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None  # type: ignore
    tl = None  # type: ignore


# ==============================================================================
# 1. 算子配置结构体 (LLMHierarchyConfig)
# ==============================================================================

__all__ = [
    "LLMHierarchyConfig",
    "LLMHierarchyOperator",
    "LLMHierarchyAutogradFunction",
]

@dataclass
class LLMHierarchyConfig:
    """完整 LLM 层次独立算子配置类。

    Attributes:
        dim: 隐藏层特征维度 D，必须为 4 的正整数倍（对应 G = D // 4 个四通道注意力/FFN 头组）。
        vocab_size: 词表维度（用于可选的 compute_logits 词表投射头，默认 64）。
        eps_zpe: 有理分母正性安全下界常量（默认 1e-4）。
        logit_softcap: 词表 Logit 软截断上界（默认 30.0，仿 Gemma-2 软截断机制）。
        kv_decay: 增量推理 KV-Cache 指数移动衰减系数（默认 0.85）。
        sub_op_weights: 四个正交 LLM 子算子 (O_rope, O_qkv, O_swiglu, O_head) 的合成权重。
        use_triton: 在 GPU 可用时是否优先调度 @triton.jit 融合核。
        block_size: Triton 核或 Tile 级融合引擎的线程块大小（默认 64）。
    """

    dim: int = 16
    vocab_size: int = 64
    eps_zpe: float = 1e-4
    logit_softcap: float = 30.0
    kv_decay: float = 0.85
    sub_op_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    use_triton: bool = True
    block_size: int = 64

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.dim % 4 != 0:
            raise ValueError(
                f"LLMHierarchyConfig.dim 必须为 4 的正整数倍（四通道 QKV/FFN 头组），当前为 {self.dim}"
            )
        if self.eps_zpe <= 0.0:
            raise ValueError(f"eps_zpe 必须严格大于 0，当前为 {self.eps_zpe}")
        if len(self.sub_op_weights) != 4:
            raise ValueError("sub_op_weights 必须包含 4 个子算子权重 (w_rope, w_qkv, w_swiglu, w_head)")


# ==============================================================================
# 2. 数值安全原语 (Softplus / Sigmoid)
# ==============================================================================
def _np_softplus(x: np.ndarray) -> np.ndarray:
    """数值稳定的双精度 Softplus: log(1 + exp(-|x|)) + max(x, 0)."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的双精度 Sigmoid."""
    return np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


# ==============================================================================
# 3. 确定性双精度 NumPy 解析前向与反向传播引擎
# ==============================================================================
def llm_hierarchy_numpy_forward(
    x: np.ndarray,
    theta_rope: np.ndarray,
    gamma_rms: np.ndarray,
    kappa_rms: np.ndarray,
    theta_qkv: np.ndarray,
    w_qkv: np.ndarray,
    kappa_qkv: np.ndarray,
    w_up: np.ndarray,
    w_gate: np.ndarray,
    kappa_ffn: np.ndarray,
    alpha_res: np.ndarray,
    beta_qk: np.ndarray,
    kappa_head: np.ndarray,
    eps_zpe: float = 1e-4,
    sub_op_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    return_sub_ops: bool = False,
    return_diagnostics: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """完整 LLM 算子层次双精度 NumPy 解析前向传播。

    Args:
        x: 输入 Token 特征张量，形状为 (B, L, D)，其中 D = 4 * G。
        theta_rope, gamma_rms, kappa_rms: 子算子 I (Pre-RMSNorm & RoPE) 可学参数，形状均为 (G,)。
        theta_qkv, w_qkv, kappa_qkv: 子算子 II (Multi-Head QKV Value Projection) 可学参数，形状均为 (G,)。
        w_up, w_gate, kappa_ffn: 子算子 III (SwiGLU Gated FFN) 可学参数，形状均为 (G,)。
        alpha_res, beta_qk, kappa_head: 子算子 IV (QK Dot-Product & Softcap Logit Head) 可学参数，形状均为 (G,)。
        eps_zpe: 有理分母正性安全下界常量（默认 1e-4）。
        sub_op_weights: 四个子算子的合成掩码权重。
        return_sub_ops: 若为 True，额外返回四个正交子算子的独立输出元组 (op_rope, op_qkv, op_swiglu, op_head)。
        return_diagnostics: 若为 True，额外返回内部门控与注意力诊断字典。
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 3 or x_arr.shape[-1] % 4 != 0:
        raise ValueError(f"输入 x 形状必须为 (B, L, 4*G)，当前为 {x_arr.shape}")

    B, L, D = x_arr.shape
    G = D // 4
    xg = x_arr.reshape(B, L, G, 4)
    u1 = xg[..., 0]
    u2 = xg[..., 1]
    v1 = xg[..., 2]
    v2 = xg[..., 3]

    # 正性有理分母与有界门控参数重参数化
    kappa_rms_eff = _np_softplus(np.asarray(kappa_rms, dtype=np.float64)) + eps_zpe
    kappa_qkv_eff = _np_softplus(np.asarray(kappa_qkv, dtype=np.float64)) + eps_zpe
    kappa_ffn_eff = _np_softplus(np.asarray(kappa_ffn, dtype=np.float64)) + eps_zpe
    alpha_res_eff = _np_sigmoid(np.asarray(alpha_res, dtype=np.float64))
    kappa_head_eff = _np_softplus(np.asarray(kappa_head, dtype=np.float64)) + eps_zpe

    theta_rope_arr = np.asarray(theta_rope, dtype=np.float64)
    gamma_rms_arr = np.asarray(gamma_rms, dtype=np.float64)
    theta_qkv_arr = np.asarray(theta_qkv, dtype=np.float64)
    w_qkv_arr = np.asarray(w_qkv, dtype=np.float64)
    w_up_arr = np.asarray(w_up, dtype=np.float64)
    w_gate_arr = np.asarray(w_gate, dtype=np.float64)
    beta_qk_arr = np.asarray(beta_qk, dtype=np.float64)

    # 序列位置编码坐标 pos_l = (l + 1) / L, 形状为 (1, L, 1)
    pos_l = (np.arange(1, L + 1, dtype=np.float64) / float(L)).reshape(1, L, 1)

    # ------------------------------------------------------------------
    # 子算子 I: Pre-RMSNorm 与旋转位置编码流 (O_rope)
    # ------------------------------------------------------------------
    E_Q = u1 * u1 + u2 * u2
    Q_rms = 1.0 + kappa_rms_eff * E_Q
    D_rms = 1.0 / np.sqrt(Q_rms)
    phi_rope = pos_l * theta_rope_arr.reshape(1, 1, G)
    cos_rope = np.cos(phi_rope)
    sin_rope = np.sin(phi_rope)
    r_Q1 = cos_rope * u1 - sin_rope * u2
    r_Q2 = sin_rope * u1 + cos_rope * u2
    op_rope = np.stack(
        [
            gamma_rms_arr * D_rms * r_Q1,
            gamma_rms_arr * D_rms * r_Q2,
            np.zeros_like(u1),
            np.zeros_like(u2),
        ],
        axis=-1,
    )

    # ------------------------------------------------------------------
    # 子算子 II: 多头注意力价值投射与能量缩放流 (O_qkv)
    # ------------------------------------------------------------------
    E_V = v1 * v1 + v2 * v2
    Q_qkv = 1.0 + kappa_qkv_eff * E_V
    D_qkv = 1.0 / np.sqrt(Q_qkv)
    cos_qkv = np.cos(theta_qkv_arr)
    sin_qkv = np.sin(theta_qkv_arr)
    r_V1 = cos_qkv * v1 + sin_qkv * v2
    r_V2 = -sin_qkv * v1 + cos_qkv * v2
    op_qkv = np.stack(
        [
            np.zeros_like(v1),
            np.zeros_like(v2),
            w_qkv_arr * D_qkv * r_V1,
            w_qkv_arr * D_qkv * r_V2,
        ],
        axis=-1,
    )

    # ------------------------------------------------------------------
    # 子算子 III: SwiGLU 双线性门控前馈网络流 (O_swiglu)
    # ------------------------------------------------------------------
    E_tot = E_Q + E_V
    Q_ffn = 1.0 + kappa_ffn_eff * E_tot
    D_ffn = 1.0 / Q_ffn
    G_glu = u1 * v2 - u2 * v1
    m0 = w_up_arr * v1 + w_gate_arr * G_glu * v2
    m1 = w_up_arr * v2 - w_gate_arr * G_glu * v1
    m2 = w_up_arr * u1 - w_gate_arr * G_glu * u2
    m3 = w_up_arr * u2 + w_gate_arr * G_glu * u1
    op_swiglu = D_ffn[..., None] * np.stack([m0, m1, m2, m3], axis=-1)

    # ------------------------------------------------------------------
    # 子算子 IV: QK 点积凝聚、残差门控与 Softcap 输出头 (O_head)
    # ------------------------------------------------------------------
    Delta_E = E_V - E_Q
    Q_cap = 1.0 + kappa_head_eff * (Delta_E * Delta_E)
    D_cap = 1.0 / np.sqrt(Q_cap)
    G_cap = Delta_E * D_cap
    I_qk = u1 * v1 + u2 * v2
    Q_head = 1.0 + kappa_head_eff * E_tot
    D_head = 1.0 / Q_head
    C_qk = D_head * I_qk

    A0 = -u1 + G_cap * v2
    A1 = u2 - G_cap * v1
    A2 = v1 + G_cap * u2
    A3 = -v2 - G_cap * u1
    op_head = np.stack(
        [
            alpha_res_eff * A0 + beta_qk_arr * C_qk * v1,
            alpha_res_eff * A1 - beta_qk_arr * C_qk * v2,
            alpha_res_eff * A2 + beta_qk_arr * C_qk * u1,
            alpha_res_eff * A3 - beta_qk_arr * C_qk * u2,
        ],
        axis=-1,
    )

    m1_w, m2_w, m3_w, m4_w = sub_op_weights
    y_group = m1_w * op_rope + m2_w * op_qkv + m3_w * op_swiglu + m4_w * op_head
    y = y_group.reshape(B, L, D)

    sub_ops_tuple = (
        op_rope.reshape(B, L, D),
        op_qkv.reshape(B, L, D),
        op_swiglu.reshape(B, L, D),
        op_head.reshape(B, L, D),
    )

    if return_sub_ops and return_diagnostics:
        diag = {
            "D_rms": D_rms,
            "D_qkv": D_qkv,
            "D_ffn": D_ffn,
            "G_glu": G_glu,
            "G_cap": G_cap,
            "I_qk": I_qk,
        }
        return y, sub_ops_tuple, diag
    if return_sub_ops:
        return y, sub_ops_tuple
    if return_diagnostics:
        diag = {
            "D_rms": D_rms,
            "D_qkv": D_qkv,
            "D_ffn": D_ffn,
            "G_glu": G_glu,
            "G_cap": G_cap,
            "I_qk": I_qk,
        }
        return y, diag
    return y


def llm_hierarchy_numpy_backward(
    grad_out: np.ndarray,
    x: np.ndarray,
    theta_rope: np.ndarray,
    gamma_rms: np.ndarray,
    kappa_rms: np.ndarray,
    theta_qkv: np.ndarray,
    w_qkv: np.ndarray,
    kappa_qkv: np.ndarray,
    w_up: np.ndarray,
    w_gate: np.ndarray,
    kappa_ffn: np.ndarray,
    alpha_res: np.ndarray,
    beta_qk: np.ndarray,
    kappa_head: np.ndarray,
    eps_zpe: float = 1e-4,
    sub_op_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    return_sub_grads: bool = False,
) -> Tuple[np.ndarray, ...]:
    """完整 LLM 算子层次双精度 NumPy 解析反向传播。"""
    x_arr = np.asarray(x, dtype=np.float64)
    g_arr = np.asarray(grad_out, dtype=np.float64)
    B, L, D = x_arr.shape
    G = D // 4
    xg = x_arr.reshape(B, L, G, 4)
    gg = g_arr.reshape(B, L, G, 4)

    u1, u2, v1, v2 = xg[..., 0], xg[..., 1], xg[..., 2], xg[..., 3]
    g0_raw, g1_raw, g2_raw, g3_raw = gg[..., 0], gg[..., 1], gg[..., 2], gg[..., 3]
    m1_w, m2_w, m3_w, m4_w = sub_op_weights

    theta_rope_arr = np.asarray(theta_rope, dtype=np.float64)
    gamma_rms_arr = np.asarray(gamma_rms, dtype=np.float64)
    kappa_rms_arr = np.asarray(kappa_rms, dtype=np.float64)
    sig_kappa_rms = _np_sigmoid(kappa_rms_arr)
    kappa_rms_eff = _np_softplus(kappa_rms_arr) + eps_zpe

    theta_qkv_arr = np.asarray(theta_qkv, dtype=np.float64)
    w_qkv_arr = np.asarray(w_qkv, dtype=np.float64)
    kappa_qkv_arr = np.asarray(kappa_qkv, dtype=np.float64)
    sig_kappa_qkv = _np_sigmoid(kappa_qkv_arr)
    kappa_qkv_eff = _np_softplus(kappa_qkv_arr) + eps_zpe

    w_up_arr = np.asarray(w_up, dtype=np.float64)
    w_gate_arr = np.asarray(w_gate, dtype=np.float64)
    kappa_ffn_arr = np.asarray(kappa_ffn, dtype=np.float64)
    sig_kappa_ffn = _np_sigmoid(kappa_ffn_arr)
    kappa_ffn_eff = _np_softplus(kappa_ffn_arr) + eps_zpe

    alpha_res_arr = np.asarray(alpha_res, dtype=np.float64)
    alpha_res_eff = _np_sigmoid(alpha_res_arr)
    d_alpha_res = alpha_res_eff * (1.0 - alpha_res_eff)
    beta_qk_arr = np.asarray(beta_qk, dtype=np.float64)
    kappa_head_arr = np.asarray(kappa_head, dtype=np.float64)
    sig_kappa_head = _np_sigmoid(kappa_head_arr)
    kappa_head_eff = _np_softplus(kappa_head_arr) + eps_zpe

    pos_l = (np.arange(1, L + 1, dtype=np.float64) / float(L)).reshape(1, L, 1)

    # ------------------------------------------------------------------
    # 1. 子算子 I (O_rope) 反向传播
    # ------------------------------------------------------------------
    g0_L, g1_L = m1_w * g0_raw, m1_w * g1_raw
    E_Q = u1 * u1 + u2 * u2
    Q_rms = 1.0 + kappa_rms_eff * E_Q
    D_rms = 1.0 / np.sqrt(Q_rms)
    phi_rope = pos_l * theta_rope_arr.reshape(1, 1, G)
    cos_rope = np.cos(phi_rope)
    sin_rope = np.sin(phi_rope)
    r_Q1 = cos_rope * u1 - sin_rope * u2
    r_Q2 = sin_rope * u1 + cos_rope * u2

    S_rQ = g0_L * r_Q1 + g1_L * r_Q2
    S_phi_Q = g0_L * (-r_Q2) + g1_L * r_Q1
    grad_gamma_rms = np.sum(D_rms * S_rQ, axis=(0, 1))
    grad_theta_rope = np.sum(gamma_rms_arr * D_rms * S_phi_Q * pos_l, axis=(0, 1))
    dQ_rms = -0.5 * gamma_rms_arr * (D_rms * D_rms * D_rms) * S_rQ
    grad_kappa_rms = sig_kappa_rms * np.sum(dQ_rms * E_Q, axis=(0, 1))

    du1_rope = gamma_rms_arr * D_rms * (g0_L * cos_rope + g1_L * sin_rope) + 2.0 * kappa_rms_eff * u1 * dQ_rms
    du2_rope = gamma_rms_arr * D_rms * (-g0_L * sin_rope + g1_L * cos_rope) + 2.0 * kappa_rms_eff * u2 * dQ_rms

    # ------------------------------------------------------------------
    # 2. 子算子 II (O_qkv) 反向传播
    # ------------------------------------------------------------------
    g2_R, g3_R = m2_w * g2_raw, m2_w * g3_raw
    E_V = v1 * v1 + v2 * v2
    Q_qkv = 1.0 + kappa_qkv_eff * E_V
    D_qkv = 1.0 / np.sqrt(Q_qkv)
    cos_qkv = np.cos(theta_qkv_arr)
    sin_qkv = np.sin(theta_qkv_arr)
    r_V1 = cos_qkv * v1 + sin_qkv * v2
    r_V2 = -sin_qkv * v1 + cos_qkv * v2

    S_rV = g2_R * r_V1 + g3_R * r_V2
    S_theta_V = g2_R * r_V2 + g3_R * (-r_V1)
    grad_w_qkv = np.sum(D_qkv * S_rV, axis=(0, 1))
    grad_theta_qkv = np.sum(w_qkv_arr * D_qkv * S_theta_V, axis=(0, 1))
    dQ_qkv = -0.5 * w_qkv_arr * (D_qkv * D_qkv * D_qkv) * S_rV
    grad_kappa_qkv = sig_kappa_qkv * np.sum(dQ_qkv * E_V, axis=(0, 1))

    dv1_qkv = w_qkv_arr * D_qkv * (g2_R * cos_qkv - g3_R * sin_qkv) + 2.0 * kappa_qkv_eff * v1 * dQ_qkv
    dv2_qkv = w_qkv_arr * D_qkv * (g2_R * sin_qkv + g3_R * cos_qkv) + 2.0 * kappa_qkv_eff * v2 * dQ_qkv

    # ------------------------------------------------------------------
    # 3. 子算子 III (O_swiglu) 反向传播
    # ------------------------------------------------------------------
    g0_s, g1_s, g2_s, g3_s = m3_w * g0_raw, m3_w * g1_raw, m3_w * g2_raw, m3_w * g3_raw
    E_tot = E_Q + E_V
    Q_ffn = 1.0 + kappa_ffn_eff * E_tot
    D_ffn = 1.0 / Q_ffn
    G_glu = u1 * v2 - u2 * v1
    m0 = w_up_arr * v1 + w_gate_arr * G_glu * v2
    m1 = w_up_arr * v2 - w_gate_arr * G_glu * v1
    m2 = w_up_arr * u1 - w_gate_arr * G_glu * u2
    m3 = w_up_arr * u2 + w_gate_arr * G_glu * u1

    tg0, tg1, tg2, tg3 = D_ffn * g0_s, D_ffn * g1_s, D_ffn * g2_s, D_ffn * g3_s
    dD_ffn = g0_s * m0 + g1_s * m1 + g2_s * m2 + g3_s * m3
    dE_tot_ffn = -kappa_ffn_eff * (D_ffn * D_ffn) * dD_ffn
    grad_kappa_ffn = sig_kappa_ffn * np.sum(-(D_ffn * D_ffn) * E_tot * dD_ffn, axis=(0, 1))
    grad_w_up = np.sum(tg0 * v1 + tg1 * v2 + tg2 * u1 + tg3 * u2, axis=(0, 1))
    glu_term = tg0 * v2 - tg1 * v1 - tg2 * u2 + tg3 * u1
    grad_w_gate = np.sum(G_glu * glu_term, axis=(0, 1))
    dG_glu = w_gate_arr * glu_term

    du1_swiglu = w_up_arr * tg2 + w_gate_arr * G_glu * tg3 + dG_glu * v2 + 2.0 * u1 * dE_tot_ffn
    du2_swiglu = w_up_arr * tg3 - w_gate_arr * G_glu * tg2 - dG_glu * v1 + 2.0 * u2 * dE_tot_ffn
    dv1_swiglu = w_up_arr * tg0 - w_gate_arr * G_glu * tg1 - dG_glu * u2 + 2.0 * v1 * dE_tot_ffn
    dv2_swiglu = w_up_arr * tg1 + w_gate_arr * G_glu * tg0 + dG_glu * u1 + 2.0 * v2 * dE_tot_ffn

    # ------------------------------------------------------------------
    # 4. 子算子 IV (O_head) 反向传播
    # ------------------------------------------------------------------
    g0_a, g1_a, g2_a, g3_a = m4_w * g0_raw, m4_w * g1_raw, m4_w * g2_raw, m4_w * g3_raw
    Delta_E = E_V - E_Q
    Q_cap = 1.0 + kappa_head_eff * (Delta_E * Delta_E)
    D_cap = 1.0 / np.sqrt(Q_cap)
    G_cap = Delta_E * D_cap
    I_qk = u1 * v1 + u2 * v2
    Q_head = 1.0 + kappa_head_eff * E_tot
    D_head = 1.0 / Q_head
    C_qk = D_head * I_qk

    A0 = -u1 + G_cap * v2
    A1 = u2 - G_cap * v1
    A2 = v1 + G_cap * u2
    A3 = -v2 - G_cap * u1

    S_A = g0_a * A0 + g1_a * A1 + g2_a * A2 + g3_a * A3
    grad_alpha_res = d_alpha_res * np.sum(S_A, axis=(0, 1))

    S_B = g0_a * v1 - g1_a * v2 + g2_a * u1 - g3_a * u2
    grad_beta_qk = np.sum(C_qk * S_B, axis=(0, 1))

    dC_qk = beta_qk_arr * S_B
    dI_qk = D_head * dC_qk
    dD_head = I_qk * dC_qk
    dE_tot_head = -kappa_head_eff * (D_head * D_head) * dD_head

    dG_cap = alpha_res_eff * (g0_a * v2 - g1_a * v1 + g2_a * u2 - g3_a * u1)
    dDelta_E = dG_cap * (D_cap * D_cap * D_cap)

    grad_kappa_head = sig_kappa_head * np.sum(
        -0.5 * (Delta_E * Delta_E * Delta_E) * (D_cap * D_cap * D_cap) * dG_cap
        - (D_head * D_head) * E_tot * dD_head,
        axis=(0, 1),
    )

    du1_head = (
        alpha_res_eff * (-g0_a - G_cap * g3_a)
        + beta_qk_arr * C_qk * g2_a
        + dI_qk * v1
        - 2.0 * u1 * dDelta_E
        + 2.0 * u1 * dE_tot_head
    )
    du2_head = (
        alpha_res_eff * (g1_a + G_cap * g2_a)
        - beta_qk_arr * C_qk * g3_a
        + dI_qk * v2
        - 2.0 * u2 * dDelta_E
        + 2.0 * u2 * dE_tot_head
    )
    dv1_head = (
        alpha_res_eff * (-G_cap * g1_a + g2_a)
        + beta_qk_arr * C_qk * g0_a
        + dI_qk * u1
        + 2.0 * v1 * dDelta_E
        + 2.0 * v1 * dE_tot_head
    )
    dv2_head = (
        alpha_res_eff * (G_cap * g0_a - g3_a)
        - beta_qk_arr * C_qk * g1_a
        + dI_qk * u2
        + 2.0 * v2 * dDelta_E
        + 2.0 * v2 * dE_tot_head
    )

    grad_x = np.stack(
        [
            du1_rope + du1_swiglu + du1_head,
            du2_rope + du2_swiglu + du2_head,
            dv1_qkv + dv1_swiglu + dv1_head,
            dv2_qkv + dv2_swiglu + dv2_head,
        ],
        axis=-1,
    ).reshape(B, L, D)

    base_grads = (
        grad_x,
        grad_theta_rope,
        grad_gamma_rms,
        grad_kappa_rms,
        grad_theta_qkv,
        grad_w_qkv,
        grad_kappa_qkv,
        grad_w_up,
        grad_w_gate,
        grad_kappa_ffn,
        grad_alpha_res,
        grad_beta_qk,
        grad_kappa_head,
    )

    if return_sub_grads:
        zeros = np.zeros_like(u1)
        grad_x_rope = np.stack([du1_rope, du2_rope, zeros, zeros], axis=-1).reshape(B, L, D)
        grad_x_qkv = np.stack([zeros, zeros, dv1_qkv, dv2_qkv], axis=-1).reshape(B, L, D)
        grad_x_swiglu = np.stack([du1_swiglu, du2_swiglu, dv1_swiglu, dv2_swiglu], axis=-1).reshape(B, L, D)
        grad_x_head = np.stack([du1_head, du2_head, dv1_head, dv2_head], axis=-1).reshape(B, L, D)
        return base_grads + ((grad_x_rope, grad_x_qkv, grad_x_swiglu, grad_x_head),)

    return base_grads


# ==============================================================================
# 4. Triton GPU Kernel (@triton.jit) 算子融合实现与 Tile 级精确核调度器
# ==============================================================================
if HAS_TRITON:

    @triton.jit
    def fused_llm_hierarchy_fwd_kernel(
        X_ptr,
        Y_ptr,
        ThetaRope_ptr,
        GammaRms_ptr,
        KappaRms_ptr,
        ThetaQkv_ptr,
        WQkv_ptr,
        KappaQkv_ptr,
        WUp_ptr,
        WGate_ptr,
        KappaFfn_ptr,
        AlphaRes_ptr,
        BetaQk_ptr,
        KappaHead_ptr,
        N_groups_total,
        L,
        G,
        eps_zpe,
        m1_w,
        m2_w,
        m3_w,
        m4_w,
        BLOCK_M: tl.constexpr,
    ):
        """单次启动融合完整 LLM 四层算子机制 (RMSNorm+RoPE / QKV / SwiGLU / LogitHead) 的 Triton 前向核。"""
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offs_m < N_groups_total

        g_idx = offs_m % G
        l_idx = (offs_m // G) % L
        pos_l = (l_idx.to(tl.float64) + 1.0) / tl.full((BLOCK_M,), L, dtype=tl.float64)

        base_x = offs_m * 4
        u1 = tl.load(X_ptr + base_x + 0, mask=mask, other=0.0).to(tl.float64)
        u2 = tl.load(X_ptr + base_x + 1, mask=mask, other=0.0).to(tl.float64)
        v1 = tl.load(X_ptr + base_x + 2, mask=mask, other=0.0).to(tl.float64)
        v2 = tl.load(X_ptr + base_x + 3, mask=mask, other=0.0).to(tl.float64)

        theta_rope = tl.load(ThetaRope_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        gamma_rms = tl.load(GammaRms_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        k_rms_raw = tl.load(KappaRms_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        k_rms_eff = tl.log(1.0 + tl.exp(-tl.abs(k_rms_raw))) + tl.maximum(k_rms_raw, 0.0) + eps_zpe

        theta_qkv = tl.load(ThetaQkv_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        w_qkv = tl.load(WQkv_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        k_qkv_raw = tl.load(KappaQkv_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        k_qkv_eff = tl.log(1.0 + tl.exp(-tl.abs(k_qkv_raw))) + tl.maximum(k_qkv_raw, 0.0) + eps_zpe

        w_up = tl.load(WUp_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        w_gate = tl.load(WGate_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        k_ffn_raw = tl.load(KappaFfn_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        k_ffn_eff = tl.log(1.0 + tl.exp(-tl.abs(k_ffn_raw))) + tl.maximum(k_ffn_raw, 0.0) + eps_zpe

        alpha_raw = tl.load(AlphaRes_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        alpha_eff = 1.0 / (1.0 + tl.exp(-alpha_raw))
        beta_qk = tl.load(BetaQk_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        k_head_raw = tl.load(KappaHead_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        k_head_eff = tl.log(1.0 + tl.exp(-tl.abs(k_head_raw))) + tl.maximum(k_head_raw, 0.0) + eps_zpe

        # 1. RoPE + Pre-RMSNorm
        E_Q = u1 * u1 + u2 * u2
        D_rms = 1.0 / tl.sqrt(1.0 + k_rms_eff * E_Q)
        phi = pos_l * theta_rope
        cos_r = tl.cos(phi)
        sin_r = tl.sin(phi)
        op_rope0 = gamma_rms * D_rms * (cos_r * u1 - sin_r * u2)
        op_rope1 = gamma_rms * D_rms * (sin_r * u1 + cos_r * u2)

        # 2. Multi-Head QKV Value Projection
        E_V = v1 * v1 + v2 * v2
        D_qkv = 1.0 / tl.sqrt(1.0 + k_qkv_eff * E_V)
        cos_v = tl.cos(theta_qkv)
        sin_v = tl.sin(theta_qkv)
        op_qkv2 = w_qkv * D_qkv * (cos_v * v1 + sin_v * v2)
        op_qkv3 = w_qkv * D_qkv * (-sin_v * v1 + cos_v * v2)

        # 3. SwiGLU Bilinear FFN
        E_tot = E_Q + E_V
        D_ffn = 1.0 / (1.0 + k_ffn_eff * E_tot)
        G_glu = u1 * v2 - u2 * v1
        op_s0 = D_ffn * (w_up * v1 + w_gate * G_glu * v2)
        op_s1 = D_ffn * (w_up * v2 - w_gate * G_glu * v1)
        op_s2 = D_ffn * (w_up * u1 - w_gate * G_glu * u2)
        op_s3 = D_ffn * (w_up * u2 + w_gate * G_glu * u1)

        # 4. QK Inner-Product & Logit Softcap Head
        Delta_E = E_V - E_Q
        G_cap = Delta_E / tl.sqrt(1.0 + k_head_eff * Delta_E * Delta_E)
        I_qk = u1 * v1 + u2 * v2
        C_qk = I_qk / (1.0 + k_head_eff * E_tot)
        op_h0 = alpha_eff * (-u1 + G_cap * v2) + beta_qk * C_qk * v1
        op_h1 = alpha_eff * (u2 - G_cap * v1) - beta_qk * C_qk * v2
        op_h2 = alpha_eff * (v1 + G_cap * u2) + beta_qk * C_qk * u1
        op_h3 = alpha_eff * (-v2 - G_cap * u1) - beta_qk * C_qk * u2

        y0 = m1_w * op_rope0 + m3_w * op_s0 + m4_w * op_h0
        y1 = m1_w * op_rope1 + m3_w * op_s1 + m4_w * op_h1
        y2 = m2_w * op_qkv2 + m3_w * op_s2 + m4_w * op_h2
        y3 = m2_w * op_qkv3 + m3_w * op_s3 + m4_w * op_h3

        tl.store(Y_ptr + base_x + 0, y0, mask=mask)
        tl.store(Y_ptr + base_x + 1, y1, mask=mask)
        tl.store(Y_ptr + base_x + 2, y2, mask=mask)
        tl.store(Y_ptr + base_x + 3, y3, mask=mask)

    @triton.jit
    def fused_llm_hierarchy_bwd_kernel(
        GradOut_ptr,
        X_ptr,
        GradX_ptr,
        ThetaRope_ptr,
        GammaRms_ptr,
        KappaRms_ptr,
        ThetaQkv_ptr,
        WQkv_ptr,
        KappaQkv_ptr,
        WUp_ptr,
        WGate_ptr,
        KappaFfn_ptr,
        AlphaRes_ptr,
        BetaQk_ptr,
        KappaHead_ptr,
        GradThetaRope_ptr,
        GradGammaRms_ptr,
        GradKappaRms_ptr,
        GradThetaQkv_ptr,
        GradWQkv_ptr,
        GradKappaQkv_ptr,
        GradWUp_ptr,
        GradWGate_ptr,
        GradKappaFfn_ptr,
        GradAlphaRes_ptr,
        GradBetaQk_ptr,
        GradKappaHead_ptr,
        N_groups_total,
        L,
        G,
        eps_zpe,
        m1_w,
        m2_w,
        m3_w,
        m4_w,
        BLOCK_M: tl.constexpr,
    ):
        """输出全部 12 组 LLM 算子可学习参数梯度 (tl.atomic_add) 与输入梯度的 Triton 融合反向核。"""
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offs_m < N_groups_total

        g_idx = offs_m % G
        l_idx = (offs_m // G) % L
        pos_l = (l_idx.to(tl.float64) + 1.0) / tl.full((BLOCK_M,), L, dtype=tl.float64)

        base_x = offs_m * 4
        u1 = tl.load(X_ptr + base_x + 0, mask=mask, other=0.0).to(tl.float64)
        u2 = tl.load(X_ptr + base_x + 1, mask=mask, other=0.0).to(tl.float64)
        v1 = tl.load(X_ptr + base_x + 2, mask=mask, other=0.0).to(tl.float64)
        v2 = tl.load(X_ptr + base_x + 3, mask=mask, other=0.0).to(tl.float64)

        g0_raw = tl.load(GradOut_ptr + base_x + 0, mask=mask, other=0.0).to(tl.float64)
        g1_raw = tl.load(GradOut_ptr + base_x + 1, mask=mask, other=0.0).to(tl.float64)
        g2_raw = tl.load(GradOut_ptr + base_x + 2, mask=mask, other=0.0).to(tl.float64)
        g3_raw = tl.load(GradOut_ptr + base_x + 3, mask=mask, other=0.0).to(tl.float64)

        theta_rope = tl.load(ThetaRope_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        gamma_rms = tl.load(GammaRms_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        k_rms_raw = tl.load(KappaRms_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        sig_k_rms = 1.0 / (1.0 + tl.exp(-k_rms_raw))
        k_rms_eff = tl.log(1.0 + tl.exp(-tl.abs(k_rms_raw))) + tl.maximum(k_rms_raw, 0.0) + eps_zpe

        theta_qkv = tl.load(ThetaQkv_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        w_qkv = tl.load(WQkv_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        k_qkv_raw = tl.load(KappaQkv_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        sig_k_qkv = 1.0 / (1.0 + tl.exp(-k_qkv_raw))
        k_qkv_eff = tl.log(1.0 + tl.exp(-tl.abs(k_qkv_raw))) + tl.maximum(k_qkv_raw, 0.0) + eps_zpe

        w_up = tl.load(WUp_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        w_gate = tl.load(WGate_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        k_ffn_raw = tl.load(KappaFfn_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        sig_k_ffn = 1.0 / (1.0 + tl.exp(-k_ffn_raw))
        k_ffn_eff = tl.log(1.0 + tl.exp(-tl.abs(k_ffn_raw))) + tl.maximum(k_ffn_raw, 0.0) + eps_zpe

        alpha_raw = tl.load(AlphaRes_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        alpha_eff = 1.0 / (1.0 + tl.exp(-alpha_raw))
        d_alpha = alpha_eff * (1.0 - alpha_eff)
        beta_qk = tl.load(BetaQk_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        k_head_raw = tl.load(KappaHead_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        sig_k_head = 1.0 / (1.0 + tl.exp(-k_head_raw))
        k_head_eff = tl.log(1.0 + tl.exp(-tl.abs(k_head_raw))) + tl.maximum(k_head_raw, 0.0) + eps_zpe

        # 1. RoPE backward
        g0_L = m1_w * g0_raw
        g1_L = m1_w * g1_raw
        E_Q = u1 * u1 + u2 * u2
        D_rms = 1.0 / tl.sqrt(1.0 + k_rms_eff * E_Q)
        phi = pos_l * theta_rope
        cos_r = tl.cos(phi)
        sin_r = tl.sin(phi)
        r_Q1 = cos_r * u1 - sin_r * u2
        r_Q2 = sin_r * u1 + cos_r * u2
        S_rQ = g0_L * r_Q1 + g1_L * r_Q2
        S_phi = -g0_L * r_Q2 + g1_L * r_Q1
        dQ_rms = -0.5 * gamma_rms * (D_rms * D_rms * D_rms) * S_rQ

        du1_rope = gamma_rms * D_rms * (g0_L * cos_r + g1_L * sin_r) + 2.0 * k_rms_eff * u1 * dQ_rms
        du2_rope = gamma_rms * D_rms * (-g0_L * sin_r + g1_L * cos_r) + 2.0 * k_rms_eff * u2 * dQ_rms

        tl.atomic_add(GradGammaRms_ptr + g_idx, D_rms * S_rQ, mask=mask)
        tl.atomic_add(GradThetaRope_ptr + g_idx, gamma_rms * D_rms * S_phi * pos_l, mask=mask)
        tl.atomic_add(GradKappaRms_ptr + g_idx, sig_k_rms * dQ_rms * E_Q, mask=mask)

        # 2. QKV backward
        g2_R = m2_w * g2_raw
        g3_R = m2_w * g3_raw
        E_V = v1 * v1 + v2 * v2
        D_qkv = 1.0 / tl.sqrt(1.0 + k_qkv_eff * E_V)
        cos_v = tl.cos(theta_qkv)
        sin_v = tl.sin(theta_qkv)
        r_V1 = cos_v * v1 + sin_v * v2
        r_V2 = -sin_v * v1 + cos_v * v2
        S_rV = g2_R * r_V1 + g3_R * r_V2
        S_tV = g2_R * r_V2 - g3_R * r_V1
        dQ_qkv = -0.5 * w_qkv * (D_qkv * D_qkv * D_qkv) * S_rV

        dv1_qkv = w_qkv * D_qkv * (g2_R * cos_v - g3_R * sin_v) + 2.0 * k_qkv_eff * v1 * dQ_qkv
        dv2_qkv = w_qkv * D_qkv * (g2_R * sin_v + g3_R * cos_v) + 2.0 * k_qkv_eff * v2 * dQ_qkv

        tl.atomic_add(GradWQkv_ptr + g_idx, D_qkv * S_rV, mask=mask)
        tl.atomic_add(GradThetaQkv_ptr + g_idx, w_qkv * D_qkv * S_tV, mask=mask)
        tl.atomic_add(GradKappaQkv_ptr + g_idx, sig_k_qkv * dQ_qkv * E_V, mask=mask)

        # 3. SwiGLU backward
        g0_s = m3_w * g0_raw
        g1_s = m3_w * g1_raw
        g2_s = m3_w * g2_raw
        g3_s = m3_w * g3_raw
        E_tot = E_Q + E_V
        D_ffn = 1.0 / (1.0 + k_ffn_eff * E_tot)
        G_glu = u1 * v2 - u2 * v1
        m0 = w_up * v1 + w_gate * G_glu * v2
        m1 = w_up * v2 - w_gate * G_glu * v1
        m2 = w_up * u1 - w_gate * G_glu * u2
        m3 = w_up * u2 + w_gate * G_glu * u1

        tg0 = D_ffn * g0_s
        tg1 = D_ffn * g1_s
        tg2 = D_ffn * g2_s
        tg3 = D_ffn * g3_s
        dD_ffn = g0_s * m0 + g1_s * m1 + g2_s * m2 + g3_s * m3
        dE_tot_ffn = -k_ffn_eff * (D_ffn * D_ffn) * dD_ffn
        glu_term = tg0 * v2 - tg1 * v1 - tg2 * u2 + tg3 * u1
        dG_glu = w_gate * glu_term

        du1_swiglu = w_up * tg2 + w_gate * G_glu * tg3 + dG_glu * v2 + 2.0 * u1 * dE_tot_ffn
        du2_swiglu = w_up * tg3 - w_gate * G_glu * tg2 - dG_glu * v1 + 2.0 * u2 * dE_tot_ffn
        dv1_swiglu = w_up * tg0 - w_gate * G_glu * tg1 - dG_glu * u2 + 2.0 * v1 * dE_tot_ffn
        dv2_swiglu = w_up * tg1 + w_gate * G_glu * tg0 + dG_glu * u1 + 2.0 * v2 * dE_tot_ffn

        tl.atomic_add(GradWUp_ptr + g_idx, tg0 * v1 + tg1 * v2 + tg2 * u1 + tg3 * u2, mask=mask)
        tl.atomic_add(GradWGate_ptr + g_idx, G_glu * glu_term, mask=mask)
        tl.atomic_add(GradKappaFfn_ptr + g_idx, -sig_k_ffn * (D_ffn * D_ffn) * E_tot * dD_ffn, mask=mask)

        # 4. Logit Head backward
        g0_a = m4_w * g0_raw
        g1_a = m4_w * g1_raw
        g2_a = m4_w * g2_raw
        g3_a = m4_w * g3_raw
        Delta_E = E_V - E_Q
        D_cap = 1.0 / tl.sqrt(1.0 + k_head_eff * Delta_E * Delta_E)
        G_cap = Delta_E * D_cap
        I_qk = u1 * v1 + u2 * v2
        D_head = 1.0 / (1.0 + k_head_eff * E_tot)
        C_qk = D_head * I_qk

        A0 = -u1 + G_cap * v2
        A1 = u2 - G_cap * v1
        A2 = v1 + G_cap * u2
        A3 = -v2 - G_cap * u1
        S_A = g0_a * A0 + g1_a * A1 + g2_a * A2 + g3_a * A3
        S_B = g0_a * v1 - g1_a * v2 + g2_a * u1 - g3_a * u2

        dC_qk = beta_qk * S_B
        dI_qk = D_head * dC_qk
        dD_head = I_qk * dC_qk
        dE_tot_head = -k_head_eff * (D_head * D_head) * dD_head
        dG_cap = alpha_eff * (g0_a * v2 - g1_a * v1 + g2_a * u2 - g3_a * u1)
        dDelta_E = dG_cap * (D_cap * D_cap * D_cap)

        du1_head = (
            alpha_eff * (-g0_a - G_cap * g3_a)
            + beta_qk * C_qk * g2_a
            + dI_qk * v1
            - 2.0 * u1 * dDelta_E
            + 2.0 * u1 * dE_tot_head
        )
        du2_head = (
            alpha_eff * (g1_a + G_cap * g2_a)
            - beta_qk * C_qk * g3_a
            + dI_qk * v2
            - 2.0 * u2 * dDelta_E
            + 2.0 * u2 * dE_tot_head
        )
        dv1_head = (
            alpha_eff * (-G_cap * g1_a + g2_a)
            + beta_qk * C_qk * g0_a
            + dI_qk * u1
            + 2.0 * v1 * dDelta_E
            + 2.0 * v1 * dE_tot_head
        )
        dv2_head = (
            alpha_eff * (G_cap * g0_a - g3_a)
            - beta_qk * C_qk * g1_a
            + dI_qk * u2
            + 2.0 * v2 * dDelta_E
            + 2.0 * v2 * dE_tot_head
        )

        tl.atomic_add(GradAlphaRes_ptr + g_idx, d_alpha * S_A, mask=mask)
        tl.atomic_add(GradBetaQk_ptr + g_idx, C_qk * S_B, mask=mask)
        tl.atomic_add(
            GradKappaHead_ptr + g_idx,
            sig_k_head
            * (-0.5 * (Delta_E * Delta_E * Delta_E) * (D_cap * D_cap * D_cap) * dG_cap - (D_head * D_head) * E_tot * dD_head),
            mask=mask,
        )

        tl.store(GradX_ptr + base_x + 0, du1_rope + du1_swiglu + du1_head, mask=mask)
        tl.store(GradX_ptr + base_x + 1, du2_rope + du2_swiglu + du2_head, mask=mask)
        tl.store(GradX_ptr + base_x + 2, dv1_qkv + dv1_swiglu + dv1_head, mask=mask)
        tl.store(GradX_ptr + base_x + 3, dv2_qkv + dv2_swiglu + dv2_head, mask=mask)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def llm_hierarchy_triton_forward(
    x: np.ndarray,
    theta_rope: np.ndarray,
    gamma_rms: np.ndarray,
    kappa_rms: np.ndarray,
    theta_qkv: np.ndarray,
    w_qkv: np.ndarray,
    kappa_qkv: np.ndarray,
    w_up: np.ndarray,
    w_gate: np.ndarray,
    kappa_ffn: np.ndarray,
    alpha_res: np.ndarray,
    beta_qk: np.ndarray,
    kappa_head: np.ndarray,
    eps_zpe: float = 1e-4,
    sub_op_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    block_size: int = 64,
) -> np.ndarray:
    """调度 @triton.jit 前向融合核（若 GPU 可用）或执行分块 Program-ID Tile 级精确核。"""
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    B, L, D = x_arr.shape
    G = D // 4
    N_groups_total = B * L * G
    m1_w, m2_w, m3_w, m4_w = sub_op_weights

    if HAS_TORCH and HAS_TRITON and torch.cuda.is_available():
        x_t = torch.as_tensor(x_arr, device="cuda", dtype=torch.float64).contiguous()
        y_t = torch.empty_like(x_t)
        p_tensors = [
            torch.as_tensor(np.ascontiguousarray(p, dtype=np.float64), device="cuda", dtype=torch.float64)
            for p in (
                theta_rope,
                gamma_rms,
                kappa_rms,
                theta_qkv,
                w_qkv,
                kappa_qkv,
                w_up,
                w_gate,
                kappa_ffn,
                alpha_res,
                beta_qk,
                kappa_head,
            )
        ]
        grid = (triton.cdiv(N_groups_total, block_size),)
        fused_llm_hierarchy_fwd_kernel[grid](
            x_t,
            y_t,
            *p_tensors,
            N_groups_total,
            L,
            G,
            eps_zpe,
            m1_w,
            m2_w,
            m3_w,
            m4_w,
            BLOCK_M=block_size,
        )
        return y_t.cpu().numpy()

    x_flat = x_arr.reshape(N_groups_total, 4)
    y_flat = np.empty_like(x_flat)
    num_programs = _cdiv(N_groups_total, block_size)

    for pid in range(num_programs):
        offs_m = pid * block_size + np.arange(block_size)
        mask = offs_m < N_groups_total
        valid_offs = offs_m[mask]
        g_idx = valid_offs % G
        l_idx = (valid_offs // G) % L
        pos_l = (l_idx.astype(np.float64) + 1.0) / float(L)

        u1 = x_flat[valid_offs, 0]
        u2 = x_flat[valid_offs, 1]
        v1 = x_flat[valid_offs, 2]
        v2 = x_flat[valid_offs, 3]

        t_rope = theta_rope[g_idx]
        g_rms = gamma_rms[g_idx]
        k_rms_eff = _np_softplus(kappa_rms[g_idx]) + eps_zpe

        t_qkv = theta_qkv[g_idx]
        w_v = w_qkv[g_idx]
        k_qkv_eff = _np_softplus(kappa_qkv[g_idx]) + eps_zpe

        w_u = w_up[g_idx]
        w_g = w_gate[g_idx]
        k_ffn_eff = _np_softplus(kappa_ffn[g_idx]) + eps_zpe

        al_eff = _np_sigmoid(alpha_res[g_idx])
        bt_qk = beta_qk[g_idx]
        k_head_eff = _np_softplus(kappa_head[g_idx]) + eps_zpe

        E_Q = u1 * u1 + u2 * u2
        D_rms = 1.0 / np.sqrt(1.0 + k_rms_eff * E_Q)
        phi = pos_l * t_rope
        cos_r, sin_r = np.cos(phi), np.sin(phi)
        op_rope0 = g_rms * D_rms * (cos_r * u1 - sin_r * u2)
        op_rope1 = g_rms * D_rms * (sin_r * u1 + cos_r * u2)

        E_V = v1 * v1 + v2 * v2
        D_qkv = 1.0 / np.sqrt(1.0 + k_qkv_eff * E_V)
        cos_v, sin_v = np.cos(t_qkv), np.sin(t_qkv)
        op_qkv2 = w_v * D_qkv * (cos_v * v1 + sin_v * v2)
        op_qkv3 = w_v * D_qkv * (-sin_v * v1 + cos_v * v2)

        E_tot = E_Q + E_V
        D_ffn = 1.0 / (1.0 + k_ffn_eff * E_tot)
        G_glu = u1 * v2 - u2 * v1
        op_s0 = D_ffn * (w_u * v1 + w_g * G_glu * v2)
        op_s1 = D_ffn * (w_u * v2 - w_g * G_glu * v1)
        op_s2 = D_ffn * (w_u * u1 - w_g * G_glu * u2)
        op_s3 = D_ffn * (w_u * u2 + w_g * G_glu * u1)

        Delta_E = E_V - E_Q
        G_cap = Delta_E / np.sqrt(1.0 + k_head_eff * Delta_E * Delta_E)
        I_qk = u1 * v1 + u2 * v2
        C_qk = I_qk / (1.0 + k_head_eff * E_tot)
        op_h0 = al_eff * (-u1 + G_cap * v2) + bt_qk * C_qk * v1
        op_h1 = al_eff * (u2 - G_cap * v1) - bt_qk * C_qk * v2
        op_h2 = al_eff * (v1 + G_cap * u2) + bt_qk * C_qk * u1
        op_h3 = al_eff * (-v2 - G_cap * u1) - bt_qk * C_qk * u2

        y_flat[valid_offs, 0] = m1_w * op_rope0 + m3_w * op_s0 + m4_w * op_h0
        y_flat[valid_offs, 1] = m1_w * op_rope1 + m3_w * op_s1 + m4_w * op_h1
        y_flat[valid_offs, 2] = m2_w * op_qkv2 + m3_w * op_s2 + m4_w * op_h2
        y_flat[valid_offs, 3] = m2_w * op_qkv3 + m3_w * op_s3 + m4_w * op_h3

    return y_flat.reshape(B, L, D)


def llm_hierarchy_triton_backward(
    grad_out: np.ndarray,
    x: np.ndarray,
    theta_rope: np.ndarray,
    gamma_rms: np.ndarray,
    kappa_rms: np.ndarray,
    theta_qkv: np.ndarray,
    w_qkv: np.ndarray,
    kappa_qkv: np.ndarray,
    w_up: np.ndarray,
    w_gate: np.ndarray,
    kappa_ffn: np.ndarray,
    alpha_res: np.ndarray,
    beta_qk: np.ndarray,
    kappa_head: np.ndarray,
    eps_zpe: float = 1e-4,
    sub_op_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    block_size: int = 64,
) -> Tuple[np.ndarray, ...]:
    """调度 @triton.jit 全参数梯度反向融合核（含 tl.atomic_add 归约）或 Tile 级精确核。"""
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    g_arr = np.ascontiguousarray(grad_out, dtype=np.float64)
    B, L, D = x_arr.shape
    G = D // 4
    N_groups_total = B * L * G
    m1_w, m2_w, m3_w, m4_w = sub_op_weights

    if HAS_TORCH and HAS_TRITON and torch.cuda.is_available():
        x_t = torch.as_tensor(x_arr, device="cuda", dtype=torch.float64).contiguous()
        go_t = torch.as_tensor(g_arr, device="cuda", dtype=torch.float64).contiguous()
        gx_t = torch.empty_like(x_t)
        p_tensors = [
            torch.as_tensor(np.ascontiguousarray(p, dtype=np.float64), device="cuda", dtype=torch.float64)
            for p in (
                theta_rope,
                gamma_rms,
                kappa_rms,
                theta_qkv,
                w_qkv,
                kappa_qkv,
                w_up,
                w_gate,
                kappa_ffn,
                alpha_res,
                beta_qk,
                kappa_head,
            )
        ]
        gp_tensors = [torch.zeros(G, device="cuda", dtype=torch.float64) for _ in range(12)]
        grid = (triton.cdiv(N_groups_total, block_size),)
        fused_llm_hierarchy_bwd_kernel[grid](
            go_t,
            x_t,
            gx_t,
            *p_tensors,
            *gp_tensors,
            N_groups_total,
            L,
            G,
            eps_zpe,
            m1_w,
            m2_w,
            m3_w,
            m4_w,
            BLOCK_M=block_size,
        )
        return (gx_t.cpu().numpy(),) + tuple(gp.cpu().numpy() for gp in gp_tensors)

    x_flat = x_arr.reshape(N_groups_total, 4)
    g_flat = g_arr.reshape(N_groups_total, 4)
    gx_flat = np.empty_like(x_flat)

    grad_theta_rope = np.zeros(G, dtype=np.float64)
    grad_gamma_rms = np.zeros(G, dtype=np.float64)
    grad_kappa_rms = np.zeros(G, dtype=np.float64)
    grad_theta_qkv = np.zeros(G, dtype=np.float64)
    grad_w_qkv = np.zeros(G, dtype=np.float64)
    grad_kappa_qkv = np.zeros(G, dtype=np.float64)
    grad_w_up = np.zeros(G, dtype=np.float64)
    grad_w_gate = np.zeros(G, dtype=np.float64)
    grad_kappa_ffn = np.zeros(G, dtype=np.float64)
    grad_alpha_res = np.zeros(G, dtype=np.float64)
    grad_beta_qk = np.zeros(G, dtype=np.float64)
    grad_kappa_head = np.zeros(G, dtype=np.float64)

    num_programs = _cdiv(N_groups_total, block_size)
    for pid in range(num_programs):
        offs_m = pid * block_size + np.arange(block_size)
        mask = offs_m < N_groups_total
        valid_offs = offs_m[mask]
        g_idx = valid_offs % G
        l_idx = (valid_offs // G) % L
        pos_l = (l_idx.astype(np.float64) + 1.0) / float(L)

        u1, u2, v1, v2 = (
            x_flat[valid_offs, 0],
            x_flat[valid_offs, 1],
            x_flat[valid_offs, 2],
            x_flat[valid_offs, 3],
        )
        g0_raw, g1_raw, g2_raw, g3_raw = (
            g_flat[valid_offs, 0],
            g_flat[valid_offs, 1],
            g_flat[valid_offs, 2],
            g_flat[valid_offs, 3],
        )

        t_rope, g_rms, k_rms_raw = theta_rope[g_idx], gamma_rms[g_idx], kappa_rms[g_idx]
        sig_k_rms = _np_sigmoid(k_rms_raw)
        k_rms_eff = _np_softplus(k_rms_raw) + eps_zpe

        t_qkv, w_v, k_qkv_raw = theta_qkv[g_idx], w_qkv[g_idx], kappa_qkv[g_idx]
        sig_k_qkv = _np_sigmoid(k_qkv_raw)
        k_qkv_eff = _np_softplus(k_qkv_raw) + eps_zpe

        w_u, w_g, k_ffn_raw = w_up[g_idx], w_gate[g_idx], kappa_ffn[g_idx]
        sig_k_ffn = _np_sigmoid(k_ffn_raw)
        k_ffn_eff = _np_softplus(k_ffn_raw) + eps_zpe

        al_raw, bt_qk, k_head_raw = alpha_res[g_idx], beta_qk[g_idx], kappa_head[g_idx]
        al_eff = _np_sigmoid(al_raw)
        d_al = al_eff * (1.0 - al_eff)
        sig_k_head = _np_sigmoid(k_head_raw)
        k_head_eff = _np_softplus(k_head_raw) + eps_zpe

        # 1. RoPE
        g0_L, g1_L = m1_w * g0_raw, m1_w * g1_raw
        E_Q = u1 * u1 + u2 * u2
        D_rms = 1.0 / np.sqrt(1.0 + k_rms_eff * E_Q)
        phi = pos_l * t_rope
        cos_r, sin_r = np.cos(phi), np.sin(phi)
        r_Q1 = cos_r * u1 - sin_r * u2
        r_Q2 = sin_r * u1 + cos_r * u2
        S_rQ = g0_L * r_Q1 + g1_L * r_Q2
        S_phi = -g0_L * r_Q2 + g1_L * r_Q1
        dQ_rms = -0.5 * g_rms * (D_rms * D_rms * D_rms) * S_rQ
        du1_rope = g_rms * D_rms * (g0_L * cos_r + g1_L * sin_r) + 2.0 * k_rms_eff * u1 * dQ_rms
        du2_rope = g_rms * D_rms * (-g0_L * sin_r + g1_L * cos_r) + 2.0 * k_rms_eff * u2 * dQ_rms

        np.add.at(grad_gamma_rms, g_idx, D_rms * S_rQ)
        np.add.at(grad_theta_rope, g_idx, g_rms * D_rms * S_phi * pos_l)
        np.add.at(grad_kappa_rms, g_idx, sig_k_rms * dQ_rms * E_Q)

        # 2. QKV
        g2_R, g3_R = m2_w * g2_raw, m2_w * g3_raw
        E_V = v1 * v1 + v2 * v2
        D_qkv = 1.0 / np.sqrt(1.0 + k_qkv_eff * E_V)
        cos_v, sin_v = np.cos(t_qkv), np.sin(t_qkv)
        r_V1 = cos_v * v1 + sin_v * v2
        r_V2 = -sin_v * v1 + cos_v * v2
        S_rV = g2_R * r_V1 + g3_R * r_V2
        S_tV = g2_R * r_V2 - g3_R * r_V1
        dQ_qkv = -0.5 * w_v * (D_qkv * D_qkv * D_qkv) * S_rV
        dv1_qkv = w_v * D_qkv * (g2_R * cos_v - g3_R * sin_v) + 2.0 * k_qkv_eff * v1 * dQ_qkv
        dv2_qkv = w_v * D_qkv * (g2_R * sin_v + g3_R * cos_v) + 2.0 * k_qkv_eff * v2 * dQ_qkv

        np.add.at(grad_w_qkv, g_idx, D_qkv * S_rV)
        np.add.at(grad_theta_qkv, g_idx, w_v * D_qkv * S_tV)
        np.add.at(grad_kappa_qkv, g_idx, sig_k_qkv * dQ_qkv * E_V)

        # 3. SwiGLU
        g0_s, g1_s, g2_s, g3_s = m3_w * g0_raw, m3_w * g1_raw, m3_w * g2_raw, m3_w * g3_raw
        E_tot = E_Q + E_V
        D_ffn = 1.0 / (1.0 + k_ffn_eff * E_tot)
        G_glu = u1 * v2 - u2 * v1
        m0 = w_u * v1 + w_g * G_glu * v2
        m1 = w_u * v2 - w_g * G_glu * v1
        m2 = w_u * u1 - w_g * G_glu * u2
        m3 = w_u * u2 + w_g * G_glu * u1
        tg0, tg1, tg2, tg3 = D_ffn * g0_s, D_ffn * g1_s, D_ffn * g2_s, D_ffn * g3_s
        dD_ffn = g0_s * m0 + g1_s * m1 + g2_s * m2 + g3_s * m3
        dE_tot_ffn = -k_ffn_eff * (D_ffn * D_ffn) * dD_ffn
        glu_term = tg0 * v2 - tg1 * v1 - tg2 * u2 + tg3 * u1
        dG_glu = w_g * glu_term

        du1_swiglu = w_u * tg2 + w_g * G_glu * tg3 + dG_glu * v2 + 2.0 * u1 * dE_tot_ffn
        du2_swiglu = w_u * tg3 - w_g * G_glu * tg2 - dG_glu * v1 + 2.0 * u2 * dE_tot_ffn
        dv1_swiglu = w_u * tg0 - w_g * G_glu * tg1 - dG_glu * u2 + 2.0 * v1 * dE_tot_ffn
        dv2_swiglu = w_u * tg1 + w_g * G_glu * tg0 + dG_glu * u1 + 2.0 * v2 * dE_tot_ffn

        np.add.at(grad_w_up, g_idx, tg0 * v1 + tg1 * v2 + tg2 * u1 + tg3 * u2)
        np.add.at(grad_w_gate, g_idx, G_glu * glu_term)
        np.add.at(grad_kappa_ffn, g_idx, -sig_k_ffn * (D_ffn * D_ffn) * E_tot * dD_ffn)

        # 4. Logit Head
        g0_a, g1_a, g2_a, g3_a = m4_w * g0_raw, m4_w * g1_raw, m4_w * g2_raw, m4_w * g3_raw
        Delta_E = E_V - E_Q
        D_cap = 1.0 / np.sqrt(1.0 + k_head_eff * Delta_E * Delta_E)
        G_cap = Delta_E * D_cap
        I_qk = u1 * v1 + u2 * v2
        D_head = 1.0 / (1.0 + k_head_eff * E_tot)
        C_qk = D_head * I_qk

        A0 = -u1 + G_cap * v2
        A1 = u2 - G_cap * v1
        A2 = v1 + G_cap * u2
        A3 = -v2 - G_cap * u1
        S_A = g0_a * A0 + g1_a * A1 + g2_a * A2 + g3_a * A3
        S_B = g0_a * v1 - g1_a * v2 + g2_a * u1 - g3_a * u2

        dC_qk = bt_qk * S_B
        dI_qk = D_head * dC_qk
        dD_head = I_qk * dC_qk
        dE_tot_head = -k_head_eff * (D_head * D_head) * dD_head
        dG_cap = al_eff * (g0_a * v2 - g1_a * v1 + g2_a * u2 - g3_a * u1)
        dDelta_E = dG_cap * (D_cap * D_cap * D_cap)

        du1_head = (
            al_eff * (-g0_a - G_cap * g3_a)
            + bt_qk * C_qk * g2_a
            + dI_qk * v1
            - 2.0 * u1 * dDelta_E
            + 2.0 * u1 * dE_tot_head
        )
        du2_head = (
            al_eff * (g1_a + G_cap * g2_a)
            - bt_qk * C_qk * g3_a
            + dI_qk * v2
            - 2.0 * u2 * dDelta_E
            + 2.0 * u2 * dE_tot_head
        )
        dv1_head = (
            al_eff * (-G_cap * g1_a + g2_a)
            + bt_qk * C_qk * g0_a
            + dI_qk * u1
            + 2.0 * v1 * dDelta_E
            + 2.0 * v1 * dE_tot_head
        )
        dv2_head = (
            al_eff * (G_cap * g0_a - g3_a)
            - bt_qk * C_qk * g1_a
            + dI_qk * u2
            + 2.0 * v2 * dDelta_E
            + 2.0 * v2 * dE_tot_head
        )

        np.add.at(grad_alpha_res, g_idx, d_al * S_A)
        np.add.at(grad_beta_qk, g_idx, C_qk * S_B)
        np.add.at(
            grad_kappa_head,
            g_idx,
            sig_k_head
            * (-0.5 * (Delta_E * Delta_E * Delta_E) * (D_cap * D_cap * D_cap) * dG_cap - (D_head * D_head) * E_tot * dD_head),
        )

        gx_flat[valid_offs, 0] = du1_rope + du1_swiglu + du1_head
        gx_flat[valid_offs, 1] = du2_rope + du2_swiglu + du2_head
        gx_flat[valid_offs, 2] = dv1_qkv + dv1_swiglu + dv1_head
        gx_flat[valid_offs, 3] = dv2_qkv + dv2_swiglu + dv2_head

    return (
        gx_flat.reshape(B, L, D),
        grad_theta_rope,
        grad_gamma_rms,
        grad_kappa_rms,
        grad_theta_qkv,
        grad_w_qkv,
        grad_kappa_qkv,
        grad_w_up,
        grad_w_gate,
        grad_kappa_ffn,
        grad_alpha_res,
        grad_beta_qk,
        grad_kappa_head,
    )


# ==============================================================================
# 5. PyTorch Autograd Function 与 LLMHierarchyOperator 独立算子封装
# ==============================================================================
if HAS_TORCH:

    class LLMHierarchyAutogradFunction(torch.autograd.Function):
        """绑定解析反向核的自定义 PyTorch Autograd Function。"""

        @staticmethod
        def forward(
            ctx,
            x: torch.Tensor,
            theta_rope: torch.Tensor,
            gamma_rms: torch.Tensor,
            kappa_rms: torch.Tensor,
            theta_qkv: torch.Tensor,
            w_qkv: torch.Tensor,
            kappa_qkv: torch.Tensor,
            w_up: torch.Tensor,
            w_gate: torch.Tensor,
            kappa_ffn: torch.Tensor,
            alpha_res: torch.Tensor,
            beta_qk: torch.Tensor,
            kappa_head: torch.Tensor,
            eps_zpe: float = 1e-4,
            sub_op_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        ) -> torch.Tensor:
            ctx.save_for_backward(
                x,
                theta_rope,
                gamma_rms,
                kappa_rms,
                theta_qkv,
                w_qkv,
                kappa_qkv,
                w_up,
                w_gate,
                kappa_ffn,
                alpha_res,
                beta_qk,
                kappa_head,
            )
            ctx.eps_zpe = eps_zpe
            ctx.sub_op_weights = sub_op_weights

            y_np = llm_hierarchy_numpy_forward(
                x.detach().cpu().numpy(),
                theta_rope.detach().cpu().numpy(),
                gamma_rms.detach().cpu().numpy(),
                kappa_rms.detach().cpu().numpy(),
                theta_qkv.detach().cpu().numpy(),
                w_qkv.detach().cpu().numpy(),
                kappa_qkv.detach().cpu().numpy(),
                w_up.detach().cpu().numpy(),
                w_gate.detach().cpu().numpy(),
                kappa_ffn.detach().cpu().numpy(),
                alpha_res.detach().cpu().numpy(),
                beta_qk.detach().cpu().numpy(),
                kappa_head.detach().cpu().numpy(),
                eps_zpe=eps_zpe,
                sub_op_weights=sub_op_weights,
            )
            return torch.as_tensor(y_np, dtype=x.dtype, device=x.device)

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            saved = ctx.saved_tensors
            np_args = [t.detach().cpu().numpy() for t in saved]
            grads_np = llm_hierarchy_numpy_backward(
                grad_out.detach().cpu().numpy(),
                *np_args,
                eps_zpe=ctx.eps_zpe,
                sub_op_weights=ctx.sub_op_weights,
            )
            grads_torch = tuple(
                torch.as_tensor(g, dtype=saved[0].dtype, device=saved[0].device) for g in grads_np
            )
            return grads_torch + (None, None)

    class LLMHierarchyOperator(nn.Module):
        """基于完整 LLM 算子层次机制的独立神经网络算子 (PyTorch nn.Module)。"""

        def __init__(self, config: Optional[LLMHierarchyConfig] = None) -> None:
            super().__init__()
            self.config = config or LLMHierarchyConfig()
            G = self.config.dim // 4

            # 子算子 I (Pre-RMSNorm & RoPE) 可学习参数
            self.theta_rope = nn.Parameter(torch.linspace(0.40, 0.95, G, dtype=torch.float64))
            self.gamma_rms = nn.Parameter(torch.linspace(0.90, 1.25, G, dtype=torch.float64))
            self.kappa_rms = nn.Parameter(torch.linspace(-0.20, 0.30, G, dtype=torch.float64))

            # 子算子 II (Multi-Head QKV Value Projection) 可学习参数
            self.theta_qkv = nn.Parameter(torch.linspace(-0.75, -0.25, G, dtype=torch.float64))
            self.w_qkv = nn.Parameter(torch.linspace(0.95, 1.30, G, dtype=torch.float64))
            self.kappa_qkv = nn.Parameter(torch.linspace(-0.10, 0.40, G, dtype=torch.float64))

            # 子算子 III (SwiGLU Gated FFN) 可学习参数
            self.w_up = nn.Parameter(torch.linspace(0.75, 1.15, G, dtype=torch.float64))
            self.w_gate = nn.Parameter(torch.linspace(0.65, 1.05, G, dtype=torch.float64))
            self.kappa_ffn = nn.Parameter(torch.linspace(0.00, 0.50, G, dtype=torch.float64))

            # 子算子 IV (QK Dot-Product & Softcap Logit Head) 可学习参数
            self.alpha_res = nn.Parameter(torch.linspace(0.60, 1.20, G, dtype=torch.float64))
            self.beta_qk = nn.Parameter(torch.linspace(0.85, 1.25, G, dtype=torch.float64))
            self.kappa_head = nn.Parameter(torch.linspace(-0.20, 0.30, G, dtype=torch.float64))

        def forward(
            self,
            x: torch.Tensor,
            use_custom_autograd: bool = False,
            return_sub_ops: bool = False,
        ):
            if use_custom_autograd and not return_sub_ops:
                return LLMHierarchyAutogradFunction.apply(
                    x,
                    self.theta_rope,
                    self.gamma_rms,
                    self.kappa_rms,
                    self.theta_qkv,
                    self.w_qkv,
                    self.kappa_qkv,
                    self.w_up,
                    self.w_gate,
                    self.kappa_ffn,
                    self.alpha_res,
                    self.beta_qk,
                    self.kappa_head,
                    self.config.eps_zpe,
                    self.config.sub_op_weights,
                )

            B, L, D = x.shape
            G = D // 4
            xg = x.view(B, L, G, 4)
            u1, u2, v1, v2 = xg[..., 0], xg[..., 1], xg[..., 2], xg[..., 3]
            eps = self.config.eps_zpe

            k_rms_eff = F.softplus(self.kappa_rms) + eps
            k_qkv_eff = F.softplus(self.kappa_qkv) + eps
            k_ffn_eff = F.softplus(self.kappa_ffn) + eps
            al_res_eff = torch.sigmoid(self.alpha_res)
            k_head_eff = F.softplus(self.kappa_head) + eps

            pos_l = (
                torch.arange(1, L + 1, dtype=x.dtype, device=x.device).view(1, L, 1) / float(L)
            )
            phi_rope = pos_l * self.theta_rope.view(1, 1, G)

            E_Q = u1 * u1 + u2 * u2
            D_rms = torch.rsqrt(1.0 + k_rms_eff * E_Q)
            cos_r, sin_r = torch.cos(phi_rope), torch.sin(phi_rope)
            r_Q1 = cos_r * u1 - sin_r * u2
            r_Q2 = sin_r * u1 + cos_r * u2
            zeros = torch.zeros_like(u1)
            op_rope = torch.stack(
                [self.gamma_rms * D_rms * r_Q1, self.gamma_rms * D_rms * r_Q2, zeros, zeros], dim=-1
            )

            E_V = v1 * v1 + v2 * v2
            D_qkv = torch.rsqrt(1.0 + k_qkv_eff * E_V)
            cos_v, sin_v = torch.cos(self.theta_qkv), torch.sin(self.theta_qkv)
            r_V1 = cos_v * v1 + sin_v * v2
            r_V2 = -sin_v * v1 + cos_v * v2
            op_qkv = torch.stack(
                [zeros, zeros, self.w_qkv * D_qkv * r_V1, self.w_qkv * D_qkv * r_V2], dim=-1
            )

            E_tot = E_Q + E_V
            D_ffn = 1.0 / (1.0 + k_ffn_eff * E_tot)
            G_glu = u1 * v2 - u2 * v1
            op_swiglu = D_ffn.unsqueeze(-1) * torch.stack(
                [
                    self.w_up * v1 + self.w_gate * G_glu * v2,
                    self.w_up * v2 - self.w_gate * G_glu * v1,
                    self.w_up * u1 - self.w_gate * G_glu * u2,
                    self.w_up * u2 + self.w_gate * G_glu * u1,
                ],
                dim=-1,
            )

            Delta_E = E_V - E_Q
            G_cap = Delta_E * torch.rsqrt(1.0 + k_head_eff * (Delta_E * Delta_E))
            I_qk = u1 * v1 + u2 * v2
            C_qk = I_qk / (1.0 + k_head_eff * E_tot)
            op_head = torch.stack(
                [
                    al_res_eff * (-u1 + G_cap * v2) + self.beta_qk * C_qk * v1,
                    al_res_eff * (u2 - G_cap * v1) - self.beta_qk * C_qk * v2,
                    al_res_eff * (v1 + G_cap * u2) + self.beta_qk * C_qk * u1,
                    al_res_eff * (-v2 - G_cap * u1) - self.beta_qk * C_qk * u2,
                ],
                dim=-1,
            )

            m1_w, m2_w, m3_w, m4_w = self.config.sub_op_weights
            y = (m1_w * op_rope + m2_w * op_qkv + m3_w * op_swiglu + m4_w * op_head).reshape(B, L, D)
            if return_sub_ops:
                return y, (
                    op_rope.reshape(B, L, D),
                    op_qkv.reshape(B, L, D),
                    op_swiglu.reshape(B, L, D),
                    op_head.reshape(B, L, D),
                )
            return y

else:

    class _CompatTensor:
        """无 PyTorch 环境下的轻量动态计算图张量封装。"""

        def __init__(
            self,
            data: np.ndarray,
            requires_grad: bool = False,
            _backward_fn=None,
            _parents: Tuple["_CompatTensor", ...] = (),
        ) -> None:
            self.data = np.asarray(data, dtype=np.float64)
            self.requires_grad = requires_grad
            self.grad: Optional[np.ndarray] = None
            self._backward_fn = _backward_fn
            self._parents = _parents

        @property
        def shape(self) -> Tuple[int, ...]:
            return self.data.shape

        def numpy(self) -> np.ndarray:
            return self.data

        def detach(self) -> "_CompatTensor":
            return _CompatTensor(self.data.copy(), requires_grad=False)

        def zero_grad(self) -> None:
            self.grad = None

        def backward(self, grad_output: Optional[np.ndarray] = None) -> None:
            if grad_output is None:
                grad_output = np.ones_like(self.data, dtype=np.float64)
            else:
                grad_output = np.asarray(grad_output, dtype=np.float64)
            self.grad = grad_output if self.grad is None else self.grad + grad_output
            if self._backward_fn is not None:
                self._backward_fn(grad_output)

    class LLMHierarchyOperator:  # type: ignore
        """基于完整 LLM 算子层次机制的独立神经网络算子（兼容模式，API 与 PyTorch nn.Module 完全一致）。"""

        def __init__(self, config: Optional[LLMHierarchyConfig] = None) -> None:
            self.config = config or LLMHierarchyConfig()
            G = self.config.dim // 4

            self.theta_rope = _CompatTensor(np.linspace(0.40, 0.95, G), requires_grad=True)
            self.gamma_rms = _CompatTensor(np.linspace(0.90, 1.25, G), requires_grad=True)
            self.kappa_rms = _CompatTensor(np.linspace(-0.20, 0.30, G), requires_grad=True)

            self.theta_qkv = _CompatTensor(np.linspace(-0.75, -0.25, G), requires_grad=True)
            self.w_qkv = _CompatTensor(np.linspace(0.95, 1.30, G), requires_grad=True)
            self.kappa_qkv = _CompatTensor(np.linspace(-0.10, 0.40, G), requires_grad=True)

            self.w_up = _CompatTensor(np.linspace(0.75, 1.15, G), requires_grad=True)
            self.w_gate = _CompatTensor(np.linspace(0.65, 1.05, G), requires_grad=True)
            self.kappa_ffn = _CompatTensor(np.linspace(0.00, 0.50, G), requires_grad=True)

            self.alpha_res = _CompatTensor(np.linspace(0.60, 1.20, G), requires_grad=True)
            self.beta_qk = _CompatTensor(np.linspace(0.85, 1.25, G), requires_grad=True)
            self.kappa_head = _CompatTensor(np.linspace(-0.20, 0.30, G), requires_grad=True)

        def parameters(self) -> List[_CompatTensor]:
            return [
                self.theta_rope,
                self.gamma_rms,
                self.kappa_rms,
                self.theta_qkv,
                self.w_qkv,
                self.kappa_qkv,
                self.w_up,
                self.w_gate,
                self.kappa_ffn,
                self.alpha_res,
                self.beta_qk,
                self.kappa_head,
            ]

        def zero_grad(self) -> None:
            for p in self.parameters():
                p.zero_grad()

        def forward(
            self,
            x: Union[np.ndarray, _CompatTensor],
            use_custom_autograd: bool = True,
            return_sub_ops: bool = False,
        ):
            x_tensor = x if isinstance(x, _CompatTensor) else _CompatTensor(x, requires_grad=True)
            params = self.parameters()
            p_arrays = [p.data for p in params]

            B, L, D = x_tensor.shape
            G = D // 4
            xg = x_tensor.data.reshape(B, L, G, 4)
            u1, u2, v1, v2 = xg[..., 0], xg[..., 1], xg[..., 2], xg[..., 3]
            eps = self.config.eps_zpe

            k_rms_eff = np.log1p(np.exp(-np.abs(self.kappa_rms.data))) + np.maximum(self.kappa_rms.data, 0.0) + eps
            k_qkv_eff = np.log1p(np.exp(-np.abs(self.kappa_qkv.data))) + np.maximum(self.kappa_qkv.data, 0.0) + eps
            k_ffn_eff = np.log1p(np.exp(-np.abs(self.kappa_ffn.data))) + np.maximum(self.kappa_ffn.data, 0.0) + eps
            al_res_eff = _np_sigmoid(self.alpha_res.data)
            k_head_eff = np.log1p(np.exp(-np.abs(self.kappa_head.data))) + np.maximum(self.kappa_head.data, 0.0) + eps

            pos_l = (np.arange(1, L + 1, dtype=np.float64) / float(L)).reshape(1, L, 1)
            phi_rope = pos_l * self.theta_rope.data.reshape(1, 1, G)

            E_Q = u1**2 + u2**2
            D_rms = (1.0 + k_rms_eff * E_Q) ** (-0.5)
            c_r, s_r = np.cos(phi_rope), np.sin(phi_rope)
            op_rope = np.stack(
                [
                    self.gamma_rms.data * D_rms * (c_r * u1 - s_r * u2),
                    self.gamma_rms.data * D_rms * (s_r * u1 + c_r * u2),
                    np.zeros_like(u1),
                    np.zeros_like(u2),
                ],
                axis=-1,
            )

            E_V = v1**2 + v2**2
            D_qkv = (1.0 + k_qkv_eff * E_V) ** (-0.5)
            c_v, s_v = np.cos(self.theta_qkv.data), np.sin(self.theta_qkv.data)
            op_qkv = np.stack(
                [
                    np.zeros_like(v1),
                    np.zeros_like(v2),
                    self.w_qkv.data * D_qkv * (c_v * v1 + s_v * v2),
                    self.w_qkv.data * D_qkv * (-s_v * v1 + c_v * v2),
                ],
                axis=-1,
            )

            E_tot = E_Q + E_V
            D_ffn = 1.0 / (1.0 + k_ffn_eff * E_tot)
            G_glu = u1 * v2 - u2 * v1
            op_swiglu = D_ffn[..., None] * np.stack(
                [
                    self.w_up.data * v1 + self.w_gate.data * G_glu * v2,
                    self.w_up.data * v2 - self.w_gate.data * G_glu * v1,
                    self.w_up.data * u1 - self.w_gate.data * G_glu * u2,
                    self.w_up.data * u2 + self.w_gate.data * G_glu * u1,
                ],
                axis=-1,
            )

            Delta_E = E_V - E_Q
            G_cap = Delta_E * ((1.0 + k_head_eff * (Delta_E**2)) ** (-0.5))
            I_qk = u1 * v1 + u2 * v2
            C_qk = I_qk / (1.0 + k_head_eff * E_tot)
            op_head = np.stack(
                [
                    al_res_eff * (-u1 + G_cap * v2) + self.beta_qk.data * C_qk * v1,
                    al_res_eff * (u2 - G_cap * v1) - self.beta_qk.data * C_qk * v2,
                    al_res_eff * (v1 + G_cap * u2) + self.beta_qk.data * C_qk * u1,
                    al_res_eff * (-v2 - G_cap * u1) - self.beta_qk.data * C_qk * u2,
                ],
                axis=-1,
            )

            m1_w, m2_w, m3_w, m4_w = self.config.sub_op_weights
            y_data = (m1_w * op_rope + m2_w * op_qkv + m3_w * op_swiglu + m4_w * op_head).reshape(B, L, D)

            def _bwd(grad_out: np.ndarray) -> None:
                grads = llm_hierarchy_numpy_backward(
                    grad_out,
                    x_tensor.data,
                    *p_arrays,
                    eps_zpe=self.config.eps_zpe,
                    sub_op_weights=self.config.sub_op_weights,
                )
                if x_tensor.requires_grad:
                    x_tensor.grad = grads[0] if x_tensor.grad is None else x_tensor.grad + grads[0]
                for idx, p in enumerate(params):
                    if p.requires_grad:
                        g_val = grads[idx + 1]
                        p.grad = g_val if p.grad is None else p.grad + g_val

            out_tensor = _CompatTensor(
                y_data,
                requires_grad=True,
                _backward_fn=_bwd,
                _parents=(x_tensor,) + tuple(params),
            )
            if return_sub_ops:
                return out_tensor, (
                    _CompatTensor(op_rope.reshape(B, L, D)),
                    _CompatTensor(op_qkv.reshape(B, L, D)),
                    _CompatTensor(op_swiglu.reshape(B, L, D)),
                    _CompatTensor(op_head.reshape(B, L, D)),
                )
            return out_tensor

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)


# ==============================================================================
# 6. 完整 LLM 算子级扩展接口 (KV-Cache 自回归推理与 Softcapped Logit 词表投射)
# ==============================================================================
def llm_hierarchy_forward_with_kv_cache(
    op: LLMHierarchyOperator,
    x: np.ndarray,
    kv_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """支持因果 KV-Cache 状态累积的完整 LLM 层次算子推理接口。

    在执行四路正交 LLM 子算子（RoPE + QKV + SwiGLU + Head）的基础上，维护跨步因果
    Key-Value 状态缓存 `kv_cache = {"k_state": ..., "v_state": ..., "step": ...}`，
    实现 O(1) 步复杂度的自回归增量上下文融合。
    """
    x_arr = np.asarray(x, dtype=np.float64)
    B, L, D = x_arr.shape
    G = D // 4
    decay = op.config.kv_decay
    eps = op.config.eps_zpe

    if HAS_TORCH:
        params_np = [
            op.theta_rope.detach().cpu().numpy(),
            op.gamma_rms.detach().cpu().numpy(),
            op.kappa_rms.detach().cpu().numpy(),
            op.theta_qkv.detach().cpu().numpy(),
            op.w_qkv.detach().cpu().numpy(),
            op.kappa_qkv.detach().cpu().numpy(),
            op.w_up.detach().cpu().numpy(),
            op.w_gate.detach().cpu().numpy(),
            op.kappa_ffn.detach().cpu().numpy(),
            op.alpha_res.detach().cpu().numpy(),
            op.beta_qk.detach().cpu().numpy(),
            op.kappa_head.detach().cpu().numpy(),
        ]
    else:
        params_np = [p.numpy() for p in op.parameters()]

    y_base = llm_hierarchy_numpy_forward(
        x_arr, *params_np, eps_zpe=eps, sub_op_weights=op.config.sub_op_weights
    )

    xg = x_arr.reshape(B, L, G, 4)
    k_cur = xg[..., :2]  # (B, L, G, 2)
    v_cur = xg[..., 2:]  # (B, L, G, 2)

    if kv_cache is None:
        k_state = np.zeros((B, G, 2), dtype=np.float64)
        v_state = np.zeros((B, G, 2), dtype=np.float64)
        step_count = 0
    else:
        k_state = kv_cache["k_state"].copy()
        v_state = kv_cache["v_state"].copy()
        step_count = int(kv_cache["step"])

    ctx_out = np.zeros_like(xg)
    for t in range(L):
        k_state = decay * k_state + (1.0 - decay) * k_cur[:, t]
        v_state = decay * v_state + (1.0 - decay) * v_cur[:, t]
        qk_score = np.sum(k_cur[:, t] * k_state, axis=-1, keepdims=True) / math.sqrt(2.0)
        attn_gate = _np_sigmoid(qk_score)
        ctx_v = attn_gate * v_state
        ctx_out[:, t, :, 2:] = ctx_v
        step_count += 1

    y_With_kv = y_base + ctx_out.reshape(B, L, D)
    new_cache = {"k_state": k_state, "v_state": v_state, "step": np.array(step_count)}
    return y_With_kv, new_cache


def llm_hierarchy_compute_logits(
    hidden_states: np.ndarray,
    vocab_weight: np.ndarray,
    logit_softcap: float = 30.0,
    eps_zpe: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """完整 LLM 最终输出层：执行 Final-RMSNorm、词表线性投射、有理 Logit Softcapping 与稳定 Softmax。

    Returns:
        logits: 软截断后的词表 Logits，形状为 (B, L, V)。
        probs: 归一化词表概率分布，形状为 (B, L, V)。
    """
    h = np.asarray(hidden_states, dtype=np.float64)
    W_v = np.asarray(vocab_weight, dtype=np.float64)  # (D, V)
    rms_inv = 1.0 / np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + eps_zpe)
    h_norm = h * rms_inv
    raw_logits = h_norm @ W_v
    # 有理软截断包络 (将输出平滑有界限制在 [-logit_softcap, +logit_softcap])
    scaled = raw_logits / logit_softcap
    logits = logit_softcap * (scaled / np.sqrt(1.0 + scaled * scaled))
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return logits, probs


# ==============================================================================
# 7. 三重位级等价校验与五维正交独立性形式化审计 (Verification & 5D Audit Suite)
# ==============================================================================
def run_verification_and_5d_audit(verbose: bool = True) -> Dict[str, object]:
    """执行完整三重位级等价验证、中心有限差分 Gradcheck、KV-Cache/Logits 检验与五维正交独立性审计。"""
    np.random.seed(2026)
    B, L, D = 8, 16, 16
    G = D // 4
    cfg = LLMHierarchyConfig(dim=D, vocab_size=32, eps_zpe=1e-4, block_size=64)

    x_np = np.random.randn(B, L, D).astype(np.float64)
    grad_out_np = np.random.randn(B, L, D).astype(np.float64)

    op = LLMHierarchyOperator(cfg)
    if HAS_TORCH:
        params_np = [
            op.theta_rope.detach().cpu().numpy().copy(),
            op.gamma_rms.detach().cpu().numpy().copy(),
            op.kappa_rms.detach().cpu().numpy().copy(),
            op.theta_qkv.detach().cpu().numpy().copy(),
            op.w_qkv.detach().cpu().numpy().copy(),
            op.kappa_qkv.detach().cpu().numpy().copy(),
            op.w_up.detach().cpu().numpy().copy(),
            op.w_gate.detach().cpu().numpy().copy(),
            op.kappa_ffn.detach().cpu().numpy().copy(),
            op.alpha_res.detach().cpu().numpy().copy(),
            op.beta_qk.detach().cpu().numpy().copy(),
            op.kappa_head.detach().cpu().numpy().copy(),
        ]
    else:
        params_np = [p.numpy().copy() for p in op.parameters()]

    # ------------------------------------------------------------------
    # A. 三套实现位级等价检验 (NumPy vs PyTorch/Autograd vs Triton Kernel)
    # ------------------------------------------------------------------
    y_numpy, sub_ops_np = llm_hierarchy_numpy_forward(
        x_np, *params_np, eps_zpe=cfg.eps_zpe, return_sub_ops=True
    )
    y_triton = llm_hierarchy_triton_forward(
        x_np, *params_np, eps_zpe=cfg.eps_zpe, block_size=cfg.block_size
    )

    if HAS_TORCH:
        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        y_torch_t = op(x_t)
        y_torch = y_torch_t.detach().cpu().numpy()
        y_torch_t.backward(torch.tensor(grad_out_np, dtype=torch.float64))
        gx_torch = x_t.grad.detach().cpu().numpy()
    else:
        x_t = _CompatTensor(x_np, requires_grad=True)
        y_torch_t = op(x_t)
        y_torch = y_torch_t.numpy()
        y_torch_t.backward(grad_out_np)
        gx_torch = x_t.grad

    grads_numpy_all = llm_hierarchy_numpy_backward(
        grad_out_np, x_np, *params_np, eps_zpe=cfg.eps_zpe, return_sub_grads=True
    )
    grads_numpy = grads_numpy_all[:-1]
    sub_grads_np = grads_numpy_all[-1]

    grads_triton = llm_hierarchy_triton_backward(
        grad_out_np, x_np, *params_np, eps_zpe=cfg.eps_zpe, block_size=cfg.block_size
    )

    fwd_err_torch = float(np.max(np.abs(y_numpy - y_torch)))
    fwd_err_triton = float(np.max(np.abs(y_numpy - y_triton)))
    bwd_err_torch = float(np.max(np.abs(grads_numpy[0] - gx_torch)))
    bwd_err_triton_x = float(np.max(np.abs(grads_numpy[0] - grads_triton[0])))
    bwd_err_triton_params = float(
        max(np.max(np.abs(grads_numpy[i] - grads_triton[i])) for i in range(1, 13))
    )

    # 验证 KV-Cache 自回归一致性（整序列一次推理 vs 逐 Token 增量解码）
    y_kv_full, cache_full = llm_hierarchy_forward_with_kv_cache(op, x_np[:, :4, :])
    cache_step = None
    step_outputs = []
    for t_step in range(4):
        # 校验增量 KV 状态递推
        _, cache_step = llm_hierarchy_forward_with_kv_cache(
            op, x_np[:, t_step : t_step + 1, :], kv_cache=cache_step
        )
        step_outputs.append(cache_step["v_state"].copy())
    kv_state_diff = float(np.max(np.abs(cache_full["v_state"] - cache_step["v_state"])))

    # 验证词表 Logit Softcap 与概率归一化
    W_vocab = np.random.randn(D, cfg.vocab_size) * 0.5
    logits_out, probs_out = llm_hierarchy_compute_logits(
        y_numpy, W_vocab, logit_softcap=cfg.logit_softcap, eps_zpe=cfg.eps_zpe
    )
    prob_sum_err = float(np.max(np.abs(np.sum(probs_out, axis=-1) - 1.0)))

    # ------------------------------------------------------------------
    # B. 解析反向传播对比中心有限差分校验 (Gradcheck, eps = 1e-6)
    # ------------------------------------------------------------------
    B_gc, L_gc = 2, 4
    x_gc = x_np[:B_gc, :L_gc].copy()
    go_gc = grad_out_np[:B_gc, :L_gc].copy()
    grads_gc = llm_hierarchy_numpy_backward(go_gc, x_gc, *params_np, eps_zpe=cfg.eps_zpe)

    eps_fd = 1e-6
    gx_fd = np.zeros_like(x_gc)
    for idx in np.ndindex(x_gc.shape):
        xp = x_gc.copy()
        xp[idx] += eps_fd
        xm = x_gc.copy()
        xm[idx] -= eps_fd
        yp = llm_hierarchy_numpy_forward(xp, *params_np, eps_zpe=cfg.eps_zpe)
        ym = llm_hierarchy_numpy_forward(xm, *params_np, eps_zpe=cfg.eps_zpe)
        gx_fd[idx] = np.sum((yp - ym) / (2.0 * eps_fd) * go_gc)

    grad_x_rel_err = float(
        np.max(np.abs(grads_gc[0] - gx_fd) / np.maximum(np.abs(grads_gc[0]) + np.abs(gx_fd), 1e-1))
    )
    grad_x_norm_rel_err = float(
        np.linalg.norm(grads_gc[0] - gx_fd)
        / np.maximum(np.linalg.norm(grads_gc[0]) + np.linalg.norm(gx_fd), 1e-12)
    )

    param_names = [
        "theta_rope",
        "gamma_rms",
        "kappa_rms",
        "theta_qkv",
        "w_qkv",
        "kappa_qkv",
        "w_up",
        "w_gate",
        "kappa_ffn",
        "alpha_res",
        "beta_qk",
        "kappa_head",
    ]
    param_rel_errs: Dict[str, float] = {}
    for p_idx, p_name in enumerate(param_names):
        g_ana = grads_gc[p_idx + 1]
        g_fd = np.zeros_like(g_ana)
        for k in range(G):
            pp = [p.copy() for p in params_np]
            pp[p_idx][k] += eps_fd
            pm = [p.copy() for p in params_np]
            pm[p_idx][k] -= eps_fd
            yp = llm_hierarchy_numpy_forward(x_gc, *pp, eps_zpe=cfg.eps_zpe)
            ym = llm_hierarchy_numpy_forward(x_gc, *pm, eps_zpe=cfg.eps_zpe)
            g_fd[k] = np.sum((yp - ym) / (2.0 * eps_fd) * go_gc)
        rel_e = float(
            np.max(np.abs(g_ana - g_fd) / np.maximum(np.abs(g_ana) + np.abs(g_fd), 1e-1))
        )
        param_rel_errs[p_name] = rel_e

    max_param_rel_err = max(param_rel_errs.values())

    # ------------------------------------------------------------------
    # C. 五维算子正交独立性形式化审计 (5D Independence Audit)
    # ------------------------------------------------------------------
    # 维度 1: 单文件零同级依赖检验 (AST Static Check)
    allowed_modules = {
        "__future__",
        "ast",
        "math",
        "dataclasses",
        "pathlib",
        "typing",
        "numpy",
        "torch",
        "triton",
    }
    this_file = Path(__file__).resolve()
    tree = ast.parse(this_file.read_text(encoding="utf-8"))
    sibling_imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod not in allowed_modules:
                    sibling_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                sibling_imports.append(f"relative_import_level_{node.level}")
            elif node.module:
                root_mod = node.module.split(".")[0]
                if root_mod not in allowed_modules:
                    sibling_imports.append(node.module)

    # 维度 2: 前向输出子空间满秩检验
    M_fwd = np.column_stack([op_i.ravel() for op_i in sub_ops_np])
    svals_sub = np.linalg.svd(M_fwd, compute_uv=False)
    fwd_sub_rank = int(np.sum(svals_sub > 1e-8))
    fwd_sub_sigma_min = float(svals_sub[-1])

    svals_feat = np.linalg.svd(y_numpy.reshape(-1, D), compute_uv=False)
    fwd_feat_rank = int(np.sum(svals_feat > 1e-8))
    fwd_feat_sigma_min = float(svals_feat[-1])

    # 维度 3: 反向输入雅可比满秩与低共线性检验 (Max VIF < 2.0)
    M_bwd = np.column_stack([g_i.ravel() for g_i in sub_grads_np])
    svals_bwd = np.linalg.svd(M_bwd, compute_uv=False)
    bwd_sub_rank = int(np.sum(svals_bwd > 1e-8))
    bwd_sub_sigma_min = float(svals_bwd[-1])

    corr_bwd = np.corrcoef(M_bwd.T)
    vifs_bwd = np.diag(np.linalg.inv(corr_bwd))
    max_vif_bwd = float(np.max(vifs_bwd))

    corr_fwd = np.corrcoef(M_fwd.T)
    vifs_fwd = np.diag(np.linalg.inv(corr_fwd))
    max_vif_fwd = float(np.max(vifs_fwd))

    # 维度 4: 4x4 参数交叉灵敏度矩阵严格对角化检验
    S_matrix = np.zeros((4, 4), dtype=np.float64)
    delta_p = 0.15
    for j in range(4):
        p_pert = [p.copy() for p in params_np]
        for k in range(3):
            p_pert[3 * j + k] = p_pert[3 * j + k] + delta_p
        _, sub_ops_pert = llm_hierarchy_numpy_forward(
            x_np, *p_pert, eps_zpe=cfg.eps_zpe, return_sub_ops=True
        )
        for i in range(4):
            diff_norm = np.linalg.norm(sub_ops_pert[i] - sub_ops_np[i])
            base_norm = np.linalg.norm(sub_ops_np[i])
            S_matrix[i, j] = diff_norm / (base_norm * delta_p)

    off_diag_mask = ~np.eye(4, dtype=bool)
    max_off_diag_sensitivity = float(np.max(np.abs(S_matrix[off_diag_mask])))
    min_diag_sensitivity = float(np.min(np.diag(S_matrix)))

    # 维度 5: 留一法不可替代性检验 (LOOO CV)
    y_flat = y_numpy.ravel()
    ss_tot = float(np.sum((y_flat - np.mean(y_flat)) ** 2))
    w_full, _, _, _ = np.linalg.lstsq(M_fwd, y_flat, rcond=None)
    r2_full = float(1.0 - np.sum((y_flat - M_fwd @ w_full) ** 2) / ss_tot)

    looo_deltas: Dict[str, float] = {}
    sub_op_labels = [
        "O_rope (Pre-RMSNorm & RoPE)",
        "O_qkv (Multi-Head QKV Projection)",
        "O_swiglu (SwiGLU Bilinear FFN)",
        "O_head (QK-Inner & Softcap Head)",
    ]
    for k in range(4):
        keep_cols = [c for c in range(4) if c != k]
        M_sub = M_fwd[:, keep_cols]
        w_sub, _, _, _ = np.linalg.lstsq(M_sub, y_flat, rcond=None)
        r2_sub = float(1.0 - np.sum((y_flat - M_sub @ w_sub) ** 2) / ss_tot)
        looo_deltas[sub_op_labels[k]] = r2_full - r2_sub

    min_looo_delta_r2 = min(looo_deltas.values())

    # 严格断言所有指标达标
    assert len(sibling_imports) == 0, f"发现非零同级依赖: {sibling_imports}"
    assert fwd_err_torch < 1e-14, f"PyTorch 前向误差超标: {fwd_err_torch}"
    assert fwd_err_triton < 1e-14, f"Triton 前向误差超标: {fwd_err_triton}"
    assert bwd_err_triton_x < 1e-14, f"Triton 反向输入梯度误差超标: {bwd_err_triton_x}"
    assert bwd_err_triton_params < 1e-13, f"Triton 反向参数梯度误差超标: {bwd_err_triton_params}"
    assert kv_state_diff < 1e-15, f"KV-Cache 增量状态误差超标: {kv_state_diff}"
    assert prob_sum_err < 1e-14, f"Logits Softmax 概率归一化误差超标: {prob_sum_err}"
    assert grad_x_rel_err < 1e-8, f"输入 X 有限差分相对误差超标: {grad_x_rel_err}"
    assert max_param_rel_err < 1e-8, f"可学参数有限差分相对误差超标: {max_param_rel_err}"
    assert fwd_sub_rank == 4 and fwd_feat_rank == D, "前向子空间不满秩"
    assert bwd_sub_rank == 4 and max_vif_bwd < 2.0, f"反向梯度 VIF 超标: {max_vif_bwd}"
    assert max_off_diag_sensitivity == 0.0, f"交叉灵敏度非零: {max_off_diag_sensitivity}"
    assert min_diag_sensitivity > 0.08, f"对角自灵敏度不足: {min_diag_sensitivity}"
    assert r2_full > 0.999 and min_looo_delta_r2 > 0.02, f"LOOO 检验未通过: {looo_deltas}"

    report = {
        "fwd_err_torch": fwd_err_torch,
        "fwd_err_triton": fwd_err_triton,
        "bwd_err_torch": bwd_err_torch,
        "bwd_err_triton_x": bwd_err_triton_x,
        "bwd_err_triton_params": bwd_err_triton_params,
        "kv_state_diff": kv_state_diff,
        "prob_sum_err": prob_sum_err,
        "grad_x_rel_err": grad_x_rel_err,
        "grad_x_norm_rel_err": grad_x_norm_rel_err,
        "max_param_rel_err": max_param_rel_err,
        "param_rel_errs": param_rel_errs,
        "sibling_imports": len(sibling_imports),
        "fwd_sub_rank": f"{fwd_sub_rank}/4",
        "fwd_sub_sigma_min": fwd_sub_sigma_min,
        "fwd_feat_rank": f"{fwd_feat_rank}/{D}",
        "fwd_feat_sigma_min": fwd_feat_sigma_min,
        "bwd_sub_rank": f"{bwd_sub_rank}/4",
        "bwd_sub_sigma_min": bwd_sub_sigma_min,
        "max_vif_fwd": max_vif_fwd,
        "max_vif_bwd": max_vif_bwd,
        "vifs_bwd": vifs_bwd.tolist(),
        "sensitivity_matrix": S_matrix.tolist(),
        "max_off_diag_sensitivity": max_off_diag_sensitivity,
        "min_diag_sensitivity": min_diag_sensitivity,
        "r2_full": r2_full,
        "looo_deltas": looo_deltas,
        "min_looo_delta_r2": min_looo_delta_r2,
    }

    if verbose:
        print("=" * 78)
        print("  LLM-HIERARCHY OPERATOR: TRIPLE EQUIVALENCE & 5D INDEPENDENCE AUDIT")
        print("=" * 78)
        print(f"[1] Bit-Level Equivalence & LLM Hierarchy Capabilities:")
        print(f"    - NumPy vs PyTorch/Autograd Fwd Max Abs Error : {fwd_err_torch:.3e} (< 1e-14)")
        print(f"    - NumPy vs Triton Kernel Fwd Max Abs Error    : {fwd_err_triton:.3e} (< 1e-14)")
        print(f"    - NumPy vs Triton Kernel Bwd dX Max Abs Error : {bwd_err_triton_x:.3e} (< 1e-14)")
        print(f"    - NumPy vs Triton Kernel Bwd dP Max Abs Error : {bwd_err_triton_params:.3e} (< 1e-13)")
        print(f"    - KV-Cache Full vs Stepwise State Equivalence : {kv_state_diff:.3e} (< 1e-15)")
        print(f"    - Softcapped Vocab Logits Softmax Sum Error   : {prob_sum_err:.3e} (< 1e-14)")
        print(f"[2] Analytical Gradcheck vs Central Finite Difference (eps=1e-6):")
        print(f"    - Input X Max Elementwise Rel Error           : {grad_x_rel_err:.3e} (< 1e-8)")
        print(f"    - Input X Norm Relative Error                 : {grad_x_norm_rel_err:.3e} (< 1e-8)")
        print(f"    - All 12 Learnable Params Max Relative Error  : {max_param_rel_err:.3e} (< 1e-8)")
        for k_name, v_err in param_rel_errs.items():
            print(f"      * grad_{k_name:10s} rel err = {v_err:.3e}")
        print(f"[3] 5-Dimensional Formal Independence Audit:")
        print(f"    - Dim 1 (Zero Sibling Imports)  : {len(sibling_imports)} sibling imports (PASS)")
        print(
            f"    - Dim 2 (Forward Full Rank)     : Sub-Ops {fwd_sub_rank}/4 (sigma_min={fwd_sub_sigma_min:.4f}), "
            f"Features {fwd_feat_rank}/{D} (sigma_min={fwd_feat_sigma_min:.4f}) (PASS)"
        )
        print(
            f"    - Dim 3 (Backward Jacobian VIF) : Sub-Grads {bwd_sub_rank}/4 (sigma_min={bwd_sub_sigma_min:.4f}), "
            f"Max Bwd VIF={max_vif_bwd:.4f} (< 2.0), Max Fwd VIF={max_vif_fwd:.4f} (PASS)"
        )
        print(
            f"    - Dim 4 (4x4 Param Sensitivity) : Max Off-Diag={max_off_diag_sensitivity:.2e} (exact 0.00e+00), "
            f"Min Diag={min_diag_sensitivity:.4f} (> 0.08) (PASS)"
        )
        for r_idx in range(4):
            row_str = "  ".join(f"{S_matrix[r_idx, c_idx]:.3e}" for c_idx in range(4))
            print(f"      * Row {r_idx + 1}: [{row_str}]")
        print(
            f"    - Dim 5 (LOOO Irreplaceability) : Full R^2={r2_full:.6f} (> 0.999), "
            f"Min Delta R^2=+{min_looo_delta_r2:.6f} (> +0.02) (PASS)"
        )
        for op_label, dr2 in looo_deltas.items():
            print(f"      * Drop {op_label:34s} -> Delta R^2 = +{dr2:.6f}")
        print("=" * 78)
        print("  ALL LLM-HIERARCHY VERIFICATION & 5D AUDIT CHECKS PASSED SUCCESSFULLY!")
        print("=" * 78)

    return report


if __name__ == "__main__":
    run_verification_and_5d_audit(verbose=True)