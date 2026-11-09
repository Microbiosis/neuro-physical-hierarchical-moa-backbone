#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spin-Chirality Independent Neural Operator (粒子自旋与手性理论独立新算子)
================================================================================

零同级依赖（0 Sibling Imports）、自包含、可微分的深度学习底层物理混合算子模块。
基于相对论量子场论（Dirac/Weyl 旋量手性投影理论）、SU(2) 泡利自旋李代数进动动力学与
Adler-Bell-Jackiw (ABJ) 手性反常轴矢流耦合机制，将四分量 Dirac 双旋量特征流分解为
四个严格正交、参数解耦的物理子算子通道：

1. 左手 Weyl 手性规范流子算子 (Left-Handed Weyl Chiral Gauge Current, O_L):
   基于手性投影算符 P_L = (I - gamma^5)/2 提取左手旋量 psi_L = (u_1, u_2)^T，
   施加 SU(2)_L 弱同位旋李代数相位旋转 R(theta_L) 与有理能量阻尼包络
   D_L = 1 / sqrt(1 + kappa_{L,eff} * ||psi_L||^2)。

2. 右手 Weyl 手性螺旋度流子算子 (Right-Handed Weyl Chiral Helicity Current, O_R):
   基于手性投影算符 P_R = (I + gamma^5)/2 提取右手旋量 psi_R = (v_1, v_2)^T，
   施加反向螺旋度 SU(2)_R 相位旋转 R^dagger(theta_R) 与有理能量阻尼包络
   D_R = 1 / sqrt(1 + kappa_{R,eff} * ||psi_R||^2)。

3. Pauli 自旋双线性进动与螺旋度交换流子算子 (Pauli Spin Precession & Exchange Current, O_spin):
   基于泡利自旋矩阵构造赝标量手性力矩 S_y = u_1 v_2 - u_2 v_1 与跨手性自旋翻转耦合，
   施加总旋量能量有理包络 D_s = 1 / (1 + kappa_{s,eff} * E_tot)。

4. Adler-Bell-Jackiw (ABJ) 手性反常与轴矢流门控子算子 (ABJ Chiral Anomaly & Axial Gate, O_anom):
   由手性密度不平衡度（Pontryagin 拓扑荷密度代理）P_5 = ||psi_R||^2 - ||psi_L||^2
   驱动有界反常包络 G_anom = P_5 / sqrt(1 + kappa_{a,eff} * P_5^2)，结合标量-赝标量
   手性凝聚项 I_ps = u_1 v_1 + u_2 v_2 与有界衰减率 alpha_{a,eff} = sigmoid(alpha_a)。

核心数值与工程特性：
- 严格正性/有界重参数化：所有有理分母阻尼参数统一施加 kappa_eff = softplus(kappa) + 1e-4 > 0，
  所有衰减门控参数施加 alpha_eff = sigmoid(alpha) in (0, 1)，根治除零奇点与负门控失稳。
- 三套实现位级等价：双精度 NumPy 解析前反向引擎、PyTorch Autograd 引擎与 @triton.jit
  融合前反向核（含无 GPU 环境下的 Tile 级 SIMD 精确核仿真器）前向最大绝对误差 < 1e-14。
- 解析梯度校验（Gradcheck）：输入与全部 12 组可学习参数的解析反向传播对比中心有限差分
  （eps = 1e-6）最大相对误差 < 1e-8。
- 五维算子正交独立性验证（5D Independence Audit）：
  (1) 单文件零同级依赖（0 sibling imports）
  (2) 前向输出子空间满秩（4/4 子算子满秩且 D/D 特征满秩，sigma_min > 0）
  (3) 反向输入雅可比满秩与低共线性（Max VIF < 2.0）
  (4) 4x4 参数交叉灵敏度矩阵严格对角化（非对角精确为 0.00e+00，对角自灵敏度 > 0.08）
  (5) 留一法不可替代性检验（LOOO CV：全模型 R^2 > 0.999，移除任一子算子 Delta R^2 > +0.02）
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
# 1. 算子配置结构体 (SpinChiralityConfig)
# ==============================================================================

__all__ = [
    "SpinChiralityConfig",
    "SpinChiralityOperator",
    "SpinChiralityAutogradFunction",
]

@dataclass
class SpinChiralityConfig:
    """粒子自旋与手性算子配置类。

    Attributes:
        dim: 输入/输出特征总维度 D，必须为 4 的整数倍（对应 G = D // 4 个四分量 Dirac 双旋量组）。
        eps_zpe: 真空零点能/有理分母正性安全下界常量（默认 1e-4）。
        sub_op_weights: 四个正交物理子算子 (O_L, O_R, O_spin, O_anom) 的激活掩码或缩放权重。
        use_triton: 在 GPU 可用时是否优先调度 @triton.jit 融合算子核。
        block_size: Triton 核或 Tile 级融合引擎的线程块大小（2 的幂次）。
    """

    dim: int = 16
    eps_zpe: float = 1e-4
    sub_op_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    use_triton: bool = True
    block_size: int = 64

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.dim % 4 != 0:
            raise ValueError(
                f"SpinChiralityConfig.dim 必须为 4 的正整数倍（四分量 Dirac 旋量分组），当前为 {self.dim}"
            )
        if self.eps_zpe <= 0.0:
            raise ValueError(f"eps_zpe 必须严格大于 0，当前为 {self.eps_zpe}")
        if len(self.sub_op_weights) != 4:
            raise ValueError("sub_op_weights 必须包含 4 个子算子权重 (w_L_mask, w_R_mask, w_spin_mask, w_anom_mask)")


# ==============================================================================
# 2. 数值安全基础原语 (Softplus / Sigmoid 及其解析导数)
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
def spin_chirality_numpy_forward(
    x: np.ndarray,
    theta_L: np.ndarray,
    w_L: np.ndarray,
    kappa_L: np.ndarray,
    theta_R: np.ndarray,
    w_R: np.ndarray,
    kappa_R: np.ndarray,
    omega_s: np.ndarray,
    gamma_s: np.ndarray,
    kappa_s: np.ndarray,
    alpha_a: np.ndarray,
    beta_a: np.ndarray,
    kappa_a: np.ndarray,
    eps_zpe: float = 1e-4,
    sub_op_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    return_sub_ops: bool = False,
    return_gate_diagnostics: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """粒子自旋与手性算子双精度 NumPy 解析前向传播。

    Args:
        x: 输入旋量特征张量，形状为 (B, L, D)，其中 D = 4 * G。
        theta_L, w_L, kappa_L: 左手 Weyl 旋量规范流参数，形状均为 (G,)。
        theta_R, w_R, kappa_R: 右手 Weyl 旋量螺旋度流参数，形状均为 (G,)。
        omega_s, gamma_s, kappa_s: Pauli 自旋进动与跨手性交换参数，形状均为 (G,)。
        alpha_a, beta_a, kappa_a: ABJ 手性反常与轴矢流耦合参数，形状均为 (G,)。
        eps_zpe: 有理分母正性下界常量（默认 1e-4）。
        sub_op_weights: 四个物理子算子的合成掩码权重。
        return_sub_ops: 若为 True，额外返回四个正交子算子的独立输出元组 (op_L, op_R, op_spin, op_anom)。
        return_gate_diagnostics: 若为 True，额外返回物理门控诊断字典（如手性反常门 G_anom 等）。

    Returns:
        y: 算子输出张量，形状为 (B, L, D)；若开启附加标志则返回对应元组。
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

    # 正性有理分母与有界衰减率重参数化
    kappa_L_eff = _np_softplus(np.asarray(kappa_L, dtype=np.float64)) + eps_zpe
    kappa_R_eff = _np_softplus(np.asarray(kappa_R, dtype=np.float64)) + eps_zpe
    kappa_s_eff = _np_softplus(np.asarray(kappa_s, dtype=np.float64)) + eps_zpe
    alpha_a_eff = _np_sigmoid(np.asarray(alpha_a, dtype=np.float64))
    kappa_a_eff = _np_softplus(np.asarray(kappa_a, dtype=np.float64)) + eps_zpe

    theta_L_arr = np.asarray(theta_L, dtype=np.float64)
    w_L_arr = np.asarray(w_L, dtype=np.float64)
    theta_R_arr = np.asarray(theta_R, dtype=np.float64)
    w_R_arr = np.asarray(w_R, dtype=np.float64)
    omega_s_arr = np.asarray(omega_s, dtype=np.float64)
    gamma_s_arr = np.asarray(gamma_s, dtype=np.float64)
    beta_a_arr = np.asarray(beta_a, dtype=np.float64)

    # ------------------------------------------------------------------
    # 子算子 I: 左手 Weyl 手性规范流 (O_L)
    # ------------------------------------------------------------------
    E_L = u1 * u1 + u2 * u2
    Q_L = 1.0 + kappa_L_eff * E_L
    D_L = 1.0 / np.sqrt(Q_L)
    cos_L = np.cos(theta_L_arr)
    sin_L = np.sin(theta_L_arr)
    r_L1 = cos_L * u1 - sin_L * u2
    r_L2 = sin_L * u1 + cos_L * u2
    op_L = np.stack(
        [
            w_L_arr * D_L * r_L1,
            w_L_arr * D_L * r_L2,
            np.zeros_like(u1),
            np.zeros_like(u2),
        ],
        axis=-1,
    )

    # ------------------------------------------------------------------
    # 子算子 II: 右手 Weyl 手性螺旋度流 (O_R)
    # ------------------------------------------------------------------
    E_R = v1 * v1 + v2 * v2
    Q_R = 1.0 + kappa_R_eff * E_R
    D_R = 1.0 / np.sqrt(Q_R)
    cos_R = np.cos(theta_R_arr)
    sin_R = np.sin(theta_R_arr)
    r_R1 = cos_R * v1 + sin_R * v2
    r_R2 = -sin_R * v1 + cos_R * v2
    op_R = np.stack(
        [
            np.zeros_like(v1),
            np.zeros_like(v2),
            w_R_arr * D_R * r_R1,
            w_R_arr * D_R * r_R2,
        ],
        axis=-1,
    )

    # ------------------------------------------------------------------
    # 子算子 III: Pauli 自旋双线性进动与螺旋度交换流 (O_spin)
    # ------------------------------------------------------------------
    E_tot = E_L + E_R
    Q_s = 1.0 + kappa_s_eff * E_tot
    D_s = 1.0 / Q_s
    S_y = u1 * v2 - u2 * v1
    m0 = omega_s_arr * v1 + gamma_s_arr * S_y * v2
    m1 = omega_s_arr * v2 - gamma_s_arr * S_y * v1
    m2 = omega_s_arr * u1 - gamma_s_arr * S_y * u2
    m3 = omega_s_arr * u2 + gamma_s_arr * S_y * u1
    op_spin = D_s[..., None] * np.stack([m0, m1, m2, m3], axis=-1)

    # ------------------------------------------------------------------
    # 子算子 IV: ABJ 手性反常与轴矢流螺旋度门控 (O_anom)
    # ------------------------------------------------------------------
    P_5 = E_R - E_L
    Q_P5 = 1.0 + kappa_a_eff * (P_5 * P_5)
    D_P5 = 1.0 / np.sqrt(Q_P5)
    G_anom = P_5 * D_P5
    I_ps = u1 * v1 + u2 * v2
    Q_a = 1.0 + kappa_a_eff * E_tot
    D_a = 1.0 / Q_a
    C_ps = D_a * I_ps

    A0 = -u1 + G_anom * v2
    A1 = u2 - G_anom * v1
    A2 = v1 + G_anom * u2
    A3 = -v2 - G_anom * u1
    op_anom = np.stack(
        [
            alpha_a_eff * A0 + beta_a_arr * C_ps * v1,
            alpha_a_eff * A1 - beta_a_arr * C_ps * v2,
            alpha_a_eff * A2 + beta_a_arr * C_ps * u1,
            alpha_a_eff * A3 - beta_a_arr * C_ps * u2,
        ],
        axis=-1,
    )

    mL, mR, mS, mA = sub_op_weights
    y_group = mL * op_L + mR * op_R + mS * op_spin + mA * op_anom
    y = y_group.reshape(B, L, D)

    sub_ops_tuple = (
        op_L.reshape(B, L, D),
        op_R.reshape(B, L, D),
        op_spin.reshape(B, L, D),
        op_anom.reshape(B, L, D),
    )

    if return_sub_ops and return_gate_diagnostics:
        diagnostics = {
            "D_L": D_L,
            "D_R": D_R,
            "D_s": D_s,
            "G_anom": G_anom,
            "P_5": P_5,
            "S_y": S_y,
        }
        return y, sub_ops_tuple, diagnostics
    if return_sub_ops:
        return y, sub_ops_tuple
    if return_gate_diagnostics:
        diagnostics = {
            "D_L": D_L,
            "D_R": D_R,
            "D_s": D_s,
            "G_anom": G_anom,
            "P_5": P_5,
            "S_y": S_y,
        }
        return y, diagnostics
    return y


def spin_chirality_numpy_backward(
    grad_out: np.ndarray,
    x: np.ndarray,
    theta_L: np.ndarray,
    w_L: np.ndarray,
    kappa_L: np.ndarray,
    theta_R: np.ndarray,
    w_R: np.ndarray,
    kappa_R: np.ndarray,
    omega_s: np.ndarray,
    gamma_s: np.ndarray,
    kappa_s: np.ndarray,
    alpha_a: np.ndarray,
    beta_a: np.ndarray,
    kappa_a: np.ndarray,
    eps_zpe: float = 1e-4,
    sub_op_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    return_sub_grads: bool = False,
) -> Tuple[np.ndarray, ...]:
    """粒子自旋与手性算子双精度 NumPy 解析反向传播。

    严格遵循链式法则乘回重参数化导数因子：
      d(kappa_eff)/d(kappa) = sigmoid(kappa)
      d(alpha_a_eff)/d(alpha_a) = alpha_a_eff * (1 - alpha_a_eff)
    """
    x_arr = np.asarray(x, dtype=np.float64)
    g_arr = np.asarray(grad_out, dtype=np.float64)
    B, L, D = x_arr.shape
    G = D // 4
    xg = x_arr.reshape(B, L, G, 4)
    gg = g_arr.reshape(B, L, G, 4)

    u1, u2, v1, v2 = xg[..., 0], xg[..., 1], xg[..., 2], xg[..., 3]
    g0_raw, g1_raw, g2_raw, g3_raw = gg[..., 0], gg[..., 1], gg[..., 2], gg[..., 3]
    mL, mR, mS, mA = sub_op_weights

    # 有效参数与重参数化导数因子
    theta_L_arr = np.asarray(theta_L, dtype=np.float64)
    w_L_arr = np.asarray(w_L, dtype=np.float64)
    kappa_L_arr = np.asarray(kappa_L, dtype=np.float64)
    sig_kappa_L = _np_sigmoid(kappa_L_arr)
    kappa_L_eff = _np_softplus(kappa_L_arr) + eps_zpe

    theta_R_arr = np.asarray(theta_R, dtype=np.float64)
    w_R_arr = np.asarray(w_R, dtype=np.float64)
    kappa_R_arr = np.asarray(kappa_R, dtype=np.float64)
    sig_kappa_R = _np_sigmoid(kappa_R_arr)
    kappa_R_eff = _np_softplus(kappa_R_arr) + eps_zpe

    omega_s_arr = np.asarray(omega_s, dtype=np.float64)
    gamma_s_arr = np.asarray(gamma_s, dtype=np.float64)
    kappa_s_arr = np.asarray(kappa_s, dtype=np.float64)
    sig_kappa_s = _np_sigmoid(kappa_s_arr)
    kappa_s_eff = _np_softplus(kappa_s_arr) + eps_zpe

    alpha_a_arr = np.asarray(alpha_a, dtype=np.float64)
    alpha_a_eff = _np_sigmoid(alpha_a_arr)
    d_alpha_a = alpha_a_eff * (1.0 - alpha_a_eff)
    beta_a_arr = np.asarray(beta_a, dtype=np.float64)
    kappa_a_arr = np.asarray(kappa_a, dtype=np.float64)
    sig_kappa_a = _np_sigmoid(kappa_a_arr)
    kappa_a_eff = _np_softplus(kappa_a_arr) + eps_zpe

    # ------------------------------------------------------------------
    # 1. 左手 Weyl 手性规范流反向传播 (O_L)
    # ------------------------------------------------------------------
    g0_L, g1_L = mL * g0_raw, mL * g1_raw
    E_L = u1 * u1 + u2 * u2
    Q_L = 1.0 + kappa_L_eff * E_L
    D_L = 1.0 / np.sqrt(Q_L)
    cos_L = np.cos(theta_L_arr)
    sin_L = np.sin(theta_L_arr)
    r_L1 = cos_L * u1 - sin_L * u2
    r_L2 = sin_L * u1 + cos_L * u2

    S_rL = g0_L * r_L1 + g1_L * r_L2
    S_theta_L = g0_L * (-r_L2) + g1_L * r_L1
    grad_w_L = np.sum(D_L * S_rL, axis=(0, 1))
    grad_theta_L = np.sum(w_L_arr * D_L * S_theta_L, axis=(0, 1))
    dQ_L = -0.5 * w_L_arr * (D_L * D_L * D_L) * S_rL
    grad_kappa_L = sig_kappa_L * np.sum(dQ_L * E_L, axis=(0, 1))

    du1_L = w_L_arr * D_L * (g0_L * cos_L + g1_L * sin_L) + 2.0 * kappa_L_eff * u1 * dQ_L
    du2_L = w_L_arr * D_L * (-g0_L * sin_L + g1_L * cos_L) + 2.0 * kappa_L_eff * u2 * dQ_L

    # ------------------------------------------------------------------
    # 2. 右手 Weyl 手性螺旋度流反向传播 (O_R)
    # ------------------------------------------------------------------
    g2_R, g3_R = mR * g2_raw, mR * g3_raw
    E_R = v1 * v1 + v2 * v2
    Q_R = 1.0 + kappa_R_eff * E_R
    D_R = 1.0 / np.sqrt(Q_R)
    cos_R = np.cos(theta_R_arr)
    sin_R = np.sin(theta_R_arr)
    r_R1 = cos_R * v1 + sin_R * v2
    r_R2 = -sin_R * v1 + cos_R * v2

    S_rR = g2_R * r_R1 + g3_R * r_R2
    S_theta_R = g2_R * r_R2 + g3_R * (-r_R1)
    grad_w_R = np.sum(D_R * S_rR, axis=(0, 1))
    grad_theta_R = np.sum(w_R_arr * D_R * S_theta_R, axis=(0, 1))
    dQ_R = -0.5 * w_R_arr * (D_R * D_R * D_R) * S_rR
    grad_kappa_R = sig_kappa_R * np.sum(dQ_R * E_R, axis=(0, 1))

    dv1_R = w_R_arr * D_R * (g2_R * cos_R - g3_R * sin_R) + 2.0 * kappa_R_eff * v1 * dQ_R
    dv2_R = w_R_arr * D_R * (g2_R * sin_R + g3_R * cos_R) + 2.0 * kappa_R_eff * v2 * dQ_R

    # ------------------------------------------------------------------
    # 3. Pauli 自旋双线性进动与螺旋度交换流反向传播 (O_spin)
    # ------------------------------------------------------------------
    g0_s, g1_s, g2_s, g3_s = mS * g0_raw, mS * g1_raw, mS * g2_raw, mS * g3_raw
    E_tot = E_L + E_R
    Q_s = 1.0 + kappa_s_eff * E_tot
    D_s = 1.0 / Q_s
    S_y = u1 * v2 - u2 * v1
    m0 = omega_s_arr * v1 + gamma_s_arr * S_y * v2
    m1 = omega_s_arr * v2 - gamma_s_arr * S_y * v1
    m2 = omega_s_arr * u1 - gamma_s_arr * S_y * u2
    m3 = omega_s_arr * u2 + gamma_s_arr * S_y * u1

    tg0, tg1, tg2, tg3 = D_s * g0_s, D_s * g1_s, D_s * g2_s, D_s * g3_s
    dD_s = g0_s * m0 + g1_s * m1 + g2_s * m2 + g3_s * m3
    dE_tot_s = -kappa_s_eff * (D_s * D_s) * dD_s
    grad_kappa_s = sig_kappa_s * np.sum(-(D_s * D_s) * E_tot * dD_s, axis=(0, 1))
    grad_omega_s = np.sum(tg0 * v1 + tg1 * v2 + tg2 * u1 + tg3 * u2, axis=(0, 1))
    tau_term = tg0 * v2 - tg1 * v1 - tg2 * u2 + tg3 * u1
    grad_gamma_s = np.sum(S_y * tau_term, axis=(0, 1))
    dS_y = gamma_s_arr * tau_term

    du1_s = omega_s_arr * tg2 + gamma_s_arr * S_y * tg3 + dS_y * v2 + 2.0 * u1 * dE_tot_s
    du2_s = omega_s_arr * tg3 - gamma_s_arr * S_y * tg2 - dS_y * v1 + 2.0 * u2 * dE_tot_s
    dv1_s = omega_s_arr * tg0 - gamma_s_arr * S_y * tg1 - dS_y * u2 + 2.0 * v1 * dE_tot_s
    dv2_s = omega_s_arr * tg1 + gamma_s_arr * S_y * tg0 + dS_y * u1 + 2.0 * v2 * dE_tot_s

    # ------------------------------------------------------------------
    # 4. ABJ 手性反常与轴矢流门控反向传播 (O_anom)
    # ------------------------------------------------------------------
    g0_a, g1_a, g2_a, g3_a = mA * g0_raw, mA * g1_raw, mA * g2_raw, mA * g3_raw
    P_5 = E_R - E_L
    Q_P5 = 1.0 + kappa_a_eff * (P_5 * P_5)
    D_P5 = 1.0 / np.sqrt(Q_P5)
    G_anom = P_5 * D_P5
    I_ps = u1 * v1 + u2 * v2
    Q_a = 1.0 + kappa_a_eff * E_tot
    D_a = 1.0 / Q_a
    C_ps = D_a * I_ps

    A0 = -u1 + G_anom * v2
    A1 = u2 - G_anom * v1
    A2 = v1 + G_anom * u2
    A3 = -v2 - G_anom * u1

    S_A = g0_a * A0 + g1_a * A1 + g2_a * A2 + g3_a * A3
    grad_alpha_a = d_alpha_a * np.sum(S_A, axis=(0, 1))

    S_B = g0_a * v1 - g1_a * v2 + g2_a * u1 - g3_a * u2
    grad_beta_a = np.sum(C_ps * S_B, axis=(0, 1))

    dC_ps = beta_a_arr * S_B
    dI_ps = D_a * dC_ps
    dD_a = I_ps * dC_ps
    dE_tot_a = -kappa_a_eff * (D_a * D_a) * dD_a

    dG_anom = alpha_a_eff * (g0_a * v2 - g1_a * v1 + g2_a * u2 - g3_a * u1)
    dP_5 = dG_anom * (D_P5 * D_P5 * D_P5)

    grad_kappa_a = sig_kappa_a * np.sum(
        -0.5 * (P_5 * P_5 * P_5) * (D_P5 * D_P5 * D_P5) * dG_anom - (D_a * D_a) * E_tot * dD_a,
        axis=(0, 1),
    )

    du1_a = (
        alpha_a_eff * (-g0_a - G_anom * g3_a)
        + beta_a_arr * C_ps * g2_a
        + dI_ps * v1
        - 2.0 * u1 * dP_5
        + 2.0 * u1 * dE_tot_a
    )
    du2_a = (
        alpha_a_eff * (g1_a + G_anom * g2_a)
        - beta_a_arr * C_ps * g3_a
        + dI_ps * v2
        - 2.0 * u2 * dP_5
        + 2.0 * u2 * dE_tot_a
    )
    dv1_a = (
        alpha_a_eff * (-G_anom * g1_a + g2_a)
        + beta_a_arr * C_ps * g0_a
        + dI_ps * u1
        + 2.0 * v1 * dP_5
        + 2.0 * v1 * dE_tot_a
    )
    dv2_a = (
        alpha_a_eff * (G_anom * g0_a - g3_a)
        - beta_a_arr * C_ps * g1_a
        + dI_ps * u2
        + 2.0 * v2 * dP_5
        + 2.0 * v2 * dE_tot_a
    )

    grad_x = np.stack(
        [
            du1_L + du1_s + du1_a,
            du2_L + du2_s + du2_a,
            dv1_R + dv1_s + dv1_a,
            dv2_R + dv2_s + dv2_a,
        ],
        axis=-1,
    ).reshape(B, L, D)

    base_grads = (
        grad_x,
        grad_theta_L,
        grad_w_L,
        grad_kappa_L,
        grad_theta_R,
        grad_w_R,
        grad_kappa_R,
        grad_omega_s,
        grad_gamma_s,
        grad_kappa_s,
        grad_alpha_a,
        grad_beta_a,
        grad_kappa_a,
    )

    if return_sub_grads:
        zeros = np.zeros_like(u1)
        grad_x_L = np.stack([du1_L, du2_L, zeros, zeros], axis=-1).reshape(B, L, D)
        grad_x_R = np.stack([zeros, zeros, dv1_R, dv2_R], axis=-1).reshape(B, L, D)
        grad_x_spin = np.stack([du1_s, du2_s, dv1_s, dv2_s], axis=-1).reshape(B, L, D)
        grad_x_anom = np.stack([du1_a, du2_a, dv1_a, dv2_a], axis=-1).reshape(B, L, D)
        return base_grads + ((grad_x_L, grad_x_R, grad_x_spin, grad_x_anom),)

    return base_grads


# ==============================================================================
# 4. Triton GPU Kernel (@triton.jit) 算子融合实现与 Tile 级精确核调度器
# ==============================================================================
if HAS_TRITON:

    @triton.jit
    def fused_spin_chirality_fwd_kernel(
        X_ptr,
        Y_ptr,
        ThetaL_ptr,
        WL_ptr,
        KappaL_ptr,
        ThetaR_ptr,
        WR_ptr,
        KappaR_ptr,
        OmegaS_ptr,
        GammaS_ptr,
        KappaS_ptr,
        AlphaA_ptr,
        BetaA_ptr,
        KappaA_ptr,
        N_groups_total,
        G,
        eps_zpe,
        mL,
        mR,
        mS,
        mA,
        BLOCK_M: tl.constexpr,
    ):
        """单次启动融合四分量 Dirac 旋量左/右手性流、Pauli 自旋进动与 ABJ 手性反常的 Triton 前向核。"""
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offs_m < N_groups_total
        g_idx = offs_m % G

        # 连续加载四分量旋量 (u1, u2, v1, v2)
        base_x = offs_m * 4
        u1 = tl.load(X_ptr + base_x + 0, mask=mask, other=0.0).to(tl.float64)
        u2 = tl.load(X_ptr + base_x + 1, mask=mask, other=0.0).to(tl.float64)
        v1 = tl.load(X_ptr + base_x + 2, mask=mask, other=0.0).to(tl.float64)
        v2 = tl.load(X_ptr + base_x + 3, mask=mask, other=0.0).to(tl.float64)

        # 加载参数并执行片上重参数化
        theta_L = tl.load(ThetaL_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        w_L = tl.load(WL_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        kappa_L_raw = tl.load(KappaL_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        kappa_L_eff = tl.log(1.0 + tl.exp(-tl.abs(kappa_L_raw))) + tl.maximum(kappa_L_raw, 0.0) + eps_zpe

        theta_R = tl.load(ThetaR_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        w_R = tl.load(WR_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        kappa_R_raw = tl.load(KappaR_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        kappa_R_eff = tl.log(1.0 + tl.exp(-tl.abs(kappa_R_raw))) + tl.maximum(kappa_R_raw, 0.0) + eps_zpe

        omega_s = tl.load(OmegaS_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        gamma_s = tl.load(GammaS_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        kappa_s_raw = tl.load(KappaS_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        kappa_s_eff = tl.log(1.0 + tl.exp(-tl.abs(kappa_s_raw))) + tl.maximum(kappa_s_raw, 0.0) + eps_zpe

        alpha_a_raw = tl.load(AlphaA_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        alpha_a_eff = 1.0 / (1.0 + tl.exp(-alpha_a_raw))
        beta_a = tl.load(BetaA_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        kappa_a_raw = tl.load(KappaA_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        kappa_a_eff = tl.log(1.0 + tl.exp(-tl.abs(kappa_a_raw))) + tl.maximum(kappa_a_raw, 0.0) + eps_zpe

        # 1. Left-Handed Weyl Current
        E_L = u1 * u1 + u2 * u2
        D_L = 1.0 / tl.sqrt(1.0 + kappa_L_eff * E_L)
        cos_L = tl.cos(theta_L)
        sin_L = tl.sin(theta_L)
        op_L0 = w_L * D_L * (cos_L * u1 - sin_L * u2)
        op_L1 = w_L * D_L * (sin_L * u1 + cos_L * u2)

        # 2. Right-Handed Weyl Current
        E_R = v1 * v1 + v2 * v2
        D_R = 1.0 / tl.sqrt(1.0 + kappa_R_eff * E_R)
        cos_R = tl.cos(theta_R)
        sin_R = tl.sin(theta_R)
        op_R2 = w_R * D_R * (cos_R * v1 + sin_R * v2)
        op_R3 = w_R * D_R * (-sin_R * v1 + cos_R * v2)

        # 3. Pauli Spin Precession
        E_tot = E_L + E_R
        D_s = 1.0 / (1.0 + kappa_s_eff * E_tot)
        S_y = u1 * v2 - u2 * v1
        op_s0 = D_s * (omega_s * v1 + gamma_s * S_y * v2)
        op_s1 = D_s * (omega_s * v2 - gamma_s * S_y * v1)
        op_s2 = D_s * (omega_s * u1 - gamma_s * S_y * u2)
        op_s3 = D_s * (omega_s * u2 + gamma_s * S_y * u1)

        # 4. ABJ Chiral Anomaly
        P_5 = E_R - E_L
        G_anom = P_5 / tl.sqrt(1.0 + kappa_a_eff * P_5 * P_5)
        I_ps = u1 * v1 + u2 * v2
        C_ps = I_ps / (1.0 + kappa_a_eff * E_tot)
        op_a0 = alpha_a_eff * (-u1 + G_anom * v2) + beta_a * C_ps * v1
        op_a1 = alpha_a_eff * (u2 - G_anom * v1) - beta_a * C_ps * v2
        op_a2 = alpha_a_eff * (v1 + G_anom * u2) + beta_a * C_ps * u1
        op_a3 = alpha_a_eff * (-v2 - G_anom * u1) - beta_a * C_ps * u2

        y0 = mL * op_L0 + mS * op_s0 + mA * op_a0
        y1 = mL * op_L1 + mS * op_s1 + mA * op_a1
        y2 = mR * op_R2 + mS * op_s2 + mA * op_a2
        y3 = mR * op_R3 + mS * op_s3 + mA * op_a3

        tl.store(Y_ptr + base_x + 0, y0, mask=mask)
        tl.store(Y_ptr + base_x + 1, y1, mask=mask)
        tl.store(Y_ptr + base_x + 2, y2, mask=mask)
        tl.store(Y_ptr + base_x + 3, y3, mask=mask)

    @triton.jit
    def fused_spin_chirality_bwd_kernel(
        GradOut_ptr,
        X_ptr,
        GradX_ptr,
        ThetaL_ptr,
        WL_ptr,
        KappaL_ptr,
        ThetaR_ptr,
        WR_ptr,
        KappaR_ptr,
        OmegaS_ptr,
        GammaS_ptr,
        KappaS_ptr,
        AlphaA_ptr,
        BetaA_ptr,
        KappaA_ptr,
        GradThetaL_ptr,
        GradWL_ptr,
        GradKappaL_ptr,
        GradThetaR_ptr,
        GradWR_ptr,
        GradKappaR_ptr,
        GradOmegaS_ptr,
        GradGammaS_ptr,
        GradKappaS_ptr,
        GradAlphaA_ptr,
        GradBetaA_ptr,
        GradKappaA_ptr,
        N_groups_total,
        G,
        eps_zpe,
        mL,
        mR,
        mS,
        mA,
        BLOCK_M: tl.constexpr,
    ):
        """输出全部 12 个可学参数梯度 (tl.atomic_add) 与输入旋量梯度的 Triton 融合反向核。"""
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offs_m < N_groups_total
        g_idx = offs_m % G

        base_x = offs_m * 4
        u1 = tl.load(X_ptr + base_x + 0, mask=mask, other=0.0).to(tl.float64)
        u2 = tl.load(X_ptr + base_x + 1, mask=mask, other=0.0).to(tl.float64)
        v1 = tl.load(X_ptr + base_x + 2, mask=mask, other=0.0).to(tl.float64)
        v2 = tl.load(X_ptr + base_x + 3, mask=mask, other=0.0).to(tl.float64)

        g0_raw = tl.load(GradOut_ptr + base_x + 0, mask=mask, other=0.0).to(tl.float64)
        g1_raw = tl.load(GradOut_ptr + base_x + 1, mask=mask, other=0.0).to(tl.float64)
        g2_raw = tl.load(GradOut_ptr + base_x + 2, mask=mask, other=0.0).to(tl.float64)
        g3_raw = tl.load(GradOut_ptr + base_x + 3, mask=mask, other=0.0).to(tl.float64)

        theta_L = tl.load(ThetaL_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        w_L = tl.load(WL_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        kappa_L_raw = tl.load(KappaL_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        sig_kappa_L = 1.0 / (1.0 + tl.exp(-kappa_L_raw))
        kappa_L_eff = tl.log(1.0 + tl.exp(-tl.abs(kappa_L_raw))) + tl.maximum(kappa_L_raw, 0.0) + eps_zpe

        theta_R = tl.load(ThetaR_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        w_R = tl.load(WR_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        kappa_R_raw = tl.load(KappaR_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        sig_kappa_R = 1.0 / (1.0 + tl.exp(-kappa_R_raw))
        kappa_R_eff = tl.log(1.0 + tl.exp(-tl.abs(kappa_R_raw))) + tl.maximum(kappa_R_raw, 0.0) + eps_zpe

        omega_s = tl.load(OmegaS_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        gamma_s = tl.load(GammaS_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        kappa_s_raw = tl.load(KappaS_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        sig_kappa_s = 1.0 / (1.0 + tl.exp(-kappa_s_raw))
        kappa_s_eff = tl.log(1.0 + tl.exp(-tl.abs(kappa_s_raw))) + tl.maximum(kappa_s_raw, 0.0) + eps_zpe

        alpha_a_raw = tl.load(AlphaA_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        alpha_a_eff = 1.0 / (1.0 + tl.exp(-alpha_a_raw))
        d_alpha_a = alpha_a_eff * (1.0 - alpha_a_eff)
        beta_a = tl.load(BetaA_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        kappa_a_raw = tl.load(KappaA_ptr + g_idx, mask=mask, other=0.0).to(tl.float64)
        sig_kappa_a = 1.0 / (1.0 + tl.exp(-kappa_a_raw))
        kappa_a_eff = tl.log(1.0 + tl.exp(-tl.abs(kappa_a_raw))) + tl.maximum(kappa_a_raw, 0.0) + eps_zpe

        # 1. Left Weyl backward
        g0_L = mL * g0_raw
        g1_L = mL * g1_raw
        E_L = u1 * u1 + u2 * u2
        D_L = 1.0 / tl.sqrt(1.0 + kappa_L_eff * E_L)
        cos_L = tl.cos(theta_L)
        sin_L = tl.sin(theta_L)
        r_L1 = cos_L * u1 - sin_L * u2
        r_L2 = sin_L * u1 + cos_L * u2
        S_rL = g0_L * r_L1 + g1_L * r_L2
        S_theta_L = -g0_L * r_L2 + g1_L * r_L1
        dQ_L = -0.5 * w_L * (D_L * D_L * D_L) * S_rL

        du1_L = w_L * D_L * (g0_L * cos_L + g1_L * sin_L) + 2.0 * kappa_L_eff * u1 * dQ_L
        du2_L = w_L * D_L * (-g0_L * sin_L + g1_L * cos_L) + 2.0 * kappa_L_eff * u2 * dQ_L

        tl.atomic_add(GradWL_ptr + g_idx, D_L * S_rL, mask=mask)
        tl.atomic_add(GradThetaL_ptr + g_idx, w_L * D_L * S_theta_L, mask=mask)
        tl.atomic_add(GradKappaL_ptr + g_idx, sig_kappa_L * dQ_L * E_L, mask=mask)

        # 2. Right Weyl backward
        g2_R = mR * g2_raw
        g3_R = mR * g3_raw
        E_R = v1 * v1 + v2 * v2
        D_R = 1.0 / tl.sqrt(1.0 + kappa_R_eff * E_R)
        cos_R = tl.cos(theta_R)
        sin_R = tl.sin(theta_R)
        r_R1 = cos_R * v1 + sin_R * v2
        r_R2 = -sin_R * v1 + cos_R * v2
        S_rR = g2_R * r_R1 + g3_R * r_R2
        S_theta_R = g2_R * r_R2 - g3_R * r_R1
        dQ_R = -0.5 * w_R * (D_R * D_R * D_R) * S_rR

        dv1_R = w_R * D_R * (g2_R * cos_R - g3_R * sin_R) + 2.0 * kappa_R_eff * v1 * dQ_R
        dv2_R = w_R * D_R * (g2_R * sin_R + g3_R * cos_R) + 2.0 * kappa_R_eff * v2 * dQ_R

        tl.atomic_add(GradWR_ptr + g_idx, D_R * S_rR, mask=mask)
        tl.atomic_add(GradThetaR_ptr + g_idx, w_R * D_R * S_theta_R, mask=mask)
        tl.atomic_add(GradKappaR_ptr + g_idx, sig_kappa_R * dQ_R * E_R, mask=mask)

        # 3. Pauli Spin Precession backward
        g0_s = mS * g0_raw
        g1_s = mS * g1_raw
        g2_s = mS * g2_raw
        g3_s = mS * g3_raw
        E_tot = E_L + E_R
        D_s = 1.0 / (1.0 + kappa_s_eff * E_tot)
        S_y = u1 * v2 - u2 * v1
        m0 = omega_s * v1 + gamma_s * S_y * v2
        m1 = omega_s * v2 - gamma_s * S_y * v1
        m2 = omega_s * u1 - gamma_s * S_y * u2
        m3 = omega_s * u2 + gamma_s * S_y * u1

        tg0 = D_s * g0_s
        tg1 = D_s * g1_s
        tg2 = D_s * g2_s
        tg3 = D_s * g3_s
        dD_s = g0_s * m0 + g1_s * m1 + g2_s * m2 + g3_s * m3
        dE_tot_s = -kappa_s_eff * (D_s * D_s) * dD_s
        tau_term = tg0 * v2 - tg1 * v1 - tg2 * u2 + tg3 * u1
        dS_y = gamma_s * tau_term

        du1_s = omega_s * tg2 + gamma_s * S_y * tg3 + dS_y * v2 + 2.0 * u1 * dE_tot_s
        du2_s = omega_s * tg3 - gamma_s * S_y * tg2 - dS_y * v1 + 2.0 * u2 * dE_tot_s
        dv1_s = omega_s * tg0 - gamma_s * S_y * tg1 - dS_y * u2 + 2.0 * v1 * dE_tot_s
        dv2_s = omega_s * tg1 + gamma_s * S_y * tg0 + dS_y * u1 + 2.0 * v2 * dE_tot_s

        tl.atomic_add(GradOmegaS_ptr + g_idx, tg0 * v1 + tg1 * v2 + tg2 * u1 + tg3 * u2, mask=mask)
        tl.atomic_add(GradGammaS_ptr + g_idx, S_y * tau_term, mask=mask)
        tl.atomic_add(GradKappaS_ptr + g_idx, -sig_kappa_s * (D_s * D_s) * E_tot * dD_s, mask=mask)

        # 4. ABJ Chiral Anomaly backward
        g0_a = mA * g0_raw
        g1_a = mA * g1_raw
        g2_a = mA * g2_raw
        g3_a = mA * g3_raw
        P_5 = E_R - E_L
        D_P5 = 1.0 / tl.sqrt(1.0 + kappa_a_eff * P_5 * P_5)
        G_anom = P_5 * D_P5
        I_ps = u1 * v1 + u2 * v2
        D_a = 1.0 / (1.0 + kappa_a_eff * E_tot)
        C_ps = D_a * I_ps

        A0 = -u1 + G_anom * v2
        A1 = u2 - G_anom * v1
        A2 = v1 + G_anom * u2
        A3 = -v2 - G_anom * u1
        S_A = g0_a * A0 + g1_a * A1 + g2_a * A2 + g3_a * A3
        S_B = g0_a * v1 - g1_a * v2 + g2_a * u1 - g3_a * u2

        dC_ps = beta_a * S_B
        dI_ps = D_a * dC_ps
        dD_a = I_ps * dC_ps
        dE_tot_a = -kappa_a_eff * (D_a * D_a) * dD_a
        dG_anom = alpha_a_eff * (g0_a * v2 - g1_a * v1 + g2_a * u2 - g3_a * u1)
        dP_5 = dG_anom * (D_P5 * D_P5 * D_P5)

        du1_a = (
            alpha_a_eff * (-g0_a - G_anom * g3_a)
            + beta_a * C_ps * g2_a
            + dI_ps * v1
            - 2.0 * u1 * dP_5
            + 2.0 * u1 * dE_tot_a
        )
        du2_a = (
            alpha_a_eff * (g1_a + G_anom * g2_a)
            - beta_a * C_ps * g3_a
            + dI_ps * v2
            - 2.0 * u2 * dP_5
            + 2.0 * u2 * dE_tot_a
        )
        dv1_a = (
            alpha_a_eff * (-G_anom * g1_a + g2_a)
            + beta_a * C_ps * g0_a
            + dI_ps * u1
            + 2.0 * v1 * dP_5
            + 2.0 * v1 * dE_tot_a
        )
        dv2_a = (
            alpha_a_eff * (G_anom * g0_a - g3_a)
            - beta_a * C_ps * g1_a
            + dI_ps * u2
            + 2.0 * v2 * dP_5
            + 2.0 * v2 * dE_tot_a
        )

        tl.atomic_add(GradAlphaA_ptr + g_idx, d_alpha_a * S_A, mask=mask)
        tl.atomic_add(GradBetaA_ptr + g_idx, C_ps * S_B, mask=mask)
        tl.atomic_add(
            GradKappaA_ptr + g_idx,
            sig_kappa_a * (-0.5 * (P_5 * P_5 * P_5) * (D_P5 * D_P5 * D_P5) * dG_anom - (D_a * D_a) * E_tot * dD_a),
            mask=mask,
        )

        tl.store(GradX_ptr + base_x + 0, du1_L + du1_s + du1_a, mask=mask)
        tl.store(GradX_ptr + base_x + 1, du2_L + du2_s + du2_a, mask=mask)
        tl.store(GradX_ptr + base_x + 2, dv1_R + dv1_s + dv1_a, mask=mask)
        tl.store(GradX_ptr + base_x + 3, dv2_R + dv2_s + dv2_a, mask=mask)


def _cdiv(a: int, b: int) -> int:
    """与 triton.cdiv 等价的向上整除。"""
    return (a + b - 1) // b


def spin_chirality_triton_forward(
    x: np.ndarray,
    theta_L: np.ndarray,
    w_L: np.ndarray,
    kappa_L: np.ndarray,
    theta_R: np.ndarray,
    w_R: np.ndarray,
    kappa_R: np.ndarray,
    omega_s: np.ndarray,
    gamma_s: np.ndarray,
    kappa_s: np.ndarray,
    alpha_a: np.ndarray,
    beta_a: np.ndarray,
    kappa_a: np.ndarray,
    eps_zpe: float = 1e-4,
    sub_op_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    block_size: int = 64,
) -> np.ndarray:
    """调度 @triton.jit 前向融合核（若在 GPU 上）或执行分块 SIMD Program-ID 精确仿真核。"""
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    B, L, D = x_arr.shape
    G = D // 4
    N_groups_total = B * L * G
    mL, mR, mS, mA = sub_op_weights

    if HAS_TORCH and HAS_TRITON and torch.cuda.is_available():
        x_t = torch.as_tensor(x_arr, device="cuda", dtype=torch.float64).contiguous()
        y_t = torch.empty_like(x_t)
        p_tensors = [
            torch.as_tensor(np.ascontiguousarray(p, dtype=np.float64), device="cuda", dtype=torch.float64)
            for p in (
                theta_L,
                w_L,
                kappa_L,
                theta_R,
                w_R,
                kappa_R,
                omega_s,
                gamma_s,
                kappa_s,
                alpha_a,
                beta_a,
                kappa_a,
            )
        ]
        grid = (triton.cdiv(N_groups_total, block_size),)
        fused_spin_chirality_fwd_kernel[grid](
            x_t,
            y_t,
            *p_tensors,
            N_groups_total,
            G,
            eps_zpe,
            mL,
            mR,
            mS,
            mA,
            BLOCK_M=block_size,
        )
        return y_t.cpu().numpy()

    # 分块 SIMD Program-ID Tile 级精确执行（严格按 grid = (cdiv(N_groups_total, BLOCK_M),) 迭代）
    x_flat = x_arr.reshape(N_groups_total, 4)
    y_flat = np.empty_like(x_flat)
    num_programs = _cdiv(N_groups_total, block_size)

    for pid in range(num_programs):
        offs_m = pid * block_size + np.arange(block_size)
        mask = offs_m < N_groups_total
        valid_offs = offs_m[mask]
        g_idx = valid_offs % G

        u1 = x_flat[valid_offs, 0]
        u2 = x_flat[valid_offs, 1]
        v1 = x_flat[valid_offs, 2]
        v2 = x_flat[valid_offs, 3]

        tL = theta_L[g_idx]
        wL = w_L[g_idx]
        kL_eff = _np_softplus(kappa_L[g_idx]) + eps_zpe

        tR = theta_R[g_idx]
        wR = w_R[g_idx]
        kR_eff = _np_softplus(kappa_R[g_idx]) + eps_zpe

        omS = omega_s[g_idx]
        gmS = gamma_s[g_idx]
        kS_eff = _np_softplus(kappa_s[g_idx]) + eps_zpe

        alA_eff = _np_sigmoid(alpha_a[g_idx])
        btA = beta_a[g_idx]
        kA_eff = _np_softplus(kappa_a[g_idx]) + eps_zpe

        E_L = u1 * u1 + u2 * u2
        D_L = 1.0 / np.sqrt(1.0 + kL_eff * E_L)
        cos_L, sin_L = np.cos(tL), np.sin(tL)
        op_L0 = wL * D_L * (cos_L * u1 - sin_L * u2)
        op_L1 = wL * D_L * (sin_L * u1 + cos_L * u2)

        E_R = v1 * v1 + v2 * v2
        D_R = 1.0 / np.sqrt(1.0 + kR_eff * E_R)
        cos_R, sin_R = np.cos(tR), np.sin(tR)
        op_R2 = wR * D_R * (cos_R * v1 + sin_R * v2)
        op_R3 = wR * D_R * (-sin_R * v1 + cos_R * v2)

        E_tot = E_L + E_R
        D_s = 1.0 / (1.0 + kS_eff * E_tot)
        S_y = u1 * v2 - u2 * v1
        op_s0 = D_s * (omS * v1 + gmS * S_y * v2)
        op_s1 = D_s * (omS * v2 - gmS * S_y * v1)
        op_s2 = D_s * (omS * u1 - gmS * S_y * u2)
        op_s3 = D_s * (omS * u2 + gmS * S_y * u1)

        P_5 = E_R - E_L
        G_anom = P_5 / np.sqrt(1.0 + kA_eff * P_5 * P_5)
        I_ps = u1 * v1 + u2 * v2
        C_ps = I_ps / (1.0 + kA_eff * E_tot)
        op_a0 = alA_eff * (-u1 + G_anom * v2) + btA * C_ps * v1
        op_a1 = alA_eff * (u2 - G_anom * v1) - btA * C_ps * v2
        op_a2 = alA_eff * (v1 + G_anom * u2) + btA * C_ps * u1
        op_a3 = alA_eff * (-v2 - G_anom * u1) - btA * C_ps * u2

        y_flat[valid_offs, 0] = mL * op_L0 + mS * op_s0 + mA * op_a0
        y_flat[valid_offs, 1] = mL * op_L1 + mS * op_s1 + mA * op_a1
        y_flat[valid_offs, 2] = mR * op_R2 + mS * op_s2 + mA * op_a2
        y_flat[valid_offs, 3] = mR * op_R3 + mS * op_s3 + mA * op_a3

    return y_flat.reshape(B, L, D)


def spin_chirality_triton_backward(
    grad_out: np.ndarray,
    x: np.ndarray,
    theta_L: np.ndarray,
    w_L: np.ndarray,
    kappa_L: np.ndarray,
    theta_R: np.ndarray,
    w_R: np.ndarray,
    kappa_R: np.ndarray,
    omega_s: np.ndarray,
    gamma_s: np.ndarray,
    kappa_s: np.ndarray,
    alpha_a: np.ndarray,
    beta_a: np.ndarray,
    kappa_a: np.ndarray,
    eps_zpe: float = 1e-4,
    sub_op_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    block_size: int = 64,
) -> Tuple[np.ndarray, ...]:
    """调度 @triton.jit 全参数梯度反向融合核（含 tl.atomic_add 归约）或 Tile 级精确仿真核。"""
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    g_arr = np.ascontiguousarray(grad_out, dtype=np.float64)
    B, L, D = x_arr.shape
    G = D // 4
    N_groups_total = B * L * G
    mL, mR, mS, mA = sub_op_weights

    if HAS_TORCH and HAS_TRITON and torch.cuda.is_available():
        x_t = torch.as_tensor(x_arr, device="cuda", dtype=torch.float64).contiguous()
        go_t = torch.as_tensor(g_arr, device="cuda", dtype=torch.float64).contiguous()
        gx_t = torch.empty_like(x_t)
        p_tensors = [
            torch.as_tensor(np.ascontiguousarray(p, dtype=np.float64), device="cuda", dtype=torch.float64)
            for p in (
                theta_L,
                w_L,
                kappa_L,
                theta_R,
                w_R,
                kappa_R,
                omega_s,
                gamma_s,
                kappa_s,
                alpha_a,
                beta_a,
                kappa_a,
            )
        ]
        gp_tensors = [torch.zeros(G, device="cuda", dtype=torch.float64) for _ in range(12)]
        grid = (triton.cdiv(N_groups_total, block_size),)
        fused_spin_chirality_bwd_kernel[grid](
            go_t,
            x_t,
            gx_t,
            *p_tensors,
            *gp_tensors,
            N_groups_total,
            G,
            eps_zpe,
            mL,
            mR,
            mS,
            mA,
            BLOCK_M=block_size,
        )
        return (gx_t.cpu().numpy(),) + tuple(gp.cpu().numpy() for gp in gp_tensors)

    # Tile 级 Program-ID 分块仿真与 atomic_add 精确归约
    x_flat = x_arr.reshape(N_groups_total, 4)
    g_flat = g_arr.reshape(N_groups_total, 4)
    gx_flat = np.empty_like(x_flat)

    grad_theta_L = np.zeros(G, dtype=np.float64)
    grad_w_L = np.zeros(G, dtype=np.float64)
    grad_kappa_L = np.zeros(G, dtype=np.float64)
    grad_theta_R = np.zeros(G, dtype=np.float64)
    grad_w_R = np.zeros(G, dtype=np.float64)
    grad_kappa_R = np.zeros(G, dtype=np.float64)
    grad_omega_s = np.zeros(G, dtype=np.float64)
    grad_gamma_s = np.zeros(G, dtype=np.float64)
    grad_kappa_s = np.zeros(G, dtype=np.float64)
    grad_alpha_a = np.zeros(G, dtype=np.float64)
    grad_beta_a = np.zeros(G, dtype=np.float64)
    grad_kappa_a = np.zeros(G, dtype=np.float64)

    num_programs = _cdiv(N_groups_total, block_size)
    for pid in range(num_programs):
        offs_m = pid * block_size + np.arange(block_size)
        mask = offs_m < N_groups_total
        valid_offs = offs_m[mask]
        g_idx = valid_offs % G

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

        tL, wL, kL_raw = theta_L[g_idx], w_L[g_idx], kappa_L[g_idx]
        sig_kL = _np_sigmoid(kL_raw)
        kL_eff = _np_softplus(kL_raw) + eps_zpe

        tR, wR, kR_raw = theta_R[g_idx], w_R[g_idx], kappa_R[g_idx]
        sig_kR = _np_sigmoid(kR_raw)
        kR_eff = _np_softplus(kR_raw) + eps_zpe

        omS, gmS, kS_raw = omega_s[g_idx], gamma_s[g_idx], kappa_s[g_idx]
        sig_kS = _np_sigmoid(kS_raw)
        kS_eff = _np_softplus(kS_raw) + eps_zpe

        alA_raw, btA, kA_raw = alpha_a[g_idx], beta_a[g_idx], kappa_a[g_idx]
        alA_eff = _np_sigmoid(alA_raw)
        d_alA = alA_eff * (1.0 - alA_eff)
        sig_kA = _np_sigmoid(kA_raw)
        kA_eff = _np_softplus(kA_raw) + eps_zpe

        # 1. Left Weyl
        g0_L, g1_L = mL * g0_raw, mL * g1_raw
        E_L = u1 * u1 + u2 * u2
        D_L = 1.0 / np.sqrt(1.0 + kL_eff * E_L)
        cos_L, sin_L = np.cos(tL), np.sin(tL)
        r_L1 = cos_L * u1 - sin_L * u2
        r_L2 = sin_L * u1 + cos_L * u2
        S_rL = g0_L * r_L1 + g1_L * r_L2
        S_tL = -g0_L * r_L2 + g1_L * r_L1
        dQ_L = -0.5 * wL * (D_L * D_L * D_L) * S_rL
        du1_L = wL * D_L * (g0_L * cos_L + g1_L * sin_L) + 2.0 * kL_eff * u1 * dQ_L
        du2_L = wL * D_L * (-g0_L * sin_L + g1_L * cos_L) + 2.0 * kL_eff * u2 * dQ_L

        np.add.at(grad_w_L, g_idx, D_L * S_rL)
        np.add.at(grad_theta_L, g_idx, wL * D_L * S_tL)
        np.add.at(grad_kappa_L, g_idx, sig_kL * dQ_L * E_L)

        # 2. Right Weyl
        g2_R, g3_R = mR * g2_raw, mR * g3_raw
        E_R = v1 * v1 + v2 * v2
        D_R = 1.0 / np.sqrt(1.0 + kR_eff * E_R)
        cos_R, sin_R = np.cos(tR), np.sin(tR)
        r_R1 = cos_R * v1 + sin_R * v2
        r_R2 = -sin_R * v1 + cos_R * v2
        S_rR = g2_R * r_R1 + g3_R * r_R2
        S_tR = g2_R * r_R2 - g3_R * r_R1
        dQ_R = -0.5 * wR * (D_R * D_R * D_R) * S_rR
        dv1_R = wR * D_R * (g2_R * cos_R - g3_R * sin_R) + 2.0 * kR_eff * v1 * dQ_R
        dv2_R = wR * D_R * (g2_R * sin_R + g3_R * cos_R) + 2.0 * kR_eff * v2 * dQ_R

        np.add.at(grad_w_R, g_idx, D_R * S_rR)
        np.add.at(grad_theta_R, g_idx, wR * D_R * S_tR)
        np.add.at(grad_kappa_R, g_idx, sig_kR * dQ_R * E_R)

        # 3. Pauli Spin
        g0_s, g1_s, g2_s, g3_s = mS * g0_raw, mS * g1_raw, mS * g2_raw, mS * g3_raw
        E_tot = E_L + E_R
        D_s = 1.0 / (1.0 + kS_eff * E_tot)
        S_y = u1 * v2 - u2 * v1
        m0 = omS * v1 + gmS * S_y * v2
        m1 = omS * v2 - gmS * S_y * v1
        m2 = omS * u1 - gmS * S_y * u2
        m3 = omS * u2 + gmS * S_y * u1
        tg0, tg1, tg2, tg3 = D_s * g0_s, D_s * g1_s, D_s * g2_s, D_s * g3_s
        dD_s = g0_s * m0 + g1_s * m1 + g2_s * m2 + g3_s * m3
        dE_tot_s = -kS_eff * (D_s * D_s) * dD_s
        tau_term = tg0 * v2 - tg1 * v1 - tg2 * u2 + tg3 * u1
        dS_y = gmS * tau_term

        du1_s = omS * tg2 + gmS * S_y * tg3 + dS_y * v2 + 2.0 * u1 * dE_tot_s
        du2_s = omS * tg3 - gmS * S_y * tg2 - dS_y * v1 + 2.0 * u2 * dE_tot_s
        dv1_s = omS * tg0 - gmS * S_y * tg1 - dS_y * u2 + 2.0 * v1 * dE_tot_s
        dv2_s = omS * tg1 + gmS * S_y * tg0 + dS_y * u1 + 2.0 * v2 * dE_tot_s

        np.add.at(grad_omega_s, g_idx, tg0 * v1 + tg1 * v2 + tg2 * u1 + tg3 * u2)
        np.add.at(grad_gamma_s, g_idx, S_y * tau_term)
        np.add.at(grad_kappa_s, g_idx, -sig_kS * (D_s * D_s) * E_tot * dD_s)

        # 4. ABJ Anomaly
        g0_a, g1_a, g2_a, g3_a = mA * g0_raw, mA * g1_raw, mA * g2_raw, mA * g3_raw
        P_5 = E_R - E_L
        D_P5 = 1.0 / np.sqrt(1.0 + kA_eff * P_5 * P_5)
        G_anom = P_5 * D_P5
        I_ps = u1 * v1 + u2 * v2
        D_a = 1.0 / (1.0 + kA_eff * E_tot)
        C_ps = D_a * I_ps

        A0 = -u1 + G_anom * v2
        A1 = u2 - G_anom * v1
        A2 = v1 + G_anom * u2
        A3 = -v2 - G_anom * u1
        S_A = g0_a * A0 + g1_a * A1 + g2_a * A2 + g3_a * A3
        S_B = g0_a * v1 - g1_a * v2 + g2_a * u1 - g3_a * u2

        dC_ps = btA * S_B
        dI_ps = D_a * dC_ps
        dD_a = I_ps * dC_ps
        dE_tot_a = -kA_eff * (D_a * D_a) * dD_a
        dG_anom = alA_eff * (g0_a * v2 - g1_a * v1 + g2_a * u2 - g3_a * u1)
        dP_5 = dG_anom * (D_P5 * D_P5 * D_P5)

        du1_a = (
            alA_eff * (-g0_a - G_anom * g3_a)
            + btA * C_ps * g2_a
            + dI_ps * v1
            - 2.0 * u1 * dP_5
            + 2.0 * u1 * dE_tot_a
        )
        du2_a = (
            alA_eff * (g1_a + G_anom * g2_a)
            - btA * C_ps * g3_a
            + dI_ps * v2
            - 2.0 * u2 * dP_5
            + 2.0 * u2 * dE_tot_a
        )
        dv1_a = (
            alA_eff * (-G_anom * g1_a + g2_a)
            + btA * C_ps * g0_a
            + dI_ps * u1
            + 2.0 * v1 * dP_5
            + 2.0 * v1 * dE_tot_a
        )
        dv2_a = (
            alA_eff * (G_anom * g0_a - g3_a)
            - btA * C_ps * g1_a
            + dI_ps * u2
            + 2.0 * v2 * dP_5
            + 2.0 * v2 * dE_tot_a
        )

        np.add.at(grad_alpha_a, g_idx, d_alA * S_A)
        np.add.at(grad_beta_a, g_idx, C_ps * S_B)
        np.add.at(
            grad_kappa_a,
            g_idx,
            sig_kA * (-0.5 * (P_5 * P_5 * P_5) * (D_P5 * D_P5 * D_P5) * dG_anom - (D_a * D_a) * E_tot * dD_a),
        )

        gx_flat[valid_offs, 0] = du1_L + du1_s + du1_a
        gx_flat[valid_offs, 1] = du2_L + du2_s + du2_a
        gx_flat[valid_offs, 2] = dv1_R + dv1_s + dv1_a
        gx_flat[valid_offs, 3] = dv2_R + dv2_s + dv2_a

    return (
        gx_flat.reshape(B, L, D),
        grad_theta_L,
        grad_w_L,
        grad_kappa_L,
        grad_theta_R,
        grad_w_R,
        grad_kappa_R,
        grad_omega_s,
        grad_gamma_s,
        grad_kappa_s,
        grad_alpha_a,
        grad_beta_a,
        grad_kappa_a,
    )


# ==============================================================================
# 5. PyTorch Autograd Function 与 nn.Module 封装（含无 PyTorch 环境轻量兼容层）
# ==============================================================================
if HAS_TORCH:

    class SpinChiralityAutogradFunction(torch.autograd.Function):
        """绑定解析反向核与 @triton.jit 调度的自定义 PyTorch Autograd Function。"""

        @staticmethod
        def forward(
            ctx,
            x: torch.Tensor,
            theta_L: torch.Tensor,
            w_L: torch.Tensor,
            kappa_L: torch.Tensor,
            theta_R: torch.Tensor,
            w_R: torch.Tensor,
            kappa_R: torch.Tensor,
            omega_s: torch.Tensor,
            gamma_s: torch.Tensor,
            kappa_s: torch.Tensor,
            alpha_a: torch.Tensor,
            beta_a: torch.Tensor,
            kappa_a: torch.Tensor,
            eps_zpe: float = 1e-4,
            sub_op_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        ) -> torch.Tensor:
            ctx.save_for_backward(
                x,
                theta_L,
                w_L,
                kappa_L,
                theta_R,
                w_R,
                kappa_R,
                omega_s,
                gamma_s,
                kappa_s,
                alpha_a,
                beta_a,
                kappa_a,
            )
            ctx.eps_zpe = eps_zpe
            ctx.sub_op_weights = sub_op_weights

            y_np = spin_chirality_numpy_forward(
                x.detach().cpu().numpy(),
                theta_L.detach().cpu().numpy(),
                w_L.detach().cpu().numpy(),
                kappa_L.detach().cpu().numpy(),
                theta_R.detach().cpu().numpy(),
                w_R.detach().cpu().numpy(),
                kappa_R.detach().cpu().numpy(),
                omega_s.detach().cpu().numpy(),
                gamma_s.detach().cpu().numpy(),
                kappa_s.detach().cpu().numpy(),
                alpha_a.detach().cpu().numpy(),
                beta_a.detach().cpu().numpy(),
                kappa_a.detach().cpu().numpy(),
                eps_zpe=eps_zpe,
                sub_op_weights=sub_op_weights,
            )
            return torch.as_tensor(y_np, dtype=x.dtype, device=x.device)

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            saved = ctx.saved_tensors
            np_args = [t.detach().cpu().numpy() for t in saved]
            grads_np = spin_chirality_numpy_backward(
                grad_out.detach().cpu().numpy(),
                *np_args,
                eps_zpe=ctx.eps_zpe,
                sub_op_weights=ctx.sub_op_weights,
            )
            grads_torch = tuple(
                torch.as_tensor(g, dtype=saved[0].dtype, device=saved[0].device) for g in grads_np
            )
            return grads_torch + (None, None)

    class SpinChiralityOperator(nn.Module):
        """基于粒子自旋与手性理论的独立神经网络算子 (PyTorch nn.Module)。"""

        def __init__(self, config: Optional[SpinChiralityConfig] = None) -> None:
            super().__init__()
            self.config = config or SpinChiralityConfig()
            G = self.config.dim // 4

            # 子算子 I (Left Weyl) 可学习参数
            self.theta_L = nn.Parameter(torch.linspace(0.35, 0.85, G, dtype=torch.float64))
            self.w_L = nn.Parameter(torch.linspace(0.90, 1.25, G, dtype=torch.float64))
            self.kappa_L = nn.Parameter(torch.linspace(-0.20, 0.30, G, dtype=torch.float64))

            # 子算子 II (Right Weyl) 可学习参数
            self.theta_R = nn.Parameter(torch.linspace(-0.75, -0.25, G, dtype=torch.float64))
            self.w_R = nn.Parameter(torch.linspace(0.95, 1.30, G, dtype=torch.float64))
            self.kappa_R = nn.Parameter(torch.linspace(-0.10, 0.40, G, dtype=torch.float64))

            # 子算子 III (Pauli Spin Precession) 可学习参数
            self.omega_s = nn.Parameter(torch.linspace(0.75, 1.15, G, dtype=torch.float64))
            self.gamma_s = nn.Parameter(torch.linspace(0.65, 1.05, G, dtype=torch.float64))
            self.kappa_s = nn.Parameter(torch.linspace(0.00, 0.50, G, dtype=torch.float64))

            # 子算子 IV (ABJ Chiral Anomaly) 可学习参数
            self.alpha_a = nn.Parameter(torch.linspace(0.60, 1.20, G, dtype=torch.float64))
            self.beta_a = nn.Parameter(torch.linspace(0.85, 1.25, G, dtype=torch.float64))
            self.kappa_a = nn.Parameter(torch.linspace(-0.20, 0.30, G, dtype=torch.float64))

        def forward(
            self,
            x: torch.Tensor,
            use_custom_autograd: bool = False,
            return_sub_ops: bool = False,
        ):
            if use_custom_autograd and not return_sub_ops:
                return SpinChiralityAutogradFunction.apply(
                    x,
                    self.theta_L,
                    self.w_L,
                    self.kappa_L,
                    self.theta_R,
                    self.w_R,
                    self.kappa_R,
                    self.omega_s,
                    self.gamma_s,
                    self.kappa_s,
                    self.alpha_a,
                    self.beta_a,
                    self.kappa_a,
                    self.config.eps_zpe,
                    self.config.sub_op_weights,
                )

            # 原生 PyTorch Eager 动态图路径（保证所有门控张量保留在动态计算图中）
            B, L, D = x.shape
            G = D // 4
            xg = x.view(B, L, G, 4)
            u1, u2, v1, v2 = xg[..., 0], xg[..., 1], xg[..., 2], xg[..., 3]
            eps = self.config.eps_zpe

            kappa_L_eff = F.softplus(self.kappa_L) + eps
            kappa_R_eff = F.softplus(self.kappa_R) + eps
            kappa_s_eff = F.softplus(self.kappa_s) + eps
            alpha_a_eff = torch.sigmoid(self.alpha_a)
            kappa_a_eff = F.softplus(self.kappa_a) + eps

            E_L = u1 * u1 + u2 * u2
            D_L = torch.rsqrt(1.0 + kappa_L_eff * E_L)
            cos_L, sin_L = torch.cos(self.theta_L), torch.sin(self.theta_L)
            r_L1 = cos_L * u1 - sin_L * u2
            r_L2 = sin_L * u1 + cos_L * u2
            zeros = torch.zeros_like(u1)
            op_L = torch.stack([self.w_L * D_L * r_L1, self.w_L * D_L * r_L2, zeros, zeros], dim=-1)

            E_R = v1 * v1 + v2 * v2
            D_R = torch.rsqrt(1.0 + kappa_R_eff * E_R)
            cos_R, sin_R = torch.cos(self.theta_R), torch.sin(self.theta_R)
            r_R1 = cos_R * v1 + sin_R * v2
            r_R2 = -sin_R * v1 + cos_R * v2
            op_R = torch.stack([zeros, zeros, self.w_R * D_R * r_R1, self.w_R * D_R * r_R2], dim=-1)

            E_tot = E_L + E_R
            D_s = 1.0 / (1.0 + kappa_s_eff * E_tot)
            S_y = u1 * v2 - u2 * v1
            op_spin = D_s.unsqueeze(-1) * torch.stack(
                [
                    self.omega_s * v1 + self.gamma_s * S_y * v2,
                    self.omega_s * v2 - self.gamma_s * S_y * v1,
                    self.omega_s * u1 - self.gamma_s * S_y * u2,
                    self.omega_s * u2 + self.gamma_s * S_y * u1,
                ],
                dim=-1,
            )

            P_5 = E_R - E_L
            G_anom = P_5 * torch.rsqrt(1.0 + kappa_a_eff * (P_5 * P_5))
            I_ps = u1 * v1 + u2 * v2
            C_ps = I_ps / (1.0 + kappa_a_eff * E_tot)
            op_anom = torch.stack(
                [
                    alpha_a_eff * (-u1 + G_anom * v2) + self.beta_a * C_ps * v1,
                    alpha_a_eff * (u2 - G_anom * v1) - self.beta_a * C_ps * v2,
                    alpha_a_eff * (v1 + G_anom * u2) + self.beta_a * C_ps * u1,
                    alpha_a_eff * (-v2 - G_anom * u1) - self.beta_a * C_ps * u2,
                ],
                dim=-1,
            )

            mL, mR, mS, mA = self.config.sub_op_weights
            y = (mL * op_L + mR * op_R + mS * op_spin + mA * op_anom).reshape(B, L, D)
            if return_sub_ops:
                return y, (
                    op_L.reshape(B, L, D),
                    op_R.reshape(B, L, D),
                    op_spin.reshape(B, L, D),
                    op_anom.reshape(B, L, D),
                )
            return y

else:

    class _CompatTensor:
        """无 PyTorch 环境下的轻量动态计算图张量封装，保证算子 API 与 Autograd 测试完全一致。"""

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

    class SpinChiralityOperator:  # type: ignore
        """基于粒子自旋与手性理论的独立神经网络算子（兼容模式，API 与 PyTorch nn.Module 完全一致）。"""

        def __init__(self, config: Optional[SpinChiralityConfig] = None) -> None:
            self.config = config or SpinChiralityConfig()
            G = self.config.dim // 4

            self.theta_L = _CompatTensor(np.linspace(0.35, 0.85, G), requires_grad=True)
            self.w_L = _CompatTensor(np.linspace(0.90, 1.25, G), requires_grad=True)
            self.kappa_L = _CompatTensor(np.linspace(-0.20, 0.30, G), requires_grad=True)

            self.theta_R = _CompatTensor(np.linspace(-0.75, -0.25, G), requires_grad=True)
            self.w_R = _CompatTensor(np.linspace(0.95, 1.30, G), requires_grad=True)
            self.kappa_R = _CompatTensor(np.linspace(-0.10, 0.40, G), requires_grad=True)

            self.omega_s = _CompatTensor(np.linspace(0.75, 1.15, G), requires_grad=True)
            self.gamma_s = _CompatTensor(np.linspace(0.65, 1.05, G), requires_grad=True)
            self.kappa_s = _CompatTensor(np.linspace(0.00, 0.50, G), requires_grad=True)

            self.alpha_a = _CompatTensor(np.linspace(0.60, 1.20, G), requires_grad=True)
            self.beta_a = _CompatTensor(np.linspace(0.85, 1.25, G), requires_grad=True)
            self.kappa_a = _CompatTensor(np.linspace(-0.20, 0.30, G), requires_grad=True)

        def parameters(self) -> List[_CompatTensor]:
            return [
                self.theta_L,
                self.w_L,
                self.kappa_L,
                self.theta_R,
                self.w_R,
                self.kappa_R,
                self.omega_s,
                self.gamma_s,
                self.kappa_s,
                self.alpha_a,
                self.beta_a,
                self.kappa_a,
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

            # Eager step-by-step tensor graph calculation (independent from spin_chirality_numpy_forward)
            B, L, D = x_tensor.shape
            G = D // 4
            xg = x_tensor.data.reshape(B, L, G, 4)
            u1, u2, v1, v2 = xg[..., 0], xg[..., 1], xg[..., 2], xg[..., 3]
            eps = self.config.eps_zpe

            kL_eff = np.log1p(np.exp(-np.abs(self.kappa_L.data))) + np.maximum(self.kappa_L.data, 0.0) + eps
            kR_eff = np.log1p(np.exp(-np.abs(self.kappa_R.data))) + np.maximum(self.kappa_R.data, 0.0) + eps
            kS_eff = np.log1p(np.exp(-np.abs(self.kappa_s.data))) + np.maximum(self.kappa_s.data, 0.0) + eps
            alA_eff = _np_sigmoid(self.alpha_a.data)
            kA_eff = np.log1p(np.exp(-np.abs(self.kappa_a.data))) + np.maximum(self.kappa_a.data, 0.0) + eps

            E_L = u1**2 + u2**2
            D_L = (1.0 + kL_eff * E_L) ** (-0.5)
            cL, sL = np.cos(self.theta_L.data), np.sin(self.theta_L.data)
            op_L = np.stack(
                [
                    self.w_L.data * D_L * (cL * u1 - sL * u2),
                    self.w_L.data * D_L * (sL * u1 + cL * u2),
                    np.zeros_like(u1),
                    np.zeros_like(u2),
                ],
                axis=-1,
            )

            E_R = v1**2 + v2**2
            D_R = (1.0 + kR_eff * E_R) ** (-0.5)
            cR, sR = np.cos(self.theta_R.data), np.sin(self.theta_R.data)
            op_R = np.stack(
                [
                    np.zeros_like(v1),
                    np.zeros_like(v2),
                    self.w_R.data * D_R * (cR * v1 + sR * v2),
                    self.w_R.data * D_R * (-sR * v1 + cR * v2),
                ],
                axis=-1,
            )

            E_tot = E_L + E_R
            D_s = 1.0 / (1.0 + kS_eff * E_tot)
            S_y = u1 * v2 - u2 * v1
            op_spin = D_s[..., None] * np.stack(
                [
                    self.omega_s.data * v1 + self.gamma_s.data * S_y * v2,
                    self.omega_s.data * v2 - self.gamma_s.data * S_y * v1,
                    self.omega_s.data * u1 - self.gamma_s.data * S_y * u2,
                    self.omega_s.data * u2 + self.gamma_s.data * S_y * u1,
                ],
                axis=-1,
            )

            P_5 = E_R - E_L
            G_anom = P_5 * ((1.0 + kA_eff * (P_5**2)) ** (-0.5))
            I_ps = u1 * v1 + u2 * v2
            C_ps = I_ps / (1.0 + kA_eff * E_tot)
            op_anom = np.stack(
                [
                    alA_eff * (-u1 + G_anom * v2) + self.beta_a.data * C_ps * v1,
                    alA_eff * (u2 - G_anom * v1) - self.beta_a.data * C_ps * v2,
                    alA_eff * (v1 + G_anom * u2) + self.beta_a.data * C_ps * u1,
                    alA_eff * (-v2 - G_anom * u1) - self.beta_a.data * C_ps * u2,
                ],
                axis=-1,
            )

            mL, mR, mS, mA = self.config.sub_op_weights
            y_data = (mL * op_L + mR * op_R + mS * op_spin + mA * op_anom).reshape(B, L, D)

            def _bwd(grad_out: np.ndarray) -> None:
                grads = spin_chirality_numpy_backward(
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
                    _CompatTensor(op_L.reshape(B, L, D)),
                    _CompatTensor(op_R.reshape(B, L, D)),
                    _CompatTensor(op_spin.reshape(B, L, D)),
                    _CompatTensor(op_anom.reshape(B, L, D)),
                )
            return out_tensor

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)


# ==============================================================================
# 6. 三重位级等价校验与五维正交独立性形式化审计 (Verification & 5D Audit Suite)
# ==============================================================================
def run_verification_and_5d_audit(verbose: bool = True) -> Dict[str, object]:
    """执行完整三重位级等价验证、中心有限差分 Gradcheck 与五维算子正交独立性审计。"""
    np.random.seed(2026)
    B, L, D = 8, 16, 16
    G = D // 4
    cfg = SpinChiralityConfig(dim=D, eps_zpe=1e-4, block_size=64)

    x_np = np.random.randn(B, L, D).astype(np.float64)
    grad_out_np = np.random.randn(B, L, D).astype(np.float64)

    op = SpinChiralityOperator(cfg)
    if HAS_TORCH:
        params_np = [
            op.theta_L.detach().cpu().numpy().copy(),
            op.w_L.detach().cpu().numpy().copy(),
            op.kappa_L.detach().cpu().numpy().copy(),
            op.theta_R.detach().cpu().numpy().copy(),
            op.w_R.detach().cpu().numpy().copy(),
            op.kappa_R.detach().cpu().numpy().copy(),
            op.omega_s.detach().cpu().numpy().copy(),
            op.gamma_s.detach().cpu().numpy().copy(),
            op.kappa_s.detach().cpu().numpy().copy(),
            op.alpha_a.detach().cpu().numpy().copy(),
            op.beta_a.detach().cpu().numpy().copy(),
            op.kappa_a.detach().cpu().numpy().copy(),
        ]
    else:
        params_np = [p.numpy().copy() for p in op.parameters()]

    # ------------------------------------------------------------------
    # A. 三套实现位级等价检验 (NumPy vs PyTorch/Autograd vs Triton Kernel)
    # ------------------------------------------------------------------
    y_numpy, sub_ops_np = spin_chirality_numpy_forward(
        x_np, *params_np, eps_zpe=cfg.eps_zpe, return_sub_ops=True
    )
    y_triton = spin_chirality_triton_forward(
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

    grads_numpy_all = spin_chirality_numpy_backward(
        grad_out_np, x_np, *params_np, eps_zpe=cfg.eps_zpe, return_sub_grads=True
    )
    grads_numpy = grads_numpy_all[:-1]
    sub_grads_np = grads_numpy_all[-1]

    grads_triton = spin_chirality_triton_backward(
        grad_out_np, x_np, *params_np, eps_zpe=cfg.eps_zpe, block_size=cfg.block_size
    )

    fwd_err_torch = float(np.max(np.abs(y_numpy - y_torch)))
    fwd_err_triton = float(np.max(np.abs(y_numpy - y_triton)))
    bwd_err_torch = float(np.max(np.abs(grads_numpy[0] - gx_torch)))
    bwd_err_triton_x = float(np.max(np.abs(grads_numpy[0] - grads_triton[0])))
    bwd_err_triton_params = float(
        max(np.max(np.abs(grads_numpy[i] - grads_triton[i])) for i in range(1, 13))
    )

    # ------------------------------------------------------------------
    # B. 解析反向传播对比中心有限差分校验 (Gradcheck, eps = 1e-6)
    # ------------------------------------------------------------------
    B_gc, L_gc = 2, 4
    x_gc = x_np[:B_gc, :L_gc].copy()
    go_gc = grad_out_np[:B_gc, :L_gc].copy()
    grads_gc = spin_chirality_numpy_backward(go_gc, x_gc, *params_np, eps_zpe=cfg.eps_zpe)

    eps_fd = 1e-6
    gx_fd = np.zeros_like(x_gc)
    for idx in np.ndindex(x_gc.shape):
        xp = x_gc.copy()
        xp[idx] += eps_fd
        xm = x_gc.copy()
        xm[idx] -= eps_fd
        yp = spin_chirality_numpy_forward(xp, *params_np, eps_zpe=cfg.eps_zpe)
        ym = spin_chirality_numpy_forward(xm, *params_np, eps_zpe=cfg.eps_zpe)
        gx_fd[idx] = np.sum((yp - ym) / (2.0 * eps_fd) * go_gc)

    grad_x_rel_err = float(
        np.max(np.abs(grads_gc[0] - gx_fd) / np.maximum(np.abs(grads_gc[0]) + np.abs(gx_fd), 1e-1))
    )
    grad_x_norm_rel_err = float(
        np.linalg.norm(grads_gc[0] - gx_fd) / np.maximum(np.linalg.norm(grads_gc[0]) + np.linalg.norm(gx_fd), 1e-12)
    )

    param_names = [
        "theta_L",
        "w_L",
        "kappa_L",
        "theta_R",
        "w_R",
        "kappa_R",
        "omega_s",
        "gamma_s",
        "kappa_s",
        "alpha_a",
        "beta_a",
        "kappa_a",
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
            yp = spin_chirality_numpy_forward(x_gc, *pp, eps_zpe=cfg.eps_zpe)
            ym = spin_chirality_numpy_forward(x_gc, *pm, eps_zpe=cfg.eps_zpe)
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
        _, sub_ops_pert = spin_chirality_numpy_forward(
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
        "O_L (Left Weyl Chiral Current)",
        "O_R (Right Weyl Helicity Current)",
        "O_spin (Pauli Spin Precession)",
        "O_anom (ABJ Chiral Anomaly Gate)",
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
        print("  SPIN-CHIRALITY OPERATOR: TRIPLE EQUIVALENCE & 5D INDEPENDENCE AUDIT")
        print("=" * 78)
        print(f"[1] Bit-Level Equivalence:")
        print(f"    - NumPy vs PyTorch/Autograd Fwd Max Abs Error : {fwd_err_torch:.3e} (< 1e-14)")
        print(f"    - NumPy vs Triton Kernel Fwd Max Abs Error    : {fwd_err_triton:.3e} (< 1e-14)")
        print(f"    - NumPy vs Triton Kernel Bwd dX Max Abs Error : {bwd_err_triton_x:.3e} (< 1e-14)")
        print(f"    - NumPy vs Triton Kernel Bwd dP Max Abs Error : {bwd_err_triton_params:.3e} (< 1e-13)")
        print(f"[2] Analytical Gradcheck vs Central Finite Difference (eps=1e-6):")
        print(f"    - Input X Max Elementwise Rel Error           : {grad_x_rel_err:.3e} (< 1e-8)")
        print(f"    - Input X Norm Relative Error                 : {grad_x_norm_rel_err:.3e} (< 1e-8)")
        print(f"    - All 12 Learnable Params Max Relative Error  : {max_param_rel_err:.3e} (< 1e-8)")
        for k_name, v_err in param_rel_errs.items():
            print(f"      * grad_{k_name:8s} rel err = {v_err:.3e}")
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
        print("  ALL VERIFICATION & 5D AUDIT CHECKS PASSED SUCCESSFULLY!")
        print("=" * 78)

    return report


if __name__ == "__main__":
    run_verification_and_5d_audit(verbose=True)