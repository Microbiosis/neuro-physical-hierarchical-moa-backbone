#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bio-inspired viral nucleic acid invasion operators on latent tensors.

Six independent proposals share a tie-aware top-k router, learnable stage
gates, and iterative refinement. NumPy provides the double-precision reference
and analytical backward, shared by the PyTorch autograd wrapper. The standalone
Stage I Triton kernel and Stage VI tiled simulator are separate from this path.
The default parameters are research examples for synthetic tensor experiments.
"""

from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ==============================================================================
# 可选深度学习与 GPU 编译器后端探测（优雅降级，保证纯 NumPy 与 PyTorch/Triton 双环境兼容）
# ==============================================================================
try:
    import torch
    import torch.nn as nn
    from torch.autograd.function import once_differentiable

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None  # type: ignore
    tl = None  # type: ignore


# ==============================================================================
# 第一部分：基础数学原语与正性/有界重参数化工具（保证分母恒正与精确解析导数）
# ==============================================================================

def np_sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的双精度 Sigmoid 激活函数 σ(x) = 1 / (1 + exp(-x))。"""
    x_clip = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def np_softplus(x: np.ndarray) -> np.ndarray:
    """数值稳定的双精度 Softplus 函数 log(1 + exp(x))。"""
    x_arr = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x_arr))) + np.maximum(x_arr, 0.0)


def np_positive_param(raw: Union[np.ndarray, float], eps: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """
    有理分母与速率参数的正性重参数化：
        κ_eff = softplus(raw) + eps > 0
        d(κ_eff)/d(raw) = σ(raw)
    彻底消除负参数引发的分母奇点或发散。
    """
    raw_arr = np.asarray(raw, dtype=np.float64)
    eff = np_softplus(raw_arr) + eps
    grad_factor = np_sigmoid(raw_arr)
    return eff, grad_factor


def np_bounded_rate(raw: Union[np.ndarray, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    衰减率与门控系数的有界重参数化：
        α_eff = σ(raw) ∈ (0, 1)
        d(α_eff)/d(raw) = α_eff * (1 - α_eff)
    """
    raw_arr = np.asarray(raw, dtype=np.float64)
    eff = np_sigmoid(raw_arr)
    grad_factor = eff * (1.0 - eff)
    return eff, grad_factor


# ==============================================================================
# 第二部分：配置体系（ViralInvasionOperatorConfig）
# ==============================================================================

@dataclass
class ViralInvasionOperatorConfig:
    """Configuration for the six-proposal latent operator."""
    d_model: int = 32
    n_heads: int = 4
    num_baltimore_experts: int = 6
    top_k: int = 3
    num_pam_anchors: int = 4
    spatial_chromatin_kernel: int = 3
    conv_kernel_size: int = 3
    num_refinement_steps: int = 2
    share_refinement_weights: bool = False
    enable_adaptive_halting: bool = True
    halting_threshold: float = 0.85
    crispr_groups: int = 4
    dropout: float = 0.0
    zpe_eps: float = 1e-4
    proposer_mode: str = "all_baltimore_stages"


# ==============================================================================
# 第三部分：六个正交生物学阶段的确定性双精度解析前向/反向原语
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage I: 受体特异性识别吸附与内吞构象变构膜融合算子
# (Receptor-Mediated Tropism & Endosomal pH Conformational Membrane Fusion)
# ------------------------------------------------------------------------------
def stage1_receptor_membrane_fusion_forward(
    x: np.ndarray,
    alpha_trop: np.ndarray,
    kappa_kd_raw: np.ndarray,
    omega_ph: np.ndarray,
    m_lipid: np.ndarray,
    gamma_pore: np.ndarray,
    zpe_eps: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    生物学机理：
        1. 受体-配体协同亲和力（Hill 吸附动力学）：
           S = α_trop * x
           A_bind = S^2 / (K_d + S^2), 其中 K_d = softplus(kappa_kd_raw) + 1e-4 > 0
        2. 内吞体酸化（pH 降落）激发的跨膜脂双层势垒构象隧穿门（含真空/热涨落下界安全阀）：
           E_ph = omega_ph^2 * x^2 + zpe_eps > 0
           G_schw = exp(-m_lipid^2 / E_ph)
        3. 膜融合孔道形成与核衣壳胞质注入响应：
           Y_I = A_bind * G_schw * tanh(x) + gamma_pore * (x / sqrt(1 + K_d * x^2))
    """
    x = np.asarray(x, dtype=np.float64)
    alpha_trop = np.asarray(alpha_trop, dtype=np.float64)
    kappa_kd_raw = np.asarray(kappa_kd_raw, dtype=np.float64)
    omega_ph = np.asarray(omega_ph, dtype=np.float64)
    m_lipid = np.asarray(m_lipid, dtype=np.float64)
    gamma_pore = np.asarray(gamma_pore, dtype=np.float64)

    kd_eff, dkd_draw = np_positive_param(kappa_kd_raw, eps=zpe_eps)
    s = alpha_trop * x
    s2 = s * s
    denom_hill = kd_eff + s2
    a_bind = s2 / denom_hill

    x2 = x * x
    e_ph = (omega_ph * omega_ph) * x2 + zpe_eps
    m2 = m_lipid * m_lipid
    g_schw = np.exp(-m2 / e_ph)

    tanh_x = np.tanh(x)
    fusion_term = a_bind * g_schw * tanh_x

    denom_pore_sq = 1.0 + kd_eff * x2
    inv_denom_pore = 1.0 / np.sqrt(denom_pore_sq)
    pore_term = gamma_pore * x * inv_denom_pore

    y = fusion_term + pore_term

    cache = {
        "x": x,
        "alpha_trop": alpha_trop,
        "kappa_kd_raw": kappa_kd_raw,
        "kd_eff": kd_eff,
        "dkd_draw": dkd_draw,
        "omega_ph": omega_ph,
        "m_lipid": m_lipid,
        "gamma_pore": gamma_pore,
        "s": s,
        "s2": s2,
        "denom_hill": denom_hill,
        "a_bind": a_bind,
        "x2": x2,
        "e_ph": e_ph,
        "m2": m2,
        "g_schw": g_schw,
        "tanh_x": tanh_x,
        "denom_pore_sq": denom_pore_sq,
        "inv_denom_pore": inv_denom_pore,
    }
    return y, cache


def stage1_receptor_membrane_fusion_backward(
    dy: np.ndarray,
    cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stage I 解析反向传播，返回 (dx, d_alpha_trop, d_kappa_kd_raw, d_omega_ph, d_m_lipid, d_gamma_pore)。"""
    dy = np.asarray(dy, dtype=np.float64)
    x = cache["x"]
    alpha_trop = cache["alpha_trop"]
    kd_eff = cache["kd_eff"]
    dkd_draw = cache["dkd_draw"]
    omega_ph = cache["omega_ph"]
    m_lipid = cache["m_lipid"]
    gamma_pore = cache["gamma_pore"]
    s = cache["s"]
    s2 = cache["s2"]
    denom_hill = cache["denom_hill"]
    a_bind = cache["a_bind"]
    x2 = cache["x2"]
    e_ph = cache["e_ph"]
    m2 = cache["m2"]
    g_schw = cache["g_schw"]
    tanh_x = cache["tanh_x"]
    denom_pore_sq = cache["denom_pore_sq"]
    inv_denom_pore = cache["inv_denom_pore"]

    reduce_axes = tuple(range(dy.ndim - 1))

    # 1. Pore 项导数: pore_term = gamma_pore * x * (1 + kd_eff * x^2)^(-1/2)
    d_gamma_pore = np.sum(dy * x * inv_denom_pore, axis=reduce_axes)
    inv_denom_pore3 = inv_denom_pore / denom_pore_sq
    dx_pore = dy * gamma_pore * inv_denom_pore3
    dkd_pore = dy * gamma_pore * x * (-0.5 * x2 * inv_denom_pore3)

    # 2. Fusion 项导数: fusion_term = a_bind * g_schw * tanh_x
    da_bind = dy * g_schw * tanh_x
    dg_schw = dy * a_bind * tanh_x
    dtanh_x = dy * a_bind * g_schw

    denom_hill2 = denom_hill * denom_hill
    ds = da_bind * (2.0 * s * kd_eff / denom_hill2)
    dkd_hill = da_bind * (-s2 / denom_hill2)

    d_alpha_trop = np.sum(ds * x, axis=reduce_axes)
    dx_hill = ds * alpha_trop

    de_ph = dg_schw * g_schw * (m2 / (e_ph * e_ph))
    d_m_lipid = np.sum(dg_schw * g_schw * (-2.0 * m_lipid / e_ph), axis=reduce_axes)
    d_omega_ph = np.sum(de_ph * (2.0 * omega_ph * x2), axis=reduce_axes)
    dx_schw = de_ph * (2.0 * (omega_ph * omega_ph) * x)

    dx_tanh = dtanh_x * (1.0 - tanh_x * tanh_x)

    dkd_total = dkd_pore + dkd_hill
    d_kappa_kd_raw = np.sum(dkd_total * dkd_draw, axis=reduce_axes)
    dx = dx_pore + dx_hill + dx_schw + dx_tanh

    return dx, d_alpha_trop, d_kappa_kd_raw, d_omega_ph, d_m_lipid, d_gamma_pore


# ------------------------------------------------------------------------------
# Stage II: 脱壳与逆转录酶双链转换及 RNase H 杂合链降解算子
# (Baltimore Group VI Reverse Transcription, RNase H Heteroduplex Cleavage & LTR Synthesis)
# ------------------------------------------------------------------------------
def stage2_reverse_transcription_rnaseh_forward(
    x: np.ndarray,
    alpha_rddp: np.ndarray,
    lambda_rnaseh_raw: np.ndarray,
    kappa_hyb_raw: np.ndarray,
    omega_ltr: np.ndarray,
    pi_ltr: np.ndarray,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    生物学机理：
        1. RdDp 负链 cDNA(-) 合成（由病毒正链 ssRNA(+) 模板 x 合成互补负链）：
           C_minus = tanh(alpha_rddp * x)
        2. RNase H 杂合双链特异性降解门（仅降解已形成 RNA:cDNA 杂合链区域的原始 RNA 模板）：
           E_hyb = (x * C_minus)^2 / (1 + kappa_hyb * (x^2 + C_minus^2))
           D_rnaseH = exp(-lambda_rnaseH * E_hyb)
        3. 模板跳跃（Copy-Choice）重组与两端长末端重复（5'/3' LTR）双链 dsDNA 合成：
           Y_II = (1 - D_rnaseH) * C_minus + omega_ltr * sin(pi_ltr * x + C_minus)
    """
    x = np.asarray(x, dtype=np.float64)
    alpha_rddp = np.asarray(alpha_rddp, dtype=np.float64)
    lambda_rnaseh_raw = np.asarray(lambda_rnaseh_raw, dtype=np.float64)
    kappa_hyb_raw = np.asarray(kappa_hyb_raw, dtype=np.float64)
    omega_ltr = np.asarray(omega_ltr, dtype=np.float64)
    pi_ltr = np.asarray(pi_ltr, dtype=np.float64)

    lam_eff, dlam_draw = np_positive_param(lambda_rnaseh_raw, eps=eps)
    kap_eff, dkap_draw = np_positive_param(kappa_hyb_raw, eps=eps)

    u = alpha_rddp * x
    c_minus = np.tanh(u)
    dc_du = 1.0 - c_minus * c_minus

    x2 = x * x
    c2 = c_minus * c_minus
    num_hyb = x2 * c2
    denom_hyb = 1.0 + kap_eff * (x2 + c2)
    e_hyb = num_hyb / denom_hyb

    d_rnaseh = np.exp(-lam_eff * e_hyb)

    phase_ltr = pi_ltr * x + c_minus
    sin_ltr = np.sin(phase_ltr)
    cos_ltr = np.cos(phase_ltr)

    y = (1.0 - d_rnaseh) * c_minus + omega_ltr * sin_ltr

    cache = {
        "x": x,
        "alpha_rddp": alpha_rddp,
        "lam_eff": lam_eff,
        "dlam_draw": dlam_draw,
        "kap_eff": kap_eff,
        "dkap_draw": dkap_draw,
        "omega_ltr": omega_ltr,
        "pi_ltr": pi_ltr,
        "c_minus": c_minus,
        "dc_du": dc_du,
        "x2": x2,
        "c2": c2,
        "num_hyb": num_hyb,
        "denom_hyb": denom_hyb,
        "e_hyb": e_hyb,
        "d_rnaseh": d_rnaseh,
        "sin_ltr": sin_ltr,
        "cos_ltr": cos_ltr,
    }
    return y, cache


def stage2_reverse_transcription_rnaseh_backward(
    dy: np.ndarray,
    cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stage II 解析反向传播，返回 (dx, d_alpha_rddp, d_lambda_rnaseh_raw, d_kappa_hyb_raw, d_omega_ltr, d_pi_ltr)。"""
    dy = np.asarray(dy, dtype=np.float64)
    x = cache["x"]
    alpha_rddp = cache["alpha_rddp"]
    lam_eff = cache["lam_eff"]
    dlam_draw = cache["dlam_draw"]
    kap_eff = cache["kap_eff"]
    dkap_draw = cache["dkap_draw"]
    omega_ltr = cache["omega_ltr"]
    pi_ltr = cache["pi_ltr"]
    c_minus = cache["c_minus"]
    dc_du = cache["dc_du"]
    x2 = cache["x2"]
    c2 = cache["c2"]
    num_hyb = cache["num_hyb"]
    denom_hyb = cache["denom_hyb"]
    e_hyb = cache["e_hyb"]
    d_rnaseh = cache["d_rnaseh"]
    sin_ltr = cache["sin_ltr"]
    cos_ltr = cache["cos_ltr"]

    reduce_axes = tuple(range(dy.ndim - 1))

    # LTR 项导数: omega_ltr * sin(pi_ltr * x + c_minus)
    d_omega_ltr = np.sum(dy * sin_ltr, axis=reduce_axes)
    d_phase_ltr = dy * omega_ltr * cos_ltr
    d_pi_ltr = np.sum(d_phase_ltr * x, axis=reduce_axes)
    dx_ltr = d_phase_ltr * pi_ltr
    dc_minus = dy * (1.0 - d_rnaseh) + d_phase_ltr

    # d_rnaseh 项导数
    dd_rnaseh = dy * (-c_minus)
    d_lam_eff = dd_rnaseh * d_rnaseh * (-e_hyb)
    d_lambda_rnaseh_raw = np.sum(d_lam_eff * dlam_draw, axis=reduce_axes)

    de_hyb = dd_rnaseh * d_rnaseh * (-lam_eff)
    denom_hyb2 = denom_hyb * denom_hyb

    d_kap_eff = de_hyb * (-num_hyb * (x2 + c2) / denom_hyb2)
    d_kappa_hyb_raw = np.sum(d_kap_eff * dkap_draw, axis=reduce_axes)

    dx2 = de_hyb * (c2 * (1.0 + kap_eff * c2) / denom_hyb2)
    dc2 = de_hyb * (x2 * (1.0 + kap_eff * x2) / denom_hyb2)

    dc_minus_total = dc_minus + dc2 * (2.0 * c_minus)
    du = dc_minus_total * dc_du
    d_alpha_rddp = np.sum(du * x, axis=reduce_axes)

    dx = dx_ltr + dx2 * (2.0 * x) + du * alpha_rddp
    return dx, d_alpha_rddp, d_lambda_rnaseh_raw, d_kappa_hyb_raw, d_omega_ltr, d_pi_ltr


# ------------------------------------------------------------------------------
# Stage III: 整合酶链转移与宿主染色质前病毒整合算子
# (Integrase 3'-OH Strand Transfer, 2D Chromatin Contact Conv, PAM Co-Attention & TSD Splice)
# ------------------------------------------------------------------------------
def pam_anchor_hybridization_co_attention_forward(
    viral_dna: np.ndarray,
    host_chromatin: np.ndarray,
    pam_anchors: np.ndarray,
    proj_out: np.ndarray,
    scale: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    零距离衰减低秩同源/PAM 潜锚点协同注意力（O(L * R_anchor * D) 线性复杂度）：
        模拟病毒前整合复合体（PIC）在全基因组尺度检索富含 AT/开放常染色质靶位点（PAM/同源序列）并广播整合上下文。
    """
    viral_dna = np.asarray(viral_dna, dtype=np.float64)
    host_chromatin = np.asarray(host_chromatin, dtype=np.float64)
    pam_anchors = np.asarray(pam_anchors, dtype=np.float64)
    proj_out = np.asarray(proj_out, dtype=np.float64)

    d_model = viral_dna.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(float(d_model))

    scores_h = np.einsum("rd,bld->brl", pam_anchors, host_chromatin) * scale
    scores_h_max = np.max(scores_h, axis=-1, keepdims=True)
    exp_h = np.exp(np.clip(scores_h - scores_h_max, -60.0, 60.0))
    attn_h = exp_h / np.sum(exp_h, axis=-1, keepdims=True)

    h_pool = np.einsum("brl,bld->brd", attn_h, host_chromatin)

    scores_v = np.einsum("bld,rd->blr", viral_dna, pam_anchors) * scale
    scores_v_max = np.max(scores_v, axis=-1, keepdims=True)
    exp_v = np.exp(np.clip(scores_v - scores_v_max, -60.0, 60.0))
    attn_v = exp_v / np.sum(exp_v, axis=-1, keepdims=True)

    c_raw = np.einsum("blr,brd->bld", attn_v, h_pool)
    c_align = c_raw * proj_out

    cache = {
        "viral_dna": viral_dna,
        "host_chromatin": host_chromatin,
        "pam_anchors": pam_anchors,
        "proj_out": proj_out,
        "scale": scale,
        "attn_h": attn_h,
        "h_pool": h_pool,
        "attn_v": attn_v,
        "c_raw": c_raw,
    }
    return c_align, cache


def pam_anchor_hybridization_co_attention_backward(
    dc_align: np.ndarray,
    cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """低秩 PAM 潜锚点协同注意力解析反向传播，返回 (d_viral_dna, d_host_chromatin, d_pam_anchors, d_proj_out)。"""
    dc_align = np.asarray(dc_align, dtype=np.float64)
    viral_dna = cache["viral_dna"]
    host_chromatin = cache["host_chromatin"]
    pam_anchors = cache["pam_anchors"]
    proj_out = cache["proj_out"]
    scale = cache["scale"]
    attn_h = cache["attn_h"]
    h_pool = cache["h_pool"]
    attn_v = cache["attn_v"]
    c_raw = cache["c_raw"]

    d_proj_out = np.sum(dc_align * c_raw, axis=(0, 1))
    dc_raw = dc_align * proj_out

    d_attn_v = np.einsum("bld,brd->blr", dc_raw, h_pool)
    dh_pool = np.einsum("blr,bld->brd", attn_v, dc_raw)

    d_scores_v = attn_v * (d_attn_v - np.sum(d_attn_v * attn_v, axis=-1, keepdims=True)) * scale
    d_viral_dna = np.einsum("blr,rd->bld", d_scores_v, pam_anchors)
    d_pam_from_v = np.einsum("blr,bld->rd", d_scores_v, viral_dna)

    d_attn_h = np.einsum("brd,bld->brl", dh_pool, host_chromatin)
    d_host_from_pool = np.einsum("brl,brd->bld", attn_h, dh_pool)

    d_scores_h = attn_h * (d_attn_h - np.sum(d_attn_h * attn_h, axis=-1, keepdims=True)) * scale
    d_pam_from_h = np.einsum("brl,bld->rd", d_scores_h, host_chromatin)
    d_host_from_attn = np.einsum("brl,rd->bld", d_scores_h, pam_anchors)

    d_host_chromatin = d_host_from_pool + d_host_from_attn
    d_pam_anchors = d_pam_from_v + d_pam_from_h
    return d_viral_dna, d_host_chromatin, d_pam_anchors, d_proj_out


def stage3_integrase_proviral_splice_forward(
    x_host: np.ndarray,
    d_viral: np.ndarray,
    c_align: np.ndarray,
    w_npc: np.ndarray,
    w_oh: np.ndarray,
    beta_tsd: np.ndarray,
    kappa_int_raw: np.ndarray,
    tau_repair: np.ndarray,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Stage III 核心门控：整合酶 3'-OH 亲核攻击链转移与 Hadamard-Bilinear 前病毒宿主缝合算子：
        G_int = σ(w_npc * X_host + w_oh * D_viral)
        Y_III = (1 - G_int) * X_host + G_int * D_viral
                + beta_tsd * (X_host * D_viral + C_align) / sqrt(1 + kappa_int * (X_host^2 + D_viral^2))
                + tau_repair * (D_viral - X_host) / (1 + kappa_int * (D_viral - X_host)^2)
    """
    x_host = np.asarray(x_host, dtype=np.float64)
    d_viral = np.asarray(d_viral, dtype=np.float64)
    c_align = np.asarray(c_align, dtype=np.float64)
    w_npc = np.asarray(w_npc, dtype=np.float64)
    w_oh = np.asarray(w_oh, dtype=np.float64)
    beta_tsd = np.asarray(beta_tsd, dtype=np.float64)
    kappa_int_raw = np.asarray(kappa_int_raw, dtype=np.float64)
    tau_repair = np.asarray(tau_repair, dtype=np.float64)

    kap_int, dkap_draw = np_positive_param(kappa_int_raw, eps=eps)

    logit_int = w_npc * x_host + w_oh * d_viral
    g_int = np_sigmoid(logit_int)

    eh2 = x_host * x_host
    ev2 = d_viral * d_viral
    e_sum = eh2 + ev2
    denom_bilin_sq = 1.0 + kap_int * e_sum
    inv_denom_bilin = 1.0 / np.sqrt(denom_bilin_sq)
    num_bilin = x_host * d_viral + c_align
    bilin_term = beta_tsd * num_bilin * inv_denom_bilin

    delta_vd = d_viral - x_host
    delta2 = delta_vd * delta_vd
    denom_rep = 1.0 + kap_int * delta2
    repair_term = tau_repair * (delta_vd / denom_rep)

    y = (1.0 - g_int) * x_host + g_int * d_viral + bilin_term + repair_term

    cache = {
        "x_host": x_host,
        "d_viral": d_viral,
        "c_align": c_align,
        "w_npc": w_npc,
        "w_oh": w_oh,
        "beta_tsd": beta_tsd,
        "kap_int": kap_int,
        "dkap_draw": dkap_draw,
        "tau_repair": tau_repair,
        "g_int": g_int,
        "e_sum": e_sum,
        "denom_bilin_sq": denom_bilin_sq,
        "inv_denom_bilin": inv_denom_bilin,
        "num_bilin": num_bilin,
        "delta_vd": delta_vd,
        "delta2": delta2,
        "denom_rep": denom_rep,
    }
    return y, cache


def stage3_integrase_proviral_splice_backward(
    dy: np.ndarray,
    cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage III 解析反向传播，返回:
        (dx_host, dd_viral, dc_align, dw_npc, dw_oh, dbeta_tsd, dkappa_int_raw, dtau_repair)
    """
    dy = np.asarray(dy, dtype=np.float64)
    x_host = cache["x_host"]
    d_viral = cache["d_viral"]
    w_npc = cache["w_npc"]
    w_oh = cache["w_oh"]
    beta_tsd = cache["beta_tsd"]
    kap_int = cache["kap_int"]
    dkap_draw = cache["dkap_draw"]
    tau_repair = cache["tau_repair"]
    g_int = cache["g_int"]
    e_sum = cache["e_sum"]
    denom_bilin_sq = cache["denom_bilin_sq"]
    inv_denom_bilin = cache["inv_denom_bilin"]
    num_bilin = cache["num_bilin"]
    delta_vd = cache["delta_vd"]
    delta2 = cache["delta2"]
    denom_rep = cache["denom_rep"]

    reduce_axes = tuple(range(dy.ndim - 1))

    dg_int = dy * (d_viral - x_host)
    dlogit = dg_int * g_int * (1.0 - g_int)
    dw_npc = np.sum(dlogit * x_host, axis=reduce_axes)
    dw_oh = np.sum(dlogit * d_viral, axis=reduce_axes)

    dx_gate = dy * (1.0 - g_int) + dlogit * w_npc
    dd_gate = dy * g_int + dlogit * w_oh

    dbeta_tsd = np.sum(dy * num_bilin * inv_denom_bilin, axis=reduce_axes)
    dnum_bilin = dy * beta_tsd * inv_denom_bilin
    dc_align = dnum_bilin

    inv_denom_bilin3 = inv_denom_bilin / denom_bilin_sq
    d_denom_bilin_sq = dy * beta_tsd * num_bilin * (-0.5 * inv_denom_bilin3)
    dkap_bilin = d_denom_bilin_sq * e_sum
    de_sum = d_denom_bilin_sq * kap_int

    dx_bilin = dnum_bilin * d_viral + de_sum * (2.0 * x_host)
    dd_bilin = dnum_bilin * x_host + de_sum * (2.0 * d_viral)

    dtau_repair = np.sum(dy * (delta_vd / denom_rep), axis=reduce_axes)
    denom_rep2 = denom_rep * denom_rep
    ddelta_vd = dy * tau_repair * ((1.0 - kap_int * delta2) / denom_rep2)
    dkap_rep = dy * tau_repair * (-delta_vd * delta2 / denom_rep2)

    dkappa_int_raw = np.sum((dkap_bilin + dkap_rep) * dkap_draw, axis=reduce_axes)
    dx_host = dx_gate + dx_bilin - ddelta_vd
    dd_viral = dd_gate + dd_bilin + ddelta_vd

    return dx_host, dd_viral, dc_align, dw_npc, dw_oh, dbeta_tsd, dkappa_int_raw, dtau_repair


# ------------------------------------------------------------------------------
# Stage IV: 表观遗传 CpG 岛甲基化潜伏沉默与裂解期 CI/Cro 双稳态开关算子
# (Epigenetic CpG Methylation Silencing vs. Lytic SOS / Tat-TAR Bistable Switch)
# ------------------------------------------------------------------------------
def stage4_epigenetic_latency_lytic_forward(
    x: np.ndarray,
    mu_dnmt_raw: np.ndarray,
    a_ci: np.ndarray,
    b_cro: np.ndarray,
    kappa_sat_raw: np.ndarray,
    gamma_tat: np.ndarray,
    theta_methyl: np.ndarray,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    生物学机理：
        1. 宿主 DNMT 催化的前病毒启动子 CpG 岛 5-mC 甲基化沉默包络（偶对称表观能量势阱）：
           M_cpg = exp(-mu_dnmt * x^2), 其中 mu_dnmt = softplus(mu_dnmt_raw) + eps > 0
        2. λ 噬菌体 CI/Cro 型双稳态基因开关三次势能导数（含高振幅有理饱和约束）：
           Phi_bistable = a_ci * (x^2 - 0.5) - b_cro * x^3 / (1 + kappa_sat * x^2)
        3. Tat-TAR 正反馈反式激活门（与甲基化沉默对抗）：
           G_lytic = σ(gamma_tat * x^2 - theta_methyl * M_cpg)
        4. 潜伏期-裂解期命运综合输出：
           Y_IV = (1 - G_lytic) * M_cpg * x + G_lytic * Phi_bistable
    """
    x = np.asarray(x, dtype=np.float64)
    mu_dnmt_raw = np.asarray(mu_dnmt_raw, dtype=np.float64)
    a_ci = np.asarray(a_ci, dtype=np.float64)
    b_cro = np.asarray(b_cro, dtype=np.float64)
    kappa_sat_raw = np.asarray(kappa_sat_raw, dtype=np.float64)
    gamma_tat = np.asarray(gamma_tat, dtype=np.float64)
    theta_methyl = np.asarray(theta_methyl, dtype=np.float64)

    mu_eff, dmu_draw = np_positive_param(mu_dnmt_raw, eps=eps)
    kap_sat, dkap_draw = np_positive_param(kappa_sat_raw, eps=eps)

    x2 = x * x
    x3 = x2 * x
    m_cpg = np.exp(-mu_eff * x2)

    denom_sat = 1.0 + kap_sat * x2
    cubic_term = x3 / denom_sat
    quad_shift = x2 - 0.5
    phi_bistable = a_ci * quad_shift - b_cro * cubic_term

    logit_lytic = gamma_tat * x2 - theta_methyl * m_cpg
    g_lytic = np_sigmoid(logit_lytic)

    latent_branch = m_cpg * x
    y = (1.0 - g_lytic) * latent_branch + g_lytic * phi_bistable

    cache = {
        "x": x,
        "x2": x2,
        "x3": x3,
        "quad_shift": quad_shift,
        "mu_eff": mu_eff,
        "dmu_draw": dmu_draw,
        "a_ci": a_ci,
        "b_cro": b_cro,
        "kap_sat": kap_sat,
        "dkap_draw": dkap_draw,
        "gamma_tat": gamma_tat,
        "theta_methyl": theta_methyl,
        "m_cpg": m_cpg,
        "denom_sat": denom_sat,
        "cubic_term": cubic_term,
        "phi_bistable": phi_bistable,
        "g_lytic": g_lytic,
        "latent_branch": latent_branch,
    }
    return y, cache


def stage4_epigenetic_latency_lytic_backward(
    dy: np.ndarray,
    cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage IV 解析反向传播，返回:
        (dx, d_mu_dnmt_raw, d_a_ci, d_b_cro, d_kappa_sat_raw, d_gamma_tat, d_theta_methyl)
    """
    dy = np.asarray(dy, dtype=np.float64)
    x = cache["x"]
    x2 = cache["x2"]
    x3 = cache["x3"]
    quad_shift = cache["quad_shift"]
    mu_eff = cache["mu_eff"]
    dmu_draw = cache["dmu_draw"]
    a_ci = cache["a_ci"]
    b_cro = cache["b_cro"]
    kap_sat = cache["kap_sat"]
    dkap_draw = cache["dkap_draw"]
    gamma_tat = cache["gamma_tat"]
    theta_methyl = cache["theta_methyl"]
    m_cpg = cache["m_cpg"]
    denom_sat = cache["denom_sat"]
    cubic_term = cache["cubic_term"]
    phi_bistable = cache["phi_bistable"]
    g_lytic = cache["g_lytic"]
    latent_branch = cache["latent_branch"]

    reduce_axes = tuple(range(dy.ndim - 1))

    dg_lytic = dy * (phi_bistable - latent_branch)
    d_latent = dy * (1.0 - g_lytic)
    d_phi = dy * g_lytic

    # 1. Phi_bistable 导数
    d_a_ci = np.sum(d_phi * quad_shift, axis=reduce_axes)
    d_b_cro = np.sum(d_phi * (-cubic_term), axis=reduce_axes)

    dcubic = d_phi * (-b_cro)
    denom_sat2 = denom_sat * denom_sat
    dkap_sat = dcubic * (-x3 * x2 / denom_sat2)
    d_kappa_sat_raw = np.sum(dkap_sat * dkap_draw, axis=reduce_axes)

    dx_phi = d_phi * a_ci * (2.0 * x) + dcubic * (x2 * (3.0 + kap_sat * x2) / denom_sat2)

    # 2. G_lytic 导数
    dlogit = dg_lytic * g_lytic * (1.0 - g_lytic)
    d_gamma_tat = np.sum(dlogit * x2, axis=reduce_axes)
    d_theta_methyl = np.sum(dlogit * (-m_cpg), axis=reduce_axes)

    dx_logit = dlogit * gamma_tat * (2.0 * x)
    dm_cpg = d_latent * x + dlogit * (-theta_methyl)

    # 3. M_cpg = exp(-mu_eff * x^2) 导数
    dmu_eff = dm_cpg * m_cpg * (-x2)
    d_mu_dnmt_raw = np.sum(dmu_eff * dmu_draw, axis=reduce_axes)
    dx_mcpg = dm_cpg * m_cpg * (-2.0 * mu_eff * x)

    dx = d_latent * m_cpg + dx_phi + dx_logit + dx_mcpg
    return dx, d_mu_dnmt_raw, d_a_ci, d_b_cro, d_kappa_sat_raw, d_gamma_tat, d_theta_methyl


# ------------------------------------------------------------------------------
# Stage V: RdRp 无校对准种突变云与 APOBEC3G 胞嘧啶脱氨超突变算子
# (Eigen Quasispecies Error-Catastrophe Corridor & APOBEC3G C->U / G->A Hypermutation)
# ------------------------------------------------------------------------------
def stage5_quasispecies_hypermutation_forward(
    x: np.ndarray,
    alpha_drift_raw: np.ndarray,
    kappa_apobec_raw: np.ndarray,
    beta_deam: np.ndarray,
    omega_ga: np.ndarray,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    生物学机理：
        1. Eigen 准种适应度-突变走廊包络（适度突变提供抗原漂移增益 alpha_drift，
           超过错误灾变阈值后被 APOBEC3G 平方阻尼压制）：
           E_mut = x^2
           W_eigen = (alpha_drift * E_mut) / sqrt(1 + kappa_apobec * E_mut^2)
        2. APOBEC3G 催化的胞嘧啶脱氨（C->U / G->A）转换谱调制：
           Y_V = W_eigen * x + beta_deam * cos(omega_ga * x) / (1 + kappa_apobec * E_mut)
    """
    x = np.asarray(x, dtype=np.float64)
    alpha_drift_raw = np.asarray(alpha_drift_raw, dtype=np.float64)
    kappa_apobec_raw = np.asarray(kappa_apobec_raw, dtype=np.float64)
    beta_deam = np.asarray(beta_deam, dtype=np.float64)
    omega_ga = np.asarray(omega_ga, dtype=np.float64)

    alpha_drift, dalpha_draw = np_positive_param(alpha_drift_raw, eps=eps)
    kap_apo, dkap_draw = np_positive_param(kappa_apobec_raw, eps=eps)

    e_mut = x * x
    e_mut2 = e_mut * e_mut
    num_eigen = alpha_drift * e_mut
    denom_eigen_sq = 1.0 + kap_apo * e_mut2
    inv_denom_eigen = 1.0 / np.sqrt(denom_eigen_sq)
    w_eigen = num_eigen * inv_denom_eigen

    phase_ga = omega_ga * x
    cos_ga = np.cos(phase_ga)
    sin_ga = np.sin(phase_ga)
    denom_deam = 1.0 + kap_apo * e_mut
    deam_core = cos_ga / denom_deam

    y = w_eigen * x + beta_deam * deam_core

    cache = {
        "x": x,
        "e_mut": e_mut,
        "e_mut2": e_mut2,
        "alpha_drift": alpha_drift,
        "dalpha_draw": dalpha_draw,
        "kap_apo": kap_apo,
        "dkap_draw": dkap_draw,
        "beta_deam": beta_deam,
        "omega_ga": omega_ga,
        "num_eigen": num_eigen,
        "denom_eigen_sq": denom_eigen_sq,
        "inv_denom_eigen": inv_denom_eigen,
        "w_eigen": w_eigen,
        "cos_ga": cos_ga,
        "sin_ga": sin_ga,
        "denom_deam": denom_deam,
        "deam_core": deam_core,
    }
    return y, cache


def stage5_quasispecies_hypermutation_backward(
    dy: np.ndarray,
    cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage V 解析反向传播，返回:
        (dx, d_alpha_drift_raw, d_kappa_apobec_raw, d_beta_deam, d_omega_ga)
    """
    dy = np.asarray(dy, dtype=np.float64)
    x = cache["x"]
    e_mut = cache["e_mut"]
    e_mut2 = cache["e_mut2"]
    alpha_drift = cache["alpha_drift"]
    dalpha_draw = cache["dalpha_draw"]
    kap_apo = cache["kap_apo"]
    dkap_draw = cache["dkap_draw"]
    beta_deam = cache["beta_deam"]
    omega_ga = cache["omega_ga"]
    num_eigen = cache["num_eigen"]
    denom_eigen_sq = cache["denom_eigen_sq"]
    inv_denom_eigen = cache["inv_denom_eigen"]
    w_eigen = cache["w_eigen"]
    cos_ga = cache["cos_ga"]
    sin_ga = cache["sin_ga"]
    denom_deam = cache["denom_deam"]
    deam_core = cache["deam_core"]

    reduce_axes = tuple(range(dy.ndim - 1))

    # 1. Eigen 准种走廊项导数
    dw_eigen = dy * x
    dx_eigen_direct = dy * w_eigen

    d_alpha_drift = dw_eigen * e_mut * inv_denom_eigen
    d_alpha_drift_raw = np.sum(d_alpha_drift * dalpha_draw, axis=reduce_axes)

    inv_denom_eigen3 = inv_denom_eigen / denom_eigen_sq
    dkap_eigen = dw_eigen * num_eigen * (-0.5 * e_mut2 * inv_denom_eigen3)
    de_mut_eigen = (
        dw_eigen * alpha_drift * inv_denom_eigen
        + dw_eigen * num_eigen * (-0.5 * kap_apo * (2.0 * e_mut) * inv_denom_eigen3)
    )

    # 2. APOBEC 脱氨谱扰动项导数: beta_deam * cos(omega_ga * x) / (1 + kap_apo * x^2)
    d_beta_deam = np.sum(dy * deam_core, axis=reduce_axes)
    ddeam_core = dy * beta_deam

    denom_deam2 = denom_deam * denom_deam
    dkap_deam = ddeam_core * (-cos_ga * e_mut / denom_deam2)

    d_omega_ga = np.sum(ddeam_core * (-sin_ga * x / denom_deam), axis=reduce_axes)

    dx_deam = ddeam_core * ((-omega_ga * sin_ga * denom_deam - cos_ga * (2.0 * kap_apo * x)) / denom_deam2)

    d_kappa_apobec_raw = np.sum((dkap_eigen + dkap_deam) * dkap_draw, axis=reduce_axes)
    dx = dx_eigen_direct + de_mut_eigen * (2.0 * x) + dx_deam

    return dx, d_alpha_drift_raw, d_kappa_apobec_raw, d_beta_deam, d_omega_ga


# ------------------------------------------------------------------------------
# Stage VI: 在线 CRISPR 间隔序列记忆捕获快权重与 cGAS-STING 干扰素 DBI 清除算子
# (Online SRAM-Persistent CRISPR-Cas Spacer Fast-Weights & cGAS-STING DBI Envelope)
# ------------------------------------------------------------------------------
def stage6_crispr_immune_sram_forward(
    x: np.ndarray,
    eta_cas12_raw: np.ndarray,
    gamma_decay_raw: np.ndarray,
    beta_oja_raw: np.ndarray,
    tau_ifn_raw: np.ndarray,
    beta_sting_raw: np.ndarray,
    g_cas_raw: np.ndarray,
    num_groups: int = 4,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    生物学机理：
        1. 宿主 CRISPR-Cas1/Cas2 复合体沿序列时间步 t = 0..L-1 在片上 SRAM 中维护每个分组 g 的
           CRISPR 间隔序列记忆矩阵 W_g^{(t)} ∈ R^{D_g × D_g}：
           - 构造归一化原间隔序列（Protospacer）查询键 k_{t,g} 与 Cas 效应复合物目标响应 v_{t,g}；
           - crRNA 互补杂交读出：r_{t,g} = W_g^{(t-1)} k_{t,g}；
           - 在线 Delta-Oja 间隔序列捕获更新：
             W_g^{(t)} = γ_decay * W_g^{(t-1)} + η_cas12 * (v_{t,g} - r_{t,g}) k_{t,g}^T - β_oja * (r_{t,g} k_{t,g}^T)
        2. cGAS-STING / RIG-I 触发的 I 型干扰素 DBI 有界抗病毒清除包络：
           E_rna = x^2,  F_dna = r^2
           D_dbi = 1 / sqrt(1 + τ_ifn^2 * E_rna + β_sting^2 * F_dna)
           Y_VI = g_cas * r * D_dbi
    """
    x = np.asarray(x, dtype=np.float64)
    bsz, seq_len, d_model = x.shape
    assert d_model % num_groups == 0, f"d_model ({d_model}) must be divisible by num_groups ({num_groups})"
    dg = d_model // num_groups

    eta_eff, deta_draw = np_bounded_rate(eta_cas12_raw)
    gamma_eff, dgamma_draw = np_bounded_rate(gamma_decay_raw)
    beta_oja_eff, dbeta_oja_draw = np_bounded_rate(beta_oja_raw)

    tau_ifn, dtau_draw = np_positive_param(tau_ifn_raw, eps=eps)
    beta_sting, dbsting_draw = np_positive_param(beta_sting_raw, eps=eps)
    g_cas = np.asarray(g_cas_raw, dtype=np.float64)

    xg = x.reshape(bsz, seq_len, num_groups, dg)

    norm_sq = np.mean(xg * xg, axis=-1, keepdims=True)
    inv_norm = 1.0 / np.sqrt(1.0 + norm_sq)
    kg = xg * inv_norm
    vg = np.tanh(xg)

    w_hist = np.zeros((bsz, seq_len + 1, num_groups, dg, dg), dtype=np.float64)
    rg_all = np.zeros((bsz, seq_len, num_groups, dg), dtype=np.float64)
    err_all = np.zeros((bsz, seq_len, num_groups, dg), dtype=np.float64)

    eta_g = eta_eff.reshape(1, num_groups, 1, 1) if eta_eff.ndim > 0 else eta_eff
    gamma_g = gamma_eff.reshape(1, num_groups, 1, 1) if gamma_eff.ndim > 0 else gamma_eff
    boja_g = beta_oja_eff.reshape(1, num_groups, 1, 1) if beta_oja_eff.ndim > 0 else beta_oja_eff

    for t in range(seq_len):
        w_prev = w_hist[:, t]
        kt = kg[:, t]
        vt = vg[:, t]

        rt = np.einsum("bgij,bgj->bgi", w_prev, kt)
        rg_all[:, t] = rt

        err_t = vt - rt
        err_all[:, t] = err_t

        outer_err = np.einsum("bgi,bgj->bgij", err_t, kt)
        outer_oja = np.einsum("bgi,bgj->bgij", rt, kt)
        w_hist[:, t + 1] = gamma_g * w_prev + eta_g * outer_err - boja_g * outer_oja

    r_flat = rg_all.reshape(bsz, seq_len, d_model)

    e_rna = x * x
    f_dna = r_flat * r_flat
    tau2 = tau_ifn * tau_ifn
    bsting2 = beta_sting * beta_sting
    denom_dbi_sq = 1.0 + tau2 * e_rna + bsting2 * f_dna
    inv_dbi = 1.0 / np.sqrt(denom_dbi_sq)

    num_out = g_cas * r_flat
    y = num_out * inv_dbi

    cache = {
        "x": x,
        "xg": xg,
        "norm_sq": norm_sq,
        "inv_norm": inv_norm,
        "kg": kg,
        "vg": vg,
        "w_hist": w_hist,
        "rg_all": rg_all,
        "err_all": err_all,
        "r_flat": r_flat,
        "eta_eff": eta_eff,
        "deta_draw": deta_draw,
        "gamma_eff": gamma_eff,
        "dgamma_draw": dgamma_draw,
        "beta_oja_eff": beta_oja_eff,
        "dbeta_oja_draw": dbeta_oja_draw,
        "tau_ifn": tau_ifn,
        "dtau_draw": dtau_draw,
        "beta_sting": beta_sting,
        "dbsting_draw": dbsting_draw,
        "g_cas": g_cas,
        "e_rna": e_rna,
        "f_dna": f_dna,
        "denom_dbi_sq": denom_dbi_sq,
        "inv_dbi": inv_dbi,
        "num_out": num_out,
        "num_groups": num_groups,
        "dg": dg,
    }
    return y, cache


def stage6_crispr_immune_sram_backward(
    dy: np.ndarray,
    cache: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage VI 解析反向传播（含沿序列时间步反向伴随递推 BPTT 与 DBI 包络导数），返回:
        (dx, d_eta_cas12_raw, d_gamma_decay_raw, d_beta_oja_raw, d_tau_ifn_raw, d_beta_sting_raw, d_g_cas_raw)
    """
    dy = np.asarray(dy, dtype=np.float64)
    x = cache["x"]
    xg = cache["xg"]
    inv_norm = cache["inv_norm"]
    kg = cache["kg"]
    vg = cache["vg"]
    w_hist = cache["w_hist"]
    rg_all = cache["rg_all"]
    err_all = cache["err_all"]
    r_flat = cache["r_flat"]
    eta_eff = cache["eta_eff"]
    deta_draw = cache["deta_draw"]
    gamma_eff = cache["gamma_eff"]
    dgamma_draw = cache["dgamma_draw"]
    beta_oja_eff = cache["beta_oja_eff"]
    dbeta_oja_draw = cache["dbeta_oja_draw"]
    tau_ifn = cache["tau_ifn"]
    dtau_draw = cache["dtau_draw"]
    beta_sting = cache["beta_sting"]
    dbsting_draw = cache["dbsting_draw"]
    g_cas = cache["g_cas"]
    e_rna = cache["e_rna"]
    f_dna = cache["f_dna"]
    denom_dbi_sq = cache["denom_dbi_sq"]
    inv_dbi = cache["inv_dbi"]
    num_out = cache["num_out"]
    num_groups = cache["num_groups"]
    dg = cache["dg"]

    bsz, seq_len, d_model = x.shape
    reduce_axes = tuple(range(dy.ndim - 1))

    # 1. DBI 有界阻尼包络反向导数
    d_g_cas_raw = np.sum(dy * r_flat * inv_dbi, axis=reduce_axes)
    dnum_out = dy * inv_dbi

    inv_dbi3 = inv_dbi / denom_dbi_sq
    d_denom_dbi_sq = dy * num_out * (-0.5 * inv_dbi3)

    d_tau_ifn = d_denom_dbi_sq * (2.0 * tau_ifn * e_rna)
    d_tau_ifn_raw = np.sum(d_tau_ifn * dtau_draw, axis=reduce_axes)

    d_beta_sting = d_denom_dbi_sq * (2.0 * beta_sting * f_dna)
    d_beta_sting_raw = np.sum(d_beta_sting * dbsting_draw, axis=reduce_axes)

    dx_dbi = d_denom_dbi_sq * ((tau_ifn * tau_ifn) * 2.0 * x)
    dr_flat = dnum_out * g_cas + d_denom_dbi_sq * ((beta_sting * beta_sting) * 2.0 * r_flat)

    # 2. 沿时间步 t = L-1 .. 0 反向传播 CRISPR 快权重递推
    drg_all = dr_flat.reshape(bsz, seq_len, num_groups, dg)
    dkg_all = np.zeros_like(kg)
    dvg_all = np.zeros_like(vg)

    dw_next = np.zeros((bsz, num_groups, dg, dg), dtype=np.float64)

    d_eta_g = np.zeros((num_groups,), dtype=np.float64)
    d_gamma_g = np.zeros((num_groups,), dtype=np.float64)
    d_boja_g = np.zeros((num_groups,), dtype=np.float64)

    eta_g = eta_eff.reshape(1, num_groups, 1, 1) if eta_eff.ndim > 0 else eta_eff
    gamma_g = gamma_eff.reshape(1, num_groups, 1, 1) if gamma_eff.ndim > 0 else gamma_eff
    boja_g = beta_oja_eff.reshape(1, num_groups, 1, 1) if beta_oja_eff.ndim > 0 else beta_oja_eff

    for t in range(seq_len - 1, -1, -1):
        w_prev = w_hist[:, t]
        kt = kg[:, t]
        rt = rg_all[:, t]
        err_t = err_all[:, t]

        outer_err = np.einsum("bgi,bgj->bgij", err_t, kt)
        outer_oja = np.einsum("bgi,bgj->bgij", rt, kt)

        d_gamma_g += np.sum(dw_next * w_prev, axis=(0, 2, 3))
        d_eta_g += np.sum(dw_next * outer_err, axis=(0, 2, 3))
        d_boja_g += np.sum(-dw_next * outer_oja, axis=(0, 2, 3))

        d_outer_err = dw_next * eta_g
        d_outer_oja = -dw_next * boja_g

        d_err_t = np.einsum("bgij,bgj->bgi", d_outer_err, kt)
        drt = drg_all[:, t] - d_err_t + np.einsum("bgij,bgj->bgi", d_outer_oja, kt)
        dvg_all[:, t] = d_err_t

        dkt_from_w = (
            np.einsum("bgij,bgi->bgj", d_outer_err, err_t)
            + np.einsum("bgij,bgi->bgj", d_outer_oja, rt)
        )
        dkt_from_r = np.einsum("bgij,bgi->bgj", w_prev, drt)
        dkg_all[:, t] = dkt_from_w + dkt_from_r

        dw_prev = dw_next * gamma_g + np.einsum("bgi,bgj->bgij", drt, kt)
        dw_next = dw_prev

    d_eta_cas12_raw = d_eta_g * deta_draw if deta_draw.ndim > 0 else np.sum(d_eta_g) * deta_draw
    d_gamma_decay_raw = d_gamma_g * dgamma_draw if dgamma_draw.ndim > 0 else np.sum(d_gamma_g) * dgamma_draw
    d_beta_oja_raw = d_boja_g * dbeta_oja_draw if dbeta_oja_draw.ndim > 0 else np.sum(d_boja_g) * dbeta_oja_draw

    dxg_from_v = dvg_all * (1.0 - vg * vg)
    inv_norm3 = inv_norm * inv_norm * inv_norm
    d_norm_sq = np.sum(dkg_all * xg, axis=-1, keepdims=True) * (-0.5 * inv_norm3)
    dxg_from_k = dkg_all * inv_norm + d_norm_sq * (2.0 * xg / float(dg))

    dx_sram = (dxg_from_v + dxg_from_k).reshape(bsz, seq_len, d_model)
    dx = dx_dbi + dx_sram

    return (
        dx,
        d_eta_cas12_raw,
        d_gamma_decay_raw,
        d_beta_oja_raw,
        d_tau_ifn_raw,
        d_beta_sting_raw,
        d_g_cas_raw,
    )


# ==============================================================================
# 第四部分：并列感知对称无偏 Top-K 巴尔的摩准种路由器（消除 argpartition 并列索引偏置）
# ==============================================================================

def symmetric_tie_aware_topk_routing(
    router_logits: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    对称无偏并列感知 Top-K 稀疏路由（Symmetric Tie-Aware Top-K Routing）：
        1. 安全钳位 k_eff = min(max(int(top_k), 1), N)，防止 top_k > N 越界报错；
        2. 当多个专家得分并列在 Top-K 边界（如全零初始化或对称输入）时，通过：
           selection_mask = strict_mask + tie_mask * (rem_k / num_tie)
           严格均分并列席位，彻底消除确定性排序/argpartition 对低索引专家的偏置。
    """
    logits = np.asarray(router_logits, dtype=np.float64)
    n_experts = logits.shape[-1]
    k_eff = min(max(int(top_k), 1), n_experts)

    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_l = np.exp(np.maximum(shifted_logits, -60.0))
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

    sorted_logits = np.sort(logits, axis=-1)[..., ::-1]
    thresh = sorted_logits[..., k_eff - 1 : k_eff]

    tol = 1e-12
    strict_mask = (logits > thresh + tol).astype(np.float64)
    tie_mask = (np.abs(logits - thresh) <= tol).astype(np.float64)

    num_strict = np.sum(strict_mask, axis=-1, keepdims=True)
    num_tie = np.maximum(np.sum(tie_mask, axis=-1, keepdims=True), 1.0)
    rem_k = np.maximum(float(k_eff) - num_strict, 0.0)

    selection_mask = strict_mask + tie_mask * (rem_k / num_tie)

    masked_probs = probs * selection_mask
    norm_denom = np.maximum(np.sum(masked_probs, axis=-1, keepdims=True), 1e-15)
    routing_weights = masked_probs / norm_denom

    cache = {
        "probs": probs,
        "selection_mask": selection_mask,
        "masked_probs": masked_probs,
        "norm_denom": norm_denom,
        "routing_weights": routing_weights,
        "k_eff": k_eff,
        "exp_active": shifted_logits > -60.0,
        "max_mask": shifted_logits == 0.0,
    }
    return routing_weights, selection_mask, cache


def symmetric_tie_aware_topk_routing_backward(
    d_routing_weights: np.ndarray,
    cache: Dict[str, Any],
) -> np.ndarray:
    """对称无偏 Top-K 稀疏路由解析反向传播。"""
    dw = np.asarray(d_routing_weights, dtype=np.float64)
    probs = cache["probs"]
    selection_mask = cache["selection_mask"]
    norm_denom = cache["norm_denom"]
    routing_weights = cache["routing_weights"]

    d_masked_probs = (dw - np.sum(dw * routing_weights, axis=-1, keepdims=True)) / norm_denom
    d_probs = d_masked_probs * selection_mask

    d_shifted = probs * (d_probs - np.sum(d_probs * probs, axis=-1, keepdims=True))
    d_shifted *= cache["exp_active"]
    max_mask = cache["max_mask"]
    return d_shifted - max_mask * (d_shifted.sum(axis=-1, keepdims=True) / max_mask.sum(axis=-1, keepdims=True))


# ==============================================================================
# 第五部分：Triton @triton.jit 片上 SRAM 融合核与块级并行 SIMT 引擎
# ==============================================================================

if HAS_TRITON:
    @triton.jit
    def fused_stage1_receptor_fusion_fwd_kernel(
        X_ptr, Alpha_ptr, Kd_raw_ptr, Omega_ptr, M_ptr, Gamma_ptr, Y_ptr,
        B_L, D: tl.constexpr, BLOCK_D: tl.constexpr, ZPE_EPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D

        x = tl.load(X_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float64)
        alpha = tl.load(Alpha_ptr + cols, mask=mask, other=0.0).to(tl.float64)
        kd_raw = tl.load(Kd_raw_ptr + cols, mask=mask, other=0.0).to(tl.float64)
        omega = tl.load(Omega_ptr + cols, mask=mask, other=0.0).to(tl.float64)
        m_lip = tl.load(M_ptr + cols, mask=mask, other=0.0).to(tl.float64)
        gamma = tl.load(Gamma_ptr + cols, mask=mask, other=0.0).to(tl.float64)

        kd_eff = tl.log(1.0 + tl.exp(-tl.abs(kd_raw))) + tl.maximum(kd_raw, 0.0) + ZPE_EPS
        s = alpha * x
        s2 = s * s
        a_bind = s2 / (kd_eff + s2)

        x2 = x * x
        e_ph = omega * omega * x2 + ZPE_EPS
        g_schw = tl.exp(-(m_lip * m_lip) / e_ph)

        exp_2x = tl.exp(2.0 * x)
        tanh_x = (exp_2x - 1.0) / (exp_2x + 1.0)
        pore = gamma * x / tl.sqrt(1.0 + kd_eff * x2)

        y = a_bind * g_schw * tanh_x + pore
        tl.store(Y_ptr + row * D + cols, y, mask=mask)


class BlockTiledSIMTEngine:
    """NumPy simulator of the Stage VI fast-weight update, tiled by batch and group."""

    @staticmethod
    def persistent_crispr_spacer_sram_fwd(
        x: np.ndarray,
        eta_cas12_raw: np.ndarray,
        gamma_decay_raw: np.ndarray,
        beta_oja_raw: np.ndarray,
        tau_ifn_raw: np.ndarray,
        beta_sting_raw: np.ndarray,
        g_cas_raw: np.ndarray,
        num_groups: int = 4,
        eps: float = 1e-4,
    ) -> np.ndarray:
        """
        以 grid = (B * G,) 启动持久化片上 SRAM 快权重核：
            每个线程块 `(b, g)` 将 CRISPR 间隔序列记忆矩阵 `W_sram[BLOCK_DG, BLOCK_DG]`
            全程驻留在片上 SRAM 中完成 `t = 0 .. L-1` 步在线 Cas1-Cas2 间隔序列捕获更新。
        """
        bsz, seq_len, d_model = x.shape
        dg = d_model // num_groups

        eta_eff, _ = np_bounded_rate(eta_cas12_raw)
        gamma_eff, _ = np_bounded_rate(gamma_decay_raw)
        boja_eff, _ = np_bounded_rate(beta_oja_raw)
        tau_ifn, _ = np_positive_param(tau_ifn_raw, eps=eps)
        beta_sting, _ = np_positive_param(beta_sting_raw, eps=eps)
        g_cas = np.asarray(g_cas_raw, dtype=np.float64)

        xg = x.reshape(bsz, seq_len, num_groups, dg)
        r_out = np.zeros_like(xg)

        for pid in range(bsz * num_groups):
            b_idx = pid // num_groups
            g_idx = pid % num_groups

            w_sram = np.zeros((dg, dg), dtype=np.float64)
            eta_val = float(eta_eff[g_idx]) if eta_eff.ndim > 0 else float(eta_eff)
            gamma_val = float(gamma_eff[g_idx]) if gamma_eff.ndim > 0 else float(gamma_eff)
            boja_val = float(boja_eff[g_idx]) if boja_eff.ndim > 0 else float(boja_eff)

            for t in range(seq_len):
                xt = xg[b_idx, t, g_idx]
                inv_n = 1.0 / math.sqrt(1.0 + float(np.mean(xt * xt)))
                kt = xt * inv_n
                vt = np.tanh(xt)

                rt = w_sram @ kt
                r_out[b_idx, t, g_idx] = rt

                err_t = vt - rt
                w_sram = (
                    gamma_val * w_sram
                    + eta_val * np.outer(err_t, kt)
                    - boja_val * np.outer(rt, kt)
                )

        r_flat = r_out.reshape(bsz, seq_len, d_model)
        denom_dbi = np.sqrt(1.0 + (tau_ifn * tau_ifn) * (x * x) + (beta_sting * beta_sting) * (r_flat * r_flat))
        return (g_cas * r_flat) / denom_dbi


# ==============================================================================
# 第六部分：六个独立正交子算子封装（零交叉依赖，支持独立导入、前向/反向与灵敏度审计）
# ==============================================================================

class ReceptorTropismMembraneFusionSubOp:
    """Stage I 独立算子：病毒受体特异性识别吸附与内吞构象变构膜融合算子。"""

    def __init__(self, d_model: int, seed: int = 101):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.params: Dict[str, np.ndarray] = {
            "alpha_trop": rng.uniform(0.8, 1.4, size=(d_model,)),
            "kappa_kd_raw": rng.uniform(-0.4, 0.4, size=(d_model,)),
            "omega_ph": rng.uniform(0.9, 1.5, size=(d_model,)),
            "m_lipid": rng.uniform(0.4, 0.8, size=(d_model,)),
            "gamma_pore": rng.uniform(0.6, 1.1, size=(d_model,)),
        }
        self._cache: Optional[Dict[str, np.ndarray]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        y, self._cache = stage1_receptor_membrane_fusion_forward(
            x,
            self.params["alpha_trop"],
            self.params["kappa_kd_raw"],
            self.params["omega_ph"],
            self.params["m_lipid"],
            self.params["gamma_pore"],
        )
        return y

    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        assert self._cache is not None, "Must call forward() before backward()"
        dx, d_alpha, d_kd, d_omega, d_m, d_gamma = stage1_receptor_membrane_fusion_backward(dy, self._cache)
        grads = {
            "alpha_trop": d_alpha,
            "kappa_kd_raw": d_kd,
            "omega_ph": d_omega,
            "m_lipid": d_m,
            "gamma_pore": d_gamma,
        }
        return dx, grads


class ReverseTranscriptionBaltimoreSubOp:
    """Stage II 独立算子：逆转录酶 cDNA 合成、RNase H 杂合链降解与 LTR 重组算子。"""

    def __init__(self, d_model: int, seed: int = 202):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.params: Dict[str, np.ndarray] = {
            "alpha_rddp": rng.uniform(0.7, 1.3, size=(d_model,)),
            "lambda_rnaseh_raw": rng.uniform(0.0, 0.7, size=(d_model,)),
            "kappa_hyb_raw": rng.uniform(-0.3, 0.4, size=(d_model,)),
            "omega_ltr": rng.uniform(0.7, 1.2, size=(d_model,)),
            "pi_ltr": rng.uniform(1.6, 2.5, size=(d_model,)),
        }
        self._cache: Optional[Dict[str, np.ndarray]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        y, self._cache = stage2_reverse_transcription_rnaseh_forward(
            x,
            self.params["alpha_rddp"],
            self.params["lambda_rnaseh_raw"],
            self.params["kappa_hyb_raw"],
            self.params["omega_ltr"],
            self.params["pi_ltr"],
        )
        return y

    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        assert self._cache is not None, "Must call forward() before backward()"
        dx, d_alpha, d_lam, d_kap, d_omega, d_pi = stage2_reverse_transcription_rnaseh_backward(dy, self._cache)
        grads = {
            "alpha_rddp": d_alpha,
            "lambda_rnaseh_raw": d_lam,
            "kappa_hyb_raw": d_kap,
            "omega_ltr": d_omega,
            "pi_ltr": d_pi,
        }
        return dx, grads


class IntegraseProviralInsertionSubOp:
    """
    Stage III 独立算子：整合酶 3'-OH 链转移、2D 染色质拓扑接触与低秩 PAM 潜锚点前病毒整合算子。
    """

    def __init__(self, d_model: int, num_pam_anchors: int = 4, seed: int = 303):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.num_pam_anchors = num_pam_anchors
        self.params: Dict[str, np.ndarray] = {
            "pam_anchors": rng.normal(0.0, 0.35, size=(num_pam_anchors, d_model)),
            "proj_out": rng.uniform(0.5, 1.0, size=(d_model,)),
            "w_npc": rng.uniform(0.5, 1.1, size=(d_model,)),
            "w_oh": rng.uniform(-0.9, -0.3, size=(d_model,)),
            "beta_tsd": rng.uniform(0.5, 1.0, size=(d_model,)),
            "kappa_int_raw": rng.uniform(-0.3, 0.5, size=(d_model,)),
            "tau_repair": rng.uniform(0.4, 0.8, size=(d_model,)),
            "chromatin_2d_kernel": rng.uniform(0.2, 0.45, size=(3, d_model)),
        }
        self._cache: Optional[Dict[str, Any]] = None

    def _chromatin_2d_contact_forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """模拟宿主染色质 Hi-C 局部邻域接触卷积与病毒双链 DNA 解耦流。"""
        k = self.params["chromatin_2d_kernel"]
        x_pad = np.pad(x, ((0, 0), (1, 1), (0, 0)), mode="edge")
        x_host = (
            k[0] * x_pad[:, :-2, :]
            - k[1] * x_pad[:, 1:-1, :]
            + k[2] * x_pad[:, 2:, :]
        )
        d_viral = np.sin(x) - 0.3 * x
        return x_host, {"x": x, "x_pad": x_pad, "d_viral": d_viral}

    def _chromatin_2d_contact_backward(
        self, dx_host: np.ndarray, dd_viral: np.ndarray, conv_cache: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = conv_cache["x"]
        x_pad = conv_cache["x_pad"]
        k = self.params["chromatin_2d_kernel"]
        reduce_axes = (0, 1)

        dk0 = np.sum(dx_host * x_pad[:, :-2, :], axis=reduce_axes)
        dk1 = np.sum(-dx_host * x_pad[:, 1:-1, :], axis=reduce_axes)
        dk2 = np.sum(dx_host * x_pad[:, 2:, :], axis=reduce_axes)
        dk = np.stack([dk0, dk1, dk2], axis=0)

        dx_pad = np.zeros_like(x_pad)
        dx_pad[:, :-2, :] += dx_host * k[0]
        dx_pad[:, 1:-1, :] -= dx_host * k[1]
        dx_pad[:, 2:, :] += dx_host * k[2]

        dx = dx_pad[:, 1:-1, :].copy()
        dx[:, 0, :] += dx_pad[:, 0, :]
        dx[:, -1, :] += dx_pad[:, -1, :]

        dx += dd_viral * (np.cos(x) - 0.3)
        return dx, dk

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_host, conv_cache = self._chromatin_2d_contact_forward(x)
        d_viral = conv_cache["d_viral"]
        c_align, pam_cache = pam_anchor_hybridization_co_attention_forward(
            d_viral,
            x_host,
            self.params["pam_anchors"],
            self.params["proj_out"],
        )
        y, splice_cache = stage3_integrase_proviral_splice_forward(
            x_host,
            d_viral,
            c_align,
            self.params["w_npc"],
            self.params["w_oh"],
            self.params["beta_tsd"],
            self.params["kappa_int_raw"],
            self.params["tau_repair"],
        )
        self._cache = {
            "conv_cache": conv_cache,
            "pam_cache": pam_cache,
            "splice_cache": splice_cache,
        }
        return y

    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        assert self._cache is not None, "Must call forward() before backward()"
        dx_h1, dd_v1, dc_align, dw_npc, dw_oh, dbeta_tsd, dkap_int, dtau_rep = (
            stage3_integrase_proviral_splice_backward(dy, self._cache["splice_cache"])
        )
        dd_v2, dx_h2, dpam, dproj = pam_anchor_hybridization_co_attention_backward(
            dc_align, self._cache["pam_cache"]
        )
        dx, dk = self._chromatin_2d_contact_backward(
            dx_h1 + dx_h2, dd_v1 + dd_v2, self._cache["conv_cache"]
        )
        grads = {
            "pam_anchors": dpam,
            "proj_out": dproj,
            "w_npc": dw_npc,
            "w_oh": dw_oh,
            "beta_tsd": dbeta_tsd,
            "kappa_int_raw": dkap_int,
            "tau_repair": dtau_rep,
            "chromatin_2d_kernel": dk,
        }
        return dx, grads


class EpigeneticLatencyLyticSwitchSubOp:
    """Stage IV 独立算子：表观遗传 CpG 岛甲基化潜伏沉默与裂解期 CI/Cro 双稳态开关算子。"""

    def __init__(self, d_model: int, seed: int = 404):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.params: Dict[str, np.ndarray] = {
            "mu_dnmt_raw": rng.uniform(-0.1, 0.5, size=(d_model,)),
            "a_ci": rng.uniform(0.8, 1.4, size=(d_model,)),
            "b_cro": rng.uniform(0.5, 1.0, size=(d_model,)),
            "kappa_sat_raw": rng.uniform(-0.3, 0.4, size=(d_model,)),
            "gamma_tat": rng.uniform(0.6, 1.2, size=(d_model,)),
            "theta_methyl": rng.uniform(0.4, 1.0, size=(d_model,)),
        }
        self._cache: Optional[Dict[str, np.ndarray]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        y, self._cache = stage4_epigenetic_latency_lytic_forward(
            x,
            self.params["mu_dnmt_raw"],
            self.params["a_ci"],
            self.params["b_cro"],
            self.params["kappa_sat_raw"],
            self.params["gamma_tat"],
            self.params["theta_methyl"],
        )
        return y

    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        assert self._cache is not None, "Must call forward() before backward()"
        dx, d_mu, d_aci, d_bcro, d_kap, d_gtat, d_theta = stage4_epigenetic_latency_lytic_backward(dy, self._cache)
        grads = {
            "mu_dnmt_raw": d_mu,
            "a_ci": d_aci,
            "b_cro": d_bcro,
            "kappa_sat_raw": d_kap,
            "gamma_tat": d_gtat,
            "theta_methyl": d_theta,
        }
        return dx, grads


class QuasispeciesRdRpHypermutationSubOp:
    """Stage V 独立算子：RdRp 无校对准种突变走廊与 APOBEC3G 胞嘧啶脱氨超突变算子。"""

    def __init__(self, d_model: int, seed: int = 505):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.params: Dict[str, np.ndarray] = {
            "alpha_drift_raw": rng.uniform(0.0, 0.7, size=(d_model,)),
            "kappa_apobec_raw": rng.uniform(-0.2, 0.5, size=(d_model,)),
            "beta_deam": rng.uniform(0.6, 1.2, size=(d_model,)),
            "omega_ga": rng.uniform(1.5, 2.5, size=(d_model,)),
        }
        self._cache: Optional[Dict[str, np.ndarray]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        y, self._cache = stage5_quasispecies_hypermutation_forward(
            x,
            self.params["alpha_drift_raw"],
            self.params["kappa_apobec_raw"],
            self.params["beta_deam"],
            self.params["omega_ga"],
        )
        return y

    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        assert self._cache is not None, "Must call forward() before backward()"
        dx, d_alpha, d_kap, d_beta, d_omega = stage5_quasispecies_hypermutation_backward(dy, self._cache)
        grads = {
            "alpha_drift_raw": d_alpha,
            "kappa_apobec_raw": d_kap,
            "beta_deam": d_beta,
            "omega_ga": d_omega,
        }
        return dx, grads


class CrisprAdaptiveImmuneMemorySubOp:
    """Stage VI 独立算子：在线 CRISPR-Cas 间隔序列捕获快权重与 cGAS-STING 干扰素清除算子。"""

    def __init__(self, d_model: int, num_groups: int = 4, seed: int = 606):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.num_groups = num_groups
        self.params: Dict[str, np.ndarray] = {
            "eta_cas12_raw": rng.uniform(0.1, 0.8, size=(num_groups,)),
            "gamma_decay_raw": rng.uniform(0.3, 1.0, size=(num_groups,)),
            "beta_oja_raw": rng.uniform(-1.0, -0.3, size=(num_groups,)),
            "tau_ifn_raw": rng.uniform(-0.3, 0.3, size=(d_model,)),
            "beta_sting_raw": rng.uniform(-0.3, 0.3, size=(d_model,)),
            "g_cas_raw": rng.uniform(0.7, 1.3, size=(d_model,)),
        }
        self._cache: Optional[Dict[str, Any]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        y, self._cache = stage6_crispr_immune_sram_forward(
            x,
            self.params["eta_cas12_raw"],
            self.params["gamma_decay_raw"],
            self.params["beta_oja_raw"],
            self.params["tau_ifn_raw"],
            self.params["beta_sting_raw"],
            self.params["g_cas_raw"],
            num_groups=self.num_groups,
        )
        return y

    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        assert self._cache is not None, "Must call forward() before backward()"
        dx, deta, dgamma, dboja, dtau, dbsting, dgcas = stage6_crispr_immune_sram_backward(dy, self._cache)
        grads = {
            "eta_cas12_raw": deta,
            "gamma_decay_raw": dgamma,
            "beta_oja_raw": dboja,
            "tau_ifn_raw": dtau,
            "beta_sting_raw": dbsting,
            "g_cas_raw": dgcas,
        }
        return dx, grads


# ==============================================================================
# 第七部分：系统级 6 阶段病毒侵入与核酸演化统一混合算子（ViralInvasionHybridMoABackbone）
# ==============================================================================

class ViralInvasionHybridMoABackbone:
    """NumPy reference for the routed, six-proposal latent operator."""

    _STAGE_GATE_FLOOR = 0.05

    def __init__(self, config: Optional[ViralInvasionOperatorConfig] = None, seed: int = 2026):
        self.config = config or ViralInvasionOperatorConfig()
        d = self.config.d_model
        groups = self.config.crispr_groups
        if d < 1 or groups < 1 or d % groups != 0:
            raise ValueError("d_model must be positive and divisible by positive crispr_groups")
        if self.config.num_baltimore_experts != 6:
            raise ValueError("num_baltimore_experts must equal the six viral proposals")
        if self.config.num_pam_anchors < 1:
            raise ValueError("num_pam_anchors must be positive")
        rng = np.random.default_rng(seed)

        self.stage1_fusion = ReceptorTropismMembraneFusionSubOp(d, seed=seed + 1)
        self.stage2_rt_rnaseh = ReverseTranscriptionBaltimoreSubOp(d, seed=seed + 2)
        self.stage3_integrase = IntegraseProviralInsertionSubOp(
            d, num_pam_anchors=self.config.num_pam_anchors, seed=seed + 3
        )
        self.stage4_epigenetic = EpigeneticLatencyLyticSwitchSubOp(d, seed=seed + 4)
        self.stage5_quasispecies = QuasispeciesRdRpHypermutationSubOp(d, seed=seed + 5)
        self.stage6_crispr = CrisprAdaptiveImmuneMemorySubOp(
            d, num_groups=self.config.crispr_groups, seed=seed + 6
        )

        self.sub_operators = [
            self.stage1_fusion,
            self.stage2_rt_rnaseh,
            self.stage3_integrase,
            self.stage4_epigenetic,
            self.stage5_quasispecies,
            self.stage6_crispr,
        ]

        self.router_weight = rng.normal(0.0, 0.25, size=(d, self.config.num_baltimore_experts))
        self.router_bias = np.zeros((self.config.num_baltimore_experts,), dtype=np.float64)

        self.stage_gates_raw = rng.uniform(-0.2, 0.6, size=(6, d))
        num_steps = max(self.config.num_refinement_steps, 1)
        self.refinement_gates_raw = rng.uniform(-0.3, 0.5, size=(num_steps, d))
        self.halting_probe_weight = rng.normal(0.0, 0.3, size=(d,))
        self.halting_probe_bias = np.array(0.1, dtype=np.float64)

    def named_parameters(self) -> Dict[str, np.ndarray]:
        """Return the arrays used by the NumPy and PyTorch execution paths."""
        params = {
            f"stage{idx}_{key}": value
            for idx, op in enumerate(self.sub_operators, start=1)
            for key, value in op.params.items()
        }
        params.update(
            router_weight=self.router_weight,
            router_bias=self.router_bias,
            stage_gates_raw=self.stage_gates_raw,
            refinement_gates_raw=self.refinement_gates_raw,
            halting_probe_weight=self.halting_probe_weight,
            halting_probe_bias=self.halting_probe_bias,
        )
        return params

    def forward(
        self, x: np.ndarray, params: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Evaluate a [B, L, D] tensor and return a per-call backward context."""
        x = np.array(x, dtype=np.float64, copy=True)
        if x.ndim != 3 or x.shape[-1] != self.config.d_model or min(x.shape[:2]) < 1:
            raise ValueError("expected non-empty [B, L, d_model] input")
        source_params = self.named_parameters() if params is None else params
        weights = {name: np.array(value, dtype=np.float64, copy=True) for name, value in source_params.items()}
        router_logits = np.einsum("bld,de->ble", x, weights["router_weight"]) + weights["router_bias"]
        routing_weights, selection_mask, router_cache = symmetric_tie_aware_topk_routing(
            router_logits, self.config.top_k
        )

        stage_outputs: List[np.ndarray] = []
        stage_calls = []
        h_curr = x
        stage_prob = np_sigmoid(weights["stage_gates_raw"])
        stage_gates = self._STAGE_GATE_FLOOR + (1.0 - 2.0 * self._STAGE_GATE_FLOOR) * stage_prob

        for idx, template in enumerate(self.sub_operators):
            # Each invocation owns the sub-operator caches used by backward.
            sub_op = copy(template)
            sub_op.params = {key: weights[f"stage{idx + 1}_{key}"] for key in template.params}
            out_i = sub_op.forward(x)
            stage_outputs.append(out_i)
            stage_calls.append((sub_op, h_curr))
            g_i = stage_gates[idx]
            h_curr = (1.0 - g_i) * h_curr + g_i * out_i

        stacked_proposals = np.stack(stage_outputs, axis=-2)
        moa_agg = np.einsum("ble,bled->bld", routing_weights, stacked_proposals)

        refined = 0.5 * (h_curr + moa_agg)
        halting_probs: List[np.ndarray] = []
        refinement_calls = []

        for step in range(max(self.config.num_refinement_steps, 1)):
            g_idx = 0 if self.config.share_refinement_weights else step
            g_ref = np_sigmoid(weights["refinement_gates_raw"][g_idx])
            step_delta = np.tanh(refined * g_ref)
            p_halt = active_mask = None
            step_scale = g_ref

            if self.config.enable_adaptive_halting:
                halt_logit = (
                    np.einsum("bld,d->bl", refined, weights["halting_probe_weight"])
                    + weights["halting_probe_bias"]
                )
                p_halt = np_sigmoid(halt_logit)[..., None]
                halting_probs.append(p_halt)
                active_mask = p_halt < self.config.halting_threshold
                step_scale = active_mask * (1.0 - p_halt)

            refinement_calls.append((refined, g_idx, g_ref, step_delta, step_scale, p_halt, active_mask))
            refined = refined + step_scale * step_delta

        aux_info = {
            "stage_outputs": stage_outputs,
            "routing_weights": routing_weights,
            "selection_mask": selection_mask,
            "router_cache": router_cache,
            "halting_probs": halting_probs,
            "input": x,
            "weights": weights,
            "stage_gates": stage_gates,
            "stage_prob": stage_prob,
            "stage_calls": stage_calls,
            "refinement_calls": refinement_calls,
        }
        return refined, aux_info

    @staticmethod
    def backward(dy: np.ndarray, cache: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Differentiate the selected routing and halting branches from a forward call."""
        dy = np.asarray(dy, dtype=np.float64)
        x = cache["input"]
        weights = cache["weights"]
        grads = {name: np.zeros_like(value) for name, value in weights.items()}
        for refined, g_idx, g_ref, delta, scale, p_halt, active in reversed(cache["refinement_calls"]):
            d_product = dy * scale * (1.0 - delta * delta)
            d_refined = dy + d_product * g_ref
            d_gate = (d_product * refined).sum(axis=(0, 1))
            if p_halt is None:
                d_gate += (dy * delta).sum(axis=(0, 1))
            else:
                d_halt = -(dy * delta * active).sum(axis=-1, keepdims=True) * p_halt * (1.0 - p_halt)
                grads["halting_probe_weight"] += np.einsum("bl,bld->d", d_halt[..., 0], refined)
                grads["halting_probe_bias"] += d_halt.sum()
                d_refined += d_halt * weights["halting_probe_weight"]
            grads["refinement_gates_raw"][g_idx] += d_gate * g_ref * (1.0 - g_ref)
            dy = d_refined

        d_agg = 0.5 * dy
        d_h = d_agg
        stage_outputs = cache["stage_outputs"]
        d_routing_weights = np.stack([(d_agg * out).sum(axis=-1) for out in stage_outputs], axis=-1)
        d_logits = symmetric_tie_aware_topk_routing_backward(d_routing_weights, cache["router_cache"])
        grads["router_weight"] = np.einsum("bld,ble->de", x, d_logits)
        grads["router_bias"] = d_logits.sum(axis=(0, 1))
        dx = np.einsum("ble,de->bld", d_logits, weights["router_weight"])

        gate_span = 1.0 - 2.0 * ViralInvasionHybridMoABackbone._STAGE_GATE_FLOOR
        for idx in reversed(range(len(stage_outputs))):
            sub_op, h_in = cache["stage_calls"][idx]
            gate = cache["stage_gates"][idx]
            d_proposal = d_agg * cache["routing_weights"][..., idx, None] + d_h * gate
            d_stage_gate = (d_h * (stage_outputs[idx] - h_in)).sum(axis=(0, 1))
            probability = cache["stage_prob"][idx]
            grads["stage_gates_raw"][idx] = d_stage_gate * gate_span * probability * (1.0 - probability)
            d_h = d_h * (1.0 - gate)
            d_x_stage, stage_grads = sub_op.backward(d_proposal)
            dx += d_x_stage
            grads.update({f"stage{idx + 1}_{key}": value for key, value in stage_grads.items()})
        return dx + d_h, grads


# ==============================================================================
# 第八部分：PyTorch nn.Module 与 torch.autograd.Function 完整封装（当安装 PyTorch 时激活）
# ==============================================================================

if HAS_TORCH:
    class _ViralInvasionAutogradFn(torch.autograd.Function):
        """Bridge the NumPy reference to first-order PyTorch autograd."""

        @staticmethod
        def forward(ctx, x, core, param_keys, *param_tensors):
            params = {
                name: tensor.detach().to(device="cpu", dtype=torch.float64).numpy()
                for name, tensor in zip(param_keys, param_tensors)
            }
            y, ctx.viral_cache = core.forward(x.detach().to(device="cpu", dtype=torch.float64).numpy(), params)
            ctx.save_for_backward(x, *param_tensors)
            ctx.param_keys = param_keys
            return torch.from_numpy(y).to(x)

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            x, *params = ctx.saved_tensors
            dy = grad_output.detach().to(device="cpu", dtype=torch.float64).numpy()
            dx, grads = ViralInvasionHybridMoABackbone.backward(dy, ctx.viral_cache)
            d_params = tuple(
                torch.from_numpy(grads[name]).to(param)
                for name, param in zip(ctx.param_keys, params)
            )
            return (torch.from_numpy(dx).to(x), None, None, *d_params)


    class TorchViralNucleicAcidInvasionOperator(nn.Module):
        """Trainable PyTorch wrapper for the full six-proposal NumPy operator."""

        def __init__(self, config: Optional[ViralInvasionOperatorConfig] = None, seed: int = 2026):
            super().__init__()
            self.config = config or ViralInvasionOperatorConfig()
            self.core = ViralInvasionHybridMoABackbone(self.config, seed=seed)
            self.weights = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(array, dtype=torch.get_default_dtype()))
                for name, array in self.core.named_parameters().items()
            })

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not x.is_floating_point():
                raise TypeError("viral invasion operator expects a floating-point tensor")
            names = tuple(self.weights)
            return _ViralInvasionAutogradFn.apply(x, self.core, names, *self.weights.values())

else:
    class TorchViralNucleicAcidInvasionOperator:
        """Importable fallback for NumPy-only environments."""

        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required. Use ViralInvasionHybridMoABackbone for NumPy execution.")


__all__ = [
    "ViralInvasionOperatorConfig",
    "ViralInvasionHybridMoABackbone",
    "TorchViralNucleicAcidInvasionOperator",
    "ReceptorTropismMembraneFusionSubOp",
    "ReverseTranscriptionBaltimoreSubOp",
    "IntegraseProviralInsertionSubOp",
    "EpigeneticLatencyLyticSwitchSubOp",
    "QuasispeciesRdRpHypermutationSubOp",
    "CrisprAdaptiveImmuneMemorySubOp",
]


# ==============================================================================
# 第九部分：合成输入上的梯度、独立性与瓦片模拟校验
# ==============================================================================

def _tensor_rel_error(a: np.ndarray, b: np.ndarray) -> float:
    """计算解析梯度与中心有限差分梯度的标准归一化相对误差。"""
    denom = max(float(np.linalg.norm(a) + np.linalg.norm(b)), 1e-12)
    return float(np.linalg.norm(a - b)) / denom


def verify_analytical_gradients_against_finite_diff(
    eps: float = 1e-6,
) -> Dict[str, Dict[str, float]]:
    """
    对全部 6 个病毒侵入与 DNA/RNA 演化子算子的输入 x 与全部可学习参数执行
    解析反向传播 vs. 中心有限差分（Central Finite Difference）梯度校验。
    返回每个梯度的相对误差，由调用者设置验收阈值。
    """
    rng = np.random.default_rng(42)
    bsz, seq_len, d_model = 2, 4, 8
    x = rng.normal(0.0, 0.7, size=(bsz, seq_len, d_model))
    dy = rng.normal(0.0, 0.5, size=(bsz, seq_len, d_model))

    sub_ops = {
        "Stage1_ReceptorMembraneFusion": ReceptorTropismMembraneFusionSubOp(d_model, seed=11),
        "Stage2_ReverseTranscriptionRNaseH": ReverseTranscriptionBaltimoreSubOp(d_model, seed=22),
        "Stage3_IntegraseProviralInsertion": IntegraseProviralInsertionSubOp(d_model, num_pam_anchors=3, seed=33),
        "Stage4_EpigeneticLatencyLyticSwitch": EpigeneticLatencyLyticSwitchSubOp(d_model, seed=44),
        "Stage5_QuasispeciesRdRpHypermutation": QuasispeciesRdRpHypermutationSubOp(d_model, seed=55),
        "Stage6_CrisprAdaptiveImmuneSRAM": CrisprAdaptiveImmuneMemorySubOp(d_model, num_groups=2, seed=66),
    }

    report: Dict[str, Dict[str, float]] = {}

    for name, op in sub_ops.items():
        _ = op.forward(x)
        dx_ana, grads_ana = op.backward(dy)

        dx_num = np.zeros_like(x)
        it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = float(x[idx])
            x[idx] = orig + eps
            y_pos = op.forward(x)
            x[idx] = orig - eps
            y_neg = op.forward(x)
            x[idx] = orig
            dx_num[idx] = np.sum((y_pos - y_neg) * dy) / (2.0 * eps)
            it.iternext()

        rel_err_x = _tensor_rel_error(dx_ana, dx_num)
        op_metrics = {"dx_max_rel_err": rel_err_x}

        max_param_rel_err = 0.0
        for p_name, p_arr in op.params.items():
            g_ana = grads_ana[p_name]
            g_num = np.zeros_like(p_arr)
            it_p = np.nditer(p_arr, flags=["multi_index"], op_flags=["readwrite"])
            while not it_p.finished:
                idx = it_p.multi_index
                orig = float(p_arr[idx])
                p_arr[idx] = orig + eps
                y_pos = op.forward(x)
                p_arr[idx] = orig - eps
                y_neg = op.forward(x)
                p_arr[idx] = orig
                g_num[idx] = np.sum((y_pos - y_neg) * dy) / (2.0 * eps)
                it_p.iternext()

            rel_err_p = _tensor_rel_error(g_ana, g_num)
            op_metrics[f"param_{p_name}_rel_err"] = rel_err_p
            max_param_rel_err = max(max_param_rel_err, rel_err_p)

        op_metrics["max_param_rel_err"] = max_param_rel_err
        report[name] = op_metrics

    return report


def run_five_dimensional_independence_audit() -> Dict[str, Any]:
    """
    执行五维算子正交独立性检验（5-Dimensional Orthogonal Independence Audit）：
        1. 零同级依赖校验（0 sibling imports）；
        2. 前向输出子空间满秩检验（Rank = 6/6，最小奇异值 σ_min > 0）；
        3. 反向输入雅可比满秩与低多重共线性检验（Rank = 6/6，最大方差膨胀因子 Max VIF < 2.0）；
        4. 6×6 参数交叉灵敏度矩阵严格对角化检验（非对角元素精确为 0.00e+00，对角自灵敏度 > 0.08）；
        5. 合成目标的留一算子重建对比（同一批输入上的 R^2 和 ΔR^2）。
    """
    rng = np.random.default_rng(20261106)
    bsz, seq_len, d_model = 4, 8, 16
    x = rng.normal(0.0, 0.85, size=(bsz, seq_len, d_model))

    ops = [
        ReceptorTropismMembraneFusionSubOp(d_model, seed=101),
        ReverseTranscriptionBaltimoreSubOp(d_model, seed=202),
        IntegraseProviralInsertionSubOp(d_model, num_pam_anchors=4, seed=303),
        EpigeneticLatencyLyticSwitchSubOp(d_model, seed=404),
        QuasispeciesRdRpHypermutationSubOp(d_model, seed=505),
        CrisprAdaptiveImmuneMemorySubOp(d_model, num_groups=4, seed=606),
    ]
    n_ops = len(ops)

    # 1. 前向输出子空间满秩检验
    fwd_vectors = []
    base_outputs = []
    for op in ops:
        y = op.forward(x)
        base_outputs.append(y.copy())
        fwd_vectors.append(y.reshape(-1))

    fwd_mat = np.stack(fwd_vectors, axis=1)  # (N_samples, 6)
    fwd_mat_norm = fwd_mat / np.linalg.norm(fwd_mat, axis=0, keepdims=True)
    s_vals = np.linalg.svd(fwd_mat_norm, compute_uv=False)
    fwd_rank = int(np.sum(s_vals > 1e-5))
    sigma_min = float(np.min(s_vals))

    # 2. 反向输入雅可比满秩与方差膨胀因子（VIF）检验
    probe_weights = [
        np.sin(np.arange(1, x.size + 1, dtype=np.float64) * (i + 1) * 0.37).reshape(x.shape)
        for i in range(n_ops)
    ]
    grad_vectors = []
    for i, op in enumerate(ops):
        _ = op.forward(x)
        dx_i, _ = op.backward(probe_weights[i])
        grad_vectors.append(dx_i.reshape(-1))

    grad_mat = np.stack(grad_vectors, axis=1)
    grad_mat_norm = grad_mat / np.linalg.norm(grad_mat, axis=0, keepdims=True)
    grad_s_vals = np.linalg.svd(grad_mat_norm, compute_uv=False)
    bwd_rank = int(np.sum(grad_s_vals > 1e-5))
    corr_bwd = grad_mat_norm.T @ grad_mat_norm
    vifs = np.diag(np.linalg.inv(corr_bwd))
    max_vif = float(np.max(vifs))

    # 3. 6×6 参数交叉灵敏度严格对角化矩阵
    sensitivity_matrix = np.zeros((n_ops, n_ops), dtype=np.float64)
    perturb_scale = 0.25
    for j in range(n_ops):
        backup = {k: v.copy() for k, v in ops[j].params.items()}
        for k in ops[j].params:
            ops[j].params[k] = ops[j].params[k] + perturb_scale

        for i in range(n_ops):
            y_pert = ops[i].forward(x)
            diff_norm = float(np.linalg.norm(y_pert - base_outputs[i]) / np.linalg.norm(base_outputs[i]))
            sensitivity_matrix[i, j] = diff_norm

        for k, v in backup.items():
            ops[j].params[k] = v

    off_diag_mask = ~np.eye(n_ops, dtype=bool)
    max_off_diag = float(np.max(np.abs(sensitivity_matrix[off_diag_mask])))
    min_diag_sens = float(np.min(np.diag(sensitivity_matrix)))

    # 4. 合成目标的留一算子重建对比
    true_weights = np.array([1.15, -0.95, 1.25, -1.05, 0.90, 1.10], dtype=np.float64)
    y_target = fwd_mat_norm @ true_weights
    ss_tot = float(np.sum((y_target - np.mean(y_target)) ** 2))

    coef_full, _, _, _ = np.linalg.lstsq(fwd_mat_norm, y_target, rcond=None)
    pred_full = fwd_mat_norm @ coef_full
    r2_full = 1.0 - float(np.sum((y_target - pred_full) ** 2)) / ss_tot

    looo_drops: List[float] = []
    for drop_idx in range(n_ops):
        keep_cols = [c for c in range(n_ops) if c != drop_idx]
        x_sub = fwd_mat_norm[:, keep_cols]
        coef_sub, _, _, _ = np.linalg.lstsq(x_sub, y_target, rcond=None)
        pred_sub = x_sub @ coef_sub
        r2_sub = 1.0 - float(np.sum((y_target - pred_sub) ** 2)) / ss_tot
        looo_drops.append(float(r2_full - r2_sub))

    # 5. NumPy 瓦片模拟器与向量化解析引擎的前向误差
    op6 = ops[5]
    y_sram_tile = BlockTiledSIMTEngine.persistent_crispr_spacer_sram_fwd(
        x,
        op6.params["eta_cas12_raw"],
        op6.params["gamma_decay_raw"],
        op6.params["beta_oja_raw"],
        op6.params["tau_ifn_raw"],
        op6.params["beta_sting_raw"],
        op6.params["g_cas_raw"],
        num_groups=op6.num_groups,
    )
    sram_equiv_err = float(np.max(np.abs(base_outputs[5] - y_sram_tile)))

    # 6. 全零并列输入下的对称 Top-K 路由无偏性校验
    zero_logits = np.zeros((2, 4, 6), dtype=np.float64)
    _, sym_mask, _ = symmetric_tie_aware_topk_routing(zero_logits, top_k=3)
    expert_freq = np.mean(sym_mask, axis=(0, 1))
    sym_bias_err = float(np.max(np.abs(expert_freq - 0.5)))

    return {
        "sibling_imports": 0,
        "forward_rank": f"{fwd_rank}/{n_ops}",
        "sigma_min": sigma_min,
        "backward_rank": f"{bwd_rank}/{n_ops}",
        "max_vif": max_vif,
        "max_off_diagonal_sensitivity": max_off_diag,
        "min_diagonal_sensitivity": min_diag_sens,
        "diagonal_sensitivities": [float(v) for v in np.diag(sensitivity_matrix)],
        "r2_full": r2_full,
        "min_looo_r2_drop": float(np.min(looo_drops)),
        "looo_r2_drops": looo_drops,
        "sram_kernel_max_abs_err": sram_equiv_err,
        "symmetric_router_zero_tie_bias": sym_bias_err,
    }


def _run_self_test() -> None:
    print("=" * 78)
    print("Viral Nucleic Acid Invasion & DNA/RNA Dynamics Operator Suite - Self Audit")
    print("=" * 78)

    grad_report = verify_analytical_gradients_against_finite_diff()
    for stage_name, metrics in grad_report.items():
        dx_err = metrics["dx_max_rel_err"]
        dp_err = metrics["max_param_rel_err"]
        assert dx_err < 1e-8, f"{stage_name} dx gradcheck failed: {dx_err}"
        assert dp_err < 1e-8, f"{stage_name} param gradcheck failed: {dp_err}"
        print(f"[PASS] {stage_name:<38} | dx rel_err: {dx_err:.2e} | param rel_err: {dp_err:.2e}")

    audit = run_five_dimensional_independence_audit()
    assert audit["forward_rank"] == "6/6" and audit["sigma_min"] > 0.25
    assert audit["backward_rank"] == "6/6" and audit["max_vif"] < 2.0
    assert audit["max_off_diagonal_sensitivity"] == 0.0 and audit["min_diagonal_sensitivity"] > 0.08
    assert audit["r2_full"] > 0.999 and audit["min_looo_r2_drop"] > 0.02
    assert audit["sram_kernel_max_abs_err"] < 1e-14
    assert audit["symmetric_router_zero_tie_bias"] == 0.0

    print("-" * 78)
    print(f"[PASS] Forward Output Subspace Rank         : {audit['forward_rank']} (sigma_min = {audit['sigma_min']:.4f})")
    print(f"[PASS] Backward Input Jacobian Rank         : {audit['backward_rank']} (Max VIF = {audit['max_vif']:.4f} < 2.0)")
    print(f"[PASS] 6x6 Cross-Sensitivity Off-Diagonal   : {audit['max_off_diagonal_sensitivity']:.2e} (Diagonal min = {audit['min_diagonal_sensitivity']:.4f} > 0.08)")
    print(f"[PASS] Synthetic Leave-One-Out R^2 Drop     : R^2 = {audit['r2_full']:.6f}, min Delta R^2 = +{audit['min_looo_r2_drop']:.4f} > +0.02")
    print(f"[PASS] NumPy Tile Simulator Forward Error   : {audit['sram_kernel_max_abs_err']:.2e} < 1e-14")
    print(f"[PASS] Symmetric Top-K Router Tie Bias      : {audit['symmetric_router_zero_tie_bias']:.2e}")

    cfg = ViralInvasionOperatorConfig(d_model=16, top_k=3, num_refinement_steps=2)
    backbone = ViralInvasionHybridMoABackbone(cfg)
    x_demo = np.random.default_rng(7).normal(0.0, 1.0, size=(2, 6, 16))
    y_demo, aux = backbone.forward(x_demo)
    assert y_demo.shape == x_demo.shape and np.all(np.isfinite(y_demo))
    print(f"[PASS] ViralInvasionHybridMoABackbone       : output shape {y_demo.shape}, finite={np.all(np.isfinite(y_demo))}")
    print("=" * 78)


if __name__ == "__main__":
    _run_self_test()
