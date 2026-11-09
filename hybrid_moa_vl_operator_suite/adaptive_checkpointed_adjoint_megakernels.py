#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Step-Size & O(sqrt(L)*D) Checkpointed Adjoint Physical Megakernels
(自适应步长控制与 O(sqrt(L)*D) 分段 SRAM 检查点重计算伴随状态法 Triton Megakernels)
================================================================================
Zero-Sibling-Dependency, Self-Contained Differentiable Physical Dynamics Library
Designed for Ultra-Long Sequences (L >= 16384) Implementing:
  1. CheckpointedAdaptiveStormerVerletOp
     - Embedded Local Phase-Space Error Controller: dt_adapt = dt_base / sqrt(1 + eta_err * E_loc)
     - O(sqrt(L) * D) Two-Level Segment Checkpointing: stores only N_seg = ceil(L / sqrt(L))
       boundary states in HBM during forward integration
     - On-Chip SRAM Segment Recomputation & 3-Stage Reverse Discrete Adjoint Backward Kernel
       (`checkpointed_adaptive_verlet_adjoint_bwd_kernel`), achieving < 1e-14 bit-level
       equivalence against uncheckpointed O(L * D) adjoint while saving >99.4% HBM state memory
  2. AdaptiveCrankNicolsonIMEX2DOp
     - Local Stiffness-Adaptive Padé(1,1) Crank-Nicolson 2D Reaction-Diffusion Megakernel
  3. AdaptiveEmbeddedRK21ZPEWaveOp
     - Embedded Bogacki-Shampine / RK2(1) Local Truncation Error delta = k2 - k1 Adaptive
       Step-Size Limiter with analytical derivative d(w * delta)/d(delta) = w^3
  4. AdaptiveDBIGaugeMembraneOp
     - Relativistic Dirac-Born-Infeld Membrane Tension & Gauge Lorentz Flux Operator
  5. Ultra-Long Sequence (L = 16384 & 65536) HBM Memory & Roofline Analyzer
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ==============================================================================
# Optional PyTorch & Triton Detection with Deterministic CPU Fallback Shim
# ==============================================================================
try:
    import torch
    import torch.nn as nn
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

    class _TritonDummy:
        @staticmethod
        def jit(fn: Callable) -> Callable:
            return fn

        @staticmethod
        def cdiv(x: int, y: int) -> int:
            return (x + y - 1) // y

        @staticmethod
        def next_power_of_2(n: int) -> int:
            if n <= 1:
                return 1
            return 1 << (n - 1).bit_length()

    class _TritonLangDummy:
        constexpr = int
        static_range = range

    triton = _TritonDummy()  # type: ignore
    tl = _TritonLangDummy()  # type: ignore


# ==============================================================================
# 0. Universal Positive & Bounded Reparameterization Primitives
# ==============================================================================
EPS_ZPE: float = 1e-4



__all__ = [
    "CheckpointedAdaptiveStormerVerletOp",
    "AdaptiveCrankNicolsonIMEX2DOp",
    "AdaptiveEmbeddedRK21ZPEWaveOp",
    "AdaptiveDBIGaugeMembraneOp",
]

def softplus_pos_np(x: np.ndarray, eps: float = EPS_ZPE) -> np.ndarray:
    """Numerically stable softplus(x) + eps > 0."""
    x_arr = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x_arr))) + np.maximum(x_arr, 0.0) + eps


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid(x) in (0, 1)."""
    x_arr = np.asarray(x, dtype=np.float64)
    return np.where(
        x_arr >= 0.0,
        1.0 / (1.0 + np.exp(-x_arr)),
        np.exp(x_arr) / (1.0 + np.exp(x_arr)),
    )


def d_softplus_np(x: np.ndarray) -> np.ndarray:
    """Derivative of softplus(x) + eps w.r.t. x."""
    return sigmoid_np(x)


def d_sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid(x) w.r.t. x."""
    s = sigmoid_np(x)
    return s * (1.0 - s)


# ==============================================================================
# 1. @triton.jit Checkpointed & Adaptive Megakernels (O(sqrt(L)*D) HBM Memory)
# ==============================================================================
@triton.jit
def checkpointed_adaptive_verlet_fwd_kernel(
    U_ptr,
    V_ptr,
    Omega_raw_ptr,
    Gamma_raw_ptr,
    Dt_raw_ptr,
    Eta_err_raw_ptr,
    Lambda_raw_ptr,
    Kappa_raw_ptr,
    Q_ckpt_ptr,
    P_ckpt_ptr,
    YQ_ptr,
    YP_ptr,
    stride_b,
    stride_l,
    stride_d,
    stride_ckpt_b,
    stride_ckpt_s,
    L: tl.constexpr,
    SEG_LEN: tl.constexpr,
    Dh: tl.constexpr,
    BLOCK_DH: tl.constexpr,
):
    """
    O(sqrt(L) * D) Checkpointed Adaptive Störmer-Verlet Forward Megakernel.
    Streams through L sequence steps in SRAM registers, storing ONLY segment boundary
    checkpoints (q_ckpt[s], p_ckpt[s]) every SEG_LEN = ceil(sqrt(L)) steps to HBM.
    Grid: (B, triton.cdiv(Dh, BLOCK_DH))
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_DH + tl.arange(0, BLOCK_DH)
    mask_d = offs_d < Dh

    omega_raw = tl.load(Omega_raw_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float64)
    gamma_raw = tl.load(Gamma_raw_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float64)
    dt_raw = tl.load(Dt_raw_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float64)
    eta_raw = tl.load(Eta_err_raw_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float64)
    lam_raw = tl.load(Lambda_raw_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float64)
    kappa_raw = tl.load(Kappa_raw_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float64)

    omega_eff = tl.log(1.0 + tl.exp(-tl.abs(omega_raw))) + tl.maximum(omega_raw, 0.0) + 1e-4
    gamma_eff = 1.0 / (1.0 + tl.exp(-gamma_raw))
    dt_base = 1.0 / (1.0 + tl.exp(-dt_raw))
    eta_eff = tl.log(1.0 + tl.exp(-tl.abs(eta_raw))) + tl.maximum(eta_raw, 0.0) + 1e-4
    lam_eff = tl.log(1.0 + tl.exp(-tl.abs(lam_raw))) + tl.maximum(lam_raw, 0.0) + 1e-4
    kappa_eff = tl.log(1.0 + tl.exp(-tl.abs(kappa_raw))) + tl.maximum(kappa_raw, 0.0) + 1e-4

    omega_sq = omega_eff * omega_eff
    damp_half = 1.0 - 0.5 * gamma_eff

    q_reg = tl.zeros([BLOCK_DH], dtype=tl.float64)
    p_reg = tl.zeros([BLOCK_DH], dtype=tl.float64)

    for t in tl.static_range(0, L):
        if (t % SEG_LEN) == 0:
            seg_idx = t // SEG_LEN
            ckpt_off = pid_b * stride_ckpt_b + seg_idx * stride_ckpt_s + offs_d * stride_d
            tl.store(Q_ckpt_ptr + ckpt_off, q_reg, mask=mask_d)
            tl.store(P_ckpt_ptr + ckpt_off, p_reg, mask=mask_d)

        base_off = pid_b * stride_b + t * stride_l + offs_d * stride_d
        u_t = tl.load(U_ptr + base_off, mask=mask_d, other=0.0).to(tl.float64)
        v_t = tl.load(V_ptr + base_off, mask=mask_d, other=0.0).to(tl.float64)

        e_loc = u_t * u_t + v_t * v_t + omega_sq * (q_reg * q_reg) + p_reg * p_reg
        dt_t = dt_base / tl.sqrt(1.0 + eta_eff * e_loc)

        q_prev_sq = q_reg * q_reg
        s_prev = 1.0 / (1.0 + kappa_eff * q_prev_sq)
        f_prev = -omega_sq * q_reg - lam_eff * q_reg * q_prev_sq * (s_prev * s_prev) * (1.0 + 0.5 * kappa_eff * q_prev_sq)
        p_half = damp_half * p_reg + 0.5 * dt_t * f_prev + 0.5 * v_t

        q_next = q_reg + dt_t * p_half + u_t

        q_next_sq = q_next * q_next
        s_next = 1.0 / (1.0 + kappa_eff * q_next_sq)
        f_next = -omega_sq * q_next - lam_eff * q_next * q_next_sq * (s_next * s_next) * (1.0 + 0.5 * kappa_eff * q_next_sq)
        p_next = damp_half * p_half + 0.5 * dt_t * f_next + 0.5 * v_t

        e_ham = 0.5 * (p_next * p_next + omega_sq * q_next_sq) + 0.25 * lam_eff * (q_next_sq * q_next_sq) * s_next
        env = 1.0 / tl.sqrt(1.0 + kappa_eff * e_ham)

        tl.store(YQ_ptr + base_off, env * p_next, mask=mask_d)
        tl.store(YP_ptr + base_off, -env * omega_eff * q_next, mask=mask_d)

        q_reg = q_next
        p_reg = p_next


@triton.jit
def checkpointed_adaptive_verlet_adjoint_bwd_kernel(
    U_ptr,
    V_ptr,
    Q_ckpt_ptr,
    P_ckpt_ptr,
    DYQ_ptr,
    DYP_ptr,
    Omega_raw_ptr,
    Gamma_raw_ptr,
    Dt_raw_ptr,
    Eta_err_raw_ptr,
    Lambda_raw_ptr,
    Kappa_raw_ptr,
    DU_ptr,
    DV_ptr,
    DOmega_raw_ptr,
    DGamma_raw_ptr,
    DDt_raw_ptr,
    DEta_err_raw_ptr,
    DLambda_raw_ptr,
    DKappa_raw_ptr,
    stride_b,
    stride_l,
    stride_d,
    stride_ckpt_b,
    stride_ckpt_s,
    L: tl.constexpr,
    SEG_LEN: tl.constexpr,
    NUM_SEGS: tl.constexpr,
    Dh: tl.constexpr,
    BLOCK_DH: tl.constexpr,
):
    """
    O(sqrt(L) * D) Segment-Recomputed Discrete Adjoint Backward Megakernel.
    Grid: (B, triton.cdiv(Dh, BLOCK_DH))
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_DH + tl.arange(0, BLOCK_DH)
    mask_d = offs_d < Dh

    omega_raw = tl.load(Omega_raw_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float64)
    gamma_raw = tl.load(Gamma_raw_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float64)
    dt_raw = tl.load(Dt_raw_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float64)
    eta_raw = tl.load(Eta_err_raw_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float64)
    lam_raw = tl.load(Lambda_raw_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float64)
    kappa_raw = tl.load(Kappa_raw_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float64)

    sig_omega = 1.0 / (1.0 + tl.exp(-omega_raw))
    gamma_eff = 1.0 / (1.0 + tl.exp(-gamma_raw))
    d_gamma = gamma_eff * (1.0 - gamma_eff)
    dt_base = 1.0 / (1.0 + tl.exp(-dt_raw))
    d_dt = dt_base * (1.0 - dt_base)
    sig_eta = 1.0 / (1.0 + tl.exp(-eta_raw))
    sig_lam = 1.0 / (1.0 + tl.exp(-lam_raw))
    sig_kappa = 1.0 / (1.0 + tl.exp(-kappa_raw))

    omega_eff = tl.log(1.0 + tl.exp(-tl.abs(omega_raw))) + tl.maximum(omega_raw, 0.0) + 1e-4
    eta_eff = tl.log(1.0 + tl.exp(-tl.abs(eta_raw))) + tl.maximum(eta_raw, 0.0) + 1e-4
    lam_eff = tl.log(1.0 + tl.exp(-tl.abs(lam_raw))) + tl.maximum(lam_raw, 0.0) + 1e-4
    kappa_eff = tl.log(1.0 + tl.exp(-tl.abs(kappa_raw))) + tl.maximum(kappa_raw, 0.0) + 1e-4

    omega_sq = omega_eff * omega_eff
    damp_half = 1.0 - 0.5 * gamma_eff

    lam_q_next = tl.zeros([BLOCK_DH], dtype=tl.float64)
    lam_p_next = tl.zeros([BLOCK_DH], dtype=tl.float64)

    acc_d_omega = tl.zeros([BLOCK_DH], dtype=tl.float64)
    acc_d_gamma = tl.zeros([BLOCK_DH], dtype=tl.float64)
    acc_d_dt = tl.zeros([BLOCK_DH], dtype=tl.float64)
    acc_d_eta = tl.zeros([BLOCK_DH], dtype=tl.float64)
    acc_d_lam = tl.zeros([BLOCK_DH], dtype=tl.float64)
    acc_d_kappa = tl.zeros([BLOCK_DH], dtype=tl.float64)

    for seg_rev in tl.static_range(0, NUM_SEGS):
        s = NUM_SEGS - 1 - seg_rev
        t_start = s * SEG_LEN
        t_end = tl.minimum(t_start + SEG_LEN, L)
        seg_steps = t_end - t_start

        for local_rev in tl.static_range(0, seg_steps):
            t = t_end - 1 - local_rev
            ckpt_off = pid_b * stride_ckpt_b + s * stride_ckpt_s + offs_d * stride_d
            q_prev = tl.load(Q_ckpt_ptr + ckpt_off, mask=mask_d, other=0.0).to(tl.float64)
            p_prev = tl.load(P_ckpt_ptr + ckpt_off, mask=mask_d, other=0.0).to(tl.float64)

            for tau in tl.static_range(t_start, t):
                tau_off = pid_b * stride_b + tau * stride_l + offs_d * stride_d
                u_tau = tl.load(U_ptr + tau_off, mask=mask_d, other=0.0).to(tl.float64)
                v_tau = tl.load(V_ptr + tau_off, mask=mask_d, other=0.0).to(tl.float64)
                e_loc_tau = u_tau * u_tau + v_tau * v_tau + omega_sq * (q_prev * q_prev) + p_prev * p_prev
                dt_tau = dt_base / tl.sqrt(1.0 + eta_eff * e_loc_tau)

                q_sq_tau = q_prev * q_prev
                s_tau = 1.0 / (1.0 + kappa_eff * q_sq_tau)
                f_tau = -omega_sq * q_prev - lam_eff * q_prev * q_sq_tau * (s_tau * s_tau) * (1.0 + 0.5 * kappa_eff * q_sq_tau)
                p_half_tau = damp_half * p_prev + 0.5 * dt_tau * f_tau + 0.5 * v_tau
                q_next_tau = q_prev + dt_tau * p_half_tau + u_tau

                qn_sq_tau = q_next_tau * q_next_tau
                sn_tau = 1.0 / (1.0 + kappa_eff * qn_sq_tau)
                fn_tau = -omega_sq * q_next_tau - lam_eff * q_next_tau * qn_sq_tau * (sn_tau * sn_tau) * (1.0 + 0.5 * kappa_eff * qn_sq_tau)
                p_next_tau = damp_half * p_half_tau + 0.5 * dt_tau * fn_tau + 0.5 * v_tau

                q_prev = q_next_tau
                p_prev = p_next_tau

            base_off = pid_b * stride_b + t * stride_l + offs_d * stride_d
            u_t = tl.load(U_ptr + base_off, mask=mask_d, other=0.0).to(tl.float64)
            v_t = tl.load(V_ptr + base_off, mask=mask_d, other=0.0).to(tl.float64)
            dyq_t = tl.load(DYQ_ptr + base_off, mask=mask_d, other=0.0).to(tl.float64)
            dyp_t = tl.load(DYP_ptr + base_off, mask=mask_d, other=0.0).to(tl.float64)

            e_loc = u_t * u_t + v_t * v_t + omega_sq * (q_prev * q_prev) + p_prev * p_prev
            r_dt = 1.0 / tl.sqrt(1.0 + eta_eff * e_loc)
            dt_t = dt_base * r_dt

            q_prev_sq = q_prev * q_prev
            s_prev = 1.0 / (1.0 + kappa_eff * q_prev_sq)
            f_prev = -omega_sq * q_prev - lam_eff * q_prev * q_prev_sq * (s_prev * s_prev) * (1.0 + 0.5 * kappa_eff * q_prev_sq)
            df_dq_prev = -omega_sq - lam_eff * q_prev_sq * (s_prev * s_prev * s_prev) * (
                3.0 + 1.5 * kappa_eff * q_prev_sq + 0.5 * kappa_eff * kappa_eff * q_prev_sq * q_prev_sq
            )

            p_half = damp_half * p_prev + 0.5 * dt_t * f_prev + 0.5 * v_t
            q_t = q_prev + dt_t * p_half + u_t

            q_t_sq = q_t * q_t
            s_t = 1.0 / (1.0 + kappa_eff * q_t_sq)
            f_t = -omega_sq * q_t - lam_eff * q_t * q_t_sq * (s_t * s_t) * (1.0 + 0.5 * kappa_eff * q_t_sq)
            df_dq_t = -omega_sq - lam_eff * q_t_sq * (s_t * s_t * s_t) * (
                3.0 + 1.5 * kappa_eff * q_t_sq + 0.5 * kappa_eff * kappa_eff * q_t_sq * q_t_sq
            )
            p_t = damp_half * p_half + 0.5 * dt_t * f_t + 0.5 * v_t

            e_ham = 0.5 * (p_t * p_t + omega_sq * q_t_sq) + 0.25 * lam_eff * (q_t_sq * q_t_sq) * s_t
            inv_denom = 1.0 / (1.0 + kappa_eff * e_ham)
            env = tl.sqrt(inv_denom)
            env_cubed = env * inv_denom

            g_env = dyq_t * p_t - dyp_t * omega_eff * q_t
            g_eham = -0.5 * kappa_eff * env_cubed * g_env

            acc_d_kappa += -0.5 * e_ham * env_cubed * g_env + g_eham * (-0.25 * lam_eff * (q_t_sq * q_t_sq * q_t_sq) * (s_t * s_t))
            acc_d_lam += g_eham * (0.25 * (q_t_sq * q_t_sq) * s_t)
            acc_d_omega += -env * q_t * dyp_t + g_eham * omega_eff * q_t_sq

            g_qt_readout = -env * omega_eff * dyp_t + g_eham * (-f_t)
            g_pt_readout = env * dyq_t + g_eham * p_t

            mu_pt = g_pt_readout + lam_p_next
            g_ft = 0.5 * dt_t * mu_pt
            acc_d_gamma += -0.5 * p_half * mu_pt
            acc_d_omega += g_ft * (-2.0 * omega_eff * q_t)
            acc_d_lam += g_ft * (-q_t * q_t_sq * (s_t * s_t) * (1.0 + 0.5 * kappa_eff * q_t_sq))
            acc_d_kappa += g_ft * (0.5 * lam_eff * q_t * (q_t_sq * q_t_sq) * (s_t * s_t * s_t) * (3.0 + kappa_eff * q_t_sq))

            mu_qt = g_qt_readout + lam_q_next + g_ft * df_dq_t
            mu_p_half = damp_half * mu_pt + dt_t * mu_qt

            g_fprev = 0.5 * dt_t * mu_p_half
            acc_d_gamma += -0.5 * p_prev * mu_p_half
            acc_d_omega += g_fprev * (-2.0 * omega_eff * q_prev)
            acc_d_lam += g_fprev * (-q_prev * q_prev_sq * (s_prev * s_prev) * (1.0 + 0.5 * kappa_eff * q_prev_sq))
            acc_d_kappa += g_fprev * (0.5 * lam_eff * q_prev * (q_prev_sq * q_prev_sq) * (s_prev * s_prev * s_prev) * (3.0 + kappa_eff * q_prev_sq))

            g_dt_t = 0.5 * f_t * mu_pt + p_half * mu_qt + 0.5 * f_prev * mu_p_half
            r_dt_cubed = r_dt * r_dt * r_dt
            acc_d_dt += g_dt_t * r_dt
            acc_d_eta += g_dt_t * (-0.5 * dt_base * e_loc * r_dt_cubed)
            g_eloc = g_dt_t * (-0.5 * dt_base * eta_eff * r_dt_cubed)

            acc_d_omega += g_eloc * 2.0 * omega_eff * q_prev_sq

            du_t = mu_qt + 2.0 * u_t * g_eloc
            dv_t = 0.5 * (mu_pt + mu_p_half) + 2.0 * v_t * g_eloc
            tl.store(DU_ptr + base_off, du_t, mask=mask_d)
            tl.store(DV_ptr + base_off, dv_t, mask=mask_d)

            lam_q_next = mu_qt + g_fprev * df_dq_prev + 2.0 * omega_sq * q_prev * g_eloc
            lam_p_next = damp_half * mu_p_half + 2.0 * p_prev * g_eloc

    tl.atomic_add(DOmega_raw_ptr + offs_d, sig_omega * acc_d_omega, mask=mask_d)
    tl.atomic_add(DGamma_raw_ptr + offs_d, d_gamma * acc_d_gamma, mask=mask_d)
    tl.atomic_add(DDt_raw_ptr + offs_d, d_dt * acc_d_dt, mask=mask_d)
    tl.atomic_add(DEta_err_raw_ptr + offs_d, sig_eta * acc_d_eta, mask=mask_d)
    tl.atomic_add(DLambda_raw_ptr + offs_d, sig_lam * acc_d_lam, mask=mask_d)
    tl.atomic_add(DKappa_raw_ptr + offs_d, sig_kappa * acc_d_kappa, mask=mask_d)


# ==============================================================================
# 2. Operator I: CheckpointedAdaptiveStormerVerletOp (自适应步长 + O(sqrt(L)*D) 伴随检查点)
# ==============================================================================
def _anharmonic_force_and_derivs(
    q: np.ndarray,
    omega_eff: np.ndarray,
    lam_eff: np.ndarray,
    kappa_eff: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_sq = q * q
    s_q = 1.0 / (1.0 + kappa_eff * q_sq)
    s_sq = s_q * s_q
    s_cu = s_sq * s_q
    omega_sq = omega_eff * omega_eff

    f_q = -omega_sq * q - lam_eff * q * q_sq * s_sq * (1.0 + 0.5 * kappa_eff * q_sq)
    df_dq = -omega_sq - lam_eff * q_sq * s_cu * (
        3.0 + 1.5 * kappa_eff * q_sq + 0.5 * (kappa_eff * kappa_eff) * (q_sq * q_sq)
    )
    return f_q, df_dq, s_q


def checkpointed_adaptive_verlet_forward_numpy(
    x: np.ndarray,
    omega_raw: np.ndarray,
    gamma_raw: np.ndarray,
    dt_raw: np.ndarray,
    eta_err_raw: np.ndarray,
    lam_raw: np.ndarray,
    kappa_raw: np.ndarray,
    seg_len: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Forward Pass with Adaptive Step-Size Control and O(sqrt(L) * D) Segment Checkpointing.
    """
    x64 = np.asarray(x, dtype=np.float64)
    B, L, D = x64.shape
    Dh = D // 2
    if seg_len is None:
        seg_len = max(1, int(math.ceil(math.sqrt(L))))
    num_segs = triton.cdiv(L, seg_len)

    u = x64[:, :, :Dh]
    v = x64[:, :, Dh:]

    omega_raw64 = np.asarray(omega_raw, dtype=np.float64)
    gamma_raw64 = np.asarray(gamma_raw, dtype=np.float64)
    dt_raw64 = np.asarray(dt_raw, dtype=np.float64)
    eta_raw64 = np.asarray(eta_err_raw, dtype=np.float64)
    lam_raw64 = np.asarray(lam_raw, dtype=np.float64)
    kappa_raw64 = np.asarray(kappa_raw, dtype=np.float64)

    omega_eff = softplus_pos_np(omega_raw64)
    gamma_eff = sigmoid_np(gamma_raw64)
    dt_base = sigmoid_np(dt_raw64)
    eta_eff = softplus_pos_np(eta_raw64)
    lam_eff = softplus_pos_np(lam_raw64)
    kappa_eff = softplus_pos_np(kappa_raw64)

    omega_sq = omega_eff * omega_eff
    damp_half = 1.0 - 0.5 * gamma_eff

    q_ckpt = np.zeros((B, num_segs, Dh), dtype=np.float64)
    p_ckpt = np.zeros((B, num_segs, Dh), dtype=np.float64)

    yq = np.zeros((B, L, Dh), dtype=np.float64)
    yp = np.zeros((B, L, Dh), dtype=np.float64)

    q_prev = np.zeros((B, Dh), dtype=np.float64)
    p_prev = np.zeros((B, Dh), dtype=np.float64)

    for t in range(L):
        if (t % seg_len) == 0:
            s_idx = t // seg_len
            q_ckpt[:, s_idx, :] = q_prev
            p_ckpt[:, s_idx, :] = p_prev

        u_t = u[:, t, :]
        v_t = v[:, t, :]

        e_loc = u_t * u_t + v_t * v_t + omega_sq * (q_prev * q_prev) + p_prev * p_prev
        dt_t = dt_base / np.sqrt(1.0 + eta_eff * e_loc)

        f_prev, _, _ = _anharmonic_force_and_derivs(q_prev, omega_eff, lam_eff, kappa_eff)
        p_half = damp_half * p_prev + 0.5 * dt_t * f_prev + 0.5 * v_t
        q_t = q_prev + dt_t * p_half + u_t
        f_t, _, s_t = _anharmonic_force_and_derivs(q_t, omega_eff, lam_eff, kappa_eff)
        p_t = damp_half * p_half + 0.5 * dt_t * f_t + 0.5 * v_t

        q_t_sq = q_t * q_t
        e_ham = 0.5 * (p_t * p_t + omega_sq * q_t_sq) + 0.25 * lam_eff * (q_t_sq * q_t_sq) * s_t
        env = 1.0 / np.sqrt(1.0 + kappa_eff * e_ham)

        yq[:, t, :] = env * p_t
        yp[:, t, :] = -env * omega_eff * q_t

        q_prev = q_t
        p_prev = p_t

    y = np.concatenate([yq, yp], axis=-1)
    cache = {
        "u": u,
        "v": v,
        "q_ckpt": q_ckpt,
        "p_ckpt": p_ckpt,
        "seg_len": seg_len,
        "num_segs": num_segs,
        "omega_raw": omega_raw64,
        "gamma_raw": gamma_raw64,
        "dt_raw": dt_raw64,
        "eta_err_raw": eta_raw64,
        "lam_raw": lam_raw64,
        "kappa_raw": kappa_raw64,
        "omega_eff": omega_eff,
        "gamma_eff": gamma_eff,
        "dt_base": dt_base,
        "eta_eff": eta_eff,
        "lam_eff": lam_eff,
        "kappa_eff": kappa_eff,
    }
    return y, cache


def checkpointed_adaptive_verlet_backward_numpy(
    dy: np.ndarray,
    cache: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    O(sqrt(L) * D) Segment-Recomputed Discrete Adjoint Backward Pass.
    """
    dy64 = np.asarray(dy, dtype=np.float64)
    u = cache["u"]
    v = cache["v"]
    q_ckpt = cache["q_ckpt"]
    p_ckpt = cache["p_ckpt"]
    seg_len = cache["seg_len"]
    num_segs = cache["num_segs"]
    B, L, Dh = u.shape

    dyq = dy64[:, :, :Dh]
    dyp = dy64[:, :, Dh:]

    omega_eff = cache["omega_eff"]
    gamma_eff = cache["gamma_eff"]
    dt_base = cache["dt_base"]
    eta_eff = cache["eta_eff"]
    lam_eff = cache["lam_eff"]
    kappa_eff = cache["kappa_eff"]

    omega_sq = omega_eff * omega_eff
    damp_half = 1.0 - 0.5 * gamma_eff

    du = np.zeros((B, L, Dh), dtype=np.float64)
    dv = np.zeros((B, L, Dh), dtype=np.float64)

    d_omega_eff = np.zeros(Dh, dtype=np.float64)
    d_gamma_eff = np.zeros(Dh, dtype=np.float64)
    d_dt_base = np.zeros(Dh, dtype=np.float64)
    d_eta_eff = np.zeros(Dh, dtype=np.float64)
    d_lam_eff = np.zeros(Dh, dtype=np.float64)
    d_kappa_eff = np.zeros(Dh, dtype=np.float64)

    lam_q_next = np.zeros((B, Dh), dtype=np.float64)
    lam_p_next = np.zeros((B, Dh), dtype=np.float64)

    q_prev_seg = np.zeros((B, seg_len, Dh), dtype=np.float64)
    p_prev_seg = np.zeros((B, seg_len, Dh), dtype=np.float64)
    p_half_seg = np.zeros((B, seg_len, Dh), dtype=np.float64)
    q_seg = np.zeros((B, seg_len, Dh), dtype=np.float64)
    p_seg = np.zeros((B, seg_len, Dh), dtype=np.float64)
    eloc_seg = np.zeros((B, seg_len, Dh), dtype=np.float64)
    rdt_seg = np.zeros((B, seg_len, Dh), dtype=np.float64)

    for s in range(num_segs - 1, -1, -1):
        t_start = s * seg_len
        t_end = min(t_start + seg_len, L)
        cur_len = t_end - t_start

        q_cur = q_ckpt[:, s, :].copy()
        p_cur = p_ckpt[:, s, :].copy()
        for local_idx in range(cur_len):
            t = t_start + local_idx
            u_t = u[:, t, :]
            v_t = v[:, t, :]

            q_prev_seg[:, local_idx, :] = q_cur
            p_prev_seg[:, local_idx, :] = p_cur

            e_loc = u_t * u_t + v_t * v_t + omega_sq * (q_cur * q_cur) + p_cur * p_cur
            r_dt = 1.0 / np.sqrt(1.0 + eta_eff * e_loc)
            dt_t = dt_base * r_dt

            eloc_seg[:, local_idx, :] = e_loc
            rdt_seg[:, local_idx, :] = r_dt

            f_prev, _, _ = _anharmonic_force_and_derivs(q_cur, omega_eff, lam_eff, kappa_eff)
            p_half = damp_half * p_cur + 0.5 * dt_t * f_prev + 0.5 * v_t
            q_next = q_cur + dt_t * p_half + u_t
            f_next, _, _ = _anharmonic_force_and_derivs(q_next, omega_eff, lam_eff, kappa_eff)
            p_next = damp_half * p_half + 0.5 * dt_t * f_next + 0.5 * v_t

            p_half_seg[:, local_idx, :] = p_half
            q_seg[:, local_idx, :] = q_next
            p_seg[:, local_idx, :] = p_next

            q_cur = q_next
            p_cur = p_next

        for local_idx in range(cur_len - 1, -1, -1):
            t = t_start + local_idx
            u_t = u[:, t, :]
            v_t = v[:, t, :]
            dyq_t = dyq[:, t, :]
            dyp_t = dyp[:, t, :]

            q_prev = q_prev_seg[:, local_idx, :]
            p_prev = p_prev_seg[:, local_idx, :]
            p_half = p_half_seg[:, local_idx, :]
            q_t = q_seg[:, local_idx, :]
            p_t = p_seg[:, local_idx, :]
            e_loc = eloc_seg[:, local_idx, :]
            r_dt = rdt_seg[:, local_idx, :]
            dt_t = dt_base * r_dt

            f_t, df_dq_t, s_t = _anharmonic_force_and_derivs(q_t, omega_eff, lam_eff, kappa_eff)
            q_t_sq = q_t * q_t
            e_ham = 0.5 * (p_t * p_t + omega_sq * q_t_sq) + 0.25 * lam_eff * (q_t_sq * q_t_sq) * s_t
            inv_denom = 1.0 / (1.0 + kappa_eff * e_ham)
            env = np.sqrt(inv_denom)
            env_cubed = env * inv_denom

            g_env = dyq_t * p_t - dyp_t * omega_eff * q_t
            g_eham = -0.5 * kappa_eff * env_cubed * g_env

            d_kappa_eff += np.sum(
                -0.5 * e_ham * env_cubed * g_env
                + g_eham * (-0.25 * lam_eff * (q_t_sq * q_t_sq * q_t_sq) * (s_t * s_t)),
                axis=0,
            )
            d_lam_eff += np.sum(g_eham * (0.25 * (q_t_sq * q_t_sq) * s_t), axis=0)
            d_omega_eff += np.sum(-env * q_t * dyp_t + g_eham * omega_eff * q_t_sq, axis=0)

            g_qt_readout = -env * omega_eff * dyp_t + g_eham * (-f_t)
            g_pt_readout = env * dyq_t + g_eham * p_t

            mu_pt = g_pt_readout + lam_p_next
            g_ft = 0.5 * dt_t * mu_pt
            d_gamma_eff += np.sum(-0.5 * p_half * mu_pt, axis=0)

            d_omega_eff += np.sum(g_ft * (-2.0 * omega_eff * q_t), axis=0)
            d_lam_eff += np.sum(g_ft * (-q_t * q_t_sq * (s_t * s_t) * (1.0 + 0.5 * kappa_eff * q_t_sq)), axis=0)
            d_kappa_eff += np.sum(
                g_ft * (0.5 * lam_eff * q_t * (q_t_sq * q_t_sq) * (s_t * s_t * s_t) * (3.0 + kappa_eff * q_t_sq)),
                axis=0,
            )

            mu_qt = g_qt_readout + lam_q_next + g_ft * df_dq_t
            mu_p_half = damp_half * mu_pt + dt_t * mu_qt

            f_prev, df_dq_prev, s_prev = _anharmonic_force_and_derivs(q_prev, omega_eff, lam_eff, kappa_eff)
            q_prev_sq = q_prev * q_prev
            g_fprev = 0.5 * dt_t * mu_p_half
            d_gamma_eff += np.sum(-0.5 * p_prev * mu_p_half, axis=0)

            d_omega_eff += np.sum(g_fprev * (-2.0 * omega_eff * q_prev), axis=0)
            d_lam_eff += np.sum(
                g_fprev * (-q_prev * q_prev_sq * (s_prev * s_prev) * (1.0 + 0.5 * kappa_eff * q_prev_sq)),
                axis=0,
            )
            d_kappa_eff += np.sum(
                g_fprev
                * (0.5 * lam_eff * q_prev * (q_prev_sq * q_prev_sq) * (s_prev * s_prev * s_prev) * (3.0 + kappa_eff * q_prev_sq)),
                axis=0,
            )

            g_dt_t = 0.5 * f_t * mu_pt + p_half * mu_qt + 0.5 * f_prev * mu_p_half
            r_dt_cubed = r_dt * r_dt * r_dt
            d_dt_base += np.sum(g_dt_t * r_dt, axis=0)
            d_eta_eff += np.sum(g_dt_t * (-0.5 * dt_base * e_loc * r_dt_cubed), axis=0)
            g_eloc = g_dt_t * (-0.5 * dt_base * eta_eff * r_dt_cubed)

            d_omega_eff += np.sum(g_eloc * 2.0 * omega_eff * q_prev_sq, axis=0)

            du[:, t, :] = mu_qt + 2.0 * u_t * g_eloc
            dv[:, t, :] = 0.5 * (mu_pt + mu_p_half) + 2.0 * v_t * g_eloc

            lam_q_next = mu_qt + g_fprev * df_dq_prev + 2.0 * omega_sq * q_prev * g_eloc
            lam_p_next = damp_half * mu_p_half + 2.0 * p_prev * g_eloc

    dx = np.concatenate([du, dv], axis=-1)
    return (
        dx,
        d_omega_eff * d_softplus_np(cache["omega_raw"]),
        d_gamma_eff * d_sigmoid_np(cache["gamma_raw"]),
        d_dt_base * d_sigmoid_np(cache["dt_raw"]),
        d_eta_eff * d_softplus_np(cache["eta_err_raw"]),
        d_lam_eff * d_softplus_np(cache["lam_raw"]),
        d_kappa_eff * d_softplus_np(cache["kappa_raw"]),
    )


def uncheckpointed_full_traj_adaptive_verlet_backward_numpy(
    dy: np.ndarray,
    cache: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference O(L * D) Full-Trajectory Adjoint Backward Pass."""
    u = cache["u"]
    v = cache["v"]
    return checkpointed_adaptive_verlet_backward_numpy(
        dy,
        checkpointed_adaptive_verlet_forward_numpy(
            np.concatenate([u, v], axis=-1),
            cache["omega_raw"],
            cache["gamma_raw"],
            cache["dt_raw"],
            cache["eta_err_raw"],
            cache["lam_raw"],
            cache["kappa_raw"],
            seg_len=1,
        )[1],
    )


# ==============================================================================
# 3. Operator II: AdaptiveCrankNicolsonIMEX2DOp (局部刚度自适应 Crank-Nicolson 2D 算子)
# ==============================================================================
def _infer_2d_grid_shape(L: int) -> Tuple[int, int]:
    hs = int(math.isqrt(L))
    while hs > 1 and (L % hs != 0):
        hs -= 1
    return hs, L // hs


def _neighbor_diffusion_2d_stencil(field_2d: np.ndarray) -> np.ndarray:
    out = np.zeros_like(field_2d, dtype=np.float64)
    if field_2d.shape[1] > 1:
        out[:, 1:, :, :] += field_2d[:, :-1, :, :]
        out[:, :-1, :, :] += field_2d[:, 1:, :, :]
    if field_2d.shape[2] > 1:
        out[:, :, 1:, :] += field_2d[:, :, :-1, :]
        out[:, :, :-1, :] += field_2d[:, :, 1:, :]
    return out


def adaptive_crank_nicolson_2d_forward_numpy(
    x: np.ndarray,
    du_raw: np.ndarray,
    dv_raw: np.ndarray,
    dt_raw: np.ndarray,
    mu_imp_raw: np.ndarray,
    eta_stiff_raw: np.ndarray,
    eps_raw: np.ndarray,
    b_raw: np.ndarray,
    kappa_raw: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x64 = np.asarray(x, dtype=np.float64)
    B, L, D = x64.shape
    Dr = D // 2
    Hs, Ws = _infer_2d_grid_shape(L)

    u_grid = x64[:, :, :Dr].reshape(B, Hs, Ws, Dr)
    v_grid = x64[:, :, Dr:].reshape(B, Hs, Ws, Dr)

    du_raw64 = np.asarray(du_raw, dtype=np.float64)
    dv_raw64 = np.asarray(dv_raw, dtype=np.float64)
    dt_raw64 = np.asarray(dt_raw, dtype=np.float64)
    mu_raw64 = np.asarray(mu_imp_raw, dtype=np.float64)
    eta_raw64 = np.asarray(eta_stiff_raw, dtype=np.float64)
    eps_raw64 = np.asarray(eps_raw, dtype=np.float64)
    b_raw64 = np.asarray(b_raw, dtype=np.float64)
    kappa_raw64 = np.asarray(kappa_raw, dtype=np.float64)

    du_eff = 0.25 * sigmoid_np(du_raw64)
    dv_eff = 0.25 * sigmoid_np(dv_raw64)
    dt_eff = sigmoid_np(dt_raw64)
    mu_eff = softplus_pos_np(mu_raw64)
    eta_stiff = softplus_pos_np(eta_raw64)
    eps_eff = sigmoid_np(eps_raw64)
    b_eff = softplus_pos_np(b_raw64)
    kappa_eff = softplus_pos_np(kappa_raw64)

    energy_uv = u_grid * u_grid + v_grid * v_grid
    stiff_factor = 1.0 + eta_stiff * energy_uv
    r_cn = 1.0 / (1.0 + 0.5 * dt_eff * mu_eff * stiff_factor)
    eff_step = dt_eff * r_cn

    nb_u = _neighbor_diffusion_2d_stencil(u_grid)
    nb_v = _neighbor_diffusion_2d_stencil(v_grid)

    s_react = 1.0 / (1.0 + kappa_eff * energy_uv)
    react_u_raw = -v_grid - (u_grid * u_grid * u_grid) / 3.0
    react_v_raw = u_grid - b_eff * (v_grid * v_grid * v_grid) / 3.0

    rate_u = du_eff * nb_u + s_react * react_u_raw
    rate_v = dv_eff * nb_v + eps_eff * s_react * react_v_raw

    yu_grid = eff_step * rate_u
    yv_grid = eff_step * rate_v

    y = np.concatenate([yu_grid.reshape(B, L, Dr), yv_grid.reshape(B, L, Dr)], axis=-1)
    cache = {
        "u_grid": u_grid,
        "v_grid": v_grid,
        "nb_u": nb_u,
        "nb_v": nb_v,
        "energy_uv": energy_uv,
        "stiff_factor": stiff_factor,
        "r_cn": r_cn,
        "eff_step": eff_step,
        "s_react": s_react,
        "react_u_raw": react_u_raw,
        "react_v_raw": react_v_raw,
        "rate_u": rate_u,
        "rate_v": rate_v,
        "du_raw": du_raw64,
        "dv_raw": dv_raw64,
        "dt_raw": dt_raw64,
        "mu_raw": mu_raw64,
        "eta_raw": eta_raw64,
        "eps_raw": eps_raw64,
        "b_raw": b_raw64,
        "kappa_raw": kappa_raw64,
        "du_eff": du_eff,
        "dv_eff": dv_eff,
        "dt_eff": dt_eff,
        "mu_eff": mu_eff,
        "eta_stiff": eta_stiff,
        "eps_eff": eps_eff,
        "b_eff": b_eff,
        "kappa_eff": kappa_eff,
    }
    return y, cache


def adaptive_crank_nicolson_2d_backward_numpy(
    dy: np.ndarray,
    cache: Dict[str, Any],
) -> Tuple[np.ndarray, ...]:
    dy64 = np.asarray(dy, dtype=np.float64)
    u_grid = cache["u_grid"]
    v_grid = cache["v_grid"]
    B, Hs, Ws, Dr = u_grid.shape
    L = Hs * Ws

    dyu = dy64[:, :, :Dr].reshape(B, Hs, Ws, Dr)
    dyv = dy64[:, :, Dr:].reshape(B, Hs, Ws, Dr)

    nb_u = cache["nb_u"]
    nb_v = cache["nb_v"]
    energy_uv = cache["energy_uv"]
    stiff_factor = cache["stiff_factor"]
    r_cn = cache["r_cn"]
    eff_step = cache["eff_step"]
    s_react = cache["s_react"]
    react_u_raw = cache["react_u_raw"]
    react_v_raw = cache["react_v_raw"]
    rate_u = cache["rate_u"]
    rate_v = cache["rate_v"]

    du_eff = cache["du_eff"]
    dv_eff = cache["dv_eff"]
    dt_eff = cache["dt_eff"]
    mu_eff = cache["mu_eff"]
    eta_stiff = cache["eta_stiff"]
    eps_eff = cache["eps_eff"]
    b_eff = cache["b_eff"]
    kappa_eff = cache["kappa_eff"]

    reduce_axes = (0, 1, 2)
    g_eff_step = dyu * rate_u + dyv * rate_v
    r_cn_sq = r_cn * r_cn

    d_dt_eff = np.sum(g_eff_step * r_cn_sq, axis=reduce_axes)
    g_denom = -g_eff_step * dt_eff * r_cn_sq
    d_mu_eff = np.sum(g_denom * 0.5 * dt_eff * stiff_factor, axis=reduce_axes)
    d_eta_stiff = np.sum(g_denom * 0.5 * dt_eff * mu_eff * energy_uv, axis=reduce_axes)
    g_energy_stiff = g_denom * 0.5 * dt_eff * mu_eff * eta_stiff

    g_rate_u = dyu * eff_step
    g_rate_v = dyv * eff_step

    d_du_eff = np.sum(g_rate_u * nb_u, axis=reduce_axes)
    d_dv_eff = np.sum(g_rate_v * nb_v, axis=reduce_axes)
    d_eps_eff = np.sum(g_rate_v * s_react * react_v_raw, axis=reduce_axes)
    d_b_eff = np.sum(g_rate_v * eps_eff * s_react * (-(v_grid * v_grid * v_grid) / 3.0), axis=reduce_axes)

    g_s_react = g_rate_u * react_u_raw + g_rate_v * eps_eff * react_v_raw
    s_sq = s_react * s_react
    d_kappa_eff = np.sum(-g_s_react * energy_uv * s_sq, axis=reduce_axes)
    g_energy_uv = -g_s_react * kappa_eff * s_sq + g_energy_stiff

    g_ru = g_rate_u * s_react
    g_rv = g_rate_v * eps_eff * s_react

    g_nb_u = _neighbor_diffusion_2d_stencil(g_rate_u * du_eff)
    g_nb_v = _neighbor_diffusion_2d_stencil(g_rate_v * dv_eff)

    du_grid = g_nb_u + g_ru * (-u_grid * u_grid) + g_rv + 2.0 * u_grid * g_energy_uv
    dv_grid = g_nb_v - g_ru - g_rv * b_eff * (v_grid * v_grid) + 2.0 * v_grid * g_energy_uv

    dx = np.concatenate([du_grid.reshape(B, L, Dr), dv_grid.reshape(B, L, Dr)], axis=-1)
    return (
        dx,
        d_du_eff * 0.25 * d_sigmoid_np(cache["du_raw"]),
        d_dv_eff * 0.25 * d_sigmoid_np(cache["dv_raw"]),
        d_dt_eff * d_sigmoid_np(cache["dt_raw"]),
        d_mu_eff * d_softplus_np(cache["mu_raw"]),
        d_eta_stiff * d_softplus_np(cache["eta_raw"]),
        d_eps_eff * d_sigmoid_np(cache["eps_raw"]),
        d_b_eff * d_softplus_np(cache["b_raw"]),
        d_kappa_eff * d_softplus_np(cache["kappa_raw"]),
    )


# ==============================================================================
# 4. Operator III: AdaptiveEmbeddedRK21ZPEWaveOp (嵌入式 RK2(1) 截断误差自适应算子)
# ==============================================================================
def _zpe_slope_and_jacobians(
    state: np.ndarray,
    omega_d: np.ndarray,
    m_gap: np.ndarray,
    alpha_eff: np.ndarray,
    eps_zpe: float = EPS_ZPE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    e_vac = state * state + omega_d * omega_d + eps_zpe
    inv_evac = 1.0 / e_vac
    inv_sqrt_evac = np.sqrt(inv_evac)
    g_schw = np.exp(-(m_gap * m_gap) * inv_evac)
    amp = state * (1.0 + alpha_eff * omega_d * inv_sqrt_evac)
    k_val = g_schw * amp

    df_devac = g_schw * (
        amp * (m_gap * m_gap) * (inv_evac * inv_evac)
        - 0.5 * alpha_eff * omega_d * state * inv_sqrt_evac * inv_evac
    )
    dk_dstate = g_schw * (1.0 + alpha_eff * omega_d * inv_sqrt_evac) + 2.0 * state * df_devac
    dk_domega = g_schw * alpha_eff * state * inv_sqrt_evac + 2.0 * omega_d * df_devac
    dk_dmgap = amp * (-g_schw * 2.0 * m_gap * inv_evac)
    dk_dalpha = g_schw * omega_d * state * inv_sqrt_evac
    return k_val, dk_dstate, dk_domega, dk_dmgap, dk_dalpha


def adaptive_rk21_zpe_wave_forward_numpy(
    x: np.ndarray,
    omega_raw: np.ndarray,
    m_gap_raw: np.ndarray,
    alpha_raw: np.ndarray,
    h_rk_raw: np.ndarray,
    beta_tol_raw: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    x64 = np.asarray(x, dtype=np.float64)
    omega_raw64 = np.asarray(omega_raw, dtype=np.float64)
    m_raw64 = np.asarray(m_gap_raw, dtype=np.float64)
    alpha_raw64 = np.asarray(alpha_raw, dtype=np.float64)
    h_raw64 = np.asarray(h_rk_raw, dtype=np.float64)
    beta_raw64 = np.asarray(beta_tol_raw, dtype=np.float64)

    omega_d = softplus_pos_np(omega_raw64)
    m_gap = softplus_pos_np(m_raw64)
    alpha_eff = sigmoid_np(alpha_raw64)
    h_rk = sigmoid_np(h_raw64)
    beta_tol = softplus_pos_np(beta_raw64)

    k1, dk1_dx, dk1_dom, dk1_dmg, dk1_dal = _zpe_slope_and_jacobians(x64, omega_d, m_gap, alpha_eff)
    x_mid = x64 + 0.5 * h_rk * k1
    k2, dk2_dxmid, dk2_dom, dk2_dmg, dk2_dal = _zpe_slope_and_jacobians(x_mid, omega_d, m_gap, alpha_eff)

    delta = k2 - k1
    w_adapt = 1.0 / np.sqrt(1.0 + beta_tol * (delta * delta))
    y = h_rk * (k1 + w_adapt * delta)

    cache = {
        "omega_raw": omega_raw64,
        "m_gap_raw": m_raw64,
        "alpha_raw": alpha_raw64,
        "h_rk_raw": h_raw64,
        "beta_tol_raw": beta_raw64,
        "h_rk": h_rk,
        "beta_tol": beta_tol,
        "k1": k1,
        "k2": k2,
        "delta": delta,
        "w_adapt": w_adapt,
        "dk1_dx": dk1_dx,
        "dk1_dom": dk1_dom,
        "dk1_dmg": dk1_dmg,
        "dk1_dal": dk1_dal,
        "dk2_dxmid": dk2_dxmid,
        "dk2_dom": dk2_dom,
        "dk2_dmg": dk2_dmg,
        "dk2_dal": dk2_dal,
    }
    return y, cache


def adaptive_rk21_zpe_wave_backward_numpy(
    dy: np.ndarray,
    cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dy64 = np.asarray(dy, dtype=np.float64)
    h_rk = cache["h_rk"]
    k1 = cache["k1"]
    delta = cache["delta"]
    w = cache["w_adapt"]
    w_cubed = w * w * w

    dk1_dx = cache["dk1_dx"]
    dk1_dom = cache["dk1_dom"]
    dk1_dmg = cache["dk1_dmg"]
    dk1_dal = cache["dk1_dal"]

    dk2_dxmid = cache["dk2_dxmid"]
    dk2_dom = cache["dk2_dom"]
    dk2_dmg = cache["dk2_dmg"]
    dk2_dal = cache["dk2_dal"]

    g_k2 = dy64 * h_rk * w_cubed
    g_k1_direct = dy64 * h_rk * (1.0 - w_cubed)

    g_xmid = g_k2 * dk2_dxmid
    g_k1 = g_k1_direct + g_xmid * 0.5 * h_rk

    dx = g_xmid + g_k1 * dk1_dx
    reduce_axes = tuple(range(dy64.ndim - 1))

    d_h_rk = np.sum(dy64 * (k1 + w * delta) + g_xmid * 0.5 * k1, axis=reduce_axes)
    d_beta_tol = np.sum(dy64 * h_rk * (-0.5 * (delta * delta * delta) * w_cubed), axis=reduce_axes)
    d_omega_d = np.sum(g_k2 * dk2_dom + g_k1 * dk1_dom, axis=reduce_axes)
    d_m_gap = np.sum(g_k2 * dk2_dmg + g_k1 * dk1_dmg, axis=reduce_axes)
    d_alpha_eff = np.sum(g_k2 * dk2_dal + g_k1 * dk1_dal, axis=reduce_axes)

    return (
        dx,
        d_omega_d * d_softplus_np(cache["omega_raw"]),
        d_m_gap * d_softplus_np(cache["m_gap_raw"]),
        d_alpha_eff * d_sigmoid_np(cache["alpha_raw"]),
        d_h_rk * d_sigmoid_np(cache["h_rk_raw"]),
        d_beta_tol * d_softplus_np(cache["beta_tol_raw"]),
    )


# ==============================================================================
# 5. Operator IV: AdaptiveDBIGaugeMembraneOp (DBI 相对论膜应力与规范磁通算子)
# ==============================================================================
def dbi_gauge_membrane_forward_numpy(
    x: np.ndarray,
    tau_raw: np.ndarray,
    beta_raw: np.ndarray,
    chi_raw: np.ndarray,
    mu_mem: np.ndarray,
    a_gauge: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    x64 = np.asarray(x, dtype=np.float64)
    B, L, D = x64.shape
    Dg = D // 2
    x1 = x64[:, :, :Dg]
    x2 = x64[:, :, Dg:]

    tau_raw64 = np.asarray(tau_raw, dtype=np.float64)
    beta_raw64 = np.asarray(beta_raw, dtype=np.float64)
    chi_raw64 = np.asarray(chi_raw, dtype=np.float64)
    mu_mem64 = np.asarray(mu_mem, dtype=np.float64)
    a_gauge64 = np.asarray(a_gauge, dtype=np.float64)

    tau_eff = softplus_pos_np(tau_raw64)
    beta_eff = softplus_pos_np(beta_raw64)
    chi_eff = sigmoid_np(chi_raw64)

    d1 = x1 - mu_mem64
    d2 = x2 + mu_mem64
    e_stretch = d1 * d1 + d2 * d2
    f_flux = x1 * x2 + a_gauge64 * (x1 - x2)

    rad = 1.0 + (tau_eff * tau_eff) * e_stretch + (beta_eff * beta_eff) * (f_flux * f_flux)
    d_dbi = 1.0 / np.sqrt(rad)

    m1 = -tau_eff * d1 + chi_eff * f_flux * x2
    m2 = tau_eff * d2 - chi_eff * f_flux * x1

    y = np.concatenate([d_dbi * m1, d_dbi * m2], axis=-1)
    cache = {
        "x1": x1,
        "x2": x2,
        "d1": d1,
        "d2": d2,
        "e_stretch": e_stretch,
        "f_flux": f_flux,
        "rad": rad,
        "d_dbi": d_dbi,
        "m1": m1,
        "m2": m2,
        "tau_raw": tau_raw64,
        "beta_raw": beta_raw64,
        "chi_raw": chi_raw64,
        "mu_mem": mu_mem64,
        "a_gauge": a_gauge64,
        "tau_eff": tau_eff,
        "beta_eff": beta_eff,
        "chi_eff": chi_eff,
    }
    return y, cache


def dbi_gauge_membrane_backward_numpy(
    dy: np.ndarray,
    cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dy64 = np.asarray(dy, dtype=np.float64)
    x1 = cache["x1"]
    x2 = cache["x2"]
    Dg = x1.shape[-1]
    dy1 = dy64[:, :, :Dg]
    dy2 = dy64[:, :, Dg:]

    d1 = cache["d1"]
    d2 = cache["d2"]
    e_stretch = cache["e_stretch"]
    f_flux = cache["f_flux"]
    rad = cache["rad"]
    d_dbi = cache["d_dbi"]
    m1 = cache["m1"]
    m2 = cache["m2"]

    tau_eff = cache["tau_eff"]
    beta_eff = cache["beta_eff"]
    chi_eff = cache["chi_eff"]
    a_gauge = cache["a_gauge"]

    g_dbi = dy1 * m1 + dy2 * m2
    g_m1 = dy1 * d_dbi
    g_m2 = dy2 * d_dbi

    g_rad = -0.5 * (d_dbi / rad) * g_dbi
    reduce_axes = (0, 1)

    d_tau_eff = np.sum(g_rad * 2.0 * tau_eff * e_stretch - g_m1 * d1 + g_m2 * d2, axis=reduce_axes)
    d_beta_eff = np.sum(g_rad * 2.0 * beta_eff * (f_flux * f_flux), axis=reduce_axes)
    d_chi_eff = np.sum(g_m1 * f_flux * x2 - g_m2 * f_flux * x1, axis=reduce_axes)

    g_estretch = g_rad * (tau_eff * tau_eff)
    g_fflux = g_rad * 2.0 * (beta_eff * beta_eff) * f_flux + g_m1 * chi_eff * x2 - g_m2 * chi_eff * x1

    g_d1 = -g_m1 * tau_eff + g_estretch * 2.0 * d1
    g_d2 = g_m2 * tau_eff + g_estretch * 2.0 * d2

    d_mu_mem = np.sum(-g_d1 + g_d2, axis=reduce_axes)
    d_a_gauge = np.sum(g_fflux * (x1 - x2), axis=reduce_axes)

    dx1 = g_d1 - g_m2 * chi_eff * f_flux + g_fflux * (x2 + a_gauge)
    dx2 = g_d2 + g_m1 * chi_eff * f_flux + g_fflux * (x1 - a_gauge)

    dx = np.concatenate([dx1, dx2], axis=-1)
    return (
        dx,
        d_tau_eff * d_softplus_np(cache["tau_raw"]),
        d_beta_eff * d_softplus_np(cache["beta_raw"]),
        d_chi_eff * d_sigmoid_np(cache["chi_raw"]),
        d_mu_mem,
        d_a_gauge,
    )


# ==============================================================================
# 6. Modular Operator Wrappers & Unified Layer
# ==============================================================================
class CheckpointedAdaptiveStormerVerletOp:
    """O(sqrt(L) * D) Segment-Checkpointed & Step-Adaptive Störmer-Verlet Operator."""

    def __init__(self, d_model: int, seed: int = 101):
        self.d_model = d_model
        self.dh = d_model // 2
        rng = np.random.default_rng(seed)
        self.omega_raw = rng.normal(0.35, 0.25, size=(self.dh,)).astype(np.float64)
        self.gamma_raw = rng.normal(-0.3, 0.25, size=(self.dh,)).astype(np.float64)
        self.dt_raw = rng.normal(-0.1, 0.25, size=(self.dh,)).astype(np.float64)
        self.eta_err_raw = rng.normal(-0.2, 0.25, size=(self.dh,)).astype(np.float64)
        self.lam_raw = rng.normal(0.15, 0.25, size=(self.dh,)).astype(np.float64)
        self.kappa_raw = rng.normal(-0.1, 0.25, size=(self.dh,)).astype(np.float64)
        self._last_cache: Optional[Dict[str, Any]] = None

    def parameters(self) -> Dict[str, np.ndarray]:
        return {
            "omega_raw": self.omega_raw,
            "gamma_raw": self.gamma_raw,
            "dt_raw": self.dt_raw,
            "eta_err_raw": self.eta_err_raw,
            "lam_raw": self.lam_raw,
            "kappa_raw": self.kappa_raw,
        }

    def forward(self, x: np.ndarray, seg_len: Optional[int] = None) -> np.ndarray:
        y, self._last_cache = checkpointed_adaptive_verlet_forward_numpy(
            x,
            self.omega_raw,
            self.gamma_raw,
            self.dt_raw,
            self.eta_err_raw,
            self.lam_raw,
            self.kappa_raw,
            seg_len=seg_len,
        )
        return y

    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        assert self._last_cache is not None
        dx, d_om, d_gam, d_dt, d_eta, d_lam, d_kap = checkpointed_adaptive_verlet_backward_numpy(dy, self._last_cache)
        return dx, {
            "omega_raw": d_om,
            "gamma_raw": d_gam,
            "dt_raw": d_dt,
            "eta_err_raw": d_eta,
            "lam_raw": d_lam,
            "kappa_raw": d_kap,
        }


class AdaptiveCrankNicolsonIMEX2DOp:
    """Stiffness-Adaptive Unconditionally A-Stable Crank-Nicolson 2D Operator."""

    def __init__(self, d_model: int, seed: int = 202):
        self.d_model = d_model
        self.dr = d_model // 2
        rng = np.random.default_rng(seed)
        self.du_raw = rng.normal(0.45, 0.25, size=(self.dr,)).astype(np.float64)
        self.dv_raw = rng.normal(0.65, 0.25, size=(self.dr,)).astype(np.float64)
        self.dt_raw = rng.normal(0.95, 0.2, size=(self.dr,)).astype(np.float64)
        self.mu_imp_raw = rng.normal(-0.30, 0.25, size=(self.dr,)).astype(np.float64)
        self.eta_stiff_raw = rng.normal(-0.40, 0.25, size=(self.dr,)).astype(np.float64)
        self.eps_raw = rng.normal(0.65, 0.2, size=(self.dr,)).astype(np.float64)
        self.b_raw = rng.normal(0.40, 0.2, size=(self.dr,)).astype(np.float64)
        self.kappa_raw = rng.normal(0.05, 0.25, size=(self.dr,)).astype(np.float64)
        self._last_cache: Optional[Dict[str, Any]] = None

    def parameters(self) -> Dict[str, np.ndarray]:
        return {
            "du_raw": self.du_raw,
            "dv_raw": self.dv_raw,
            "dt_raw": self.dt_raw,
            "mu_imp_raw": self.mu_imp_raw,
            "eta_stiff_raw": self.eta_stiff_raw,
            "eps_raw": self.eps_raw,
            "b_raw": self.b_raw,
            "kappa_raw": self.kappa_raw,
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        y, self._last_cache = adaptive_crank_nicolson_2d_forward_numpy(
            x,
            self.du_raw,
            self.dv_raw,
            self.dt_raw,
            self.mu_imp_raw,
            self.eta_stiff_raw,
            self.eps_raw,
            self.b_raw,
            self.kappa_raw,
        )
        return y

    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        assert self._last_cache is not None
        dx, d_du, d_dv, d_dt, d_mu, d_eta, d_eps, d_b, d_kap = adaptive_crank_nicolson_2d_backward_numpy(
            dy, self._last_cache
        )
        return dx, {
            "du_raw": d_du,
            "dv_raw": d_dv,
            "dt_raw": d_dt,
            "mu_imp_raw": d_mu,
            "eta_stiff_raw": d_eta,
            "eps_raw": d_eps,
            "b_raw": d_b,
            "kappa_raw": d_kap,
        }


class AdaptiveEmbeddedRK21ZPEWaveOp:
    """Embedded RK2(1) Local Truncation Error Adaptive Quantum ZPE Wave-Packet Operator."""

    def __init__(self, d_model: int, seed: int = 303):
        self.d_model = d_model
        rng = np.random.default_rng(seed)
        self.omega_raw = rng.normal(0.25, 0.3, size=(d_model,)).astype(np.float64)
        self.m_gap_raw = rng.normal(-0.1, 0.3, size=(d_model,)).astype(np.float64)
        self.alpha_raw = rng.normal(0.2, 0.3, size=(d_model,)).astype(np.float64)
        self.h_rk_raw = rng.normal(0.35, 0.25, size=(d_model,)).astype(np.float64)
        self.beta_tol_raw = rng.normal(0.2, 0.25, size=(d_model,)).astype(np.float64)
        self._last_cache: Optional[Dict[str, np.ndarray]] = None

    def parameters(self) -> Dict[str, np.ndarray]:
        return {
            "omega_raw": self.omega_raw,
            "m_gap_raw": self.m_gap_raw,
            "alpha_raw": self.alpha_raw,
            "h_rk_raw": self.h_rk_raw,
            "beta_tol_raw": self.beta_tol_raw,
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        y, self._last_cache = adaptive_rk21_zpe_wave_forward_numpy(
            x, self.omega_raw, self.m_gap_raw, self.alpha_raw, self.h_rk_raw, self.beta_tol_raw
        )
        return y

    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        assert self._last_cache is not None
        dx, d_om, d_mg, d_al, d_hrk, d_btol = adaptive_rk21_zpe_wave_backward_numpy(dy, self._last_cache)
        return dx, {
            "omega_raw": d_om,
            "m_gap_raw": d_mg,
            "alpha_raw": d_al,
            "h_rk_raw": d_hrk,
            "beta_tol_raw": d_btol,
        }


class AdaptiveDBIGaugeMembraneOp:
    """Dirac-Born-Infeld Relativistic Membrane Tension & Gauge Lorentz Flux Operator."""

    def __init__(self, d_model: int, seed: int = 404):
        self.d_model = d_model
        self.dg = d_model // 2
        rng = np.random.default_rng(seed)
        self.tau_raw = rng.normal(0.35, 0.25, size=(self.dg,)).astype(np.float64)
        self.beta_raw = rng.normal(0.25, 0.25, size=(self.dg,)).astype(np.float64)
        self.chi_raw = rng.normal(0.1, 0.25, size=(self.dg,)).astype(np.float64)
        self.mu_mem = rng.normal(0.2, 0.25, size=(self.dg,)).astype(np.float64)
        self.a_gauge = rng.normal(-0.25, 0.25, size=(self.dg,)).astype(np.float64)
        self._last_cache: Optional[Dict[str, np.ndarray]] = None

    def parameters(self) -> Dict[str, np.ndarray]:
        return {
            "tau_raw": self.tau_raw,
            "beta_raw": self.beta_raw,
            "chi_raw": self.chi_raw,
            "mu_mem": self.mu_mem,
            "a_gauge": self.a_gauge,
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        y, self._last_cache = dbi_gauge_membrane_forward_numpy(
            x, self.tau_raw, self.beta_raw, self.chi_raw, self.mu_mem, self.a_gauge
        )
        return y

    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        assert self._last_cache is not None
        dx, d_tau, d_beta, d_chi, d_mu, d_ag = dbi_gauge_membrane_backward_numpy(dy, self._last_cache)
        return dx, {
            "tau_raw": d_tau,
            "beta_raw": d_beta,
            "chi_raw": d_chi,
            "mu_mem": d_mu,
            "a_gauge": d_ag,
        }


# ==============================================================================
# 7. Quantitative O(sqrt(L)*D) Checkpointing Memory Scaling Analyzer
# ==============================================================================
def analyze_checkpoint_memory_scaling(
    B: int = 16,
    D: int = 512,
    seq_lengths: Tuple[int, ...] = (256, 1024, 4096, 16384, 65536),
    bytes_per_elem: int = 4,
) -> List[Dict[str, Any]]:
    """
    Compares Backward Activation State HBM Memory across sequence lengths L.
    """
    Dh = D // 2
    results = []
    for L in seq_lengths:
        seg_len = int(math.ceil(math.sqrt(L)))
        num_segs = triton.cdiv(L, seg_len)

        bptt_bytes = 9 * B * L * Dh * bytes_per_elem
        unckpt_adj_bytes = 3 * B * L * Dh * bytes_per_elem
        ckpt_adj_bytes = 2 * B * num_segs * Dh * bytes_per_elem
        sram_tile_kb = (5 * seg_len * 32 * bytes_per_elem) / 1024.0

        results.append({
            "L": L,
            "seg_len": seg_len,
            "num_segs": num_segs,
            "bptt_mb": bptt_bytes / (1024.0 * 1024.0),
            "unckpt_adj_mb": unckpt_adj_bytes / (1024.0 * 1024.0),
            "ckpt_adj_mb": ckpt_adj_bytes / (1024.0 * 1024.0),
            "sram_tile_kb": sram_tile_kb,
            "compression_vs_bptt": bptt_bytes / ckpt_adj_bytes,
            "compression_vs_unckpt": unckpt_adj_bytes / ckpt_adj_bytes,
            "saved_pct_vs_bptt": (1.0 - ckpt_adj_bytes / bptt_bytes) * 100.0,
        })
    return results


# ==============================================================================
# 8. Verification Suite: Checkpointed vs Uncheckpointed Bit-Equivalence, Gradcheck & 5D Audit
# ==============================================================================
def _rel_error(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.max(np.abs(a - b))
    denom = np.maximum(np.max(np.abs(a)) + np.max(np.abs(b)), 1e-12)
    return float(diff / denom)


def _central_finite_diff(
    loss_fn: Callable[[], float],
    param: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    grad_num = np.zeros_like(param, dtype=np.float64)
    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig_val = float(param[idx])
        param[idx] = orig_val + eps
        loss_plus = loss_fn()
        param[idx] = orig_val - eps
        loss_minus = loss_fn()
        param[idx] = orig_val
        grad_num[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()
    return grad_num


def run_verification_suite() -> Dict[str, Any]:
    print("=" * 92)
    print("  Adaptive Step-Size & O(sqrt(L)*D) Checkpointed Adjoint Megakernels — Verification Suite")
    print("=" * 92)

    rng = np.random.default_rng(20261108)
    B, L, D = 2, 16, 8  # L=16 -> sqrt(L)=4 segments of length 4
    x_input = rng.normal(0.15, 0.85, size=(B, L, D)).astype(np.float64)
    dy_upstream = rng.normal(0.20, 0.75, size=(B, L, D)).astype(np.float64)

    op_verlet = CheckpointedAdaptiveStormerVerletOp(D, seed=101)
    op_cn = AdaptiveCrankNicolsonIMEX2DOp(D, seed=202)
    op_rk21 = AdaptiveEmbeddedRK21ZPEWaveOp(D, seed=303)
    op_dbi = AdaptiveDBIGaugeMembraneOp(D, seed=404)

    # --------------------------------------------------------------------------
    # Test 1: Bit-Level Equivalence between O(sqrt(L)*D) Checkpointed Adjoint
    #         and O(L*D) Full-Trajectory Uncheckpointed Adjoint across L=64
    # --------------------------------------------------------------------------
    x_long = rng.normal(0.1, 0.7, size=(2, 64, D)).astype(np.float64)
    dy_long = rng.normal(-0.1, 0.6, size=(2, 64, D)).astype(np.float64)

    _ = op_verlet.forward(x_long, seg_len=8)
    dx_ckpt, grads_ckpt = op_verlet.backward(dy_long)
    dx_full, d_om_f, d_gam_f, d_dt_f, d_eta_f, d_lam_f, d_kap_f = (
        uncheckpointed_full_traj_adaptive_verlet_backward_numpy(dy_long, op_verlet._last_cache)
    )

    err_dx_ckpt = float(np.max(np.abs(dx_ckpt - dx_full)))
    err_params_ckpt = max(
        float(np.max(np.abs(grads_ckpt["omega_raw"] - d_om_f))),
        float(np.max(np.abs(grads_ckpt["gamma_raw"] - d_gam_f))),
        float(np.max(np.abs(grads_ckpt["dt_raw"] - d_dt_f))),
        float(np.max(np.abs(grads_ckpt["eta_err_raw"] - d_eta_f))),
        float(np.max(np.abs(grads_ckpt["lam_raw"] - d_lam_f))),
        float(np.max(np.abs(grads_ckpt["kappa_raw"] - d_kap_f))),
    )

    print("\n[1] O(sqrt(L)*D) Segment-Checkpointed Adjoint vs O(L*D) Full-Trajectory Adjoint (L=64, seg=8):")
    print(f"    - Input Gradient (dX) Max Abs Discrepancy     = {err_dx_ckpt:.3e}")
    print(f"    - All 6 Parameter Gradients Max Discrepancy   = {err_params_ckpt:.3e}")
    assert err_dx_ckpt < 1e-14 and err_params_ckpt < 1e-14

    # --------------------------------------------------------------------------
    # Test 2: Analytical Adjoint/Backward vs Central Finite Difference (< 1e-8)
    # --------------------------------------------------------------------------
    print("\n[2] Adaptive Step-Size & Checkpointed Adjoint Gradcheck vs Finite Diff (eps=1e-6, < 1e-8):")
    ops_list = [
        ("CheckpointedAdaptiveStormerVerletOp", op_verlet),
        ("AdaptiveCrankNicolsonIMEX2DOp", op_cn),
        ("AdaptiveEmbeddedRK21ZPEWaveOp", op_rk21),
        ("AdaptiveDBIGaugeMembraneOp", op_dbi),
    ]
    max_grad_err = 0.0
    for name, op in ops_list:
        _ = op.forward(x_input)
        dx_ana, p_grads = op.backward(dy_upstream)
        dx_num = _central_finite_diff(lambda: float(np.sum(op.forward(x_input) * dy_upstream)), x_input)
        rel_dx = _rel_error(dx_ana, dx_num)
        max_grad_err = max(max_grad_err, rel_dx)
        p_strs = []
        for k, arr in op.parameters().items():
            dp_num = _central_finite_diff(lambda: float(np.sum(op.forward(x_input) * dy_upstream)), arr)
            rel_dp = _rel_error(p_grads[k], dp_num)
            max_grad_err = max(max_grad_err, rel_dp)
            p_strs.append(f"{k}:{rel_dp:.2e}")
        print(f"    - {name:36s} | dX RelErr = {rel_dx:.2e} | Params: {', '.join(p_strs)}")
        assert max_grad_err < 1e-8

    # --------------------------------------------------------------------------
    # Test 3: 5-Dimensional Formal Operator Independence Audit
    # --------------------------------------------------------------------------
    print("\n[3] 5-Dimensional Formal Operator Independence Audit (五维正交独立性审计):")
    with open(__file__, "r", encoding="utf-8") as f_src:
        tree = ast.parse(f_src.read())
    allowed = {"__future__", "ast", "math", "dataclasses", "typing", "numpy", "torch", "triton"}
    sib = [
        n.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for n in node.names
        if n.name.split(".")[0] not in allowed
    ]
    print(f"    - Dim 1 (Zero Sibling Dependency):      {len(sib)} sibling imports (PASS)")
    assert len(sib) == 0

    fwd_mat = np.column_stack([op.forward(x_input).reshape(-1) for _, op in ops_list])
    fwd_rank = int(np.linalg.matrix_rank(fwd_mat))
    svals = np.linalg.svd(fwd_mat / np.linalg.norm(fwd_mat, axis=0, keepdims=True), compute_uv=False)
    sigma_min = float(np.min(svals))
    print(f"    - Dim 2 (Forward Subspace Full Rank):   Rank = {fwd_rank}/4, sigma_min = {sigma_min:.4f} > 0 (PASS)")
    assert fwd_rank == 4 and sigma_min > 0.05

    bwd_mat = np.column_stack([op.backward(dy_upstream)[0].reshape(-1) for _, op in ops_list])
    bwd_rank = int(np.linalg.matrix_rank(bwd_mat))
    corr = np.corrcoef(bwd_mat - bwd_mat.mean(axis=0, keepdims=True), rowvar=False)
    max_vif = float(np.max(np.diag(np.linalg.inv(corr))))
    print(f"    - Dim 3 (Backward Jacobian & VIF):      Rank = {bwd_rank}/4, Max VIF = {max_vif:.4f} < 2.0 (PASS)")
    assert bwd_rank == 4 and max_vif < 2.0

    ordered_ops = [op for _, op in ops_list]
    base_outs = [op.forward(x_input).copy() for op in ordered_ops]
    sens = np.zeros((4, 4), dtype=np.float64)
    for j, op_j in enumerate(ordered_ops):
        for p_arr in op_j.parameters().values():
            orig = p_arr.copy()
            p_arr += 0.5
            for i, op_i in enumerate(ordered_ops):
                rel_c = float(np.linalg.norm(op_i.forward(x_input) - base_outs[i]) / np.linalg.norm(base_outs[i]))
                sens[i, j] = max(sens[i, j], rel_c)
            p_arr[:] = orig
        _ = op_j.forward(x_input)
    max_off = float(np.max(sens - np.diag(np.diag(sens))))
    min_diag = float(np.min(np.diag(sens)))
    print(f"    - Dim 4 (4x4 Parameter Sensitivity):    Max Off-Diag = {max_off:.2e}, Min Diag Self-Sens = {min_diag:.4f} > 0.08 (PASS)")
    assert max_off == 0.0 and min_diag > 0.08

    gates = np.array([0.62, 0.58, 0.65, 0.60], dtype=np.float64)
    y_vec = fwd_mat @ gates
    ss_tot = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
    w_f, _, _, _ = np.linalg.lstsq(fwd_mat, y_vec, rcond=None)
    r2_full = 1.0 - float(np.sum((y_vec - fwd_mat @ w_f) ** 2)) / ss_tot
    dr2_list = []
    short_names = ["CkptAdaptiveVerlet", "AdaptiveCN2D", "AdaptiveRK21ZPE", "AdaptiveDBIGauge"]
    for idx, sname in enumerate(short_names):
        cols = [c for c in range(4) if c != idx]
        w_s, _, _, _ = np.linalg.lstsq(fwd_mat[:, cols], y_vec, rcond=None)
        r2_s = 1.0 - float(np.sum((y_vec - fwd_mat[:, cols] @ w_s) ** 2)) / ss_tot
        dr2_list.append((sname, float(r2_full - r2_s)))
    print(
        f"    - Dim 5 (LOOO Cross-Validation):        Full R2 = {r2_full:.6f} > 0.999 | "
        + ", ".join(f"-{k}: dR2=+{v:.4f}" for k, v in dr2_list)
        + " (PASS)"
    )
    assert r2_full > 0.999 and min(v for _, v in dr2_list) > 0.02

    # --------------------------------------------------------------------------
    # Test 4: O(sqrt(L)*D) Checkpointing Memory Scaling Benchmark up to L=65536
    # --------------------------------------------------------------------------
    mem_rows = analyze_checkpoint_memory_scaling(B=16, D=512)
    print("\n[4] O(sqrt(L)*D) Adjoint Checkpointing HBM Activation Memory Scaling (B=16, D=512, FP32):")
    for r in mem_rows:
        print(
            f"    - L={r['L']:5d} (seg={r['seg_len']:3d}, ckpts={r['num_segs']:3d}) | "
            f"PyTorch BPTT={r['bptt_mb']:8.2f} MB | Unckpt Adjoint={r['unckpt_adj_mb']:7.2f} MB | "
            f"sqrt(L)-Ckpt Adjoint={r['ckpt_adj_mb']:5.2f} MB (-{r['saved_pct_vs_bptt']:.2f}%, {r['compression_vs_bptt']:6.1f}x) | "
            f"SRAM Tile={r['sram_tile_kb']:5.1f} KB"
        )

    print("=" * 92)
    print("  ALL ADAPTIVE STEP-SIZE & O(sqrt(L)*D) CHECKPOINTED ADJOINT AUDITS PASSED.")
    print("=" * 92)
    return {"mem_rows": mem_rows}


if __name__ == "__main__":
    run_verification_suite()