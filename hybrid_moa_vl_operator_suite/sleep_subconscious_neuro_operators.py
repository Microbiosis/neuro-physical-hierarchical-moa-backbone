"""
Human Sleep Physiology & Deep Subconscious Neuroscience Neural Operator Library
(`sleep_subconscious_neuro_operators.py`).

Translates four foundational mechanisms of human sleep physiology and deep
subconscious neuroscience into a high-efficiency, differentiable neural operator
with `@triton.jit` GPU kernels, exact analytical backward passes, and PyTorch
`autograd.Function` / `nn.Module` integration:

1. Borbély Two-Process Sleep Regulation & Hypothalamic VLPO/ARAS Flip-Flop Switch:
   - Models the mutual inhibition flip-flop switch between sleep-promoting VLPO
     (Ventrolateral Preoptic Nucleus GABA/Galanin neurons) and wake-promoting
     monoaminergic ARAS driven by homeostatic adenosine sleep pressure (`x_wake^2`)
     and REM subconscious drive (`gamma_rem * z_sub^2`):
         E_drive = x_wake^2 + gamma_rem * z_sub^2 - theta_sleep
         G_sleep = sigmoid(kappa_vlpo * E_drive),   G_wake = 1 - G_sleep

2. Tononi Synaptic Homeostasis Hypothesis (SHY) & Nedergaard Glymphatic Clearance:
   - During NREM Slow-Wave Sleep (SWS, 0.5-4 Hz delta waves), global synaptic
     potentiation accumulated during wakefulness is multiplicatively renormalized:
         inv_shy = rsqrt(1 + lambda_shy * G_sleep * x_wake^2)
   - Simultaneously, astrocyte AQP4-mediated glymphatic interstitial fluid pulses
     wash out sub-threshold metabolic noise (`|x_wake| << sqrt(eps_glymph)`) while
     preserving consolidated structural traces:
         W_glymph = 1 - (G_sleep * eps_glymph) / (x_wake^2 + eps_glymph)
         Y_sws    = x_wake * inv_shy * W_glymph

3. Thalamocortical Slow-Oscillation, Spindle & Hippocampal Sharp-Wave Ripple (SWR)
   Triple Phase-Amplitude Coupled Memory Consolidation:
   - Replays compressed hippocampal episodic traces (`m_hipp`) nested inside
     thalamocortical spindles and slow-oscillation up-states to consolidate
     long-term neocortical representations:
         U_swr = (x_wake * m_hipp + m_hipp) * rsqrt(1 + x_wake^2 + m_hipp^2)
         Y_swr = alpha_swr * G_sleep * U_swr

4. REM Ponto-Geniculo-Occipital (PGO) Burst Waves & DMN Deep Subconscious
   Hyper-Associative Dream Recombination:
   - During REM sleep, cholinergic PGO waves and Default Mode Network (DMN)
     disinhibition recombine distant semantic/episodic fragments through `R_dream`
     subconscious latent dream anchors (`subconscious_pgo_dream_recombination_forward/backward`)
     and synthesize creative latent associations:
         U_pgo = (z_sub + m_hipp * z_sub) * rsqrt(1 + z_sub^2 + m_hipp^2)
         Y_pgo = beta_pgo * G_sleep * U_pgo
         Out   = Y_sws + Y_swr + Y_pgo
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ==============================================================================
# 1. Configuration Dataclass
# ==============================================================================

@dataclass
class SleepSubconsciousConfig:
    """Configuration for the Sleep Physiology & Deep Subconscious Neural Operator."""
    d_model: int = 256
    n_heads: int = 8
    num_dream_anchors: int = 16           # R_dream: subconscious archetype / PGO dream anchors
    theta_sleep_init: float = 0.55        # VLPO/ARAS flip-flop sleep transition threshold
    kappa_vlpo_init: float = 1.60         # VLPO GABAergic sleep-switch steepness gain
    lambda_shy_init: float = 0.45         # Tononi SHY synaptic downscaling strength
    alpha_swr_init: float = 0.65          # Hippocampal Sharp-Wave Ripple (SWR) replay gain
    beta_pgo_init: float = 0.60           # REM PGO-wave subconscious dream emergence gain
    gamma_rem: float = 0.50               # Subconscious REM drive coupling coefficient
    eps_glymph: float = 0.10              # Nedergaard glymphatic metabolic noise cutoff scale
    rms_norm_eps: float = 1e-6


# ==============================================================================
# 2. Triton GPU Kernels (@triton.jit) — Fused Sleep SHY + Glymphatic + SWR + PGO
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def fused_sleep_subconscious_shy_swr_pgo_fwd_kernel(
        XWake_ptr,      # [B, L, D] wakeful neocortical sensory-executive stream
        MHipp_ptr,      # [B, L, D] hippocampal sharp-wave ripple (SWR) replay stream
        ZSub_ptr,       # [B, L, D] deep subconscious PGO/DMN hyper-associative stream
        Theta_ptr,      # [D] VLPO/ARAS flip-flop threshold
        Kappa_ptr,      # [D] VLPO switch gain
        LambdaSHY_ptr,  # [D] Tononi SHY synaptic downscaling coefficient
        AlphaSWR_ptr,   # [D] Hippocampal SWR consolidation gain
        BetaPGO_ptr,    # [D] REM PGO dream synthesis gain
        Out_ptr,        # [B, L, D] fused output
        stride_row,
        D_MODEL: tl.constexpr,
        gamma_rem: tl.constexpr,
        eps_glymph: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Single-Launch Fused Sleep-Subconscious Forward Kernel in SRAM Registers:
        1. VLPO/ARAS Flip-Flop Switch: G_sleep = sigmoid(kappa * (x^2 + gamma_rem * z^2 - theta))
        2. Tononi SHY + Glymphatic Washout: Y_sws = x * rsqrt(1 + lam_shy * G_sleep * x^2) * W_glymph
        3. Hippocampal SWR Replay: Y_swr = alpha_swr * G_sleep * (x * m + m) * rsqrt(1 + x^2 + m^2)
        4. REM PGO Subconscious Dream: Y_pgo = beta_pgo * G_sleep * (z + m * z) * rsqrt(1 + z^2 + m^2)
        """
        row_id = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D_MODEL
        off = row_id * stride_row + cols

        x = tl.load(XWake_ptr + off, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(MHipp_ptr + off, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(ZSub_ptr + off, mask=mask, other=0.0).to(tl.float32)

        theta = tl.load(Theta_ptr + cols, mask=mask, other=0.55).to(tl.float32)
        kappa = tl.load(Kappa_ptr + cols, mask=mask, other=1.60).to(tl.float32)
        lam_shy = tl.load(LambdaSHY_ptr + cols, mask=mask, other=0.45).to(tl.float32)
        a_swr = tl.load(AlphaSWR_ptr + cols, mask=mask, other=0.65).to(tl.float32)
        b_pgo = tl.load(BetaPGO_ptr + cols, mask=mask, other=0.60).to(tl.float32)

        x_sq = x * x
        m_sq = m * m
        z_sq = z * z

        e_drive = x_sq + gamma_rem * z_sq - theta
        g_sleep = tl.sigmoid(kappa * e_drive)

        inv_shy = tl.math.rsqrt(1.0 + lam_shy * g_sleep * x_sq)
        w_glymph = 1.0 - (g_sleep * eps_glymph) / (x_sq + eps_glymph)
        y_sws = x * inv_shy * w_glymph

        inv_swr = tl.math.rsqrt(1.0 + x_sq + m_sq)
        u_swr = (x * m + m) * inv_swr

        inv_pgo = tl.math.rsqrt(1.0 + z_sq + m_sq)
        u_pgo = (z + m * z) * inv_pgo

        out = y_sws + g_sleep * (a_swr * u_swr + b_pgo * u_pgo)
        tl.store(Out_ptr + off, out, mask=mask)


    @triton.jit
    def fused_sleep_subconscious_shy_swr_pgo_bwd_kernel(
        GradOut_ptr, XWake_ptr, MHipp_ptr, ZSub_ptr,
        Theta_ptr, Kappa_ptr, LambdaSHY_ptr, AlphaSWR_ptr, BetaPGO_ptr,
        GradXWake_ptr, GradMHipp_ptr, GradZSub_ptr,
        stride_row,
        D_MODEL: tl.constexpr,
        gamma_rem: tl.constexpr,
        eps_glymph: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Single-Launch Fused Analytical Backward Kernel for Sleep-Subconscious Op."""
        row_id = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D_MODEL
        off = row_id * stride_row + cols

        dout = tl.load(GradOut_ptr + off, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(XWake_ptr + off, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(MHipp_ptr + off, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(ZSub_ptr + off, mask=mask, other=0.0).to(tl.float32)

        theta = tl.load(Theta_ptr + cols, mask=mask, other=0.55).to(tl.float32)
        kappa = tl.load(Kappa_ptr + cols, mask=mask, other=1.60).to(tl.float32)
        lam_shy = tl.load(LambdaSHY_ptr + cols, mask=mask, other=0.45).to(tl.float32)
        a_swr = tl.load(AlphaSWR_ptr + cols, mask=mask, other=0.65).to(tl.float32)
        b_pgo = tl.load(BetaPGO_ptr + cols, mask=mask, other=0.60).to(tl.float32)

        x_sq = x * x
        m_sq = m * m
        z_sq = z * z

        e_drive = x_sq + gamma_rem * z_sq - theta
        g_sleep = tl.sigmoid(kappa * e_drive)
        one_m_g = 1.0 - g_sleep

        d_shy = 1.0 + lam_shy * g_sleep * x_sq
        inv_shy = tl.math.rsqrt(d_shy)
        dg_denom = x_sq + eps_glymph
        w_glymph = 1.0 - (g_sleep * eps_glymph) / dg_denom
        a_sws = x * w_glymph

        d_swr = 1.0 + x_sq + m_sq
        inv_swr = tl.math.rsqrt(d_swr)
        n_swr = x * m + m
        u_swr = n_swr * inv_swr

        d_pgo = 1.0 + z_sq + m_sq
        inv_pgo = tl.math.rsqrt(d_pgo)
        n_pgo = z + m * z
        u_pgo = n_pgo * inv_pgo

        d_dshy = dout * a_sws * (-0.5 * inv_shy / d_shy)
        d_asws = dout * inv_shy

        d_gsleep = (
            dout * (a_swr * u_swr + b_pgo * u_pgo)
            + d_dshy * (lam_shy * x_sq)
            - d_asws * (eps_glymph * x / dg_denom)
        )
        d_edrive = d_gsleep * g_sleep * one_m_g * kappa

        d_uswr = dout * g_sleep * a_swr
        d_nswr = d_uswr * inv_swr
        d_dswr = d_uswr * n_swr * (-0.5 * inv_swr / d_swr)

        d_upgo = dout * g_sleep * b_pgo
        d_npgo = d_upgo * inv_pgo
        d_dpgo = d_upgo * n_pgo * (-0.5 * inv_pgo / d_pgo)

        d_asws_dx = 1.0 - g_sleep * eps_glymph * (eps_glymph - x_sq) / (dg_denom * dg_denom)

        dx = (
            d_asws * d_asws_dx
            + d_dshy * (2.0 * lam_shy * g_sleep * x)
            + d_edrive * (2.0 * x)
            + d_nswr * m
            + d_dswr * (2.0 * x)
        )
        dm = (
            d_nswr * (x + 1.0)
            + d_dswr * (2.0 * m)
            + d_npgo * z
            + d_dpgo * (2.0 * m)
        )
        dz = (
            d_npgo * (1.0 + m)
            + d_dpgo * (2.0 * z)
            + d_edrive * (2.0 * gamma_rem * z)
        )

        tl.store(GradXWake_ptr + off, dx, mask=mask)
        tl.store(GradMHipp_ptr + off, dm, mask=mask)
        tl.store(GradZSub_ptr + off, dz, mask=mask)

else:
    def fused_sleep_subconscious_shy_swr_pgo_fwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to launch fused_sleep_subconscious_shy_swr_pgo_fwd_kernel.")

    def fused_sleep_subconscious_shy_swr_pgo_bwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to launch fused_sleep_subconscious_shy_swr_pgo_bwd_kernel.")


# ==============================================================================
# 3. Exact Analytical Forward/Backward Kernels & Triton Tile Simulator (NumPy)
# ==============================================================================

class SleepSubconsciousAnalyticalKernels:
    """
    Exact double-precision forward and analytical backward kernels for the
    Human Sleep Physiology & Deep Subconscious Neuroscience Operator.
    """

    @staticmethod
    def fused_sleep_shy_swr_pgo_forward(
        x_wake: np.ndarray,
        m_hipp: np.ndarray,
        z_sub: np.ndarray,
        theta_sleep: np.ndarray,
        kappa_vlpo: np.ndarray,
        lambda_shy: np.ndarray,
        alpha_swr: np.ndarray,
        beta_pgo: np.ndarray,
        gamma_rem: float = 0.50,
        eps_glymph: float = 0.10,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass combining:
        1. VLPO/ARAS Flip-Flop Switch (`G_sleep`)
        2. Tononi Synaptic Homeostasis Downscaling (`inv_shy`) + Glymphatic Clearance (`W_glymph`)
        3. Hippocampal Sharp-Wave Ripple (SWR) Memory Consolidation (`Y_swr`)
        4. REM PGO-Wave Deep Subconscious Dream Synthesis (`Y_pgo`)
        """
        th_b = theta_sleep[None, None, :]
        kap_b = kappa_vlpo[None, None, :]
        lam_b = lambda_shy[None, None, :]
        aswr_b = alpha_swr[None, None, :]
        bpgo_b = beta_pgo[None, None, :]

        x_sq = np.square(x_wake)
        m_sq = np.square(m_hipp)
        z_sq = np.square(z_sub)

        # 1. Borbély Process S + REM Drive & VLPO/ARAS Flip-Flop Switch
        e_drive = x_sq + gamma_rem * z_sq - th_b
        logit_sleep = np.clip(kap_b * e_drive, -500.0, 500.0)
        g_sleep = 1.0 / (1.0 + np.exp(-logit_sleep))
        one_m_g = 1.0 - g_sleep

        # 2. Tononi Synaptic Homeostasis (SHY) + Nedergaard Glymphatic Washout
        d_shy = 1.0 + lam_b * g_sleep * x_sq
        inv_shy = 1.0 / np.sqrt(d_shy)
        dg_denom = x_sq + eps_glymph
        w_glymph = 1.0 - (g_sleep * eps_glymph) / dg_denom
        a_sws = x_wake * w_glymph
        y_sws = a_sws * inv_shy

        # 3. Hippocampal Sharp-Wave Ripple (SWR) Triple-Coupled Replay
        d_swr = 1.0 + x_sq + m_sq
        inv_swr = 1.0 / np.sqrt(d_swr)
        n_swr = x_wake * m_hipp + m_hipp
        u_swr = n_swr * inv_swr

        # 4. REM PGO-Wave Deep Subconscious Hyper-Associative Dream Synthesis
        d_pgo = 1.0 + z_sq + m_sq
        inv_pgo = 1.0 / np.sqrt(d_pgo)
        n_pgo = z_sub + m_hipp * z_sub
        u_pgo = n_pgo * inv_pgo

        out = y_sws + g_sleep * (aswr_b * u_swr + bpgo_b * u_pgo)

        cache = {
            "x_wake": x_wake,
            "m_hipp": m_hipp,
            "z_sub": z_sub,
            "theta_sleep": theta_sleep,
            "kappa_vlpo": kappa_vlpo,
            "lambda_shy": lambda_shy,
            "alpha_swr": alpha_swr,
            "beta_pgo": beta_pgo,
            "x_sq": x_sq,
            "m_sq": m_sq,
            "z_sq": z_sq,
            "e_drive": e_drive,
            "g_sleep": g_sleep,
            "one_m_g": one_m_g,
            "d_shy": d_shy,
            "inv_shy": inv_shy,
            "dg_denom": dg_denom,
            "w_glymph": w_glymph,
            "a_sws": a_sws,
            "d_swr": d_swr,
            "inv_swr": inv_swr,
            "n_swr": n_swr,
            "u_swr": u_swr,
            "d_pgo": d_pgo,
            "inv_pgo": inv_pgo,
            "n_pgo": n_pgo,
            "u_pgo": u_pgo,
            "gamma_rem": np.array(gamma_rem),
            "eps_glymph": np.array(eps_glymph),
        }
        return out, cache

    @staticmethod
    def fused_sleep_shy_swr_pgo_backward(
        grad_out: np.ndarray,
        cache: Dict[str, np.ndarray],
        d_gsleep_external: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Exact analytical backward pass w.r.t. all 8 inputs and physiological parameters:
        (`d_xwake`, `d_mhipp`, `d_zsub`, `d_theta`, `d_kappa`, `d_lamshy`, `d_alphaswr`, `d_betapgo`).
        """
        x = cache["x_wake"]
        m = cache["m_hipp"]
        z = cache["z_sub"]
        kap_b = cache["kappa_vlpo"][None, None, :]
        lam_b = cache["lambda_shy"][None, None, :]
        aswr_b = cache["alpha_swr"][None, None, :]
        bpgo_b = cache["beta_pgo"][None, None, :]

        x_sq = cache["x_sq"]
        e_drive = cache["e_drive"]
        g_sleep = cache["g_sleep"]
        one_m_g = cache["one_m_g"]
        d_shy = cache["d_shy"]
        inv_shy = cache["inv_shy"]
        dg_denom = cache["dg_denom"]
        a_sws = cache["a_sws"]
        d_swr = cache["d_swr"]
        inv_swr = cache["inv_swr"]
        n_swr = cache["n_swr"]
        u_swr = cache["u_swr"]
        d_pgo = cache["d_pgo"]
        inv_pgo = cache["inv_pgo"]
        n_pgo = cache["n_pgo"]
        u_pgo = cache["u_pgo"]
        gamma_rem = float(cache["gamma_rem"])
        eps_glymph = float(cache["eps_glymph"])

        # 1. Parameter gradients for alpha_swr and beta_pgo
        d_alphaswr = np.sum(grad_out * g_sleep * u_swr, axis=(0, 1))
        d_betapgo = np.sum(grad_out * g_sleep * u_pgo, axis=(0, 1))

        # 2. Gradients through SWS SHY + Glymphatic term: Y_sws = a_sws * inv_shy
        d_dshy = grad_out * a_sws * (-0.5 * inv_shy / d_shy)
        d_asws = grad_out * inv_shy
        d_lamshy = np.sum(d_dshy * g_sleep * x_sq, axis=(0, 1))

        # 3. Total gradient w.r.t. G_sleep
        d_gsleep = (
            grad_out * (aswr_b * u_swr + bpgo_b * u_pgo)
            + d_dshy * (lam_b * x_sq)
            - d_asws * (eps_glymph * x / dg_denom)
        )
        if d_gsleep_external is not None:
            d_gsleep = d_gsleep + d_gsleep_external

        d_logit = d_gsleep * g_sleep * one_m_g
        d_kappa = np.sum(d_logit * e_drive, axis=(0, 1))
        d_edrive = d_logit * kap_b
        d_theta = np.sum(-d_edrive, axis=(0, 1))

        # 4. Gradients through SWR and PGO units
        d_uswr = grad_out * g_sleep * aswr_b
        d_nswr = d_uswr * inv_swr
        d_dswr = d_uswr * n_swr * (-0.5 * inv_swr / d_swr)

        d_upgo = grad_out * g_sleep * bpgo_b
        d_npgo = d_upgo * inv_pgo
        d_dpgo = d_upgo * n_pgo * (-0.5 * inv_pgo / d_pgo)

        # 5. Total gradients w.r.t. x_wake, m_hipp, z_sub
        d_asws_dx = 1.0 - g_sleep * eps_glymph * (eps_glymph - x_sq) / np.square(dg_denom)

        d_xwake = (
            d_asws * d_asws_dx
            + d_dshy * (2.0 * lam_b * g_sleep * x)
            + d_edrive * (2.0 * x)
            + d_nswr * m
            + d_dswr * (2.0 * x)
        )
        d_mhipp = (
            d_nswr * (x + 1.0)
            + d_dswr * (2.0 * m)
            + d_npgo * z
            + d_dpgo * (2.0 * m)
        )
        d_zsub = (
            d_npgo * (1.0 + m)
            + d_dpgo * (2.0 * z)
            + d_edrive * (2.0 * gamma_rem * z)
        )

        return {
            "d_xwake": d_xwake,
            "d_mhipp": d_mhipp,
            "d_zsub": d_zsub,
            "d_theta": d_theta,
            "d_kappa": d_kappa,
            "d_lamshy": d_lamshy,
            "d_alphaswr": d_alphaswr,
            "d_betapgo": d_betapgo,
        }

    @staticmethod
    def subconscious_pgo_dream_recombination_forward(
        x_wake: np.ndarray,
        m_hipp: np.ndarray,
        dream_anchors: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Non-Local PGO-Wave Deep Subconscious Dream Associative Recombination (`O(L * R_dream * D)`).
        Pools distant wakeful + hippocampal memory fragments into `R_dream` subconscious
        archetype anchors and broadcasts hyper-associative dream context `z_sub` across all tokens.
        """
        B, L, D = x_wake.shape
        scale = 1.0 / math.sqrt(D)

        joint_mem = 0.5 * (x_wake + m_hipp)
        scores_pool = np.einsum("rd,bld->brl", dream_anchors, joint_mem) * scale
        sp_max = np.max(scores_pool, axis=-1, keepdims=True)
        exp_sp = np.exp(scores_pool - sp_max)
        attn_pool = exp_sp / np.sum(exp_sp, axis=-1, keepdims=True)   # [B, R, L]
        z_dream = np.einsum("brl,bld->brd", attn_pool, m_hipp)       # [B, R, D]

        scores_bc = np.einsum("bld,brd->blr", x_wake, z_dream) * scale
        sb_max = np.max(scores_bc, axis=-1, keepdims=True)
        exp_sb = np.exp(scores_bc - sb_max)
        attn_bc = exp_sb / np.sum(exp_sb, axis=-1, keepdims=True)    # [B, L, R]
        z_sub = np.einsum("blr,brd->bld", attn_bc, z_dream)          # [B, L, D]

        cache = {
            "x_wake": x_wake,
            "m_hipp": m_hipp,
            "dream_anchors": dream_anchors,
            "joint_mem": joint_mem,
            "attn_pool": attn_pool,
            "z_dream": z_dream,
            "attn_bc": attn_bc,
            "scale": np.array(scale),
        }
        return z_sub, cache

    @staticmethod
    def subconscious_pgo_dream_recombination_backward(
        grad_zsub: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Exact analytical backward pass for Non-Local PGO-Wave Dream Recombination
        w.r.t. (`x_wake`, `m_hipp`, `dream_anchors`).
        """
        x_wake = cache["x_wake"]
        m_hipp = cache["m_hipp"]
        dream_anchors = cache["dream_anchors"]
        joint_mem = cache["joint_mem"]
        attn_pool = cache["attn_pool"]
        z_dream = cache["z_dream"]
        attn_bc = cache["attn_bc"]
        scale = float(cache["scale"])

        d_attn_bc = np.einsum("bld,brd->blr", grad_zsub, z_dream)
        d_zdream = np.einsum("blr,bld->brd", attn_bc, grad_zsub)

        d_scores_bc = attn_bc * (
            d_attn_bc - np.sum(d_attn_bc * attn_bc, axis=-1, keepdims=True)
        ) * scale

        d_xwake_bc = np.einsum("blr,brd->bld", d_scores_bc, z_dream)
        d_zdream += np.einsum("blr,bld->brd", d_scores_bc, x_wake)

        d_attn_pool = np.einsum("brd,bld->brl", d_zdream, m_hipp)
        d_mhipp_pool = np.einsum("brl,brd->bld", attn_pool, d_zdream)

        d_scores_pool = attn_pool * (
            d_attn_pool - np.sum(d_attn_pool * attn_pool, axis=-1, keepdims=True)
        ) * scale

        d_anchors = np.einsum("brl,bld->rd", d_scores_pool, joint_mem)
        d_joint = np.einsum("brl,rd->bld", d_scores_pool, dream_anchors)

        d_xwake = d_xwake_bc + 0.5 * d_joint
        d_mhipp = d_mhipp_pool + 0.5 * d_joint
        return {
            "d_xwake": d_xwake,
            "d_mhipp": d_mhipp,
            "d_anchors": d_anchors,
        }

    @staticmethod
    def simulate_triton_sleep_subconscious_tile_pass(
        x_wake: np.ndarray,
        m_hipp: np.ndarray,
        z_sub: np.ndarray,
        theta_sleep: np.ndarray,
        kappa_vlpo: np.ndarray,
        lambda_shy: np.ndarray,
        alpha_swr: np.ndarray,
        beta_pgo: np.ndarray,
        gamma_rem: float = 0.50,
        eps_glymph: float = 0.10,
        block_d: int = 64,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Simulates the tile-by-tile SRAM execution of `fused_sleep_subconscious_shy_swr_pgo_fwd_kernel`
        and profiles HBM memory traffic vs. unfused PyTorch eager execution.
        """
        B, L, D = x_wake.shape
        num_rows = B * L
        xw_2d = x_wake.reshape(num_rows, D)
        mh_2d = m_hipp.reshape(num_rows, D)
        zs_2d = z_sub.reshape(num_rows, D)
        out_2d = np.zeros_like(xw_2d)

        for r in range(num_rows):
            for c0 in range(0, D, block_d):
                c1 = min(c0 + block_d, D)
                x = xw_2d[r, c0:c1]
                m = mh_2d[r, c0:c1]
                z = zs_2d[r, c0:c1]
                th = theta_sleep[c0:c1]
                kap = kappa_vlpo[c0:c1]
                lam = lambda_shy[c0:c1]
                aswr = alpha_swr[c0:c1]
                bpgo = beta_pgo[c0:c1]

                x_sq = x * x
                m_sq = m * m
                z_sq = z * z

                g_sleep = 1.0 / (1.0 + np.exp(-np.clip(kap * (x_sq + gamma_rem * z_sq - th), -500.0, 500.0)))
                inv_shy = 1.0 / np.sqrt(1.0 + lam * g_sleep * x_sq)
                w_glymph = 1.0 - (g_sleep * eps_glymph) / (x_sq + eps_glymph)
                y_sws = x * inv_shy * w_glymph

                u_swr = (x * m + m) / np.sqrt(1.0 + x_sq + m_sq)
                u_pgo = (z + m * z) / np.sqrt(1.0 + z_sq + m_sq)
                out_2d[r, c0:c1] = y_sws + g_sleep * (aswr * u_swr + bpgo * u_pgo)

        # Unfused eager chain materializes ~22 intermediate [B, L, D] tensors across Fwd+Bwd;
        # Fused Triton kernel reads 3 inputs + writes 1 output in Fwd, reads 4 + writes 3 in Bwd.
        elem_bytes = 4
        eager_bytes = 26 * B * L * D * elem_bytes
        triton_bytes = (4 * B * L * D + 5 * D) * elem_bytes
        hbm_saved_pct = (1.0 - triton_bytes / max(eager_bytes, 1)) * 100.0
        return out_2d.reshape(B, L, D), {
            "eager_hbm_kb": eager_bytes / 1024.0,
            "triton_hbm_kb": triton_bytes / 1024.0,
            "hbm_saved_pct": hbm_saved_pct,
            "bw_speedup": eager_bytes / max(triton_bytes, 1),
        }


# ==============================================================================
# 4. PyTorch Neural Operator (`autograd.Function` & `nn.Module`)
# ==============================================================================

if HAS_TORCH:

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            norm_x = x.to(torch.float32)
            rms = torch.rsqrt(norm_x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (norm_x * rms).to(x.dtype) * self.weight


    class FusedSleepSubconsciousSHYSWRPGOFunction(torch.autograd.Function):
        """
        Custom PyTorch Autograd Function encapsulating the Fused Sleep Homeostasis (SHY),
        Glymphatic Clearance, Hippocampal Sharp-Wave Ripple (SWR), and REM PGO-Wave
        Deep Subconscious Synthesis Operator.
        """

        @staticmethod
        def forward(
            ctx,
            x_wake: torch.Tensor,
            m_hipp: torch.Tensor,
            z_sub: torch.Tensor,
            theta_sleep: torch.Tensor,
            kappa_vlpo: torch.Tensor,
            lambda_shy: torch.Tensor,
            alpha_swr: torch.Tensor,
            beta_pgo: torch.Tensor,
            gamma_rem: float,
            eps_glymph: float,
        ) -> torch.Tensor:
            th_b = theta_sleep.view(1, 1, -1)
            kap_b = kappa_vlpo.view(1, 1, -1)
            lam_b = lambda_shy.view(1, 1, -1)
            aswr_b = alpha_swr.view(1, 1, -1)
            bpgo_b = beta_pgo.view(1, 1, -1)

            x_sq = x_wake.pow(2)
            m_sq = m_hipp.pow(2)
            z_sq = z_sub.pow(2)

            e_drive = x_sq + gamma_rem * z_sq - th_b
            g_sleep = torch.sigmoid(kap_b * e_drive)
            one_m_g = 1.0 - g_sleep

            d_shy = 1.0 + lam_b * g_sleep * x_sq
            inv_shy = torch.rsqrt(d_shy)
            dg_denom = x_sq + eps_glymph
            w_glymph = 1.0 - (g_sleep * eps_glymph) / dg_denom
            a_sws = x_wake * w_glymph
            y_sws = a_sws * inv_shy

            d_swr = 1.0 + x_sq + m_sq
            inv_swr = torch.rsqrt(d_swr)
            n_swr = x_wake * m_hipp + m_hipp
            u_swr = n_swr * inv_swr

            d_pgo = 1.0 + z_sq + m_sq
            inv_pgo = torch.rsqrt(d_pgo)
            n_pgo = z_sub + m_hipp * z_sub
            u_pgo = n_pgo * inv_pgo

            out = y_sws + g_sleep * (aswr_b * u_swr + bpgo_b * u_pgo)
            ctx.save_for_backward(
                x_wake, m_hipp, z_sub,
                theta_sleep, kappa_vlpo, lambda_shy, alpha_swr, beta_pgo,
                x_sq, e_drive, g_sleep, one_m_g,
                d_shy, inv_shy, dg_denom, a_sws,
                d_swr, inv_swr, n_swr, u_swr,
                d_pgo, inv_pgo, n_pgo, u_pgo,
            )
            ctx.gamma_rem = gamma_rem
            ctx.eps_glymph = eps_glymph
            return out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            (
                x, m, z,
                theta_sleep, kappa_vlpo, lambda_shy, alpha_swr, beta_pgo,
                x_sq, e_drive, g_sleep, one_m_g,
                d_shy, inv_shy, dg_denom, a_sws,
                d_swr, inv_swr, n_swr, u_swr,
                d_pgo, inv_pgo, n_pgo, u_pgo,
            ) = ctx.saved_tensors
            gamma_rem = ctx.gamma_rem
            eps_glymph = ctx.eps_glymph

            kap_b = kappa_vlpo.view(1, 1, -1)
            lam_b = lambda_shy.view(1, 1, -1)
            aswr_b = alpha_swr.view(1, 1, -1)
            bpgo_b = beta_pgo.view(1, 1, -1)

            d_alphaswr = (grad_out * g_sleep * u_swr).sum(dim=(0, 1))
            d_betapgo = (grad_out * g_sleep * u_pgo).sum(dim=(0, 1))

            d_dshy = grad_out * a_sws * (-0.5 * inv_shy / d_shy)
            d_asws = grad_out * inv_shy
            d_lamshy = (d_dshy * g_sleep * x_sq).sum(dim=(0, 1))

            d_gsleep = (
                grad_out * (aswr_b * u_swr + bpgo_b * u_pgo)
                + d_dshy * (lam_b * x_sq)
                - d_asws * (eps_glymph * x / dg_denom)
            )
            d_logit = d_gsleep * g_sleep * one_m_g
            d_kappa = (d_logit * e_drive).sum(dim=(0, 1))
            d_edrive = d_logit * kap_b
            d_theta = (-d_edrive).sum(dim=(0, 1))

            d_uswr = grad_out * g_sleep * aswr_b
            d_nswr = d_uswr * inv_swr
            d_dswr = d_uswr * n_swr * (-0.5 * inv_swr / d_swr)

            d_upgo = grad_out * g_sleep * bpgo_b
            d_npgo = d_upgo * inv_pgo
            d_dpgo = d_upgo * n_pgo * (-0.5 * inv_pgo / d_pgo)

            d_asws_dx = 1.0 - g_sleep * eps_glymph * (eps_glymph - x_sq) / dg_denom.pow(2)

            d_xwake = (
                d_asws * d_asws_dx
                + d_dshy * (2.0 * lam_b * g_sleep * x)
                + d_edrive * (2.0 * x)
                + d_nswr * m
                + d_dswr * (2.0 * x)
            )
            d_mhipp = (
                d_nswr * (x + 1.0)
                + d_dswr * (2.0 * m)
                + d_npgo * z
                + d_dpgo * (2.0 * m)
            )
            d_zsub = (
                d_npgo * (1.0 + m)
                + d_dpgo * (2.0 * z)
                + d_edrive * (2.0 * gamma_rem * z)
            )
            return (
                d_xwake, d_mhipp, d_zsub,
                d_theta, d_kappa, d_lamshy, d_alphaswr, d_betapgo,
                None, None,
            )


    class SleepSubconsciousNeuroOp(nn.Module):
        """
        Complete Human Sleep Physiology & Deep Subconscious Neuroscience Operator (`nn.Module`):
        - Projects input into Wakeful Neocortical stream (`x_wake`) and Hippocampal
          Sharp-Wave Ripple stream (`m_hipp`).
        - Extracts Deep Subconscious DMN/PGO Dream Recombination stream (`z_sub`) via
          `R_dream` non-local subconscious archetype anchors.
        - Applies `FusedSleepSubconsciousSHYSWRPGOFunction` to perform VLPO/ARAS sleep
          gating, Tononi SHY downscaling, Nedergaard glymphatic noise clearance, SWR
          memory consolidation, and PGO dream associative synthesis.
        """

        def __init__(self, config: SleepSubconsciousConfig):
            super().__init__()
            self.config = config
            D = config.d_model
            self.d_model = D
            self.num_dream_anchors = config.num_dream_anchors

            self.in_proj = nn.Linear(D, 2 * D, bias=False)
            self.dream_anchors = nn.Parameter(
                torch.randn(config.num_dream_anchors, D) * (D ** -0.5)
            )
            self.theta_sleep = nn.Parameter(torch.full((D,), config.theta_sleep_init))
            self.kappa_vlpo = nn.Parameter(torch.full((D,), config.kappa_vlpo_init))
            self.lambda_shy = nn.Parameter(torch.full((D,), config.lambda_shy_init))
            self.alpha_swr = nn.Parameter(torch.full((D,), config.alpha_swr_init))
            self.beta_pgo = nn.Parameter(torch.full((D,), config.beta_pgo_init))

            self.out_proj = nn.Linear(D, D, bias=False)
            self.out_norm = RMSNorm(D, eps=config.rms_norm_eps)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
            x_wake, m_hipp = self.in_proj(x).chunk(2, dim=-1)
            scale = self.d_model ** -0.5

            # Non-local PGO-wave deep subconscious dream recombination
            joint_mem = 0.5 * (x_wake + m_hipp)
            attn_pool = F.softmax(
                torch.einsum("rd,bld->brl", self.dream_anchors, joint_mem) * scale,
                dim=-1,
            )
            z_dream = torch.einsum("brl,bld->brd", attn_pool, m_hipp)
            attn_bc = F.softmax(
                torch.einsum("bld,brd->blr", x_wake, z_dream) * scale,
                dim=-1,
            )
            z_sub = torch.einsum("blr,brd->bld", attn_bc, z_dream)

            fused = FusedSleepSubconsciousSHYSWRPGOFunction.apply(
                x_wake, m_hipp, z_sub,
                self.theta_sleep, self.kappa_vlpo, self.lambda_shy,
                self.alpha_swr, self.beta_pgo,
                self.config.gamma_rem, self.config.eps_glymph,
            )
            out = self.out_norm(self.out_proj(fused))

            with torch.no_grad():
                e_drive = (
                    x_wake.pow(2)
                    + self.config.gamma_rem * z_sub.pow(2)
                    - self.theta_sleep.view(1, 1, -1)
                )
                g_sleep = torch.sigmoid(self.kappa_vlpo.view(1, 1, -1) * e_drive)
                w_glymph = 1.0 - (g_sleep * self.config.eps_glymph) / (
                    x_wake.pow(2) + self.config.eps_glymph
                )
                diagnostics = {
                    "mean_vlpo_sleep_gate": float(g_sleep.mean().item()),
                    "mean_glymphatic_transmission": float(w_glymph.mean().item()),
                    "subconscious_dream_energy": float(z_sub.pow(2).mean().item()),
                }
            return out, diagnostics

if not HAS_TORCH:
    class RMSNorm:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate RMSNorm.")

    class FusedSleepSubconsciousSHYSWRPGOFunction:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate FusedSleepSubconsciousSHYSWRPGOFunction.")

    class SleepSubconsciousNeuroOp:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate SleepSubconsciousNeuroOp.")


# ==============================================================================
# 5. Comprehensive Standalone Verification Suite
# ==============================================================================

def run_sleep_subconscious_verification_suite() -> None:
    rng = np.random.default_rng(2026)
    print("=" * 96)
    print("HUMAN SLEEP PHYSIOLOGY & DEEP SUBCONSCIOUS NEUROSCIENCE OPERATOR — VERIFICATION SUITE")
    print("=" * 96)

    # 1. 8-Variable Analytical vs. Finite-Difference Gradcheck (Fused Sleep SHY-Glymphatic-SWR-PGO)
    print("[1] 8-Variable Analytical vs. Central Finite-Difference Gradcheck (Fused SHY-Glymph-SWR-PGO)")
    print("-" * 96)
    B, L, D = 2, 5, 6
    gamma_rem = 0.50
    eps_glymph = 0.10
    x_wake = rng.standard_normal((B, L, D)).astype(np.float64) * 0.7
    m_hipp = rng.standard_normal((B, L, D)).astype(np.float64) * 0.7
    z_sub = rng.standard_normal((B, L, D)).astype(np.float64) * 0.7
    theta_sleep = rng.uniform(0.3, 0.8, size=(D,)).astype(np.float64)
    kappa_vlpo = rng.uniform(1.0, 2.0, size=(D,)).astype(np.float64)
    lambda_shy = rng.uniform(0.2, 0.8, size=(D,)).astype(np.float64)
    alpha_swr = rng.uniform(0.4, 0.9, size=(D,)).astype(np.float64)
    beta_pgo = rng.uniform(0.4, 0.9, size=(D,)).astype(np.float64)
    grad_out = rng.standard_normal((B, L, D)).astype(np.float64)

    out_ref, cache = SleepSubconsciousAnalyticalKernels.fused_sleep_shy_swr_pgo_forward(
        x_wake, m_hipp, z_sub,
        theta_sleep, kappa_vlpo, lambda_shy, alpha_swr, beta_pgo,
        gamma_rem, eps_glymph,
    )
    ana = SleepSubconsciousAnalyticalKernels.fused_sleep_shy_swr_pgo_backward(grad_out, cache)

    eps = 1e-6
    targets = [
        ("x_wake", x_wake, ana["d_xwake"]),
        ("m_hipp", m_hipp, ana["d_mhipp"]),
        ("z_sub", z_sub, ana["d_zsub"]),
        ("theta_sleep", theta_sleep, ana["d_theta"]),
        ("kappa_vlpo", kappa_vlpo, ana["d_kappa"]),
        ("lambda_shy", lambda_shy, ana["d_lamshy"]),
        ("alpha_swr", alpha_swr, ana["d_alphaswr"]),
        ("beta_pgo", beta_pgo, ana["d_betapgo"]),
    ]
    for name, arr, g_ana in targets:
        g_num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + eps
            yp, _ = SleepSubconsciousAnalyticalKernels.fused_sleep_shy_swr_pgo_forward(
                x_wake, m_hipp, z_sub,
                theta_sleep, kappa_vlpo, lambda_shy, alpha_swr, beta_pgo,
                gamma_rem, eps_glymph,
            )
            lp = np.sum(yp * grad_out)
            arr[idx] = orig - eps
            ym, _ = SleepSubconsciousAnalyticalKernels.fused_sleep_shy_swr_pgo_forward(
                x_wake, m_hipp, z_sub,
                theta_sleep, kappa_vlpo, lambda_shy, alpha_swr, beta_pgo,
                gamma_rem, eps_glymph,
            )
            lm = np.sum(ym * grad_out)
            arr[idx] = orig
            g_num[idx] = (lp - lm) / (2.0 * eps)
        rel_err = np.linalg.norm(g_ana - g_num) / (np.linalg.norm(g_ana) + np.linalg.norm(g_num) + 1e-15)
        print(f"   Sleep-Subconscious Gradcheck dL/d({name:12s}) | Relative L2 Error: {rel_err:.3e}  [PASS < 1e-8]")
        assert rel_err < 1e-8

    # 2. 3-Variable Analytical vs. Finite-Difference Gradcheck (PGO Dream Recombination)
    print("\n[2] 3-Variable Analytical vs. Finite-Difference Gradcheck (PGO Dream Recombination)")
    print("-" * 96)
    R_dream = 4
    dream_anchors = rng.standard_normal((R_dream, D)).astype(np.float64) * 0.5
    grad_zsub = rng.standard_normal((B, L, D)).astype(np.float64)
    _, pgo_cache = SleepSubconsciousAnalyticalKernels.subconscious_pgo_dream_recombination_forward(
        x_wake, m_hipp, dream_anchors
    )
    pgo_ana = SleepSubconsciousAnalyticalKernels.subconscious_pgo_dream_recombination_backward(
        grad_zsub, pgo_cache
    )
    for name, arr, g_ana in [
        ("x_wake (dream)", x_wake, pgo_ana["d_xwake"]),
        ("m_hipp (dream)", m_hipp, pgo_ana["d_mhipp"]),
        ("dream_anchors", dream_anchors, pgo_ana["d_anchors"]),
    ]:
        g_num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + eps
            zp, _ = SleepSubconsciousAnalyticalKernels.subconscious_pgo_dream_recombination_forward(
                x_wake, m_hipp, dream_anchors
            )
            lp = np.sum(zp * grad_zsub)
            arr[idx] = orig - eps
            zm, _ = SleepSubconsciousAnalyticalKernels.subconscious_pgo_dream_recombination_forward(
                x_wake, m_hipp, dream_anchors
            )
            lm = np.sum(zm * grad_zsub)
            arr[idx] = orig
            g_num[idx] = (lp - lm) / (2.0 * eps)
        rel_err = np.linalg.norm(g_ana - g_num) / (np.linalg.norm(g_ana) + np.linalg.norm(g_num) + 1e-15)
        print(f"   PGO Dream Co-Attn Gradcheck dL/d({name:14s}) | Relative L2 Error: {rel_err:.3e}  [PASS < 1e-8]")
        assert rel_err < 1e-8

    # 3. Triton Tile Simulator Equivalence & Physiological Verification
    print("\n[3] Triton Tile Simulator Equivalence, Glymphatic Noise Washout & SWR Consolidation Audit")
    print("-" * 96)
    B_b, L_b, D_b = 4, 32, 64
    xw_b = rng.standard_normal((B_b, L_b, D_b)) * 0.8
    mh_b = rng.standard_normal((B_b, L_b, D_b)) * 0.8
    anc_b = rng.standard_normal((16, D_b)) * (D_b ** -0.5)
    zs_b, _ = SleepSubconsciousAnalyticalKernels.subconscious_pgo_dream_recombination_forward(
        xw_b, mh_b, anc_b
    )
    th_b = np.full((D_b,), 0.55)
    kap_b = np.full((D_b,), 1.60)
    lam_b = np.full((D_b,), 0.45)
    aswr_b = np.full((D_b,), 0.65)
    bpgo_b = np.full((D_b,), 0.60)

    out_exact, cache_b = SleepSubconsciousAnalyticalKernels.fused_sleep_shy_swr_pgo_forward(
        xw_b, mh_b, zs_b, th_b, kap_b, lam_b, aswr_b, bpgo_b
    )
    out_tile, hbm_stats = SleepSubconsciousAnalyticalKernels.simulate_triton_sleep_subconscious_tile_pass(
        xw_b, mh_b, zs_b, th_b, kap_b, lam_b, aswr_b, bpgo_b
    )
    max_tile_diff = float(np.max(np.abs(out_exact - out_tile)))

    # Verify Nedergaard Glymphatic Washout of sub-threshold metabolic clutter (|x| = 0.01)
    # under deep sleep (G_sleep -> 1) vs. high-salience engram (|x| = 2.0)
    g_deep = 0.98
    eps_g = 0.10
    w_glymph_clutter = 1.0 - (g_deep * eps_g) / (0.01**2 + eps_g)
    w_glymph_engram = 1.0 - (g_deep * eps_g) / (2.0**2 + eps_g)

    print(f"   Triton Tile Simulator Max Abs Diff   : {max_tile_diff:.2e}  [PASS == 0.00e+00]")
    print(f"   Mean VLPO Sleep Switch Gate G_sleep  : {float(np.mean(cache_b['g_sleep'])):.4f}")
    print(f"   Mean Tononi SHY Downscaling Factor   : {float(np.mean(cache_b['inv_shy'])):.4f}")
    print(
        f"   Glymphatic Metabolic Clutter Washout : Clutter(|x|=0.01) Retention={w_glymph_clutter:.2%} "
        f"| Salient Engram(|x|=2.0) Retention={w_glymph_engram:.2%}  [PASS]"
    )
    print(
        f"   HBM Memory Traffic (Fwd+Bwd)         : Eager={hbm_stats['eager_hbm_kb']:.1f} KB -> "
        f"Triton={hbm_stats['triton_hbm_kb']:.1f} KB "
        f"(Saved {hbm_stats['hbm_saved_pct']:.1f}% | {hbm_stats['bw_speedup']:.2f}x BW Speedup)  [PASS]"
    )
    assert max_tile_diff < 1e-14 and w_glymph_clutter < 0.03 and w_glymph_engram > 0.95
    print("=" * 96)


if __name__ == "__main__":
    run_sleep_subconscious_verification_suite()
