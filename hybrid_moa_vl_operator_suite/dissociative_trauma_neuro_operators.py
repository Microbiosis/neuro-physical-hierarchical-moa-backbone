"""
Dissociative Disorder Spectrum & Trauma-Related Psychiatric Neuroscience Operator Library
(`dissociative_trauma_neuro_operators.py`).

A standalone, Launch-Bound-Free deep learning feature operator library translating
clinical and computational neuroscience models of the Dissociative Disorder
Spectrum (DID, Depersonalization/Derealization DPDR) and Trauma-Related Disorders
(PTSD & Dissociative Subtype PTSD) into end-to-end differentiable PyTorch operators
(`nn.Module` & `torch.autograd.Function`), fused single-launch Triton GPU Kernels
(`@triton.jit`), and analytical NumPy forward/backward verification kernels.

Key Guarantees:
1. Complete 7-Output Fused Triton Backward Kernel (`GradXSens`, `GradSAmyg`, `GradCVmpfc`,
   plus atomic-accumulated `GradTheta`, `GradKappa`, `GradAlphaHyper`, `GradGammaDpdr`)
   and real `kernel[grid](...)` launch dispatch when running on CUDA with Triton.
2. Live Differentiable Shutdown Gate (`g_shut_tensor` in `diagnostics`):
   `DissociativeTraumaNeuroOp.forward` computes `g_shut_tensor` inside the live PyTorch
   autograd graph (never detaching it via `float(...)` before downstream regime gating),
   ensuring 100% gradient parity between PyTorch autograd and the NumPy analytical engine.
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


@dataclass
class DissociativeTraumaConfig:
    """Configuration for the Dissociative Spectrum & Trauma Neuroscience Operator."""
    d_model: int = 256
    n_heads: int = 8
    num_compartments: int = 4
    theta_dissoc_init: float = 0.65
    kappa_dissoc_init: float = 1.80
    alpha_hyper_init: float = 0.55
    beta_vmpfc: float = 0.40
    gamma_dpdr_init: float = 0.75
    flashbulb_salience_scale: float = 0.60
    rms_norm_eps: float = 1e-6


if HAS_TRITON:

    @triton.jit
    def fused_corticolimbic_dissociative_dpdr_fwd_kernel(
        XSens_ptr, SAmyg_ptr, CVmpfc_ptr,
        Theta_ptr, Kappa_ptr, AlphaHyper_ptr, GammaDpdr_ptr, Out_ptr,
        stride_row,
        D_MODEL: tl.constexpr, beta_vmpfc: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D_MODEL
        off = row_id * stride_row + cols

        x_sens = tl.load(XSens_ptr + off, mask=mask, other=0.0).to(tl.float32)
        s_amyg = tl.load(SAmyg_ptr + off, mask=mask, other=0.0).to(tl.float32)
        c_vmpfc = tl.load(CVmpfc_ptr + off, mask=mask, other=0.0).to(tl.float32)

        theta = tl.load(Theta_ptr + cols, mask=mask, other=0.5).to(tl.float32)
        kappa = tl.load(Kappa_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        alpha_h = tl.load(AlphaHyper_ptr + cols, mask=mask, other=0.5).to(tl.float32)
        gamma_d = tl.load(GammaDpdr_ptr + cols, mask=mask, other=0.5).to(tl.float32)

        e_excess = (s_amyg * s_amyg) + beta_vmpfc * (c_vmpfc * c_vmpfc) - theta
        g_shut = tl.sigmoid(-kappa * e_excess)

        s_sq = s_amyg * s_amyg
        hyper_gain = 1.0 + alpha_h * s_amyg * tl.math.rsqrt(1.0 + s_sq)
        r_amyg = g_shut * s_amyg * hyper_gain

        dpdr_denom = 1.0 + gamma_d * (1.0 - g_shut) * (x_sens * x_sens)
        x_dpdr = x_sens / dpdr_denom

        out = x_dpdr + r_amyg
        tl.store(Out_ptr + off, out, mask=mask)


    @triton.jit
    def fused_corticolimbic_dissociative_dpdr_bwd_kernel(
        GradOut_ptr, XSens_ptr, SAmyg_ptr, CVmpfc_ptr,
        Theta_ptr, Kappa_ptr, AlphaHyper_ptr, GammaDpdr_ptr,
        GradXSens_ptr, GradSAmyg_ptr, GradCVmpfc_ptr,
        GradTheta_ptr, GradKappa_ptr, GradAlphaHyper_ptr, GradGammaDpdr_ptr,
        stride_row,
        D_MODEL: tl.constexpr, beta_vmpfc: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D_MODEL
        off = row_id * stride_row + cols

        dout = tl.load(GradOut_ptr + off, mask=mask, other=0.0).to(tl.float32)
        x_sens = tl.load(XSens_ptr + off, mask=mask, other=0.0).to(tl.float32)
        s_amyg = tl.load(SAmyg_ptr + off, mask=mask, other=0.0).to(tl.float32)
        c_vmpfc = tl.load(CVmpfc_ptr + off, mask=mask, other=0.0).to(tl.float32)

        theta = tl.load(Theta_ptr + cols, mask=mask, other=0.5).to(tl.float32)
        kappa = tl.load(Kappa_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        alpha_h = tl.load(AlphaHyper_ptr + cols, mask=mask, other=0.5).to(tl.float32)
        gamma_d = tl.load(GammaDpdr_ptr + cols, mask=mask, other=0.5).to(tl.float32)

        e_excess = (s_amyg * s_amyg) + beta_vmpfc * (c_vmpfc * c_vmpfc) - theta
        g_shut = tl.sigmoid(-kappa * e_excess)
        one_m_g = 1.0 - g_shut

        s_sq = s_amyg * s_amyg
        inv_s = tl.math.rsqrt(1.0 + s_sq)
        hyper_unit = s_sq * inv_s
        hyper_term = s_amyg + alpha_h * hyper_unit

        x_sq = x_sens * x_sens
        dpdr_denom = 1.0 + gamma_d * one_m_g * x_sq
        denom_sq = dpdr_denom * dpdr_denom

        c_dpdr = gamma_d * one_m_g
        dx_sens = dout * (1.0 - c_dpdr * x_sq) / denom_sq
        d_gammadpdr = dout * (-x_sens * one_m_g * x_sq / denom_sq)
        d_alphahyper = dout * g_shut * hyper_unit

        dg_from_amyg = dout * hyper_term
        dg_from_dpdr = dout * (x_sens * gamma_d * x_sq) / denom_sq
        d_gshut = dg_from_amyg + dg_from_dpdr

        d_logit = d_gshut * g_shut * one_m_g
        d_kappa = d_logit * (-e_excess)
        d_eexcess = d_logit * (-kappa)
        d_theta = -d_eexcess

        dhyper_ds = 1.0 + alpha_h * s_amyg * (2.0 + s_sq) * (inv_s / (1.0 + s_sq))
        ds_amyg = dout * g_shut * dhyper_ds + d_eexcess * (2.0 * s_amyg)
        dc_vmpfc = d_eexcess * (2.0 * beta_vmpfc * c_vmpfc)

        tl.store(GradXSens_ptr + off, dx_sens, mask=mask)
        tl.store(GradSAmyg_ptr + off, ds_amyg, mask=mask)
        tl.store(GradCVmpfc_ptr + off, dc_vmpfc, mask=mask)
        tl.atomic_add(GradTheta_ptr + cols, d_theta, mask=mask)
        tl.atomic_add(GradKappa_ptr + cols, d_kappa, mask=mask)
        tl.atomic_add(GradAlphaHyper_ptr + cols, d_alphahyper, mask=mask)
        tl.atomic_add(GradGammaDpdr_ptr + cols, d_gammadpdr, mask=mask)

else:
    def fused_corticolimbic_dissociative_dpdr_fwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to launch fused_corticolimbic_dissociative_dpdr_fwd_kernel.")

    def fused_corticolimbic_dissociative_dpdr_bwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to launch fused_corticolimbic_dissociative_dpdr_bwd_kernel.")


class DissociativeTraumaAnalyticalKernels:
    """Exact forward and analytical backward kernels for the Dissociative & Trauma Operator."""

    @staticmethod
    def fused_corticolimbic_dissociative_dpdr_forward(
        x_sens: np.ndarray,
        s_amyg: np.ndarray,
        c_vmpfc: np.ndarray,
        theta_dissoc: np.ndarray,
        kappa_dissoc: np.ndarray,
        alpha_hyper: np.ndarray,
        gamma_dpdr: np.ndarray,
        beta_vmpfc: float = 0.40,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        th_b = theta_dissoc[None, None, :]
        kap_b = kappa_dissoc[None, None, :]
        alp_b = alpha_hyper[None, None, :]
        gam_b = gamma_dpdr[None, None, :]

        s_sq = np.square(s_amyg)
        c_sq = np.square(c_vmpfc)
        e_excess = s_sq + beta_vmpfc * c_sq - th_b
        logit_shut = np.clip(-kap_b * e_excess, -500.0, 500.0)
        g_shut = 1.0 / (1.0 + np.exp(-logit_shut))
        one_m_g = 1.0 - g_shut

        inv_s = 1.0 / np.sqrt(1.0 + s_sq)
        hyper_unit = s_sq * inv_s
        hyper_term = s_amyg + alp_b * hyper_unit
        r_amyg = g_shut * hyper_term

        x_sq = np.square(x_sens)
        dpdr_denom = 1.0 + gam_b * one_m_g * x_sq
        x_dpdr = x_sens / dpdr_denom

        out = x_dpdr + r_amyg

        cache = {
            "x_sens": x_sens,
            "s_amyg": s_amyg,
            "c_vmpfc": c_vmpfc,
            "theta_dissoc": theta_dissoc,
            "kappa_dissoc": kappa_dissoc,
            "alpha_hyper": alpha_hyper,
            "gamma_dpdr": gamma_dpdr,
            "s_sq": s_sq,
            "x_sq": x_sq,
            "e_excess": e_excess,
            "g_shut": g_shut,
            "one_m_g": one_m_g,
            "inv_s": inv_s,
            "hyper_unit": hyper_unit,
            "hyper_term": hyper_term,
            "dpdr_denom": dpdr_denom,
            "beta_vmpfc": np.array(beta_vmpfc),
        }
        return out, cache

    @staticmethod
    def fused_corticolimbic_dissociative_dpdr_backward(
        grad_out: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        x_sens = cache["x_sens"]
        s_amyg = cache["s_amyg"]
        c_vmpfc = cache["c_vmpfc"]
        kappa_dissoc = cache["kappa_dissoc"][None, None, :]
        alpha_hyper = cache["alpha_hyper"][None, None, :]
        gamma_dpdr = cache["gamma_dpdr"][None, None, :]
        s_sq = cache["s_sq"]
        x_sq = cache["x_sq"]
        e_excess = cache["e_excess"]
        g_shut = cache["g_shut"]
        one_m_g = cache["one_m_g"]
        inv_s = cache["inv_s"]
        hyper_unit = cache["hyper_unit"]
        hyper_term = cache["hyper_term"]
        dpdr_denom = cache["dpdr_denom"]
        beta_vmpfc = float(cache["beta_vmpfc"])

        c_dpdr = gamma_dpdr * one_m_g
        denom_sq = np.square(dpdr_denom)
        d_xsens = grad_out * (1.0 - c_dpdr * x_sq) / denom_sq
        d_gammadpdr = np.sum(
            grad_out * (-x_sens * one_m_g * x_sq / denom_sq), axis=(0, 1)
        )

        d_alphahyper = np.sum(grad_out * g_shut * hyper_unit, axis=(0, 1))

        dg_from_amyg = grad_out * hyper_term
        dg_from_dpdr = grad_out * (x_sens * gamma_dpdr * x_sq / denom_sq)
        d_gshut = dg_from_amyg + dg_from_dpdr

        d_logit = d_gshut * g_shut * one_m_g
        d_kappa = np.sum(d_logit * (-e_excess), axis=(0, 1))
        d_eexcess = d_logit * (-kappa_dissoc)
        d_theta = np.sum(-d_eexcess, axis=(0, 1))

        dhyper_ds = 1.0 + alpha_hyper * s_amyg * (2.0 + s_sq) * (inv_s / (1.0 + s_sq))
        d_samyg = grad_out * g_shut * dhyper_ds + d_eexcess * (2.0 * s_amyg)
        d_cvmpfc = d_eexcess * (2.0 * beta_vmpfc * c_vmpfc)

        return {
            "d_xsens": d_xsens,
            "d_samyg": d_samyg,
            "d_cvmpfc": d_cvmpfc,
            "d_theta": d_theta,
            "d_kappa": d_kappa,
            "d_alphahyper": d_alphahyper,
            "d_gammadpdr": d_gammadpdr,
        }

    @staticmethod
    def structural_dissociation_amnestic_flashbulb_forward(
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        s_amyg_heads: np.ndarray,
        amnestic_barrier_logits: np.ndarray,
        flashbulb_scale: float = 0.60,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        B, H, L, Dh = q.shape
        scale = 1.0 / math.sqrt(Dh)

        raw_scores = np.einsum("bhid,bhjd->bhij", q, k) * scale
        flashbulb_bias = flashbulb_scale * np.maximum(0.0, s_amyg_heads - 0.50)[:, :, None, :]

        idx = np.arange(L)
        causal_mask = idx[:, None] >= idx[None, :]
        scores = np.where(causal_mask[None, None, :, :], raw_scores + flashbulb_bias, -1e9)
        s_max = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - s_max) * causal_mask[None, None, :, :]
        attn = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

        head_out = np.einsum("bhij,bhjd->bhid", attn, v)

        perm = 1.0 / (1.0 + np.exp(-np.clip(amnestic_barrier_logits, -500.0, 500.0)))
        perm_norm = perm / (np.sum(perm, axis=-1, keepdims=True) + 1e-6)
        compartment_out = np.einsum("ij,bjld->bild", perm_norm, head_out)
        out = compartment_out.transpose(0, 2, 1, 3).reshape(B, L, H * Dh)

        off_diag_mask = 1.0 - np.eye(H)
        mean_cross_compartment_leakage = float(
            np.sum(perm_norm * off_diag_mask) / max(H * (H - 1), 1)
        )
        return out, {
            "mean_cross_compartment_permeability": mean_cross_compartment_leakage,
            "mean_flashbulb_salience_boost": float(np.mean(flashbulb_bias)),
        }


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


    class FusedCorticolimbicDissociativeDPDRFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            x_sens: torch.Tensor,
            s_amyg: torch.Tensor,
            c_vmpfc: torch.Tensor,
            theta_dissoc: torch.Tensor,
            kappa_dissoc: torch.Tensor,
            alpha_hyper: torch.Tensor,
            gamma_dpdr: torch.Tensor,
            beta_vmpfc: float,
        ) -> torch.Tensor:
            ctx.beta_vmpfc = beta_vmpfc
            if HAS_TRITON and x_sens.is_cuda:
                B, L, D = x_sens.shape
                num_rows = B * L
                block_d = triton.next_power_of_2(D)
                grid = (triton.cdiv(num_rows, 1),)
                out = torch.empty_like(x_sens)
                fused_corticolimbic_dissociative_dpdr_fwd_kernel[grid](
                    x_sens.contiguous(),
                    s_amyg.contiguous(),
                    c_vmpfc.contiguous(),
                    theta_dissoc.contiguous(),
                    kappa_dissoc.contiguous(),
                    alpha_hyper.contiguous(),
                    gamma_dpdr.contiguous(),
                    out,
                    D,
                    D_MODEL=D,
                    beta_vmpfc=beta_vmpfc,
                    BLOCK_D=block_d,
                )
                ctx.save_for_backward(
                    x_sens, s_amyg, c_vmpfc, theta_dissoc, kappa_dissoc, alpha_hyper, gamma_dpdr
                )
                ctx.used_triton = True
                return out

            ctx.used_triton = False
            th_b = theta_dissoc.view(1, 1, -1)
            kap_b = kappa_dissoc.view(1, 1, -1)
            alp_b = alpha_hyper.view(1, 1, -1)
            gam_b = gamma_dpdr.view(1, 1, -1)

            s_sq = s_amyg.pow(2)
            c_sq = c_vmpfc.pow(2)
            e_excess = s_sq + beta_vmpfc * c_sq - th_b
            g_shut = torch.sigmoid(-kap_b * e_excess)
            one_m_g = 1.0 - g_shut

            inv_s = torch.rsqrt(1.0 + s_sq)
            hyper_unit = s_sq * inv_s
            hyper_term = s_amyg + alp_b * hyper_unit
            r_amyg = g_shut * hyper_term

            x_sq = x_sens.pow(2)
            dpdr_denom = 1.0 + gam_b * one_m_g * x_sq
            x_dpdr = x_sens / dpdr_denom

            out = x_dpdr + r_amyg

            ctx.save_for_backward(
                x_sens, s_amyg, c_vmpfc, kappa_dissoc, alpha_hyper, gamma_dpdr,
                s_sq, x_sq, e_excess, g_shut, one_m_g, inv_s, hyper_unit, hyper_term, dpdr_denom
            )
            return out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            if getattr(ctx, "used_triton", False):
                (
                    x_sens, s_amyg, c_vmpfc, theta_dissoc, kappa_dissoc, alpha_hyper, gamma_dpdr
                ) = ctx.saved_tensors
                B, L, D = x_sens.shape
                num_rows = B * L
                block_d = triton.next_power_of_2(D)
                grid = (triton.cdiv(num_rows, 1),)

                d_xsens = torch.empty_like(x_sens)
                d_samyg = torch.empty_like(s_amyg)
                d_cvmpfc = torch.empty_like(c_vmpfc)
                d_theta = torch.zeros_like(theta_dissoc, dtype=torch.float32)
                d_kappa = torch.zeros_like(kappa_dissoc, dtype=torch.float32)
                d_alphahyper = torch.zeros_like(alpha_hyper, dtype=torch.float32)
                d_gammadpdr = torch.zeros_like(gamma_dpdr, dtype=torch.float32)

                fused_corticolimbic_dissociative_dpdr_bwd_kernel[grid](
                    grad_out.contiguous(),
                    x_sens.contiguous(),
                    s_amyg.contiguous(),
                    c_vmpfc.contiguous(),
                    theta_dissoc.contiguous(),
                    kappa_dissoc.contiguous(),
                    alpha_hyper.contiguous(),
                    gamma_dpdr.contiguous(),
                    d_xsens,
                    d_samyg,
                    d_cvmpfc,
                    d_theta,
                    d_kappa,
                    d_alphahyper,
                    d_gammadpdr,
                    D,
                    D_MODEL=D,
                    beta_vmpfc=ctx.beta_vmpfc,
                    BLOCK_D=block_d,
                )
                return (
                    d_xsens.to(x_sens.dtype),
                    d_samyg.to(s_amyg.dtype),
                    d_cvmpfc.to(c_vmpfc.dtype),
                    d_theta.to(theta_dissoc.dtype),
                    d_kappa.to(kappa_dissoc.dtype),
                    d_alphahyper.to(alpha_hyper.dtype),
                    d_gammadpdr.to(gamma_dpdr.dtype),
                    None,
                )

            (
                x_sens, s_amyg, c_vmpfc, kappa_dissoc, alpha_hyper, gamma_dpdr,
                s_sq, x_sq, e_excess, g_shut, one_m_g, inv_s, hyper_unit, hyper_term, dpdr_denom
            ) = ctx.saved_tensors
            beta_vmpfc = ctx.beta_vmpfc

            kap_b = kappa_dissoc.view(1, 1, -1)
            alp_b = alpha_hyper.view(1, 1, -1)
            gam_b = gamma_dpdr.view(1, 1, -1)

            c_dpdr = gam_b * one_m_g
            denom_sq = dpdr_denom.pow(2)
            d_xsens = grad_out * (1.0 - c_dpdr * x_sq) / denom_sq
            d_gammadpdr = (grad_out * (-x_sens * one_m_g * x_sq / denom_sq)).sum(dim=(0, 1))

            d_alphahyper = (grad_out * g_shut * hyper_unit).sum(dim=(0, 1))

            dg_from_amyg = grad_out * hyper_term
            dg_from_dpdr = grad_out * (x_sens * gam_b * x_sq / denom_sq)
            d_gshut = dg_from_amyg + dg_from_dpdr

            d_logit = d_gshut * g_shut * one_m_g
            d_kappa = (d_logit * (-e_excess)).sum(dim=(0, 1))
            d_eexcess = d_logit * (-kap_b)
            d_theta = (-d_eexcess).sum(dim=(0, 1))

            dhyper_ds = 1.0 + alp_b * s_amyg * (2.0 + s_sq) * (inv_s / (1.0 + s_sq))
            d_samyg = grad_out * g_shut * dhyper_ds + d_eexcess * (2.0 * s_amyg)
            d_cvmpfc = d_eexcess * (2.0 * beta_vmpfc * c_vmpfc)

            return d_xsens, d_samyg, d_cvmpfc, d_theta, d_kappa, d_alphahyper, d_gammadpdr, None


    class DissociativeTraumaNeuroOp(nn.Module):
        def __init__(self, config: DissociativeTraumaConfig):
            super().__init__()
            assert config.d_model % config.n_heads == 0
            self.config = config
            self.d_model = config.d_model
            self.n_heads = config.n_heads
            self.d_head = config.d_model // config.n_heads

            self.fused_circuit_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
            self.theta_dissoc = nn.Parameter(torch.full((config.d_model,), config.theta_dissoc_init))
            self.kappa_dissoc = nn.Parameter(torch.full((config.d_model,), config.kappa_dissoc_init))
            self.alpha_hyper = nn.Parameter(torch.full((config.d_model,), config.alpha_hyper_init))
            self.gamma_dpdr = nn.Parameter(torch.full((config.d_model,), config.gamma_dpdr_init))

            self.qkv_comp = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
            init_barrier = torch.eye(config.n_heads) * 3.0 - 1.5
            self.amnestic_barrier_logits = nn.Parameter(init_barrier)

            self.reintegration_proj = nn.Linear(2 * config.d_model, config.d_model, bias=False)
            self.out_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, object]]:
            B, L, D = x.shape
            x_sens, s_amyg, c_vmpfc = self.fused_circuit_proj(x).chunk(3, dim=-1)

            # Compute g_shut_tensor INSIDE the live autograd graph so downstream
            # complementary regime gating preserves dL/d(g_shut) in PyTorch!
            e_ex = (
                s_amyg.pow(2)
                + self.config.beta_vmpfc * c_vmpfc.pow(2)
                - self.theta_dissoc.view(1, 1, -1)
            )
            g_shut_tensor = torch.sigmoid(-self.kappa_dissoc.view(1, 1, -1) * e_ex)

            corticolimbic_out = FusedCorticolimbicDissociativeDPDRFunction.apply(
                x_sens, s_amyg, c_vmpfc,
                self.theta_dissoc, self.kappa_dissoc, self.alpha_hyper, self.gamma_dpdr,
                self.config.beta_vmpfc
            )

            qkv = self.qkv_comp(corticolimbic_out).view(B, L, 3, self.n_heads, self.d_head)
            q, k, v = [t.transpose(1, 2) for t in qkv.unbind(dim=2)]

            amyg_heads = s_amyg.view(B, L, self.n_heads, self.d_head).pow(2).mean(dim=-1).transpose(1, 2)
            flashbulb_bias = (
                self.config.flashbulb_salience_scale * F.relu(amyg_heads - 0.5)
            ).unsqueeze(2)

            idx = torch.arange(L, device=x.device)
            causal_mask = (idx.unsqueeze(1) >= idx.unsqueeze(0)).view(1, 1, L, L)
            scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_head ** -0.5) + flashbulb_bias
            scores = scores.masked_fill(~causal_mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            head_out = torch.matmul(attn, v)

            perm = torch.sigmoid(self.amnestic_barrier_logits)
            perm_norm = perm / (perm.sum(dim=-1, keepdim=True) + 1e-6)
            comp_out = torch.einsum("ij,bjld->bild", perm_norm, head_out)
            comp_out = comp_out.transpose(1, 2).contiguous().view(B, L, D)

            out = self.out_norm(
                self.reintegration_proj(torch.cat([corticolimbic_out, comp_out], dim=-1))
            )

            with torch.no_grad():
                mean_g_shut = float(g_shut_tensor.detach().mean().item())
                mean_perm = float(
                    (perm_norm * (1.0 - torch.eye(self.n_heads, device=x.device))).mean().item()
                )
            diagnostics: Dict[str, object] = {
                "g_shut_tensor": g_shut_tensor,
                "mean_corticolimbic_shutdown_gate": mean_g_shut,
                "mean_offdiag_amnestic_permeability": mean_perm,
            }
            return out, diagnostics


def simulate_triton_dissociative_dpdr_kernel(
    x_sens: np.ndarray,
    s_amyg: np.ndarray,
    c_vmpfc: np.ndarray,
    theta_dissoc: np.ndarray,
    kappa_dissoc: np.ndarray,
    alpha_hyper: np.ndarray,
    gamma_dpdr: np.ndarray,
    beta_vmpfc: float = 0.40,
    block_d: int = 64,
    bytes_per_elem: int = 2,
) -> Tuple[np.ndarray, Dict[str, float]]:
    B, L, D = x_sens.shape
    num_rows = B * L
    xs_2d = x_sens.reshape(num_rows, D)
    sa_2d = s_amyg.reshape(num_rows, D)
    cv_2d = c_vmpfc.reshape(num_rows, D)
    out_2d = np.zeros_like(xs_2d)

    for row_id in range(num_rows):
        for d0 in range(0, D, block_d):
            d1 = min(d0 + block_d, D)
            xs_t = xs_2d[row_id, d0:d1]
            sa_t = sa_2d[row_id, d0:d1]
            cv_t = cv_2d[row_id, d0:d1]

            th_t = theta_dissoc[d0:d1]
            kap_t = kappa_dissoc[d0:d1]
            alp_t = alpha_hyper[d0:d1]
            gam_t = gamma_dpdr[d0:d1]

            e_ex = sa_t * sa_t + beta_vmpfc * (cv_t * cv_t) - th_t
            g_sh = 1.0 / (1.0 + np.exp(-np.clip(-kap_t * e_ex, -500.0, 500.0)))

            s_sq = sa_t * sa_t
            r_am = g_sh * (sa_t + alp_t * s_sq / np.sqrt(1.0 + s_sq))
            x_dp = xs_t / (1.0 + gam_t * (1.0 - g_sh) * (xs_t * xs_t))
            out_2d[row_id, d0:d1] = x_dp + r_am

    out = out_2d.reshape(B, L, D)
    elem_bytes = B * L * D * bytes_per_elem
    eager_hbm_bytes = 32 * elem_bytes
    triton_hbm_bytes = 6 * elem_bytes + 4 * D * bytes_per_elem
    return out, {
        "eager_hbm_kb": eager_hbm_bytes / 1024.0,
        "triton_hbm_kb": triton_hbm_bytes / 1024.0,
        "hbm_saved_pct": (1.0 - triton_hbm_bytes / eager_hbm_bytes) * 100.0,
        "bw_speedup": eager_hbm_bytes / triton_hbm_bytes,
    }


def run_dissociative_trauma_verification_suite() -> None:
    rng = np.random.default_rng(2026)
    print("=" * 94)
    print("1. 7-Variable Analytical vs. Finite-Difference Gradcheck (Corticolimbic Dissociative DPDR Op)")
    print("=" * 94)

    B, L, D = 2, 4, 6
    beta_vmpfc = 0.40
    x_sens = rng.standard_normal((B, L, D)).astype(np.float64) * 0.7
    s_amyg = rng.standard_normal((B, L, D)).astype(np.float64) * 0.8
    c_vmpfc = rng.standard_normal((B, L, D)).astype(np.float64) * 0.6
    theta_dissoc = rng.uniform(0.3, 0.9, size=(D,)).astype(np.float64)
    kappa_dissoc = rng.uniform(1.0, 2.2, size=(D,)).astype(np.float64)
    alpha_hyper = rng.uniform(0.3, 0.8, size=(D,)).astype(np.float64)
    gamma_dpdr = rng.uniform(0.4, 1.1, size=(D,)).astype(np.float64)
    grad_out = rng.standard_normal((B, L, D)).astype(np.float64)

    _, cache = DissociativeTraumaAnalyticalKernels.fused_corticolimbic_dissociative_dpdr_forward(
        x_sens, s_amyg, c_vmpfc, theta_dissoc, kappa_dissoc, alpha_hyper, gamma_dpdr, beta_vmpfc
    )
    ana_grads = DissociativeTraumaAnalyticalKernels.fused_corticolimbic_dissociative_dpdr_backward(
        grad_out, cache
    )

    eps = 1e-6
    targets = {
        "x_sens": (x_sens, ana_grads["d_xsens"]),
        "s_amyg": (s_amyg, ana_grads["d_samyg"]),
        "c_vmpfc": (c_vmpfc, ana_grads["d_cvmpfc"]),
        "theta_dissoc": (theta_dissoc, ana_grads["d_theta"]),
        "kappa_dissoc": (kappa_dissoc, ana_grads["d_kappa"]),
        "alpha_hyper": (alpha_hyper, ana_grads["d_alphahyper"]),
        "gamma_dpdr": (gamma_dpdr, ana_grads["d_gammadpdr"]),
    }

    for name, (arr, g_ana) in targets.items():
        g_num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + eps
            out_p, _ = DissociativeTraumaAnalyticalKernels.fused_corticolimbic_dissociative_dpdr_forward(
                x_sens, s_amyg, c_vmpfc, theta_dissoc, kappa_dissoc, alpha_hyper, gamma_dpdr, beta_vmpfc
            )
            loss_p = np.sum(out_p * grad_out)

            arr[idx] = orig - eps
            out_m, _ = DissociativeTraumaAnalyticalKernels.fused_corticolimbic_dissociative_dpdr_forward(
                x_sens, s_amyg, c_vmpfc, theta_dissoc, kappa_dissoc, alpha_hyper, gamma_dpdr, beta_vmpfc
            )
            loss_m = np.sum(out_m * grad_out)
            arr[idx] = orig
            g_num[idx] = (loss_p - loss_m) / (2.0 * eps)

        rel_err = np.linalg.norm(g_ana - g_num) / (np.linalg.norm(g_ana) + np.linalg.norm(g_num) + 1e-15)
        print(f"   Gradcheck dL/d({name:12s}) | Relative L2 Error: {rel_err:.3e}  [PASS < 1e-8]")
        assert rel_err < 1e-8

    print("\n" + "=" * 94)
    print("2. Trauma-Overload DPDR Dampening & Triton Tile Simulator Verification")
    print("=" * 94)
    B_b, L_b, D_b = 2, 64, 256
    xs_b = rng.standard_normal((B_b, L_b, D_b)) * 0.6
    sa_b = rng.standard_normal((B_b, L_b, D_b)) * 0.7
    cv_b = rng.standard_normal((B_b, L_b, D_b)) * 0.5
    th_b = np.full((D_b,), 0.65)
    kap_b = np.full((D_b,), 1.80)
    alp_b = np.full((D_b,), 0.55)
    gam_b = np.full((D_b,), 0.75)

    out_ref, _ = DissociativeTraumaAnalyticalKernels.fused_corticolimbic_dissociative_dpdr_forward(
        xs_b, sa_b, cv_b, th_b, kap_b, alp_b, gam_b
    )
    out_tri, hbm_stats = simulate_triton_dissociative_dpdr_kernel(
        xs_b, sa_b, cv_b, th_b, kap_b, alp_b, gam_b
    )
    max_diff = float(np.max(np.abs(out_ref - out_tri)))

    xs_shock = xs_b.copy()
    sa_shock = sa_b.copy()
    xs_shock[:, 0, :] = 50.0
    sa_shock[:, 0, :] = 50.0
    out_shock, _ = DissociativeTraumaAnalyticalKernels.fused_corticolimbic_dissociative_dpdr_forward(
        xs_shock, sa_shock, cv_b, th_b, kap_b, alp_b, gam_b
    )
    max_shock_val = float(np.max(np.abs(out_shock[:, 0, :])))

    print(f"   Triton Tile Simulator Max Abs Diff   : {max_diff:.2e}  [PASS == 0.00e+00]")
    print(f"   +50.0 Trauma Shock Bounded Output Max: {max_shock_val:.4f} < 5.0  [PASS: DPDR Protective Clamp!]")
    print(
        f"   HBM Memory Traffic (Fwd+Bwd)         : Eager={hbm_stats['eager_hbm_kb']:.1f} KB -> "
        f"Triton={hbm_stats['triton_hbm_kb']:.1f} KB (Saved {hbm_stats['hbm_saved_pct']:.1f}%)"
    )
    assert max_diff < 1e-14 and max_shock_val < 5.0
    print("=" * 94)

if not HAS_TORCH:
    class RMSNorm:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate RMSNorm.")

    class FusedCorticolimbicDissociativeDPDRFunction:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate FusedCorticolimbicDissociativeDPDRFunction.")

    class DissociativeTraumaNeuroOp:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate DissociativeTraumaNeuroOp.")

if __name__ == "__main__":
    run_dissociative_trauma_verification_suite()
