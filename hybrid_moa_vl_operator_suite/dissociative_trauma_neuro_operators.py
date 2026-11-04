"""
Dissociative Disorder Spectrum & Trauma-Related Psychiatric Neuroscience Operator Library
(`dissociative_trauma_neuro_operators.py`).

A standalone, Launch-Bound-Free deep learning feature operator library translating
clinical and computational neuroscience models of the Dissociative Disorder
Spectrum (DID, Depersonalization/Derealization DPDR) and Trauma-Related Disorders
(PTSD & Dissociative Subtype PTSD) into end-to-end differentiable PyTorch operators
(`nn.Module` & `torch.autograd.Function`), fused single-launch Triton GPU Kernels
(`@triton.jit`), and analytical NumPy forward/backward verification kernels.
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
        hyper_term = s_amyg + alpha_h * s_sq * inv_s

        x_sq = x_sens * x_sens
        dpdr_denom = 1.0 + gamma_d * one_m_g * x_sq

        c_dpdr = gamma_d * one_m_g
        dx_sens = dout * (1.0 - c_dpdr * x_sq) / (dpdr_denom * dpdr_denom)

        dg_from_amyg = dout * hyper_term
        dg_from_dpdr = dout * (x_sens * gamma_d * x_sq) / (dpdr_denom * dpdr_denom)
        d_gshut = dg_from_amyg + dg_from_dpdr

        d_eexcess = d_gshut * (-kappa * g_shut * one_m_g)
        dhyper_ds = 1.0 + alpha_h * s_amyg * (2.0 + s_sq) * (inv_s / (1.0 + s_sq))
        ds_amyg = dout * g_shut * dhyper_ds + d_eexcess * (2.0 * s_amyg)
        dc_vmpfc = d_eexcess * (2.0 * beta_vmpfc * c_vmpfc)

        tl.store(GradXSens_ptr + off, dx_sens, mask=mask)
        tl.store(GradSAmyg_ptr + off, ds_amyg, mask=mask)
        tl.store(GradCVmpfc_ptr + off, dc_vmpfc, mask=mask)


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
        trauma_salience: np.ndarray,
        amnestic_barrier_logits: np.ndarray,
        flashbulb_scale: float = 0.60,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        B, H, L, D_h = q.shape
        scale = 1.0 / math.sqrt(D_h)

        flashbulb_intrusion = flashbulb_scale * np.maximum(trauma_salience - 0.5, 0.0)
        intrusion_bias = flashbulb_intrusion[:, :, None, :]

        idx = np.arange(L)
        causal_mask = (idx[:, None] >= idx[None, :])[None, None, :, :]

        scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale + intrusion_bias
        scores = np.where(causal_mask, scores, -np.inf)

        s_max = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - s_max)
        attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        head_out = np.matmul(attn, v)

        perm_matrix = 1.0 / (1.0 + np.exp(-amnestic_barrier_logits))
        perm_norm = perm_matrix / (np.sum(perm_matrix, axis=-1, keepdims=True) + 1e-6)
        compartment_out = np.einsum("ij,bjld->bild", perm_norm, head_out)

        return compartment_out, {
            "amnestic_permeability": perm_norm,
            "mean_flashbulb_intrusion": flashbulb_intrusion,
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
            ctx.beta_vmpfc = beta_vmpfc
            return out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
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

            return (
                d_xsens, d_samyg, d_cvmpfc,
                d_theta, d_kappa, d_alphahyper, d_gammadpdr,
                None
            )


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

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
            B, L, D = x.shape
            x_sens, s_amyg, c_vmpfc = self.fused_circuit_proj(x).chunk(3, dim=-1)

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
                e_ex = s_amyg.pow(2) + self.config.beta_vmpfc * c_vmpfc.pow(2) - self.theta_dissoc.view(1, 1, -1)
                g_shut = torch.sigmoid(-self.kappa_dissoc.view(1, 1, -1) * e_ex)
                diagnostics = {
                    "mean_corticolimbic_shutdown_gate": float(g_shut.mean().item()),
                    "mean_offdiag_amnestic_permeability": float(
                        (perm_norm * (1.0 - torch.eye(self.n_heads, device=x.device))).mean().item()
                    ),
                }
            return out, diagnostics


# ==============================================================================
# 4. Standalone Verification Suite (7-Var Gradcheck & +50.0 Shock Safety)
# ==============================================================================

def run_dissociative_trauma_verification_suite() -> None:
    rng = np.random.default_rng(2026)
    print("=" * 92)
    print("DISSOCIATIVE TRAUMA & CORTICOLIMBIC DEFENSE OPERATOR — STANDALONE VERIFICATION SUITE")
    print("=" * 92)

    B, L, D = 2, 5, 6
    beta_vmpfc = 0.45
    x_sens = rng.standard_normal((B, L, D)).astype(np.float64) * 0.7
    s_amyg = rng.standard_normal((B, L, D)).astype(np.float64) * 0.7
    c_vmpfc = rng.standard_normal((B, L, D)).astype(np.float64) * 0.7
    theta_dissoc = rng.uniform(0.4, 0.9, size=(D,)).astype(np.float64)
    kappa_dissoc = rng.uniform(1.0, 2.2, size=(D,)).astype(np.float64)
    alpha_hyper = rng.uniform(0.3, 0.8, size=(D,)).astype(np.float64)
    gamma_dpdr = rng.uniform(0.4, 1.0, size=(D,)).astype(np.float64)
    grad_out = rng.standard_normal((B, L, D)).astype(np.float64)

    _, cache = DissociativeTraumaAnalyticalKernels.fused_corticolimbic_dissociative_dpdr_forward(
        x_sens, s_amyg, c_vmpfc, theta_dissoc, kappa_dissoc, alpha_hyper, gamma_dpdr, beta_vmpfc
    )
    ana = DissociativeTraumaAnalyticalKernels.fused_corticolimbic_dissociative_dpdr_backward(
        grad_out, cache
    )

    eps = 1e-6
    checks = [
        ("x_sens", x_sens, ana["d_xsens"]),
        ("s_amyg", s_amyg, ana["d_samyg"]),
        ("c_vmpfc", c_vmpfc, ana["d_cvmpfc"]),
        ("theta_dissoc", theta_dissoc, ana["d_theta"]),
        ("kappa_dissoc", kappa_dissoc, ana["d_kappa"]),
        ("alpha_hyper", alpha_hyper, ana["d_alphahyper"]),
        ("gamma_dpdr", gamma_dpdr, ana["d_gammadpdr"]),
    ]
    for name, arr, g_ana in checks:
        g_num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + eps
            yp, _ = DissociativeTraumaAnalyticalKernels.fused_corticolimbic_dissociative_dpdr_forward(
                x_sens, s_amyg, c_vmpfc, theta_dissoc, kappa_dissoc, alpha_hyper, gamma_dpdr, beta_vmpfc
            )
            lp = np.sum(yp * grad_out)
            arr[idx] = orig - eps
            ym, _ = DissociativeTraumaAnalyticalKernels.fused_corticolimbic_dissociative_dpdr_forward(
                x_sens, s_amyg, c_vmpfc, theta_dissoc, kappa_dissoc, alpha_hyper, gamma_dpdr, beta_vmpfc
            )
            lm = np.sum(ym * grad_out)
            arr[idx] = orig
            g_num[idx] = (lp - lm) / (2.0 * eps)
        rel_err = np.linalg.norm(g_ana - g_num) / (np.linalg.norm(g_ana) + np.linalg.norm(g_num) + 1e-15)
        print(f"   Gradcheck dL/d({name:13s}) | Relative L2 Error: {rel_err:.3e}  [PASS < 1e-8]")
        assert rel_err < 1e-8

    # Extreme +50.0 Trauma Shock Protection Verification
    x_shock = x_sens.copy()
    s_shock = s_amyg.copy()
    x_shock[:, 2, :] += 50.0
    s_shock[:, 2, :] += 50.0
    out_shock, shock_cache = DissociativeTraumaAnalyticalKernels.fused_corticolimbic_dissociative_dpdr_forward(
        x_shock, s_shock, c_vmpfc, theta_dissoc, kappa_dissoc, alpha_hyper, gamma_dpdr, beta_vmpfc
    )
    max_norm = float(np.max(np.abs(out_shock)))
    mean_shut_at_shock = float(np.mean(shock_cache["g_shut"][:, 2, :]))
    print(f"   Extreme +50.0 Shock Max Output Magnitude: {max_norm:.4f} (Bounded < 5.0) | PAG Shutdown Gate: {mean_shut_at_shock:.3e}")
    assert np.all(np.isfinite(out_shock)) and max_norm < 5.0
    print("=" * 92)

if not HAS_TORCH:
    class RMSNorm:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate RMSNorm.")

    class FusedCorticolimbicDissociativeDPDRFunction:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate FusedCorticolimbicDissociativeDPDRFunction.")

    class DissociativeTraumaNeuroOp:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate DissociativeTraumaNeuroOp.")

if __name__ == "__main__":
    run_dissociative_trauma_verification_suite()
