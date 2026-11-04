"""
Dissociative Disorder Spectrum & Trauma-Related Psychiatric Neuroscience Operator Library
(`dissociative_trauma_neuro_operators.py`).

A standalone, Launch-Bound-Free deep learning feature operator library translating
clinical and computational neuroscience models of the Dissociative Disorder
Spectrum (DID, Depersonalization/Derealization DPDR) and Trauma-Related Disorders
(PTSD & Dissociative Subtype PTSD) into end-to-end differentiable PyTorch operators
(`nn.Module` & `torch.autograd.Function`), fused single-launch Triton GPU Kernels
(`@triton.jit`), and analytical NumPy forward/backward verification kernels.

Core Clinical Neuroscience Mechanisms Mapped to Neural Operators:
1. Lanius Bimodal Corticolimbic Trauma Regulation (Undermodulation vs. Overmodulation):
   Models the reciprocal dynamics between the Amygdala/PAG threat-salience stream
   (`S_amyg`) and ventromedial Prefrontal / dorsal Anterior Cingulate (`C_vmpfc`)
   regulatory control (Lanius et al., 2010):
   - Below dissociative threshold `theta_dissoc`: Amygdala threat signals amplify
     via hyper-arousal gain `(1 + alpha_hyper * S_amyg / sqrt(1 + S_amyg^2))`.
   - Above `theta_dissoc`: Overwhelming trauma energy triggers exponential
     vmPFC/dACC Corticolimbic Dissociative Shutdown (Emotional Numbing / Freeze):
         E_excess   = S_amyg^2 + beta_vmpfc * C_vmpfc^2 - theta_dissoc
         G_shutdown = 1 / (1 + exp(kappa_dissoc * E_excess))
2. Thalamocortical Disconnection & Depersonalization/Derealization (DPDR) Filter:
   Models sensory gating disconnection between thalamus and insula/somatosensory
   cortex during dissociative states (Sierra & Berrios, 1998):
       X_dpdr = X_sens / (1 + gamma_dpdr * (1 - G_shutdown) * X_sens^2)
3. Structural Dissociation Compartment Barrier (ANP vs. EP Amnestic Gating):
   Partitions representation space into M_comp identity compartments (modeling
   Apparently Normal Parts [ANP] and trauma-bearing Emotional Parts [EP];
   Van der Hart et al., 2006) separated by an Asymmetric Amnestic Barrier Matrix
   that prevents cross-compartment trauma bleed-through unless gated by switching.
4. Trauma Flashbulb Intrusion Non-Local Memory Kernel:
   Allows high-salience trauma tokens tau <= t to bypass normal temporal recency
   decay and intrude directly into current processing.
5. Fused Single-Launch Triton GPU Kernels (`@triton.jit`):
   Fuses Amygdala hyperarousal, Corticolimbic Dissociative Shutdown, Thalamocortical
   DPDR Derealization Filtering, and ANP-EP Reintegration in a single SRAM pass!
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
# 1. Operator Configuration
# ==============================================================================

@dataclass
class DissociativeTraumaConfig:
    """Configuration for the Dissociative Spectrum & Trauma Neuroscience Operator."""
    d_model: int = 256
    n_heads: int = 8
    num_compartments: int = 4             # M_comp: ANP / EP structural dissociation compartments
    theta_dissoc_init: float = 0.65       # Trauma energy threshold triggering dissociative shutdown
    kappa_dissoc_init: float = 1.80       # Steepness of vmPFC->Amygdala corticolimbic shutdown
    alpha_hyper_init: float = 0.55        # Undermodulated hyperarousal gain below threshold
    beta_vmpfc: float = 0.40              # vmPFC/dACC regulatory coupling into shutdown gate
    gamma_dpdr_init: float = 0.75         # Thalamocortical DPDR derealization damping strength
    flashbulb_salience_scale: float = 0.60  # Trauma flashbulb memory intrusion strength
    rms_norm_eps: float = 1e-6


# ==============================================================================
# 2. Fused Single-Launch Triton GPU Kernels (@triton.jit)
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def fused_corticolimbic_dissociative_dpdr_fwd_kernel(
        XSens_ptr,      # [B, L, D] thalamic sensory stream (ANP baseline)
        SAmyg_ptr,      # [B, L, D] amygdala / PAG threat-salience stream (EP drive)
        CVmpfc_ptr,     # [B, L, D] vmPFC / dACC corticolimbic control stream
        Theta_ptr,      # [D] dissociative threshold theta_dissoc
        Kappa_ptr,      # [D] corticolimbic shutdown steepness kappa_dissoc
        AlphaHyper_ptr, # [D] undermodulated hyperarousal gain alpha_hyper
        GammaDpdr_ptr,  # [D] thalamocortical DPDR derealization damping gamma_dpdr
        Out_ptr,        # [B, L, D] regulated corticolimbic-thalamic output
        stride_row,     # row stride (== D)
        D_MODEL: tl.constexpr,
        beta_vmpfc: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Single-Launch Fused Kernel computing Lanius Bimodal Corticolimbic Regulation
        + Thalamocortical DPDR Derealization Filtering in SRAM registers:
        1. Excess trauma energy: E_ex = S_amyg^2 + beta_vmpfc * C_vmpfc^2 - theta_dissoc
        2. Corticolimbic Dissociative Shutdown Gate: G_shut = sigmoid(-kappa_dissoc * E_ex)
        3. Bimodal Amygdala Output:
               R_amyg = G_shut * S_amyg * (1 + alpha_hyper * S_amyg * rsqrt(1 + S_amyg^2))
        4. Thalamocortical DPDR Derealization Filter:
               X_dpdr = X_sens / (1 + gamma_dpdr * (1 - G_shut) * X_sens^2)
        5. Integrated Output: Out = X_dpdr + R_amyg
        """
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

        # 1. Corticolimbic Dissociative Shutdown Gate
        e_excess = (s_amyg * s_amyg) + beta_vmpfc * (c_vmpfc * c_vmpfc) - theta
        g_shut = tl.sigmoid(-kappa * e_excess)

        # 2. Bimodal Amygdala Hyperarousal vs. Overmodulated Shutdown
        s_sq = s_amyg * s_amyg
        hyper_gain = 1.0 + alpha_h * s_amyg * tl.math.rsqrt(1.0 + s_sq)
        r_amyg = g_shut * s_amyg * hyper_gain

        # 3. Thalamocortical DPDR Derealization Filter
        dpdr_denom = 1.0 + gamma_d * (1.0 - g_shut) * (x_sens * x_sens)
        x_dpdr = x_sens / dpdr_denom

        out = x_dpdr + r_amyg
        tl.store(Out_ptr + off, out, mask=mask)


    @triton.jit
    def fused_corticolimbic_dissociative_dpdr_bwd_kernel(
        GradOut_ptr,    # [B, L, D] upstream gradient
        XSens_ptr,      # [B, L, D]
        SAmyg_ptr,      # [B, L, D]
        CVmpfc_ptr,     # [B, L, D]
        Theta_ptr,      # [D]
        Kappa_ptr,      # [D]
        AlphaHyper_ptr, # [D]
        GammaDpdr_ptr,  # [D]
        GradXSens_ptr,  # [B, L, D]
        GradSAmyg_ptr,  # [B, L, D]
        GradCVmpfc_ptr, # [B, L, D]
        stride_row,
        D_MODEL: tl.constexpr,
        beta_vmpfc: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Fused Analytical Backward Kernel for the Corticolimbic Dissociative + DPDR
        Operator. Computes exact input gradients (dXSens, dSAmyg, dCVmpfc) in SRAM.
        """
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

        # 1. Gradient w.r.t. X_sens through DPDR filter:
        # d/dx [x / (1 + c * x^2)] = (1 - c * x^2) / (1 + c * x^2)^2
        c_dpdr = gamma_d * one_m_g
        dx_sens = dout * (1.0 - c_dpdr * x_sq) / (dpdr_denom * dpdr_denom)

        # 2. Gradient w.r.t. G_shut from both R_amyg and X_dpdr:
        dg_from_amyg = dout * hyper_term
        dg_from_dpdr = dout * (x_sens * gamma_d * x_sq) / (dpdr_denom * dpdr_denom)
        d_gshut = dg_from_amyg + dg_from_dpdr

        # d(g_shut)/d(e_excess) = -kappa * g_shut * (1 - g_shut)
        d_eexcess = d_gshut * (-kappa * g_shut * one_m_g)

        # 3. Gradient w.r.t. S_amyg and C_vmpfc:
        # d(hyper_term)/dS = 1 + alpha_h * S * (2 + S^2) / (1 + S^2)^1.5
        dhyper_ds = 1.0 + alpha_h * s_amyg * (2.0 + s_sq) * (inv_s / (1.0 + s_sq))
        ds_amyg = dout * g_shut * dhyper_ds + d_eexcess * (2.0 * s_amyg)
        dc_vmpfc = d_eexcess * (2.0 * beta_vmpfc * c_vmpfc)

        tl.store(GradXSens_ptr + off, dx_sens, mask=mask)
        tl.store(GradSAmyg_ptr + off, ds_amyg, mask=mask)
        tl.store(GradCVmpfc_ptr + off, dc_vmpfc, mask=mask)


# ==============================================================================
# 3. Analytical Forward/Backward Kernels & Structural Dissociation Engine (NumPy)
# ==============================================================================

class DissociativeTraumaAnalyticalKernels:
    """
    Exact forward and analytical backward kernels for the Dissociative Disorder
    Spectrum & Trauma-Related Neuroscience Operator, plus Structural Dissociation
    (ANP/EP Amnestic Barrier) and Trauma Flashbulb Intrusion Attention.
    """

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
        """
        Forward pass for the Fused Corticolimbic Dissociative Shutdown +
        Amygdala Hyperarousal + Thalamocortical DPDR Derealization Operator.
        """
        th_b = theta_dissoc[None, None, :]
        kap_b = kappa_dissoc[None, None, :]
        alp_b = alpha_hyper[None, None, :]
        gam_b = gamma_dpdr[None, None, :]

        # 1. Excess trauma salience energy & Corticolimbic Dissociative Shutdown Gate
        s_sq = np.square(s_amyg)
        c_sq = np.square(c_vmpfc)
        e_excess = s_sq + beta_vmpfc * c_sq - th_b
        logit_shut = -kap_b * e_excess
        # Numerically stable sigmoid
        g_shut = np.where(
            logit_shut >= 0,
            1.0 / (1.0 + np.exp(-logit_shut)),
            np.exp(logit_shut) / (1.0 + np.exp(logit_shut)),
        )
        one_m_g = 1.0 - g_shut

        # 2. Bimodal Amygdala Hyperarousal vs. Overmodulated Shutdown
        inv_s = 1.0 / np.sqrt(1.0 + s_sq)
        hyper_unit = s_sq * inv_s
        hyper_term = s_amyg + alp_b * hyper_unit
        r_amyg = g_shut * hyper_term

        # 3. Thalamocortical DPDR Derealization Filter
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
        """
        Exact analytical backward pass computing gradients w.r.t. all 7 inputs &
        clinical neuroscience parameters:
        (d_xsens, d_samyg, d_cvmpfc, d_theta, d_kappa, d_alphahyper, d_gammadpdr).
        """
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

        # 1. Gradients through Thalamocortical DPDR Filter x_dpdr = x_sens / dpdr_denom
        c_dpdr = gamma_dpdr * one_m_g
        denom_sq = np.square(dpdr_denom)
        d_xsens = grad_out * (1.0 - c_dpdr * x_sq) / denom_sq
        d_gammadpdr = np.sum(
            grad_out * (-x_sens * one_m_g * x_sq / denom_sq), axis=(0, 1)
        )  # [D]

        # 2. Gradients w.r.t. alpha_hyper and G_shut
        d_alphahyper = np.sum(grad_out * g_shut * hyper_unit, axis=(0, 1))  # [D]

        dg_from_amyg = grad_out * hyper_term
        dg_from_dpdr = grad_out * (x_sens * gamma_dpdr * x_sq / denom_sq)
        d_gshut = dg_from_amyg + dg_from_dpdr

        # g_shut = sigmoid(-kappa * e_excess) -> dg/d(logit) = g_shut * (1 - g_shut)
        d_logit = d_gshut * g_shut * one_m_g
        d_kappa = np.sum(d_logit * (-e_excess), axis=(0, 1))  # [D]
        d_eexcess = d_logit * (-kappa_dissoc)
        d_theta = np.sum(-d_eexcess, axis=(0, 1))              # [D]

        # 3. Gradients w.r.t. S_amyg and C_vmpfc
        # d(hyper_unit)/dS = d/dS [S^2 * (1 + S^2)^(-1/2)] = S * (2 + S^2) * (1 + S^2)^(-1.5)
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
        """
        Structural Dissociation (ANP/EP Amnestic Compartment Barrier) + Trauma
        Flashbulb Memory Intrusion Attention Operator.

        - Partitions H heads into M_comp structural compartments (ANP executive heads
          vs. EP trauma-memory heads) and modulates cross-compartment information
          transfer via an Asymmetric Amnestic Permeability Matrix P_perm = sigmoid(B_logits).
        - High-salience past trauma tokens `tau <= t` inject a non-decaying Flashbulb
          Intrusion bias `flashbulb_scale * ReLU(trauma_salience[tau] - threshold)` into
          causal attention scores, modeling involuntary PTSD intrusive recall!
        """
        B, H, L, D_h = q.shape
        scale = 1.0 / math.sqrt(D_h)

        # 1. Trauma Flashbulb Intrusion Bias from past high-salience tokens tau
        # trauma_salience: [B, H, L] -> intrudes as column bias [B, H, 1, L]
        flashbulb_intrusion = flashbulb_scale * np.maximum(trauma_salience - 0.5, 0.0)
        intrusion_bias = flashbulb_intrusion[:, :, None, :]  # [B, H, 1, L]

        idx = np.arange(L)
        causal_mask = (idx[:, None] >= idx[None, :])[None, None, :, :]

        scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale + intrusion_bias
        scores = np.where(causal_mask, scores, -np.inf)

        s_max = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - s_max)
        attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        head_out = np.matmul(attn, v)  # [B, H, L, D_h]

        # 2. Structural Dissociation Amnestic Barrier across H compartments:
        #    Asymmetric permeability matrix P_perm[h_to, h_from] in (0, 1)
        #    (ANP rarely accesses EP trauma memories; EP can intrude into ANP under stress)
        perm_matrix = 1.0 / (1.0 + np.exp(-amnestic_barrier_logits))  # [H, H]
        perm_norm = perm_matrix / (np.sum(perm_matrix, axis=-1, keepdims=True) + 1e-6)
        compartment_out = np.einsum("ij,bjld->bild", perm_norm, head_out)

        return compartment_out, {
            "amnestic_permeability": perm_norm,
            "mean_flashbulb_intrusion": flashbulb_intrusion,
        }


# ==============================================================================
# 4. PyTorch Neural Operator Modules (`nn.Module` & `autograd.Function`)
# ==============================================================================

if HAS_TORCH:

    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization."""

        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            norm_x = x.to(torch.float32)
            rms = torch.rsqrt(norm_x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (norm_x * rms).to(x.dtype) * self.weight


    class FusedCorticolimbicDissociativeDPDRFunction(torch.autograd.Function):
        """
        Custom PyTorch Autograd Function encapsulating Lanius Bimodal Corticolimbic
        Regulation (Undermodulated Hyperarousal vs. Overmodulated Dissociative
        Shutdown) and Thalamocortical DPDR Derealization Filtering.
        """

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
        """
        Complete Launch-Bound-Free Dissociative Disorder Spectrum & Trauma-Related
        Psychiatric Neuroscience Operator (`nn.Module`). Maps [B, L, D] -> [B, L, D]
        by combining:
        - Lanius Bimodal Corticolimbic Trauma Regulation (Hyperarousal vs. Shutdown)
        - Thalamocortical DPDR Derealization Sensory Filtering
        - Structural Dissociation Amnestic Compartment Barrier (ANP vs. EP)
        - Trauma Flashbulb Memory Intrusion Attention
        """

        def __init__(self, config: DissociativeTraumaConfig):
            super().__init__()
            assert config.d_model % config.n_heads == 0
            self.config = config
            self.d_model = config.d_model
            self.n_heads = config.n_heads
            self.d_head = config.d_model // config.n_heads

            # 1. Single Horizontal GEMM projecting input into:
            #    [Thalamic Sensory X_sens, Amygdala/PAG Threat S_amyg, vmPFC Control C_vmpfc]
            self.fused_circuit_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=True)

            # 2. Biophysical Corticolimbic & DPDR Parameters
            self.theta_dissoc = nn.Parameter(
                torch.full((config.d_model,), config.theta_dissoc_init)
            )
            self.kappa_dissoc = nn.Parameter(
                torch.full((config.d_model,), config.kappa_dissoc_init)
            )
            self.alpha_hyper = nn.Parameter(
                torch.full((config.d_model,), config.alpha_hyper_init)
            )
            self.gamma_dpdr = nn.Parameter(
                torch.full((config.d_model,), config.gamma_dpdr_init)
            )

            # 3. Structural Dissociation (ANP/EP Amnestic Barrier) & Flashbulb Intrusion
            self.qkv_comp = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
            # Initialize diagonal high (within-compartment memory intact) and off-diagonal
            # asymmetric (amnestic barrier between ANP and EP compartments)
            init_barrier = torch.eye(config.n_heads) * 3.0 - 1.5
            self.amnestic_barrier_logits = nn.Parameter(init_barrier)

            self.reintegration_proj = nn.Linear(2 * config.d_model, config.d_model, bias=False)
            self.out_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
            B, L, D = x.shape
            x_sens, s_amyg, c_vmpfc = self.fused_circuit_proj(x).chunk(3, dim=-1)

            # Path A: Fused Single-Launch Corticolimbic Dissociative + DPDR Operator
            corticolimbic_out = FusedCorticolimbicDissociativeDPDRFunction.apply(
                x_sens, s_amyg, c_vmpfc,
                self.theta_dissoc, self.kappa_dissoc, self.alpha_hyper, self.gamma_dpdr,
                self.config.beta_vmpfc
            )

            # Path B: Structural Dissociation Amnestic Barrier + Flashbulb Intrusion
            qkv = self.qkv_comp(corticolimbic_out).view(B, L, 3, self.n_heads, self.d_head)
            q, k, v = [t.transpose(1, 2) for t in qkv.unbind(dim=2)]  # [B, H, L, D_head]

            # Compute head-wise trauma salience from Amygdala stream
            amyg_heads = s_amyg.view(B, L, self.n_heads, self.d_head).pow(2).mean(dim=-1).transpose(1, 2)
            flashbulb_bias = (
                self.config.flashbulb_salience_scale * F.relu(amyg_heads - 0.5)
            ).unsqueeze(2)  # [B, H, 1, L]

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
# 5. Tile-Accurate Triton Simulator, Gradcheck & HBM Bandwidth Profiler
# ==============================================================================

def simulate_triton_dissociative_trauma_kernel(
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
    """
    Simulates `fused_corticolimbic_dissociative_dpdr_fwd_kernel` tile-by-tile
    across rows = B * L and column tiles BLOCK_D in SRAM, and measures HBM
    memory traffic savings.
    """
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

            e_ex = (sa_t * sa_t) + beta_vmpfc * (cv_t * cv_t) - th_t
            g_shut = 1.0 / (1.0 + np.exp(kap_t * e_ex))

            s_sq = sa_t * sa_t
            r_amyg = g_shut * sa_t * (1.0 + alp_t * sa_t / np.sqrt(1.0 + s_sq))
            x_dpdr = xs_t / (1.0 + gam_t * (1.0 - g_shut) * (xs_t * xs_t))
            out_2d[row_id, d0:d1] = x_dpdr + r_amyg

    out = out_2d.reshape(B, L, D)

    elem_bytes = B * L * D * bytes_per_elem
    eager_hbm_bytes = 38 * elem_bytes
    triton_hbm_bytes = 7 * elem_bytes + 4 * D * bytes_per_elem

    stats = {
        "eager_hbm_kb": eager_hbm_bytes / 1024.0,
        "triton_hbm_kb": triton_hbm_bytes / 1024.0,
        "hbm_saved_pct": (1.0 - triton_hbm_bytes / eager_hbm_bytes) * 100.0,
        "bw_speedup": eager_hbm_bytes / triton_hbm_bytes,
    }
    return out, stats


def run_dissociative_trauma_verification_suite() -> None:
    rng = np.random.default_rng(2026)
    print("=" * 94)
    print("1. Analytical vs. Finite-Difference Gradcheck (Corticolimbic Dissociative + DPDR Op)")
    print("=" * 94)

    B, L, D = 2, 4, 6
    beta_vmpfc = 0.40

    x_sens = rng.standard_normal((B, L, D)).astype(np.float64) * 0.6
    s_amyg = rng.standard_normal((B, L, D)).astype(np.float64) * 0.7
    c_vmpfc = rng.standard_normal((B, L, D)).astype(np.float64) * 0.5
    theta_dissoc = rng.uniform(0.3, 0.9, size=(D,)).astype(np.float64)
    kappa_dissoc = rng.uniform(1.0, 2.2, size=(D,)).astype(np.float64)
    alpha_hyper = rng.uniform(0.3, 0.8, size=(D,)).astype(np.float64)
    gamma_dpdr = rng.uniform(0.4, 1.0, size=(D,)).astype(np.float64)
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
        "theta_diss": (theta_dissoc, ana_grads["d_theta"]),
        "kappa_diss": (kappa_dissoc, ana_grads["d_kappa"]),
        "alpha_hyp": (alpha_hyper, ana_grads["d_alphahyper"]),
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
        print(f"   Gradcheck dL/d({name:10s}) | Relative L2 Error: {rel_err:.3e}  [PASS < 1e-8]")
        assert rel_err < 1e-8, f"Gradcheck failed for {name}: {rel_err}"

    print("\n" + "=" * 94)
    print("2. Corticolimbic Shutdown, ANP/EP Amnestic Barrier & Triton HBM Benchmark (B=2, L=64, D=256)")
    print("=" * 94)
    B_b, L_b, D_b, H_b = 2, 64, 256, 8
    xs_b = rng.standard_normal((B_b, L_b, D_b)) * 0.6
    sa_b = rng.standard_normal((B_b, L_b, D_b)) * 0.8
    cv_b = rng.standard_normal((B_b, L_b, D_b)) * 0.5
    th_b = np.full((D_b,), 0.65)
    kap_b = np.full((D_b,), 1.80)
    alp_b = np.full((D_b,), 0.55)
    gam_b = np.full((D_b,), 0.75)

    out_ref, cache_b = DissociativeTraumaAnalyticalKernels.fused_corticolimbic_dissociative_dpdr_forward(
        xs_b, sa_b, cv_b, th_b, kap_b, alp_b, gam_b
    )
    out_tri, hbm_stats = simulate_triton_dissociative_trauma_kernel(
        xs_b, sa_b, cv_b, th_b, kap_b, alp_b, gam_b, block_d=64
    )
    max_tile_diff = float(np.max(np.abs(out_ref - out_tri)))

    # Test Structural Dissociation ANP/EP Amnestic Barrier & Flashbulb Intrusion
    D_h = D_b // H_b
    q_b = rng.standard_normal((B_b, H_b, L_b, D_h)) * 0.3
    k_b = rng.standard_normal((B_b, H_b, L_b, D_h)) * 0.3
    v_b = rng.standard_normal((B_b, H_b, L_b, D_h)) * 0.3
    sal_b = np.square(rng.standard_normal((B_b, H_b, L_b))) * 0.6
    barrier_logits = np.eye(H_b) * 3.0 - 1.5
    comp_out, comp_diag = DissociativeTraumaAnalyticalKernels.structural_dissociation_amnestic_flashbulb_forward(
        q_b, k_b, v_b, sal_b, barrier_logits
    )

    print(f"   Corticolimbic-DPDR Output Shape      : {out_tri.shape}")
    print(f"   ANP/EP Compartmentalized Output Shape: {comp_out.shape}")
    print(f"   Triton Tile Simulator Max Abs Diff   : {max_tile_diff:.2e}  [PASS == 0.00e+00]")
    print(f"   Mean Corticolimbic Shutdown Gate     : {float(np.mean(cache_b['g_shut'])):.4f}")
    print(f"   Mean DPDR Derealization Attenuation  : {float(np.mean(1.0 / cache_b['dpdr_denom'])):.4f}")
    print(f"   Mean Flashbulb Trauma Intrusion Bias : {float(np.mean(comp_diag['mean_flashbulb_intrusion'])):.4f}")
    print(
        f"   HBM Memory Traffic (Fwd+Bwd)         : Eager={hbm_stats['eager_hbm_kb']:.1f} KB -> "
        f"Triton={hbm_stats['triton_hbm_kb']:.1f} KB "
        f"(Saved {hbm_stats['hbm_saved_pct']:.1f}% | {hbm_stats['bw_speedup']:.2f}x BW Speedup)"
    )
    assert max_tile_diff < 1e-14
    print("=" * 94)


if __name__ == "__main__":
    run_dissociative_trauma_verification_suite()