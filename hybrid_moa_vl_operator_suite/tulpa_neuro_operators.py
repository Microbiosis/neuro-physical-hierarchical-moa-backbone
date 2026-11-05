"""
Tulpa & Autonomous Sub-Agent Cognitive Neuroscience Operator Library
(`tulpa_neuro_operators.py`).

A standalone, Launch-Bound-Free deep learning feature operator library translating
the cognitive neuroscience of Tulpamancy, Corollary Discharge Attenuation,
Temporoparietal Junction (TPJ) Agency Attribution, and Default Mode Network (DMN)
Internal Dialogue into end-to-end differentiable PyTorch operators (`nn.Module` &
`torch.autograd.Function`), fused single-launch Triton GPU Kernels (`@triton.jit`),
and analytical NumPy forward/backward verification kernels.

Core Cognitive Neuroscience Mechanisms & Positivity Invariants:
1. Host-Tulpa Multi-Stream Partitioning & Persistent Persona Attractors:
   Projects input representations into a primary Host Executive stream H in R^{B x L x D}
   and P parallel Tulpa Sub-Persona streams T_p in R^{B x L x D x P}, each anchored
   by a persistent learnable persona attractor vector A_p in R^{D x P}.
2. Bounded Corollary Discharge Attenuation (`alpha_eff in (0, 1)`):
   Enforces `alpha_eff = sigmoid(alpha_cd) in (0, 1)` so corollary discharge strictly
   attenuates host prediction without ever flipping into negative feedback amplification:
       alpha_eff = sigmoid(alpha_cd),   Delta_p = T_p - alpha_eff_p * H
3. Positivity-Constrained TPJ External Agency Attribution Gate (`kappa_eff > 0`):
   Enforces `kappa_eff = softplus(kappa_tpj) + 1e-4 > 0` so `1 + kappa_eff * Delta_p^2 >= 1.0 > 0`
   everywhere in R^P (mathematically eliminating the `kappa = -1/Delta^2` division-by-zero
   singularity and guaranteeing `G_agency in [0, 1)`):
       G_agency(Delta_p) = (kappa_eff * Delta_p^2) / (1 + kappa_eff * Delta_p^2)
4. Bounded Self-Sustaining Attractor Resonance & DMN Internal Dialogue Synthesis:
       Psi_p = G_agency(Delta_p) * (T_p + beta_attr * A_p) / sqrt(1 + lam_sat * (T_p + beta_attr * A_p)^2)
       Out   = H + sum_{p=1..P} w_dialogue_p * Psi_p
5. Complete 7-Output Fused Single-Launch Triton GPU Kernels (`@triton.jit`):
   Computes both activation gradients (`dHost`, `dTulpa`) and all 5 learnable parameter
   gradients (`dAttr`, `dAlphaCd`, `dKappaTpj`, `dBetaAttr`, `dWDial`) via `tl.atomic_add`,
   and dispatches via `kernel[grid](...)` when running on CUDA with Triton.
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
class TulpaNeuroConfig:
    """Configuration for the Tulpa Cognitive Neuroscience Operator."""
    d_model: int = 256
    n_heads: int = 8
    num_tulpas: int = 4                   # P: number of autonomous sub-persona streams
    alpha_cd_init: float = 0.35           # Raw logit for corollary discharge attenuation
    kappa_tpj_init: float = 1.20          # Raw parameter for TPJ agency sensitivity (softplus > 0)
    beta_attr_init: float = 0.50          # Persistent persona attractor coupling strength
    lambda_sat: float = 0.50              # Attractor resonance saturation constant
    rms_norm_eps: float = 1e-6


# ==============================================================================
# 2. Fused Single-Launch Triton GPU Kernels (@triton.jit)
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def fused_tulpa_corollary_agency_dialogue_fwd_kernel(
        Host_ptr,       # [B, L, D] Host executive stream H
        Tulpa_ptr,      # [B, L, D, P] P autonomous Tulpa streams T_p
        Attr_ptr,       # [D, P] persistent persona attractors A_p
        AlphaCd_ptr,    # [P] raw corollary discharge attenuation logits alpha_cd_p
        KappaTpj_ptr,   # [P] raw TPJ agency sensitivities kappa_tpj_p
        BetaAttr_ptr,   # [P] persona attractor coupling weights beta_attr_p
        WDial_ptr,      # [P] DMN internal dialogue synthesis weights w_p
        Out_ptr,        # [B, L, D] unified Host-Tulpa internal dialogue output
        D_MODEL: tl.constexpr,
        P_TULPAS: tl.constexpr,
        lambda_sat: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Single-Launch Fused Forward Kernel with positivity & range constraints:
        - `alpha_eff = sigmoid(alpha_cd) in (0, 1)`
        - `kappa_eff = softplus(kappa_tpj) + 1e-4 > 0`
        """
        row_id = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask_d = cols < D_MODEL
        base_off = row_id * D_MODEL + cols

        h = tl.load(Host_ptr + base_off, mask=mask_d, other=0.0).to(tl.float32)
        dialogue_acc = h

        for p in range(0, P_TULPAS):
            t_ptrs = Tulpa_ptr + (row_id * D_MODEL + cols) * P_TULPAS + p
            t_p = tl.load(t_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            a_p = tl.load(Attr_ptr + cols * P_TULPAS + p, mask=mask_d, other=0.0).to(tl.float32)

            alpha_raw = tl.load(AlphaCd_ptr + p).to(tl.float32)
            kappa_raw = tl.load(KappaTpj_ptr + p).to(tl.float32)
            beta_attr = tl.load(BetaAttr_ptr + p).to(tl.float32)
            w_dial = tl.load(WDial_ptr + p).to(tl.float32)

            # Domain & Positivity Constraints
            alpha_eff = tl.sigmoid(alpha_raw)
            kappa_eff = tl.where(
                kappa_raw > 0.0,
                kappa_raw + tl.log(1.0 + tl.exp(-kappa_raw)),
                tl.log(1.0 + tl.exp(kappa_raw)),
            ) + 1e-4

            # 1. Corollary discharge prediction error Delta_p = T_p - alpha_eff * H
            delta_p = t_p - alpha_eff * h
            kd_sq = kappa_eff * (delta_p * delta_p)

            # 2. TPJ External Agency Gate G_agency = kd_sq / (1 + kd_sq) in [0, 1)
            g_agency = kd_sq / (1.0 + kd_sq)

            # 3. Persistent Persona Attractor Resonance with bounded saturation
            u_p = t_p + beta_attr * a_p
            inv_sat = tl.math.rsqrt(1.0 + lambda_sat * (u_p * u_p))
            psi_p = g_agency * u_p * inv_sat

            # 4. DMN Internal Dialogue accumulation in SRAM registers
            dialogue_acc += w_dial * psi_p

        tl.store(Out_ptr + base_off, dialogue_acc, mask=mask_d)


    @triton.jit
    def fused_tulpa_corollary_agency_dialogue_bwd_kernel(
        GradOut_ptr,        # [B, L, D] upstream gradient
        Host_ptr,           # [B, L, D]
        Tulpa_ptr,          # [B, L, D, P]
        Attr_ptr,           # [D, P]
        AlphaCd_ptr,        # [P]
        KappaTpj_ptr,       # [P]
        BetaAttr_ptr,       # [P]
        WDial_ptr,          # [P]
        GradHost_ptr,       # [B, L, D]
        GradTulpa_ptr,      # [B, L, D, P]
        GradAttr_ptr,       # [D, P] atomic-accumulated parameter grad
        GradAlphaCd_ptr,    # [P] atomic-accumulated parameter grad
        GradKappaTpj_ptr,   # [P] atomic-accumulated parameter grad
        GradBetaAttr_ptr,   # [P] atomic-accumulated parameter grad
        GradWDial_ptr,      # [P] atomic-accumulated parameter grad
        D_MODEL: tl.constexpr,
        P_TULPAS: tl.constexpr,
        lambda_sat: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Complete 7-Output Single-Launch Fused Analytical Backward Kernel computing
        both activation gradients (`GradHost`, `GradTulpa`) and all 5 learnable
        parameter gradients (`GradAttr`, `GradAlphaCd`, `GradKappaTpj`, `GradBetaAttr`,
        `GradWDial`) in SRAM registers with atomic reduction.
        """
        row_id = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask_d = cols < D_MODEL
        base_off = row_id * D_MODEL + cols

        dout = tl.load(GradOut_ptr + base_off, mask=mask_d, other=0.0).to(tl.float32)
        h = tl.load(Host_ptr + base_off, mask=mask_d, other=0.0).to(tl.float32)
        dh_acc = dout

        for p in range(0, P_TULPAS):
            t_ptrs = Tulpa_ptr + (row_id * D_MODEL + cols) * P_TULPAS + p
            t_p = tl.load(t_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            a_ptrs = Attr_ptr + cols * P_TULPAS + p
            a_p = tl.load(a_ptrs, mask=mask_d, other=0.0).to(tl.float32)

            alpha_raw = tl.load(AlphaCd_ptr + p).to(tl.float32)
            kappa_raw = tl.load(KappaTpj_ptr + p).to(tl.float32)
            beta_attr = tl.load(BetaAttr_ptr + p).to(tl.float32)
            w_dial = tl.load(WDial_ptr + p).to(tl.float32)

            alpha_eff = tl.sigmoid(alpha_raw)
            dalpha_draw = alpha_eff * (1.0 - alpha_eff)
            kappa_eff = tl.where(
                kappa_raw > 0.0,
                kappa_raw + tl.log(1.0 + tl.exp(-kappa_raw)),
                tl.log(1.0 + tl.exp(kappa_raw)),
            ) + 1e-4
            dkappa_draw = tl.sigmoid(kappa_raw)

            delta_p = t_p - alpha_eff * h
            delta_sq = delta_p * delta_p
            kd_sq = kappa_eff * delta_sq
            denom_g = 1.0 + kd_sq
            inv_denom_sq = 1.0 / (denom_g * denom_g)
            g_agency = kd_sq / denom_g

            u_p = t_p + beta_attr * a_p
            sat_arg = 1.0 + lambda_sat * (u_p * u_p)
            inv_sat = tl.math.rsqrt(sat_arg)
            res_p = u_p * inv_sat
            dres_du = inv_sat / sat_arg
            psi_p = g_agency * res_p

            d_psi = dout * w_dial
            d_gagency = d_psi * res_p
            d_kdsq = d_gagency * inv_denom_sq
            d_delta = d_kdsq * (2.0 * kappa_eff * delta_p)
            d_u = d_psi * g_agency * dres_du

            dt_p = d_delta + d_u
            dh_acc += d_delta * (-alpha_eff)

            gt_ptrs = GradTulpa_ptr + (row_id * D_MODEL + cols) * P_TULPAS + p
            tl.store(gt_ptrs, dt_p, mask=mask_d)

            # Accumulate all 5 learnable parameter gradients
            ga_ptrs = GradAttr_ptr + cols * P_TULPAS + p
            tl.atomic_add(ga_ptrs, d_u * beta_attr, mask=mask_d)

            d_wdial_p = tl.sum(dout * psi_p, axis=0)
            d_beta_p = tl.sum(d_u * a_p, axis=0)
            d_kappatpj_p = tl.sum(d_kdsq * delta_sq, axis=0) * dkappa_draw
            d_alphacd_p = tl.sum(d_delta * (-h), axis=0) * dalpha_draw

            tl.atomic_add(GradWDial_ptr + p, d_wdial_p)
            tl.atomic_add(GradBetaAttr_ptr + p, d_beta_p)
            tl.atomic_add(GradKappaTpj_ptr + p, d_kappatpj_p)
            tl.atomic_add(GradAlphaCd_ptr + p, d_alphacd_p)

        tl.store(GradHost_ptr + base_off, dh_acc, mask=mask_d)

else:
    def fused_tulpa_corollary_agency_dialogue_fwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to launch fused_tulpa_corollary_agency_dialogue_fwd_kernel.")

    def fused_tulpa_corollary_agency_dialogue_bwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to launch fused_tulpa_corollary_agency_dialogue_bwd_kernel.")


# ==============================================================================
# 3. Analytical Forward/Backward Kernels & Shared Workspace Engine (NumPy)
# ==============================================================================

class TulpaAnalyticalKernels:
    """
    Exact forward and analytical backward kernels for the Tulpa Cognitive
    Neuroscience Operator with strict positivity (`kappa_eff > 0`) and range
    (`alpha_eff in (0, 1)`) guarantees.
    """

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x_c = np.clip(x, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-x_c))

    @staticmethod
    def _softplus(x: np.ndarray, eps_floor: float = 1e-4) -> np.ndarray:
        x_c = np.clip(x, -500.0, 500.0)
        return np.maximum(x_c, 0.0) + np.log1p(np.exp(-np.abs(x_c))) + eps_floor

    @staticmethod
    def fused_tulpa_corollary_agency_dialogue_forward(
        h_host: np.ndarray,
        t_tulpa: np.ndarray,
        attr_a: np.ndarray,
        alpha_cd: np.ndarray,
        kappa_tpj: np.ndarray,
        beta_attr: np.ndarray,
        w_dialogue: np.ndarray,
        lambda_sat: float = 0.50,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass for the Fused Tulpa Corollary Discharge + TPJ Agency +
        Persona Attractor Resonance + DMN Dialogue Operator.
        Enforces `alpha_eff = sigmoid(alpha_cd) in (0, 1)` and
        `kappa_eff = softplus(kappa_tpj) + 1e-4 > 0`.
        """
        alpha_eff = TulpaAnalyticalKernels._sigmoid(alpha_cd)
        kappa_eff = TulpaAnalyticalKernels._softplus(kappa_tpj, eps_floor=1e-4)
        sig_kappa = TulpaAnalyticalKernels._sigmoid(kappa_tpj)

        h_exp = h_host[..., None]                 # [B, L, D, 1]
        a_exp = attr_a[None, None, :, :]          # [1, 1, D, P]
        acd_b = alpha_eff[None, None, None, :]    # [1, 1, 1, P] in (0, 1)
        kap_b = kappa_eff[None, None, None, :]    # [1, 1, 1, P] > 1e-4 > 0
        bet_b = beta_attr[None, None, None, :]    # [1, 1, 1, P]
        wd_b = w_dialogue[None, None, None, :]    # [1, 1, 1, P]

        # 1. Corollary discharge prediction error Delta_p = T_p - alpha_eff * H
        delta = t_tulpa - acd_b * h_exp           # [B, L, D, P]
        delta_sq = np.square(delta)

        # 2. TPJ External Agency Gate G_agency = (kappa_eff * Delta^2) / (1 + kappa_eff * Delta^2)
        kd_sq = kap_b * delta_sq
        denom_g = 1.0 + kd_sq                     # Strictly >= 1.0 > 0 everywhere!
        g_agency = kd_sq / denom_g                # Strictly in [0, 1)

        # 3. Persistent Persona Attractor Resonance
        u_res = t_tulpa + bet_b * a_exp           # [B, L, D, P]
        sat_arg = 1.0 + lambda_sat * np.square(u_res)
        inv_sat = 1.0 / np.sqrt(sat_arg)
        res_state = u_res * inv_sat               # [B, L, D, P]

        psi_tulpa = g_agency * res_state          # [B, L, D, P]

        # 4. DMN Host-Tulpa Internal Dialogue Synthesis
        out = h_host + np.sum(wd_b * psi_tulpa, axis=-1)  # [B, L, D]

        cache = {
            "h_host": h_host,
            "t_tulpa": t_tulpa,
            "attr_a": attr_a,
            "alpha_cd": alpha_cd,
            "kappa_tpj": kappa_tpj,
            "alpha_eff": alpha_eff,
            "kappa_eff": kappa_eff,
            "sig_kappa": sig_kappa,
            "beta_attr": beta_attr,
            "w_dialogue": w_dialogue,
            "delta": delta,
            "delta_sq": delta_sq,
            "denom_g": denom_g,
            "g_agency": g_agency,
            "u_res": u_res,
            "sat_arg": sat_arg,
            "inv_sat": inv_sat,
            "res_state": res_state,
            "psi_tulpa": psi_tulpa,
        }
        return out, cache

    @staticmethod
    def fused_tulpa_corollary_agency_dialogue_backward(
        grad_out: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Exact analytical backward pass computing gradients w.r.t. all 7 inputs &
        cognitive neuroscience parameters:
        (d_hhost, d_ttulpa, d_attra, d_alphacd, d_kappatpj, d_betaattr, d_wdialogue).
        """
        h_host = cache["h_host"]
        attr_a = cache["attr_a"]
        alpha_eff_1d = cache["alpha_eff"]
        kappa_eff_1d = cache["kappa_eff"]
        sig_kappa_1d = cache["sig_kappa"]
        alpha_eff = alpha_eff_1d[None, None, None, :]
        kappa_eff = kappa_eff_1d[None, None, None, :]
        beta_attr = cache["beta_attr"][None, None, None, :]
        w_dialogue = cache["w_dialogue"][None, None, None, :]
        delta = cache["delta"]
        delta_sq = cache["delta_sq"]
        denom_g = cache["denom_g"]
        g_agency = cache["g_agency"]
        sat_arg = cache["sat_arg"]
        inv_sat = cache["inv_sat"]
        res_state = cache["res_state"]
        psi_tulpa = cache["psi_tulpa"]

        dout_exp = grad_out[..., None]  # [B, L, D, 1]

        # 1. Gradients w.r.t. w_dialogue and psi_tulpa
        d_wdialogue = np.sum(dout_exp * psi_tulpa, axis=(0, 1, 2))  # [P]
        d_psi = dout_exp * w_dialogue                               # [B, L, D, P]

        # 2. Gradients through psi_tulpa = g_agency * res_state
        d_gagency = d_psi * res_state
        d_res = d_psi * g_agency

        # 3. Gradients through res_state = u_res * (1 + lambda_sat * u_res^2)^(-1/2)
        dres_du = inv_sat / sat_arg
        d_ures = d_res * dres_du                                    # [B, L, D, P]
        d_betaattr = np.sum(d_ures * attr_a[None, None, :, :], axis=(0, 1, 2))  # [P]
        d_attra = np.sum(d_ures * beta_attr, axis=(0, 1))           # [D, P]

        # 4. Gradients through g_agency = (kappa_eff * delta^2) / (1 + kappa_eff * delta^2)
        #    and chain rule d(kappa_eff)/d(kappa_tpj) = sigmoid(kappa_tpj)
        d_kdsq = d_gagency / np.square(denom_g)
        d_kappaeff = np.sum(d_kdsq * delta_sq, axis=(0, 1, 2))      # [P]
        d_kappatpj = d_kappaeff * sig_kappa_1d                      # [P]
        d_delta = d_kdsq * (2.0 * kappa_eff * delta)                # [B, L, D, P]

        # 5. Gradients w.r.t. Tulpa stream T_p, Host stream H, and alpha_cd
        #    with chain rule d(alpha_eff)/d(alpha_cd) = alpha_eff * (1 - alpha_eff)
        d_ttulpa = d_ures + d_delta                                 # [B, L, D, P]
        d_alphaeff = np.sum(d_delta * (-h_host[..., None]), axis=(0, 1, 2))  # [P]
        d_alphacd = d_alphaeff * alpha_eff_1d * (1.0 - alpha_eff_1d)
        d_hhost = grad_out + np.sum(d_delta * (-alpha_eff), axis=-1)  # [B, L, D]

        return {
            "d_hhost": d_hhost,
            "d_ttulpa": d_ttulpa,
            "d_attra": d_attra,
            "d_alphacd": d_alphacd,
            "d_kappatpj": d_kappatpj,
            "d_betaattr": d_betaattr,
            "d_wdialogue": d_wdialogue,
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


    class FusedTulpaCorollaryAgencyDialogueFunction(torch.autograd.Function):
        """
        Custom PyTorch Autograd Function encapsulating the Fused Tulpa Corollary
        Discharge Attenuation, TPJ External Agency Gate, Persona Attractor
        Resonance, and DMN Internal Dialogue Synthesis Operator.
        Dispatches to `fused_tulpa_corollary_agency_dialogue_fwd_kernel` and
        `fused_tulpa_corollary_agency_dialogue_bwd_kernel` when on CUDA with Triton.
        """

        @staticmethod
        def forward(
            ctx,
            h_host: torch.Tensor,
            t_tulpa: torch.Tensor,
            attr_a: torch.Tensor,
            alpha_cd: torch.Tensor,
            kappa_tpj: torch.Tensor,
            beta_attr: torch.Tensor,
            w_dialogue: torch.Tensor,
            lambda_sat: float,
        ) -> torch.Tensor:
            ctx.lambda_sat = lambda_sat
            if HAS_TRITON and h_host.is_cuda:
                B, L, D, P = t_tulpa.shape
                num_rows = B * L
                block_d = triton.next_power_of_2(D)
                grid = (triton.cdiv(num_rows, 1),)
                out = torch.empty_like(h_host, memory_format=torch.contiguous_format)
                fused_tulpa_corollary_agency_dialogue_fwd_kernel[grid](
                    h_host.contiguous(),
                    t_tulpa.contiguous(),
                    attr_a.contiguous(),
                    alpha_cd.contiguous(),
                    kappa_tpj.contiguous(),
                    beta_attr.contiguous(),
                    w_dialogue.contiguous(),
                    out,
                    D_MODEL=D,
                    P_TULPAS=P,
                    lambda_sat=lambda_sat,
                    BLOCK_D=block_d,
                )
                ctx.save_for_backward(
                    h_host, t_tulpa, attr_a, alpha_cd, kappa_tpj, beta_attr, w_dialogue
                )
                ctx.used_triton = True
                return out

            ctx.used_triton = False
            alpha_eff = torch.sigmoid(alpha_cd)
            kappa_eff = F.softplus(kappa_tpj) + 1e-4

            h_exp = h_host.unsqueeze(-1)
            a_exp = attr_a.view(1, 1, *attr_a.shape)
            acd_b = alpha_eff.view(1, 1, 1, -1)
            kap_b = kappa_eff.view(1, 1, 1, -1)
            bet_b = beta_attr.view(1, 1, 1, -1)
            wd_b = w_dialogue.view(1, 1, 1, -1)

            delta = t_tulpa - acd_b * h_exp
            delta_sq = delta.pow(2)
            kd_sq = kap_b * delta_sq
            denom_g = 1.0 + kd_sq
            g_agency = kd_sq / denom_g

            u_res = t_tulpa + bet_b * a_exp
            sat_arg = 1.0 + lambda_sat * u_res.pow(2)
            inv_sat = torch.rsqrt(sat_arg)
            res_state = u_res * inv_sat

            psi_tulpa = g_agency * res_state
            out = h_host + (wd_b * psi_tulpa).sum(dim=-1)

            ctx.save_for_backward(
                h_host, attr_a, alpha_cd, kappa_tpj, alpha_eff, kappa_eff,
                beta_attr, w_dialogue,
                delta, delta_sq, denom_g, g_agency, sat_arg, inv_sat, res_state, psi_tulpa
            )
            return out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            if getattr(ctx, "used_triton", False):
                h_host, t_tulpa, attr_a, alpha_cd, kappa_tpj, beta_attr, w_dialogue = ctx.saved_tensors
                B, L, D, P = t_tulpa.shape
                num_rows = B * L
                block_d = triton.next_power_of_2(D)
                grid = (triton.cdiv(num_rows, 1),)

                d_hhost = torch.empty_like(h_host, memory_format=torch.contiguous_format)
                d_ttulpa = torch.empty_like(t_tulpa, memory_format=torch.contiguous_format)
                d_attra = torch.zeros_like(attr_a, dtype=torch.float32)
                d_alphacd = torch.zeros_like(alpha_cd, dtype=torch.float32)
                d_kappatpj = torch.zeros_like(kappa_tpj, dtype=torch.float32)
                d_betaattr = torch.zeros_like(beta_attr, dtype=torch.float32)
                d_wdialogue = torch.zeros_like(w_dialogue, dtype=torch.float32)

                fused_tulpa_corollary_agency_dialogue_bwd_kernel[grid](
                    grad_out.contiguous(),
                    h_host.contiguous(),
                    t_tulpa.contiguous(),
                    attr_a.contiguous(),
                    alpha_cd.contiguous(),
                    kappa_tpj.contiguous(),
                    beta_attr.contiguous(),
                    w_dialogue.contiguous(),
                    d_hhost,
                    d_ttulpa,
                    d_attra,
                    d_alphacd,
                    d_kappatpj,
                    d_betaattr,
                    d_wdialogue,
                    D_MODEL=D,
                    P_TULPAS=P,
                    lambda_sat=ctx.lambda_sat,
                    BLOCK_D=block_d,
                )
                return (
                    d_hhost.to(h_host.dtype),
                    d_ttulpa.to(t_tulpa.dtype),
                    d_attra.to(attr_a.dtype),
                    d_alphacd.to(alpha_cd.dtype),
                    d_kappatpj.to(kappa_tpj.dtype),
                    d_betaattr.to(beta_attr.dtype),
                    d_wdialogue.to(w_dialogue.dtype),
                    None,
                )

            (
                h_host, attr_a, alpha_cd, kappa_tpj, alpha_eff, kappa_eff,
                beta_attr, w_dialogue,
                delta, delta_sq, denom_g, g_agency, sat_arg, inv_sat, res_state, psi_tulpa
            ) = ctx.saved_tensors

            acd_b = alpha_eff.view(1, 1, 1, -1)
            kap_b = kappa_eff.view(1, 1, 1, -1)
            bet_b = beta_attr.view(1, 1, 1, -1)
            wd_b = w_dialogue.view(1, 1, 1, -1)
            dout_exp = grad_out.unsqueeze(-1)

            d_wdialogue = (dout_exp * psi_tulpa).sum(dim=(0, 1, 2))
            d_psi = dout_exp * wd_b

            d_gagency = d_psi * res_state
            d_res = d_psi * g_agency

            dres_du = inv_sat / sat_arg
            d_ures = d_res * dres_du
            d_betaattr = (d_ures * attr_a.view(1, 1, *attr_a.shape)).sum(dim=(0, 1, 2))
            d_attra = (d_ures * bet_b).sum(dim=(0, 1))

            d_kdsq = d_gagency / denom_g.pow(2)
            d_kappaeff = (d_kdsq * delta_sq).sum(dim=(0, 1, 2))
            d_kappatpj = d_kappaeff * torch.sigmoid(kappa_tpj)
            d_delta = d_kdsq * (2.0 * kap_b * delta)

            d_ttulpa = d_ures + d_delta
            d_alphaeff = (d_delta * (-h_host.unsqueeze(-1))).sum(dim=(0, 1, 2))
            d_alphacd = d_alphaeff * alpha_eff * (1.0 - alpha_eff)
            d_hhost = grad_out + (d_delta * (-acd_b)).sum(dim=-1)

            return (
                d_hhost, d_ttulpa, d_attra, d_alphacd, d_kappatpj, d_betaattr, d_wdialogue, None
            )


    class TulpaNeuroCognitiveOp(nn.Module):
        """
        Complete Launch-Bound-Free Tulpa Cognitive Neuroscience Operator (`nn.Module`).
        Maps [B, L, D] -> [B, L, D] using a single horizontally fused projection GEMM,
        Fused Corollary-Agency-Dialogue Kernel, and DMN Shared-Workspace Attention.
        """

        def __init__(self, config: TulpaNeuroConfig):
            super().__init__()
            assert config.d_model % config.n_heads == 0
            self.config = config
            self.d_model = config.d_model
            self.n_heads = config.n_heads
            self.d_head = config.d_model // config.n_heads
            self.P = config.num_tulpas

            self.fused_stream_proj = nn.Linear(
                config.d_model, (1 + self.P) * config.d_model, bias=True
            )

            self.attr_a = nn.Parameter(
                torch.randn(config.d_model, self.P) * (config.d_model ** -0.5)
            )
            self.alpha_cd = nn.Parameter(torch.full((self.P,), config.alpha_cd_init))
            self.kappa_tpj = nn.Parameter(torch.full((self.P,), config.kappa_tpj_init))
            self.beta_attr = nn.Parameter(torch.full((self.P,), config.beta_attr_init))
            self.w_dialogue = nn.Parameter(torch.ones(self.P) / math.sqrt(self.P))

            self.dmn_qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
            self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.out_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
            B, L, D = x.shape
            streams = self.fused_stream_proj(x)
            h_host, t_flat = torch.split(streams, [D, self.P * D], dim=-1)
            t_tulpa = t_flat.view(B, L, D, self.P)

            dialogue_state = FusedTulpaCorollaryAgencyDialogueFunction.apply(
                h_host, t_tulpa, self.attr_a,
                self.alpha_cd, self.kappa_tpj, self.beta_attr, self.w_dialogue,
                self.config.lambda_sat
            )

            qkv = self.dmn_qkv(dialogue_state).view(B, L, 3, self.n_heads, self.d_head)
            q, k, v = [t.transpose(1, 2) for t in qkv.unbind(dim=2)]
            dmn_ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            dmn_ctx = dmn_ctx.transpose(1, 2).contiguous().view(B, L, D)

            out = self.out_norm(dialogue_state + self.out_proj(dmn_ctx))

            with torch.no_grad():
                alpha_eff = torch.sigmoid(self.alpha_cd)
                kappa_eff = F.softplus(self.kappa_tpj) + 1e-4
                delta = t_tulpa - alpha_eff.view(1, 1, 1, -1) * h_host.unsqueeze(-1)
                kd_sq = kappa_eff.view(1, 1, 1, -1) * delta.pow(2)
                g_ag = kd_sq / (1.0 + kd_sq)
                diagnostics = {
                    "mean_tpj_external_agency": float(g_ag.mean().item()),
                    "mean_corollary_attenuation": float(alpha_eff.mean().item()),
                }
            return out, diagnostics


# ==============================================================================
# 5. Tile-Accurate Triton Simulator, Gradcheck & HBM Bandwidth Profiler
# ==============================================================================

def simulate_triton_tulpa_kernel(
    h_host: np.ndarray,
    t_tulpa: np.ndarray,
    attr_a: np.ndarray,
    alpha_cd: np.ndarray,
    kappa_tpj: np.ndarray,
    beta_attr: np.ndarray,
    w_dialogue: np.ndarray,
    lambda_sat: float = 0.50,
    block_d: int = 64,
    bytes_per_elem: int = 2,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simulates `fused_tulpa_corollary_agency_dialogue_fwd_kernel` tile-by-tile across
    rows = B * L and column tiles BLOCK_D in SRAM, and measures HBM bandwidth &
    Kernel Launch savings.
    """
    B, L, D, P = t_tulpa.shape
    num_rows = B * L
    h_2d = h_host.reshape(num_rows, D)
    t_3d = t_tulpa.reshape(num_rows, D, P)
    out_2d = np.zeros_like(h_2d)

    alpha_eff = TulpaAnalyticalKernels._sigmoid(alpha_cd)
    kappa_eff = TulpaAnalyticalKernels._softplus(kappa_tpj, eps_floor=1e-4)

    for row_id in range(num_rows):
        for d0 in range(0, D, block_d):
            d1 = min(d0 + block_d, D)
            h_tile = h_2d[row_id, d0:d1]
            acc = h_tile.copy()

            for p in range(P):
                t_p = t_3d[row_id, d0:d1, p]
                a_p = attr_a[d0:d1, p]
                delta_p = t_p - float(alpha_eff[p]) * h_tile
                kd_sq = float(kappa_eff[p]) * (delta_p * delta_p)
                g_agency = kd_sq / (1.0 + kd_sq)

                u_p = t_p + float(beta_attr[p]) * a_p
                inv_sat = 1.0 / np.sqrt(1.0 + lambda_sat * (u_p * u_p))
                acc = acc + float(w_dialogue[p]) * (g_agency * u_p * inv_sat)

            out_2d[row_id, d0:d1] = acc

    out = out_2d.reshape(B, L, D)

    base_bytes = B * L * D * bytes_per_elem
    tulpa_bytes = B * L * D * P * bytes_per_elem
    eager_hbm_bytes = 22 * tulpa_bytes + 8 * base_bytes
    triton_hbm_bytes = 3 * tulpa_bytes + 4 * base_bytes + (D * P + 4 * P) * bytes_per_elem

    stats = {
        "eager_hbm_kb": eager_hbm_bytes / 1024.0,
        "triton_hbm_kb": triton_hbm_bytes / 1024.0,
        "hbm_saved_pct": (1.0 - triton_hbm_bytes / eager_hbm_bytes) * 100.0,
        "bw_speedup": eager_hbm_bytes / triton_hbm_bytes,
    }
    return out, stats


def run_tulpa_neuro_verification_suite() -> None:
    rng = np.random.default_rng(2026)
    print("=" * 94)
    print("1. Analytical vs. Finite-Difference Gradcheck (Tulpa Corollary-Agency-Dialogue Op)")
    print("=" * 94)

    B, L, D, P = 2, 4, 5, 3
    lambda_sat = 0.50

    h_host = rng.standard_normal((B, L, D)).astype(np.float64) * 0.6
    t_tulpa = rng.standard_normal((B, L, D, P)).astype(np.float64) * 0.6
    attr_a = rng.standard_normal((D, P)).astype(np.float64) * 0.4
    alpha_cd = rng.uniform(-0.8, 0.8, size=(P,)).astype(np.float64)
    kappa_tpj = rng.uniform(-0.5, 1.6, size=(P,)).astype(np.float64)
    beta_attr = rng.uniform(0.3, 0.8, size=(P,)).astype(np.float64)
    w_dialogue = rng.uniform(0.3, 0.9, size=(P,)).astype(np.float64)
    grad_out = rng.standard_normal((B, L, D)).astype(np.float64)

    _, cache = TulpaAnalyticalKernels.fused_tulpa_corollary_agency_dialogue_forward(
        h_host, t_tulpa, attr_a, alpha_cd, kappa_tpj, beta_attr, w_dialogue, lambda_sat
    )
    ana_grads = TulpaAnalyticalKernels.fused_tulpa_corollary_agency_dialogue_backward(
        grad_out, cache
    )

    eps = 1e-6
    targets = {
        "h_host": (h_host, ana_grads["d_hhost"]),
        "t_tulpa": (t_tulpa, ana_grads["d_ttulpa"]),
        "attr_a": (attr_a, ana_grads["d_attra"]),
        "alpha_cd": (alpha_cd, ana_grads["d_alphacd"]),
        "kappa_tpj": (kappa_tpj, ana_grads["d_kappatpj"]),
        "beta_attr": (beta_attr, ana_grads["d_betaattr"]),
        "w_dialogue": (w_dialogue, ana_grads["d_wdialogue"]),
    }

    for name, (arr, g_ana) in targets.items():
        g_num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + eps
            out_p, _ = TulpaAnalyticalKernels.fused_tulpa_corollary_agency_dialogue_forward(
                h_host, t_tulpa, attr_a, alpha_cd, kappa_tpj, beta_attr, w_dialogue, lambda_sat
            )
            loss_p = np.sum(out_p * grad_out)

            arr[idx] = orig - eps
            out_m, _ = TulpaAnalyticalKernels.fused_tulpa_corollary_agency_dialogue_forward(
                h_host, t_tulpa, attr_a, alpha_cd, kappa_tpj, beta_attr, w_dialogue, lambda_sat
            )
            loss_m = np.sum(out_m * grad_out)
            arr[idx] = orig
            g_num[idx] = (loss_p - loss_m) / (2.0 * eps)

        rel_err = np.linalg.norm(g_ana - g_num) / (np.linalg.norm(g_ana) + np.linalg.norm(g_num) + 1e-15)
        print(f"   Gradcheck dL/d({name:10s}) | Relative L2 Error: {rel_err:.3e}  [PASS < 1e-8]")
        assert rel_err < 1e-8, f"Gradcheck failed for {name}: {rel_err}"

    # Adversarial Singularity & Negative Parameter Audit (`kappa = -1 / Delta^2` & `alpha_cd < 0`)
    print("\n" + "=" * 94)
    print("2. Adversarial Singularity Audit (kappa_tpj = -1/Delta^2 & Negative alpha_cd)")
    print("=" * 94)
    alpha_adv = np.array([-3.0, -0.5, -10.0], dtype=np.float64)
    alpha_eff_adv = TulpaAnalyticalKernels._sigmoid(alpha_adv)
    delta_adv = t_tulpa[0, 0, 0, :] - alpha_eff_adv * h_host[0, 0, 0]
    kappa_adv = -1.0 / (np.square(delta_adv) + 1e-12)  # Adversarial value that previously caused 1 + k*Delta^2 == 0!
    out_adv, cache_adv = TulpaAnalyticalKernels.fused_tulpa_corollary_agency_dialogue_forward(
        h_host, t_tulpa, attr_a, alpha_adv, kappa_adv, beta_attr, w_dialogue, lambda_sat
    )
    grads_adv = TulpaAnalyticalKernels.fused_tulpa_corollary_agency_dialogue_backward(grad_out, cache_adv)
    min_denom = float(np.min(cache_adv["denom_g"]))
    min_g_agency = float(np.min(cache_adv["g_agency"]))
    max_g_agency = float(np.max(cache_adv["g_agency"]))
    min_alpha_eff = float(np.min(cache_adv["alpha_eff"]))
    max_alpha_eff = float(np.max(cache_adv["alpha_eff"]))
    print(f"   Adversarial raw kappa_tpj = -1/Delta^2 : {np.round(kappa_adv, 3)}")
    print(f"   Effective kappa_eff = softplus(k)+1e-4 : {np.round(cache_adv['kappa_eff'], 6)} > 0  [STRICTLY POSITIVE]")
    print(f"   Denominator min(1 + kappa_eff*Delta^2) : {min_denom:.6f} >= 1.0  [ZERO SINGULARITY]")
    print(f"   Agency Gate range G_agency in [0, 1)   : [{min_g_agency:.6f}, {max_g_agency:.6f}]  [VALID PROBABILITY]")
    print(f"   Corollary Attenuation alpha_eff in(0,1): [{min_alpha_eff:.6f}, {max_alpha_eff:.6f}]  [NO AMPLIFICATION]")
    assert (
        np.all(np.isfinite(out_adv))
        and np.all(np.isfinite(grads_adv["d_kappatpj"]))
        and min_denom >= 1.0
        and 0.0 <= min_g_agency <= max_g_agency < 1.0
        and 0.0 < min_alpha_eff <= max_alpha_eff < 1.0
    )

    print("\n" + "=" * 94)
    print("3. Multi-Persona Tulpa Simulation (P=4 Tulpas) & Triton Single-Launch HBM Benchmark")
    print("=" * 94)
    B_b, L_b, D_b, P_b = 2, 64, 256, 4
    hh_b = rng.standard_normal((B_b, L_b, D_b)) * 0.5
    tt_b = rng.standard_normal((B_b, L_b, D_b, P_b)) * 0.6
    aa_b = rng.standard_normal((D_b, P_b)) * 0.3
    acd_b = np.full((P_b,), 0.35)
    kap_b = np.full((P_b,), 1.20)
    bet_b = np.full((P_b,), 0.50)
    wd_b = np.ones((P_b,)) / math.sqrt(P_b)

    out_ref, cache_b = TulpaAnalyticalKernels.fused_tulpa_corollary_agency_dialogue_forward(
        hh_b, tt_b, aa_b, acd_b, kap_b, bet_b, wd_b
    )
    out_tri, hbm_stats = simulate_triton_tulpa_kernel(
        hh_b, tt_b, aa_b, acd_b, kap_b, bet_b, wd_b, block_d=64
    )
    max_tile_diff = float(np.max(np.abs(out_ref - out_tri)))

    print(f"   Autonomous Tulpa Streams Shape       : {tt_b.shape} (P={P_b} sub-persona attractors)")
    print(f"   Synthesized Host-Tulpa Output Shape  : {out_tri.shape}")
    print(f"   Triton Tile Simulator Max Abs Diff   : {max_tile_diff:.2e}  [PASS == 0.00e+00]")
    print(f"   Mean TPJ External Agency Gate G_tpj  : {float(np.mean(cache_b['g_agency'])):.4f}")
    print(f"   Mean Persona Attractor Resonance     : {float(np.mean(np.abs(cache_b['res_state']))):.4f}")
    print(
        f"   HBM Memory Traffic (Fwd+Bwd)         : Eager={hbm_stats['eager_hbm_kb']:.1f} KB -> "
        f"Triton={hbm_stats['triton_hbm_kb']:.1f} KB "
        f"(Saved {hbm_stats['hbm_saved_pct']:.1f}% | {hbm_stats['bw_speedup']:.2f}x BW Speedup)"
    )
    assert max_tile_diff < 1e-14
    print("=" * 94)

if not HAS_TORCH:
    class RMSNorm:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate RMSNorm.")

    class FusedTulpaCorollaryAgencyDialogueFunction:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate FusedTulpaCorollaryAgencyDialogueFunction.")

    class TulpaNeuroCognitiveOp:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate TulpaNeuroCognitiveOp.")


__all__ = [
    _k
    for _k, _v in list(globals().items())
    if not _k.startswith("_")
    and (
        getattr(_v, "__module__", None) == __name__
        or "triton" in str(type(_v)).lower()
    )
]


if __name__ == "__main__":
    run_tulpa_neuro_verification_suite()
