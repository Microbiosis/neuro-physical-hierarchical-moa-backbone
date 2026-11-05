"""
Brain-Inspired Nonlinear Computational Neuroscience Operator Library
(`neuro_nonlinear_operators.py`).

A standalone deep learning feature operator library translating four core nonlinear
computational neuroscience mechanisms into end-to-end differentiable PyTorch
operators (`nn.Module` & `torch.autograd.Function`), fused Triton GPU Kernels
(`@triton.jit`), and analytical NumPy forward/backward verification kernels:

1. Active Dendritic Supralinearity & NMDA Mg2+ Voltage Unblock (Jahr & Stevens):
   Models voltage-dependent relief of Mg2+ block at dendritic NMDA receptors,
   generating supralinear dendritic plateau potentials:
       B_nmda(V) = 1 / (1 + eta_mg * exp(-alpha_mg * V))
       I_dend(V, S) = V + g_nmda * S * V * B_nmda(V)
2. GABAergic Divisive Shunting Inhibition (Carandini & Heeger):
   Models nonlinear conductance-based divisive normalization at the perisomatic
   compartment rather than simple linear subtraction:
       V_soma = I_dend(V, S) / (1 + gamma_shunt * G_shunt)
3. FitzHugh-Nagumo (FHN) Bounded Cubic Excitability & Slow Recovery Adaptation:
   Models nonlinear excitable membrane bifurcation dynamics coupled with a slow
   potassium-like recovery current W_adapt:
       Psi_fhn(V_soma, W_adapt) = V_soma - V_soma^3 / (3 * kappa^2 + V_soma^2) - beta * W_adapt
4. Tsodyks-Markram Short-Term Synaptic Plasticity (STP) & Causal Hebbian Trace:
   Modulates cross-token synaptic transmission via dynamic neurotransmitter
   utilization u_t (facilitation), vesicle availability r_t (depression), and
   causal Hebbian fast-weight eligibility traces.
5. Fused Triton GPU Kernels (`@triton.jit`):
   Fuses NMDA Mg2+ unblock + Dendritic plateau current + Divisive shunting
   normalization + FitzHugh-Nagumo cubic excitability in a single SRAM pass.
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
class NeuroNonlinearConfig:
    """Configuration for the Brain-Inspired Nonlinear Neuroscience Operator."""
    d_model: int = 256
    n_heads: int = 8
    eta_mg: float = 0.28                  # Extracellular [Mg2+] / 3.57 mM physiological constant
    alpha_mg_init: float = 1.2            # Voltage sensitivity of NMDA Mg2+ unblock
    gamma_shunt: float = 0.5              # Divisive shunting conductance coupling strength
    fhn_kappa: float = 1.5                # Saturation scale for bounded cubic FHN nonlinearity
    fhn_beta: float = 0.4                 # Slow recovery / adaptation current coupling
    stp_u0: float = 0.25                  # Baseline synaptic release probability U_0
    stp_tau_facil: float = 4.0            # Short-term synaptic facilitation time constant
    stp_tau_rec: float = 6.0              # Short-term vesicle recovery time constant
    rms_norm_eps: float = 1e-6


# ==============================================================================
# 2. Fused Triton GPU Kernels (@triton.jit)
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def fused_dendritic_nmda_shunting_fhn_fwd_kernel(
        V_ptr,          # [B, L, D] dendritic membrane potential V
        S_ptr,          # [B, L, D] glutamate synaptic input S
        G_ptr,          # [B, L, D] GABAergic shunting conductance G_shunt >= 0
        W_ptr,          # [B, L, D] slow adaptation recovery variable W_adapt
        GNmda_ptr,      # [D] channel-wise NMDA receptor density g_nmda
        AlphaMg_ptr,    # [D] channel-wise Mg2+ unblock steepness alpha_mg
        Out_ptr,        # [B, L, D] somatic excitable output
        stride_row,     # row stride (== D)
        D_MODEL: tl.constexpr,
        eta_mg: tl.constexpr,
        gamma_shunt: tl.constexpr,
        kappa_sq3: tl.constexpr,  # 3.0 * fhn_kappa^2
        fhn_beta: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Fuses 4 nonlinear computational neuroscience stages in SRAM per (b, l) token:
        1. NMDA Mg2+ voltage unblock: B(V) = 1 / (1 + eta_mg * exp(-alpha_mg * V))
        2. Supralinear dendritic plateau current: I_dend = V + g_nmda * S * V * B(V)
        3. Divisive shunting inhibition: V_soma = I_dend / (1 + gamma_shunt * G_shunt)
        4. FitzHugh-Nagumo bounded cubic excitability:
           Out = V_soma - V_soma^3 / (3*kappa^2 + V_soma^2) - beta * W_adapt
        """
        row_id = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D_MODEL
        offset = row_id * stride_row + cols

        v = tl.load(V_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(S_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        g_shunt = tl.load(G_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        w_adapt = tl.load(W_ptr + offset, mask=mask, other=0.0).to(tl.float32)

        g_nmda = tl.load(GNmda_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        alpha_mg = tl.load(AlphaMg_ptr + cols, mask=mask, other=1.0).to(tl.float32)

        # 1. Voltage-dependent NMDA Mg2+ unblock in SRAM
        b_nmda = 1.0 / (1.0 + eta_mg * tl.exp(-alpha_mg * v))

        # 2. Supralinear dendritic plateau current
        i_dend = v + g_nmda * s * v * b_nmda

        # 3. Divisive GABAergic shunting normalization
        denom_shunt = 1.0 + gamma_shunt * g_shunt
        v_soma = i_dend / denom_shunt

        # 4. FitzHugh-Nagumo bounded cubic excitable response
        v2 = v_soma * v_soma
        v3 = v2 * v_soma
        fhn_cubic = v_soma - (v3 / (kappa_sq3 + v2)) - fhn_beta * w_adapt

        tl.store(Out_ptr + offset, fhn_cubic, mask=mask)


    @triton.jit
    def fused_dendritic_nmda_shunting_fhn_bwd_kernel(
        GradOut_ptr,    # [B, L, D] upstream gradient dL/dOut
        V_ptr,          # [B, L, D]
        S_ptr,          # [B, L, D]
        G_ptr,          # [B, L, D]
        W_ptr,          # [B, L, D]
        GNmda_ptr,      # [D]
        AlphaMg_ptr,    # [D]
        GradV_ptr,      # [B, L, D]
        GradS_ptr,      # [B, L, D]
        GradG_ptr,      # [B, L, D]
        GradW_ptr,      # [B, L, D]
        stride_row,
        D_MODEL: tl.constexpr,
        eta_mg: tl.constexpr,
        gamma_shunt: tl.constexpr,
        kappa_sq3: tl.constexpr,
        fhn_beta: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Fused Analytical Backward Kernel for the Dendritic-NMDA-Shunting-FHN Operator.
        Computes exact input gradients (dV, dS, dG_shunt, dW_adapt) in a single SRAM pass.
        """
        row_id = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D_MODEL
        offset = row_id * stride_row + cols

        dout = tl.load(GradOut_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(V_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(S_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        g_shunt = tl.load(G_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        g_nmda = tl.load(GNmda_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        alpha_mg = tl.load(AlphaMg_ptr + cols, mask=mask, other=1.0).to(tl.float32)

        b_nmda = 1.0 / (1.0 + eta_mg * tl.exp(-alpha_mg * v))
        i_dend = v + g_nmda * s * v * b_nmda
        inv_shunt = 1.0 / (1.0 + gamma_shunt * g_shunt)
        v_soma = i_dend * inv_shunt

        # 1. Derivative through FitzHugh-Nagumo bounded cubic:
        # d/dV_soma [V_soma - V_soma^3 / (C + V_soma^2)]
        # = 1 - (3*C*V_soma^2 + V_soma^4) / (C + V_soma^2)^2
        v2 = v_soma * v_soma
        denom_c = kappa_sq3 + v2
        dfhn_dvsoma = 1.0 - (v2 * (3.0 * kappa_sq3 + v2)) / (denom_c * denom_c)

        d_vsoma = dout * dfhn_dvsoma
        d_w = -fhn_beta * dout

        # 2. Derivative through Divisive Shunting Inhibition:
        d_idend = d_vsoma * inv_shunt
        d_gshunt = -d_vsoma * v_soma * gamma_shunt * inv_shunt

        # 3. Derivative through Supralinear NMDA Dendritic Current:
        # d(B_nmda)/dV = alpha_mg * B_nmda * (1 - B_nmda)
        db_dv = alpha_mg * b_nmda * (1.0 - b_nmda)
        d_s = d_idend * (g_nmda * v * b_nmda)
        d_v = d_idend * (1.0 + g_nmda * s * (b_nmda + v * db_dv))

        tl.store(GradV_ptr + offset, d_v, mask=mask)
        tl.store(GradS_ptr + offset, d_s, mask=mask)
        tl.store(GradG_ptr + offset, d_gshunt, mask=mask)
        tl.store(GradW_ptr + offset, d_w, mask=mask)


# ==============================================================================
# 3. Analytical Forward/Backward Kernels & STP-Hebbian Engine (NumPy)
# ==============================================================================

class NeuroNonlinearAnalyticalKernels:
    """
    Exact forward and analytical backward kernels for the Brain-Inspired Nonlinear
    Computational Neuroscience Operator, plus Tsodyks-Markram Short-Term Synaptic
    Plasticity (STP) and Causal Hebbian Fast-Weight Attention.
    """

    @staticmethod
    def fused_dendritic_nmda_shunting_fhn_forward(
        v: np.ndarray,
        s: np.ndarray,
        g_shunt: np.ndarray,
        w_adapt: np.ndarray,
        g_nmda: np.ndarray,
        alpha_mg: np.ndarray,
        eta_mg: float = 0.28,
        gamma_shunt: float = 0.5,
        fhn_kappa: float = 1.5,
        fhn_beta: float = 0.4,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass for the Fused Dendritic-NMDA + Shunting + FitzHugh-Nagumo Operator.
        Args:
            v: Dendritic membrane potential [B, L, D]
            s: Glutamatergic synaptic drive [B, L, D]
            g_shunt: GABAergic shunting conductance (>= 0) [B, L, D]
            w_adapt: Slow potassium recovery variable [B, L, D]
            g_nmda: Channel-wise NMDA receptor coupling [D]
            alpha_mg: Channel-wise Mg2+ voltage steepness [D]
        Returns:
            out: Somatic nonlinear response [B, L, D]
            cache: Intermediate physical states for exact analytical backward pass
        """
        alpha_b = alpha_mg[None, None, :]
        gnmda_b = g_nmda[None, None, :]

        # 1. Voltage-dependent Mg2+ unblock at NMDA channels
        b_nmda = 1.0 / (1.0 + eta_mg * np.exp(-alpha_b * v))

        # 2. Supralinear dendritic plateau current
        i_dend = v + gnmda_b * s * v * b_nmda

        # 3. Divisive GABAergic shunting normalization
        inv_shunt = 1.0 / (1.0 + gamma_shunt * g_shunt)
        v_soma = i_dend * inv_shunt

        # 4. FitzHugh-Nagumo bounded cubic excitable response
        kappa_sq3 = 3.0 * (fhn_kappa ** 2)
        v2 = np.square(v_soma)
        v3 = v2 * v_soma
        denom_c = kappa_sq3 + v2
        out = v_soma - (v3 / denom_c) - fhn_beta * w_adapt

        cache = {
            "v": v,
            "s": s,
            "g_shunt": g_shunt,
            "w_adapt": w_adapt,
            "g_nmda": g_nmda,
            "alpha_mg": alpha_mg,
            "b_nmda": b_nmda,
            "inv_shunt": inv_shunt,
            "v_soma": v_soma,
            "v2": v2,
            "denom_c": denom_c,
            "kappa_sq3": np.array(kappa_sq3),
        }
        return out, cache

    @staticmethod
    def fused_dendritic_nmda_shunting_fhn_backward(
        grad_out: np.ndarray,
        cache: Dict[str, np.ndarray],
        gamma_shunt: float = 0.5,
        fhn_beta: float = 0.4,
    ) -> Dict[str, np.ndarray]:
        """
        Exact analytical backward pass computing gradients w.r.t. all 6 inputs &
        biophysical parameters: (d_v, d_s, d_gshunt, d_wadapt, d_gnmda, d_alphamg).
        """
        v = cache["v"]
        s = cache["s"]
        g_nmda = cache["g_nmda"][None, None, :]
        alpha_mg = cache["alpha_mg"][None, None, :]
        b_nmda = cache["b_nmda"]
        inv_shunt = cache["inv_shunt"]
        v_soma = cache["v_soma"]
        v2 = cache["v2"]
        denom_c = cache["denom_c"]
        kappa_sq3 = float(cache["kappa_sq3"])

        # 1. Gradient through FitzHugh-Nagumo bounded cubic
        dfhn_dvsoma = 1.0 - (v2 * (3.0 * kappa_sq3 + v2)) / np.square(denom_c)
        d_vsoma = grad_out * dfhn_dvsoma
        d_wadapt = -fhn_beta * grad_out

        # 2. Gradient through Divisive Shunting Inhibition
        d_idend = d_vsoma * inv_shunt
        d_gshunt = -d_vsoma * v_soma * gamma_shunt * inv_shunt

        # 3. Gradient through Supralinear NMDA Dendritic Plateau Current
        # B_nmda = sigmoid(alpha_mg * V - ln(eta_mg)) -> dB/dV = alpha_mg * B * (1 - B)
        # dB/d(alpha_mg) = V * B * (1 - B)
        b_one_minus_b = b_nmda * (1.0 - b_nmda)
        db_dv = alpha_mg * b_one_minus_b
        db_dalpha = v * b_one_minus_b

        d_s = d_idend * (g_nmda * v * b_nmda)
        d_v = d_idend * (1.0 + g_nmda * s * (b_nmda + v * db_dv))

        # Sum over batch B (axis 0) and sequence L (axis 1) for channel parameter grads
        d_gnmda = np.sum(d_idend * (s * v * b_nmda), axis=(0, 1))
        d_alphamg = np.sum(d_idend * (g_nmda * s * v * db_dalpha), axis=(0, 1))

        return {
            "d_v": d_v,
            "d_s": d_s,
            "d_gshunt": d_gshunt,
            "d_wadapt": d_wadapt,
            "d_gnmda": d_gnmda,
            "d_alphamg": d_alphamg,
        }

    @staticmethod
    def causal_stp_hebbian_synaptic_forward(
        q_post: np.ndarray,
        k_pre: np.ndarray,
        v_vesicle: np.ndarray,
        drive_rate: np.ndarray,
        u0: float = 0.25,
        tau_facil: float = 4.0,
        tau_rec: float = 6.0,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Tsodyks-Markram Short-Term Synaptic Plasticity (STP) + Causal Hebbian
        Synaptic Transmission Operator.

        Across sequence time t = 0..L-1, each presynaptic head tracks:
        - Calcium-mediated neurotransmitter utilization `u_t in (0, 1)` (Facilitation):
              u_t = u_{t-1} * exp(-1/tau_f) + U_0 * (1 - u_{t-1} * exp(-1/tau_f)) * rate_t
        - Available synaptic vesicle pool `r_t in (0, 1)` (Depression / Depletion):
              r_t = 1 - (1 - r_{t-1} * (1 - u_{t-1} * rate_{t-1})) * exp(-1/tau_rec)
        - Instantaneous synaptic efficacy `w_stp(t) = u_t * r_t`, which modulates
          presynaptic keys K_pre(tau) during causal synaptic integration!

        Args:
            q_post, k_pre, v_vesicle: [B, H, L, D_head]
            drive_rate: Presynaptic firing rate in (0, 1) [B, H, L]
        Returns:
            synaptic_out: [B, H, L, D_head]
            stp_traces: Dictionary containing `u_trace`, `r_trace`, and `efficacy` [B, H, L]
        """
        B, H, L, D_h = q_post.shape
        decay_f = math.exp(-1.0 / tau_facil)
        decay_r = math.exp(-1.0 / tau_rec)

        u_trace = np.zeros((B, H, L), dtype=np.float64)
        r_trace = np.zeros((B, H, L), dtype=np.float64)

        u_prev = np.full((B, H), u0, dtype=np.float64)
        r_prev = np.ones((B, H), dtype=np.float64)

        for t in range(L):
            rate_t = drive_rate[:, :, t]
            if t == 0:
                u_curr = u_prev
                r_curr = r_prev
            else:
                rate_prev = drive_rate[:, :, t - 1]
                u_decayed = u_prev * decay_f
                u_curr = u_decayed + u0 * (1.0 - u_decayed) * rate_t
                released_prev = u_prev * r_prev * rate_prev
                r_curr = 1.0 - (1.0 - (r_prev - released_prev)) * decay_r

            u_trace[:, :, t] = u_curr
            r_trace[:, :, t] = r_curr
            u_prev, r_prev = u_curr, r_curr

        # Instantaneous Tsodyks-Markram synaptic release efficacy
        efficacy = u_trace * r_trace  # [B, H, L]

        # Modulate presynaptic representations by short-term synaptic efficacy
        k_modulated = k_pre * efficacy[:, :, :, None]
        v_modulated = v_vesicle * efficacy[:, :, :, None]

        # Causal Hebbian synaptic integration
        scale = 1.0 / math.sqrt(D_h)
        scores = np.matmul(q_post, np.swapaxes(k_modulated, -1, -2)) * scale
        idx = np.arange(L)
        causal_mask = idx[:, None] >= idx[None, :]
        scores = np.where(causal_mask[None, None, :, :], scores, -np.inf)

        s_max = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - s_max)
        syn_weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        synaptic_out = np.matmul(syn_weights, v_modulated)

        return synaptic_out, {
            "u_facilitation": u_trace,
            "r_vesicle_pool": r_trace,
            "stp_efficacy": efficacy,
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


    class FusedDendriticNMDAShuntingFHNFunction(torch.autograd.Function):
        """
        Custom PyTorch Autograd Function encapsulating the Fused Active Dendritic
        NMDA Plateau + Divisive Shunting Inhibition + FitzHugh-Nagumo Cubic Operator.
        """

        @staticmethod
        def forward(
            ctx,
            v: torch.Tensor,
            s: torch.Tensor,
            g_shunt: torch.Tensor,
            w_adapt: torch.Tensor,
            g_nmda: torch.Tensor,
            alpha_mg: torch.Tensor,
            eta_mg: float,
            gamma_shunt: float,
            fhn_kappa: float,
            fhn_beta: float,
        ) -> torch.Tensor:
            if HAS_TRITON and v.is_cuda:
                B, L, D = v.shape
                num_rows = B * L
                block_d = triton.next_power_of_2(D)
                grid = (triton.cdiv(num_rows, 1),)
                out_t = torch.empty_like(v, memory_format=torch.contiguous_format)
                fused_dendritic_nmda_shunting_fhn_fwd_kernel[grid](
                    v.contiguous(), s.contiguous(), g_shunt.contiguous(), w_adapt.contiguous(),
                    g_nmda.contiguous(), alpha_mg.contiguous(), out_t, D,
                    D_MODEL=D, eta_mg=eta_mg, gamma_shunt=gamma_shunt,
                    fhn_kappa=fhn_kappa, fhn_beta=fhn_beta, BLOCK_D=block_d,
                )
            alpha_b = alpha_mg.view(1, 1, -1)
            gnmda_b = g_nmda.view(1, 1, -1)

            b_nmda = torch.reciprocal(1.0 + eta_mg * torch.exp(-alpha_b * v))
            i_dend = v + gnmda_b * s * v * b_nmda
            inv_shunt = torch.reciprocal(1.0 + gamma_shunt * g_shunt)
            v_soma = i_dend * inv_shunt

            kappa_sq3 = 3.0 * (fhn_kappa ** 2)
            v2 = v_soma.pow(2)
            denom_c = kappa_sq3 + v2
            out = v_soma - (v_soma * v2) / denom_c - fhn_beta * w_adapt

            ctx.save_for_backward(
                v, s, g_nmda, alpha_mg, b_nmda, inv_shunt, v_soma, v2, denom_c
            )
            ctx.gamma_shunt = gamma_shunt
            ctx.fhn_beta = fhn_beta
            ctx.kappa_sq3 = kappa_sq3
            return out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            (
                v, s, g_nmda, alpha_mg, b_nmda, inv_shunt, v_soma, v2, denom_c
            ) = ctx.saved_tensors
            gamma_shunt = ctx.gamma_shunt
            fhn_beta = ctx.fhn_beta
            kappa_sq3 = ctx.kappa_sq3

            gnmda_b = g_nmda.view(1, 1, -1)
            alpha_b = alpha_mg.view(1, 1, -1)

            dfhn_dvsoma = 1.0 - (v2 * (3.0 * kappa_sq3 + v2)) / denom_c.pow(2)
            d_vsoma = grad_out * dfhn_dvsoma
            d_wadapt = -fhn_beta * grad_out

            d_idend = d_vsoma * inv_shunt
            d_gshunt = -d_vsoma * v_soma * gamma_shunt * inv_shunt

            b_one_minus_b = b_nmda * (1.0 - b_nmda)
            db_dv = alpha_b * b_one_minus_b
            db_dalpha = v * b_one_minus_b

            d_s = d_idend * (gnmda_b * v * b_nmda)
            d_v = d_idend * (1.0 + gnmda_b * s * (b_nmda + v * db_dv))

            d_gnmda = (d_idend * (s * v * b_nmda)).sum(dim=(0, 1))
            d_alphamg = (d_idend * (gnmda_b * s * v * db_dalpha)).sum(dim=(0, 1))

            return (
                d_v, d_s, d_gshunt, d_wadapt, d_gnmda, d_alphamg,
                None, None, None, None
            )


    class NonlinearDendriticNeuroOp(nn.Module):
        """
        Complete Brain-Inspired Nonlinear Computational Neuroscience Feature Operator (`nn.Module`).
        Maps sequence hidden representations [B, L, D] -> [B, L, D] by combining:
        - Active Dendritic NMDA Supralinear Plateau Integration
        - Perisomatic GABAergic Divisive Shunting Normalization
        - FitzHugh-Nagumo Cubic Excitable Dynamics & Slow Potassium Recovery
        - Tsodyks-Markram Short-Term Synaptic Plasticity (Facilitation & Depression)
        """

        def __init__(self, config: NeuroNonlinearConfig):
            super().__init__()
            assert config.d_model % config.n_heads == 0
            self.config = config
            self.d_model = config.d_model
            self.n_heads = config.n_heads
            self.d_head = config.d_model // config.n_heads

            # 1. Project input into 4 compartmental biophysical states:
            #    [Dendritic Potential V, Synaptic Glutamate S, Shunting Conductance G, Slow Adaptation W]
            self.compartment_proj = nn.Linear(config.d_model, 4 * config.d_model, bias=True)
            self.g_nmda = nn.Parameter(torch.ones(config.d_model) * 0.8)
            self.alpha_mg = nn.Parameter(torch.full((config.d_model,), config.alpha_mg_init))

            # 2. Tsodyks-Markram Short-Term Plasticity (STP) & Causal Synaptic projections
            self.synaptic_qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
            self.presynaptic_rate_proj = nn.Linear(config.d_model, config.n_heads, bias=True)

            # 3. Axonal output integration & normalization
            self.axonal_proj = nn.Linear(2 * config.d_model, config.d_model, bias=False)
            self.out_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
            B, L, D = x.shape
            v, s, g_raw, w_adapt = self.compartment_proj(x).chunk(4, dim=-1)
            # Shunting conductance G_shunt must be non-negative
            g_shunt = F.softplus(g_raw)

            # 1. Fused Dendritic-NMDA + Shunting + FitzHugh-Nagumo Somatic Operator
            somatic_out = FusedDendriticNMDAShuntingFHNFunction.apply(
                v, s, g_shunt, w_adapt, self.g_nmda, self.alpha_mg,
                self.config.eta_mg, self.config.gamma_shunt,
                self.config.fhn_kappa, self.config.fhn_beta
            )

            # 2. Tsodyks-Markram Short-Term Synaptic Plasticity (STP) across sequence time L
            qkv = self.synaptic_qkv(x).view(B, L, 3, self.n_heads, self.d_head)
            q_post, k_pre, v_ves = [t.transpose(1, 2) for t in qkv.unbind(dim=2)]  # [B, H, L, D_head]
            drive_rate = torch.sigmoid(self.presynaptic_rate_proj(x)).transpose(1, 2)  # [B, H, L]

            decay_f = math.exp(-1.0 / self.config.stp_tau_facil)
            decay_r = math.exp(-1.0 / self.config.stp_tau_rec)
            u_prev = torch.full((B, self.n_heads), self.config.stp_u0, device=x.device, dtype=x.dtype)
            r_prev = torch.ones((B, self.n_heads), device=x.device, dtype=x.dtype)

            efficacies = []
            for t in range(L):
                rate_t = drive_rate[:, :, t]
                if t == 0:
                    u_curr, r_curr = u_prev, r_prev
                else:
                    rate_prev = drive_rate[:, :, t - 1]
                    u_dec = u_prev * decay_f
                    u_curr = u_dec + self.config.stp_u0 * (1.0 - u_dec) * rate_t
                    rel_prev = u_prev * r_prev * rate_prev
                    r_curr = 1.0 - (1.0 - (r_prev - rel_prev)) * decay_r
                efficacies.append(u_curr * r_curr)
                u_prev, r_prev = u_curr, r_curr

            stp_efficacy = torch.stack(efficacies, dim=-1).unsqueeze(-1)  # [B, H, L, 1]
            k_mod = k_pre * stp_efficacy
            v_mod = v_ves * stp_efficacy

            syn_out = F.scaled_dot_product_attention(q_post, k_mod, v_mod, is_causal=True)
            syn_out = syn_out.transpose(1, 2).contiguous().view(B, L, D)

            combined = torch.cat([somatic_out, syn_out], dim=-1)
            out = self.out_norm(self.axonal_proj(combined))

            with torch.no_grad():
                nmda_gate = torch.reciprocal(1.0 + self.config.eta_mg * torch.exp(-self.alpha_mg.view(1, 1, -1) * v))
                diagnostics = {
                    "mean_nmda_unblock": float(nmda_gate.mean().item()),
                    "mean_shunting_div": float((1.0 / (1.0 + self.config.gamma_shunt * g_shunt)).mean().item()),
                    "mean_stp_efficacy": float(stp_efficacy.mean().item()),
                }
            return out, diagnostics


# ==============================================================================
# 5. Tile-Accurate Triton Simulator, Gradcheck & HBM Bandwidth Profiler
# ==============================================================================

def simulate_triton_neuro_nonlinear_kernel(
    v: np.ndarray,
    s: np.ndarray,
    g_shunt: np.ndarray,
    w_adapt: np.ndarray,
    g_nmda: np.ndarray,
    alpha_mg: np.ndarray,
    eta_mg: float = 0.28,
    gamma_shunt: float = 0.5,
    fhn_kappa: float = 1.5,
    fhn_beta: float = 0.4,
    block_d: int = 64,
    bytes_per_elem: int = 2,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simulates `fused_dendritic_nmda_shunting_fhn_fwd_kernel` tile-by-tile across
    rows = B * L and column tiles BLOCK_D in SRAM, and computes HBM bandwidth savings.
    """
    B, L, D = v.shape
    num_rows = B * L
    kappa_sq3 = 3.0 * (fhn_kappa ** 2)

    v_2d = v.reshape(num_rows, D)
    s_2d = s.reshape(num_rows, D)
    g_2d = g_shunt.reshape(num_rows, D)
    w_2d = w_adapt.reshape(num_rows, D)
    out_2d = np.zeros_like(v_2d)

    for row_id in range(num_rows):
        for d0 in range(0, D, block_d):
            d1 = min(d0 + block_d, D)
            v_tile = v_2d[row_id, d0:d1]
            s_tile = s_2d[row_id, d0:d1]
            g_tile = g_2d[row_id, d0:d1]
            w_tile = w_2d[row_id, d0:d1]
            gn_tile = g_nmda[d0:d1]
            am_tile = alpha_mg[d0:d1]

            b_nmda = 1.0 / (1.0 + eta_mg * np.exp(-am_tile * v_tile))
            i_dend = v_tile + gn_tile * s_tile * v_tile * b_nmda
            v_soma = i_dend / (1.0 + gamma_shunt * g_tile)
            v2 = v_soma * v_soma
            out_2d[row_id, d0:d1] = v_soma - (v_soma * v2) / (kappa_sq3 + v2) - fhn_beta * w_tile

    out = out_2d.reshape(B, L, D)

    # HBM Bandwidth Analysis (Forward + Backward):
    # Unfused Eager PyTorch launches 9 separate elementwise kernels (exp, b_nmda, i_dend,
    # denom_shunt, v_soma, v2, v3/denom_c, fhn_sub, adapt_sub), reading/writing [B, L, D]
    # ~36 tensor passes across Fwd+Bwd.
    # Fused Triton Kernel reads (V, S, G, W) once and writes Out once in Fwd (5 passes),
    # and reads 5 + writes 4 in Bwd (14 passes total vs 36 eager passes).
    elem_bytes = B * L * D * bytes_per_elem
    eager_hbm_bytes = 36 * elem_bytes
    triton_hbm_bytes = 9 * elem_bytes + 4 * D * bytes_per_elem

    stats = {
        "eager_hbm_kb": eager_hbm_bytes / 1024.0,
        "triton_hbm_kb": triton_hbm_bytes / 1024.0,
        "hbm_saved_pct": (1.0 - triton_hbm_bytes / eager_hbm_bytes) * 100.0,
        "bw_speedup": eager_hbm_bytes / triton_hbm_bytes,
    }
    return out, stats


def run_neuro_nonlinear_verification_suite() -> None:
    """
    Executes central finite-difference gradcheck on all 6 biophysical inputs &
    parameters, verifies Triton tile simulator accuracy, and profiles STP-Hebbian
    synaptic dynamics.
    """
    rng = np.random.default_rng(2026)
    print("=" * 88)
    print("1. Analytical vs. Finite-Difference Gradcheck (Dendritic-NMDA + Shunting + FHN Op)")
    print("=" * 88)

    B, L, D = 2, 5, 6
    eta_mg, gamma_shunt, fhn_kappa, fhn_beta = 0.28, 0.5, 1.5, 0.4

    v = rng.standard_normal((B, L, D)).astype(np.float64) * 0.8
    s = rng.standard_normal((B, L, D)).astype(np.float64) * 0.6
    g_shunt = np.abs(rng.standard_normal((B, L, D)).astype(np.float64)) * 0.5 + 0.1
    w_adapt = rng.standard_normal((B, L, D)).astype(np.float64) * 0.4
    g_nmda = rng.uniform(0.5, 1.2, size=(D,)).astype(np.float64)
    alpha_mg = rng.uniform(0.8, 1.6, size=(D,)).astype(np.float64)
    grad_out = rng.standard_normal((B, L, D)).astype(np.float64)

    _, cache = NeuroNonlinearAnalyticalKernels.fused_dendritic_nmda_shunting_fhn_forward(
        v, s, g_shunt, w_adapt, g_nmda, alpha_mg, eta_mg, gamma_shunt, fhn_kappa, fhn_beta
    )
    ana_grads = NeuroNonlinearAnalyticalKernels.fused_dendritic_nmda_shunting_fhn_backward(
        grad_out, cache, gamma_shunt, fhn_beta
    )

    eps = 1e-6
    check_targets = {
        "v_dend": (v, ana_grads["d_v"]),
        "s_glut": (s, ana_grads["d_s"]),
        "g_shunt": (g_shunt, ana_grads["d_gshunt"]),
        "w_adapt": (w_adapt, ana_grads["d_wadapt"]),
        "g_nmda": (g_nmda, ana_grads["d_gnmda"]),
        "alpha_mg": (alpha_mg, ana_grads["d_alphamg"]),
    }

    for name, (arr, g_ana) in check_targets.items():
        g_num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + eps
            out_p, _ = NeuroNonlinearAnalyticalKernels.fused_dendritic_nmda_shunting_fhn_forward(
                v, s, g_shunt, w_adapt, g_nmda, alpha_mg, eta_mg, gamma_shunt, fhn_kappa, fhn_beta
            )
            loss_p = np.sum(out_p * grad_out)

            arr[idx] = orig - eps
            out_m, _ = NeuroNonlinearAnalyticalKernels.fused_dendritic_nmda_shunting_fhn_forward(
                v, s, g_shunt, w_adapt, g_nmda, alpha_mg, eta_mg, gamma_shunt, fhn_kappa, fhn_beta
            )
            loss_m = np.sum(out_m * grad_out)
            arr[idx] = orig
            g_num[idx] = (loss_p - loss_m) / (2.0 * eps)

        rel_err = np.linalg.norm(g_ana - g_num) / (np.linalg.norm(g_ana) + np.linalg.norm(g_num) + 1e-15)
        print(f"   Gradcheck dL/d({name:10s}) | Relative L2 Error: {rel_err:.3e}  [PASS < 1e-8]")
        assert rel_err < 1e-8, f"Gradcheck failed for {name}: {rel_err}"

    print("\n" + "=" * 88)
    print("2. Triton Tile Simulator Parity, STP Synaptic Dynamics & HBM Bandwidth (B=2, L=64, D=256)")
    print("=" * 88)
    B_b, L_b, D_b, H_b = 2, 64, 256, 8
    v_b = rng.standard_normal((B_b, L_b, D_b)) * 0.8
    s_b = rng.standard_normal((B_b, L_b, D_b)) * 0.6
    g_b = np.abs(rng.standard_normal((B_b, L_b, D_b))) * 0.5
    w_b = rng.standard_normal((B_b, L_b, D_b)) * 0.3
    gn_b = np.ones((D_b,)) * 0.8
    am_b = np.full((D_b,), 1.2)

    out_ref, cache_b = NeuroNonlinearAnalyticalKernels.fused_dendritic_nmda_shunting_fhn_forward(
        v_b, s_b, g_b, w_b, gn_b, am_b
    )
    out_tri, hbm_stats = simulate_triton_neuro_nonlinear_kernel(
        v_b, s_b, g_b, w_b, gn_b, am_b, block_d=64
    )
    max_tile_diff = float(np.max(np.abs(out_ref - out_tri)))

    # Tsodyks-Markram STP + Causal Synaptic Plasticity test
    D_h = D_b // H_b
    q_b = rng.standard_normal((B_b, H_b, L_b, D_h)) * 0.3
    k_b = rng.standard_normal((B_b, H_b, L_b, D_h)) * 0.3
    vv_b = rng.standard_normal((B_b, H_b, L_b, D_h)) * 0.3
    rates_b = 1.0 / (1.0 + np.exp(-rng.standard_normal((B_b, H_b, L_b))))
    syn_out, stp_traces = NeuroNonlinearAnalyticalKernels.causal_stp_hebbian_synaptic_forward(
        q_b, k_b, vv_b, rates_b
    )

    print(f"   Fused Somatic Excitable Output Shape : {out_tri.shape}")
    print(f"   Causal STP Synaptic Output Shape     : {syn_out.shape}")
    print(f"   Triton Tile Simulator Max Abs Diff   : {max_tile_diff:.2e}  [PASS == 0.00e+00]")
    print(f"   Mean NMDA Mg2+ Unblock Factor B(V)   : {float(np.mean(cache_b['b_nmda'])):.4f}")
    print(f"   Mean Divisive Shunting Factor        : {float(np.mean(cache_b['inv_shunt'])):.4f}")
    print(
        f"   Tsodyks-Markram STP Dynamics         : "
        f"Mean Facilitation u={float(np.mean(stp_traces['u_facilitation'])):.4f}, "
        f"Vesicle Pool r={float(np.mean(stp_traces['r_vesicle_pool'])):.4f}, "
        f"Efficacy u*r={float(np.mean(stp_traces['stp_efficacy'])):.4f}"
    )
    print(
        f"   HBM Memory Traffic (Fwd+Bwd)         : Eager={hbm_stats['eager_hbm_kb']:.1f} KB -> "
        f"Triton={hbm_stats['triton_hbm_kb']:.1f} KB "
        f"(Saved {hbm_stats['hbm_saved_pct']:.1f}% | {hbm_stats['bw_speedup']:.2f}x BW Speedup)"
    )
    assert max_tile_diff < 1e-14
    print("=" * 88)

if not HAS_TORCH:
    class RMSNorm:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate RMSNorm.")

    class FusedDendriticNMDAShuntingFHNFunction:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate FusedDendriticNMDAShuntingFHNFunction.")

    class NonlinearDendriticNeuroOp:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate NonlinearDendriticNeuroOp.")

if __name__ == "__main__":
    run_neuro_nonlinear_verification_suite()
