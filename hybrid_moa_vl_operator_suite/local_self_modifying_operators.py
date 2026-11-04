"""
Local Nonlinear Self-Modifying Weight Operator Library
(`local_self_modifying_operators.py`).

A standalone deep learning operator library implementing local, input-dependent
nonlinear weight self-modification (Differentiable Plasticity / Fast-Weight
Programmers / Nonlinear Delta-Oja Rule) as end-to-end differentiable PyTorch
operators (`nn.Module` & `torch.autograd.Function`), fused Triton GPU Kernels
(`@triton.jit`), and analytical NumPy forward/backward verification kernels.

Core Mathematical Mechanisms of Local Nonlinear Weight Self-Modification:
1. Group-Local Subspace Weight Partitioning (`G` local blocks of dim `D_g = D / G`):
   Confines weight self-modification to local modular subspaces so each group g
   autonomously adapts its local synaptic matrix W_eff(g) in R^{D_g x D_g}:
       W_eff(x, g) = (1 - lambda_oja * E_loc(x, g)) * W_base(g) + alpha_g * dW_nl(x, g)
2. Bounded Rational Nonlinear Plasticity Rule (`Psi_nl`):
   Instead of unbounded linear Hebbian outer products (which diverge in deep nets),
   local weight updates pass through a smooth rational saturation nonlinearity:
       Psi_nl(z) = (eta * z) / (1 + (eta * z)^2),   z = V_loc(x)^T x_g
       dW_nl(x, g) x_g = U_loc(x) * Psi_nl(V_loc(x)^T x_g)
3. Oja's Local Homeostatic Weight Stabilization (`lambda_oja`):
   Normalizes the base weight response using the local presynaptic projection
   energy E_loc(x, g) = mean((V_loc(x)^T x_g)^2), preventing synaptic runaway:
       y_g = (1 - lambda_oja * E_loc(x, g)) * (W_base(g) x_g) + alpha_g * (U_loc(x) Psi_nl(z))
4. Recurrent Nonlinear Delta-Oja Fast-Weight Plasticity (`W_t` across time `t`):
   Updates a persistent local fast-weight matrix W_t in R^{D_g x D_g} via a
   nonlinear prediction-error Delta rule with Oja decay:
       err_t = tanh(v_t - W_{t-1} k_t)
       W_t   = gamma_t * W_{t-1} + eta_t * (err_t (x) k_t^T - beta_oja * (W_{t-1} k_t) (x) k_t^T)
5. Fused Triton GPU Kernels (`@triton.jit`):
   Executes the local weight perturbation, rational nonlinear plasticity, and
   Oja homeostatic scaling directly in SRAM without materializing the [B, L, G, D_g, D_g]
   modified weight tensor in HBM.
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
class LocalSelfModConfig:
    """Configuration for the Local Nonlinear Self-Modifying Weight Operator."""
    d_model: int = 256
    num_local_groups: int = 8             # G: number of local structural weight subspaces
    local_rank: int = 4                   # R: rank of instantaneous local weight modification
    eta_plasticity_init: float = 0.8      # Initial nonlinear plasticity gain eta
    lambda_oja_init: float = 0.15         # Oja's local homeostatic decay coefficient
    alpha_mod_init: float = 0.5           # Coupling scale of self-modified weight delta
    recurrent_beta_oja: float = 0.10      # Oja decay in recurrent Delta-Oja fast-weight rule
    rms_norm_eps: float = 1e-6


# ==============================================================================
# 2. Fused Triton GPU Kernels (@triton.jit)
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def fused_local_self_mod_weight_fwd_kernel(
        YBase_ptr,      # [B, L, G, D_g] output of static local base weight W_base(g) @ x_g
        U_ptr,          # [B, L, G, D_g, R] local post-synaptic basis U(x)
        Z_ptr,          # [B, L, G, R] local pre-synaptic rank projection z = V(x)^T @ x_g
        Eta_ptr,        # [G, R] learnable plasticity gain eta
        LambdaOja_ptr,  # [G] learnable Oja homeostatic coefficient lambda_oja
        AlphaMod_ptr,   # [G, D_g] learnable self-modification coupling alpha_mod
        Out_ptr,        # [B, L, G, D_g] self-modified output y_g
        D_G: tl.constexpr,
        R_RANK: tl.constexpr,
        G_GROUPS: tl.constexpr,
        BLOCK_DG: tl.constexpr,
    ):
        """
        Each program instance handles one local group instance `row_id = (b * L + l) * G + g`.
        Fuses in SRAM without materializing dW_nl [B, L, G, D_g, D_g] in HBM:
        1. Rational nonlinear plasticity on rank coordinates:
               psi_r = (eta_r * z_r) / (1 + (eta_r * z_r)^2)
        2. Local Oja homeostatic energy:
               E_loc = mean_r(z_r^2)
        3. Matrix-free contraction of self-modified weight delta:
               y_plastic[d] = sum_{r=0..R-1} U[d, r] * psi_r
        4. Homeostatic combination with base weight output:
               Out[d] = (1 - lambda_oja * E_loc) * YBase[d] + alpha_mod[d] * y_plastic[d]
        """
        row_id = tl.program_id(0)
        g_idx = row_id % G_GROUPS

        cols = tl.arange(0, BLOCK_DG)
        mask_d = cols < D_G
        base_offset = row_id * D_G + cols

        y_base = tl.load(YBase_ptr + base_offset, mask=mask_d, other=0.0).to(tl.float32)
        alpha_mod = tl.load(AlphaMod_ptr + g_idx * D_G + cols, mask=mask_d, other=0.0).to(tl.float32)
        lambda_oja = tl.load(LambdaOja_ptr + g_idx).to(tl.float32)

        # Accumulate y_plastic[d] and local Oja energy E_loc across rank r in [0..R_RANK-1]
        y_plastic = tl.zeros((BLOCK_DG,), dtype=tl.float32)
        z_sq_sum = tl.zeros((1,), dtype=tl.float32)

        for r in range(0, R_RANK):
            z_r = tl.load(Z_ptr + row_id * R_RANK + r).to(tl.float32)
            eta_r = tl.load(Eta_ptr + g_idx * R_RANK + r).to(tl.float32)

            ez = eta_r * z_r
            psi_r = ez / (1.0 + ez * ez)
            z_sq_sum += z_r * z_r

            u_ptrs = U_ptr + (row_id * D_G + cols) * R_RANK + r
            u_col = tl.load(u_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            y_plastic += u_col * psi_r

        e_loc = z_sq_sum / R_RANK
        oja_scale = 1.0 - lambda_oja * e_loc
        out = oja_scale * y_base + alpha_mod * y_plastic

        tl.store(Out_ptr + base_offset, out, mask=mask_d)


    @triton.jit
    def fused_local_self_mod_weight_bwd_kernel(
        GradOut_ptr,    # [B, L, G, D_g] upstream gradient
        YBase_ptr,      # [B, L, G, D_g]
        U_ptr,          # [B, L, G, D_g, R]
        Z_ptr,          # [B, L, G, R]
        Eta_ptr,        # [G, R]
        LambdaOja_ptr,  # [G]
        AlphaMod_ptr,   # [G, D_g]
        GradYBase_ptr,  # [B, L, G, D_g] output grad
        GradU_ptr,      # [B, L, G, D_g, R] output grad
        GradZ_ptr,      # [B, L, G, R] output grad
        D_G: tl.constexpr,
        R_RANK: tl.constexpr,
        G_GROUPS: tl.constexpr,
        BLOCK_DG: tl.constexpr,
    ):
        """
        Fused Analytical Backward Kernel for the Local Self-Modifying Weight Operator.
        Computes dYBase, dU, and dZ in a single SRAM pass per (b, l, g) instance.
        """
        row_id = tl.program_id(0)
        g_idx = row_id % G_GROUPS

        cols = tl.arange(0, BLOCK_DG)
        mask_d = cols < D_G
        base_offset = row_id * D_G + cols

        dout = tl.load(GradOut_ptr + base_offset, mask=mask_d, other=0.0).to(tl.float32)
        y_base = tl.load(YBase_ptr + base_offset, mask=mask_d, other=0.0).to(tl.float32)
        alpha_mod = tl.load(AlphaMod_ptr + g_idx * D_G + cols, mask=mask_d, other=0.0).to(tl.float32)
        lambda_oja = tl.load(LambdaOja_ptr + g_idx).to(tl.float32)

        dy_plastic = dout * alpha_mod
        dot_dout_ybase = tl.sum(dout * y_base, axis=0)

        # First pass over R_RANK: compute E_loc and gradients dU, dZ
        z_sq_sum = tl.zeros((1,), dtype=tl.float32)
        for r in range(0, R_RANK):
            z_r = tl.load(Z_ptr + row_id * R_RANK + r).to(tl.float32)
            eta_r = tl.load(Eta_ptr + g_idx * R_RANK + r).to(tl.float32)
            z_sq_sum += z_r * z_r

            ez = eta_r * z_r
            denom = 1.0 + ez * ez
            psi_r = ez / denom
            # d(psi_r)/d(z_r) = eta_r * (1 - ez^2) / (1 + ez^2)^2
            dpsi_dz = eta_r * (1.0 - ez * ez) / (denom * denom)

            u_ptrs = U_ptr + (row_id * D_G + cols) * R_RANK + r
            u_col = tl.load(u_ptrs, mask=mask_d, other=0.0).to(tl.float32)

            # dU[d, r] = dy_plastic[d] * psi_r
            du_ptrs = GradU_ptr + (row_id * D_G + cols) * R_RANK + r
            tl.store(du_ptrs, dy_plastic * psi_r, mask=mask_d)

            # dZ[r] = sum_d(dy_plastic[d] * U[d, r]) * dpsi_dz + d(oja_scale)/d(z_r) * dot(dout, y_base)
            d_psi_r = tl.sum(dy_plastic * u_col, axis=0)
            dz_oja = (-2.0 * lambda_oja * z_r / R_RANK) * dot_dout_ybase
            dz_r = d_psi_r * dpsi_dz + dz_oja
            tl.store(GradZ_ptr + row_id * R_RANK + r, dz_r)

        e_loc = z_sq_sum / R_RANK
        oja_scale = 1.0 - lambda_oja * e_loc
        dy_base = dout * oja_scale
        tl.store(GradYBase_ptr + base_offset, dy_base, mask=mask_d)


# ==============================================================================
# 3. Analytical Forward/Backward Kernels & Fast-Weight Engine (NumPy)
# ==============================================================================

class LocalSelfModifyingAnalyticalKernels:
    """
    Exact forward and analytical backward kernels for the Local Nonlinear
    Self-Modifying Weight Operator, plus Recurrent Nonlinear Delta-Oja Fast-Weight
    Plasticity across sequence steps.
    """

    @staticmethod
    def fused_local_self_mod_forward(
        y_base: np.ndarray,
        u_basis: np.ndarray,
        z_proj: np.ndarray,
        eta: np.ndarray,
        lambda_oja: np.ndarray,
        alpha_mod: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass for the Fused Local Nonlinear Weight Self-Modification Operator.
        Args:
            y_base: Static local base weight output [B, L, G, D_g]
            u_basis: Local post-synaptic basis vectors [B, L, G, D_g, R]
            z_proj: Local pre-synaptic rank coordinates V(x)^T x_g [B, L, G, R]
            eta: Nonlinear plasticity gain [G, R]
            lambda_oja: Oja's local homeostatic decay factor [G]
            alpha_mod: Self-modification coupling scale [G, D_g]
        Returns:
            out: Self-modified local output [B, L, G, D_g]
            cache: Saved intermediate tensors for analytical backward pass
        """
        _, _, G, D_g, R = u_basis.shape
        eta_b = eta[None, None, :, :]          # [1, 1, G, R]
        lam_b = lambda_oja[None, None, :, None]  # [1, 1, G, 1]
        alp_b = alpha_mod[None, None, :, :]    # [1, 1, G, D_g]

        # 1. Bounded rational nonlinear plasticity on local rank coordinates
        ez = eta_b * z_proj                    # [B, L, G, R]
        denom = 1.0 + np.square(ez)            # [B, L, G, R]
        psi = ez / denom                       # [B, L, G, R]

        # 2. Matrix-free contraction of local weight perturbation U @ psi
        y_plastic = np.einsum("blgdr,blgr->blgd", u_basis, psi)  # [B, L, G, D_g]

        # 3. Oja's local homeostatic energy E_loc = mean_r(z_r^2)
        e_loc = np.mean(np.square(z_proj), axis=-1, keepdims=True)  # [B, L, G, 1]
        oja_scale = 1.0 - lam_b * e_loc

        # 4. Self-modified output
        out = oja_scale * y_base + alp_b * y_plastic

        cache = {
            "y_base": y_base,
            "u_basis": u_basis,
            "z_proj": z_proj,
            "eta": eta,
            "lambda_oja": lambda_oja,
            "alpha_mod": alpha_mod,
            "ez": ez,
            "denom": denom,
            "psi": psi,
            "y_plastic": y_plastic,
            "e_loc": e_loc,
            "oja_scale": oja_scale,
        }
        return out, cache

    @staticmethod
    def fused_local_self_mod_backward(
        grad_out: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Exact analytical backward pass computing gradients w.r.t. all 6 inputs &
        plasticity parameters: (d_ybase, d_ubasis, d_zproj, d_eta, d_lambda_oja, d_alpha_mod).
        """
        y_base = cache["y_base"]
        u_basis = cache["u_basis"]
        z_proj = cache["z_proj"]
        eta = cache["eta"][None, None, :, :]
        lambda_oja = cache["lambda_oja"][None, None, :, None]
        alpha_mod = cache["alpha_mod"][None, None, :, :]
        ez = cache["ez"]
        denom = cache["denom"]
        psi = cache["psi"]
        y_plastic = cache["y_plastic"]
        e_loc = cache["e_loc"]
        oja_scale = cache["oja_scale"]

        R = z_proj.shape[-1]

        # 1. Gradients w.r.t. y_base, alpha_mod, and lambda_oja
        d_ybase = grad_out * oja_scale
        d_alpha_mod = np.sum(grad_out * y_plastic, axis=(0, 1))  # [G, D_g]

        dot_dout_ybase = np.sum(grad_out * y_base, axis=-1, keepdims=True)  # [B, L, G, 1]
        d_ojasca = dot_dout_ybase
        d_lambda_oja = np.sum(-d_ojasca * e_loc, axis=(0, 1, 3))  # [G]
        d_eloc = -d_ojasca * lambda_oja                           # [B, L, G, 1]

        # 2. Gradients through y_plastic = einsum("blgdr,blgr->blgd", u_basis, psi)
        dy_plastic = grad_out * alpha_mod  # [B, L, G, D_g]
        d_ubasis = np.einsum("blgd,blgr->blgdr", dy_plastic, psi)
        d_psi = np.einsum("blgd,blgdr->blgr", dy_plastic, u_basis)

        # 3. Gradients through rational nonlinearity psi = ez / (1 + ez^2)
        # d(psi)/d(ez) = (1 - ez^2) / (1 + ez^2)^2
        dpsi_dez = (1.0 - np.square(ez)) / np.square(denom)
        d_ez = d_psi * dpsi_dez

        d_eta = np.sum(d_ez * z_proj, axis=(0, 1))  # [G, R]
        d_zproj = d_ez * eta + d_eloc * (2.0 * z_proj / R)

        return {
            "d_ybase": d_ybase,
            "d_ubasis": d_ubasis,
            "d_zproj": d_zproj,
            "d_eta": d_eta,
            "d_lambda_oja": d_lambda_oja,
            "d_alpha_mod": d_alpha_mod,
        }

    @staticmethod
    def recurrent_nonlinear_delta_oja_fastweight_forward(
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        w_init: np.ndarray,
        eta_gate: np.ndarray,
        gamma_forget: np.ndarray,
        beta_oja: float = 0.10,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Recurrent Local Nonlinear Delta-Oja Fast-Weight Self-Modification Operator
        (Test-Time Training / Fast-Weight Programmer formulation).

        At each sequence step t = 0..L-1, each local group g in [0..G-1] reads and
        self-modifies its local synaptic matrix W_t[b, g] in R^{D_g x D_g}:
            1. Readout:        y_t = W_{t-1} @ q_t
            2. Local Recall:   v_hat_t = tanh(W_{t-1} @ k_t)
            3. Nonlinear Err:  err_t = tanh(v_t - v_hat_t)
            4. Self-Modify:    W_t = gamma_t * W_{t-1}
                                     + eta_t * (err_t (x) k_t^T - beta_oja * (v_hat_t^2 * (W_{t-1} @ k_t)) (x) k_t^T)

        Args:
            q, k, v: Group-partitioned representations [B, L, G, D_g]
            w_init: Initial local base weight per group [G, D_g, D_g]
            eta_gate: Token-wise local plasticity rate in (0, 1) [B, L, G]
            gamma_forget: Token-wise retention factor in (0, 1) [B, L, G]
        Returns:
            y_out: Readout sequence [B, L, G, D_g]
            w_final: Final self-modified local weight matrices [B, G, D_g, D_g]
            diagnostics: Dictionary of weight drift norm and prediction error metrics
        """
        B, L, G, D_g = q.shape
        # Broadcast initial base weights across batch B -> [B, G, D_g, D_g]
        w_curr = np.broadcast_to(w_init[None, ...], (B, G, D_g, D_g)).copy()
        y_out = np.zeros_like(q)
        err_norms = []

        for t in range(L):
            q_t = q[:, t]  # [B, G, D_g]
            k_t = k[:, t]  # [B, G, D_g]
            v_t = v[:, t]  # [B, G, D_g]

            # Normalize k_t to unit L2 norm per group for stable local Hebbian/Delta outer product
            k_norm = k_t / (np.linalg.norm(k_t, axis=-1, keepdims=True) + 1e-6)

            # 1. Readout with current self-modified local weight W_{t-1}
            y_t = np.einsum("bgij,bgj->bgi", w_curr, q_t)
            y_out[:, t] = y_t

            # 2. Local associative recall & nonlinear prediction error
            wk_t = np.einsum("bgij,bgj->bgi", w_curr, k_norm)
            v_hat_t = np.tanh(wk_t)
            err_t = np.tanh(v_t - v_hat_t)
            err_norms.append(float(np.mean(np.square(err_t))))

            # 3. Local Nonlinear Delta + Oja Homeostatic Self-Modification
            oja_term = beta_oja * np.square(v_hat_t) * wk_t
            delta_vec = err_t - oja_term  # [B, G, D_g]
            outer_update = np.einsum("bgi,bgj->bgij", delta_vec, k_norm)  # [B, G, D_g, D_g]

            gam_t = gamma_forget[:, t, :, None, None]  # [B, G, 1, 1]
            eta_t = eta_gate[:, t, :, None, None]      # [B, G, 1, 1]
            w_curr = gam_t * w_curr + eta_t * outer_update

        weight_drift = float(np.linalg.norm(w_curr - w_init[None, ...]) / math.sqrt(B * G * D_g * D_g))
        diagnostics = {
            "mean_nonlinear_delta_err": sum(err_norms) / max(len(err_norms), 1),
            "rms_fast_weight_drift": weight_drift,
        }
        return y_out, w_curr, diagnostics


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


    class FusedLocalSelfModWeightFunction(torch.autograd.Function):
        """
        Custom PyTorch Autograd Function encapsulating the Fused Local Nonlinear
        Weight Self-Modification Operator (Bounded Rational Plasticity + Oja Homeostasis).
        """

        @staticmethod
        def forward(
            ctx,
            y_base: torch.Tensor,
            u_basis: torch.Tensor,
            z_proj: torch.Tensor,
            eta: torch.Tensor,
            lambda_oja: torch.Tensor,
            alpha_mod: torch.Tensor,
        ) -> torch.Tensor:
            eta_b = eta.unsqueeze(0).unsqueeze(0)              # [1, 1, G, R]
            lam_b = lambda_oja.view(1, 1, -1, 1)               # [1, 1, G, 1]
            alp_b = alpha_mod.unsqueeze(0).unsqueeze(0)        # [1, 1, G, D_g]

            ez = eta_b * z_proj
            denom = 1.0 + ez.pow(2)
            psi = ez / denom

            y_plastic = torch.einsum("blgdr,blgr->blgd", u_basis, psi)
            e_loc = z_proj.pow(2).mean(dim=-1, keepdim=True)
            oja_scale = 1.0 - lam_b * e_loc
            out = oja_scale * y_base + alp_b * y_plastic

            ctx.save_for_backward(
                y_base, u_basis, z_proj, eta, lambda_oja, alpha_mod,
                ez, denom, psi, y_plastic, e_loc, oja_scale
            )
            return out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            (
                y_base, u_basis, z_proj, eta, lambda_oja, alpha_mod,
                ez, denom, psi, y_plastic, e_loc, oja_scale
            ) = ctx.saved_tensors
            R = z_proj.shape[-1]
            eta_b = eta.unsqueeze(0).unsqueeze(0)
            lam_b = lambda_oja.view(1, 1, -1, 1)
            alp_b = alpha_mod.unsqueeze(0).unsqueeze(0)

            d_ybase = grad_out * oja_scale
            d_alpha_mod = (grad_out * y_plastic).sum(dim=(0, 1))

            dot_dout_ybase = (grad_out * y_base).sum(dim=-1, keepdim=True)
            d_lambda_oja = (-dot_dout_ybase * e_loc).sum(dim=(0, 1, 3))
            d_eloc = -dot_dout_ybase * lam_b

            dy_plastic = grad_out * alp_b
            d_ubasis = torch.einsum("blgd,blgr->blgdr", dy_plastic, psi)
            d_psi = torch.einsum("blgd,blgdr->blgr", dy_plastic, u_basis)

            dpsi_dez = (1.0 - ez.pow(2)) / denom.pow(2)
            d_ez = d_psi * dpsi_dez

            d_eta = (d_ez * z_proj).sum(dim=(0, 1))
            d_zproj = d_ez * eta_b + d_eloc * (2.0 * z_proj / R)

            return d_ybase, d_ubasis, d_zproj, d_eta, d_lambda_oja, d_alpha_mod


    class LocalNonlinearSelfModifyingOp(nn.Module):
        """
        Complete Local Nonlinear Self-Modifying Weight Operator (`nn.Module`).
        Combines:
        1. Instantaneous Subspace-Local Weight Self-Modification (`FusedLocalSelfModWeightFunction`)
        2. Recurrent Nonlinear Delta-Oja Fast-Weight Plasticity across sequence steps
        """

        def __init__(self, config: LocalSelfModConfig):
            super().__init__()
            assert config.d_model % config.num_local_groups == 0
            self.config = config
            self.d_model = config.d_model
            self.G = config.num_local_groups
            self.D_g = config.d_model // config.num_local_groups
            self.R = config.local_rank

            # 1. Static local block-diagonal base weights W_base[G, D_g, D_g]
            self.w_base = nn.Parameter(
                torch.randn(self.G, self.D_g, self.D_g) * (self.D_g ** -0.5)
            )

            # 2. Local Hyper-Generators for U(x) and V(x) within each group g
            #    Each group g only sees its own local subspace x_g in R^{D_g} (strict locality!)
            self.u_gen_weight = nn.Parameter(
                torch.randn(self.G, self.D_g, self.D_g * self.R) * (self.D_g ** -0.5)
            )
            self.v_gen_weight = nn.Parameter(
                torch.randn(self.G, self.D_g, self.D_g * self.R) * (self.D_g ** -0.5)
            )

            # 3. Plasticity & Oja Homeostatic coefficients
            self.eta = nn.Parameter(
                torch.full((self.G, self.R), config.eta_plasticity_init)
            )
            self.lambda_oja = nn.Parameter(
                torch.full((self.G,), config.lambda_oja_init)
            )
            self.alpha_mod = nn.Parameter(
                torch.full((self.G, self.D_g), config.alpha_mod_init)
            )

            # 4. Recurrent Delta-Oja Fast-Weight projections (local per group)
            self.qkv_local = nn.Parameter(
                torch.randn(self.G, self.D_g, 3 * self.D_g) * (self.D_g ** -0.5)
            )
            self.plasticity_gates = nn.Linear(config.d_model, 2 * self.G, bias=True)
            self.out_proj = nn.Linear(2 * config.d_model, config.d_model, bias=False)
            self.out_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
            """
            Args:
                x: Input feature tensor [B, L, D]
            Returns:
                out: Locally self-modified feature representation [B, L, D]
                diagnostics: Dictionary of local plasticity and weight drift metrics
            """
            B, L, D = x.shape
            # Partition into G local subspaces: [B, L, G, D_g]
            x_loc = x.view(B, L, self.G, self.D_g)

            # Path A: Instantaneous Local Self-Modifying Weight Operator
            y_base = torch.einsum("blgi,gij->blgj", x_loc, self.w_base)
            u_basis = torch.einsum("blgi,gio->blgo", x_loc, self.u_gen_weight).view(
                B, L, self.G, self.D_g, self.R
            )
            v_basis = torch.einsum("blgi,gio->blgo", x_loc, self.v_gen_weight).view(
                B, L, self.G, self.D_g, self.R
            )
            # Local rank coordinates z = V(x)^T @ x_g: [B, L, G, R]
            z_proj = torch.einsum("blgdr,blgd->blgr", v_basis, x_loc) * (self.D_g ** -0.5)

            y_inst = FusedLocalSelfModWeightFunction.apply(
                y_base, u_basis, z_proj, self.eta, self.lambda_oja, self.alpha_mod
            ).reshape(B, L, D)

            # Path B: Recurrent Nonlinear Delta-Oja Fast-Weight Plasticity
            qkv = torch.einsum("blgi,gio->blgo", x_loc, self.qkv_local)
            q, k, v = qkv.chunk(3, dim=-1)  # Each: [B, L, G, D_g]

            gates = torch.sigmoid(self.plasticity_gates(x))  # [B, L, 2*G]
            eta_gate, gamma_forget = gates.chunk(2, dim=-1)  # Each: [B, L, G]

            w_fast = self.w_base.unsqueeze(0).expand(B, -1, -1, -1).clone()
            y_rec_list = []
            for t in range(L):
                q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
                k_norm = F.normalize(k_t, p=2, dim=-1, eps=1e-6)

                y_t = torch.einsum("bgij,bgj->bgi", w_fast, q_t)
                y_rec_list.append(y_t)

                wk_t = torch.einsum("bgij,bgj->bgi", w_fast, k_norm)
                v_hat_t = torch.tanh(wk_t)
                err_t = torch.tanh(v_t - v_hat_t)
                oja_term = self.config.recurrent_beta_oja * v_hat_t.pow(2) * wk_t
                delta_vec = err_t - oja_term

                outer_update = torch.einsum("bgi,bgj->bgij", delta_vec, k_norm)
                gam_t = gamma_forget[:, t].view(B, self.G, 1, 1)
                eta_t = (0.2 * eta_gate[:, t]).view(B, self.G, 1, 1)
                w_fast = gam_t * w_fast + eta_t * outer_update

            y_rec = torch.stack(y_rec_list, dim=1).reshape(B, L, D)
            out = self.out_norm(self.out_proj(torch.cat([y_inst, y_rec], dim=-1)))

            with torch.no_grad():
                drift = float((w_fast - self.w_base.unsqueeze(0)).norm().item() / math.sqrt(B * self.G * self.D_g * self.D_g))
                mean_oja = float((1.0 - self.lambda_oja.view(1, 1, -1, 1) * z_proj.pow(2).mean(dim=-1, keepdim=True)).mean().item())
            return out, {"rms_fast_weight_drift": drift, "mean_oja_homeostasis_scale": mean_oja}


# ==============================================================================
# 5. Tile-Accurate Triton Simulator, Gradcheck & HBM Bandwidth Profiler
# ==============================================================================

def simulate_triton_local_self_mod_kernel(
    y_base: np.ndarray,
    u_basis: np.ndarray,
    z_proj: np.ndarray,
    eta: np.ndarray,
    lambda_oja: np.ndarray,
    alpha_mod: np.ndarray,
    block_dg: int = 32,
    bytes_per_elem: int = 2,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simulates `fused_local_self_mod_weight_fwd_kernel` tile-by-tile across
    `num_rows = B * L * G` in SRAM and measures HBM bandwidth savings vs.
    materializing the explicit local weight modification matrix dW [B, L, G, D_g, D_g].
    """
    B, L, G, D_g, R = u_basis.shape
    num_rows = B * L * G

    yb_2d = y_base.reshape(num_rows, D_g)
    u_3d = u_basis.reshape(num_rows, D_g, R)
    z_2d = z_proj.reshape(num_rows, R)
    out_2d = np.zeros_like(yb_2d)

    for row_id in range(num_rows):
        g_idx = row_id % G
        lam_g = float(lambda_oja[g_idx])

        for d0 in range(0, D_g, block_dg):
            d1 = min(d0 + block_dg, D_g)
            yb_tile = yb_2d[row_id, d0:d1]
            alp_tile = alpha_mod[g_idx, d0:d1]

            y_plastic = np.zeros((d1 - d0,), dtype=np.float64)
            z_sq_sum = 0.0

            for r in range(R):
                z_r = float(z_2d[row_id, r])
                eta_r = float(eta[g_idx, r])
                ez = eta_r * z_r
                psi_r = ez / (1.0 + ez * ez)
                z_sq_sum += z_r * z_r
                y_plastic += u_3d[row_id, d0:d1, r] * psi_r

            e_loc = z_sq_sum / R
            oja_scale = 1.0 - lam_g * e_loc
            out_2d[row_id, d0:d1] = oja_scale * yb_tile + alp_tile * y_plastic

    out = out_2d.reshape(B, L, G, D_g)

    # HBM Bandwidth Comparison (Forward + Backward):
    # Unfused Eager PyTorch that materializes the local weight modification tensor
    # dW_loc [B, L, G, D_g, D_g] + intermediate psi, e_loc, oja_scale tensors reads/writes
    # ~6 * B * L * G * D_g^2 + 18 * B * L * G * D_g elements.
    # Fused Matrix-Free Triton Kernel NEVER materializes [B, L, G, D_g, D_g] in HBM!
    eager_hbm_bytes = (6 * B * L * G * D_g * D_g + 18 * B * L * G * D_g) * bytes_per_elem
    triton_hbm_bytes = (4 * B * L * G * D_g + 3 * B * L * G * D_g * R + 4 * B * L * G * R) * bytes_per_elem

    stats = {
        "eager_hbm_kb": eager_hbm_bytes / 1024.0,
        "triton_hbm_kb": triton_hbm_bytes / 1024.0,
        "hbm_saved_pct": (1.0 - triton_hbm_bytes / eager_hbm_bytes) * 100.0,
        "bw_speedup": eager_hbm_bytes / triton_hbm_bytes,
    }
    return out, stats


def run_local_self_modifying_verification_suite() -> None:
    """
    Executes central finite-difference gradcheck on all 6 inputs & plasticity
    parameters, verifies Triton tile simulator parity, and tests Recurrent
    Nonlinear Delta-Oja Fast-Weight adaptation.
    """
    rng = np.random.default_rng(2026)
    print("=" * 90)
    print("1. Analytical vs. Finite-Difference Gradcheck (FusedLocalSelfModifyingWeightOp)")
    print("=" * 90)

    B, L, G, D_g, R = 2, 3, 4, 5, 3
    y_base = rng.standard_normal((B, L, G, D_g)).astype(np.float64) * 0.6
    u_basis = rng.standard_normal((B, L, G, D_g, R)).astype(np.float64) * 0.4
    z_proj = rng.standard_normal((B, L, G, R)).astype(np.float64) * 0.7
    eta = rng.uniform(0.4, 1.2, size=(G, R)).astype(np.float64)
    lambda_oja = rng.uniform(0.05, 0.25, size=(G,)).astype(np.float64)
    alpha_mod = rng.uniform(0.3, 0.8, size=(G, D_g)).astype(np.float64)
    grad_out = rng.standard_normal((B, L, G, D_g)).astype(np.float64)

    _, cache = LocalSelfModifyingAnalyticalKernels.fused_local_self_mod_forward(
        y_base, u_basis, z_proj, eta, lambda_oja, alpha_mod
    )
    ana_grads = LocalSelfModifyingAnalyticalKernels.fused_local_self_mod_backward(
        grad_out, cache
    )

    eps = 1e-6
    targets = {
        "y_base": (y_base, ana_grads["d_ybase"]),
        "u_basis": (u_basis, ana_grads["d_ubasis"]),
        "z_proj": (z_proj, ana_grads["d_zproj"]),
        "eta": (eta, ana_grads["d_eta"]),
        "lambda_oja": (lambda_oja, ana_grads["d_lambda_oja"]),
        "alpha_mod": (alpha_mod, ana_grads["d_alpha_mod"]),
    }

    for name, (arr, g_ana) in targets.items():
        g_num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + eps
            out_p, _ = LocalSelfModifyingAnalyticalKernels.fused_local_self_mod_forward(
                y_base, u_basis, z_proj, eta, lambda_oja, alpha_mod
            )
            loss_p = np.sum(out_p * grad_out)

            arr[idx] = orig - eps
            out_m, _ = LocalSelfModifyingAnalyticalKernels.fused_local_self_mod_forward(
                y_base, u_basis, z_proj, eta, lambda_oja, alpha_mod
            )
            loss_m = np.sum(out_m * grad_out)
            arr[idx] = orig
            g_num[idx] = (loss_p - loss_m) / (2.0 * eps)

        rel_err = np.linalg.norm(g_ana - g_num) / (np.linalg.norm(g_ana) + np.linalg.norm(g_num) + 1e-15)
        print(f"   Gradcheck dL/d({name:10s}) | Relative L2 Error: {rel_err:.3e}  [PASS < 1e-8]")
        assert rel_err < 1e-8, f"Gradcheck failed for {name}: {rel_err}"

    print("\n" + "=" * 90)
    print("2. Triton Tile Simulator Parity, Fast-Weight Drift & HBM Bandwidth (B=2, L=64, D=256)")
    print("=" * 90)
    B_b, L_b, G_b, Dg_b, R_b = 2, 64, 8, 32, 4
    yb_b = rng.standard_normal((B_b, L_b, G_b, Dg_b)) * 0.5
    ub_b = rng.standard_normal((B_b, L_b, G_b, Dg_b, R_b)) * 0.3
    zp_b = rng.standard_normal((B_b, L_b, G_b, R_b)) * 0.5
    eta_b = np.full((G_b, R_b), 0.8)
    lam_b = np.full((G_b,), 0.15)
    alp_b = np.full((G_b, Dg_b), 0.5)

    out_ref, cache_b = LocalSelfModifyingAnalyticalKernels.fused_local_self_mod_forward(
        yb_b, ub_b, zp_b, eta_b, lam_b, alp_b
    )
    out_tri, hbm_stats = simulate_triton_local_self_mod_kernel(
        yb_b, ub_b, zp_b, eta_b, lam_b, alp_b, block_dg=16
    )
    max_tile_diff = float(np.max(np.abs(out_ref - out_tri)))

    # Test Recurrent Nonlinear Delta-Oja Fast-Weight Plasticity
    q_b = rng.standard_normal((B_b, L_b, G_b, Dg_b)) * 0.3
    k_b = rng.standard_normal((B_b, L_b, G_b, Dg_b)) * 0.3
    v_b = rng.standard_normal((B_b, L_b, G_b, Dg_b)) * 0.3
    w_init_b = rng.standard_normal((G_b, Dg_b, Dg_b)) * (Dg_b ** -0.5)
    eta_gate_b = np.full((B_b, L_b, G_b), 0.15)
    gam_gate_b = np.full((B_b, L_b, G_b), 0.95)
    y_rec, w_final, fw_diag = LocalSelfModifyingAnalyticalKernels.recurrent_nonlinear_delta_oja_fastweight_forward(
        q_b, k_b, v_b, w_init_b, eta_gate_b, gam_gate_b, beta_oja=0.10
    )

    print(f"   Instantaneous Self-Mod Output Shape  : {out_tri.shape} -> Flattened: {(B_b, L_b, G_b * Dg_b)}")
    print(f"   Recurrent Fast-Weight Output Shape   : {y_rec.shape} (Final W_t: {w_final.shape})")
    print(f"   Triton Tile Simulator Max Abs Diff   : {max_tile_diff:.2e}  [PASS == 0.00e+00]")
    print(f"   Mean Oja Homeostatic Scale Factor    : {float(np.mean(cache_b['oja_scale'])):.4f}")
    print(f"   Recurrent Fast-Weight RMS Drift      : {fw_diag['rms_fast_weight_drift']:.4f}")
    print(f"   Mean Nonlinear Delta Prediction Err  : {fw_diag['mean_nonlinear_delta_err']:.4f}")
    print(
        f"   HBM Memory Traffic (Fwd+Bwd)         : Eager={hbm_stats['eager_hbm_kb']:.1f} KB -> "
        f"Triton={hbm_stats['triton_hbm_kb']:.1f} KB "
        f"(Saved {hbm_stats['hbm_saved_pct']:.1f}% | {hbm_stats['bw_speedup']:.2f}x BW Speedup)"
    )
    assert max_tile_diff < 1e-14
    print("=" * 90)


if __name__ == "__main__":
    run_local_self_modifying_verification_suite()
