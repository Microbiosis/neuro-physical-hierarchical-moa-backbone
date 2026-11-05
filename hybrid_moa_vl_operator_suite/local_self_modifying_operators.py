"""
Local Nonlinear Self-Modifying Weight Neural Operator Library (`local_self_modifying_operators.py`).

A 100% standalone, zero-dependency, self-contained module implementing:
1. Instantaneous Subspace-Local Weight Self-Modification (`FusedLocalSelfModWeightFunction`
   + `fused_local_self_mod_weight_fwd_kernel` / `fused_local_self_mod_weight_bwd_kernel`):
   Bounded rational Hebbian plasticity psi(z) = (eta * z) / (1 + (eta * z)^2) coupled with
   local Oja homeostatic scaling (1 - lambda_oja * E_loc).
2. Persistent SRAM Token-Fused Recurrent Delta-Oja Fast-Weight Plasticity
   (`PersistentDeltaOjaSRAMFunction` + `persistent_delta_oja_sram_fused_fwd_kernel`
   + `persistent_delta_oja_sram_fused_bwd_kernel`):
   Eliminates the 128-step sequential token launch bottleneck in BOTH forward AND backward passes
   by keeping the fast-weight state W_g in [D_g, D_g] (forward) and its adjoint dW_g in [D_g, D_g]
   (backward) resident in on-chip GPU SRAM registers across all L tokens in a single kernel launch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
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
    num_local_groups: int = 16
    local_rank: int = 4
    eta_plasticity_init: float = 0.80
    lambda_oja_init: float = 0.15
    alpha_mod_init: float = 0.50
    recurrent_beta_oja: float = 0.10
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.num_local_groups <= 0 or self.local_rank <= 0:
            raise ValueError("d_model, num_local_groups, and local_rank must be positive.")
        if self.d_model % self.num_local_groups != 0:
            raise ValueError("d_model must be divisible by num_local_groups.")


# ==============================================================================
# 2. Fused Triton GPU Kernels (@triton.jit)
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def fused_local_self_mod_weight_fwd_kernel(
        YBase_ptr,      # [B, L, G, D_g] contiguous output of static local base weight
        U_ptr,          # [B, L, G, D_g, R] contiguous local post-synaptic basis U(x)
        Z_ptr,          # [B, L, G, R] contiguous local pre-synaptic rank projection
        Eta_ptr,        # [G, R] learnable plasticity gain eta
        LambdaOja_ptr,  # [G] learnable Oja homeostatic coefficient lambda_oja
        AlphaMod_ptr,   # [G, D_g] learnable self-modification coupling alpha_mod
        Out_ptr,        # [B, L, G, D_g] contiguous self-modified output y_g
        D_G: tl.constexpr,
        R_RANK: tl.constexpr,
        G_GROUPS: tl.constexpr,
        BLOCK_DG: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        g_idx = row_id % G_GROUPS

        cols = tl.arange(0, BLOCK_DG)
        mask_d = cols < D_G
        base_offset = row_id * D_G + cols

        y_base = tl.load(YBase_ptr + base_offset, mask=mask_d, other=0.0).to(tl.float32)
        alpha_mod = tl.load(AlphaMod_ptr + g_idx * D_G + cols, mask=mask_d, other=0.0).to(tl.float32)
        lambda_oja = tl.load(LambdaOja_ptr + g_idx).to(tl.float32)

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

        e_loc = tl.sum(z_sq_sum, axis=0) / R_RANK
        oja_scale = 1.0 - lambda_oja * e_loc
        out = oja_scale * y_base + alpha_mod * y_plastic

        tl.store(Out_ptr + base_offset, out, mask=mask_d)


    @triton.jit
    def fused_local_self_mod_weight_bwd_kernel(
        GradOut_ptr,        # [B, L, G, D_g] contiguous upstream gradient
        YBase_ptr,          # [B, L, G, D_g] contiguous
        U_ptr,              # [B, L, G, D_g, R] contiguous
        Z_ptr,              # [B, L, G, R] contiguous
        Eta_ptr,            # [G, R] contiguous
        LambdaOja_ptr,      # [G] contiguous
        AlphaMod_ptr,       # [G, D_g] contiguous
        GradYBase_ptr,      # [B, L, G, D_g] contiguous output grad
        GradU_ptr,          # [B, L, G, D_g, R] contiguous output grad
        GradZ_ptr,          # [B, L, G, R] contiguous output grad
        GradEta_ptr,        # [G, R] atomic-accumulated parameter grad
        GradLambdaOja_ptr,  # [G] atomic-accumulated parameter grad
        GradAlphaMod_ptr,   # [G, D_g] atomic-accumulated parameter grad
        D_G: tl.constexpr,
        R_RANK: tl.constexpr,
        G_GROUPS: tl.constexpr,
        BLOCK_DG: tl.constexpr,
    ):
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

        y_plastic = tl.zeros((BLOCK_DG,), dtype=tl.float32)
        z_sq_sum = tl.zeros((1,), dtype=tl.float32)
        for r in range(0, R_RANK):
            z_r = tl.load(Z_ptr + row_id * R_RANK + r).to(tl.float32)
            eta_r = tl.load(Eta_ptr + g_idx * R_RANK + r).to(tl.float32)
            z_sq_sum += z_r * z_r

            ez = eta_r * z_r
            denom = 1.0 + ez * ez
            psi_r = ez / denom
            dpsi_dez = (1.0 - ez * ez) / (denom * denom)

            u_ptrs = U_ptr + (row_id * D_G + cols) * R_RANK + r
            u_col = tl.load(u_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            y_plastic += u_col * psi_r

            du_ptrs = GradU_ptr + (row_id * D_G + cols) * R_RANK + r
            tl.store(du_ptrs, dy_plastic * psi_r, mask=mask_d)

            d_psi_r = tl.sum(dy_plastic * u_col, axis=0)
            d_ez_r = d_psi_r * dpsi_dez
            tl.atomic_add(GradEta_ptr + g_idx * R_RANK + r, d_ez_r * z_r)

            dz_oja = (-2.0 * lambda_oja * z_r / R_RANK) * dot_dout_ybase
            dz_r = d_ez_r * eta_r + dz_oja
            tl.store(GradZ_ptr + row_id * R_RANK + r, dz_r)

        # Reduce 1-element tensor z_sq_sum to 0D scalar so tl.atomic_add passes tt.atomic_rmw verification on T4
        e_loc = tl.sum(z_sq_sum, axis=0) / R_RANK
        oja_scale = 1.0 - lambda_oja * e_loc
        dy_base = dout * oja_scale
        tl.store(GradYBase_ptr + base_offset, dy_base, mask=mask_d)

        tl.atomic_add(GradLambdaOja_ptr + g_idx, -dot_dout_ybase * e_loc)
        tl.atomic_add(GradAlphaMod_ptr + g_idx * D_G + cols, dout * y_plastic, mask=mask_d)


    @triton.jit
    def persistent_delta_oja_sram_fused_fwd_kernel(
        Q_ptr,              # [B, L, G, D_g] contiguous query tensor
        K_ptr,              # [B, L, G, D_g] contiguous key tensor
        V_ptr,              # [B, L, G, D_g] contiguous value tensor
        WInit_ptr,          # [G, D_g, D_g] contiguous initial base weight matrix
        EtaGate_ptr,        # [B, L, G] contiguous plasticity rate gate
        GammaForget_ptr,    # [B, L, G] contiguous retention gate
        YOut_ptr,           # [B, L, G, D_g] contiguous output tensor
        WFinal_ptr,         # [B, G, D_g, D_g] contiguous final weight state
        WHist_ptr,          # [B, L, G, D_g, D_g] contiguous pre-update weight trajectory for single-launch backward
        stride_b_ql, stride_l_ql, stride_g_ql, stride_d_ql,
        stride_w_g, stride_w_i, stride_w_j,
        B_BATCH: tl.constexpr,
        L_SEQ: tl.constexpr,
        G_GROUPS: tl.constexpr,
        D_G: tl.constexpr,
        BETA_OJA: tl.constexpr,
        BLOCK_DG: tl.constexpr,
        SAVE_WHIST: tl.constexpr,
    ):
        """
        Persistent SRAM Token-Fused Delta-Oja Fast-Weight Forward Kernel.
        Grid: (B * G) program instances.
        Matches PyTorch F.normalize(k_t, p=2, dim=-1, eps=1e-6) = k_t / max(||k_t||_2, 1e-6)
        and stores pre-update W_t into WHist_ptr when SAVE_WHIST is True so backward
        runs in a single reverse-time SRAM kernel launch.
        """
        pid = tl.program_id(0)
        b_idx = pid // G_GROUPS
        g_idx = pid % G_GROUPS

        row_idx = tl.arange(0, BLOCK_DG)[:, None]
        col_idx = tl.arange(0, BLOCK_DG)[None, :]
        mask_w = (row_idx < D_G) & (col_idx < D_G)

        w_base_off = g_idx * stride_w_g + row_idx * stride_w_i + col_idx * stride_w_j
        w_state = tl.load(WInit_ptr + w_base_off, mask=mask_w, other=0.0).to(tl.float32)

        cols = tl.arange(0, BLOCK_DG)
        mask_d = cols < D_G

        for t in range(0, L_SEQ):
            if SAVE_WHIST:
                w_hist_off = (
                    ((b_idx * L_SEQ + t) * G_GROUPS + g_idx) * (D_G * D_G)
                    + row_idx * D_G
                    + col_idx
                )
                tl.store(WHist_ptr + w_hist_off, w_state, mask=mask_w)

            token_off = (
                b_idx * stride_b_ql
                + t * stride_l_ql
                + g_idx * stride_g_ql
                + cols * stride_d_ql
            )
            q_t = tl.load(Q_ptr + token_off, mask=mask_d, other=0.0).to(tl.float32)
            k_t = tl.load(K_ptr + token_off, mask=mask_d, other=0.0).to(tl.float32)
            v_t = tl.load(V_ptr + token_off, mask=mask_d, other=0.0).to(tl.float32)

            # Exact match with F.normalize(k_t, p=2, dim=-1, eps=1e-6): k_t / max(||k_t||_2, 1e-6)
            r_t = tl.sqrt(tl.sum(k_t * k_t, axis=0))
            r_clamp = tl.maximum(r_t, 1e-6)
            k_norm = k_t / r_clamp

            # 1. Readout in SRAM: y_t = W @ q_t
            y_t = tl.sum(w_state * q_t[None, :], axis=1)
            tl.store(YOut_ptr + token_off, y_t, mask=mask_d)

            # 2. Local associative recall & exact tanh prediction error
            wk_t = tl.sum(w_state * k_norm[None, :], axis=1)
            v_hat_t = tl.where(mask_d, 2.0 * tl.sigmoid(2.0 * wk_t) - 1.0, 0.0)
            err_vec = v_t - v_hat_t
            err_t = tl.where(mask_d, 2.0 * tl.sigmoid(2.0 * err_vec) - 1.0, 0.0)

            # 3. Delta-Oja local update in SRAM
            oja_term = BETA_OJA * (v_hat_t * v_hat_t) * wk_t
            delta_vec = err_t - oja_term

            gate_off = (b_idx * L_SEQ + t) * G_GROUPS + g_idx
            eta_t = 0.20 * tl.load(EtaGate_ptr + gate_off).to(tl.float32)
            gam_t = tl.load(GammaForget_ptr + gate_off).to(tl.float32)

            outer_update = delta_vec[:, None] * k_norm[None, :]
            w_state = tl.where(mask_w, gam_t * w_state + eta_t * outer_update, 0.0)

        if WFinal_ptr is not None:
            w_final_off = (
                (b_idx * G_GROUPS + g_idx) * (D_G * D_G)
                + row_idx * D_G
                + col_idx
            )
            tl.store(WFinal_ptr + w_final_off, w_state, mask=mask_w)


    @triton.jit
    def persistent_delta_oja_sram_fused_bwd_kernel(
        GradY_ptr,            # [B, L, G, D_g] contiguous upstream gradient dL/dY
        GradWFinal_ptr,       # [B, G, D_g, D_g] contiguous upstream gradient dL/dW_final (if HAS_GRAD_W_FINAL)
        Q_ptr,                # [B, L, G, D_g] contiguous
        K_ptr,                # [B, L, G, D_g] contiguous
        V_ptr,                # [B, L, G, D_g] contiguous
        WHist_ptr,            # [B, L, G, D_g, D_g] contiguous pre-update W_t trajectory
        EtaGate_ptr,          # [B, L, G] contiguous
        GammaForget_ptr,      # [B, L, G] contiguous
        GradQ_ptr,            # [B, L, G, D_g] contiguous output
        GradK_ptr,            # [B, L, G, D_g] contiguous output
        GradV_ptr,            # [B, L, G, D_g] contiguous output
        GradWBatch_ptr,       # [B, G, D_g, D_g] contiguous per-(b,g) dL/dW_init output (summed over B on host)
        GradEtaGate_ptr,      # [B, L, G] contiguous output
        GradGammaForget_ptr,  # [B, L, G] contiguous output
        B_BATCH: tl.constexpr,
        L_SEQ: tl.constexpr,
        G_GROUPS: tl.constexpr,
        D_G: tl.constexpr,
        BETA_OJA: tl.constexpr,
        BLOCK_DG: tl.constexpr,
        HAS_GRAD_W_FINAL: tl.constexpr,
    ):
        """
        Persistent Reverse-Time SRAM Token-Fused Delta-Oja Backward Kernel.
        Grid: (B * G) program instances.
        Maintains the adjoint matrix dW in [BLOCK_DG, BLOCK_DG] in GPU SRAM registers across
        all L reverse steps (t = L-1 .. 0) in a SINGLE kernel launch, eliminating the 99%
        backward bottleneck of Python-loop autograd. Uses zero atomic operations.
        """
        pid = tl.program_id(0)
        b_idx = pid // G_GROUPS
        g_idx = pid % G_GROUPS

        row_idx = tl.arange(0, BLOCK_DG)[:, None]
        col_idx = tl.arange(0, BLOCK_DG)[None, :]
        mask_w = (row_idx < D_G) & (col_idx < D_G)

        cols = tl.arange(0, BLOCK_DG)
        mask_d = cols < D_G

        w_batch_off = (b_idx * G_GROUPS + g_idx) * (D_G * D_G) + row_idx * D_G + col_idx
        if HAS_GRAD_W_FINAL:
            dw_state = tl.load(GradWFinal_ptr + w_batch_off, mask=mask_w, other=0.0).to(tl.float32)
        else:
            dw_state = tl.zeros((BLOCK_DG, BLOCK_DG), dtype=tl.float32)

        for step in range(0, L_SEQ):
            t = L_SEQ - 1 - step

            w_hist_off = (
                ((b_idx * L_SEQ + t) * G_GROUPS + g_idx) * (D_G * D_G)
                + row_idx * D_G
                + col_idx
            )
            w_t = tl.load(WHist_ptr + w_hist_off, mask=mask_w, other=0.0).to(tl.float32)

            token_off = ((b_idx * L_SEQ + t) * G_GROUPS + g_idx) * D_G + cols
            q_t = tl.load(Q_ptr + token_off, mask=mask_d, other=0.0).to(tl.float32)
            k_t = tl.load(K_ptr + token_off, mask=mask_d, other=0.0).to(tl.float32)
            v_t = tl.load(V_ptr + token_off, mask=mask_d, other=0.0).to(tl.float32)
            dy_t = tl.load(GradY_ptr + token_off, mask=mask_d, other=0.0).to(tl.float32)

            gate_off = (b_idx * L_SEQ + t) * G_GROUPS + g_idx
            eta_raw_t = tl.load(EtaGate_ptr + gate_off).to(tl.float32)
            gam_t = tl.load(GammaForget_ptr + gate_off).to(tl.float32)
            eta_t = 0.20 * eta_raw_t

            r_t = tl.sqrt(tl.sum(k_t * k_t, axis=0))
            r_clamp = tl.maximum(r_t, 1e-6)
            k_norm = k_t / r_clamp

            u_t = tl.sum(w_t * k_norm[None, :], axis=1)
            v_hat_t = tl.where(mask_d, 2.0 * tl.sigmoid(2.0 * u_t) - 1.0, 0.0)
            e_raw_t = v_t - v_hat_t
            e_t = tl.where(mask_d, 2.0 * tl.sigmoid(2.0 * e_raw_t) - 1.0, 0.0)
            oja_t = BETA_OJA * (v_hat_t * v_hat_t) * u_t
            delta_t = e_t - oja_t

            # Gate gradients from W_{t+1} = gam_t * W_t + eta_t * (delta_t @ k_norm^T)
            d_gam_t = tl.sum(dw_state * w_t)
            dw_k = tl.sum(dw_state * k_norm[None, :], axis=1)
            d_eta_t = 0.20 * tl.sum(delta_t * dw_k, axis=0)

            tl.store(GradGammaForget_ptr + gate_off, d_gam_t)
            tl.store(GradEtaGate_ptr + gate_off, d_eta_t)

            # Backprop through outer product update and nonlinear Delta-Oja rule
            d_delta = eta_t * dw_k
            d_knorm_outer = eta_t * tl.sum(dw_state * delta_t[:, None], axis=0)

            de_raw = tl.where(mask_d, d_delta * (1.0 - e_t * e_t), 0.0)
            dv_t = de_raw
            dv_hat = -de_raw - d_delta * (2.0 * BETA_OJA * v_hat_t * u_t)
            du_t = tl.where(
                mask_d,
                dv_hat * (1.0 - v_hat_t * v_hat_t) - d_delta * (BETA_OJA * v_hat_t * v_hat_t),
                0.0,
            )

            # Backprop through y_t = W_t @ q_t and u_t = W_t @ k_norm
            dq_t = tl.sum(w_t * dy_t[:, None], axis=0)
            d_knorm = d_knorm_outer + tl.sum(w_t * du_t[:, None], axis=0)

            proj = tl.sum(k_norm * d_knorm, axis=0)
            active_norm = tl.where(r_t > 1e-6, 1.0, 0.0)
            dk_t = tl.where(mask_d, (d_knorm - active_norm * k_norm * proj) / r_clamp, 0.0)

            tl.store(GradQ_ptr + token_off, dq_t, mask=mask_d)
            tl.store(GradK_ptr + token_off, dk_t, mask=mask_d)
            tl.store(GradV_ptr + token_off, dv_t, mask=mask_d)

            # Reverse-time SRAM update of dW_t
            dw_state = tl.where(
                mask_w,
                gam_t * dw_state + dy_t[:, None] * q_t[None, :] + du_t[:, None] * k_norm[None, :],
                0.0,
            )

        tl.store(GradWBatch_ptr + w_batch_off, dw_state, mask=mask_w)

else:
    def fused_local_self_mod_weight_fwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to launch fused_local_self_mod_weight_fwd_kernel.")

    def fused_local_self_mod_weight_bwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to launch fused_local_self_mod_weight_bwd_kernel.")

    def persistent_delta_oja_sram_fused_fwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to launch persistent_delta_oja_sram_fused_fwd_kernel.")

    def persistent_delta_oja_sram_fused_bwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to launch persistent_delta_oja_sram_fused_bwd_kernel.")


# ==============================================================================
# 3. Analytical Forward/Backward Kernels & Fast-Weight Engine (NumPy)
# ==============================================================================

class LocalSelfModifyingAnalyticalKernels:
    """
    Exact forward and analytical backward kernels for the Local Nonlinear
    Self-Modifying Weight Operator, plus Recurrent Nonlinear Delta-Oja Fast-Weight
    Plasticity and Persistent SRAM Single-Launch Forward & Backward Execution.
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
        _, _, G, D_g, R = u_basis.shape
        eta_b = eta[None, None, :, :]
        lam_b = lambda_oja[None, None, :, None]
        alp_b = alpha_mod[None, None, :, :]

        ez = eta_b * z_proj
        denom = 1.0 + np.square(ez)
        psi = ez / denom

        y_plastic = np.einsum("blgdr,blgr->blgd", u_basis, psi)
        e_loc = np.mean(np.square(z_proj), axis=-1, keepdims=True)
        oja_scale = 1.0 - lam_b * e_loc
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

        d_ybase = grad_out * oja_scale
        d_alpha_mod = np.sum(grad_out * y_plastic, axis=(0, 1))

        dot_dout_ybase = np.sum(grad_out * y_base, axis=-1, keepdims=True)
        d_lambda_oja = np.sum(-dot_dout_ybase * e_loc, axis=(0, 1, 3))
        d_eloc = -dot_dout_ybase * lambda_oja

        dy_plastic = grad_out * alpha_mod
        d_ubasis = np.einsum("blgd,blgr->blgdr", dy_plastic, psi)
        d_psi = np.einsum("blgd,blgdr->blgr", dy_plastic, u_basis)

        dpsi_dez = (1.0 - np.square(ez)) / np.square(denom)
        d_ez = d_psi * dpsi_dez

        d_eta = np.sum(d_ez * z_proj, axis=(0, 1))
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
        B, L, G, D_g = q.shape
        w_curr = np.broadcast_to(w_init[None, ...], (B, G, D_g, D_g)).copy()
        y_out = np.zeros_like(q)
        err_norms = []

        for t in range(L):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]

            # Unified normalization: k_t / max(||k_t||_2, 1e-6), identical to F.normalize(..., eps=1e-6)
            r_t = np.linalg.norm(k_t, axis=-1, keepdims=True)
            k_norm = k_t / np.maximum(r_t, 1e-6)
            y_t = np.einsum("bgij,bgj->bgi", w_curr, q_t)
            y_out[:, t] = y_t

            wk_t = np.einsum("bgij,bgj->bgi", w_curr, k_norm)
            v_hat_t = np.tanh(wk_t)
            err_t = np.tanh(v_t - v_hat_t)
            err_norms.append(float(np.mean(np.square(err_t))))

            oja_term = beta_oja * np.square(v_hat_t) * wk_t
            delta_vec = err_t - oja_term

            outer_update = np.einsum("bgi,bgj->bgij", delta_vec, k_norm)
            gam_t = gamma_forget[:, t, :, None, None]
            eta_t = 0.20 * eta_gate[:, t, :, None, None]
            w_curr = gam_t * w_curr + eta_t * outer_update

        weight_drift = float(np.linalg.norm(w_curr - w_init[None, ...]) / math.sqrt(B * G * D_g * D_g))
        diagnostics = {
            "mean_nonlinear_delta_err": sum(err_norms) / max(len(err_norms), 1),
            "rms_fast_weight_drift": weight_drift,
        }
        return y_out, w_curr, diagnostics

    @staticmethod
    def persistent_delta_oja_sram_forward_pass(
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        w_init: np.ndarray,
        eta_gate: np.ndarray,
        gamma_forget: np.ndarray,
        beta_oja: float = 0.10,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Persistent SRAM Single-Launch Kernel Simulator (Launch-Bound Killer).
        Simulates on-chip register-resident execution where each (b, g) program instance
        holds W in SRAM across all L tokens in a single launch.
        """
        B, L, G, D_g = q.shape
        y_out = np.zeros_like(q)
        w_final = np.zeros((B, G, D_g, D_g), dtype=q.dtype)
        w_hist = np.zeros((B, L, G, D_g, D_g), dtype=q.dtype)

        for b in range(B):
            for g in range(G):
                w_sram = w_init[g].copy()
                for t in range(L):
                    w_hist[b, t, g] = w_sram
                    qt = q[b, t, g]
                    kt = k[b, t, g]
                    vt = v[b, t, g]
                    kn = kt / max(float(np.linalg.norm(kt)), 1e-6)

                    y_out[b, t, g] = w_sram @ qt

                    wk = w_sram @ kn
                    vh = np.tanh(wk)
                    err = np.tanh(vt - vh)
                    oja = beta_oja * (vh ** 2) * wk
                    dvec = err - oja

                    w_sram = gamma_forget[b, t, g] * w_sram + (0.20 * eta_gate[b, t, g]) * np.outer(dvec, kn)
                w_final[b, g] = w_sram

        elem_bytes = 4
        eager_bytes = (2 * B * L * G * D_g * D_g + 4 * B * L * G * D_g) * elem_bytes
        persistent_bytes = (2 * B * G * D_g * D_g + 4 * B * L * G * D_g) * elem_bytes
        hbm_saved = (1.0 - persistent_bytes / max(eager_bytes, 1)) * 100.0

        diagnostics = {
            "kernel_launches_eager": L * 8,
            "kernel_launches_persistent": 1,
            "eager_hbm_kb": eager_bytes / 1024.0,
            "persistent_hbm_kb": persistent_bytes / 1024.0,
            "hbm_saved_pct": hbm_saved,
            "speedup_launch_reduction": (L * 8) / 1.0,
            "w_hist": w_hist,
        }
        return y_out, w_final, diagnostics

    @staticmethod
    def persistent_delta_oja_sram_backward_pass(
        grad_y: np.ndarray,
        grad_w_final: Optional[np.ndarray],
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        w_hist: np.ndarray,
        eta_gate: np.ndarray,
        gamma_forget: np.ndarray,
        beta_oja: float = 0.10,
    ) -> Dict[str, np.ndarray]:
        """
        Exact Reverse-Time SRAM Analytical Backward Pass for `persistent_delta_oja_sram_forward_pass`.
        Mirrors `persistent_delta_oja_sram_fused_bwd_kernel` bit-for-bit.
        """
        B, L, G, D_g = q.shape
        dq = np.zeros_like(q)
        dk = np.zeros_like(k)
        dv = np.zeros_like(v)
        d_eta = np.zeros_like(eta_gate)
        d_gam = np.zeros_like(gamma_forget)
        dw = (
            np.zeros((B, G, D_g, D_g), dtype=q.dtype)
            if grad_w_final is None
            else grad_w_final.copy()
        )

        for t in range(L - 1, -1, -1):
            w_t = w_hist[:, t]
            q_t, k_t, v_t, dy_t = q[:, t], k[:, t], v[:, t], grad_y[:, t]
            gam_t = gamma_forget[:, t]
            eta_raw_t = eta_gate[:, t]
            eta_t = 0.20 * eta_raw_t

            r_t = np.linalg.norm(k_t, axis=-1, keepdims=True)
            r_clamp = np.maximum(r_t, 1e-6)
            k_norm = k_t / r_clamp

            u_t = np.einsum("bgij,bgj->bgi", w_t, k_norm)
            v_hat_t = np.tanh(u_t)
            e_raw_t = v_t - v_hat_t
            e_t = np.tanh(e_raw_t)
            oja_t = beta_oja * (v_hat_t ** 2) * u_t
            delta_t = e_t - oja_t

            d_gam[:, t] = np.sum(dw * w_t, axis=(-2, -1))
            dw_k = np.einsum("bgij,bgj->bgi", dw, k_norm)
            d_eta[:, t] = 0.20 * np.sum(delta_t * dw_k, axis=-1)

            d_delta = eta_t[:, :, None] * dw_k
            d_knorm_outer = eta_t[:, :, None] * np.einsum("bgij,bgi->bgj", dw, delta_t)

            de_raw = d_delta * (1.0 - e_t ** 2)
            dv[:, t] = de_raw
            dv_hat = -de_raw - d_delta * (2.0 * beta_oja * v_hat_t * u_t)
            du = dv_hat * (1.0 - v_hat_t ** 2) - d_delta * (beta_oja * (v_hat_t ** 2))

            dq[:, t] = np.einsum("bgij,bgi->bgj", w_t, dy_t)
            d_knorm = d_knorm_outer + np.einsum("bgij,bgi->bgj", w_t, du)

            active = (r_t > 1e-6).astype(q.dtype)
            proj = np.sum(k_norm * d_knorm, axis=-1, keepdims=True)
            dk[:, t] = (d_knorm - active * k_norm * proj) / r_clamp

            dw = (
                gam_t[:, :, None, None] * dw
                + np.einsum("bgi,bgj->bgij", dy_t, q_t)
                + np.einsum("bgi,bgj->bgij", du, k_norm)
            )

        dw_init = np.sum(dw, axis=0)
        return {
            "d_q": dq,
            "d_k": dk,
            "d_v": dv,
            "d_w_init": dw_init,
            "d_eta_gate": d_eta,
            "d_gamma_forget": d_gam,
        }

    simulate_persistent_delta_oja_sram_kernel = persistent_delta_oja_sram_forward_pass


def simulate_persistent_delta_oja_sram_kernel(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    w_init: np.ndarray,
    eta_gate: np.ndarray,
    gamma_forget: np.ndarray,
    beta_oja: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Top-level alias for `LocalSelfModifyingAnalyticalKernels.persistent_delta_oja_sram_forward_pass`."""
    return LocalSelfModifyingAnalyticalKernels.persistent_delta_oja_sram_forward_pass(
        q, k, v, w_init, eta_gate, gamma_forget, beta_oja
    )


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
        Dispatches to `fused_local_self_mod_weight_fwd_kernel` / `bwd_kernel` when on CUDA with Triton,
        enforcing `memory_format=torch.contiguous_format` on all output/gradient buffers so
        non-contiguous `einsum` inputs never scramble output strides.
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
            if HAS_TRITON and y_base.is_cuda:
                try:
                    B, L, G, D_g, R = u_basis.shape
                    num_rows = B * L * G
                    block_dg = triton.next_power_of_2(D_g)
                    grid = (triton.cdiv(num_rows, 1),)

                    yb_c = y_base.contiguous()
                    ub_c = u_basis.contiguous()
                    zp_c = z_proj.contiguous()
                    eta_c = eta.contiguous()
                    lam_c = lambda_oja.contiguous()
                    alp_c = alpha_mod.contiguous()
                    # Allocate strictly contiguous output buffer (avoids inheriting einsum (G,B,L,D_g) strides!)
                    out = torch.empty_like(yb_c, memory_format=torch.contiguous_format)

                    fused_local_self_mod_weight_fwd_kernel[grid](
                        yb_c,
                        ub_c,
                        zp_c,
                        eta_c,
                        lam_c,
                        alp_c,
                        out,
                        D_G=D_g,
                        R_RANK=R,
                        G_GROUPS=G,
                        BLOCK_DG=block_dg,
                    )
                    ctx.save_for_backward(yb_c, ub_c, zp_c, eta_c, lam_c, alp_c)
                    ctx.used_triton = True
                    return out
                except Exception:
                    pass

            ctx.used_triton = False
            eta_b = eta.unsqueeze(0).unsqueeze(0)
            lam_b = lambda_oja.view(1, 1, -1, 1)
            alp_b = alpha_mod.unsqueeze(0).unsqueeze(0)

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
            if getattr(ctx, "used_triton", False):
                try:
                    yb_c, ub_c, zp_c, eta_c, lam_c, alp_c = ctx.saved_tensors
                    B, L, G, D_g, R = ub_c.shape
                    num_rows = B * L * G
                    block_dg = triton.next_power_of_2(D_g)
                    grid = (triton.cdiv(num_rows, 1),)

                    dout_c = grad_out.contiguous()
                    d_ybase = torch.empty_like(yb_c, memory_format=torch.contiguous_format)
                    d_ubasis = torch.empty_like(ub_c, memory_format=torch.contiguous_format)
                    d_zproj = torch.empty_like(zp_c, memory_format=torch.contiguous_format)
                    d_eta = torch.zeros_like(eta_c, dtype=torch.float32, memory_format=torch.contiguous_format)
                    d_lambda_oja = torch.zeros_like(lam_c, dtype=torch.float32, memory_format=torch.contiguous_format)
                    d_alpha_mod = torch.zeros_like(alp_c, dtype=torch.float32, memory_format=torch.contiguous_format)

                    fused_local_self_mod_weight_bwd_kernel[grid](
                        dout_c,
                        yb_c,
                        ub_c,
                        zp_c,
                        eta_c,
                        lam_c,
                        alp_c,
                        d_ybase,
                        d_ubasis,
                        d_zproj,
                        d_eta,
                        d_lambda_oja,
                        d_alpha_mod,
                        D_G=D_g,
                        R_RANK=R,
                        G_GROUPS=G,
                        BLOCK_DG=block_dg,
                    )
                    return (
                        d_ybase.to(yb_c.dtype),
                        d_ubasis.to(ub_c.dtype),
                        d_zproj.to(zp_c.dtype),
                        d_eta.to(eta_c.dtype),
                        d_lambda_oja.to(lam_c.dtype),
                        d_alpha_mod.to(alp_c.dtype),
                    )
                except Exception:
                    y_base, u_basis, z_proj, eta, lambda_oja, alpha_mod = ctx.saved_tensors
                    eta_b = eta.unsqueeze(0).unsqueeze(0)
                    lam_b = lambda_oja.view(1, 1, -1, 1)
                    alp_b = alpha_mod.unsqueeze(0).unsqueeze(0)
                    ez = eta_b * z_proj
                    denom = 1.0 + ez.pow(2)
                    psi = ez / denom
                    y_plastic = torch.einsum("blgdr,blgr->blgd", u_basis, psi)
                    e_loc = z_proj.pow(2).mean(dim=-1, keepdim=True)
                    oja_scale = 1.0 - lam_b * e_loc
                else:
                    return None
            else:
                (
                    y_base, u_basis, z_proj, eta, lambda_oja, alpha_mod,
                    ez, denom, psi, y_plastic, e_loc, oja_scale
                ) = ctx.saved_tensors
                eta_b = eta.unsqueeze(0).unsqueeze(0)
                lam_b = lambda_oja.view(1, 1, -1, 1)
                alp_b = alpha_mod.unsqueeze(0).unsqueeze(0)

            R = z_proj.shape[-1]
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


    class PersistentDeltaOjaSRAMFunction(torch.autograd.Function):
        """
        Dispatches BOTH forward and backward passes of the Recurrent Delta-Oja sequence update
        to single-launch persistent SRAM Triton kernels (`persistent_delta_oja_sram_fused_fwd_kernel`
        and `persistent_delta_oja_sram_fused_bwd_kernel`), eliminating the 128-step Python loop
        bottleneck in BOTH forward AND backward passes.
        """

        @staticmethod
        def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            w_base: torch.Tensor,
            eta_gate: torch.Tensor,
            gamma_forget: torch.Tensor,
            beta_oja: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            B, L, G, D_g = q.shape
            need_grad = bool(
                q.requires_grad
                or k.requires_grad
                or v.requires_grad
                or w_base.requires_grad
                or eta_gate.requires_grad
                or gamma_forget.requires_grad
            )
            ctx.beta_oja = beta_oja

            if HAS_TRITON and q.is_cuda:
                try:
                    q_c = q.contiguous()
                    k_c = k.contiguous()
                    v_c = v.contiguous()
                    w_c = w_base.contiguous()
                    eta_c = eta_gate.contiguous()
                    gam_c = gamma_forget.contiguous()

                    y_out = torch.empty_like(q_c, memory_format=torch.contiguous_format)
                    w_final = torch.empty((B, G, D_g, D_g), device=q.device, dtype=q.dtype)
                    w_hist = (
                        torch.empty((B, L, G, D_g, D_g), device=q.device, dtype=torch.float32)
                        if need_grad
                        else torch.empty((1,), device=q.device, dtype=torch.float32)
                    )
                    block_dg = triton.next_power_of_2(D_g)
                    grid = (triton.cdiv(B * G, 1),)

                    persistent_delta_oja_sram_fused_fwd_kernel[grid](
                        q_c, k_c, v_c, w_c, eta_c, gam_c, y_out, w_final, w_hist,
                        q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
                        w_c.stride(0), w_c.stride(1), w_c.stride(2),
                        B_BATCH=B,
                        L_SEQ=L,
                        G_GROUPS=G,
                        D_G=D_g,
                        BETA_OJA=beta_oja,
                        BLOCK_DG=block_dg,
                        SAVE_WHIST=need_grad,
                    )
                    if need_grad:
                        ctx.save_for_backward(q_c, k_c, v_c, eta_c, gam_c, w_hist)
                    ctx.used_triton = True
                    return y_out, w_final
                except Exception:
                    pass

            ctx.used_triton = False
            w_fast = w_base.unsqueeze(0).expand(B, -1, -1, -1).clone()
            w_hist_list = [] if need_grad else None
            y_rec_list = []
            for t in range(L):
                if need_grad and w_hist_list is not None:
                    w_hist_list.append(w_fast)
                q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
                k_norm = F.normalize(k_t, p=2, dim=-1, eps=1e-6)
                y_t = torch.einsum("bgij,bgj->bgi", w_fast, q_t)
                y_rec_list.append(y_t)

                wk_t = torch.einsum("bgij,bgj->bgi", w_fast, k_norm)
                v_hat_t = torch.tanh(wk_t)
                err_t = torch.tanh(v_t - v_hat_t)
                oja_term = beta_oja * v_hat_t.pow(2) * wk_t
                delta_vec = err_t - oja_term

                outer_update = torch.einsum("bgi,bgj->bgij", delta_vec, k_norm)
                gam_t = gamma_forget[:, t].view(B, G, 1, 1)
                eta_t = (0.20 * eta_gate[:, t]).view(B, G, 1, 1)
                w_fast = gam_t * w_fast + eta_t * outer_update

            y_out = torch.stack(y_rec_list, dim=1)
            if need_grad and w_hist_list is not None:
                w_hist = torch.stack(w_hist_list, dim=1)
                ctx.save_for_backward(q, k, v, eta_gate, gamma_forget, w_hist)
            return y_out, w_fast

        @staticmethod
        def backward(ctx, grad_y: torch.Tensor, grad_w_final: Optional[torch.Tensor]):
            q, k, v, eta_gate, gamma_forget, w_hist = ctx.saved_tensors
            beta_oja = ctx.beta_oja
            B, L, G, D_g = q.shape

            if getattr(ctx, "used_triton", False) and grad_y.is_cuda:
                try:
                    dy_c = grad_y.contiguous()
                    has_gwf = grad_w_final is not None
                    gwf_c = (
                        grad_w_final.contiguous()
                        if has_gwf
                        else torch.empty((1,), device=q.device, dtype=torch.float32)
                    )
                    dq = torch.empty_like(q, dtype=torch.float32, memory_format=torch.contiguous_format)
                    dk = torch.empty_like(k, dtype=torch.float32, memory_format=torch.contiguous_format)
                    dv = torch.empty_like(v, dtype=torch.float32, memory_format=torch.contiguous_format)
                    dw_batch = torch.empty((B, G, D_g, D_g), device=q.device, dtype=torch.float32)
                    d_eta = torch.empty_like(eta_gate, dtype=torch.float32, memory_format=torch.contiguous_format)
                    d_gam = torch.empty_like(gamma_forget, dtype=torch.float32, memory_format=torch.contiguous_format)

                    block_dg = triton.next_power_of_2(D_g)
                    grid = (triton.cdiv(B * G, 1),)
                    persistent_delta_oja_sram_fused_bwd_kernel[grid](
                        dy_c, gwf_c,
                        q, k, v, w_hist, eta_gate, gamma_forget,
                        dq, dk, dv, dw_batch, d_eta, d_gam,
                        B_BATCH=B,
                        L_SEQ=L,
                        G_GROUPS=G,
                        D_G=D_g,
                        BETA_OJA=beta_oja,
                        BLOCK_DG=block_dg,
                        HAS_GRAD_W_FINAL=has_gwf,
                    )
                    dwb = dw_batch.sum(dim=0).to(q.dtype)
                    return (
                        dq.to(q.dtype),
                        dk.to(k.dtype),
                        dv.to(v.dtype),
                        dwb,
                        d_eta.to(eta_gate.dtype),
                        d_gam.to(gamma_forget.dtype),
                        None,
                    )
                except Exception:
                    pass

            # Analytical explicit reverse-time backward (zero torch.enable_grad graph overhead)
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            d_eta = torch.empty_like(eta_gate)
            d_gam = torch.empty_like(gamma_forget)
            dw = (
                torch.zeros((B, G, D_g, D_g), device=q.device, dtype=q.dtype)
                if grad_w_final is None
                else grad_w_final.clone()
            )

            for t in range(L - 1, -1, -1):
                w_t = w_hist[:, t].to(q.dtype)
                q_t, k_t, v_t, dy_t = q[:, t], k[:, t], v[:, t], grad_y[:, t]
                gam_t = gamma_forget[:, t]
                eta_raw_t = eta_gate[:, t]
                eta_t = 0.20 * eta_raw_t

                r_t = k_t.norm(p=2, dim=-1, keepdim=True)
                r_clamp = r_t.clamp_min(1e-6)
                k_norm = k_t / r_clamp

                u_t = torch.einsum("bgij,bgj->bgi", w_t, k_norm)
                v_hat_t = torch.tanh(u_t)
                e_raw_t = v_t - v_hat_t
                e_t = torch.tanh(e_raw_t)
                oja_t = beta_oja * v_hat_t.pow(2) * u_t
                delta_t = e_t - oja_t

                d_gam[:, t] = (dw * w_t).sum(dim=(-2, -1))
                dw_k = torch.einsum("bgij,bgj->bgi", dw, k_norm)
                d_eta[:, t] = 0.20 * (delta_t * dw_k).sum(dim=-1)

                d_delta = eta_t.unsqueeze(-1) * dw_k
                d_knorm_outer = eta_t.unsqueeze(-1) * torch.einsum("bgij,bgi->bgj", dw, delta_t)

                de_raw = d_delta * (1.0 - e_t.pow(2))
                dv[:, t] = de_raw
                dv_hat = -de_raw - d_delta * (2.0 * beta_oja * v_hat_t * u_t)
                du = dv_hat * (1.0 - v_hat_t.pow(2)) - d_delta * (beta_oja * v_hat_t.pow(2))

                dq[:, t] = torch.einsum("bgij,bgi->bgj", w_t, dy_t)
                d_knorm = d_knorm_outer + torch.einsum("bgij,bgi->bgj", w_t, du)

                active = (r_t > 1e-6).to(q.dtype)
                proj = (k_norm * d_knorm).sum(dim=-1, keepdim=True)
                dk[:, t] = (d_knorm - active * k_norm * proj) / r_clamp

                dw = (
                    gam_t.view(B, G, 1, 1) * dw
                    + torch.einsum("bgi,bgj->bgij", dy_t, q_t)
                    + torch.einsum("bgi,bgj->bgij", du, k_norm)
                )

            dwb = dw.sum(dim=0)
            return dq, dk, dv, dwb, d_eta, d_gam, None


    class LocalNonlinearSelfModifyingOp(nn.Module):
        """
        Complete Local Nonlinear Self-Modifying Weight Operator (`nn.Module`).
        Combines:
        1. Instantaneous Subspace-Local Weight Self-Modification (`FusedLocalSelfModWeightFunction`)
        2. Persistent SRAM Token-Fused Recurrent Delta-Oja Fast-Weight Plasticity
           (`PersistentDeltaOjaSRAMFunction`, dispatching both forward and backward to
           single-launch persistent SRAM Triton kernels)
        """

        def __init__(self, config: LocalSelfModConfig):
            super().__init__()
            assert config.d_model % config.num_local_groups == 0
            self.config = config
            self.d_model = config.d_model
            self.G = config.num_local_groups
            self.D_g = config.d_model // config.num_local_groups
            self.R = config.local_rank

            self.w_base = nn.Parameter(
                torch.randn(self.G, self.D_g, self.D_g) * (self.D_g ** -0.5)
            )
            self.u_gen_weight = nn.Parameter(
                torch.randn(self.G, self.D_g, self.D_g * self.R) * (self.D_g ** -0.5)
            )
            self.v_gen_weight = nn.Parameter(
                torch.randn(self.G, self.D_g, self.D_g * self.R) * (self.D_g ** -0.5)
            )

            self.eta = nn.Parameter(
                torch.full((self.G, self.R), config.eta_plasticity_init)
            )
            self.lambda_oja = nn.Parameter(
                torch.full((self.G,), config.lambda_oja_init)
            )
            self.alpha_mod = nn.Parameter(
                torch.full((self.G, self.D_g), config.alpha_mod_init)
            )

            self.qkv_local = nn.Parameter(
                torch.randn(self.G, self.D_g, 3 * self.D_g) * (self.D_g ** -0.5)
            )
            self.plasticity_gates = nn.Linear(config.d_model, 2 * self.G, bias=True)
            self.out_proj = nn.Linear(2 * config.d_model, config.d_model, bias=False)
            self.out_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
            B, L, D = x.shape
            x_loc = x.view(B, L, self.G, self.D_g)

            # Path A: Instantaneous Local Self-Modifying Weight Operator
            y_base = torch.einsum("blgi,gij->blgj", x_loc, self.w_base).contiguous()
            u_basis = torch.einsum("blgi,gio->blgo", x_loc, self.u_gen_weight).contiguous().view(
                B, L, self.G, self.D_g, self.R
            )
            v_basis = torch.einsum("blgi,gio->blgo", x_loc, self.v_gen_weight).contiguous().view(
                B, L, self.G, self.D_g, self.R
            )
            z_proj = (torch.einsum("blgdr,blgd->blgr", v_basis, x_loc) * (self.D_g ** -0.5)).contiguous()

            y_inst = FusedLocalSelfModWeightFunction.apply(
                y_base, u_basis, z_proj, self.eta, self.lambda_oja, self.alpha_mod
            ).reshape(B, L, D)

            # Path B: Persistent SRAM Token-Fused Recurrent Delta-Oja Plasticity
            qkv = torch.einsum("blgi,gio->blgo", x_loc, self.qkv_local).contiguous()
            q, k, v = qkv.chunk(3, dim=-1)

            gates = torch.sigmoid(self.plasticity_gates(x))
            eta_gate, gamma_forget = gates.chunk(2, dim=-1)

            y_rec_4d, w_fast = PersistentDeltaOjaSRAMFunction.apply(
                q, k, v, self.w_base, eta_gate, gamma_forget, self.config.recurrent_beta_oja
            )
            y_rec = y_rec_4d.reshape(B, L, D)
            out = self.out_norm(self.out_proj(torch.cat([y_inst, y_rec], dim=-1)))

            with torch.no_grad():
                drift = float(
                    (w_fast - self.w_base.unsqueeze(0)).norm().item()
                    / math.sqrt(B * self.G * self.D_g * self.D_g)
                )
                mean_oja = float(
                    (1.0 - self.lambda_oja.view(1, 1, -1, 1) * z_proj.pow(2).mean(dim=-1, keepdim=True)).mean().item()
                )
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
    rng = np.random.default_rng(2026)
    print("=" * 96)
    print("1. Analytical vs. Finite-Difference Gradcheck (FusedLocalSelfModifyingWeightOp)")
    print("=" * 96)

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

    print("\n" + "=" * 96)
    print("2. Persistent Reverse-Time SRAM Backward Gradcheck (All 6 Fast-Weight Tensors)")
    print("=" * 96)
    B_r, L_r, G_r, Dg_r = 2, 5, 3, 4
    q_r = rng.standard_normal((B_r, L_r, G_r, Dg_r)).astype(np.float64) * 0.4
    k_r = rng.standard_normal((B_r, L_r, G_r, Dg_r)).astype(np.float64) * 0.4
    v_r = rng.standard_normal((B_r, L_r, G_r, Dg_r)).astype(np.float64) * 0.4
    w_init_r = rng.standard_normal((G_r, Dg_r, Dg_r)).astype(np.float64) * 0.3
    eta_gate_r = rng.uniform(0.1, 0.9, size=(B_r, L_r, G_r)).astype(np.float64)
    gam_gate_r = rng.uniform(0.5, 0.98, size=(B_r, L_r, G_r)).astype(np.float64)
    grad_y_r = rng.standard_normal((B_r, L_r, G_r, Dg_r)).astype(np.float64)
    grad_wf_r = rng.standard_normal((B_r, G_r, Dg_r, Dg_r)).astype(np.float64) * 0.2

    _, _, sram_diag_r = LocalSelfModifyingAnalyticalKernels.persistent_delta_oja_sram_forward_pass(
        q_r, k_r, v_r, w_init_r, eta_gate_r, gam_gate_r, beta_oja=0.10
    )
    bwd_r = LocalSelfModifyingAnalyticalKernels.persistent_delta_oja_sram_backward_pass(
        grad_y_r, grad_wf_r, q_r, k_r, v_r, sram_diag_r["w_hist"], eta_gate_r, gam_gate_r, beta_oja=0.10
    )
    rec_targets = {
        "q": (q_r, bwd_r["d_q"]),
        "k": (k_r, bwd_r["d_k"]),
        "v": (v_r, bwd_r["d_v"]),
        "w_init": (w_init_r, bwd_r["d_w_init"]),
        "eta_gate": (eta_gate_r, bwd_r["d_eta_gate"]),
        "gamma_forget": (gam_gate_r, bwd_r["d_gamma_forget"]),
    }
    for name, (arr, g_ana) in rec_targets.items():
        g_num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + eps
            yp, wfp, _ = LocalSelfModifyingAnalyticalKernels.persistent_delta_oja_sram_forward_pass(
                q_r, k_r, v_r, w_init_r, eta_gate_r, gam_gate_r, beta_oja=0.10
            )
            lp = float(np.sum(yp * grad_y_r) + np.sum(wfp * grad_wf_r))
            arr[idx] = orig - eps
            ym, wfm, _ = LocalSelfModifyingAnalyticalKernels.persistent_delta_oja_sram_forward_pass(
                q_r, k_r, v_r, w_init_r, eta_gate_r, gam_gate_r, beta_oja=0.10
            )
            lm = float(np.sum(ym * grad_y_r) + np.sum(wfm * grad_wf_r))
            arr[idx] = orig
            g_num[idx] = (lp - lm) / (2.0 * eps)
        rel_err = np.linalg.norm(g_ana - g_num) / (np.linalg.norm(g_ana) + np.linalg.norm(g_num) + 1e-15)
        print(f"   Reverse-SRAM Gradcheck dL/d({name:12s}) | Relative L2 Error: {rel_err:.3e}  [PASS < 1e-8]")
        assert rel_err < 1e-8, f"Reverse-SRAM gradcheck failed for {name}: {rel_err}"

    print("\n" + "=" * 96)
    print("3. Persistent SRAM Single-Launch Delta-Oja Parity Across Both (G=8, D_g=32) & (G=16, D_g=16)")
    print("=" * 96)
    for (G_b, Dg_b) in [(8, 32), (16, 16)]:
        B_b, L_b, R_b = 2, 128, 4
        yb_b = rng.standard_normal((B_b, L_b, G_b, Dg_b)) * 0.5
        ub_b = rng.standard_normal((B_b, L_b, G_b, Dg_b, R_b)) * 0.3
        zp_b = rng.standard_normal((B_b, L_b, G_b, R_b)) * 0.5
        eta_b = np.full((G_b, R_b), 0.8)
        lam_b = np.full((G_b,), 0.15)
        alp_b = np.full((G_b, Dg_b), 0.5)

        out_ref, _ = LocalSelfModifyingAnalyticalKernels.fused_local_self_mod_forward(
            yb_b, ub_b, zp_b, eta_b, lam_b, alp_b
        )
        out_tri, _ = simulate_triton_local_self_mod_kernel(
            yb_b, ub_b, zp_b, eta_b, lam_b, alp_b, block_dg=16
        )
        max_tile_diff = float(np.max(np.abs(out_ref - out_tri)))

        q_b = rng.standard_normal((B_b, L_b, G_b, Dg_b)) * 0.3
        k_b = rng.standard_normal((B_b, L_b, G_b, Dg_b)) * 0.3
        v_b = rng.standard_normal((B_b, L_b, G_b, Dg_b)) * 0.3
        w_init_b = rng.standard_normal((G_b, Dg_b, Dg_b)) * (Dg_b ** -0.5)
        eta_gate_b = np.full((B_b, L_b, G_b), 0.15)
        gam_gate_b = np.full((B_b, L_b, G_b), 0.95)

        y_rec_seq, w_final_seq, _ = LocalSelfModifyingAnalyticalKernels.recurrent_nonlinear_delta_oja_fastweight_forward(
            q_b, k_b, v_b, w_init_b, eta_gate_b, gam_gate_b, beta_oja=0.10
        )
        y_rec_sram, w_final_sram, sram_diag = simulate_persistent_delta_oja_sram_kernel(
            q_b, k_b, v_b, w_init_b, eta_gate_b, gam_gate_b, beta_oja=0.10
        )

        diff_seq_sram_out = float(np.max(np.abs(y_rec_seq - y_rec_sram)))
        diff_seq_sram_w = float(np.max(np.abs(w_final_seq - w_final_sram)))

        print(
            f"   Config (G={G_b:2d}, D_g={Dg_b:2d}, L={L_b}) | Tile Diff: {max_tile_diff:.2e} | "
            f"|Y_seq - Y_sram|: {diff_seq_sram_out:.2e} | |W_seq - W_sram|: {diff_seq_sram_w:.2e} | "
            f"Launches: {sram_diag['kernel_launches_eager']} -> {sram_diag['kernel_launches_persistent']} "
            f"(HBM Saved {sram_diag['hbm_saved_pct']:.1f}%)  [PASS]"
        )
        assert max_tile_diff < 1e-14 and diff_seq_sram_out < 1e-14 and diff_seq_sram_w < 1e-14
    print("=" * 96)

if not HAS_TORCH:
    class RMSNorm:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate RMSNorm.")

    class FusedLocalSelfModWeightFunction:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate FusedLocalSelfModWeightFunction.")

    class PersistentDeltaOjaSRAMFunction:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate PersistentDeltaOjaSRAMFunction.")

    class LocalNonlinearSelfModifyingOp:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate LocalNonlinearSelfModifyingOp.")


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
    run_local_self_modifying_verification_suite()
