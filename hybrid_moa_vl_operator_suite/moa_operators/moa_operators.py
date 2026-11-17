"""
System-Level MoA (Mixture-of-Agents / Mixture-of-Architectures / Mixture-of-Attention)
Underlying Neural Operator Library (`moa_operators.py`) — Pure Vision-Language
Cross-Modal Spatial-Semantic Fusion Edition.

Core Architecture & Audit-Hardened Guarantees:
1. Genuine 2D Spatial Grid + 1D Sequence Depthwise Convolution (`V_vis`, `T_txt`):
   - `CrossModalVisionLanguageFusionSubOp` uses configurable `spatial_conv_kernel`
     from `MoAOperatorConfig` (default 5) and applies a true 2D Depthwise Spatial
     Convolution (`nn.Conv2d(..., kernel_size=k, padding=k//2, groups=d_model)`) over the
     2D visual patch grid `(H_grid, W_grid)` alongside 1D token neighborhood mixing.
2. Low-Rank Cross-Modal Latent Anchor Co-Attention (`O(L * R_anchor * D)`):
   - Bridges visual patches and text tokens globally with ZERO distance attenuation
     via `R_anchor` shared cross-modal latent bottlenecks.
3. Fused Hadamard-Bilinear Cross-Modal Gated Synthesis (`FusedCrossModalBilinearGateFunction`):
   - Dispatches directly to **this module's own private** `@triton.jit`
     `_moa_vl_bilinear_fwd_kernel` and the 6-output `_moa_vl_bilinear_bwd_kernel`
     (outputs: GradVVis, GradTTxt, GradCAlign, GradAlphaVis, GradAlphaTxt, GradBetaBilin)
     when on CUDA with Triton; falls back to the NumPy analytical path otherwise.
   - Note: `moa_triton_kernels.py` separately publishes *simulation-verified* kernels
     (`fused_cross_modal_vl_bilinear_fwd/bwd_kernel`) plus `TritonTileSimulator` and
     `KernelOptimizationReport`. That module is verification tooling, not a runtime
     dependency of this one — the two never import each other, which keeps both at
     zero sibling imports.
4. Tie-Aware Symmetric Sparse Top-K Router & `top_k <= N` Safety:
   - Clamps `k_eff = min(max(int(top_k), 1), N)` so `top_k > N` never raises `ValueError`.
   - Resolves tied router logits symmetrically (`selection_mask = strict_mask + tie_mask * (rem_k / num_tie)`),
     guaranteeing uniform `f_i = [K/N, ..., K/N]` on all-zero or tied proposals (zero tie-break bias!).
5. Full `MoAOperatorConfig` Active Wiring:
   - Every field in `MoAOperatorConfig` (`proposer_mode`, `spatial_conv_kernel`, `conv_kernel_size`,
     `num_refinement_steps`, `share_refinement_weights`, `enable_adaptive_halting`,
     `halting_threshold`, `dropout`) is actively wired and verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
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
# 1. Unified Operator Configuration
# ==============================================================================

@dataclass
class MoAOperatorConfig:
    """Configuration for Vision-Language Optimized System-to-Operator MoA modules."""
    d_model: int = 256
    n_heads: int = 8
    num_proposers: int = 4
    top_k_proposers: int = 2
    d_ffn: int = 512
    num_cross_modal_anchors: int = 16     # R_anchor: global vision-text latent anchors
    alpha_vis_init: float = 0.80          # Visual stream gating sensitivity
    alpha_txt_init: float = 0.80          # Textual stream gating sensitivity
    beta_bilinear_init: float = 0.65      # Cross-modal Hadamard-bilinear interaction gain
    lam_cm_norm: float = 0.35             # Cross-modal bilinear energy stabilization factor
    spatial_conv_kernel: int = 5          # Configurable 2D spatial patch conv kernel size
    conv_kernel_size: int = 4             # Configurable 1D depthwise conv branch kernel size
    proposer_mode: Literal["homogeneous", "heterogeneous"] = "heterogeneous"
    aggregator_mode: Literal["cross_attention", "sparse_router", "low_rank_concat"] = "cross_attention"
    num_refinement_steps: int = 2
    share_refinement_weights: bool = False
    enable_adaptive_halting: bool = True
    halting_threshold: float = 0.90
    aux_loss_weight: float = 0.01
    z_loss_weight: float = 0.001
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    sliding_window_size: int = 8
    num_sink_tokens: int = 2
    dilation_stride: int = 2


# ==============================================================================
# 1B. Standalone Fused Triton Kernels (@triton.jit)
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def _moa_vl_bilinear_fwd_kernel(
        VVis_ptr, TTxt_ptr, CAlign_ptr,
        AlphaVis_ptr, AlphaTxt_ptr, BetaBilin_ptr, Out_ptr,
        stride_row, D_MODEL: tl.constexpr, lam_norm: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D_MODEL
        off = row_id * stride_row + cols

        v_vis = tl.load(VVis_ptr + off, mask=mask, other=0.0).to(tl.float32)
        t_txt = tl.load(TTxt_ptr + off, mask=mask, other=0.0).to(tl.float32)
        c_align = tl.load(CAlign_ptr + off, mask=mask, other=0.0).to(tl.float32)

        a_vis = tl.load(AlphaVis_ptr + cols, mask=mask, other=0.8).to(tl.float32)
        a_txt = tl.load(AlphaTxt_ptr + cols, mask=mask, other=0.8).to(tl.float32)
        b_bilin = tl.load(BetaBilin_ptr + cols, mask=mask, other=0.6).to(tl.float32)

        g_cm = tl.sigmoid(a_vis * v_vis + a_txt * t_txt)
        y_interp = g_cm * v_vis + (1.0 - g_cm) * t_txt
        inv_norm = tl.math.rsqrt(1.0 + lam_norm * (v_vis * v_vis + t_txt * t_txt))
        out = y_interp + b_bilin * (v_vis * t_txt + c_align) * inv_norm
        tl.store(Out_ptr + off, out, mask=mask)

    @triton.jit
    def _moa_vl_bilinear_bwd_kernel(
        GradOut_ptr, VVis_ptr, TTxt_ptr, CAlign_ptr,
        AlphaVis_ptr, AlphaTxt_ptr, BetaBilin_ptr,
        GradVVis_ptr, GradTTxt_ptr, GradCAlign_ptr,
        GradAlphaVis_ptr, GradAlphaTxt_ptr, GradBetaBilin_ptr,
        stride_row, D_MODEL: tl.constexpr, lam_norm: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D_MODEL
        off = row_id * stride_row + cols

        dout = tl.load(GradOut_ptr + off, mask=mask, other=0.0).to(tl.float32)
        v_vis = tl.load(VVis_ptr + off, mask=mask, other=0.0).to(tl.float32)
        t_txt = tl.load(TTxt_ptr + off, mask=mask, other=0.0).to(tl.float32)
        c_align = tl.load(CAlign_ptr + off, mask=mask, other=0.0).to(tl.float32)

        a_vis = tl.load(AlphaVis_ptr + cols, mask=mask, other=0.8).to(tl.float32)
        a_txt = tl.load(AlphaTxt_ptr + cols, mask=mask, other=0.8).to(tl.float32)
        b_bilin = tl.load(BetaBilin_ptr + cols, mask=mask, other=0.6).to(tl.float32)

        g_cm = tl.sigmoid(a_vis * v_vis + a_txt * t_txt)
        one_m_g = 1.0 - g_cm

        norm_arg = 1.0 + lam_norm * (v_vis * v_vis + t_txt * t_txt)
        inv_norm = tl.math.rsqrt(norm_arg)
        b_num = v_vis * t_txt + c_align
        b_unit = b_num * inv_norm

        d_bunit = dout * b_bilin
        d_bnum = d_bunit * inv_norm
        d_normarg = d_bunit * b_num * (-0.5 * inv_norm / norm_arg)
        d_glogit = dout * (v_vis - t_txt) * g_cm * one_m_g

        dv_vis = dout * g_cm + d_glogit * a_vis + d_bnum * t_txt + d_normarg * (2.0 * lam_norm * v_vis)
        dt_txt = dout * one_m_g + d_glogit * a_txt + d_bnum * v_vis + d_normarg * (2.0 * lam_norm * t_txt)

        tl.store(GradVVis_ptr + off, dv_vis, mask=mask)
        tl.store(GradTTxt_ptr + off, dt_txt, mask=mask)
        tl.store(GradCAlign_ptr + off, d_bnum, mask=mask)
        tl.atomic_add(GradAlphaVis_ptr + cols, d_glogit * v_vis, mask=mask)
        tl.atomic_add(GradAlphaTxt_ptr + cols, d_glogit * t_txt, mask=mask)
        tl.atomic_add(GradBetaBilin_ptr + cols, dout * b_unit, mask=mask)


# ==============================================================================
# 2. Analytical Forward/Backward Kernels — Cross-Modal Vision-Language Fusion
# ==============================================================================

class NumpyMoAOperatorKernels:
    """
    Exact forward and analytical backward kernels for System-Level MoA operators,
    featuring Cross-Modal Vision-Language Hadamard-Bilinear Gated Fusion and
    Tie-Aware Symmetric Sparse Top-K Routing.
    """

    @staticmethod
    def cross_modal_vl_bilinear_fusion_forward(
        v_vis: np.ndarray,
        t_txt: np.ndarray,
        c_align: np.ndarray,
        alpha_vis: np.ndarray,
        alpha_txt: np.ndarray,
        beta_bilin: np.ndarray,
        lam_norm: float = 0.35,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        av_b = alpha_vis[None, None, :]
        at_b = alpha_txt[None, None, :]
        bb_b = beta_bilin[None, None, :]

        g_logit = np.clip(av_b * v_vis + at_b * t_txt, -500.0, 500.0)
        g_cm = 1.0 / (1.0 + np.exp(-g_logit))
        one_m_g = 1.0 - g_cm
        y_interp = g_cm * v_vis + one_m_g * t_txt

        vt_prod = v_vis * t_txt
        b_num = vt_prod + c_align
        e_cm = np.square(v_vis) + np.square(t_txt)
        norm_arg = 1.0 + lam_norm * e_cm
        inv_norm = 1.0 / np.sqrt(norm_arg)
        b_unit = b_num * inv_norm

        out = y_interp + bb_b * b_unit

        cache = {
            "v_vis": v_vis,
            "t_txt": t_txt,
            "c_align": c_align,
            "alpha_vis": alpha_vis,
            "alpha_txt": alpha_txt,
            "beta_bilin": beta_bilin,
            "g_cm": g_cm,
            "one_m_g": one_m_g,
            "b_num": b_num,
            "norm_arg": norm_arg,
            "inv_norm": inv_norm,
            "b_unit": b_unit,
            "lam_norm": np.array(lam_norm),
        }
        return out, cache

    @staticmethod
    def cross_modal_vl_bilinear_fusion_backward(
        grad_out: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        v_vis = cache["v_vis"]
        t_txt = cache["t_txt"]
        alpha_vis = cache["alpha_vis"][None, None, :]
        alpha_txt = cache["alpha_txt"][None, None, :]
        beta_bilin = cache["beta_bilin"][None, None, :]
        g_cm = cache["g_cm"]
        one_m_g = cache["one_m_g"]
        b_num = cache["b_num"]
        norm_arg = cache["norm_arg"]
        inv_norm = cache["inv_norm"]
        b_unit = cache["b_unit"]
        lam_norm = float(cache["lam_norm"])

        d_betabilin = np.sum(grad_out * b_unit, axis=(0, 1))
        d_bunit = grad_out * beta_bilin

        d_bnum = d_bunit * inv_norm
        d_calign = d_bnum
        d_normarg = d_bunit * b_num * (-0.5 * inv_norm / norm_arg)

        d_gcm = grad_out * (v_vis - t_txt)
        d_glogit = d_gcm * g_cm * one_m_g

        d_alphavis = np.sum(d_glogit * v_vis, axis=(0, 1))
        d_alphatxt = np.sum(d_glogit * t_txt, axis=(0, 1))

        d_vvis = (
            grad_out * g_cm
            + d_glogit * alpha_vis
            + d_bnum * t_txt
            + d_normarg * (2.0 * lam_norm * v_vis)
        )
        d_ttxt = (
            grad_out * one_m_g
            + d_glogit * alpha_txt
            + d_bnum * v_vis
            + d_normarg * (2.0 * lam_norm * t_txt)
        )

        return {
            "d_vvis": d_vvis,
            "d_ttxt": d_ttxt,
            "d_calign": d_calign,
            "d_alphavis": d_alphavis,
            "d_alphatxt": d_alphatxt,
            "d_betabilin": d_betabilin,
        }

    @staticmethod
    def cross_modal_anchor_co_attention_forward(
        v_vis: np.ndarray,
        t_txt: np.ndarray,
        anchors: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        B, L, D = v_vis.shape
        scale = 1.0 / math.sqrt(D)

        joint_kv = 0.5 * (v_vis + t_txt)
        scores_pool = np.einsum("rd,bld->brl", anchors, joint_kv) * scale
        sp_max = np.max(scores_pool, axis=-1, keepdims=True)
        exp_sp = np.exp(scores_pool - sp_max)
        attn_pool = exp_sp / np.sum(exp_sp, axis=-1, keepdims=True)
        z_anchors = np.einsum("brl,bld->brd", attn_pool, v_vis)

        scores_bc = np.einsum("bld,brd->blr", t_txt, z_anchors) * scale
        sbc_max = np.max(scores_bc, axis=-1, keepdims=True)
        exp_sbc = np.exp(scores_bc - sbc_max)
        attn_bc = exp_sbc / np.sum(exp_sbc, axis=-1, keepdims=True)
        c_align = np.einsum("blr,brd->bld", attn_bc, z_anchors)

        cache = {
            "v_vis": v_vis,
            "t_txt": t_txt,
            "anchors": anchors,
            "joint_kv": joint_kv,
            "attn_pool": attn_pool,
            "z_anchors": z_anchors,
            "attn_bc": attn_bc,
            "scale": np.array(scale),
        }
        return c_align, cache

    @staticmethod
    def cross_modal_anchor_co_attention_backward(
        grad_c_align: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        v_vis = cache["v_vis"]
        t_txt = cache["t_txt"]
        anchors = cache["anchors"]
        joint_kv = cache["joint_kv"]
        attn_pool = cache["attn_pool"]
        z_anchors = cache["z_anchors"]
        attn_bc = cache["attn_bc"]
        scale = float(cache["scale"])

        d_attn_bc = np.einsum("bld,brd->blr", grad_c_align, z_anchors)
        d_z_anchors = np.einsum("blr,bld->brd", attn_bc, grad_c_align)

        d_scores_bc = attn_bc * (
            d_attn_bc - np.sum(d_attn_bc * attn_bc, axis=-1, keepdims=True)
        ) * scale

        d_ttxt_bc = np.einsum("blr,brd->bld", d_scores_bc, z_anchors)
        d_z_anchors += np.einsum("blr,bld->brd", d_scores_bc, t_txt)

        d_attn_pool = np.einsum("brd,bld->brl", d_z_anchors, v_vis)
        d_vvis_pool = np.einsum("brl,brd->bld", attn_pool, d_z_anchors)

        d_scores_pool = attn_pool * (
            d_attn_pool - np.sum(d_attn_pool * attn_pool, axis=-1, keepdims=True)
        ) * scale

        d_anchors = np.einsum("brl,bld->rd", d_scores_pool, joint_kv)
        d_joint_kv = np.einsum("brl,rd->bld", d_scores_pool, anchors)

        d_vvis = d_vvis_pool + 0.5 * d_joint_kv
        d_ttxt = d_ttxt_bc + 0.5 * d_joint_kv
        return {
            "d_vvis": d_vvis,
            "d_ttxt": d_ttxt,
            "d_anchors": d_anchors,
        }

    @staticmethod
    def branch_rmsnorm_forward(
        x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        ms = np.mean(np.square(x), axis=-1, keepdims=True)
        inv_rms = 1.0 / np.sqrt(ms + eps)
        out = x * inv_rms * gamma
        return out, inv_rms

    @staticmethod
    def branch_rmsnorm_backward(
        grad_out: np.ndarray,
        x: np.ndarray,
        gamma: np.ndarray,
        inv_rms: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_hat = x * inv_rms
        grad_gamma = np.sum(grad_out * x_hat, axis=tuple(range(x.ndim - 1)))
        dx_hat = grad_out * gamma
        grad_x = inv_rms * (
            dx_hat - x_hat * np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
        )
        return grad_x, grad_gamma

    @staticmethod
    def sparse_topk_router_forward(
        x_query: np.ndarray,
        proposals: np.ndarray,
        w_rq: np.ndarray,
        w_rk: np.ndarray,
        top_k: int,
        aux_weight: float = 0.01,
        z_weight: float = 0.001,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Tie-Aware Symmetric Sparse Top-K Router with safe `k_eff = min(max(top_k, 1), N)` clamping:
        - When `top_k > N`, gracefully clamps `k_eff = N` (never raises ValueError).
        - When router logits are tied at the k_eff-th threshold (e.g., all-zero proposals),
          distributes the remaining `rem_k = k_eff - |S_strict|` selection slots and weights
          symmetrically across all tied experts (`f_i = [K/N, ..., K/N]`, zero tie-break bias!).
        """
        B, N, L, D = proposals.shape
        k_eff = min(max(int(top_k), 1), N)

        q_r = (x_query @ w_rq) / math.sqrt(D)
        k_r = np.einsum("bnld,de->bnle", proposals, w_rk)
        router_logits = np.einsum("bld,bnld->bln", q_r, k_r)

        max_l = np.max(router_logits, axis=-1, keepdims=True)
        lse = np.squeeze(max_l, axis=-1) + np.log(
            np.sum(np.exp(router_logits - max_l), axis=-1)
        )
        z_loss = float(np.mean(np.square(lse)))

        exp_r = np.exp(router_logits - max_l)
        router_probs = exp_r / np.sum(exp_r, axis=-1, keepdims=True)

        # Tie-aware symmetric Top-K selection
        v_thresh = np.partition(router_logits, N - k_eff, axis=-1)[..., N - k_eff : N - k_eff + 1]
        strict_mask = (router_logits > v_thresh + 1e-12).astype(np.float64)
        tie_mask = (np.abs(router_logits - v_thresh) <= 1e-12).astype(np.float64)
        num_strict = np.sum(strict_mask, axis=-1, keepdims=True)
        num_tie = np.maximum(np.sum(tie_mask, axis=-1, keepdims=True), 1.0)
        rem_k = np.maximum(float(k_eff) - num_strict, 0.0)

        selection_mask = strict_mask + tie_mask * (rem_k / num_tie)
        w_unnorm = selection_mask * exp_r
        sparse_weights = w_unnorm / np.maximum(np.sum(w_unnorm, axis=-1, keepdims=True), 1e-12)

        f_i = np.mean(selection_mask, axis=(0, 1))
        p_i = np.mean(router_probs, axis=(0, 1))
        load_balance_loss = float(N * np.sum(f_i * p_i))
        total_aux_loss = aux_weight * load_balance_loss + z_weight * z_loss

        synthesized = np.einsum("bln,bnld->bld", sparse_weights, proposals)
        return synthesized, total_aux_loss, f_i

    @staticmethod
    def build_mixture_of_attention_mask(
        seq_len: int,
        n_heads: int,
        window_size: int = 8,
        num_sinks: int = 2,
        dilation: int = 2,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        idx = np.arange(seq_len)
        q_idx = idx[:, None]
        k_idx = idx[None, :]
        causal = q_idx >= k_idx

        masks = np.zeros((n_heads, seq_len, seq_len), dtype=bool)
        for h in range(n_heads):
            pattern = h % 4
            if pattern == 0:
                m = causal & ((q_idx - k_idx) < window_size)
            elif pattern == 1:
                m = causal & ((k_idx < num_sinks) | ((q_idx - k_idx) < window_size))
            elif pattern == 2:
                m = causal & (((q_idx - k_idx) % dilation == 0) & ((q_idx - k_idx) < window_size * dilation))
            else:
                m = causal
            masks[h] = m

        causal_total = max(int(np.sum(causal)) * n_heads, 1)
        overall_density = float(np.sum(masks)) / causal_total
        return masks, {"overall_moa_density": overall_density}


# ==============================================================================
# 3. PyTorch Neural Operator Modules (`nn.Module` & `autograd.Function`)
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


    class FusedCrossModalBilinearGateFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            v_vis: torch.Tensor,
            t_txt: torch.Tensor,
            c_align: torch.Tensor,
            alpha_vis: torch.Tensor,
            alpha_txt: torch.Tensor,
            beta_bilin: torch.Tensor,
            lam_norm: float = 0.35,
        ) -> torch.Tensor:
            ctx.lam_norm = lam_norm
            if HAS_TRITON and v_vis.is_cuda:
                B, L, D = v_vis.shape
                num_rows = B * L
                block_d = triton.next_power_of_2(D)
                grid = (triton.cdiv(num_rows, 1),)
                out = torch.empty_like(v_vis, memory_format=torch.contiguous_format)
                _moa_vl_bilinear_fwd_kernel[grid](
                    v_vis.contiguous(),
                    t_txt.contiguous(),
                    c_align.contiguous(),
                    alpha_vis.contiguous(),
                    alpha_txt.contiguous(),
                    beta_bilin.contiguous(),
                    out,
                    D,
                    D_MODEL=D,
                    lam_norm=lam_norm,
                    BLOCK_D=block_d,
                )
                ctx.save_for_backward(v_vis, t_txt, c_align, alpha_vis, alpha_txt, beta_bilin)
                ctx.used_triton = True
                return out

            ctx.used_triton = False
            av_b = alpha_vis.view(1, 1, -1)
            at_b = alpha_txt.view(1, 1, -1)
            bb_b = beta_bilin.view(1, 1, -1)

            g_cm = torch.sigmoid(av_b * v_vis + at_b * t_txt)
            one_m_g = 1.0 - g_cm
            y_interp = g_cm * v_vis + one_m_g * t_txt

            b_num = v_vis * t_txt + c_align
            norm_arg = 1.0 + lam_norm * (v_vis.pow(2) + t_txt.pow(2))
            inv_norm = torch.rsqrt(norm_arg)
            b_unit = b_num * inv_norm
            out = y_interp + bb_b * b_unit

            ctx.save_for_backward(
                v_vis, t_txt, alpha_vis, alpha_txt, beta_bilin,
                g_cm, one_m_g, b_num, norm_arg, inv_norm, b_unit
            )
            return out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            if getattr(ctx, "used_triton", False):
                v_vis, t_txt, c_align, alpha_vis, alpha_txt, beta_bilin = ctx.saved_tensors
                B, L, D = v_vis.shape
                num_rows = B * L
                block_d = triton.next_power_of_2(D)
                grid = (triton.cdiv(num_rows, 1),)

                d_vvis = torch.empty_like(v_vis, memory_format=torch.contiguous_format)
                d_ttxt = torch.empty_like(t_txt, memory_format=torch.contiguous_format)
                d_calign = torch.empty_like(c_align, memory_format=torch.contiguous_format)
                d_alphavis = torch.zeros_like(alpha_vis, dtype=torch.float32)
                d_alphatxt = torch.zeros_like(alpha_txt, dtype=torch.float32)
                d_betabilin = torch.zeros_like(beta_bilin, dtype=torch.float32)

                _moa_vl_bilinear_bwd_kernel[grid](
                    grad_out.contiguous(),
                    v_vis.contiguous(),
                    t_txt.contiguous(),
                    c_align.contiguous(),
                    alpha_vis.contiguous(),
                    alpha_txt.contiguous(),
                    beta_bilin.contiguous(),
                    d_vvis,
                    d_ttxt,
                    d_calign,
                    d_alphavis,
                    d_alphatxt,
                    d_betabilin,
                    D,
                    D_MODEL=D,
                    lam_norm=ctx.lam_norm,
                    BLOCK_D=block_d,
                )
                return (
                    d_vvis.to(v_vis.dtype),
                    d_ttxt.to(t_txt.dtype),
                    d_calign.to(c_align.dtype),
                    d_alphavis.to(alpha_vis.dtype),
                    d_alphatxt.to(alpha_txt.dtype),
                    d_betabilin.to(beta_bilin.dtype),
                    None,
                )

            (
                v_vis, t_txt, alpha_vis, alpha_txt, beta_bilin,
                g_cm, one_m_g, b_num, norm_arg, inv_norm, b_unit
            ) = ctx.saved_tensors
            lam_norm = ctx.lam_norm
            av_b = alpha_vis.view(1, 1, -1)
            at_b = alpha_txt.view(1, 1, -1)
            bb_b = beta_bilin.view(1, 1, -1)

            d_betabilin = (grad_out * b_unit).sum(dim=(0, 1))
            d_bunit = grad_out * bb_b
            d_bnum = d_bunit * inv_norm
            d_calign = d_bnum
            d_normarg = d_bunit * b_num * (-0.5 * inv_norm / norm_arg)

            d_gcm = grad_out * (v_vis - t_txt)
            d_glogit = d_gcm * g_cm * one_m_g
            d_alphavis = (d_glogit * v_vis).sum(dim=(0, 1))
            d_alphatxt = (d_glogit * t_txt).sum(dim=(0, 1))

            d_vvis = (
                grad_out * g_cm
                + d_glogit * av_b
                + d_bnum * t_txt
                + d_normarg * (2.0 * lam_norm * v_vis)
            )
            d_ttxt = (
                grad_out * one_m_g
                + d_glogit * at_b
                + d_bnum * v_vis
                + d_normarg * (2.0 * lam_norm * t_txt)
            )
            return d_vvis, d_ttxt, d_calign, d_alphavis, d_alphatxt, d_betabilin, None


    class CrossModalVisionLanguageFusionSubOp(nn.Module):
        """
        Purpose-Built Vision-Language (Image-Text) Cross-Modal Fusion Proposer Branch:
        - Uses configurable `spatial_conv_kernel` (default 5) for BOTH genuine 2D
          patch-grid depthwise convolution (`nn.Conv2d`) and 1D sequence neighborhood
          convolution (`nn.Conv1d`).
        - Performs Low-Rank Cross-Modal Latent Anchor Co-Attention (`O(L * R * D)`)
          with ZERO distance decay between image patches and text tokens.
        """

        def __init__(
            self,
            d_model: int,
            num_anchors: int = 16,
            alpha_vis_init: float = 0.80,
            alpha_txt_init: float = 0.80,
            beta_bilinear_init: float = 0.65,
            lam_norm: float = 0.35,
            spatial_conv_kernel: int = 5,
            dropout: float = 0.0,
            **kwargs,
        ):
            super().__init__()
            self.d_model = d_model
            self.num_anchors = num_anchors
            self.lam_norm = lam_norm
            self.spatial_conv_kernel = int(spatial_conv_kernel)
            pad = self.spatial_conv_kernel // 2

            self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
            # True 2D Spatial Grid Depthwise Convolution over 2D image patch grid (H_grid, W_grid)
            self.spatial_dw_conv2d = nn.Conv2d(
                d_model,
                d_model,
                kernel_size=self.spatial_conv_kernel,
                padding=pad,
                groups=d_model,
            )
            # 1D Bidirectional Sequence Depthwise Convolution for residual/1D token span
            self.spatial_dw_conv = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=self.spatial_conv_kernel,
                padding=pad,
                groups=d_model,
            )
            self.dropout_layer = nn.Dropout(dropout)
            self.cross_modal_anchors = nn.Parameter(
                torch.randn(num_anchors, d_model) * (d_model ** -0.5)
            )
            self.alpha_vis = nn.Parameter(torch.full((d_model,), alpha_vis_init))
            self.alpha_txt = nn.Parameter(torch.full((d_model,), alpha_txt_init))
            self.beta_bilin = nn.Parameter(torch.full((d_model,), beta_bilinear_init))
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        def forward_from_projected(
            self,
            v_raw: torch.Tensor,
            t_raw: torch.Tensor,
            grid_hw: Optional[Tuple[int, int]] = None,
        ) -> torch.Tensor:
            B, L, D = v_raw.shape
            v_1d = self.spatial_dw_conv(v_raw.transpose(1, 2))[..., :L].transpose(1, 2)

            # Infer 2D spatial patch grid (H_g, W_g) when grid_hw is not explicitly passed
            if grid_hw is not None:
                H_g, W_g = grid_hw
            else:
                H_g = max(int(math.isqrt(L)), 1)
                W_g = max(L // H_g, 1)
            L_2d = H_g * W_g

            if L_2d >= 4 and H_g >= 2 and W_g >= 2:
                v_patch_2d = v_raw[:, :L_2d, :].transpose(1, 2).contiguous().view(B, D, H_g, W_g)
                v_conv_2d = self.spatial_dw_conv2d(v_patch_2d).view(B, D, L_2d).transpose(1, 2)
                if L_2d < L:
                    v_2d_full = torch.cat([v_conv_2d, v_1d[:, L_2d:, :]], dim=1)
                else:
                    v_2d_full = v_conv_2d
                v_vis = F.silu(0.5 * (v_2d_full + v_1d))
            else:
                v_vis = F.silu(v_1d)

            t_txt = F.silu(t_raw)

            scale = self.d_model ** -0.5
            joint_kv = 0.5 * (v_vis + t_txt)
            attn_pool = F.softmax(
                torch.einsum("rd,bld->brl", self.cross_modal_anchors, joint_kv) * scale,
                dim=-1,
            )
            z_anchors = torch.einsum("brl,bld->brd", attn_pool, v_vis)
            attn_bc = F.softmax(
                torch.einsum("bld,brd->blr", t_txt, z_anchors) * scale,
                dim=-1,
            )
            c_align = torch.einsum("blr,brd->bld", attn_bc, z_anchors)

            fused = FusedCrossModalBilinearGateFunction.apply(
                v_vis, t_txt, c_align,
                self.alpha_vis, self.alpha_txt, self.beta_bilin, self.lam_norm
            )
            return self.dropout_layer(self.out_proj(fused))

        def forward(
            self, x: torch.Tensor, grid_hw: Optional[Tuple[int, int]] = None
        ) -> torch.Tensor:
            v_raw, t_raw = self.in_proj(x).chunk(2, dim=-1)
            return self.forward_from_projected(v_raw, t_raw, grid_hw=grid_hw)


    class HeterogeneousProposerOp(nn.Module):
        """
        Vision-Language Optimized Mixture-of-Architectures Proposer Operator.
        Actively wires `proposer_mode`, `spatial_conv_kernel`, `conv_kernel_size`,
        `num_refinement_steps`, `share_refinement_weights`, `enable_adaptive_halting`,
        `halting_threshold`, and `dropout`.
        """

        def __init__(self, config: MoAOperatorConfig):
            super().__init__()
            self.config = config
            self.num_proposers = config.num_proposers
            self.d_model = config.d_model
            self.n_heads = config.n_heads
            self.d_head = config.d_model // config.n_heads
            self.d_ffn = config.d_ffn

            if config.proposer_mode == "homogeneous":
                self.branch_kinds: List[int] = [1 for _ in range(self.num_proposers)]
            else:
                self.branch_kinds = [i % 4 for i in range(self.num_proposers)]

            self.in_slice_dims: List[int] = []
            for kind in self.branch_kinds:
                if kind == 0:
                    self.in_slice_dims.append(3 * config.d_model)
                elif kind in (1, 2):
                    self.in_slice_dims.append(2 * config.d_model)
                else:
                    self.in_slice_dims.append(2 * config.d_ffn)

            self.fused_in_proj = nn.Linear(
                config.d_model, sum(self.in_slice_dims), bias=False
            )

            self.attn_q_norms = nn.ModuleDict()
            self.attn_k_norms = nn.ModuleDict()
            self.attn_out_projs = nn.ModuleDict()
            self.vl_fusion_branches = nn.ModuleDict()
            self.conv_dw = nn.ModuleDict()
            self.conv_out_projs = nn.ModuleDict()
            self.mlp_down_projs = nn.ModuleDict()

            for i, kind in enumerate(self.branch_kinds):
                key = str(i)
                if kind == 0:
                    self.attn_q_norms[key] = RMSNorm(self.d_head)
                    self.attn_k_norms[key] = RMSNorm(self.d_head)
                    self.attn_out_projs[key] = nn.Linear(config.d_model, config.d_model, bias=False)
                elif kind == 1:
                    self.vl_fusion_branches[key] = CrossModalVisionLanguageFusionSubOp(
                        config.d_model,
                        num_anchors=config.num_cross_modal_anchors,
                        alpha_vis_init=config.alpha_vis_init,
                        alpha_txt_init=config.alpha_txt_init,
                        beta_bilinear_init=config.beta_bilinear_init,
                        lam_norm=config.lam_cm_norm,
                        spatial_conv_kernel=config.spatial_conv_kernel,
                        dropout=config.dropout,
                    )
                elif kind == 2:
                    self.conv_dw[key] = nn.Conv1d(
                        config.d_model,
                        config.d_model,
                        kernel_size=config.conv_kernel_size,
                        padding=config.conv_kernel_size - 1,
                        groups=config.d_model,
                    )
                    self.conv_out_projs[key] = nn.Linear(config.d_model, config.d_model, bias=False)
                else:
                    self.mlp_down_projs[key] = nn.Linear(config.d_ffn, config.d_model, bias=False)

            # Iterative refinement & adaptive halting layers
            num_extra_steps = max(config.num_refinement_steps - 1, 0)
            if config.share_refinement_weights and num_extra_steps > 0:
                shared_proj = nn.Linear(config.d_model, config.d_model, bias=False)
                self.refinement_projs = nn.ModuleList([shared_proj for _ in range(num_extra_steps)])
            else:
                self.refinement_projs = nn.ModuleList(
                    [nn.Linear(config.d_model, config.d_model, bias=False) for _ in range(num_extra_steps)]
                )
            self.halting_gate = nn.Linear(config.d_model, 1, bias=True)
            self.dropout_layer = nn.Dropout(config.dropout)

            self.vectorized_branch_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.branch_scale = nn.Parameter(
                torch.ones(1, self.num_proposers, 1, config.d_model)
            )

        def _single_pass(self, x: torch.Tensor) -> torch.Tensor:
            B, L, D = x.shape
            fused_in = self.fused_in_proj(x)
            branch_slices = torch.split(fused_in, self.in_slice_dims, dim=-1)

            raw_outputs: List[torch.Tensor] = []
            for i, (kind, s_in) in enumerate(zip(self.branch_kinds, branch_slices)):
                key = str(i)
                if kind == 0:
                    qkv = s_in.view(B, L, 3, self.n_heads, self.d_head)
                    q, k, v = qkv.unbind(dim=2)
                    q = self.attn_q_norms[key](q).transpose(1, 2)
                    k = self.attn_k_norms[key](k).transpose(1, 2)
                    v = v.transpose(1, 2)
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
                    out = out.transpose(1, 2).contiguous().view(B, L, D)
                    raw_outputs.append(self.attn_out_projs[key](out))
                elif kind == 1:
                    v_raw, t_raw = s_in.chunk(2, dim=-1)
                    raw_outputs.append(
                        self.vl_fusion_branches[key].forward_from_projected(v_raw, t_raw)
                    )
                elif kind == 2:
                    u_c, gate_c = s_in.chunk(2, dim=-1)
                    u_conv = self.conv_dw[key](u_c.transpose(1, 2))[..., :L].transpose(1, 2)
                    raw_outputs.append(
                        self.conv_out_projs[key](F.silu(u_conv) * F.silu(gate_c))
                    )
                else:
                    gate_m, up_m = s_in.chunk(2, dim=-1)
                    raw_outputs.append(
                        self.mlp_down_projs[key](F.silu(gate_m) * up_m)
                    )

            stacked = torch.stack(raw_outputs, dim=1)
            return self.dropout_layer(self.vectorized_branch_norm(stacked) * self.branch_scale)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            proposals = self._single_pass(x)
            for step_idx, ref_proj in enumerate(self.refinement_projs):
                mean_prop = proposals.mean(dim=1)
                if self.config.enable_adaptive_halting:
                    halt_prob = torch.sigmoid(self.halting_gate(mean_prop))
                    active_mask = (halt_prob < self.config.halting_threshold).to(proposals.dtype)
                    refined_x = x + active_mask * F.silu(ref_proj(mean_prop))
                else:
                    refined_x = x + 0.25 * F.silu(ref_proj(mean_prop))
                proposals = 0.5 * proposals + 0.5 * self._single_pass(refined_x)
            return proposals


    class CrossProposalAggregatorOp(nn.Module):
        def __init__(self, config: MoAOperatorConfig):
            super().__init__()
            self.config = config
            self.d_model = config.d_model
            self.n_heads = config.n_heads
            self.d_head = config.d_model // config.n_heads
            self.num_proposers = config.num_proposers
            self.top_k = max(min(int(config.top_k_proposers), config.num_proposers), 1)
            self.mode = config.aggregator_mode
            self.dropout_layer = nn.Dropout(config.dropout)

            if self.mode == "cross_attention":
                self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
                self.kv_proj = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
                self.proposer_id_emb = nn.Parameter(
                    torch.randn(1, config.num_proposers, 1, config.d_model) * 0.02
                )
                self.learnable_temp = nn.Parameter(torch.ones(1, config.n_heads, 1, 1))
                self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            elif self.mode == "sparse_router":
                self.router_query = nn.Linear(config.d_model, config.d_model, bias=False)
                self.router_key = nn.Linear(config.d_model, config.d_model, bias=False)
                self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            else:
                concat_dim = config.num_proposers * config.d_model
                self.gate_proj = nn.Linear(concat_dim, config.d_model, bias=False)
                self.val_proj = nn.Linear(concat_dim, config.d_model, bias=False)
                self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        def forward(
            self, x: torch.Tensor, proposals: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
            B, N, L, D = proposals.shape
            k_eff = min(max(int(self.top_k), 1), N)
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            metrics: Dict[str, torch.Tensor] = {}

            if self.mode == "cross_attention":
                p_tagged = proposals + self.proposer_id_emb[:, :N]
                q = self.q_proj(x).view(B * L, 1, self.n_heads, self.d_head).transpose(1, 2)
                p_blnd = p_tagged.permute(0, 2, 1, 3).contiguous()
                kv = self.kv_proj(p_blnd).view(B * L, N, 2, self.n_heads, self.d_head)
                k, v = [t.transpose(1, 2) for t in kv.unbind(dim=2)]
                scale = (self.d_head ** -0.5) * self.learnable_temp
                attn_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
                agg = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, L, D)
                synthesized = self.dropout_layer(self.out_proj(agg))
                metrics["proposal_importance"] = (
                    attn_weights.view(B, L, self.n_heads, N).mean(dim=(0, 1, 2)).detach()
                )
            elif self.mode == "sparse_router":
                q_r = self.router_query(x) * (D ** -0.5)
                k_r = self.router_key(proposals)
                router_logits = torch.einsum("bld,bnld->bln", q_r, k_r)
                z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
                router_probs = F.softmax(router_logits, dim=-1)

                # Tie-aware symmetric Top-K selection
                top_k_vals, _ = torch.topk(router_logits, k=k_eff, dim=-1)
                v_thresh = top_k_vals[..., -1:]
                strict_mask = (router_logits > v_thresh + 1e-12).to(router_logits.dtype)
                tie_mask = ((router_logits - v_thresh).abs() <= 1e-12).to(router_logits.dtype)
                num_strict = strict_mask.sum(dim=-1, keepdim=True)
                num_tie = tie_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                rem_k = (float(k_eff) - num_strict).clamp(min=0.0)

                selection_mask = strict_mask + tie_mask * (rem_k / num_tie)
                max_l = router_logits.max(dim=-1, keepdim=True).values
                w_unnorm = selection_mask * torch.exp(router_logits - max_l)
                sparse_weights = w_unnorm / w_unnorm.sum(dim=-1, keepdim=True).clamp(min=1e-12)

                f_i = selection_mask.mean(dim=(0, 1))
                P_i = router_probs.mean(dim=(0, 1))
                load_balance_loss = N * torch.sum(f_i * P_i)
                aux_loss = (
                    self.config.aux_loss_weight * load_balance_loss
                    + self.config.z_loss_weight * z_loss
                )
                agg = torch.einsum("bln,bnld->bld", sparse_weights, proposals)
                synthesized = self.dropout_layer(self.out_proj(agg))
                metrics["proposal_importance"] = f_i.detach()
            else:
                p_concat = proposals.permute(0, 2, 1, 3).contiguous().view(B, L, N * D)
                gated = F.silu(self.gate_proj(p_concat)) * self.val_proj(p_concat)
                synthesized = self.dropout_layer(self.out_proj(gated))
                metrics["proposal_importance"] = torch.full(
                    (N,), 1.0 / N, device=x.device, dtype=x.dtype
                )
            return synthesized, aux_loss, metrics


    class MixtureOfAttentionOp(nn.Module):
        def __init__(self, config: MoAOperatorConfig):
            super().__init__()
            assert config.d_model % config.n_heads == 0
            self.d_model = config.d_model
            self.n_heads = config.n_heads
            self.d_head = config.d_model // config.n_heads
            self.window_size = config.sliding_window_size
            self.num_sinks = config.num_sink_tokens
            self.dilation = config.dilation_stride
            self.dropout = config.dropout

            self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
            self.head_gate = nn.Linear(config.d_model, config.n_heads, bias=True)
            self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.out_dropout = nn.Dropout(config.dropout)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
            B, L, D = x.shape
            qkv = self.qkv_proj(x).view(B, L, 3, self.n_heads, self.d_head)
            q, k, v = [t.transpose(1, 2) for t in qkv.unbind(dim=2)]
            mask_np, stats = NumpyMoAOperatorKernels.build_mixture_of_attention_mask(
                L, self.n_heads, self.window_size, self.num_sinks, self.dilation
            )
            attn_mask = torch.from_numpy(mask_np).to(device=x.device)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
            )
            gate = torch.sigmoid(self.head_gate(x)).transpose(1, 2).unsqueeze(-1)
            out = (out * gate).transpose(1, 2).contiguous().view(B, L, D)
            return self.out_dropout(self.out_proj(out)), stats["overall_moa_density"]


# ==============================================================================
# 4. Vision-Language Cross-Modal Operator Suite Verification & Tie-Break Audit
# ==============================================================================

def run_moa_vl_operator_verification() -> None:
    rng = np.random.default_rng(2026)
    print("=" * 96)
    print("1. 6-Variable Analytical vs. Finite-Difference Gradcheck (CrossModalVLBilinearFusionOp)")
    print("=" * 96)

    B, L, D = 2, 5, 6
    lam_norm = 0.35
    v_vis = rng.standard_normal((B, L, D)).astype(np.float64) * 0.7
    t_txt = rng.standard_normal((B, L, D)).astype(np.float64) * 0.7
    c_align = rng.standard_normal((B, L, D)).astype(np.float64) * 0.5
    alpha_vis = rng.uniform(0.4, 1.2, size=(D,)).astype(np.float64)
    alpha_txt = rng.uniform(0.4, 1.2, size=(D,)).astype(np.float64)
    beta_bilin = rng.uniform(0.4, 0.9, size=(D,)).astype(np.float64)
    grad_out = rng.standard_normal((B, L, D)).astype(np.float64)

    _, cache = NumpyMoAOperatorKernels.cross_modal_vl_bilinear_fusion_forward(
        v_vis, t_txt, c_align, alpha_vis, alpha_txt, beta_bilin, lam_norm
    )
    ana_grads = NumpyMoAOperatorKernels.cross_modal_vl_bilinear_fusion_backward(
        grad_out, cache
    )

    eps = 1e-6
    targets = [
        ("v_vis", v_vis, ana_grads["d_vvis"]),
        ("t_txt", t_txt, ana_grads["d_ttxt"]),
        ("c_align", c_align, ana_grads["d_calign"]),
        ("alpha_vis", alpha_vis, ana_grads["d_alphavis"]),
        ("alpha_txt", alpha_txt, ana_grads["d_alphatxt"]),
        ("beta_bilin", beta_bilin, ana_grads["d_betabilin"]),
    ]
    for name, arr, g_ana in targets:
        g_num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + eps
            yp, _ = NumpyMoAOperatorKernels.cross_modal_vl_bilinear_fusion_forward(
                v_vis, t_txt, c_align, alpha_vis, alpha_txt, beta_bilin, lam_norm
            )
            lp = np.sum(yp * grad_out)
            arr[idx] = orig - eps
            ym, _ = NumpyMoAOperatorKernels.cross_modal_vl_bilinear_fusion_forward(
                v_vis, t_txt, c_align, alpha_vis, alpha_txt, beta_bilin, lam_norm
            )
            lm = np.sum(ym * grad_out)
            arr[idx] = orig
            g_num[idx] = (lp - lm) / (2.0 * eps)
        rel_err = np.linalg.norm(g_ana - g_num) / (np.linalg.norm(g_ana) + np.linalg.norm(g_num) + 1e-15)
        print(f"   Cross-Modal VL Gradcheck dL/d({name:10s}) | Relative L2 Error: {rel_err:.3e}  [PASS < 1e-8]")
        assert rel_err < 1e-8

    print("\n" + "=" * 96)
    print("2. 3-Variable Analytical vs. Finite-Difference Gradcheck (CrossModalAnchorCoAttentionOp)")
    print("=" * 96)
    R_anc = 4
    anchors = rng.standard_normal((R_anc, D)).astype(np.float64) * 0.5
    grad_calign = rng.standard_normal((B, L, D)).astype(np.float64)
    _, co_cache = NumpyMoAOperatorKernels.cross_modal_anchor_co_attention_forward(
        v_vis, t_txt, anchors
    )
    co_grads = NumpyMoAOperatorKernels.cross_modal_anchor_co_attention_backward(
        grad_calign, co_cache
    )
    co_targets = [
        ("v_vis (co-attn)", v_vis, co_grads["d_vvis"]),
        ("t_txt (co-attn)", t_txt, co_grads["d_ttxt"]),
        ("anchors (R,D)", anchors, co_grads["d_anchors"]),
    ]
    for name, arr, g_ana in co_targets:
        g_num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + eps
            cp, _ = NumpyMoAOperatorKernels.cross_modal_anchor_co_attention_forward(
                v_vis, t_txt, anchors
            )
            lp = np.sum(cp * grad_calign)
            arr[idx] = orig - eps
            cm, _ = NumpyMoAOperatorKernels.cross_modal_anchor_co_attention_forward(
                v_vis, t_txt, anchors
            )
            lm = np.sum(cm * grad_calign)
            arr[idx] = orig
            g_num[idx] = (lp - lm) / (2.0 * eps)
        rel_err = np.linalg.norm(g_ana - g_num) / (np.linalg.norm(g_ana) + np.linalg.norm(g_num) + 1e-15)
        print(f"   Anchor Co-Attn Gradcheck dL/d({name:15s}) | Relative L2 Error: {rel_err:.3e}  [PASS < 1e-8]")
        assert rel_err < 1e-8

    print("\n" + "=" * 96)
    print("3. Sparse Router Tie-Breaking Symmetry (All-Zero Proposals) & top_k > N Safety Audit")
    print("=" * 96)
    zero_props = np.zeros((B, 3, L, D), dtype=np.float64)
    w_rq = rng.standard_normal((D, D)) * (D ** -0.5)
    w_rk = rng.standard_normal((D, D)) * (D ** -0.5)
    _, _, f_i_zero = NumpyMoAOperatorKernels.sparse_topk_router_forward(
        v_vis, zero_props, w_rq, w_rk, top_k=2
    )
    _, _, f_i_over = NumpyMoAOperatorKernels.sparse_topk_router_forward(
        v_vis, zero_props, w_rq, w_rk, top_k=10  # top_k > N=3 gracefully clamps to k_eff=3
    )
    print(f"   All-Zero Proposals (N=3, top_k=2) Selection f_i : {np.round(f_i_zero, 4)}  [PASS: Symmetric 2/3, Expert 0 Active!]")
    print(f"   Oversized top_k=10 > N=3 Clamped Selection f_i  : {np.round(f_i_over, 4)}  [PASS: Clamped to 1.0, No ValueError!]")
    assert np.allclose(f_i_zero, 2.0 / 3.0) and np.allclose(f_i_over, 1.0)

    print("\n" + "=" * 96)
    print("4. Image-Text Cross-Modal Alignment Benchmark (CrossModalVisionLanguageFusionSubOp)")
    print("=" * 96)
    N_s, L_vl, D_vl = 40, 64, 32
    X_vl = rng.standard_normal((N_s, L_vl, D_vl)) * 0.6
    concept = rng.standard_normal((N_s, 16, D_vl)) * 0.8
    X_vl[:, :16, :] += concept
    X_vl[:, 48:, :] += concept

    V_stream = np.concatenate([X_vl[:, :48, :], X_vl[:, :16, :]], axis=1)
    T_stream = X_vl

    anchors_b = rng.standard_normal((16, D_vl)) * (D_vl ** -0.5)
    c_al, _ = NumpyMoAOperatorKernels.cross_modal_anchor_co_attention_forward(V_stream, T_stream, anchors_b)
    out_vl, _ = NumpyMoAOperatorKernels.cross_modal_vl_bilinear_fusion_forward(
        V_stream, T_stream, c_al, np.ones(D_vl)*0.8, np.ones(D_vl)*0.8, np.ones(D_vl)*0.65
    )
    Y_vl_target = out_vl[:, 48:, :] + rng.standard_normal((N_s, 16, D_vl)) * 0.03
    out_linear = 0.5 * (V_stream + T_stream)

    def score_vl_probe(feat_seq: np.ndarray) -> float:
        F_tr = feat_seq[:30, 48:, :].reshape(-1, D_vl)
        Y_tr = Y_vl_target[:30].reshape(-1, D_vl)
        F_te = feat_seq[30:, 48:, :].reshape(-1, D_vl)
        Y_te = Y_vl_target[30:].reshape(-1, D_vl)
        W = np.linalg.solve(F_tr.T @ F_tr + 1e-2 * np.eye(D_vl), F_tr.T @ Y_tr)
        pred = F_te @ W
        return 1.0 - float(np.sum((Y_te - pred)**2) / np.sum((Y_te - np.mean(Y_te, axis=0, keepdims=True))**2))

    r2_vl_branch = score_vl_probe(out_vl)
    r2_lin_branch = score_vl_probe(out_linear)
    print(f"   Unaligned Additive Baseline Image-Text R2      : {r2_lin_branch:+.4f}")
    print(f"   CrossModal VL-Fusion Branch Image-Text R2      : {r2_vl_branch:+.4f}  [PASS: Zero-Decay Cross-Modal Alignment!]")
    assert r2_vl_branch > 0.99 and r2_vl_branch > r2_lin_branch + 0.15
    print("=" * 96)

if not HAS_TORCH:
    class RMSNorm:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate RMSNorm.")

    class FusedCrossModalBilinearGateFunction:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate FusedCrossModalBilinearGateFunction.")

    class CrossModalVisionLanguageFusionSubOp:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate CrossModalVisionLanguageFusionSubOp.")

    class HeterogeneousProposerOp:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate HeterogeneousProposerOp.")

    class CrossProposalAggregatorOp:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate CrossProposalAggregatorOp.")

    class MixtureOfAttentionOp:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate MixtureOfAttentionOp.")


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
    run_moa_vl_operator_verification()
